"""
FastAPI router for the Workflow Research Assistant.

Provides endpoints for creating, executing, approving, and
managing workflow sessions.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agent.planner import planner
from agent.session_store import session_store
from agent.workflow_engine import WorkflowEngine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/workflow", tags=["workflow"])

# Track running engines for cancel/resume
_running_engines: Dict[str, WorkflowEngine] = {}


# ---------------------------------------------------------------------------
# Request/Response models
# ---------------------------------------------------------------------------

class CreateWorkflowRequest(BaseModel):
    goal: str
    language: Optional[str] = None
    map_context: Optional[Dict[str, Any]] = None


class ApproveStepRequest(BaseModel):
    step_id: str


class TemplateRequest(BaseModel):
    region_name: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/create")
async def create_workflow(req: CreateWorkflowRequest):
    """Create a workflow plan from a natural-language goal.

    Returns the plan (steps, narrative) without starting execution.
    """
    try:
        session = await planner.create_plan(
            user_goal=req.goal,
            language=req.language,
            map_context=req.map_context,
        )
        session_store.save(session)

        return {
            "session_id": session.session_id,
            "user_goal": session.user_goal,
            "language": session.language,
            "agent_type": session.agent_type,
            "plan_narrative": session.plan_narrative,
            "state": session.state,
            "steps": [s.model_dump() for s in session.steps],
        }
    except Exception as e:
        logger.exception(f"Failed to create workflow: {e}")
        raise HTTPException(500, f"Planning failed: {e}")


@router.post("/template/{template_name}")
async def create_from_template(template_name: str, req: TemplateRequest):
    """Instantiate a prebuilt workflow template."""
    try:
        if template_name == "lda_sharad":
            from agent.templates.lda_sharad import create_lda_sharad_workflow
            session = create_lda_sharad_workflow(
                region_name=req.region_name or "Arcadia Planitia",
            )
        elif template_name == "terrace_epsilon":
            from agent.templates.terrace_epsilon import create_terrace_epsilon_workflow
            session = create_terrace_epsilon_workflow(
                region_name=req.region_name,
                lat=req.lat,
                lon=req.lon,
            )
        elif template_name == "ice_search":
            from agent.templates.ice_search import create_ice_search_workflow
            session = create_ice_search_workflow(
                region_name=req.region_name or "Arcadia Planitia",
            )
        else:
            raise HTTPException(404, f"Unknown template: {template_name}")

        session_store.save(session)
        return {
            "session_id": session.session_id,
            "user_goal": session.user_goal,
            "agent_type": session.agent_type,
            "plan_narrative": session.plan_narrative,
            "state": session.state,
            "steps": [s.model_dump() for s in session.steps],
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to create template workflow: {e}")
        raise HTTPException(500, str(e))


@router.post("/{session_id}/start")
async def start_workflow(session_id: str):
    """Start executing a workflow. Returns an SSE stream."""
    session = session_store.load(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    engine = WorkflowEngine(session)
    _running_engines[session_id] = engine

    async def event_stream():
        try:
            async for event in engine.run():
                session_store.save(session)
                yield f"data: {json.dumps(event, default=str)}\n\n"
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.exception(f"Workflow stream error: {e}")
            yield f"data: {json.dumps({'event': 'error', 'data': {'error': str(e)}})}\n\n"
        finally:
            _running_engines.pop(session_id, None)
            session_store.save(session)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/{session_id}/approve")
async def approve_step(session_id: str, req: ApproveStepRequest):
    """Approve a pending step. After approval, client should call /resume."""
    session = session_store.load(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    engine = _running_engines.get(session_id) or WorkflowEngine(session)
    if engine.approve_step(req.step_id):
        session_store.save(session)
        return {"status": "approved", "step_id": req.step_id}
    else:
        raise HTTPException(400, f"Step {req.step_id} is not awaiting approval")


@router.post("/{session_id}/deny")
async def deny_step(session_id: str, req: ApproveStepRequest):
    """Deny (skip) a pending step. After denial, client should call /resume."""
    session = session_store.load(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    engine = _running_engines.get(session_id) or WorkflowEngine(session)
    if engine.deny_step(req.step_id):
        session_store.save(session)
        return {"status": "skipped", "step_id": req.step_id}
    else:
        raise HTTPException(400, f"Step {req.step_id} is not awaiting approval")


@router.get("/{session_id}/resume")
async def resume_workflow(session_id: str):
    """Resume a paused workflow (after approval/denial). Returns SSE stream."""
    session = session_store.load(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    engine = WorkflowEngine(session)
    _running_engines[session_id] = engine

    async def event_stream():
        try:
            async for event in engine.run():
                session_store.save(session)
                yield f"data: {json.dumps(event, default=str)}\n\n"
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.exception(f"Resume stream error: {e}")
            yield f"data: {json.dumps({'event': 'error', 'data': {'error': str(e)}})}\n\n"
        finally:
            _running_engines.pop(session_id, None)
            session_store.save(session)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/sessions/list")
async def list_sessions():
    """List all workflow sessions (for history UI)."""
    return session_store.list_sessions()


@router.get("/{session_id}")
async def get_session(session_id: str):
    """Get current session state (polling fallback)."""
    session = session_store.load(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    return {
        "session_id": session.session_id,
        "user_goal": session.user_goal,
        "state": session.state,
        "agent_type": session.agent_type,
        "plan_narrative": session.plan_narrative,
        "current_step_index": session.current_step_index,
        "steps": [s.model_dump() for s in session.steps],
        "context_keys": list(session.context.keys()),
        "report_markdown": session.context.get("report_markdown"),
    }


@router.post("/{session_id}/cancel")
async def cancel_workflow(session_id: str):
    """Cancel a running workflow."""
    engine = _running_engines.get(session_id)
    if engine:
        engine.cancel()

    session = session_store.load(session_id)
    if session:
        session.state = "cancelled"
        session_store.save(session)
        return {"status": "cancelled"}
    raise HTTPException(404, "Session not found")
