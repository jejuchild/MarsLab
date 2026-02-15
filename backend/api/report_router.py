"""
Landing Site Report Router — API endpoints for multi-region comparison reports.

POST /api/report/generate       — Start report generation (SSE stream)
GET  /api/report/regions        — List all available regions for picker
GET  /api/report/resume/{id}    — Reconnect to running/completed session (SSE replay)
GET  /api/report/session/{id}   — Get session state (polling fallback)
GET  /api/report/download/{id}  — Download report (md or pdf)
"""

import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, field_validator

from .mars_regions import list_all_regions
from .report_orchestrator import (
    GroundRules,
    ReportConfig,
    ReportSession,
    cancel_report_session,
    get_report_session,
    list_report_sessions,
    start_report_background,
    stream_report_events,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/report", tags=["landing-site-report"])


# =============================================================================
# Request Models
# =============================================================================


class GroundRulesRequest(BaseModel):
    lat_min: Optional[float] = None
    lat_max: Optional[float] = None
    lon_min: Optional[float] = None
    lon_max: Optional[float] = None
    include_regions: List[str] = []
    exclude_regions: List[str] = []
    include_tags: List[str] = []
    exclude_tags: List[str] = []
    min_slope_safety: Optional[str] = None
    custom_notes: str = ""

    @field_validator("min_slope_safety")
    @classmethod
    def validate_slope_safety(cls, v):
        if v is not None and v not in ("FAVORABLE", "MARGINAL"):
            raise ValueError("min_slope_safety must be FAVORABLE or MARGINAL")
        return v


class ReportGenerateRequest(BaseModel):
    ground_rules: GroundRulesRequest = GroundRulesRequest()
    analyses: List[str] = ["slope", "subsurface", "mineral"]
    auto_download: bool = True
    instruments: List[str] = [
        "CRISM",
        "SHARAD",
        "SHARAD_HIGHRES",
        "CTX",
        "HIRISE_DTM",
    ]
    max_regions: int = 5

    @field_validator("analyses")
    @classmethod
    def validate_analyses(cls, v):
        valid = {"slope", "subsurface", "mineral"}
        for a in v:
            if a not in valid:
                raise ValueError(f"Invalid analysis type: {a}. Must be one of {valid}")
        return v

    @field_validator("max_regions")
    @classmethod
    def validate_max_regions(cls, v):
        if v < 1 or v > 10:
            raise ValueError("max_regions must be between 1 and 10")
        return v


# =============================================================================
# SSE Helper
# =============================================================================


def _sse_response(session: ReportSession, from_index: int = 0) -> StreamingResponse:
    async def event_generator():
        try:
            async for event in stream_report_events(session, from_index=from_index):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            logger.error(f"Report SSE error for {session.session_id}: {e}")
            yield f"data: {json.dumps({'event': 'error', 'data': {'error': str(e)}})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# =============================================================================
# Endpoints
# =============================================================================


@router.get("/regions")
async def report_regions():
    """List all available Mars regions for the region picker."""
    regions = list_all_regions()
    return [
        {
            "region_id": r.region_id,
            "display_name": r.display_name,
            "center_lat": round(r.center_lat, 2),
            "center_lon": round(r.center_lon, 2),
            "tags": r.tags,
            "lat_min": r.lat_min,
            "lat_max": r.lat_max,
            "lon_min": r.lon_min,
            "lon_max": r.lon_max,
        }
        for r in regions
    ]


@router.post("/generate")
async def report_generate(request: ReportGenerateRequest):
    """Start report generation and stream SSE events."""
    # Convert request to config
    gr = request.ground_rules
    ground_rules = GroundRules(
        lat_min=gr.lat_min,
        lat_max=gr.lat_max,
        lon_min=gr.lon_min,
        lon_max=gr.lon_max,
        include_regions=gr.include_regions,
        exclude_regions=gr.exclude_regions,
        include_tags=gr.include_tags,
        exclude_tags=gr.exclude_tags,
        min_slope_safety=gr.min_slope_safety,
        custom_notes=gr.custom_notes,
    )

    config = ReportConfig(
        ground_rules=ground_rules,
        analyses=request.analyses,
        auto_download=request.auto_download,
        instruments=request.instruments,
        max_regions=request.max_regions,
    )

    session = start_report_background(config)
    return _sse_response(session, from_index=0)


@router.get("/resume/{session_id}")
async def report_resume(session_id: str, from_index: int = Query(0, ge=0)):
    """Reconnect to a running or completed report session."""
    session = get_report_session(session_id)
    if not session:
        raise HTTPException(
            status_code=404, detail=f"Report session {session_id} not found"
        )
    return _sse_response(session, from_index=from_index)


@router.post("/stop/{session_id}")
async def report_stop(session_id: str):
    """Cancel a running report session."""
    cancelled = await cancel_report_session(session_id)
    if not cancelled:
        raise HTTPException(status_code=400, detail="Session not found or already finished")
    return {"status": "cancelled", "session_id": session_id}


@router.get("/session/{session_id}")
async def report_session(session_id: str):
    """Get current state of a report session (polling fallback)."""
    session = get_report_session(session_id)
    if not session:
        raise HTTPException(
            status_code=404, detail=f"Report session {session_id} not found"
        )
    return session.to_dict()


@router.get("/sessions")
async def report_sessions():
    """List all past report sessions."""
    sessions = list_report_sessions()
    result = []
    for s in sessions:
        recommended = None
        if s.comparison and isinstance(s.comparison, dict):
            rec = s.comparison.get("recommended")
            if rec:
                recommended = {
                    "region_name": rec.get("region_name"),
                    "score": rec.get("score"),
                }
        region_names = [r.region_name for r in s.regions]
        result.append({
            "session_id": s.session_id,
            "status": s.status,
            "region_count": len(s.regions),
            "region_names": region_names[:5],
            "created_at": s.created_at,
            "recommended": recommended,
        })
    return result


@router.get("/download/{session_id}")
async def report_download(
    session_id: str,
    format: str = Query("md", pattern="^(md|pdf)$"),
):
    """Download the finished report as Markdown or PDF."""
    session = get_report_session(session_id)
    if not session:
        raise HTTPException(
            status_code=404, detail=f"Report session {session_id} not found"
        )
    if session.status != "done":
        raise HTTPException(status_code=400, detail="Report has not completed yet")

    md_content = session.report_markdown
    if not md_content:
        raise HTTPException(status_code=500, detail="Report content is empty")

    base_filename = f"marslab_comparison_report_{session_id}"

    if format == "pdf":
        try:
            import markdown as md_lib
            from weasyprint import HTML

            html_body = md_lib.markdown(md_content, extensions=["tables"])
            styled_html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
  body {{ font-family: 'Helvetica Neue', Arial, sans-serif; margin: 40px; color: #1a1a2e; line-height: 1.6; font-size: 11pt; }}
  h1 {{ color: #16213e; border-bottom: 2px solid #0f3460; padding-bottom: 8px; font-size: 20pt; }}
  h2 {{ color: #0f3460; margin-top: 24px; font-size: 14pt; }}
  h3 {{ color: #1a4a6e; margin-top: 16px; font-size: 12pt; }}
  table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
  th, td {{ border: 1px solid #c8d6e5; padding: 6px 10px; text-align: left; font-size: 10pt; }}
  th {{ background: #0f3460; color: white; }}
  tr:nth-child(even) {{ background: #f0f4f8; }}
  strong {{ color: #0f3460; }}
  hr {{ border: none; border-top: 1px solid #c8d6e5; margin: 24px 0; }}
  em {{ color: #6b7c9c; }}
  blockquote {{ border-left: 3px solid #e74c3c; padding-left: 12px; color: #c0392b; margin: 12px 0; }}
</style>
</head><body>{html_body}</body></html>"""

            pdf_bytes = HTML(string=styled_html).write_pdf()
            return Response(
                content=pdf_bytes,
                media_type="application/pdf",
                headers={
                    "Content-Disposition": f'attachment; filename="{base_filename}.pdf"'
                },
            )
        except ImportError:
            logger.warning(
                "weasyprint or markdown not installed, falling back to Markdown"
            )
        except Exception as e:
            logger.error(f"PDF generation failed: {e}")

    return Response(
        content=md_content.encode("utf-8"),
        media_type="text/markdown; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="{base_filename}.md"'
        },
    )
