"""
Workflow execution engine with state machine and approval gates.

Executes workflow steps sequentially, pausing at approval gates
and yielding SSE events for real-time frontend updates.
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, Optional

from agent.schemas import (
    StepStatus,
    WorkflowEvent,
    WorkflowSession,
    WorkflowState,
    WorkflowStep,
)
from agent.tool_registry import workflow_tools

logger = logging.getLogger(__name__)


def _resolve_template_refs(value: Any, context: Dict[str, Any]) -> Any:
    """Replace {{context.key}} references in params with actual context values.

    Supports nested dotted paths: {{context.region_bbox.min_lat}}
    If the entire value is a template string, return the resolved object directly.
    If it's part of a larger string, do string interpolation.
    """
    if isinstance(value, str):
        # Check for exact match: entire string is one template ref
        exact = re.fullmatch(r"\{\{context\.([a-zA-Z0-9_.]+)\}\}", value)
        if exact:
            path = exact.group(1)
            return _get_nested(context, path, value)

        # Partial interpolation: replace {{context.xxx}} within string
        def _replacer(m: re.Match) -> str:
            path = m.group(1)
            resolved = _get_nested(context, path, m.group(0))
            return str(resolved)

        return re.sub(r"\{\{context\.([a-zA-Z0-9_.]+)\}\}", _replacer, value)

    elif isinstance(value, dict):
        return {k: _resolve_template_refs(v, context) for k, v in value.items()}
    elif isinstance(value, list):
        return [_resolve_template_refs(v, context) for v in value]
    return value


def _get_nested(d: Dict, path: str, default: Any = None) -> Any:
    """Resolve a dotted path like 'region_bbox.min_lat' in a dict."""
    parts = path.split(".")
    obj: Any = d
    for part in parts:
        if isinstance(obj, dict):
            obj = obj.get(part, default)
        else:
            return default
    return obj


class WorkflowEngine:
    """Executes a WorkflowSession step-by-step with approval gates."""

    def __init__(self, session: WorkflowSession) -> None:
        self.session = session
        self._cancelled = False

    # ------------------------------------------------------------------
    # Main execution loop
    # ------------------------------------------------------------------

    async def run(self) -> AsyncIterator[Dict[str, Any]]:
        """Execute steps, yielding SSE event dicts.

        Pauses and returns when:
        - A step requires approval (state → AWAITING_APPROVAL)
        - Workflow completes
        - Workflow fails
        - Cancelled
        """
        self.session.state = WorkflowState.EXECUTING
        self.session.log("workflow_started")

        while self.session.current_step_index < len(self.session.steps):
            if self._cancelled:
                self.session.state = WorkflowState.CANCELLED
                self.session.log("workflow_cancelled")
                yield _event("workflow_cancelled", {"session_id": self.session.session_id})
                return

            step = self.session.steps[self.session.current_step_index]

            # --- Approval gate ---
            if step.requires_approval and step.status == StepStatus.PENDING:
                step.status = StepStatus.AWAITING_APPROVAL
                self.session.state = WorkflowState.AWAITING_APPROVAL
                self.session.log("approval_required", step_id=step.step_id, tool=step.tool_name)

                # Distinguish step-by-step gates from original approval gates
                is_step_gate = getattr(step, "_step_gate", False)

                yield _event("approval_required", {
                    "session_id": self.session.session_id,
                    "step_id": step.step_id,
                    "step_index": self.session.current_step_index,
                    "tool_name": step.tool_name,
                    "description": step.description,
                    "approval_message": step.approval_message or f"Approve: {step.description}?",
                    "is_step_gate": is_step_gate,
                })
                return  # Pause — client must call approve/deny then resume

            # --- Skip if still awaiting ---
            if step.status == StepStatus.AWAITING_APPROVAL:
                return

            # --- Skip already completed/skipped ---
            if step.status in (StepStatus.COMPLETED, StepStatus.SKIPPED):
                self.session.current_step_index += 1
                continue

            # --- Execute step ---
            # Yield step_started event BEFORE execution so frontend shows spinner
            step.status = StepStatus.RUNNING
            step.started_at = datetime.now(timezone.utc)
            self.session.log("step_started", step_id=step.step_id, tool=step.tool_name)

            yield _event("step_started", {
                "session_id": self.session.session_id,
                "step_id": step.step_id,
                "step_index": self.session.current_step_index,
                "tool_name": step.tool_name,
                "description": step.description,
            })

            yield await self._execute_step(step)

            if step.status == StepStatus.FAILED:
                self.session.state = WorkflowState.FAILED
                self.session.log("workflow_failed", step_id=step.step_id)
                yield _event("workflow_failed", {
                    "session_id": self.session.session_id,
                    "step_id": step.step_id,
                    "error": step.error,
                })
                return

            self.session.current_step_index += 1

            # --- Step-by-step gate ---
            # After each step completes, mark the next step for approval
            # so the user can review the result before continuing.
            if (
                self.session.step_by_step
                and self.session.current_step_index < len(self.session.steps)
            ):
                next_step = self.session.steps[self.session.current_step_index]
                if (
                    next_step.status == StepStatus.PENDING
                    and not next_step.requires_approval
                ):
                    next_step.requires_approval = True
                    next_step._step_gate = True  # type: ignore[attr-defined]

                    if next_step.tool_name == "generate_report":
                        next_step.approval_message = (
                            "Generate final analysis report?"
                            if self.session.language != "ko"
                            else "최종 분석 보고서를 생성하시겠습니까?"
                        )
                    else:
                        next_step.approval_message = (
                            f"Continue to: {next_step.description}?"
                            if self.session.language != "ko"
                            else f"계속: {next_step.description}?"
                        )

        # All steps done
        self.session.state = WorkflowState.COMPLETED
        self.session.log("workflow_completed")
        yield _event("workflow_completed", {
            "session_id": self.session.session_id,
            "completed_steps": self.session.completed_step_count,
            "total_steps": len(self.session.steps),
            "context_keys": list(self.session.context.keys()),
            "report_markdown": self.session.context.get("report_markdown"),
        })

    # ------------------------------------------------------------------
    # Step execution
    # ------------------------------------------------------------------

    async def _execute_step(self, step: WorkflowStep) -> Dict[str, Any]:
        """Execute a single step and return its SSE event.

        Note: step.status is already RUNNING and step.started_at is set
        by the caller (run loop) before the step_started event is yielded.
        """
        # Resolve template references in params
        resolved_params = _resolve_template_refs(step.params, self.session.context)

        try:
            result = await workflow_tools.execute(
                step.tool_name,
                self.session.context,
                resolved_params,
            )

            step.result = result
            success = result.get("success", True)
            step.status = StepStatus.COMPLETED if success else StepStatus.FAILED
            step.completed_at = datetime.now(timezone.utc)

            if not success:
                step.error = result.get("error", "Unknown error")

            self.session.log(
                "step_completed" if success else "step_failed",
                step_id=step.step_id,
                tool=step.tool_name,
                success=success,
            )

            event_type = "step_completed" if success else "step_failed"
            return _event(event_type, {
                "session_id": self.session.session_id,
                "step_id": step.step_id,
                "step_index": self.session.current_step_index,
                "tool_name": step.tool_name,
                "description": step.description,
                "status": step.status,
                "summary": result.get("summary", ""),
                "error": step.error,
                "elapsed_s": (step.completed_at - step.started_at).total_seconds(),
            })

        except Exception as e:
            step.status = StepStatus.FAILED
            step.error = str(e)
            step.completed_at = datetime.now(timezone.utc)
            self.session.log("step_failed", step_id=step.step_id, error=str(e))
            logger.exception(f"Step {step.step_id} ({step.tool_name}) failed: {e}")

            return _event("step_failed", {
                "session_id": self.session.session_id,
                "step_id": step.step_id,
                "step_index": self.session.current_step_index,
                "tool_name": step.tool_name,
                "status": "failed",
                "error": str(e),
            })

    # ------------------------------------------------------------------
    # Approval / control
    # ------------------------------------------------------------------

    def approve_step(self, step_id: str) -> bool:
        """Mark a step as approved so execution can resume."""
        for step in self.session.steps:
            if step.step_id == step_id and step.status == StepStatus.AWAITING_APPROVAL:
                step.status = StepStatus.PENDING  # Reset to pending → will execute on next run()
                step.requires_approval = False  # Don't re-trigger approval gate
                self.session.state = WorkflowState.EXECUTING
                self.session.log("step_approved", step_id=step_id)
                return True
        return False

    def deny_step(self, step_id: str) -> bool:
        """Skip a denied step and continue with the workflow."""
        for step in self.session.steps:
            if step.step_id == step_id and step.status == StepStatus.AWAITING_APPROVAL:
                step.status = StepStatus.SKIPPED
                step.error = "Denied by user"
                self.session.state = WorkflowState.EXECUTING
                self.session.current_step_index += 1
                self.session.log("step_denied", step_id=step_id)
                return True
        return False

    def cancel(self) -> None:
        """Cancel the workflow."""
        self._cancelled = True
        self.session.state = WorkflowState.CANCELLED
        self.session.log("workflow_cancelled")


def _event(event_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Create an SSE-ready event dict."""
    return {"event": event_type, "data": data}
