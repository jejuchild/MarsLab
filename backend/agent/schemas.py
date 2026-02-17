"""
Pydantic schemas for the MarsLab Workflow Research Assistant.

Defines the data structures for sessions, steps, tool specs,
and the workflow state machine.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class StepStatus(str, Enum):
    PENDING = "pending"
    AWAITING_APPROVAL = "awaiting_approval"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowState(str, Enum):
    PLANNING = "planning"
    AWAITING_APPROVAL = "awaiting_approval"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ---------------------------------------------------------------------------
# Workflow Step
# ---------------------------------------------------------------------------

class WorkflowStep(BaseModel):
    """A single step in a workflow plan."""

    step_id: str
    tool_name: str
    description: str  # Human-readable, may be Korean or English
    params: Dict[str, Any] = Field(default_factory=dict)

    status: StepStatus = StepStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    requires_approval: bool = False
    approval_message: Optional[str] = None  # e.g. "Download 5 files (~200 MB)?"

    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    class Config:
        use_enum_values = True


# ---------------------------------------------------------------------------
# Workflow Session
# ---------------------------------------------------------------------------

class WorkflowSession(BaseModel):
    """Full session state for a workflow execution."""

    session_id: str
    user_goal: str  # Original natural-language request
    language: str = "en"  # "en" or "ko"

    state: WorkflowState = WorkflowState.PLANNING
    steps: List[WorkflowStep] = Field(default_factory=list)
    current_step_index: int = 0

    # Shared context: tool outputs feed into later steps via {{context.key}}
    context: Dict[str, Any] = Field(default_factory=dict)

    # LLM-generated explanation of the plan
    plan_narrative: Optional[str] = None

    # Step-by-step mode: pause after each step for user confirmation
    step_by_step: bool = True

    # Which agent template was used (None = generic planner)
    agent_type: Optional[str] = None  # "lda_sharad" | "terrace_epsilon" | "ice_search"

    # Artifacts produced during execution (file paths, figure base64, etc.)
    artifacts: List[Dict[str, Any]] = Field(default_factory=list)

    # Append-only structured logs
    logs: List[Dict[str, Any]] = Field(default_factory=list)

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Config:
        use_enum_values = True

    # -- helpers --

    @property
    def current_step(self) -> Optional[WorkflowStep]:
        if 0 <= self.current_step_index < len(self.steps):
            return self.steps[self.current_step_index]
        return None

    @property
    def completed_step_count(self) -> int:
        return sum(1 for s in self.steps if s.status == StepStatus.COMPLETED)

    @property
    def is_done(self) -> bool:
        return self.state in (
            WorkflowState.COMPLETED,
            WorkflowState.FAILED,
            WorkflowState.CANCELLED,
        )

    def log(self, event: str, **kwargs: Any) -> None:
        self.logs.append({
            "event": event,
            "ts": datetime.now(timezone.utc).isoformat(),
            **kwargs,
        })
        self.updated_at = datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Tool Spec (for planner LLM)
# ---------------------------------------------------------------------------

class ToolSpec(BaseModel):
    """Describes a tool available to the workflow planner."""

    name: str
    description: str
    params_schema: Dict[str, Any] = Field(default_factory=dict)
    requires_approval: bool = False
    approval_template: Optional[str] = None  # e.g. "Download {count} files (~{size_mb} MB)?"


# ---------------------------------------------------------------------------
# SSE Event envelope
# ---------------------------------------------------------------------------

class WorkflowEvent(BaseModel):
    """SSE event sent to frontend."""

    event: str  # plan_created | step_started | step_completed | step_failed
                # | approval_required | workflow_completed | workflow_failed
    data: Dict[str, Any] = Field(default_factory=dict)
