from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


ModelName = Literal["v2", "mars-bench"]
JobState = Literal["queued", "processing", "completed", "failed"]


class ClassifyRequest(BaseModel):
    product_id: str
    model: ModelName = "v2"
    include_heatmap: bool = True
    lat: float | None = None
    lon: float | None = None


class TileResult(BaseModel):
    x: int
    y: int
    attention_weight: float = Field(..., ge=0.0, le=1.0)
    lat: float
    lon: float


class PredictionResult(BaseModel):
    top_class: str
    probabilities: dict[str, float] = Field(default_factory=dict)
    confidence: float = Field(..., ge=0.0, le=1.0)


class AgentReasoningStep(BaseModel):
    """A single step from the VLM ReACT agent reasoning chain."""
    step: int
    action: str | None = None
    action_input: dict[str, Any] | None = None
    observation: dict[str, Any] | str | None = None
    thought: str | None = None
    vlm_response: str | None = None
    error: str | None = None
    forced_final: bool = False


class AgentReasoningResult(BaseModel):
    """VLM agent reasoning output — included when confidence < threshold."""
    enabled: bool = False
    mode: str = "fast"
    landform_class: str | None = None
    confidence: float | None = None
    reasoning_chain: list[AgentReasoningStep] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)
    num_steps: int = 0
    error: str | None = None


class ClassifyResult(BaseModel):
    product_id: str
    model_used: str
    prediction: PredictionResult
    tiles: list[TileResult] = Field(default_factory=list)
    heatmap_url: str | None = None
    processing_time_s: float = 0.0
    agent_reasoning: AgentReasoningResult | None = None
    num_tiles: int = 0
    device: str = "cpu"


class JobStatus(BaseModel):
    job_id: str
    status: JobState
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    submitted_at: str
    result: ClassifyResult | None = None
    error: str | None = None
