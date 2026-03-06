from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

ModelName = Literal["v3", "v2"]
JobState = Literal["queued", "processing", "completed", "failed"]

V3_CLASSES = ["LDA", "LVF", "CCF", "OTHER", "SCT"]
UNCERTAIN_CLASS = "Uncertain"

# Per-class confidence thresholds (optimized on 11 images, 1563 tiles vs Levy polygons)
# Below threshold → 'Uncertain'. Prevents LVF wipeout from single global threshold.
CLASS_THRESHOLDS: dict[str, float] = {
    "LDA": 0.65,   # global optimal 0.63, per-image mean 0.61±0.10
    "LVF": 0.50,   # global optimal 0.53, per-image mean 0.47±0.20
    "CCF": 0.45,   # model barely classifies CCF (F1=3.5%), low bar
    "OTHER": 0.55, # catch-all class, moderate threshold
    "SCT": 0.55,   # scalloped terrain — calibrate after V5 training
}

class ClassifyRequest(BaseModel):
    product_id: str
    model: ModelName = "v3"
    include_heatmap: bool = True
    use_crf: bool = True  # CRF spatial smoothing on tile grid (+0.6% F1)
    confidence_threshold: float | None = None  # None = use per-class CLASS_THRESHOLDS
    lat: float | None = None
    lon: float | None = None

class TilePrediction(BaseModel):
    x: int
    y: int
    predicted_class: str  # V3_CLASSES or 'Uncertain'
    raw_class: str = ""  # Original prediction before threshold
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: dict[str, float] = Field(default_factory=dict)
    lat: float
    lon: float


class ClassSummary(BaseModel):
    class_name: str
    tile_count: int
    percentage: float
    mean_confidence: float


class AgentReasoningStep(BaseModel):
    step: int
    action: str | None = None
    action_input: dict[str, Any] | None = None
    observation: dict[str, Any] | str | None = None
    thought: str | None = None
    vlm_response: str | None = None
    error: str | None = None
    forced_final: bool = False


class AgentReasoningResult(BaseModel):
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
    tile_predictions: list[TilePrediction] = Field(default_factory=list)
    class_summary: list[ClassSummary] = Field(default_factory=list)
    dominant_class: str = "OTHER"
    dominant_confidence: float = 0.0
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
