from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


ModelName = Literal["v2", "mars-bench"]
JobState = Literal["queued", "processing", "completed", "failed"]


class ClassifyRequest(BaseModel):
    product_id: str
    model: ModelName
    include_heatmap: bool = True


class TileResult(BaseModel):
    x: int
    y: int
    attention_weight: float = Field(..., ge=0.0, le=1.0)
    lat: float
    lon: float


class PredictionResult(BaseModel):
    top_class: str
    probabilities: Dict[str, float] = Field(default_factory=dict)
    confidence: float = Field(..., ge=0.0, le=1.0)


class ClassifyResult(BaseModel):
    product_id: str
    model_used: str
    prediction: PredictionResult
    tiles: List[TileResult] = Field(default_factory=list)
    heatmap_url: Optional[str] = None


class JobStatus(BaseModel):
    job_id: str
    status: JobState
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    submitted_at: str
    result: Optional[ClassifyResult] = None
    error: Optional[str] = None
