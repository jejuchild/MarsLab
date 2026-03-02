from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ConsistencyPointResponse(BaseModel):
    lat: float
    lon: float
    consistency_0_1m: Optional[float] = None
    consistency_1_5m: Optional[float] = None
    consistency_5m_plus: Optional[float] = None
    method_scores: Dict[str, Optional[float]] = Field(default_factory=dict)
    mode: Literal["precomputed", "live"]
    depth_to_ice_estimate_m: Optional[float] = None


class ConsistencyRegionResponse(BaseModel):
    bounds: Dict[str, float]
    stats_0_1m: Dict[str, Any]
    stats_1_5m: Dict[str, Any]
    stats_5m_plus: Dict[str, Any]
    tile_urls: Dict[str, str]


class CustomFusionRequest(BaseModel):
    lat: float
    lon: float
    enabled_methods: List[str] = Field(default_factory=list)
    custom_weights: Optional[Dict[str, float]] = None
