from typing import Literal

from pydantic import BaseModel, Field


class ConsistencyPointResponse(BaseModel):
    lat: float
    lon: float
    consistency_0_1m: float | None = None
    consistency_1_5m: float | None = None
    consistency_5m_plus: float | None = None
    method_scores: dict[str, float | None] = Field(default_factory=dict)
    mode: Literal["precomputed", "live"]
    depth_to_ice_estimate_m: float | None = None


class ConsistencyRegionResponse(BaseModel):
    bounds: dict[str, float]
    stats_0_1m: dict[str, object]
    stats_1_5m: dict[str, object]
    stats_5m_plus: dict[str, object]
    tile_urls: dict[str, str]


class CustomFusionRequest(BaseModel):
    lat: float
    lon: float
    enabled_methods: list[str] = Field(default_factory=list)
    custom_weights: dict[str, float] | None = None
