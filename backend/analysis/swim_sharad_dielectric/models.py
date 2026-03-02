from typing import Literal

from pydantic import BaseModel


class DielectricPointResponse(BaseModel):
    lat: float
    lon: float
    consistency_score_1_5m: float | None
    consistency_score_5m_plus: float | None
    estimated_epsilon: float | None
    depth_ranges: list[str]
    nearest_track_id: str | None


class DielectricRegionResponse(BaseModel):
    bounds: dict[str, float]
    stats: dict[str, float]
    depth: Literal["1-5m", "5m-plus"]
    tile_url: str
