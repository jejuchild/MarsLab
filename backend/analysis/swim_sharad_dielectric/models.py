from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel


class DielectricPointResponse(BaseModel):
    lat: float
    lon: float
    consistency_score_1_5m: Optional[float]
    consistency_score_5m_plus: Optional[float]
    estimated_epsilon: Optional[float]
    depth_ranges: List[str]
    nearest_track_id: Optional[str]


class DielectricRegionResponse(BaseModel):
    bounds: Dict[str, float]
    stats: Dict[str, Any]
    depth: Literal["1-5m", "5m-plus"]
    tile_url: str
