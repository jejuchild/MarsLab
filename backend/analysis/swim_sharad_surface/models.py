from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel


class SharadSurfacePointResponse(BaseModel):
    lat: float
    lon: float
    consistency_score: Optional[float]
    surface_power_excess_db: Optional[float]
    nearest_track_id: Optional[str]
    depth_range: str
    data_quality: Literal["nominal", "interpolated", "no_data"]


class SharadSurfaceRegionResponse(BaseModel):
    bounds: Dict[str, float]
    stats: Dict[str, Any]
    tile_url: str
