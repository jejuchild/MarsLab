from typing import Literal

from pydantic import BaseModel


class PointResponse(BaseModel):
    lat: float
    lon: float
    consistency_score: float | None
    surface_power_excess_db: float | None
    nearest_track_id: str | None
    depth_range: str
    data_quality: Literal["nominal", "interpolated", "no_data"]


class RegionResponse(BaseModel):
    bounds: dict[str, float]
    stats: dict[str, float | str | bool | None]
    tile_url: str
