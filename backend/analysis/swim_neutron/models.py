from typing import Literal

from pydantic import BaseModel


class PointResponse(BaseModel):
    lat: float
    lon: float
    consistency_score: float | None
    water_equivalent_h: float | None
    depth_range: str
    data_quality: Literal["nominal", "interpolated", "no_data"]
    native_resolution_note: str


class RegionResponse(BaseModel):
    bounds: dict[str, float]
    stats: dict[str, float | str | bool | None]
    tile_url: str
