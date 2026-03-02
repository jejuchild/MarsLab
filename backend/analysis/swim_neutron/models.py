from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel


class NeutronPointResponse(BaseModel):
    lat: float
    lon: float
    consistency_score: Optional[float]
    water_equivalent_h: Optional[float]
    depth_range: str
    data_quality: Literal["nominal", "interpolated", "no_data"]
    native_resolution_note: str


class NeutronRegionResponse(BaseModel):
    bounds: Dict[str, float]
    stats: Dict[str, Any]
    tile_url: str
