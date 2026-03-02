from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel


class ThermalPointResponse(BaseModel):
    lat: float
    lon: float
    consistency_score: Optional[float]
    thermal_inertia_tiu: Optional[float]
    interpretation: Literal["ice_cemented", "ambiguous", "dry_fines"]
    depth_range: str
    data_quality: Literal["nominal", "interpolated", "no_data"]


class ThermalRegionResponse(BaseModel):
    bounds: Dict[str, float]
    stats: Dict[str, Any]
    tile_url: str
