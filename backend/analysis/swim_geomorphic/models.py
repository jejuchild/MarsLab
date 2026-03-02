from typing import Literal

from pydantic import BaseModel


class GeomorphicPointResponse(BaseModel):
    lat: float
    lon: float
    consistency_shallow: float | None
    consistency_intermediate: float | None
    consistency_deep: float | None
    landforms_detected: list[str]
    hirise_classification: dict[str, str | float | None] | None


class GeomorphicRegionResponse(BaseModel):
    bounds: dict[str, float]
    stats: dict[str, float]
    depth: Literal["0-1m", "1-5m", "5m-plus"]
    tile_url: str
