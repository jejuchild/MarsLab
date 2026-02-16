"""
Terrain Features In-View Filter.

POST /api/terrain/features_in_view — Filter detected features by map viewport bbox.
"""

import logging
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/terrain", tags=["Terrain Features"])


class BoundsModel(BaseModel):
    west: float = Field(..., description="Western longitude (-180 to 360)")
    south: float = Field(..., description="Southern latitude (-90 to 90)")
    east: float = Field(..., description="Eastern longitude (-180 to 360)")
    north: float = Field(..., description="Northern latitude (-90 to 90)")


class FeaturesInViewRequest(BaseModel):
    features: list[dict[str, Any]] = Field(..., description="Detected features from MOLA scan")
    bounds: BoundsModel = Field(..., description="Map viewport bounding box")


@router.post("/features_in_view")
async def features_in_view(req: FeaturesInViewRequest):
    """Filter detected features to those visible within the map viewport."""
    west = req.bounds.west
    east = req.bounds.east
    south = req.bounds.south
    north = req.bounds.north

    visible = []
    for f in req.features:
        lat = f.get("lat", 0)
        lon = f.get("lon", 0)

        # Latitude check
        if lat < south or lat > north:
            continue

        # Longitude check (handle wrap-around)
        if west <= east:
            if lon < west or lon > east:
                continue
        else:
            # Viewport crosses antimeridian
            if lon < west and lon > east:
                continue

        visible.append(f)

    return JSONResponse(content={
        "visible": visible,
        "total": len(req.features),
        "filtered": len(visible),
    })
