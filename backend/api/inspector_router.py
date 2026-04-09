"""
Inspector at-point aggregator.

`GET /api/inspector/at-point?lat=&lon=&radius_km=`

Given a coordinate, returns the products from each of the 4 lanes
(SHARAD, CRISM, HIRISE, CTX) within a small radius. Drives the new
4-lane Inspector UX (Phase 3 of the MarsLab refactoring).

Implementation notes:
- Reads from the in-memory `_geojson_cache` populated at app startup,
  so this endpoint is fast (<50ms typical).
- Each lane is computed in parallel via `asyncio.to_thread`.
- HiRISE/HiRISE-DTM are merged into a single `HIRISE` lane with a
  `variant` field. Same for SHARAD/SHARAD-HIGHRES → `SHARAD`,
  CRISM/CRISM-TRR3 → `CRISM`. CTX/CTX-MOSAIC → `CTX`.
"""

from __future__ import annotations

import asyncio
import math
from typing import Any, Iterable, Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

router = APIRouter(prefix="/api/inspector", tags=["Inspector"])

# ======================================================
# Constants
# ======================================================

MARS_RADIUS_KM = 3389.5
DEFAULT_RADIUS_KM = 5.0

Lane = Literal["SHARAD", "CRISM", "HIRISE", "CTX"]

# Maps the in-memory cache key → (Lane, variant) tuple
_CACHE_TO_LANE: dict[str, tuple[Lane, str]] = {
    "sharad": ("SHARAD", "standard"),
    "sharad_highres": ("SHARAD", "highres"),
    "crism": ("CRISM", "standard"),
    "crism_trr3": ("CRISM", "trr3"),
    "hirise": ("HIRISE", "image"),
    "hirise_dtm": ("HIRISE", "dtm"),
    "ctx": ("CTX", "image"),
    "ctx_mosaic": ("CTX", "mosaic"),
}

# ======================================================
# Models
# ======================================================


class LaneProduct(BaseModel):
    product_id: str
    title: str | None = None
    lat: float | None = None
    lon: float | None = None
    variant: str
    distance_km: float | None = None


class AtPointResponse(BaseModel):
    lat: float
    lon: float
    radius_km: float
    lanes: dict[Lane, list[LaneProduct]]
    counts: dict[Lane, int]


# ======================================================
# Helpers
# ======================================================


def _normalize_lon_180(lon: float) -> float:
    lon = lon % 360.0
    if lon > 180.0:
        lon -= 360.0
    return lon


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance on Mars in km."""
    lat1_r = math.radians(lat1)
    lat2_r = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    return 2 * MARS_RADIUS_KM * math.asin(min(1.0, math.sqrt(a)))


def _feature_center(feature: dict[str, Any]) -> tuple[float, float] | None:
    """Compute (lat, lon) center for a feature.

    Tries `properties.{west,east,south,north}` first (DTM/CTX), then geometry centroid.
    """
    props = feature.get("properties") or {}
    west = props.get("west")
    east = props.get("east")
    south = props.get("south")
    north = props.get("north")
    if (
        isinstance(west, (int, float))
        and isinstance(east, (int, float))
        and isinstance(south, (int, float))
        and isinstance(north, (int, float))
    ):
        return ((south + north) / 2.0, (west + east) / 2.0)

    geom = feature.get("geometry") or {}
    coords = geom.get("coordinates")
    gtype = geom.get("type")
    if not coords:
        return None

    try:
        if gtype == "Point" and isinstance(coords, list) and len(coords) >= 2:
            return (float(coords[1]), float(coords[0]))
        if gtype == "Polygon" and isinstance(coords, list) and coords:
            ring = coords[0]
            if isinstance(ring, list) and ring:
                lons = [float(c[0]) for c in ring if isinstance(c, list) and len(c) >= 2]
                lats = [float(c[1]) for c in ring if isinstance(c, list) and len(c) >= 2]
                if lons and lats:
                    return (sum(lats) / len(lats), sum(lons) / len(lons))
        if gtype == "LineString" and isinstance(coords, list) and coords:
            mid = coords[len(coords) // 2]
            if isinstance(mid, list) and len(mid) >= 2:
                return (float(mid[1]), float(mid[0]))
    except (TypeError, ValueError):
        return None

    return None


def _bbox_intersects(
    feature_bounds: tuple[float, float, float, float],
    point_lat: float,
    point_lon: float,
    radius_km: float,
) -> bool:
    """Quick rejection: does feature's bounding box come within radius of point?"""
    fwest, fsouth, feast, fnorth = feature_bounds
    # Crude angular padding (1 deg lat ≈ 59 km on Mars)
    pad_deg = radius_km / 59.0
    if point_lat < fsouth - pad_deg or point_lat > fnorth + pad_deg:
        return False
    # Longitude wrap not handled in fast path; fine for small radius
    if point_lon < fwest - pad_deg or point_lon > feast + pad_deg:
        return False
    return True


def _feature_bounds(feature: dict[str, Any]) -> tuple[float, float, float, float] | None:
    """Compute (west, south, east, north) for a feature."""
    props = feature.get("properties") or {}
    west = props.get("west")
    east = props.get("east")
    south = props.get("south")
    north = props.get("north")
    if all(isinstance(v, (int, float)) for v in (west, east, south, north)):
        return (float(west), float(south), float(east), float(north))

    geom = feature.get("geometry") or {}
    coords = geom.get("coordinates")
    gtype = geom.get("type")
    if not coords:
        return None

    try:
        if gtype == "Point" and isinstance(coords, list) and len(coords) >= 2:
            lon, lat = float(coords[0]), float(coords[1])
            return (lon, lat, lon, lat)
        if gtype == "Polygon" and isinstance(coords, list) and coords:
            ring = coords[0]
            lons = [float(c[0]) for c in ring if isinstance(c, list) and len(c) >= 2]
            lats = [float(c[1]) for c in ring if isinstance(c, list) and len(c) >= 2]
            if lons and lats:
                return (min(lons), min(lats), max(lons), max(lats))
        if gtype == "LineString" and isinstance(coords, list) and coords:
            lons = [float(c[0]) for c in coords if isinstance(c, list) and len(c) >= 2]
            lats = [float(c[1]) for c in coords if isinstance(c, list) and len(c) >= 2]
            if lons and lats:
                return (min(lons), min(lats), max(lons), max(lats))
    except (TypeError, ValueError):
        return None

    return None


# ======================================================
# Per-cache search (synchronous, runs in to_thread)
# ======================================================


def _search_cache_at_point(
    cache_key: str,
    features: Iterable[dict[str, Any]],
    lat: float,
    lon: float,
    radius_km: float,
    max_results: int = 25,
) -> list[LaneProduct]:
    """Find features whose footprint comes within `radius_km` of the point."""
    out: list[LaneProduct] = []
    lane_info = _CACHE_TO_LANE.get(cache_key)
    if not lane_info:
        return out
    _lane, variant = lane_info

    for feature in features:
        bounds = _feature_bounds(feature)
        if bounds is None:
            continue
        if not _bbox_intersects(bounds, lat, lon, radius_km):
            continue

        center = _feature_center(feature)
        if center is None:
            continue
        clat, clon = center
        dist = _haversine_km(lat, lon, clat, clon)
        if dist > radius_km:
            continue

        props = feature.get("properties") or {}
        product_id = props.get("product_id") or props.get("id")
        if not product_id:
            continue

        out.append(
            LaneProduct(
                product_id=str(product_id),
                title=props.get("title") or props.get("name"),
                lat=clat,
                lon=clon,
                variant=variant,
                distance_km=round(dist, 2),
            )
        )

    out.sort(key=lambda p: p.distance_km if p.distance_km is not None else 1e9)
    return out[:max_results]


# ======================================================
# Endpoint
# ======================================================


@router.get("/at-point", response_model=AtPointResponse)
async def at_point(
    lat: float = Query(..., ge=-90, le=90, description="Latitude in degrees (-90..90)"),
    lon: float = Query(..., ge=-360, le=360, description="Longitude in degrees (any 0/180-based)"),
    radius_km: float = Query(DEFAULT_RADIUS_KM, gt=0, le=200, description="Search radius in km (max 200)"),
):
    """Aggregate products from 4 lanes at a single point."""
    # Lazy import to avoid a circular reference at app startup time.
    try:
        from app import _geojson_cache  # type: ignore
    except ImportError:
        raise HTTPException(status_code=500, detail="GeoJSON cache unavailable")

    norm_lon = _normalize_lon_180(lon)

    async def _search(cache_key: str) -> tuple[Lane, list[LaneProduct]]:
        lane_info = _CACHE_TO_LANE.get(cache_key)
        if lane_info is None:
            return ("CTX", [])
        lane, _variant = lane_info
        cache = _geojson_cache.get(cache_key)
        if not cache:
            return (lane, [])
        features = cache.get("features", [])
        results = await asyncio.to_thread(
            _search_cache_at_point, cache_key, features, lat, norm_lon, radius_km
        )
        return (lane, results)

    cache_keys = list(_CACHE_TO_LANE.keys())
    pairs = await asyncio.gather(*[_search(k) for k in cache_keys])

    lanes: dict[Lane, list[LaneProduct]] = {"SHARAD": [], "CRISM": [], "HIRISE": [], "CTX": []}
    for lane, results in pairs:
        lanes[lane].extend(results)

    # Re-sort each lane by distance after merging variants
    for lane in lanes:
        lanes[lane].sort(key=lambda p: p.distance_km if p.distance_km is not None else 1e9)
        lanes[lane] = lanes[lane][:25]

    return AtPointResponse(
        lat=lat,
        lon=norm_lon,
        radius_km=radius_km,
        lanes=lanes,
        counts={lane: len(products) for lane, products in lanes.items()},
    )
