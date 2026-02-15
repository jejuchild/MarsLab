"""
Product Proximity / Overlap Search API.

Given a selected product, finds:
1. Overlapping products (geometry intersection) across instruments
2. Nearest products (ranked by centroid distance)

Supports cross-instrument search (e.g. HiRISE → nearby CRISM + SHARAD).
"""

import math
import logging
import asyncio
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from .registry import get_registry
from .mars_regions import MarsRegion, MARS_REGIONS, find_region, list_all_regions
from .ode_client import (
    search_ode_spatial, search_sharad_spatial, search_sharad_highres_spatial,
    Instrument, ODEProduct,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/proximity", tags=["proximity"])

MARS_RADIUS_KM = 3389.5


# =============================================================================
# Models
# =============================================================================

class ProximityResult(BaseModel):
    """A single product found near/overlapping the source product."""
    product_id: str
    instrument: str
    lat: Optional[float] = None
    lon: Optional[float] = None
    distance_km: Optional[float] = None
    overlap: bool = False


class ProximityResponse(BaseModel):
    """Response from proximity search."""
    source_product_id: str
    source_instrument: str
    source_bbox: Optional[Dict[str, float]] = None
    query_mode: str  # "overlap" or "nearest"
    target_instruments: List[str]
    results: List[ProximityResult]
    total_count: int


class RegionInfo(BaseModel):
    """Region information for API response."""
    region_id: str
    display_name: str
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    description: str = ""
    tags: List[str] = []


# =============================================================================
# Helpers
# =============================================================================

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance on Mars in km."""
    lat1_r, lat2_r = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    return MARS_RADIUS_KM * 2 * math.asin(math.sqrt(min(a, 1.0)))


def _bbox_from_geometry(geom: Dict) -> Optional[Dict[str, float]]:
    """Extract bounding box from a GeoJSON geometry."""
    coords = []
    geom_type = geom.get("type", "")

    if geom_type == "Polygon":
        rings = geom.get("coordinates", [])
        if rings:
            coords = rings[0]
    elif geom_type == "LineString":
        coords = geom.get("coordinates", [])
    elif geom_type == "Point":
        c = geom.get("coordinates", [])
        if len(c) >= 2:
            return {"lon_min": c[0], "lon_max": c[0], "lat_min": c[1], "lat_max": c[1]}
    elif geom_type == "MultiPolygon":
        for polygon in geom.get("coordinates", []):
            if polygon:
                coords.extend(polygon[0])

    if not coords:
        return None

    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return {
        "lon_min": min(lons),
        "lon_max": max(lons),
        "lat_min": min(lats),
        "lat_max": max(lats),
    }


def _centroid_from_geometry(geom: Dict) -> Optional[Dict[str, float]]:
    """Get centroid from a GeoJSON geometry."""
    bbox = _bbox_from_geometry(geom)
    if not bbox:
        return None
    return {
        "lat": (bbox["lat_min"] + bbox["lat_max"]) / 2,
        "lon": (bbox["lon_min"] + bbox["lon_max"]) / 2,
    }


def _bboxes_overlap(a: Dict, b: Dict) -> bool:
    """Check if two bounding boxes overlap."""
    if a["lat_max"] < b["lat_min"] or b["lat_max"] < a["lat_min"]:
        return False
    if a["lon_max"] < b["lon_min"] or b["lon_max"] < a["lon_min"]:
        return False
    return True


def _liang_barsky_clip(x1: float, y1: float, x2: float, y2: float,
                       xmin: float, ymin: float, xmax: float, ymax: float) -> bool:
    """Liang-Barsky line-clipping: returns True if segment (x1,y1)-(x2,y2) intersects the rect."""
    dx = x2 - x1
    dy = y2 - y1
    t0, t1 = 0.0, 1.0

    for p, q in [(-dx, x1 - xmin), (dx, xmax - x1),
                 (-dy, y1 - ymin), (dy, ymax - y1)]:
        if p == 0:
            if q < 0:
                return False
        else:
            r = q / p
            if p < 0:
                t0 = max(t0, r)
            else:
                t1 = min(t1, r)
            if t0 > t1:
                return False
    return True


def _line_intersects_bbox(coords: list, bbox: Dict) -> bool:
    """Check if a LineString (list of [lon, lat]) intersects a bounding box.

    Tests each segment of the polyline against the bbox using Liang-Barsky clipping.
    """
    xmin, xmax = bbox["lon_min"], bbox["lon_max"]
    ymin, ymax = bbox["lat_min"], bbox["lat_max"]

    for i in range(len(coords) - 1):
        x1, y1 = coords[i][0], coords[i][1]
        x2, y2 = coords[i + 1][0], coords[i + 1][1]
        # Skip antimeridian-crossing segments (large longitude jump).
        # These are orbital track dateline transitions, not real ground coverage.
        if abs(x2 - x1) > 180:
            continue
        if _liang_barsky_clip(x1, y1, x2, y2, xmin, ymin, xmax, ymax):
            return True
    return False


def _geometries_overlap(geom_a: Optional[Dict], bbox_a: Dict,
                        geom_b: Optional[Dict], bbox_b: Dict) -> bool:
    """Geometry-aware overlap test.

    For LineString geometries (SHARAD), checks actual segment-vs-bbox intersection
    instead of inflated bbox-vs-bbox, which produces many false positives.
    """
    a_is_line = geom_a and geom_a.get("type") == "LineString"
    b_is_line = geom_b and geom_b.get("type") == "LineString"

    if a_is_line and b_is_line:
        # Both lines: quick bbox pre-check then segment-vs-segment (use bbox as proxy)
        return _bboxes_overlap(bbox_a, bbox_b) and (
            _line_intersects_bbox(geom_a["coordinates"], bbox_b) or
            _line_intersects_bbox(geom_b["coordinates"], bbox_a)
        )
    elif a_is_line:
        # Source is line, target is polygon/rect: check line segments vs target bbox
        return _line_intersects_bbox(geom_a["coordinates"], bbox_b)
    elif b_is_line:
        # Source is polygon/rect, target is line: check line segments vs source bbox
        return _line_intersects_bbox(geom_b["coordinates"], bbox_a)
    else:
        # Both polygons/rects: standard bbox overlap
        return _bboxes_overlap(bbox_a, bbox_b)


def _expand_bbox(bbox: Dict, margin_deg: float) -> Dict:
    """Expand a bbox by a margin in degrees."""
    return {
        "lat_min": max(-90, bbox["lat_min"] - margin_deg),
        "lat_max": min(90, bbox["lat_max"] + margin_deg),
        "lon_min": bbox["lon_min"] - margin_deg,
        "lon_max": bbox["lon_max"] + margin_deg,
    }


def _find_product_in_index(product_id: str, instrument_id: str) -> Optional[Dict]:
    """Find a product in the local GeoJSON index and return its feature."""
    registry = get_registry()
    try:
        index = registry.load_index(instrument_id)
    except Exception:
        return None

    if not index:
        return None

    features = index.get("features", [])
    pid_lower = product_id.lower()
    for f in features:
        props = f.get("properties", {})
        fid = (props.get("product_id") or props.get("ProductId") or
               props.get("id") or props.get("PRODUCT_ID") or "")
        if fid.lower() == pid_lower or pid_lower in fid.lower():
            return f

    return None


# =============================================================================
# Search Logic
# =============================================================================

def _search_index_for_overlaps(
    source_bbox: Dict,
    instrument_id: str,
    exclude_product_id: str,
    limit: int = 50,
    source_geom: Optional[Dict] = None,
) -> List[ProximityResult]:
    """Search a single instrument's index for overlapping products.

    Uses geometry-aware intersection: for LineString geometries (SHARAD),
    checks actual line-segment vs bbox instead of inflated bbox-vs-bbox.
    """
    registry = get_registry()
    try:
        index = registry.load_index(instrument_id)
    except Exception:
        return []

    if not index:
        return []

    results = []
    source_center = {
        "lat": (source_bbox["lat_min"] + source_bbox["lat_max"]) / 2,
        "lon": (source_bbox["lon_min"] + source_bbox["lon_max"]) / 2,
    }

    for feature in index.get("features", []):
        props = feature.get("properties", {})
        pid = (props.get("product_id") or props.get("ProductId") or
               props.get("id") or props.get("PRODUCT_ID") or "")

        if not pid or pid.lower() == exclude_product_id.lower():
            continue

        geom = feature.get("geometry")
        if not geom:
            continue

        feat_bbox = _bbox_from_geometry(geom)
        if not feat_bbox:
            continue

        is_overlap = _geometries_overlap(source_geom, source_bbox, geom, feat_bbox)
        if not is_overlap:
            continue

        centroid = _centroid_from_geometry(geom)
        dist = None
        if centroid:
            dist = haversine_km(
                source_center["lat"], source_center["lon"],
                centroid["lat"], centroid["lon"]
            )

        results.append(ProximityResult(
            product_id=pid,
            instrument=instrument_id.upper(),
            lat=centroid["lat"] if centroid else None,
            lon=centroid["lon"] if centroid else None,
            distance_km=round(dist, 2) if dist is not None else None,
            overlap=True,
        ))

    # Sort by distance
    results.sort(key=lambda r: r.distance_km if r.distance_km is not None else float("inf"))
    return results[:limit]


def _search_index_for_nearest(
    source_center: Dict,
    instrument_id: str,
    exclude_product_id: str,
    search_radius_deg: float = 5.0,
    limit: int = 50,
) -> List[ProximityResult]:
    """Search a single instrument's index for nearest products."""
    registry = get_registry()
    try:
        index = registry.load_index(instrument_id)
    except Exception:
        return []

    if not index:
        return []

    search_bbox = {
        "lat_min": max(-90, source_center["lat"] - search_radius_deg),
        "lat_max": min(90, source_center["lat"] + search_radius_deg),
        "lon_min": source_center["lon"] - search_radius_deg,
        "lon_max": source_center["lon"] + search_radius_deg,
    }

    results = []

    for feature in index.get("features", []):
        props = feature.get("properties", {})
        pid = (props.get("product_id") or props.get("ProductId") or
               props.get("id") or props.get("PRODUCT_ID") or "")

        if not pid or pid.lower() == exclude_product_id.lower():
            continue

        geom = feature.get("geometry")
        if not geom:
            continue

        centroid = _centroid_from_geometry(geom)
        if not centroid:
            continue

        # Quick bbox pre-filter
        if (centroid["lat"] < search_bbox["lat_min"] or
            centroid["lat"] > search_bbox["lat_max"] or
            centroid["lon"] < search_bbox["lon_min"] or
            centroid["lon"] > search_bbox["lon_max"]):
            continue

        dist = haversine_km(
            source_center["lat"], source_center["lon"],
            centroid["lat"], centroid["lon"]
        )

        feat_bbox = _bbox_from_geometry(geom)
        is_overlap = False
        if feat_bbox:
            # Check if this product also overlaps the source
            src_bbox = {
                "lat_min": source_center["lat"] - 0.01,
                "lat_max": source_center["lat"] + 0.01,
                "lon_min": source_center["lon"] - 0.01,
                "lon_max": source_center["lon"] + 0.01,
            }
            is_overlap = _bboxes_overlap(src_bbox, feat_bbox)

        results.append(ProximityResult(
            product_id=pid,
            instrument=instrument_id.upper(),
            lat=centroid["lat"],
            lon=centroid["lon"],
            distance_km=round(dist, 2),
            overlap=is_overlap,
        ))

    # Sort by distance
    results.sort(key=lambda r: r.distance_km if r.distance_km is not None else float("inf"))
    return results[:limit]


# =============================================================================
# ODE Search (for instruments available via ODE REST API)
# =============================================================================

# Instruments that can be queried via ODE spatial search
ODE_INSTRUMENTS = {"CRISM", "HIRISE", "SHARAD", "SHARAD_HIGHRES"}
# Instruments only available as local indices
LOCAL_ONLY_INSTRUMENTS = {"CTX", "HIRISE_DTM"}


def _to_ode_lon(lon: float) -> float:
    """Convert -180/180 longitude to ODE's 0-360 convention."""
    return lon + 360 if lon < 0 else lon


async def _search_ode_for_bbox(
    source_bbox: Dict,
    target_instrument: str,
    source_product_id: str,
    limit: int,
) -> List[ProximityResult]:
    """Query ODE for products within a bbox for a single instrument."""
    minlat = source_bbox["lat_min"]
    maxlat = source_bbox["lat_max"]
    westernlon = _to_ode_lon(source_bbox["lon_min"])
    easternlon = _to_ode_lon(source_bbox["lon_max"])

    source_center_lat = (minlat + maxlat) / 2
    source_center_lon = (source_bbox["lon_min"] + source_bbox["lon_max"]) / 2

    ode_products: List[ODEProduct] = []

    try:
        if target_instrument == "CRISM":
            ode_products = await search_ode_spatial(
                minlat, maxlat, westernlon, easternlon,
                instrument=Instrument.CRISM, max_results=limit,
            )
        elif target_instrument == "HIRISE":
            ode_products = await search_ode_spatial(
                minlat, maxlat, westernlon, easternlon,
                instrument=Instrument.HIRISE, max_results=limit,
            )
        elif target_instrument == "SHARAD":
            ode_products = await search_sharad_spatial(
                minlat, maxlat, westernlon, easternlon,
                max_results=limit,
            )
        elif target_instrument == "SHARAD_HIGHRES":
            ode_products = await search_sharad_highres_spatial(
                minlat, maxlat, westernlon, easternlon,
                max_results=limit,
            )
    except Exception as e:
        logger.error(f"ODE proximity search error for {target_instrument}: {e}")
        return []

    results = []
    for p in ode_products:
        if p.product_id.lower() == source_product_id.lower():
            continue

        plat = p.lat
        plon = p.lon
        dist = None
        if plat is not None and plon is not None:
            dist = haversine_km(source_center_lat, source_center_lon, plat, plon)

        results.append(ProximityResult(
            product_id=p.product_id,
            instrument=target_instrument,
            lat=plat,
            lon=plon,
            distance_km=round(dist, 2) if dist is not None else None,
            overlap=True,  # ODE spatial query returns products within the bbox
        ))

    results.sort(key=lambda r: r.distance_km if r.distance_km is not None else float("inf"))
    return results[:limit]


# =============================================================================
# API Endpoints
# =============================================================================

@router.get("/search", response_model=ProximityResponse)
async def proximity_search(
    product_id: str = Query(..., description="Source product ID"),
    instrument: str = Query(..., description="Source product instrument"),
    mode: str = Query("overlap", description="Search mode: overlap or nearest"),
    target_instruments: str = Query("all", description="Comma-separated target instruments, or 'all'"),
    limit: int = Query(50, ge=1, le=200),
    search_radius_deg: float = Query(5.0, ge=0.5, le=30.0, description="Search radius for nearest mode"),
):
    """
    Find products that overlap with or are near a given product.

    Queries ODE for CRISM/HiRISE/SHARAD, falls back to local index for CTX/HIRISE_DTM.
    """
    registry = get_registry()
    instrument_upper = instrument.upper()

    # Resolve target instruments (normalize to uppercase)
    if target_instruments.lower() == "all":
        target_ids = [i.upper() for i in registry.ids()]
    else:
        target_ids = [t.strip().upper() for t in target_instruments.split(",")]

    # Find the source product in local index to get its geometry/bbox
    feature = _find_product_in_index(product_id, instrument_upper)

    source_bbox = None
    source_center = None
    source_geom = None

    if feature and feature.get("geometry"):
        source_geom = feature["geometry"]
        source_bbox = _bbox_from_geometry(source_geom)
        source_center = _centroid_from_geometry(source_geom)

    if not source_bbox and not source_center:
        raise HTTPException(
            status_code=404,
            detail=f"Product {product_id} not found in {instrument_upper} index"
        )

    all_results: List[ProximityResult] = []

    # For "nearest" mode, expand the bbox to create a search area
    search_bbox = source_bbox
    if mode == "nearest" and source_center:
        search_bbox = _expand_bbox({
            "lat_min": source_center["lat"],
            "lat_max": source_center["lat"],
            "lon_min": source_center["lon"],
            "lon_max": source_center["lon"],
        }, search_radius_deg)

    # Search each target instrument
    ode_tasks = []
    local_targets = []

    for target_id in target_ids:
        tid = target_id.upper()
        if tid in ODE_INSTRUMENTS and search_bbox:
            # Query ODE (async, run in parallel)
            ode_tasks.append(_search_ode_for_bbox(
                search_bbox, tid, product_id, limit,
            ))
        elif tid in LOCAL_ONLY_INSTRUMENTS or tid == "CUSTOM":
            local_targets.append(tid)
        else:
            # Unknown instrument — try local index anyway
            local_targets.append(tid)

    # Run ODE queries in parallel
    if ode_tasks:
        ode_results_list = await asyncio.gather(*ode_tasks, return_exceptions=True)
        for ode_results in ode_results_list:
            if isinstance(ode_results, Exception):
                logger.error(f"ODE search task failed: {ode_results}")
                continue
            all_results.extend(ode_results)

    # Search local indices for CTX/HIRISE_DTM
    for tid in local_targets:
        if mode == "overlap" and source_bbox:
            results = _search_index_for_overlaps(
                source_bbox, tid, product_id, limit,
                source_geom=source_geom,
            )
        elif source_center:
            results = _search_index_for_nearest(
                source_center, tid, product_id, search_radius_deg, limit,
            )
        else:
            continue
        all_results.extend(results)

    # Sort all results by distance
    all_results.sort(key=lambda r: r.distance_km if r.distance_km is not None else float("inf"))
    all_results = all_results[:limit]

    return ProximityResponse(
        source_product_id=product_id,
        source_instrument=instrument_upper,
        source_bbox=source_bbox,
        query_mode=mode,
        target_instruments=target_ids,
        results=all_results,
        total_count=len(all_results),
    )


# =============================================================================
# Region API Endpoints
# =============================================================================

@router.get("/regions", response_model=List[RegionInfo])
async def list_regions_api(
    tag: Optional[str] = Query(None, description="Filter by tag"),
    query: Optional[str] = Query(None, description="Search by name"),
):
    """List all Mars regions with bounding boxes."""
    from .mars_regions import find_regions_by_tag, find_all_matching_regions

    if tag:
        regions = find_regions_by_tag(tag)
    elif query:
        regions = find_all_matching_regions(query)
    else:
        regions = list_all_regions()

    return [
        RegionInfo(
            region_id=r.region_id,
            display_name=r.display_name,
            lat_min=r.lat_min,
            lat_max=r.lat_max,
            lon_min=r.lon_min,
            lon_max=r.lon_max,
            description=r.description,
            tags=r.tags,
        )
        for r in regions
    ]


@router.get("/regions/{region_id}", response_model=RegionInfo)
async def get_region_api(region_id: str):
    """Get a specific region by ID."""
    from .mars_regions import get_region_by_id

    region = get_region_by_id(region_id)
    if not region:
        raise HTTPException(status_code=404, detail=f"Region '{region_id}' not found")

    return RegionInfo(
        region_id=region.region_id,
        display_name=region.display_name,
        lat_min=region.lat_min,
        lat_max=region.lat_max,
        lon_min=region.lon_min,
        lon_max=region.lon_max,
        description=region.description,
        tags=region.tags,
    )
