# api/footprints_router.py
"""
Viewport-based footprint API with LOD support and geometry simplification.

This module provides spatial queries for footprint data with:
- Bounding box filtering
- Level-of-detail (LOD) support: none, point, poly
- Geometry simplification using Douglas-Peucker algorithm
- Antimeridian crossing support
- Result limiting with truncation metadata
"""

from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional, Literal
from functools import lru_cache
import json
import os

# Import instrument registry
from .registry import get_registry

# Try to import shapely for geometry operations
try:
    from shapely.geometry import shape, box, Point, mapping
    from shapely.ops import transform
    from shapely.validation import make_valid
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False
    print("[WARNING] shapely not installed - geometry simplification disabled")

router = APIRouter()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# LOD types
LODType = Literal["none", "point", "poly"]
SimplifyLevel = Literal["low", "mid", "high"]

# Simplification tolerances in degrees (approximate)
SIMPLIFY_TOLERANCES = {
    "low": 0.01,    # ~1km at equator
    "mid": 0.005,   # ~500m at equator
    "high": 0.001,  # ~100m at equator
}

# Default limits
DEFAULT_LIMIT = 2000
MAX_LIMIT = 5000


@lru_cache(maxsize=8)
def load_geojson_index(instrument: str) -> dict:
    """Load and cache GeoJSON index for an instrument using the registry.

    Tries the app-level preloaded cache first for instant access,
    falls back to disk read if not preloaded.
    """
    # Try app-level preloaded cache first (loaded at startup)
    try:
        from app import _geojson_cache
        cache_key = instrument.lower()
        if cache_key in _geojson_cache:
            return _geojson_cache[cache_key]
    except ImportError:
        pass

    registry = get_registry()
    config = registry.get(instrument)

    if not config:
        raise HTTPException(status_code=400, detail=f"Unknown instrument: {instrument}")

    index_path = config.get_index_path(registry.base_dir)

    if not index_path.exists():
        raise HTTPException(status_code=404, detail=f"{instrument} index.geojson not found at {index_path}")

    with open(index_path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_lon(lon: float) -> float:
    """Normalize longitude to -180 to 180 range."""
    while lon > 180:
        lon -= 360
    while lon < -180:
        lon += 360
    return lon


def feature_intersects_bbox(feature: dict, min_lon: float, min_lat: float,
                            max_lon: float, max_lat: float,
                            crosses_antimeridian: bool) -> bool:
    """
    Check if a feature intersects the bounding box.

    All longitudes are in -180 to 180 range.
    If crosses_antimeridian is True, the bbox wraps: [min_lon, 180] U [-180, max_lon]
    """
    geom = feature.get("geometry")
    if not geom:
        return False

    geom_type = geom.get("type")
    coords = geom.get("coordinates")

    if not coords:
        return False

    # Extract all coordinate points from geometry
    if geom_type == "Polygon":
        ring = coords[0] if coords else []
        lons = [normalize_lon(c[0]) for c in ring]
        lats = [c[1] for c in ring]
    elif geom_type == "LineString":
        lons = [normalize_lon(c[0]) for c in coords]
        lats = [c[1] for c in coords]
    elif geom_type == "Point":
        lons = [normalize_lon(coords[0])]
        lats = [coords[1]]
    else:
        return False

    if not lons or not lats:
        return False

    feat_min_lon = min(lons)
    feat_max_lon = max(lons)
    feat_min_lat = min(lats)
    feat_max_lat = max(lats)

    # Latitude check (simple rectangle overlap)
    if feat_max_lat < min_lat or feat_min_lat > max_lat:
        return False

    # Check if feature itself crosses antimeridian (large longitude span)
    feat_crosses_antimeridian = (feat_max_lon - feat_min_lon) > 180

    # Longitude check
    if crosses_antimeridian:
        # Query bbox crosses antimeridian: covers [min_lon, 180] U [-180, max_lon]
        # A feature intersects if ANY of its points are in either segment

        if feat_crosses_antimeridian:
            # Both cross antimeridian - almost certainly intersect
            return True

        # Check if feature is in the positive segment [min_lon, 180]
        in_positive_segment = feat_max_lon >= min_lon and feat_min_lon <= 180

        # Check if feature is in the negative segment [-180, max_lon]
        in_negative_segment = feat_min_lon <= max_lon and feat_max_lon >= -180

        return in_positive_segment or in_negative_segment
    else:
        # Normal case: standard rectangle overlap
        if feat_crosses_antimeridian:
            # Feature crosses antimeridian but query doesn't
            # Feature covers [feat_min_lon, 180] U [-180, feat_max_lon]
            # Check if query bbox overlaps either segment
            overlaps_positive = min_lon <= 180 and max_lon >= feat_min_lon
            overlaps_negative = min_lon <= feat_max_lon and max_lon >= -180
            # Actually for normal query, just check if any overlap
            return not (max_lon < feat_min_lon and min_lon > feat_max_lon)
        else:
            # Simple overlap check
            if feat_max_lon < min_lon or feat_min_lon > max_lon:
                return False

    return True


def compute_centroid(feature: dict) -> Optional[dict]:
    """Compute centroid of a feature geometry."""
    geom = feature.get("geometry")
    if not geom:
        return None

    geom_type = geom.get("type")
    coords = geom.get("coordinates")

    if not coords:
        return None

    if geom_type == "Point":
        return {"type": "Point", "coordinates": [normalize_lon(coords[0]), coords[1]]}
    elif geom_type == "Polygon":
        ring = coords[0] if coords else []
        if not ring:
            return None
        lons = [c[0] for c in ring]
        lats = [c[1] for c in ring]
        center_lon = sum(lons) / len(lons)
        center_lat = sum(lats) / len(lats)
        return {"type": "Point", "coordinates": [normalize_lon(center_lon), center_lat]}
    elif geom_type == "LineString":
        if not coords:
            return None
        mid_idx = len(coords) // 2
        mid_coord = coords[mid_idx]
        return {"type": "Point", "coordinates": [normalize_lon(mid_coord[0]), mid_coord[1]]}

    return None


def normalize_geometry_coords(geom: dict) -> dict:
    """Normalize all longitude values in a geometry to -180..180 range."""
    geom_type = geom.get("type")
    coords = geom.get("coordinates")
    if not coords:
        return geom

    if geom_type == "Point":
        return {"type": "Point", "coordinates": [normalize_lon(coords[0]), coords[1]]}
    elif geom_type == "LineString":
        return {"type": "LineString", "coordinates": [[normalize_lon(c[0]), c[1]] for c in coords]}
    elif geom_type == "Polygon":
        new_rings = []
        for ring in coords:
            new_rings.append([[normalize_lon(c[0]), c[1]] for c in ring])
        return {"type": "Polygon", "coordinates": new_rings}
    return geom


def simplify_geometry(geom: dict, tolerance: float) -> dict:
    """Simplify geometry using Douglas-Peucker algorithm."""
    if not SHAPELY_AVAILABLE:
        return geom

    try:
        shapely_geom = shape(geom)

        # Make valid if needed
        if not shapely_geom.is_valid:
            shapely_geom = make_valid(shapely_geom)

        # Simplify with topology preservation
        simplified = shapely_geom.simplify(tolerance, preserve_topology=True)

        # Ensure result is valid
        if not simplified.is_valid:
            simplified = make_valid(simplified)

        # Convert back to GeoJSON
        result = mapping(simplified)

        # Normalize coordinates
        if result.get("type") == "Polygon":
            new_coords = []
            for ring in result.get("coordinates", []):
                new_ring = [[normalize_lon(c[0]), c[1]] for c in ring]
                new_coords.append(new_ring)
            result["coordinates"] = new_coords

        return result
    except Exception as e:
        # If simplification fails, return original
        print(f"[WARNING] Geometry simplification failed: {e}")
        return geom


# Server-side LOD enforcement thresholds (in km)
LOD_THRESHOLDS_KM = {
    "FAR": 15000,   # > 15,000 km: no footprints
    "MID": 5000,    # 5,000-15,000 km: points only
    # < 5,000 km: polygons allowed
}


@router.get("/api/footprints")
async def get_footprints(
    instrument: str = Query(..., description="Instrument: CRISM, HIRISE, or SHARAD"),
    bbox: str = Query(..., description="Bounding box: minLon,minLat,maxLon,maxLat"),
    lod: LODType = Query("poly", description="Level of detail: none, point, or poly"),
    simplify: Optional[SimplifyLevel] = Query(None, description="Simplification level: low, mid, high"),
    limit: int = Query(DEFAULT_LIMIT, ge=1, le=MAX_LIMIT, description="Maximum features to return"),
    camera_height_km: Optional[float] = Query(None, description="Camera height in km for server-side LOD enforcement"),
):
    """
    Get footprints within a bounding box.

    Returns a GeoJSON FeatureCollection with metadata:
    - truncated: boolean indicating if results were limited
    - returned: number of features returned
    - total_estimate: estimated total features in bbox (if truncated)
    - lod: the LOD level used (may be downgraded by server)
    - simplify: the simplification level used (if any)
    - lod_enforced: true if server downgraded the LOD

    Server-side LOD enforcement:
    - If camera_height_km > 2000: forces lod=none
    - If camera_height_km > 500 and lod=poly: downgrades to lod=point
    """
    lod_enforced = False
    original_lod = lod

    # Server-side LOD enforcement based on camera height
    if camera_height_km is not None:
        if camera_height_km > LOD_THRESHOLDS_KM["FAR"]:
            # Too zoomed out - no footprints
            if lod != "none":
                lod = "none"
                lod_enforced = True
        elif camera_height_km > LOD_THRESHOLDS_KM["MID"]:
            # Mid zoom - points only
            if lod == "poly":
                lod = "point"
                lod_enforced = True
    # Parse bbox
    try:
        parts = [float(x.strip()) for x in bbox.split(",")]
        if len(parts) != 4:
            raise ValueError("Expected 4 values")
        min_lon, min_lat, max_lon, max_lat = parts
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid bbox format: {e}")

    # Validate coordinates
    if min_lat < -90 or max_lat > 90:
        raise HTTPException(status_code=400, detail="Latitude must be between -90 and 90")

    # Normalize longitudes to -180 to 180
    min_lon = normalize_lon(min_lon)
    max_lon = normalize_lon(max_lon)

    # Check for antimeridian crossing
    crosses_antimeridian = min_lon > max_lon

    # If LOD is none, return empty result immediately
    if lod == "none":
        return JSONResponse(content={
            "type": "FeatureCollection",
            "features": [],
            "metadata": {
                "truncated": False,
                "returned": 0,
                "total_estimate": 0,
                "lod": lod,
                "original_lod": original_lod,
                "lod_enforced": lod_enforced,
                "simplify": simplify,
                "bbox": [min_lon, min_lat, max_lon, max_lat],
                "instrument": instrument.upper(),
                "reason": "Zoom in to see footprints" if lod_enforced else None,
            }
        })

    # Load index
    geojson = load_geojson_index(instrument)
    all_features = geojson.get("features", [])

    # Filter features by bbox
    is_crism = instrument.upper() in ("CRISM", "CRISM_TRR3")
    matching_features = []
    for feature in all_features:
        if feature_intersects_bbox(feature, min_lon, min_lat, max_lon, max_lat, crosses_antimeridian):
            # For CRISM layers, only show FRT (Full Resolution Targeted) products
            if is_crism:
                pid = feature.get("properties", {}).get("product_id", "")
                if pid and not pid.lower().startswith("frt"):
                    continue
            matching_features.append(feature)

    total_matching = len(matching_features)
    truncated = total_matching > limit

    # Limit results
    if truncated:
        matching_features = matching_features[:limit]

    # Process features based on LOD
    result_features = []

    for feature in matching_features:
        props = feature.get("properties", {}).copy()

        if lod == "point":
            # Return centroids only
            centroid = compute_centroid(feature)
            if centroid:
                result_features.append({
                    "type": "Feature",
                    "properties": props,
                    "geometry": centroid,
                })
        else:  # lod == "poly"
            geom = feature.get("geometry")
            if geom:
                # Normalize coordinates to -180..180 range
                geom = normalize_geometry_coords(geom)
                # Apply simplification if requested
                if simplify and SHAPELY_AVAILABLE:
                    tolerance = SIMPLIFY_TOLERANCES.get(simplify, 0.005)
                    geom = simplify_geometry(geom, tolerance)

                result_features.append({
                    "type": "Feature",
                    "properties": props,
                    "geometry": geom,
                })

    return JSONResponse(
        content={
            "type": "FeatureCollection",
            "features": result_features,
            "metadata": {
                "truncated": truncated,
                "returned": len(result_features),
                "total_estimate": total_matching,
                "lod": lod,
                "original_lod": original_lod,
                "lod_enforced": lod_enforced,
                "simplify": simplify,
                "bbox": [min_lon, min_lat, max_lon, max_lat],
                "instrument": instrument.upper(),
            }
        },
        headers={"Cache-Control": "public, max-age=300"},
    )


@router.get("/api/footprints/stats")
async def get_footprints_stats(instrument: str = Query(..., description="Instrument: CRISM, HIRISE, or SHARAD")):
    """Get statistics about footprint data for an instrument."""
    geojson = load_geojson_index(instrument)
    features = geojson.get("features", [])

    # Compute bounds
    all_lons = []
    all_lats = []

    for feature in features:
        geom = feature.get("geometry")
        if not geom:
            continue
        coords = geom.get("coordinates", [])
        geom_type = geom.get("type")

        if geom_type == "Polygon" and coords:
            ring = coords[0]
            all_lons.extend([c[0] for c in ring])
            all_lats.extend([c[1] for c in ring])
        elif geom_type == "LineString":
            all_lons.extend([c[0] for c in coords])
            all_lats.extend([c[1] for c in coords])
        elif geom_type == "Point":
            all_lons.append(coords[0])
            all_lats.append(coords[1])

    bounds = None
    if all_lons and all_lats:
        bounds = {
            "min_lon": min(all_lons),
            "max_lon": max(all_lons),
            "min_lat": min(all_lats),
            "max_lat": max(all_lats),
        }

    return JSONResponse(content={
        "instrument": instrument.upper(),
        "total_features": len(features),
        "bounds": bounds,
        "shapely_available": SHAPELY_AVAILABLE,
    })
