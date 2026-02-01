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


@lru_cache(maxsize=4)
def load_geojson_index(instrument: str) -> dict:
    """Load and cache GeoJSON index for an instrument."""
    instrument_lower = instrument.lower()

    if instrument_lower == "crism":
        path = os.path.join(BASE_DIR, "crism_data", "index.geojson")
    elif instrument_lower == "hirise":
        path = os.path.join(BASE_DIR, "hirise_data", "index.geojson")
    elif instrument_lower == "sharad":
        path = os.path.join(BASE_DIR, "sharad_data", "index.geojson")
    else:
        raise HTTPException(status_code=400, detail=f"Unknown instrument: {instrument}")

    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"{instrument} index.geojson not found")

    with open(path, "r", encoding="utf-8") as f:
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
    """Check if a feature intersects the bounding box."""
    geom = feature.get("geometry")
    if not geom:
        return False

    geom_type = geom.get("type")
    coords = geom.get("coordinates")

    if not coords:
        return False

    # Extract bounds from geometry
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

    # Latitude check (simple)
    if feat_max_lat < min_lat or feat_min_lat > max_lat:
        return False

    # Longitude check
    if crosses_antimeridian:
        # Query bbox crosses antimeridian: min_lon > max_lon
        # Feature intersects if it's in [min_lon, 180] OR [-180, max_lon]
        in_western_part = feat_max_lon >= min_lon  # Feature reaches into western part
        in_eastern_part = feat_min_lon <= max_lon  # Feature reaches into eastern part

        # Feature itself might cross antimeridian
        feat_crosses = feat_max_lon - feat_min_lon > 180
        if feat_crosses:
            return True  # Feature crosses antimeridian, likely intersects

        return in_western_part or in_eastern_part
    else:
        # Normal case: simple overlap check
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
        return geom
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
    matching_features = []
    for feature in all_features:
        if feature_intersects_bbox(feature, min_lon, min_lat, max_lon, max_lat, crosses_antimeridian):
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
                # Apply simplification if requested
                if simplify and SHAPELY_AVAILABLE:
                    tolerance = SIMPLIFY_TOLERANCES.get(simplify, 0.005)
                    geom = simplify_geometry(geom, tolerance)

                result_features.append({
                    "type": "Feature",
                    "properties": props,
                    "geometry": geom,
                })

    return JSONResponse(content={
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
    })


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
