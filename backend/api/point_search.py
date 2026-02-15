"""
Point-based coordinate search for Mars datasets via ODE.

Answers the question: "What datasets cover this exact location on Mars?"

Queries ODE (Orbital Data Explorer) for products near a given coordinate.
This is completely independent of spatial/bbox search - it uses a small
bounding box centered on the point to find all relevant products.
"""

import asyncio
from typing import List, Dict, Optional
from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from .ode_client import (
    Instrument,
    search_ode_spatial,
    search_sharad_spatial,
    search_sharad_highres_spatial,
)
from .download_manager import check_local_existence_detailed
import json
from pathlib import Path

router = APIRouter(prefix="/api/search", tags=["Point Search"])

# =============================================================================
# Constants
# =============================================================================

# Default search radius in degrees (approximately 60km at equator on Mars)
DEFAULT_SEARCH_RADIUS_DEG = 1.0

# Mars mean radius in km (for distance calculations)
MARS_RADIUS_KM = 3389.5

import math

# =============================================================================
# Helper Functions
# =============================================================================

def normalize_longitude_180(lon: float) -> float:
    """Normalize longitude to -180 to 180 range."""
    lon = lon % 360
    if lon > 180:
        lon -= 360
    return lon


def normalize_longitude_360(lon: float) -> float:
    """Normalize longitude to 0-360 range."""
    lon = lon % 360
    if lon < 0:
        lon += 360
    return lon


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate great-circle distance between two points on Mars.

    Args:
        lat1, lon1: First point (degrees)
        lat2, lon2: Second point (degrees)

    Returns:
        Distance in kilometers
    """
    lat1_r = math.radians(lat1)
    lat2_r = math.radians(lat2)
    lon1_r = math.radians(lon1)
    lon2_r = math.radians(lon2)

    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(min(a, 1.0)))

    return MARS_RADIUS_KM * c


def point_to_line_distance_km(
    point_lat: float, point_lon: float,
    line_start_lat: float, line_start_lon: float,
    line_end_lat: float, line_end_lon: float
) -> float:
    """
    Calculate the shortest distance from a point to a line segment on Mars (great circle).

    Uses cross-track distance formula for spherical geometry.
    If the perpendicular foot falls outside the segment, returns distance to nearest endpoint.

    Args:
        point_lat, point_lon: Query point (degrees)
        line_start_lat, line_start_lon: Line segment start (degrees)
        line_end_lat, line_end_lon: Line segment end (degrees)

    Returns:
        Distance in kilometers from point to the closest point on the line segment
    """
    # Convert to radians
    p_lat = math.radians(point_lat)
    p_lon = math.radians(point_lon)
    a_lat = math.radians(line_start_lat)
    a_lon = math.radians(line_start_lon)
    b_lat = math.radians(line_end_lat)
    b_lon = math.radians(line_end_lon)

    # Distance from point to start (d_pa) and end (d_pb)
    d_pa = haversine_distance_km(point_lat, point_lon, line_start_lat, line_start_lon)
    d_pb = haversine_distance_km(point_lat, point_lon, line_end_lat, line_end_lon)

    # Distance from start to end (d_ab)
    d_ab = haversine_distance_km(line_start_lat, line_start_lon, line_end_lat, line_end_lon)

    # If line segment is essentially a point, return distance to that point
    if d_ab < 0.001:  # Less than 1 meter
        return d_pa

    # Calculate cross-track distance using spherical trigonometry
    # Angular distances (in radians)
    delta_pa = d_pa / MARS_RADIUS_KM
    delta_pb = d_pb / MARS_RADIUS_KM
    delta_ab = d_ab / MARS_RADIUS_KM

    # Bearing from A to P
    y = math.sin(p_lon - a_lon) * math.cos(p_lat)
    x = math.cos(a_lat) * math.sin(p_lat) - math.sin(a_lat) * math.cos(p_lat) * math.cos(p_lon - a_lon)
    bearing_ap = math.atan2(y, x)

    # Bearing from A to B
    y = math.sin(b_lon - a_lon) * math.cos(b_lat)
    x = math.cos(a_lat) * math.sin(b_lat) - math.sin(a_lat) * math.cos(b_lat) * math.cos(b_lon - a_lon)
    bearing_ab = math.atan2(y, x)

    # Cross-track distance (perpendicular distance from P to great circle through A-B)
    cross_track = math.asin(
        min(1.0, max(-1.0, math.sin(delta_pa) * math.sin(bearing_ap - bearing_ab)))
    )
    cross_track_km = abs(cross_track) * MARS_RADIUS_KM

    # Along-track distance from A to the closest point on the great circle
    along_track = math.acos(
        min(1.0, max(-1.0, math.cos(delta_pa) / max(0.0001, math.cos(cross_track))))
    )
    along_track_km = along_track * MARS_RADIUS_KM

    # Check if the perpendicular foot falls within the segment
    # If along_track is negative or > d_ab, the closest point is an endpoint
    if along_track_km < 0:
        return d_pa  # Closest to start
    elif along_track_km > d_ab:
        return d_pb  # Closest to end
    else:
        return cross_track_km  # Perpendicular distance


def create_bbox_around_point(lat: float, lon: float, radius_deg: float = DEFAULT_SEARCH_RADIUS_DEG):
    """
    Create a bounding box around a point.

    Args:
        lat: Center latitude
        lon: Center longitude
        radius_deg: Search radius in degrees

    Returns:
        Tuple of (minlat, maxlat, westernlon, easternlon)
    """
    minlat = max(-90, lat - radius_deg)
    maxlat = min(90, lat + radius_deg)

    # Handle longitude wrapping
    westernlon = lon - radius_deg
    easternlon = lon + radius_deg

    return minlat, maxlat, westernlon, easternlon


# =============================================================================
# API Endpoint
# =============================================================================

@router.get("/point")
async def search_by_point(
    lat: float = Query(..., ge=-90, le=90, description="Latitude in degrees (-90 to 90)"),
    lon: float = Query(..., description="Longitude in degrees (any range, normalized internally)"),
    radius: float = Query(
        DEFAULT_SEARCH_RADIUS_DEG,
        ge=0.1,
        le=10,
        description="Search radius in degrees (default 1.0, roughly 60km)"
    ),
    limit: int = Query(20, ge=1, le=100, description="Maximum results per instrument"),
):
    """
    Search ODE for all datasets that cover a given coordinate.

    This queries ODE using a bounding box centered on the point to find
    all products from CRISM, HiRISE, SHARAD, and SHARAD High-Res that
    may cover or pass near this location.

    ## Search Logic

    Creates a bounding box around the point (default ±1 degree) and queries
    ODE for each instrument type.

    ## Longitude Handling

    Input longitude can be in any range (-180 to 180, 0 to 360, or beyond).
    It is normalized internally for the ODE query.

    ## Response

    Results are grouped by instrument. Each result includes:
    - `product_id`: Dataset identifier
    - `instrument`: Instrument name
    - `lat`, `lon`: Product center coordinates
    - `exists`: Whether product is already downloaded locally
    - `has_core`, `has_browse`: Local file status
    - `missing_files`: List of missing file types

    ## Example

    ```
    GET /api/search/point?lat=18.5&lon=-77.0&radius=1.0
    ```
    """
    try:
        # Normalize longitude
        lon_normalized = normalize_longitude_180(lon)

        # Create bounding box around point
        minlat, maxlat, westernlon, easternlon = create_bbox_around_point(
            lat, lon_normalized, radius
        )

        # Query all instruments in parallel
        results = {
            "CRISM": [],
            "HIRISE": [],
            "SHARAD": [],
            "SHARAD_HIGHRES": [],
            "HIRISE_DTM": [],
        }

        # Helper to calculate distance and add to result
        def make_result(p, instrument_name):
            existence = check_local_existence_detailed(p.product_id, p.instrument)

            # Calculate distance from query point
            distance_km = None

            # For SHARAD products with track coordinates, calculate distance to the track line
            if instrument_name in ("sharad", "sharad_highres") and \
               p.start_lat is not None and p.start_lon is not None and \
               p.stop_lat is not None and p.stop_lon is not None:
                distance_km = round(point_to_line_distance_km(
                    lat, lon_normalized,
                    p.start_lat, p.start_lon,
                    p.stop_lat, p.stop_lon
                ), 2)
            # Otherwise use center point distance
            elif p.lat is not None and p.lon is not None:
                distance_km = round(haversine_distance_km(lat, lon_normalized, p.lat, p.lon), 2)

            return {
                "product_id": p.product_id,
                "instrument": instrument_name,
                "lat": p.lat,
                "lon": p.lon,
                "distance_km": distance_km,
                "exists": existence.exists,
                "has_core": existence.has_core,
                "has_browse": existence.has_browse,
                "missing_files": existence.missing_files,
                # Include track endpoints for SHARAD (useful for frontend)
                "start_lat": p.start_lat if hasattr(p, 'start_lat') else None,
                "start_lon": p.start_lon if hasattr(p, 'start_lon') else None,
                "stop_lat": p.stop_lat if hasattr(p, 'stop_lat') else None,
                "stop_lon": p.stop_lon if hasattr(p, 'stop_lon') else None,
            }

        # Run all searches in parallel
        async def search_crism():
            try:
                products = await search_ode_spatial(
                    minlat, maxlat, westernlon, easternlon,
                    Instrument.CRISM, limit
                )
                for p in products:
                    results["CRISM"].append(make_result(p, "crism"))
                # Sort by distance
                results["CRISM"].sort(key=lambda x: x.get("distance_km") or float('inf'))
            except Exception as e:
                print(f"CRISM search error: {e}")

        async def search_hirise():
            try:
                products = await search_ode_spatial(
                    minlat, maxlat, westernlon, easternlon,
                    Instrument.HIRISE, limit
                )
                for p in products:
                    results["HIRISE"].append(make_result(p, "hirise"))
                # Sort by distance
                results["HIRISE"].sort(key=lambda x: x.get("distance_km") or float('inf'))
            except Exception as e:
                print(f"HiRISE search error: {e}")

        async def search_sharad():
            try:
                products = await search_sharad_spatial(
                    minlat, maxlat, westernlon, easternlon, limit
                )
                for p in products:
                    results["SHARAD"].append(make_result(p, "sharad"))
                # Sort by distance
                results["SHARAD"].sort(key=lambda x: x.get("distance_km") or float('inf'))
            except Exception as e:
                print(f"SHARAD search error: {e}")

        async def search_sharad_highres():
            try:
                products = await search_sharad_highres_spatial(
                    minlat, maxlat, westernlon, easternlon, limit
                )
                for p in products:
                    results["SHARAD_HIGHRES"].append(make_result(p, "sharad_highres"))
                # Sort by distance
                results["SHARAD_HIGHRES"].sort(key=lambda x: x.get("distance_km") or float('inf'))
            except Exception as e:
                print(f"SHARAD High-Res search error: {e}")

        async def search_hirise_dtm():
            try:
                index_path = Path(__file__).parent.parent / "hirise_dtm_data" / "index.geojson"
                if not index_path.exists():
                    return
                with open(index_path) as f:
                    data = json.load(f)
                for feature in data.get("features", []):
                    props = feature.get("properties", {})
                    pid = props.get("product_id", "")
                    if not pid:
                        continue
                    geom = feature.get("geometry", {})
                    coords = geom.get("coordinates", [])
                    if geom.get("type") != "Polygon" or not coords:
                        continue
                    ring = coords[0]
                    clat = sum(c[1] for c in ring) / len(ring)
                    clon = sum(c[0] for c in ring) / len(ring)
                    if clat < minlat or clat > maxlat:
                        continue
                    if westernlon <= easternlon:
                        if clon < westernlon or clon > easternlon:
                            continue
                    else:
                        if clon < westernlon and clon > easternlon:
                            continue
                    distance_km = round(haversine_distance_km(lat, lon_normalized, clat, clon), 2)
                    results["HIRISE_DTM"].append({
                        "product_id": pid,
                        "instrument": "hirise_dtm",
                        "lat": clat,
                        "lon": clon,
                        "distance_km": distance_km,
                        "exists": True,
                        "has_core": True,
                        "has_browse": True,
                        "missing_files": [],
                    })
                results["HIRISE_DTM"].sort(key=lambda x: x.get("distance_km") or float('inf'))
                results["HIRISE_DTM"] = results["HIRISE_DTM"][:limit]
            except Exception as e:
                print(f"HiRISE DTM search error: {e}")

        # Run all searches concurrently
        await asyncio.gather(
            search_crism(),
            search_hirise(),
            search_sharad(),
            search_sharad_highres(),
            search_hirise_dtm(),
        )

        # Count total results
        total = sum(len(v) for v in results.values())

        return JSONResponse(content={
            "query": {
                "lat": lat,
                "lon": round(lon_normalized, 6),
                "lon_360": round(normalize_longitude_360(lon), 6),
                "radius_deg": radius,
            },
            "total_count": total,
            "results": results,
        })

    except Exception as e:
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )
