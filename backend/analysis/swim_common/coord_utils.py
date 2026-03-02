"""
Coordinate utilities for SWIM modules.

Handles:
  - 0-360°E ↔ -180-180°E longitude conversion
  - SWIM region bounds validation
  - Elevation checks (SWIM excludes >1 km)
"""

from typing import Tuple, Optional


# SWIM study region limits
SWIM_LAT_MIN = -60.0
SWIM_LAT_MAX = 60.0
SWIM_LON_MIN = -180.0
SWIM_LON_MAX = 180.0
SWIM_MAX_ELEVATION_M = 1000.0  # SWIM excludes elevations above 1 km
SWIM_MAX_REGION_SIZE_DEG = 10.0  # Max query region size


def lon_360_to_180(lon: float) -> float:
    """Convert longitude from 0-360°E to -180-180°E."""
    if lon > 180.0:
        return lon - 360.0
    return lon


def lon_180_to_360(lon: float) -> float:
    """Convert longitude from -180-180°E to 0-360°E."""
    if lon < 0.0:
        return lon + 360.0
    return lon


def validate_swim_bounds(
    lat: float, lon: float
) -> Tuple[bool, Optional[str]]:
    """
    Check if a point is within the SWIM study region.

    Returns:
        (is_valid, error_message) — error_message is None if valid
    """
    if lat < SWIM_LAT_MIN or lat > SWIM_LAT_MAX:
        return False, f"Latitude {lat}° outside SWIM coverage ({SWIM_LAT_MIN}° to {SWIM_LAT_MAX}°)"
    if lon < SWIM_LON_MIN or lon > SWIM_LON_MAX:
        return False, f"Longitude {lon}° outside SWIM coverage"
    return True, None


def validate_region_size(
    north: float, south: float, east: float, west: float
) -> Tuple[bool, Optional[str]]:
    """
    Validate that a region query is within size limits.

    Returns:
        (is_valid, error_message)
    """
    lat_span = north - south
    lon_span = east - west

    if lat_span <= 0 or lon_span <= 0:
        return False, "Invalid region: north must be > south, east must be > west"

    if lat_span > SWIM_MAX_REGION_SIZE_DEG or lon_span > SWIM_MAX_REGION_SIZE_DEG:
        return False, (
            f"Region too large ({lat_span:.1f}° × {lon_span:.1f}°). "
            f"Maximum is {SWIM_MAX_REGION_SIZE_DEG}° × {SWIM_MAX_REGION_SIZE_DEG}°"
        )

    return True, None


def clamp_to_swim_region(
    north: float, south: float, east: float, west: float
) -> Tuple[float, float, float, float]:
    """Clamp a bounding box to the SWIM study region."""
    return (
        min(north, SWIM_LAT_MAX),
        max(south, SWIM_LAT_MIN),
        min(east, SWIM_LON_MAX),
        max(west, SWIM_LON_MIN),
    )
