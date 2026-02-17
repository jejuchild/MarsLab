"""Coordinate transforms and distance calculations for Mars.

Consolidates utilities previously duplicated across sharad_highres_router,
sharad_pick, sharad_reflectors, terrain_router, and RTE pipeline.

Longitude convention:
  - SHARAD RDR geometry uses 0-360 East-positive
  - Cesium / output uses -180..180
  - MOLA DEM is indexed 0-360
  Use lon_to_180() at output boundaries, not internally.

Latitude convention:
  - SHARAD RDR uses planetocentric
  - Cesium / output uses planetographic (geodetic)
  Use centric_to_graphic() when converting for display.
"""

import math
import numpy as np

from .constants import MARS_AB2, MARS_MEAN_RADIUS_M, MARS_MEAN_RADIUS_KM


# ── Longitude conversion ─────────────────────────────────────────

def lon_to_180(lon):
    """Convert 0..360 East-positive longitude to -180..180.

    Accepts scalar float or numpy array.
    """
    if isinstance(lon, np.ndarray):
        return np.where(lon > 180, lon - 360, lon)
    return lon - 360 if lon > 180 else lon


# ── Latitude conversion ──────────────────────────────────────────

def centric_to_graphic(lat_centric):
    """Convert planetocentric → planetographic latitude.

    tan(lat_graphic) = (a/b)^2 * tan(lat_centric)

    Accepts scalar float or numpy array.
    """
    if isinstance(lat_centric, np.ndarray):
        rad = np.radians(lat_centric)
        return np.degrees(np.arctan(MARS_AB2 * np.tan(rad)))
    rad = math.radians(lat_centric)
    return math.degrees(math.atan(MARS_AB2 * math.tan(rad)))


# ── Great-circle distance (scalar) ───────────────────────────────

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points on Mars (km)."""
    rlat1, rlat2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2)
    return MARS_MEAN_RADIUS_KM * 2 * math.asin(math.sqrt(min(a, 1.0)))


# ── Cumulative along-track distance (vectorized) ─────────────────

def cumulative_distance_m(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Cumulative great-circle distance on Mars (metres), vectorized.

    Returns array of length len(lats) with distances[0] = 0.0.
    """
    lat_r = np.radians(lats)
    lon_r = np.radians(lons)
    dlat = np.diff(lat_r)
    dlon = np.diff(lon_r)
    a = (np.sin(dlat / 2) ** 2
         + np.cos(lat_r[:-1]) * np.cos(lat_r[1:]) * np.sin(dlon / 2) ** 2)
    seg = MARS_MEAN_RADIUS_M * 2 * np.arcsin(np.sqrt(np.minimum(a, 1.0)))
    return np.concatenate([[0.0], np.cumsum(seg)])
