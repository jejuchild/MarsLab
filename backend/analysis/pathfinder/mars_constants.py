"""Mars physical and environmental constants for pathfinder computations.

Constants sourced from IAU 2000/2015 reports and NASA Mars Fact Sheet.
Reuses MARS_EQUATORIAL_RADIUS / MARS_POLAR_RADIUS already defined
in api.terrain_router; this module adds rover-specific constants.

References:
    IAU 2015 Report on Cartographic Coordinates and Rotational Elements
    NASA Mars Fact Sheet: https://nssdc.gsfc.nasa.gov/planetary/factsheet/marsfact.html
"""

import math

# ── Mars Ellipsoid (IAU 2000) ──────────────────────────────────
MARS_EQUATORIAL_RADIUS_M = 3_396_190.0
MARS_POLAR_RADIUS_M = 3_376_200.0
MARS_MEAN_RADIUS_M = (2 * MARS_EQUATORIAL_RADIUS_M + MARS_POLAR_RADIUS_M) / 3

# ── Gravity & Atmosphere ───────────────────────────────────────
MARS_GRAVITY_M_S2 = 3.72076           # surface gravity (m/s²)
MARS_ATM_PRESSURE_PA = 636.0          # mean surface pressure (Pa)
MARS_ATM_DENSITY_KG_M3 = 0.020       # approx surface air density

# ── Sol & Time ─────────────────────────────────────────────────
MARS_SOL_SECONDS = 88_775.244         # one sol in Earth seconds
MARS_SOL_HOURS = MARS_SOL_SECONDS / 3600.0  # ~24.66 hours

# ── Communication ──────────────────────────────────────────────
EARTH_MARS_LIGHT_DELAY_MIN = 4.3      # minimum one-way (opposition)
EARTH_MARS_LIGHT_DELAY_MAX = 24.0     # maximum one-way (conjunction)
EARTH_MARS_LIGHT_DELAY_MEAN = 12.5    # mean one-way

# ── Terrain Traversability Thresholds ──────────────────────────
# Based on Perseverance operational limits and MER experience
MAX_SAFE_SLOPE_DEG = 15.0             # hard limit — no traverse above this
MARGINAL_SLOPE_DEG = 10.0             # caution zone begins
COMFORTABLE_SLOPE_DEG = 5.0           # nominal safe traverse

MAX_ROUGHNESS_TRI = 500.0             # terrain ruggedness index limit (m)
SAFE_ROUGHNESS_TRI = 50.0             # smooth terrain threshold

# Elevation thresholds (higher = thinner atmosphere = riskier EDL)
IDEAL_ELEVATION_M = -2000.0           # MOLA datum reference
MAX_LANDING_ELEVATION_M = 2000.0      # upper limit for safe EDL

# ── Coordinate Helpers ─────────────────────────────────────────

def haversine_mars(lat1: float, lon1: float,
                   lat2: float, lon2: float) -> float:
    """Great-circle distance on Mars (meters) using haversine formula.

    Args:
        lat1, lon1: Point 1 coordinates in degrees
        lat2, lon2: Point 2 coordinates in degrees

    Returns:
        Distance in meters on Mars surface
    """
    lat1_r = math.radians(lat1)
    lon1_r = math.radians(lon1)
    lat2_r = math.radians(lat2)
    lon2_r = math.radians(lon2)

    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r

    a = (math.sin(dlat / 2) ** 2
         + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2)
    c = 2 * math.asin(min(1.0, math.sqrt(a)))
    return MARS_MEAN_RADIUS_M * c


def meters_per_degree_lat(lat: float) -> float:
    """Meters per degree of latitude at given latitude on Mars."""
    return (2 * math.pi * MARS_POLAR_RADIUS_M) / 360.0


def meters_per_degree_lon(lat: float) -> float:
    """Meters per degree of longitude at given latitude on Mars."""
    return (2 * math.pi * MARS_EQUATORIAL_RADIUS_M
            * math.cos(math.radians(lat))) / 360.0


def bearing_deg(lat1: float, lon1: float,
                lat2: float, lon2: float) -> float:
    """Initial bearing from point 1 to point 2 in degrees (0=N, 90=E)."""
    lat1_r = math.radians(lat1)
    lat2_r = math.radians(lat2)
    dlon_r = math.radians(lon2 - lon1)

    x = math.sin(dlon_r) * math.cos(lat2_r)
    y = (math.cos(lat1_r) * math.sin(lat2_r)
         - math.sin(lat1_r) * math.cos(lat2_r) * math.cos(dlon_r))
    bearing = math.degrees(math.atan2(x, y))
    return bearing % 360.0
