"""
Shared DEM extraction utilities for MOLA landform detection.

Reuses the global MOLA DEM infrastructure from terrain_router.
Provides a SharedDEMContext to avoid redundant I/O and computation.
"""

import logging
import math
from dataclasses import dataclass, field

import numpy as np
import rasterio
from rasterio.windows import Window
from scipy.ndimage import gaussian_filter

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from api.terrain_router import _get_dem, MARS_EQUATORIAL_RADIUS, MARS_POLAR_RADIUS, MARS_MEAN_RADIUS

logger = logging.getLogger(__name__)

MAX_SCAN_RADIUS_KM = 500  # Cap to prevent OOM


# ---------------------------------------------------------------------------
# Shared DEM context — extract once, reuse everywhere
# ---------------------------------------------------------------------------

@dataclass
class SharedDEMContext:
    """Pre-computed DEM data shared across all detectors in a single scan."""
    elev: np.ndarray              # raw elevation (NaN for nodata)
    elev_filled: np.ndarray       # NaN-filled elevation
    elev_smooth: np.ndarray       # Gaussian-smoothed (sigma=2)
    slope_deg: np.ndarray         # slope map in degrees
    neg_laplacian: np.ndarray     # -(d2z/dx2 + d2z/dy2); >0 = valley, <0 = ridge
    meta: dict = field(default_factory=dict)
    px_m_ew: float = 0.0
    px_m_ns: float = 0.0
    px_m: float = 0.0             # mean pixel size


def build_shared_context(lat0: float, lon0: float, radius_km: float) -> SharedDEMContext:
    """
    Extract DEM once and pre-compute all shared derivatives.

    This eliminates 4x redundant DEM reads and duplicate slope/Laplacian
    computations across detectors.
    """
    elev, meta = extract_dem_window(lat0, lon0, radius_km)
    px_m_ew = meta["px_m_ew"]
    px_m_ns = meta["px_m_ns"]
    px_m = (px_m_ew + px_m_ns) / 2.0

    # Fill NaN
    nan_mask = np.isnan(elev)
    fill_val = float(np.nanmean(elev)) if not np.all(nan_mask) else 0.0
    elev_filled = np.where(nan_mask, fill_val, elev)

    # Gaussian smooth (sigma=2, shared across all detectors)
    elev_smooth = gaussian_filter(elev_filled, sigma=2.0)

    # Slope map (shared: used by craters, ridges, LDAs)
    slope_deg = compute_slope_map(elev_smooth, px_m_ns, px_m_ew)

    # Laplacian curvature (shared: used by channels + ridges)
    neg_laplacian = _compute_laplacian(elev_smooth, px_m_ew, px_m_ns)

    logger.info(
        "SharedDEMContext: %dx%d window (%.0f km radius) at (%.3f, %.3f)",
        elev.shape[1], elev.shape[0], radius_km, lat0, lon0,
    )

    return SharedDEMContext(
        elev=elev,
        elev_filled=elev_filled,
        elev_smooth=elev_smooth,
        slope_deg=slope_deg,
        neg_laplacian=neg_laplacian,
        meta=meta,
        px_m_ew=px_m_ew,
        px_m_ns=px_m_ns,
        px_m=px_m,
    )


def _compute_laplacian(
    elev_smooth: np.ndarray, px_m_ew: float, px_m_ns: float,
) -> np.ndarray:
    """
    Negative Laplacian of the elevation surface.

    Returns -(d2z/dx2 + d2z/dy2).
    Positive = concave-up (valleys); negative = convex-up (ridges).
    """
    dz_dy = np.gradient(elev_smooth, px_m_ns, axis=0)
    dz_dx = np.gradient(elev_smooth, px_m_ew, axis=1)
    d2z_dy2 = np.gradient(dz_dy, px_m_ns, axis=0)
    d2z_dx2 = np.gradient(dz_dx, px_m_ew, axis=1)
    return -(d2z_dx2 + d2z_dy2)


def extract_dem_window(
    lat0: float, lon0: float, radius_km: float
) -> tuple:
    """
    Extract a DEM patch from the global MOLA DEM.

    Args:
        lat0: Center latitude (degrees, -90 to 90)
        lon0: Center longitude (degrees, -180 to 180)
        radius_km: Scan radius in km (capped at 500)

    Returns:
        (elev, meta) where:
            elev: 2D float64 array of elevation values (NaN for nodata)
            meta: dict with:
                px_m_ew: pixel size in meters (east-west)
                px_m_ns: pixel size in meters (north-south)
                row0, col0: top-left pixel coords in global raster
                nrows, ncols: patch dimensions
                lat0, lon0: center coordinates
                radius_km: actual scan radius used
    """
    radius_km = min(radius_km, MAX_SCAN_RADIUS_KM)
    ds = _get_dem()

    # Normalize longitude to 0-360 (DEM convention)
    lon0_360 = lon0 % 360
    if lon0_360 < 0:
        lon0_360 += 360

    # Pixel sizes in degrees
    px_deg_ew = abs(ds.transform.a)
    px_deg_ns = abs(ds.transform.e)

    # Meters per degree at this latitude
    meters_per_deg_lat = (math.pi / 180.0) * MARS_MEAN_RADIUS
    meters_per_deg_lon = meters_per_deg_lat * math.cos(math.radians(lat0))

    px_m_ew = px_deg_ew * max(meters_per_deg_lon, 1.0)
    px_m_ns = px_deg_ns * meters_per_deg_lat

    radius_m = radius_km * 1000.0

    # Number of pixels for radius
    half_ew = int(math.ceil(radius_m / px_m_ew)) + 2
    half_ns = int(math.ceil(radius_m / px_m_ns)) + 2

    # Center pixel
    row_c, col_c = ds.index(lon0_360, lat0)
    row_c = max(0, min(ds.height - 1, row_c))
    col_c = max(0, min(ds.width - 1, col_c))

    # Window bounds
    row0 = max(0, row_c - half_ns)
    row1 = min(ds.height, row_c + half_ns + 1)
    col0 = max(0, col_c - half_ew)
    col1 = min(ds.width, col_c + half_ew + 1)

    window = Window(col0, row0, col1 - col0, row1 - row0)
    elev = ds.read(1, window=window).astype(np.float64)

    # Mark nodata as NaN
    if ds.nodata is not None:
        elev[elev == ds.nodata] = np.nan

    meta = {
        "px_m_ew": px_m_ew,
        "px_m_ns": px_m_ns,
        "px_deg_ew": px_deg_ew,
        "px_deg_ns": px_deg_ns,
        "row0": row0,
        "col0": col0,
        "row_c_local": row_c - row0,
        "col_c_local": col_c - col0,
        "nrows": elev.shape[0],
        "ncols": elev.shape[1],
        "lat0": lat0,
        "lon0": lon0,
        "lon0_360": lon0_360,
        "radius_km": radius_km,
        "transform_a": ds.transform.a,  # pixel width in degrees
        "transform_f": ds.transform.f,  # top-left latitude
        "transform_c": ds.transform.c,  # top-left longitude
    }

    return elev, meta


def compute_slope_map(
    elev: np.ndarray, px_m_ns: float, px_m_ew: float
) -> np.ndarray:
    """Compute slope in degrees from elevation grid."""
    dz_dy, dz_dx = np.gradient(elev, px_m_ns, px_m_ew)
    slope_deg = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))
    return slope_deg


def pixel_to_latlon(row: int, col: int, meta: dict) -> tuple:
    """
    Convert local pixel (row, col) in extracted window to geographic (lat, lon).

    Returns (lat, lon) in degrees with lon in -180/180 convention.
    """
    # Global pixel coordinates
    abs_row = meta["row0"] + row
    abs_col = meta["col0"] + col

    # Transform from pixel to geographic
    lon_360 = meta["transform_c"] + abs_col * meta["px_deg_ew"]
    lat = meta["transform_f"] - abs_row * meta["px_deg_ns"]

    # Convert 0-360 to -180/180
    lon = lon_360 if lon_360 <= 180 else lon_360 - 360

    return lat, lon


def disk_footprint(radius_px: int) -> np.ndarray:
    """Create a circular disk structuring element."""
    y, x = np.ogrid[-radius_px:radius_px + 1, -radius_px:radius_px + 1]
    return (x**2 + y**2 <= radius_px**2).astype(np.uint8)


def pixels_to_km(n_pixels: float, px_m: float) -> float:
    """Convert pixel count to kilometers."""
    return n_pixels * px_m / 1000.0


def km_to_pixels(km: float, px_m: float) -> int:
    """Convert kilometers to pixel count."""
    return max(1, int(round(km * 1000.0 / px_m)))
