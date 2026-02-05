"""
Terrain slope analysis router.

Computes slope statistics within a radius around a point on Mars
using the HRSC/MOLA Blend DEM (~200m/pixel).
"""

import os
import math

import numpy as np
import rasterio
from rasterio.windows import Window
from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/terrain", tags=["Terrain"])

# Path to the global Mars DEM (project root = backend/../)
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJECT_ROOT = os.path.dirname(_BACKEND_DIR)
DEM_PATH = os.path.join(
    _PROJECT_ROOT,
    "Mars_HRSC_MOLA_BlendDEM_Global_200mp_v2.tif",
)

# Mars ellipsoid parameters (IAU 2000)
MARS_EQUATORIAL_RADIUS = 3_396_190.0  # meters
MARS_POLAR_RADIUS = 3_376_200.0  # meters
MARS_MEAN_RADIUS = (2 * MARS_EQUATORIAL_RADIUS + MARS_POLAR_RADIUS) / 3

# Cached DEM dataset handle
_dem_ds = None


def _get_dem():
    """Open (and cache) the DEM dataset."""
    global _dem_ds
    if _dem_ds is None:
        if not os.path.exists(DEM_PATH):
            raise FileNotFoundError(f"DEM file not found: {DEM_PATH}")
        _dem_ds = rasterio.open(DEM_PATH)
    return _dem_ds


def _haversine_vectorized(
    lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """Vectorised great-circle distance on Mars (meters)."""
    lat1_r = np.radians(lat1)
    lon1_r = np.radians(lon1)
    lat2_r = np.radians(lat2)
    lon2_r = np.radians(lon2)

    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return MARS_MEAN_RADIUS * c


def compute_slope_stats(
    lat0: float, lon0: float, radius_m: float = 1000
) -> dict:
    """
    Compute slope statistics within *radius_m* of (*lat0*, *lon0*).

    Returns a dict with mean_slope, std_slope, max_slope, count,
    elevation_m, and a slope distribution histogram.
    """
    ds = _get_dem()

    # Centre pixel
    row_c, col_c = ds.index(lon0, lat0)

    # Pixel sizes in degrees
    px_deg_ew = abs(ds.transform.a)
    px_deg_ns = abs(ds.transform.e)

    # Metre-per-degree at this latitude
    ew_m_per_deg = (2 * math.pi * MARS_EQUATORIAL_RADIUS * math.cos(math.radians(lat0))) / 360
    ns_m_per_deg = (2 * math.pi * MARS_POLAR_RADIUS) / 360

    px_m_ew = px_deg_ew * ew_m_per_deg
    px_m_ns = px_deg_ns * ns_m_per_deg

    # Window half-size (pixels) with 2-pixel margin for gradient edges
    margin = 2
    half_ew = int(math.ceil(radius_m / px_m_ew)) + margin
    half_ns = int(math.ceil(radius_m / px_m_ns)) + margin

    row0 = max(0, row_c - half_ns)
    row1 = min(ds.height, row_c + half_ns + 1)
    col0 = max(0, col_c - half_ew)
    col1 = min(ds.width, col_c + half_ew + 1)

    window = Window(col0, row0, col1 - col0, row1 - row0)
    elev = ds.read(1, window=window).astype(np.float64)

    # Mark nodata as NaN
    if ds.nodata is not None:
        elev[elev == ds.nodata] = np.nan

    # Slope from elevation gradients (degrees)
    dz_dy, dz_dx = np.gradient(elev, px_m_ns, px_m_ew)
    slope_deg = np.degrees(np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2)))

    # Distance mask: vectorised haversine
    nrows, ncols = elev.shape
    abs_rows = row0 + np.arange(nrows)
    abs_cols = col0 + np.arange(ncols)
    col_grid, row_grid = np.meshgrid(abs_cols, abs_rows)

    t = ds.transform
    lons_grid = t.c + col_grid * t.a  # t.b is 0 for north-up
    lats_grid = t.f + row_grid * t.e

    dists = _haversine_vectorized(lat0, lon0, lats_grid, lons_grid)
    mask = dists <= radius_m

    valid = mask & ~np.isnan(slope_deg)

    if not np.any(valid):
        return {
            "mean_slope": 0,
            "std_slope": 0,
            "max_slope": 0,
            "count": 0,
            "elevation_m": 0,
            "distribution": {"0_3": 0, "3_5": 0, "5_plus": 0},
        }

    slopes = slope_deg[valid]
    total = len(slopes)

    # Distribution percentages (5° safety threshold)
    dist_0_3 = float(np.sum(slopes < 3) / total * 100)
    dist_3_5 = float(np.sum((slopes >= 3) & (slopes < 5)) / total * 100)
    dist_5_plus = float(np.sum(slopes >= 5) / total * 100)

    # Centre-pixel elevation
    cr = row_c - row0
    cc = col_c - col0
    center_elev = float(elev[cr, cc]) if not np.isnan(elev[cr, cc]) else 0.0

    return {
        "mean_slope": round(float(np.mean(slopes)), 2),
        "std_slope": round(float(np.std(slopes)), 2),
        "max_slope": round(float(np.max(slopes)), 2),
        "count": int(total),
        "elevation_m": round(center_elev, 1),
        "distribution": {
            "0_3": round(dist_0_3, 1),
            "3_5": round(dist_3_5, 1),
            "5_plus": round(dist_5_plus, 1),
        },
    }


# ──────────────────────────────────────────────
# API endpoint
# ──────────────────────────────────────────────
@router.get("/slope_stats")
async def get_slope_stats(
    lat: float = Query(..., description="Centre latitude (degrees)"),
    lon: float = Query(..., description="Centre longitude (degrees)"),
    radius_m: float = Query(1000, description="Analysis radius (metres)"),
):
    """Compute terrain slope statistics within a radius on Mars."""
    try:
        result = compute_slope_stats(lat, lon, radius_m)

        # Safety assessment (5° landing-slope constraint)
        if result["count"] == 0:
            safety = "UNKNOWN"
            safety_desc = "No valid terrain data available at this location."
        else:
            pct_below_5 = result["distribution"]["0_3"] + result["distribution"]["3_5"]
            pct_unsafe = result["distribution"]["5_plus"]

            if pct_below_5 >= 100.0:
                safety = "FAVORABLE"
                safety_desc = (
                    f"Slope values within the analysis radius are 100.0% within "
                    f"the mission safety threshold (< 5\u00b0). "
                    f"Mean slope {result['mean_slope']:.1f}\u00b0 is well below "
                    f"the maximum allowable slope for landing safety."
                )
            elif result["mean_slope"] < 5 and pct_unsafe < 10:
                safety = "MARGINAL"
                safety_desc = (
                    f"{pct_unsafe:.1f}% of pixels exceed the mission safety threshold "
                    f"(\u2265 5\u00b0). Mean slope {result['mean_slope']:.1f}\u00b0 is "
                    f"below the limit, but localised steep areas require further analysis."
                )
            else:
                safety = "UNFAVORABLE"
                safety_desc = (
                    f"{pct_unsafe:.1f}% of pixels exceed the mission safety threshold "
                    f"(\u2265 5\u00b0), indicating potential landing hazards. "
                    f"Mean slope {result['mean_slope']:.1f}\u00b0. "
                    f"Not recommended for landing."
                )

        result["safety"] = safety
        result["safety_description"] = safety_desc
        result["lat"] = lat
        result["lon"] = lon
        result["radius_m"] = radius_m

        return JSONResponse(content=result)

    except FileNotFoundError as e:
        return JSONResponse(content={"error": str(e)}, status_code=404)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# ──────────────────────────────────────────────
# Line profile endpoint
# ──────────────────────────────────────────────
def _haversine_scalar(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points on Mars (metres)."""
    lat1_r, lon1_r = math.radians(lat1), math.radians(lon1)
    lat2_r, lon2_r = math.radians(lat2), math.radians(lon2)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    return MARS_MEAN_RADIUS * 2 * math.asin(math.sqrt(min(a, 1.0)))


def compute_line_profile(
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
    num_samples: int = 200,
) -> list:
    """
    Extract elevation profile along a straight line between two points.

    Returns a list of dicts: {distance_km, elevation_m, lat, lon}
    """
    ds = _get_dem()

    # Generate sample points along the line (linear interpolation in lat/lon)
    fracs = np.linspace(0.0, 1.0, num_samples)
    lats = start_lat + fracs * (end_lat - start_lat)
    lons = start_lon + fracs * (end_lon - start_lon)

    # Convert lat/lon to pixel coordinates (vectorised)
    inv_transform = ~ds.transform
    cols, rows = inv_transform * (lons, lats)
    cols = np.round(cols).astype(int)
    rows = np.round(rows).astype(int)

    # Clamp to raster bounds
    cols = np.clip(cols, 0, ds.width - 1)
    rows = np.clip(rows, 0, ds.height - 1)

    # Read a minimal bounding window for efficiency
    row_min, row_max = int(rows.min()), int(rows.max())
    col_min, col_max = int(cols.min()), int(cols.max())
    window = Window(col_min, row_min, col_max - col_min + 1, row_max - row_min + 1)
    elev_block = ds.read(1, window=window).astype(np.float64)

    if ds.nodata is not None:
        elev_block[elev_block == ds.nodata] = np.nan

    # Extract elevations for each sample
    local_rows = rows - row_min
    local_cols = cols - col_min
    elevations = elev_block[local_rows, local_cols]

    # Compute cumulative distance along the profile
    distances_m = np.zeros(num_samples)
    for i in range(1, num_samples):
        distances_m[i] = distances_m[i - 1] + _haversine_scalar(
            lats[i - 1], lons[i - 1], lats[i], lons[i]
        )

    # Build result
    profile = []
    for i in range(num_samples):
        elev_val = float(elevations[i])
        if np.isnan(elev_val):
            elev_val = None
        profile.append({
            "distance_km": round(distances_m[i] / 1000.0, 3),
            "elevation_m": round(elev_val, 1) if elev_val is not None else None,
            "lat": round(float(lats[i]), 6),
            "lon": round(float(lons[i]), 6),
        })

    return profile


@router.get("/line_profile")
async def get_line_profile(
    start_lat: float = Query(..., description="Start latitude (degrees)"),
    start_lon: float = Query(..., description="Start longitude (degrees)"),
    end_lat: float = Query(..., description="End latitude (degrees)"),
    end_lon: float = Query(..., description="End longitude (degrees)"),
    num_samples: int = Query(200, ge=10, le=2000, description="Number of sample points"),
):
    """Extract elevation profile along a line between two points on Mars."""
    try:
        profile = compute_line_profile(start_lat, start_lon, end_lat, end_lon, num_samples)
        total_dist_km = profile[-1]["distance_km"] if profile else 0

        return JSONResponse(content={
            "profile": profile,
            "total_distance_km": total_dist_km,
            "num_samples": len(profile),
            "start": {"lat": start_lat, "lon": start_lon},
            "end": {"lat": end_lat, "lon": end_lon},
        })

    except FileNotFoundError as e:
        return JSONResponse(content={"error": str(e)}, status_code=404)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# ──────────────────────────────────────────────
# DEM patch endpoint (for 3D terrain viewer)
# ──────────────────────────────────────────────
def compute_dem_patch(
    lat0: float,
    lon0: float,
    radius_m: float = 5000,
    grid_size: int = 128,
) -> dict:
    """
    Extract a DEM patch centered at (lat0, lon0) with given radius.

    Returns a dict with:
      - elevations: flattened array of elevation values (row-major order)
      - rows, cols: grid dimensions
      - spacing_m: approximate grid spacing in meters
      - bounds: {west, east, south, north} in degrees
      - center_elevation_m: elevation at center point
      - min_elevation_m, max_elevation_m: elevation range
      - slope_mean, slope_max: slope statistics for the patch
    """
    ds = _get_dem()

    # Centre pixel
    row_c, col_c = ds.index(lon0, lat0)

    # Pixel sizes in degrees
    px_deg_ew = abs(ds.transform.a)
    px_deg_ns = abs(ds.transform.e)

    # Approximate metres per degree at this latitude
    meters_per_deg_lat = (math.pi / 180.0) * MARS_MEAN_RADIUS
    meters_per_deg_lon = meters_per_deg_lat * math.cos(math.radians(lat0))

    # Pixel size in metres
    px_m_ns = px_deg_ns * meters_per_deg_lat
    px_m_ew = px_deg_ew * meters_per_deg_lon

    # Number of pixels to cover radius
    n_pix_ns = int(math.ceil(radius_m / px_m_ns))
    n_pix_ew = int(math.ceil(radius_m / px_m_ew))

    # Define window
    row_start = max(0, row_c - n_pix_ns)
    row_end = min(ds.height, row_c + n_pix_ns + 1)
    col_start = max(0, col_c - n_pix_ew)
    col_end = min(ds.width, col_c + n_pix_ew + 1)

    window = Window(col_start, row_start, col_end - col_start, row_end - row_start)
    elev = ds.read(1, window=window).astype(np.float64)

    if ds.nodata is not None:
        elev[elev == ds.nodata] = np.nan

    # Resample to target grid size if needed
    original_rows, original_cols = elev.shape
    if original_rows != grid_size or original_cols != grid_size:
        from scipy.ndimage import zoom
        zoom_factor_y = grid_size / original_rows
        zoom_factor_x = grid_size / original_cols
        elev = zoom(elev, (zoom_factor_y, zoom_factor_x), order=1)  # bilinear interpolation

    rows, cols = elev.shape

    # Compute bounds
    west_lon = ds.transform.c + col_start * ds.transform.a
    north_lat = ds.transform.f + row_start * ds.transform.e
    east_lon = ds.transform.c + col_end * ds.transform.a
    south_lat = ds.transform.f + row_end * ds.transform.e

    # Grid spacing in meters (after resampling)
    spacing_m_ew = (radius_m * 2) / cols
    spacing_m_ns = (radius_m * 2) / rows

    # Compute slope (in degrees)
    dz_dy, dz_dx = np.gradient(elev, spacing_m_ns, spacing_m_ew)
    slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    slope_deg = np.degrees(slope_rad)

    # Mask invalid values
    valid_mask = ~np.isnan(elev)
    valid_slopes = slope_deg[valid_mask]

    # Statistics
    center_row, center_col = rows // 2, cols // 2
    center_elev = float(elev[center_row, center_col])
    if np.isnan(center_elev):
        center_elev = float(np.nanmean(elev))

    # Replace NaN with mean for output
    elev_clean = np.nan_to_num(elev, nan=float(np.nanmean(elev)))

    return {
        "elevations": elev_clean.flatten().tolist(),
        "rows": rows,
        "cols": cols,
        "spacing_m": round((spacing_m_ew + spacing_m_ns) / 2, 2),
        "bounds": {
            "west": round(west_lon, 6),
            "east": round(east_lon, 6),
            "south": round(south_lat, 6),
            "north": round(north_lat, 6),
        },
        "center": {"lat": lat0, "lon": lon0},
        "center_elevation_m": round(center_elev, 1),
        "min_elevation_m": round(float(np.nanmin(elev)), 1),
        "max_elevation_m": round(float(np.nanmax(elev)), 1),
        "elevation_range_m": round(float(np.nanmax(elev) - np.nanmin(elev)), 1),
        "slope_mean": round(float(np.mean(valid_slopes)), 2) if len(valid_slopes) > 0 else 0,
        "slope_max": round(float(np.max(valid_slopes)), 2) if len(valid_slopes) > 0 else 0,
        "radius_m": radius_m,
    }


@router.get("/dem_patch")
async def get_dem_patch(
    lat: float = Query(..., description="Centre latitude (degrees)"),
    lon: float = Query(..., description="Centre longitude (degrees)"),
    radius_m: float = Query(5000, ge=500, le=50000, description="Patch radius (metres)"),
    grid_size: int = Query(128, ge=32, le=512, description="Output grid size (rows/cols)"),
):
    """
    Get a DEM patch for 3D terrain visualization.

    Returns elevation grid data with metadata for rendering in a 3D viewer.
    """
    try:
        result = compute_dem_patch(lat, lon, radius_m, grid_size)
        return JSONResponse(content=result)

    except FileNotFoundError as e:
        return JSONResponse(content={"error": str(e)}, status_code=404)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)
