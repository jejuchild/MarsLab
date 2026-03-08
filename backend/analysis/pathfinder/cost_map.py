"""Mars terrain traversability cost map generator.

Computes per-pixel traversal cost from DEM elevation, slope, roughness,
and hazard data.  Supports multi-resolution DEM:
  - HiRISE DTM (~1 m/px) when available for the region
  - HRSC/MOLA Blend (~200 m/px) global fallback

The cost map feeds directly into the Field D* path planner.

References:
    [1] Ferguson & Stentz, "Field D*," J. Field Robotics, 2006
    [2] Carsten et al., "Global Path Planning on Board the Mars Rovers," IEEE Aerospace, 2007
"""

import io
import json
import logging
import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import numpy as np
import rasterio
from rasterio.windows import Window
from scipy.ndimage import gaussian_filter

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from api.terrain_router import _get_dem, MARS_EQUATORIAL_RADIUS, MARS_POLAR_RADIUS, MARS_MEAN_RADIUS

from .mars_constants import (
    MAX_SAFE_SLOPE_DEG,
    MARGINAL_SLOPE_DEG,
    COMFORTABLE_SLOPE_DEG,
    meters_per_degree_lat,
    meters_per_degree_lon,
)
from .rover_models import RoverModel, PERSEVERANCE

logger = logging.getLogger(__name__)

# ── Default Cost Weights ────────────────────────────────────────
DEFAULT_WEIGHTS = {
    "slope": 0.40,
    "roughness": 0.25,
    "hazard": 0.25,
    "elevation": 0.10,
}


# ── Data Classes ────────────────────────────────────────────────

@dataclass
class CostMapResult:
    """Result of a cost map computation."""

    cost_grid: np.ndarray          # float32, np.inf = impassable
    slope_grid: np.ndarray         # float32, degrees
    elevation_grid: np.ndarray     # float32, meters
    hazard_mask: np.ndarray        # bool, True = hazard detected
    meta: dict[str, Any] = field(default_factory=dict)
    # meta keys: lat_min, lat_max, lon_min, lon_max, rows, cols,
    #            px_m_ns, px_m_ew, px_deg_ns, px_deg_ew


# ── HiRISE DTM Index ───────────────────────────────────────────

_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
_DTM_DIR = _BACKEND_DIR / "hirise_dtm_data"
_DTM_INDEX = _DTM_DIR / "index.geojson"


@lru_cache(maxsize=1)
def _load_dtm_index() -> list[dict]:
    """Load HiRISE DTM index as a list of {product_id, dtm_file, west, east, south, north}."""
    if not _DTM_INDEX.exists():
        return []
    try:
        with open(_DTM_INDEX) as f:
            data = json.load(f)
        return [feat["properties"] for feat in data.get("features", [])]
    except Exception as e:
        logger.warning("Failed to load HiRISE DTM index: %s", e)
        return []


def _find_best_hirise_dtm(
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
) -> Optional[dict]:
    """Find the HiRISE DTM with the best coverage for the given bbox.

    Returns the DTM properties dict with the largest overlap, or None
    if no DTM covers at least 10% of the request bbox.
    """
    index = _load_dtm_index()
    if not index:
        return None

    req_area = (lat_max - lat_min) * (lon_max - lon_min)
    if req_area <= 0:
        return None

    best_dtm = None
    best_overlap = 0.0

    for dtm in index:
        dtm_file = dtm.get("dtm_file", "")
        if not dtm_file:
            continue
        # Check file exists on disk
        dtm_path = _DTM_DIR / dtm_file
        if not dtm_path.exists():
            continue

        dw = dtm.get("west", 0)
        de = dtm.get("east", 0)
        ds_ = dtm.get("south", 0)
        dn = dtm.get("north", 0)

        # Compute overlap area
        o_lat_min = max(lat_min, ds_)
        o_lat_max = min(lat_max, dn)
        o_lon_min = max(lon_min, dw)
        o_lon_max = min(lon_max, de)

        if o_lat_min >= o_lat_max or o_lon_min >= o_lon_max:
            continue

        overlap = (o_lat_max - o_lat_min) * (o_lon_max - o_lon_min)
        if overlap > best_overlap:
            best_overlap = overlap
            best_dtm = dtm

    # Require at least 10% coverage of request bbox
    if best_dtm and best_overlap / req_area >= 0.10:
        return best_dtm
    return None


# ── DEM Extraction ──────────────────────────────────────────────


def _extract_hirise_dtm_bbox(
    dtm_props: dict,
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Extract elevation from a HiRISE DTM .IMG file for the given bbox.

    HiRISE DTMs use Mars equirectangular projection. We manually convert
    lat/lon to projected metres using the CRS parameters from the raster.

    Returns (elevation_array, meta_dict).
    """
    dtm_path = _DTM_DIR / dtm_props["dtm_file"]
    ds = rasterio.open(dtm_path)

    # Parse equirectangular CRS parameters from the WKT
    crs_wkt = ds.crs.to_wkt() if ds.crs else ""
    import re
    std_par_m = re.search(r'standard_parallel_1",([-\d.]+)', crs_wkt)
    cen_mer_m = re.search(r'central_meridian",([-\d.]+)', crs_wkt)
    spheroid_m = re.search(r'SPHEROID\["[^"]+",([-\d.]+)', crs_wkt)

    std_parallel = float(std_par_m.group(1)) if std_par_m else 0.0
    central_meridian = float(cen_mer_m.group(1)) if cen_mer_m else 0.0
    mars_radius = float(spheroid_m.group(1)) if spheroid_m else 3389500.0

    # Convert central meridian from 0-360 to -180/180 if needed
    if central_meridian > 180:
        central_meridian -= 360

    cos_std = math.cos(math.radians(std_parallel))

    def geo_to_proj(lat: float, lon: float) -> tuple[float, float]:
        """Geographic (lat, lon in degrees) -> projected (x, y in meters)."""
        x = math.radians(lon - central_meridian) * mars_radius * cos_std
        y = math.radians(lat) * mars_radius
        return x, y

    # Convert bbox corners to projected coordinates
    x_min, y_min = geo_to_proj(lat_min, lon_min)
    x_max, y_max = geo_to_proj(lat_max, lon_max)

    # Ensure min < max
    if x_min > x_max:
        x_min, x_max = x_max, x_min
    if y_min > y_max:
        y_min, y_max = y_max, y_min
    # Convert to pixel coordinates
    inv_transform = ~ds.transform
    col_min_f, row_min_f = inv_transform * (x_min, y_max)  # y_max = top
    col_max_f, row_max_f = inv_transform * (x_max, y_min)  # y_min = bottom

    col_min = max(0, int(col_min_f))
    col_max = min(ds.width, int(col_max_f) + 1)
    row_min = max(0, int(row_min_f))
    row_max = min(ds.height, int(row_max_f) + 1)

    if col_max <= col_min or row_max <= row_min:
        ds.close()
        raise ValueError(f"HiRISE DTM bbox produces empty window")

    # Limit grid size for planner performance.
    # A* on grids >1000x1000 may timeout. Subsample to keep
    # effective resolution ~2-5 m/px (still 40-100x better than MOLA 200m).
    max_dim = 1000
    width = col_max - col_min
    height = row_max - row_min
    step = 1
    if width > max_dim or height > max_dim:
        step = max(-(-width // max_dim), -(-height // max_dim), 1)  # ceiling division
        logger.info(
            "HiRISE DTM subsampling by %dx (%dx%d -> ~%dx%d, ~%.1f m/px)",
            step, height, width, height // step, width // step,
            abs(ds.transform.a) * step,
        )

    window = Window(col_min, row_min, col_max - col_min, row_max - row_min)
    elev = ds.read(1, window=window, out_shape=(
        (row_max - row_min + step - 1) // step,
        (col_max - col_min + step - 1) // step,
    )).astype(np.float32)

    # Mark nodata
    if ds.nodata is not None:
        elev[elev == ds.nodata] = np.nan

    # Pixel sizes in meters (from the DTM's native resolution)
    px_m_ew = abs(ds.transform.a) * step  # meters per pixel in x
    px_m_ns = abs(ds.transform.e) * step  # meters per pixel in y

    # Compute degree equivalents for meta
    mid_lat = (lat_min + lat_max) / 2.0
    mars_circ_lat = 2 * math.pi * MARS_MEAN_RADIUS
    mars_circ_lon = 2 * math.pi * MARS_MEAN_RADIUS * math.cos(math.radians(mid_lat))
    px_deg_ns = px_m_ns / (mars_circ_lat / 360.0) if mars_circ_lat > 0 else 0.001
    px_deg_ew = px_m_ew / (mars_circ_lon / 360.0) if mars_circ_lon > 0 else 0.001

    ds.close()

    meta = {
        "lat_min": lat_min,
        "lat_max": lat_max,
        "lon_min": lon_min,
        "lon_max": lon_max,
        "rows": elev.shape[0],
        "cols": elev.shape[1],
        "px_m_ns": px_m_ns,
        "px_m_ew": px_m_ew,
        "px_deg_ns": px_deg_ns,
        "px_deg_ew": px_deg_ew,
        "dem_source": "HiRISE_DTM",
        "dem_product_id": dtm_props.get("product_id", ""),
        "dem_resolution_m": round(max(px_m_ns, px_m_ew), 2),
    }

    logger.info(
        "HiRISE DTM loaded: %s — %dx%d grid @ %.1f m/px",
        dtm_props.get("product_id", "?"), elev.shape[0], elev.shape[1],
        max(px_m_ns, px_m_ew),
    )

    return elev, meta


def _extract_mola_dem_bbox(
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Extract DEM window from global HRSC/MOLA blend (~200 m/px).

    Returns (elevation_array, meta_dict).
    """
    ds = _get_dem()

    # Normalize longitudes to DEM convention (-180..180)
    lon_min_n = ((lon_min + 180) % 360) - 180
    lon_max_n = ((lon_max + 180) % 360) - 180

    px_deg_ew = abs(ds.transform.a)
    px_deg_ns = abs(ds.transform.e)

    # Convert bbox to pixel coordinates
    col_min = int((lon_min_n - ds.transform.c) / ds.transform.a)
    col_max = int((lon_max_n - ds.transform.c) / ds.transform.a)
    row_min = int((lat_max - ds.transform.f) / ds.transform.e)  # note: e is negative
    row_max = int((lat_min - ds.transform.f) / ds.transform.e)

    # Clamp to raster bounds
    col_min = max(0, min(ds.width - 1, col_min))
    col_max = max(col_min + 1, min(ds.width, col_max + 1))
    row_min = max(0, min(ds.height - 1, row_min))
    row_max = max(row_min + 1, min(ds.height, row_max + 1))

    window = Window(col_min, row_min, col_max - col_min, row_max - row_min)
    elev = ds.read(1, window=window).astype(np.float32)

    # Mark nodata
    if ds.nodata is not None:
        elev[elev == ds.nodata] = np.nan

    # Pixel sizes in meters
    mid_lat = (lat_min + lat_max) / 2.0
    px_m_ns = px_deg_ns * (math.pi / 180.0) * MARS_MEAN_RADIUS
    px_m_ew = px_deg_ew * (math.pi / 180.0) * MARS_MEAN_RADIUS * math.cos(math.radians(mid_lat))
    px_m_ew = max(px_m_ew, 1.0)

    meta = {
        "lat_min": lat_min,
        "lat_max": lat_max,
        "lon_min": lon_min,
        "lon_max": lon_max,
        "rows": elev.shape[0],
        "cols": elev.shape[1],
        "px_m_ns": px_m_ns,
        "px_m_ew": px_m_ew,
        "px_deg_ns": px_deg_ns,
        "px_deg_ew": px_deg_ew,
        "dem_source": "MOLA",
        "dem_resolution_m": round(max(px_m_ns, px_m_ew), 1),
        "row0": row_min,
        "col0": col_min,
    }

    return elev, meta


def _extract_dem_bbox(
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Extract DEM window for a geographic bounding box.

    Multi-resolution strategy:
      1. Search for HiRISE DTM (~1 m/px) covering the bbox
      2. Fall back to global HRSC/MOLA blend (~200 m/px)

    Returns (elevation_array, meta_dict).
    """
    # Try HiRISE DTM first
    hirise = _find_best_hirise_dtm(lat_min, lat_max, lon_min, lon_max)
    if hirise:
        try:
            return _extract_hirise_dtm_bbox(hirise, lat_min, lat_max, lon_min, lon_max)
        except Exception as e:
            logger.warning(
                "HiRISE DTM %s failed, falling back to MOLA: %s",
                hirise.get("product_id", "?"), e,
            )

    # Fallback: global MOLA
    return _extract_mola_dem_bbox(lat_min, lat_max, lon_min, lon_max)

# ── Terrain Derivatives ─────────────────────────────────────────

def _compute_slope(elev: np.ndarray, px_m_ns: float, px_m_ew: float) -> np.ndarray:
    """Compute slope in degrees from elevation grid."""
    dz_dy, dz_dx = np.gradient(elev, px_m_ns, px_m_ew)
    slope_deg = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))
    return slope_deg.astype(np.float32)


def _compute_tri(elev: np.ndarray) -> np.ndarray:
    """Compute Terrain Ruggedness Index (TRI).

    TRI = sum of absolute elevation differences between center pixel
    and its 8 neighbors.  Vectorized via shifted arrays.
    """
    rows, cols = elev.shape
    tri = np.zeros_like(elev, dtype=np.float32)

    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            # Shifted slices
            r_start = max(0, dr)
            r_end = rows + min(0, dr)
            c_start = max(0, dc)
            c_end = cols + min(0, dc)

            r_src_start = max(0, -dr)
            r_src_end = rows + min(0, -dr)
            c_src_start = max(0, -dc)
            c_src_end = cols + min(0, -dc)

            tri[r_src_start:r_src_end, c_src_start:c_src_end] += np.abs(
                elev[r_start:r_end, c_start:c_end]
                - elev[r_src_start:r_src_end, c_src_start:c_src_end]
            )

    return tri


def _compute_laplacian(elev: np.ndarray, px_m_ew: float, px_m_ns: float) -> np.ndarray:
    """Negative Laplacian of elevation surface.

    Positive = concave-up (valleys/crater interiors)
    Large negative = convex-up (crater rims, ridges)
    """
    dz_dy = np.gradient(elev, px_m_ns, axis=0)
    dz_dx = np.gradient(elev, px_m_ew, axis=1)
    d2z_dy2 = np.gradient(dz_dy, px_m_ns, axis=0)
    d2z_dx2 = np.gradient(dz_dx, px_m_ew, axis=1)
    return -(d2z_dx2 + d2z_dy2)


def _detect_hazards(
    neg_laplacian: np.ndarray,
    slope_deg: np.ndarray,
    rover: RoverModel,
) -> np.ndarray:
    """Detect hazardous terrain cells.

    Hazards include:
    - Crater rims (high negative Laplacian curvature)
    - Excessively steep terrain (above rover physical limit)
    """
    # Crater rims: strong convex curvature (negative Laplacian)
    lap_std = np.nanstd(neg_laplacian)
    crater_rim_mask = neg_laplacian < -2.0 * lap_std

    # Impassable slopes
    steep_mask = slope_deg > rover.max_slope_deg

    return (crater_rim_mask | steep_mask).astype(bool)


# ── Cost Computation ────────────────────────────────────────────

def _compute_slope_cost(slope_deg: np.ndarray, rover: RoverModel) -> np.ndarray:
    """Slope cost: 0 at flat, 1.0 at safe_slope, inf above max_slope."""
    cost = np.zeros_like(slope_deg, dtype=np.float32)

    # Linear ramp from 0 to 1 over [0, safe_slope_deg]
    safe = rover.safe_slope_deg
    mask_normal = slope_deg <= safe
    cost[mask_normal] = slope_deg[mask_normal] / safe

    # Above safe but below max: ramp from 1.0 to 5.0 (very expensive)
    mask_caution = (slope_deg > safe) & (slope_deg <= rover.max_slope_deg)
    t = (slope_deg[mask_caution] - safe) / (rover.max_slope_deg - safe)
    cost[mask_caution] = 1.0 + 4.0 * t

    # Above max: impassable
    cost[slope_deg > rover.max_slope_deg] = np.inf

    return cost


def _compute_roughness_cost(tri: np.ndarray, rover: RoverModel) -> np.ndarray:
    """Roughness cost: 0 at smooth, 1.0 at max_roughness_tri."""
    cost = np.clip(tri / rover.max_roughness_tri, 0.0, 1.0)
    return cost.astype(np.float32)


def _compute_elevation_cost(elev: np.ndarray) -> np.ndarray:
    """Elevation cost: normalized 0-1 based on elevation range in window."""
    e_min = np.nanmin(elev)
    e_max = np.nanmax(elev)
    rng = e_max - e_min
    if rng < 1.0:
        return np.zeros_like(elev, dtype=np.float32)
    return ((elev - e_min) / rng).astype(np.float32)


def compute_cost_map(
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
    rover: RoverModel = PERSEVERANCE,
    weights: dict[str, float] | None = None,
) -> CostMapResult:
    """Compute traversability cost map for a geographic bounding box.

    Args:
        lat_min, lat_max: Latitude bounds (degrees)
        lon_min, lon_max: Longitude bounds (degrees)
        rover: Rover model with slope/roughness limits
        weights: Optional dict overriding default cost weights
            Keys: slope, roughness, hazard, elevation (0-1 each)

    Returns:
        CostMapResult with cost_grid, slope_grid, elevation_grid, hazard_mask, meta
    """
    w = {**DEFAULT_WEIGHTS, **(weights or {})}

    # 1. Extract DEM
    elev_raw, meta = _extract_dem_bbox(lat_min, lat_max, lon_min, lon_max)

    # 2. Fill NaN and smooth
    nan_mask = np.isnan(elev_raw)
    fill_val = float(np.nanmean(elev_raw)) if not np.all(nan_mask) else 0.0
    elev_filled = np.where(nan_mask, fill_val, elev_raw).astype(np.float32)
    elev_smooth = gaussian_filter(elev_filled, sigma=1.0).astype(np.float32)

    px_m_ns = meta["px_m_ns"]
    px_m_ew = meta["px_m_ew"]

    # 3. Compute derivatives
    slope_deg = _compute_slope(elev_smooth, px_m_ns, px_m_ew)
    tri = _compute_tri(elev_smooth)
    neg_lap = _compute_laplacian(elev_smooth, px_m_ew, px_m_ns)

    # 4. Detect hazards
    hazard_mask = _detect_hazards(neg_lap, slope_deg, rover)

    # 5. Component costs
    slope_cost = _compute_slope_cost(slope_deg, rover)
    rough_cost = _compute_roughness_cost(tri, rover)
    elev_cost = _compute_elevation_cost(elev_smooth)
    hazard_cost = hazard_mask.astype(np.float32)  # 0 or 1

    # 6. Weighted combination
    cost_grid = (
        w["slope"] * slope_cost
        + w["roughness"] * rough_cost
        + w["hazard"] * hazard_cost
        + w["elevation"] * elev_cost
    ).astype(np.float32)

    # Ensure impassable cells stay impassable
    cost_grid[slope_deg > rover.max_slope_deg] = np.inf
    cost_grid[nan_mask] = np.inf

    # Minimum traversable cost is a small positive value
    finite_mask = np.isfinite(cost_grid)
    if np.any(finite_mask):
        cost_grid[finite_mask] = np.maximum(cost_grid[finite_mask], 0.01)

    logger.info(
        "Cost map: %dx%d grid, %.1f%% traversable, slope range [%.1f, %.1f]°",
        meta["rows"], meta["cols"],
        100.0 * np.sum(np.isfinite(cost_grid)) / cost_grid.size,
        float(np.nanmin(slope_deg)), float(np.nanmax(slope_deg)),
    )

    return CostMapResult(
        cost_grid=cost_grid,
        slope_grid=slope_deg,
        elevation_grid=elev_smooth,
        hazard_mask=hazard_mask,
        meta=meta,
    )


def compute_cost_map_for_route(
    start_lat: float, start_lon: float,
    goal_lat: float, goal_lon: float,
    margin_km: float = 5.0,
    rover: RoverModel = PERSEVERANCE,
    weights: dict[str, float] | None = None,
) -> CostMapResult:
    """Compute cost map with bounding box derived from start/goal + margin.

    The bounding box is the smallest rectangle enclosing both points,
    expanded by margin_km on each side.
    """
    mid_lat = (start_lat + goal_lat) / 2.0
    margin_deg_lat = margin_km / (meters_per_degree_lat(mid_lat) / 1000.0)
    margin_deg_lon = margin_km / (meters_per_degree_lon(mid_lat) / 1000.0)

    lat_min = min(start_lat, goal_lat) - margin_deg_lat
    lat_max = max(start_lat, goal_lat) + margin_deg_lat
    lon_min = min(start_lon, goal_lon) - margin_deg_lon
    lon_max = max(start_lon, goal_lon) + margin_deg_lon

    # Clamp latitude
    lat_min = max(-90.0, lat_min)
    lat_max = min(90.0, lat_max)

    return compute_cost_map(lat_min, lat_max, lon_min, lon_max, rover, weights)


# ── Tile Rendering ──────────────────────────────────────────────

def render_cost_map_tile(cost_result: CostMapResult, tile_size: int = 256) -> bytes:
    """Render cost map as a PNG tile for map overlay.

    Color mapping: green (easy) → yellow (moderate) → red (hard) → black (impassable).

    Returns PNG bytes.
    """
    import matplotlib.colors as mcolors
    from PIL import Image

    cost = cost_result.cost_grid.copy()

    # Normalize finite costs to [0, 1]
    finite = cost[np.isfinite(cost)]
    if len(finite) == 0:
        # All impassable — solid black tile
        rgba = np.zeros((tile_size, tile_size, 4), dtype=np.uint8)
        rgba[:, :, 3] = 180
        return _encode_png(rgba)

    vmin = float(np.percentile(finite, 2))
    vmax = float(np.percentile(finite, 98))
    if vmax - vmin < 0.001:
        vmax = vmin + 1.0

    # Normalize
    norm = np.clip((cost - vmin) / (vmax - vmin), 0.0, 1.0)

    # Custom colormap: green → yellow → red
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "traversability",
        [(0.0, "#22c55e"), (0.4, "#eab308"), (0.7, "#ef4444"), (1.0, "#1e1e1e")],
    )

    # Apply colormap → RGBA float [0,1]
    rgba_float = cmap(norm)  # shape (rows, cols, 4)

    # Mark impassable as transparent black
    impassable = ~np.isfinite(cost)
    rgba_float[impassable] = [0.0, 0.0, 0.0, 0.7]

    # Convert to uint8
    rgba = (rgba_float * 255).astype(np.uint8)

    # Semi-transparent overlay
    rgba[:, :, 3] = np.where(impassable, 180, 160)

    # Resize to tile_size
    img = Image.fromarray(rgba, "RGBA")
    img = img.resize((tile_size, tile_size), Image.Resampling.BILINEAR)

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _encode_png(rgba: np.ndarray) -> bytes:
    """Encode RGBA numpy array as PNG bytes."""
    from PIL import Image

    img = Image.fromarray(rgba, "RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
