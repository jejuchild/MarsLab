"""MOLA-pixel-level accessibility scoring for HiRISE image footprints.

Simplified three-factor formula (2025-03):

    MOLA Accessibility = S_elev × S_lat × S_slope

Using a 5 km × 5 km rolling window around each MOLA pixel (~200 m):

    S_elev  — 95th-percentile elevation (km) in the window
              z ≤ -2 km  → 1
              -2 < z < 0 → -z / 2
              z ≥ 0      → 0

    S_slope — mean slope (deg) in the window
              slope ≤ 2°      → 1
              2° < slope < 5° → (5 - slope) / 3
              slope ≥ 5°      → 0

    S_lat   — absolute latitude of the pixel (deg)
              |lat| < 20°       → 1
              20° ≤ |lat| ≤ 30° → (30 - |lat|) / 10
              |lat| > 30°       → 0

Category mapping (from composite score):
    excellent      score ≥ 0.8
    good           0.5 ≤ score < 0.8
    moderate       0.2 ≤ score < 0.5
    challenging    0 < score < 0.2
    inaccessible   score = 0
"""

from __future__ import annotations

import logging
from typing import Any  # noqa: F401 — kept for dict annotation compatibility

import numpy as np
from scipy import ndimage

logger = logging.getLogger(__name__)

DEM_RESOLUTION_M = 200.0
WINDOW_KM = 5.0  # 5 km × 5 km analysis window
WINDOW_PX = max(1, int(WINDOW_KM * 1000 / DEM_RESOLUTION_M))  # ~25 pixels


# ── Sub-score functions (vectorised) ─────────────────────────────────────────

def _score_elev(elev_95_km: np.ndarray) -> np.ndarray:
    """S_elev from 95th-percentile elevation (km).

    z ≤ -2 → 1,  -2 < z < 0 → -z/2,  z ≥ 0 → 0
    """
    s = np.zeros_like(elev_95_km, dtype=np.float32)
    s[elev_95_km <= -2.0] = 1.0
    mid = (elev_95_km > -2.0) & (elev_95_km < 0.0)
    s[mid] = -elev_95_km[mid] / 2.0
    return s


def _score_slope(mean_slope_deg: np.ndarray) -> np.ndarray:
    """S_slope from mean slope (degrees).

    slope ≤ 2° → 1,  2° < slope < 5° → (5-slope)/3,  slope ≥ 5° → 0
    """
    s = np.zeros_like(mean_slope_deg, dtype=np.float32)
    s[mean_slope_deg <= 2.0] = 1.0
    mid = (mean_slope_deg > 2.0) & (mean_slope_deg < 5.0)
    s[mid] = (5.0 - mean_slope_deg[mid]) / 3.0
    return s


def _score_lat(lat_abs: np.ndarray) -> np.ndarray:
    """S_lat from absolute latitude (degrees).

    |lat| < 20° → 1,  20° ≤ |lat| ≤ 30° → (30-|lat|)/10,  |lat| > 30° → 0
    """
    s = np.zeros_like(lat_abs, dtype=np.float32)
    s[lat_abs < 20.0] = 1.0
    mid = (lat_abs >= 20.0) & (lat_abs <= 30.0)
    s[mid] = (30.0 - lat_abs[mid]) / 10.0
    return s


# ── Slope helper ─────────────────────────────────────────────────────────────

def _compute_slope(elev: np.ndarray, res: float = DEM_RESOLUTION_M) -> np.ndarray:
    """Compute slope in degrees from elevation grid."""
    if elev.shape[0] < 3 or elev.shape[1] < 3:
        return np.zeros_like(elev)
    dy, dx = np.gradient(elev, res)
    return np.degrees(np.arctan(np.sqrt(dx ** 2 + dy ** 2)))


# ── Score → category ─────────────────────────────────────────────────────────

def _score_to_category(score: float) -> str:
    if score >= 0.8:
        return "excellent"
    if score >= 0.5:
        return "good"
    if score >= 0.2:
        return "moderate"
    if score > 0:
        return "challenging"
    return "inaccessible"


_score_to_category_vec = np.vectorize(_score_to_category)


# ── Core computation ─────────────────────────────────────────────────────────

def compute_mola_accessibility(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    mola_path: str,
    pad_deg: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Compute per-MOLA-pixel accessibility for a HiRISE footprint.

    Parameters
    ----------
    lat_min, lat_max : float
        Latitude bounds of the HiRISE image (degrees N).
    lon_min, lon_max : float
        Longitude bounds (degrees E, 0-360).
    mola_path : str
        Path to MOLA DEM GeoTIFF.
    pad_deg : float
        Padding in degrees for rolling-window edge handling.

    Returns
    -------
    scores : (H, W) float32
        Accessibility scores [0, 1] for the core (unpadded) area.
    categories : (H, W) str
        Category labels per pixel.
    metadata : dict
        Summary statistics.
    """
    import rasterio
    from rasterio.windows import from_bounds

    # Convert lon to -180..180 for DEM lookup
    lon180_min = ((lon_min + 180) % 360) - 180
    lon180_max = ((lon_max + 180) % 360) - 180
    if lon180_max < lon180_min:
        lon180_max += 360

    west = lon180_min - pad_deg
    east = lon180_max + pad_deg
    south = lat_min - pad_deg
    north = lat_max + pad_deg

    with rasterio.open(mola_path) as src:
        window = from_bounds(west, south, east, north, src.transform)
        dem = src.read(1, window=window).astype(np.float64)
        nodata = src.nodata
        if nodata is not None:
            dem[dem == nodata] = np.nan
        win_transform = src.window_transform(window)

    h, w = dem.shape
    if h < 5 or w < 5:
        logger.warning("MOLA window too small (%d x %d)", h, w)
        empty_s = np.zeros((1, 1), dtype=np.float32)
        empty_c = np.full((1, 1), "insufficient", dtype="<U15")
        return empty_s, empty_c, {"mean_score": 0.0}

    # Fill NaN with median for derivative computation
    med = float(np.nanmedian(dem))
    dem_filled = np.where(np.isnan(dem), med, dem)

    # ── Latitude grid (degrees) ──
    # win_transform: y = f + e * row  (e is negative for north-up rasters)
    lat_per_row = win_transform.f + win_transform.e * (np.arange(h) + 0.5)
    lat_grid = np.broadcast_to(lat_per_row[:, np.newaxis], (h, w)).copy()

    # ── Rolling statistics within 5 km × 5 km window ──
    # 95th percentile elevation (m → km)
    elev_95_m = ndimage.percentile_filter(
        dem_filled, percentile=95, size=WINDOW_PX, mode="nearest",
    )
    elev_95_km = elev_95_m.astype(np.float64) / 1000.0

    # Slope (degrees) then mean slope in window
    slope = _compute_slope(dem_filled).astype(np.float32)
    mean_slope = ndimage.uniform_filter(slope, size=WINDOW_PX, mode="nearest")

    # ── Sub-scores ──
    s_elev = _score_elev(elev_95_km.astype(np.float32))
    s_slope = _score_slope(mean_slope)
    s_lat = _score_lat(np.abs(lat_grid).astype(np.float32))

    # ── Final score = S_elev × S_lat × S_slope ──
    raw_scores = np.clip(s_elev * s_lat * s_slope, 0.0, 1.0).astype(np.float32)

    # ── Categories ──
    categories = _score_to_category_vec(raw_scores).astype("<U15")

    # ── Trim padding → core area ──
    total_lat = north - south
    total_lon = east - west
    core_r0 = max(0, int(pad_deg / total_lat * h))
    core_r1 = min(h, h - core_r0)
    core_c0 = max(0, int(pad_deg / total_lon * w))
    core_c1 = min(w, w - core_c0)
    core_r1 = max(core_r1, core_r0 + 1)
    core_c1 = max(core_c1, core_c0 + 1)

    core_scores = raw_scores[core_r0:core_r1, core_c0:core_c1]
    core_cats = categories[core_r0:core_r1, core_c0:core_c1]
    core_slope = slope[core_r0:core_r1, core_c0:core_c1]
    core_elev = dem_filled[core_r0:core_r1, core_c0:core_c1]
    core_lat = lat_grid[core_r0:core_r1, core_c0:core_c1]

    # ── Metadata ──
    metadata: dict = {
        "mean_score": round(float(np.nanmean(core_scores)), 4),
        "mean_slope_deg": round(float(np.nanmean(core_slope)), 2),
        "mean_elev_m": round(float(np.nanmean(core_elev)), 1),
        "mean_lat_deg": round(float(np.nanmean(core_lat)), 2),
        "grid_shape": [int(core_scores.shape[0]), int(core_scores.shape[1])],
        "formula": "S_elev × S_lat × S_slope",
        "window_km": WINDOW_KM,
    }
    total_px = int(core_cats.size)
    for cat in ("excellent", "good", "moderate", "challenging", "inaccessible"):
        n = int(np.sum(core_cats == cat))
        if n > 0:
            metadata[f"n_{cat}"] = n
            metadata[f"pct_{cat}"] = round(n / total_px * 100, 1)

    logger.info(
        "MOLA accessibility: %d x %d grid, mean=%.3f, slope=%.1f deg, elev=%.0f m",
        core_scores.shape[0], core_scores.shape[1],
        metadata["mean_score"], metadata["mean_slope_deg"],
        metadata["mean_elev_m"],
    )
    return core_scores, core_cats, metadata
