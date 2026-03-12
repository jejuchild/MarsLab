"""MOLA-pixel-level accessibility scoring for HiRISE image footprints.

Ported from: /disk1/cspark/accessibility_analysis.py

Computes per-MOLA-pixel (200 m) accessibility scores using multi-scale
geomorphometric features and neighbourhood slope analysis.

Scoring weights (total = 10):
    slope_1km   4.0   Primary constraint (local traversability)
    slope_5km   2.0   Regional approach route
    slope_std   1.5   Terrain predictability
    TRI         1.2   Surface ruggedness
    roughness   0.8   Local obstacles
    curvature   0.5   Terrain shape

Neighbourhood categories (5x5 MOLA pixels):
    excellent     ALL neighbours < 5 deg
    good          < 5 % neighbours in 5-10 deg
    moderate      < 10 % in 10-15 deg AND < 1 % at 15 deg+
    challenging   >= 1 % at 15 deg+
    inaccessible  > 20 % at 15 deg+ OR mean slope > 25 deg
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy import ndimage

logger = logging.getLogger(__name__)

DEM_RESOLUTION_M = 200.0
NEIGHBOR_WINDOW = 5


# ── Geomorphometric helpers (from accessibility_analysis.py) ─────────────────

def _compute_slope(elev: np.ndarray, res: float = DEM_RESOLUTION_M) -> np.ndarray:
    if elev.shape[0] < 3 or elev.shape[1] < 3:
        return np.zeros_like(elev)
    dy, dx = np.gradient(elev, res)
    return np.degrees(np.arctan(np.sqrt(dx ** 2 + dy ** 2)))


def _compute_curvature(elev: np.ndarray, res: float = DEM_RESOLUTION_M) -> np.ndarray:
    if elev.shape[0] < 3 or elev.shape[1] < 3:
        return np.zeros_like(elev)
    dy, dx = np.gradient(elev, res)
    dyy, _ = np.gradient(dy, res)
    _, dxx = np.gradient(dx, res)
    return dyy + dxx


def _compute_tri(elev: np.ndarray) -> np.ndarray:
    if elev.shape[0] < 3 or elev.shape[1] < 3:
        return np.zeros_like(elev)
    padded = np.pad(elev, 1, mode="edge")
    h, w = elev.shape
    total = np.zeros_like(elev, dtype=np.float64)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            total += np.abs(padded[1 + dr : h + 1 + dr, 1 + dc : w + 1 + dc] - elev)
    return total / 8.0


def _compute_roughness(elev: np.ndarray) -> np.ndarray:
    return ndimage.maximum_filter(elev, size=3) - ndimage.minimum_filter(elev, size=3)


# ── Core computation ─────────────────────────────────────────────────────────

def compute_mola_accessibility(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    mola_path: str,
    pad_deg: float = 0.5,
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
        Padding in degrees for multi-scale analysis (default 0.5 ~ 30 km).

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

    h, w = dem.shape
    if h < 5 or w < 5:
        logger.warning("MOLA window too small (%d x %d)", h, w)
        empty_s = np.zeros((1, 1), dtype=np.float32)
        empty_c = np.full((1, 1), "insufficient", dtype="<U15")
        return empty_s, empty_c, {"mean_score": 0.0}

    # Fill NaN with median for derivative computation
    med = float(np.nanmedian(dem))
    dem_filled = np.where(np.isnan(dem), med, dem)

    # ── Base geomorphometric maps ──
    slope = _compute_slope(dem_filled).astype(np.float32)
    curvature = _compute_curvature(dem_filled).astype(np.float32)
    tri = _compute_tri(dem_filled).astype(np.float32)
    roughness = _compute_roughness(dem_filled).astype(np.float32)

    # ── Multi-scale statistics via uniform_filter ──
    r_1km = max(1, int(1000 / DEM_RESOLUTION_M))  # 5 pixels
    r_5km = max(1, int(5000 / DEM_RESOLUTION_M))  # 25 pixels
    k_1km = 2 * r_1km + 1  # 11
    k_5km = 2 * r_5km + 1  # 51

    slope_1km = ndimage.uniform_filter(slope, size=k_1km, mode="nearest")
    slope_sq = ndimage.uniform_filter(slope ** 2, size=k_1km, mode="nearest")
    slope_std_1km = np.sqrt(np.maximum(slope_sq - slope_1km ** 2, 0))
    slope_5km = ndimage.uniform_filter(slope, size=k_5km, mode="nearest")
    curvature_1km = ndimage.uniform_filter(np.abs(curvature), size=k_1km, mode="nearest")
    tri_1km = ndimage.uniform_filter(tri, size=k_1km, mode="nearest")
    roughness_1km = ndimage.uniform_filter(roughness, size=k_1km, mode="nearest")

    # ── Normalize (reference thresholds, 2024-03) ──
    slope_norm = np.clip(slope_1km / 15.0, 0, 1)
    slope_5km_norm = np.clip(slope_5km / 15.0, 0, 1)
    slope_std_norm = np.clip(slope_std_1km / 5.0, 0, 1)
    tri_norm = np.clip(tri_1km / 30.0, 0, 1)
    roughness_norm = np.clip(roughness_1km / 50.0, 0, 1)
    curvature_norm = np.clip(curvature_1km / 0.005, 0, 1)

    # ── Weighted penalty (total = 10.0) ──
    penalty = (
        4.0 * slope_norm
        + 2.0 * slope_5km_norm
        + 1.5 * slope_std_norm
        + 1.2 * tri_norm
        + 0.8 * roughness_norm
        + 0.5 * curvature_norm
    ) / 10.0

    raw_scores = (1.0 - np.clip(penalty, 0, 1)).astype(np.float32)

    # ── Neighbourhood category adjustment (5 x 5) ──
    final_scores, categories = _apply_neighborhood_categories(slope, raw_scores)

    # ── Trim padding → core area ──
    total_lat = north - south
    total_lon = east - west
    core_r0 = max(0, int(pad_deg / total_lat * h))
    core_r1 = min(h, h - core_r0)
    core_c0 = max(0, int(pad_deg / total_lon * w))
    core_c1 = min(w, w - core_c0)
    core_r1 = max(core_r1, core_r0 + 1)
    core_c1 = max(core_c1, core_c0 + 1)

    core_scores = final_scores[core_r0:core_r1, core_c0:core_c1]
    core_cats = categories[core_r0:core_r1, core_c0:core_c1]
    core_slope = slope[core_r0:core_r1, core_c0:core_c1]

    # ── Metadata ──
    metadata: dict = {
        "mean_score": round(float(np.nanmean(core_scores)), 4),
        "mean_slope_deg": round(float(np.nanmean(core_slope)), 2),
        "grid_shape": [int(core_scores.shape[0]), int(core_scores.shape[1])],
    }
    total_px = int(core_cats.size)
    for cat in ("excellent", "good", "moderate", "challenging", "inaccessible"):
        n = int(np.sum(core_cats == cat))
        if n > 0:
            metadata[f"n_{cat}"] = n
            metadata[f"pct_{cat}"] = round(n / total_px * 100, 1)

    logger.info(
        "MOLA accessibility: %d x %d grid, mean=%.3f, slope=%.1f deg",
        core_scores.shape[0], core_scores.shape[1],
        metadata["mean_score"], metadata["mean_slope_deg"],
    )
    return core_scores, core_cats, metadata


def _apply_neighborhood_categories(
    slope_map: np.ndarray,
    raw_scores: np.ndarray,
    window_size: int = NEIGHBOR_WINDOW,
) -> tuple[np.ndarray, np.ndarray]:
    """Classify each pixel by neighbourhood slope distribution (5x5).

    Criteria (from accessibility_analysis.py, 2024-03):
        Inaccessible  > 20% at 15+  OR  mean slope > 25 deg
        Challenging    >= 1% at 15+
        Moderate       >= 10% in 10-15 deg  (and < 1% at 15+)
        Good           >= 5% in 5-10 deg
        Excellent      ALL neighbours < 5 deg
    """
    h, w = slope_map.shape
    categories = np.full((h, w), "good", dtype="<U15")
    final_scores = raw_scores.copy() * 0.9  # default: "good" with mild penalty

    half = window_size // 2

    for r in range(h):
        for c in range(w):
            r0, r1 = max(0, r - half), min(h, r + half + 1)
            c0, c1 = max(0, c - half), min(w, c + half + 1)
            nb = slope_map[r0:r1, c0:c1]
            valid = nb[~np.isnan(nb)]
            if len(valid) == 0:
                categories[r, c] = "inaccessible"
                final_scores[r, c] = raw_scores[r, c] * 0.1
                continue

            n = len(valid)
            pct_5_10 = float(np.sum((valid >= 5) & (valid < 10))) / n * 100
            pct_10_15 = float(np.sum((valid >= 10) & (valid < 15))) / n * 100
            pct_15 = float(np.sum(valid >= 15)) / n * 100
            mean_s = float(np.mean(valid))

            if pct_15 > 20 or mean_s > 25:
                categories[r, c] = "inaccessible"
                final_scores[r, c] = raw_scores[r, c] * 0.1
            elif pct_15 >= 1:
                categories[r, c] = "challenging"
                final_scores[r, c] = raw_scores[r, c] * 0.4
            elif pct_10_15 >= 10:
                categories[r, c] = "moderate"
                final_scores[r, c] = raw_scores[r, c] * 0.65
            elif pct_5_10 >= 5:
                categories[r, c] = "good"
                final_scores[r, c] = raw_scores[r, c] * 0.85
            elif np.sum(valid < 5) == n:
                categories[r, c] = "excellent"
                final_scores[r, c] = raw_scores[r, c]  # no penalty
            else:
                categories[r, c] = "good"
                final_scores[r, c] = raw_scores[r, c] * 0.9

    return final_scores, categories
