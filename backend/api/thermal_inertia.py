"""
TES Thermal Inertia Analysis for Agent Scoring.

Loads the global TES thermal inertia map (Putzig & Mellon 2007, 20 ppd)
and queries it for regional analysis. Thermal inertia (TI) is a key
indicator of surface material consolidation:

  Low TI  (< 150):  Fine dust, loose regolith
  Mid TI  (150-300): Mixed, moderately consolidated
  High TI (300-600): Rocky, cemented, or ice-rich
  Very high TI (> 600): Exposed bedrock

High TI combined with ice spectral signatures strongly supports
ice-cemented or near-surface ice deposits.

Data source: TES nightside thermal inertia, 20 ppd (~3 km/pixel)
             Putzig & Mellon (2007), Icarus 191, 68-94.
"""

import os
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR = os.path.join(_BACKEND_DIR, "data")
_TI_NPY_PATH = os.path.join(_DATA_DIR, "tes_thermal_inertia.npy")

# Grid spec for the TES 20ppd dataset
_TES_PPD = 20  # pixels per degree
_TES_COLS = 360 * _TES_PPD   # 7200
_TES_ROWS = 180 * _TES_PPD   # 3600
_TES_LAT_RANGE = (-90.0, 90.0)
_TES_LON_RANGE = (0.0, 360.0)

# Coverage limits: TES TI only reliable between ~60S-60N
_TES_LAT_MIN = -60.0
_TES_LAT_MAX = 60.0

# Lazy-loaded TI grid
_ti_grid: Optional[np.ndarray] = None


def _load_ti_grid() -> Optional[np.ndarray]:
    """Load TES TI numpy array. Returns None if unavailable."""
    global _ti_grid
    if _ti_grid is not None:
        return _ti_grid

    if not os.path.exists(_TI_NPY_PATH):
        logger.warning(f"TES TI data not found at {_TI_NPY_PATH}")
        return None

    try:
        _ti_grid = np.load(_TI_NPY_PATH)
        logger.info(f"Loaded TES TI grid: shape={_ti_grid.shape}, dtype={_ti_grid.dtype}")
        return _ti_grid
    except Exception as e:
        logger.error(f"Failed to load TES TI data: {e}")
        return None


def get_thermal_inertia(lat: float, lon: float) -> Optional[float]:
    """
    Query thermal inertia at a single point.

    Args:
        lat: Latitude in degrees (-90 to 90)
        lon: Longitude in degrees (-180 to 180 or 0 to 360)

    Returns:
        Thermal inertia in J/(m²·K·s^0.5), or None if unavailable.
    """
    grid = _load_ti_grid()
    if grid is None:
        return None

    # Check latitude coverage
    if lat < _TES_LAT_MIN or lat > _TES_LAT_MAX:
        return None

    # Convert lon to 0-360
    lon360 = lon % 360

    # Convert to pixel coordinates
    # Row 0 = +90°N (top), last row = -90°S (bottom)
    row = int((_TES_LAT_RANGE[1] - lat) * _TES_PPD)
    col = int((lon360 - _TES_LON_RANGE[0]) * _TES_PPD)

    # Clamp
    row = max(0, min(row, grid.shape[0] - 1))
    col = max(0, min(col, grid.shape[1] - 1))

    val = float(grid[row, col])

    # Filter invalid values
    if np.isnan(val) or val <= 0 or val > 2000:
        return None

    return val


def sample_region_ti(
    min_lat: float, max_lat: float, min_lon: float, max_lon: float,
    max_samples: int = 100,
) -> dict:
    """
    Sample thermal inertia across a bounding box region.

    Returns dict with statistics and classification.
    """
    grid = _load_ti_grid()
    if grid is None:
        return {
            "available": False,
            "error": "TES thermal inertia data not loaded",
        }

    # Clamp to TES coverage
    eff_min_lat = max(min_lat, _TES_LAT_MIN)
    eff_max_lat = min(max_lat, _TES_LAT_MAX)

    if eff_min_lat >= eff_max_lat:
        return {
            "available": False,
            "error": f"Region outside TES TI coverage ({_TES_LAT_MIN}° to {_TES_LAT_MAX}°)",
        }

    # Convert to 0-360 lon
    lon_min_360 = min_lon % 360
    lon_max_360 = max_lon % 360
    if lon_max_360 <= lon_min_360:
        lon_max_360 += 360

    # Pixel range
    row_start = int((_TES_LAT_RANGE[1] - eff_max_lat) * _TES_PPD)
    row_end = int((_TES_LAT_RANGE[1] - eff_min_lat) * _TES_PPD)
    col_start = int((lon_min_360 - _TES_LON_RANGE[0]) * _TES_PPD)
    col_end = int((lon_max_360 - _TES_LON_RANGE[0]) * _TES_PPD)

    row_start = max(0, min(row_start, grid.shape[0] - 1))
    row_end = max(0, min(row_end, grid.shape[0]))
    col_start = max(0, col_start)
    col_end = min(col_end, grid.shape[1])

    if row_end <= row_start or col_end <= col_start:
        return {"available": False, "error": "Region too small for TI sampling"}

    # Extract sub-grid (handle wrap-around)
    if col_end <= grid.shape[1]:
        sub = grid[row_start:row_end, col_start:col_end]
    else:
        # Wraps around 360
        sub1 = grid[row_start:row_end, col_start:]
        sub2 = grid[row_start:row_end, :col_end - grid.shape[1]]
        sub = np.concatenate([sub1, sub2], axis=1)

    # Flatten and filter valid
    values = sub.flatten()
    valid = values[(values > 0) & (values < 2000) & np.isfinite(values)]

    if len(valid) == 0:
        return {"available": False, "error": "No valid TI data in region"}

    # Subsample if too many
    if len(valid) > max_samples:
        indices = np.random.default_rng(42).choice(len(valid), max_samples, replace=False)
        sampled = valid[indices]
    else:
        sampled = valid

    # Statistics
    ti_mean = float(np.mean(sampled))
    ti_median = float(np.median(sampled))
    ti_std = float(np.std(sampled))
    ti_min = float(np.min(sampled))
    ti_max = float(np.max(sampled))

    # Percentile distribution
    pct_low = float(np.sum(sampled < 150)) / len(sampled) * 100     # dusty
    pct_mid = float(np.sum((sampled >= 150) & (sampled < 300))) / len(sampled) * 100  # mixed
    pct_high = float(np.sum((sampled >= 300) & (sampled < 600))) / len(sampled) * 100  # consolidated
    pct_vhigh = float(np.sum(sampled >= 600)) / len(sampled) * 100  # bedrock

    # Classification
    if ti_median >= 400:
        classification = "ice-cemented or rocky"
    elif ti_median >= 300:
        classification = "consolidated regolith"
    elif ti_median >= 200:
        classification = "moderately consolidated"
    elif ti_median >= 150:
        classification = "mixed dust and rock"
    else:
        classification = "fine dust"

    return {
        "available": True,
        "ti_mean": round(ti_mean, 1),
        "ti_median": round(ti_median, 1),
        "ti_std": round(ti_std, 1),
        "ti_min": round(ti_min, 1),
        "ti_max": round(ti_max, 1),
        "classification": classification,
        "valid_pixels": len(valid),
        "sampled_pixels": len(sampled),
        "distribution_pct": {
            "dusty_lt150": round(pct_low, 1),
            "mixed_150_300": round(pct_mid, 1),
            "consolidated_300_600": round(pct_high, 1),
            "bedrock_gt600": round(pct_vhigh, 1),
        },
    }


def thermal_inertia_score(ti_stats: dict, has_ice_signal: bool = False) -> tuple:
    """
    Compute thermal inertia contribution to agent score (0-10).

    Args:
        ti_stats: Output from sample_region_ti()
        has_ice_signal: Whether CRISM/SHARAD detected ice

    Returns:
        (score, explanation_string)
    """
    if not ti_stats.get("available"):
        return 0, "Thermal inertia data unavailable"

    ti_median = ti_stats.get("ti_median", 0)
    classification = ti_stats.get("classification", "unknown")

    if ti_median >= 300 and has_ice_signal:
        # Strong corroboration: high TI + ice signals
        return 10, (
            f"High thermal inertia (TI={ti_median:.0f}) consistent with "
            f"ice-cemented regolith — strongly corroborates ice detections"
        )
    elif ti_median >= 300:
        # High TI but no ice signal — consolidated surface, good for landing
        return 5, (
            f"High thermal inertia (TI={ti_median:.0f}, {classification}) — "
            f"consolidated surface favorable for operations"
        )
    elif ti_median >= 150:
        # Mixed — moderate confidence
        return 3, (
            f"Moderate thermal inertia (TI={ti_median:.0f}, {classification}) — "
            f"mixed surface material"
        )
    else:
        # Dusty — low confidence in any ice detection
        score = 1 if has_ice_signal else 0
        return score, (
            f"Low thermal inertia (TI={ti_median:.0f}, {classification}) — "
            f"fine dust cover may mask subsurface ice or indicate absence"
        )


def thermal_inertia_analysis_for_region(
    min_lat: float, max_lat: float, min_lon: float, max_lon: float,
    has_ice_signal: bool = False,
) -> dict:
    """
    Full thermal inertia analysis for a bounding box region.
    Returns a dict suitable for TaskResult.data.
    """
    ti_stats = sample_region_ti(min_lat, max_lat, min_lon, max_lon)
    score, explanation = thermal_inertia_score(ti_stats, has_ice_signal)

    return {
        **ti_stats,
        "ti_score": score,
        "ti_explanation": explanation,
        "has_ice_signal": has_ice_signal,
    }
