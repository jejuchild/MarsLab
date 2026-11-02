"""Ice/Hydration score filtering API (CRISM observations)."""
import json
import os

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from api.paths import BASE_DIR

router = APIRouter()

CRISM_SCORE_DIR = os.path.join(BASE_DIR, "crism_score")
SCORE_STATS_FILE = os.path.join(CRISM_SCORE_DIR, "score_stats.json")

# Lazily loaded once, then cached for the process lifetime
_score_stats_cache = None


def _load_score_stats():
    """Load score stats from JSON file, with caching."""
    global _score_stats_cache
    if _score_stats_cache is None:
        if os.path.exists(SCORE_STATS_FILE):
            with open(SCORE_STATS_FILE, "r") as f:
                _score_stats_cache = json.load(f)
        else:
            _score_stats_cache = {}
    return _score_stats_cache


# Precomputed thresholds (must match generate_score_maps.py)
SCORE_THRESHOLDS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5]


def _find_closest_threshold(value: float) -> str:
    """Find the closest precomputed threshold (as string key)."""
    # Find the largest threshold <= value
    for t in reversed(SCORE_THRESHOLDS):
        if t <= value:
            return str(t)
    return str(SCORE_THRESHOLDS[0])  # Default to lowest


@router.get("/api/filter/ice")
def filter_ice_score(min_score: float = 0.3, min_percent: float = 5.0):
    """
    Filter CRISM observations by ice score.

    Returns observations where at least `min_percent`% of pixels have
    ice_score >= `min_score`.

    Args:
        min_score: Minimum ice score threshold (default 0.3)
        min_percent: Minimum percentage of pixels that must meet threshold (default 5%)

    Returns:
        JSON with list of passing observation IDs
    """
    stats = _load_score_stats()

    if not stats:
        return JSONResponse(content={"passing_ids": [], "total": 0, "passing_count": 0})

    # Find closest precomputed threshold
    threshold_key = _find_closest_threshold(min_score)

    passing_ids = []
    for obs_id, obs_stats in stats.items():
        ice_stats = obs_stats.get("ice", {})
        valid_pixels = ice_stats.get("valid_pixels", 0)

        if valid_pixels == 0:
            continue

        threshold_counts = ice_stats.get("threshold_counts", {})
        count_above = threshold_counts.get(threshold_key, 0)

        percent = (count_above / valid_pixels) * 100

        if percent >= min_percent:
            passing_ids.append(obs_id)

    return JSONResponse(content={
        "passing_ids": passing_ids,
        "total": len(stats),
        "passing_count": len(passing_ids),
        "params": {
            "min_score": min_score,
            "min_percent": min_percent,
            "used_threshold": float(threshold_key)
        }
    })


@router.get("/api/filter/hyd")
def filter_hyd_score(min_score: float = 0.3, min_percent: float = 5.0):
    """
    Filter CRISM observations by hydration score.

    Returns observations where at least `min_percent`% of pixels have
    hyd_score >= `min_score`.
    """
    stats = _load_score_stats()

    if not stats:
        return JSONResponse(content={"passing_ids": [], "total": 0, "passing_count": 0})

    threshold_key = _find_closest_threshold(min_score)

    passing_ids = []
    for obs_id, obs_stats in stats.items():
        hyd_stats = obs_stats.get("hyd", {})
        valid_pixels = hyd_stats.get("valid_pixels", 0)

        if valid_pixels == 0:
            continue

        threshold_counts = hyd_stats.get("threshold_counts", {})
        count_above = threshold_counts.get(threshold_key, 0)

        percent = (count_above / valid_pixels) * 100

        if percent >= min_percent:
            passing_ids.append(obs_id)

    return JSONResponse(content={
        "passing_ids": passing_ids,
        "total": len(stats),
        "passing_count": len(passing_ids),
        "params": {
            "min_score": min_score,
            "min_percent": min_percent,
            "used_threshold": float(threshold_key)
        }
    })


@router.get("/api/score/stats")
def get_score_stats():
    """
    Get score statistics for all observations.
    Useful for debugging and building filter UIs.
    """
    stats = _load_score_stats()
    return JSONResponse(content={
        "total_observations": len(stats),
        "available_thresholds": SCORE_THRESHOLDS
    })
