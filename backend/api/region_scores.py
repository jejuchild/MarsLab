"""
Real region scoring for the Mars Region Dashboard.

Computes scores from actual data sources:
- Slope: Global HRSC/MOLA DEM (200m/pixel)
- Ice: CRISM score_stats.json + crism_data/index.geojson
- Coverage: Local GeoJSON indices (CRISM, SHARAD_HIGHRES, CTX, HIRISE_DTM)
- Subsurface: SHARAD_HIGHRES track count from local index

Scores are cached to disk after first computation.
"""

import asyncio
import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Query

from .mars_regions import MARS_REGIONS, MarsRegion
from .terrain_router import compute_slope_stats

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/regions", tags=["regions"])

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CACHE_DIR = os.path.join(_BACKEND_DIR, "cache")
_CACHE_FILE = os.path.join(_CACHE_DIR, "region_scores.json")

# Index paths
_CRISM_INDEX = os.path.join(_BACKEND_DIR, "crism_data", "index.geojson")
_CRISM_SCORES = os.path.join(_BACKEND_DIR, "crism_score", "score_stats.json")
_SHARAD_INDEX = os.path.join(_BACKEND_DIR, "sharad_highres_data", "index.geojson")
_CTX_INDEX = os.path.join(_BACKEND_DIR, "ctx_data", "index.geojson")
_HIRISE_DTM_INDEX = os.path.join(_BACKEND_DIR, "hirise_dtm_data", "index.geojson")
_TRR3_DATA_DIR = os.path.join(_BACKEND_DIR, "mineral_cnn_data")


# ─────────────────────────────────────────────────────────────────────
# Geometry helpers (mirror proximity_router.py patterns)
# ─────────────────────────────────────────────────────────────────────

def _point_in_bbox(lat: float, lon: float, region: MarsRegion) -> bool:
    """Check if point is inside region bbox (handles antimeridian wrapping)."""
    return region.contains_point(lat, lon)


def _polygon_centroid(coords: List[List[List[float]]]) -> Tuple[float, float]:
    """Compute centroid of a GeoJSON Polygon (first ring). Returns (lat, lon)."""
    ring = coords[0]
    n = len(ring)
    if n == 0:
        return 0.0, 0.0
    avg_lon = sum(pt[0] for pt in ring) / n
    avg_lat = sum(pt[1] for pt in ring) / n
    return avg_lat, avg_lon


def _polygon_bbox_overlaps(coords: List[List[List[float]]], region: MarsRegion) -> bool:
    """Check if polygon bbox overlaps region bbox (handles antimeridian wrapping)."""
    ring = coords[0]
    lons = [pt[0] for pt in ring]
    lats = [pt[1] for pt in ring]
    poly_lon_min, poly_lon_max = min(lons), max(lons)
    poly_lat_min, poly_lat_max = min(lats), max(lats)
    if poly_lat_max < region.lat_min or poly_lat_min > region.lat_max:
        return False
    if region.lon_min <= region.lon_max:
        # Normal case: no antimeridian wrapping
        if poly_lon_max < region.lon_min or poly_lon_min > region.lon_max:
            return False
    else:
        # Antimeridian wrapping: region spans [lon_min, 180] ∪ [-180, lon_max]
        # Polygon must overlap at least one half
        in_east = poly_lon_max >= region.lon_min   # overlaps [lon_min, 180]
        in_west = poly_lon_min <= region.lon_max   # overlaps [-180, lon_max]
        if not in_east and not in_west:
            return False
    return True


def _liang_barsky_clip(x1: float, y1: float, x2: float, y2: float,
                       xmin: float, ymin: float, xmax: float, ymax: float) -> bool:
    """Liang-Barsky line-clipping: True if segment intersects the rect."""
    dx = x2 - x1
    dy = y2 - y1
    t0, t1 = 0.0, 1.0
    for p, q in [(-dx, x1 - xmin), (dx, xmax - x1),
                 (-dy, y1 - ymin), (dy, ymax - y1)]:
        if p == 0:
            if q < 0:
                return False
        else:
            r = q / p
            if p < 0:
                t0 = max(t0, r)
            else:
                t1 = min(t1, r)
            if t0 > t1:
                return False
    return True


def _line_intersects_bbox(coords: list, region: MarsRegion) -> bool:
    """Check if a LineString intersects a region bbox (handles antimeridian wrapping)."""
    ymin, ymax = region.lat_min, region.lat_max
    wraps = region.lon_min > region.lon_max

    for i in range(len(coords) - 1):
        x1, y1 = coords[i][0], coords[i][1]
        x2, y2 = coords[i + 1][0], coords[i + 1][1]
        if abs(x2 - x1) > 180:
            continue
        if wraps:
            # Split into two rectangles: [lon_min, 180] and [-180, lon_max]
            if (_liang_barsky_clip(x1, y1, x2, y2, region.lon_min, ymin, 180.0, ymax) or
                _liang_barsky_clip(x1, y1, x2, y2, -180.0, ymin, region.lon_max, ymax)):
                return True
        else:
            if _liang_barsky_clip(x1, y1, x2, y2, region.lon_min, ymin, region.lon_max, ymax):
                return True
    return False


# ─────────────────────────────────────────────────────────────────────
# Index loaders
# ─────────────────────────────────────────────────────────────────────

def _load_geojson(path: str) -> List[Dict]:
    """Load features from a GeoJSON file. Returns empty list if missing."""
    if not os.path.exists(path):
        logger.warning("Index not found: %s", path)
        return []
    with open(path, "r") as f:
        data = json.load(f)
    return data.get("features", [])


def _load_score_stats() -> Dict[str, Dict]:
    """Load CRISM ice/hyd score stats keyed by obs_id."""
    if not os.path.exists(_CRISM_SCORES):
        logger.warning("CRISM score_stats.json not found: %s", _CRISM_SCORES)
        return {}
    with open(_CRISM_SCORES, "r") as f:
        return json.load(f)


def _extract_obs_id(product_id: str) -> Optional[str]:
    """Extract CRISM observation ID from product ID."""
    m = re.match(r'^([a-z]{3}[0-9a-f]{8})', product_id.lower())
    return m.group(1) if m else None


# ─────────────────────────────────────────────────────────────────────
# Build CRISM observation map: obs_id → (lat, lon, ice_pct)
# ─────────────────────────────────────────────────────────────────────

def _list_trr3_obs_ids() -> set:
    """Return set of obs_ids that have local TRR3 cubes in mineral_cnn_data/."""
    if not os.path.isdir(_TRR3_DATA_DIR):
        return set()
    trr3_obs = set()
    for name in os.listdir(_TRR3_DATA_DIR):
        if os.path.isdir(os.path.join(_TRR3_DATA_DIR, name)):
            obs_id = _extract_obs_id(name)
            if obs_id:
                trr3_obs.add(obs_id)
    return trr3_obs


def _build_crism_obs_map() -> List[Dict[str, Any]]:
    """
    Join CRISM index geometries with score_stats to produce
    a list of {obs_id, lat, lon, ice_pct, has_score, has_trr3}.
    """
    features = _load_geojson(_CRISM_INDEX)
    score_stats = _load_score_stats()
    trr3_obs_ids = _list_trr3_obs_ids()

    obs_map: List[Dict[str, Any]] = []
    seen_obs: set = set()

    for feat in features:
        props = feat.get("properties", {})
        pid = props.get("product_id", "")
        obs_id = _extract_obs_id(pid)
        if not obs_id or obs_id in seen_obs:
            continue
        seen_obs.add(obs_id)

        geom = feat.get("geometry", {})
        if geom.get("type") != "Polygon" or not geom.get("coordinates"):
            continue

        lat, lon = _polygon_centroid(geom["coordinates"])

        stats = score_stats.get(obs_id, {})
        ice_stats = stats.get("ice", {})
        valid = ice_stats.get("valid_pixels", 0)
        ice_pct = 0.0
        if valid > 0:
            ice_above = ice_stats.get("threshold_counts", {}).get("0.3", 0)
            ice_pct = round((ice_above / valid) * 100, 1)

        obs_map.append({
            "obs_id": obs_id,
            "lat": lat,
            "lon": lon,
            "ice_pct": ice_pct,
            "has_score": bool(stats),
            "has_trr3": obs_id in trr3_obs_ids,
        })

    return obs_map


# ─────────────────────────────────────────────────────────────────────
# Scoring functions
# ─────────────────────────────────────────────────────────────────────

def _compute_slope_score(region: MarsRegion) -> Dict[str, Any]:
    """
    Sample a 3×3 grid across the region bbox with compute_slope_stats().
    Returns {score: int, safety: str, mean_slope: float, favorable_pct: float}.
    """
    n = 3
    grid_results = []

    lat_step = (region.lat_max - region.lat_min) / (n - 1) if n > 1 else 0
    lon_step = (region.lon_max - region.lon_min) / (n - 1) if n > 1 else 0

    for i in range(n):
        for j in range(n):
            g_lat = region.lat_min + i * lat_step
            g_lon = region.lon_min + j * lon_step
            try:
                stats = compute_slope_stats(g_lat, g_lon, radius_m=2000)
                pct_safe = stats["distribution"]["0_3"] + stats["distribution"]["3_5"]
                # Safety classification (mirrors agent_tasks._assess_safety)
                if pct_safe >= 95:
                    safety = "FAVORABLE"
                elif stats["mean_slope"] < 5 and stats["distribution"]["5_plus"] < 10:
                    safety = "MARGINAL"
                else:
                    safety = "UNFAVORABLE"
                grid_results.append({
                    "mean_slope": stats["mean_slope"],
                    "pct_safe": pct_safe,
                    "safety": safety,
                })
            except Exception:
                continue

    if not grid_results:
        return {"score": None, "safety": "UNKNOWN", "mean_slope": None, "favorable_pct": None}

    favorable_count = sum(1 for p in grid_results if p["safety"] == "FAVORABLE")
    marginal_count = sum(1 for p in grid_results if p["safety"] == "MARGINAL")
    total = len(grid_results)
    favorable_frac = favorable_count / total

    # Overall safety label
    if favorable_count > total * 0.6:
        overall = "FAVORABLE"
    elif (favorable_count + marginal_count) > total * 0.5:
        overall = "MARGINAL"
    else:
        overall = "UNFAVORABLE"

    # Continuous score from average pct_safe across grid (0-100)
    avg_pct_safe = sum(p["pct_safe"] for p in grid_results) / total
    score = round(min(100, max(0, avg_pct_safe)))

    mean_slope = round(sum(p["mean_slope"] for p in grid_results) / total, 2)

    return {
        "score": score,
        "safety": overall,
        "mean_slope": mean_slope,
        "favorable_pct": round(favorable_frac * 100, 1),
    }


def _compute_ice_score(
    region: MarsRegion,
    crism_obs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Match CRISM observations to region, compute ice score.
    Returns {score: int|None, crism_count: int, high_ice_count: int, scored_count: int}.
    """
    in_region = [o for o in crism_obs if _point_in_bbox(o["lat"], o["lon"], region)]
    scored = [o for o in in_region if o["has_score"]]
    high_ice = [o for o in scored if o["ice_pct"] >= 5.0]

    if not in_region:
        return {"score": None, "crism_count": 0, "scored_count": 0, "high_ice_count": 0}

    n_high = len(high_ice)

    # Score ladder (CRISM component of normalize_surface_ice)
    if n_high >= 3:
        # Check for spatial clustering (hotspot)
        lats = [o["lat"] for o in high_ice]
        lons = [o["lon"] for o in high_ice]
        lat_spread = max(lats) - min(lats)
        lon_spread = max(lons) - min(lons)
        region_lat_span = region.lat_max - region.lat_min
        region_lon_span = region.lon_max - region.lon_min
        is_clustered = (lat_spread < region_lat_span * 0.5
                        and lon_spread < region_lon_span * 0.5)
        score = 90 if is_clustered else 70
    elif n_high >= 1:
        score = 40
    elif len(scored) > 0:
        # CRISM data exists but no high-ice detections
        score = 10
    else:
        # CRISM footprints exist but no scores available
        score = None

    return {
        "score": score,
        "crism_count": len(in_region),
        "scored_count": len(scored),
        "high_ice_count": n_high,
    }


def _compute_coverage_score(
    region: MarsRegion,
    crism_obs: List[Dict[str, Any]],
    sharad_features: List[Dict],
    ctx_features: List[Dict],
    dtm_features: List[Dict],
) -> Dict[str, Any]:
    """
    Count exact products per instrument type in region.
    Score = (instruments_with_data / 4) * 100.
    Returns per-type product counts for display.
    """
    product_counts: Dict[str, int] = {}

    # CRISM MTRDR (from obs map — each obs_id = 1 MTRDR product)
    crism_in = [o for o in crism_obs if _point_in_bbox(o["lat"], o["lon"], region)]
    mtrdr_count = len(crism_in)
    trr3_count = sum(1 for o in crism_in if o.get("has_trr3"))
    if mtrdr_count > 0:
        product_counts["CRISM_MTRDR"] = mtrdr_count
    if trr3_count > 0:
        product_counts["CRISM_TRR3"] = trr3_count

    # SHARAD_HIGHRES
    sharad_count = 0
    for feat in sharad_features:
        geom = feat.get("geometry", {})
        if geom.get("type") == "LineString" and geom.get("coordinates"):
            if _line_intersects_bbox(geom["coordinates"], region):
                sharad_count += 1
    if sharad_count > 0:
        product_counts["SHARAD_HIGHRES"] = sharad_count

    # CTX
    ctx_count = 0
    for feat in ctx_features:
        geom = feat.get("geometry", {})
        if geom.get("type") == "Polygon" and geom.get("coordinates"):
            if _polygon_bbox_overlaps(geom["coordinates"], region):
                ctx_count += 1
    if ctx_count > 0:
        product_counts["CTX"] = ctx_count

    # HIRISE_DTM
    dtm_count = 0
    for feat in dtm_features:
        geom = feat.get("geometry", {})
        if geom.get("type") == "Polygon" and geom.get("coordinates"):
            if _polygon_bbox_overlaps(geom["coordinates"], region):
                dtm_count += 1
    if dtm_count > 0:
        product_counts["HIRISE_DTM"] = dtm_count

    # Instrument families (CRISM counts as 1 even if MTRDR + TRR3 both present)
    families = set()
    if mtrdr_count > 0 or trr3_count > 0:
        families.add("CRISM")
    if sharad_count > 0:
        families.add("SHARAD_HIGHRES")
    if ctx_count > 0:
        families.add("CTX")
    if dtm_count > 0:
        families.add("HIRISE_DTM")

    score = round((len(families) / 4) * 100)

    return {
        "score": score,
        "product_counts": product_counts,
        "instruments_with_data": sorted(families),
        "instruments_total": 4,
    }


def _compute_subsurface_score(
    region: MarsRegion,
    sharad_features: List[Dict],
) -> Dict[str, Any]:
    """
    Count SHARAD_HIGHRES tracks intersecting region.
    Returns {score: int|None, track_count: int}.
    """
    track_count = 0
    for feat in sharad_features:
        geom = feat.get("geometry", {})
        if geom.get("type") == "LineString" and geom.get("coordinates"):
            if _line_intersects_bbox(geom["coordinates"], region):
                track_count += 1

    if track_count == 0:
        return {"score": None, "track_count": 0}

    # Score ladder based on track density
    if track_count >= 6:
        score = 80
    elif track_count >= 3:
        score = 55
    else:
        score = 30

    return {"score": score, "track_count": track_count}


def _compute_composite(
    slope: Optional[int],
    ice: Optional[int],
    subsurface: Optional[int],
    coverage: Optional[int],
) -> Optional[int]:
    """
    Weighted composite of available scores.
    Weights: subsurface=0.30, ice=0.25, slope=0.25, coverage=0.20.
    Null sub-scores are excluded and weights renormalized.
    """
    parts: List[Tuple[float, float]] = []
    if subsurface is not None:
        parts.append((subsurface, 0.30))
    if ice is not None:
        parts.append((ice, 0.25))
    if slope is not None:
        parts.append((slope, 0.25))
    if coverage is not None:
        parts.append((coverage, 0.20))

    if not parts:
        return None

    total_weight = sum(w for _, w in parts)
    composite = sum(v * (w / total_weight) for v, w in parts)
    return round(composite)


# ─────────────────────────────────────────────────────────────────────
# Main computation
# ─────────────────────────────────────────────────────────────────────

def compute_all_region_scores() -> Dict[str, Any]:
    """Compute scores for all regions. Returns the full response dict."""
    t0 = time.time()
    logger.info("Computing region scores for %d regions...", len(MARS_REGIONS))

    # Load all indices once
    sharad_features = _load_geojson(_SHARAD_INDEX)
    ctx_features = _load_geojson(_CTX_INDEX)
    dtm_features = _load_geojson(_HIRISE_DTM_INDEX)

    # Build CRISM observation map (obs_id → lat/lon/ice_pct)
    crism_obs = _build_crism_obs_map()
    logger.info("Built CRISM obs map: %d observations", len(crism_obs))

    scores: Dict[str, Any] = {}

    for region_id, region in MARS_REGIONS.items():
        # Slope
        slope_data = _compute_slope_score(region)

        # Ice
        ice_data = _compute_ice_score(region, crism_obs)

        # Coverage
        cov_data = _compute_coverage_score(
            region, crism_obs, sharad_features, ctx_features, dtm_features,
        )

        # Subsurface
        sub_data = _compute_subsurface_score(region, sharad_features)

        # Composite
        composite = _compute_composite(
            slope_data["score"],
            ice_data["score"],
            sub_data["score"],
            cov_data["score"],
        )

        scores[region_id] = {
            "ice": ice_data["score"],
            "slope": slope_data["score"],
            "subsurface": sub_data["score"],
            "coverage": cov_data["score"],
            "composite": composite,
            "details": {
                "slope_safety": slope_data.get("safety"),
                "slope_mean": slope_data.get("mean_slope"),
                "slope_favorable_pct": slope_data.get("favorable_pct"),
                "crism_count": ice_data.get("crism_count", 0),
                "scored_count": ice_data.get("scored_count", 0),
                "high_ice_count": ice_data.get("high_ice_count", 0),
                "sharad_track_count": sub_data.get("track_count", 0),
                "instruments_with_data": cov_data.get("instruments_with_data", []),
                "product_counts": cov_data.get("product_counts", {}),
            },
        }

    elapsed = round(time.time() - t0, 1)
    logger.info("Region scores computed in %.1fs", elapsed)

    result = {
        "scores": scores,
        "computed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "computation_time_s": elapsed,
        "regions_scored": len(scores),
        "is_cached": False,
    }

    # Cache to disk
    os.makedirs(_CACHE_DIR, exist_ok=True)
    with open(_CACHE_FILE, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Region scores cached to %s", _CACHE_FILE)

    return result


def _load_cached() -> Optional[Dict[str, Any]]:
    """Load cached scores from disk if available."""
    if not os.path.exists(_CACHE_FILE):
        return None
    try:
        with open(_CACHE_FILE, "r") as f:
            data = json.load(f)
        data["is_cached"] = True
        return data
    except (json.JSONDecodeError, KeyError):
        return None


# ─────────────────────────────────────────────────────────────────────
# API endpoint
# ─────────────────────────────────────────────────────────────────────

@router.get("/scores")
async def get_region_scores(refresh: bool = Query(False)):
    """
    Return real scores for all Mars regions.

    On first call, computes scores from DEM + local indices (may take 30-60s).
    Subsequent calls return cached data instantly.
    Use ?refresh=true to force recomputation.
    """
    if not refresh:
        cached = _load_cached()
        if cached is not None:
            return cached

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, compute_all_region_scores)
    return result
