"""
CRISM ice/hydration evidence for ice probability.

Finds nearest CRISM products with ice and hydration scores,
applies distance-weighted evidence decay, and produces E4 score.
"""

from __future__ import annotations

import json
import logging
import math
import os
from typing import Dict, List, Optional

import numpy as np

from .models import E4Crism

logger = logging.getLogger(__name__)

R_MARS = 3_389_500.0  # metres


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    la1, la2 = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    dlat = la2 - la1
    a = math.sin(dlat / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin(dlon / 2) ** 2
    return R_MARS * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)) / 1000.0


def evaluate_crism_evidence(
    candidate_lat: float,
    candidate_lon: float,
    crism_products: Optional[List[str]] = None,
    search_radius_km: float = 200.0,
    distance_scale_km: float = 50.0,
) -> E4Crism:
    """Evaluate CRISM ice/hydration evidence near a candidate.

    Uses distance-weighted scoring:
      score = max_over_products( ice_score * exp(-dist_km / scale) )

    Args:
        candidate_lat/lon: Location
        crism_products: Specific product IDs; if None, search index
        search_radius_km: Max distance for CRISM search
        distance_scale_km: Exponential decay scale

    Returns:
        E4Crism evidence score
    """
    backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Load CRISM index to get product locations
    crism_index = _load_crism_index(backend_dir)
    if not crism_index:
        return E4Crism(score=0.0, notes="No CRISM index available")

    # Load score stats
    score_stats = _load_score_stats(backend_dir)

    # Find nearby products
    nearby = _find_nearby_crism(
        candidate_lat, candidate_lon,
        crism_index, crism_products,
        search_radius_km,
    )

    if not nearby:
        return E4Crism(
            score=0.0,
            notes=f"No CRISM products found within {search_radius_km} km",
        )

    # Score each product
    best_ice_weighted = 0.0
    best_hyd_weighted = 0.0
    best_ice_raw = 0.0
    best_hyd_raw = 0.0
    min_dist_km = float("inf")
    products_with_scores = 0

    for product in nearby:
        obs_id = product["obs_id"]
        dist_km = product["distance_km"]
        min_dist_km = min(min_dist_km, dist_km)

        # Get ice/hyd scores from score_stats
        stats = score_stats.get(obs_id, {})
        ice_stats = stats.get("ice", {})
        hyd_stats = stats.get("hyd", {})

        # Use mean score as proxy (or fraction above threshold)
        ice_score = ice_stats.get("mean", 0.0)
        hyd_score = hyd_stats.get("mean", 0.0)

        # If no precomputed stats, try to derive from threshold counts
        if ice_score == 0 and ice_stats.get("threshold_counts"):
            counts = ice_stats["threshold_counts"]
            valid = ice_stats.get("valid_pixels", 1)
            # Fraction above 0.3 as proxy
            above = counts.get("0.3", counts.get("0.25", 0))
            ice_score = above / max(valid, 1)

        if hyd_score == 0 and hyd_stats.get("threshold_counts"):
            counts = hyd_stats["threshold_counts"]
            valid = hyd_stats.get("valid_pixels", 1)
            above = counts.get("0.3", counts.get("0.25", 0))
            hyd_score = above / max(valid, 1)

        if ice_score > 0 or hyd_score > 0:
            products_with_scores += 1

        # Distance-weighted
        weight = math.exp(-dist_km / distance_scale_km)
        ice_w = ice_score * weight
        hyd_w = hyd_score * weight

        best_ice_weighted = max(best_ice_weighted, ice_w)
        best_hyd_weighted = max(best_hyd_weighted, hyd_w)
        best_ice_raw = max(best_ice_raw, ice_score)
        best_hyd_raw = max(best_hyd_raw, hyd_score)

    # Final score: ice dominates, hydration adds secondary signal
    score = 0.7 * best_ice_weighted + 0.3 * best_hyd_weighted

    notes = (
        f"Analyzed {len(nearby)} CRISM products ({products_with_scores} with scores). "
        f"Best ice={best_ice_raw:.3f}, best hyd={best_hyd_raw:.3f}, "
        f"nearest={min_dist_km:.0f} km. "
    )

    if best_ice_raw > 0.3:
        notes += "Strong ice spectral signature detected."
    elif best_ice_raw > 0.1:
        notes += "Moderate ice signature — evidence supports ice-consistent mineralogy."
    elif best_hyd_raw > 0.3:
        notes += "Hydration signature present but no strong ice detection."
    else:
        notes += "Weak spectral evidence for ice/hydration."

    return E4Crism(
        score=round(score, 3),
        ice_score=round(best_ice_raw, 3),
        hyd_score=round(best_hyd_raw, 3),
        distance_km=round(min_dist_km, 1) if min_dist_km < float("inf") else 0.0,
        notes=notes,
    )


def _load_crism_index(backend_dir: str) -> List[Dict]:
    """Load CRISM index features with lat/lon centroids."""
    index_path = os.path.join(backend_dir, "crism_data", "index.geojson")
    if not os.path.exists(index_path):
        return []

    with open(index_path, "r") as f:
        data = json.load(f)

    features = []
    for feat in data.get("features", []):
        props = feat.get("properties", {})
        geom = feat.get("geometry", {})
        coords = geom.get("coordinates", [])

        # Compute centroid
        if geom.get("type") == "Polygon" and coords:
            ring = coords[0]
            clat = np.mean([c[1] for c in ring])
            clon = np.mean([c[0] for c in ring])
        elif geom.get("type") == "Point" and len(coords) >= 2:
            clon, clat = coords[0], coords[1]
        else:
            continue

        pid = props.get("product_id", "")
        # Extract observation ID
        parts = pid.lower().split("_")
        obs_id = parts[0] if parts else pid

        features.append({
            "product_id": pid,
            "obs_id": obs_id,
            "lat": float(clat),
            "lon": float(clon),
        })

    return features


def _load_score_stats(backend_dir: str) -> Dict:
    """Load precomputed CRISM score stats."""
    stats_path = os.path.join(backend_dir, "crism_score", "score_stats.json")
    if not os.path.exists(stats_path):
        return {}
    with open(stats_path, "r") as f:
        return json.load(f)


def _find_nearby_crism(
    lat: float, lon: float,
    index: List[Dict],
    specific_products: Optional[List[str]],
    radius_km: float,
) -> List[Dict]:
    """Find CRISM products near a location."""
    nearby = []
    for feat in index:
        if specific_products and feat["product_id"] not in specific_products and feat["obs_id"] not in specific_products:
            continue
        dist = _haversine_km(lat, lon, feat["lat"], feat["lon"])
        if dist <= radius_km:
            nearby.append({**feat, "distance_km": dist})

    nearby.sort(key=lambda x: x["distance_km"])
    return nearby[:50]  # cap
