"""
SHARAD subsurface reflector evidence for ice probability.

Detects coherent subsurface reflectors and scores them by SNR,
continuity, and proximity to candidate locations.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from .models import E2Reflector

logger = logging.getLogger(__name__)

SPEED_OF_LIGHT = 299_792_458.0
SHARAD_DT_S = 3.75e-8  # s per bin
R_MARS = 3_389_500.0


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance on Mars in km."""
    la1, la2 = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    dlat = la2 - la1
    a = math.sin(dlat / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin(dlon / 2) ** 2
    return R_MARS * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)) / 1000.0


def _detect_reflectors_in_track(
    power: np.ndarray,
    surface: np.ndarray,
    min_depth_bins: int = 20,
    max_depth_bins: int = 200,
    snr_threshold: float = 3.0,
) -> List[Dict]:
    """Detect subsurface reflector segments in a single SHARAD track.

    Returns list of reflector segments with start/end traces, mean SNR,
    mean depth_bins, and continuity (fraction of valid traces).
    """
    n_traces, n_bins = power.shape
    picks = np.full(n_traces, -1, dtype=np.int32)
    snrs = np.zeros(n_traces, dtype=np.float64)

    for t in range(n_traces):
        s = int(surface[t])
        if s < 0:
            continue
        b_lo = s + min_depth_bins
        b_hi = min(n_bins, s + max_depth_bins)
        if b_hi <= b_lo:
            continue
        band = power[t, b_lo:b_hi].astype(np.float64)
        median_bg = float(np.median(band))
        if median_bg <= 0:
            continue
        peak_idx = int(np.argmax(band))
        peak_val = band[peak_idx]
        local_snr = peak_val / median_bg
        if local_snr >= snr_threshold:
            picks[t] = b_lo + peak_idx
            snrs[t] = local_snr

    # Find coherent segments (consecutive valid picks within ±5 bins)
    segments = []
    seg_start = -1
    for t in range(n_traces):
        if picks[t] >= 0:
            if seg_start < 0:
                seg_start = t
        else:
            if seg_start >= 0 and t - seg_start >= 10:
                seg_picks = picks[seg_start:t]
                seg_snrs = snrs[seg_start:t]
                valid = seg_picks >= 0
                segments.append({
                    "start_trace": seg_start,
                    "end_trace": t - 1,
                    "n_traces": int(valid.sum()),
                    "mean_depth_bins": float(np.mean(seg_picks[valid] - surface[seg_start:t][valid])),
                    "mean_snr": float(np.mean(seg_snrs[valid])),
                    "continuity": float(valid.sum() / (t - seg_start)),
                })
            seg_start = -1

    # Handle last segment
    if seg_start >= 0 and n_traces - seg_start >= 10:
        seg_picks = picks[seg_start:]
        seg_snrs = snrs[seg_start:]
        valid = seg_picks >= 0
        if valid.sum() > 0:
            segments.append({
                "start_trace": seg_start,
                "end_trace": n_traces - 1,
                "n_traces": int(valid.sum()),
                "mean_depth_bins": float(np.mean(seg_picks[valid] - surface[seg_start:][valid])),
                "mean_snr": float(np.mean(seg_snrs[valid])),
                "continuity": float(valid.sum() / (n_traces - seg_start)),
            })

    return segments


def evaluate_reflector_evidence(
    candidate_lat: float,
    candidate_lon: float,
    sharad_tracks: Optional[List[str]] = None,
    search_radius_km: float = 200.0,
) -> E2Reflector:
    """Evaluate SHARAD subsurface reflector evidence near a candidate location.

    Args:
        candidate_lat/lon: Candidate location
        sharad_tracks: Specific product IDs; if None, search index
        search_radius_km: Max distance to consider tracks

    Returns:
        E2Reflector evidence score
    """
    import sys
    backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)

    from api.sharad_highres_router import _get_power, _pick_surface, _get_geometry, _centric_to_graphic

    # Find tracks near candidate
    if not sharad_tracks:
        sharad_tracks = _find_nearby_tracks(candidate_lat, candidate_lon, search_radius_km)

    if not sharad_tracks:
        return E2Reflector(
            score=0.0,
            notes="No SHARAD tracks found within search radius",
        )

    all_segments = []
    best_snr = 0.0
    best_continuity = 0.0
    min_dist_km = float("inf")

    for pid in sharad_tracks[:5]:  # limit to 5 tracks
        try:
            power, total_rows = _get_power(pid)
            surface = _pick_surface(pid, power)
            geom, _ = _get_geometry(pid)
            lats = _centric_to_graphic(geom["lat"])
            lons = geom["lon"].copy()
            lons[lons > 180] -= 360

            segments = _detect_reflectors_in_track(power, surface)

            for seg in segments:
                # Distance from candidate to segment midpoint
                mid_trace = (seg["start_trace"] + seg["end_trace"]) // 2
                if mid_trace < len(lats):
                    dist = _haversine_km(candidate_lat, candidate_lon,
                                         float(lats[mid_trace]), float(lons[mid_trace]))
                    seg["distance_km"] = dist
                    seg["product_id"] = pid
                    min_dist_km = min(min_dist_km, dist)
                    best_snr = max(best_snr, seg["mean_snr"])
                    best_continuity = max(best_continuity, seg["continuity"])
                    all_segments.append(seg)

        except Exception as e:
            logger.warning(f"Error processing track {pid}: {e}")
            continue

    if not all_segments:
        return E2Reflector(
            score=0.0,
            notes=f"No subsurface reflectors detected in {len(sharad_tracks)} tracks",
        )

    # Score: combine SNR, continuity, and proximity
    # SNR score: 0-1 mapped from 3-15
    snr_score = min(1.0, max(0.0, (best_snr - 3.0) / 12.0))
    # Continuity score: direct 0-1
    cont_score = best_continuity
    # Distance score: exponential decay
    dist_score = math.exp(-min_dist_km / 100.0) if min_dist_km < float("inf") else 0.0

    score = 0.4 * snr_score + 0.3 * cont_score + 0.3 * dist_score

    n_segs = len(all_segments)
    notes = (
        f"Found {n_segs} reflector segment(s) in {len(sharad_tracks)} tracks. "
        f"Best SNR={best_snr:.1f}, best continuity={best_continuity:.0%}, "
        f"nearest={min_dist_km:.0f} km from candidate."
    )
    if best_snr > 5 and best_continuity > 0.5:
        notes += " Strong coherent subsurface reflector detected — evidence supports subsurface interface."
    elif best_snr > 3:
        notes += " Moderate reflector evidence — subsurface interface possible but not definitive."

    return E2Reflector(
        score=round(score, 3),
        snr_stats={"best_snr": round(best_snr, 2), "best_continuity": round(best_continuity, 3)},
        continuity=round(best_continuity, 3),
        notes=notes,
    )


def _find_nearby_tracks(lat: float, lon: float, radius_km: float) -> List[str]:
    """Search SHARAD_HIGHRES index for tracks near a location."""
    import json
    backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    index_path = os.path.join(backend_dir, "sharad_highres_data", "index.geojson")
    if not os.path.exists(index_path):
        return []

    with open(index_path, "r") as f:
        index = json.load(f)

    tracks = []
    for feat in index.get("features", []):
        props = feat.get("properties", {})
        pid = props.get("product_id", "")
        # Check if any geometry point is within radius
        geom = feat.get("geometry", {})
        coords = geom.get("coordinates", [])
        if geom.get("type") == "LineString" and coords:
            for coord in coords[::max(1, len(coords) // 10)]:
                c_lon, c_lat = coord[0], coord[1]
                dist = _haversine_km(lat, lon, c_lat, c_lon)
                if dist <= radius_km:
                    tracks.append(pid)
                    break

    return tracks
