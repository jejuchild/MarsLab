"""Near-crater εr inversion via distance-weighted SHARAD track triangulation."""

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from analysis.shared.coordinates import haversine_km

logger = logging.getLogger(__name__)


@dataclass
class NearCraterEpsilonResult:
    """Result of near-crater εr estimation."""

    epsilon_r_est: Optional[float]  # Weighted mean εr, None if no candidates
    epsilon_r_std: Optional[float]  # Weighted standard deviation
    n_used: int  # Number of contributing traces
    candidates: List[Dict]  # Details on each candidate
    quality_note: str  # Human-readable quality assessment


class NearCraterEpsilonEstimator:
    """Estimate εr from SHARAD tracks passing near a crater using distance weighting."""

    @staticmethod
    def estimate(
        crater_lat: float,
        crater_lon: float,
        crater_depth_m: float,
        candidate_product_ids: List[str],
        max_distance_km: float = 10.0,
        distance_weight_sigma_km: float = 3.0,
        min_confidence: float = 0.6,
    ) -> NearCraterEpsilonResult:
        """Estimate εr from nearby SHARAD tracks via depth triangulation.

        Algorithm:
        1. For each candidate product_id:
           a. Load geometry (lat/lon arrays)
           b. Compute min distance from crater center to track polyline
           c. Find nearest trace index
           d. Get detected_peaks (delta_bins) for that trace
           e. If detected: compute TWT, then εr = (c * TWT / (2 * crater_depth_m))^2
           f. Check: distance <= max_distance_km, confidence >= min_confidence
           g. Assign weight w_i = exp(-(distance^2) / (2 * sigma_km^2))
        2. Compute weighted mean: εr = Σ(w_i * εr_i) / Σ(w_i)
        3. Compute weighted std: σ = sqrt(Σ(w_i * (εr_i - εr)^2) / Σ(w_i))
        4. Return result with quality metrics

        Args:
            crater_lat: Crater center latitude (planetocentric or graphic, must match data)
            crater_lon: Crater center longitude (-180..180)
            crater_depth_m: Known crater depth in meters
            candidate_product_ids: List of SHARAD product IDs to search
            max_distance_km: Maximum distance from crater to use (default 10 km)
            distance_weight_sigma_km: Gaussian weight sigma (default 3 km)
            min_confidence: Minimum confidence threshold (default 0.6)

        Returns:
            NearCraterEpsilonResult with epsilon_r_est, std, and diagnostics
        """
        candidates = []
        weights = []
        epsilon_estimates = []

        # Late import to avoid circular deps
        from api.sharad_highres_router import _get_geometry

        for product_id in candidate_product_ids:
            try:
                geom, _ = _get_geometry(product_id)
                track_lats = geom["lat"]
                track_lons = geom["lon"]

                if len(track_lats) < 2:
                    logger.debug(f"Skipping {product_id}: insufficient traces")
                    continue

                # Compute min distance from crater to track polyline
                min_dist_km = float("inf")
                nearest_idx = 0

                for i in range(len(track_lats)):
                    dist = haversine_km(
                        crater_lat, crater_lon, track_lats[i], track_lons[i]
                    )
                    if dist < min_dist_km:
                        min_dist_km = dist
                        nearest_idx = i

                # Filtering: distance and confidence
                if min_dist_km > max_distance_km:
                    logger.debug(
                        f"{product_id}: too far (dist={min_dist_km:.1f} km > {max_distance_km} km)"
                    )
                    continue

                # Note: In real implementation, would load RTE result for confidence
                # For MVP, assume detected = confidence available
                # This is a placeholder; actual integration would query RTE results
                weight = math.exp(-((min_dist_km ** 2) / (2 * distance_weight_sigma_km ** 2)))

                # Placeholder: assume crater_depth and TWT are available
                # Real: would retrieve from RTE result at nearest_idx
                # Simplified: assume crater crossing detected at depth = crater_depth_m
                # Then: εr = (c * TWT / (2 * crater_depth_m))^2
                # For MVP: skip actual computation, just log candidate
                candidate_dict = {
                    "product_id": product_id,
                    "distance_km": round(min_dist_km, 2),
                    "nearest_trace_idx": nearest_idx,
                    "weight": round(weight, 3),
                }
                candidates.append(candidate_dict)

                logger.debug(f"Candidate {product_id}: dist={min_dist_km:.1f} km, weight={weight:.3f}")

            except Exception as e:
                logger.debug(f"Error processing {product_id}: {e}")
                continue

        # Compute weighted mean if candidates available
        if len(candidates) == 0:
            logger.warning("No suitable candidates found for near-crater εr estimation")
            return NearCraterEpsilonResult(
                epsilon_r_est=None,
                epsilon_r_std=None,
                n_used=0,
                candidates=[],
                quality_note="No candidates found within max_distance",
            )

        # Placeholder: compute mean/std from candidates
        # In real implementation, would extract epsilon_r estimates and compute weighted stats
        epsilon_r_est = None  # Would be computed from candidates
        epsilon_r_std = None

        quality_note = f"MVP: {len(candidates)} candidate(s) found. Full integration pending RTE result querying."

        return NearCraterEpsilonResult(
            epsilon_r_est=epsilon_r_est,
            epsilon_r_std=epsilon_r_std,
            n_used=len(candidates),
            candidates=candidates,
            quality_note=quality_note,
        )
