"""
Dielectric constant back-calculation from terraced crater depth + SHARAD travel time.

εr = (c * t / (2 * depth_true))²

where:
  c = speed of light (299,792,458 m/s)
  t = SHARAD two-way travel time (seconds)
  depth_true = DTM-derived terrace depth (meters)
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

SPEED_OF_LIGHT = 299792458.0  # m/s

# Quality weights for aggregation (Phase 2)
_QUALITY_WEIGHTS: Dict[str, float] = {
    "good": 1.0,
    "marginal": 0.4,
    "unreliable": 0.1,
}


@dataclass
class EpsilonEstimate:
    """A single dielectric constant estimate."""
    crater_id: str
    depth_metric: str         # e.g., 'terrace_to_floor', 'upper_to_lower_terrace'
    depth_true_m: float
    depth_unc_m: float
    twt_us: float             # Two-way travel time in µs
    twt_unc_us: float
    epsilon_r: float          # Best estimate
    epsilon_low: float        # Lower bound (max depth, min t)
    epsilon_high: float       # Upper bound (min depth, max t)
    interpretation: str       # Material classification
    quality: str = "good"     # "good", "marginal", "unreliable"
    quality_note: str = ""
    sharad_product: str = ""
    n_sharad_picks: int = 0


@dataclass
class EpsilonResult:
    """Full result of epsilon estimation for one or more craters."""
    estimates: List[EpsilonEstimate] = field(default_factory=list)
    summary: str = ""
    error: Optional[str] = None
    aggregate: Optional[Dict] = None  # Phase 2: quality-weighted aggregate


def _interpret_epsilon(eps: float) -> str:
    """Classify material based on dielectric constant."""
    if eps < 2.0:
        return "very porous / dry dust (unlikely subsurface interface)"
    elif eps < 2.8:
        return "porous ice or dry regolith"
    elif eps < 3.5:
        return "consistent with water ice (εr~3.1 for pure ice)"
    elif eps < 5.0:
        return "ice-cemented regolith or ice-rock mixture"
    elif eps < 8.0:
        return "basaltic regolith or dense rock"
    else:
        return "very dense material (metallic content or measurement artifact)"


def compute_epsilon(
    crater_id: str,
    depth_true_m: float,
    depth_unc_m: float,
    twt_us: float,
    twt_unc_us: float,
    depth_metric: str = "terrace_to_floor",
    sharad_product: str = "",
    n_picks: int = 0,
) -> Optional[EpsilonEstimate]:
    """
    Back-calculate dielectric constant from a single depth/twt pair.

    εr = (c * t / (2 * d))²

    Uncertainty propagation uses min/max bounds:
    - ε_low = (c * (t - δt) / (2 * (d + δd)))²   (fastest wave → lowest εr)
    - ε_high = (c * (t + δt) / (2 * (d - δd)))²   (slowest wave → highest εr)
    """
    if depth_true_m <= 0 or twt_us <= 0:
        return None

    t_s = twt_us * 1e-6  # Convert µs to seconds
    dt_s = twt_unc_us * 1e-6

    # Central estimate
    epsilon_r = (SPEED_OF_LIGHT * t_s / (2.0 * depth_true_m)) ** 2

    # Uncertainty bounds
    t_low = max(t_s - dt_s, t_s * 0.5)  # Don't go below half
    t_high = t_s + dt_s
    d_low = max(depth_true_m - depth_unc_m, depth_true_m * 0.5)
    d_high = depth_true_m + depth_unc_m

    epsilon_low = (SPEED_OF_LIGHT * t_low / (2.0 * d_high)) ** 2
    epsilon_high = (SPEED_OF_LIGHT * t_high / (2.0 * d_low)) ** 2

    # Quality assessment
    quality = "good"
    quality_note = ""
    if epsilon_r > 10:
        quality = "unreliable"
        quality_note = "εr > 10: terrace boundary likely not same as SHARAD reflector"
    elif epsilon_r > 6:
        quality = "marginal"
        quality_note = "εr > 6: possible mismatch between terrace and SHARAD interface"
    elif epsilon_r < 1.5:
        quality = "marginal"
        quality_note = "εr < 1.5: unrealistically low, possible depth overestimate"

    # Uncertainty quality
    if epsilon_high > 0 and epsilon_low > 0:
        ratio = epsilon_high / epsilon_low
        if ratio > 10:
            quality = "marginal" if quality == "good" else quality
            quality_note += "; wide uncertainty bounds (10x range)"

    # Reject physically impossible
    if epsilon_r > 50:
        return None

    return EpsilonEstimate(
        crater_id=crater_id,
        depth_metric=depth_metric,
        depth_true_m=round(depth_true_m, 1),
        depth_unc_m=round(depth_unc_m, 1),
        twt_us=round(twt_us, 4),
        twt_unc_us=round(twt_unc_us, 4),
        epsilon_r=round(epsilon_r, 2),
        epsilon_low=round(epsilon_low, 2),
        epsilon_high=round(epsilon_high, 2),
        interpretation=_interpret_epsilon(epsilon_r),
        quality=quality,
        quality_note=quality_note,
        sharad_product=sharad_product,
        n_sharad_picks=n_picks,
    )


def quality_weighted_aggregate(
    estimates: List[EpsilonEstimate],
) -> Optional[Dict]:
    """
    Compute quality-weighted aggregate εr from multiple estimates.

    Weighting scheme (Phase 2):
      - Quality weight: good=1.0, marginal=0.4, unreliable=0.1
      - Pick-count weight: log2(n_picks + 1) / log2(max_picks + 1)
      - Depth-uncertainty weight: 1 / (1 + depth_unc_m / depth_true_m)
      - Combined weight = quality_w * pick_w * depth_unc_w

    Returns dict with weighted_mean_epsilon_r, weighted_median_epsilon_r,
    confidence_interval_68, confidence_interval_95, per-estimate weights,
    or None if no valid estimates.
    """
    if not estimates:
        return None

    # Filter out unreliable-only if we have better ones
    non_unreliable = [e for e in estimates if e.quality != "unreliable"]
    pool = non_unreliable if non_unreliable else estimates

    max_picks = max(e.n_sharad_picks for e in pool) if pool else 1
    log_max = math.log2(max_picks + 1) if max_picks > 0 else 1.0

    weights = []
    eps_vals = []
    per_estimate = []

    for est in pool:
        q_w = _QUALITY_WEIGHTS.get(est.quality, 0.1)
        pick_w = math.log2(est.n_sharad_picks + 1) / log_max if log_max > 0 else 1.0
        depth_rel_unc = est.depth_unc_m / est.depth_true_m if est.depth_true_m > 0 else 1.0
        depth_w = 1.0 / (1.0 + depth_rel_unc)
        combined_w = q_w * pick_w * depth_w

        weights.append(combined_w)
        eps_vals.append(est.epsilon_r)
        per_estimate.append({
            "crater_id": est.crater_id,
            "depth_metric": est.depth_metric,
            "epsilon_r": est.epsilon_r,
            "quality": est.quality,
            "weight": round(combined_w, 4),
            "quality_weight": q_w,
            "pick_weight": round(pick_w, 4),
            "depth_weight": round(depth_w, 4),
        })

    total_w = sum(weights)
    if total_w == 0:
        return None

    # Weighted mean
    weighted_mean = sum(e * w for e, w in zip(eps_vals, weights)) / total_w

    # Weighted median (sort by eps, find cumulative weight midpoint)
    sorted_pairs = sorted(zip(eps_vals, weights))
    cum_w = 0.0
    half_w = total_w / 2.0
    weighted_median = sorted_pairs[-1][0]  # fallback
    for eps, w in sorted_pairs:
        cum_w += w
        if cum_w >= half_w:
            weighted_median = eps
            break

    # Confidence intervals via weighted variance
    weighted_var = sum(w * (e - weighted_mean) ** 2 for e, w in zip(eps_vals, weights)) / total_w
    weighted_std = math.sqrt(weighted_var) if weighted_var > 0 else 0.0

    # Also use min/max of estimate bounds for hard CI
    all_lows = [e.epsilon_low for e in pool]
    all_highs = [e.epsilon_high for e in pool]
    bound_low = min(all_lows) if all_lows else weighted_mean
    bound_high = max(all_highs) if all_highs else weighted_mean

    ci_68 = (
        round(max(weighted_mean - weighted_std, bound_low), 2),
        round(min(weighted_mean + weighted_std, bound_high), 2),
    )
    ci_95 = (
        round(max(weighted_mean - 2 * weighted_std, bound_low), 2),
        round(min(weighted_mean + 2 * weighted_std, bound_high), 2),
    )

    return {
        "weighted_mean_epsilon_r": round(weighted_mean, 2),
        "weighted_median_epsilon_r": round(weighted_median, 2),
        "weighted_std": round(weighted_std, 3),
        "n_estimates": len(pool),
        "n_good": sum(1 for e in pool if e.quality == "good"),
        "n_marginal": sum(1 for e in pool if e.quality == "marginal"),
        "confidence_interval_68": ci_68,
        "confidence_interval_95": ci_95,
        "interpretation": _interpret_epsilon(weighted_median),
        "per_estimate_weights": per_estimate,
    }


def run_epsilon_analysis(
    terrace_results: list,
    sharad_pick_results: list,
    min_depth_m: float = 5.0,
) -> EpsilonResult:
    """
    Combine terrace depth measurements with SHARAD picks to estimate εr.

    Args:
        terrace_results: List of TerraceResult objects
        sharad_pick_results: List of (TerraceResult, SharadPickResult) tuples
        min_depth_m: Minimum terrace depth to consider (default 5m, lowered from 10m in Phase 2)
    """
    result = EpsilonResult()
    estimates = []

    for terrace_result, pick_result in sharad_pick_results:
        if pick_result.error or not pick_result.picks:
            continue
        if not terrace_result.terrace_depths:
            continue

        crater_id = terrace_result.dtm_file.split('/')[-1].replace('.IMG', '')
        twt_us = pick_result.median_twt_us
        twt_unc = pick_result.twt_unc_us

        # Try each depth metric
        for depth_info in terrace_result.terrace_depths:
            depth_m = depth_info['depth_m']
            unc_m = depth_info['uncertainty_m']
            metric = depth_info['metric']

            if depth_m < min_depth_m:
                continue

            est = compute_epsilon(
                crater_id=crater_id,
                depth_true_m=depth_m,
                depth_unc_m=unc_m,
                twt_us=twt_us,
                twt_unc_us=twt_unc,
                depth_metric=metric,
                sharad_product=pick_result.picks[0].product_id if pick_result.picks else "",
                n_picks=len(pick_result.picks),
            )
            if est is not None:
                estimates.append(est)

    result.estimates = estimates

    if estimates:
        # Quality-weighted aggregation (Phase 2)
        agg = quality_weighted_aggregate(estimates)
        if agg:
            result.summary = (
                f"{len(estimates)} εr estimates from {len(sharad_pick_results)} craters. "
                f"Weighted εr = {agg['weighted_median_epsilon_r']:.2f} "
                f"(68% CI: {agg['confidence_interval_68'][0]:.2f}–{agg['confidence_interval_68'][1]:.2f}). "
                f"{agg['interpretation']}"
            )
            result.aggregate = agg
        else:
            eps_vals = [e.epsilon_r for e in estimates]
            mean_eps = sum(eps_vals) / len(eps_vals)
            result.summary = (
                f"{len(estimates)} εr estimates from {len(sharad_pick_results)} craters. "
                f"Mean εr = {mean_eps:.2f} ({_interpret_epsilon(mean_eps)})"
            )
    else:
        result.summary = "No valid εr estimates produced"

    return result
