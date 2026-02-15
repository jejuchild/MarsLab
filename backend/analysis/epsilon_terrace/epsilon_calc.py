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
from typing import List, Optional

SPEED_OF_LIGHT = 299792458.0  # m/s


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


def run_epsilon_analysis(
    terrace_results: list,
    sharad_pick_results: list,
) -> EpsilonResult:
    """
    Combine terrace depth measurements with SHARAD picks to estimate εr.

    Args:
        terrace_results: List of TerraceResult objects
        sharad_pick_results: List of (TerraceResult, SharadPickResult) tuples
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

            if depth_m < 10:  # Skip very shallow depths (< 10m)
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
        eps_vals = [e.epsilon_r for e in estimates]
        mean_eps = sum(eps_vals) / len(eps_vals)
        result.summary = (
            f"{len(estimates)} εr estimates from {len(sharad_pick_results)} craters. "
            f"Mean εr = {mean_eps:.2f} ({_interpret_epsilon(mean_eps)})"
        )
    else:
        result.summary = "No valid εr estimates produced"

    return result
