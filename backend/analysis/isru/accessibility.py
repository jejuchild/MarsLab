"""
ISRU (In-Situ Resource Utilization) accessibility assessment.

Converts SHARAD observables + physics-based εr into excavation feasibility
metrics for Mars ice-mining mission planning.

Key rule: Depth is ONLY computed when physics-based εr is available.
"""
import math
from dataclasses import dataclass, field
from typing import Optional, List

_SPEED_OF_LIGHT = 299_792_458.0  # m/s


@dataclass
class IsruAccessibility:
    """ISRU accessibility assessment for a candidate site."""

    # Depth from physics (None if εr not measured)
    depth_m: Optional[float] = None
    depth_uncertainty_m: Optional[float] = None
    depth_source: str = "not_available"  # "physics_inversion", "terrace_dielectric", "not_available"

    # ISRU classification
    accessibility_category: str = "depth_unknown"
    # Categories: "easy_excavation", "moderate_excavation", "significant_excavation",
    #             "major_infrastructure", "not_practical", "depth_unknown"
    accessibility_score: float = 0.0  # 0.0 - 1.0
    isru_tier: str = "unknown"  # "tier_1", "tier_2", "tier_3", "not_suitable", "unknown"

    # Slope penalty
    slope_penalty_factor: float = 1.0  # 1.0 = no penalty, higher = harder
    slope_stability: str = "stable"  # "stable", "marginal", "unstable"

    # Ice quality indicators
    epsilon_r: Optional[float] = None
    epsilon_r_ci: Optional[List[float]] = None
    ice_purity_estimate: str = "unknown"  # "high_purity_ice", "icy_regolith", "rocky", "unknown"

    # Feasibility notes
    notes: List[str] = field(default_factory=list)


def compute_depth_from_twt(
    twt_us: float,
    epsilon_r: float,
    epsilon_r_sigma: Optional[float] = None,
) -> tuple:
    """
    Convert two-way travel time to depth using measured εr.

    Formula: depth_m = (c * twt_us * 1e-6) / (2 * sqrt(εr))

    Args:
        twt_us: Two-way travel time in microseconds.
        epsilon_r: Measured dielectric constant (must be > 1.0).
        epsilon_r_sigma: 1-sigma uncertainty on εr (optional).

    Returns:
        (depth_m, depth_uncertainty_m) — uncertainty is None if sigma not provided.
    """
    if epsilon_r <= 1.0:
        raise ValueError(f"εr must be > 1.0 (got {epsilon_r})")
    if twt_us <= 0:
        raise ValueError(f"TWT must be > 0 (got {twt_us})")

    twt_s = twt_us * 1e-6
    velocity = _SPEED_OF_LIGHT / math.sqrt(epsilon_r)
    depth_m = (velocity * twt_s) / 2.0

    depth_unc_m = None
    if epsilon_r_sigma is not None and epsilon_r_sigma > 0:
        # Gaussian error propagation:
        # d = c * t / (2 * sqrt(εr))
        # dd/dεr = -c * t / (4 * εr^(3/2))
        dd_deps = _SPEED_OF_LIGHT * twt_s / (4.0 * epsilon_r ** 1.5)
        depth_unc_m = abs(dd_deps * epsilon_r_sigma)

    return depth_m, depth_unc_m


def _classify_ice_purity(epsilon_r: Optional[float]) -> str:
    """Classify ice purity from measured εr."""
    if epsilon_r is None:
        return "unknown"
    if 2.5 <= epsilon_r <= 3.5:
        return "high_purity_ice"
    elif 3.5 < epsilon_r <= 5.0:
        return "icy_regolith"
    elif epsilon_r > 5.0:
        return "rocky"
    elif 1.0 < epsilon_r < 2.5:
        return "high_purity_ice"  # Very low εr suggests clean ice or porous ice
    return "unknown"


def _compute_slope_penalty(mean_slope_deg: float) -> tuple:
    """
    Compute excavation slope penalty and stability classification.

    Returns:
        (penalty_factor, stability_class)
        penalty_factor: 1.0 = no penalty, higher = more difficult
    """
    if mean_slope_deg < 5.0:
        return 1.0, "stable"
    elif mean_slope_deg < 10.0:
        return 1.1, "stable"
    elif mean_slope_deg < 15.0:
        return 1.3, "marginal"
    elif mean_slope_deg < 20.0:
        return 1.6, "marginal"
    else:
        return 2.5, "unstable"


def compute_isru_accessibility(
    twt_us: Optional[float] = None,
    epsilon_r: Optional[float] = None,
    epsilon_r_sigma: Optional[float] = None,
    epsilon_r_source: str = "not_estimated",
    epsilon_r_ci: Optional[List[float]] = None,
    mean_slope_deg: float = 0.0,
    elevation_m: float = 0.0,
    ice_fraction_crism: float = 0.0,
) -> IsruAccessibility:
    """
    Compute ISRU accessibility assessment.

    Only computes depth when physics-based εr is available.
    When εr is not available, returns depth_unknown category.

    Args:
        twt_us: Median two-way travel time from SHARAD (µs). None if no reflectors.
        epsilon_r: Measured dielectric constant. None if not measured.
        epsilon_r_sigma: 1-sigma uncertainty on εr.
        epsilon_r_source: How εr was obtained ("physics_inversion", "terrace_dielectric", etc.)
        epsilon_r_ci: [lo, hi] confidence interval on εr.
        mean_slope_deg: Mean surface slope in degrees.
        elevation_m: Surface elevation in meters.
        ice_fraction_crism: CRISM-derived water ice fraction (0-1).

    Returns:
        IsruAccessibility dataclass with all fields populated.
    """
    result = IsruAccessibility()
    notes = []

    # Step 1: Slope penalty (always computable)
    slope_penalty, slope_stability = _compute_slope_penalty(mean_slope_deg)
    result.slope_penalty_factor = slope_penalty
    result.slope_stability = slope_stability

    if slope_stability == "unstable":
        notes.append(
            f"Slope {mean_slope_deg:.1f} deg exceeds 20 deg — excavation requires "
            f"slope stabilization engineering, significantly increasing mission cost."
        )
    elif slope_stability == "marginal":
        notes.append(
            f"Slope {mean_slope_deg:.1f} deg is marginal for surface operations — "
            f"excavation penalty factor {slope_penalty:.1f}x."
        )

    # Step 2: Ice purity from εr
    result.ice_purity_estimate = _classify_ice_purity(epsilon_r)
    result.epsilon_r = epsilon_r
    result.epsilon_r_ci = epsilon_r_ci

    # Step 3: Compute depth (ONLY if physics-based εr available)
    physics_sources = ("physics_inversion", "terrace_dielectric")
    has_physics_depth = (
        twt_us is not None
        and twt_us > 0
        and epsilon_r is not None
        and epsilon_r > 1.0
        and epsilon_r_source in physics_sources
    )

    if has_physics_depth:
        depth_m, depth_unc_m = compute_depth_from_twt(twt_us, epsilon_r, epsilon_r_sigma)
        result.depth_m = round(depth_m, 2)
        result.depth_uncertainty_m = round(depth_unc_m, 2) if depth_unc_m else None
        result.depth_source = epsilon_r_source

        # Step 4: Classify accessibility based on depth
        if depth_m <= 5.0:
            result.accessibility_category = "easy_excavation"
            result.accessibility_score = 1.0
            result.isru_tier = "tier_1"
            notes.append(
                f"Depth {depth_m:.1f} m — within easy excavation range (<5 m). "
                f"Shallow trenching or auger drilling sufficient."
            )
        elif depth_m <= 10.0:
            result.accessibility_category = "moderate_excavation"
            result.accessibility_score = 0.9
            result.isru_tier = "tier_1"
            notes.append(
                f"Depth {depth_m:.1f} m — moderate excavation (5-10 m). "
                f"Standard rotary drilling with casing achievable."
            )
        elif depth_m <= 20.0:
            result.accessibility_category = "significant_excavation"
            # Linear interpolation: 0.8 at 10m, 0.5 at 20m
            result.accessibility_score = 0.8 - 0.3 * (depth_m - 10.0) / 10.0
            result.isru_tier = "tier_2"
            notes.append(
                f"Depth {depth_m:.1f} m — significant excavation (10-20 m). "
                f"Requires dedicated drilling infrastructure."
            )
        elif depth_m <= 30.0:
            result.accessibility_category = "major_infrastructure"
            result.accessibility_score = 0.5 - 0.3 * (depth_m - 20.0) / 10.0
            result.isru_tier = "tier_3"
            notes.append(
                f"Depth {depth_m:.1f} m — major infrastructure required (20-30 m). "
                f"Not feasible for first-generation ISRU missions."
            )
        else:
            result.accessibility_category = "not_practical"
            result.accessibility_score = max(0.0, 0.2 - 0.01 * (depth_m - 30.0))
            result.isru_tier = "not_suitable"
            notes.append(
                f"Depth {depth_m:.1f} m — not practical for ISRU (>30 m). "
                f"Exceeds projected drilling capability of near-term Mars missions."
            )

        # Apply slope penalty to accessibility score
        result.accessibility_score = round(
            max(0.0, result.accessibility_score / slope_penalty), 4
        )

        # Depth uncertainty note
        if depth_unc_m:
            notes.append(
                f"Depth uncertainty ±{depth_unc_m:.1f} m (1σ from εr={epsilon_r:.2f}±{epsilon_r_sigma:.2f})."
            )
            # If uncertainty pushes depth across a tier boundary, note it
            if depth_m - depth_unc_m <= 10.0 < depth_m:
                notes.append(
                    "Lower uncertainty bound is within ≤10 m ISRU-accessible range."
                )
            elif depth_m + depth_unc_m > 10.0 >= depth_m:
                notes.append(
                    "Upper uncertainty bound extends beyond 10 m ISRU threshold."
                )

        # Ice purity note
        if result.ice_purity_estimate == "high_purity_ice":
            notes.append(
                f"εr={epsilon_r:.2f} consistent with high-purity water ice — "
                f"favorable for direct sublimation extraction."
            )
        elif result.ice_purity_estimate == "icy_regolith":
            notes.append(
                f"εr={epsilon_r:.2f} suggests ice-cemented regolith — "
                f"extraction efficiency reduced, heating/separation required."
            )
        elif result.ice_purity_estimate == "rocky":
            notes.append(
                f"εr={epsilon_r:.2f} indicates rocky material — "
                f"minimal ice content, ISRU water extraction not viable."
            )

    else:
        # No physics-based depth available
        result.depth_m = None
        result.depth_uncertainty_m = None
        result.depth_source = "not_available"
        result.accessibility_category = "depth_unknown"
        result.accessibility_score = 0.0
        result.isru_tier = "unknown"

        if twt_us is not None and twt_us > 0:
            notes.append(
                f"SHARAD reflector detected (TWT={twt_us:.4f} µs) but depth cannot be "
                f"computed without physics-based εr. Dielectric inversion required."
            )
        else:
            notes.append(
                "No SHARAD subsurface reflector detected — depth is unknown. "
                "ISRU accessibility cannot be assessed."
            )

        # If CRISM shows ice, note indirect evidence
        if ice_fraction_crism > 0.02:
            notes.append(
                f"CRISM spectral analysis indicates {ice_fraction_crism:.1%} surface ice fraction. "
                f"Subsurface depth remains unknown without SHARAD+dielectric measurement."
            )

    result.notes = notes
    return result
