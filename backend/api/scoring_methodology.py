"""
Transparent, Scientifically Defensible Scoring Methodology for Mars Mission Assessment.

This module implements a weighted composite scoring model with explicit
normalization functions, sensitivity analysis, and evidence classification.
It replaces the implicit point-based scoring previously embedded in
synthesize_results and compare_regions.

Importable by:
  - agent_tasks.py  (single-region assessment)
  - report_orchestrator.py  (multi-region comparison)

All sub-scores are normalized to 0.0-1.0.  The composite score is a weighted
linear combination with optional science-distance penalty.

Score = w1 * subsurface_potential + w2 * surface_ice_score
      + w3 * terrain_safety      + w4 * climate_score

No external dependencies beyond the Python standard library.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


# =============================================================================
# Default Weights
# =============================================================================

DEFAULT_WEIGHTS: Dict[str, float] = {
    "w1": 0.30,  # subsurface potential (SHARAD)
    "w2": 0.25,  # surface ice (CRISM spectral + CNN H2O)
    "w3": 0.25,  # terrain safety (slope engineering)
    "w4": 0.20,  # climate suitability
}

WEIGHT_JUSTIFICATION: str = (
    "w1=0.30 (Subsurface Potential): Subsurface ice is the primary mission "
    "objective; SHARAD provides the only direct volumetric ice measurement "
    "capability from orbit, making it the single most discriminating dataset "
    "for landing site selection.\n"
    "w2=0.25 (Surface Ice): Surface ice confirmation via CRISM spectral "
    "analysis and CNN mineral classification provides independent validation "
    "of ice accessibility.  Two complementary detection methods (spectral "
    "indices and deep-learning classification) increase confidence when they "
    "agree.\n"
    "w3=0.25 (Terrain Safety): Terrain safety is a binary gate -- a site must "
    "be landable regardless of science value.  Slope statistics, roughness, "
    "and favorable-zone fraction determine whether EDL (Entry, Descent, and "
    "Landing) is feasible.\n"
    "w4=0.20 (Climate Suitability): Climate affects the operational window "
    "(dust storms, CO2 frost, wind loads) but is less discriminating between "
    "mid-latitude sites because Mars climate varies more with latitude than "
    "with longitude at the same latitude band."
)


# =============================================================================
# Internal helpers
# =============================================================================

def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp *value* to [lo, hi]."""
    return max(lo, min(hi, value))


# =============================================================================
# Sub-score Normalization Functions
# =============================================================================

def normalize_subsurface(
    subsurface_data: Dict[str, Any],
    dielectric_data: Dict[str, Any],
    cross_instrument_data: Dict[str, Any],
    isru_data: Optional[Dict[str, Any]] = None,
) -> Tuple[float, Dict[str, Any]]:
    """Normalize SHARAD subsurface potential to 0-1.

    Scoring ladder (when depth is known from physics-based εr):
      0.0  No SHARAD data
      0.2  SHARAD analyzed, no reflectors detected
      Uses ISRU accessibility_score when physics-based depth available

    Scoring ladder (fallback when no physics εr):
      0.0  No SHARAD data
      0.2  SHARAD analyzed, no reflectors detected
      0.4  Reflectors detected, depth > 80 m (assumed εr)
      0.5  Reflectors detected, depth unknown (no εr at all)
      0.6  Reflectors at 30-80 m depth
      0.8  Reflectors at 10-30 m, multiple tracks
      1.0  Reflectors at <10 m, multiple tracks, dielectric er 2.5-3.5

    Bonuses / penalties (capped to [0, 1]):
      +0.1  cross-instrument consistency == "consistent"
      -0.1  cross-instrument consistency == "inconsistent"

    Parameters
    ----------
    subsurface_data : dict
        ``synthesis["subsurface_coverage"]`` from ``synthesize_results``.
    dielectric_data : dict
        ``synthesis["dielectric_analysis"]`` from ``synthesize_results``.
    cross_instrument_data : dict
        ``synthesis["cross_instrument"]`` from ``synthesize_results``.
    isru_data : dict or None
        ``synthesis["isru_assessment"]`` from ``synthesize_results``.
        When provided with a physics-based depth, ISRU accessibility_score
        is used as the base score for the subsurface component.

    Returns
    -------
    tuple[float, dict]
        (normalized_score, breakdown)
    """
    breakdown: Dict[str, Any] = {
        "base_score": 0.0,
        "depth_category": "no_data",
        "n_detections": 0,
        "dielectric_bonus": 0.0,
        "cross_instrument_adjustment": 0.0,
        "isru_tier": None,
        "isru_accessibility_score": None,
        "notes": [],
    }

    analyzed_count = subsurface_data.get("analyzed_count", 0)
    n_detections = subsurface_data.get("subsurface_detections", 0)
    depth_summary = subsurface_data.get("depth_summary") or {}
    median_depth = depth_summary.get("median_depth_m")
    breakdown["n_detections"] = n_detections

    # --- Base score from SHARAD detection and depth ---
    if analyzed_count == 0 and n_detections == 0:
        base = 0.0
        breakdown["depth_category"] = "no_data"
        breakdown["notes"].append("No SHARAD data available.")
    elif n_detections == 0:
        base = 0.2
        breakdown["depth_category"] = "analyzed_no_reflectors"
        breakdown["notes"].append(
            f"SHARAD analyzed ({analyzed_count} track(s)), no reflectors detected."
        )
    else:
        # We have detections -- check if ISRU provides physics-based depth
        isru_depth = None
        isru_score = None
        isru_tier = None
        if isru_data and isru_data.get("depth_source") in ("physics_inversion", "terrace_dielectric"):
            isru_depth = isru_data.get("depth_m")
            isru_score = isru_data.get("accessibility_score", 0.0)
            isru_tier = isru_data.get("isru_tier", "unknown")

        if isru_depth is not None and isru_score is not None:
            # ISRU-aware scoring: use physics-based depth and accessibility score
            base = max(0.4, isru_score)  # Floor at 0.4 since reflectors were detected
            breakdown["isru_tier"] = isru_tier
            breakdown["isru_accessibility_score"] = isru_score
            cat = isru_data.get("accessibility_category", "unknown")
            breakdown["depth_category"] = f"isru_{cat}"
            breakdown["notes"].append(
                f"Physics-based depth {isru_depth:.1f} m (εr from "
                f"{isru_data['depth_source']}). ISRU tier: {isru_tier}, "
                f"accessibility score: {isru_score:.2f}."
            )
            if isru_tier == "tier_1":
                breakdown["notes"].append(
                    "ISRU Tier 1: Ice within ≤10 m — accessible with standard "
                    "drilling or trenching for near-term ISRU missions."
                )
            elif isru_tier == "tier_2":
                breakdown["notes"].append(
                    "ISRU Tier 2: Ice at 10-20 m — requires dedicated drilling "
                    "infrastructure but feasible for advanced ISRU missions."
                )
            elif isru_tier == "tier_3":
                breakdown["notes"].append(
                    "ISRU Tier 3: Ice at 20-30 m — major infrastructure required, "
                    "not viable for first-generation missions."
                )
        else:
            # Fallback: resolve depth from depth_summary (may be None or from assumed εr)
            depth_m: Optional[float] = None
            if median_depth is not None:
                try:
                    depth_m = float(median_depth)
                except (TypeError, ValueError):
                    depth_m = None

            if depth_m is None:
                # Depth unknown -- give intermediate credit
                base = 0.5 if n_detections >= 2 else 0.4
                breakdown["depth_category"] = "depth_unknown"
                breakdown["notes"].append(
                    f"Reflector(s) detected ({n_detections} track(s)), depth unknown "
                    "(no physics-based εr). ISRU accessibility cannot be assessed."
                )
            elif depth_m > 80:
                base = 0.4
                breakdown["depth_category"] = "deep_>80m"
                breakdown["notes"].append(
                    f"Reflectors detected at ~{depth_m:.0f} m depth -- too deep for "
                    "practical extraction without major drilling infrastructure."
                )
            elif depth_m > 30:
                base = 0.6
                breakdown["depth_category"] = "moderate_30-80m"
                breakdown["notes"].append(
                    f"Reflectors at ~{depth_m:.0f} m depth -- moderate drilling required."
                )
            elif depth_m > 10:
                base = 0.8 if n_detections >= 2 else 0.7
                breakdown["depth_category"] = "shallow_10-30m"
                breakdown["notes"].append(
                    f"Reflectors at ~{depth_m:.0f} m depth with {n_detections} track(s) "
                    "-- accessible with moderate excavation."
                )
            else:
                # <= 10 m
                base = 1.0 if n_detections >= 2 else 0.85
                breakdown["depth_category"] = "very_shallow_<10m"
                breakdown["notes"].append(
                    f"Very shallow reflectors at ~{depth_m:.0f} m with {n_detections} "
                    "track(s) -- highly accessible."
                )

    breakdown["base_score"] = round(base, 4)

    # --- Dielectric bonus (physics inversion preferred over assumed εr) ---
    dielectric_bonus = 0.0
    fallback_penalty = 0.0
    is_physics_inversion = dielectric_data.get("method") in ("physics_inversion", "terraced_crater")
    physics_attempted = dielectric_data.get("physics_attempted", False)
    eps = dielectric_data.get("median_epsilon_r")
    if eps is not None and dielectric_data.get("estimates_count", 0) > 0:
        try:
            eps_f = float(eps)
        except (TypeError, ValueError):
            eps_f = None
        if eps_f is not None:
            if is_physics_inversion:
                # Physics-based εr: stronger bonus/penalty since measurement is independent
                if 2.5 <= eps_f <= 3.5 and n_detections >= 1 and base >= 0.6:
                    dielectric_bonus = min(1.0 - base, 0.15)
                    breakdown["notes"].append(
                        f"PHYSICS-BASED εr={eps_f:.2f} from DTM depth inversion "
                        f"(not assumed). Consistent with ice-rich regolith (εr 2.5-3.5). "
                        f"Enhanced confidence bonus applied."
                    )
                elif eps_f > 5.0:
                    dielectric_bonus = -0.1
                    breakdown["notes"].append(
                        f"PHYSICS-BASED εr={eps_f:.2f} from DTM depth inversion "
                        f"indicates rocky/basaltic subsurface, not ice."
                    )
                else:
                    breakdown["notes"].append(
                        f"PHYSICS-BASED εr={eps_f:.2f} from DTM depth inversion. "
                        f"Intermediate value — ambiguous interpretation."
                    )
                breakdown["notes"].append(
                    "Dielectric constant was COMPUTED via εr = (c·Δt / 2d)², "
                    "not assumed. Depth measured independently from DTM terrace geometry."
                )
            else:
                # Assumed-εr pathway: NO bonus (circular reasoning risk).
                # Assumed εr cannot strengthen ice evidence — it only reflects
                # the assumption itself, not an independent measurement.
                if eps_f > 5.0:
                    breakdown["notes"].append(
                        f"Dielectric constant εr={eps_f:.2f} suggests rocky/basaltic "
                        "subsurface rather than ice."
                    )
                else:
                    breakdown["notes"].append(
                        f"Dielectric constant εr={eps_f:.2f} is ASSUMED (not measured). "
                        "No scoring bonus applied — assumed εr introduces circular "
                        "reasoning risk and cannot independently support ice presence."
                    )
                # Apply fallback penalty if physics inversion was attempted but failed
                if physics_attempted:
                    fallback_penalty = -0.05
                    breakdown["notes"].append(
                        "FALLBACK PENALTY (-0.05): Physics-based dielectric inversion "
                        "was attempted but failed. Depth estimates rely on assumed "
                        "εr=3.15, reducing confidence in ice interpretation."
                    )
    elif n_detections > 0 and not is_physics_inversion:
        # Reflectors detected but no dielectric estimates at all
        breakdown["notes"].append(
            "No dielectric constant estimation was performed. "
            "Depth estimates use assumed εr=3.15 (water-ice) — "
            "this is a non-physical fallback, not evidence of ice."
        )
        if physics_attempted:
            fallback_penalty = -0.05
            breakdown["notes"].append(
                "FALLBACK PENALTY (-0.05): Physics-based inversion attempted but "
                "failed; no independent depth constraint available."
            )

    breakdown["dielectric_bonus"] = round(dielectric_bonus, 4)
    breakdown["fallback_penalty"] = round(fallback_penalty, 4)
    breakdown["dielectric_method"] = dielectric_data.get("method", "assumed") if is_physics_inversion else "assumed"

    # --- Cross-instrument adjustment ---
    consistency = cross_instrument_data.get("evidence_consistency", "insufficient_data")
    cross_adj = 0.0
    if consistency == "consistent":
        cross_adj = 0.1
        breakdown["notes"].append(
            "Cross-instrument consistency bonus: SHARAD + surface ice spatially coincide."
        )
    elif consistency == "inconsistent":
        cross_adj = -0.1
        breakdown["notes"].append(
            "Cross-instrument inconsistency penalty: SHARAD and surface evidence "
            "do not spatially coincide."
        )
    breakdown["cross_instrument_adjustment"] = cross_adj

    score = _clamp(base + dielectric_bonus + fallback_penalty + cross_adj)
    return round(score, 4), breakdown


def normalize_surface_ice(
    mineral_data: Dict[str, Any],
    cnn_data: Dict[str, Any],
) -> Tuple[float, Dict[str, Any]]:
    """Normalize surface ice evidence to 0-1.

    Two independent components, each contributing 0-0.5:

    CRISM spectral component (0-0.5):
      0.0   No ice detected
      0.2   1+ high-ice observations
      0.4   3+ high-ice observations
      0.5   3+ high-ice observations with spatial coherence (ice_hotspot)

    CNN H2O component (0-0.5):
      0.0   No H2O detected
      0.2   Trace H2O (h2o_observations >= 1 but h2o_rich_observations == 0)
      0.35  1+ H2O-rich observations (>= 1% H2O pixels)
      0.5   3+ H2O-rich observations

    Parameters
    ----------
    mineral_data : dict
        ``synthesis["mineral_signatures"]`` from ``synthesize_results``.
    cnn_data : dict
        ``synthesis["cnn_mineral_classification"]`` from ``synthesize_results``.

    Returns
    -------
    tuple[float, dict]
        (normalized_score, breakdown)
    """
    breakdown: Dict[str, Any] = {
        "crism_component": 0.0,
        "cnn_component": 0.0,
        "crism_high_ice_count": 0,
        "cnn_h2o_rich_observations": 0,
        "cnn_h2o_observations": 0,
        "has_spatial_coherence": False,
        "notes": [],
    }

    # --- CRISM spectral component ---
    high_ice_count = mineral_data.get("high_ice_count", 0)
    ice_hotspot = mineral_data.get("ice_hotspot")
    breakdown["crism_high_ice_count"] = high_ice_count

    if high_ice_count >= 3 and ice_hotspot is not None:
        crism_score = 0.5
        breakdown["has_spatial_coherence"] = True
        breakdown["notes"].append(
            f"Strong CRISM ice: {high_ice_count} high-ice observations with "
            "spatial coherence (hotspot detected)."
        )
    elif high_ice_count >= 3:
        crism_score = 0.4
        breakdown["notes"].append(
            f"Good CRISM ice coverage: {high_ice_count} high-ice observations."
        )
    elif high_ice_count >= 1:
        crism_score = 0.2
        breakdown["notes"].append(
            f"Moderate CRISM ice: {high_ice_count} high-ice observation(s)."
        )
    else:
        crism_score = 0.0
        breakdown["notes"].append("No CRISM spectral ice signatures detected.")
    breakdown["crism_component"] = crism_score

    # --- CNN H2O component ---
    h2o_rich_n = cnn_data.get("h2o_rich_observations", 0)
    h2o_obs_n = cnn_data.get("h2o_observations", 0)
    breakdown["cnn_h2o_rich_observations"] = h2o_rich_n
    breakdown["cnn_h2o_observations"] = h2o_obs_n

    if h2o_rich_n >= 3:
        cnn_score = 0.5
        breakdown["notes"].append(
            f"Strong CNN surface H2O ice: {h2o_rich_n} H2O-rich observations "
            "(>= 1% H2O pixels at 95% confidence)."
        )
    elif h2o_rich_n >= 1:
        cnn_score = 0.35
        breakdown["notes"].append(
            f"CNN confirmed surface H2O ice in {h2o_rich_n} observation(s)."
        )
    elif h2o_obs_n >= 1:
        cnn_score = 0.2
        breakdown["notes"].append(
            f"Trace CNN H2O in {h2o_obs_n} observation(s) (below 1% threshold)."
        )
    else:
        cnn_score = 0.0
        if cnn_data.get("observations_classified", 0) > 0:
            breakdown["notes"].append(
                "CNN classified TRR3 data but detected no H2O ice phases."
            )
        else:
            breakdown["notes"].append("No CNN mineral classification data available.")
    breakdown["cnn_component"] = cnn_score

    # --- CRISM Spectral Analysis component (physics-based SAM + band params) ---
    # This adds additional information from the new physics pipeline when available
    spectral_data = mineral_data.get("spectral_analysis", {})
    if spectral_data.get("observations_analyzed", 0) > 0:
        water_frac = spectral_data.get("water_ice_overall_fraction", 0)
        bd1500 = spectral_data.get("mean_band_params", {}).get("BD1500")
        if water_frac > 0.05 and bd1500 is not None and bd1500 > 0.02:
            spectral_bonus = 0.1
            breakdown["notes"].append(
                f"SAM spectral analysis confirms water ice: fraction={water_frac:.1%}, "
                f"BD1500={bd1500:.3f}. Independent physics-based validation."
            )
        elif water_frac > 0.01:
            spectral_bonus = 0.05
            breakdown["notes"].append(
                f"SAM spectral analysis: trace water ice (fraction={water_frac:.1%}). "
                f"Weak but detectable 1.5µm absorption."
            )
        else:
            spectral_bonus = 0.0
            breakdown["notes"].append(
                f"SAM spectral analysis: no significant water ice (fraction={water_frac:.1%})."
            )
        breakdown["spectral_analysis_bonus"] = spectral_bonus
        score = min(1.0, crism_score + cnn_score + spectral_bonus)
    else:
        score = min(1.0, crism_score + cnn_score)

    return round(score, 4), breakdown


def normalize_terrain(
    engineering_data: Dict[str, Any],
) -> Tuple[float, Dict[str, Any]]:
    """Normalize terrain safety to 0-1.

    Scoring:
      1.0  FAVORABLE safety AND >60% grid cells favorable
      0.7  FAVORABLE safety (any favorable-zone fraction)
      0.5  MARGINAL safety
      0.2  UNFAVORABLE safety
      0.0  Unknown / no data

    Roughness penalty: -0.1 for high roughness (if available).

    Parameters
    ----------
    engineering_data : dict
        ``synthesis["engineering_feasibility"]`` from ``synthesize_results``.

    Returns
    -------
    tuple[float, dict]
        (normalized_score, breakdown)
    """
    breakdown: Dict[str, Any] = {
        "safety_class": "UNKNOWN",
        "base_score": 0.0,
        "favorable_fraction": None,
        "roughness_penalty": 0.0,
        "mean_slope": None,
        "max_slope": None,
        "notes": [],
    }

    safety = engineering_data.get("safety", "UNKNOWN")
    breakdown["safety_class"] = safety
    breakdown["mean_slope"] = engineering_data.get("mean_slope")
    breakdown["max_slope"] = engineering_data.get("max_slope")

    # Compute favorable fraction if grid data available
    favorable_zones = engineering_data.get("favorable_zones", 0)
    grid_size = engineering_data.get("grid_size", 0)
    favorable_frac: Optional[float] = None
    if grid_size > 0:
        favorable_frac = favorable_zones / grid_size
        breakdown["favorable_fraction"] = round(favorable_frac, 3)

    if safety == "FAVORABLE":
        if favorable_frac is not None and favorable_frac > 0.6:
            base = 1.0
            breakdown["notes"].append(
                f"FAVORABLE terrain with {favorable_frac:.0%} favorable grid zones -- "
                "excellent landing conditions."
            )
        else:
            base = 0.7
            frac_str = (
                f" ({favorable_frac:.0%} favorable zones)"
                if favorable_frac is not None
                else ""
            )
            breakdown["notes"].append(
                f"FAVORABLE terrain assessment{frac_str}."
            )
    elif safety == "MARGINAL":
        base = 0.5
        breakdown["notes"].append(
            "MARGINAL terrain -- some zones exceed 5 deg slope threshold."
        )
    elif safety == "UNFAVORABLE":
        base = 0.2
        breakdown["notes"].append(
            "UNFAVORABLE terrain -- steep slopes dominate, limiting EDL options."
        )
    else:
        base = 0.0
        breakdown["notes"].append("Slope data unavailable for engineering assessment.")

    breakdown["base_score"] = round(base, 4)

    # Roughness penalty (if roughness info provided)
    roughness = engineering_data.get("roughness")
    roughness_penalty = 0.0
    if roughness is not None:
        try:
            roughness_str = str(roughness).lower()
        except (TypeError, ValueError):
            roughness_str = ""
        if roughness_str in ("high", "very_high"):
            roughness_penalty = -0.1
            breakdown["notes"].append(
                f"Surface roughness is {roughness_str} -- penalty applied."
            )
    breakdown["roughness_penalty"] = roughness_penalty

    score = _clamp(base + roughness_penalty)
    return round(score, 4), breakdown


def normalize_climate(
    climate_data: Dict[str, Any],
) -> Tuple[float, Dict[str, Any]]:
    """Normalize climate suitability to 0-1.

    Uses ``annual_stats`` when available for a formula-based score:

      Temperature  (weight 0.30):  1.0 if 180<=mean<=250 K, 0.5 if 160-180 or 250-270, else 0.0
      Dust         (weight 0.25):  1.0 if tau_mean<0.4, 0.5 if <0.7, else 0.0
      Wind         (weight 0.20):  1.0 if mean<7 m/s, 0.5 if <10, else 0.0
      Frost        (weight 0.25):  1.0 if frost_prob<0.05, 0.5 if <0.3, else 0.0

    Fallback: If ``annual_stats`` is absent, scale the legacy ``climate_score``
    (0-10 integer) to [0, 1].

    Parameters
    ----------
    climate_data : dict
        ``synthesis["climate"]`` from ``synthesize_results``.

    Returns
    -------
    tuple[float, dict]
        (normalized_score, breakdown)
    """
    breakdown: Dict[str, Any] = {
        "temperature_component": 0.0,
        "dust_component": 0.0,
        "wind_component": 0.0,
        "frost_component": 0.0,
        "method": "annual_stats",
        "notes": [],
    }

    if not climate_data:
        breakdown["method"] = "no_data"
        breakdown["notes"].append("No climate data available.")
        return 0.0, breakdown

    annual = climate_data.get("annual_stats", {})

    if annual:
        # --- Temperature ---
        mean_k = annual.get("temp_mean_k")
        if mean_k is not None:
            if 180 <= mean_k <= 250:
                t_raw = 1.0
            elif (160 <= mean_k < 180) or (250 < mean_k <= 270):
                t_raw = 0.5
            else:
                t_raw = 0.0
            breakdown["temperature_component"] = round(0.30 * t_raw, 4)
            breakdown["notes"].append(
                f"Annual mean temperature {mean_k:.0f} K -> raw {t_raw:.1f}"
            )
        else:
            breakdown["notes"].append("Temperature data missing.")

        # --- Dust ---
        tau_mean = annual.get("dust_tau_mean")
        if tau_mean is not None:
            if tau_mean < 0.4:
                d_raw = 1.0
            elif tau_mean < 0.7:
                d_raw = 0.5
            else:
                d_raw = 0.0
            breakdown["dust_component"] = round(0.25 * d_raw, 4)
            breakdown["notes"].append(
                f"Mean dust optical depth tau={tau_mean:.2f} -> raw {d_raw:.1f}"
            )
        else:
            breakdown["notes"].append("Dust optical depth data missing.")

        # --- Wind ---
        wind_mean = annual.get("wind_mean_ms")
        if wind_mean is not None:
            if wind_mean < 7:
                w_raw = 1.0
            elif wind_mean < 10:
                w_raw = 0.5
            else:
                w_raw = 0.0
            breakdown["wind_component"] = round(0.20 * w_raw, 4)
            breakdown["notes"].append(
                f"Mean wind speed {wind_mean:.1f} m/s -> raw {w_raw:.1f}"
            )
        else:
            breakdown["notes"].append("Wind speed data missing.")

        # --- Frost ---
        frost_prob = annual.get("frost_max_probability")
        if frost_prob is not None:
            if frost_prob < 0.05:
                f_raw = 1.0
            elif frost_prob < 0.3:
                f_raw = 0.5
            else:
                f_raw = 0.0
            breakdown["frost_component"] = round(0.25 * f_raw, 4)
            breakdown["notes"].append(
                f"Max frost probability {frost_prob:.2f} -> raw {f_raw:.1f}"
            )
        else:
            breakdown["notes"].append("Frost probability data missing.")

        score = (
            breakdown["temperature_component"]
            + breakdown["dust_component"]
            + breakdown["wind_component"]
            + breakdown["frost_component"]
        )
        score = _clamp(score)
    else:
        # Fallback: legacy climate_score 0-10 -> 0-1
        breakdown["method"] = "legacy_score_fallback"
        legacy = climate_data.get("climate_score", 0)
        try:
            legacy_f = float(legacy)
        except (TypeError, ValueError):
            legacy_f = 0.0
        score = _clamp(legacy_f / 10.0)
        breakdown["notes"].append(
            f"Used legacy climate_score={legacy} (scaled to {score:.2f})."
        )

    return round(score, 4), breakdown


# =============================================================================
# Science-Distance Penalty
# =============================================================================

def apply_science_distance_penalty(
    composite_score: float,
    w1: float,
    w2: float,
    subsurface_score: float,
    surface_ice_score: float,
    distance_to_science_km: float,
) -> Tuple[float, float]:
    """Apply a distance penalty to the science components of the composite score.

    If the science target (nearest SHARAD reflector or ice hotspot) is more
    than 50 km from the landing site, the science-weighted components are
    discounted proportionally.

    ``penalty_factor = max(0.5, 1.0 - distance_km / 200.0)``

    At 50 km the penalty is mild (factor ~0.75); at 200+ km it saturates at
    0.5 (halving the science contribution).

    Parameters
    ----------
    composite_score : float
        Pre-penalty composite score.
    w1, w2 : float
        Weights for subsurface and surface-ice components.
    subsurface_score, surface_ice_score : float
        Normalized sub-scores (0-1).
    distance_to_science_km : float
        Distance from landing site to nearest science target.

    Returns
    -------
    tuple[float, float]
        (adjusted_score, penalty_factor)
    """
    if distance_to_science_km <= 50.0:
        return composite_score, 1.0

    penalty_factor = max(0.5, 1.0 - distance_to_science_km / 200.0)
    science_contribution = w1 * subsurface_score + w2 * surface_ice_score
    reduction = science_contribution * (1.0 - penalty_factor)
    adjusted = max(0.0, composite_score - reduction)
    return round(adjusted, 6), round(penalty_factor, 4)


# =============================================================================
# Composite Score
# =============================================================================

def compute_composite_score(
    subsurface_data: Dict[str, Any],
    dielectric_data: Dict[str, Any],
    cross_instrument_data: Dict[str, Any],
    mineral_data: Dict[str, Any],
    cnn_data: Dict[str, Any],
    engineering_data: Dict[str, Any],
    climate_data: Dict[str, Any],
    science_distance_km: Optional[float] = None,
    weights: Optional[Dict[str, float]] = None,
    isru_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute the full weighted composite score with all sub-scores.

    This is the main entry point for both ``agent_tasks.py`` (single region)
    and ``report_orchestrator.py`` (multi-region comparison).

    Parameters
    ----------
    subsurface_data : dict
        ``synthesis["subsurface_coverage"]``.
    dielectric_data : dict
        ``synthesis["dielectric_analysis"]`` (may be empty ``{}``).
    cross_instrument_data : dict
        ``synthesis["cross_instrument"]``.
    mineral_data : dict
        ``synthesis["mineral_signatures"]``.
    cnn_data : dict
        ``synthesis["cnn_mineral_classification"]`` (may be empty ``{}``).
    engineering_data : dict
        ``synthesis["engineering_feasibility"]``.
    climate_data : dict
        ``synthesis["climate"]`` (may be empty ``{}``).
    science_distance_km : float or None
        Optional distance to nearest science target for penalty.
    weights : dict or None
        Override default weights (keys ``w1``, ``w2``, ``w3``, ``w4``).
    isru_data : dict or None
        ``synthesis["isru_assessment"]`` for ISRU-aware depth scoring.

    Returns
    -------
    dict
        Complete scoring result with sub-scores, weights, justification,
        optional distance penalty, and final score.
    """
    w = dict(DEFAULT_WEIGHTS)
    if weights is not None:
        for key in ("w1", "w2", "w3", "w4"):
            if key in weights:
                w[key] = float(weights[key])

    # Normalize sub-scores
    sub_score, sub_bd = normalize_subsurface(
        subsurface_data, dielectric_data, cross_instrument_data,
        isru_data=isru_data,
    )
    ice_score, ice_bd = normalize_surface_ice(mineral_data, cnn_data)
    terrain_score, terrain_bd = normalize_terrain(engineering_data)
    climate_score, climate_bd = normalize_climate(climate_data)

    # Weighted composite
    composite = (
        w["w1"] * sub_score
        + w["w2"] * ice_score
        + w["w3"] * terrain_score
        + w["w4"] * climate_score
    )
    composite = round(_clamp(composite), 6)

    result: Dict[str, Any] = {
        "composite_score": composite,
        "composite_score_100": round(composite * 100),
        "weights": {
            "w1": w["w1"],
            "w2": w["w2"],
            "w3": w["w3"],
            "w4": w["w4"],
        },
        "weight_justification": WEIGHT_JUSTIFICATION,
        "sub_scores": {
            "subsurface_potential": {"score": round(sub_score, 4), "breakdown": sub_bd},
            "surface_ice": {"score": round(ice_score, 4), "breakdown": ice_bd},
            "terrain_safety": {"score": round(terrain_score, 4), "breakdown": terrain_bd},
            "climate": {"score": round(climate_score, 4), "breakdown": climate_bd},
        },
        "science_distance_penalty": None,
        "final_score": composite,
        "final_score_100": round(composite * 100),
    }

    # Apply distance penalty if applicable
    if science_distance_km is not None and science_distance_km > 50.0:
        adjusted, factor = apply_science_distance_penalty(
            composite, w["w1"], w["w2"], sub_score, ice_score, science_distance_km,
        )
        result["science_distance_penalty"] = {
            "applied": True,
            "distance_km": round(science_distance_km, 1),
            "factor": factor,
        }
        result["final_score"] = adjusted
        result["final_score_100"] = round(adjusted * 100)
    elif science_distance_km is not None:
        result["science_distance_penalty"] = {
            "applied": False,
            "distance_km": round(science_distance_km, 1),
            "factor": 1.0,
        }

    return result


# =============================================================================
# Sensitivity Analysis
# =============================================================================

def sensitivity_analysis(
    subsurface_data: Dict[str, Any],
    dielectric_data: Dict[str, Any],
    cross_instrument_data: Dict[str, Any],
    mineral_data: Dict[str, Any],
    cnn_data: Dict[str, Any],
    engineering_data: Dict[str, Any],
    climate_data: Dict[str, Any],
    science_distance_km: Optional[float] = None,
) -> Dict[str, Any]:
    """Run sensitivity analysis on the composite score.

    Varies each weight by +/-0.05 (one at a time, renormalizing so weights
    still sum to 1.0) and each sub-score by +/-0.1, then records the
    resulting composite score range.

    Parameters
    ----------
    subsurface_data, dielectric_data, cross_instrument_data,
    mineral_data, cnn_data, engineering_data, climate_data : dict
        Same inputs as ``compute_composite_score``.
    science_distance_km : float or None
        Optional distance for penalty.

    Returns
    -------
    dict
        Sensitivity results with baseline, per-weight deltas, per-subscore
        deltas, and overall min/max range.
    """
    # Compute baseline
    baseline_result = compute_composite_score(
        subsurface_data, dielectric_data, cross_instrument_data,
        mineral_data, cnn_data, engineering_data, climate_data,
        science_distance_km=science_distance_km,
    )
    baseline = baseline_result["final_score"]

    # Extract baseline sub-scores
    ss = baseline_result["sub_scores"]
    base_sub = ss["subsurface_potential"]["score"]
    base_ice = ss["surface_ice"]["score"]
    base_terrain = ss["terrain_safety"]["score"]
    base_climate = ss["climate"]["score"]

    w = dict(DEFAULT_WEIGHTS)
    all_extremes: List[float] = [baseline]

    # --- Weight sensitivity ---
    weight_keys = [
        ("w1", "w1_subsurface"),
        ("w2", "w2_surface_ice"),
        ("w3", "w3_terrain"),
        ("w4", "w4_climate"),
    ]
    weight_sensitivity: List[Dict[str, Any]] = []

    for wk, label in weight_keys:
        results_for_key: List[float] = []
        for delta in (-0.05, 0.05):
            test_w = dict(w)
            test_w[wk] = w[wk] + delta
            # Renormalize so weights sum to 1
            total_w = sum(test_w.values())
            if total_w > 0:
                test_w = {k: v / total_w for k, v in test_w.items()}
            res = compute_composite_score(
                subsurface_data, dielectric_data, cross_instrument_data,
                mineral_data, cnn_data, engineering_data, climate_data,
                science_distance_km=science_distance_km,
                weights=test_w,
            )
            results_for_key.append(res["final_score"])
            all_extremes.append(res["final_score"])

        low_val = min(results_for_key)
        high_val = max(results_for_key)
        weight_sensitivity.append({
            "parameter": label,
            "low": round(low_val, 4),
            "high": round(high_val, 4),
            "delta": round(high_val - low_val, 4),
        })

    # --- Sub-score sensitivity ---
    sub_scores_list = [base_sub, base_ice, base_terrain, base_climate]
    sub_labels = ["subsurface_potential", "surface_ice", "terrain_safety", "climate"]
    subscore_sensitivity: List[Dict[str, Any]] = []

    for i, (base_val, label) in enumerate(zip(sub_scores_list, sub_labels)):
        results_for_sub: List[float] = []
        for delta in (-0.1, 0.1):
            perturbed = _clamp(base_val + delta)
            # Build perturbed sub-scores array
            perturbed_scores = list(sub_scores_list)
            perturbed_scores[i] = perturbed
            # Compute composite directly from weights and perturbed sub-scores
            wvals = [w["w1"], w["w2"], w["w3"], w["w4"]]
            composite = sum(wv * ps for wv, ps in zip(wvals, perturbed_scores))
            composite = _clamp(composite)

            # Apply distance penalty if needed
            if science_distance_km is not None and science_distance_km > 50.0:
                adjusted, _ = apply_science_distance_penalty(
                    composite, w["w1"], w["w2"],
                    perturbed_scores[0], perturbed_scores[1],
                    science_distance_km,
                )
                composite = adjusted

            results_for_sub.append(composite)
            all_extremes.append(composite)

        low_val = min(results_for_sub)
        high_val = max(results_for_sub)
        subscore_sensitivity.append({
            "parameter": label,
            "low": round(low_val, 4),
            "high": round(high_val, 4),
            "delta": round(high_val - low_val, 4),
        })

    return {
        "baseline_score": round(baseline, 4),
        "weight_sensitivity": weight_sensitivity,
        "subscore_sensitivity": subscore_sensitivity,
        "overall_range": {
            "min": round(min(all_extremes), 4),
            "max": round(max(all_extremes), 4),
        },
    }


# =============================================================================
# Evidence Strength Classification
# =============================================================================

_STRENGTH_ORDER: List[str] = ["INCONCLUSIVE", "WEAK", "MODERATE", "STRONG"]


def classify_evidence_strength(
    sub_scores: Dict[str, Any],
    cross_instrument_data: Dict[str, Any],
) -> Dict[str, str]:
    """Classify per-instrument evidence strength.

    Classification rules per sub-score:
      STRONG:        score >= 0.7 AND cross-instrument corroboration
      MODERATE:      score >= 0.4 OR single-instrument with high confidence
      WEAK:          score >= 0.2
      INCONCLUSIVE:  score < 0.2

    Overall classification is the minimum (weakest) of all components.

    Parameters
    ----------
    sub_scores : dict
        The ``"sub_scores"`` dict from ``compute_composite_score`` result.
    cross_instrument_data : dict
        ``synthesis["cross_instrument"]``.

    Returns
    -------
    dict
        Per-component and overall evidence strength classification.
    """
    consistency = cross_instrument_data.get("evidence_consistency", "insufficient_data")
    has_corroboration = consistency in ("consistent", "surface_multi", "partial")

    def _classify(score: float, can_corroborate: bool) -> str:
        if score >= 0.7 and can_corroborate and has_corroboration:
            return "STRONG"
        if score >= 0.4:
            return "MODERATE"
        if score >= 0.2:
            return "WEAK"
        return "INCONCLUSIVE"

    subsurface_s = sub_scores.get("subsurface_potential", {}).get("score", 0.0)
    surface_ice_s = sub_scores.get("surface_ice", {}).get("score", 0.0)
    terrain_s = sub_scores.get("terrain_safety", {}).get("score", 0.0)
    climate_s = sub_scores.get("climate", {}).get("score", 0.0)

    classifications: Dict[str, str] = {
        "subsurface": _classify(subsurface_s, True),
        "surface_ice": _classify(surface_ice_s, True),
        "terrain": _classify(terrain_s, False),   # terrain is independent
        "climate": _classify(climate_s, False),    # climate is independent
    }

    # Overall = weakest link
    min_idx = min(
        _STRENGTH_ORDER.index(v) for v in classifications.values()
    )
    classifications["overall"] = _STRENGTH_ORDER[min_idx]

    return classifications


# =============================================================================
# Recommendation Classification
# =============================================================================

def classify_recommendation(
    final_score: float,
    evidence_strength: Dict[str, str],
    cross_instrument_consistency: str,
) -> Dict[str, Any]:
    """Classify the overall site recommendation.

    CRITICAL RULE: If ``cross_instrument_consistency`` is ``"insufficient_data"``
    or ``"inconsistent"`` and the consistency_score is effectively zero (no
    coincident detections), the site MUST NOT be recommended as a primary
    science target.

    Classification logic:
      - score >= 0.75 AND evidence overall >= MODERATE AND cross != non_correlated
        -> "Science-ready candidate", HIGH confidence
      - score >= 0.55 AND evidence overall >= WEAK
        -> "Screening-level", MEDIUM confidence
      - If cross == non_correlated or overall evidence INCONCLUSIVE with low score
        -> At best "Engineering-safe only", LOW confidence
        -> Add critical flag: "Zero cross-instrument corroboration"
      - score < 0.35
        -> "Low priority"

    Parameters
    ----------
    final_score : float
        Post-penalty final composite score (0-1).
    evidence_strength : dict
        Output of ``classify_evidence_strength``.
    cross_instrument_consistency : str
        ``synthesis["cross_instrument"]["evidence_consistency"]``.

    Returns
    -------
    dict
        Classification with confidence, rationale, and critical flags.
    """
    overall_evidence = evidence_strength.get("overall", "INCONCLUSIVE")
    evidence_idx = _STRENGTH_ORDER.index(overall_evidence)

    critical_flags: List[str] = []
    rationale_parts: List[str] = []

    # Detect non-corroboration situations
    no_corroboration = cross_instrument_consistency in (
        "insufficient_data", "inconsistent",
    )

    if no_corroboration:
        critical_flags.append("Zero cross-instrument corroboration")

    # Detect terrain safety issues
    terrain_evidence = evidence_strength.get("terrain", "INCONCLUSIVE")
    if terrain_evidence == "INCONCLUSIVE":
        critical_flags.append("No terrain safety data available")

    # --- Classification decision tree ---

    # Rule 1: Non-correlated data caps recommendation
    if no_corroboration:
        if final_score >= 0.55 and terrain_evidence in ("STRONG", "MODERATE"):
            classification = "Engineering-safe only"
            confidence = "LOW"
            rationale_parts.append(
                f"Score {final_score:.2f} is promising, but no cross-instrument "
                "corroboration exists. Site may be engineering-safe but cannot be "
                "recommended as a primary science target without independent "
                "confirmation of ice resources."
            )
        elif final_score >= 0.35:
            classification = "Engineering-safe only"
            confidence = "LOW"
            rationale_parts.append(
                f"Score {final_score:.2f} with insufficient cross-instrument "
                "corroboration. Site evaluation is at screening level at best."
            )
        else:
            classification = "Low priority"
            confidence = "LOW"
            rationale_parts.append(
                f"Score {final_score:.2f} with no cross-instrument corroboration. "
                "Site does not meet minimum thresholds."
            )

    # Rule 2: Science-ready candidate
    elif (
        final_score >= 0.75
        and evidence_idx >= _STRENGTH_ORDER.index("MODERATE")
    ):
        classification = "Science-ready candidate"
        confidence = "HIGH"
        rationale_parts.append(
            f"Score {final_score:.2f} exceeds science-ready threshold (0.75). "
            f"Overall evidence is {overall_evidence} with "
            f"{cross_instrument_consistency} cross-instrument consistency."
        )
        if evidence_idx >= _STRENGTH_ORDER.index("STRONG"):
            rationale_parts.append(
                "Multiple independent datasets corroborate the ice assessment."
            )

    # Rule 3: Screening-level
    elif (
        final_score >= 0.55
        and evidence_idx >= _STRENGTH_ORDER.index("WEAK")
    ):
        classification = "Screening-level"
        confidence = "MEDIUM"
        rationale_parts.append(
            f"Score {final_score:.2f} passes screening threshold (0.55). "
            f"Evidence is {overall_evidence}. Further investigation recommended "
            "before inclusion in final candidate shortlist."
        )

    # Rule 4: Low priority
    elif final_score < 0.35:
        classification = "Low priority"
        confidence = "LOW"
        rationale_parts.append(
            f"Score {final_score:.2f} is below the minimum threshold (0.35). "
            "Insufficient evidence or unfavorable conditions."
        )

    # Rule 5: Intermediate scores without clear classification
    else:
        classification = "Screening-level"
        confidence = "LOW"
        rationale_parts.append(
            f"Score {final_score:.2f} is in the intermediate range. "
            f"Evidence is {overall_evidence}. Additional data collection "
            "is needed before a definitive recommendation."
        )

    return {
        "classification": classification,
        "confidence": confidence,
        "rationale": " ".join(rationale_parts),
        "critical_flags": critical_flags,
    }


# =============================================================================
# Convenience: Full Assessment Pipeline
# =============================================================================

def full_assessment(
    synthesis: Dict[str, Any],
    science_distance_km: Optional[float] = None,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Run the complete scoring pipeline from a synthesis dict.

    This is a convenience wrapper that extracts the relevant sub-dicts from
    a ``synthesize_results`` output and calls ``compute_composite_score``,
    ``classify_evidence_strength``, ``classify_recommendation``, and
    ``sensitivity_analysis`` in sequence.

    Parameters
    ----------
    synthesis : dict
        Complete synthesis dict from ``agent_tasks.synthesize_results``.
    science_distance_km : float or None
        Distance to nearest science target (km).
    weights : dict or None
        Optional weight overrides.

    Returns
    -------
    dict
        Combined result with scoring, evidence, recommendation, and
        sensitivity analysis.
    """
    subsurface_data = synthesis.get("subsurface_coverage", {})
    dielectric_data = synthesis.get("dielectric_analysis", {})
    cross_instrument_data = synthesis.get("cross_instrument", {})
    mineral_data = synthesis.get("mineral_signatures", {})
    cnn_data = synthesis.get("cnn_mineral_classification", {})
    engineering_data = synthesis.get("engineering_feasibility", {})
    climate_data = synthesis.get("climate", {})

    # 1. Composite score
    scoring = compute_composite_score(
        subsurface_data=subsurface_data,
        dielectric_data=dielectric_data,
        cross_instrument_data=cross_instrument_data,
        mineral_data=mineral_data,
        cnn_data=cnn_data,
        engineering_data=engineering_data,
        climate_data=climate_data,
        science_distance_km=science_distance_km,
        weights=weights,
    )

    # 2. Evidence strength
    evidence = classify_evidence_strength(
        scoring["sub_scores"], cross_instrument_data,
    )

    # 3. Recommendation
    consistency = cross_instrument_data.get("evidence_consistency", "insufficient_data")
    recommendation = classify_recommendation(
        scoring["final_score"], evidence, consistency,
    )

    # 4. Sensitivity analysis
    sensitivity = sensitivity_analysis(
        subsurface_data=subsurface_data,
        dielectric_data=dielectric_data,
        cross_instrument_data=cross_instrument_data,
        mineral_data=mineral_data,
        cnn_data=cnn_data,
        engineering_data=engineering_data,
        climate_data=climate_data,
        science_distance_km=science_distance_km,
    )

    return {
        "scoring": scoring,
        "evidence_strength": evidence,
        "recommendation": recommendation,
        "sensitivity": sensitivity,
    }
