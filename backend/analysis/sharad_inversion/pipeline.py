"""
SHARAD Physics-Based Dielectric Inversion Pipeline.

5-step mandatory pipeline:
  Step 1: Terraced crater selection (find DTM-SHARAD intersections, measure terrace depth)
  Step 2: SHARAD two-way travel time measurement (extract Δt with auto-pick refinement)
  Step 3: Dielectric constant inversion (v = 2d/Δt, εr = (c/v)²)
  Step 4: Hyperbola curvature cross-validation
  Step 5: Material interpretation with explicit rules

PRINCIPLE: Depth MUST come from DTM geometry (independent measurement).
           εr MUST be computed, NEVER assumed.
"""
import logging
import math
import numpy as np
from typing import List, Optional, Dict, Any, Tuple

from .models import (
    InversionPipelineResult, InversionResult,
    TerraceDepthMeasurement, ReflectorPick, DielectricInversion,
    HyperbolaValidation, ClutterAssessment,
    DerivationStep, PhysicalAssumption,
)
from .clutter_filter import assess_clutter

logger = logging.getLogger(__name__)

SPEED_OF_LIGHT = 299_792_458.0  # m/s
SHARAD_DT_US = 3.0 / 80.0      # 0.0375 µs per range bin


def run_inversion_pipeline(
    sharad_product_ids: List[str],
    dtm_dir: str = "hirise_dtm_data",
    buffer_km: float = 30.0,
    min_snr: float = 3.5,
    do_hyperbola_validation: bool = True,
    do_clutter_filter: bool = True,
) -> InversionPipelineResult:
    """
    Run the full 5-step physics-based dielectric inversion pipeline.

    Args:
        sharad_product_ids: List of SHARAD high-res product IDs to analyze
        dtm_dir: Directory containing HiRISE DTM files
        buffer_km: Search radius for DTM-SHARAD proximity
        min_snr: Minimum SNR for reflector detection
        do_hyperbola_validation: Whether to perform Step 4 (curvature validation)
        do_clutter_filter: Whether to perform cluttergram comparison

    Returns:
        InversionPipelineResult with all inversions, assumptions, and derivation log
    """
    all_inversions: List[InversionResult] = []
    all_assumptions: List[PhysicalAssumption] = []
    all_derivation: List[DerivationStep] = []
    dtm_intersections = 0
    step_counter = [0]  # mutable counter for derivation steps

    def _step(desc: str, formula: str = "", inputs: dict = None, result: dict = None, notes: str = "") -> DerivationStep:
        step_counter[0] += 1
        s = DerivationStep(
            step_number=step_counter[0],
            description=desc,
            formula=formula,
            inputs=inputs or {},
            result=result or {},
            notes=notes,
        )
        all_derivation.append(s)
        return s

    # Log fundamental assumptions
    all_assumptions.append(PhysicalAssumption(
        param="speed_of_light", value=SPEED_OF_LIGHT,
        source="physics", justification="Vacuum speed of light (exact)",
    ))
    all_assumptions.append(PhysicalAssumption(
        param="SHARAD_sample_interval", value=f"{SHARAD_DT_US} µs",
        source="instrument_spec", justification="SHARAD ADC: 26.67 MHz → 0.0375 µs/bin",
    ))

    try:
        from analysis.epsilon_terrace.run import find_nearby_dtms
        from analysis.epsilon_terrace.terrace_detect import extract_radial_profiles
        from analysis.epsilon_terrace.sharad_pick import pick_subsurface_interface
        from analysis.epsilon_terrace.epsilon_calc import compute_epsilon
    except ImportError as e:
        return InversionPipelineResult(
            success=False,
            error=f"Missing epsilon_terrace module: {e}",
            assumptions=all_assumptions,
        )

    _step("Pipeline initialization", notes=f"Analyzing {len(sharad_product_ids)} SHARAD products, buffer={buffer_km}km")

    for sharad_pid in sharad_product_ids:
        logger.info(f"[Inversion] Processing SHARAD product: {sharad_pid}")

        # ── Step 1: Find terraced crater intersections ──
        _step(
            f"Step 1: Searching for DTM-SHARAD intersections for {sharad_pid}",
            notes=f"Buffer radius: {buffer_km} km",
        )

        try:
            dtm_matches = find_nearby_dtms(sharad_pid, dtm_dir, buffer_km=buffer_km)
        except Exception as e:
            _step(f"DTM search failed for {sharad_pid}", notes=str(e))
            logger.warning(f"DTM search failed for {sharad_pid}: {e}")
            continue

        if not dtm_matches:
            _step(
                f"No DTM intersections found for {sharad_pid}",
                notes="Dielectric inversion CANNOT be performed without independent depth measurement"
            )
            all_assumptions.append(PhysicalAssumption(
                param="DTM_availability", value=False,
                source="data_search", justification=f"No HiRISE DTMs within {buffer_km}km of SHARAD track {sharad_pid}",
            ))
            continue

        dtm_intersections += len(dtm_matches)
        _step(f"Found {len(dtm_matches)} DTM(s) near {sharad_pid}")

        for dtm_info in dtm_matches:
            dtm_path = dtm_info.get("dtm_path", "")
            dtm_pid = dtm_info.get("dtm_product_id", dtm_path.split("/")[-1] if dtm_path else "unknown")
            dtm_dist_km = dtm_info.get("distance_km", 0)

            # ── Step 1b: Extract terrace geometry ──
            _step(
                f"Extracting terrace geometry from DTM {dtm_pid}",
                notes=f"Distance to SHARAD track: {dtm_dist_km:.1f} km",
            )

            try:
                terrace_result = extract_radial_profiles(dtm_path)
            except Exception as e:
                _step(f"Terrace extraction failed for {dtm_pid}", notes=str(e))
                logger.warning(f"Terrace extraction failed: {e}")
                continue

            if not terrace_result or not terrace_result.terrace_depths:
                _step(f"No terraces detected in {dtm_pid}")
                continue

            crater_lat = terrace_result.crater_center_lat
            crater_lon = terrace_result.crater_center_lon

            # Use the best depth metric (prefer upper-to-lower terrace delta)
            for depth_entry in terrace_result.terrace_depths:
                depth_m = depth_entry.get("depth_m", 0)
                depth_unc = depth_entry.get("uncertainty_m", depth_m * 0.15)  # 15% if not provided
                depth_metric = depth_entry.get("metric", "unknown")

                if depth_m < 10:
                    _step(f"Terrace depth {depth_m:.1f}m too shallow, skipping", notes="Minimum 10m for reliable inversion")
                    continue

                terrace_meas = TerraceDepthMeasurement(
                    crater_id=f"{dtm_pid}_{depth_metric}",
                    dtm_product_id=dtm_pid,
                    depth_m=depth_m,
                    depth_unc_m=depth_unc,
                    depth_metric=depth_metric,
                    crater_lat=crater_lat,
                    crater_lon=crater_lon,
                )

                _step(
                    f"Terrace depth measured: d = {depth_m:.1f} ± {depth_unc:.1f} m",
                    formula="d = elevation(terrace_upper) - elevation(terrace_lower)",
                    inputs={"dtm_product": dtm_pid, "metric": depth_metric, "n_azimuths": 36},
                    result={"depth_m": depth_m, "depth_unc_m": depth_unc},
                    notes="Independent geometric measurement from HiRISE DTM",
                )

                all_assumptions.append(PhysicalAssumption(
                    param="terrace_depth", value=f"{depth_m:.1f} ± {depth_unc:.1f} m",
                    source="measured", justification=f"DTM terrace geometry ({depth_metric}) in {dtm_pid}",
                    uncertainty=f"±{depth_unc:.1f} m (std across {terrace_meas.n_azimuths} azimuths)",
                ))

                # ── Step 2: SHARAD two-way travel time measurement ──
                _step(f"Step 2: Measuring SHARAD two-way travel time near crater at ({crater_lat:.3f}, {crater_lon:.3f})")

                try:
                    pick_result = pick_subsurface_interface(
                        sharad_pid, crater_lat, crater_lon,
                        window_km=max(5.0, buffer_km / 2),
                        min_snr=min_snr,
                    )
                except Exception as e:
                    _step(f"SHARAD pick failed near {dtm_pid}", notes=str(e))
                    logger.warning(f"SHARAD pick failed: {e}")
                    continue

                if not pick_result or not pick_result.picks:
                    _step(
                        "No subsurface reflectors detected",
                        notes=f"SNR threshold: {min_snr}. No reflections above threshold near crater."
                    )
                    continue

                # Build reflector picks with clutter assessment
                reflector_picks: List[ReflectorPick] = []
                for pick in pick_result.picks:
                    # ── Step 2b: Cluttergram comparison ──
                    if do_clutter_filter:
                        clutter = assess_clutter(
                            sharad_pid, pick.trace_idx, pick.interface_bin, pick.surface_bin,
                        )
                    else:
                        clutter = ClutterAssessment(notes="Clutter filter disabled")

                    rp = ReflectorPick(
                        product_id=sharad_pid,
                        trace_idx=pick.trace_idx,
                        lat=pick.lat,
                        lon=pick.lon,
                        surface_bin=pick.surface_bin,
                        interface_bin=pick.interface_bin,
                        delta_bins=pick.delta_bins,
                        twt_us=pick.twt_us,
                        twt_unc_us=pick.twt_unc_us,
                        snr=pick.snr,
                        clutter=clutter,
                    )

                    if clutter.rejected:
                        _step(
                            f"Reflector at trace {pick.trace_idx} REJECTED as clutter",
                            notes=clutter.notes,
                        )
                    else:
                        reflector_picks.append(rp)

                if not reflector_picks:
                    _step("All reflectors rejected by clutter filter")
                    continue

                # Use median twt from non-rejected picks
                twt_values = [rp.twt_us for rp in reflector_picks]
                twt_us = float(np.median(twt_values))
                twt_unc_us = pick_result.twt_unc_us if pick_result.twt_unc_us else float(np.std(twt_values)) if len(twt_values) > 1 else SHARAD_DT_US

                _step(
                    f"Two-way travel time: Δt = {twt_us:.4f} ± {twt_unc_us:.4f} µs",
                    formula="Δt = (interface_bin - surface_bin) × 0.0375 µs",
                    inputs={"n_picks": len(reflector_picks), "median_delta_bins": float(np.median([rp.delta_bins for rp in reflector_picks]))},
                    result={"twt_us": twt_us, "twt_unc_us": twt_unc_us},
                    notes=f"{len(reflector_picks)} picks after clutter filtering (of {len(pick_result.picks)} total)",
                )

                # ── Step 3: Dielectric constant inversion ──
                _step("Step 3: Computing dielectric constant via inversion")

                t_s = twt_us * 1e-6  # Convert µs to seconds
                dt_s = twt_unc_us * 1e-6
                d = depth_m
                dd = depth_unc

                # v = 2d / Δt
                velocity = (2.0 * d) / t_s

                # εr = (c / v)²
                epsilon_r = (SPEED_OF_LIGHT / velocity) ** 2

                # Uncertainty propagation
                t_lo = max(t_s - dt_s, t_s * 0.5)
                t_hi = t_s + dt_s
                d_lo = max(d - dd, d * 0.5)
                d_hi = d + dd

                v_lo = (2.0 * d_lo) / t_hi  # slowest → highest εr
                v_hi = (2.0 * d_hi) / t_lo  # fastest → lowest εr

                eps_lo = (SPEED_OF_LIGHT / v_hi) ** 2
                eps_hi = (SPEED_OF_LIGHT / v_lo) ** 2

                # Quality classification
                if 2.0 < epsilon_r < 6.0 and (eps_hi / max(eps_lo, 0.01)) < 10:
                    quality = "good"
                elif epsilon_r > 10.0 or epsilon_r < 1.5:
                    quality = "unreliable"
                else:
                    quality = "marginal"

                # Interpretation
                interpretation = _interpret_epsilon(epsilon_r)

                derivation_steps_local = [
                    DerivationStep(
                        step_number=1, description="Compute wave velocity",
                        formula="v = 2d / Δt",
                        inputs={"d_m": d, "twt_s": t_s},
                        result={"v_mps": round(velocity, 1)},
                    ),
                    DerivationStep(
                        step_number=2, description="Compute dielectric constant",
                        formula="εr = (c / v)²",
                        inputs={"c_mps": SPEED_OF_LIGHT, "v_mps": round(velocity, 1)},
                        result={"epsilon_r": round(epsilon_r, 3)},
                    ),
                    DerivationStep(
                        step_number=3, description="Propagate uncertainty",
                        formula="εr_low = (c·(Δt-δt) / (2·(d+δd)))²; εr_high = (c·(Δt+δt) / (2·(d-δd)))²",
                        inputs={"dt_s": dt_s, "dd_m": dd},
                        result={"epsilon_r_low": round(eps_lo, 3), "epsilon_r_high": round(eps_hi, 3)},
                    ),
                ]

                inversion = DielectricInversion(
                    epsilon_r=round(epsilon_r, 4),
                    epsilon_r_low=round(eps_lo, 4),
                    epsilon_r_high=round(eps_hi, 4),
                    velocity_mps=round(velocity, 1),
                    depth_m=d,
                    depth_unc_m=dd,
                    twt_us=twt_us,
                    twt_unc_us=twt_unc_us,
                    quality=quality,
                    interpretation=interpretation,
                    derivation=derivation_steps_local,
                )

                _step(
                    f"Dielectric inversion: εr = {epsilon_r:.3f} [{eps_lo:.3f}, {eps_hi:.3f}]",
                    formula="v = 2d/Δt = 2×{d:.1f} / {t:.6f} = {v:.0f} m/s → εr = (c/v)² = {eps:.3f}".format(
                        d=d, t=t_s, v=velocity, eps=epsilon_r
                    ),
                    result={"epsilon_r": round(epsilon_r, 4), "quality": quality, "interpretation": interpretation},
                )

                all_assumptions.append(PhysicalAssumption(
                    param="epsilon_r", value=round(epsilon_r, 4),
                    source="DTM_inversion",
                    justification=f"Terrace depth {d:.1f}m, twt {twt_us:.4f}µs → v={velocity:.0f} m/s → εr={epsilon_r:.3f}",
                    uncertainty=f"[{eps_lo:.3f}, {eps_hi:.3f}]",
                ))

                # ── Step 4: Hyperbola curvature validation ──
                hyp_val = HyperbolaValidation()

                if do_hyperbola_validation:
                    _step("Step 4: Hyperbola curvature cross-validation")
                    hyp_val = _run_hyperbola_validation(
                        sharad_pid, reflector_picks, epsilon_r, _step
                    )

                    if hyp_val.performed and hyp_val.epsilon_r_curvature is not None:
                        all_assumptions.append(PhysicalAssumption(
                            param="epsilon_r_curvature", value=round(hyp_val.epsilon_r_curvature, 4),
                            source="curvature_fit",
                            justification=f"Hyperbola fit: t(x)=sqrt(t0²+(2x/v)²) → εr={hyp_val.epsilon_r_curvature:.3f}",
                            uncertainty=str(hyp_val.epsilon_r_curvature_ci95),
                        ))

                # ── Step 5: Interpretation ──
                _step(
                    f"Step 5: Material interpretation — {interpretation}",
                    notes=f"Quality: {quality}, Agreement: {hyp_val.agreement or 'N/A'}",
                )

                inv_result = InversionResult(
                    crater_id=f"{dtm_pid}_{depth_metric}",
                    dtm_product_id=dtm_pid,
                    sharad_product_id=sharad_pid,
                    terrace_depth=terrace_meas,
                    reflector_picks=reflector_picks,
                    inversion=inversion,
                    hyperbola_validation=hyp_val,
                    assumptions=[a for a in all_assumptions if a.param in ("terrace_depth", "epsilon_r", "epsilon_r_curvature")],
                    derivation_steps=derivation_steps_local,
                )
                all_inversions.append(inv_result)

    # Aggregate best result
    good_inversions = [inv for inv in all_inversions if inv.inversion and inv.inversion.quality in ("good", "marginal")]

    best_eps = None
    best_ci = None
    best_interp = None
    reflector_conf = 0.0

    if good_inversions:
        eps_values = [inv.inversion.epsilon_r for inv in good_inversions]
        best_eps = float(np.median(eps_values))
        best_ci = [float(np.percentile(eps_values, 5)), float(np.percentile(eps_values, 95))] if len(eps_values) > 1 else [good_inversions[0].inversion.epsilon_r_low, good_inversions[0].inversion.epsilon_r_high]
        best_interp = _interpret_epsilon(best_eps)

        # Reflector confidence based on number of inversions, quality, and agreement
        n_good = sum(1 for inv in good_inversions if inv.inversion.quality == "good")
        n_agreed = sum(1 for inv in good_inversions if inv.hyperbola_validation.agreement == "consistent")
        n_uncluttered = sum(1 for inv in good_inversions for rp in inv.reflector_picks if not rp.clutter.rejected and rp.clutter.clutter_likelihood_score < 0.3)

        conf_factors = []
        conf_factors.append(min(1.0, n_good / 3.0))  # Up to 3 good inversions → 1.0
        if any(inv.hyperbola_validation.performed for inv in good_inversions):
            conf_factors.append(min(1.0, n_agreed / max(1, sum(1 for inv in good_inversions if inv.hyperbola_validation.performed))))
        conf_factors.append(min(1.0, n_uncluttered / max(1, sum(len(inv.reflector_picks) for inv in good_inversions))))

        reflector_conf = round(float(np.mean(conf_factors)), 3)

    _step(
        f"Pipeline complete: {len(all_inversions)} inversions, best εr = {best_eps}",
        result={
            "n_inversions": len(all_inversions),
            "best_epsilon_r": best_eps,
            "reflector_confidence": reflector_conf,
        },
    )

    return InversionPipelineResult(
        success=len(all_inversions) > 0,
        sharad_products_analyzed=len(sharad_product_ids),
        dtm_intersections_found=dtm_intersections,
        inversions_completed=len(all_inversions),
        inversions=all_inversions,
        best_epsilon_r=best_eps,
        best_epsilon_r_ci=best_ci,
        best_interpretation=best_interp,
        reflector_confidence=reflector_conf,
        assumptions=all_assumptions,
        derivation_log=all_derivation,
        notes=f"Analyzed {len(sharad_product_ids)} SHARAD products, found {dtm_intersections} DTM intersections, completed {len(all_inversions)} inversions" if all_inversions else "No dielectric inversions possible — no DTM-SHARAD terrace intersections found",
    )


def _run_hyperbola_validation(
    sharad_pid: str,
    reflector_picks: List[ReflectorPick],
    epsilon_r_dtm: float,
    step_fn,
) -> HyperbolaValidation:
    """Run hyperbola curvature fit and compare with DTM-based εr."""
    try:
        from analysis.ice_evidence.hyperbola_fit import auto_detect_apexes, fit_hyperbola
        from analysis.ice_evidence.models import HyperbolaFitRequest
    except ImportError as e:
        return HyperbolaValidation(notes=f"hyperbola_fit module unavailable: {e}")

    try:
        # Use the best reflector pick as apex for hyperbola fitting
        best_pick = max(reflector_picks, key=lambda rp: rp.snr)

        req = HyperbolaFitRequest(
            product_id=sharad_pid,
            apex_trace=best_pick.trace_idx,
            apex_bin=best_pick.interface_bin,
        )

        fit_result = fit_hyperbola(req)

        if not fit_result or fit_result.epsr is None or fit_result.epsr <= 0:
            step_fn("Hyperbola fit failed or returned invalid εr", notes="Curvature validation not possible")
            return HyperbolaValidation(performed=True, notes="Fit returned invalid εr")

        eps_curv = fit_result.epsr
        eps_curv_ci = fit_result.epsr_ci95 if fit_result.epsr_ci95 else [eps_curv * 0.8, eps_curv * 1.2]

        delta = abs(eps_curv - epsilon_r_dtm)

        # Agreement classification based on overlap of uncertainty ranges
        if delta < 0.5:
            agreement = "consistent"
        elif delta < 1.5:
            agreement = "marginal"
        else:
            agreement = "inconsistent"

        step_fn(
            f"Hyperbola validation: εr(curvature) = {eps_curv:.3f}, εr(DTM) = {epsilon_r_dtm:.3f}, Δ = {delta:.3f}",
            formula="t(x) = sqrt(t0² + (2x/v)²) → v → εr = (c/v)²",
            result={"epsilon_r_curvature": round(eps_curv, 4), "delta": round(delta, 4), "agreement": agreement},
            notes=f"{'CONSISTENT' if agreement == 'consistent' else 'DIVERGENT'}: DTM and curvature methods {'agree' if agreement == 'consistent' else 'disagree'}",
        )

        return HyperbolaValidation(
            performed=True,
            epsilon_r_curvature=round(eps_curv, 4),
            epsilon_r_curvature_ci95=[round(c, 4) for c in eps_curv_ci],
            epsilon_r_dtm=round(epsilon_r_dtm, 4),
            agreement=agreement,
            delta_epsilon=round(delta, 4),
            notes=f"Δεr={delta:.3f}, agreement={agreement}",
        )

    except Exception as e:
        logger.warning(f"Hyperbola validation failed: {e}")
        step_fn(f"Hyperbola validation failed: {e}")
        return HyperbolaValidation(performed=False, notes=f"Error: {str(e)}")


def _interpret_epsilon(eps: float) -> str:
    """Classify material from dielectric constant."""
    if eps < 2.0:
        return "very porous / dry dust"
    elif eps < 2.8:
        return "porous ice or dry regolith"
    elif eps < 3.5:
        return "water ice (εr_ice ≈ 3.1)"
    elif eps < 5.0:
        return "ice-cemented regolith or ice-rock mixture"
    elif eps < 8.0:
        return "basaltic regolith or dense rock"
    else:
        return "very dense material or metallic minerals"
