"""
B. Climate-Mineral Stability Map.

Maps mineral phase stability across Mars using climate (T, P) fields
from the Neural Climate Emulator and thermodynamic stability envelopes
for key mineral groups detected by CRISM.

Key mineral systems:
  - Phyllosilicates (clays): stable under broad T range, indicate past water
  - Sulfates: monohydrated ↔ polyhydrated phase transitions (T-dependent)
  - Perchlorates: deliquescence at T > eutectic → transient liquid brines
  - Carbonates: CO2-pressure-dependent stability
  - Water ice: sublimation/condensation boundary

The module produces a stability assessment for any (lat, lon) point,
answering: "Which minerals are thermodynamically stable HERE and NOW?"

Integration points:
  - Neural Climate Emulator → T(lat, lon, Ls), P(lat, lon, Ls)
  - Parametric climate model → fallback T, P
  - CRISM spectral data → observed mineral detections for validation
  - Thermal inertia → surface material context
"""

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Thermodynamic constants (published values)
# ---------------------------------------------------------------------------

# Water ice sublimation curve (Clausius-Clapeyron simplified)
# P_sub(T) = P_ref * exp(-L/R * (1/T - 1/T_ref))
WATER_ICE = {
    "L_sub_J_mol": 51_058.0,   # Latent heat of sublimation (J/mol)
    "R_gas": 8.314,             # Universal gas constant (J/mol·K)
    "T_ref_k": 273.15,         # Reference temperature
    "P_ref_pa": 611.657,       # Triple point pressure (Pa)
}

# Sulfate hydration phase boundaries (Chipera & Vaniman 2007)
# Mg-sulfate: epsomite (7H2O) → hexahydrite (6H2O) → kieserite (1H2O)
SULFATE_TRANSITIONS = {
    "epsomite_to_hexahydrite_k": 275,   # 7→6 H2O, ~2°C
    "hexahydrite_to_kieserite_k": 340,  # 6→1 H2O, ~67°C (lab; lower on Mars due to P)
    "mars_adjusted_hex_to_kies_k": 260, # On Mars, low aH2O shifts this down
    "kieserite_stable_below_k": 260,    # Below this, kieserite is stable on Mars
}

# Perchlorate eutectic temperatures (Chevrier et al. 2009, Toner et al. 2014)
PERCHLORATE_EUTECTICS = {
    "Ca(ClO4)2": {"eutectic_k": 199, "deliquescence_rh": 0.40},
    "Mg(ClO4)2": {"eutectic_k": 206, "deliquescence_rh": 0.42},
    "NaClO4":    {"eutectic_k": 236, "deliquescence_rh": 0.50},
    "Fe(ClO4)3": {"eutectic_k": 208, "deliquescence_rh": 0.45},
}

# Carbonate stability: CaCO3 stable where CO2 partial pressure is sufficient
# On Mars, pCO2 ~ 95.3% of surface pressure
CARBONATE = {
    "co2_fraction": 0.953,
    "min_pco2_pa_for_stability": 10.0,  # Very stable on Mars (always enough CO2)
    "decomposition_temp_k": 1100,       # CaCO3 decomposes above ~1100K (irrelevant for Mars surface)
}

# Phyllosilicate stability ranges
PHYLLOSILICATE = {
    "formation_requires_water": True,
    "stable_once_formed": True,   # Clays persist indefinitely once formed
    "dehydroxylation_k": 700,     # Lose structural OH above ~700K (irrelevant for Mars)
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MineralStability:
    """Stability assessment for a single mineral group."""
    mineral_group: str
    is_stable: bool
    phase: str                    # Current stable phase name
    confidence: float             # 0–1
    temperature_k: float          # Local temperature used
    pressure_pa: float            # Local pressure used
    notes: str
    brine_possible: bool = False  # Only for perchlorates


@dataclass
class StabilityMapResult:
    """Complete mineral stability assessment for a location."""
    lat: float
    lon: float
    ls: float
    temperature_k: float
    pressure_pa: float
    minerals: List[MineralStability]
    brine_candidates: List[str]       # Perchlorate salts that could form brines
    water_ice_stable: bool
    dominant_sulfate_phase: str
    climate_source: str               # "neural_climate" or "parametric"
    thermal_inertia: Optional[float]  # TES TI value, if available
    crism_validation: Optional[Dict]  # CRISM mineral detections nearby, if any
    summary: str


# ---------------------------------------------------------------------------
# Thermodynamic functions
# ---------------------------------------------------------------------------

def water_ice_sublimation_pressure(temp_k: float) -> float:
    """Clausius-Clapeyron sublimation pressure of H2O ice at temperature T.

    Returns pressure in Pa below which ice sublimates.
    """
    if temp_k <= 0:
        return 0.0
    L = WATER_ICE["L_sub_J_mol"]
    R = WATER_ICE["R_gas"]
    T_ref = WATER_ICE["T_ref_k"]
    P_ref = WATER_ICE["P_ref_pa"]
    return P_ref * math.exp(-L / R * (1.0 / temp_k - 1.0 / T_ref))


def is_water_ice_stable(temp_k: float, pressure_pa: float) -> Tuple[bool, str]:
    """Check if surface water ice is thermodynamically stable.

    Ice is stable when the ambient partial pressure of H2O exceeds
    the sublimation pressure. On Mars, pH2O ~ 0.03% of total P.
    """
    p_sub = water_ice_sublimation_pressure(temp_k)
    # Mars atmospheric H2O partial pressure (very low, ~0.03% or ~0.2 Pa typical)
    p_h2o = pressure_pa * 0.0003
    stable = p_h2o >= p_sub
    if stable:
        note = f"Ice stable: pH2O={p_h2o:.3f} Pa ≥ P_sub={p_sub:.3f} Pa at {temp_k:.1f}K"
    else:
        note = f"Ice unstable: pH2O={p_h2o:.3f} Pa < P_sub={p_sub:.3f} Pa at {temp_k:.1f}K — sublimation expected"
    return stable, note


def sulfate_phase(temp_k: float) -> Tuple[str, str]:
    """Determine stable Mg-sulfate hydration phase on Mars.

    Returns (phase_name, explanation).
    """
    if temp_k < SULFATE_TRANSITIONS["kieserite_stable_below_k"]:
        return "kieserite (MgSO4·H2O)", (
            f"T={temp_k:.1f}K < {SULFATE_TRANSITIONS['kieserite_stable_below_k']}K — "
            "monohydrated sulfate stable under low Mars humidity"
        )
    elif temp_k < SULFATE_TRANSITIONS["epsomite_to_hexahydrite_k"]:
        return "hexahydrite (MgSO4·6H2O)", (
            f"T={temp_k:.1f}K in transition zone — polyhydrated sulfate possible "
            "if local humidity sufficient"
        )
    else:
        return "epsomite (MgSO4·7H2O)", (
            f"T={temp_k:.1f}K — fully hydrated sulfate, requires substantial water activity"
        )


def perchlorate_brine_check(temp_k: float) -> List[Dict]:
    """Check which perchlorate salts could form liquid brines at temp_k.

    Returns list of salts with brine potential.
    """
    results = []
    for salt, props in PERCHLORATE_EUTECTICS.items():
        eutectic = props["eutectic_k"]
        if temp_k >= eutectic:
            results.append({
                "salt": salt,
                "eutectic_k": eutectic,
                "margin_k": round(temp_k - eutectic, 1),
                "brine_possible": True,
                "note": (
                    f"{salt}: T={temp_k:.1f}K ≥ eutectic={eutectic}K "
                    f"(+{temp_k - eutectic:.1f}K margin) — transient liquid brine thermodynamically possible"
                ),
            })
        else:
            results.append({
                "salt": salt,
                "eutectic_k": eutectic,
                "margin_k": round(temp_k - eutectic, 1),
                "brine_possible": False,
                "note": f"{salt}: T={temp_k:.1f}K < eutectic={eutectic}K — no liquid brine",
            })
    return results


def carbonate_stability(pressure_pa: float) -> Tuple[bool, str]:
    """Check if carbonate (CaCO3) is stable on the surface.

    On Mars, carbonates are thermodynamically stable because pCO2
    is always above the minimum stability threshold.
    """
    pco2 = pressure_pa * CARBONATE["co2_fraction"]
    stable = pco2 >= CARBONATE["min_pco2_pa_for_stability"]
    note = (
        f"pCO2={pco2:.1f} Pa ({'≥' if stable else '<'} "
        f"{CARBONATE['min_pco2_pa_for_stability']} Pa) — "
        f"carbonates {'stable' if stable else 'unstable (extremely low pressure)'}"
    )
    return stable, note


def phyllosilicate_stability(temp_k: float) -> Tuple[bool, str]:
    """Check phyllosilicate (clay) stability.

    Clays are kinetically stable once formed — they persist on Mars
    indefinitely under all current surface conditions.
    """
    stable = temp_k < PHYLLOSILICATE["dehydroxylation_k"]
    note = (
        f"Phyllosilicates stable at {temp_k:.1f}K "
        f"(dehydroxylation above {PHYLLOSILICATE['dehydroxylation_k']}K). "
        "If present, clay minerals persist under current conditions."
    )
    return stable, note


# ---------------------------------------------------------------------------
# Subsystem adapters
# ---------------------------------------------------------------------------

def _get_climate(lat: float, lon: float, ls: float) -> Tuple[Dict, str]:
    """Get temperature and pressure from best available climate source."""
    # Try neural climate first
    try:
        from neural_climate.predictor import get_predictor, is_model_trained
        if is_model_trained():
            pred = get_predictor()
            result = pred.predict(lat=lat, lon=lon, ls=ls)
            return {
                "temperature_k": result.get("temperature_mean_k", 215),
                "temperature_max_k": result.get("temperature_max_k", 250),
                "temperature_min_k": result.get("temperature_min_k", 180),
                "pressure_pa": result.get("pressure_pa", 636),
            }, "neural_climate"
    except Exception as exc:
        logger.debug("Neural climate unavailable: %s", exc)

    # Fallback to parametric
    try:
        from api.mars_climate import (
            get_elevation_m, surface_temperature_k,
            surface_pressure_pa,
        )
        elev = get_elevation_m(lat, lon)
        temp = surface_temperature_k(lat, ls, elev)
        pressure = surface_pressure_pa(elev)
        return {
            "temperature_k": temp["mean_k"],
            "temperature_max_k": temp["max_k"],
            "temperature_min_k": temp["min_k"],
            "pressure_pa": pressure,
        }, "parametric"
    except Exception as exc:
        logger.warning("Parametric climate failed: %s", exc)

    # Last resort fallback
    return {
        "temperature_k": 215.0,
        "temperature_max_k": 250.0,
        "temperature_min_k": 180.0,
        "pressure_pa": 636.0,
    }, "fallback"


def _get_thermal_inertia(lat: float, lon: float) -> Optional[float]:
    """Query TES thermal inertia at location."""
    try:
        from api.thermal_inertia import get_thermal_inertia
        return get_thermal_inertia(lat, lon)
    except Exception:
        return None


def _get_crism_context(lat: float, lon: float) -> Optional[Dict]:
    """Get CRISM mineral detection context near location."""
    try:
        from analysis.ice_evidence.crism_proxy import evaluate_crism_evidence
        result = evaluate_crism_evidence(lat, lon)
        return {
            "ice_score": result.score,
            "hyd_score": result.hyd_score,
            "distance_km": result.distance_km,
            "notes": result.notes,
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

def assess_mineral_stability(
    lat: float,
    lon: float,
    ls: float = 0.0,
) -> StabilityMapResult:
    """
    Assess mineral phase stability at (lat, lon) for season Ls.

    Evaluates stability of 5 mineral systems under local climate
    conditions and checks for transient liquid brine potential.

    Parameters
    ----------
    lat : float
        Latitude in degrees (-90 to 90).
    lon : float
        Longitude in degrees (-180 to 360).
    ls : float
        Solar longitude (0-360). Default 0 = northern spring equinox.

    Returns
    -------
    StabilityMapResult with per-mineral stability and brine assessment.
    """
    # 1. Get climate
    climate, source = _get_climate(lat, lon, ls)
    t_mean = climate["temperature_k"]
    t_max = climate["temperature_max_k"]
    t_min = climate["temperature_min_k"]
    p = climate["pressure_pa"]

    # 2. Get context data
    ti = _get_thermal_inertia(lat, lon)
    crism = _get_crism_context(lat, lon)

    # 3. Evaluate each mineral system
    minerals = []

    # 3a. Water ice (use min temperature — most favorable for ice)
    ice_stable, ice_note = is_water_ice_stable(t_min, p)
    minerals.append(MineralStability(
        mineral_group="water_ice",
        is_stable=ice_stable,
        phase="H2O ice (Ih)" if ice_stable else "H2O vapor",
        confidence=0.9 if ice_stable and abs(lat) > 50 else 0.7 if ice_stable else 0.8,
        temperature_k=t_min,
        pressure_pa=p,
        notes=ice_note,
    ))

    # 3b. Sulfates (use mean temperature)
    sulf_phase, sulf_note = sulfate_phase(t_mean)
    minerals.append(MineralStability(
        mineral_group="sulfates",
        is_stable=True,  # Some sulfate phase always stable
        phase=sulf_phase,
        confidence=0.85,
        temperature_k=t_mean,
        pressure_pa=p,
        notes=sulf_note,
    ))

    # 3c. Perchlorates — check all salts at max temperature (warmest = most brine potential)
    brine_results = perchlorate_brine_check(t_max)
    brine_candidates = [r["salt"] for r in brine_results if r["brine_possible"]]
    any_brine = len(brine_candidates) > 0

    perc_notes = "; ".join(r["note"] for r in brine_results if r["brine_possible"])
    if not perc_notes:
        perc_notes = f"No perchlorate brines at T_max={t_max:.1f}K — all eutectics above current temperature"

    minerals.append(MineralStability(
        mineral_group="perchlorates",
        is_stable=True,  # Perchlorate salts are always stable; question is liquid phase
        phase="liquid brine (transient)" if any_brine else "solid salt",
        confidence=0.75 if any_brine else 0.85,
        temperature_k=t_max,
        pressure_pa=p,
        notes=perc_notes,
        brine_possible=any_brine,
    ))

    # 3d. Carbonates
    carb_stable, carb_note = carbonate_stability(p)
    minerals.append(MineralStability(
        mineral_group="carbonates",
        is_stable=carb_stable,
        phase="CaCO3 / MgCO3 (stable)" if carb_stable else "unstable",
        confidence=0.95,  # Very well constrained
        temperature_k=t_mean,
        pressure_pa=p,
        notes=carb_note,
    ))

    # 3e. Phyllosilicates
    phyl_stable, phyl_note = phyllosilicate_stability(t_mean)
    minerals.append(MineralStability(
        mineral_group="phyllosilicates",
        is_stable=phyl_stable,
        phase="stable clays (smectite, nontronite, etc.)",
        confidence=0.95,
        temperature_k=t_mean,
        pressure_pa=p,
        notes=phyl_note,
    ))

    # 4. Build summary
    summary_parts = []
    summary_parts.append(
        f"Location ({lat:.1f}°, {lon:.1f}°) at Ls={ls:.0f}°: "
        f"T={t_mean:.1f}K, P={p:.0f}Pa"
    )

    if ice_stable:
        summary_parts.append("Water ice is thermodynamically stable at surface nighttime temperatures.")
    else:
        summary_parts.append("Surface water ice is unstable — sublimation expected.")

    summary_parts.append(f"Dominant sulfate phase: {sulf_phase}.")

    if any_brine:
        summary_parts.append(
            f"Transient liquid brines possible via {', '.join(brine_candidates)} "
            f"at T_max={t_max:.1f}K."
        )
    else:
        summary_parts.append("No liquid brines expected at current temperatures.")

    if ti is not None:
        if ti > 300:
            summary_parts.append(f"High thermal inertia ({ti:.0f}) suggests consolidated/ice-cemented surface.")
        elif ti < 150:
            summary_parts.append(f"Low thermal inertia ({ti:.0f}) indicates fine dust cover.")
        else:
            summary_parts.append(f"Moderate thermal inertia ({ti:.0f}) — mixed regolith.")

    if crism and crism["ice_score"] > 0.1:
        summary_parts.append(
            f"CRISM validation: ice spectral score={crism['ice_score']:.2f}, "
            f"hydration={crism['hyd_score']:.2f} at {crism['distance_km']:.0f}km."
        )

    return StabilityMapResult(
        lat=lat,
        lon=lon,
        ls=ls,
        temperature_k=t_mean,
        pressure_pa=p,
        minerals=minerals,
        brine_candidates=brine_candidates,
        water_ice_stable=ice_stable,
        dominant_sulfate_phase=sulf_phase,
        climate_source=source,
        thermal_inertia=ti,
        crism_validation=crism,
        summary=" ".join(summary_parts),
    )


def seasonal_stability_profile(
    lat: float,
    lon: float,
    n_seasons: int = 12,
) -> List[StabilityMapResult]:
    """
    Compute mineral stability across a full Mars year.

    Samples n_seasons evenly spaced Ls values from 0-360.

    Parameters
    ----------
    lat : float
        Latitude in degrees.
    lon : float
        Longitude in degrees.
    n_seasons : int
        Number of seasonal samples (default 12 = monthly).

    Returns
    -------
    List of StabilityMapResult, one per Ls sample.
    """
    ls_values = np.linspace(0, 360, n_seasons, endpoint=False)
    results = []
    for ls in ls_values:
        result = assess_mineral_stability(lat, lon, float(ls))
        results.append(result)
    return results


def brine_habitability_window(
    lat: float,
    lon: float,
    n_seasons: int = 36,
) -> Dict:
    """
    Determine the seasonal window where liquid brines are possible.

    Evaluates perchlorate brine potential at high temporal resolution
    across the Mars year.

    Parameters
    ----------
    lat : float
        Latitude in degrees.
    lon : float
        Longitude in degrees.
    n_seasons : int
        Number of seasonal samples (default 36 = ~10° Ls resolution).

    Returns
    -------
    Dict with brine window statistics:
      - brine_fraction: fraction of year with brine potential
      - brine_ls_ranges: list of (start_ls, end_ls) brine windows
      - salts_involved: which perchlorate salts contribute
      - max_temperature_k: peak temperature across year
      - total_brine_hours_est: rough estimate of annual brine duration
    """
    ls_values = np.linspace(0, 360, n_seasons, endpoint=False)
    brine_flags = []
    all_salts = set()
    max_t = 0.0

    for ls in ls_values:
        climate, _ = _get_climate(lat, lon, float(ls))
        t_max = climate["temperature_max_k"]
        max_t = max(max_t, t_max)

        brines = perchlorate_brine_check(t_max)
        active = [b for b in brines if b["brine_possible"]]
        brine_flags.append(len(active) > 0)
        for b in active:
            all_salts.add(b["salt"])

    # Find contiguous brine windows
    brine_fraction = sum(brine_flags) / len(brine_flags)

    ranges = []
    in_window = False
    start_ls = 0.0
    for i, (ls, has_brine) in enumerate(zip(ls_values, brine_flags)):
        if has_brine and not in_window:
            start_ls = float(ls)
            in_window = True
        elif not has_brine and in_window:
            ranges.append((start_ls, float(ls_values[i - 1])))
            in_window = False
    if in_window:
        ranges.append((start_ls, float(ls_values[-1])))

    # Rough brine duration estimate: Mars year ~668.6 sols, ~24.66 hr/sol
    _mars_year_hours = 668.6 * 24.66  # noqa: F841
    # Brines only possible for ~2-4 hours/sol during warmest part of day
    brine_hours_per_sol = 3.0  # conservative estimate
    total_brine_hours = brine_fraction * 668.6 * brine_hours_per_sol

    return {
        "lat": lat,
        "lon": lon,
        "brine_fraction": round(brine_fraction, 3),
        "brine_ls_ranges": ranges,
        "salts_involved": sorted(all_salts),
        "max_temperature_k": round(max_t, 1),
        "total_brine_hours_est": round(total_brine_hours, 1),
        "n_samples": n_seasons,
    }
