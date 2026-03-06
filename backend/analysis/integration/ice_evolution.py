"""
C. Subsurface Ice Evolution Model.

Combines surface climate (Neural Climate Emulator or parametric model)
with interior heat flow (PINNs model) to compute subsurface temperature
profiles and ice stability depth.

Physics:
  The steady-state subsurface temperature at depth z is:
    T(z) = T_surface + (dT/dz) * z
  where dT/dz is the geothermal gradient derived from PINNs interior
  velocity anomalies (hotter interior → steeper gradient).

  Ice is stable at depth z when:
    T(z) < T_sublimation(P(z))
  where P(z) accounts for lithostatic overburden.

  The ice stability depth is the shallowest z where ice can persist
  through the annual thermal wave.

Validation:
  - SHARAD radar detections of subsurface reflectors (ice table depth)
  - SWIM ice probability scores
  - Thermal inertia (high TI → ice-cemented regolith)

Integration points:
  - Neural Climate / Parametric → surface T(lat, lon, Ls)
  - PINNs Interior → geothermal gradient proxy
  - SHARAD → observed ice table depth for validation
  - SWIM → multi-criteria ice probability
  - Thermal inertia → surface material indicator
"""

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

# Mars thermal properties
REGOLITH_THERMAL_CONDUCTIVITY = 0.04    # W/(m·K), dry Mars regolith
ICE_CEMENTED_CONDUCTIVITY = 2.0         # W/(m·K), ice-cemented soil
REGOLITH_DENSITY = 1500.0               # kg/m³
REGOLITH_SPECIFIC_HEAT = 800.0          # J/(kg·K)
MARS_GRAVITY = 3.72                     # m/s²

# Thermal wave penetration
MARS_YEAR_SECONDS = 5.94e7             # ~668.6 sols × 88,775 s/sol
MARS_DAY_SECONDS = 88_775.0            # 1 sol in seconds

# Default geothermal gradient (Plesa et al. 2016)
DEFAULT_GEOTHERMAL_GRADIENT = 0.005    # K/m (~5 K/km, Mars interior)

# Water ice sublimation (Clausius-Clapeyron)
WATER_ICE_L_SUB = 51_058.0            # J/mol
R_GAS = 8.314                          # J/(mol·K)
TRIPLE_POINT_T = 273.15               # K
TRIPLE_POINT_P = 611.657              # Pa


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SubsurfaceProfile:
    """Temperature profile at a single depth."""
    depth_m: float
    temperature_k: float
    ice_stable: bool
    sublimation_pressure_pa: float
    pore_pressure_pa: float


@dataclass
class IceStabilityResult:
    """Ice stability depth calculation for a location."""
    lat: float
    lon: float
    ls: float
    ice_stability_depth_m: Optional[float]   # None if ice never stable
    ice_stable_surface: bool                  # True if stable at surface
    surface_temperature_k: float
    annual_max_surface_temp_k: float
    annual_min_surface_temp_k: float
    geothermal_gradient_k_per_m: float
    thermal_skin_depth_m: float              # Annual thermal wave penetration
    profile: List[SubsurfaceProfile]         # T(z) profile
    geothermal_source: str                   # "pinns" or "default"
    climate_source: str


@dataclass
class IceEvolutionResult:
    """Complete ice evolution assessment with validation."""
    lat: float
    lon: float
    stability: IceStabilityResult
    sharad_validation: Optional[Dict]        # SHARAD-observed ice depth
    swim_probability: Optional[float]        # SWIM ice probability
    thermal_inertia: Optional[float]         # TES TI value
    consistency_score: float                 # Agreement between model and observations
    summary: str


# ---------------------------------------------------------------------------
# Thermal physics
# ---------------------------------------------------------------------------

def sublimation_pressure(temp_k: float) -> float:
    """Clausius-Clapeyron sublimation pressure of H2O ice."""
    if temp_k <= 0:
        return 0.0
    return TRIPLE_POINT_P * math.exp(
        -WATER_ICE_L_SUB / R_GAS * (1.0 / temp_k - 1.0 / TRIPLE_POINT_T)
    )


def pore_pressure_at_depth(depth_m: float, surface_pressure_pa: float) -> float:
    """Estimate pore space water vapor pressure at depth.

    Assumes diffusive equilibrium with atmosphere.
    Mars atmospheric H2O is ~0.03% of total pressure.
    Pore pressure increases slightly with lithostatic overburden.
    """
    # Atmospheric H2O contribution
    p_atm_h2o = surface_pressure_pa * 0.0003

    # At depth, some additional vapor from regolith desorption
    # This is a simplified model — in reality depends on porosity, diffusion
    adsorption_bonus = depth_m * 0.001  # ~0.001 Pa/m from adsorbed water release

    return p_atm_h2o + adsorption_bonus


def annual_thermal_skin_depth(
    conductivity: float = REGOLITH_THERMAL_CONDUCTIVITY,
    density: float = REGOLITH_DENSITY,
    specific_heat: float = REGOLITH_SPECIFIC_HEAT,
) -> float:
    """Compute annual thermal skin depth.

    δ = sqrt(2 * k * P / (ρ * c * 2π))
    where P is the Mars year period.
    """
    thermal_diffusivity = conductivity / (density * specific_heat)
    return math.sqrt(2.0 * thermal_diffusivity * MARS_YEAR_SECONDS / (2.0 * math.pi))


def subsurface_temperature(
    depth_m: float,
    t_mean_surface: float,
    t_amplitude: float,
    geothermal_gradient: float,
    ls: float,
    skin_depth: float,
) -> float:
    """Compute subsurface temperature at depth z and season Ls.

    T(z, t) = T_mean + dT/dz * z + A * exp(-z/δ) * cos(ωt - z/δ)

    Parameters
    ----------
    depth_m : float
        Depth below surface (m).
    t_mean_surface : float
        Annual mean surface temperature (K).
    t_amplitude : float
        Half-amplitude of annual temperature variation (K).
    geothermal_gradient : float
        dT/dz (K/m), positive = increasing with depth.
    ls : float
        Solar longitude (0-360°), proxy for seasonal phase.
    skin_depth : float
        Annual thermal skin depth (m).
    """
    omega_t = 2.0 * math.pi * ls / 360.0  # Phase from Ls

    # Mean temperature + geothermal
    t_mean_at_depth = t_mean_surface + geothermal_gradient * depth_m

    # Attenuated annual wave
    if skin_depth > 0 and depth_m < 20 * skin_depth:
        attenuation = math.exp(-depth_m / skin_depth)
        phase_shift = depth_m / skin_depth
        t_wave = t_amplitude * attenuation * math.cos(omega_t - phase_shift)
    else:
        t_wave = 0.0  # Below thermal wave penetration

    return t_mean_at_depth + t_wave


def max_annual_temperature_at_depth(
    depth_m: float,
    t_mean_surface: float,
    t_amplitude: float,
    geothermal_gradient: float,
    skin_depth: float,
) -> float:
    """Maximum temperature reached at depth z over the Mars year.

    T_max(z) = T_mean + dT/dz * z + A * exp(-z/δ)
    """
    t_mean_at_depth = t_mean_surface + geothermal_gradient * depth_m

    if skin_depth > 0:
        attenuation = math.exp(-depth_m / skin_depth)
        return t_mean_at_depth + t_amplitude * attenuation
    return t_mean_at_depth


# ---------------------------------------------------------------------------
# Geothermal gradient from PINNs
# ---------------------------------------------------------------------------

def _estimate_geothermal_gradient() -> Tuple[float, str]:
    """Derive geothermal gradient from PINNs interior model.

    If PINNs predicts anomalous crustal velocity, interpret as
    thermal anomaly and adjust the geothermal gradient.

    Lower Vp → hotter crust → steeper gradient.
    Higher Vp → cooler crust → shallower gradient.
    """
    try:
        from pinns_interior.predictor import get_predictor, is_model_trained
        if not is_model_trained():
            return DEFAULT_GEOTHERMAL_GRADIENT, "default"

        pred = get_predictor()
        # Sample at 30 km depth (shallow crust)
        vp_shallow = float(pred.predict_velocity(depth_km=30.0))

        # Reference crustal Vp ~4.5 km/s at 30 km
        ref_vp_shallow = 4.5
        anomaly = (ref_vp_shallow - vp_shallow) / ref_vp_shallow

        # Negative anomaly (low Vp) → hotter → higher gradient
        # Typical Mars: 5 K/km; anomaly scales ±50%
        gradient_k_per_km = 5.0 * (1.0 + anomaly * 2.0)
        gradient_k_per_km = max(2.0, min(15.0, gradient_k_per_km))  # Clamp to reasonable range

        return gradient_k_per_km / 1000.0, "pinns"

    except Exception as exc:
        logger.debug("PINNs unavailable for geothermal gradient: %s", exc)
        return DEFAULT_GEOTHERMAL_GRADIENT, "default"


# ---------------------------------------------------------------------------
# Climate adapters
# ---------------------------------------------------------------------------

def _get_surface_climate(lat: float, lon: float, ls: float) -> Tuple[Dict, str]:
    """Get surface temperature statistics from best available source."""
    # Try neural climate
    try:
        from neural_climate.predictor import get_predictor, is_model_trained
        if is_model_trained():
            pred = get_predictor()
            result = pred.predict(lat=lat, lon=lon, ls=ls)
            return {
                "t_mean_k": result.get("temperature_mean_k", 215),
                "t_max_k": result.get("temperature_max_k", 250),
                "t_min_k": result.get("temperature_min_k", 180),
                "pressure_pa": result.get("pressure_pa", 636),
            }, "neural_climate"
    except Exception:
        pass

    # Parametric fallback
    try:
        from api.mars_climate import (
            get_elevation_m, surface_temperature_k, surface_pressure_pa,
        )
        elev = get_elevation_m(lat, lon)
        temp = surface_temperature_k(lat, ls, elev)
        pressure = surface_pressure_pa(elev)
        return {
            "t_mean_k": temp["mean_k"],
            "t_max_k": temp["max_k"],
            "t_min_k": temp["min_k"],
            "pressure_pa": pressure,
        }, "parametric"
    except Exception:
        pass

    return {
        "t_mean_k": 215.0,
        "t_max_k": 250.0,
        "t_min_k": 180.0,
        "pressure_pa": 636.0,
    }, "fallback"


def _get_thermal_inertia(lat: float, lon: float) -> Optional[float]:
    """Query TES thermal inertia."""
    try:
        from api.thermal_inertia import get_thermal_inertia
        return get_thermal_inertia(lat, lon)
    except Exception:
        return None


def _get_sharad_ice_depth(lat: float, lon: float) -> Optional[Dict]:
    """Query SHARAD for observed subsurface ice reflectors."""
    try:
        from analysis.ice_evidence.crism_proxy import evaluate_crism_evidence
        from analysis.ice_evidence.fusion import fuse_evidence
        from analysis.ice_evidence.models import (
            CandidateLocation, E1Hyperbola, E2Reflector, E3Terrain,
            E4Crism, EvidenceParams,
        )

        # Use CRISM proxy for ice evidence
        crism_result = evaluate_crism_evidence(lat, lon)

        # Build a minimal evidence assessment
        candidate = CandidateLocation(lat=lat, lon=lon, id=f"ice_evo_{lat}_{lon}")
        e1 = E1Hyperbola()  # No hyperbola fit data
        e2 = E2Reflector()  # No reflector data
        e3 = E3Terrain()    # No terrain
        e4 = E4Crism(
            score=crism_result.score,
            ice_score=crism_result.ice_score,
            hyd_score=crism_result.hyd_score,
            distance_km=crism_result.distance_km,
        )

        result = fuse_evidence(candidate, e1, e2, e3, e4, EvidenceParams())
        return {
            "ice_probability": result.ice_probability,
            "confidence": result.confidence,
            "crism_ice_score": crism_result.ice_score,
            "crism_distance_km": crism_result.distance_km,
        }
    except Exception:
        return None


def _get_swim_ice_probability(lat: float, lon: float) -> Optional[float]:
    """Query SWIM multi-criteria ice probability."""
    try:
        from analysis.swim_fusion.pipeline import SwimFusionPipeline
        pipeline = SwimFusionPipeline()
        result = pipeline.query_point(lat, lon)
        # ConsistencyPointResponse has consistency scores for different depths
        # Use the shallowest (0-1m) as ice probability proxy
        if result.consistency_0_1m is not None:
            return result.consistency_0_1m
        if result.consistency_1_5m is not None:
            return result.consistency_1_5m
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------

def compute_ice_stability(
    lat: float,
    lon: float,
    ls: float = 0.0,
    max_depth_m: float = 20.0,
    depth_resolution_m: float = 0.5,
) -> IceStabilityResult:
    """
    Compute subsurface ice stability depth at (lat, lon).

    Finds the shallowest depth where water ice can persist through
    the entire Mars year (worst-case = maximum annual temperature).

    Parameters
    ----------
    lat : float
        Latitude in degrees (-90 to 90).
    lon : float
        Longitude in degrees.
    ls : float
        Solar longitude for current season profile.
    max_depth_m : float
        Maximum depth to probe (default 20 m).
    depth_resolution_m : float
        Depth step size (default 0.5 m).

    Returns
    -------
    IceStabilityResult with stability depth and T(z) profile.
    """
    # 1. Get surface climate
    climate, climate_source = _get_surface_climate(lat, lon, ls)
    t_mean = climate["t_mean_k"]
    t_max = climate["t_max_k"]
    t_min = climate["t_min_k"]
    pressure = climate["pressure_pa"]

    # Annual temperature amplitude
    t_amplitude = (t_max - t_min) / 2.0

    # 2. Get geothermal gradient
    geo_gradient, geo_source = _estimate_geothermal_gradient()

    # 3. Compute thermal skin depth
    # Use higher conductivity if thermal inertia suggests ice-cemented regolith
    ti = _get_thermal_inertia(lat, lon)
    if ti is not None and ti > 300:
        conductivity = ICE_CEMENTED_CONDUCTIVITY
    else:
        conductivity = REGOLITH_THERMAL_CONDUCTIVITY

    skin_depth = annual_thermal_skin_depth(conductivity)

    # 4. Compute temperature profile and find ice stability depth
    depths = np.arange(0, max_depth_m + depth_resolution_m, depth_resolution_m)
    profile = []
    ice_stability_depth = None

    for z in depths:
        # Use maximum annual temperature (worst case for ice)
        t_max_at_z = max_annual_temperature_at_depth(
            z, t_mean, t_amplitude, geo_gradient, skin_depth
        )

        # Also compute current temperature for profile
        t_current = subsurface_temperature(
            z, t_mean, t_amplitude, geo_gradient, ls, skin_depth
        )

        # Sublimation pressure at this temperature
        p_sub = sublimation_pressure(t_max_at_z)

        # Pore vapor pressure at this depth
        p_pore = pore_pressure_at_depth(z, pressure)

        # Ice is stable if pore vapor pressure ≥ sublimation pressure
        # OR if temperature is low enough that sublimation is negligible
        ice_ok = p_pore >= p_sub or t_max_at_z < 150.0

        profile.append(SubsurfaceProfile(
            depth_m=round(float(z), 2),
            temperature_k=round(t_current, 2),
            ice_stable=ice_ok,
            sublimation_pressure_pa=round(p_sub, 4),
            pore_pressure_pa=round(p_pore, 4),
        ))

        if ice_ok and ice_stability_depth is None:
            ice_stability_depth = round(float(z), 2)

    # Check if ice is stable at the surface
    surface_ice_stable = profile[0].ice_stable if profile else False

    return IceStabilityResult(
        lat=lat,
        lon=lon,
        ls=ls,
        ice_stability_depth_m=ice_stability_depth,
        ice_stable_surface=surface_ice_stable,
        surface_temperature_k=round(t_mean, 2),
        annual_max_surface_temp_k=round(t_max, 2),
        annual_min_surface_temp_k=round(t_min, 2),
        geothermal_gradient_k_per_m=round(geo_gradient, 6),
        thermal_skin_depth_m=round(skin_depth, 3),
        profile=profile,
        geothermal_source=geo_source,
        climate_source=climate_source,
    )


def assess_ice_evolution(
    lat: float,
    lon: float,
    ls: float = 0.0,
) -> IceEvolutionResult:
    """
    Full ice evolution assessment with observational validation.

    Computes modeled ice stability depth and compares against
    SHARAD, SWIM, and thermal inertia observations.

    Parameters
    ----------
    lat : float
        Latitude in degrees.
    lon : float
        Longitude in degrees.
    ls : float
        Solar longitude.

    Returns
    -------
    IceEvolutionResult with model predictions and validation.
    """
    # 1. Compute modeled ice stability
    stability = compute_ice_stability(lat, lon, ls)

    # 2. Get validation data
    sharad = _get_sharad_ice_depth(lat, lon)
    swim = _get_swim_ice_probability(lat, lon)
    ti = _get_thermal_inertia(lat, lon)

    # 3. Compute consistency score
    consistency = _compute_consistency(stability, sharad, swim, ti)

    # 4. Build summary
    summary_parts = []
    summary_parts.append(
        f"Ice evolution at ({lat:.1f}°, {lon:.1f}°), Ls={ls:.0f}°:"
    )

    if stability.ice_stability_depth_m is not None:
        if stability.ice_stable_surface:
            summary_parts.append(
                "Water ice is stable at the surface under current conditions."
            )
        else:
            summary_parts.append(
                f"Ice stability depth: {stability.ice_stability_depth_m:.1f} m below surface."
            )
    else:
        summary_parts.append(
            "No ice stability zone found within the modeled depth range — "
            "subsurface temperatures too warm for ice persistence."
        )

    summary_parts.append(
        f"Geothermal gradient: {stability.geothermal_gradient_k_per_m * 1000:.1f} K/km "
        f"(source: {stability.geothermal_source})."
    )

    summary_parts.append(
        f"Annual thermal skin depth: {stability.thermal_skin_depth_m:.2f} m."
    )

    if sharad and sharad["ice_probability"] > 0.3:
        summary_parts.append(
            f"SHARAD/CRISM validation: ice probability {sharad['ice_probability']:.2f} "
            f"(confidence {sharad['confidence']:.2f})."
        )

    if swim is not None:
        summary_parts.append(f"SWIM ice probability: {swim:.2f}.")

    if ti is not None:
        if ti > 300:
            summary_parts.append(
                f"High thermal inertia ({ti:.0f}) — consistent with ice-cemented regolith."
            )
        elif ti < 150:
            summary_parts.append(
                f"Low thermal inertia ({ti:.0f}) — loose dust, ice unlikely near surface."
            )

    summary_parts.append(f"Model-observation consistency: {consistency:.2f}.")

    return IceEvolutionResult(
        lat=lat,
        lon=lon,
        stability=stability,
        sharad_validation=sharad,
        swim_probability=swim,
        thermal_inertia=ti,
        consistency_score=round(consistency, 3),
        summary=" ".join(summary_parts),
    )


def depth_stability_map(
    lat: float,
    lon: float,
    n_seasons: int = 12,
    max_depth_m: float = 20.0,
) -> Dict:
    """
    Compute ice stability depth across a full Mars year.

    Returns a map of ice stability depth vs. Ls, showing
    how the ice table retreats/advances with the seasons.

    Parameters
    ----------
    lat, lon : float
        Location.
    n_seasons : int
        Number of Ls samples (default 12).
    max_depth_m : float
        Maximum depth to probe.

    Returns
    -------
    Dict with seasonal stability data.
    """
    ls_values = np.linspace(0, 360, n_seasons, endpoint=False)
    results = []

    for ls in ls_values:
        result = compute_ice_stability(
            lat, lon, float(ls), max_depth_m=max_depth_m
        )
        results.append({
            "ls": float(ls),
            "ice_stability_depth_m": result.ice_stability_depth_m,
            "ice_stable_surface": result.ice_stable_surface,
            "surface_temperature_k": result.surface_temperature_k,
        })

    # Summary statistics
    stable_depths = [
        r["ice_stability_depth_m"] for r in results
        if r["ice_stability_depth_m"] is not None
    ]

    return {
        "lat": lat,
        "lon": lon,
        "seasonal_data": results,
        "min_stability_depth_m": min(stable_depths) if stable_depths else None,
        "max_stability_depth_m": max(stable_depths) if stable_depths else None,
        "mean_stability_depth_m": (
            round(sum(stable_depths) / len(stable_depths), 2)
            if stable_depths else None
        ),
        "fraction_year_surface_stable": (
            sum(1 for r in results if r["ice_stable_surface"]) / len(results)
        ),
        "n_seasons": n_seasons,
    }


# ---------------------------------------------------------------------------
# Consistency scoring
# ---------------------------------------------------------------------------

def _compute_consistency(
    stability: IceStabilityResult,
    sharad: Optional[Dict],
    swim: Optional[float],
    ti: Optional[float],
) -> float:
    """Score consistency between modeled ice stability and observations.

    Returns 0–1 where 1 = perfect agreement.
    """
    scores = []

    # Model predicts ice?
    model_has_ice = stability.ice_stability_depth_m is not None

    # 1. SHARAD consistency
    if sharad is not None:
        sharad_sees_ice = sharad["ice_probability"] > 0.3
        if model_has_ice and sharad_sees_ice:
            scores.append(0.9)  # Both agree: ice present
        elif not model_has_ice and not sharad_sees_ice:
            scores.append(0.8)  # Both agree: no ice
        else:
            scores.append(0.2)  # Disagree

    # 2. SWIM consistency
    if swim is not None:
        swim_sees_ice = swim > 0.5
        if model_has_ice and swim_sees_ice:
            scores.append(0.9)
        elif not model_has_ice and not swim_sees_ice:
            scores.append(0.8)
        elif model_has_ice and swim > 0.3:
            scores.append(0.5)  # Partial agreement
        else:
            scores.append(0.2)

    # 3. Thermal inertia consistency
    if ti is not None:
        if model_has_ice and ti > 250:
            scores.append(0.8)  # Ice + high TI = consistent
        elif model_has_ice and ti < 150:
            scores.append(0.3)  # Ice predicted but dusty surface
        elif not model_has_ice and ti < 200:
            scores.append(0.7)  # No ice + low TI = consistent
        else:
            scores.append(0.5)

    # 4. Latitude consistency (ice more expected at high latitudes)
    if abs(stability.lat) > 50:
        if model_has_ice:
            scores.append(0.9)
        else:
            scores.append(0.4)  # Surprising if no ice at high lat
    elif abs(stability.lat) < 30:
        if not model_has_ice:
            scores.append(0.8)
        else:
            scores.append(0.5)  # Possible but less expected near equator

    if not scores:
        return 0.5  # No validation data

    return sum(scores) / len(scores)
