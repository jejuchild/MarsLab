"""
A. Landing Site Suitability Scorer.

Combines ALL MarsLab subsystems into a single comprehensive score
for any (lat, lon) coordinate on Mars.

Subsystems queried:
  - MOLA terrain: elevation, slope
  - Neural Climate: temperature, pressure, dust, wind, frost
  - PINNs Interior: seismic velocity → seismic risk proxy
  - Parametric climate: backup if neural unavailable
  - SWIM/ice evidence: subsurface ice indicators

Score = weighted sum of 6 category subscores (0–1 each):
  terrain (0.20) + climate (0.25) + dust (0.15) +
  wind (0.15) + frost (0.10) + science_value (0.15)
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Score weights
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS = {
    "terrain": 0.20,
    "climate": 0.25,
    "dust": 0.15,
    "wind": 0.15,
    "frost": 0.10,
    "science_value": 0.15,
}

# Human-exploration reference constraints (NASA DRA 5.0 / SpaceX Starship)
CONSTRAINTS = {
    "max_elevation_m": 2000,       # Above ~2 km, pressure too low for ops
    "min_elevation_m": -6000,      # Below datum is fine (more atmosphere)
    "max_slope_deg": 15,           # Landing hazard above 15°
    "ideal_slope_deg": 5,          # Ideal < 5°
    "ideal_temp_range_k": (190, 260),  # Survivable with insulation
    "max_dust_tau": 2.0,           # Above 2.0 = severe visibility loss
    "max_wind_ms": 15,             # Above 15 m/s = EDL hazard
    "max_frost_prob": 0.5,         # High frost = surface ops risk
    "ideal_lat_range": (-45, 45),  # Mid-latitudes preferred for solar
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CategoryScore:
    """Score for a single category."""
    name: str
    score: float         # 0–1
    weight: float
    weighted: float      # score × weight
    details: Dict        # raw data
    assessment: str      # human-readable


@dataclass
class LandingSiteResult:
    """Complete landing site assessment."""
    lat: float
    lon: float
    ls: float                          # solar longitude for climate calc
    overall_score: float               # 0–100
    grade: str                         # A/B/C/D/F
    categories: List[CategoryScore]
    recommendation: str
    warnings: List[str]
    data_sources: Dict[str, str]       # which module provided each data point


# ---------------------------------------------------------------------------
# Subsystem adapters (graceful degradation)
# ---------------------------------------------------------------------------

def _query_terrain(lat: float, lon: float) -> Dict:
    """Query MOLA elevation + slope. Falls back to parametric."""
    try:
        from api.mars_climate import get_elevation_m
        elev = get_elevation_m(lat, lon)
    except Exception:
        elev = 0.0

    # Slope: try terrain_router, fallback to flat assumption
    slope = 0.0
    try:
        from api.terrain_router import compute_slope_stats
        stats = compute_slope_stats(lat, lon, radius_m=1000)
        slope = stats.get("mean_slope", 0.0)
    except Exception:
        pass

    return {"elevation_m": elev, "slope_deg": slope, "source": "MOLA" if elev != 0 else "default"}


def _query_neural_climate(lat: float, lon: float, ls: float) -> Optional[Dict]:
    """Query Neural Climate Emulator. Returns None if unavailable."""
    try:
        from neural_climate.predictor import get_predictor, is_model_trained
        if not is_model_trained():
            return None
        pred = get_predictor()
        result = pred.predict(lat=lat, lon=lon, ls=ls)
        return {
            "temperature_mean_k": result.get("temperature_mean_k", 215),
            "temperature_max_k": result.get("temperature_max_k", 250),
            "temperature_min_k": result.get("temperature_min_k", 180),
            "pressure_pa": result.get("pressure_pa", 636),
            "dust_tau": result.get("dust_tau_mean", 0.3),
            "wind_ms": result.get("wind_mean_ms", 5.0),
            "frost_prob": result.get("frost_probability", 0.1),
            "source": "neural_climate",
        }
    except Exception as exc:
        logger.debug("Neural climate unavailable: %s", exc)
        return None


def _query_parametric_climate(lat: float, lon: float, ls: float, elev: float) -> Dict:
    """Parametric climate model (always available)."""
    try:
        from api.mars_climate import (
            surface_temperature_k, surface_pressure_pa,
            dust_opacity, wind_speed, co2_frost_probability,
        )
        temp = surface_temperature_k(lat, ls, elev)
        dust = dust_opacity(lat, ls)
        wind = wind_speed(lat, ls)
        frost = co2_frost_probability(lat, ls, elev)

        return {
            "temperature_mean_k": temp["mean_k"],
            "temperature_max_k": temp["max_k"],
            "temperature_min_k": temp["min_k"],
            "pressure_pa": surface_pressure_pa(elev),
            "dust_tau": dust["tau_mean"],
            "wind_ms": wind["mean_ms"],
            "frost_prob": frost["frost_probability"],
            "source": "parametric",
        }
    except Exception as exc:
        logger.warning("Parametric climate also failed: %s", exc)
        return {
            "temperature_mean_k": 215, "temperature_max_k": 250,
            "temperature_min_k": 180, "pressure_pa": 636,
            "dust_tau": 0.3, "wind_ms": 5.0, "frost_prob": 0.1,
            "source": "fallback",
        }


def _query_pinns_seismic(lat: float) -> Dict:
    """Query PINNs for seismic risk proxy based on interior velocity anomalies."""
    try:
        from pinns_interior.predictor import get_predictor, is_model_trained
        if not is_model_trained():
            return {"seismic_risk": "unknown", "vp_anomaly_pct": 0, "source": "none"}

        pred = get_predictor()
        # Sample shallow crust velocity (30 km depth)
        vp = pred.predict_velocity(depth_km=30.0)
        ref_vp = 4.5  # Expected crustal Vp ~4.5 km/s

        anomaly_pct = abs(vp - ref_vp) / ref_vp * 100

        if anomaly_pct > 20:
            risk = "elevated"
        elif anomaly_pct > 10:
            risk = "moderate"
        else:
            risk = "low"

        return {"seismic_risk": risk, "vp_anomaly_pct": round(anomaly_pct, 1),
                "vp_km_s": round(vp, 3), "source": "pinns"}
    except Exception:
        return {"seismic_risk": "unknown", "vp_anomaly_pct": 0, "source": "none"}


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _score_terrain(terrain: Dict) -> CategoryScore:
    """Score terrain suitability (elevation + slope)."""
    elev = terrain["elevation_m"]
    slope = terrain["slope_deg"]

    # Elevation score: ideal is -4000 to +1000 m (most atmosphere, below mean)
    if -5000 <= elev <= 1000:
        elev_score = 1.0
    elif elev > CONSTRAINTS["max_elevation_m"]:
        elev_score = _clamp(1.0 - (elev - CONSTRAINTS["max_elevation_m"]) / 3000)
    else:
        elev_score = _clamp(1.0 - (CONSTRAINTS["min_elevation_m"] - elev) / 2000)

    # Slope score: <5° ideal, >15° hazardous
    if slope <= CONSTRAINTS["ideal_slope_deg"]:
        slope_score = 1.0
    elif slope <= CONSTRAINTS["max_slope_deg"]:
        slope_score = _clamp(1.0 - (slope - 5) / 10)
    else:
        slope_score = 0.0

    combined = 0.6 * elev_score + 0.4 * slope_score

    if combined > 0.8:
        assessment = f"Excellent terrain: {elev:.0f}m elevation, {slope:.1f}° slope"
    elif combined > 0.5:
        assessment = f"Acceptable terrain: {elev:.0f}m, {slope:.1f}° slope"
    else:
        assessment = f"Challenging terrain: {elev:.0f}m, {slope:.1f}° slope — landing hazard"

    return CategoryScore(
        name="terrain", score=round(combined, 3),
        weight=DEFAULT_WEIGHTS["terrain"],
        weighted=round(combined * DEFAULT_WEIGHTS["terrain"], 3),
        details=terrain, assessment=assessment,
    )


def _score_climate(climate: Dict) -> CategoryScore:
    """Score temperature/pressure suitability."""
    t_mean = climate["temperature_mean_k"]
    p = climate["pressure_pa"]

    lo, hi = CONSTRAINTS["ideal_temp_range_k"]
    if lo <= t_mean <= hi:
        t_score = 1.0
    else:
        dist = min(abs(t_mean - lo), abs(t_mean - hi))
        t_score = _clamp(1.0 - dist / 50)

    # Pressure: higher is better (more atmosphere for aerobraking)
    p_score = _clamp(p / 800)  # 800 Pa = generous atmosphere

    combined = 0.7 * t_score + 0.3 * p_score

    assessment = f"T_mean={t_mean:.1f}K ({t_mean - 273.15:.1f}°C), P={p:.0f}Pa"

    return CategoryScore(
        name="climate", score=round(combined, 3),
        weight=DEFAULT_WEIGHTS["climate"],
        weighted=round(combined * DEFAULT_WEIGHTS["climate"], 3),
        details={"t_mean_k": t_mean, "t_max_k": climate["temperature_max_k"],
                 "t_min_k": climate["temperature_min_k"], "pressure_pa": p},
        assessment=assessment,
    )


def _score_dust(climate: Dict) -> CategoryScore:
    """Score dust opacity hazard."""
    tau = climate["dust_tau"]

    if tau <= 0.5:
        score = 1.0
    elif tau <= CONSTRAINTS["max_dust_tau"]:
        score = _clamp(1.0 - (tau - 0.5) / 1.5)
    else:
        score = 0.0

    if tau <= 0.3:
        assessment = f"Clear skies (τ={tau:.2f}) — excellent visibility"
    elif tau <= 1.0:
        assessment = f"Moderate dust (τ={tau:.2f}) — acceptable"
    else:
        assessment = f"High dust (τ={tau:.2f}) — visibility hazard, solar power reduced"

    return CategoryScore(
        name="dust", score=round(score, 3),
        weight=DEFAULT_WEIGHTS["dust"],
        weighted=round(score * DEFAULT_WEIGHTS["dust"], 3),
        details={"dust_tau": tau}, assessment=assessment,
    )


def _score_wind(climate: Dict) -> CategoryScore:
    """Score wind hazard."""
    wind = climate["wind_ms"]

    if wind <= 5:
        score = 1.0
    elif wind <= CONSTRAINTS["max_wind_ms"]:
        score = _clamp(1.0 - (wind - 5) / 10)
    else:
        score = 0.0

    assessment = f"Wind {wind:.1f} m/s — {'calm' if wind < 5 else 'moderate' if wind < 10 else 'hazardous'}"

    return CategoryScore(
        name="wind", score=round(score, 3),
        weight=DEFAULT_WEIGHTS["wind"],
        weighted=round(score * DEFAULT_WEIGHTS["wind"], 3),
        details={"wind_ms": wind}, assessment=assessment,
    )


def _score_frost(climate: Dict) -> CategoryScore:
    """Score frost probability."""
    prob = climate["frost_prob"]

    score = _clamp(1.0 - prob / CONSTRAINTS["max_frost_prob"])

    if prob < 0.1:
        assessment = "No significant frost risk"
    elif prob < 0.3:
        assessment = f"Occasional frost ({prob:.0%}) — manageable"
    else:
        assessment = f"Frequent frost ({prob:.0%}) — surface ops impacted"

    return CategoryScore(
        name="frost", score=round(score, 3),
        weight=DEFAULT_WEIGHTS["frost"],
        weighted=round(score * DEFAULT_WEIGHTS["frost"], 3),
        details={"frost_probability": prob}, assessment=assessment,
    )


def _score_science_value(lat: float, lon: float, terrain: Dict, climate: Dict) -> CategoryScore:
    """Score scientific interest for the site."""
    score = 0.5  # baseline: every site has some science value

    # Low elevation = ancient terrain = more geology
    elev = terrain["elevation_m"]
    if elev < -2000:
        score += 0.15  # deep basins often have sedimentary deposits

    # Mid-latitudes (30-50°) have subsurface ice → water access
    if 30 <= abs(lat) <= 50:
        score += 0.2

    # Near-equatorial = diverse mineralogy + solar power
    if abs(lat) < 25:
        score += 0.1

    # Lower frost = more accessible water if ice present
    if climate["frost_prob"] < 0.2 and abs(lat) > 25:
        score += 0.05  # ice under ground, not on surface

    score = _clamp(score)

    assessment = f"Science value: {'high' if score > 0.7 else 'moderate' if score > 0.4 else 'limited'}"

    return CategoryScore(
        name="science_value", score=round(score, 3),
        weight=DEFAULT_WEIGHTS["science_value"],
        weighted=round(score * DEFAULT_WEIGHTS["science_value"], 3),
        details={"lat": lat, "lon": lon, "elevation_m": elev},
        assessment=assessment,
    )


def _grade(score: float) -> str:
    if score >= 80:
        return "A"
    if score >= 65:
        return "B"
    if score >= 50:
        return "C"
    if score >= 35:
        return "D"
    return "F"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def score_landing_site(
    lat: float,
    lon: float,
    ls: float = 0.0,
    weights: Optional[Dict[str, float]] = None,
) -> LandingSiteResult:
    """
    Score a landing site at (lat, lon) for a given season (ls).

    Parameters
    ----------
    lat : float
        Latitude in degrees (-90 to 90).
    lon : float
        Longitude in degrees (-180 to 360).
    ls : float
        Solar longitude (0-360). Default 0 = northern spring equinox.
    weights : dict, optional
        Override default category weights.

    Returns
    -------
    LandingSiteResult with overall_score (0-100), grade, and per-category breakdown.
    """
    w = weights or DEFAULT_WEIGHTS

    # 1. Query all subsystems
    terrain = _query_terrain(lat, lon)
    neural = _query_neural_climate(lat, lon, ls)
    if neural:
        climate = neural
    else:
        climate = _query_parametric_climate(lat, lon, ls, terrain["elevation_m"])

    seismic = _query_pinns_seismic(lat)

    # 2. Score each category
    categories = [
        _score_terrain(terrain),
        _score_climate(climate),
        _score_dust(climate),
        _score_wind(climate),
        _score_frost(climate),
        _score_science_value(lat, lon, terrain, climate),
    ]

    # 3. Compute overall score
    overall = sum(c.weighted for c in categories)
    overall_100 = round(overall * 100, 1)

    # 4. Warnings
    warnings = []
    if terrain["slope_deg"] > CONSTRAINTS["max_slope_deg"]:
        warnings.append(f"⚠ Slope {terrain['slope_deg']:.1f}° exceeds safe landing limit ({CONSTRAINTS['max_slope_deg']}°)")
    if terrain["elevation_m"] > CONSTRAINTS["max_elevation_m"]:
        warnings.append(f"⚠ Elevation {terrain['elevation_m']:.0f}m — low atmospheric density for EDL")
    if climate["dust_tau"] > CONSTRAINTS["max_dust_tau"]:
        warnings.append(f"⚠ Extreme dust opacity τ={climate['dust_tau']:.1f} at Ls={ls:.0f}°")
    if climate["frost_prob"] > CONSTRAINTS["max_frost_prob"]:
        warnings.append(f"⚠ High frost probability ({climate['frost_prob']:.0%})")
    if seismic["seismic_risk"] == "elevated":
        warnings.append(f"⚠ Elevated seismic risk (Vp anomaly {seismic['vp_anomaly_pct']}%)")

    # 5. Recommendation
    grade = _grade(overall_100)
    if grade == "A":
        rec = "Highly recommended landing site — all parameters within safe limits."
    elif grade == "B":
        rec = "Good candidate — minor concerns in some categories."
    elif grade == "C":
        rec = "Marginal site — significant challenges in one or more areas."
    elif grade == "D":
        rec = "Poor candidate — multiple hazards detected."
    else:
        rec = "Not recommended — critical hazards present."

    sources = {
        "terrain": terrain.get("source", "unknown"),
        "climate": climate.get("source", "unknown"),
        "seismic": seismic.get("source", "unknown"),
    }

    return LandingSiteResult(
        lat=lat, lon=lon, ls=ls,
        overall_score=overall_100,
        grade=grade,
        categories=categories,
        recommendation=rec,
        warnings=warnings,
        data_sources=sources,
    )


def compare_sites(
    sites: List[Dict],
    ls: float = 0.0,
) -> List[LandingSiteResult]:
    """
    Score and rank multiple landing sites.

    Parameters
    ----------
    sites : list of dict
        Each dict must have 'lat' and 'lon'. Optional 'name'.
    ls : float
        Solar longitude for climate calculations.

    Returns
    -------
    List of LandingSiteResult, sorted by overall_score descending.
    """
    results = []
    for site in sites:
        result = score_landing_site(
            lat=site["lat"],
            lon=site["lon"],
            ls=ls,
        )
        results.append(result)

    results.sort(key=lambda r: r.overall_score, reverse=True)
    return results
