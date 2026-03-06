"""
Parametric Mars Climate Model for Agent Scoring.

Computes surface temperature, pressure, dust opacity, wind speed, and
CO2 frost probability from (lat, lon, Ls) using published MCD v6.1
statistics and well-known Mars atmospheric relationships.

References:
  - Millour et al. (2018) MCD v5.3 design document
  - Forget et al. (1999) Mars GCM
  - Haberle et al. (2001) surface pressure
"""

import math
import os
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
CO2_FROST_POINT_K = 148.0     # CO2 condensation temperature at ~6 mbar
MEAN_SURFACE_PRESSURE_PA = 636.0  # Viking Lander reference pressure
SCALE_HEIGHT_M = 10_800       # Mars atmospheric scale height

# MOLA DEM for elevation lookups
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_MOLA_PATH = os.path.join(_PROJECT_ROOT, "Mars_HRSC_MOLA_BlendDEM_Global_200mp_v2.tif")

# Lazy-loaded MOLA dataset handle
_mola_ds = None


def _get_mola():
    """Lazy-load MOLA GeoTIFF (via rasterio) for elevation queries."""
    global _mola_ds
    if _mola_ds is None:
        try:
            import rasterio
            _mola_ds = rasterio.open(_MOLA_PATH)
        except Exception as e:
            logger.warning(f"MOLA DEM not available: {e}")
            return None
    return _mola_ds


def get_elevation_m(lat: float, lon: float) -> float:
    """Get surface elevation (m) from MOLA DEM. Returns 0 if unavailable."""
    ds = _get_mola()
    if ds is None:
        return 0.0
    try:
        # MOLA DEM uses -180/180 lon convention
        if lon > 180:
            lon -= 360
        elif lon < -180:
            lon += 360
        row, col = ds.index(lon, lat)
        elev = ds.read(1, window=((row, row + 1), (col, col + 1)))[0, 0]
        if np.isnan(elev) or elev < -20_000 or elev > 30_000:
            return 0.0
        return float(elev)
    except Exception:
        return 0.0

# ---------------------------------------------------------------------------
# Parametric climate functions
# ---------------------------------------------------------------------------

def surface_temperature_k(lat_deg: float, ls_deg: float, elevation_m: float = 0.0) -> dict:
    """
    Estimate surface temperature statistics for a given location and season.

    Returns dict with mean_k, max_k, min_k, diurnal_swing_k.

    Model: T = T_mean(lat) + T_seasonal(lat, Ls) + T_elevation(z)
    """
    lat_r = math.radians(lat_deg)
    ls_r = math.radians(ls_deg)

    # Mean annual temperature decreases with latitude
    # Equator ~215K, poles ~160K
    t_mean = 215.0 - 55.0 * (math.sin(lat_r) ** 2)

    # Seasonal variation: largest at high latitudes
    # Peak warm at summer solstice (Ls=90 for N, Ls=270 for S)
    if lat_deg >= 0:
        t_seasonal = 25.0 * math.cos(ls_r - math.pi / 2) * abs(math.sin(lat_r))
    else:
        t_seasonal = 25.0 * math.cos(ls_r - 3 * math.pi / 2) * abs(math.sin(lat_r))

    # Elevation lapse rate: ~1.5 K per 1000m (thinner atmosphere = colder)
    t_elev = -1.5 * (elevation_m / 1000.0)

    t_base = t_mean + t_seasonal + t_elev

    # Diurnal swing: larger at equator (~60K), smaller at poles (~20K)
    diurnal = 60.0 * math.cos(lat_r) ** 2 + 20.0

    # Clamp above CO2 frost point
    t_base = max(t_base, CO2_FROST_POINT_K)
    t_max = max(t_base + diurnal / 2, CO2_FROST_POINT_K)
    t_min = max(t_base - diurnal / 2, CO2_FROST_POINT_K)

    return {
        "mean_k": round(t_base, 1),
        "max_k": round(t_max, 1),
        "min_k": round(t_min, 1),
        "diurnal_swing_k": round(diurnal, 1),
    }


def surface_pressure_pa(elevation_m: float) -> float:
    """Surface pressure from barometric formula. Returns Pa."""
    return MEAN_SURFACE_PRESSURE_PA * math.exp(-elevation_m / SCALE_HEIGHT_M)


def dust_opacity(lat_deg: float, ls_deg: float) -> dict:
    """
    Estimate column dust opacity (tau) at 9.3 µm.

    Background tau ~0.2-0.5. Dust storm season (Ls 180-360, low latitudes)
    raises tau to 1-5.

    Returns dict with tau_mean, tau_peak, storm_risk (str).
    """
    lat_r = math.radians(lat_deg)

    # Background dust: higher near equator, lower at poles
    tau_bg = 0.25 + 0.15 * math.cos(lat_r) ** 2

    # Dust storm season: Ls 180-360 (southern spring/summer)
    # Peak around Ls ~270
    if 150 < ls_deg < 360:
        storm_factor = math.sin(math.pi * (ls_deg - 150) / 210) ** 2
        # Storms mainly affect latitudes between 60S and 40N
        if -60 < lat_deg < 40:
            lat_factor = 1.0 - 0.5 * (abs(lat_deg) / 60.0) ** 2
            tau_storm = 2.0 * storm_factor * max(lat_factor, 0)
        else:
            tau_storm = 0.3 * storm_factor
    else:
        tau_storm = 0.0

    tau_mean = tau_bg + tau_storm * 0.3  # Mean includes some storm contribution
    tau_peak = tau_bg + tau_storm

    # Risk classification
    if tau_peak > 2.0:
        storm_risk = "HIGH"
    elif tau_peak > 1.0:
        storm_risk = "MODERATE"
    else:
        storm_risk = "LOW"

    return {
        "tau_mean": round(tau_mean, 2),
        "tau_peak": round(tau_peak, 2),
        "storm_risk": storm_risk,
    }


def wind_speed(lat_deg: float, ls_deg: float) -> dict:
    """
    Estimate mean surface wind speed (m/s).

    Driven by Hadley circulation, thermal tides, and topographic flows.
    Typical: 2-7 m/s. Storm conditions: up to 25 m/s.

    Returns dict with mean_ms, gust_ms, wind_hazard (str).
    """
    lat_r = math.radians(lat_deg)

    # Base wind from Hadley circulation: stronger near equator
    base_wind = 4.0 + 2.0 * math.cos(lat_r)

    # Seasonal modulation: stronger during dust storm season
    if 180 < ls_deg < 330:
        seasonal = 1.5 * math.sin(math.pi * (ls_deg - 180) / 150) ** 2
    else:
        seasonal = 0.0

    # Polar jets at high latitudes during winter
    if lat_deg > 50 and (ls_deg > 270 or ls_deg < 90):
        polar_jet = 2.0
    elif lat_deg < -50 and 90 < ls_deg < 270:
        polar_jet = 2.0
    else:
        polar_jet = 0.0

    mean_wind = base_wind + seasonal + polar_jet
    gust_wind = mean_wind * 2.5  # Gusts ~2.5x mean

    if gust_wind > 20:
        hazard = "HIGH"
    elif gust_wind > 12:
        hazard = "MODERATE"
    else:
        hazard = "LOW"

    return {
        "mean_ms": round(mean_wind, 1),
        "gust_ms": round(gust_wind, 1),
        "wind_hazard": hazard,
    }


def co2_frost_probability(lat_deg: float, ls_deg: float, elevation_m: float = 0.0) -> dict:
    """
    Estimate CO2 frost probability (seasonal and diurnal).

    CO2 frost forms when surface temp drops below ~148K.
    Seasonal CO2 caps extend to ~50° latitude in winter.

    Returns dict with frost_probability (0-1), seasonal_frost (bool), description.
    """
    temp = surface_temperature_k(lat_deg, ls_deg, elevation_m)

    # If minimum temperature is below frost point, frost is possible
    frost_margin = temp["min_k"] - CO2_FROST_POINT_K

    if frost_margin < 0:
        # Below frost point — seasonal cap territory
        prob = min(1.0, abs(frost_margin) / 20.0)
        seasonal = True
    elif frost_margin < 15:
        # Close to frost point — occasional nighttime frost
        prob = max(0.0, 0.3 * (1.0 - frost_margin / 15.0))
        seasonal = False
    else:
        prob = 0.0
        seasonal = False

    if prob > 0.7:
        desc = "Persistent seasonal CO2 frost cap — limits surface operations"
    elif prob > 0.3:
        desc = "Occasional CO2 frost — mostly nighttime, manageable"
    elif prob > 0:
        desc = "Rare frost events — minimal operational impact"
    else:
        desc = "No CO2 frost expected at this latitude/season"

    return {
        "frost_probability": round(prob, 2),
        "seasonal_frost": seasonal,
        "description": desc,
    }


# ---------------------------------------------------------------------------
# Composite climate analysis for agent task
# ---------------------------------------------------------------------------

@dataclass
class ClimateResult:
    """Full climate assessment for a region."""
    lat: float
    lon: float
    elevation_m: float
    temperature: dict
    pressure_pa: float
    dust: dict
    wind: dict
    frost: dict
    climate_score: int          # 0-10 for agent scoring (backward compat)
    climate_summary: str
    annual_stats: dict          # Aggregated over 12 Ls bins
    seasonal_profile: list      # Per-Ls-bin climate data (12 entries)
    climate_subscore: float     # 0-1 formula-based score
    climate_score_formula: str  # Human-readable formula explanation


_LS_BINS = list(range(0, 360, 30))  # 12 bins: 0, 30, 60, ..., 330

_LS_LABELS = [
    "Ls 0-30 (N Early Spring)",
    "Ls 30-60 (N Mid Spring)",
    "Ls 60-90 (N Late Spring)",
    "Ls 90-120 (N Early Summer)",
    "Ls 120-150 (N Mid Summer)",
    "Ls 150-180 (N Late Summer)",
    "Ls 180-210 (N Early Autumn)",
    "Ls 210-240 (N Mid Autumn)",
    "Ls 240-270 (N Late Autumn)",
    "Ls 270-300 (N Early Winter)",
    "Ls 300-330 (N Mid Winter)",
    "Ls 330-360 (N Late Winter)",
]

_CLIMATE_SCORE_FORMULA = (
    "climate_subscore = 0.30 * t_score + 0.25 * d_score + 0.20 * w_score + 0.25 * f_score "
    "where t_score = clamp(1 - |annual_t_mean - 215| / 70, 0, 1) if outside 180-250K else 1.0; "
    "d_score = clamp(1 - annual_tau_mean / 1.0, 0, 1); "
    "w_score = clamp(1 - (annual_wind_mean - 3.0) / 12.0, 0, 1); "
    "f_score = 1 - max_frost_prob"
)


def analyze_climate(lat: float, lon: float) -> ClimateResult:
    """
    Full climate analysis for a single point.
    Computes stats across 12 Ls bins (every 30 deg) and produces a
    formula-based 0-1 climate_subscore plus backward-compatible 0-10 score.
    """
    elevation = get_elevation_m(lat, lon)
    n_bins = len(_LS_BINS)

    # Sample 12 Ls bins
    all_temps = []
    all_dust = []
    all_wind = []
    all_frost = []
    seasonal_profile = []

    for i, ls in enumerate(_LS_BINS):
        t = surface_temperature_k(lat, ls, elevation)
        d = dust_opacity(lat, ls)
        w = wind_speed(lat, ls)
        f = co2_frost_probability(lat, ls, elevation)
        all_temps.append(t)
        all_dust.append(d)
        all_wind.append(w)
        all_frost.append(f)
        seasonal_profile.append({
            "ls": ls,
            "ls_label": _LS_LABELS[i],
            "temperature": {"mean_k": t["mean_k"], "max_k": t["max_k"], "min_k": t["min_k"]},
            "pressure_pa": round(surface_pressure_pa(elevation), 1),
            "dust_tau": d["tau_mean"],
            "wind_mean_ms": w["mean_ms"],
            "frost_probability": f["frost_probability"],
        })

    # Annual aggregates
    annual_t_min = min(t["min_k"] for t in all_temps)
    annual_t_max = max(t["max_k"] for t in all_temps)
    annual_t_mean = sum(t["mean_k"] for t in all_temps) / n_bins
    annual_tau_mean = sum(d["tau_mean"] for d in all_dust) / n_bins
    annual_tau_peak = max(d["tau_peak"] for d in all_dust)
    annual_wind_mean = sum(w["mean_ms"] for w in all_wind) / n_bins
    annual_wind_gust = max(w["gust_ms"] for w in all_wind)
    max_frost_prob = max(f["frost_probability"] for f in all_frost)
    any_seasonal_frost = any(f["seasonal_frost"] for f in all_frost)

    # Derived physical quantities
    seasonal_means = [t["mean_k"] for t in all_temps]
    temperature_amplitude_k = max(seasonal_means) - min(seasonal_means)

    all_tau_means = [d["tau_mean"] for d in all_dust]
    dust_variability = max(all_tau_means) - min(all_tau_means)

    # Pressure is elevation-dependent (constant across Ls in this model),
    # but include for completeness — bins share the same pressure here.
    bin_pressure = surface_pressure_pa(elevation)
    pressure_range_pa = 0.0  # Same elevation → same pressure per bin

    frost_duration_ls = sum(
        1 for f in all_frost if f["frost_probability"] > 0.3
    )

    annual_stats = {
        "temp_min_k": round(annual_t_min, 1),
        "temp_max_k": round(annual_t_max, 1),
        "temp_mean_k": round(annual_t_mean, 1),
        "temp_range_k": round(annual_t_max - annual_t_min, 1),
        "temperature_amplitude_k": round(temperature_amplitude_k, 1),
        "dust_tau_mean": round(annual_tau_mean, 2),
        "dust_tau_peak": round(annual_tau_peak, 2),
        "dust_variability": round(dust_variability, 3),
        "wind_mean_ms": round(annual_wind_mean, 1),
        "wind_gust_max_ms": round(annual_wind_gust, 1),
        "frost_max_probability": round(max_frost_prob, 2),
        "frost_duration_ls": frost_duration_ls,
        "seasonal_frost": any_seasonal_frost,
        "pressure_pa": round(bin_pressure, 1),
        "pressure_range_pa": round(pressure_range_pa, 1),
        "elevation_m": round(elevation, 0),
        "seasonal_breakdown": seasonal_profile,
    }

    # ── Formula-based climate subscore (0-1) ──
    # Temperature suitability (0-1): best if mean 180-250K
    if 180 <= annual_t_mean <= 250:
        t_score = 1.0
    else:
        t_score = max(0.0, 1.0 - abs(annual_t_mean - 215) / 70.0)

    # Dust suitability (0-1): lower dust = better
    d_score = max(0.0, 1.0 - annual_tau_mean / 1.0)

    # Wind suitability (0-1): lower wind = better
    w_score = max(0.0, 1.0 - (annual_wind_mean - 3.0) / 12.0)

    # Frost penalty (0-1): no frost = 1.0
    f_score = 1.0 - max_frost_prob

    # Composite with explicit weights
    climate_subscore = (
        0.30 * t_score + 0.25 * d_score + 0.20 * w_score + 0.25 * f_score
    )
    climate_subscore = round(max(0.0, min(1.0, climate_subscore)), 4)

    # Backward-compatible integer score (0-10)
    climate_score = round(climate_subscore * 10)

    # Build summary
    parts = []
    parts.append(f"Annual mean temperature {annual_t_mean:.0f} K ({annual_t_min:.0f}\u2013{annual_t_max:.0f} K range)")
    parts.append(f"Surface pressure {annual_stats['pressure_pa']:.0f} Pa at {elevation:.0f} m elevation")

    if annual_tau_peak > 2.0:
        parts.append(f"HIGH dust storm risk (peak tau {annual_tau_peak:.1f})")
    elif annual_tau_peak > 1.0:
        parts.append(f"Moderate dust activity (peak tau {annual_tau_peak:.1f})")
    else:
        parts.append(f"Low dust environment (peak tau {annual_tau_peak:.1f})")

    if any_seasonal_frost:
        parts.append(f"Seasonal CO2 frost present ({frost_duration_ls}/12 Ls bins) \u2014 limits winter operations")
    elif max_frost_prob > 0:
        parts.append(f"Minor frost risk ({max_frost_prob:.0%} probability)")

    summary = ". ".join(parts) + "."

    # Use best-season temp for the primary display
    best_season_idx = max(range(n_bins), key=lambda i: all_temps[i]["mean_k"])

    return ClimateResult(
        lat=lat,
        lon=lon,
        elevation_m=elevation,
        temperature=all_temps[best_season_idx],
        pressure_pa=annual_stats["pressure_pa"],
        dust=all_dust[best_season_idx],
        wind=all_wind[best_season_idx],
        frost=all_frost[best_season_idx],
        climate_score=climate_score,
        climate_summary=summary,
        annual_stats=annual_stats,
        seasonal_profile=seasonal_profile,
        climate_subscore=climate_subscore,
        climate_score_formula=_CLIMATE_SCORE_FORMULA,
    )


# ---------------------------------------------------------------------------
# Ice Stability Analysis (Phase 4)
# ---------------------------------------------------------------------------

# Clausius-Clapeyron constants for water ice sublimation
# Ref: Murphy & Koop (2005), Buck (1981)
_H2O_SUBLIMATION_HEAT_J = 51_058.0  # J/mol latent heat of sublimation
_R_GAS = 8.314  # J/(mol·K) universal gas constant
_TRIPLE_POINT_T_K = 273.16  # Water triple point temperature
_TRIPLE_POINT_P_PA = 611.657  # Water triple point pressure


def _water_vapor_pressure_pa(temp_k: float) -> float:
    """
    Water vapor saturation pressure over ice via Clausius-Clapeyron.

    P(T) = P_triple * exp(-ΔH_sub/R * (1/T - 1/T_triple))

    Valid for T < 273.16 K (ice phase).
    """
    if temp_k <= 0:
        return 0.0
    return _TRIPLE_POINT_P_PA * math.exp(
        -_H2O_SUBLIMATION_HEAT_J / _R_GAS * (1.0 / temp_k - 1.0 / _TRIPLE_POINT_T_K)
    )


def compute_ice_stability(
    lat_deg: float,
    lon: float = 0.0,
    elevation_m: float = 0.0,
) -> dict:
    """
    Determine whether near-surface water ice is thermodynamically stable.

    Uses annual mean ground temperature to compute equilibrium vapor
    pressure vs. ambient atmospheric water vapor partial pressure.

    On Mars, atmospheric H2O column ~10 pr-µm → surface mixing ratio
    yields partial pressure ~0.01-0.1 Pa. Ice is stable where the
    equilibrium vapor pressure at ground temperature is below this.

    Returns dict with:
        ice_table_stable (bool): Whether ice can persist at shallow depth
        annual_mean_temp_k (float): Mean annual ground temperature
        equilibrium_vapor_pressure_pa (float): H2O vapor pressure at mean T
        atmospheric_h2o_pa (float): Estimated atmospheric water vapor pressure
        stability_margin (float): ratio atm_h2o / equil_vp (>1 = stable)
        estimated_ice_table_depth_m (float or None): Estimated depth to ice table
        sublimation_regime (str): "stable", "marginal", "sublimating"
    """
    # Compute annual mean ground temperature (average across 12 Ls bins)
    temps = []
    for ls in _LS_BINS:
        t = surface_temperature_k(lat_deg, ls, elevation_m)
        temps.append(t["mean_k"])

    annual_mean_t = sum(temps) / len(temps)
    annual_min_t = min(temps)

    # Equilibrium vapor pressure at annual mean temperature
    equil_vp = _water_vapor_pressure_pa(annual_mean_t)

    # Atmospheric H2O partial pressure
    # Mars atmospheric H2O: ~10 precipitable microns at equator, ~30 at poles
    # At ~636 Pa surface pressure, mixing ratio ~1.5e-4
    # Partial pressure: ~0.01-0.1 Pa, latitude dependent
    atm_pressure = surface_pressure_pa(elevation_m)
    # Higher water vapor at mid-latitudes (seasonal sublimation from caps)
    abs_lat = abs(lat_deg)
    if abs_lat > 60:
        mixing_ratio = 3e-4  # Polar: higher near caps
    elif abs_lat > 30:
        mixing_ratio = 1.5e-4  # Mid-latitude
    else:
        mixing_ratio = 0.5e-4  # Equatorial: drier
    atm_h2o_pa = atm_pressure * mixing_ratio

    # Stability: ice is stable if equilibrium VP <= atmospheric H2O PP
    # A stability margin > 1 means ice can persist
    if equil_vp > 0:
        stability_margin = atm_h2o_pa / equil_vp
    else:
        stability_margin = float("inf")

    ice_stable = stability_margin >= 1.0

    # Sublimation regime classification
    if stability_margin >= 2.0:
        regime = "stable"
    elif stability_margin >= 0.5:
        regime = "marginal"
    else:
        regime = "sublimating"

    # Estimated ice table depth (Schorghofer 2005 approximation)
    # Depth increases toward equator as surface gets warmer
    # At polar latitudes: ~0 cm (surface ice), mid-lats: ~5-100 cm
    if ice_stable:
        if abs_lat > 60:
            ice_table_depth = 0.0  # Surface or very shallow
        elif abs_lat > 45:
            ice_table_depth = 0.05 + (60 - abs_lat) * 0.02  # 5-35 cm
        elif abs_lat > 30:
            ice_table_depth = 0.35 + (45 - abs_lat) * 0.1  # 0.35-1.85 m
        else:
            ice_table_depth = 2.0 + (30 - abs_lat) * 0.3  # 2-11 m
    elif regime == "marginal":
        # Marginally stable: ice may persist at greater depth
        ice_table_depth = 5.0 + (1.0 / max(stability_margin, 0.01)) * 2.0
    else:
        ice_table_depth = None  # Ice not expected to persist

    return {
        "ice_table_stable": ice_stable,
        "annual_mean_temp_k": round(annual_mean_t, 1),
        "annual_min_temp_k": round(annual_min_t, 1),
        "equilibrium_vapor_pressure_pa": round(equil_vp, 6),
        "atmospheric_h2o_pa": round(atm_h2o_pa, 6),
        "stability_margin": round(stability_margin, 3),
        "estimated_ice_table_depth_m": round(ice_table_depth, 2) if ice_table_depth is not None else None,
        "sublimation_regime": regime,
    }


def compute_seasonal_operation_window(
    lat_deg: float,
    lon: float = 0.0,
    elevation_m: float = 0.0,
) -> dict:
    """
    Determine which Ls bins are safe for surface operations.

    A bin is "safe" if:
      - No seasonal CO2 frost (frost_probability < 0.3)
      - Low dust risk (tau_peak < 2.0)
      - Acceptable wind (gust < 20 m/s)
      - Temperature above operational minimum (> 160 K mean)

    Returns dict with:
        safe_bins: list of Ls values that are safe
        n_safe_bins: count (out of 12)
        operational_fraction: fraction of Mars year safe for operations
        best_season_ls: Ls bin with best combined conditions
        worst_season_ls: Ls bin with worst conditions
        constraints: list of limiting factors
    """
    safe_bins = []
    bin_scores = []
    constraints = set()

    for i, ls in enumerate(_LS_BINS):
        t = surface_temperature_k(lat_deg, ls, elevation_m)
        d = dust_opacity(lat_deg, ls)
        w = wind_speed(lat_deg, ls)
        f = co2_frost_probability(lat_deg, ls, elevation_m)

        is_safe = True
        bin_score = 0.0

        # Frost check
        if f["frost_probability"] >= 0.3:
            is_safe = False
            constraints.add("CO2_frost")
        else:
            bin_score += 0.25 * (1.0 - f["frost_probability"])

        # Dust check
        if d["tau_peak"] >= 2.0:
            is_safe = False
            constraints.add("dust_storms")
        else:
            bin_score += 0.25 * max(0, 1.0 - d["tau_peak"] / 2.0)

        # Wind check
        if w["gust_ms"] >= 20:
            is_safe = False
            constraints.add("high_wind")
        else:
            bin_score += 0.25 * max(0, 1.0 - w["gust_ms"] / 20.0)

        # Temperature check
        if t["mean_k"] <= 160:
            is_safe = False
            constraints.add("extreme_cold")
        else:
            bin_score += 0.25 * min(1.0, (t["mean_k"] - 160) / 55.0)

        if is_safe:
            safe_bins.append(ls)
        bin_scores.append((ls, round(bin_score, 3)))

    n_safe = len(safe_bins)
    best = max(bin_scores, key=lambda x: x[1])
    worst = min(bin_scores, key=lambda x: x[1])

    return {
        "safe_bins": safe_bins,
        "n_safe_bins": n_safe,
        "total_bins": len(_LS_BINS),
        "operational_fraction": round(n_safe / len(_LS_BINS), 2),
        "best_season_ls": best[0],
        "best_season_score": best[1],
        "worst_season_ls": worst[0],
        "worst_season_score": worst[1],
        "constraints": sorted(constraints),
        "bin_scores": bin_scores,
    }


def climate_analysis_for_region(
    min_lat: float, max_lat: float, min_lon: float, max_lon: float,
) -> dict:
    """
    Climate analysis for a bounding box region.
    Samples center + 4 corners, returns aggregated stats.

    Returns a dict suitable for TaskResult.data.
    """
    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2

    # Sample points: center + 4 corners
    points = [
        (center_lat, center_lon),
        (min_lat, min_lon),
        (min_lat, max_lon),
        (max_lat, min_lon),
        (max_lat, max_lon),
    ]

    results = []
    for lat, lon in points:
        try:
            r = analyze_climate(lat, lon)
            results.append(r)
        except Exception as e:
            logger.warning(f"Climate analysis failed at ({lat}, {lon}): {e}")

    if not results:
        return {
            "success": False,
            "error": "Climate analysis failed for all sample points",
            "climate_score": 0,
            "climate_subscore": 0.0,
        }

    # Aggregate
    scores = [r.climate_score for r in results]
    subscores = [r.climate_subscore for r in results]
    avg_score = round(sum(scores) / len(scores))
    avg_subscore = round(sum(subscores) / len(subscores), 4)

    # Use center point as primary
    center = results[0]

    # Phase 4: Ice stability and seasonal operation window
    ice_stability = compute_ice_stability(center_lat, center_lon, center.elevation_m)
    seasonal_window = compute_seasonal_operation_window(center_lat, center_lon, center.elevation_m)

    return {
        "success": True,
        "center_lat": center_lat,
        "center_lon": center_lon,
        "elevation_m": center.elevation_m,
        "climate_score": avg_score,
        "climate_subscore": avg_subscore,
        "climate_score_formula": center.climate_score_formula,
        "climate_summary": center.climate_summary,
        "annual_stats": center.annual_stats,
        "seasonal_profile": center.seasonal_profile,
        "sample_points": len(results),
        "score_range": {"min": min(scores), "max": max(scores)},
        "subscore_range": {
            "min": round(min(subscores), 4),
            "max": round(max(subscores), 4),
        },
        "ice_stability": ice_stability,
        "seasonal_operation_window": seasonal_window,
    }
