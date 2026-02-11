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
        # MOLA uses 0-360 lon
        lon360 = lon % 360
        row, col = ds.index(lon360, lat)
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
    ls_r = math.radians(ls_deg)

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
    ls_r = math.radians(ls_deg)

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
    climate_score: int          # 0-10 for agent scoring
    climate_summary: str
    annual_stats: dict          # Aggregated over 4 seasons


def analyze_climate(lat: float, lon: float) -> ClimateResult:
    """
    Full climate analysis for a single point.
    Computes stats across 4 seasons (Ls = 0, 90, 180, 270).
    """
    elevation = get_elevation_m(lat, lon)

    # Sample 4 seasons
    seasons = [0, 90, 180, 270]
    season_names = ["N Spring", "N Summer", "N Autumn", "N Winter"]
    all_temps = []
    all_dust = []
    all_wind = []
    all_frost = []
    seasonal_data = []

    for ls in seasons:
        t = surface_temperature_k(lat, ls, elevation)
        d = dust_opacity(lat, ls)
        w = wind_speed(lat, ls)
        f = co2_frost_probability(lat, ls, elevation)
        all_temps.append(t)
        all_dust.append(d)
        all_wind.append(w)
        all_frost.append(f)
        seasonal_data.append({"ls": ls, "temp": t, "dust": d, "wind": w, "frost": f})

    # Annual aggregates
    annual_t_min = min(t["min_k"] for t in all_temps)
    annual_t_max = max(t["max_k"] for t in all_temps)
    annual_t_mean = sum(t["mean_k"] for t in all_temps) / 4
    annual_tau_mean = sum(d["tau_mean"] for d in all_dust) / 4
    annual_tau_peak = max(d["tau_peak"] for d in all_dust)
    annual_wind_mean = sum(w["mean_ms"] for w in all_wind) / 4
    annual_wind_gust = max(w["gust_ms"] for w in all_wind)
    max_frost_prob = max(f["frost_probability"] for f in all_frost)
    any_seasonal_frost = any(f["seasonal_frost"] for f in all_frost)

    annual_stats = {
        "temp_min_k": round(annual_t_min, 1),
        "temp_max_k": round(annual_t_max, 1),
        "temp_mean_k": round(annual_t_mean, 1),
        "temp_range_k": round(annual_t_max - annual_t_min, 1),
        "dust_tau_mean": round(annual_tau_mean, 2),
        "dust_tau_peak": round(annual_tau_peak, 2),
        "wind_mean_ms": round(annual_wind_mean, 1),
        "wind_gust_max_ms": round(annual_wind_gust, 1),
        "frost_max_probability": round(max_frost_prob, 2),
        "seasonal_frost": any_seasonal_frost,
        "pressure_pa": round(surface_pressure_pa(elevation), 1),
        "elevation_m": round(elevation, 0),
        "seasonal_breakdown": seasonal_data,
    }

    # ── Climate score (0-10) ──
    score = 0

    # Temperature livability (0-3): favorable if mean 180-250K
    if 180 <= annual_t_mean <= 250:
        score += 3
    elif 160 <= annual_t_mean < 180 or 250 < annual_t_mean <= 270:
        score += 1

    # Dust (0-2): low annual dust is better
    if annual_tau_mean < 0.4:
        score += 2
    elif annual_tau_mean < 0.7:
        score += 1

    # Wind (0-2): low wind is better for operations
    if annual_wind_mean < 7:
        score += 2
    elif annual_wind_mean < 10:
        score += 1

    # CO2 frost (0-3): no persistent frost is better
    if max_frost_prob < 0.05:
        score += 3
    elif max_frost_prob < 0.3:
        score += 2
    elif not any_seasonal_frost:
        score += 1

    # Build summary
    parts = []
    parts.append(f"Annual mean temperature {annual_t_mean:.0f} K ({annual_t_min:.0f}–{annual_t_max:.0f} K range)")
    parts.append(f"Surface pressure {annual_stats['pressure_pa']:.0f} Pa at {elevation:.0f} m elevation")

    if annual_tau_peak > 2.0:
        parts.append(f"HIGH dust storm risk (peak tau {annual_tau_peak:.1f})")
    elif annual_tau_peak > 1.0:
        parts.append(f"Moderate dust activity (peak tau {annual_tau_peak:.1f})")
    else:
        parts.append(f"Low dust environment (peak tau {annual_tau_peak:.1f})")

    if any_seasonal_frost:
        parts.append("Seasonal CO2 frost present — limits winter operations")
    elif max_frost_prob > 0:
        parts.append(f"Minor frost risk ({max_frost_prob:.0%} probability)")

    summary = ". ".join(parts) + "."

    # Use best-season temp for the primary display
    best_season_idx = max(range(4), key=lambda i: all_temps[i]["mean_k"])

    return ClimateResult(
        lat=lat,
        lon=lon,
        elevation_m=elevation,
        temperature=all_temps[best_season_idx],
        pressure_pa=annual_stats["pressure_pa"],
        dust=all_dust[best_season_idx],
        wind=all_wind[best_season_idx],
        frost=all_frost[best_season_idx],
        climate_score=score,
        climate_summary=summary,
        annual_stats=annual_stats,
    )


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
        }

    # Aggregate
    scores = [r.climate_score for r in results]
    avg_score = round(sum(scores) / len(scores))

    # Use center point as primary
    center = results[0]

    return {
        "success": True,
        "center_lat": center_lat,
        "center_lon": center_lon,
        "elevation_m": center.elevation_m,
        "climate_score": avg_score,
        "climate_summary": center.climate_summary,
        "annual_stats": center.annual_stats,
        "sample_points": len(results),
        "score_range": {"min": min(scores), "max": max(scores)},
    }
