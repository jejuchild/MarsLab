"""
Dust contamination risk assessment for CRISM spectral data.

Uses Mars atmospheric dust opacity model + spectral slope analysis to flag
spectra likely affected by atmospheric dust interference.

Dust characteristics:
- Raises continuum reflectance uniformly
- Suppresses diagnostic absorption band depths by 30-50% at high tau
- Red spectral slope in 1-2.5 um from ferric dust absorption
- High dust: Ls 150-360 (southern spring/summer), mainly -60 to +40 lat
"""

import math
from dataclasses import dataclass


@dataclass
class DustAssessment:
    tau_estimated: float
    risk_level: str
    spectral_slope: float | None
    band_depth_suppression_pct: float
    warning_message: str | None


def assess_dust_risk(
    lat: float,
    lon: float,
    ls: float | None = None,
    wavelengths: list[float | None] | None = None,
    reflectance: list[float | None] | None = None,
) -> DustAssessment:
    _ = lon
    tau = _estimate_tau(lat, ls)

    slope = None
    if wavelengths and reflectance:
        slope = _compute_spectral_slope(wavelengths, reflectance)

    risk_level = "LOW"
    suppression = 0.0
    warning = None

    if tau > 2.0:
        risk_level = "HIGH"
        suppression = min(60.0, tau * 15.0)
        warning = f"High atmospheric dust opacity (tau~{tau:.1f}). Band depths may be suppressed by ~{suppression:.0f}%. Mineral identifications should be treated with caution."
    elif tau > 1.0:
        risk_level = "MODERATE"
        suppression = min(40.0, tau * 12.0)
        warning = f"Moderate dust opacity (tau~{tau:.1f}). Band depths may be suppressed by ~{suppression:.0f}%."
    elif slope is not None and slope > 0.08:
        risk_level = "MODERATE"
        suppression = 20.0
        warning = "Red spectral slope detected - possible surface dust coating affecting mineral signatures."

    return DustAssessment(
        tau_estimated=round(tau, 3),
        risk_level=risk_level,
        spectral_slope=round(slope, 5) if slope is not None else None,
        band_depth_suppression_pct=round(suppression, 1),
        warning_message=warning,
    )


def _estimate_tau(lat: float, ls: float | None) -> float:
    base_tau = 0.3 + 0.15 * math.exp(-(lat ** 2) / (2 * 30 ** 2))

    if ls is None:
        return base_tau

    if 150 <= ls <= 360:
        season_factor = math.exp(-((ls - 270) ** 2) / (2 * 50 ** 2))
        lat_factor = 1.0 if -60 <= lat <= 40 else 0.3
        storm_tau = 1.5 * season_factor * lat_factor
        return base_tau + storm_tau

    return base_tau


def _compute_spectral_slope(wavelengths: list[float | None], reflectance: list[float | None]) -> float | None:
    if not wavelengths or not reflectance:
        return None

    idx_short = None
    idx_long = None
    for i, w in enumerate(wavelengths):
        if i >= len(reflectance):
            continue
        r = reflectance[i]
        if w is not None and 1.25 <= w <= 1.35 and r is not None:
            idx_short = i
        if w is not None and 2.25 <= w <= 2.35 and r is not None:
            idx_long = i

    if idx_short is None or idx_long is None:
        return None

    r_short = reflectance[idx_short]
    r_long = reflectance[idx_long]
    w_short = wavelengths[idx_short]
    w_long = wavelengths[idx_long]

    if (
        r_short is None
        or r_long is None
        or w_short is None
        or w_long is None
        or r_short <= 0
        or w_long == w_short
    ):
        return None

    return (r_long - r_short) / (r_short * (w_long - w_short))
