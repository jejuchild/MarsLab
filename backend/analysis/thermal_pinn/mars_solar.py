"""
Mars solar flux model for thermal inversion.

Computes solar insolation at the Mars surface accounting for:
- Mars orbital mechanics (eccentricity, obliquity, Ls → season)
- Solar zenith angle (latitude, Ls, local solar time)
- Atmospheric dust opacity (simple Beer-Lambert with diffuse correction)

All functions are vectorized and support both numpy and scalar inputs.
"""
from __future__ import annotations

import numpy as np

# ── Mars orbital constants ──────────────────────────────────────
MARS_OBLIQUITY_DEG = 25.19          # axial tilt
MARS_ECCENTRICITY = 0.0934          # orbital eccentricity
MARS_LS_PERIHELION = 251.0          # Ls at perihelion (degrees)
MARS_SEMI_MAJOR_AU = 1.524          # semi-major axis (AU)
SOLAR_CONSTANT = 1361.0             # W/m² at 1 AU
STEFAN_BOLTZMANN = 5.670374419e-8   # W/m²/K⁴
MARS_SOL = 88_775.0                 # seconds per sol
MARS_YEAR_SOLS = 668.6
MARS_YEAR_SEC = MARS_YEAR_SOLS * MARS_SOL

# Default surface parameters for Arcadia Planitia
DEFAULT_ALBEDO = 0.25               # Bond albedo
DEFAULT_EMISSIVITY = 0.95           # thermal emissivity
DEFAULT_DUST_TAU = 0.5              # visible dust optical depth


def solar_declination(Ls_deg):
    """
    Solar declination angle (degrees) as function of Ls.

    δ = arcsin(sin(obliquity) · sin(Ls))
    """
    obl = np.radians(MARS_OBLIQUITY_DEG)
    Ls = np.radians(np.asarray(Ls_deg, dtype=np.float64))
    return np.degrees(np.arcsin(np.sin(obl) * np.sin(Ls)))


def heliocentric_distance(Ls_deg):
    """
    Mars-Sun distance in AU as function of Ls.

    r = a(1-e²) / (1 + e·cos(ν))
    where ν = Ls - Ls_perihelion (true anomaly approximation).
    """
    e = MARS_ECCENTRICITY
    nu = np.radians(np.asarray(Ls_deg, dtype=np.float64) - MARS_LS_PERIHELION)
    return MARS_SEMI_MAJOR_AU * (1 - e**2) / (1 + e * np.cos(nu))


def cos_solar_zenith(lat_deg, Ls_deg, local_time_hr):
    """
    Cosine of solar zenith angle, clamped ≥ 0 (below horizon = 0).

    cos(θ) = sin(δ)sin(φ) + cos(δ)cos(φ)cos(h)

    Parameters:
        lat_deg: latitude (degrees N, scalar or array)
        Ls_deg: solar longitude (degrees, scalar or array)
        local_time_hr: local solar time (hours 0-24, scalar or array)
    """
    dec = np.radians(solar_declination(Ls_deg))
    lat = np.radians(np.asarray(lat_deg, dtype=np.float64))
    h = np.radians((np.asarray(local_time_hr, dtype=np.float64) - 12.0) * 15.0)

    cos_z = np.sin(dec) * np.sin(lat) + np.cos(dec) * np.cos(lat) * np.cos(h)
    return np.maximum(cos_z, 0.0)


def surface_solar_flux(lat_deg, Ls_deg, local_time_hr,
                       albedo=DEFAULT_ALBEDO, tau=DEFAULT_DUST_TAU):
    """
    Absorbed solar flux at Mars surface (W/m²).

    Includes:
    - Inverse-square law for Mars-Sun distance
    - Cosine projection (zenith angle)
    - Bond albedo reflection
    - Atmospheric dust extinction (Beer-Lambert direct + isotropic diffuse)

    Parameters:
        lat_deg: latitude (degrees N)
        Ls_deg: solar longitude (degrees)
        local_time_hr: local solar time (hours)
        albedo: surface Bond albedo
        tau: dust visible optical depth

    Returns:
        Q_abs: absorbed flux (W/m²), 0 when sun is below horizon
    """
    r_au = heliocentric_distance(Ls_deg)
    cos_z = cos_solar_zenith(lat_deg, Ls_deg, local_time_hr)

    # Top-of-atmosphere flux
    F_toa = SOLAR_CONSTANT / r_au**2

    # Direct beam (Beer-Lambert through atmosphere)
    # Avoid division by zero when sun is at horizon
    cos_z_safe = np.where(cos_z > 0.01, cos_z, 0.01)
    airmass = 1.0 / cos_z_safe
    F_direct = F_toa * cos_z * np.exp(-tau * airmass)

    # Diffuse (scattered) component: ~40% of absorbed atmospheric flux
    # Pollack et al. (1990) parameterization
    F_diffuse = 0.4 * F_toa * (1 - np.exp(-tau)) * cos_z

    # Total absorbed by surface
    Q_abs = (1 - albedo) * (F_direct + F_diffuse)

    # Zero when sun below horizon
    Q_abs = np.where(cos_z > 0.0, Q_abs, 0.0)
    return Q_abs


def surface_solar_flux_timeseries(lat_deg, Ls_start, n_steps, dt_sec,
                                  albedo=DEFAULT_ALBEDO, tau=DEFAULT_DUST_TAU):
    """
    Generate solar flux timeseries for FDM integration.

    Parameters:
        lat_deg: latitude (degrees N)
        Ls_start: starting Ls (degrees)
        n_steps: number of timesteps
        dt_sec: timestep in seconds
        albedo: surface Bond albedo
        tau: dust optical depth

    Returns:
        Q_series: (n_steps,) array of absorbed surface flux (W/m²)
        Ls_series: (n_steps,) array of Ls values
        lt_series: (n_steps,) array of local times (hours)
    """
    t_sec = np.arange(n_steps, dtype=np.float64) * dt_sec

    # Local solar time: cycles every sol
    lt_series = (t_sec % MARS_SOL) / MARS_SOL * 24.0

    # Ls progression: ~0.54°/sol
    Ls_series = Ls_start + t_sec / MARS_YEAR_SEC * 360.0

    Q_series = surface_solar_flux(lat_deg, Ls_series, lt_series, albedo, tau)

    return Q_series, Ls_series, lt_series
