"""
Band parameter extraction for CRISM spectra.

Computes diagnostic band depths:
  BD1500 - 1.5 um H2O ice absorption
  BD1900 - 1.9 um H2O absorption
  BD2100 - 2.1 um mono-hydrated sulfate / ice
  BD2200 - 2.2 um Al-OH absorption

Band depth = 1 - R(center) / R(continuum)
where R(continuum) is linearly interpolated between shoulders.
"""
import numpy as np
from typing import Optional, Tuple


def _find_nearest_band(wavelengths: np.ndarray, target_um: float) -> int:
    """Find index of nearest wavelength to target."""
    return int(np.argmin(np.abs(wavelengths - target_um)))


def _band_depth(reflectance: np.ndarray, wavelengths: np.ndarray,
                center_um: float, shoulder_lo_um: float, shoulder_hi_um: float) -> Optional[float]:
    """
    Compute band depth using linear continuum between two shoulders.

    BD = 1 - R_center / R_continuum
    R_continuum = linear interpolation of R at shoulder wavelengths
    """
    idx_center = _find_nearest_band(wavelengths, center_um)
    idx_lo = _find_nearest_band(wavelengths, shoulder_lo_um)
    idx_hi = _find_nearest_band(wavelengths, shoulder_hi_um)

    r_center = reflectance[idx_center]
    r_lo = reflectance[idx_lo]
    r_hi = reflectance[idx_hi]

    # Check for invalid values
    if not (np.isfinite(r_center) and np.isfinite(r_lo) and np.isfinite(r_hi)):
        return None
    if r_lo <= 0 or r_hi <= 0 or r_center <= 0:
        return None

    # Linear continuum at center wavelength
    wl_center = wavelengths[idx_center]
    wl_lo = wavelengths[idx_lo]
    wl_hi = wavelengths[idx_hi]

    if abs(wl_hi - wl_lo) < 1e-6:
        return None

    frac = (wl_center - wl_lo) / (wl_hi - wl_lo)
    r_continuum = r_lo + frac * (r_hi - r_lo)

    if r_continuum <= 0:
        return None

    bd = 1.0 - (r_center / r_continuum)
    return float(bd)


def compute_BD1500(reflectance: np.ndarray, wavelengths: np.ndarray) -> Optional[float]:
    """
    BD1500: 1.5 um H2O ice absorption band depth.
    Center: 1.50 um, Shoulders: 1.33 um, 1.81 um
    """
    return _band_depth(reflectance, wavelengths, 1.50, 1.33, 1.81)


def compute_BD1900(reflectance: np.ndarray, wavelengths: np.ndarray) -> Optional[float]:
    """
    BD1900: 1.9 um H2O absorption band depth.
    Center: 1.93 um, Shoulders: 1.85 um, 2.06 um
    """
    return _band_depth(reflectance, wavelengths, 1.93, 1.85, 2.06)


def compute_BD2100(reflectance: np.ndarray, wavelengths: np.ndarray) -> Optional[float]:
    """
    BD2100: 2.1 um mono-hydrated sulfate / ice band depth.
    Center: 2.13 um, Shoulders: 1.93 um, 2.25 um
    """
    return _band_depth(reflectance, wavelengths, 2.13, 1.93, 2.25)


def compute_BD2200(reflectance: np.ndarray, wavelengths: np.ndarray) -> Optional[float]:
    """
    BD2200: 2.2 um Al-OH absorption band depth.
    Center: 2.21 um, Shoulders: 2.14 um, 2.29 um
    """
    return _band_depth(reflectance, wavelengths, 2.21, 2.14, 2.29)


def compute_all_band_params(reflectance: np.ndarray, wavelengths: np.ndarray) -> dict:
    """Compute all four band parameters for a single spectrum."""
    return {
        "BD1500": compute_BD1500(reflectance, wavelengths),
        "BD1900": compute_BD1900(reflectance, wavelengths),
        "BD2100": compute_BD2100(reflectance, wavelengths),
        "BD2200": compute_BD2200(reflectance, wavelengths),
    }
