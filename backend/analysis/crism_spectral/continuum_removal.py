"""
Continuum removal via convex hull method.

For each pixel spectrum, computes the upper convex hull envelope
and divides the spectrum by this envelope to normalize absorption
features to a [0, 1] scale where 1 = continuum level.
"""
import numpy as np
from typing import Tuple


def compute_convex_hull_continuum(wavelengths: np.ndarray, reflectance: np.ndarray) -> np.ndarray:
    """
    Compute the convex hull continuum for a single spectrum.

    Args:
        wavelengths: 1D array of wavelength values (um)
        reflectance: 1D array of reflectance values

    Returns:
        continuum: 1D array of continuum values (same shape as reflectance)
    """
    n = len(wavelengths)
    if n < 3:
        return np.ones_like(reflectance)

    # Build upper convex hull using Andrew's monotone chain
    # Points are (wavelength, reflectance)
    points = list(zip(wavelengths, reflectance))

    # Upper hull
    upper = []
    for p in points:
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) >= 0:
            upper.pop()
        upper.append(p)

    if len(upper) < 2:
        return np.ones_like(reflectance) * np.max(reflectance)

    # Interpolate continuum at each wavelength
    hull_wl = np.array([p[0] for p in upper])
    hull_ref = np.array([p[1] for p in upper])

    continuum = np.interp(wavelengths, hull_wl, hull_ref)

    # Ensure continuum >= reflectance (numerical safety)
    continuum = np.maximum(continuum, reflectance)

    return continuum


def remove_continuum(wavelengths: np.ndarray, reflectance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform continuum removal on a spectrum.

    Args:
        wavelengths: 1D array of wavelength values
        reflectance: 1D array of reflectance values

    Returns:
        (cr_spectrum, continuum): continuum-removed spectrum and the continuum itself
        cr_spectrum = reflectance / continuum, range [0, 1]
    """
    # Mask invalid values
    valid = np.isfinite(reflectance) & (reflectance > 0) & (reflectance < 10.0)

    if np.sum(valid) < 10:
        return np.ones_like(reflectance), np.ones_like(reflectance)

    # Work only with valid points for hull computation
    wl_valid = wavelengths[valid]
    ref_valid = reflectance[valid]

    continuum_valid = compute_convex_hull_continuum(wl_valid, ref_valid)

    # Interpolate continuum back to full wavelength grid
    continuum = np.interp(wavelengths, wl_valid, continuum_valid)

    # Avoid division by zero
    safe_continuum = np.where(continuum > 1e-8, continuum, 1.0)
    cr_spectrum = reflectance / safe_continuum

    # Clamp to [0, 1]
    cr_spectrum = np.clip(cr_spectrum, 0.0, 1.0)

    return cr_spectrum, continuum


def remove_continuum_batch(wavelengths: np.ndarray, spectra: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Batch continuum removal for multiple pixels.

    Args:
        wavelengths: 1D array of wavelength values (n_bands,)
        spectra: 2D array of reflectance (n_pixels, n_bands)

    Returns:
        (cr_spectra, continua): both (n_pixels, n_bands)
    """
    n_pixels, n_bands = spectra.shape
    cr_spectra = np.ones_like(spectra)
    continua = np.ones_like(spectra)

    for i in range(n_pixels):
        cr_spectra[i], continua[i] = remove_continuum(wavelengths, spectra[i])

    return cr_spectra, continua


def _cross(O, A, B):
    """Cross product for 2D points (used in convex hull)."""
    return (A[0] - O[0]) * (B[1] - O[1]) - (A[1] - O[1]) * (B[0] - O[0])
