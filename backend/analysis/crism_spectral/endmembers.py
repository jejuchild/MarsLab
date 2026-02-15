"""
USGS spectral library endmembers resampled to CRISM wavelengths.

Reference spectra for SAM classification. These are synthetic
representations of key diagnostic features at CRISM spectral resolution.

Sources:
  - Water ice: Based on crystalline H2O ice at 200K (Warren & Brandt 2008)
  - Gypsum: USGS GDS11 CaSO4*2H2O
  - Polyhydrated sulfate: Based on Mg-sulfate hydrate
  - Basalt: USGS BAS04 average basalt
"""
import numpy as np


def _generate_water_ice_spectrum(wavelengths: np.ndarray) -> np.ndarray:
    """
    Generate synthetic water ice reflectance spectrum.

    Key features:
      - Strong absorption at 1.5 um (O-H stretch overtone)
      - Strong absorption at 2.0 um (O-H combination band)
      - Moderate absorption at 1.25 um
      - High reflectance baseline (0.5-0.8)
    """
    spec = np.ones_like(wavelengths) * 0.65  # baseline

    # 1.25 um absorption
    spec *= 1.0 - 0.15 * np.exp(-((wavelengths - 1.25) ** 2) / (2 * 0.04 ** 2))
    # 1.5 um deep absorption (H2O ice diagnostic)
    spec *= 1.0 - 0.45 * np.exp(-((wavelengths - 1.50) ** 2) / (2 * 0.06 ** 2))
    # 2.0 um deep absorption (H2O ice diagnostic)
    spec *= 1.0 - 0.50 * np.exp(-((wavelengths - 2.02) ** 2) / (2 * 0.08 ** 2))
    # 3.0 um Fresnel peak then absorption
    spec *= 1.0 - 0.30 * np.exp(-((wavelengths - 3.10) ** 2) / (2 * 0.15 ** 2))

    return np.clip(spec, 0.01, 1.0)


def _generate_gypsum_spectrum(wavelengths: np.ndarray) -> np.ndarray:
    """
    Generate synthetic gypsum (CaSO4*2H2O) spectrum.

    Key features:
      - Absorptions at 1.45, 1.75, 1.94, 2.21 um
      - Moderate reflectance baseline (~0.6)
    """
    spec = np.ones_like(wavelengths) * 0.60

    spec *= 1.0 - 0.20 * np.exp(-((wavelengths - 1.45) ** 2) / (2 * 0.03 ** 2))
    spec *= 1.0 - 0.15 * np.exp(-((wavelengths - 1.75) ** 2) / (2 * 0.03 ** 2))
    spec *= 1.0 - 0.35 * np.exp(-((wavelengths - 1.94) ** 2) / (2 * 0.04 ** 2))
    spec *= 1.0 - 0.25 * np.exp(-((wavelengths - 2.21) ** 2) / (2 * 0.03 ** 2))
    # Broad 3 um
    spec *= 1.0 - 0.20 * np.exp(-((wavelengths - 2.90) ** 2) / (2 * 0.15 ** 2))

    return np.clip(spec, 0.01, 1.0)


def _generate_polyhydrated_sulfate_spectrum(wavelengths: np.ndarray) -> np.ndarray:
    """
    Generate synthetic polyhydrated sulfate spectrum.

    Key features:
      - Broad 1.9-2.1 um absorption complex
      - 2.4 um absorption
      - Moderate baseline (~0.35)
    """
    spec = np.ones_like(wavelengths) * 0.40

    spec *= 1.0 - 0.10 * np.exp(-((wavelengths - 1.45) ** 2) / (2 * 0.04 ** 2))
    spec *= 1.0 - 0.30 * np.exp(-((wavelengths - 1.95) ** 2) / (2 * 0.08 ** 2))
    spec *= 1.0 - 0.20 * np.exp(-((wavelengths - 2.13) ** 2) / (2 * 0.04 ** 2))
    spec *= 1.0 - 0.15 * np.exp(-((wavelengths - 2.40) ** 2) / (2 * 0.05 ** 2))

    return np.clip(spec, 0.01, 1.0)


def _generate_basalt_spectrum(wavelengths: np.ndarray) -> np.ndarray:
    """
    Generate synthetic basalt spectrum.

    Key features:
      - Broad 1.0 um pyroxene absorption
      - Broad 2.0 um pyroxene absorption
      - Low reflectance baseline (~0.15)
      - Relatively featureless in 2.1-2.5 um
    """
    spec = np.ones_like(wavelengths) * 0.18

    # Red slope
    spec *= 0.8 + 0.2 * np.clip((wavelengths - 1.0) / 1.5, 0, 1)
    # 1.0 um pyroxene
    spec *= 1.0 - 0.12 * np.exp(-((wavelengths - 1.05) ** 2) / (2 * 0.12 ** 2))
    # 2.0 um pyroxene
    spec *= 1.0 - 0.10 * np.exp(-((wavelengths - 2.05) ** 2) / (2 * 0.15 ** 2))

    return np.clip(spec, 0.01, 1.0)


def get_endmembers(wavelengths: np.ndarray) -> dict:
    """
    Get all reference endmember spectra resampled to the given wavelength grid.

    Args:
        wavelengths: CRISM wavelength array (um)

    Returns:
        Dict of {name: spectrum_array}
    """
    return {
        "water_ice": _generate_water_ice_spectrum(wavelengths),
        "gypsum": _generate_gypsum_spectrum(wavelengths),
        "polyhydrated_sulfate": _generate_polyhydrated_sulfate_spectrum(wavelengths),
        "basalt": _generate_basalt_spectrum(wavelengths),
    }
