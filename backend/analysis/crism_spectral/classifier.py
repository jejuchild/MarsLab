"""
Per-pixel mineral classification combining SAM and band parameters.
"""
import numpy as np
from typing import Dict, Optional, Tuple
from .band_parameters import compute_all_band_params
from .spectral_angle_mapper import classify_pixel_sam
from .continuum_removal import remove_continuum
from .models import PixelClassification, BandParameterResult


def classify_pixel(
    spectrum: np.ndarray,
    wavelengths: np.ndarray,
    endmembers: Dict[str, np.ndarray],
    max_sam_angle: float = 0.3,
) -> PixelClassification:
    """
    Full classification pipeline for a single pixel.

    1. Continuum removal
    2. Band parameter extraction
    3. SAM classification
    4. Combined confidence
    """
    # Band parameters on raw reflectance
    bp = compute_all_band_params(spectrum, wavelengths)

    # SAM on raw reflectance (SAM is illumination-invariant)
    best_match, sam_conf, angles = classify_pixel_sam(spectrum, endmembers, max_sam_angle)

    # Band strength: max of non-None band depths
    bd_vals = [v for v in bp.values() if v is not None and v > 0]
    band_strength = max(bd_vals) if bd_vals else 0.0

    # Cross-validate: if SAM says water_ice, check BD1500 and BD2100
    if best_match == "water_ice":
        bd1500 = bp.get("BD1500")
        bd2100 = bp.get("BD2100")
        # Water ice should have BD1500 > 0.02 and BD2100 > 0.01
        if bd1500 is not None and bd1500 < 0.02 and bd2100 is not None and bd2100 < 0.01:
            # Weak band depth contradicts SAM — reduce confidence
            sam_conf *= 0.5

    return PixelClassification(
        mineral_class=best_match,
        sam_confidence=round(sam_conf, 4),
        band_params=BandParameterResult(**bp),
        band_strength=round(band_strength, 4),
    )
