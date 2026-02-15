"""
Spectral Angle Mapper (SAM) implementation.

SAM measures the spectral similarity between an observed spectrum
and reference endmember spectra using the angle between vectors:

  theta = arccos( dot(t, r) / (|t| * |r|) )

Smaller angle = more similar. Insensitive to illumination intensity.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple


def spectral_angle(target: np.ndarray, reference: np.ndarray) -> float:
    """
    Compute spectral angle between target and reference spectrum.

    theta = arccos( dot(t, r) / (|t| * |r|) )

    Args:
        target: observed spectrum (1D array)
        reference: endmember spectrum (1D array, same length)

    Returns:
        angle in radians (0 = identical, pi/2 = orthogonal)
    """
    # Mask invalid values in either spectrum
    valid = np.isfinite(target) & np.isfinite(reference) & (target > 0) & (reference > 0)

    if np.sum(valid) < 10:
        return np.pi / 2  # Maximum angle if insufficient data

    t = target[valid]
    r = reference[valid]

    dot_product = np.dot(t, r)
    norm_t = np.linalg.norm(t)
    norm_r = np.linalg.norm(r)

    if norm_t < 1e-10 or norm_r < 1e-10:
        return np.pi / 2

    cos_theta = dot_product / (norm_t * norm_r)
    # Clamp to [-1, 1] for numerical safety
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    return float(np.arccos(cos_theta))


def classify_pixel_sam(
    spectrum: np.ndarray,
    endmembers: Dict[str, np.ndarray],
    max_angle_rad: float = 0.3,
) -> Tuple[str, float, Dict[str, float]]:
    """
    Classify a single pixel using SAM against all endmembers.

    Args:
        spectrum: observed spectrum
        endmembers: dict of {name: reference_spectrum}
        max_angle_rad: maximum angle for valid classification (reject if all angles exceed this)

    Returns:
        (best_match_name, confidence, {name: angle_rad})
    """
    angles = {}
    for name, ref in endmembers.items():
        angles[name] = spectral_angle(spectrum, ref)

    if not angles:
        return "unclassified", 0.0, {}

    best_name = min(angles, key=angles.get)
    best_angle = angles[best_name]

    if best_angle > max_angle_rad:
        return "unclassified", 0.0, angles

    # Confidence: 1 at angle=0, 0 at angle=max_angle
    confidence = max(0.0, 1.0 - (best_angle / max_angle_rad))

    return best_name, float(confidence), angles
