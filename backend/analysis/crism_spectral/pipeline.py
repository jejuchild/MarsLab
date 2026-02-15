"""
Full CRISM spectral analysis pipeline.

Load TRR3 -> continuum removal -> band parameters -> SAM -> classification
"""
import logging
import time
import numpy as np
from typing import Optional

from .models import CrismSpectralResult
from .endmembers import get_endmembers
from .classifier import classify_pixel
from .band_parameters import compute_all_band_params

logger = logging.getLogger(__name__)

# Maximum pixels to process (for memory/time safety)
MAX_PIXELS = 200_000
FILL_VALUE = 65535.0


def run_spectral_analysis(
    obs_id: str,
    max_sam_angle: float = 0.3,
    subsample: int = 1,
) -> CrismSpectralResult:
    """
    Run full spectral analysis on a CRISM TRR3 observation.

    Args:
        obs_id: CRISM observation ID (e.g., 'frt0000c702')
        max_sam_angle: Maximum SAM angle for valid classification (radians)
        subsample: Spatial subsample factor (1=full res, 2=every other pixel, etc.)

    Returns:
        CrismSpectralResult with mineral map, band parameters, confidence
    """
    t0 = time.time()

    try:
        # Import CRISM data loader
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'api'))
        from crism_trr3.loader import get_cube_memmap, get_wavelengths
    except ImportError as e:
        return CrismSpectralResult(
            obs_id=obs_id,
            error=f"Cannot import CRISM loader: {e}",
        )

    # Load data
    try:
        memmap, rows, cols, n_bands = get_cube_memmap(obs_id)
        wavelengths = get_wavelengths()
    except Exception as e:
        return CrismSpectralResult(
            obs_id=obs_id,
            error=f"Failed to load TRR3 cube: {e}",
        )

    logger.info(f"[CRISM Spectral] Loaded {obs_id}: {rows}x{cols}, {n_bands} bands")

    # Get endmember spectra at CRISM wavelengths
    endmembers = get_endmembers(wavelengths)

    # Process pixels
    class_counts = {}
    ice_confidences = []
    ice_bd1500s = []
    ice_bd1900s = []
    ice_bd2100s = []
    ice_bd2200s = []
    all_confidences = []
    n_valid = 0
    n_classified = 0

    # Subsample for performance
    row_indices = range(0, rows, subsample)
    col_indices = range(0, cols, subsample)
    total_pixels = len(row_indices) * len(col_indices)

    if total_pixels > MAX_PIXELS:
        # Auto-increase subsample
        factor = int(np.ceil(np.sqrt(total_pixels / MAX_PIXELS)))
        subsample = max(subsample, factor)
        row_indices = range(0, rows, subsample)
        col_indices = range(0, cols, subsample)
        logger.info(
            f"[CRISM Spectral] Auto-subsampled to {subsample}x "
            f"({len(row_indices) * len(col_indices)} pixels)"
        )

    processed = 0
    for r in row_indices:
        for c in col_indices:
            # memmap shape is (rows, bands, cols) — Line-Interleaved Layout
            spectrum = np.array(memmap[r, :, c], dtype=np.float64)

            # Check for fill values
            n_valid_bands = np.sum(
                np.isfinite(spectrum) & (spectrum < FILL_VALUE) & (spectrum > 0)
            )
            if n_valid_bands < 100:
                continue

            n_valid += 1

            # Classify
            result = classify_pixel(spectrum, wavelengths, endmembers, max_sam_angle)

            if result.mineral_class != "unclassified":
                n_classified += 1
                class_counts[result.mineral_class] = class_counts.get(result.mineral_class, 0) + 1
                all_confidences.append(result.sam_confidence)

                if result.mineral_class == "water_ice":
                    ice_confidences.append(result.sam_confidence)
                    if result.band_params.BD1500 is not None:
                        ice_bd1500s.append(result.band_params.BD1500)
                    if result.band_params.BD1900 is not None:
                        ice_bd1900s.append(result.band_params.BD1900)
                    if result.band_params.BD2100 is not None:
                        ice_bd2100s.append(result.band_params.BD2100)
                    if result.band_params.BD2200 is not None:
                        ice_bd2200s.append(result.band_params.BD2200)

            processed += 1
            if processed % 5000 == 0:
                logger.info(f"[CRISM Spectral] Processed {processed}/{total_pixels} pixels")

    elapsed = time.time() - t0

    # Compute fractions
    class_fractions = {k: v / max(n_classified, 1) for k, v in class_counts.items()}

    # Dominant mineral
    dominant = max(class_counts, key=class_counts.get) if class_counts else None
    dominant_frac = class_fractions.get(dominant, 0) if dominant else 0

    # Ice metrics
    ice_pixels = class_counts.get("water_ice", 0)
    ice_frac = ice_pixels / max(n_valid, 1)
    ice_mean_conf = float(np.mean(ice_confidences)) if ice_confidences else 0.0

    # Atmospheric uncertainty note
    atm_notes = _assess_atmospheric_uncertainty(
        wavelengths, class_counts, n_valid, ice_frac
    )

    return CrismSpectralResult(
        obs_id=obs_id,
        rows=rows,
        cols=cols,
        n_valid_pixels=n_valid,
        n_classified_pixels=n_classified,
        class_counts=class_counts,
        class_fractions={k: round(v, 4) for k, v in class_fractions.items()},
        dominant_mineral=dominant,
        dominant_fraction=round(dominant_frac, 4),
        water_ice_fraction=round(ice_frac, 4),
        water_ice_pixels=ice_pixels,
        water_ice_mean_confidence=round(ice_mean_conf, 4),
        mean_BD1500=round(float(np.mean(ice_bd1500s)), 4) if ice_bd1500s else None,
        mean_BD1900=round(float(np.mean(ice_bd1900s)), 4) if ice_bd1900s else None,
        mean_BD2100=round(float(np.mean(ice_bd2100s)), 4) if ice_bd2100s else None,
        mean_BD2200=round(float(np.mean(ice_bd2200s)), 4) if ice_bd2200s else None,
        mean_sam_confidence=round(float(np.mean(all_confidences)), 4) if all_confidences else 0.0,
        atmospheric_uncertainty_notes=atm_notes,
        success=True,
        elapsed_seconds=round(elapsed, 2),
    )


def _assess_atmospheric_uncertainty(
    wavelengths: np.ndarray,
    class_counts: dict,
    n_valid: int,
    ice_fraction: float,
) -> str:
    """Generate atmospheric uncertainty discussion."""
    notes = []

    # CO2 atmospheric bands at ~2.0 um and ~2.7 um can contaminate mineral signatures
    notes.append(
        "CRISM spectra are affected by Mars atmospheric CO2 absorption bands "
        "near 2.0 um and 2.7 um. JCAT atmospheric correction mitigates this "
        "but residual artifacts may affect BD1900 and BD2100 values."
    )

    if ice_fraction > 0.5:
        notes.append(
            "WARNING: >50% water ice classification may indicate atmospheric "
            "CO2 frost contamination rather than surface ice deposits. "
            "Cross-validate with thermal inertia and season of observation."
        )

    if ice_fraction > 0 and ice_fraction < 0.01:
        notes.append(
            "Trace water ice detection (<1% of pixels). This may be below "
            "the reliable detection limit for SAM classification."
        )

    return " ".join(notes)
