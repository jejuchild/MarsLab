"""
CNN Mineral Classification — JCAT Atmospheric Correction Bridge

Wraps the external autojcat module for per-pixel atmospheric correction.
Runs correction in a thread pool with progress reporting.
"""

import sys
import os
from typing import Callable, Optional

import numpy as np

from .constants import (
    JCAT_AUTOMATION_DIR, IR_START, IR_COUNT, BANDS, FILL_VALUE,
)
from .data_loader import load_vs_adr

# Add JCAT_Automation to sys.path once at import time
if JCAT_AUTOMATION_DIR not in sys.path:
    sys.path.insert(0, JCAT_AUTOMATION_DIR)
import autojcat


def correct_cube(
    cube: np.ndarray,
    ina_rad: float,
    adr_path: str,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> np.ndarray:
    """
    Apply JCAT atmospheric correction to IR bands of a TRR3 cube.

    Args:
        cube: Full TRR3 cube (rows, cols, total_bands)
        ina_rad: Incidence angle in radians (from DDR)
        adr_path: Path to VS ADR .IMG file
        progress_callback: Optional callback(done, total) for progress reporting

    Returns: Corrected cube (rows, cols, IR_COUNT) — IR bands only
    """
    rows, cols, _ = cube.shape
    cube_ir = cube[:, :, IR_START:IR_START + IR_COUNT].copy()

    # Load ADR reference spectra
    vstrans, vsart = load_vs_adr(adr_path, IR_COUNT)

    wl = BANDS.tolist()
    vstrans_list = vstrans.tolist()
    vsart_list = vsart.tolist()

    cube_corrected = np.empty_like(cube_ir)
    total_pixels = rows * cols
    done = 0

    for r in range(rows):
        for c in range(cols):
            spec = cube_ir[r, c, :].tolist()
            if autojcat.get_median(spec) >= FILL_VALUE:
                cube_corrected[r, c, :] = cube_ir[r, c, :]
            else:
                cube_corrected[r, c, :] = autojcat.jcat_correction_pipeline(
                    intensity=spec,
                    wavelength=wl,
                    vstrans=vstrans_list,
                    vsart=vsart_list,
                    ina_rad=ina_rad,
                )
            done += 1
            if progress_callback and done % 500 == 0:
                progress_callback(done, total_pixels)

    if progress_callback:
        progress_callback(total_pixels, total_pixels)

    return cube_corrected
