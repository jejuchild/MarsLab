"""
CRISM TRR3 — Memory-mapped cube loader for spectrum + RGB endpoints.

Uses np.memmap to avoid loading full 546 MB cubes into RAM.
Reuses parse_pds3_label and resolve_trr_files from mineral_cnn.data_loader.
"""

import numpy as np
from functools import lru_cache

from api.mineral_cnn.data_loader import parse_pds3_label, resolve_trr_files
from api.mineral_cnn.constants import BANDS as IR_WAVELENGTHS, IR_START, FILL_VALUE


@lru_cache(maxsize=4)
def get_cube_memmap(obs_id: str):
    """
    Return a read-only np.memmap of shape (rows, bands, cols) as stored
    in the PDS3 Line-Interleaved Layout (LIL) file, plus metadata.

    Returns: (memmap, rows, cols, bands)
    """
    files = resolve_trr_files(obs_id)
    meta = parse_pds3_label(files["trr_lbl"])

    rows = int(meta["LINES"])
    cols = int(meta["LINE_SAMPLES"])
    bands = int(meta["BANDS"])

    mm = np.memmap(files["trr_img"], dtype=np.float32, mode="r",
                   shape=(rows, bands, cols))
    return mm, rows, cols, bands


def get_shape(obs_id: str) -> dict:
    """Return {rows, cols, bands} for a TRR3 observation."""
    _, rows, cols, bands = get_cube_memmap(obs_id)
    return {"rows": rows, "cols": cols, "bands": bands}


SUPPORTED_BANDS = 438


class UnsupportedCubeError(ValueError):
    """Cube whose band set has no wavelength table here (e.g. VNIR/S-detector
    MSV 90-band or 107-band cubes, whose CDR WA tables are not bundled)."""


def require_supported_bands(bands: int) -> None:
    if bands != SUPPORTED_BANDS:
        raise UnsupportedCubeError(
            f"This observation has {bands} bands (VNIR / multispectral survey cube). "
            f"Spectra and band composites are only available for {SUPPORTED_BANDS}-band "
            "IR (TRR3 L-detector) cubes; the wavelength table for this band set is not available."
        )


def get_wavelengths() -> np.ndarray:
    """
    Build full 438-band wavelength array.
    Bands 0-87: VNIR (linearly spaced 0.362 – 1.014 µm)
    Bands 88-437: IR from mineral_cnn constants (1.021 – 3.477 µm)
    """
    vnir_count = IR_START  # 88
    vnir = np.linspace(0.362, 1.014, vnir_count)
    return np.concatenate([vnir, IR_WAVELENGTHS])


def get_pixel_spectrum(obs_id: str, line: int, sample: int):
    """
    Extract spectrum for a single pixel.
    Returns (wavelengths, values) arrays.
    """
    mm, rows, cols, bands = get_cube_memmap(obs_id)
    require_supported_bands(bands)

    line = max(0, min(rows - 1, line))
    sample = max(0, min(cols - 1, sample))

    # memmap shape: (rows, bands, cols) — LIL layout
    values = np.array(mm[line, :, sample], dtype=np.float32)

    wavelengths = get_wavelengths()
    if len(wavelengths) != bands:
        # Fallback: just use IR bands
        wavelengths = IR_WAVELENGTHS
        values = values[IR_START:IR_START + len(IR_WAVELENGTHS)]

    return wavelengths, values


