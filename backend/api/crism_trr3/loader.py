"""
CRISM TRR3 — Memory-mapped cube loader for spectrum + RGB endpoints.

Uses np.memmap to avoid loading full 546 MB cubes into RAM.
Reuses parse_pds3_label and resolve_trr_files from mineral_cnn.data_loader.
"""

import re
from functools import lru_cache
from pathlib import Path

import numpy as np

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

# CRISM wavelength CDRs (center wavelength in nm of every detector pixel),
# named by each cube label's MRO:WAVELENGTH_FILE_NAME.
WA_DIR = Path(__file__).resolve().parents[2] / "crism_cdr_wa"
WA_URL = "https://pds-geosciences.wustl.edu/mro/mro-m-crism-2-edr-v1/mrocr_0001/cdr/wa/"
_WA_NAME = re.compile(r"^CDR[0-9A-Z_]+$")


class UnsupportedCubeError(ValueError):
    """Cube whose wavelengths cannot be determined (no WA table, or a
    non-spectral product such as a 3-band browse cube)."""


def require_supported_bands(bands: int) -> None:
    """Kept for callers that only handle the 438-band IR layout."""
    if bands != SUPPORTED_BANDS:
        raise UnsupportedCubeError(
            f"This observation has {bands} bands; this operation needs a "
            f"{SUPPORTED_BANDS}-band IR (TRR3 L-detector) cube."
        )


def get_wavelengths() -> np.ndarray:
    """
    Wavelength labels of the mineral-CNN convention (NOT the physical band
    wavelengths of a TRR3 cube; use get_cube_wavelengths for those).
    Bands 0-87: VNIR (linearly spaced 0.362 – 1.014 µm)
    Bands 88-437: IR from mineral_cnn constants (1.021 – 3.477 µm)
    """
    vnir_count = IR_START  # 88
    vnir = np.linspace(0.362, 1.014, vnir_count)
    return np.concatenate([vnir, IR_WAVELENGTHS])


def _wa_file(name: str) -> Path:
    """Local copy of a WA CDR image, downloaded from PDS on first use."""
    stem = name.rsplit(".", 1)[0].upper()
    if not _WA_NAME.match(stem):
        raise UnsupportedCubeError(f"Unexpected wavelength file name: {name}")
    path = WA_DIR / f"{stem.lower()}.img"
    if not path.exists():
        import urllib.request
        WA_DIR.mkdir(parents=True, exist_ok=True)
        try:
            with urllib.request.urlopen(WA_URL + path.name, timeout=30) as r:
                data = r.read()
        except Exception as exc:
            raise UnsupportedCubeError(
                f"Wavelength table {stem} is not available locally and could not be downloaded."
            ) from exc
        import os
        import threading
        tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.part")
        tmp.write_bytes(data)
        tmp.replace(path)  # atomic; concurrent first requests each write their own temp
    return path


@lru_cache(maxsize=16)
def _wa_table(name: str, bands: int) -> np.ndarray:
    """(bands, samples) center wavelengths in µm; NaN where the CDR has fill."""
    raw = np.fromfile(_wa_file(name), dtype="<f4")
    samples = raw.size // bands
    # Image is followed by a small row-number table; samples is a power of 2 fraction of 640.
    for s in (640, 320, 128, 64, 32):
        if s <= samples:
            samples = s
            break
    if samples < 1 or raw.size < bands * samples:
        raise UnsupportedCubeError(f"Wavelength table {name} does not match a {bands}-band cube.")
    table = raw[: bands * samples].reshape(bands, samples).astype(np.float64) / 1000.0
    table[(table <= 0.2) | (table >= 5.0)] = np.nan
    return table


def get_cube_wavelengths(obs_id: str, sample: int | None = None) -> np.ndarray:
    """
    Wavelengths (µm) of every band of this cube, in the cube's band order,
    from the CRISM WA CDR named in its label: at the given detector column
    (spectral smile) or the median over columns. NaN marks bands the CDR has
    no wavelength for (unusable detector rows).

    Note: L-detector (438-band) cubes are stored long→short wavelength
    (band 0 ≈ 3.94 µm, band 437 ≈ 1.00 µm).
    """
    _, _, cols, bands = get_cube_memmap(obs_id)
    meta = parse_pds3_label(resolve_trr_files(obs_id)["trr_lbl"])
    name = meta.get("MRO:WAVELENGTH_FILE_NAME")
    if not name or bands < 10:
        raise UnsupportedCubeError(
            f"This observation has {bands} bands and no wavelength table; it is not a spectral cube."
        )
    table = _wa_table(name, bands)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows → NaN
        wl = np.nanmedian(table, axis=1)
    if np.isfinite(wl).sum() < bands // 2:
        raise UnsupportedCubeError(f"Wavelength table {name} has too few valid values.")
    if sample is not None and table.shape[1] == cols:
        col = table[:, max(0, min(cols - 1, sample))]
        # Per-column values where valid, the column median elsewhere.
        wl = np.where(np.isfinite(col), col, wl)
    return wl


def get_pixel_spectrum(obs_id: str, line: int, sample: int):
    """
    Extract spectrum for a single pixel.
    Returns (wavelengths, values) arrays.
    """
    mm, rows, cols, bands = get_cube_memmap(obs_id)

    line = max(0, min(rows - 1, line))
    sample = max(0, min(cols - 1, sample))
    wavelengths = get_cube_wavelengths(obs_id, sample)

    # memmap shape: (rows, bands, cols) — LIL layout
    values = np.array(mm[line, :, sample], dtype=np.float32)

    # Ascending wavelength order for plotting and band lookups; drop bands
    # without a wavelength.
    keep = np.flatnonzero(np.isfinite(wavelengths))
    order = keep[np.argsort(wavelengths[keep], kind="stable")]
    return wavelengths[order], values[order]


