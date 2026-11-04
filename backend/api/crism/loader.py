# api/crism/loader.py
import logging
import numpy as np
import spectral as sp

logger = logging.getLogger(__name__)
def load_cube(hdr_path: str) -> np.ndarray:
    img = sp.open_image(hdr_path)
    # spectral returns (lines, samples, bands), we need (bands, lines, samples)
    cube = img.load().astype(np.float32)
    cube = np.transpose(cube, (2, 0, 1))  # (bands, lines, samples)
    return cube

def read_pixel_spectrum(hdr_path: str, line: int, sample: int):
    """
    One pixel's spectrum without loading the cube: (spectrum, (bands, lines, samples)).
    Same values as load_cube(hdr)[:, line, sample] (spectral applies the same
    scale factor) but reads a few KB instead of the whole ~1 GB cube.
    Raises IndexError for an out-of-range pixel.
    """
    img = sp.open_image(hdr_path)
    n_lines, n_samples, n_bands = img.shape
    if not (0 <= line < n_lines and 0 <= sample < n_samples):
        raise IndexError((n_lines, n_samples))
    spectrum = np.asarray(img.read_pixel(line, sample), dtype=np.float32).copy()
    return spectrum, (n_bands, n_lines, n_samples)


def read_bands(hdr_path: str, band_indices: list[int]) -> list[np.ndarray]:
    """Read only the given bands, each as float32 (lines, samples)."""
    img = sp.open_image(hdr_path)
    return [np.asarray(img.read_band(int(i)), dtype=np.float32) for i in band_indices]


def load_wavelength(wv_tab_path: str) -> np.ndarray:
    wv = []
    with open(wv_tab_path) as f:
        for line in f:
            if line.strip().startswith("#") or "," not in line:
                continue
            parts = line.split(",")
            try:
                wv.append(float(parts[2]))  # wavelength column (in nanometers)
            except (ValueError, IndexError) as e:
                logger.debug(f"Skipping malformed wavelength line: {e}")
    wv = np.array(wv, dtype=np.float32)
    # Convert from nanometers to micrometers
    wv = wv / 1000.0
    return wv
