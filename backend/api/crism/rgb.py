# api/crism/rgb.py
import numpy as np

# CRISM MTRDR data ignore value (from header)
IGNORE_VALUE = 65535.0

def nearest_band(wavelengths: np.ndarray, target_um: float) -> int:
    return int(np.argmin(np.abs(wavelengths - target_um)))

def normalize_band(band, vmin, vmax):
    band = np.clip(band, vmin, vmax)
    return (band - vmin) / (vmax - vmin)

def make_rgb(
    cube: np.ndarray,          # (bands, lines, samples)
    wavelengths: np.ndarray,   # (bands,) in micrometers
    r_um: float,
    g_um: float,
    b_um: float,
    vmin: float,
    vmax: float,
):
    print(f"[make_rgb] Cube shape: {cube.shape}")
    print(f"[make_rgb] Wavelengths range: {wavelengths.min():.3f} - {wavelengths.max():.3f} um")
    print(f"[make_rgb] Target wavelengths: R={r_um}, G={g_um}, B={b_um} um")

    # 1. Find band indices
    r_idx = nearest_band(wavelengths, r_um)
    g_idx = nearest_band(wavelengths, g_um)
    b_idx = nearest_band(wavelengths, b_um)
    print(f"[make_rgb] Band indices: R={r_idx} ({wavelengths[r_idx]:.3f}um), G={g_idx} ({wavelengths[g_idx]:.3f}um), B={b_idx} ({wavelengths[b_idx]:.3f}um)")

    return stretch_rgb(cube[r_idx], cube[g_idx], cube[b_idx], vmin, vmax)


def clean_band(band: np.ndarray) -> np.ndarray:
    """Copy of one band with ignore/invalid/negative values set to NaN."""
    band = np.array(band, dtype=np.float32, copy=True)
    band[band >= IGNORE_VALUE - 1] = np.nan  # 65535 is the ignore value
    band[~np.isfinite(band)] = np.nan
    band[band < 0] = np.nan
    return band


def stretch_rgb(R: np.ndarray, G: np.ndarray, B: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """
    Linear-stretch three (lines, samples) bands into an RGB image in [0, 1].
    Cleaning is per pixel, so working on three bands gives exactly the
    result of cleaning the whole cube first (as make_rgb used to).
    """
    R, G, B = clean_band(R), clean_band(G), clean_band(B)

    # Footprint mask (where data is valid)
    mask = np.isfinite(G)

    def stretch(img):
        out = img.copy()
        out[~mask] = 0  # Set masked areas to 0 (will be black)
        out = np.clip(out, vmin, vmax)
        return (out - vmin) / (vmax - vmin)

    Rn = stretch(R)
    Gn = stretch(G)
    Bn = stretch(B)

    rgb = np.dstack([Rn, Gn, Bn])
    return rgb
