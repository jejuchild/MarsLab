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

    # 2. Clean cube - replace ignore values with NaN
    cube = cube.copy()
    cube[cube >= IGNORE_VALUE - 1] = np.nan  # 65535 is the ignore value
    cube[~np.isfinite(cube)] = np.nan
    cube[cube < 0] = np.nan

    # Extract RGB bands
    R = cube[r_idx]
    G = cube[g_idx]
    B = cube[b_idx]

    print(f"[make_rgb] R stats: min={np.nanmin(R):.4f}, max={np.nanmax(R):.4f}, mean={np.nanmean(R):.4f}")
    print(f"[make_rgb] G stats: min={np.nanmin(G):.4f}, max={np.nanmax(G):.4f}, mean={np.nanmean(G):.4f}")
    print(f"[make_rgb] B stats: min={np.nanmin(B):.4f}, max={np.nanmax(B):.4f}, mean={np.nanmean(B):.4f}")
    print(f"[make_rgb] NaN ratios: R={np.isnan(R).mean():.2%}, G={np.isnan(G).mean():.2%}, B={np.isnan(B).mean():.2%}")

    # 3. Create footprint mask (where data is valid)
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
    print(f"[make_rgb] Output RGB shape: {rgb.shape}, min={rgb.min():.3f}, max={rgb.max():.3f}")
    return rgb
