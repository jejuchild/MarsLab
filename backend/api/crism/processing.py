import numpy as np

def clean_cube(cube, vmax=1.0):
    out = cube.copy()
    out[~np.isfinite(out)] = np.nan
    out[out < 0] = np.nan
    out[out > vmax] = np.nan
    return out

def band_at(wavelength, target_um):
    return int(np.nanargmin(np.abs(wavelength - target_um)))

def stretch(img, mask, vmin=0.02, vmax=0.25):
    out = img.copy()
    out[~mask] = vmin
    out = np.clip(out, vmin, vmax)
    return (out - vmin) / (vmax - vmin)

def make_rgb(cube, wavelength, r_um, g_um, b_um,
             vmin=0.02, vmax=0.25):

    r_band = band_at(wavelength, r_um)
    g_band = band_at(wavelength, g_um)
    b_band = band_at(wavelength, b_um)

    mask = np.isfinite(cube[g_band])

    R = stretch(cube[r_band], mask, vmin, vmax)
    G = stretch(cube[g_band], mask, vmin, vmax)
    B = stretch(cube[b_band], mask, vmin, vmax)

    return np.dstack([R, G, B])
