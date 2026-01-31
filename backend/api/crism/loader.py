# api/crism/loader.py
import numpy as np
import spectral as sp

def load_cube(hdr_path: str) -> np.ndarray:
    img = sp.open_image(hdr_path)
    # spectral returns (lines, samples, bands), we need (bands, lines, samples)
    cube = img.load().astype(np.float32)
    cube = np.transpose(cube, (2, 0, 1))  # (bands, lines, samples)
    return cube

def load_wavelength(wv_tab_path: str) -> np.ndarray:
    wv = []
    with open(wv_tab_path) as f:
        for line in f:
            if line.strip().startswith("#") or "," not in line:
                continue
            parts = line.split(",")
            try:
                wv.append(float(parts[2]))  # wavelength column (in nanometers)
            except:
                pass
    wv = np.array(wv, dtype=np.float32)
    # Convert from nanometers to micrometers
    wv = wv / 1000.0
    return wv
