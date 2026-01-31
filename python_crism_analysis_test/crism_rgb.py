#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import spectral as sp

# =========================================================
# 0. 경로 설정
# =========================================================
HDR_PATH = "../backend/crism_data/FRT0001FD76_07_IF166J_MTR3.HDR"
WV_PATH  = "../backend/crism_data/FRT0001FD76_07_WV166J_MTR3.TAB"

# =========================================================
# 1. ENVI header 기반 cube 로드
# =========================================================
# ENVI 기준 shape: (lines, samples, bands)
img = sp.open_image(HDR_PATH)
cube = img.load().astype(np.float32)

# (bands, lines, samples)
cube = cube.transpose(2, 0, 1)

print("[INFO] cube shape (bands, lines, samples):", cube.shape)

# =========================================================
# 2. CRISM reflectance 필터링
# =========================================================
def clean_crism_cube(cube, vmax=1.0):
    out = cube.copy()
    out[~np.isfinite(out)] = np.nan
    out[out < 0] = np.nan
    out[out > vmax] = np.nan
    return out

cube_clean = clean_crism_cube(cube, vmax=1.0)

# =========================================================
# 3. WV.TAB 로드
# =========================================================
def load_wavelength_tab(path):
    """
    CRISM *_WV*.TAB reader
    Returns wavelength array in microns (µm)
    """
    wv = []

    with open(path) as f:
        for line in f:
            line = line.strip()

            # skip empty / comment
            if not line or line.startswith("#"):
                continue

            # comma-separated
            parts = [p.strip() for p in line.split(",")]

            # safety check
            if len(parts) < 3:
                continue

            try:
                # 3rd column = wavelength in nm
                wv_nm = float(parts[2])
                wv.append(wv_nm / 1000.0)  # nm → µm
            except ValueError:
                continue

    return np.array(wv, dtype=np.float32)

wavelength = load_wavelength_tab(WV_PATH)

assert len(wavelength) == cube_clean.shape[0], "Band count mismatch!"

print("[INFO] wavelength range (µm):",
      wavelength.min(), "–", wavelength.max())

# =========================================================
# 4. wavelength → band index 매핑
# =========================================================
def band_at(wavelength, target_um):
    """
    Return band index closest to target wavelength (µm)
    """
    return int(np.nanargmin(np.abs(wavelength - target_um)))

# =========================================================
# 5. stretch 함수
# =========================================================
def stretch(img, mask, vmin=0.02, vmax=0.25):
    out = img.copy()
    out[~mask] = vmin
    out = np.clip(out, vmin, vmax)
    return (out - vmin) / (vmax - vmin)

# =========================================================
# 6. 사용자 입력 (파장 기반 RGB)
# =========================================================
# ---- 여기만 바꾸면 됨 ----
R_WV = 0.75   # µm
G_WV = 0.55   # µm
B_WV = 0.44   # µm
# -------------------------

R_BAND = band_at(wavelength, R_WV)
G_BAND = band_at(wavelength, G_WV)
B_BAND = band_at(wavelength, B_WV)

print("[INFO] Selected RGB bands:")
print(f"  R: {R_WV} µm -> band {R_BAND} ({wavelength[R_BAND]:.4f} µm)")
print(f"  G: {G_WV} µm -> band {G_BAND} ({wavelength[G_BAND]:.4f} µm)")
print(f"  B: {B_WV} µm -> band {B_BAND} ({wavelength[B_BAND]:.4f} µm)")

# =========================================================
# 7. footprint mask
# =========================================================
mask = np.isfinite(cube_clean[G_BAND])
print("[INFO] footprint coverage:", mask.mean())

# =========================================================
# 8. RGB 구성
# =========================================================
R = stretch(cube_clean[R_BAND], mask)
G = stretch(cube_clean[G_BAND], mask)
B = stretch(cube_clean[B_BAND], mask)

rgb = np.dstack([R, G, B])

# =========================================================
# 9. 시각화
# =========================================================
plt.figure(figsize=(7, 7))
plt.imshow(rgb)
plt.axis("off")
plt.title(
    "CRISM RGB (wavelength-based)\n"
    f"R={R_WV}µm, G={G_WV}µm, B={B_WV}µm"
)
plt.tight_layout()
plt.show()
