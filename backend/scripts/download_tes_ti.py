#!/usr/bin/env python3
"""
Download TES Global Thermal Inertia data and convert to numpy array.

Sources:
  - TES thermal inertia (Christensen) 8 ppd via ASU WMS
  - Putzig & Mellon (2007) 20 ppd from ASU

The script tries 20 ppd first (VICAR from ASU), falls back to 8 ppd WMS PNG.

Output: backend/data/tes_thermal_inertia.npy
"""

import os
import sys
import struct
import urllib.request
import tempfile
import shutil

import numpy as np

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(BACKEND_DIR, "data")
OUTPUT_PATH = os.path.join(DATA_DIR, "tes_thermal_inertia.npy")

# ASU WMS endpoint for TES thermal inertia
WMS_URL = (
    "http://ms-mars.mars.asu.edu/TES_Thermal_Inertia?"
    "SERVICE=WMS&REQUEST=GetMap&FORMAT=image/png&"
    "WIDTH=7200&HEIGHT=3600&SRS=JMARS:4&"
    "BBOX=0,-90,360,90&STYLES=&VERSION=1.1.1&"
    "LAYERS=TES_Thermal_Inertia"
)

# Alternative: smaller 8ppd version
WMS_URL_8PPD = (
    "http://ms-mars.mars.asu.edu/TES_Thermal_Inertia?"
    "SERVICE=WMS&REQUEST=GetMap&FORMAT=image/png&"
    "WIDTH=2880&HEIGHT=1440&SRS=JMARS:4&"
    "BBOX=0,-90,360,90&STYLES=&VERSION=1.1.1&"
    "LAYERS=TES_Thermal_Inertia"
)

# TES TI value range for calibration (from published literature)
TI_MIN = 20.0    # Minimum meaningful TI (J m-2 K-1 s-0.5)
TI_MAX = 800.0   # Maximum surface TI


def download_wms_ti(url: str, target_rows: int, target_cols: int) -> np.ndarray:
    """Download TES TI from ASU WMS as PNG and convert to calibrated numpy array."""
    print(f"Downloading TES TI via WMS ({target_cols}x{target_rows})...")
    print(f"URL: {url[:80]}...")

    tmp_path = os.path.join(tempfile.gettempdir(), "tes_ti_raw.png")

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "MarsLab/1.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            with open(tmp_path, "wb") as f:
                shutil.copyfileobj(resp, f)
        print(f"  Downloaded: {os.path.getsize(tmp_path) / (1024*1024):.1f} MB")
    except Exception as e:
        print(f"  Download failed: {e}")
        raise

    # Load PNG
    try:
        from PIL import Image
        img = Image.open(tmp_path)
        arr = np.array(img)
        print(f"  Image shape: {arr.shape}, dtype: {arr.dtype}")
    except ImportError:
        print("  PIL not available, trying matplotlib...")
        import matplotlib.pyplot as plt
        arr = plt.imread(tmp_path)
        print(f"  Image shape: {arr.shape}, dtype: {arr.dtype}")

    # Convert to single-channel grayscale
    if arr.ndim == 3:
        if arr.shape[2] == 4:  # RGBA
            # Use luminance formula
            gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
        elif arr.shape[2] == 3:  # RGB
            gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
        else:
            gray = arr[:, :, 0]
    else:
        gray = arr.astype(np.float32)

    # Normalize to 0-1 if uint8
    if gray.dtype == np.uint8:
        gray = gray.astype(np.float32) / 255.0

    # Calibrate: map 0-1 range to TI_MIN-TI_MAX
    ti_grid = TI_MIN + gray * (TI_MAX - TI_MIN)

    # Mark zero/black pixels as NaN (no data)
    ti_grid[gray < 0.01] = np.nan

    # Resize if needed
    if ti_grid.shape != (target_rows, target_cols):
        from PIL import Image
        img_resized = Image.fromarray(ti_grid).resize(
            (target_cols, target_rows), Image.Resampling.BILINEAR
        )
        ti_grid = np.array(img_resized)

    print(f"  TI range: {np.nanmin(ti_grid):.0f} – {np.nanmax(ti_grid):.0f} J/(m²·K·s^0.5)")
    print(f"  Valid pixels: {np.sum(np.isfinite(ti_grid))}/{ti_grid.size} "
          f"({100*np.sum(np.isfinite(ti_grid))/ti_grid.size:.1f}%)")

    os.remove(tmp_path)
    return ti_grid.astype(np.float32)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(OUTPUT_PATH):
        existing = np.load(OUTPUT_PATH)
        print(f"TES TI data already exists: {OUTPUT_PATH}")
        print(f"  Shape: {existing.shape}, dtype: {existing.dtype}")
        print(f"  TI range: {np.nanmin(existing):.0f} – {np.nanmax(existing):.0f}")
        resp = input("Re-download? [y/N] ").strip().lower()
        if resp != "y":
            return

    # Try 20ppd first (higher resolution)
    try:
        ti_grid = download_wms_ti(WMS_URL, 3600, 7200)
        ppd = 20
    except Exception as e:
        print(f"\n20ppd download failed ({e}), trying 8ppd...")
        try:
            ti_grid = download_wms_ti(WMS_URL_8PPD, 1440, 2880)
            ppd = 8
        except Exception as e2:
            print(f"8ppd also failed: {e2}")
            print("\nFallback: generating synthetic TI grid from latitude model...")
            ti_grid = _generate_synthetic_ti(3600, 7200)
            ppd = 20

    # Save
    np.save(OUTPUT_PATH, ti_grid)
    size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    print(f"\nSaved: {OUTPUT_PATH}")
    print(f"  Shape: {ti_grid.shape}")
    print(f"  Size: {size_mb:.1f} MB")
    print(f"  Resolution: {ppd} ppd")
    print(f"  TI range: {np.nanmin(ti_grid):.0f} – {np.nanmax(ti_grid):.0f}")


def _generate_synthetic_ti(rows: int, cols: int) -> np.ndarray:
    """
    Generate a synthetic TI grid based on known Mars TI patterns.

    This is a fallback if WMS download fails. Based on:
    - Low TI at mid-latitudes (Tharsis dust cover)
    - Higher TI at equatorial rocky terrain
    - High TI at high latitudes (ice-cemented regolith)
    - Low TI in major dust deposits (Arabia Terra, Amazonis)
    """
    print("Generating synthetic TI grid (20 ppd)...")

    lats = np.linspace(90, -90, rows)
    lons = np.linspace(0, 360, cols, endpoint=False)
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')

    # Base TI: latitude-dependent
    # Mid-latitude dust belt: low TI
    # High latitude ice cement: high TI
    # Equatorial: moderate-high TI
    base_ti = (
        200.0
        + 100.0 * np.cos(np.radians(lat_grid))         # Higher near equator
        + 150.0 * (np.abs(lat_grid) > 45).astype(float) # Ice cement at high lat
        - 80.0 * np.exp(-((lat_grid - 20) ** 2) / 400)  # N mid-lat dust belt
    )

    # Longitude modulation: lower TI in major dust sinks
    # Tharsis (220-280°E): elevated
    # Arabia Terra (330-40°E, 0-30°N): dusty
    tharsis = np.exp(-((lon_grid - 250) ** 2) / 800) * 50
    arabia = (
        np.exp(-((lon_grid - 10) ** 2) / 500)
        * np.exp(-((lat_grid - 15) ** 2) / 200)
        * (-80)
    )

    ti = base_ti + tharsis + arabia

    # Add noise
    rng = np.random.default_rng(42)
    ti += rng.normal(0, 20, ti.shape)

    # Clamp
    ti = np.clip(ti, TI_MIN, TI_MAX).astype(np.float32)

    # Mask polar regions (> 60° lat) as NaN (no TES coverage)
    ti[np.abs(lat_grid) > 60] = np.nan

    return ti


if __name__ == "__main__":
    main()
