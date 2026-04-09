#!/usr/bin/env python3
"""Build MarsOrtho patch dataset from existing ortho PNGs.

For each sol with both ortho_meta.json and ortho.png:
1. Load Mastcam-Z ortho (HR, ~6.25cm/px)
2. Extract HiRISE patch at the same lon/lat (LR, 25cm/px)
3. Crop into aligned LR-HR patch pairs
4. Save as PNG for Kaggle Dataset upload
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image

# Paths
BASE = Path("/disk1/cspark/mastcam")
OUTPUT_DIR = BASE / "coregister_data" / "output"
HIRISE_JP2 = BASE / "coregister_data" / "pds_cache" / "hirise" / "ESP_065431_1985_RED.JP2"
DATASET_DIR = BASE / "remote_gpu" / "dataset" / "marsortho-benchmark"

# Patch settings
HR_PATCH = 256
LR_PATCH = 64  # x4 SR
SCALE = 4
MIN_VALID_FRAC = 0.3  # Skip patches with too many transparent pixels
MAX_PATCHES_PER_SOL = 20  # Cap to keep dataset size reasonable

# HiRISE CRS parameters (Equirectangular MARS)
HIR_R = 3394839.8133163
HIR_LAT_REF = 15.0


def lonlat_to_hirise_meters(lon, lat):
    """Convert lon/lat (degrees) to HiRISE projected coordinates (meters)."""
    proj_x = HIR_R * np.cos(np.radians(HIR_LAT_REF)) * (lon - 180.0) * np.pi / 180.0
    proj_y = HIR_R * lat * np.pi / 180.0
    return proj_x, proj_y


def extract_patches_from_sol(sol_num: int, hir_ds, train_dir: Path, test_dir: Path,
                              is_test: bool = False) -> int:
    """Extract LR-HR patch pairs for a single sol."""
    sol_str = f"sol{sol_num:05d}"
    ortho_png = OUTPUT_DIR / f"{sol_str}_ortho.png"
    ortho_meta = OUTPUT_DIR / f"{sol_str}_ortho_meta.json"

    if not ortho_png.exists() or not ortho_meta.exists():
        return 0

    # Load HR ortho (Mastcam-Z, RGBA)
    hr_img = np.array(Image.open(ortho_png).convert("RGBA"))
    h, w = hr_img.shape[:2]
    if h < HR_PATCH or w < HR_PATCH:
        return 0

    # Load metadata for grid info
    with open(ortho_meta) as f:
        meta = json.load(f)

    out_dir = test_dir if is_test else train_dir
    n_saved = 0

    # Need lon/lat info to map to HiRISE
    combined_npz = OUTPUT_DIR / sol_str / "combined_lonlat.npz"
    if not combined_npz.exists():
        return 0

    data = np.load(str(combined_npz))
    lon = data["lon"]
    lat = data["lat"]

    if len(lon) == 0:
        return 0

    lon_min, lon_max = float(lon.min()), float(lon.max())
    lat_min, lat_max = float(lat.min()), float(lat.max())

    # HR ortho geographic bounds (assume linear mapping over the ortho image)
    # Sample patches by sliding window over valid (alpha > 0) regions
    alpha = hr_img[:, :, 3]

    # Compute lon/lat for each pixel in the ortho
    # ortho.png is laid out: top=lat_max, bottom=lat_min, left=lon_min, right=lon_max
    px_lons = np.linspace(lon_min, lon_max, w)
    px_lats = np.linspace(lat_max, lat_min, h)

    stride = HR_PATCH // 2
    for y0 in range(0, h - HR_PATCH + 1, stride):
        for x0 in range(0, w - HR_PATCH + 1, stride):
            if n_saved >= MAX_PATCHES_PER_SOL:
                break

            patch_alpha = alpha[y0:y0+HR_PATCH, x0:x0+HR_PATCH]
            valid_frac = (patch_alpha > 0).mean()
            if valid_frac < MIN_VALID_FRAC:
                continue

            # Get HR RGB patch
            hr_rgb = hr_img[y0:y0+HR_PATCH, x0:x0+HR_PATCH, :3]

            # Get geographic center
            center_lon = px_lons[x0 + HR_PATCH // 2]
            center_lat = px_lats[y0 + HR_PATCH // 2]

            # Convert lon/lat → projected coords → HiRISE pixel
            try:
                proj_x, proj_y = lonlat_to_hirise_meters(center_lon, center_lat)
                hir_row, hir_col = hir_ds.index(proj_x, proj_y)
            except Exception:
                continue

            # Extract LR patch from HiRISE
            half = LR_PATCH // 2
            hr_row_min = int(hir_row - half)
            hr_col_min = int(hir_col - half)

            if (hr_row_min < 0 or hr_col_min < 0 or
                hr_row_min + LR_PATCH > hir_ds.height or
                hr_col_min + LR_PATCH > hir_ds.width):
                continue

            window = rasterio.windows.Window(hr_col_min, hr_row_min, LR_PATCH, LR_PATCH)
            try:
                lr_data = hir_ds.read(1, window=window)
            except Exception:
                continue

            if lr_data.shape != (LR_PATCH, LR_PATCH):
                continue

            # Normalize LR to uint8
            valid_lr = lr_data[lr_data > 0]
            if len(valid_lr) < 100:
                continue
            vmin, vmax = np.percentile(valid_lr, [2, 98])
            lr_uint8 = np.clip((lr_data - vmin) / max(vmax - vmin, 1) * 255, 0, 255).astype(np.uint8)
            lr_rgb = np.stack([lr_uint8] * 3, axis=-1)

            # Save patches
            patch_name = f"{sol_str}_y{y0:04d}_x{x0:04d}"
            (out_dir / "lr").mkdir(parents=True, exist_ok=True)
            (out_dir / "hr").mkdir(parents=True, exist_ok=True)

            Image.fromarray(lr_rgb).save(out_dir / "lr" / f"{patch_name}.png")
            Image.fromarray(hr_rgb).save(out_dir / "hr" / f"{patch_name}.png")
            n_saved += 1

        if n_saved >= MAX_PATCHES_PER_SOL:
            break

    return n_saved


def main():
    print("=== MarsOrtho Dataset Builder ===\n")

    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)
    DATASET_DIR.mkdir(parents=True)

    train_dir = DATASET_DIR / "train"
    test_dir = DATASET_DIR / "test"

    # Find all sols with ortho data
    ortho_files = sorted(OUTPUT_DIR.glob("sol?????_ortho.png"))
    sol_nums = []
    for f in ortho_files:
        try:
            sol_nums.append(int(f.stem.replace("sol", "").replace("_ortho", "")))
        except ValueError:
            pass

    print(f"Found {len(sol_nums)} sols with ortho data")

    if not HIRISE_JP2.exists():
        print(f"ERROR: HiRISE JP2 not found: {HIRISE_JP2}")
        return

    # Train/test split: 85/15
    n_test = max(1, len(sol_nums) // 7)
    test_sols = set(sol_nums[-n_test:])
    train_sols = set(sol_nums) - test_sols

    print(f"Train sols: {len(train_sols)}, Test sols: {len(test_sols)}")

    # Open HiRISE once
    print(f"\nOpening HiRISE: {HIRISE_JP2.name}")
    with rasterio.open(HIRISE_JP2) as hir_ds:
        print(f"  Size: {hir_ds.width}x{hir_ds.height}")
        print(f"  CRS: {hir_ds.crs}")
        print(f"  Bounds: {hir_ds.bounds}")
        print()

        n_train_total = 0
        n_test_total = 0

        for sol in sol_nums:
            is_test = sol in test_sols
            n = extract_patches_from_sol(sol, hir_ds, train_dir, test_dir, is_test)
            split = "test" if is_test else "train"
            print(f"  sol{sol:05d} ({split}): {n} patches")
            if is_test:
                n_test_total += n
            else:
                n_train_total += n

    print(f"\n{'='*50}")
    print(f"Train: {n_train_total} patches")
    print(f"Test:  {n_test_total} patches")
    print(f"Output: {DATASET_DIR}")
    print(f"{'='*50}")

    # Write dataset metadata for Kaggle
    dataset_meta = {
        "title": "MarsOrtho Benchmark",
        "id": "carsonparksnu/marsortho-benchmark",
        "licenses": [{"name": "CC0-1.0"}],
        "keywords": ["mars", "super-resolution", "remote-sensing", "planetary-science"],
        "subtitle": "Cross-sensor SR benchmark from Mastcam-Z + HiRISE",
        "description": (
            f"MarsOrtho is the first cross-sensor super-resolution benchmark for Mars, "
            f"derived from Perseverance Mastcam-Z (HR, ~6.25cm/px) and MRO HiRISE "
            f"(LR, ~25cm/px) co-registered via SPICE kernels.\n\n"
            f"- Train: {n_train_total} patches from {len(train_sols)} sols\n"
            f"- Test: {n_test_total} patches from {len(test_sols)} sols\n"
            f"- Patch sizes: HR 256x256, LR 64x64 (x4 SR)\n"
            f"- Coverage: Jezero Crater, Mars\n"
        ),
    }
    meta_path = DATASET_DIR / "dataset-metadata.json"
    with open(meta_path, "w") as f:
        json.dump(dataset_meta, f, indent=2)
    print(f"\nMetadata: {meta_path}")
    print(f"\nUpload to Kaggle:")
    print(f"  kaggle datasets create -p {DATASET_DIR}")


if __name__ == "__main__":
    main()
