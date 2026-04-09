#!/usr/bin/env python3
"""Build paired HiRISE → Mastcam-Z dataset for Image-to-Image Translation.

For each Mastcam ortho patch:
1. Find its lon/lat bounds
2. Extract the corresponding HiRISE region
3. Resample both to 256x256 (HiRISE upsampled by bicubic)
4. Save as paired (input_hirise, target_mastcam)

Output structure:
    dataset/marsortho-i2i/
        train/
            hirise/  (256x256x1, grayscale, bicubic upsampled)
            mastcam/ (256x256x3, RGB, target)
        test/
            hirise/
            mastcam/
"""

import json
import shutil
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image

# Paths
BASE = Path("/disk1/cspark/mastcam")
OUTPUT_DIR = BASE / "coregister_data" / "output"
HIRISE_JP2 = BASE / "coregister_data" / "pds_cache" / "hirise" / "ESP_065431_1985_RED.JP2"
DATASET_DIR = BASE / "remote_gpu" / "dataset" / "marsortho-i2i"

# Patch settings
PATCH_SIZE = 256  # Both inputs and outputs at this size
MIN_VALID_FRAC = 0.3
MAX_PATCHES_PER_SOL = 30  # More patches now

# HiRISE CRS (Equirectangular MARS)
HIR_R = 3394839.8133163
HIR_LAT_REF = 15.0


def lonlat_to_hirise_meters(lon, lat):
    """Convert lon/lat (degrees) to HiRISE projected coordinates (meters)."""
    proj_x = HIR_R * np.cos(np.radians(HIR_LAT_REF)) * (lon - 180.0) * np.pi / 180.0
    proj_y = HIR_R * lat * np.pi / 180.0
    return proj_x, proj_y


def extract_paired_patches(sol_num, hir_ds, train_dir, test_dir, is_test=False):
    """Extract HiRISE-Mastcam paired patches for one sol."""
    sol_str = f"sol{sol_num:05d}"
    ortho_png = OUTPUT_DIR / f"{sol_str}_ortho.png"
    combined_npz = OUTPUT_DIR / sol_str / "combined_lonlat.npz"

    if not ortho_png.exists() or not combined_npz.exists():
        return 0

    # Load Mastcam ortho (target)
    mastcam_img = np.array(Image.open(ortho_png).convert("RGBA"))
    h, w = mastcam_img.shape[:2]
    if h < PATCH_SIZE or w < PATCH_SIZE:
        return 0

    # Get lon/lat extent of the ortho
    data = np.load(str(combined_npz))
    lon = data["lon"]
    lat = data["lat"]
    if len(lon) == 0:
        return 0

    lon_min, lon_max = float(lon.min()), float(lon.max())
    lat_min, lat_max = float(lat.min()), float(lat.max())

    # Pixel → lon/lat mapping (ortho image is laid out top=lat_max, left=lon_min)
    px_lons = np.linspace(lon_min, lon_max, w)
    px_lats = np.linspace(lat_max, lat_min, h)

    out_dir = test_dir if is_test else train_dir
    n_saved = 0
    alpha = mastcam_img[:, :, 3]

    stride = PATCH_SIZE // 2
    for y0 in range(0, h - PATCH_SIZE + 1, stride):
        for x0 in range(0, w - PATCH_SIZE + 1, stride):
            if n_saved >= MAX_PATCHES_PER_SOL:
                break

            # Check Mastcam validity
            patch_alpha = alpha[y0:y0+PATCH_SIZE, x0:x0+PATCH_SIZE]
            valid_frac = (patch_alpha > 0).mean()
            if valid_frac < MIN_VALID_FRAC:
                continue

            # Get Mastcam patch (target)
            mastcam_patch = mastcam_img[y0:y0+PATCH_SIZE, x0:x0+PATCH_SIZE, :3]

            # Get geographic bounds of this patch
            patch_lon_min = px_lons[x0]
            patch_lon_max = px_lons[min(x0+PATCH_SIZE-1, w-1)]
            patch_lat_max = px_lats[y0]
            patch_lat_min = px_lats[min(y0+PATCH_SIZE-1, h-1)]

            # Convert to HiRISE projected coordinates
            pmin_x, pmin_y = lonlat_to_hirise_meters(patch_lon_min, patch_lat_min)
            pmax_x, pmax_y = lonlat_to_hirise_meters(patch_lon_max, patch_lat_max)

            try:
                # Get HiRISE pixel window for this geographic bounds
                row_top, col_left = hir_ds.index(min(pmin_x, pmax_x), max(pmin_y, pmax_y))
                row_bot, col_right = hir_ds.index(max(pmin_x, pmax_x), min(pmin_y, pmax_y))
            except Exception:
                continue

            r0 = max(0, int(min(row_top, row_bot)))
            r1 = min(hir_ds.height, int(max(row_top, row_bot)))
            c0 = max(0, int(min(col_left, col_right)))
            c1 = min(hir_ds.width, int(max(col_left, col_right)))

            if r1 - r0 < 4 or c1 - c0 < 4:
                continue

            # Read HiRISE window
            window = rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0)
            try:
                hir_data = hir_ds.read(1, window=window)
            except Exception:
                continue

            if hir_data.size == 0:
                continue

            # Skip patches with too many zero pixels
            if (hir_data == 0).mean() > 0.5:
                continue

            # Normalize HiRISE to uint8
            valid_lr = hir_data[hir_data > 0]
            if len(valid_lr) < 100:
                continue
            vmin, vmax = np.percentile(valid_lr, [2, 98])
            hir_norm = np.clip((hir_data - vmin) / max(vmax - vmin, 1) * 255, 0, 255).astype(np.uint8)

            # Bicubic upsample HiRISE to 256x256 (matches Mastcam patch size)
            hir_pil = Image.fromarray(hir_norm).resize((PATCH_SIZE, PATCH_SIZE), Image.BICUBIC)
            hir_upsampled = np.array(hir_pil)

            # Save paired patches
            patch_name = f"{sol_str}_y{y0:04d}_x{x0:04d}"
            (out_dir / "hirise").mkdir(parents=True, exist_ok=True)
            (out_dir / "mastcam").mkdir(parents=True, exist_ok=True)

            Image.fromarray(hir_upsampled).save(out_dir / "hirise" / f"{patch_name}.png")
            Image.fromarray(mastcam_patch).save(out_dir / "mastcam" / f"{patch_name}.png")
            n_saved += 1

        if n_saved >= MAX_PATCHES_PER_SOL:
            break

    return n_saved


def main():
    print("=== MarsOrtho I2I Dataset Builder ===\n")

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

    print(f"Found {len(sol_nums)} sols")

    if not HIRISE_JP2.exists():
        print(f"ERROR: HiRISE JP2 not found")
        return

    # Train/test split: hold out last 1/7 sols
    n_test = max(1, len(sol_nums) // 7)
    test_sols = set(sol_nums[-n_test:])
    print(f"Train: {len(sol_nums) - len(test_sols)} sols, Test: {len(test_sols)} sols\n")

    with rasterio.open(HIRISE_JP2) as hir_ds:
        print(f"HiRISE: {hir_ds.width}x{hir_ds.height}, CRS: {hir_ds.crs.to_string()[:50]}...\n")

        n_train = n_test_total = 0
        for sol in sol_nums:
            is_test = sol in test_sols
            n = extract_paired_patches(sol, hir_ds, train_dir, test_dir, is_test)
            split = "test" if is_test else "train"
            print(f"  sol{sol:05d} ({split}): {n} patches")
            if is_test:
                n_test_total += n
            else:
                n_train += n

    print(f"\n{'='*50}")
    print(f"Train: {n_train}, Test: {n_test_total}")
    print(f"Output: {DATASET_DIR}")
    print(f"{'='*50}")

    # Kaggle dataset metadata
    meta = {
        "title": "MarsOrtho I2I",
        "id": "carsonparksnu/marsortho-i2i",
        "licenses": [{"name": "CC0-1.0"}],
        "subtitle": "HiRISE → Mastcam-Z paired image-to-image translation dataset",
        "description": (
            f"Paired HiRISE (orbital, 1ch) and Mastcam-Z (ground-projected, 3ch) "
            f"patches for cross-perspective texture translation on Mars.\n\n"
            f"- Train: {n_train} patches\n"
            f"- Test: {n_test_total} patches\n"
            f"- Patch size: 256x256 (both)\n"
            f"- Coverage: Jezero Crater\n"
        ),
    }
    with open(DATASET_DIR / "dataset-metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
