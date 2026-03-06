#!/usr/bin/env python3
"""
Prepare V5 training data for Colab — SCT expansion.

Outputs a tar.gz with:
  - tiles/{image_id}/tile_{row:03d}_{col:03d}.jpg  (JPEG tiles)
  - mola_features_by_tile.npy                       (MOLA 25-dim per tile)
  - tile_labels_v5.json                             (merged labels)
  - tile_splits_v5.json                             (train/val/test split indices)
  - exemplar_buffer_v5.json                         (old-class exemplar indices)
  - marslandform_v4b_deploy.pt                      (teacher checkpoint)

Usage:
  python prepare_v5_colab_data.py [--output-dir PATH] [--skip-tiles] [--workers N]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(os.getenv("MARSLAB_ROOT", "/disk1/cspark/MarsLab"))

# Inputs
V5_LABELS = ROOT / "Data/HiRISE/v3_output/tile_labels_v5.json"
V5_SPLITS = ROOT / "Data/HiRISE/v3_output/tile_splits_v5.json"
V5_EXEMPLARS = ROOT / "Data/HiRISE/v3_output/exemplar_buffer_v5.json"
BROWSE_DIR = ROOT / "Data/HiRISE/midlat_browse"
MOLA_DEM = ROOT / "Mars_HRSC_MOLA_BlendDEM_Global_200mp_v2.tif"
V4B_CHECKPOINT = ROOT / "Data/HiRISE/v3_output/models/marslandform_v4b_deploy.pt"

# Existing V4 MOLA features (reuse — 8313 images already computed)
V4_MOLA = ROOT / "Data/HiRISE/v4_colab_data_expanded/mola_features_by_tile.npy"

TILE_SIZE = 224
MIN_CONTENT = 0.3
JPEG_QUALITY = 85
MARS_RADIUS_M = 3389500.0
DEM_RESOLUTION_M = 200.0


def extract_tiles_from_image(browse_path: Path):
    """Extract 224x224 tiles from browse image."""
    try:
        img = Image.open(browse_path).convert("RGB")
    except Exception:
        return []

    w, h = img.size
    arr = np.array(img)
    tiles = []

    for row in range(0, h - TILE_SIZE + 1, TILE_SIZE):
        for col in range(0, w - TILE_SIZE + 1, TILE_SIZE):
            tile_arr = arr[row:row + TILE_SIZE, col:col + TILE_SIZE]
            content_frac = np.mean(tile_arr > 10)
            if content_frac < MIN_CONTENT:
                continue
            tiles.append({
                "tile_row": row // TILE_SIZE,
                "tile_col": col // TILE_SIZE,
                "tile_array": tile_arr,
            })
    return tiles


def save_image_tiles(args_tuple):
    """Worker: extract and save tiles for one image."""
    img_id, browse_dir, tiles_dir, needed_tiles, jpeg_quality = args_tuple

    browse_path = Path(browse_dir) / f"{img_id}_RED.abrowse.jpg"
    if not browse_path.exists():
        matches = list(Path(browse_dir).glob(f"{img_id}*"))
        if matches:
            browse_path = matches[0]
        else:
            return img_id, {}, 0

    tiles = extract_tiles_from_image(browse_path)
    if not tiles:
        return img_id, {}, 0

    img_dir = Path(tiles_dir) / img_id
    img_dir.mkdir(exist_ok=True, parents=True)

    tile_index = {}
    n_saved = 0
    needed_set = set(needed_tiles.get(img_id, []))

    for tile in tiles:
        tr, tc = tile["tile_row"], tile["tile_col"]
        if needed_set and (tr, tc) not in needed_set:
            continue

        tile_fname = f"tile_{tr:03d}_{tc:03d}.jpg"
        tile_path = img_dir / tile_fname
        Image.fromarray(tile["tile_array"]).save(tile_path, quality=jpeg_quality)

        key = f"{img_id}_{tr}_{tc}"
        tile_index[key] = f"tiles/{img_id}/{tile_fname}"
        n_saved += 1

    return img_id, tile_index, n_saved


def main():
    parser = argparse.ArgumentParser(description="Prepare V5 Colab data package")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "Data/HiRISE/v5_colab_data")
    parser.add_argument("--skip-tiles", action="store_true", help="Skip tile extraction (reuse existing)")
    parser.add_argument("--skip-mola", action="store_true", help="Skip MOLA extraction (reuse existing)")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load labels and splits
    logger.info("Loading V5 labels and splits...")
    with open(V5_LABELS) as f:
        labels = json.load(f)
    with open(V5_SPLITS) as f:
        splits = json.load(f)

    # Collect all needed (image_id, tile_row, tile_col)
    all_indices = []
    for split_name in ["train", "val", "test"]:
        all_indices.extend(splits[split_name])
    all_indices = sorted(set(all_indices))

    needed = {}
    for idx in all_indices:
        if idx < len(labels):
            t = labels[idx]
            iid = t["image_id"]
            needed.setdefault(iid, []).append((t["tile_row"], t["tile_col"]))

    unique_images = list(needed.keys())
    logger.info(f"Need tiles from {len(unique_images)} images ({len(all_indices)} tiles)")

    # 2. Extract tiles
    tiles_dir = out_dir / "tiles"
    if not args.skip_tiles:
        logger.info("Extracting tiles...")
        tiles_dir.mkdir(parents=True, exist_ok=True)

        work_items = [
            (img_id, str(BROWSE_DIR), str(tiles_dir), needed, JPEG_QUALITY)
            for img_id in unique_images
        ]

        total_saved = 0
        tile_index = {}
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(save_image_tiles, w): w[0] for w in work_items}
            for i, future in enumerate(as_completed(futures)):
                img_id, idx, n = future.result()
                tile_index.update(idx)
                total_saved += n
                if (i + 1) % 100 == 0:
                    logger.info(f"  [{i+1}/{len(futures)}] {total_saved} tiles saved")

        logger.info(f"Saved {total_saved} tiles to {tiles_dir}")

        with open(out_dir / "tile_index.json", "w") as f:
            json.dump(tile_index, f)
    else:
        logger.info("Skipping tile extraction (--skip-tiles)")

    # 3. MOLA features
    mola_out = out_dir / "mola_features_by_tile.npy"
    if not args.skip_mola:
        if V4_MOLA.exists():
            logger.info(f"Copying existing MOLA features from {V4_MOLA}")
            # Load existing, filter to needed images, and save
            existing_mola = np.load(V4_MOLA, allow_pickle=True).item()
            # Keep all existing + we'll need to compute new ones for SCT images
            needed_images = set(unique_images)
            existing_images = set(existing_mola.keys())
            missing_images = needed_images - existing_images
            logger.info(f"  Existing MOLA: {len(existing_images & needed_images)}, missing: {len(missing_images)}")

            if missing_images:
                logger.info(f"Computing MOLA features for {len(missing_images)} new images...")
                import rasterio
                ds = rasterio.open(str(MOLA_DEM))

                # Simplified MOLA extraction for missing images
                sys.path.insert(0, str(ROOT))
                from scripts.marslandform_v2.prepare_v4_colab_data import extract_mola_features_for_tile

                new_mola = {}
                for i, img_id in enumerate(missing_images):
                    img_mola = {}
                    for tr, tc in needed.get(img_id, []):
                        t_label = next(
                            (l for l in labels if l["image_id"] == img_id and l["tile_row"] == tr and l["tile_col"] == tc),
                            None
                        )
                        if t_label and t_label.get("lat") and t_label.get("lon"):
                            feats = extract_mola_features_for_tile(ds, t_label["lat"], t_label["lon"])
                            img_mola[f"{tr}_{tc}"] = feats
                    if img_mola:
                        new_mola[img_id] = img_mola
                    if (i + 1) % 50 == 0:
                        logger.info(f"  MOLA [{i+1}/{len(missing_images)}]")

                ds.close()
                existing_mola.update(new_mola)
                logger.info(f"  Computed MOLA for {len(new_mola)} new images")

            np.save(mola_out, existing_mola)
            logger.info(f"Saved MOLA features to {mola_out}")
        else:
            logger.warning(f"No existing MOLA file at {V4_MOLA}. Computing from scratch...")
            logger.warning("This will take a long time. Consider running prepare_v4_colab_data.py first.")
    else:
        logger.info("Skipping MOLA extraction (--skip-mola)")

    # 4. Copy labels, splits, exemplars
    for src, name in [
        (V5_LABELS, "tile_labels_v5.json"),
        (V5_SPLITS, "tile_splits_v5.json"),
        (V5_EXEMPLARS, "exemplar_buffer_v5.json"),
    ]:
        dst = out_dir / name
        shutil.copy2(src, dst)
        logger.info(f"Copied {name}")

    # 5. Copy V4b checkpoint (teacher)
    if V4B_CHECKPOINT.exists():
        dst = out_dir / "marslandform_v4b_deploy.pt"
        shutil.copy2(V4B_CHECKPOINT, dst)
        logger.info(f"Copied V4b checkpoint ({V4B_CHECKPOINT.stat().st_size / 1e6:.1f} MB)")
    else:
        logger.warning(f"V4b checkpoint not found: {V4B_CHECKPOINT}")

    # 6. Summary
    logger.info("\n=== V5 Colab Data Package ===")
    logger.info(f"Output: {out_dir}")
    total_size = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file())
    logger.info(f"Total size: {total_size / 1e9:.2f} GB")

    label_counts = Counter(labels[i]["label"] for i in all_indices if i < len(labels))
    logger.info(f"Labels: {dict(label_counts)}")
    logger.info(f"\nTo upload: tar -czf v5_colab_data.tar.gz -C {out_dir.parent} {out_dir.name}")


if __name__ == "__main__":
    main()
