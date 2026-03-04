#!/usr/bin/env python3
"""
Prepare V4 training data for Colab DINOv2-LoRA fine-tuning.

V4 key changes vs V3:
  - 8,490 images (was 639) — 3.7x more Levy-overlapping images from ODE download
  - CCF: 15,827 train tiles (was 536!) — the critical bottleneck is solved
  - Subsampling: keep all CCF/LVF, cap LDA~20K, OTHER~20K to balance classes
  - MOLA features: 25-dim per tile (7 features × 3 scales + 2 global + 2 relative)

Outputs:
  1. tiles/ — JPEG tile images: tiles/{image_id}/tile_{row:03d}_{col:03d}.jpg
  2. mola_features_by_tile.npy — {image_id: {row_col: ndarray(25,)}}
  3. ssl_lora_weights.pt — LoRA-only weights from SSL pretraining (~3.5MB)
  4. tile_index.json — (image_id, tile_row, tile_col) → relative JPEG path
  5. tile_labels_v4.json — subsampled labels (list of dicts)
  6. tile_splits_v4.json — train/val/test indices into tile_labels_v4
  7. label_stats_v4.json — class distribution summary

Usage:
  python prepare_v4_colab_data.py [--output-dir PATH] [--skip-mola] [--skip-tiles] [--workers N]
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────
TILE_SIZE = 224
MIN_CONTENT = 0.3  # minimum fraction of non-black pixels
MARS_RADIUS_M = 3389500.0
DEM_RESOLUTION_M = 200.0
JPEG_QUALITY = 85

# ─── Subsampling config ──────────────────────────────────────────────────────
# Keep ALL CCF and LVF (minority classes), cap LDA and OTHER
SUBSAMPLE_CAPS = {
    "CCF": None,      # keep all (~15K train)
    "LVF": None,      # keep all (~16K train)
    "LDA": 20000,     # cap at 20K per split ratio (was 71K train)
    "OTHER": 20000,   # cap at 20K per split ratio (was 681K train)
}


def subsample_labels(
    all_labels: List[dict],
    splits: Dict[str, List[int]],
    caps: Dict[str, int],
    seed: int = 42,
) -> Tuple[List[dict], Dict[str, List[int]]]:
    """
    Subsample labels to balance classes. Keeps all minority classes,
    caps majority classes per split to maintain train/val/test proportions.

    Returns: (subsampled_labels, new_splits)
    """
    rng = np.random.RandomState(seed)

    # Build reverse index: original_idx → split_name
    idx_to_split = {}
    for split_name, indices in splits.items():
        for idx in indices:
            idx_to_split[idx] = split_name

    # Group by (split, label)
    groups = defaultdict(list)  # (split, label) → [original_indices]
    for idx, tile in enumerate(all_labels):
        label = tile.get("label", "UNLABELED")
        if label == "UNLABELED":
            continue
        split_name = idx_to_split.get(idx)
        if split_name is None:
            continue
        groups[(split_name, label)].append(idx)

    # Compute split ratios from original data
    split_sizes = {s: len(idxs) for s, idxs in splits.items()}
    total = sum(split_sizes.values())
    split_ratios = {s: sz / total for s, sz in split_sizes.items()}

    # Subsample each group
    selected_indices = []
    for (split_name, label), indices in groups.items():
        cap = caps.get(label)
        if cap is not None:
            # Allocate cap proportionally to split
            split_cap = max(1, int(cap * split_ratios[split_name]))
            if len(indices) > split_cap:
                chosen = rng.choice(indices, size=split_cap, replace=False).tolist()
            else:
                chosen = indices
        else:
            chosen = indices
        selected_indices.extend(chosen)

    # Sort selected indices to maintain order
    selected_indices = sorted(selected_indices)

    # Build new labels list and new splits
    new_labels = []
    new_splits = {"train": [], "val": [], "test": []}
    old_to_new = {}

    for new_idx, old_idx in enumerate(selected_indices):
        old_to_new[old_idx] = new_idx
        new_labels.append(all_labels[old_idx])
        split_name = idx_to_split[old_idx]
        new_splits[split_name].append(new_idx)

    return new_labels, new_splits


def extract_tiles_from_image(image_path: Path) -> List[dict]:
    """Extract 224×224 tiles with content check. Returns list of tile info."""
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
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


def _extract_and_save_image_tiles(args_tuple):
    """Worker function for parallel tile extraction."""
    img_id, browse_dir, tiles_dir, needed_tiles, jpeg_quality = args_tuple

    browse_path = Path(browse_dir) / f"{img_id}_RED.abrowse.jpg"
    if not browse_path.exists():
        # Try glob fallback
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

    # Build set of needed (row, col) for this image
    needed_set = set(needed_tiles.get(img_id, []))

    for tile in tiles:
        tr, tc = tile["tile_row"], tile["tile_col"]
        # Only save tiles we actually need (in our subsampled label set)
        if needed_set and (tr, tc) not in needed_set:
            continue

        tile_fname = f"tile_{tr:03d}_{tc:03d}.jpg"
        tile_path = img_dir / tile_fname
        Image.fromarray(tile["tile_array"]).save(tile_path, quality=jpeg_quality)

        key = f"{img_id}_{tr}_{tc}"
        tile_index[key] = f"tiles/{img_id}/{tile_fname}"
        n_saved += 1

    return img_id, tile_index, n_saved


def extract_mola_features_for_tile(ds, lat: float, lon: float, scales_km=(1.0, 5.0, 20.0)) -> np.ndarray:
    """Extract 25 MOLA features for one tile center location."""

    def latlon_to_pixel(lat_, lon_):
        row, col = ds.index(lon_, lat_)
        row = max(0, min(int(row), ds.height - 1))
        col = max(0, min(int(col), ds.width - 1))
        return row, col

    def extract_window(lat_, lon_, radius_km):
        radius_px = max(1, int(radius_km * 1000 / DEM_RESOLUTION_M))
        r, c = latlon_to_pixel(lat_, lon_)
        r0 = max(0, r - radius_px)
        r1 = min(ds.height, r + radius_px + 1)
        c0 = max(0, c - radius_px)
        c1 = min(ds.width, c + radius_px + 1)
        window = ds.read(1, window=((r0, r1), (c0, c1)))
        nodata = ds.nodata
        if nodata is not None:
            window = window.astype(np.float64)
            window[window == nodata] = np.nan
        return window

    def compute_slope(elev, cell=DEM_RESOLUTION_M):
        if elev.shape[0] < 3 or elev.shape[1] < 3:
            return np.zeros_like(elev)
        dy, dx = np.gradient(elev, cell)
        return np.degrees(np.arctan(np.sqrt(dx ** 2 + dy ** 2)))

    def compute_curvature(elev, cell=DEM_RESOLUTION_M):
        if elev.shape[0] < 3 or elev.shape[1] < 3:
            return np.zeros_like(elev)
        dy, dx = np.gradient(elev, cell)
        dyy, _ = np.gradient(dy, cell)
        _, dxx = np.gradient(dx, cell)
        return dyy + dxx

    def compute_tpi(elev, radius_px=5):
        from scipy.ndimage import uniform_filter
        kernel = 2 * radius_px + 1
        mean_e = uniform_filter(elev, size=kernel, mode="nearest")
        return elev - mean_e

    def compute_tri(elev):
        from scipy.ndimage import uniform_filter
        mean_elev = uniform_filter(elev, size=3, mode='nearest')
        sq_mean = uniform_filter(elev**2, size=3, mode='nearest')
        return np.sqrt(np.maximum(sq_mean - mean_elev**2, 0))

    def compute_roughness(elev):
        from scipy.ndimage import maximum_filter, minimum_filter
        return maximum_filter(elev, size=3) - minimum_filter(elev, size=3)

    features = []

    # Multi-scale features (7 × 3 = 21)
    for scale in scales_km:
        window = extract_window(lat, lon, scale)
        if window.size == 0 or np.all(np.isnan(window)):
            features.extend([0.0] * 7)
            continue

        valid = window[~np.isnan(window)]
        median_val = float(np.median(valid)) if len(valid) > 0 else 0.0
        wf = np.where(np.isnan(window), median_val, window)

        slope = compute_slope(wf)
        curv = compute_curvature(wf)
        tpi_r = max(1, wf.shape[0] // 4)
        tpi = compute_tpi(wf, radius_px=tpi_r)
        tri = compute_tri(wf)
        rough = compute_roughness(wf)

        mean_slope = np.nanmean(slope)
        lobateness = float(np.nanmax(slope) / mean_slope) if mean_slope > 0.1 else 0.0

        features.extend([
            float(np.nanmean(slope)),
            float(np.nanstd(slope)),
            float(np.nanmean(curv)),
            float(np.nanmean(tpi)),
            float(np.nanmean(tri)),
            float(np.nanmean(rough)),
            lobateness,
        ])

    # Global features (+2 = 23)
    w1 = extract_window(lat, lon, 1.0)
    elev_mean = float(np.nanmean(w1)) if w1.size > 0 else 0.0
    features.append(elev_mean)
    features.append(abs(lat))

    # Relative features (+2 = 25): placeholder (filled per-image later)
    features.extend([0.0, 0.0])

    return np.array(features, dtype=np.float32)


def extract_lora_weights(ssl_ckpt_path: Path, output_path: Path):
    """Extract only LoRA weights from SSL checkpoint."""
    import torch
    ckpt = torch.load(ssl_ckpt_path, map_location="cpu")
    student_backbone = ckpt["student_backbone"]

    lora_state = {k: v for k, v in student_backbone.items() if "lora" in k.lower()}
    torch.save({"lora_state_dict": lora_state, "lora_config": {
        "r": 16, "alpha": 32, "dropout": 0.1,
        "target_modules": ["query", "key", "value"],
    }}, output_path)

    total_params = sum(v.numel() for v in lora_state.values())
    logger.info(f"Saved {len(lora_state)} LoRA tensors ({total_params:,} params) to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare V4 training data for Colab")
    parser.add_argument("--output-dir", type=str,
                        default="/disk1/cspark/MarsLab/Data/HiRISE/v4_colab_data_expanded")
    parser.add_argument("--browse-dir", type=str,
                        default="/disk1/cspark/MarsLab/Data/HiRISE/midlat_browse")
    parser.add_argument("--labels", type=str,
                        default="/disk1/cspark/MarsLab/Data/HiRISE/v4_output/tile_labels_v3.json")
    parser.add_argument("--splits", type=str,
                        default="/disk1/cspark/MarsLab/Data/HiRISE/v4_output/tile_splits_v3.json")
    parser.add_argument("--metadata", type=str,
                        default="/disk1/cspark/MarsLab/Data/HiRISE/midlat_metadata.json")
    parser.add_argument("--dem", type=str,
                        default="/disk1/cspark/MarsLab/Mars_HRSC_MOLA_BlendDEM_Global_200mp_v2.tif")
    parser.add_argument("--ssl-weights", type=str,
                        default="/disk1/cspark/MarsLab/Data/HiRISE/v2_output/ssl_lora_weights/best_model.pt")
    parser.add_argument("--skip-mola", action="store_true", help="Skip MOLA extraction (slow)")
    parser.add_argument("--skip-tiles", action="store_true", help="Skip tile extraction")
    parser.add_argument("--skip-subsample", action="store_true", help="Use full labels without subsampling")
    parser.add_argument("--jpeg-quality", type=int, default=JPEG_QUALITY)
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers for tile extraction")
    parser.add_argument("--lda-cap", type=int, default=20000, help="Max LDA tiles to keep")
    parser.add_argument("--other-cap", type=int, default=20000, help="Max OTHER tiles to keep")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load raw labels and splits ────────────────────────────────────────────
    logger.info("Loading labels and splits...")
    with open(args.labels) as f:
        all_labels = json.load(f)
    with open(args.splits) as f:
        splits = json.load(f)

    # Count original distribution
    orig_dist = Counter(t["label"] for t in all_labels if t["label"] != "UNLABELED")
    logger.info(f"Original labels: {len(all_labels)} total, distribution: {dict(orig_dist)}")

    # ── Step 0: Subsample ─────────────────────────────────────────────────────
    if not args.skip_subsample:
        logger.info("=== Step 0: Subsampling labels ===")
        caps = {
            "CCF": None,
            "LVF": None,
            "LDA": args.lda_cap,
            "OTHER": args.other_cap,
        }
        labels, new_splits = subsample_labels(all_labels, splits, caps)

        # Stats
        sub_dist = Counter(t["label"] for t in labels)
        logger.info(f"Subsampled: {len(labels)} tiles")
        logger.info(f"  Distribution: {dict(sub_dist)}")
        for split_name in ["train", "val", "test"]:
            split_dist = Counter(labels[i]["label"] for i in new_splits[split_name])
            logger.info(f"  {split_name}: {len(new_splits[split_name])} tiles — {dict(split_dist)}")
    else:
        # Filter out UNLABELED, keep everything
        labels = [t for t in all_labels if t["label"] != "UNLABELED"]
        # Rebuild splits (filter to non-UNLABELED indices)
        old_to_new = {}
        for new_idx, t in enumerate(labels):
            pass  # need original index
        # Simpler: just use all labels with original splits filtered
        logger.info("Skipping subsampling, using full label set (minus UNLABELED)")
        labeled_indices = set()
        for idx, t in enumerate(all_labels):
            if t["label"] != "UNLABELED":
                labeled_indices.add(idx)
        new_splits = {}
        old_to_new = {}
        new_idx = 0
        for idx in sorted(labeled_indices):
            old_to_new[idx] = new_idx
            new_idx += 1
        labels = [all_labels[i] for i in sorted(labeled_indices)]
        for split_name in ["train", "val", "test"]:
            new_splits[split_name] = [old_to_new[i] for i in splits[split_name] if i in old_to_new]

    # ── Save subsampled labels and splits ─────────────────────────────────────
    with open(output_dir / "tile_labels_v4.json", "w") as f:
        json.dump(labels, f)
    with open(output_dir / "tile_splits_v4.json", "w") as f:
        json.dump(new_splits, f)

    # Save stats
    stats = {
        "total_tiles": len(labels),
        "classes": ["LDA", "LVF", "CCF", "OTHER"],
        "distribution": dict(Counter(t["label"] for t in labels)),
        "split_sizes": {k: len(v) for k, v in new_splits.items()},
        "split_distribution": {},
    }
    for split_name in ["train", "val", "test"]:
        stats["split_distribution"][split_name] = dict(
            Counter(labels[i]["label"] for i in new_splits[split_name])
        )
    with open(output_dir / "label_stats_v4.json", "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Saved tile_labels_v4.json, tile_splits_v4.json, label_stats_v4.json")

    # ── Build needed tiles index (image_id → set of (row, col)) ──────────────
    needed_tiles = defaultdict(set)
    for t in labels:
        needed_tiles[t["image_id"]].add((t["tile_row"], t["tile_col"]))
    unique_images = sorted(needed_tiles.keys())
    logger.info(f"Need tiles from {len(unique_images)} unique images")

    # ── Step 1: Extract tile images ───────────────────────────────────────────
    if not args.skip_tiles:
        logger.info(f"=== Step 1: Extracting tile images ({args.workers} workers) ===")
        tiles_dir = output_dir / "tiles"
        tiles_dir.mkdir(exist_ok=True)

        # Convert needed_tiles to serializable format for workers
        needed_tiles_list = {
            img_id: list(coords) for img_id, coords in needed_tiles.items()
        }

        # Prepare worker args
        worker_args = [
            (img_id, args.browse_dir, str(tiles_dir), needed_tiles_list, args.jpeg_quality)
            for img_id in unique_images
        ]

        tile_index = {}
        total_saved = 0
        t0 = time.time()

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(_extract_and_save_image_tiles, wa): wa[0]
                for wa in worker_args
            }
            done_count = 0
            for future in as_completed(futures):
                img_id = futures[future]
                try:
                    _, img_tile_index, n_saved = future.result()
                    tile_index.update(img_tile_index)
                    total_saved += n_saved
                except Exception as e:
                    logger.warning(f"Failed {img_id}: {e}")

                done_count += 1
                if done_count % 200 == 0:
                    elapsed = time.time() - t0
                    rate = done_count / elapsed if elapsed > 0 else 0
                    logger.info(f"  Tiled {done_count}/{len(unique_images)} images "
                               f"({total_saved} tiles, {rate:.1f} img/s)")

        elapsed = time.time() - t0
        logger.info(f"Tile extraction done: {total_saved} tiles from {len(unique_images)} images "
                    f"in {elapsed:.0f}s ({total_saved / elapsed:.0f} tiles/s)")

        with open(output_dir / "tile_index.json", "w") as f:
            json.dump(tile_index, f)
        logger.info(f"Saved tile_index.json ({len(tile_index)} entries)")

    # ── Step 2: Compute MOLA features ─────────────────────────────────────────
    if not args.skip_mola:
        logger.info("=== Step 2: Computing MOLA features ===")
        try:
            import rasterio
        except ImportError:
            logger.error("rasterio required: pip install rasterio")
            sys.exit(1)

        ds = rasterio.open(args.dem)
        logger.info(f"DEM loaded: {ds.width}×{ds.height} ({DEM_RESOLUTION_M}m/px)")

        # Group tiles by image for batch MOLA extraction
        tiles_by_image = defaultdict(list)
        for t in labels:
            tiles_by_image[t["image_id"]].append(t)

        mola_features = {}
        n_computed = 0
        n_failed = 0
        t0 = time.time()

        for i, img_id in enumerate(unique_images):
            img_tiles = tiles_by_image.get(img_id, [])
            img_features = {}

            for t in img_tiles:
                lat = t.get("lat")
                lon = t.get("lon")
                if lat is None or lon is None:
                    n_failed += 1
                    continue

                try:
                    feats = extract_mola_features_for_tile(ds, float(lat), float(lon))
                    key = f"{t['tile_row']}_{t['tile_col']}"
                    img_features[key] = feats
                    n_computed += 1
                except Exception as e:
                    n_failed += 1

            if img_features:
                mola_features[img_id] = img_features

            if (i + 1) % 200 == 0:
                elapsed = time.time() - t0
                rate = n_computed / elapsed if elapsed > 0 else 0
                logger.info(f"  MOLA {i+1}/{len(unique_images)} images "
                           f"({n_computed} tiles, {rate:.0f} tiles/s, {n_failed} failed)")

        ds.close()

        # Fill relative features (per-image normalization)
        for img_id, tiles_dict in mola_features.items():
            if not tiles_dict:
                continue
            all_feats = np.stack(list(tiles_dict.values()))
            img_mean_elev = np.mean(all_feats[:, 21])  # elevation_mean
            img_mean_slope = np.mean(all_feats[:, 0])   # slope_mean at 1km
            for key in tiles_dict:
                tiles_dict[key][23] = tiles_dict[key][21] - img_mean_elev  # elev_rel
                tiles_dict[key][24] = tiles_dict[key][0] - img_mean_slope  # slope_rel

        np.save(output_dir / "mola_features_by_tile.npy", mola_features, allow_pickle=True)
        elapsed = time.time() - t0
        logger.info(f"Saved mola_features_by_tile.npy ({n_computed} tiles, {len(mola_features)} images, "
                    f"{n_failed} failed) in {elapsed:.0f}s")

    # ── Step 3: Extract LoRA weights ──────────────────────────────────────────
    logger.info("=== Step 3: Extracting SSL LoRA weights ===")
    ssl_path = Path(args.ssl_weights)
    if ssl_path.exists():
        extract_lora_weights(ssl_path, output_dir / "ssl_lora_weights.pt")
    else:
        logger.warning(f"SSL weights not found: {ssl_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("\n=== V4 Data Preparation Summary ===")
    for f in sorted(output_dir.iterdir()):
        if f.is_file():
            size_mb = f.stat().st_size / 1e6
            logger.info(f"  {f.name}: {size_mb:.1f}MB")

    tiles_dir = output_dir / "tiles"
    if tiles_dir.exists():
        n_tile_dirs = sum(1 for d in tiles_dir.iterdir() if d.is_dir())
        n_tile_files = sum(1 for _ in tiles_dir.rglob("*.jpg"))
        tiles_size_mb = sum(f.stat().st_size for f in tiles_dir.rglob("*.jpg")) / 1e6
        logger.info(f"  tiles/: {n_tile_files} JPEGs in {n_tile_dirs} dirs, {tiles_size_mb:.1f}MB total")

    logger.info(f"\nOutput directory: {output_dir}")
    logger.info("Next: tar czf v4_colab_data_expanded.tar.gz -C <output_dir> .")


if __name__ == "__main__":
    main()
