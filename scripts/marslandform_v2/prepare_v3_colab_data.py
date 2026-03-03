"""
Prepare V3 training data for Colab end-to-end DINOv2 fine-tuning.

Outputs:
  1. tiles/ directory: JPEG tile images organized as tiles/{image_id}/tile_{row}_{col}.jpg
  2. mola_features_by_tile.npy: {image_id: ndarray(max_tile_idx+1, 25)} MOLA features
  3. ssl_lora_weights.pt: LoRA-only weights from SSL pretraining (~3.5MB)
  4. tile_index.json: mapping from (image_id, tile_row, tile_col) -> tile JPEG path
  5. Copies tile_labels_v3.json and tile_splits_v3.json

Usage:
  python prepare_v3_colab_data.py --output-dir /disk1/cspark/MarsLab/Data/HiRISE/v3_colab_data
"""

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TILE_SIZE = 224
MIN_CONTENT = 0.3
MARS_RADIUS_M = 3389500.0
DEM_RESOLUTION_M = 200.0


def extract_tiles_from_image(image_path: Path) -> list[dict]:
    """Extract 224x224 tiles with content check. Returns list of tile metadata."""
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        logger.warning(f"Cannot open {image_path}: {e}")
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
        # Vectorized TRI: mean absolute difference from center
        mean_elev = uniform_filter(elev, size=3, mode='nearest')
        sq_mean = uniform_filter(elev**2, size=3, mode='nearest')
        # TRI approx = sqrt(sum_of_squared_diffs / 8) ≈ std of 3x3 neighborhood
        return np.sqrt(np.maximum(sq_mean - mean_elev**2, 0))

    def compute_roughness(elev):
        from scipy.ndimage import maximum_filter, minimum_filter
        return maximum_filter(elev, size=3) - minimum_filter(elev, size=3)

    features = []

    # Multi-scale features (7 x 3 = 21)
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

    # Relative features (+2 = 25): placeholder (filled later per-image)
    features.extend([0.0, 0.0])

    return np.array(features, dtype=np.float32)


def extract_lora_weights(ssl_ckpt_path: Path, output_path: Path):
    """Extract only LoRA weights from SSL checkpoint."""
    import torch
    ckpt = torch.load(ssl_ckpt_path, map_location="cpu")
    student_backbone = ckpt["student_backbone"]

    # Extract LoRA-only weights
    lora_state = {k: v for k, v in student_backbone.items() if "lora" in k.lower()}
    torch.save({"lora_state_dict": lora_state, "lora_config": {
        "r": 16, "alpha": 32, "dropout": 0.1,
        "target_modules": ["query", "key", "value"],
    }}, output_path)

    total_params = sum(v.numel() for v in lora_state.values())
    logger.info(f"Saved {len(lora_state)} LoRA tensors ({total_params:,} params) to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Prepare V3 training data for Colab")
    parser.add_argument("--output-dir", type=str,
                        default="/disk1/cspark/MarsLab/Data/HiRISE/v3_colab_data")
    parser.add_argument("--browse-dir", type=str,
                        default="/disk1/cspark/MarsLab/Data/HiRISE/midlat_browse")
    parser.add_argument("--labels", type=str,
                        default="/disk1/cspark/MarsLab/Data/HiRISE/v3_output/tile_labels_v3.json")
    parser.add_argument("--splits", type=str,
                        default="/disk1/cspark/MarsLab/Data/HiRISE/v3_output/tile_splits_v3.json")
    parser.add_argument("--metadata", type=str,
                        default="/disk1/cspark/MarsLab/Data/HiRISE/midlat_metadata.json")
    parser.add_argument("--dem", type=str,
                        default="/disk1/cspark/MarsLab/Mars_HRSC_MOLA_BlendDEM_Global_200mp_v2.tif")
    parser.add_argument("--ssl-weights", type=str,
                        default="/disk1/cspark/MarsLab/Data/HiRISE/v2_output/ssl_lora_weights/best_model.pt")
    parser.add_argument("--skip-mola", action="store_true", help="Skip MOLA extraction (slow)")
    parser.add_argument("--skip-tiles", action="store_true", help="Skip tile extraction")
    parser.add_argument("--jpeg-quality", type=int, default=85)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load labels and splits
    logger.info("Loading labels and splits...")
    with open(args.labels) as f:
        tile_labels = json.load(f)
    with open(args.splits) as f:
        splits = json.load(f)
    with open(args.metadata) as f:
        metadata = json.load(f)
    if isinstance(metadata, list):
        metadata = {m["image_id"]: m for m in metadata}

    # Build lookup: (image_id, tile_row, tile_col) -> label info
    label_lookup = {}
    for t in tile_labels:
        key = (t["image_id"], t["tile_row"], t["tile_col"])
        label_lookup[key] = t

    # Get unique images from labeled tiles
    labeled_images = sorted(set(t["image_id"] for t in tile_labels))
    logger.info(f"Labels: {len(tile_labels)} tiles, {len(labeled_images)} images")

    # ── Step 1: Extract tile images ─────────────────────────────────────────
    if not args.skip_tiles:
        logger.info("=== Step 1: Extracting tile images ===")
        tiles_dir = output_dir / "tiles"
        tiles_dir.mkdir(exist_ok=True)
        tile_index = {}  # maps (image_id, tile_row, tile_col) -> relative path
        total_tiles = 0
        total_matched = 0

        for i, img_id in enumerate(labeled_images):
            browse_path = Path(args.browse_dir) / f"{img_id}_RED.abrowse.jpg"
            if not browse_path.exists():
                # Try glob
                matches = list(Path(args.browse_dir).glob(f"{img_id}*"))
                if matches:
                    browse_path = matches[0]
                else:
                    continue

            tiles = extract_tiles_from_image(browse_path)
            if not tiles:
                continue

            img_dir = tiles_dir / img_id
            img_dir.mkdir(exist_ok=True)

            for tile in tiles:
                tr, tc = tile["tile_row"], tile["tile_col"]
                key = (img_id, tr, tc)

                # Save tile JPEG
                tile_fname = f"tile_{tr:03d}_{tc:03d}.jpg"
                tile_path = img_dir / tile_fname
                Image.fromarray(tile["tile_array"]).save(tile_path, quality=args.jpeg_quality)

                rel_path = f"tiles/{img_id}/{tile_fname}"
                tile_index[f"{img_id}_{tr}_{tc}"] = rel_path
                total_tiles += 1

                if key in label_lookup:
                    total_matched += 1

            if (i + 1) % 50 == 0:
                logger.info(f"  Tiled {i+1}/{len(labeled_images)} images ({total_tiles} tiles, {total_matched} matched)")

        logger.info(f"Tile extraction done: {total_tiles} tiles, {total_matched} matched to labels")

        # Save tile index
        with open(output_dir / "tile_index.json", "w") as f:
            json.dump(tile_index, f)
        logger.info(f"Saved tile_index.json ({len(tile_index)} entries)")

    # ── Step 2: Compute MOLA features ───────────────────────────────────────
    if not args.skip_mola:
        logger.info("=== Step 2: Computing MOLA features ===")
        try:
            import rasterio
        except ImportError:
            logger.error("rasterio required: pip install rasterio")
            sys.exit(1)

        ds = rasterio.open(args.dem)
        logger.info(f"DEM loaded: {ds.width}x{ds.height} ({DEM_RESOLUTION_M}m/px)")

        # Pre-index tiles by image for O(1) lookup
        tiles_by_image = defaultdict(list)
        for t in tile_labels:
            if t.get('label') != 'UNLABELED':  # Skip unlabeled tiles for MOLA
                tiles_by_image[t['image_id']].append(t)
        logger.info(f"Computing MOLA for {sum(len(v) for v in tiles_by_image.values())} non-UNLABELED tiles")
        
        mola_features = {}  # image_id -> {tile_key -> features}
        n_computed = 0
        t0 = time.time()

        for i, img_id in enumerate(labeled_images):
            img_tiles = tiles_by_image.get(img_id, [])
            img_features = {}

            for t in img_tiles:
                lat = t.get("lat")
                lon = t.get("lon")
                if lat is None or lon is None:
                    continue

                try:
                    feats = extract_mola_features_for_tile(ds, float(lat), float(lon))
                    key = f"{t['tile_row']}_{t['tile_col']}"
                    img_features[key] = feats
                    n_computed += 1
                except Exception as e:
                    pass

            if img_features:
                mola_features[img_id] = img_features

            if (i + 1) % 50 == 0:
                elapsed = time.time() - t0
                rate = n_computed / elapsed if elapsed > 0 else 0
                logger.info(f"  MOLA {i+1}/{len(labeled_images)} images ({n_computed} tiles, {rate:.0f} tiles/s)")

        ds.close()

        # Fill relative features (per-image normalization)
        for img_id, tiles_dict in mola_features.items():
            if not tiles_dict:
                continue
            all_feats = np.stack(list(tiles_dict.values()))
            # Feature index 21 = elevation_mean, index 0 = slope_mean at 1km
            img_mean_elev = np.mean(all_feats[:, 21])
            img_mean_slope = np.mean(all_feats[:, 0])
            for key in tiles_dict:
                tiles_dict[key][23] = tiles_dict[key][21] - img_mean_elev  # elev_rel
                tiles_dict[key][24] = tiles_dict[key][0] - img_mean_slope  # slope_rel

        # Save as numpy dict
        np.save(output_dir / "mola_features_by_tile.npy", mola_features, allow_pickle=True)
        logger.info(f"Saved mola_features_by_tile.npy ({n_computed} tiles, {len(mola_features)} images)")

    # ── Step 3: Extract LoRA weights ────────────────────────────────────────
    logger.info("=== Step 3: Extracting SSL LoRA weights ===")
    ssl_path = Path(args.ssl_weights)
    if ssl_path.exists():
        extract_lora_weights(ssl_path, output_dir / "ssl_lora_weights.pt")
    else:
        logger.warning(f"SSL weights not found: {ssl_path}")

    # ── Step 4: Copy labels and splits ──────────────────────────────────────
    logger.info("=== Step 4: Copying labels and splits ===")
    import shutil
    shutil.copy2(args.labels, output_dir / "tile_labels_v3.json")
    shutil.copy2(args.splits, output_dir / "tile_splits_v3.json")
    logger.info("Copied tile_labels_v3.json and tile_splits_v3.json")

    # ── Summary ─────────────────────────────────────────────────────────────
    logger.info("\n=== Summary ===")
    for f in sorted(output_dir.rglob("*")):
        if f.is_file() and "tiles/" not in str(f.relative_to(output_dir)):
            size_mb = f.stat().st_size / 1e6
            logger.info(f"  {f.relative_to(output_dir)}: {size_mb:.1f}MB")

    tiles_dir = output_dir / "tiles"
    if tiles_dir.exists():
        n_tile_files = sum(1 for _ in tiles_dir.rglob("*.jpg"))
        tiles_size_mb = sum(f.stat().st_size for f in tiles_dir.rglob("*.jpg")) / 1e6
        logger.info(f"  tiles/: {n_tile_files} JPEGs, {tiles_size_mb:.1f}MB total")

    logger.info(f"\nOutput directory: {output_dir}")
    logger.info("Next: tar czf v3_colab_e2e_data.tar.gz -C output_dir .")


if __name__ == "__main__":
    main()
