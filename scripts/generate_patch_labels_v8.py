#!/usr/bin/env python3
"""
Generate patch-level labels from Levy 2014 expert polygons for V8 segmentation.

For each HiRISE tile (224×224 @ ~25 m/px), DINOv2 ViT-L/14 produces a 16×16
grid of patch tokens. Each patch covers 14×14 browse pixels ≈ 350m × 350m.

This script assigns a class label to each patch using POINT-IN-POLYGON test
on the patch center. Much faster than full intersection computation.

Output: patch_labels_v8.npy — dict of dicts:
  {product_id: {row_col: np.array(16, 16, dtype=uint8)}}
  Values: 0=LDA, 1=LVF, 2=CCF, 3=OTHER, 255=UNLABELED (ignore in loss)
"""
from __future__ import annotations

import json
import logging
import math
import time
from collections import Counter
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import Point, box as shapely_box
from shapely.strtree import STRtree
from shapely.prepared import prep

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path("/disk1/cspark/MarsLab")
SHAPEFILE = ROOT / "Data/external_datasets/levy_2014_glacial/extracted/AreaIceUnits.shp"
TILE_LABELS = ROOT / "Data/HiRISE/v5_retrain/tile_labels_v5.json"
OUTPUT_DIR = ROOT / "Data/HiRISE/v8_segmentation"

# ── Constants ──────────────────────────────────────────────────────────────────
MARS_RADIUS_M = 3396190.0
DEG_TO_RAD = math.pi / 180.0

PATCHES_PER_SIDE = 16
PATCH_SIZE_PX = 14
TILE_SIZE_PX = 224
BROWSE_SCALE_M_PER_PX = 25.0

TILE_SIZE_M = TILE_SIZE_PX * BROWSE_SCALE_M_PER_PX  # 5600 m
PATCH_SIZE_M = PATCH_SIZE_PX * BROWSE_SCALE_M_PER_PX  # 350 m

CLASS_NAMES = ["LDA", "LVF", "CCF", "OTHER"]
CLASS_MAP = {"LDA": 0, "LVF": 1, "CCF": 2, "OTHER": 3}
UNLABELED = 255
LEVY_CODE_MAP = {"ccf": "CCF", "lda": "LDA", "lvf": "LVF"}


def latlon_to_equirect(lat_deg: float, lon_deg: float) -> tuple[float, float]:
    x = lon_deg * DEG_TO_RAD * MARS_RADIUS_M
    y = lat_deg * DEG_TO_RAD * MARS_RADIUS_M
    return x, y


def compute_patch_centers_equirect(
    tile_lat: float, tile_lon: float,
) -> np.ndarray:
    """
    Compute equirectangular (x, y) for each 16×16 patch center.
    Returns: (16, 16, 2) array of (x, y) in meters.
    """
    cx, cy = latlon_to_equirect(tile_lat, tile_lon)
    centers = np.zeros((PATCHES_PER_SIDE, PATCHES_PER_SIDE, 2), dtype=np.float64)
    for pr in range(PATCHES_PER_SIDE):
        for pc in range(PATCHES_PER_SIDE):
            offset_y = (7.5 - pr) * PATCH_SIZE_M
            offset_x = (pc - 7.5) * PATCH_SIZE_M
            centers[pr, pc, 0] = cx + offset_x
            centers[pr, pc, 1] = cy + offset_y
    return centers


def classify_patches_for_tile(
    tile_lat: float,
    tile_lon: float,
    gdf: gpd.GeoDataFrame,
    tree: STRtree,
    tile_label_type: str,
    prepared_geoms: list,
) -> np.ndarray:
    """
    Assign class labels using point-in-polygon on patch centers.
    Returns: (16, 16) uint8 array.
    """
    labels = np.full((PATCHES_PER_SIDE, PATCHES_PER_SIDE), UNLABELED, dtype=np.uint8)

    # Fast path: negative tiles → all OTHER
    if tile_label_type == "negative":
        labels[:] = CLASS_MAP["OTHER"]
        return labels

    # Query polygons near this tile
    tile_cx, tile_cy = latlon_to_equirect(tile_lat, tile_lon)
    half = TILE_SIZE_M / 2 + PATCH_SIZE_M
    tile_bbox = shapely_box(tile_cx - half, tile_cy - half, tile_cx + half, tile_cy + half)
    candidate_indices = tree.query(tile_bbox)

    if len(candidate_indices) == 0:
        return labels  # UNLABELED

    # Get patch centers in equirectangular coordinates
    centers = compute_patch_centers_equirect(tile_lat, tile_lon)

    # Point-in-polygon test for each patch center
    for pr in range(PATCHES_PER_SIDE):
        for pc in range(PATCHES_PER_SIDE):
            px, py = centers[pr, pc]
            pt = Point(px, py)

            # Check each candidate polygon
            for idx in candidate_indices:
                code = gdf.iloc[idx]["code"]
                cls = LEVY_CODE_MAP.get(code, code.upper())
                if cls not in CLASS_MAP or cls == "OTHER":
                    continue

                # Use prepared geometry for fast contains check
                if prepared_geoms[idx].contains(pt):
                    labels[pr, pc] = CLASS_MAP[cls]
                    break  # First match wins (polygons shouldn't overlap much)

            # If no polygon contains this point and we're in ambiguous zone → UNLABELED
            # (already default)

    return labels


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Load Levy shapefile
    logger.info("=" * 60)
    logger.info("Step 1: Loading Levy 2014 shapefile...")
    gdf = gpd.read_file(str(SHAPEFILE))
    logger.info(f"  {len(gdf)} polygons: {dict(gdf['code'].value_counts())}")
    tree = STRtree(gdf.geometry)

    # Prepare geometries for fast contains() queries
    logger.info("  Preparing geometries for fast point-in-polygon...")
    prepared_geoms = [prep(geom) for geom in gdf.geometry]
    logger.info("  Done")

    # Step 2: Load tile metadata
    logger.info("=" * 60)
    logger.info("Step 2: Loading tile labels...")
    with open(TILE_LABELS) as f:
        tiles = json.load(f)

    valid_tiles = [t for t in tiles if t.get("lat") is not None]
    logger.info(f"  {len(valid_tiles)} tiles with valid lat/lon")

    # Count by type
    type_counts = Counter(t.get("label_type", "?") for t in valid_tiles)
    logger.info(f"  Types: {dict(type_counts)}")
    logger.info(f"  negative (fast path): {type_counts.get('negative', 0)}")
    logger.info(f"  Need polygon check: {len(valid_tiles) - type_counts.get('negative', 0)}")

    # Step 3: Compute patch labels
    logger.info("=" * 60)
    logger.info("Step 3: Computing patch-level labels (point-in-polygon)...")
    logger.info(f"  Patch grid: {PATCHES_PER_SIDE}×{PATCHES_PER_SIDE} per tile")
    logger.info(f"  Patch size: {PATCH_SIZE_M:.0f}m")

    patch_labels: dict[str, dict[str, np.ndarray]] = {}
    t0 = time.time()
    total_patches = 0
    class_counts = Counter()

    for i, tile in enumerate(valid_tiles):
        pid = tile["image_id"]
        key = f"{tile['tile_row']}_{tile['tile_col']}"
        label_type = tile.get("label_type", "ambiguous")

        labels_16x16 = classify_patches_for_tile(
            tile_lat=tile["lat"],
            tile_lon=tile["lon"],
            gdf=gdf,
            tree=tree,
            tile_label_type=label_type,
            prepared_geoms=prepared_geoms,
        )

        if pid not in patch_labels:
            patch_labels[pid] = {}
        patch_labels[pid][key] = labels_16x16

        # Count
        unique, counts = np.unique(labels_16x16, return_counts=True)
        for val, cnt in zip(unique, counts):
            if val == UNLABELED:
                class_counts["UNLABELED"] += cnt
            else:
                class_counts[CLASS_NAMES[val]] += cnt
        total_patches += labels_16x16.size

        if (i + 1) % 1000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(valid_tiles) - i - 1) / rate
            logger.info(
                f"  {i + 1}/{len(valid_tiles)} tiles "
                f"({elapsed:.0f}s elapsed, ETA {eta:.0f}s, {rate:.1f} tiles/s)"
            )

    elapsed = time.time() - t0
    logger.info(f"  Done in {elapsed:.1f}s ({len(valid_tiles) / elapsed:.1f} tiles/s)")

    # Step 4: Stats
    logger.info("=" * 60)
    logger.info("Step 4: Statistics")
    logger.info(f"  Total patches: {total_patches:,}")
    logger.info(f"  Products: {len(patch_labels):,}")
    for cls, cnt in sorted(class_counts.items()):
        pct = 100 * cnt / total_patches
        logger.info(f"    {cls}: {cnt:,} ({pct:.1f}%)")

    trainable = total_patches - class_counts.get("UNLABELED", 0)
    logger.info(f"  Trainable patches: {trainable:,} ({100 * trainable / total_patches:.1f}%)")

    # Step 5: Save
    logger.info("=" * 60)
    logger.info("Step 5: Saving...")
    out_path = OUTPUT_DIR / "patch_labels_v8.npy"
    np.save(out_path, patch_labels, allow_pickle=True)
    logger.info(f"  Saved: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    stats = {
        "total_tiles": len(valid_tiles),
        "total_patches": total_patches,
        "trainable_patches": trainable,
        "patches_per_tile": PATCHES_PER_SIDE ** 2,
        "patch_size_m": PATCH_SIZE_M,
        "tile_size_m": TILE_SIZE_M,
        "class_distribution": {k: int(v) for k, v in class_counts.items()},
        "class_names": CLASS_NAMES,
        "class_map": CLASS_MAP,
        "unlabeled_value": UNLABELED,
    }
    stats_path = OUTPUT_DIR / "patch_label_stats_v8.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"  Saved: {stats_path}")

    logger.info("=" * 60)
    logger.info("DONE")


if __name__ == "__main__":
    main()
