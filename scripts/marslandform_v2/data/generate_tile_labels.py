#!/usr/bin/env python3
"""
Generate tile-level labels from Levy 2014 expert polygons.

For each HiRISE tile (224×224 @ ~25 m/px ≈ 5.6 km), computes polygon overlap
fraction per class and assigns:
  - CONFIDENT: max coverage ≥ 0.6 → hard label
  - MIXED:     max coverage 0.2–0.6 → soft label (coverage fractions)
  - OTHER:     max coverage < 0.2 AND >10 km from any polygon → negative
  - UNLABELED: max coverage < 0.2 AND ≤10 km → masked from loss

Output: tile_labels_v3.json
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from collections import Counter
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box as shapely_box
from shapely.geometry import Point
from shapely.strtree import STRtree

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path("/disk1/cspark/MarsLab")
SHAPEFILE = ROOT / "Data/external_datasets/levy_2014_glacial/extracted/AreaIceUnits.shp"
TILE_METADATA = ROOT / "Data/HiRISE/v2_output/tile_metadata.csv"
METADATA_JSON = ROOT / "Data/HiRISE/midlat_metadata.json"
OUTPUT_DIR = ROOT / "Data/HiRISE/v3_output"

# Mars equirectangular CRS (from Levy shapefile)
MARS_RADIUS_M = 3396190.0
DEG_TO_RAD = math.pi / 180.0

# Tile physical size
TILE_PX = 224
BROWSE_SCALE_M_PER_PX = 25.0  # HiRISE browse ~25 m/px
TILE_SIZE_M = TILE_PX * BROWSE_SCALE_M_PER_PX  # 5600 m

# Label thresholds
CONFIDENT_THRESHOLD = 0.6   # coverage ≥ this → hard label
MIXED_THRESHOLD = 0.2       # coverage ≥ this but < confident → soft label
OTHER_DISTANCE_M = 10_000   # >10 km from any polygon → OTHER

V3_CLASSES = ["LDA", "LVF", "CCF", "OTHER"]
LEVY_CODE_MAP = {"ccf": "CCF", "lda": "LDA", "lvf": "LVF"}


def latlon_to_equirect(lat_deg: float, lon_deg: float) -> tuple[float, float]:
    """Convert Mars lat/lon (degrees) → equirectangular meters."""
    x = lon_deg * DEG_TO_RAD * MARS_RADIUS_M
    y = lat_deg * DEG_TO_RAD * MARS_RADIUS_M
    return x, y


def tile_bbox_equirect(
    lat: float, lon: float, half_size_m: float = TILE_SIZE_M / 2
) -> tuple[float, float, float, float]:
    """Compute tile bounding box in equirectangular CRS (xmin, ymin, xmax, ymax)."""
    cx, cy = latlon_to_equirect(lat, lon)
    return cx - half_size_m, cy - half_size_m, cx + half_size_m, cy + half_size_m


def compute_tile_labels(
    tile_df: pd.DataFrame,
    gdf: gpd.GeoDataFrame,
    tree: STRtree,
) -> list[dict[str, Any]]:
    """Compute per-tile labels from Levy polygon overlap."""
    results: list[dict[str, Any]] = []
    n_total = len(tile_df)

    for i, row in enumerate(tile_df.itertuples()):
        lat = getattr(row, "lat", None)
        lon = getattr(row, "lon", None)
        image_id = getattr(row, "image_id", "")
        tile_idx = getattr(row, "tile_idx", 0)
        tile_row = getattr(row, "tile_row", 0)
        tile_col = getattr(row, "tile_col", 0)

        if lat is None or lon is None or pd.isna(lat) or pd.isna(lon):
            results.append({
                "image_id": image_id,
                "tile_idx": int(tile_idx),
                "tile_row": int(tile_row),
                "tile_col": int(tile_col),
                "lat": None, "lon": None,
                "label": "UNLABELED",
                "label_type": "no_coords",
                "coverage": {},
                "max_coverage": 0.0,
            })
            continue

        lat_f = float(lat)
        lon_f = float(lon)

        # Build tile bounding box in equirectangular CRS
        xmin, ymin, xmax, ymax = tile_bbox_equirect(lat_f, lon_f)
        tile_geom = shapely_box(xmin, ymin, xmax, ymax)
        tile_area = tile_geom.area

        # Query spatial index
        candidates = tree.query(tile_geom)

        # Compute per-class coverage fraction
        coverage: dict[str, float] = {"CCF": 0.0, "LDA": 0.0, "LVF": 0.0}
        min_distance = float("inf")

        for idx in candidates:
            poly = gdf.geometry.iloc[idx]
            code = gdf.iloc[idx]["code"]
            cls = LEVY_CODE_MAP.get(code, code.upper())
            if cls not in coverage:
                continue

            intersection = tile_geom.intersection(poly)
            if not intersection.is_empty:
                frac = intersection.area / tile_area
                coverage[cls] += frac

            dist = tile_geom.distance(poly)
            if dist < min_distance:
                min_distance = dist

        # If no candidates found, compute distance to nearest polygon
        if len(candidates) == 0:
            pt = Point(*latlon_to_equirect(lat_f, lon_f))
            # Use a large buffer query for distance estimation
            buffered = pt.buffer(OTHER_DISTANCE_M * 1.5)
            nearby = tree.query(buffered)
            if len(nearby) > 0:
                for idx in nearby:
                    d = gdf.geometry.iloc[idx].distance(pt)
                    if d < min_distance:
                        min_distance = d
            # If still no nearby, it's very far
            if min_distance == float("inf"):
                min_distance = OTHER_DISTANCE_M * 2

        # Clamp coverages
        for cls in coverage:
            coverage[cls] = min(1.0, coverage[cls])

        max_cov = max(coverage.values())
        max_cls = max(coverage, key=lambda c: coverage[c]) if max_cov > 0 else None

        # Assign label
        if max_cov >= CONFIDENT_THRESHOLD:
            label = max_cls
            label_type = "confident"
        elif max_cov >= MIXED_THRESHOLD:
            label = max_cls
            label_type = "mixed"
        elif min_distance > OTHER_DISTANCE_M:
            label = "OTHER"
            label_type = "negative"
        else:
            label = "UNLABELED"
            label_type = "ambiguous"

        results.append({
            "image_id": image_id,
            "tile_idx": int(tile_idx),
            "tile_row": int(tile_row),
            "tile_col": int(tile_col),
            "lat": lat_f,
            "lon": lon_f,
            "label": label,
            "label_type": label_type,
            "coverage": {k: round(v, 4) for k, v in coverage.items()},
            "max_coverage": round(max_cov, 4),
            "distance_to_polygon_m": round(min_distance, 1) if min_distance < float("inf") else None,
        })

        if (i + 1) % 10000 == 0:
            logger.info(f"  Processed {i + 1}/{n_total} tiles")

    return results


def create_spatial_splits(
    tile_labels: list[dict[str, Any]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
    radius_km: float = 20.0,
) -> dict[str, list[int]]:
    """
    Spatial split at IMAGE level (all tiles from same image in same split),
    then return tile indices per split.
    """
    import random
    rng = random.Random(seed)

    # Group tiles by image
    image_tiles: dict[str, list[int]] = {}
    image_coords: dict[str, tuple[float, float]] = {}
    image_classes: dict[str, list[str]] = {}

    for idx, t in enumerate(tile_labels):
        if t["label"] in ("UNLABELED",):
            continue
        img = t["image_id"]
        if img not in image_tiles:
            image_tiles[img] = []
            image_classes[img] = []
        image_tiles[img].append(idx)
        image_classes[img].append(t["label"])
        if t.get("lat") and img not in image_coords:
            image_coords[img] = (t["lat"], t["lon"])

    # Spatial clustering (same as V2)
    MARS_R = 3389.5
    def haversine(lat1, lon1, lat2, lon2):
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
        return 2 * MARS_R * math.asin(math.sqrt(min(a, 1.0)))

    images = list(image_tiles.keys())
    rng.shuffle(images)

    # Cluster nearby images
    clusters: list[list[str]] = []
    assigned: set[str] = set()
    for img in images:
        if img in assigned:
            continue
        cluster = [img]
        assigned.add(img)
        if img in image_coords:
            lat1, lon1 = image_coords[img]
            for other in images:
                if other in assigned or other not in image_coords:
                    continue
                lat2, lon2 = image_coords[other]
                if haversine(lat1, lon1, lat2, lon2) < radius_km:
                    cluster.append(other)
                    assigned.add(other)
        clusters.append(cluster)

    # Assign clusters to splits
    total_tiles = sum(len(image_tiles[img]) for c in clusters for img in c)
    target_train = int(total_tiles * train_ratio)
    target_val = int(total_tiles * val_ratio)

    splits: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    counts = {"train": 0, "val": 0, "test": 0}

    rng.shuffle(clusters)
    for cluster in clusters:
        n = sum(len(image_tiles[img]) for img in cluster)
        # Pick split with most remaining capacity
        if counts["train"] < target_train:
            split = "train"
        elif counts["val"] < target_val:
            split = "val"
        else:
            split = "test"
        for img in cluster:
            splits[split].extend(image_tiles[img])
        counts[split] += n

    return splits


def main():
    parser = argparse.ArgumentParser(description="Generate V3 tile-level labels from Levy 2014")
    parser.add_argument("--shapefile", default=str(SHAPEFILE))
    parser.add_argument("--tile-metadata", default=str(TILE_METADATA))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load Levy shapefile
    logger.info("=" * 60)
    logger.info("Step 1: Loading Levy 2014 shapefile...")
    gdf = gpd.read_file(args.shapefile)
    logger.info(f"  {len(gdf)} polygons: {dict(gdf['code'].value_counts())}")

    tree = STRtree(gdf.geometry)
    logger.info("  Spatial index built")

    # Step 2: Load tile metadata
    logger.info("=" * 60)
    logger.info("Step 2: Loading tile metadata...")
    tile_df = pd.read_csv(args.tile_metadata)
    logger.info(f"  {len(tile_df)} tiles from {tile_df['image_id'].nunique()} images")

    # Step 3: Compute tile labels
    logger.info("=" * 60)
    logger.info("Step 3: Computing tile-level labels...")
    tile_labels = compute_tile_labels(tile_df, gdf, tree)

    # Stats
    label_counts = Counter(t["label"] for t in tile_labels)
    type_counts = Counter(t["label_type"] for t in tile_labels)
    logger.info(f"\nLabel distribution:")
    for cls, cnt in sorted(label_counts.items()):
        logger.info(f"  {cls}: {cnt} ({100*cnt/len(tile_labels):.1f}%)")
    logger.info(f"\nLabel type distribution:")
    for lt, cnt in sorted(type_counts.items()):
        logger.info(f"  {lt}: {cnt}")

    # Step 4: Spatial splits
    logger.info("=" * 60)
    logger.info("Step 4: Creating spatial splits...")
    splits = create_spatial_splits(tile_labels, seed=args.seed)
    for split_name, indices in splits.items():
        split_labels = Counter(tile_labels[i]["label"] for i in indices)
        logger.info(f"  {split_name}: {len(indices)} tiles — {dict(split_labels)}")

    # Step 5: Save
    logger.info("=" * 60)
    logger.info("Step 5: Saving outputs...")

    with open(output_dir / "tile_labels_v3.json", "w") as f:
        json.dump(tile_labels, f, indent=2, default=str)
    logger.info(f"  Saved tile_labels_v3.json ({len(tile_labels)} tiles)")

    with open(output_dir / "tile_splits_v3.json", "w") as f:
        json.dump(splits, f, indent=2)
    logger.info(f"  Saved tile_splits_v3.json")

    stats = {
        "total_tiles": len(tile_labels),
        "total_images": tile_df["image_id"].nunique(),
        "classes": V3_CLASSES,
        "label_distribution": dict(label_counts),
        "label_type_distribution": dict(type_counts),
        "thresholds": {
            "confident": CONFIDENT_THRESHOLD,
            "mixed": MIXED_THRESHOLD,
            "other_distance_m": OTHER_DISTANCE_M,
        },
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "split_class_distribution": {
            split: dict(Counter(tile_labels[i]["label"] for i in indices))
            for split, indices in splits.items()
        },
    }
    with open(output_dir / "tile_label_stats_v3.json", "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"  Saved tile_label_stats_v3.json")

    logger.info("=" * 60)
    logger.info("DONE")


if __name__ == "__main__":
    main()
