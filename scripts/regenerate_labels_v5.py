#!/usr/bin/env python3
"""
V5b Label Regeneration — Recompute tile labels using PDS extent-based coordinates.

The V3/V4 labels used pixel_scale_m=25.0 to compute tile lat/lon, which was ~6.6x
too large. This script uses actual PDS LBL extents to compute correct tile positions,
then recomputes Levy 2014 polygon overlap to generate accurate labels.

Usage:
  nohup python3 regenerate_labels_v5.py > /tmp/regen_labels_v5.log 2>&1 &
"""
from __future__ import annotations

import json
import logging
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
from PIL import Image
from shapely.geometry import Point
from shapely.geometry import box as shapely_box
from shapely.strtree import STRtree

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("/tmp/regen_labels_v5.log")],
)
logger = logging.getLogger(__name__)

# ── Paths ──
ROOT = Path("/disk1/cspark/MarsLab")
SHAPEFILE = ROOT / "Data/external_datasets/levy_2014_glacial/extracted/AreaIceUnits.shp"
BROWSE_DIR = ROOT / "Data/HiRISE/midlat_browse"
V4_DIR = ROOT / "Data/HiRISE/v4_colab_data_expanded"
V5_DIR = ROOT / "Data/HiRISE/v5_retrain"

# ── Constants ──
TILE_PX = 224
MARS_RADIUS_M = 3396190.0
DEG_TO_RAD = math.pi / 180.0

# Label thresholds (same as V3/V4)
CONFIDENT_THRESHOLD = 0.6
MIXED_THRESHOLD = 0.2
OTHER_DISTANCE_M = 10_000  # >10 km from any polygon → OTHER

CLASSES = ["LDA", "LVF", "CCF", "OTHER"]
LEVY_CODE_MAP = {"ccf": "CCF", "lda": "LDA", "lvf": "LVF"}


def latlon_to_equirect(lat_deg: float, lon_deg: float) -> tuple[float, float]:
    """Convert Mars lat/lon (degrees) → equirectangular meters."""
    x = lon_deg * DEG_TO_RAD * MARS_RADIUS_M
    y = lat_deg * DEG_TO_RAD * MARS_RADIUS_M
    return x, y


def compute_tile_latlon(
    lat_min: float, lat_max: float, lon_min: float, lon_max: float,
    img_w: int, img_h: int, tile_row: int, tile_col: int,
) -> tuple[float, float]:
    """Compute tile center lat/lon from PDS extent + browse image dimensions."""
    tile_center_row = tile_row * TILE_PX + TILE_PX / 2
    tile_center_col = tile_col * TILE_PX + TILE_PX / 2
    # Row 0 = top = lat_max (north)
    lat = lat_max - (tile_center_row / img_h) * (lat_max - lat_min)
    # Col 0 = left = lon_min (west)
    lon = lon_min + (tile_center_col / img_w) * (lon_max - lon_min)
    return float(lat), float(lon)


def tile_bbox_equirect(lat: float, lon: float, lat_span_per_px: float, lon_span_per_px: float) -> tuple[float, float, float, float]:
    """
    Compute tile bounding box in equirectangular CRS.
    Uses actual tile size derived from PDS extent, not hardcoded 25 m/px.
    """
    half_lat = TILE_PX * lat_span_per_px / 2
    half_lon = TILE_PX * lon_span_per_px / 2

    lat_min = lat - half_lat
    lat_max = lat + half_lat
    lon_min = lon - half_lon
    lon_max = lon + half_lon

    xmin, ymin = latlon_to_equirect(lat_min, lon_min)
    xmax, ymax = latlon_to_equirect(lat_max, lon_max)
    return xmin, ymin, xmax, ymax


def get_browse_dimensions(image_id: str) -> tuple[int, int] | None:
    """Get browse image (width, height) for a HiRISE image."""
    patterns = [
        f"{image_id}_RED.abrowse.jpg",
        f"{image_id}.jpg",
        f"{image_id}_RED.browse.jpg",
    ]
    for name in patterns:
        p = BROWSE_DIR / name
        if p.exists():
            try:
                img = Image.open(p)
                return img.size  # (width, height)
            except Exception:
                continue
    return None


def compute_tile_labels_v5(
    tiles: list[dict],
    gdf: gpd.GeoDataFrame,
    tree: STRtree,
) -> list[dict[str, Any]]:
    """Compute per-tile labels from Levy polygon overlap with corrected coordinates."""
    results: list[dict[str, Any]] = []
    n_total = len(tiles)

    for i, tile in enumerate(tiles):
        lat = tile["lat"]
        lon = tile["lon"]
        lat_span = tile["lat_span_per_px"]
        lon_span = tile["lon_span_per_px"]

        # Build tile bounding box in equirectangular CRS
        xmin, ymin, xmax, ymax = tile_bbox_equirect(lat, lon, lat_span, lon_span)
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

        # If no candidates, estimate distance to nearest polygon
        if len(candidates) == 0:
            pt = Point(latlon_to_equirect(lat, lon))
            buffered = pt.buffer(OTHER_DISTANCE_M * 1.5)
            nearby = tree.query(buffered)
            if len(nearby) > 0:
                for idx in nearby:
                    d = gdf.geometry.iloc[idx].distance(pt)
                    if d < min_distance:
                        min_distance = d
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
            "image_id": tile["image_id"],
            "tile_idx": tile["tile_idx"],
            "tile_row": tile["tile_row"],
            "tile_col": tile["tile_col"],
            "lat": round(lat, 6),
            "lon": round(lon, 6),
            "label": label,
            "label_type": label_type,
            "coverage": {k: round(v, 4) for k, v in coverage.items()},
            "max_coverage": round(max_cov, 4),
            "distance_to_polygon_m": round(min_distance, 1) if min_distance < float("inf") else 20000.0,
        })

        if (i + 1) % 5000 == 0:
            logger.info(f"  Processed {i + 1}/{n_total} tiles")

    return results


def create_spatial_splits(
    tile_labels: list[dict[str, Any]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
    radius_km: float = 20.0,
) -> dict[str, list[int]]:
    """Spatial split at IMAGE level (identical logic to V3/V4)."""
    rng = random.Random(seed)

    image_tiles: dict[str, list[int]] = {}
    image_coords: dict[str, tuple[float, float]] = {}

    for idx, t in enumerate(tile_labels):
        if t["label"] == "UNLABELED":
            continue
        img = t["image_id"]
        if img not in image_tiles:
            image_tiles[img] = []
        image_tiles[img].append(idx)
        if t.get("lat") and img not in image_coords:
            image_coords[img] = (t["lat"], t["lon"])

    MARS_R = 3389.5

    def haversine(lat1, lon1, lat2, lon2):
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        return 2 * MARS_R * math.asin(math.sqrt(min(a, 1.0)))

    images = list(image_tiles.keys())
    rng.shuffle(images)

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

    total_tiles = sum(len(image_tiles[img]) for c in clusters for img in c)
    target_train = int(total_tiles * train_ratio)
    target_val = int(total_tiles * val_ratio)

    splits: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    counts = {"train": 0, "val": 0, "test": 0}

    rng.shuffle(clusters)
    for cluster in clusters:
        n = sum(len(image_tiles[img]) for img in cluster)
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
    logger.info("=" * 60)
    logger.info("V5b Label Regeneration — PDS extent-based coordinates")
    logger.info("=" * 60)

    # 1. Load tile index
    logger.info("Step 1: Loading tile index...")
    with open(V4_DIR / "tile_index.json") as f:
        tile_index = json.load(f)
    logger.info(f"  {len(tile_index)} tiles")

    # 2. Load PDS extents
    logger.info("Step 2: Loading PDS extents...")
    with open(V5_DIR / "pds_extents.json") as f:
        pds_extents = json.load(f)
    logger.info(f"  {len(pds_extents)} extents")

    # 3. Compute correct tile lat/lon from PDS extent + browse dimensions
    logger.info("Step 3: Computing corrected tile coordinates...")

    # Cache browse image dimensions per image_id
    browse_dims: dict[str, tuple[int, int]] = {}
    image_ids = sorted(set(k.rsplit("_", 2)[0] for k in tile_index))

    missing_browse = 0
    missing_extent = 0
    for img_id in image_ids:
        dims = get_browse_dimensions(img_id)
        if dims:
            browse_dims[img_id] = dims
        else:
            missing_browse += 1

    logger.info(f"  Browse dimensions loaded: {len(browse_dims)} images ({missing_browse} missing)")
    logger.info(f"  PDS extents available: {len(pds_extents)} images")

    # Build tile list with corrected coordinates
    tiles: list[dict] = []
    skipped = 0
    tile_idx_counter: dict[str, int] = {}

    for key in tile_index:
        parts = key.rsplit("_", 2)
        image_id = parts[0]
        tile_row = int(parts[1])
        tile_col = int(parts[2])

        if image_id not in pds_extents or image_id not in browse_dims:
            skipped += 1
            continue

        ext = pds_extents[image_id]
        w, h = browse_dims[image_id]

        lat, lon = compute_tile_latlon(
            ext["lat_min"], ext["lat_max"], ext["lon_min"], ext["lon_max"],
            w, h, tile_row, tile_col,
        )

        lat_span_per_px = (ext["lat_max"] - ext["lat_min"]) / h
        lon_span_per_px = (ext["lon_max"] - ext["lon_min"]) / w

        idx = tile_idx_counter.get(image_id, 0)
        tile_idx_counter[image_id] = idx + 1

        tiles.append({
            "key": key,
            "image_id": image_id,
            "tile_idx": idx,
            "tile_row": tile_row,
            "tile_col": tile_col,
            "lat": lat,
            "lon": lon,
            "lat_span_per_px": lat_span_per_px,
            "lon_span_per_px": lon_span_per_px,
        })

    logger.info(f"  Tiles with corrected coords: {len(tiles)} ({skipped} skipped)")

    # 4. Load Levy shapefile
    logger.info("Step 4: Loading Levy 2014 shapefile...")
    gdf = gpd.read_file(str(SHAPEFILE))
    logger.info(f"  {len(gdf)} polygons: {dict(gdf['code'].value_counts())}")
    tree = STRtree(gdf.geometry)
    logger.info("  Spatial index built")

    # 5. Compute tile labels
    logger.info("Step 5: Computing tile labels with Levy polygon overlap...")
    tile_labels = compute_tile_labels_v5(tiles, gdf, tree)

    # Stats
    label_counts = Counter(t["label"] for t in tile_labels)
    type_counts = Counter(t["label_type"] for t in tile_labels)
    logger.info(f"\nLabel distribution:")
    for cls, cnt in sorted(label_counts.items()):
        logger.info(f"  {cls}: {cnt} ({100 * cnt / len(tile_labels):.1f}%)")
    logger.info(f"\nLabel type distribution:")
    for lt, cnt in sorted(type_counts.items()):
        logger.info(f"  {lt}: {cnt}")

    # 6. Spatial splits
    logger.info("\nStep 6: Creating spatial splits...")
    splits = create_spatial_splits(tile_labels, seed=42)
    for split_name, indices in splits.items():
        split_labels = Counter(tile_labels[i]["label"] for i in indices)
        logger.info(f"  {split_name}: {len(indices)} tiles — {dict(split_labels)}")

    # 7. Save
    logger.info("\nStep 7: Saving outputs...")
    V5_DIR.mkdir(parents=True, exist_ok=True)

    with open(V5_DIR / "tile_labels_v5.json", "w") as f:
        json.dump(tile_labels, f, indent=1, default=str)
    logger.info(f"  Saved tile_labels_v5.json ({len(tile_labels)} tiles)")

    with open(V5_DIR / "tile_splits_v5.json", "w") as f:
        json.dump(splits, f, indent=2)
    logger.info(f"  Saved tile_splits_v5.json")

    stats = {
        "total_tiles": len(tile_labels),
        "total_images": len(image_ids),
        "label_distribution": dict(label_counts),
        "label_type_distribution": dict(type_counts),
        "thresholds": {
            "confident": CONFIDENT_THRESHOLD,
            "mixed": MIXED_THRESHOLD,
            "other_distance_m": OTHER_DISTANCE_M,
        },
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "coordinate_source": "PDS LBL extent (corrected from pixel_scale_m=25.0)",
    }
    with open(V5_DIR / "label_stats_v5.json", "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"  Saved label_stats_v5.json")

    logger.info("=" * 60)
    logger.info("DONE — Labels regenerated with PDS extent-based coordinates")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
