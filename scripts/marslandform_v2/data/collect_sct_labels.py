#!/usr/bin/env python3
"""
Generate SCT (Scalloped Terrain) tile labels from Wang et al. 2026 GeoTIFF.

The Wang GeoTIFF is a binary segmentation mask at 5m/px resolution covering
the northern hemisphere of Mars. This script:

1. Loads HiRISE index to find images in the SCT region (38-52°N, 60-110°E)
2. For each candidate image, determines tile positions (224×224 @ ~25 m/px)
3. Reads the Wang GeoTIFF window at each tile's location
4. Computes SCT pixel fraction per tile
5. Labels tiles as SCT (coverage ≥ threshold) or OTHER (far from SCT)

Output: sct_tile_labels.json — same format as tile_labels_v3.json
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.windows import Window, from_bounds
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(os.getenv("MARSLAB_ROOT", "/disk1/cspark/MarsLab"))
HIRISE_INDEX = ROOT / "backend" / "hirise_data" / "index.geojson"
BROWSE_DIR = ROOT / "Data" / "HiRISE" / "midlat_browse"
WANG_GEOTIFF = ROOT / "Data" / "external_datasets" / "wang_2026_scalloped" / "Scalloped Terrain.tif"
OUTPUT_DIR = ROOT / "Data" / "HiRISE" / "v3_output"

# Mars constants
MARS_RADIUS_M = 3396190.0
DEG_TO_RAD = math.pi / 180.0

# Tile config
TILE_PX = 224
BROWSE_SCALE_M_PER_PX = 25.0  # HiRISE browse ~25 m/px
TILE_SIZE_M = TILE_PX * BROWSE_SCALE_M_PER_PX  # 5600 m

# Label thresholds
SCT_CONFIDENT_THRESHOLD = 0.10  # ≥10% SCT pixels → confident SCT label
SCT_MIXED_THRESHOLD = 0.03     # ≥3% → soft/mixed label
OTHER_DISTANCE_TILES = 3        # ≥3 tiles (~17km) away from any SCT tile → OTHER

# SCT geographic region (Wang et al. 2026 coverage)
SCT_LAT_MIN, SCT_LAT_MAX = 37.0, 52.0
SCT_LON_MIN, SCT_LON_MAX = 60.0, 110.0  # 0-360°E


def latlon_to_equirect(lat_deg: float, lon_deg: float) -> tuple[float, float]:
    """Convert Mars lat/lon (degrees) → equirectangular meters (Wang CRS)."""
    x = lon_deg * DEG_TO_RAD * MARS_RADIUS_M
    y = lat_deg * DEG_TO_RAD * MARS_RADIUS_M
    return x, y


def get_hirise_candidates(index_path: Path) -> list[dict]:
    """Load HiRISE index and filter to SCT region."""
    with open(index_path) as f:
        idx = json.load(f)

    candidates = []
    for feat in idx["features"]:
        props = feat["properties"]
        lat = props.get("center_lat")
        lon = props.get("center_lon")

        if lat is None or lon is None:
            geom = feat.get("geometry", {})
            if geom.get("type") == "Point":
                lon, lat = geom["coordinates"]
            elif geom.get("type") == "Polygon":
                coords = geom["coordinates"][0]
                lat = sum(c[1] for c in coords) / len(coords)
                lon = sum(c[0] for c in coords) / len(coords)
            else:
                continue

        # Normalize to 0-360
        lon_360 = lon if lon >= 0 else lon + 360

        if SCT_LAT_MIN <= lat <= SCT_LAT_MAX and SCT_LON_MIN <= lon_360 <= SCT_LON_MAX:
            pid = props.get("product_id") or props.get("id") or feat.get("id", "?")
            candidates.append({
                "product_id": pid,
                "lat": lat,
                "lon_360": lon_360,
            })

    return candidates


def get_tile_positions(image_id: str, browse_dir: Path) -> list[dict] | None:
    """
    Compute tile positions for a HiRISE browse image.
    Returns list of {tile_row, tile_col, lat, lon} or None if image not found.
    """
    # Find browse image — naming convention: {product_id}_RED.abrowse.jpg
    browse_path = None
    patterns = [
        f"{image_id}_RED.abrowse.jpg",
        f"{image_id}.jpg",
        f"{image_id}_RED.browse.jpg",
        f"{image_id}.jpeg",
        f"{image_id}.png",
    ]
    for name in patterns:
        p = browse_dir / name
        if p.exists():
            browse_path = p
            break

    if browse_path is None:
        return None

    try:
        img = Image.open(browse_path)
        w, h = img.size
    except Exception:
        return None

    n_rows = h // TILE_PX
    n_cols = w // TILE_PX

    if n_rows == 0 or n_cols == 0:
        return None

    return {"width": w, "height": h, "n_rows": n_rows, "n_cols": n_cols}


def compute_sct_coverage_for_image(
    image_id: str,
    center_lat: float,
    center_lon: float,
    img_info: dict,
    wang_src: rasterio.DatasetReader,
) -> list[dict[str, Any]]:
    """
    For each tile in a HiRISE image, compute SCT coverage from Wang GeoTIFF.
    """
    n_rows = img_info["n_rows"]
    n_cols = img_info["n_cols"]
    img_w = img_info["width"]
    img_h = img_info["height"]

    # Image center in equirectangular meters
    cx_m, cy_m = latlon_to_equirect(center_lat, center_lon)

    # Image extent in meters
    img_h_m = img_h * BROWSE_SCALE_M_PER_PX
    img_w_m = img_w * BROWSE_SCALE_M_PER_PX

    # Top-left corner of image in equirectangular meters
    x0_m = cx_m - img_w_m / 2
    y0_m = cy_m + img_h_m / 2  # top = higher latitude

    results = []

    for tr in range(n_rows):
        for tc in range(n_cols):
            # Tile center in meters
            tile_cx_m = x0_m + (tc + 0.5) * TILE_SIZE_M
            tile_cy_m = y0_m - (tr + 0.5) * TILE_SIZE_M

            # Tile bounds in equirectangular meters
            tile_xmin = tile_cx_m - TILE_SIZE_M / 2
            tile_xmax = tile_cx_m + TILE_SIZE_M / 2
            tile_ymin = tile_cy_m - TILE_SIZE_M / 2
            tile_ymax = tile_cy_m + TILE_SIZE_M / 2

            # Convert back to lat/lon for output
            tile_lat = tile_cy_m / (MARS_RADIUS_M * DEG_TO_RAD)
            tile_lon = tile_cx_m / (MARS_RADIUS_M * DEG_TO_RAD)

            # Check if tile bounds are within Wang GeoTIFF bounds
            wang_bounds = wang_src.bounds
            if (tile_xmax < wang_bounds.left or tile_xmin > wang_bounds.right or
                tile_ymax < wang_bounds.bottom or tile_ymin > wang_bounds.top):
                # Outside Wang coverage
                results.append({
                    "image_id": image_id,
                    "tile_row": tr,
                    "tile_col": tc,
                    "lat": tile_lat,
                    "lon": tile_lon,
                    "sct_coverage": 0.0,
                    "sct_pixels": 0,
                    "total_valid_pixels": 0,
                    "in_wang_coverage": False,
                })
                continue

            # Clamp to Wang bounds
            tile_xmin_c = max(tile_xmin, wang_bounds.left)
            tile_xmax_c = min(tile_xmax, wang_bounds.right)
            tile_ymin_c = max(tile_ymin, wang_bounds.bottom)
            tile_ymax_c = min(tile_ymax, wang_bounds.top)

            try:
                # Get window in Wang GeoTIFF
                win = from_bounds(
                    tile_xmin_c, tile_ymin_c, tile_xmax_c, tile_ymax_c,
                    wang_src.transform
                )

                # Ensure valid window dimensions
                if win.width < 1 or win.height < 1:
                    results.append({
                        "image_id": image_id,
                        "tile_row": tr,
                        "tile_col": tc,
                        "lat": tile_lat,
                        "lon": tile_lon,
                        "sct_coverage": 0.0,
                        "sct_pixels": 0,
                        "total_valid_pixels": 0,
                        "in_wang_coverage": True,
                    })
                    continue

                # Read Wang raster at this window
                data = wang_src.read(1, window=win)

                # Count SCT and valid pixels
                sct_pixels = int(np.count_nonzero(data == 1))
                nodata_val = wang_src.nodata or 16
                valid_pixels = int(np.count_nonzero(data != nodata_val))

                sct_coverage = sct_pixels / max(1, valid_pixels)

                results.append({
                    "image_id": image_id,
                    "tile_row": tr,
                    "tile_col": tc,
                    "lat": tile_lat,
                    "lon": tile_lon,
                    "sct_coverage": round(sct_coverage, 4),
                    "sct_pixels": sct_pixels,
                    "total_valid_pixels": valid_pixels,
                    "in_wang_coverage": True,
                })

            except Exception as e:
                logger.debug(f"Window read error for {image_id} tile ({tr},{tc}): {e}")
                results.append({
                    "image_id": image_id,
                    "tile_row": tr,
                    "tile_col": tc,
                    "lat": tile_lat,
                    "lon": tile_lon,
                    "sct_coverage": 0.0,
                    "sct_pixels": 0,
                    "total_valid_pixels": 0,
                    "in_wang_coverage": False,
                })

    return results


def assign_labels(tile_data: list[dict], conf_thresh: float = SCT_CONFIDENT_THRESHOLD, mix_thresh: float = SCT_MIXED_THRESHOLD) -> list[dict[str, Any]]:
    """
    Assign SCT / OTHER / UNLABELED labels based on coverage thresholds.
    """
    # Group by image to compute distance-based OTHER labels
    by_image: dict[str, list[dict]] = {}
    for td in tile_data:
        by_image.setdefault(td["image_id"], []).append(td)

    labeled_tiles = []
    tile_idx_counter: dict[str, int] = {}

    for image_id, tiles in by_image.items():
        # Find tiles with SCT
        sct_tiles = set()
        for t in tiles:
            if t["sct_coverage"] >= mix_thresh:
                sct_tiles.add((t["tile_row"], t["tile_col"]))

        for t in tiles:
            idx = tile_idx_counter.get(image_id, 0)
            tile_idx_counter[image_id] = idx + 1

            cov = t["sct_coverage"]

            if cov >= conf_thresh:
                label = "SCT"
                label_type = "confident"
            elif cov >= mix_thresh:
                label = "SCT"
                label_type = "mixed"
            else:
                # Check distance to nearest SCT tile
                tr, tc = t["tile_row"], t["tile_col"]
                min_dist = float("inf")
                for sr, sc in sct_tiles:
                    dist = max(abs(tr - sr), abs(tc - sc))  # Chebyshev distance in tiles
                    min_dist = min(min_dist, dist)

                if min_dist >= OTHER_DISTANCE_TILES or len(sct_tiles) == 0:
                    label = "OTHER"
                    label_type = "negative"
                else:
                    label = "UNLABELED"
                    label_type = "near_sct"

            # Distance to nearest polygon in meters (approximate)
            if len(sct_tiles) == 0:
                dist_m = 20000.0
            else:
                tr, tc = t["tile_row"], t["tile_col"]
                min_d = min(
                    math.sqrt((tr - sr) ** 2 + (tc - sc) ** 2) * TILE_SIZE_M
                    for sr, sc in sct_tiles
                )
                dist_m = round(min_d, 1)

            labeled_tiles.append({
                "image_id": t["image_id"],
                "tile_idx": idx,
                "tile_row": t["tile_row"],
                "tile_col": t["tile_col"],
                "lat": t["lat"],
                "lon": t["lon"],
                "label": label,
                "label_type": label_type,
                "coverage": {"SCT": t["sct_coverage"]},
                "max_coverage": t["sct_coverage"],
                "distance_to_polygon_m": dist_m,
            })

    return labeled_tiles


def main():
    parser = argparse.ArgumentParser(description="Generate SCT tile labels from Wang 2026 GeoTIFF")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "sct_tile_labels.json")
    parser.add_argument("--wang-tif", type=Path, default=WANG_GEOTIFF)
    parser.add_argument("--hirise-index", type=Path, default=HIRISE_INDEX)
    parser.add_argument("--browse-dir", type=Path, default=BROWSE_DIR)
    parser.add_argument("--confident-threshold", type=float, default=SCT_CONFIDENT_THRESHOLD)
    parser.add_argument("--mixed-threshold", type=float, default=SCT_MIXED_THRESHOLD)
    args = parser.parse_args()

    conf_thresh = args.confident_threshold
    mix_thresh = args.mixed_threshold

    # 1. Find HiRISE candidates in SCT region
    logger.info("Loading HiRISE index...")
    candidates = get_hirise_candidates(args.hirise_index)
    logger.info(f"Found {len(candidates)} HiRISE images in SCT region")

    # 2. Open Wang GeoTIFF
    logger.info(f"Opening Wang GeoTIFF: {args.wang_tif}")
    wang_src = rasterio.open(str(args.wang_tif))

    # 3. Process each candidate image
    all_tile_data = []
    processed = 0
    skipped = 0

    for i, cand in enumerate(candidates):
        pid = cand["product_id"]

        # Get tile positions
        img_info = get_tile_positions(pid, args.browse_dir)
        if img_info is None:
            skipped += 1
            continue

        # Compute SCT coverage per tile
        tile_data = compute_sct_coverage_for_image(
            pid, cand["lat"], cand["lon_360"], img_info, wang_src
        )
        all_tile_data.extend(tile_data)
        processed += 1

        if (i + 1) % 50 == 0:
            sct_so_far = sum(1 for t in all_tile_data if t["sct_coverage"] >= SCT_CONFIDENT_THRESHOLD)
            logger.info(
                f"  [{i+1}/{len(candidates)}] {processed} processed, {skipped} skipped, "
                f"{len(all_tile_data)} tiles, {sct_so_far} SCT"
            )

    wang_src.close()

    logger.info(f"\nProcessed {processed} images ({skipped} skipped)")
    logger.info(f"Total tiles computed: {len(all_tile_data)}")

    # Stats on SCT coverage
    coverages = [t["sct_coverage"] for t in all_tile_data if t["in_wang_coverage"]]
    nonzero = [c for c in coverages if c > 0]
    logger.info(f"Tiles with any SCT: {len(nonzero)} / {len(coverages)}")
    if nonzero:
        logger.info(f"  SCT coverage: mean={np.mean(nonzero):.4f}, median={np.median(nonzero):.4f}, max={max(nonzero):.4f}")

    # 4. Assign labels
    logger.info("Assigning labels...")
    labeled_tiles = assign_labels(all_tile_data, conf_thresh=conf_thresh, mix_thresh=mix_thresh)

    # Stats
    label_counts = Counter(t["label"] for t in labeled_tiles)
    logger.info(f"\nLabel distribution:")
    for label, count in sorted(label_counts.items()):
        logger.info(f"  {label}: {count}")

    # 5. Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(labeled_tiles, f, indent=1)
    logger.info(f"Saved {len(labeled_tiles)} tile labels to {args.output}")

    # Save stats
    stats = {
        "total_candidates": len(candidates),
        "processed": processed,
        "skipped": skipped,
        "total_tiles": len(labeled_tiles),
        "label_distribution": dict(label_counts),
        "sct_confident_threshold": conf_thresh,
        "sct_mixed_threshold": mix_thresh,
        "wang_geotiff": str(args.wang_tif),
        "nonzero_sct_tiles": len(nonzero),
        "mean_sct_coverage": float(np.mean(nonzero)) if nonzero else 0.0,
    }
    stats_path = args.output.with_suffix(".stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Saved stats to {stats_path}")


if __name__ == "__main__":
    main()
