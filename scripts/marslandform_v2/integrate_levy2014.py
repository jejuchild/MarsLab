#!/usr/bin/env python3
"""
Integrate Levy 2014 LDA/LVF/CCF polygon shapefiles into HiRISE labels.

Levy et al. (2014) "Sequestered glacial ice contribution to the global Martian
water budget" — expert-labeled polygons for CCF (5115), LDA (1018), LVF (252).

This script:
1. Loads the Levy 2014 AreaIceUnits shapefile (Mars equirectangular CRS)
2. Converts each HiRISE image center (lat/lon) → shapefile CRS
3. Checks point-in-polygon for each HiRISE image
4. Updates labels_simple.json with Levy labels (highest priority)
5. Saves a report of changes
"""

import json
import math
import logging
from pathlib import Path
from collections import Counter

import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from shapely.strtree import STRtree

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path("/disk1/cspark/MarsLab")
SHAPEFILE = ROOT / "Data/external_datasets/levy_2014_glacial/extracted/AreaIceUnits.shp"
METADATA = ROOT / "Data/HiRISE/midlat_metadata.json"
LABELS_PATH = ROOT / "Data/HiRISE/v2_output/labels_simple.json"
UNIFIED_PATH = ROOT / "Data/HiRISE/v2_output/unified_labels.json"
OUTPUT_DIR = ROOT / "Data/HiRISE/v2_output"

# Mars equirectangular CRS parameters (from shapefile)
MARS_RADIUS_M = 3396190.0
DEG_TO_RAD = math.pi / 180.0


def latlon_to_equirectangular(lat_deg: float, lon_deg: float) -> tuple:
    """Convert Mars lat/lon (degrees) to equirectangular (meters)."""
    # Standard parallel = 0 (equator), so scale factor = 1
    x = lon_deg * DEG_TO_RAD * MARS_RADIUS_M
    y = lat_deg * DEG_TO_RAD * MARS_RADIUS_M
    return x, y


def main():
    logger.info("=" * 60)
    logger.info("Integrating Levy 2014 LDA/LVF/CCF polygons into labels")
    logger.info("=" * 60)

    # Load shapefile
    logger.info(f"Loading shapefile: {SHAPEFILE}")
    gdf = gpd.read_file(SHAPEFILE)
    logger.info(f"  Loaded {len(gdf)} polygons")
    logger.info(f"  Classes: {dict(gdf['code'].value_counts())}")

    # Build spatial index
    logger.info("Building spatial index...")
    tree = STRtree(gdf.geometry)

    # Load HiRISE metadata for coordinates
    logger.info(f"Loading metadata: {METADATA}")
    with open(METADATA) as f:
        metadata = json.load(f)
    meta_by_id = {m["image_id"]: m for m in metadata}
    logger.info(f"  {len(meta_by_id)} images in metadata")

    # Load current labels
    logger.info(f"Loading labels: {LABELS_PATH}")
    with open(LABELS_PATH) as f:
        labels = json.load(f)
    logger.info(f"  {len(labels)} labeled images")
    logger.info(f"  Current distribution: {dict(Counter(labels.values()))}")

    # Also load unified labels for updating source info
    unified = {}
    if UNIFIED_PATH.exists():
        with open(UNIFIED_PATH) as f:
            unified = json.load(f)

    # Match HiRISE images to Levy polygons
    logger.info("\nSpatial matching HiRISE → Levy 2014 polygons...")
    levy_matches = {}
    levy_code_map = {"ccf": "CCF", "lda": "LDA", "lvf": "LVF"}

    # Also search a small buffer around each point (HiRISE images are ~6×12 km)
    # 6km ≈ 6000m, but we use the center point only for now
    BUFFER_M = 3000.0  # 3km buffer to account for image extent

    for img_id in labels:
        meta = meta_by_id.get(img_id, {})
        lat = meta.get("lat")
        lon = meta.get("lon")
        if lat is None or lon is None:
            continue

        x, y = latlon_to_equirectangular(float(lat), float(lon))
        point = Point(x, y)

        # Check exact point-in-polygon first
        candidates = tree.query(point)
        matched_code = None
        min_dist = float("inf")

        for idx in candidates:
            geom = gdf.geometry.iloc[idx]
            if geom.contains(point):
                matched_code = gdf.iloc[idx]["code"]
                min_dist = 0.0
                break

        # If no exact hit, try buffered point (3km radius)
        if matched_code is None:
            buffered = point.buffer(BUFFER_M)
            candidates_buf = tree.query(buffered)
            for idx in candidates_buf:
                geom = gdf.geometry.iloc[idx]
                if geom.intersects(buffered):
                    dist = geom.distance(point)
                    if dist < min_dist:
                        min_dist = dist
                        matched_code = gdf.iloc[idx]["code"]

        if matched_code is not None:
            levy_class = levy_code_map.get(matched_code, matched_code.upper())
            levy_matches[img_id] = {
                "class": levy_class,
                "code": matched_code,
                "distance_m": min_dist,
                "lat": lat,
                "lon": lon,
            }

    logger.info(f"\nLevy 2014 matches: {len(levy_matches)} images")
    match_classes = Counter(m["class"] for m in levy_matches.values())
    logger.info(f"  By class: {dict(match_classes)}")

    # Report exact vs buffered matches
    exact = sum(1 for m in levy_matches.values() if m["distance_m"] == 0.0)
    buffered = len(levy_matches) - exact
    logger.info(f"  Exact point-in-polygon: {exact}")
    logger.info(f"  Buffered (within 3km): {buffered}")

    # Update labels — Levy 2014 is highest priority (expert polygon data)
    changes = {"upgraded": [], "changed": [], "new": [], "confirmed": []}
    old_dist = Counter(labels.values())

    for img_id, match in levy_matches.items():
        old_class = labels.get(img_id)
        new_class = match["class"]

        if old_class == new_class:
            changes["confirmed"].append(img_id)
        elif old_class is None:
            changes["new"].append(img_id)
            labels[img_id] = new_class
        elif old_class == "BACKGROUND":
            changes["upgraded"].append(img_id)
            labels[img_id] = new_class
        else:
            changes["changed"].append(img_id)
            labels[img_id] = new_class

        # Update unified labels
        if img_id in unified:
            unified[img_id]["label_sources"].append({
                "source": "levy_2014_polygon",
                "class": new_class,
                "levy_code": match["code"],
                "distance_m": match["distance_m"],
                "method": "point_in_polygon" if match["distance_m"] == 0.0 else "buffered_3km",
            })
            unified[img_id]["final_class"] = new_class
            unified[img_id]["label_confidence"] = "expert"
            unified[img_id]["label_method"] = "levy_2014_polygon"

    new_dist = Counter(labels.values())

    logger.info(f"\nLabel changes:")
    logger.info(f"  Confirmed (same class): {len(changes['confirmed'])}")
    logger.info(f"  Changed (different class): {len(changes['changed'])}")
    logger.info(f"  Upgraded from BACKGROUND: {len(changes['upgraded'])}")
    logger.info(f"  New labels: {len(changes['new'])}")

    if changes["changed"]:
        logger.info(f"\n  Changed images:")
        for img_id in changes["changed"][:20]:
            old = old_dist  # Can't get per-image from Counter, log the match
            logger.info(f"    {img_id}: {levy_matches[img_id]['class']} (was: check unified)")

    logger.info(f"\nDistribution before: {dict(old_dist)}")
    logger.info(f"Distribution after:  {dict(new_dist)}")

    # Save updated labels
    with open(LABELS_PATH, "w") as f:
        json.dump(labels, f, indent=2)
    logger.info(f"\nSaved updated labels_simple.json ({len(labels)} images)")

    if unified:
        with open(UNIFIED_PATH, "w") as f:
            json.dump(unified, f, indent=2, default=str)
        logger.info(f"Saved updated unified_labels.json")

    # Save Levy match report
    report = {
        "total_matches": len(levy_matches),
        "by_class": dict(match_classes),
        "exact_matches": exact,
        "buffered_matches": buffered,
        "changes": {k: len(v) for k, v in changes.items()},
        "old_distribution": dict(old_dist),
        "new_distribution": dict(new_dist),
        "matched_images": levy_matches,
    }
    report_path = OUTPUT_DIR / "levy2014_integration_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
