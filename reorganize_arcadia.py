#!/usr/bin/env python3
"""
Reorganize arcadia_browse files and update index.geojson.

- VNIR quickview images → backend/crism_quickview/
- HYD, ICE, IC2 browse images → backend/crism_browse/
- LBL files → backend/crism_data/
- Updates backend/crism_data/index.geojson with new entries including coordinates
"""

import os
import json
import shutil
import re
from pathlib import Path
from collections import defaultdict

# Paths
PROJECT_ROOT = Path(__file__).parent
ARCADIA_BROWSE = PROJECT_ROOT / "arcadia_browse" / "browse"
ARCADIA_LBL = PROJECT_ROOT / "arcadia_browse" / "lbl"
CRISM_QUICKVIEW = PROJECT_ROOT / "backend" / "crism_quickview"
CRISM_BROWSE = PROJECT_ROOT / "backend" / "crism_browse"
CRISM_DATA = PROJECT_ROOT / "backend" / "crism_data"
INDEX_PATH = CRISM_DATA / "index.geojson"

# File type classification
VNIR_SUFFIX = "_VNIR.png"
BROWSE_SUFFIXES = ["_HYD.png", "_ICE.png", "_IC2.png"]


def ensure_dirs():
    """Create destination directories if they don't exist."""
    CRISM_QUICKVIEW.mkdir(parents=True, exist_ok=True)
    CRISM_BROWSE.mkdir(parents=True, exist_ok=True)
    CRISM_DATA.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Ensured directories exist:")
    print(f"       - {CRISM_QUICKVIEW}")
    print(f"       - {CRISM_BROWSE}")
    print(f"       - {CRISM_DATA}")


def parse_lbl_coords(lbl_path: Path) -> dict:
    """Parse coordinates from LBL file."""
    coords = {}
    try:
        content = lbl_path.read_text()

        # Extract lat/lon bounds
        patterns = {
            "min_lat": r"MINIMUM_LATITUDE\s*=\s*([-\d.]+)",
            "max_lat": r"MAXIMUM_LATITUDE\s*=\s*([-\d.]+)",
            "west_lon": r"WESTERNMOST_LONGITUDE\s*=\s*([-\d.]+)",
            "east_lon": r"EASTERNMOST_LONGITUDE\s*=\s*([-\d.]+)",
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, content)
            if match:
                coords[key] = float(match.group(1))

        # Calculate center point if we have all bounds
        if len(coords) == 4:
            coords["center_lat"] = (coords["min_lat"] + coords["max_lat"]) / 2
            coords["center_lon"] = (coords["west_lon"] + coords["east_lon"]) / 2

    except Exception as e:
        print(f"[WARN] Failed to parse LBL {lbl_path.name}: {e}")

    return coords


def copy_lbl_files():
    """Copy LBL files to crism_data directory and parse coordinates."""
    if not ARCADIA_LBL.exists():
        print(f"[ERROR] LBL directory not found: {ARCADIA_LBL}")
        return {}

    lbl_files = list(ARCADIA_LBL.glob("*.lbl"))
    print(f"[INFO] Found {len(lbl_files)} LBL files in {ARCADIA_LBL}")

    # Map: base_obs_id -> {product_id, lbl_file, coords}
    obs_data = {}
    copied = 0
    skipped = 0

    for lbl_path in lbl_files:
        filename = lbl_path.name
        dest_path = CRISM_DATA / filename

        # Copy if not exists
        if not dest_path.exists():
            shutil.copy2(lbl_path, dest_path)
            copied += 1
        else:
            skipped += 1

        # Parse product info from filename (e.g., frt00003156_07_brcarj_mtr3.lbl)
        # We prefer brcarj files as they contain the browse coordinates
        if "_brcarj_" in filename:
            parts = filename.replace(".lbl", "").split("_")
            if len(parts) >= 4:
                base_obs_id = parts[0]  # e.g., frt00003156
                product_id = filename.replace(".lbl", "")  # full product name

                coords = parse_lbl_coords(lbl_path)

                obs_data[base_obs_id] = {
                    "product_id": product_id,
                    "lbl_file": filename,
                    "coords": coords,
                }

    print(f"[INFO] LBL files: {copied} copied, {skipped} skipped (already exist)")
    print(f"[INFO] Parsed coordinates for {len(obs_data)} observations")

    return obs_data


def classify_and_move_files():
    """Classify files and move to appropriate directories."""
    if not ARCADIA_BROWSE.exists():
        print(f"[ERROR] Source directory not found: {ARCADIA_BROWSE}")
        return {}, {}

    moved_vnir = {}
    moved_browse = defaultdict(list)
    skipped = []

    files = list(ARCADIA_BROWSE.glob("*.png"))
    print(f"[INFO] Found {len(files)} PNG files in {ARCADIA_BROWSE}")

    for src_path in files:
        filename = src_path.name

        # Classify by suffix
        if filename.endswith(VNIR_SUFFIX):
            dest_dir = CRISM_QUICKVIEW
            obs_id = filename.replace(VNIR_SUFFIX, "")
            file_type = "VNIR"
        elif any(filename.endswith(s) for s in BROWSE_SUFFIXES):
            dest_dir = CRISM_BROWSE
            for s in BROWSE_SUFFIXES:
                if filename.endswith(s):
                    obs_id = filename.replace(s, "")
                    file_type = s.replace("_", "").replace(".png", "")
                    break
        else:
            print(f"[WARN] Unknown file type, skipping: {filename}")
            continue

        dest_path = dest_dir / filename

        # Move or skip if exists
        if dest_path.exists():
            skipped.append(filename)
        else:
            shutil.copy2(src_path, dest_path)

        # Track for index update
        if file_type == "VNIR":
            moved_vnir[obs_id] = f"/crism/quickview/{filename}"
        else:
            moved_browse[obs_id].append({
                "type": file_type,
                "path": f"/crism/browse/{filename}"
            })

    print(f"[INFO] Processed files:")
    print(f"       - VNIR quickviews: {len(moved_vnir)}")
    print(f"       - Browse products: {sum(len(v) for v in moved_browse.values())}")
    print(f"       - Skipped (already exist): {len(skipped)}")

    return moved_vnir, moved_browse


def update_index(obs_data, moved_vnir, moved_browse):
    """Update index.geojson with new entries including coordinates from LBL."""
    # Load existing index
    if INDEX_PATH.exists():
        with open(INDEX_PATH, "r") as f:
            index = json.load(f)
    else:
        index = {"type": "FeatureCollection", "features": []}

    # Build lookup of existing features by product_id and base observation ID
    existing_product_ids = set()
    existing_base_ids = set()
    for feat in index["features"]:
        props = feat.get("properties", {})
        product_id = props.get("product_id", "")
        existing_product_ids.add(product_id)
        # Extract base observation ID
        base_id = product_id.split("_")[0] if "_" in product_id else product_id
        existing_base_ids.add(base_id)

    # Remove old arcadia entries that have placeholder coordinates
    # (entries we created before with incomplete data)
    new_features = []
    removed_count = 0
    for feat in index["features"]:
        props = feat.get("properties", {})
        product_id = props.get("product_id", "")
        base_id = product_id.split("_")[0] if "_" in product_id else product_id
        geom = feat.get("geometry", {})
        coords = geom.get("coordinates", [0, 0])

        # Keep if: has real coordinates OR is not an arcadia observation
        is_arcadia = base_id in obs_data
        has_real_coords = coords != [0, 0]

        if not is_arcadia or has_real_coords:
            new_features.append(feat)
        else:
            removed_count += 1

    if removed_count > 0:
        print(f"[INFO] Removed {removed_count} old arcadia entries with placeholder coordinates")

    index["features"] = new_features

    # Rebuild existing lookups after removal
    existing_product_ids = set()
    existing_base_ids = set()
    for feat in index["features"]:
        props = feat.get("properties", {})
        product_id = props.get("product_id", "")
        existing_product_ids.add(product_id)
        base_id = product_id.split("_")[0] if "_" in product_id else product_id
        existing_base_ids.add(base_id)

    # Add new features for arcadia observations
    added_features = []
    all_obs_ids = set(moved_vnir.keys()) | set(moved_browse.keys())

    for obs_id in sorted(all_obs_ids):
        # Get LBL data if available
        lbl_info = obs_data.get(obs_id, {})
        coords = lbl_info.get("coords", {})

        # Skip if no coordinate data
        if not coords:
            print(f"[WARN] No LBL coordinates for {obs_id}, skipping")
            continue

        product_id = lbl_info.get("product_id", obs_id)
        lbl_file = lbl_info.get("lbl_file")

        # Skip if already exists
        if product_id in existing_product_ids:
            continue

        # Create new feature
        props = {
            "instrument": "CRISM",
            "product_id": product_id,
            "base_key": obs_id,
            "mtr3_img": None,  # No IMG data for browse-only products
            "mtr3_lbl": lbl_file,
        }

        if obs_id in moved_vnir:
            props["quicklook"] = moved_vnir[obs_id]

        if obs_id in moved_browse:
            for browse in moved_browse[obs_id]:
                props[f"browse_{browse['type'].lower()}"] = browse["path"]

        # Use center coordinates from LBL
        center_lon = coords.get("center_lon", 0)
        center_lat = coords.get("center_lat", 0)

        added_features.append({
            "type": "Feature",
            "properties": props,
            "geometry": {
                "type": "Point",
                "coordinates": [center_lon, center_lat]
            }
        })

    # Append new features
    index["features"].extend(added_features)

    # Write updated index
    with open(INDEX_PATH, "w") as f:
        json.dump(index, f, indent=2)

    print(f"[INFO] Updated index.geojson:")
    print(f"       - Existing features kept: {len(index['features']) - len(added_features)}")
    print(f"       - New features added: {len(added_features)}")
    print(f"       - Total features: {len(index['features'])}")


def main():
    print("=" * 60)
    print("Arcadia Browse Reorganization Script")
    print("=" * 60)

    ensure_dirs()
    obs_data = copy_lbl_files()
    moved_vnir, moved_browse = classify_and_move_files()
    update_index(obs_data, moved_vnir, moved_browse)

    print("=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
