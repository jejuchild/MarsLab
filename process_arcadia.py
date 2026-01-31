#!/usr/bin/env python3
"""
Process Arcadia browse files:
1. Copy browse maps (HYD, IC2, ICE) to crism_browse/
2. Copy quickview (VNIR) to crism_quickview/
3. Update crism_data/index.geojson with coordinates from LBL files
"""

import os
import shutil
import json
import re
from pathlib import Path

# Directories
ARCADIA_BROWSE = Path("/home/cspark/MarsLab/arcadia_browse/browse")
ARCADIA_LBL = Path("/home/cspark/MarsLab/arcadia_browse/lbl")
ARCADIA_BROWSE_NEW = Path("/home/cspark/MarsLab/arcadia_browse/arcadia_browse/browse")
ARCADIA_LBL_NEW = Path("/home/cspark/MarsLab/arcadia_browse/arcadia_browse/lbl")

CRISM_BROWSE = Path("/home/cspark/MarsLab/backend/crism_browse")
CRISM_QUICKVIEW = Path("/home/cspark/MarsLab/backend/crism_quickview")
INDEX_FILE = Path("/home/cspark/MarsLab/backend/crism_data/index.geojson")

def parse_lbl(lbl_path):
    """Parse LBL file to extract coordinates."""
    content = lbl_path.read_text()

    def get_value(key):
        match = re.search(rf'{key}\s*=\s*([-+]?[0-9.]+)', content)
        return float(match.group(1)) if match else None

    min_lat = get_value('MINIMUM_LATITUDE')
    max_lat = get_value('MAXIMUM_LATITUDE')
    west_lon = get_value('WESTERNMOST_LONGITUDE')
    east_lon = get_value('EASTERNMOST_LONGITUDE')

    if all(v is not None for v in [min_lat, max_lat, west_lon, east_lon]):
        # Calculate center point
        center_lat = (min_lat + max_lat) / 2
        center_lon = (west_lon + east_lon) / 2
        return {
            'center_lat': center_lat,
            'center_lon': center_lon,
            'min_lat': min_lat,
            'max_lat': max_lat,
            'west_lon': west_lon,
            'east_lon': east_lon
        }
    return None

def get_base_id(filename):
    """Extract base observation ID from filename (e.g., frt00003156 from frt00003156_HYD.png)."""
    match = re.match(r'(frt[0-9a-f]+)', filename.lower())
    return match.group(1) if match else None

def main():
    # Collect all source directories
    browse_dirs = [ARCADIA_BROWSE]
    lbl_dirs = [ARCADIA_LBL]

    if ARCADIA_BROWSE_NEW.exists():
        browse_dirs.append(ARCADIA_BROWSE_NEW)
    if ARCADIA_LBL_NEW.exists():
        lbl_dirs.append(ARCADIA_LBL_NEW)

    # Track processed observations
    processed_obs = {}
    copied_browse = 0
    copied_quickview = 0

    # Process LBL files first to get coordinates
    for lbl_dir in lbl_dirs:
        if not lbl_dir.exists():
            continue
        for lbl_file in lbl_dir.glob("*.lbl"):
            base_id = get_base_id(lbl_file.name)
            if base_id and base_id not in processed_obs:
                coords = parse_lbl(lbl_file)
                if coords:
                    processed_obs[base_id] = coords
                    print(f"[LBL] {base_id}: lat={coords['center_lat']:.4f}, lon={coords['center_lon']:.4f}")

    # Copy browse and quickview files
    for browse_dir in browse_dirs:
        if not browse_dir.exists():
            continue
        for png_file in browse_dir.glob("*.png"):
            base_id = get_base_id(png_file.name)
            if not base_id:
                continue

            filename = png_file.name

            # Determine destination based on file type
            if "_VNIR" in filename.upper():
                # Quickview file
                dest = CRISM_QUICKVIEW / filename
                if not dest.exists():
                    shutil.copy2(png_file, dest)
                    copied_quickview += 1
                    print(f"[QUICKVIEW] Copied: {filename}")
            elif any(x in filename.upper() for x in ["_HYD", "_ICE", "_IC2"]):
                # Browse file
                dest = CRISM_BROWSE / filename
                if not dest.exists():
                    shutil.copy2(png_file, dest)
                    copied_browse += 1
                    print(f"[BROWSE] Copied: {filename}")

    print(f"\nCopied {copied_browse} browse files, {copied_quickview} quickview files")

    # Load existing index
    with open(INDEX_FILE, 'r') as f:
        index_data = json.load(f)

    # Get existing product IDs
    existing_ids = set()
    for feature in index_data['features']:
        props = feature.get('properties', {})
        pid = props.get('product_id', '')
        base = get_base_id(pid)
        if base:
            existing_ids.add(base)

    # Add new features for arcadia observations
    added = 0
    for base_id, coords in processed_obs.items():
        if base_id in existing_ids:
            # Update coordinates if they exist but have 0,0
            for feature in index_data['features']:
                props = feature.get('properties', {})
                pid = props.get('product_id', '')
                if get_base_id(pid) == base_id:
                    geom = feature.get('geometry', {})
                    if geom.get('coordinates') == [0, 0]:
                        # Update with real coordinates
                        feature['geometry'] = {
                            'type': 'Polygon',
                            'coordinates': [[
                                [coords['west_lon'], coords['min_lat']],
                                [coords['east_lon'], coords['min_lat']],
                                [coords['east_lon'], coords['max_lat']],
                                [coords['west_lon'], coords['max_lat']],
                                [coords['west_lon'], coords['min_lat']]
                            ]]
                        }
                        print(f"[UPDATE] Updated coordinates for {base_id}")
            continue

        # Create new feature
        feature = {
            'type': 'Feature',
            'properties': {
                'instrument': 'CRISM',
                'product_id': f'{base_id}_07_brcarj_mtr3',
                'base_key': base_id,
                'mtr3_img': None,
                'mtr3_lbl': f'{base_id}_07_brcarj_mtr3.lbl',
                'quicklook': f'/crism/quickview/{base_id}_VNIR.png'
            },
            'geometry': {
                'type': 'Polygon',
                'coordinates': [[
                    [coords['west_lon'], coords['min_lat']],
                    [coords['east_lon'], coords['min_lat']],
                    [coords['east_lon'], coords['max_lat']],
                    [coords['west_lon'], coords['max_lat']],
                    [coords['west_lon'], coords['min_lat']]
                ]]
            }
        }
        index_data['features'].append(feature)
        added += 1
        print(f"[ADD] Added {base_id} at ({coords['center_lat']:.4f}, {coords['center_lon']:.4f})")

    # Save updated index
    with open(INDEX_FILE, 'w') as f:
        json.dump(index_data, f, indent=2)

    print(f"\nAdded {added} new features to index.geojson")
    print(f"Total features: {len(index_data['features'])}")

if __name__ == '__main__':
    main()
