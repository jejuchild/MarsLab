#!/usr/bin/env python3
"""
Process SHARAD THM data:
1. Copy JPG files to sharad_quickview/
2. Parse LBL files for coordinates
3. Create sharad_data/index.geojson
"""

import os
import shutil
import json
import re
from pathlib import Path

# Directories
SHARAD_SRC = Path("/home/cspark/MarsLab/arcadia_sharad_thm")
SHARAD_QUICKVIEW = Path("/home/cspark/MarsLab/backend/sharad_quickview")
SHARAD_DATA = Path("/home/cspark/MarsLab/backend/sharad_data")

def parse_sharad_lbl(lbl_path):
    """Parse SHARAD LBL file to extract coordinates."""
    try:
        content = lbl_path.read_text()
    except:
        return None

    def get_value(key):
        # Handle both regular and MRO: prefixed keys
        match = re.search(rf'{key}\s*=\s*([-+]?[0-9.]+)', content)
        return float(match.group(1)) if match else None

    start_lon = get_value('MRO:START_SUB_SPACECRAFT_LONGITUDE')
    start_lat = get_value('MRO:START_SUB_SPACECRAFT_LATITUDE')
    stop_lon = get_value('MRO:STOP_SUB_SPACECRAFT_LONGITUDE')
    stop_lat = get_value('MRO:STOP_SUB_SPACECRAFT_LATITUDE')

    # Get product ID
    match = re.search(r'PRODUCT_ID\s*=\s*"?([^"\s]+)"?', content)
    product_id = match.group(1) if match else None

    if all(v is not None for v in [start_lon, start_lat, stop_lon, stop_lat, product_id]):
        # Normalize longitude to -180 to 180
        if start_lon > 180:
            start_lon -= 360
        if stop_lon > 180:
            stop_lon -= 360

        return {
            'product_id': product_id,
            'start_lon': start_lon,
            'start_lat': start_lat,
            'stop_lon': stop_lon,
            'stop_lat': stop_lat
        }
    return None

def coords_to_linestring(coords):
    """Convert start/stop coords to GeoJSON LineString."""
    return {
        'type': 'LineString',
        'coordinates': [
            [coords['start_lon'], coords['start_lat']],
            [coords['stop_lon'], coords['stop_lat']]
        ]
    }

def main():
    jpg_dir = SHARAD_SRC / "jpg"
    lbl_dir = SHARAD_SRC / "lbl"

    # Copy JPG files
    copied = 0
    for jpg_file in jpg_dir.glob("*.jpg"):
        dest = SHARAD_QUICKVIEW / jpg_file.name
        if not dest.exists():
            shutil.copy2(jpg_file, dest)
            copied += 1
    print(f"Copied {copied} JPG files to sharad_quickview/")

    # Process LBL files and create index
    features = []
    for lbl_file in lbl_dir.glob("*.lbl"):
        coords = parse_sharad_lbl(lbl_file)
        if not coords:
            print(f"[SKIP] Could not parse: {lbl_file.name}")
            continue

        product_id = coords['product_id']

        feature = {
            'type': 'Feature',
            'properties': {
                'instrument': 'SHARAD',
                'product_id': product_id,
                'quickview': f'/sharad/quickview/{product_id.lower()}.jpg',
                'highres': f'/sharad/highres/{product_id.lower()}.tif',  # Placeholder
                'start_lat': coords['start_lat'],
                'start_lon': coords['start_lon'],
                'stop_lat': coords['stop_lat'],
                'stop_lon': coords['stop_lon']
            },
            'geometry': coords_to_linestring(coords)
        }
        features.append(feature)
        print(f"[OK] {product_id}: ({coords['start_lat']:.2f}, {coords['start_lon']:.2f}) -> ({coords['stop_lat']:.2f}, {coords['stop_lon']:.2f})")

    # Save index
    index_data = {
        'type': 'FeatureCollection',
        'features': features
    }

    index_file = SHARAD_DATA / "index.geojson"
    with open(index_file, 'w') as f:
        json.dump(index_data, f, indent=2)

    print(f"\nCreated {len(features)} features in {index_file}")

if __name__ == '__main__':
    main()
