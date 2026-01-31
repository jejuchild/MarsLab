#!/usr/bin/env python3
"""
Update index.geojson to convert Point geometries to Polygon by reading LBL files.
"""

import os
import json
import re
from pathlib import Path

CRISM_DATA_DIR = Path("/home/cspark/MarsLab/backend/crism_data")
INDEX_FILE = CRISM_DATA_DIR / "index.geojson"

def parse_lbl_coords(lbl_path):
    """Parse LBL file to extract bounding box coordinates."""
    try:
        content = lbl_path.read_text()
    except:
        return None

    def get_value(key):
        match = re.search(rf'{key}\s*=\s*([-+]?[0-9.]+)', content)
        return float(match.group(1)) if match else None

    min_lat = get_value('MINIMUM_LATITUDE')
    max_lat = get_value('MAXIMUM_LATITUDE')
    west_lon = get_value('WESTERNMOST_LONGITUDE')
    east_lon = get_value('EASTERNMOST_LONGITUDE')

    if all(v is not None for v in [min_lat, max_lat, west_lon, east_lon]):
        return {
            'min_lat': min_lat,
            'max_lat': max_lat,
            'west_lon': west_lon,
            'east_lon': east_lon
        }
    return None

def coords_to_polygon(coords):
    """Convert bounding box to GeoJSON Polygon."""
    west = coords['west_lon']
    east = coords['east_lon']
    south = coords['min_lat']
    north = coords['max_lat']

    return {
        'type': 'Polygon',
        'coordinates': [[
            [west, south],
            [east, south],
            [east, north],
            [west, north],
            [west, south]
        ]]
    }

def main():
    # Load index
    with open(INDEX_FILE, 'r') as f:
        data = json.load(f)

    updated = 0
    skipped = 0

    for feature in data['features']:
        geom = feature.get('geometry', {})
        props = feature.get('properties', {})
        product_id = props.get('product_id', '')

        # Skip if already Polygon with valid coords
        if geom.get('type') == 'Polygon':
            coords = geom.get('coordinates', [[]])
            if coords and coords[0] and coords[0][0] != [0, 0]:
                continue

        # Try to find LBL file
        lbl_file = CRISM_DATA_DIR / f"{product_id}.lbl"
        if not lbl_file.exists():
            # Try alternative names
            base_id = product_id.split('_')[0] if '_' in product_id else product_id
            for lbl in CRISM_DATA_DIR.glob(f"{base_id}*.lbl"):
                lbl_file = lbl
                break

        if not lbl_file.exists():
            skipped += 1
            continue

        coords = parse_lbl_coords(lbl_file)
        if coords:
            feature['geometry'] = coords_to_polygon(coords)
            updated += 1
        else:
            skipped += 1

    # Save updated index
    with open(INDEX_FILE, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Updated {updated} features to Polygon geometry")
    print(f"Skipped {skipped} features (no LBL or invalid coords)")
    print(f"Total features: {len(data['features'])}")

if __name__ == '__main__':
    main()
