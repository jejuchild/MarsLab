#!/usr/bin/env python3
"""
Update HiRISE DTM index.geojson with a newly generated custom DTM.

Reads geographic bounds from a GeoTIFF DTM file using GDAL, generates
a GeoJSON Feature entry, and appends it to the index.

Usage:
    python asp_index_updater.py \
        --dtm /path/to/run-DEM.tif \
        --obs-a ESP_060706_2195 \
        --obs-b ESP_060416_2195 \
        [--ortho /path/to/ortho.tif] \
        [--install-dir /path/to/hirise_dtm_data]
"""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Optional

try:
    from osgeo import gdal
    gdal.UseExceptions()
except ImportError:
    print("Error: GDAL Python bindings required. Install with: pip install GDAL", file=sys.stderr)
    sys.exit(1)

BACKEND_DIR = Path(__file__).parent.parent
DEFAULT_INDEX = BACKEND_DIR / "hirise_dtm_data" / "index.geojson"
DEFAULT_INSTALL_DIR = BACKEND_DIR / "hirise_dtm_data"


def extract_bounds(tif_path: str) -> dict:
    """Extract geographic bounding box from a GeoTIFF.

    Handles both geographic and projected coordinate systems.
    For projected CRS (e.g. Sinusoidal from point2dem), reprojects
    corner coordinates to geographic lat/lon.
    """
    from osgeo import osr

    ds = gdal.Open(tif_path)
    if ds is None:
        raise FileNotFoundError(f"Cannot open: {tif_path}")

    gt = ds.GetGeoTransform()
    cols = ds.RasterXSize
    rows = ds.RasterYSize

    # Corner coordinates in the file's CRS
    x_min = gt[0]
    x_max = gt[0] + cols * gt[1]
    y_max = gt[3]  # origin is top-left
    y_min = gt[3] + rows * gt[5]  # gt[5] is negative

    resolution_m = abs(gt[1])  # pixel size in file units

    # Check if the CRS is projected (meters) vs geographic (degrees)
    srs = osr.SpatialReference()
    srs.ImportFromWkt(ds.GetProjection())

    if srs.IsProjected():
        # Transform corners from projected CRS to geographic lat/lon
        geo_srs = srs.CloneGeogCS()
        transform = osr.CoordinateTransformation(srs, geo_srs)

        corners = [
            (x_min, y_min), (x_max, y_min),
            (x_max, y_max), (x_min, y_max),
        ]
        lons, lats = [], []
        for x, y in corners:
            lon, lat, _ = transform.TransformPoint(x, y)
            lons.append(lon)
            lats.append(lat)

        west, east = min(lons), max(lons)
        south, north = min(lats), max(lats)
        # resolution_m is already in meters for projected CRS
    else:
        west, east = min(x_min, x_max), max(x_min, x_max)
        south, north = min(y_min, y_max), max(y_min, y_max)
        # Convert degree pixel size to meters (Mars radius ~3389.5 km)
        resolution_m = abs(gt[1]) * 3389500 * 3.14159265 / 180.0

    # Convert 0-360 longitudes to -180/180
    if west > 180:
        west -= 360
    if east > 180:
        east -= 360

    ds = None
    return {
        "west": round(west, 6),
        "east": round(east, 6),
        "south": round(south, 6),
        "north": round(north, 6),
        "resolution_m": round(resolution_m, 1),
    }


def generate_product_id(obs_a: str, obs_b: str) -> str:
    """Generate DTM product ID from two observation IDs.

    Format: DTEEC_{orbit_a}_{target_a}_{orbit_b}_{target_b}_A01
    e.g. ESP_060706_2195 + ESP_060416_2195 → DTEEC_060706_2195_060416_2195_A01
    """
    pattern = re.compile(r'^(?:ESP|PSP|TRA)_(\d{6})_(\d{4})$')
    m_a = pattern.match(obs_a)
    m_b = pattern.match(obs_b)
    if not m_a or not m_b:
        raise ValueError(
            f"Invalid observation IDs: {obs_a}, {obs_b}. "
            "Expected format: ESP_NNNNNN_NNNN"
        )
    return f"DTEEC_{m_a.group(1)}_{m_a.group(2)}_{m_b.group(1)}_{m_b.group(2)}_A01"


def build_feature(bounds: dict, product_id: str, dtm_filename: str,
                  ortho_filename: Optional[str] = None) -> dict:
    """Build a GeoJSON Feature for the DTM."""
    w, e, s, n = bounds["west"], bounds["east"], bounds["south"], bounds["north"]
    coords = [[
        [w, s], [e, s], [e, n], [w, n], [w, s]
    ]]
    props = {
        "product_id": product_id,
        "title": f"{product_id} (CUSTOM)",
        "dtm_file": dtm_filename,
        "ortho_file": ortho_filename or "",
        "custom": True,
        "west": w,
        "east": e,
        "south": s,
        "north": n,
        "resolution_m": bounds["resolution_m"],
    }
    return {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": coords},
        "properties": props,
    }


def update_index(feature: dict, index_path: Path) -> None:
    """Append a feature to the GeoJSON index, avoiding duplicates."""
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
    else:
        index = {"type": "FeatureCollection", "features": []}

    pid = feature["properties"]["product_id"]
    # Remove existing entry with same product_id (for re-runs)
    index["features"] = [
        f for f in index["features"]
        if f.get("properties", {}).get("product_id") != pid
    ]
    index["features"].append(feature)

    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"  Index updated: {pid} → {index_path}")


def install_dtm(src_path: Path, install_dir: Path, product_id: str) -> str:
    """Copy or symlink DTM file into the install directory."""
    install_dir.mkdir(parents=True, exist_ok=True)
    ext = src_path.suffix  # .tif or .IMG
    dest_name = f"{product_id}{ext}"
    dest = install_dir / dest_name
    if dest.exists():
        dest.unlink()
    shutil.copy2(src_path, dest)
    print(f"  DTM installed: {dest}")
    return dest_name


def main():
    parser = argparse.ArgumentParser(
        description="Add a custom ASP-generated DTM to the HiRISE DTM index"
    )
    parser.add_argument("--dtm", required=True, help="Path to DTM GeoTIFF")
    parser.add_argument("--obs-a", required=True,
                        help="First observation ID (e.g. ESP_060706_2195)")
    parser.add_argument("--obs-b", required=True,
                        help="Second observation ID (e.g. ESP_060416_2195)")
    parser.add_argument("--ortho", default=None,
                        help="Path to orthoimage (optional)")
    parser.add_argument("--index", default=str(DEFAULT_INDEX),
                        help=f"Path to index.geojson (default: {DEFAULT_INDEX})")
    parser.add_argument("--install-dir", default=str(DEFAULT_INSTALL_DIR),
                        help=f"Directory to copy DTM into (default: {DEFAULT_INSTALL_DIR})")
    parser.add_argument("--no-install", action="store_true",
                        help="Skip copying DTM file, just update index")

    args = parser.parse_args()

    dtm_path = Path(args.dtm)
    if not dtm_path.exists():
        print(f"Error: DTM file not found: {dtm_path}", file=sys.stderr)
        sys.exit(1)

    # 1. Generate product ID
    product_id = generate_product_id(args.obs_a, args.obs_b)
    print(f"  Product ID: {product_id}")

    # 2. Extract geographic bounds
    bounds = extract_bounds(str(dtm_path))
    print(f"  Bounds: W={bounds['west']:.4f} E={bounds['east']:.4f} "
          f"S={bounds['south']:.4f} N={bounds['north']:.4f}")
    print(f"  Resolution: ~{bounds['resolution_m']:.1f} m/px")

    # 3. Install DTM file
    install_dir = Path(args.install_dir)
    if args.no_install:
        dtm_filename = dtm_path.name
    else:
        dtm_filename = install_dtm(dtm_path, install_dir, product_id)

    # 4. Install ortho if provided
    ortho_filename = None
    if args.ortho:
        ortho_path = Path(args.ortho)
        if ortho_path.exists():
            ortho_dest = install_dir / f"{product_id}_ORTHO{ortho_path.suffix}"
            shutil.copy2(ortho_path, ortho_dest)
            ortho_filename = ortho_dest.name
            print(f"  Ortho installed: {ortho_dest}")

    # 5. Build feature and update index
    feature = build_feature(bounds, product_id, dtm_filename, ortho_filename)
    update_index(feature, Path(args.index))

    print(f"\n  Done! Custom DTM '{product_id}' added to index.")


if __name__ == "__main__":
    main()
