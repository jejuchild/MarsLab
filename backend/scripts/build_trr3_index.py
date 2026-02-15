"""
Build index.geojson for CRISM TRR3 observations from DDR lat/lon data.

Reads DDR1 files in mineral_cnn_data/ to extract footprint polygons.
DDR Band 4 = Latitude (areocentric, deg N)
DDR Band 5 = Longitude (areocentric, deg E, 0-360)

Uses actual image corner coordinates for accurate quadrilateral footprints.
"""

import json
import os
import re
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "mineral_cnn_data"
OUTPUT = DATA_DIR / "index.geojson"


def parse_ddr_label(lbl_path: Path) -> dict:
    """Parse DDR .LBL file for image dimensions."""
    text = lbl_path.read_text(errors="ignore")
    info = {}

    m = re.search(r"LINES\s*=\s*(\d+)", text)
    if m:
        info["lines"] = int(m.group(1))
    m = re.search(r"LINE_SAMPLES\s*=\s*(\d+)", text)
    if m:
        info["samples"] = int(m.group(1))
    m = re.search(r"BANDS\s*=\s*(\d+)", text)
    if m:
        info["bands"] = int(m.group(1))

    return info


def read_ddr_band(img_path: Path, info: dict, band_index: int) -> np.ndarray:
    """Read a single band from BSQ DDR IMG file."""
    lines = info["lines"]
    samples = info["samples"]
    band_size = lines * samples * 4  # float32
    offset = band_index * band_size

    with open(img_path, "rb") as f:
        f.seek(offset)
        data = f.read(band_size)

    return np.frombuffer(data, dtype="<f4").reshape(lines, samples)


def norm_lon(lon: float) -> float:
    """Convert 0-360 longitude to -180/180."""
    return lon - 360 if lon > 180 else lon


def extract_footprint(lat_band: np.ndarray, lon_band: np.ndarray) -> dict | None:
    """Extract a polygon footprint from DDR lat/lon bands.

    Uses the 4 image corners to build an accurate quadrilateral.
    For long strips, adds intermediate edge points for curvature.
    DDR longitude is 0-360; converted to -180/180.
    """
    lines, samples = lat_band.shape

    # Collect left and right edge points (top-to-bottom)
    # Subsample along the length for accuracy on long strips
    n_edge = max(2, min(lines, 20))  # 2-20 edge points per side
    step = max(1, (lines - 1) // (n_edge - 1))

    left_edge = []  # top to bottom
    right_edge = []  # top to bottom

    for i in range(0, lines, step):
        lat_l, lon_l = float(lat_band[i, 0]), norm_lon(float(lon_band[i, 0]))
        lat_r, lon_r = float(lat_band[i, -1]), norm_lon(float(lon_band[i, -1]))

        # Skip fill values
        if abs(lat_l) > 90 or abs(lon_l) > 180:
            continue
        if abs(lat_r) > 90 or abs(lon_r) > 180:
            continue

        left_edge.append((lon_l, lat_l))
        right_edge.append((lon_r, lat_r))

    # Always include last row
    last = lines - 1
    lat_l, lon_l = float(lat_band[last, 0]), norm_lon(float(lon_band[last, 0]))
    lat_r, lon_r = float(lat_band[last, -1]), norm_lon(float(lon_band[last, -1]))
    if abs(lat_l) <= 90 and abs(lon_l) <= 180:
        if not left_edge or left_edge[-1] != (lon_l, lat_l):
            left_edge.append((lon_l, lat_l))
    if abs(lat_r) <= 90 and abs(lon_r) <= 180:
        if not right_edge or right_edge[-1] != (lon_r, lat_r):
            right_edge.append((lon_r, lat_r))

    if len(left_edge) < 2 or len(right_edge) < 2:
        return None

    # Build polygon: left edge (top→bottom) + right edge (bottom→top) + close
    polygon = left_edge + list(reversed(right_edge))
    polygon.append(polygon[0])  # close ring

    return {
        "type": "Polygon",
        "coordinates": [polygon],
    }


def build_index():
    """Build index.geojson from all TRR3 observations."""
    features = []

    if not DATA_DIR.exists():
        print(f"Data directory not found: {DATA_DIR}")
        return

    obs_dirs = sorted([
        d for d in DATA_DIR.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])

    print(f"Found {len(obs_dirs)} observation directories")

    for obs_dir in obs_dirs:
        ddr_lbl = None
        ddr_img = None
        trr3_lbl = None

        for f in obs_dir.iterdir():
            name_upper = f.name.upper()
            if name_upper.endswith("_DDR1.LBL"):
                ddr_lbl = f
            elif name_upper.endswith("_DDR1.IMG"):
                ddr_img = f
            elif name_upper.endswith("_TRR3.LBL"):
                trr3_lbl = f

        if not ddr_lbl or not ddr_img:
            print(f"  SKIP {obs_dir.name}: no DDR files")
            continue

        try:
            info = parse_ddr_label(ddr_lbl)
            if not all(k in info for k in ["lines", "samples", "bands"]):
                print(f"  SKIP {obs_dir.name}: incomplete DDR label")
                continue

            lat_band = read_ddr_band(ddr_img, info, 3)
            lon_band = read_ddr_band(ddr_img, info, 4)

            geometry = extract_footprint(lat_band, lon_band)
            if not geometry:
                print(f"  SKIP {obs_dir.name}: no valid coordinates")
                continue

            obs_id = obs_dir.name

            # Extract TRR3 product ID from label
            trr3_product_id = None
            if trr3_lbl:
                text = trr3_lbl.read_text(errors="ignore")
                m = re.search(r'PRODUCT_ID\s*=\s*"?([^"\s]+)"?', text)
                if m:
                    trr3_product_id = m.group(1)

            feature = {
                "type": "Feature",
                "properties": {
                    "product_id": obs_id,
                    "trr3_product_id": trr3_product_id,
                    "instrument": "CRISM_TRR3",
                    "obs_dir": obs_dir.name,
                },
                "geometry": geometry,
            }
            features.append(feature)

            n_verts = len(geometry["coordinates"][0])
            print(f"  OK {obs_dir.name}: {info['lines']}x{info['samples']}, {n_verts} vertices")

        except Exception as e:
            print(f"  ERROR {obs_dir.name}: {e}")
            continue

    geojson = {
        "type": "FeatureCollection",
        "features": features,
    }

    OUTPUT.write_text(json.dumps(geojson, indent=2))
    print(f"\nWrote {len(features)} features to {OUTPUT}")


if __name__ == "__main__":
    build_index()
