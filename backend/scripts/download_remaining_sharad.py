#!/usr/bin/env python3
"""
Download remaining SHARAD RDR products near Jezero crater.
Wider bbox: lat 15-22, lon 74-80 (296 products on ODE).
Skips products already on disk, downloads .dat + .lbl via aria2c,
then updates index.geojson.
"""

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError

BACKEND_DIR = Path(__file__).parent.parent
SHARAD_HR_DIR = BACKEND_DIR / "sharad_highres"
SHARAD_HR_INDEX = BACKEND_DIR / "sharad_highres_data" / "index.geojson"
ODE_REST = "https://oderest.rsl.wustl.edu/live2"

# Wider Jezero bbox
MINLAT = 15
MAXLAT = 22
WLON = 74
ELON = 80


def lon_to_180(lon):
    return ((lon + 180) % 360) - 180


def parse_footprint_wkt(wkt):
    numbers = re.findall(r'(-?\d+\.?\d*)\s+(-?\d+\.?\d*)', str(wkt))
    if not numbers:
        return None
    return [[float(n[0]), float(n[1])] for n in numbers]


def query_ode():
    """Query ODE for all SHARAD RDR products in the Jezero area."""
    url = (
        f"{ODE_REST}?"
        f"target=mars&ihid=mro&iid=sharad&pt=RDR&"
        f"minlat={MINLAT}&maxlat={MAXLAT}&"
        f"westernlon={WLON}&easternlon={ELON}&"
        f"output=json&results=m&limit=1000"
    )
    print(f"Querying ODE: lat [{MINLAT},{MAXLAT}], lon [{WLON},{ELON}]")
    print(f"URL: {url}")

    for attempt in range(3):
        try:
            req = Request(url, headers={"User-Agent": "MarsLab/1.0"})
            with urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode())
            return data
        except (URLError, Exception) as e:
            print(f"  ODE error (attempt {attempt+1}/3): {e}")
            if attempt < 2:
                time.sleep(2 ** attempt)
    return None


def get_existing_on_disk():
    """Get set of product IDs already downloaded (uppercase stems)."""
    existing = set()
    if SHARAD_HR_DIR.exists():
        for f in SHARAD_HR_DIR.iterdir():
            if f.suffix == ".dat":
                existing.add(f.stem.upper())
    return existing


def run_aria2(download_list, concurrent=4):
    """Batch download via aria2c."""
    if not download_list:
        return 0, 0

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False,
                                     prefix="aria2_sharad_") as f:
        input_path = f.name
        for url, out_dir, filename in download_list:
            f.write(f"{url}\n")
            f.write(f"  dir={out_dir}\n")
            f.write(f"  out={filename}\n\n")

    print(f"\naria2c: {len(download_list)} files, {concurrent} concurrent")

    cmd = [
        "aria2c",
        "-i", input_path,
        "-j", str(concurrent),
        "-x", "16", "-s", "16", "-k", "1M",
        "--continue=true",
        "--file-allocation=trunc",
        "--auto-file-renaming=false",
        "--allow-overwrite=false",
        "--retry-wait=3",
        "--max-tries=5",
        "--timeout=600",
        "--connect-timeout=30",
        "--summary-interval=15",
    ]

    try:
        result = subprocess.run(cmd)
        os.remove(input_path)
        if result.returncode == 0:
            return len(download_list), 0
        ok = sum(1 for _, d, fn in download_list if os.path.exists(os.path.join(d, fn)))
        return ok, len(download_list) - ok
    except FileNotFoundError:
        print("ERROR: aria2c not found!")
        os.remove(input_path)
        return 0, len(download_list)


def update_index(tracks):
    """Add new SHARAD tracks to index.geojson."""
    SHARAD_HR_INDEX.parent.mkdir(parents=True, exist_ok=True)

    if SHARAD_HR_INDEX.exists():
        with open(SHARAD_HR_INDEX) as f:
            index = json.load(f)
    else:
        index = {"type": "FeatureCollection", "features": []}

    existing_ids = {feat["properties"]["product_id"].upper()
                    for feat in index["features"]}

    added = 0
    for t in tracks:
        pid = t["pid"]
        if pid in existing_ids:
            continue
        dat_path = SHARAD_HR_DIR / f"{pid.lower()}.dat"
        if not dat_path.exists():
            continue

        footprint_wkt = t.get("footprint_wkt", "")
        coords = parse_footprint_wkt(footprint_wkt)

        if coords and len(coords) >= 2:
            geometry = {"type": "LineString", "coordinates": coords}
        else:
            lon = lon_to_180(t["lon"]) if t["lon"] else 0
            geometry = {"type": "Point", "coordinates": [lon, t["lat"] or 0]}

        feature = {
            "type": "Feature",
            "geometry": geometry,
            "properties": {
                "instrument": "SHARAD_HIGHRES",
                "product_id": pid,
                "dat_file": f"{pid.lower()}.dat",
                "lbl_file": f"{pid.lower()}.lbl",
            }
        }
        index["features"].append(feature)
        existing_ids.add(pid)
        added += 1

    with open(SHARAD_HR_INDEX, "w") as f:
        json.dump(index, f, indent=2)
    print(f"Index: +{added} entries (total: {len(index['features'])})")


def main():
    SHARAD_HR_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Query ODE
    data = query_ode()
    if not data:
        print("ODE query failed, aborting")
        sys.exit(1)

    products = data.get("ODEResults", {}).get("Products", {}).get("Product", [])
    if isinstance(products, dict):
        products = [products]
    print(f"Found {len(products)} SHARAD RDR products on ODE")

    if not products:
        print("No products found")
        return

    # 2. Extract product info
    tracks = []
    for p in products:
        label_url = p.get("LabelURL", "")
        if not label_url:
            continue
        pid = label_url.rsplit("/", 1)[-1]
        if "." in pid:
            pid = pid.rsplit(".", 1)[0]
        pid = pid.upper()
        if not pid:
            continue

        try:
            lat = float(p.get("Center_latitude", 0))
        except (TypeError, ValueError):
            lat = None
        try:
            lon = float(p.get("Center_longitude", 0))
        except (TypeError, ValueError):
            lon = None

        footprint_wkt = p.get("Footprint_C0_geometry", "")
        tracks.append({
            "pid": pid, "lat": lat, "lon": lon,
            "label_url": label_url,
            "footprint_wkt": footprint_wkt,
        })

    # 3. Filter out existing
    existing = get_existing_on_disk()
    new_tracks = [t for t in tracks if t["pid"] not in existing]

    print(f"Total products: {len(tracks)}")
    print(f"Already on disk: {len(tracks) - len(new_tracks)}")
    print(f"To download: {len(new_tracks)}")

    if not new_tracks:
        print("All products already downloaded!")
        # Still update index in case some are missing from it
        update_index(tracks)
        return

    # 4. Build download list
    download_list = []
    for t in new_tracks:
        label_url = t["label_url"]
        pid_lower = t["pid"].lower()
        base_url = label_url.rsplit("/", 1)[0]
        dat_url = f"{base_url}/{pid_lower}.dat"
        lbl_url = label_url

        download_list.append((dat_url, str(SHARAD_HR_DIR), f"{pid_lower}.dat"))
        download_list.append((lbl_url, str(SHARAD_HR_DIR), f"{pid_lower}.lbl"))

    # 5. Download
    t0 = time.time()
    ok, fail = run_aria2(download_list, concurrent=4)
    elapsed = time.time() - t0
    print(f"\nDownloaded: {ok}, Failed: {fail} ({elapsed:.0f}s)")

    # 6. Update index with ALL tracks (existing + new)
    update_index(tracks)

    # 7. Summary
    dat_count = len(list(SHARAD_HR_DIR.glob("*.dat")))
    total_size_gb = sum(f.stat().st_size for f in SHARAD_HR_DIR.glob("*.dat")) / 1024**3
    print(f"\nFinal: {dat_count} .dat files on disk, {total_size_gb:.1f} GB total")


if __name__ == "__main__":
    main()
