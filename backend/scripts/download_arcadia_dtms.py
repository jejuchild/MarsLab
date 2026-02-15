#!/usr/bin/env python3
"""
Download all HiRISE DTM products in Arcadia Planitia from PDS via ODE,
using aria2c for fast multi-connection downloads.

Steps:
1. Query ODE REST API for HiRISE DTM products in Arcadia Planitia bbox
2. Filter to products we don't already have on disk
3. Extract DTM .IMG + orthoimage .JP2 URLs from ODE response (Product_files field)
4. Download all files via aria2c (16 connections per file, 4 concurrent)
5. Update hirise_dtm_data/index.geojson with new entries
"""

import asyncio
import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import aiohttp

BACKEND_DIR = Path(__file__).parent.parent
DTM_DIR = BACKEND_DIR / "hirise_dtm_data"
DTM_INDEX = DTM_DIR / "index.geojson"

ODE_REST = "https://oderest.rsl.wustl.edu/live2"

# Arcadia Planitia search bbox (0-360 lon for ODE)
SEARCH_BBOX = {
    "minlat": 35, "maxlat": 55,
    "westernlon": 180, "easternlon": 215,
}


def load_existing_ids() -> set:
    if not DTM_INDEX.exists():
        return set()
    with open(DTM_INDEX) as f:
        idx = json.load(f)
    return {feat["properties"]["product_id"] for feat in idx["features"]}


def load_existing_files() -> set:
    """Get set of filenames already on disk."""
    return {f.name for f in DTM_DIR.glob("*.IMG") if f.stat().st_size > 0}


def lon_to_180(lon: float) -> float:
    return lon - 360 if lon > 180 else lon


async def query_ode_dtms() -> List[Dict]:
    """Query ODE for HiRISE DTM products in the search bbox."""
    url = (
        f"{ODE_REST}?target=mars&ihid=mro&iid=hirise&pt=DTM&"
        f"minlat={SEARCH_BBOX['minlat']}&maxlat={SEARCH_BBOX['maxlat']}&"
        f"westernlon={SEARCH_BBOX['westernlon']}&easternlon={SEARCH_BBOX['easternlon']}&"
        f"output=json&results=pmf&limit=500"
    )
    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=90)) as r:
            data = await r.json()

    products = data.get("ODEResults", {}).get("Products", {}).get("Product", [])
    if isinstance(products, dict):
        products = [products]
    return products


def parse_footprint_wkt(wkt: str) -> Optional[List]:
    """Parse WKT POLYGON into coordinate ring [[lon,lat], ...]."""
    numbers = re.findall(r'(-?\d+\.?\d*)\s+(-?\d+\.?\d*)', str(wkt))
    if not numbers:
        return None
    return [[float(n[0]), float(n[1])] for n in numbers]


def extract_files_from_product(product: Dict) -> Dict[str, Optional[str]]:
    """Extract DTM and ortho file URLs directly from ODE REST response."""
    files = {"dtm_url": None, "dtm_filename": None, "ortho_url": None, "ortho_filename": None}

    # Method 1: Use LabelURL (direct link to DTM .IMG)
    label_url = product.get("LabelURL", "")
    label_fn = product.get("LabelFileName", "")
    if label_url and label_fn:
        fn_upper = label_fn.upper()
        if fn_upper.startswith("DTE") and fn_upper.endswith(".IMG"):
            files["dtm_url"] = label_url
            files["dtm_filename"] = label_fn.upper()

    # Method 2: Parse Product_files for DTM + ortho
    pf = product.get("Product_files", {})
    file_list = pf.get("Product_file", [])
    if isinstance(file_list, dict):
        file_list = [file_list]

    for entry in file_list:
        fn = entry.get("FileName", "")
        url = entry.get("URL", "")
        fn_upper = fn.upper()

        if fn_upper.startswith("DTE") and fn_upper.endswith(".IMG") and not files["dtm_url"]:
            files["dtm_url"] = url
            files["dtm_filename"] = fn

        if fn_upper.endswith("_ORTHO.JP2") and "_C_" in fn_upper and not files["ortho_url"]:
            files["ortho_url"] = url
            files["ortho_filename"] = fn

    # Fallback: any ortho JP2
    if not files["ortho_url"]:
        for entry in file_list:
            fn = entry.get("FileName", "")
            url = entry.get("URL", "")
            if fn.upper().endswith("_ORTHO.JP2"):
                files["ortho_url"] = url
                files["ortho_filename"] = fn
                break

    return files


def build_feature(product: Dict, dtm_filename: str, ortho_filename: str) -> Dict:
    """Build a GeoJSON feature from ODE product metadata + downloaded file."""
    pid = product.get("pdsid", "")
    center_lat = float(product.get("Center_latitude", 0))
    center_lon = float(product.get("Center_longitude", 0))

    # Extract footprint
    footprint_wkt = product.get("Footprint_C0_geometry", "")
    coords = parse_footprint_wkt(footprint_wkt)

    if coords and len(coords) >= 4:
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        west, east = min(lons), max(lons)
        south, north = min(lats), max(lats)
        ring = [[west, south], [east, south], [east, north], [west, north], [west, south]]
    else:
        clon = lon_to_180(center_lon)
        west, east = clon - 0.1, clon + 0.1
        south, north = center_lat - 0.1, center_lat + 0.1
        ring = [[west, south], [east, south], [east, north], [west, north], [west, south]]

    ext_url = product.get("External_url", "")

    # Try to read resolution from the DTM file
    resolution_m = 1.0
    dtm_path = DTM_DIR / dtm_filename
    if dtm_path.exists():
        try:
            import rasterio
            with rasterio.open(dtm_path) as ds:
                resolution_m = round(abs(ds.res[0]) * 3389500 * 3.14159 / 180, 2)
        except Exception:
            pass

    return {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [ring]},
        "properties": {
            "product_id": pid,
            "title": f"HiRISE DTM - {pid}",
            "dtm_file": dtm_filename,
            "ortho_file": ortho_filename,
            "source_url": ext_url or "",
            "west": west, "east": east,
            "south": south, "north": north,
            "resolution_m": resolution_m,
        }
    }


def run_aria2(input_file: str, label: str = "") -> bool:
    """Run aria2c with an input file for batch download."""
    cmd = [
        "aria2c",
        "-x", "16",              # 16 connections per server
        "-s", "16",              # 16 segments per file
        "-j", "4",               # 4 concurrent downloads
        "-k", "1M",              # 1MB min split
        "--continue=true",       # Resume partial downloads
        "--file-allocation=trunc",
        "--auto-file-renaming=false",
        "--allow-overwrite=false",
        "--retry-wait=5",
        "--max-tries=5",
        "--connect-timeout=30",
        "--timeout=600",
        "--max-overall-download-limit=0",  # No speed limit
        "-i", input_file,
    ]
    print(f"\n  Running aria2c {label}...", flush=True)
    print(f"  Command: aria2c -x16 -s16 -j4 -i {input_file}", flush=True)
    result = subprocess.run(cmd, timeout=7200)  # 2 hour timeout
    return result.returncode == 0


def save_index(new_features: List[Dict]):
    """Save index.geojson with new features appended."""
    if DTM_INDEX.exists():
        with open(DTM_INDEX) as f:
            index = json.load(f)
    else:
        index = {"type": "FeatureCollection", "features": []}

    existing_in_index = {feat["properties"]["product_id"] for feat in index["features"]}
    added = 0
    for feat in new_features:
        pid = feat["properties"]["product_id"]
        if pid not in existing_in_index:
            index["features"].append(feat)
            existing_in_index.add(pid)
            added += 1

    with open(DTM_INDEX, "w") as f:
        json.dump(index, f, indent=2)

    print(f"  Added {added} new entries to index.geojson", flush=True)
    print(f"  Total in index: {len(index['features'])}", flush=True)


async def main():
    DTM_DIR.mkdir(parents=True, exist_ok=True)

    # Check aria2c
    try:
        v = subprocess.run(["aria2c", "--version"], capture_output=True, text=True, timeout=5)
        ver = v.stdout.split("\n")[0] if v.returncode == 0 else "unknown"
    except FileNotFoundError:
        print("ERROR: aria2c not found. Install with: sudo apt install aria2", flush=True)
        return

    print("=" * 70, flush=True)
    print("  HiRISE DTM Bulk Download — Arcadia Planitia (aria2)", flush=True)
    print(f"  {ver}", flush=True)
    print("=" * 70, flush=True)

    existing_ids = load_existing_ids()
    existing_files = load_existing_files()
    print(f"\n  Existing DTMs in index: {len(existing_ids)}", flush=True)
    print(f"  Existing .IMG files on disk: {len(existing_files)}", flush=True)

    # Step 1: Query ODE
    print(f"\n  [1] Querying ODE for HiRISE DTMs...", flush=True)
    products = await query_ode_dtms()
    print(f"  Found {len(products)} DTMs in Arcadia Planitia (ODE)", flush=True)

    # Filter to products not yet in index
    new_products = [p for p in products if p.get("pdsid", "") not in existing_ids]
    print(f"  Not yet in index: {len(new_products)}", flush=True)

    if not new_products:
        print("\n  All DTMs already in index!", flush=True)
        return

    # Step 2: Extract URLs and build aria2 input file
    print(f"\n  [2] Extracting download URLs...", flush=True)
    download_plan = []  # (product, dtm_url, dtm_filename, ortho_url, ortho_filename)

    for p in new_products:
        pid = p.get("pdsid", "?")
        files = extract_files_from_product(p)
        if not files["dtm_url"]:
            print(f"    {pid}: No DTM .IMG URL found — skipping", flush=True)
            continue

        dtm_fn = files["dtm_filename"]
        already_on_disk = dtm_fn in existing_files

        lat = p.get("Center_latitude", "?")
        lon = lon_to_180(float(p.get("Center_longitude", 0)))
        status = "ON DISK" if already_on_disk else "TO DOWNLOAD"
        print(f"    {pid}  lat={lat} lon={lon:.2f}  [{status}]", flush=True)

        download_plan.append((p, files))

    if not download_plan:
        print("\n  No downloadable DTMs found.", flush=True)
        return

    # Build aria2 input file for DTMs not yet on disk
    dtm_entries = []
    ortho_entries = []
    for product, files in download_plan:
        dtm_fn = files["dtm_filename"]
        if dtm_fn not in existing_files:
            dtm_entries.append((files["dtm_url"], dtm_fn))
        if files["ortho_url"] and files["ortho_filename"]:
            ortho_fn = files["ortho_filename"]
            if not (DTM_DIR / ortho_fn).exists():
                ortho_entries.append((files["ortho_url"], ortho_fn))

    # Step 3: Download DTMs via aria2c
    if dtm_entries:
        print(f"\n  [3] Downloading {len(dtm_entries)} DTM files via aria2c...", flush=True)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, prefix="aria2_dtm_") as f:
            aria2_input = f.name
            for url, fn in dtm_entries:
                f.write(f"{url}\n")
                f.write(f"  out={fn}\n")
                f.write(f"  dir={DTM_DIR}\n")

        try:
            run_aria2(aria2_input, label=f"({len(dtm_entries)} DTM files)")
        finally:
            Path(aria2_input).unlink(missing_ok=True)
    else:
        print(f"\n  [3] All DTM files already on disk.", flush=True)

    # Step 4: Download ortho images via aria2c
    if ortho_entries:
        print(f"\n  [4] Downloading {len(ortho_entries)} ortho files via aria2c...", flush=True)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, prefix="aria2_ortho_") as f:
            aria2_input = f.name
            for url, fn in ortho_entries:
                f.write(f"{url}\n")
                f.write(f"  out={fn}\n")
                f.write(f"  dir={DTM_DIR}\n")

        try:
            run_aria2(aria2_input, label=f"({len(ortho_entries)} ortho files)")
        finally:
            Path(aria2_input).unlink(missing_ok=True)
    else:
        print(f"\n  [4] No ortho files to download.", flush=True)

    # Step 5: Build GeoJSON features for all successfully downloaded products
    print(f"\n  [5] Updating index.geojson...", flush=True)
    new_features = []
    for product, files in download_plan:
        dtm_fn = files["dtm_filename"]
        dtm_path = DTM_DIR / dtm_fn
        if dtm_path.exists() and dtm_path.stat().st_size > 0:
            ortho_fn = files["ortho_filename"] or ""
            feat = build_feature(product, dtm_fn, ortho_fn)
            new_features.append(feat)
            size_mb = dtm_path.stat().st_size / 1024 / 1024
            print(f"    {product.get('pdsid', '?')}: {size_mb:.0f} MB", flush=True)
        else:
            print(f"    {product.get('pdsid', '?')}: MISSING — download failed", flush=True)

    save_index(new_features)

    # Summary
    total_size = sum(
        f.stat().st_size for f in DTM_DIR.glob("*.IMG") if f.is_file()
    ) / 1024 / 1024 / 1024
    print(f"\n{'=' * 70}", flush=True)
    print(f"  SUMMARY", flush=True)
    print(f"{'=' * 70}", flush=True)
    print(f"  ODE DTMs found: {len(products)}", flush=True)
    print(f"  Previously in index: {len(existing_ids)}", flush=True)
    print(f"  New features added: {len(new_features)}", flush=True)
    print(f"  Total DTM data on disk: {total_size:.1f} GB", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
