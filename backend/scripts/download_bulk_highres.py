#!/usr/bin/env python3
"""
Bulk high-resolution data downloader for MarsLab.

Downloads:
  1. Jezero Crater — SHARAD High-Res RDR (.dat + .lbl)
  2. Jezero Crater — CRISM TRR3 + DDR (L-sensor I/F)
  3. Jezero Crater — HiRISE RDR RED (.JP2 + .lbl → GeoTIFF)
  4. Arcadia Planitia — HiRISE RDR RED (.JP2 + .lbl → GeoTIFF)

Usage:
    python download_bulk_highres.py                    # Run all 4 targets
    python download_bulk_highres.py --target jezero-sharad
    python download_bulk_highres.py --target jezero-hirise --max-obs 50
    python download_bulk_highres.py --dry-run          # List without downloading
    python download_bulk_highres.py --skip-gdal         # Skip JP2→GeoTIFF conversion
"""

import argparse
import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

import aiohttp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_lib import (  # noqa: E402  (shared with background_fetch.py)
    ODE_REST,
    _float,
    convert_jp2_to_tif,
    run_aria2,
    update_hirise_index,
    update_sharad_index,
)

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
BACKEND_DIR = SCRIPT_DIR.parent

HIRISE_DATA_DIR = BACKEND_DIR / "hirise_data"
HIRISE_INDEX = HIRISE_DATA_DIR / "index.geojson"

SHARAD_HR_DIR = BACKEND_DIR / "sharad_highres"
SHARAD_HR_INDEX = BACKEND_DIR / "sharad_highres_data" / "index.geojson"

CRISM_CNN_DIR = BACKEND_DIR / "mineral_cnn_data"

# ─── Region definitions (minlat, maxlat, westernlon_360, easternlon_360) ─────
REGIONS = {
    "jezero":  (17, 20, 76, 79),
    "arcadia": (38, 55, 190, 220),
}

# ─── Target definitions ──────────────────────────────────────────────────────
TARGETS = [
    "jezero-sharad",
    "jezero-crism",
    "jezero-hirise",
    "arcadia-hirise",
]


def parse_args():
    p = argparse.ArgumentParser(description="Bulk high-res Mars data downloader")
    p.add_argument("--target", choices=TARGETS + ["all"], default="all",
                   help="Which target to download (default: all)")
    p.add_argument("--max-obs", type=int, default=200,
                   help="Max observations per target (default: 200)")
    p.add_argument("--dry-run", action="store_true",
                   help="List products without downloading")
    p.add_argument("--skip-gdal", action="store_true",
                   help="Skip JP2 → GeoTIFF conversion")
    p.add_argument("--concurrent", type=int, default=4,
                   help="Max concurrent aria2 downloads (default: 4)")
    return p.parse_args()


# ═════════════════════════════════════════════════════════════════════════════
# Utility functions
# ═════════════════════════════════════════════════════════════════════════════

async def ode_query_with_retry(session, url, retries=3, timeout=120):
    """Query ODE with retry on transient errors."""
    for attempt in range(retries):
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                if resp.status != 200:
                    print(f"    ODE HTTP {resp.status} (attempt {attempt + 1}/{retries})")
                    if attempt < retries - 1:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    return None
                return await resp.json()
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            print(f"    ODE error: {type(e).__name__} (attempt {attempt + 1}/{retries})")
            if attempt < retries - 1:
                await asyncio.sleep(2 ** attempt)
            else:
                print(f"    ODE failed after {retries} attempts")
                return None
    return None


# ═════════════════════════════════════════════════════════════════════════════
# 1) SHARAD High-Res for Jezero
# ═════════════════════════════════════════════════════════════════════════════

async def download_jezero_sharad(args):
    """Download SHARAD High-Res RDR tracks over Jezero Crater."""
    minlat, maxlat, wlon, elon = REGIONS["jezero"]
    # Expand bbox for SHARAD (long orbital tracks)
    margin = 3.0
    q_minlat = max(-90, minlat - margin)
    q_maxlat = min(90, maxlat + margin)
    q_wlon = wlon - margin
    q_elon = elon + margin

    print(f"\n{'='*60}")
    print(f"  [1/4] SHARAD High-Res — Jezero Crater")
    print(f"  Search bbox: lat [{q_minlat},{q_maxlat}], lon [{q_wlon},{q_elon}]")
    print(f"{'='*60}")

    SHARAD_HR_DIR.mkdir(parents=True, exist_ok=True)
    SHARAD_HR_INDEX.parent.mkdir(parents=True, exist_ok=True)

    # Load existing products
    existing_on_disk = set()
    if SHARAD_HR_DIR.exists():
        for f in SHARAD_HR_DIR.iterdir():
            if f.suffix == ".dat":
                existing_on_disk.add(f.stem.upper())

    async with aiohttp.ClientSession() as session:
        # Query ODE for SHARAD RDR tracks (metadata only for discovery)
        url = (
            f"{ODE_REST}?"
            f"target=mars&ihid=mro&iid=sharad&pt=RDR&"
            f"minlat={q_minlat}&maxlat={q_maxlat}&"
            f"westernlon={q_wlon}&easternlon={q_elon}&"
            f"output=json&results=m&limit=500"
        )

        print(f"  Querying ODE ...")
        data = await ode_query_with_retry(session, url)
        if not data:
            print(f"  ODE query failed")
            return

        products = data.get("ODEResults", {}).get("Products", {}).get("Product", [])
        if isinstance(products, dict):
            products = [products]
        print(f"  Found {len(products)} SHARAD tracks in bbox")

        if not products:
            return

        # Extract product IDs from LabelURL (SHARAD RDR has no Product_id field)
        tracks = []
        for p in products[:args.max_obs]:
            label_url = p.get("LabelURL", "")
            if not label_url:
                continue
            # Extract filename from URL: .../r_0400201_001_ss19_700_a.lbl
            pid = label_url.rsplit("/", 1)[-1]
            # Strip extension (.lbl) if present
            if "." in pid:
                pid = pid.rsplit(".", 1)[0]
            pid = pid.upper()
            if not pid:
                continue
            lat = _float(p.get("Center_latitude"))
            lon = _float(p.get("Center_longitude"))
            footprint_wkt = p.get("Footprint_C0_geometry", "")
            tracks.append({
                "pid": pid, "lat": lat, "lon": lon,
                "label_url": label_url,
                "footprint_wkt": footprint_wkt,
                "product": p,
            })

        new_tracks = [t for t in tracks if t["pid"] not in existing_on_disk]
        print(f"  Total: {len(tracks)}, already on disk: {len(tracks) - len(new_tracks)}, "
              f"to download: {len(new_tracks)}")

        if args.dry_run or not new_tracks:
            for t in new_tracks[:20]:
                print(f"    {t['pid']}  lat={t['lat']}  lon={t['lon']}")
            if len(new_tracks) > 20:
                print(f"    ... and {len(new_tracks) - 20} more")
            return

        # Resolve download URLs from LabelURL (construct .dat URL from .lbl path)
        download_list = []
        for t in new_tracks:
            label_url = t["label_url"]
            if not label_url:
                print(f"    {t['pid']}: no LabelURL, skipping")
                continue
            pid_lower = t["pid"].lower()
            base_url = label_url.rsplit("/", 1)[0]
            dat_url = f"{base_url}/{pid_lower}.dat"
            lbl_url = label_url

            download_list.append((dat_url, str(SHARAD_HR_DIR), f"{pid_lower}.dat"))
            download_list.append((lbl_url, str(SHARAD_HR_DIR), f"{pid_lower}.lbl"))

        ok, fail = run_aria2(download_list, args.concurrent, label="(SHARAD)")
        print(f"  Downloaded: {ok}, Failed: {fail}")

        # Update index
        await _update_sharad_index(new_tracks)


async def _update_sharad_index(tracks: List[Dict]):
    """Add new SHARAD tracks to sharad_highres_data/index.geojson."""
    added, total = update_sharad_index(tracks, SHARAD_HR_DIR, SHARAD_HR_INDEX)
    print(f"  Index: +{added} entries (total: {total})")


# ═════════════════════════════════════════════════════════════════════════════
# 2) CRISM TRR3 for Jezero — delegates to existing script
# ═════════════════════════════════════════════════════════════════════════════

async def download_jezero_crism(args):
    """Download CRISM TRR3+DDR for Jezero via existing script."""
    print(f"\n{'='*60}")
    print(f"  [2/4] CRISM TRR3 + DDR — Jezero Crater")
    print(f"{'='*60}")

    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "download_crism_trr3.py"),
        "--region", "jezero_crater",
        "--max-obs", str(args.max_obs),
        "--concurrent", str(args.concurrent),
        "--type", "frt",
    ]
    if args.dry_run:
        cmd.append("--dry-run")
    # Skip post-processing for now (can run separately)
    cmd.append("--no-post-process")

    print(f"  Running: {' '.join(cmd[1:])}")
    result = subprocess.run(cmd, cwd=str(BACKEND_DIR))
    if result.returncode != 0:
        print(f"  CRISM download exited with code {result.returncode}")


# ═════════════════════════════════════════════════════════════════════════════
# 3) & 4) HiRISE RDR for a region
# ═════════════════════════════════════════════════════════════════════════════

async def download_hirise_region(args, region_name: str, step_label: str):
    """Download HiRISE RDR RED products for a bbox region."""
    minlat, maxlat, wlon, elon = REGIONS[region_name]

    print(f"\n{'='*60}")
    print(f"  [{step_label}] HiRISE RDR RED — {region_name.title()}")
    print(f"  Search bbox: lat [{minlat},{maxlat}], lon [{wlon},{elon}] (ODE 0-360)")
    print(f"{'='*60}")

    HIRISE_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing products
    existing_ids = set()
    if HIRISE_INDEX.exists():
        with open(HIRISE_INDEX) as f:
            idx = json.load(f)
        for feat in idx["features"]:
            existing_ids.add(feat["properties"].get("product_id", ""))

    # Also check on disk
    existing_files = {f.stem.replace("_RED", "") for f in HIRISE_DATA_DIR.glob("*_RED.JP2")}
    existing_files.update({f.stem.replace("_RED", "") for f in HIRISE_DATA_DIR.glob("*_RED.jp2")})
    existing_tifs = {f.stem.replace("_RED", "") for f in HIRISE_DATA_DIR.glob("*_RED.tif")}

    async with aiohttp.ClientSession() as session:
        # Query ODE for HiRISE RDRV11 products
        limit = min(args.max_obs * 3, 2000)  # Request 3x to account for RED filtering
        url = (
            f"{ODE_REST}?"
            f"target=mars&ihid=mro&iid=hirise&pt=RDRV11&"
            f"minlat={minlat}&maxlat={maxlat}&"
            f"westernlon={wlon}&easternlon={elon}&"
            f"output=json&results=pmf&limit={limit}"
        )

        print(f"  Querying ODE ...")
        data = await ode_query_with_retry(session, url)
        if not data:
            print(f"  ODE query failed")
            return

        products = data.get("ODEResults", {}).get("Products", {}).get("Product", [])
        if isinstance(products, dict):
            products = [products]

        # Filter to RED products only
        red_products = []
        seen_base = set()
        for p in products:
            pdsid = p.get("pdsid", "")
            if not pdsid.upper().endswith("_RED"):
                continue
            # Extract base ID: ESP_024943_2345_RED -> ESP_024943_2345
            base_id = pdsid.upper().replace("_RED", "")
            if base_id in seen_base:
                continue
            seen_base.add(base_id)
            red_products.append({"base_id": base_id, "pdsid": pdsid, "product": p})

        # Limit
        red_products = red_products[:args.max_obs]

        # Skip existing
        new_products = [rp for rp in red_products
                        if rp["base_id"] not in existing_ids
                        and rp["base_id"] not in existing_files]

        print(f"  Found {len(red_products)} RED products, "
              f"{len(red_products) - len(new_products)} already on disk, "
              f"{len(new_products)} to download")

        if args.dry_run:
            for rp in new_products[:20]:
                p = rp["product"]
                lat = p.get("Center_latitude", "?")
                lon = p.get("Center_longitude", "?")
                print(f"    {rp['base_id']}  lat={lat}  lon={lon}")
            if len(new_products) > 20:
                print(f"    ... and {len(new_products) - 20} more")
            return

        if not new_products:
            print(f"  All products already downloaded!")
            return

        # Extract JP2 + LBL URLs from Product_files
        download_list = []
        download_meta = []  # Track (base_id, jp2_filename, lat, lon, footprint)

        for rp in new_products:
            p = rp["product"]
            pf = p.get("Product_files", {}).get("Product_file", [])
            if isinstance(pf, dict):
                pf = [pf]

            jp2_url = jp2_fn = lbl_url = lbl_fn = None
            for f in pf:
                fn = f.get("FileName", "")
                furl = f.get("URL", "")
                fn_upper = fn.upper()

                # RED JP2 (map-projected, not NOMAP/QLOOK)
                if ("RED" in fn_upper and fn_upper.endswith(".JP2")
                        and "NOMAP" not in fn_upper and "QLOOK" not in fn_upper):
                    jp2_url, jp2_fn = furl, fn

                # RED label
                elif fn_upper.endswith(".LBL") and "RED" in fn_upper:
                    lbl_url, lbl_fn = furl, fn

            if not jp2_url:
                print(f"    {rp['base_id']}: no JP2 URL, skipping")
                continue

            download_list.append((jp2_url, str(HIRISE_DATA_DIR), jp2_fn))
            if lbl_url:
                download_list.append((lbl_url, str(HIRISE_DATA_DIR), lbl_fn))

            lat = _float(p.get("Center_latitude"))
            lon = _float(p.get("Center_longitude"))
            footprint_wkt = p.get("Footprint_C0_geometry", "")
            download_meta.append({
                "base_id": rp["base_id"],
                "jp2_fn": jp2_fn,
                "lat": lat,
                "lon": lon,
                "footprint_wkt": footprint_wkt,
            })

        # Download via aria2
        t0 = time.time()
        ok, fail = run_aria2(download_list, args.concurrent,
                             label=f"(HiRISE {region_name})")
        elapsed = time.time() - t0
        print(f"  Downloaded: {ok}, Failed: {fail} ({elapsed:.0f}s)")

        # JP2 → GeoTIFF conversion
        if not args.skip_gdal:
            print(f"\n  Converting JP2 → GeoTIFF ...")
            converted = 0
            for meta in download_meta:
                jp2_path = HIRISE_DATA_DIR / meta["jp2_fn"]
                if jp2_path.exists():
                    tif = convert_jp2_to_tif(jp2_path)
                    if tif:
                        converted += 1
                        size_mb = tif.stat().st_size / 1024 / 1024
                        print(f"    {meta['base_id']}: {size_mb:.0f} MB")
            print(f"  Converted: {converted}/{len(download_meta)}")

        # Update index
        _update_hirise_index(download_meta)


def _update_hirise_index(new_items: List[Dict]):
    """Add new HiRISE products to hirise_data/index.geojson."""
    added, total = update_hirise_index(new_items, HIRISE_DATA_DIR, HIRISE_INDEX)
    print(f"  Index: +{added} entries (total: {total})")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

async def main():
    args = parse_args()

    targets = TARGETS if args.target == "all" else [args.target]

    print("=" * 60)
    print("  MarsLab Bulk High-Res Downloader")
    print(f"  Targets: {', '.join(targets)}")
    print(f"  Max obs per target: {args.max_obs}")
    print(f"  Dry run: {args.dry_run}")
    print("=" * 60)

    t_total = time.time()

    for target in targets:
        if target == "jezero-sharad":
            await download_jezero_sharad(args)

        elif target == "jezero-crism":
            await download_jezero_crism(args)

        elif target == "jezero-hirise":
            await download_hirise_region(args, "jezero", "3/4")

        elif target == "arcadia-hirise":
            await download_hirise_region(args, "arcadia", "4/4")

    elapsed = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"  ALL DONE ({elapsed:.0f}s)")
    print(f"{'='*60}")

    # Print storage summary
    dirs = [
        ("SHARAD High-Res", SHARAD_HR_DIR, "*.dat"),
        ("CRISM TRR3", CRISM_CNN_DIR, "**/*.img"),
        ("HiRISE RDR", HIRISE_DATA_DIR, "*.JP2"),
        ("HiRISE TIF", HIRISE_DATA_DIR, "*.tif"),
    ]
    print(f"\n  Storage Summary:")
    for label, d, pattern in dirs:
        if d.exists():
            files = list(d.glob(pattern))
            size_gb = sum(f.stat().st_size for f in files) / 1024**3
            print(f"    {label:20s}: {len(files):4d} files, {size_gb:7.1f} GB")


if __name__ == "__main__":
    asyncio.run(main())
