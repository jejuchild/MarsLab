#!/usr/bin/env python3
"""
Download CRISM TRR3 + DDR data for mineral CNN classification.

Phase 1: Queries ODE REST API for CRISM TRDR observations, discovers file URLs.
Phase 2: Generates aria2 input file and downloads with aria2c (16 connections/file).

Usage:
    python download_crism_trr3.py                         # Arcadia Planitia, 50 obs
    python download_crism_trr3.py --max-obs 20            # Limit to 20
    python download_crism_trr3.py --region jezero_crater   # Different region
    python download_crism_trr3.py --dry-run               # List without downloading
    python download_crism_trr3.py --bbox 38 60 190 220    # Custom bbox (ODE 0-360 lon)
"""

import argparse
import asyncio
import os
import re
import subprocess
import sys
import tempfile
import time

import aiohttp

# ─── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(BACKEND_DIR, "mineral_cnn_data")

# ─── ODE endpoint ───────────────────────────────────────────────────────────
ODE_REST = "https://oderest.rsl.wustl.edu/live2"

# ─── Predefined regions (minlat, maxlat, westernlon_360, easternlon_360) ────
REGIONS = {
    "arcadia_planitia":    (38, 60, 190, 220),
    "arcadia_wide":        (25, 75, 160, 250),
    "utopia_planitia":     (30, 55, 100, 140),
    "amazonis_planitia":   (15, 40, 190, 215),
    "elysium_planitia":    (0, 20, 145, 175),
    "jezero_crater":       (17, 20, 76, 79),
    "isidis_planitia":     (5, 20, 82, 95),
    "acidalia_planitia":   (35, 55, 325, 355),
    "hellas_basin":        (-55, -30, 55, 80),
    "valles_marineris":    (-15, 0, 275, 310),
    "syrtis_major":        (5, 22, 65, 78),
    "deuteronilus_mensae": (38, 50, 15, 40),
    "nili_fossae":         (18, 25, 73, 80),
}


def parse_args():
    p = argparse.ArgumentParser(description="Download CRISM TRR3 + DDR for mineral CNN")
    p.add_argument("--region", default="arcadia_planitia",
                    help=f"Predefined region. Available: {', '.join(REGIONS.keys())}")
    p.add_argument("--bbox", nargs=4, type=float,
                    metavar=("MINLAT", "MAXLAT", "WLON360", "ELON360"),
                    help="Custom bbox (overrides --region). Lon in 0-360 ODE format.")
    p.add_argument("--max-obs", type=int, default=50,
                    help="Max observations to download (default: 50)")
    p.add_argument("--dry-run", action="store_true",
                    help="List products without downloading")
    p.add_argument("--concurrent", type=int, default=4,
                    help="Max concurrent file downloads for aria2 (default: 4)")
    p.add_argument("--type", default="frt",
                    help="Filter by observation type prefix, e.g. 'frt', 'hrl' (default: frt)")
    p.add_argument("--no-post-process", action="store_true",
                    help="Skip post-download CNN classification + quickview generation")
    return p.parse_args()


def _float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def extract_base_key(pdsid: str) -> str:
    """frt0001fd76_07_if166l_trr3 -> frt0001fd76_07"""
    parts = pdsid.lower().split("_")
    return f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else pdsid.lower()


def derive_ddr_pdsid(trr_pdsid: str) -> str:
    """frt00009312_07_if166l_trr3 -> frt00009312_07_de166l_ddr1"""
    parts = trr_pdsid.lower().split("_")
    if len(parts) >= 4:
        activity = parts[2]
        return f"{parts[0]}_{parts[1]}_de{activity[2:]}_ddr1"
    return ""


# ─── ODE queries ─────────────────────────────────────────────────────────────

async def discover_targeted_obs(session, minlat, maxlat, wlon, elon, obs_type=None):
    """Discover targeted CRISM observations via MTRDR product type.

    Returns list of base_keys (e.g. 'frt00004fb0_07') with lat/lon.
    This avoids the TRDR limit problem (33k+ products dominated by MSV/HSV).
    """
    url = (
        f"{ODE_REST}?"
        f"target=mars&ihid=mro&iid=crism&pt=MTRDR&"
        f"minlat={minlat}&maxlat={maxlat}&"
        f"westernlon={wlon}&easternlon={elon}&"
        f"output=json&results=fpm&limit=800"
    )
    print(f"Discovering targeted observations via MTRDR ...")

    async with session.get(url, timeout=aiohttp.ClientTimeout(total=300)) as resp:
        if resp.status != 200:
            print(f"  ODE returned HTTP {resp.status}")
            return []
        data = await resp.json()

    ode = data.get("ODEResults", {})
    if ode.get("Status") != "Success":
        print(f"  ODE query failed: {ode.get('Status')}")
        return []

    product_list = ode.get("Products", {}).get("Product", [])
    if isinstance(product_list, dict):
        product_list = [product_list]

    # Deduplicate by base_key
    seen = set()
    observations = []
    for p in product_list:
        pdsid = p.get("pdsid", "")
        bk = extract_base_key(pdsid)
        prefix = bk.split("0")[0] if "0" in bk else bk[:3]

        if obs_type and not bk.startswith(obs_type.lower()):
            continue
        if bk in seen:
            continue
        seen.add(bk)
        observations.append({
            "base_key": bk,
            "type": prefix,
            "lat": _float(p.get("Center_latitude")),
            "lon": _float(p.get("Center_longitude")),
        })

    # Count by type
    type_counts = {}
    for o in observations:
        type_counts[o["type"]] = type_counts.get(o["type"], 0) + 1
    type_str = ", ".join(f"{t}: {c}" for t, c in sorted(type_counts.items(), key=lambda x: -x[1]))

    print(f"  {len(product_list)} MTRDR products -> {len(observations)} unique observations ({type_str})")
    return observations


async def query_trdr_for_obs(session, base_key):
    """Query ODE for TRDR TRR3 files for a specific observation base_key.

    Derives the L-sensor TRDR product ID from the base_key and queries ODE.
    """
    # Query using productid prefix search
    url = (
        f"{ODE_REST}?"
        f"target=mars&ihid=mro&iid=crism&pt=TRDR&"
        f"productid={base_key}*&"
        f"output=json&results=fpm&limit=20"
    )
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()
    except Exception as e:
        print(f"    TRDR query error for {base_key}: {e}")
        return None

    ode = data.get("ODEResults", {})
    prods = ode.get("Products", {}).get("Product", [])
    if isinstance(prods, dict):
        prods = [prods]
    if not prods:
        return None

    # Find the L-sensor I/F TRR3 product
    _IF_L_RE = re.compile(r"_if\d+l_trr3", re.IGNORECASE)
    for p in prods:
        pdsid = p.get("pdsid", "")
        if not _IF_L_RE.search(pdsid.lower()):
            continue

        pf = p.get("Product_files", {}).get("Product_file", [])
        if isinstance(pf, dict):
            pf = [pf]

        files = {}
        for f in pf:
            name = f.get("FileName", "").upper()
            f_url = f.get("URL", "")
            kb = int(f.get("KBytes", 0))
            if name.endswith("_TRR3.IMG"):
                files["trr3_img"] = {"url": f_url, "name": f.get("FileName"), "size": kb * 1024}
            elif name.endswith("_TRR3.LBL"):
                files["trr3_lbl"] = {"url": f_url, "name": f.get("FileName"), "size": kb * 1024}

        if "trr3_img" in files and "trr3_lbl" in files:
            return {
                "pdsid": pdsid,
                "lat": _float(p.get("Center_latitude")),
                "lon": _float(p.get("Center_longitude")),
                "files": files,
            }

    return None


async def query_ddr_files(session, ddr_pdsid):
    """Query ODE for DDR product file URLs."""
    url = (
        f"{ODE_REST}?"
        f"target=mars&ihid=mro&iid=crism&"
        f"productid={ddr_pdsid}&"
        f"output=json&results=fpm&limit=1"
    )
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()
    except Exception as e:
        print(f"    DDR query error: {e}")
        return None

    ode = data.get("ODEResults", {})
    prods = ode.get("Products", {}).get("Product", [])
    if isinstance(prods, dict):
        prods = [prods]
    if not prods:
        return None

    pf = prods[0].get("Product_files", {}).get("Product_file", [])
    if isinstance(pf, dict):
        pf = [pf]

    files = {}
    for f in pf:
        name = f.get("FileName", "").upper()
        f_url = f.get("URL", "")
        kb = int(f.get("KBytes", 0))
        if name.endswith("_DDR1.IMG"):
            files["ddr_img"] = {"url": f_url, "name": f.get("FileName"), "size": kb * 1024}
        elif name.endswith("_DDR1.LBL"):
            files["ddr_lbl"] = {"url": f_url, "name": f.get("FileName"), "size": kb * 1024}

    return files if "ddr_img" in files and "ddr_lbl" in files else None


# ─── aria2 download ─────────────────────────────────────────────────────────

def run_aria2_download(download_list, concurrent=4):
    """
    Download all files using aria2c for maximum throughput.

    download_list: list of (url, output_dir, filename) tuples
    """
    if not download_list:
        return 0, 0

    # Write aria2 input file
    # Format:  URL\n  dir=...\n  out=...\n\n
    input_path = os.path.join(DATA_DIR, "_aria2_input.txt")
    with open(input_path, "w") as f:
        for url, out_dir, filename in download_list:
            f.write(f"{url}\n")
            f.write(f"  dir={out_dir}\n")
            f.write(f"  out={filename}\n")
            f.write(f"\n")

    print(f"\nStarting aria2c: {len(download_list)} files, "
          f"{concurrent} concurrent, 16 connections/file ...")
    print(f"Input file: {input_path}\n")

    cmd = [
        "aria2c",
        "-i", input_path,
        "-j", str(concurrent),     # concurrent downloads
        "-x", "16",                # connections per server
        "-s", "16",                # split segments
        "-k", "1M",                # min split size
        "--continue=true",
        "--file-allocation=trunc",
        "--auto-file-renaming=false",
        "--allow-overwrite=true",
        "--retry-wait=3",
        "--max-tries=5",
        "--timeout=300",
        "--connect-timeout=30",
        "--summary-interval=10",
    ]

    try:
        result = subprocess.run(cmd, cwd=DATA_DIR)
        # Clean up input file
        os.remove(input_path)

        if result.returncode == 0:
            return len(download_list), 0
        else:
            print(f"\naria2c exited with code {result.returncode}")
            # Count actual downloads by checking files
            ok = sum(1 for _, d, fn in download_list if os.path.exists(os.path.join(d, fn)))
            return ok, len(download_list) - ok

    except FileNotFoundError:
        print("ERROR: aria2c not found! Install with: conda install -c conda-forge aria2")
        os.remove(input_path)
        return 0, len(download_list)


# ─── Main ────────────────────────────────────────────────────────────────────

async def main():
    args = parse_args()

    if args.bbox:
        minlat, maxlat, wlon, elon = args.bbox
        region_name = "custom"
    else:
        if args.region not in REGIONS:
            print(f"Unknown region: {args.region}")
            print(f"Available: {', '.join(REGIONS.keys())}")
            sys.exit(1)
        minlat, maxlat, wlon, elon = REGIONS[args.region]
        region_name = args.region

    print(f"=== CRISM TRR3+DDR Downloader (aria2) ===")
    print(f"Region: {region_name}")
    print(f"Bbox: lat [{minlat}, {maxlat}], lon [{wlon}, {elon}]")
    print(f"Max observations: {args.max_obs}")
    print(f"Output: {DATA_DIR}")
    if args.dry_run:
        print("*** DRY RUN ***")
    print()

    os.makedirs(DATA_DIR, exist_ok=True)

    async with aiohttp.ClientSession() as session:
        # ── Phase 1: Discover targeted observations via MTRDR ─────────
        observations = await discover_targeted_obs(
            session, minlat, maxlat, wlon, elon,
            obs_type=args.type,
        )
        if not observations:
            print("No targeted CRISM observations found.")
            return

        observations = observations[:args.max_obs]

        # Skip existing (already have L-sensor TRR3 + DDR)
        _L_CHECK = re.compile(r"_if\d+l_trr3\.img$", re.IGNORECASE)
        to_discover = []
        skipped = 0
        for obs in observations:
            bk = obs["base_key"]
            dest_dir = os.path.join(DATA_DIR, bk)
            if os.path.isdir(dest_dir):
                existing = os.listdir(dest_dir)
                has_l_trr3 = any(_L_CHECK.search(f) for f in existing)
                has_ddr = any(f.lower().endswith("_ddr1.img") for f in existing)
                if has_l_trr3 and has_ddr:
                    skipped += 1
                    continue
            to_discover.append(obs)

        print(f"\n{len(observations)} targeted observations, {skipped} already on disk")
        print(f"Discovering L-sensor TRR3 + DDR files for {len(to_discover)} observations ...\n")

        # Discover TRR3 + DDR files for each observation
        download_list = []  # (url, output_dir, filename)
        total_size = 0
        to_process = []
        no_trr3 = 0

        for i, obs in enumerate(to_discover, 1):
            bk = obs["base_key"]
            dest_dir = os.path.join(DATA_DIR, bk)
            lat_s = f"{obs['lat']:.2f}" if obs['lat'] else "?"
            lon_s = f"{obs['lon']:.2f}" if obs['lon'] else "?"

            # Query ODE for L-sensor TRR3 files
            trdr = await query_trdr_for_obs(session, bk)
            if not trdr:
                no_trr3 += 1
                print(f"  [{i}/{len(to_discover)}] {bk}: no L-sensor TRR3 found, skipping")
                continue

            files = dict(trdr["files"])

            # Query DDR
            ddr_pdsid = derive_ddr_pdsid(trdr["pdsid"])
            if ddr_pdsid:
                ddr_files = await query_ddr_files(session, ddr_pdsid)
                if ddr_files:
                    files.update(ddr_files)

            obs_size = sum(f.get("size", 0) for f in files.values())
            total_size += obs_size
            file_types = sorted(files.keys())

            print(f"  [{i}/{len(to_discover)}] {bk} (lat={lat_s}, lon={lon_s}): "
                  f"{', '.join(file_types)} ({obs_size / (1024*1024):.0f} MB)")

            to_process.append(obs)
            if not args.dry_run:
                os.makedirs(dest_dir, exist_ok=True)
                for key, f in files.items():
                    dest_path = os.path.join(dest_dir, f["name"])
                    # Skip if already downloaded with correct size
                    if os.path.exists(dest_path):
                        existing_size = os.path.getsize(dest_path)
                        if f["size"] and abs(existing_size - f["size"]) < 1024:
                            continue
                    download_list.append((f["url"], dest_dir, f["name"]))

        if no_trr3:
            print(f"\n  {no_trr3} observations had no L-sensor TRR3 available")

    print(f"\nTotal: {len(to_process)} observations, "
          f"{total_size / (1024*1024*1024):.1f} GB, "
          f"{len(download_list)} files to download")

    if args.dry_run:
        print("Dry run complete.")
        return

    if not download_list:
        print("All files already downloaded!")
        return

    # ── Phase 2: aria2 batch download ────────────────────────────────
    t0 = time.time()
    ok, failed = run_aria2_download(download_list, concurrent=args.concurrent)
    elapsed = time.time() - t0

    print(f"\n=== Download Done ({elapsed:.0f}s) ===")
    print(f"Downloaded: {ok} files")
    print(f"Failed: {failed} files")
    print(f"Skipped (existing): {skipped} observations")
    print(f"Data: {DATA_DIR}")

    # ── Phase 3: Post-process (CNN classification + quickview) ────
    if not args.no_post_process:
        print(f"\n=== Phase 3: Post-processing (CNN + Quickview) ===")
        await _run_post_process()


async def _run_post_process():
    """Run batch classification + quickview for all downloaded observations."""
    # Import here to avoid circular/heavy imports when just downloading
    sys.path.insert(0, BACKEND_DIR)
    from api.mineral_cnn.pipeline import run_classification, has_cached_result
    from api.mineral_cnn.acquire import _generate_quickview
    from api.mineral_cnn.data_loader import resolve_trr_files
    import numpy as np

    # Discover L-sensor observations (CNN requires IR data with 438+ bands)
    _L_SENSOR_RE = re.compile(r"_if\d+l_", re.IGNORECASE)
    obs_ids = []
    skipped_s = 0
    for entry in sorted(os.listdir(DATA_DIR)):
        d = os.path.join(DATA_DIR, entry)
        if not os.path.isdir(d):
            continue
        files = os.listdir(d)
        has_trr3 = any(f.upper().endswith("_TRR3.IMG") for f in files)
        has_ddr = any(f.upper().endswith("_DDR1.IMG") for f in files)
        if has_trr3 and has_ddr:
            # Check for L-sensor (IR) data — S-sensor (VNIR) can't be classified
            has_l = any(_L_SENSOR_RE.search(f) for f in files)
            if has_l:
                obs_ids.append(entry)
            else:
                skipped_s += 1

    if skipped_s:
        print(f"  Skipped {skipped_s} S-sensor/VNIR observations (CNN requires L-sensor/IR)")
    if not obs_ids:
        print("No L-sensor TRR3+DDR observations to process.")
        return

    total = len(obs_ids)
    qv_gen, qv_cached, qv_fail = 0, 0, 0
    cnn_gen, cnn_cached, cnn_fail = 0, 0, 0

    for i, obs_id in enumerate(obs_ids, 1):
        print(f"\n[{i}/{total}] {obs_id}")

        # Quickview
        try:
            files = resolve_trr_files(obs_id)
            obs_dir = os.path.dirname(files["trr_img"])
            qv_path = os.path.join(obs_dir, "quickview.png")
            if os.path.exists(qv_path):
                qv_cached += 1
            else:
                result = _generate_quickview(obs_dir, files["trr_img"], files["trr_lbl"])
                if result:
                    qv_gen += 1
                    print(f"  Quickview: generated")
                else:
                    qv_fail += 1
                    print(f"  Quickview: FAILED")
        except Exception as e:
            qv_fail += 1
            print(f"  Quickview error: {e}")

        # CNN classification
        if has_cached_result(obs_id):
            cnn_cached += 1
        else:
            try:
                t_cnn = time.time()
                async for event in run_classification(obs_id):
                    evt = event.get("event", "")
                    data = event.get("data", {})
                    if evt == "status":
                        print(f"  {data.get('message', '')}")
                    elif evt == "progress":
                        pct = data.get("percent", 0)
                        if pct % 25 < 1:
                            print(f"  JCAT: {pct:.0f}%")
                    elif evt == "complete":
                        c = data.get("classified_pixels", 0)
                        v = data.get("valid_pixels", 0)
                        print(f"  CNN: {c}/{v} pixels ({time.time() - t_cnn:.1f}s)")
                        cnn_gen += 1
                    elif evt == "error":
                        print(f"  CNN ERROR: {data.get('error', '')}")
                        cnn_fail += 1
            except Exception as e:
                cnn_fail += 1
                print(f"  CNN error: {e}")

    print(f"\n=== Post-processing Done ===")
    print(f"Quickview: {qv_gen} generated, {qv_cached} cached, {qv_fail} failed")
    print(f"CNN:       {cnn_gen} classified, {cnn_cached} cached, {cnn_fail} failed")


if __name__ == "__main__":
    asyncio.run(main())
