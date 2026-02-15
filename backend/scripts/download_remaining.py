#!/usr/bin/env python3
"""Download remaining SHARAD High-Res products (retry failed ones)."""

import asyncio
import sys
import os
import json
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import aiohttp
import aiofiles
from api.ode_client import resolve_sharad_highres_bundle, get_sharad_highres_footprint

SHARAD_HR_DIR = Path(__file__).parent.parent / "sharad_highres"
SHARAD_HR_INDEX = Path(__file__).parent.parent / "sharad_highres_data" / "index.geojson"

# All 54 matching products from the ODE search
ALL_MATCHING = [
    'R_0577801_001_SS19_700_A', 'R_0614702_001_SS19_700_A', 'R_0650301_001_SS19_700_A',
    'R_0684602_001_SS19_700_A', 'R_0709601_001_SS19_700_A', 'R_0780801_001_SS19_700_A',
    'R_1274004_001_SS19_700_A', 'R_1297701_001_SS19_700_A', 'R_1332001_001_SS19_700_A',
    'R_1723601_001_SS19_700_A', 'R_2177201_001_SS19_700_A', 'R_2252401_001_SS04_700_A',
    'R_2336801_001_SS04_700_A', 'R_2466401_001_SS04_700_A', 'R_3578901_001_SS19_700_A',
    'R_3710801_001_SS19_700_A', 'R_3908602_001_SS19_700_A', 'R_3913901_001_SS19_700_A',
    'R_3933702_001_SS19_700_A', 'R_3940302_001_SS19_700_A', 'R_3985901_001_SS19_700_A',
    'R_4018103_001_SS19_700_A', 'R_4033901_001_SS19_700_A', 'R_4043102_001_SS19_700_A',
    'R_4893602_001_SS19_700_A', 'R_5488301_001_SS19_700_A', 'R_5677601_001_SS19_700_A',
    'R_5712501_001_SS19_700_A', 'R_5867501_001_SS19_700_A', 'R_6170101_001_SS4_700_A',
    'R_6523601_001_SS4_700_A', 'R_6524202_001_SS4_700_A', 'R_6535401_001_SS4_700_A',
    'R_6553202_001_SS4_700_A', 'R_6629003_001_SS4_700_A', 'R_6646203_001_SS4_700_A',
    'R_6647503_001_SS4_700_A', 'R_6654102_001_SS4_700_A', 'R_6657401_001_SS4_700_A',
    'R_6711902_001_SS4_700_A', 'R_6744603_001_SS4_700_A', 'R_6777202_001_SS4_700_A',
    'R_6995301_001_SS4_700_A', 'R_7151501_001_SS4_700_A', 'R_7490301_001_SS4_700_A',
    'R_7590502_001_SS4_700_A', 'R_7714503_001_SS4_700_A', 'R_7748801_001_SS4_700_A',
    'R_7860201_001_SS4_700_A', 'R_7891201_001_SS4_700_A', 'R_8343501_001_SS4_700_A',
    'R_8496501_001_SS4_700_A', 'R_8667201_001_SS4_700_A', 'R_8675101_001_SS4_700_A',
]


async def download_file(session, url, dest_path, max_retries=3):
    """Download a single file with retries."""
    for attempt in range(max_retries):
        try:
            timeout = aiohttp.ClientTimeout(total=600, sock_read=120)
            async with session.get(url, timeout=timeout) as resp:
                if resp.status == 200:
                    async with aiofiles.open(dest_path, "wb") as f:
                        async for chunk in resp.content.iter_chunked(65536):
                            await f.write(chunk)
                    return True
                else:
                    print(f"      HTTP {resp.status} (attempt {attempt+1})", flush=True)
        except Exception as e:
            print(f"      {type(e).__name__}: {e} (attempt {attempt+1})", flush=True)
            # Clean up partial file
            if dest_path.exists():
                dest_path.unlink()
        if attempt < max_retries - 1:
            await asyncio.sleep(2 * (attempt + 1))
    return False


async def download_product(pid, session):
    """Download .dat + .lbl for one product."""
    pid_lower = pid.lower()
    dat_path = SHARAD_HR_DIR / f"{pid_lower}.dat"
    lbl_path = SHARAD_HR_DIR / f"{pid_lower}.lbl"

    need_dat = not dat_path.exists()
    need_lbl = not lbl_path.exists()

    if not need_dat and not need_lbl:
        return True

    # Resolve bundle URLs
    try:
        bundle = await resolve_sharad_highres_bundle(pid, session)
    except Exception as e:
        print(f"    SKIP {pid}: resolve failed: {type(e).__name__}: {e}", flush=True)
        return False

    if not bundle.dat_file or not bundle.lbl_file:
        print(f"    SKIP {pid}: no files in bundle", flush=True)
        return False

    ok = True

    if need_dat:
        print(f"    DAT {pid_lower} ...", flush=True)
        if await download_file(session, bundle.dat_file.url, dat_path):
            mb = dat_path.stat().st_size / 1024 / 1024
            print(f"      OK ({mb:.1f} MB)", flush=True)
        else:
            print(f"      FAILED", flush=True)
            ok = False

    if need_lbl:
        print(f"    LBL {pid_lower} ...", flush=True)
        if await download_file(session, bundle.lbl_file.url, lbl_path):
            print(f"      OK", flush=True)
        else:
            print(f"      FAILED", flush=True)
            ok = False

    return ok


async def update_index(pids_on_disk):
    """Update sharad_highres_data/index.geojson with all products on disk."""
    if SHARAD_HR_INDEX.exists():
        async with aiofiles.open(SHARAD_HR_INDEX, "r") as f:
            index = json.loads(await f.read())
    else:
        index = {"type": "FeatureCollection", "features": []}

    existing_ids = {
        feat["properties"]["product_id"].upper()
        for feat in index["features"]
    }

    added = 0
    async with aiohttp.ClientSession() as session:
        for pid in pids_on_disk:
            pid_upper = pid.upper()
            if pid_upper in existing_ids:
                continue

            # Fetch footprint for LineString geometry
            try:
                coords = await get_sharad_highres_footprint(pid, session)
                await asyncio.sleep(0.3)
            except Exception:
                coords = []

            if coords and len(coords) >= 2:
                geometry = {"type": "LineString", "coordinates": coords}
            else:
                geometry = {"type": "Point", "coordinates": [0, 0]}

            feature = {
                "type": "Feature",
                "geometry": geometry,
                "properties": {
                    "instrument": "SHARAD_HIGHRES",
                    "product_id": pid_upper,
                    "dat_file": f"{pid.lower()}.dat",
                    "lbl_file": f"{pid.lower()}.lbl",
                }
            }
            index["features"].append(feature)
            existing_ids.add(pid_upper)
            added += 1
            if added % 5 == 0:
                print(f"    Indexed {added} so far...", flush=True)

    SHARAD_HR_INDEX.parent.mkdir(parents=True, exist_ok=True)
    async with aiofiles.open(SHARAD_HR_INDEX, "w") as f:
        await f.write(json.dumps(index, indent=2))

    print(f"  Added {added} new entries (total: {len(index['features'])})", flush=True)


async def main():
    print("=" * 60, flush=True)
    print("SHARAD High-Res: Download Remaining + Update Index", flush=True)
    print("=" * 60, flush=True)

    # Find what needs downloading
    existing_dat = {f.stem.upper() for f in SHARAD_HR_DIR.iterdir() if f.suffix == ".dat"}
    existing_lbl = {f.stem.upper() for f in SHARAD_HR_DIR.iterdir() if f.suffix == ".lbl"}

    need_download = []
    for pid in ALL_MATCHING:
        pu = pid.upper()
        if pu not in existing_dat or pu not in existing_lbl:
            need_download.append(pid)

    print(f"\n[1] Status: {len(existing_dat & set(p.upper() for p in ALL_MATCHING))}/54 on disk, "
          f"{len(need_download)} need download", flush=True)

    if need_download:
        print(f"\n[2] Downloading {len(need_download)} products (with retries)...", flush=True)
        connector = aiohttp.TCPConnector(limit=2)  # Limit concurrent connections
        async with aiohttp.ClientSession(connector=connector) as session:
            succeeded = 0
            failed = 0
            for i, pid in enumerate(need_download):
                print(f"\n  [{i+1}/{len(need_download)}] {pid}", flush=True)
                ok = await download_product(pid, session)
                if ok:
                    succeeded += 1
                else:
                    failed += 1
                # Small delay between products to be polite to PDS server
                await asyncio.sleep(1)

        print(f"\n  Results: {succeeded} succeeded, {failed} failed", flush=True)
    else:
        print(f"\n[2] All products already downloaded!", flush=True)

    # Update index
    print(f"\n[3] Updating index.geojson...", flush=True)
    pids_on_disk = [pid for pid in ALL_MATCHING
                    if (SHARAD_HR_DIR / f"{pid.lower()}.dat").exists()]
    await update_index(pids_on_disk)

    # Final summary
    final_dat = sum(1 for pid in ALL_MATCHING
                    if (SHARAD_HR_DIR / f"{pid.lower()}.dat").exists())
    print(f"\n{'=' * 60}", flush=True)
    print(f"DONE: {final_dat}/54 products on disk and indexed", flush=True)
    print(f"{'=' * 60}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
