#!/usr/bin/env python3
"""Update sharad_highres_data/index.geojson with all 54 matching products on disk."""

import asyncio
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import aiohttp
import aiofiles
from api.ode_client import get_sharad_highres_footprint

SHARAD_HR_DIR = Path(__file__).parent.parent / "sharad_highres"
SHARAD_HR_INDEX = Path(__file__).parent.parent / "sharad_highres_data" / "index.geojson"

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


async def main():
    # Load existing index
    if SHARAD_HR_INDEX.exists():
        with open(SHARAD_HR_INDEX) as f:
            index = json.load(f)
    else:
        index = {"type": "FeatureCollection", "features": []}

    existing_ids = {
        feat["properties"]["product_id"].upper()
        for feat in index["features"]
    }

    # Find products on disk that need indexing
    on_disk = [pid for pid in ALL_MATCHING
               if (SHARAD_HR_DIR / f"{pid.lower()}.dat").exists()]
    need_indexing = [pid for pid in on_disk if pid.upper() not in existing_ids]

    print(f"Products on disk: {len(on_disk)}/54", flush=True)
    print(f"Already indexed: {len(on_disk) - len(need_indexing)}", flush=True)
    print(f"Need indexing: {len(need_indexing)}", flush=True)

    if not need_indexing:
        print("Index already up to date!", flush=True)
        return

    # Fetch footprints concurrently
    sem = asyncio.Semaphore(8)
    footprints = {}

    async def fetch_one(pid, session):
        async with sem:
            try:
                coords = await get_sharad_highres_footprint(pid, session)
                if coords and len(coords) >= 2:
                    footprints[pid] = coords
                    print(f"  {pid}: {len(coords)} points", flush=True)
                else:
                    print(f"  {pid}: no footprint", flush=True)
            except Exception as e:
                print(f"  {pid}: error: {e}", flush=True)

    print(f"\nFetching footprints for {len(need_indexing)} products...", flush=True)
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_one(pid, session) for pid in need_indexing]
        await asyncio.gather(*tasks)

    # Add to index
    added = 0
    for pid in need_indexing:
        coords = footprints.get(pid, [])
        if coords and len(coords) >= 2:
            geometry = {"type": "LineString", "coordinates": coords}
        else:
            geometry = {"type": "Point", "coordinates": [0, 0]}

        feature = {
            "type": "Feature",
            "geometry": geometry,
            "properties": {
                "instrument": "SHARAD_HIGHRES",
                "product_id": pid.upper(),
                "dat_file": f"{pid.lower()}.dat",
                "lbl_file": f"{pid.lower()}.lbl",
            }
        }
        index["features"].append(feature)
        added += 1

    # Write
    SHARAD_HR_INDEX.parent.mkdir(parents=True, exist_ok=True)
    with open(SHARAD_HR_INDEX, "w") as f:
        json.dump(index, f, indent=2)

    print(f"\nAdded {added} entries to index (total: {len(index['features'])})", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
