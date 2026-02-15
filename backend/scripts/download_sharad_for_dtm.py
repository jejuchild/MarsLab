#!/usr/bin/env python3
"""
Download SHARAD High-Res RDR tracks from ODE that spatially intersect
HiRISE DTM products whose title contains "terraced crater".

Steps:
1. Load DTM index → filter by "terraced" keyword in title
2. Compute bounding boxes per DTM cluster (to minimize ODE queries)
3. Query ODE spatial API for SHARAD RDR tracks in each cluster bbox
4. Fetch LineString footprints for each candidate
5. Post-filter: keep only tracks that actually intersect DTM polygons (Liang-Barsky)
6. Download .dat + .lbl for new tracks
7. Update sharad_highres_data/index.geojson
"""

import asyncio
import json
import os
import sys
import aiohttp
import aiofiles
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.ode_client import (
    search_sharad_highres_spatial,
    get_sharad_highres_footprint,
    resolve_sharad_highres_bundle,
    ODEProduct,
)

# ── Paths ────────────────────────────────────────────────
BACKEND_DIR = Path(__file__).parent.parent
DTM_INDEX = BACKEND_DIR / "hirise_dtm_data" / "index.geojson"
SHARAD_HR_INDEX = BACKEND_DIR / "sharad_highres_data" / "index.geojson"
SHARAD_HR_DIR = BACKEND_DIR / "sharad_highres"


# ── Spatial helpers (copied from proximity_router.py) ────
def _bbox_from_geometry(geom: Dict) -> Optional[Dict[str, float]]:
    coords = []
    geom_type = geom.get("type", "")
    if geom_type == "Polygon":
        rings = geom.get("coordinates", [])
        if rings:
            coords = rings[0]
    elif geom_type == "LineString":
        coords = geom.get("coordinates", [])
    elif geom_type == "Point":
        c = geom.get("coordinates", [])
        if len(c) >= 2:
            return {"lon_min": c[0], "lon_max": c[0], "lat_min": c[1], "lat_max": c[1]}
    if not coords:
        return None
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return {"lon_min": min(lons), "lon_max": max(lons), "lat_min": min(lats), "lat_max": max(lats)}


def _liang_barsky_clip(x1, y1, x2, y2, xmin, ymin, xmax, ymax) -> bool:
    dx = x2 - x1
    dy = y2 - y1
    t0, t1 = 0.0, 1.0
    for p, q in [(-dx, x1 - xmin), (dx, xmax - x1), (-dy, y1 - ymin), (dy, ymax - y1)]:
        if p == 0:
            if q < 0:
                return False
        else:
            r = q / p
            if p < 0:
                t0 = max(t0, r)
            else:
                t1 = min(t1, r)
            if t0 > t1:
                return False
    return True


def _line_intersects_bbox(coords: list, bbox: Dict) -> bool:
    xmin, xmax = bbox["lon_min"], bbox["lon_max"]
    ymin, ymax = bbox["lat_min"], bbox["lat_max"]
    for i in range(len(coords) - 1):
        x1, y1 = coords[i][0], coords[i][1]
        x2, y2 = coords[i + 1][0], coords[i + 1][1]
        if abs(x2 - x1) > 180:
            continue
        if _liang_barsky_clip(x1, y1, x2, y2, xmin, ymin, xmax, ymax):
            return True
    return False


def _geometries_overlap(geom_a, bbox_a, geom_b, bbox_b) -> bool:
    a_is_line = geom_a and geom_a.get("type") == "LineString"
    b_is_line = geom_b and geom_b.get("type") == "LineString"
    if a_is_line and b_is_line:
        return _bboxes_overlap(bbox_a, bbox_b) and (
            _line_intersects_bbox(geom_a["coordinates"], bbox_b) or
            _line_intersects_bbox(geom_b["coordinates"], bbox_a)
        )
    elif a_is_line:
        return _line_intersects_bbox(geom_a["coordinates"], bbox_b)
    elif b_is_line:
        return _line_intersects_bbox(geom_b["coordinates"], bbox_a)
    else:
        return _bboxes_overlap(bbox_a, bbox_b)


def _bboxes_overlap(a, b) -> bool:
    if a["lat_max"] < b["lat_min"] or b["lat_max"] < a["lat_min"]:
        return False
    if a["lon_max"] < b["lon_min"] or b["lon_max"] < a["lon_min"]:
        return False
    return True


def _expand_bbox(bbox, margin_deg=2.0):
    """Expand bbox for ODE query (SHARAD tracks are long, need generous margin)."""
    return {
        "lat_min": max(-90, bbox["lat_min"] - margin_deg),
        "lat_max": min(90, bbox["lat_max"] + margin_deg),
        "lon_min": bbox["lon_min"] - margin_deg,
        "lon_max": bbox["lon_max"] + margin_deg,
    }


def _lon_to_360(lon: float) -> float:
    """Convert -180/180 longitude to 0-360 for ODE API."""
    return lon % 360


# ── Main logic ───────────────────────────────────────────

def load_terraced_dtms() -> List[Dict]:
    """Load DTM features whose title contains 'terraced'."""
    with open(DTM_INDEX) as f:
        index = json.load(f)

    results = []
    for feat in index["features"]:
        title = (feat.get("properties", {}).get("title") or "").lower()
        if "terraced" in title:
            results.append(feat)

    return results


async def query_ode_per_dtm(
    dtm: Dict,
    session: aiohttp.ClientSession,
    margin_deg: float = 1.5
) -> List[str]:
    """Query ODE for SHARAD RDR track product_ids near a single DTM.

    Uses direct ODE REST API call with a small bbox (DTM + margin).
    Returns list of product IDs.
    """
    bbox = _bbox_from_geometry(dtm["geometry"])
    if not bbox:
        return []

    expanded = _expand_bbox(bbox, margin_deg)
    westernlon = _lon_to_360(expanded["lon_min"])
    easternlon = _lon_to_360(expanded["lon_max"])

    url = (
        f"https://oderest.rsl.wustl.edu/live2?"
        f"target=mars&ihid=mro&iid=sharad&pt=RDR&"
        f"minlat={expanded['lat_min']}&maxlat={expanded['lat_max']}&"
        f"westernlon={westernlon}&easternlon={easternlon}&"
        f"output=json&results=pmf&limit=500"
    )

    dtm_pid = dtm["properties"]["product_id"]
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as resp:
            if resp.status == 200:
                data = await resp.json()
                products = data.get("ODEResults", {}).get("Products", {}).get("Product", [])
                if isinstance(products, dict):
                    products = [products]
                pids = [p.get("Product_id", p.get("pdsid", "")).upper() for p in products]
                pids = [p for p in pids if p]
                print(f"  {dtm_pid}: {len(pids)} SHARAD tracks in bbox "
                      f"lat [{expanded['lat_min']:.1f},{expanded['lat_max']:.1f}] "
                      f"lon [{expanded['lon_min']:.1f},{expanded['lon_max']:.1f}]",
                      flush=True)
                return pids
            else:
                print(f"  {dtm_pid}: ODE HTTP {resp.status}", flush=True)
    except Exception as e:
        print(f"  {dtm_pid}: ODE error: {e}", flush=True)

    return []


async def fetch_footprint_batch(
    product_ids: List[str],
    session: aiohttp.ClientSession,
    concurrency: int = 5
) -> Dict[str, List[List[float]]]:
    """Fetch LineString footprints for multiple products concurrently."""
    sem = asyncio.Semaphore(concurrency)
    results = {}

    async def fetch_one(pid: str):
        async with sem:
            try:
                coords = await get_sharad_highres_footprint(pid, session)
                if coords and len(coords) >= 2:
                    results[pid] = coords
            except Exception as e:
                print(f"    Warning: Failed to fetch footprint for {pid}: {e}")

    tasks = [fetch_one(pid) for pid in product_ids]
    await asyncio.gather(*tasks)
    return results


def filter_by_intersection(
    footprints: Dict[str, List[List[float]]],
    dtms: List[Dict]
) -> List[str]:
    """Post-filter: keep SHARAD tracks that actually intersect a DTM polygon."""
    matching = []

    for pid, coords in footprints.items():
        sharad_geom = {"type": "LineString", "coordinates": coords}
        sharad_bbox = _bbox_from_geometry(sharad_geom)
        if not sharad_bbox:
            continue

        for dtm in dtms:
            dtm_geom = dtm["geometry"]
            dtm_bbox = _bbox_from_geometry(dtm_geom)
            if not dtm_bbox:
                continue

            if _geometries_overlap(sharad_geom, sharad_bbox, dtm_geom, dtm_bbox):
                matching.append(pid)
                break  # Only need to match one DTM

    return matching


def get_existing_products() -> set:
    """Get set of product IDs already downloaded (on disk)."""
    existing = set()
    if SHARAD_HR_DIR.exists():
        for f in SHARAD_HR_DIR.iterdir():
            if f.suffix == ".dat":
                existing.add(f.stem.upper())
    return existing


def get_indexed_products() -> set:
    """Get set of product IDs already in the index."""
    indexed = set()
    if SHARAD_HR_INDEX.exists():
        with open(SHARAD_HR_INDEX) as f:
            index = json.load(f)
        for feat in index["features"]:
            pid = feat.get("properties", {}).get("product_id", "")
            indexed.add(pid.upper())
    return indexed


async def download_product(pid: str, session: aiohttp.ClientSession) -> bool:
    """Download .dat and .lbl for a single SHARAD product."""
    try:
        bundle = await resolve_sharad_highres_bundle(pid, session)

        if not bundle.dat_file or not bundle.lbl_file:
            print(f"    SKIP {pid}: No files resolved from ODE")
            return False

        pid_lower = pid.lower()
        dat_path = SHARAD_HR_DIR / f"{pid_lower}.dat"
        lbl_path = SHARAD_HR_DIR / f"{pid_lower}.lbl"

        # Download DAT
        if not dat_path.exists():
            print(f"    Downloading {bundle.dat_file.filename} ...")
            async with session.get(bundle.dat_file.url, timeout=aiohttp.ClientTimeout(total=300)) as resp:
                if resp.status == 200:
                    async with aiofiles.open(dat_path, "wb") as f:
                        async for chunk in resp.content.iter_chunked(65536):
                            await f.write(chunk)
                    size_mb = dat_path.stat().st_size / 1024 / 1024
                    print(f"      DAT: {size_mb:.1f} MB")
                else:
                    print(f"    FAIL {pid} DAT: HTTP {resp.status}")
                    return False

        # Download LBL
        if not lbl_path.exists():
            print(f"    Downloading {bundle.lbl_file.filename} ...")
            async with session.get(bundle.lbl_file.url, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                if resp.status == 200:
                    async with aiofiles.open(lbl_path, "wb") as f:
                        async for chunk in resp.content.iter_chunked(65536):
                            await f.write(chunk)
                else:
                    print(f"    FAIL {pid} LBL: HTTP {resp.status}")
                    return False

        return True

    except Exception as e:
        print(f"    ERROR downloading {pid}: {e}")
        return False


async def update_index(new_pids: List[str], footprints: Dict[str, List[List[float]]]):
    """Update sharad_highres_data/index.geojson with new products."""
    if SHARAD_HR_INDEX.exists():
        async with aiofiles.open(SHARAD_HR_INDEX, "r") as f:
            content = await f.read()
            index = json.loads(content)
    else:
        index = {"type": "FeatureCollection", "features": []}

    existing_ids = {
        feat["properties"]["product_id"].upper()
        for feat in index["features"]
    }

    added = 0
    for pid in new_pids:
        pid_upper = pid.upper()
        if pid_upper in existing_ids:
            continue

        coords = footprints.get(pid, footprints.get(pid.upper(), []))
        if coords and len(coords) >= 2:
            geometry = {"type": "LineString", "coordinates": coords}
        else:
            # Try to fetch footprint
            try:
                coords = await get_sharad_highres_footprint(pid)
                if coords and len(coords) >= 2:
                    geometry = {"type": "LineString", "coordinates": coords}
                else:
                    geometry = {"type": "Point", "coordinates": [0, 0]}
            except Exception:
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

    SHARAD_HR_INDEX.parent.mkdir(parents=True, exist_ok=True)
    async with aiofiles.open(SHARAD_HR_INDEX, "w") as f:
        await f.write(json.dumps(index, indent=2))

    print(f"\nIndex updated: {added} new entries added (total: {len(index['features'])})")


async def main():
    print("=" * 60, flush=True)
    print("SHARAD High-Res Download for Terraced Crater DTMs", flush=True)
    print("=" * 60, flush=True)

    # Step 1: Load terraced crater DTMs
    dtms = load_terraced_dtms()
    print(f"\n[1] Found {len(dtms)} DTMs matching 'terraced':", flush=True)
    for dtm in dtms:
        props = dtm["properties"]
        print(f"  - {props['product_id']}: {props['title']}", flush=True)

    # Step 2: Query ODE per-DTM for SHARAD tracks (small bboxes, fast queries)
    print(f"\n[2] Querying ODE for SHARAD RDR tracks per-DTM...", flush=True)
    all_candidate_pids = set()

    async with aiohttp.ClientSession() as session:
        for dtm in dtms:
            pids = await query_ode_per_dtm(dtm, session, margin_deg=1.5)
            all_candidate_pids.update(pids)

        unique_count = len(all_candidate_pids)
        print(f"\n  Total unique SHARAD candidates: {unique_count}", flush=True)

        # Step 3: Fetch footprints
        print(f"\n[3] Fetching LineString footprints for {unique_count} candidates...", flush=True)
        footprints = await fetch_footprint_batch(
            list(all_candidate_pids),
            session,
            concurrency=8,
        )
        print(f"  Got footprints for {len(footprints)} products", flush=True)

        # Step 4: Post-filter by actual intersection
        print(f"\n[4] Post-filtering by spatial intersection with DTM polygons...", flush=True)
        matching_pids = filter_by_intersection(footprints, dtms)
        print(f"  {len(matching_pids)} SHARAD tracks actually intersect terraced crater DTMs", flush=True)
        for pid in sorted(matching_pids):
            print(f"    - {pid}", flush=True)

        # Step 5: Check what's already downloaded
        existing_on_disk = get_existing_products()
        indexed = get_indexed_products()
        new_to_download = [pid for pid in matching_pids if pid.upper() not in existing_on_disk]
        already_downloaded = [pid for pid in matching_pids if pid.upper() in existing_on_disk]

        print(f"\n[5] Download status:", flush=True)
        print(f"  Already on disk: {len(already_downloaded)}", flush=True)
        print(f"  Need to download: {len(new_to_download)}", flush=True)

        # Step 6: Download new products
        if new_to_download:
            print(f"\n[6] Downloading {len(new_to_download)} new SHARAD High-Res products...", flush=True)
            downloaded = []
            for pid in new_to_download:
                success = await download_product(pid, session)
                if success:
                    downloaded.append(pid)

            print(f"\n  Successfully downloaded: {len(downloaded)}/{len(new_to_download)}", flush=True)
        else:
            print(f"\n[6] All matching products already downloaded!", flush=True)
            downloaded = []

        # Step 7: Update index
        all_matching_on_disk = already_downloaded + downloaded
        needs_indexing = [pid for pid in all_matching_on_disk if pid.upper() not in indexed]
        if needs_indexing or downloaded:
            print(f"\n[7] Updating index.geojson...", flush=True)
            await update_index(all_matching_on_disk, footprints)
        else:
            print(f"\n[7] Index already up to date", flush=True)

    # Summary
    print(f"\n{'=' * 60}", flush=True)
    print(f"SUMMARY", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(f"DTMs matched: {len(dtms)}", flush=True)
    print(f"ODE candidates queried: {unique_count}", flush=True)
    print(f"Footprints fetched: {len(footprints)}", flush=True)
    print(f"Spatial intersections found: {len(matching_pids)}", flush=True)
    print(f"Newly downloaded: {len(downloaded) if new_to_download else 0}", flush=True)
    print(f"Total on disk + indexed: {len(all_matching_on_disk)}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
