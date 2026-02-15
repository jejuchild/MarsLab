"""
Index Repair: Detect orphaned downloads and add them to index.geojson.

Scans each instrument's data directory for products that exist on disk
but are missing from the corresponding index.geojson. For each orphan,
attempts to fetch footprint geometry from ODE API, then adds the product
to the index.

Supported instruments: SHARAD_HIGHRES, CRISM, HiRISE, SHARAD, HiRISE_DTM.
CTX is skipped (remote tile service, no local data files).
"""

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import aiohttp
import aiofiles

ODE_REST_BASE = "https://oderest.rsl.wustl.edu/live2"
ODE_TIMEOUT = aiohttp.ClientTimeout(total=10)
INTER_REQUEST_DELAY = 0.2  # seconds between ODE requests

BASE_DIR = Path(__file__).parent.parent

# Data directories (must match download_manager.py)
CRISM_DATA_DIR = BASE_DIR / "crism_data"
HIRISE_DATA_DIR = BASE_DIR / "hirise_data"
SHARAD_DATA_DIR = BASE_DIR / "sharad_data"
SHARAD_HIGHRES_DIR = BASE_DIR / "sharad_highres"
SHARAD_HIGHRES_INDEX_DIR = BASE_DIR / "sharad_highres_data"
HIRISE_DTM_DIR = BASE_DIR / "hirise_dtm_data"


# =========================================================================
# WKT Parsing helpers
# =========================================================================

def _parse_wkt_polygon(wkt: str) -> Optional[List[List[float]]]:
    """Parse WKT POLYGON into list of [lon, lat] ring coordinates."""
    match = re.search(r'POLYGON\s*\(\((.*?)\)\)', wkt, re.IGNORECASE)
    if not match:
        return None
    coords_str = match.group(1)
    coordinates = []
    for pair in coords_str.split(','):
        parts = pair.strip().split()
        if len(parts) >= 2:
            try:
                lon = float(parts[0])
                lat = float(parts[1])
                # ODE uses 0-360 longitude; convert to -180/180
                if lon > 180:
                    lon -= 360
                coordinates.append([lon, lat])
            except ValueError:
                continue
    return coordinates if len(coordinates) >= 3 else None


def _parse_wkt_linestring(wkt: str) -> Optional[List[List[float]]]:
    """Parse WKT LINESTRING into list of [lon, lat] coordinates."""
    match = re.search(r'LINESTRING\s*\((.*)\)', wkt, re.IGNORECASE)
    if not match:
        return None
    coords_str = match.group(1)
    coordinates = []
    for pair in coords_str.split(','):
        parts = pair.strip().split()
        if len(parts) >= 2:
            try:
                lon = float(parts[0])
                lat = float(parts[1])
                if lon > 180:
                    lon -= 360
                coordinates.append([lon, lat])
            except ValueError:
                continue
    return coordinates if len(coordinates) >= 2 else None


# =========================================================================
# ODE footprint fetching
# =========================================================================

async def _fetch_ode_footprint(
    session: aiohttp.ClientSession,
    product_id: str,
    iid: str,
    pt: str,
) -> Tuple[Optional[Dict], Optional[float], Optional[float]]:
    """
    Fetch product footprint from ODE REST API.

    Returns:
        (geometry_dict, center_lat, center_lon) or (None, None, None) on failure.
        geometry_dict is a GeoJSON-compatible geometry (Polygon or LineString).
    """
    url = (
        f"{ODE_REST_BASE}?"
        f"target=mars&ihid=mro&iid={iid}&"
        f"productid={product_id}*&pt={pt}&"
        f"output=json&results=pmf&limit=1"
    )

    try:
        async with session.get(url, timeout=ODE_TIMEOUT) as resp:
            if resp.status != 200:
                return None, None, None
            data = await resp.json()
    except Exception as e:
        print(f"  [REPAIR] ODE fetch failed for {product_id}: {e}")
        return None, None, None

    products_data = data.get("ODEResults", {}).get("Products", {})
    # ODE returns "No Products Found" string when empty
    if not isinstance(products_data, dict):
        return None, None, None
    products = products_data.get("Product", [])
    if isinstance(products, dict):
        products = [products]
    if not products:
        return None, None, None

    p = products[0]
    footprint_raw = p.get("Footprint_C0_geometry", "")
    # ODE may return footprint as WKT string or dict — normalize to string
    footprint_wkt = footprint_raw if isinstance(footprint_raw, str) else ""
    center_lat = _safe_float(p.get("Center_latitude"))
    center_lon = _safe_float(p.get("Center_longitude"))

    # Convert center_lon from 0-360 to -180/180
    if center_lon is not None and center_lon > 180:
        center_lon -= 360

    geometry = None

    # Try POLYGON first (CRISM, HiRISE, DTM)
    if footprint_wkt and "POLYGON" in footprint_wkt.upper():
        coords = _parse_wkt_polygon(footprint_wkt)
        if coords:
            geometry = {"type": "Polygon", "coordinates": [coords]}

    # Try LINESTRING (SHARAD)
    if geometry is None and footprint_wkt and "LINESTRING" in footprint_wkt.upper():
        coords = _parse_wkt_linestring(footprint_wkt)
        if coords:
            geometry = {"type": "LineString", "coordinates": coords}

    # Fallback: extract start/stop for SHARAD tracks
    if geometry is None and iid == "sharad":
        start_lat = _safe_float(p.get("Start_latitude"))
        start_lon = _safe_float(p.get("Start_longitude"))
        stop_lat = _safe_float(p.get("Stop_latitude"))
        stop_lon = _safe_float(p.get("Stop_longitude"))
        if all(v is not None for v in [start_lat, start_lon, stop_lat, stop_lon]):
            sl = start_lon - 360 if start_lon > 180 else start_lon
            el = stop_lon - 360 if stop_lon > 180 else stop_lon
            geometry = {
                "type": "LineString",
                "coordinates": [[sl, start_lat], [el, stop_lat]],
            }

    return geometry, center_lat, center_lon


def _safe_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


# =========================================================================
# Index I/O helpers
# =========================================================================

def _load_index(path: Path) -> dict:
    """Load index.geojson or return empty FeatureCollection."""
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"type": "FeatureCollection", "features": []}


async def _save_index(path: Path, index: dict):
    """Write index.geojson atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    async with aiofiles.open(path, "w") as f:
        await f.write(json.dumps(index, indent=2))


# =========================================================================
# Per-instrument repair functions
# =========================================================================

async def _repair_sharad_highres(session: aiohttp.ClientSession) -> Dict:
    """Repair SHARAD_HIGHRES index by scanning sharad_highres/ for .dat files."""
    index_path = SHARAD_HIGHRES_INDEX_DIR / "index.geojson"
    data_dir = SHARAD_HIGHRES_DIR

    if not data_dir.exists():
        return {"scanned": 0, "orphaned": 0, "added": 0, "failed": 0}

    # Scan disk for .dat files
    dat_files = [f for f in os.listdir(data_dir) if f.lower().endswith(".dat")]
    disk_ids = {os.path.splitext(f)[0].upper(): f for f in dat_files}

    # Load index and get existing product IDs (case-insensitive)
    index = _load_index(index_path)
    indexed_ids = set()
    for feat in index["features"]:
        pid = feat.get("properties", {}).get("product_id", "")
        indexed_ids.add(pid.upper())

    # Find orphans
    orphan_ids = {pid: fname for pid, fname in disk_ids.items() if pid not in indexed_ids}

    if not orphan_ids:
        return {"scanned": len(dat_files), "orphaned": 0, "added": 0, "failed": 0}

    print(f"[REPAIR] SHARAD_HIGHRES: {len(orphan_ids)} orphaned products found")

    added = 0
    failed = 0

    for pid, dat_filename in sorted(orphan_ids.items()):
        # Check for matching .lbl file
        stem = os.path.splitext(dat_filename)[0]
        lbl_filename = stem + ".lbl"
        has_lbl = (data_dir / lbl_filename).exists()

        # Fetch footprint from ODE
        geometry, center_lat, center_lon = await _fetch_ode_footprint(
            session, pid, "sharad", "RDR"
        )

        if geometry is None:
            # Fallback: Point at (0, 0)
            geometry = {"type": "Point", "coordinates": [0, 0]}
            print(f"  [REPAIR] {pid}: ODE footprint unavailable, using Point(0,0)")

        feature = {
            "type": "Feature",
            "properties": {
                "product_id": pid,
                "instrument": "SHARAD_HIGHRES",
                "dat_file": dat_filename,
                "lbl_file": lbl_filename if has_lbl else None,
            },
            "geometry": geometry,
        }

        index["features"].append(feature)
        added += 1
        print(f"  [REPAIR] {pid}: added ({geometry['type']})")

        await asyncio.sleep(INTER_REQUEST_DELAY)

    if added > 0:
        await _save_index(index_path, index)
        print(f"[REPAIR] SHARAD_HIGHRES: wrote {added} new entries to index")

    return {"scanned": len(dat_files), "orphaned": len(orphan_ids), "added": added, "failed": failed}


async def _repair_crism(session: aiohttp.ClientSession) -> Dict:
    """Repair CRISM index by scanning crism_data/*/  subdirs for .img files."""
    index_path = CRISM_DATA_DIR / "index.geojson"

    if not CRISM_DATA_DIR.exists():
        return {"scanned": 0, "orphaned": 0, "added": 0, "failed": 0}

    # Scan subdirs for .img files
    disk_products = {}  # base_key -> {img, lbl, quicklook}
    for entry in os.scandir(CRISM_DATA_DIR):
        if not entry.is_dir():
            continue
        base_key = entry.name
        files = os.listdir(entry.path)
        imgs = [f for f in files if f.lower().endswith(".img")]
        if not imgs:
            continue

        img_file = imgs[0]  # Primary .img file
        lbl_file = next((f for f in files if f.lower().endswith(".lbl")), None)
        # Check for browse VNAB or BRNA quicklook
        quicklook = next(
            (f"/crism/quickview/{f}" for f in files if "_brvna" in f.lower()),
            None,
        )
        disk_products[base_key] = {
            "img": img_file,
            "lbl": lbl_file,
            "quicklook": quicklook,
        }

    # Load index and get existing base_keys
    index = _load_index(index_path)
    indexed_keys = set()
    for feat in index["features"]:
        bk = feat.get("properties", {}).get("base_key", "")
        if bk:
            indexed_keys.add(bk)
        else:
            # Fallback: derive base_key from product_id
            pid = feat.get("properties", {}).get("product_id", "")
            if pid:
                # base_key is the observation+version prefix (e.g. frt0001fd76_07)
                parts = pid.split("_")
                if len(parts) >= 2:
                    indexed_keys.add("_".join(parts[:2]))

    # Find orphans
    orphan_keys = {k: v for k, v in disk_products.items() if k not in indexed_keys}

    if not orphan_keys:
        return {"scanned": len(disk_products), "orphaned": 0, "added": 0, "failed": 0}

    print(f"[REPAIR] CRISM: {len(orphan_keys)} orphaned products found")

    added = 0
    failed = 0

    for base_key, info in sorted(orphan_keys.items()):
        product_id = info["img"].replace(".img", "") if info["img"] else base_key

        # Fetch footprint from ODE
        geometry, center_lat, center_lon = await _fetch_ode_footprint(
            session, base_key, "crism", "MTRDR"
        )

        # If MTRDR fails, try TRDR (some products are TRR3 not MTR3)
        if geometry is None:
            geometry, center_lat, center_lon = await _fetch_ode_footprint(
                session, base_key, "crism", "TRDR"
            )

        if geometry is None:
            # Fallback: Point at (0, 0)
            geometry = {"type": "Point", "coordinates": [0, 0]}
            print(f"  [REPAIR] {base_key}: ODE footprint unavailable, using Point(0,0)")

        feature = {
            "type": "Feature",
            "properties": {
                "instrument": "CRISM",
                "product_id": product_id,
                "base_key": base_key,
                "mtr3_img": info["img"],
                "mtr3_lbl": info["lbl"],
                "quicklook": info["quicklook"],
            },
            "geometry": geometry,
        }

        index["features"].append(feature)
        added += 1
        print(f"  [REPAIR] {base_key}: added ({geometry['type']})")

        await asyncio.sleep(INTER_REQUEST_DELAY)

    if added > 0:
        await _save_index(index_path, index)
        print(f"[REPAIR] CRISM: wrote {added} new entries to index")

    return {"scanned": len(disk_products), "orphaned": len(orphan_keys), "added": added, "failed": failed}


async def _repair_hirise(session: aiohttp.ClientSession) -> Dict:
    """Repair HiRISE index by scanning hirise_data/*/ subdirs for .tif/.JP2 files."""
    index_path = HIRISE_DATA_DIR / "index.geojson"

    if not HIRISE_DATA_DIR.exists():
        return {"scanned": 0, "orphaned": 0, "added": 0, "failed": 0}

    # Scan subdirs for .tif or .JP2 files
    disk_products = {}  # product_id -> {tif_file}
    for entry in os.scandir(HIRISE_DATA_DIR):
        if not entry.is_dir():
            continue
        product_id = entry.name
        files = os.listdir(entry.path)
        tifs = [f for f in files if f.lower().endswith(".tif")]
        jp2s = [f for f in files if f.lower().endswith(".jp2")]
        if not tifs and not jp2s:
            continue
        disk_products[product_id] = {
            "red_tif": tifs[0] if tifs else None,
            "jp2": jp2s[0] if jp2s else None,
        }

    # Load index and get existing product IDs
    # Normalize _RED suffix: index may have ESP_xxx or ESP_xxx_RED inconsistently
    index = _load_index(index_path)
    indexed_ids = set()
    for feat in index["features"]:
        pid = feat.get("properties", {}).get("product_id", "")
        indexed_ids.add(pid)
        # Also add the _RED variant (or stripped variant) for matching
        if pid.endswith("_RED"):
            indexed_ids.add(pid[:-4])  # strip _RED
        else:
            indexed_ids.add(pid + "_RED")  # add _RED

    # Find orphans (check both with and without _RED)
    orphan_ids = {}
    for k, v in disk_products.items():
        if k not in indexed_ids:
            base = k[:-4] if k.endswith("_RED") else k
            if base not in indexed_ids and base + "_RED" not in indexed_ids:
                orphan_ids[k] = v

    if not orphan_ids:
        return {"scanned": len(disk_products), "orphaned": 0, "added": 0, "failed": 0}

    print(f"[REPAIR] HiRISE: {len(orphan_ids)} orphaned products found")

    added = 0
    failed = 0

    for product_id, info in sorted(orphan_ids.items()):
        # Fetch footprint from ODE
        geometry, center_lat, center_lon = await _fetch_ode_footprint(
            session, product_id, "hirise", "RDRV11"
        )

        if geometry is None:
            geometry = {"type": "Point", "coordinates": [0, 0]}
            print(f"  [REPAIR] {product_id}: ODE footprint unavailable, using Point(0,0)")

        feature = {
            "type": "Feature",
            "properties": {
                "instrument": "HIRISE",
                "product_id": product_id,
                "quicklook": f"/hirise/quickview/{product_id}.jpg",
                "red_tif": info["red_tif"],
            },
            "geometry": geometry,
        }

        index["features"].append(feature)
        added += 1
        print(f"  [REPAIR] {product_id}: added ({geometry['type']})")

        await asyncio.sleep(INTER_REQUEST_DELAY)

    if added > 0:
        await _save_index(index_path, index)
        print(f"[REPAIR] HiRISE: wrote {added} new entries to index")

    return {"scanned": len(disk_products), "orphaned": len(orphan_ids), "added": added, "failed": failed}


async def _repair_sharad(session: aiohttp.ClientSession) -> Dict:
    """Repair SHARAD index by scanning sharad_data/ for .tif files."""
    index_path = SHARAD_DATA_DIR / "index.geojson"

    if not SHARAD_DATA_DIR.exists():
        return {"scanned": 0, "orphaned": 0, "added": 0, "failed": 0}

    # Scan for .tif files (SHARAD radargram images)
    tif_files = [f for f in os.listdir(SHARAD_DATA_DIR)
                 if f.lower().endswith(".tif") and f != "index.geojson"]
    # Derive product IDs: s_07705302_tiff.tif -> S_07705302_THM
    disk_ids = {}
    for f in tif_files:
        stem = os.path.splitext(f)[0]  # e.g. s_07705302_tiff
        # Extract orbit number portion
        parts = stem.split("_")
        if len(parts) >= 2:
            # Reconstruct as THM product id (matching index convention)
            pid = f"{parts[0]}_{parts[1]}_THM".upper()
            disk_ids[pid] = f

    # Load index
    index = _load_index(index_path)
    indexed_ids = set()
    for feat in index["features"]:
        pid = feat.get("properties", {}).get("product_id", "")
        indexed_ids.add(pid.upper())

    # Find orphans
    orphan_ids = {k: v for k, v in disk_ids.items() if k not in indexed_ids}

    if not orphan_ids:
        return {"scanned": len(tif_files), "orphaned": 0, "added": 0, "failed": 0}

    print(f"[REPAIR] SHARAD: {len(orphan_ids)} orphaned products found")

    added = 0

    for pid, tif_filename in sorted(orphan_ids.items()):
        feature = {
            "type": "Feature",
            "properties": {
                "instrument": "SHARAD",
                "product_id": pid,
                "quickview": f"/sharad/quickview/{pid.lower()}.jpg",
            },
            "geometry": {"type": "Point", "coordinates": [0, 0]},
        }
        index["features"].append(feature)
        added += 1
        print(f"  [REPAIR] {pid}: added (Point fallback)")

    if added > 0:
        await _save_index(index_path, index)
        print(f"[REPAIR] SHARAD: wrote {added} new entries to index")

    return {"scanned": len(tif_files), "orphaned": len(orphan_ids), "added": added, "failed": 0}


async def _repair_hirise_dtm(session: aiohttp.ClientSession) -> Dict:
    """Repair HiRISE_DTM index by scanning hirise_dtm_data/ for .IMG files."""
    index_path = HIRISE_DTM_DIR / "index.geojson"

    if not HIRISE_DTM_DIR.exists():
        return {"scanned": 0, "orphaned": 0, "added": 0, "failed": 0}

    # Scan for DTM .IMG files (product IDs start with DTE)
    img_files = [f for f in os.listdir(HIRISE_DTM_DIR)
                 if f.upper().endswith(".IMG") and f.upper().startswith("DTE")]
    disk_ids = {os.path.splitext(f)[0]: f for f in img_files}

    # Load index
    index = _load_index(index_path)
    indexed_ids = set(
        feat.get("properties", {}).get("product_id", "")
        for feat in index["features"]
    )

    # Find orphans
    orphan_ids = {k: v for k, v in disk_ids.items() if k not in indexed_ids}

    if not orphan_ids:
        return {"scanned": len(img_files), "orphaned": 0, "added": 0, "failed": 0}

    print(f"[REPAIR] HiRISE_DTM: {len(orphan_ids)} orphaned products found")

    added = 0

    for pid, img_filename in sorted(orphan_ids.items()):
        # Check for matching ortho JP2
        ortho_file = ""
        for f in os.listdir(HIRISE_DTM_DIR):
            if f.upper().endswith(".JP2") and pid[:20] in f.upper():
                ortho_file = f
                break

        # Check if this DTM has crawl metadata (downloaded from PDS)
        has_source = False
        crawl_path = HIRISE_DTM_DIR / "crawl_metadata.json"
        if crawl_path.exists():
            import json as _json
            try:
                with open(crawl_path) as _f:
                    crawl = _json.load(_f)
                has_source = any(
                    pid in m.get("dtm_filename", "")
                    for m in crawl
                )
            except Exception:
                pass

        title = pid if has_source else f"{pid} (CUSTOM)"
        feature = {
            "type": "Feature",
            "properties": {
                "product_id": pid,
                "title": title,
                "dtm_file": img_filename,
                "ortho_file": ortho_file,
                "custom": not has_source,
            },
            "geometry": {"type": "Point", "coordinates": [0, 0]},
        }
        index["features"].append(feature)
        added += 1
        print(f"  [REPAIR] {pid}: added {'(CUSTOM)' if not has_source else '(downloaded)'}")

    if added > 0:
        await _save_index(index_path, index)
        print(f"[REPAIR] HiRISE_DTM: wrote {added} new entries to index")

    return {"scanned": len(img_files), "orphaned": len(orphan_ids), "added": added, "failed": 0}


# =========================================================================
# Main entry point
# =========================================================================

async def repair_all_indices(
    session: Optional[aiohttp.ClientSession] = None,
) -> Dict[str, Dict]:
    """
    Scan all instrument data directories for orphaned downloads
    and add them to their respective index.geojson files.

    Args:
        session: Optional aiohttp session (creates one if not provided).

    Returns:
        Dict mapping instrument name to repair stats:
        {"scanned": int, "orphaned": int, "added": int, "failed": int}
    """
    close_session = session is None
    if session is None:
        session = aiohttp.ClientSession()

    try:
        results = {}
        for name, fn in [
            ("sharad_highres", _repair_sharad_highres),
            ("crism", _repair_crism),
            ("hirise", _repair_hirise),
            ("sharad", _repair_sharad),
            ("hirise_dtm", _repair_hirise_dtm),
        ]:
            try:
                results[name] = await fn(session)
            except Exception as e:
                print(f"[REPAIR] {name} repair failed: {e}")
                results[name] = {"scanned": 0, "orphaned": 0, "added": 0, "failed": 0, "error": str(e)}
        return results
    finally:
        if close_session:
            await session.close()
