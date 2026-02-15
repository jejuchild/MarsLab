"""
CNN Mineral Classification — Acquire & Classify Pipeline

End-to-end pipeline: ODE discovery → download TRR3+DDR → JCAT+CNN → quickview → index update.
Yields SSE events for real-time progress in the frontend.
"""

import os
import json
import re
import logging
import asyncio
from typing import AsyncGenerator, Optional, List

import aiohttp
import numpy as np

from .constants import TRR_DATA_DIR, RESULTS_DIR
from .pipeline import run_classification, has_cached_result, _validate_obs_id
from .data_loader import resolve_trr_files, load_trr_cube

logger = logging.getLogger(__name__)

ODE_REST = "https://oderest.rsl.wustl.edu/live2"
ODE_TIMEOUT = aiohttp.ClientTimeout(total=120)


# ============================================================
# ODE Discovery helpers (adapted from scripts/download_crism_trr3.py)
# ============================================================

def _extract_base_key(pdsid: str) -> str:
    """frt0001fd76_07_if166l_trr3 -> frt0001fd76_07"""
    parts = pdsid.lower().split("_")
    return f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else pdsid.lower()


def _derive_ddr_pdsid(trr_pdsid: str) -> str:
    """frt00009312_07_if166l_trr3 -> frt00009312_07_de166l_ddr1"""
    parts = trr_pdsid.lower().split("_")
    if len(parts) >= 4:
        activity = parts[2]  # e.g. 'if166l'
        return f"{parts[0]}_{parts[1]}_de{activity[2:]}_ddr1"
    return ""


def _is_l_sensor(pdsid: str) -> bool:
    """Check if product ID indicates L-sensor (IR) data.
    Pattern: ..._IF{NNN}L_TRR3  (L before _TRR3 = L-sensor)
    """
    return bool(re.search(r'_if\d+l_trr3', pdsid, re.IGNORECASE))


def _parse_wkt_polygon(wkt: str) -> Optional[List[List[float]]]:
    """Parse WKT POLYGON into [[lon, lat], ...]. Converts ODE 0-360 lon to -180/180."""
    match = re.search(r'POLYGON\s*\(\((.*?)\)\)', wkt, re.IGNORECASE)
    if not match:
        return None
    coords = []
    for pair in match.group(1).split(','):
        parts = pair.strip().split()
        if len(parts) >= 2:
            try:
                lon = float(parts[0])
                lat = float(parts[1])
                if lon > 180:
                    lon -= 360
                coords.append([lon, lat])
            except ValueError:
                continue
    return coords if len(coords) >= 3 else None


async def _discover_trr3(session: aiohttp.ClientSession, obs_id: str) -> dict:
    """
    Query ODE for TRR3 + DDR file URLs and footprint.

    Returns dict with keys:
      trr3_pdsid, base_key, files (list of {url, name, size, key}),
      footprint (GeoJSON coords or None)
    """
    # Query ODE for TRDR products matching this obs_id
    url = (
        f"{ODE_REST}?target=mars&ihid=mro&iid=crism&pt=TRDR&"
        f"productid={obs_id}*&output=json&results=fpm&limit=20"
    )

    async with session.get(url, timeout=ODE_TIMEOUT) as resp:
        if resp.status != 200:
            raise ConnectionError(f"ODE returned HTTP {resp.status}")
        data = await resp.json()

    ode = data.get("ODEResults", {})
    if ode.get("Status") != "Success":
        raise ValueError(f"ODE query failed: {ode.get('Status')}")

    product_list = ode.get("Products", {}).get("Product", [])
    if isinstance(product_list, dict):
        product_list = [product_list]

    # Find L-sensor TRR3 product (prefer L-sensor for CNN)
    trr3_product = None
    for p in product_list:
        pdsid = p.get("pdsid", "")
        if "_if" in pdsid.lower() and "_trr3" in pdsid.lower() and _is_l_sensor(pdsid):
            trr3_product = p
            break

    # Fallback: accept any TRR3 I/F product
    if not trr3_product:
        for p in product_list:
            pdsid = p.get("pdsid", "")
            if "_if" in pdsid.lower() and "_trr3" in pdsid.lower():
                trr3_product = p
                break

    if not trr3_product:
        raise FileNotFoundError(f"No TRDR products found for {obs_id}")

    pdsid = trr3_product["pdsid"]
    base_key = _extract_base_key(pdsid)

    # Extract file URLs from embedded Product_files
    pf = trr3_product.get("Product_files", {}).get("Product_file", [])
    if isinstance(pf, dict):
        pf = [pf]

    files = []
    for f in pf:
        name = f.get("FileName", "")
        f_url = f.get("URL", "")
        kb = int(f.get("KBytes", 0) or 0)
        upper = name.upper()
        if upper.endswith("_TRR3.IMG"):
            files.append({"url": f_url, "name": name, "size": kb * 1024, "key": "trr3_img"})
        elif upper.endswith("_TRR3.LBL"):
            files.append({"url": f_url, "name": name, "size": kb * 1024, "key": "trr3_lbl"})

    if not any(f["key"] == "trr3_img" for f in files):
        raise FileNotFoundError(f"TRR3.IMG file not found in ODE for {pdsid}")

    # Query DDR files
    ddr_pdsid = _derive_ddr_pdsid(pdsid)
    if ddr_pdsid:
        ddr_url = (
            f"{ODE_REST}?target=mars&ihid=mro&iid=crism&"
            f"productid={ddr_pdsid}&output=json&results=fpm&limit=1"
        )
        try:
            async with session.get(ddr_url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status == 200:
                    ddr_data = await resp.json()
                    ddr_prods = ddr_data.get("ODEResults", {}).get("Products", {}).get("Product", [])
                    if isinstance(ddr_prods, dict):
                        ddr_prods = [ddr_prods]
                    if ddr_prods:
                        dpf = ddr_prods[0].get("Product_files", {}).get("Product_file", [])
                        if isinstance(dpf, dict):
                            dpf = [dpf]
                        for f in dpf:
                            name = f.get("FileName", "")
                            f_url = f.get("URL", "")
                            kb = int(f.get("KBytes", 0) or 0)
                            upper = name.upper()
                            if upper.endswith("_DDR1.IMG"):
                                files.append({"url": f_url, "name": name, "size": kb * 1024, "key": "ddr_img"})
                            elif upper.endswith("_DDR1.LBL"):
                                files.append({"url": f_url, "name": name, "size": kb * 1024, "key": "ddr_lbl"})
        except Exception as e:
            logger.warning(f"DDR query failed for {ddr_pdsid}: {e}")

    if not any(f["key"] == "ddr_img" for f in files):
        raise FileNotFoundError(f"DDR files not found for {obs_id} (tried {ddr_pdsid})")

    # Extract footprint
    footprint_wkt = trr3_product.get("Footprint_C0_geometry", "")
    footprint = _parse_wkt_polygon(footprint_wkt) if footprint_wkt else None

    # Find the full TRR3 product ID (for index)
    trr3_img_name = next((f["name"] for f in files if f["key"] == "trr3_img"), "")
    trr3_product_id = trr3_img_name.replace(".IMG", "").replace(".img", "")

    return {
        "pdsid": pdsid,
        "base_key": base_key,
        "trr3_product_id": trr3_product_id,
        "files": files,
        "footprint": footprint,
        "is_l_sensor": _is_l_sensor(pdsid),
    }


# ============================================================
# Download helpers
# ============================================================

async def _download_file_aiohttp(
    session: aiohttp.ClientSession,
    url: str, dest_path: str,
    progress_callback=None,
) -> bool:
    """Download a file with aiohttp, with optional progress callback."""
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=600)) as resp:
            if resp.status != 200:
                logger.error(f"Download failed: HTTP {resp.status} for {url}")
                return False
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            with open(dest_path, "wb") as f:
                async for chunk in resp.content.iter_chunked(1024 * 256):  # 256KB chunks
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress_callback and total > 0:
                        progress_callback(downloaded, total)
        return True
    except Exception as e:
        logger.error(f"Download error for {url}: {e}")
        return False


async def _download_files(
    files: list, target_dir: str, yield_event,
) -> bool:
    """Download all files to target_dir. Returns True on success."""
    os.makedirs(target_dir, exist_ok=True)

    # Try aria2 for large files, aiohttp for small
    try:
        from ..aria2_downloader import download_single_file, Aria2Status
        has_aria2 = True
    except ImportError:
        has_aria2 = False

    async with aiohttp.ClientSession() as session:
        for fi in files:
            dest_path = os.path.join(target_dir, fi["name"])
            size_mb = fi["size"] / (1024 * 1024) if fi["size"] else 0

            # Skip if already exists
            if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
                await yield_event({
                    "event": "download_complete",
                    "data": {"file": fi["name"], "skipped": True},
                })
                continue

            await yield_event({
                "event": "status", "data": {
                    "step": "download",
                    "message": f"Downloading {fi['name']} ({size_mb:.0f} MB)...",
                },
            })

            if has_aria2 and fi["size"] > 1024 * 1024:  # aria2 for files > 1MB
                result = await download_single_file(
                    url=fi["url"],
                    output_dir=target_dir,
                    filename=fi["name"],
                )
                if result.status != Aria2Status.COMPLETED:
                    raise IOError(f"Failed to download {fi['name']}: {result.error}")
            else:
                # aiohttp for small files or if no aria2
                last_pct = [0]

                def on_progress(done, total):
                    pct = round(100 * done / total, 1) if total else 0
                    if pct - last_pct[0] >= 5:
                        last_pct[0] = pct

                ok = await _download_file_aiohttp(
                    session, fi["url"], dest_path, on_progress,
                )
                if not ok:
                    raise IOError(f"Failed to download {fi['name']}")

            await yield_event({
                "event": "download_complete",
                "data": {"file": fi["name"], "skipped": False},
            })

    return True


# ============================================================
# Quickview generation
# ============================================================

def _generate_quickview(obs_dir: str, trr_img: str, trr_lbl: str) -> str:
    """Generate a VNIR quickview PNG from TRR3 data. Returns path to PNG."""
    from PIL import Image

    cache_path = os.path.join(obs_dir, "quickview.png")
    if os.path.exists(cache_path):
        return cache_path

    cube, rows, cols = load_trr_cube(trr_img, trr_lbl)
    n_bands = cube.shape[2]
    band = int(n_bands * 0.6)  # ~0.7 µm for VNIR
    single = cube[:, :, band].astype(np.float64)

    valid = (single > 0) & (single < 1.5) & np.isfinite(single)
    if valid.sum() < 10:
        return ""

    p2 = float(np.percentile(single[valid], 2))
    p98 = float(np.percentile(single[valid], 98))
    if p98 <= p2:
        p98 = p2 + 0.01

    stretched = np.clip((single - p2) / (p98 - p2) * 255, 0, 255).astype(np.uint8)
    stretched[~valid] = 0

    img = Image.fromarray(stretched, mode="L")
    img.save(cache_path, format="PNG")
    return cache_path


# ============================================================
# Index update
# ============================================================

_index_lock = asyncio.Lock()

INDEX_PATH = os.path.join(TRR_DATA_DIR, "index.geojson")


async def _update_index(base_key: str, trr3_product_id: str, footprint_coords):
    """Add new observation to index.geojson and refresh in-memory cache."""
    async with _index_lock:
        # Load existing index
        if os.path.exists(INDEX_PATH):
            with open(INDEX_PATH) as f:
                index = json.load(f)
        else:
            index = {"type": "FeatureCollection", "features": []}

        # Check for duplicate
        existing_ids = {ft["properties"].get("product_id") for ft in index["features"]}
        if base_key in existing_ids:
            return False  # Already exists

        # Build feature
        feature = {
            "type": "Feature",
            "properties": {
                "product_id": base_key,
                "trr3_product_id": trr3_product_id,
                "instrument": "CRISM_TRR3",
                "obs_dir": base_key,
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [footprint_coords],
            } if footprint_coords else None,
        }

        index["features"].append(feature)

        with open(INDEX_PATH, "w") as f:
            json.dump(index, f, indent=2)

    # Refresh in-memory cache
    try:
        from ...app import refresh_geojson_cache
        refresh_geojson_cache("crism_trr3")
    except Exception as e:
        logger.warning(f"Could not refresh GeoJSON cache: {e}")

    return True


# ============================================================
# Status check
# ============================================================

def check_acquire_status(obs_id: str) -> dict:
    """Check whether TRR3 data and classification results exist locally."""
    _validate_obs_id(obs_id)

    # Check data directory
    obs_dir = os.path.join(TRR_DATA_DIR, obs_id)
    has_dir = os.path.isdir(obs_dir)

    has_trr3 = False
    has_ddr = False
    if has_dir:
        files = os.listdir(obs_dir)
        has_trr3 = any(f.upper().endswith("_TRR3.IMG") for f in files)
        has_ddr = any(f.upper().endswith("_DDR1.IMG") for f in files)

    has_results = has_cached_result(obs_id)

    return {
        "obs_id": obs_id,
        "has_trr3_data": has_trr3 and has_ddr,
        "has_trr3_only": has_trr3 and not has_ddr,
        "has_results": has_results,
    }


# ============================================================
# Main Pipeline (async generator → SSE events)
# ============================================================

async def acquire_and_classify(
    obs_id: str, force: bool = False,
) -> AsyncGenerator[dict, None]:
    """
    Full pipeline: discover → download → classify → quickview → index update.

    Yields SSE events for real-time progress.
    """
    try:
        _validate_obs_id(obs_id)

        # Check if already fully done
        if not force and has_cached_result(obs_id):
            from .pipeline import load_cached_result
            result = load_cached_result(obs_id)
            yield {
                "event": "cached", "data": {
                    "obs_id": obs_id,
                    "stats": result.class_stats,
                    "elapsed": result.elapsed_seconds,
                },
            }
            return

        # Also check if data already downloaded (skip download phases)
        obs_dir = os.path.join(TRR_DATA_DIR, obs_id)
        data_exists = os.path.isdir(obs_dir) and any(
            f.upper().endswith("_TRR3.IMG") for f in os.listdir(obs_dir)
        ) and any(
            f.upper().endswith("_DDR1.IMG") for f in os.listdir(obs_dir)
        )

        discovery_result = None

        if not data_exists or force:
            # ── Phase 1: ODE Discovery ──
            yield {"event": "status", "data": {
                "step": "discovery", "message": f"Querying ODE for {obs_id}...",
            }}

            async with aiohttp.ClientSession() as session:
                discovery_result = await _discover_trr3(session, obs_id)

            total_size_mb = sum(f["size"] for f in discovery_result["files"]) / (1024 * 1024)
            yield {"event": "discovery", "data": {
                "trr3_id": discovery_result["trr3_product_id"],
                "base_key": discovery_result["base_key"],
                "files": len(discovery_result["files"]),
                "total_size_mb": round(total_size_mb, 1),
                "is_l_sensor": discovery_result["is_l_sensor"],
            }}

            if not discovery_result["is_l_sensor"]:
                yield {"event": "error", "data": {
                    "error": "Only S-sensor (VNIR) data available. CNN requires L-sensor (IR) data.",
                    "type": "s_sensor_only",
                }}
                return

            # ── Phase 2: Download ──
            target_dir = os.path.join(TRR_DATA_DIR, discovery_result["base_key"])

            # Yield events via a helper that captures them
            pending_events = []

            async def yield_download_event(evt):
                pending_events.append(evt)

            await _download_files(
                discovery_result["files"], target_dir, yield_download_event,
            )

            for evt in pending_events:
                yield evt

            yield {"event": "status", "data": {
                "step": "download_done",
                "message": "All files downloaded successfully",
            }}

            # Update obs_dir to the actual downloaded directory
            obs_dir = target_dir
            obs_id_for_pipeline = discovery_result["base_key"]
        else:
            obs_id_for_pipeline = obs_id
            yield {"event": "status", "data": {
                "step": "download_skipped",
                "message": "TRR3+DDR data already exists locally",
            }}

        # ── Phase 3: Classification ──
        async for event in run_classification(obs_id_for_pipeline, force=force):
            yield event
            # Check if classification errored out
            if event.get("event") == "error":
                return

        # ── Phase 4: Post-processing ──
        # Generate quickview
        yield {"event": "status", "data": {
            "step": "quickview", "message": "Generating quickview...",
        }}
        try:
            files = resolve_trr_files(obs_id_for_pipeline)
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None, _generate_quickview,
                os.path.dirname(files["trr_img"]), files["trr_img"], files["trr_lbl"],
            )
        except Exception as e:
            logger.warning(f"Quickview generation failed: {e}")

        # Update index.geojson
        if discovery_result and discovery_result.get("footprint"):
            yield {"event": "status", "data": {
                "step": "index_update", "message": "Updating footprint index...",
            }}
            added = await _update_index(
                discovery_result["base_key"],
                discovery_result["trr3_product_id"],
                discovery_result["footprint"],
            )
            yield {"event": "status", "data": {
                "step": "index_done",
                "message": "Footprint added to index" if added else "Footprint already in index",
            }}

        yield {"event": "pipeline_complete", "data": {
            "obs_id": obs_id_for_pipeline,
            "footprint_added": bool(discovery_result and discovery_result.get("footprint")),
        }}

    except ConnectionError as e:
        yield {"event": "error", "data": {"error": str(e), "type": "ode_unavailable"}}
    except FileNotFoundError as e:
        yield {"event": "error", "data": {"error": str(e), "type": "not_found"}}
    except IOError as e:
        yield {"event": "error", "data": {"error": str(e), "type": "download_failed"}}
    except ValueError as e:
        yield {"event": "error", "data": {"error": str(e), "type": "validation"}}
    except Exception as e:
        logger.error(f"Acquire pipeline failed for {obs_id}: {e}", exc_info=True)
        yield {"event": "error", "data": {"error": str(e), "type": "internal"}}
