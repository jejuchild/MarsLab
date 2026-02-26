#!/usr/bin/env python3
"""
Download HiRISE browse images for mid-latitude Glacial/Periglacial observations.

Targets: LDA, CCF, LVF, GLF + unlabeled mid-latitude glacial imagery.
Source: hirise_ice.db → derive browse URLs → download JPEGs.

Usage:
    python scripts/download_midlat_browse.py                    # download all
    python scripts/download_midlat_browse.py --labeled-only     # only LDA/CCF/LVF/GLF titled
    python scripts/download_midlat_browse.py --limit 50         # test with 50 images
    python scripts/download_midlat_browse.py --workers 8        # parallel downloads
"""

import argparse
import json
import logging
import os
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "backend", "data", "hirise_ice.db")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "Data", "HiRISE", "midlat_browse")
METADATA_PATH = os.path.join(os.path.dirname(__file__), "..", "Data", "HiRISE", "midlat_metadata.json")

# Mid-latitude band: 25-65 deg absolute latitude
LAT_MIN = 25.0
LAT_MAX = 65.0

TIMEOUT = 30  # seconds per download
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Label classification (title → class)
# ---------------------------------------------------------------------------

def classify_title(title: str) -> str:
    """Map HiRISE observation title to landform class."""
    t = title.lower()

    # LDA
    if any(kw in t for kw in [
        "lobate debris apron", " lda", "lda ",
        "fretted terrain", "apron"
    ]):
        if "debris flow" not in t:
            return "LDA"

    # CCF
    if any(kw in t for kw in ["concentric crater fill", " ccf", "ccf ", "concentric fill"]):
        return "CCF"

    # LVF
    if any(kw in t for kw in ["lineated valley fill", "lineated valley", " lvf", "lvf "]):
        return "LVF"

    # GLF
    if any(kw in t for kw in [
        "glacier", "glacial flow", "glaciated", "glacial-like",
        "viscous flow", "rock glacier", "tongue-shaped flow",
        "crevassed", "glacial landform",
    ]):
        return "GLF"

    # GLF — flow features in known GLF regions
    if "flow" in t and any(region in t for region in [
        "deuteronilus", "protonilus", "reull", "hellas", "phlegra",
        "nilosyrtis", "promethei", "eridania",
    ]):
        return "GLF"

    return "UNLABELED"


def derive_browse_url(jp2_url: str) -> str:
    """Convert JP2 URL to browse JPEG URL."""
    return (
        jp2_url
        .replace("/download/PDS/RDR/", "/PDS/EXTRAS/RDR/")
        .replace("_RED.JP2", "_RED.abrowse.jpg")
    )


# ---------------------------------------------------------------------------
# Database query
# ---------------------------------------------------------------------------

def get_target_observations(db_path: str, labeled_only: bool = False) -> list[dict]:
    """Query DB for mid-latitude Glacial/Periglacial observations."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("""
        SELECT image_id, title, science_theme, lat, lon, jp2_url
        FROM hirise_ice
        WHERE ABS(lat) BETWEEN ? AND ?
          AND science_theme = 'Glacial/Periglacial Processes'
          AND jp2_url IS NOT NULL
          AND jp2_url != ''
    """, (LAT_MIN, LAT_MAX))

    observations = []
    for row in cur.fetchall():
        obs = dict(row)
        obs["class"] = classify_title(obs["title"])
        obs["browse_url"] = derive_browse_url(obs["jp2_url"])
        observations.append(obs)

    conn.close()

    if labeled_only:
        observations = [o for o in observations if o["class"] != "UNLABELED"]

    return observations


# ---------------------------------------------------------------------------
# Download logic
# ---------------------------------------------------------------------------

def download_one(obs: dict, output_dir: str) -> dict:
    """Download a single browse image. Returns status dict."""
    image_id = obs["image_id"]
    filename = f"{image_id}_RED.abrowse.jpg"
    filepath = os.path.join(output_dir, filename)

    # Skip if already downloaded
    if os.path.exists(filepath) and os.path.getsize(filepath) > 10_000:
        return {"image_id": image_id, "status": "exists", "size": os.path.getsize(filepath)}

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(obs["browse_url"], timeout=TIMEOUT, stream=True)
            if r.status_code == 200:
                with open(filepath, "wb") as f:
                    for chunk in r.iter_content(chunk_size=65536):
                        f.write(chunk)
                size = os.path.getsize(filepath)
                if size > 10_000:  # valid JPEG should be >10KB
                    return {"image_id": image_id, "status": "downloaded", "size": size}
                else:
                    os.remove(filepath)
                    return {"image_id": image_id, "status": "too_small", "size": size}
            elif r.status_code == 404:
                return {"image_id": image_id, "status": "not_found", "size": 0}
            else:
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY * attempt)
                    continue
                return {"image_id": image_id, "status": f"http_{r.status_code}", "size": 0}
        except (requests.RequestException, OSError) as e:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)
                continue
            return {"image_id": image_id, "status": f"error:{type(e).__name__}", "size": 0}

    return {"image_id": image_id, "status": "max_retries", "size": 0}


def download_batch(
    observations: list[dict],
    output_dir: str,
    workers: int = 4,
    limit: int | None = None,
) -> dict:
    """Download browse images in parallel. Returns summary stats."""
    os.makedirs(output_dir, exist_ok=True)

    if limit:
        observations = observations[:limit]

    total = len(observations)
    stats = {"downloaded": 0, "exists": 0, "not_found": 0, "error": 0, "total_bytes": 0}
    class_counts = {}

    logger.info(f"Downloading {total} browse images → {output_dir}")
    logger.info(f"Workers: {workers}, Retries: {MAX_RETRIES}")

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(download_one, obs, output_dir): obs
            for obs in observations
        }

        for i, future in enumerate(as_completed(futures), 1):
            obs = futures[future]
            result = future.result()
            status = result["status"]

            if status == "downloaded":
                stats["downloaded"] += 1
                stats["total_bytes"] += result["size"]
            elif status == "exists":
                stats["exists"] += 1
                stats["total_bytes"] += result["size"]
            elif status == "not_found":
                stats["not_found"] += 1
            else:
                stats["error"] += 1

            # Track per-class
            cls = obs["class"]
            if cls not in class_counts:
                class_counts[cls] = {"downloaded": 0, "exists": 0, "total": 0}
            class_counts[cls]["total"] += 1
            if status in ("downloaded", "exists"):
                class_counts[cls][status] += 1

            if i % 50 == 0 or i == total:
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                eta_min = (total - i) / rate / 60 if rate > 0 else 0
                logger.info(
                    f"[{i}/{total}] ↓{stats['downloaded']} ✓{stats['exists']} "
                    f"✗{stats['not_found']} ⚠{stats['error']} "
                    f"({stats['total_bytes']/1e9:.1f} GB) "
                    f"ETA: {eta_min:.0f}min"
                )

    stats["class_counts"] = class_counts
    stats["elapsed_sec"] = time.time() - start_time
    return stats


# ---------------------------------------------------------------------------
# Metadata export
# ---------------------------------------------------------------------------

def save_metadata(observations: list[dict], output_path: str):
    """Save observation metadata with class labels for downstream use."""
    records = []
    for obs in observations:
        records.append({
            "image_id": obs["image_id"],
            "title": obs["title"],
            "class": obs["class"],
            "lat": obs["lat"],
            "lon": obs["lon"],
        })

    with open(output_path, "w") as f:
        json.dump(records, f, indent=2)
    logger.info(f"Saved metadata for {len(records)} observations → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Download mid-lat HiRISE browse images")
    parser.add_argument("--labeled-only", action="store_true",
                        help="Only download LDA/CCF/LVF/GLF titled images")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of downloads (for testing)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel download threads (default: 4)")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                        help="Output directory for browse images")
    args = parser.parse_args()

    db_path = os.path.abspath(DB_PATH)
    if not os.path.exists(db_path):
        logger.error(f"Database not found: {db_path}")
        sys.exit(1)

    # Query observations
    observations = get_target_observations(db_path, labeled_only=args.labeled_only)
    logger.info(f"Found {len(observations)} target observations in DB")

    # Class breakdown
    class_breakdown = {}
    for obs in observations:
        cls = obs["class"]
        class_breakdown[cls] = class_breakdown.get(cls, 0) + 1
    for cls, cnt in sorted(class_breakdown.items(), key=lambda x: -x[1]):
        logger.info(f"  {cls}: {cnt}")

    # Save metadata
    metadata_path = os.path.abspath(METADATA_PATH)
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    save_metadata(observations, metadata_path)

    # Download
    output_dir = os.path.abspath(args.output_dir)
    stats = download_batch(observations, output_dir, workers=args.workers, limit=args.limit)

    # Summary
    logger.info("=" * 60)
    logger.info(f"DOWNLOAD COMPLETE in {stats['elapsed_sec']:.0f}s")
    logger.info(f"  Downloaded: {stats['downloaded']}")
    logger.info(f"  Already existed: {stats['exists']}")
    logger.info(f"  Not found (404): {stats['not_found']}")
    logger.info(f"  Errors: {stats['error']}")
    logger.info(f"  Total size: {stats['total_bytes']/1e9:.2f} GB")
    logger.info("Per-class:")
    for cls, cnt in stats.get("class_counts", {}).items():
        logger.info(f"  {cls}: {cnt['downloaded']+cnt['exists']}/{cnt['total']} downloaded")


if __name__ == "__main__":
    main()
