#!/usr/bin/env python3
"""
Download HiRISE browse JPEGs for all entries in hirise_ice.db.

Reads quickview_url from the database, skips already-downloaded files,
downloads the rest with async concurrency.

Usage:
    python scripts/landform_pipeline/download_browse.py
    python scripts/landform_pipeline/download_browse.py --limit 100
    python scripts/landform_pipeline/download_browse.py --concurrency 8
"""

import argparse
import asyncio
import logging
import os
import sqlite3
import time
from pathlib import Path

import aiohttp
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("download_browse")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = PROJECT_ROOT / "backend" / "data" / "hirise_ice.db"
OUTPUT_DIR = PROJECT_ROOT / "Data" / "HiRISE" / "midlat_browse"
ARCADIA_DIR = PROJECT_ROOT / "arcadia_hirise" / "jpeg"

TIMEOUT = aiohttp.ClientTimeout(total=120, connect=30)
MAX_RETRIES = 3
RETRY_DELAY = 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download HiRISE browse JPEGs")
    parser.add_argument("--limit", type=int, default=0, help="Max images to download (0=all)")
    parser.add_argument("--concurrency", type=int, default=6, help="Parallel downloads")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Download directory")
    parser.add_argument("--db-path", type=Path, default=DB_PATH, help="SQLite DB path")
    return parser.parse_args()


def get_download_queue(db_path: Path, output_dir: Path, limit: int) -> list[tuple[str, str, str]]:
    """Return list of (image_id, url, dest_path) for images not yet downloaded."""
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        "SELECT image_id, quickview_url FROM hirise_ice "
        "WHERE quickview_url IS NOT NULL AND quickview_url != '' "
        "ORDER BY image_id"
    ).fetchall()
    conn.close()

    # Build set of already-downloaded filenames (check both dirs)
    existing: set[str] = set()
    for d in [output_dir, ARCADIA_DIR]:
        if d.is_dir():
            for f in d.iterdir():
                if f.suffix.lower() in {".jpg", ".jpeg"}:
                    existing.add(f.name)

    queue: list[tuple[str, str, str]] = []
    for image_id, url in rows:
        # Derive expected filename from URL
        filename = url.rsplit("/", 1)[-1] if "/" in url else f"{image_id}_RED.abrowse.jpg"
        if filename in existing:
            continue
        dest = str(output_dir / filename)
        queue.append((image_id, url, dest))

    if limit > 0:
        queue = queue[:limit]

    return queue


async def download_one(
    session: aiohttp.ClientSession,
    image_id: str,
    url: str,
    dest: str,
    semaphore: asyncio.Semaphore,
    pbar: tqdm,
) -> bool:
    """Download a single file with retries."""
    async with semaphore:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                async with session.get(url) as resp:
                    if resp.status == 404:
                        log.debug("404 for %s — skipping", image_id)
                        pbar.update(1)
                        return False
                    resp.raise_for_status()
                    data = await resp.read()
                    tmp = dest + ".tmp"
                    with open(tmp, "wb") as f:
                        f.write(data)
                    os.rename(tmp, dest)
                    pbar.update(1)
                    return True
            except Exception as e:
                if attempt == MAX_RETRIES:
                    log.warning("Failed %s after %d attempts: %s", image_id, MAX_RETRIES, e)
                    pbar.update(1)
                    return False
                await asyncio.sleep(RETRY_DELAY * attempt)
    pbar.update(1)
    return False


async def run_downloads(queue: list[tuple[str, str, str]], concurrency: int) -> tuple[int, int]:
    """Download all queued files. Returns (success, failed) counts."""
    semaphore = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(limit=concurrency + 2, enable_cleanup_closed=True)

    success = 0
    failed = 0

    async with aiohttp.ClientSession(timeout=TIMEOUT, connector=connector) as session:
        pbar = tqdm(total=len(queue), desc="Downloading", unit="img")
        tasks = [
            download_one(session, image_id, url, dest, semaphore, pbar)
            for image_id, url, dest in queue
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        pbar.close()

        for r in results:
            if isinstance(r, Exception):
                failed += 1
            elif r:
                success += 1
            else:
                failed += 1

    return success, failed


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Scanning DB for download queue...")
    queue = get_download_queue(args.db_path, args.output_dir, args.limit)
    log.info("Download queue: %d images (already downloaded skipped)", len(queue))

    if not queue:
        log.info("Nothing to download — all images already present.")
        return

    est_gb = len(queue) * 3.0 / 1024  # ~3MB avg per image
    log.info("Estimated download size: ~%.1f GB", est_gb)
    log.info("Concurrency: %d", args.concurrency)

    t0 = time.time()
    success, failed = asyncio.run(run_downloads(queue, args.concurrency))
    elapsed = time.time() - t0

    log.info("Done in %.0f min (%.0f s)", elapsed / 60, elapsed)
    log.info("Success: %d, Failed: %d", success, failed)

    # Count total images now
    total = sum(1 for f in args.output_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg"})
    log.info("Total images in %s: %d", args.output_dir, total)


if __name__ == "__main__":
    main()
