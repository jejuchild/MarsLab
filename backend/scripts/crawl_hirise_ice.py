#!/usr/bin/env python3
"""
crawl_hirise_ice.py — Crawl HiRISE catalog for ice-related imagery.

Scrapes uahirise.org for two science themes:
  - Glacial/Periglacial Processes (~7K images)
  - Polar Geology (~5K images)

Stores metadata in SQLite, downloads quickview JPGs and optionally RED JP2 files.

Usage:
    python crawl_hirise_ice.py --phase catalog        # Phase 1: scrape result pages
    python crawl_hirise_ice.py --phase detail          # Phase 2: scrape individual pages
    python crawl_hirise_ice.py --phase quickview       # Phase 3: download browse JPGs
    python crawl_hirise_ice.py --phase jp2             # Phase 4: download RED JP2s
    python crawl_hirise_ice.py --phase all             # Run all phases sequentially
    python crawl_hirise_ice.py --phase catalog --limit 100  # Limit entries (for testing)
"""

import argparse
import asyncio
import logging
import os
import re
import sqlite3
import sys
import time
from pathlib import Path
from urllib.parse import quote, urljoin

import aiohttp
from bs4 import BeautifulSoup
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_URL = "https://www.uahirise.org"
RESULTS_URL = f"{BASE_URL}/results.php"
PDS_EXTRAS = "https://hirise-pds.lpl.arizona.edu/PDS/EXTRAS/RDR"
PDS_DOWNLOAD = "https://hirise-pds.lpl.arizona.edu/download/PDS/RDR"

SCIENCE_THEMES = [
    "Glacial/Periglacial Processes",
    "Polar Geology",
]

RESULTS_PER_PAGE = 24
REQUEST_DELAY = 0.5          # seconds between uahirise.org requests
DOWNLOAD_CONCURRENCY = 4     # parallel file downloads
DOWNLOAD_RETRY = 3           # retries per file
DOWNLOAD_TIMEOUT = 600       # 10 min per file

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DATA_DIR / "hirise_ice.db"
QUICKVIEW_DIR = Path(__file__).resolve().parent.parent / "hirise_ice_data" / "quickview"
JP2_DIR = Path(__file__).resolve().parent.parent / "hirise_ice_data" / "jp2"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("crawl_hirise_ice")


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------
def init_db() -> sqlite3.Connection:
    """Create / open the SQLite database and ensure schema exists."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS hirise_ice (
            image_id        TEXT PRIMARY KEY,
            title           TEXT,
            science_theme   TEXT,
            lat             REAL,
            lon             REAL,
            caption         TEXT,
            acquisition_date TEXT,
            altitude_km     REAL,
            resolution_cm   REAL,
            emission_angle  REAL,
            phase_angle     REAL,
            solar_incidence REAL,
            solar_longitude REAL,
            quickview_path  TEXT,
            jp2_path        TEXT,
            quickview_url   TEXT,
            jp2_url         TEXT,
            detail_scraped  INTEGER DEFAULT 0,
            crawled_at      TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_hirise_ice_theme
        ON hirise_ice(science_theme)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_hirise_ice_latlon
        ON hirise_ice(lat, lon)
    """)
    conn.commit()
    return conn


def db_has_image(conn: sqlite3.Connection, image_id: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM hirise_ice WHERE image_id = ?", (image_id,)
    ).fetchone()
    return row is not None


def db_add_themes(conn: sqlite3.Connection, image_id: str, theme: str):
    """If image exists, merge theme; otherwise ignore (caller inserts first)."""
    row = conn.execute(
        "SELECT science_theme FROM hirise_ice WHERE image_id = ?", (image_id,)
    ).fetchone()
    if row and row[0] and theme not in row[0]:
        merged = f"{row[0]};{theme}"
        conn.execute(
            "UPDATE hirise_ice SET science_theme = ? WHERE image_id = ?",
            (merged, image_id),
        )


# ---------------------------------------------------------------------------
# URL construction helpers
# ---------------------------------------------------------------------------
def orb_range(image_id: str) -> str:
    """Derive ORB directory range from image_id.

    ESP_011544_0985 → orbit 011544 → ORB_011500_011599
    PSP_007235_2350 → orbit 007235 → ORB_007200_007299
    """
    m = re.match(r"(ESP|PSP|TRA)_(\d{6})_\d{4}", image_id)
    if not m:
        return ""
    orbit = int(m.group(2))
    lo = (orbit // 100) * 100
    hi = lo + 99
    return f"ORB_{lo:06d}_{hi:06d}"


def prefix_dir(image_id: str) -> str:
    """Get prefix directory (ESP, PSP, TRA)."""
    return image_id[:3]


def quickview_url(image_id: str) -> str:
    orb = orb_range(image_id)
    pfx = prefix_dir(image_id)
    if not orb:
        return ""
    return f"{PDS_EXTRAS}/{pfx}/{orb}/{image_id}/{image_id}_RED.abrowse.jpg"


def jp2_url(image_id: str) -> str:
    orb = orb_range(image_id)
    pfx = prefix_dir(image_id)
    if not orb:
        return ""
    return f"{PDS_DOWNLOAD}/{pfx}/{orb}/{image_id}/{image_id}_RED.JP2"


# ---------------------------------------------------------------------------
# Phase 1: Catalog Crawl
# ---------------------------------------------------------------------------
async def crawl_catalog(conn: sqlite3.Connection, limit: int = 0):
    """Scrape search result pages for both ice themes."""
    total_new = 0

    async with aiohttp.ClientSession() as session:
        for theme in SCIENCE_THEMES:
            log.info(f"Crawling theme: {theme}")
            page = 1

            # First request to get total pages
            params = {
                "keyword": "",
                "image_all": "all",
                "lat_beg": "",
                "lat_end": "",
                "lon_beg": "",
                "lon_end": "",
                "solar_all": "all",
                "order": "release_date",
                "science_theme": theme,
                "page": str(page),
            }

            resp = await session.get(RESULTS_URL, params=params)
            html = await resp.text()
            soup = BeautifulSoup(html, "html.parser")

            # Parse total pages from "Page X of Y pages (Z images)"
            total_pages = 1
            total_images = 0
            page_text = soup.get_text()
            m = re.search(r"Page\s+\d+\s+of\s+(\d+)\s+pages?\s+\((\d[\d,]*)\s+images?\)", page_text)
            if m:
                total_pages = int(m.group(1))
                total_images = int(m.group(2).replace(",", ""))
            log.info(f"  {theme}: {total_images} images across {total_pages} pages")

            if limit:
                max_pages = (limit + RESULTS_PER_PAGE - 1) // RESULTS_PER_PAGE
                total_pages = min(total_pages, max_pages)

            pbar = tqdm(total=total_pages, desc=f"  {theme[:20]}", unit="page")

            while page <= total_pages:
                if page > 1:
                    await asyncio.sleep(REQUEST_DELAY)
                    params["page"] = str(page)
                    resp = await session.get(RESULTS_URL, params=params)
                    html = await resp.text()
                    soup = BeautifulSoup(html, "html.parser")

                new_on_page = _parse_results_page(conn, soup, theme)
                total_new += new_on_page
                pbar.update(1)
                page += 1

                if limit and total_new >= limit:
                    break

            pbar.close()
            conn.commit()

    log.info(f"Catalog crawl done. {total_new} new entries added.")
    total = conn.execute("SELECT COUNT(*) FROM hirise_ice").fetchone()[0]
    log.info(f"Total entries in DB: {total}")


def _parse_results_page(
    conn: sqlite3.Connection, soup: BeautifulSoup, theme: str
) -> int:
    """Extract image entries from a search results page. Returns count of new."""
    new_count = 0

    # Structure: <a class="cells" href="ESP_...">Title (ESP_...)</a><br/>Lat: X° Long: Y°
    cells = soup.find_all("a", class_="cells")

    for link in cells:
        href = link.get("href", "")
        m = re.search(r"((?:ESP|PSP|TRA)_\d{6}_\d{4})", href)
        if not m:
            continue
        image_id = m.group(1)

        # Extract title from link text: "Gullies (ESP_011357_2285)"
        full_text = link.get_text(strip=True)
        title_m = re.match(r"(.+?)\s*\((?:ESP|PSP|TRA)_\d{6}_\d{4}\)\s*$", full_text)
        title = title_m.group(1).strip() if title_m else full_text

        # Coordinates are in the NavigableString after the <br/> following this link
        lat, lon = None, None
        sib = link.next_sibling
        # Skip past <br/> tags to find the text node
        for _ in range(3):
            if sib is None:
                break
            if isinstance(sib, str) and "Lat:" in sib:
                coord_m = re.search(
                    r"Lat:\s*([-\d.]+)°?\s*Long:\s*([-\d.]+)°?", sib
                )
                if coord_m:
                    lat = float(coord_m.group(1))
                    lon_val = float(coord_m.group(2))
                    # Normalize 0-360 East longitude to -180..180
                    lon = ((lon_val + 180) % 360) - 180
                break
            sib = sib.next_sibling

        if db_has_image(conn, image_id):
            db_add_themes(conn, image_id, theme)
            continue

        qv = quickview_url(image_id)
        j2 = jp2_url(image_id)

        conn.execute(
            """INSERT OR IGNORE INTO hirise_ice
               (image_id, title, science_theme, lat, lon, quickview_url, jp2_url)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (image_id, title, theme, lat, lon, qv, j2),
        )
        new_count += 1

    return new_count


# ---------------------------------------------------------------------------
# Phase 2: Detail Scrape
# ---------------------------------------------------------------------------
async def scrape_details(conn: sqlite3.Connection, limit: int = 0):
    """Scrape individual image pages for detailed metadata."""
    rows = conn.execute(
        "SELECT image_id FROM hirise_ice WHERE detail_scraped = 0"
    ).fetchall()
    if limit:
        rows = rows[:limit]

    log.info(f"Scraping details for {len(rows)} images...")

    async with aiohttp.ClientSession() as session:
        for i, (image_id,) in enumerate(tqdm(rows, desc="  Details", unit="img")):
            url = f"{BASE_URL}/{image_id}"
            try:
                await asyncio.sleep(REQUEST_DELAY)
                resp = await session.get(url, timeout=aiohttp.ClientTimeout(total=30))
                if resp.status != 200:
                    log.warning(f"  {image_id}: HTTP {resp.status}")
                    continue
                html = await resp.text()
                meta = _parse_detail_page(html)
                _update_detail(conn, image_id, meta)
            except Exception as e:
                log.warning(f"  {image_id}: {e}")
                continue

            if (i + 1) % 50 == 0:
                conn.commit()

    conn.commit()
    scraped = conn.execute(
        "SELECT COUNT(*) FROM hirise_ice WHERE detail_scraped = 1"
    ).fetchone()[0]
    log.info(f"Detail scrape done. {scraped} images with full metadata.")


def _parse_detail_page(html: str) -> dict:
    """Extract metadata from an individual HiRISE image page."""
    soup = BeautifulSoup(html, "html.parser")
    meta: dict = {}

    # Caption — in <div class="caption-text">
    caption_div = soup.find("div", class_="caption-text")
    if caption_div:
        # Strip "Written by..." suffix
        cap_text = caption_div.get_text(" ", strip=True)
        written_idx = cap_text.find("Written by")
        if written_idx > 0:
            cap_text = cap_text[:written_idx].strip()
        meta["caption"] = cap_text[:2000]

    # Use image-details-container for technical metadata
    details = soup.find("div", class_="image-details-container")
    text = details.get_text(" ", strip=True) if details else soup.get_text(" ", strip=True)

    # Coordinates
    lat_m = re.search(r"Latitude\s*\(centered\)\s*([-\d.]+)", text)
    lon_m = re.search(r"Longitude\s*\(East\)\s*([-\d.]+)", text)
    if lat_m:
        meta["lat"] = float(lat_m.group(1))
    if lon_m:
        lon_val = float(lon_m.group(1))
        meta["lon"] = ((lon_val + 180) % 360) - 180

    # Acquisition date
    date_m = re.search(r"Acquisition\s+date\s*(\d{1,2}\s+\w+\s+\d{4})", text)
    if date_m:
        meta["acquisition_date"] = date_m.group(1)

    # Altitude
    alt_m = re.search(r"Spacecraft\s+altitude\s*([\d.]+)\s*km", text, re.I)
    if alt_m:
        meta["altitude_km"] = float(alt_m.group(1))

    # Resolution
    res_m = re.search(r"Original\s+image\s+scale\s+range?\s*([\d.]+)\s*cm", text, re.I)
    if not res_m:
        res_m = re.search(r"([\d.]+)\s*cm/pixel", text, re.I)
    if res_m:
        meta["resolution_cm"] = float(res_m.group(1))

    # Emission angle
    em_m = re.search(r"Emission\s+[Aa]ngle\s*([\d.]+)", text)
    if em_m:
        meta["emission_angle"] = float(em_m.group(1))

    # Phase angle
    ph_m = re.search(r"Phase\s+[Aa]ngle\s*([\d.]+)", text)
    if ph_m:
        meta["phase_angle"] = float(ph_m.group(1))

    # Solar incidence
    si_m = re.search(r"Solar\s+[Ii]ncidence\s+[Aa]ngle\s*,?\s*([\d.]+)", text)
    if si_m:
        meta["solar_incidence"] = float(si_m.group(1))

    # Solar longitude (Ls)
    sl_m = re.search(r"Solar\s+[Ll]ongitude\s*([\d.]+)", text)
    if sl_m:
        meta["solar_longitude"] = float(sl_m.group(1))

    return meta


def _update_detail(conn: sqlite3.Connection, image_id: str, meta: dict):
    """Update a row with detail-scraped metadata."""
    fields = []
    values = []
    for key in [
        "caption", "acquisition_date", "altitude_km", "resolution_cm",
        "emission_angle", "phase_angle", "solar_incidence", "solar_longitude",
    ]:
        if key in meta:
            fields.append(f"{key} = ?")
            values.append(meta[key])

    # Update lat/lon only if we got better data from the detail page
    if "lat" in meta:
        fields.append("lat = COALESCE(?, lat)")
        values.append(meta["lat"])
    if "lon" in meta:
        fields.append("lon = COALESCE(?, lon)")
        values.append(meta["lon"])

    fields.append("detail_scraped = 1")
    values.append(image_id)

    if fields:
        sql = f"UPDATE hirise_ice SET {', '.join(fields)} WHERE image_id = ?"
        conn.execute(sql, values)


# ---------------------------------------------------------------------------
# Phase 3 & 4: File Downloads
# ---------------------------------------------------------------------------
async def download_files(
    conn: sqlite3.Connection,
    file_type: str,
    limit: int = 0,
):
    """Download quickview JPGs or JP2 files.

    file_type: 'quickview' or 'jp2'
    """
    if file_type == "quickview":
        out_dir = QUICKVIEW_DIR
        url_col = "quickview_url"
        path_col = "quickview_path"
        ext = ".jpg"
    else:
        out_dir = JP2_DIR
        url_col = "jp2_url"
        path_col = "jp2_path"
        ext = ".JP2"

    out_dir.mkdir(parents=True, exist_ok=True)

    # Get rows that need downloading (no local path yet, have a URL)
    rows = conn.execute(
        f"SELECT image_id, {url_col} FROM hirise_ice "
        f"WHERE {path_col} IS NULL AND {url_col} IS NOT NULL AND {url_col} != ''"
    ).fetchall()
    if limit:
        rows = rows[:limit]

    log.info(f"Downloading {len(rows)} {file_type} files...")

    semaphore = asyncio.Semaphore(DOWNLOAD_CONCURRENCY)
    timeout = aiohttp.ClientTimeout(total=DOWNLOAD_TIMEOUT)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = []
        for image_id, url in rows:
            filename = f"{image_id}_RED{ext}" if file_type == "jp2" else f"{image_id}.jpg"
            dest = out_dir / filename
            tasks.append(
                _download_one(session, semaphore, url, dest, image_id, path_col, conn)
            )

        results = []
        for coro in tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc=f"  {file_type}",
            unit="file",
        ):
            result = await coro
            results.append(result)
            # Commit every 20 downloads
            if len(results) % 20 == 0:
                conn.commit()

    conn.commit()
    ok = sum(1 for r in results if r)
    log.info(f"Downloaded {ok}/{len(rows)} {file_type} files.")


async def _download_one(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    url: str,
    dest: Path,
    image_id: str,
    path_col: str,
    conn: sqlite3.Connection,
) -> bool:
    """Download a single file with retries."""
    if dest.exists() and dest.stat().st_size > 0:
        # Already downloaded — just update DB
        conn.execute(
            f"UPDATE hirise_ice SET {path_col} = ? WHERE image_id = ?",
            (str(dest), image_id),
        )
        return True

    async with sem:
        for attempt in range(DOWNLOAD_RETRY):
            try:
                async with session.get(url) as resp:
                    if resp.status == 404:
                        log.debug(f"  {image_id}: 404 — skipping")
                        return False
                    if resp.status != 200:
                        log.warning(f"  {image_id}: HTTP {resp.status} (attempt {attempt+1})")
                        await asyncio.sleep(2 ** attempt)
                        continue

                    # Stream to disk
                    tmp = dest.with_suffix(".tmp")
                    with open(tmp, "wb") as f:
                        async for chunk in resp.content.iter_chunked(1024 * 256):
                            f.write(chunk)
                    tmp.rename(dest)

                    conn.execute(
                        f"UPDATE hirise_ice SET {path_col} = ? WHERE image_id = ?",
                        (str(dest), image_id),
                    )
                    return True

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                log.warning(f"  {image_id}: {e} (attempt {attempt+1})")
                await asyncio.sleep(2 ** attempt)

    log.error(f"  {image_id}: failed after {DOWNLOAD_RETRY} attempts")
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def run(args):
    conn = init_db()

    phases = [p.strip() for p in args.phase.split(",")]
    run_all = "all" in phases

    if run_all or "catalog" in phases:
        log.info("=== Phase 1: Catalog Crawl ===")
        await crawl_catalog(conn, limit=args.limit)

    if run_all or "detail" in phases:
        log.info("=== Phase 2: Detail Scrape ===")
        await scrape_details(conn, limit=args.limit)

    if run_all or "quickview" in phases:
        log.info("=== Phase 3: Quickview Download ===")
        await download_files(conn, "quickview", limit=args.limit)

    if run_all or "jp2" in phases:
        log.info("=== Phase 4: JP2 Download ===")
        await download_files(conn, "jp2", limit=args.limit)

    # Summary
    total = conn.execute("SELECT COUNT(*) FROM hirise_ice").fetchone()[0]
    detailed = conn.execute("SELECT COUNT(*) FROM hirise_ice WHERE detail_scraped = 1").fetchone()[0]
    qv = conn.execute("SELECT COUNT(*) FROM hirise_ice WHERE quickview_path IS NOT NULL").fetchone()[0]
    j2 = conn.execute("SELECT COUNT(*) FROM hirise_ice WHERE jp2_path IS NOT NULL").fetchone()[0]
    log.info(f"=== Summary ===")
    log.info(f"  Total entries:     {total}")
    log.info(f"  Detail scraped:    {detailed}")
    log.info(f"  Quickviews:        {qv}")
    log.info(f"  JP2 files:         {j2}")

    conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Crawl HiRISE catalog for ice-related imagery"
    )
    parser.add_argument(
        "--phase",
        default="all",
        help="Phases to run: catalog, detail, quickview, jp2, all (comma-separated)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit entries per phase (0 = unlimited, useful for testing)",
    )
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
