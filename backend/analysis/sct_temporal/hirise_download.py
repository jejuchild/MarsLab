"""
Download full-resolution HiRISE RDR products for temporal analysis.

Uses ODE PRODUCTFILES API to discover JP2 URLs, downloads via aiohttp,
and converts to GeoTIFF via GDAL for rasterio-compatible I/O.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
from pathlib import Path
from typing import Optional

import aiohttp
import defusedxml.ElementTree as ET

logger = logging.getLogger(__name__)

ODE_PRODUCTFILES_URL = "https://ode.rsl.wustl.edu/mars/productfiles.aspx"

DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[3] / "Data" / "HiRISE" / "rdr_cache"


async def download_hirise_rdr(
    product_id: str,
    cache_dir: Optional[Path] = None,
    session: Optional[aiohttp.ClientSession] = None,
    force: bool = False,
) -> Optional[Path]:
    """
    Download a HiRISE RDR product and convert to GeoTIFF.

    Parameters
    ----------
    product_id : str
        HiRISE product ID, e.g. "ESP_016142_2240" (observation ID).
        Will search for the RED channel RDR product.
    cache_dir : Path, optional
        Directory to store downloaded files. Default: Data/HiRISE/rdr_cache/
    force : bool
        Re-download even if cached file exists.

    Returns
    -------
    Path to the GeoTIFF file, or None if download failed.
    """
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    obs_id = product_id.replace("_RED", "").strip()
    tif_path = cache_dir / f"{obs_id}_RED.tif"

    if tif_path.exists() and not force:
        logger.info(f"Using cached: {tif_path}")
        return tif_path

    # Discover JP2 URL from ODE
    jp2_url = await _resolve_jp2_url(obs_id, session)
    if jp2_url is None:
        logger.error(f"Could not resolve JP2 URL for {obs_id}")
        return None

    # Download JP2
    jp2_path = cache_dir / f"{obs_id}_RED.JP2"
    if not jp2_path.exists() or force:
        success = await _download_file(jp2_url, jp2_path, session)
        if not success:
            return None

    # Convert JP2 → GeoTIFF via GDAL
    if not tif_path.exists() or force:
        success = _jp2_to_geotiff(jp2_path, tif_path)
        if not success:
            return None

    return tif_path


async def _resolve_jp2_url(
    obs_id: str,
    session: Optional[aiohttp.ClientSession] = None,
) -> Optional[str]:
    """Resolve the JP2 download URL for a HiRISE observation via ODE PRODUCTFILES."""
    close_session = session is None
    if session is None:
        session = aiohttp.ClientSession()

    try:
        url = f"{ODE_PRODUCTFILES_URL}?ihid=MRO&iid=HIRISE&productid={obs_id}_RED*&output=XML"
        logger.info(f"Querying ODE PRODUCTFILES: {obs_id}")

        async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            if resp.status != 200:
                logger.error(f"ODE PRODUCTFILES returned {resp.status}")
                return None
            xml_text = await resp.text()
    except Exception as e:
        logger.error(f"ODE PRODUCTFILES query failed: {e}")
        return None
    finally:
        if close_session:
            await session.close()

    # Parse XML for JP2 URL
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        logger.error(f"Failed to parse ODE XML: {e}")
        return None

    # Look for the main JP2 file (not QLOOK, not NOMAP)
    for product_file in root.iter("Product_file"):
        url_elem = product_file.find("URL")
        fname_elem = product_file.find("FileName")
        if url_elem is None or fname_elem is None:
            continue

        fname = fname_elem.text or ""
        file_url = url_elem.text or ""

        if (
            fname.upper().endswith(".JP2")
            and "_RED" in fname.upper()
            and "QLOOK" not in fname.upper()
            and "NOMAP" not in fname.upper()
        ):
            logger.info(f"Found JP2: {fname} ({file_url[:80]}...)")
            return file_url

    logger.warning(f"No JP2 file found for {obs_id}")
    return None


async def _download_file(
    url: str,
    dest: Path,
    session: Optional[aiohttp.ClientSession] = None,
    chunk_size: int = 1024 * 1024,  # 1MB chunks
) -> bool:
    """Download a file with progress logging."""
    close_session = session is None
    if session is None:
        session = aiohttp.ClientSession()

    try:
        logger.info(f"Downloading: {url.split('/')[-1]}")
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=3600)) as resp:
            if resp.status != 200:
                logger.error(f"Download failed: HTTP {resp.status}")
                return False

            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            logged_pct = -1

            with open(dest, "wb") as f:
                async for chunk in resp.content.iter_chunked(chunk_size):
                    f.write(chunk)
                    downloaded += len(chunk)

                    if total > 0:
                        pct = int(100 * downloaded / total)
                        if pct >= logged_pct + 10:
                            logged_pct = pct
                            logger.info(
                                f"  {pct}% ({downloaded / 1e6:.1f} / {total / 1e6:.1f} MB)"
                            )

        logger.info(f"Downloaded: {dest.name} ({downloaded / 1e6:.1f} MB)")
        return True

    except Exception as e:
        logger.error(f"Download error: {e}")
        if dest.exists():
            dest.unlink()
        return False
    finally:
        if close_session:
            await session.close()


def _jp2_to_geotiff(jp2_path: Path, tif_path: Path) -> bool:
    """Convert JPEG2000 to GeoTIFF using GDAL."""
    try:
        result = subprocess.run(
            [
                "gdal_translate",
                "-of", "GTiff",
                "-co", "COMPRESS=LZW",
                "-co", "TILED=YES",
                "-co", "BIGTIFF=IF_SAFER",
                str(jp2_path),
                str(tif_path),
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )

        if result.returncode != 0:
            logger.error(f"gdal_translate failed: {result.stderr[:500]}")
            return False

        logger.info(f"Converted to GeoTIFF: {tif_path.name}")

        # Optionally remove JP2 to save disk space
        # jp2_path.unlink()

        return True

    except FileNotFoundError:
        logger.error(
            "gdal_translate not found. Install GDAL: "
            "apt-get install gdal-bin or conda install gdal"
        )
        return False
    except subprocess.TimeoutExpired:
        logger.error("gdal_translate timed out (>600s)")
        return False
