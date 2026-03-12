"""
Download full-resolution HiRISE RDR products for temporal analysis.

Primary method: direct download from hirise.lpl.arizona.edu using
predictable URL patterns. Fallback: ODE PRODUCTFILES API (XML).
Returns JP2 path directly (rasterio opens JP2 natively).
"""

from __future__ import annotations

import asyncio
import logging

from pathlib import Path
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)

HIRISE_BASE_URL = "https://hirise.lpl.arizona.edu/PDS"
ODE_PRODUCTFILES_URL = "https://ode.rsl.wustl.edu/mars/productfiles.aspx"

DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[3] / "Data" / "HiRISE" / "rdr_cache"


async def download_hirise_rdr(
    product_id: str,
    cache_dir: Optional[Path] = None,
    session: Optional[aiohttp.ClientSession] = None,
    force: bool = False,
) -> Optional[Path]:
    """
    Download a HiRISE RDR JP2 product.
    Returns Path to JP2 file, or None if download failed.
    """
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    obs_id = product_id.replace("_RED", "").strip()
    jp2_path = cache_dir / f"{obs_id}_RED.JP2"

    if jp2_path.exists() and not force:
        logger.info(f"Using cached JP2: {jp2_path.name}")
        return jp2_path

    jp2_url = _build_direct_url(obs_id)
    success = _download_file_wget(jp2_url, jp2_path)

    if not success:
        jp2_url_ode = await _resolve_jp2_url_ode(obs_id, session)
        if jp2_url_ode is not None:
            success = _download_file_wget(jp2_url_ode, jp2_path)

    if not success:
        logger.error(f"All download methods failed for {obs_id}")
        return None

    return jp2_path


def _build_direct_url(obs_id: str) -> str:
    """
    Build direct download URL for HiRISE RDR JP2 from observation ID.

    URL pattern:
      https://hirise.lpl.arizona.edu/PDS/RDR/{PHASE}/ORB_{LO}_{HI}/{OBS_ID}/{OBS_ID}_RED.JP2

    where PHASE = PSP or ESP, orbit range = floor/ceil to nearest 100.
    """
    parts = obs_id.split("_")
    phase = parts[0]
    orbit = int(parts[1])

    orb_lo = (orbit // 100) * 100
    orb_hi = orb_lo + 99

    return (
        f"{HIRISE_BASE_URL}/RDR/{phase}/"
        f"ORB_{orb_lo:06d}_{orb_hi:06d}/{obs_id}/{obs_id}_RED.JP2"
    )


async def _resolve_jp2_url_ode(
    obs_id: str,
    session: Optional[aiohttp.ClientSession] = None,
) -> Optional[str]:
    """Fallback: resolve JP2 URL via ODE PRODUCTFILES API."""
    close_session = session is None
    if session is None:
        session = aiohttp.ClientSession()

    try:
        url = (
            f"{ODE_PRODUCTFILES_URL}?ihid=MRO&iid=HIRISE"
            f"&productid={obs_id}_RED&output=XML"
        )
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


    try:
        import defusedxml.ElementTree as ET

        root = ET.fromstring(xml_text)
    except Exception as e:
        logger.warning(f"ODE returned non-XML response: {e}")
        return None

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
            logger.info(f"Found JP2 via ODE: {fname}")
            return file_url

    logger.warning(f"No JP2 file found via ODE for {obs_id}")
    return None


def _download_file_wget(url: str, dest: Path, retries: int = 3) -> bool:
    """Download via wget with retry and resume support for large HiRISE files (~1 GB)."""
    import subprocess

    for attempt in range(1, retries + 1):
        logger.info(f"wget attempt {attempt}/{retries}: {url.split('/')[-1]}")
        result = subprocess.run(
            [
                "wget", "-c",
                "-t", "3",
                "--timeout=60",
                "--waitretry=10",
                "-q", "--show-progress",
                "-O", str(dest),
                url,
            ],
            capture_output=True,
            text=True,
            timeout=7200,
        )
        if result.returncode == 0 and dest.exists() and dest.stat().st_size > 1_000_000:
            logger.info(f"Downloaded: {dest.name} ({dest.stat().st_size / 1e6:.0f} MB)")
            return True
        if result.returncode != 0:
            logger.warning(f"wget exit {result.returncode}: {result.stderr[:200]}")

    if dest.exists() and dest.stat().st_size < 1_000_000:
        dest.unlink()
    return False


