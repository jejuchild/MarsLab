"""
Product URL resolver — exposes direct PDS download URLs for HiRISE JP2
and CRISM browse products via ODE API.

Endpoints:
  GET /api/product-urls/hirise/{product_id}
  GET /api/product-urls/crism/{product_id}
"""

import logging
from typing import Optional

import aiohttp
from fastapi import APIRouter, HTTPException, Request
from cachetools import TTLCache

from api.ode_client import (
    resolve_hirise_bundle,
    resolve_crism_bundle,
    parse_crism_base_key,
)
from api.rate_limit import limiter
from api.validation import validate_product_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/product-urls", tags=["ProductURLs"])

# Cache resolved URLs for 30 minutes (ODE URLs are stable)
_url_cache: TTLCache = TTLCache(maxsize=512, ttl=1800)


def _cache_key(instrument: str, product_id: str) -> str:
    return f"{instrument}:{product_id}"


@router.get("/hirise/{product_id}")
@limiter.limit("30/minute")
async def get_hirise_urls(request: Request, product_id: str):
    """
    Resolve direct PDS download URLs for a HiRISE product.

    Returns JP2 image URL, label URL, and file size.
    Uses ODE PRODUCTFILES API to discover file locations.
    """
    try:
        product_id = validate_product_id(product_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid product ID format")

    cache_key = _cache_key("hirise", product_id)
    if cache_key in _url_cache:
        return _url_cache[cache_key]

    session: Optional[aiohttp.ClientSession] = getattr(
        request.app.state, "http_session", None
    )

    try:
        bundle = await resolve_hirise_bundle(product_id, session)
    except Exception as e:
        logger.error(f"Failed to resolve HiRISE bundle for {product_id}: {e}")
        raise HTTPException(status_code=502, detail="Failed to resolve URLs from ODE")

    result = {
        "product_id": product_id,
        "instrument": "hirise",
        "jp2_url": bundle.jp2_file.url if bundle.jp2_file else None,
        "jp2_size_bytes": bundle.jp2_file.size_bytes if bundle.jp2_file else None,
        "jp2_filename": bundle.jp2_file.filename if bundle.jp2_file else None,
        "lbl_url": bundle.lbl_file.url if bundle.lbl_file else None,
    }

    _url_cache[cache_key] = result
    return result


@router.get("/crism/{product_id}")
@limiter.limit("30/minute")
async def get_crism_urls(request: Request, product_id: str):
    """
    Resolve direct PDS download URLs for a CRISM product.

    Returns core product URLs and browse product URLs (HYD, ICE, IC2, VNA, etc.).
    Uses ODE REST API to discover file locations.
    """
    try:
        product_id = validate_product_id(product_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid product ID format")

    base_key = parse_crism_base_key(product_id)
    cache_key = _cache_key("crism", base_key)
    if cache_key in _url_cache:
        return _url_cache[cache_key]

    session: Optional[aiohttp.ClientSession] = getattr(
        request.app.state, "http_session", None
    )

    try:
        bundle = await resolve_crism_bundle(base_key, session)
    except Exception as e:
        logger.error(f"Failed to resolve CRISM bundle for {base_key}: {e}")
        raise HTTPException(status_code=502, detail="Failed to resolve URLs from ODE")

    # Map browse files by type abbreviation
    browse_urls: dict[str, str] = {}
    for bf in bundle.browse_files:
        # Extract browse type from filename: frt..._br{TYPE}j_mtr3.png
        fname = bf.filename.lower()
        if "_br" in fname:
            # e.g. "frt0001fd76_07_brhydj_mtr3.png" -> "hyd"
            parts = fname.split("_br")
            if len(parts) >= 2:
                type_part = parts[1]  # "hydj_mtr3.png"
                # Remove variant suffix letter + rest
                br_type = ""
                for ch in type_part:
                    if ch.isalpha() and len(br_type) < 3:
                        br_type += ch
                    else:
                        break
                if br_type:
                    browse_urls[br_type] = bf.url

    result = {
        "product_id": product_id,
        "base_key": base_key,
        "instrument": "crism",
        "product_type": bundle.product_type,
        "img_url": bundle.img_file.url if bundle.img_file else None,
        "img_filename": bundle.img_file.filename if bundle.img_file else None,
        "lbl_url": bundle.lbl_file.url if bundle.lbl_file else None,
        "browse_urls": browse_urls,
    }

    _url_cache[cache_key] = result
    return result
