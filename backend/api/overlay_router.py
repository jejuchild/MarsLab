"""Image overlay / quickview / world-tile endpoints.

HiRISE, CRISM, HiRISE-DTM and SHARAD quickview + overlay PNGs (with the
on-disk overlay cache) and the legacy HiRISE world-tile endpoints. Image math
lives in ``api.image_pipeline``; this module handles file lookup, caching and
HTTP responses.
"""
import json
import logging
import os
from functools import lru_cache

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, Response

from api.image_pipeline import (
    MAX_ZOOM,
    TILE_SIZE,
    ImagePipelineError,
    blank_png,
    cleanup_overlay_cache,
    crism_image_to_rgba,
    crop_to_aspect,
    downscale_to_width,
    ds_world_bounds,
    encode_png,
    encode_rgba_png,
    geotiff_to_png,
    gray_to_rgba,
    hirise_gray_from_jpg,
    hirise_gray_from_raster,
    open_ds,
    read_jp2_browse,
    render_world_tile,
    reproject_polar_browse,
    trr3_quickview_png,
)
from api.geojson_cache import _geojson_cache
from api.paths import BASE_DIR, HIRISE_DATA_DIR, HIRISE_DTM_DIR, OVERLAY_CACHE_DIR
from api.rate_limit import limiter
from api.validation import validate_product_id

logger = logging.getLogger("marslab")

router = APIRouter()


# ======================================================
# Utils
# ======================================================
def tif_path(name: str) -> str:
    return os.path.join(HIRISE_DATA_DIR, f"{name}_RED.tif")


@lru_cache(maxsize=1)
def world_union_extent():
    left = bottom = float("inf")
    right = top = float("-inf")

    if not os.path.exists(HIRISE_DATA_DIR):
        raise RuntimeError(f"HIRISE_DATA_DIR not found: {HIRISE_DATA_DIR}")

    for fn in os.listdir(HIRISE_DATA_DIR):
        if not fn.lower().endswith(".tif"):
            continue
        ds = open_ds(os.path.join(HIRISE_DATA_DIR, fn))
        l, b, r, t = ds_world_bounds(ds)
        left = min(left, l)
        bottom = min(bottom, b)
        right = max(right, r)
        top = max(top, t)

    if not np.isfinite(left):
        raise RuntimeError("No .tif found in DATA_DIR")

    return (left, bottom, right, top)


# ======================================================
# Meta endpoints
# ======================================================
@router.get("/meta/{name}")
def get_meta(name: str):
    path = tif_path(name)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    ds = open_ds(path)
    return {"width": ds.width, "height": ds.height}


@router.get("/world_meta")
def get_world_meta():
    try:
        left, bottom, right, top = world_union_extent()
    except Exception as e:
        # ✅ 500 대신 원인 JSON 반환 (디버깅 핵심)
        return {
            "error": str(e),
            "data_dir": HIRISE_DATA_DIR,
            "exists": os.path.exists(HIRISE_DATA_DIR),
            "files": os.listdir(HIRISE_DATA_DIR) if os.path.exists(HIRISE_DATA_DIR) else None,
        }

    return {
        "extent": [left, bottom, right, top],
        "tile_size": TILE_SIZE,
        "max_zoom": MAX_ZOOM,
    }


# ======================================================
# Tile cache
# ======================================================
@lru_cache(maxsize=8192)
def load_world_tile(name: str, z: int, x: int, y: int) -> bytes:
    path = tif_path(name)
    if not os.path.exists(path):
        return blank_png()

    ds = open_ds(path)
    return render_world_tile(ds, world_union_extent(), z, x, y)


@router.get("/world_tiles/{name}/{z}/{x}/{y}.png")
def get_world_tile(name: str, z: int, x: int, y: int):
    if z < 0 or z > MAX_ZOOM + 2:
        return Response(blank_png(), media_type="image/png")

    png = load_world_tile(name, z, x, y)
    return Response(png, media_type="image/png", headers={"Cache-Control": "public, max-age=86400"})


# Overlay cache directory with LRU cleanup
os.makedirs(OVERLAY_CACHE_DIR, exist_ok=True)

OVERLAY_CACHE_MAX_MB = 500  # Max cache size in MB
OVERLAY_CACHE_TTL_DAYS = 30  # Max age for cache entries


def _cleanup_overlay_cache():
    """Evict oldest files when cache exceeds size limit or TTL."""
    cleanup_overlay_cache(OVERLAY_CACHE_DIR, OVERLAY_CACHE_MAX_MB, OVERLAY_CACHE_TTL_DAYS)


# Run cleanup on startup (module import)
_cleanup_overlay_cache()


@router.get("/hirise/overlay/{product_id}.png")
@limiter.limit("20/minute")
def get_hirise_overlay(request: Request, product_id: str, max_size: int = 2048):
    """
    Serve HiRISE image at reduced resolution with transparent background.
    Black pixels (DN=0) are made transparent.
    Uses disk cache for faster subsequent loads.
    """
    try:
        product_id = validate_product_id(product_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid product ID format")

    # Check cache first
    cache_file = os.path.join(OVERLAY_CACHE_DIR, f"{product_id}_{max_size}.png")
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return Response(f.read(), media_type="image/png", headers={"Cache-Control": "public, max-age=86400"})

    # Fast path: use existing quickview JPG (already downsampled, instant)
    qv_jpg = os.path.join(BASE_DIR, "hirise_quickview", f"{product_id}.jpg")
    tif = tif_path(product_id)
    source = qv_jpg if os.path.exists(qv_jpg) else tif if os.path.exists(tif) else None

    if not source:
        raise HTTPException(status_code=404, detail=f"HiRISE data not found: {product_id}")

    try:
        if source.endswith(".jpg"):
            gray = hirise_gray_from_jpg(source, max_size)
            if gray is None:
                raise HTTPException(status_code=500, detail="Failed to read quickview")
        else:
            gray = hirise_gray_from_raster(open_ds(source), max_size)

        rgba = gray_to_rgba(gray, 3)

        png_bytes = encode_rgba_png(rgba, [cv2.IMWRITE_PNG_COMPRESSION, 4])
        if png_bytes is None:
            raise HTTPException(status_code=500, detail="Failed to encode PNG")

        with open(cache_file, "wb") as f:
            f.write(png_bytes)

        return Response(png_bytes, media_type="image/png", headers={"Cache-Control": "public, max-age=86400"})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _get_hirise_rdr_props(product_id: str) -> tuple[int, int] | None:
    """Get (rdr_lines, rdr_samples) from HiRISE index cache for aspect-ratio correction.
    Returns None for polar stereographic products (|proj_center_lat| >= 85)
    where equirectangular aspect-ratio correction doesn't apply."""
    cache = _geojson_cache.get("hirise")
    if not cache:
        return None
    for feat in cache.get("features", []):
        props = feat.get("properties", {})
        if props.get("product_id") == product_id:
            proj_lat = props.get("proj_center_lat")
            if proj_lat is not None and abs(proj_lat) >= 85:
                return None  # polar stereographic — skip aspect correction
            lines = props.get("rdr_lines")
            samples = props.get("rdr_samples")
            if lines and samples:
                return (int(lines), int(samples))
            return None
    return None


def _get_hirise_feature(product_id: str) -> dict | None:
    """Get full feature dict from HiRISE index cache."""
    cache = _geojson_cache.get("hirise")
    if not cache:
        return None
    for feat in cache.get("features", []):
        if feat.get("properties", {}).get("product_id") == product_id:
            return feat
    return None


@router.get("/hirise/quickview/{product_id}.png")
def get_hirise_quickview_transparent(request: Request, product_id: str):
    """
    Serve HiRISE quickview image with transparent background.
    Black pixels are made transparent.  Results are cached to disk.
    """
    try:
        product_id = validate_product_id(product_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid product ID format")

    # Check for cached transparent PNG first (fast path)
    cache_dir = os.path.join(BASE_DIR, "hirise_quickview", ".png_cache")
    cached_png = os.path.join(cache_dir, f"{product_id}.png")
    if os.path.exists(cached_png):
        return FileResponse(cached_png, media_type="image/png")

    # Try JPG first (most common)
    jpg_path = os.path.join(BASE_DIR, "hirise_quickview", f"{product_id}.jpg")
    png_path = os.path.join(BASE_DIR, "hirise_quickview", f"{product_id}.png")

    path = jpg_path if os.path.exists(jpg_path) else png_path
    if not os.path.exists(path):
        # Fallback: generate quickview from downloaded JP2 data
        jp2_dir = os.path.join(HIRISE_DATA_DIR, f"{product_id}_RED")
        jp2_path = os.path.join(jp2_dir, f"{product_id}_RED.JP2")
        if not os.path.exists(jp2_path):
            raise HTTPException(status_code=404, detail=f"Quickview not found: {product_id}")
        try:
            arr = read_jp2_browse(jp2_path)
            # Save as JPG for future use
            cv2.imwrite(jpg_path, arr)
            path = jpg_path
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to generate quickview: {e}")

    try:
        # Read image
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise HTTPException(status_code=500, detail="Failed to read image")

        # Convert to grayscale if needed
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # Crop browse image to match full product aspect ratio.
        rdr_props = _get_hirise_rdr_props(product_id)
        if rdr_props:
            rdr_lines, rdr_samples = rdr_props
            gray = crop_to_aspect(gray, rdr_lines, rdr_samples)

        # Reproject polar stereographic browse images to equirectangular
        # so the overlay aligns correctly on the geographic map
        feat = _get_hirise_feature(product_id)
        if feat:
            proj_lat = feat.get("properties", {}).get("proj_center_lat")
            if proj_lat is not None and abs(proj_lat) >= 85:
                try:
                    gray = reproject_polar_browse(gray, feat)
                except Exception as e:
                    logger.warning(f"[QuickView] Polar reprojection failed for {product_id}: {e}")

        # Downscale large images to ~1024px wide for faster overlay rendering
        gray = downscale_to_width(gray, 1024)

        # Create RGBA with transparent black pixels (threshold at 5)
        rgba = gray_to_rgba(gray, 5)

        # Encode as PNG with compression
        png_bytes = encode_rgba_png(rgba, [cv2.IMWRITE_PNG_COMPRESSION, 6])
        if png_bytes is None:
            raise HTTPException(status_code=500, detail="Failed to encode PNG")

        # Cache to disk for future requests
        os.makedirs(cache_dir, exist_ok=True)
        with open(cached_png, "wb") as f:
            f.write(png_bytes)

        return Response(png_bytes, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@lru_cache(maxsize=16)
def _cached_pds3_label(lbl_path: str) -> dict:
    """Cache PDS3 label parsing to avoid re-reading on repeated requests."""
    from api.mineral_cnn.data_loader import parse_pds3_label
    return parse_pds3_label(lbl_path)


def _generate_trr3_quickview(trr_img_path: str, trr_lbl_path: str, cache_dir: str) -> Response:
    """Generate a VNIR quickview PNG from a TRR3 cube and cache it."""
    meta = _cached_pds3_label(trr_lbl_path)
    try:
        png_bytes = trr3_quickview_png(trr_img_path, meta)
    except ImagePipelineError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Cache the quickview
    cache_path = os.path.join(cache_dir, "quickview.png")
    try:
        with open(cache_path, "wb") as f:
            f.write(png_bytes)
    except OSError:
        pass

    return Response(png_bytes, media_type="image/png",
                    headers={"Cache-Control": "public, max-age=86400"})


@router.get("/crism/quickview/{product_id}.png")
@limiter.limit("20/minute")
def get_crism_quickview_transparent(request: Request, product_id: str):
    """
    Serve CRISM quickview image with transparent background.
    Black pixels are made transparent.
    Searches multiple locations: crism_quickview/, crism_data/ browse images.
    """
    try:
        product_id = validate_product_id(product_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid product ID format")

    obs_id = product_id.split("_")[0]  # e.g. frt00008a1e
    # base = obs_id + _NN (e.g. frt00008a1e_07)
    parts = product_id.split("_")
    base = f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else product_id

    # Try all possible file locations in order of preference
    candidates = [
        os.path.join(BASE_DIR, "crism_quickview", f"{product_id}.png"),
        os.path.join(BASE_DIR, "crism_quickview", f"{product_id}.jpg"),
        os.path.join(BASE_DIR, "crism_quickview", f"{obs_id}_VNIR.png"),
        os.path.join(BASE_DIR, "crism_quickview", f"{base}.png"),
        os.path.join(BASE_DIR, "crism_quickview", f"{base}_brvnaj_mtr3.png"),
        os.path.join(BASE_DIR, "crism_data", base, f"{base}_brvnaj_mtr3.png"),
        os.path.join(BASE_DIR, "crism_data", base, "quickview.png"),  # cached TRR3 quickview
    ]

    path = None
    for c in candidates:
        if os.path.exists(c):
            path = c
            break

    if not path:
        # Last resort: generate from TRR3 cube data in crism_data/
        trr_dir = os.path.join(BASE_DIR, "crism_data", base)
        if os.path.isdir(trr_dir):
            trr_img_file = trr_lbl_file = None
            for fname in os.listdir(trr_dir):
                upper = fname.upper()
                if upper.endswith("_TRR3.IMG"):
                    trr_img_file = os.path.join(trr_dir, fname)
                elif upper.endswith("_TRR3.LBL"):
                    trr_lbl_file = os.path.join(trr_dir, fname)
            if trr_img_file and trr_lbl_file:
                return _generate_trr3_quickview(trr_img_file, trr_lbl_file, trr_dir)
        raise HTTPException(status_code=404, detail=f"CRISM quickview not found: {product_id}")

    try:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise HTTPException(status_code=500, detail="Failed to read image")

        # Handle RGB or grayscale; encoded as-is (OpenCV channel order)
        rgba = crism_image_to_rgba(img)

        png_bytes = encode_png(rgba)
        if png_bytes is None:
            raise HTTPException(status_code=500, detail="Failed to encode PNG")

        return Response(png_bytes, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ======================================================
# SHARAD quickview
# ======================================================
@router.get("/sharad/quickview/{product_id}.jpg")
@limiter.limit("20/minute")
def get_sharad_quickview(request: Request, product_id: str):
    """Serve SHARAD quickview (THM) image."""
    try:
        product_id = validate_product_id(product_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid product ID format")

    jpg_path = os.path.join(BASE_DIR, "sharad_quickview", f"{product_id}.jpg")
    if not os.path.exists(jpg_path):
        raise HTTPException(status_code=404, detail=f"SHARAD quickview not found: {product_id}")
    return FileResponse(jpg_path, media_type="image/jpeg")


# ======================================================
# HiRISE DTM overlay
# ======================================================
HIRISE_DTM_OVERLAY_CACHE_DIR = os.path.join(OVERLAY_CACHE_DIR, "hirise_dtm")
os.makedirs(HIRISE_DTM_OVERLAY_CACHE_DIR, exist_ok=True)


@router.get("/hirise_dtm/overlay/{product_id}.png")
@limiter.limit("20/minute")
def get_hirise_dtm_overlay(request: Request, product_id: str, max_size: int = 2048):
    """
    Serve HiRISE DTM orthoimage as overlay PNG with transparent background.
    Reads the JP2 orthoimage and generates a transparent PNG.
    """
    try:
        product_id = validate_product_id(product_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid product ID format")

    # Check cache first
    cache_file = os.path.join(HIRISE_DTM_OVERLAY_CACHE_DIR, f"{product_id}_{max_size}.png")
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return Response(f.read(), media_type="image/png", headers={"Cache-Control": "public, max-age=86400"})

    # Find matching orthoimage
    index_path = os.path.join(HIRISE_DTM_DIR, "index.geojson")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="HiRISE DTM index not found")

    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    ortho_file = None
    dtm_file = None
    for feature in index.get("features", []):
        props = feature.get("properties", {})
        if props.get("product_id") == product_id:
            ortho_file = props.get("ortho_file")
            dtm_file = props.get("dtm_file")
            break

    # Try orthoimage first, fall back to DTM IMG for hillshade
    source_path = None
    use_hillshade = False
    if ortho_file:
        path = os.path.join(HIRISE_DTM_DIR, ortho_file)
        if os.path.exists(path):
            source_path = path
    if not source_path and dtm_file:
        path = os.path.join(HIRISE_DTM_DIR, dtm_file)
        if os.path.exists(path):
            source_path = path
            use_hillshade = True

    if not source_path:
        raise HTTPException(status_code=404, detail=f"No orthoimage or DTM file found for: {product_id}")

    try:
        png_bytes = geotiff_to_png(source_path, max_size, use_hillshade)
        if png_bytes is None:
            raise HTTPException(status_code=500, detail="Failed to encode PNG")

        # Save to cache
        with open(cache_file, "wb") as f:
            f.write(png_bytes)

        return Response(png_bytes, media_type="image/png", headers={"Cache-Control": "public, max-age=86400"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
