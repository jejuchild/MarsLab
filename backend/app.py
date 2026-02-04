# app.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse  # ✅ (추가) CORS 확실히 타게 JSON으로 반환
import os
import json  # ✅ (추가)
from functools import lru_cache

import numpy as np
import cv2
import rasterio
from rasterio.windows import from_bounds
from rasterio.enums import Resampling

from pydantic import BaseModel
import io
from PIL import Image
from datetime import datetime

from api.crism.processing import make_rgb

# ======================================================
# App init
# ======================================================
app = FastAPI()

# ✅ CORS: 명시적으로 (iframe + fetch 안정화)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ======================================================
# Static mounts
# ======================================================

# (1) HiRISE viewer (iframe 용)
HIRISE_VIEWER_DIR = os.path.join(BASE_DIR, "hirise_viewer")
app.mount(
    "/hirise_viewer",
    StaticFiles(directory=HIRISE_VIEWER_DIR, html=True),
    name="hirise_viewer",
)

# ======================================================
# Data config (GeoTIFF)
# ======================================================
HIRISE_DATA_DIR = os.path.join(BASE_DIR, "hirise_data")
CRISM_DATA_DIR = os.path.join(BASE_DIR, "crism_data")
SHARAD_DATA_DIR = os.path.join(BASE_DIR, "sharad_data")
TILE_SIZE = 256
MAX_ZOOM = 8

# ======================================================
# Utils
# ======================================================
def tif_path(name: str) -> str:
    return os.path.join(HIRISE_DATA_DIR, f"{name}_RED.tif")

def blank_png() -> bytes:
    blank = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.uint8)
    ok, png = cv2.imencode(".png", blank)
    return png.tobytes() if ok else b""

@lru_cache(maxsize=32)
def open_ds(path: str):
    return rasterio.open(path)

def ds_world_bounds(ds):
    b = ds.bounds
    return (b.left, b.bottom, b.right, b.top)

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
@app.get("/meta/{name}")
def get_meta(name: str):
    path = tif_path(name)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    ds = open_ds(path)
    return {"width": ds.width, "height": ds.height}

@app.get("/world_meta")
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
    left, bottom, right, top = world_union_extent()

    scale = 2 ** (MAX_ZOOM - z)
    tile_px = TILE_SIZE * scale

    x0 = left + x * tile_px
    x1 = x0 + tile_px
    y1 = top - y * tile_px
    y0 = y1 - tile_px

    dl, db, dr, dt = ds_world_bounds(ds)
    if x1 <= dl or x0 >= dr or y1 <= db or y0 >= dt:
        return blank_png()

    try:
        win = from_bounds(x0, y0, x1, y1, transform=ds.transform)
        data = ds.read(
            1,
            window=win,
            out_shape=(TILE_SIZE, TILE_SIZE),
            resampling=Resampling.bilinear,
            boundless=True,
            fill_value=0,
        )
    except Exception:
        return blank_png()

    tile = np.clip(data / 4, 0, 255).astype(np.uint8)
    ok, png = cv2.imencode(".png", tile)
    return png.tobytes() if ok else blank_png()

@app.get("/world_tiles/{name}/{z}/{x}/{y}.png")
def get_world_tile(name: str, z: int, x: int, y: int):
    if z < 0 or z > MAX_ZOOM + 2:
        return Response(blank_png(), media_type="image/png")

    png = load_world_tile(name, z, x, y)
    return Response(png, media_type="image/png")


# Overlay cache directory
OVERLAY_CACHE_DIR = os.path.join(BASE_DIR, ".overlay_cache")
os.makedirs(OVERLAY_CACHE_DIR, exist_ok=True)

@app.get("/hirise/overlay/{product_id}.png")
def get_hirise_overlay(product_id: str, max_size: int = 2048):
    """
    Serve HiRISE image at reduced resolution with transparent background.
    Black pixels (DN=0) are made transparent.
    Uses disk cache for faster subsequent loads.
    """
    # Check cache first
    cache_file = os.path.join(OVERLAY_CACHE_DIR, f"{product_id}_{max_size}.png")
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return Response(f.read(), media_type="image/png")

    path = tif_path(product_id)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"HiRISE TIF not found: {product_id}")

    try:
        ds = open_ds(path)

        # Calculate downsampling factor
        scale = max(ds.width, ds.height) / max_size
        if scale < 1:
            scale = 1

        out_width = int(ds.width / scale)
        out_height = int(ds.height / scale)

        # Read with resampling
        data = ds.read(
            1,
            out_shape=(out_height, out_width),
            resampling=Resampling.bilinear
        )

        # Normalize to 0-255
        data_norm = np.clip(data / 4, 0, 255).astype(np.uint8)

        # Create RGBA image with transparency for black pixels
        rgba = np.zeros((out_height, out_width, 4), dtype=np.uint8)
        rgba[:, :, 0] = data_norm  # R
        rgba[:, :, 1] = data_norm  # G
        rgba[:, :, 2] = data_norm  # B
        # Alpha: 255 for non-zero pixels, 0 for black
        rgba[:, :, 3] = np.where(data_norm > 5, 255, 0)

        # Encode as PNG
        ok, png = cv2.imencode(".png", cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
        if not ok:
            raise HTTPException(status_code=500, detail="Failed to encode PNG")

        png_bytes = png.tobytes()

        # Save to cache
        with open(cache_file, "wb") as f:
            f.write(png_bytes)

        return Response(png_bytes, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/hirise_index.geojson")
def get_hirise_index():
    path = os.path.join(HIRISE_DATA_DIR, "index.geojson")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="HiRISE index.geojson not found")

    # ✅ (수정) FileResponse 대신 JSONResponse로 반환 (CORS 안정화)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return JSONResponse(content=data)

@app.get("/crism_index.geojson")
def get_crism_index():
    path = os.path.join(CRISM_DATA_DIR, "index.geojson")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="CRISM index.geojson not found")

    # ✅ (수정) FileResponse 대신 JSONResponse로 반환 (CORS 안정화)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return JSONResponse(content=data)

from api.crism.router import router as crism_router
from api.search_router import router as search_router
from api.footprints_router import router as footprints_router
from api.custom_router import router as custom_router

app.include_router(crism_router, prefix="/crism")
app.include_router(search_router)  # Mounts at /api/*
app.include_router(footprints_router)  # Viewport-based footprint API
app.include_router(custom_router)  # Custom user data upload

app.mount(
    "/hirise_lbl",
    StaticFiles(directory=HIRISE_DATA_DIR),
    name="hirise_lbl",
)

from api.hirise_pixel import router as hirise_pixel_router
from api.terrain_router import router as terrain_router
from api.sharad_highres_router import router as sharad_highres_router

app.include_router(
    hirise_pixel_router,
    prefix="/hirise",
    tags=["HiRISE"]
)
app.include_router(terrain_router)  # /terrain/slope_stats
app.include_router(sharad_highres_router)  # /api/sharad_highres/*

@app.get("/hirise/quickview/{product_id}.png")
def get_hirise_quickview_transparent(product_id: str):
    """
    Serve HiRISE quickview image with transparent background.
    Black pixels are made transparent.
    """
    # Try JPG first (most common)
    jpg_path = os.path.join(BASE_DIR, "hirise_quickview", f"{product_id}.jpg")
    png_path = os.path.join(BASE_DIR, "hirise_quickview", f"{product_id}.png")

    path = jpg_path if os.path.exists(jpg_path) else png_path
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Quickview not found: {product_id}")

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

        # Create RGBA with transparent black pixels
        h, w = gray.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, 0] = gray  # R
        rgba[:, :, 1] = gray  # G
        rgba[:, :, 2] = gray  # B
        # Alpha: 255 for non-black pixels, 0 for black (threshold at 5)
        rgba[:, :, 3] = np.where(gray > 5, 255, 0)

        # Encode as PNG
        ok, png = cv2.imencode(".png", cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
        if not ok:
            raise HTTPException(status_code=500, detail="Failed to encode PNG")

        return Response(png.tobytes(), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/crism/quickview/{product_id}.png")
def get_crism_quickview_transparent(product_id: str):
    """
    Serve CRISM quickview image with transparent background.
    Black pixels are made transparent.
    """
    png_path = os.path.join(BASE_DIR, "crism_quickview", f"{product_id}.png")
    jpg_path = os.path.join(BASE_DIR, "crism_quickview", f"{product_id}.jpg")

    path = png_path if os.path.exists(png_path) else jpg_path
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"CRISM quickview not found: {product_id}")

    try:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise HTTPException(status_code=500, detail="Failed to read image")

        # Handle RGB or grayscale
        if len(img.shape) == 3:
            if img.shape[2] == 4:
                # Already RGBA
                rgba = img
            else:
                # RGB - convert and add alpha
                h, w = img.shape[:2]
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                rgba = np.zeros((h, w, 4), dtype=np.uint8)
                rgba[:, :, :3] = img
                rgba[:, :, 3] = np.where(gray > 5, 255, 0)
        else:
            h, w = img.shape
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            rgba[:, :, 0] = img
            rgba[:, :, 1] = img
            rgba[:, :, 2] = img
            rgba[:, :, 3] = np.where(img > 5, 255, 0)

        ok, png = cv2.imencode(".png", rgba)
        if not ok:
            raise HTTPException(status_code=500, detail="Failed to encode PNG")

        return Response(png.tobytes(), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Static fallback for non-transparent files
app.mount(
    "/hirise/quickview",
    StaticFiles(directory="hirise_quickview"),
    name="hirise_quickview",
)

app.mount(
    "/crism/quickview",
    StaticFiles(directory="crism_quickview"),
    name="crism_quickview",
)

app.mount(
    "/crism/browse",
    StaticFiles(directory="crism_browse"),
    name="crism_browse",
)

app.mount(
    "/crism_lbl",
    StaticFiles(directory=os.path.join(BASE_DIR, "crism_data")),
    name="crism_lbl",
)

# ======================================================
# SHARAD endpoints and static mounts
# ======================================================
@app.get("/sharad_index.geojson")
def get_sharad_index():
    path = os.path.join(SHARAD_DATA_DIR, "index.geojson")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="SHARAD index.geojson not found")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return JSONResponse(content=data)

@app.get("/sharad/quickview/{product_id}.jpg")
def get_sharad_quickview(product_id: str):
    """Serve SHARAD quickview (THM) image."""
    jpg_path = os.path.join(BASE_DIR, "sharad_quickview", f"{product_id}.jpg")
    if not os.path.exists(jpg_path):
        raise HTTPException(status_code=404, detail=f"SHARAD quickview not found: {product_id}")
    return FileResponse(jpg_path, media_type="image/jpeg")

app.mount(
    "/sharad/quickview",
    StaticFiles(directory=os.path.join(BASE_DIR, "sharad_quickview")),
    name="sharad_quickview",
)

app.mount(
    "/sharad/highres",
    StaticFiles(directory=os.path.join(BASE_DIR, "sharad_highres")),
    name="sharad_highres",
)

# ======================================================
# Ice/Hydration Score Filtering API
# ======================================================
CRISM_SCORE_DIR = os.path.join(BASE_DIR, "crism_score")
SCORE_STATS_FILE = os.path.join(CRISM_SCORE_DIR, "score_stats.json")

# Preload score stats at startup for efficiency
_score_stats_cache = None

def _load_score_stats():
    """Load score stats from JSON file, with caching."""
    global _score_stats_cache
    if _score_stats_cache is None:
        if os.path.exists(SCORE_STATS_FILE):
            with open(SCORE_STATS_FILE, "r") as f:
                _score_stats_cache = json.load(f)
        else:
            _score_stats_cache = {}
    return _score_stats_cache

# Precomputed thresholds (must match generate_score_maps.py)
SCORE_THRESHOLDS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5]

def _find_closest_threshold(value: float) -> str:
    """Find the closest precomputed threshold (as string key)."""
    # Find the largest threshold <= value
    for t in reversed(SCORE_THRESHOLDS):
        if t <= value:
            return str(t)
    return str(SCORE_THRESHOLDS[0])  # Default to lowest

@app.get("/api/filter/ice")
def filter_ice_score(min_score: float = 0.3, min_percent: float = 5.0):
    """
    Filter CRISM observations by ice score.

    Returns observations where at least `min_percent`% of pixels have
    ice_score >= `min_score`.

    Args:
        min_score: Minimum ice score threshold (default 0.3)
        min_percent: Minimum percentage of pixels that must meet threshold (default 5%)

    Returns:
        JSON with list of passing observation IDs
    """
    stats = _load_score_stats()

    if not stats:
        return JSONResponse(content={"passing_ids": [], "total": 0, "passing_count": 0})

    # Find closest precomputed threshold
    threshold_key = _find_closest_threshold(min_score)

    passing_ids = []
    for obs_id, obs_stats in stats.items():
        ice_stats = obs_stats.get("ice", {})
        valid_pixels = ice_stats.get("valid_pixels", 0)

        if valid_pixels == 0:
            continue

        threshold_counts = ice_stats.get("threshold_counts", {})
        count_above = threshold_counts.get(threshold_key, 0)

        percent = (count_above / valid_pixels) * 100

        if percent >= min_percent:
            passing_ids.append(obs_id)

    return JSONResponse(content={
        "passing_ids": passing_ids,
        "total": len(stats),
        "passing_count": len(passing_ids),
        "params": {
            "min_score": min_score,
            "min_percent": min_percent,
            "used_threshold": float(threshold_key)
        }
    })

@app.get("/api/filter/hyd")
def filter_hyd_score(min_score: float = 0.3, min_percent: float = 5.0):
    """
    Filter CRISM observations by hydration score.

    Returns observations where at least `min_percent`% of pixels have
    hyd_score >= `min_score`.
    """
    stats = _load_score_stats()

    if not stats:
        return JSONResponse(content={"passing_ids": [], "total": 0, "passing_count": 0})

    threshold_key = _find_closest_threshold(min_score)

    passing_ids = []
    for obs_id, obs_stats in stats.items():
        hyd_stats = obs_stats.get("hyd", {})
        valid_pixels = hyd_stats.get("valid_pixels", 0)

        if valid_pixels == 0:
            continue

        threshold_counts = hyd_stats.get("threshold_counts", {})
        count_above = threshold_counts.get(threshold_key, 0)

        percent = (count_above / valid_pixels) * 100

        if percent >= min_percent:
            passing_ids.append(obs_id)

    return JSONResponse(content={
        "passing_ids": passing_ids,
        "total": len(stats),
        "passing_count": len(passing_ids),
        "params": {
            "min_score": min_score,
            "min_percent": min_percent,
            "used_threshold": float(threshold_key)
        }
    })

@app.get("/api/score/stats")
def get_score_stats():
    """
    Get score statistics for all observations.
    Useful for debugging and building filter UIs.
    """
    stats = _load_score_stats()
    return JSONResponse(content={
        "total_observations": len(stats),
        "available_thresholds": SCORE_THRESHOLDS
    })

# ======================================================
# Feature memo (developer notepad)
# ======================================================
MEMO_DIR = os.path.join(BASE_DIR, "private_memos")
os.makedirs(MEMO_DIR, exist_ok=True)

class FeatureMemo(BaseModel):
    text: str

@app.post("/api/feature_memo")
def save_feature_memo(memo: FeatureMemo):
    text = memo.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty memo")
    now = datetime.now()
    filename = now.strftime("feature_%Y%m%d_%H%M%S.txt")
    content = f"[{now.strftime('%Y-%m-%d %H:%M')}]\n{text}\n"
    with open(os.path.join(MEMO_DIR, filename), "w", encoding="utf-8") as f:
        f.write(content)
    return {"ok": True}

