# app.py
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
import os
import json
import gzip
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import threading

import aiohttp
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
from api.rate_limit import limiter
from api.validation import validate_product_id
# ======================================================
# Logging configuration
# ======================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("marslab")


# ======================================================
# Lifespan: shared HTTP session + parallel index preload
# ======================================================
async def _background_index_repair(app: FastAPI):
    """Run index repair as a background task after startup."""
    from api.index_repair import repair_all_indices
    try:
        repair_results = await repair_all_indices(app.state.http_session)
        total_added = sum(r.get("added", 0) for r in repair_results.values())
        if total_added > 0:
            logger.info(f"[REPAIR] Added {total_added} orphaned products to indices, refreshing caches...")
            _preload_indices_parallel()  # Refresh in-memory caches
        for name, stats in repair_results.items():
            if stats.get("orphaned", 0) > 0:
                logger.info(f"[REPAIR] {name}: scanned={stats['scanned']}, orphaned={stats['orphaned']}, added={stats['added']}")
    except Exception as e:
        logger.warning(f"[REPAIR] Index repair failed (non-fatal): {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    # Fix PROJ_DATA: rasterio bundles PROJ 9.7.1 (needs proj.db MINOR>=6)
    # while pyproj and conda ship older DBs. Use rasterio's proj.db which is compatible.
    try:
        import rasterio
        rasterio_proj = os.path.join(os.path.dirname(rasterio.__file__), "proj_data")
        if os.path.exists(os.path.join(rasterio_proj, "proj.db")):
            os.environ["PROJ_DATA"] = rasterio_proj
            os.environ.pop("PROJ_LIB", None)  # Remove conflicting legacy var
            logger.info(f"[PROJ] Set PROJ_DATA={rasterio_proj} (rasterio bundled, MINOR=6)")
        else:
            logger.warning("[PROJ] rasterio proj_data not found, PROJ may fail")
    except Exception as e:
        logger.warning(f"[PROJ] Failed to set PROJ_DATA: {e}")

    # Phase 2a: shared aiohttp session for ODE connection pooling
    app.state.http_session = aiohttp.ClientSession()

    # Phase 2c: preload GeoJSON indices in parallel using threads
    _preload_indices_parallel()

    # Phase 3: preload SWIM + Accessibility pipelines in background thread
    # (prevents cold-start latency on first user request)
    _preload_analysis_pipelines()

    # Auto-repair: run as background task (non-blocking startup)
    import asyncio
    repair_task = asyncio.create_task(_background_index_repair(app))

    yield

    # --- Shutdown ---
    repair_task.cancel()
    await app.state.http_session.close()


# ======================================================
# App init
# ======================================================
app = FastAPI(lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# GZip compression - compresses JSON responses (2.7MB CRISM → ~270KB)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# CORS: restrict to known origins (use CORS_ORIGINS env var for production)
ALLOWED_ORIGINS = os.environ.get("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Content-Type", "Authorization"],
)


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ======================================================
# In-memory GeoJSON index cache (loaded once at startup)
# ======================================================
_geojson_cache: dict[str, dict] = {}  # key → parsed JSON
_geojson_bytes_cache: dict[str, bytes] = {}  # key → serialized JSON bytes
_geojson_gz_cache: dict[str, bytes] = {}  # key → pre-compressed gzip bytes

def _preload_geojson(key: str, path: str):
    """Load a GeoJSON file into memory cache (incl. pre-compressed gzip)."""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        _geojson_cache[key] = data
        raw = json.dumps(data, separators=(",", ":")).encode("utf-8")
        _geojson_bytes_cache[key] = raw
        # Phase 2c: pre-compress gzip bytes at startup (avoid per-request gzip)
        _geojson_gz_cache[key] = gzip.compress(raw, compresslevel=6)
        logger.info(f"[CACHE] Preloaded {key}: {len(raw):,} bytes, gzip {len(_geojson_gz_cache[key]):,} bytes")
    else:
        logger.info(f"[CACHE] Skipped {key}: {path} not found")

# ======================================================
# Phase 2c: Parallel startup - load all GeoJSON files concurrently
# ======================================================
_INDEX_FILES = {
    "hirise": os.path.join(BASE_DIR, "hirise_data", "index.geojson"),
    "crism": os.path.join(BASE_DIR, "crism_data", "index.geojson"),
    "sharad": os.path.join(BASE_DIR, "sharad_data", "index.geojson"),
    "hirise_dtm": os.path.join(BASE_DIR, "hirise_dtm_data", "index.geojson"),
    "sharad_highres": os.path.join(BASE_DIR, "sharad_highres_data", "index.geojson"),
    "ctx": os.path.join(BASE_DIR, "ctx_data", "index.geojson"),
    "crism_trr3": os.path.join(BASE_DIR, "mineral_cnn_data", "index.geojson"),
}


def _preload_indices_parallel():
    """Load all GeoJSON index files in parallel using a thread pool."""
    with ThreadPoolExecutor(max_workers=len(_INDEX_FILES)) as executor:
        futures = {
            executor.submit(_preload_geojson, key, path): key
            for key, path in _INDEX_FILES.items()
        }
        for future in futures:
            future.result()  # propagate exceptions
    logger.info(f"[CACHE] All indices preloaded in parallel ({len(_geojson_cache)} total)")


def _preload_analysis_pipelines():
    """Warm up SWIM and Accessibility pipelines in background threads at startup.
    
    Prevents cold-start latency on first user request by pre-loading pipeline models.
    Both tasks run in parallel as daemon threads (non-blocking).
    """
    
    def _warm_swim_pipelines():
        """Load all SWIM pipeline models in parallel."""
        try:
            from api.swim_ice_router import (
                _get_neutron,
                _get_thermal,
                _get_surface,
                _get_dielectric,
                _get_geomorphic,
                _get_fusion,
            )
            
            logger.info("[PRELOAD] Starting SWIM pipeline warmup...")
            
            # Load each pipeline and call _ensure_loaded() if available
            pipelines = [
                ("neutron", _get_neutron),
                ("thermal", _get_thermal),
                ("surface", _get_surface),
                ("dielectric", _get_dielectric),
                ("geomorphic", _get_geomorphic),
                ("fusion", _get_fusion),
            ]
            
            for name, getter in pipelines:
                try:
                    pipeline = getter()
                    ensure_loaded = getattr(pipeline, "_ensure_loaded", None)
                    if callable(ensure_loaded):
                        ensure_loaded()
                    logger.info(f"[PRELOAD] SWIM {name} pipeline loaded")
                except Exception as e:
                    logger.warning(f"[PRELOAD] SWIM {name} pipeline warmup failed (non-fatal): {e}")
            
            logger.info("[PRELOAD] SWIM pipeline warmup complete")
        except Exception as e:
            logger.warning(f"[PRELOAD] SWIM pipeline warmup failed (non-fatal): {e}")
    
    def _warm_accessibility_pipeline():
        """Load Accessibility pipeline model."""
        try:
            from api.accessibility_router import _get_pipeline
            
            logger.info("[PRELOAD] Starting Accessibility pipeline warmup...")
            pipeline = _get_pipeline()
            ensure_loaded = getattr(pipeline, "_ensure_loaded", None)
            if callable(ensure_loaded):
                ensure_loaded()
            logger.info("[PRELOAD] Accessibility pipeline loaded")
        except Exception as e:
            logger.warning(f"[PRELOAD] Accessibility pipeline warmup failed (non-fatal): {e}")
    
    # Start both warmup tasks in parallel as daemon threads
    swim_thread = threading.Thread(target=_warm_swim_pipelines, daemon=True)
    accessibility_thread = threading.Thread(target=_warm_accessibility_pipeline, daemon=True)
    
    swim_thread.start()
    accessibility_thread.start()
    
    logger.info("[PRELOAD] Analysis pipeline warmup threads started (non-blocking)")

def refresh_geojson_cache(instrument_key: str):
    """Refresh a single instrument's GeoJSON cache from disk after download.

    Called by download_manager after _update_index() writes new data to disk.
    Also clears the footprints_router lru_cache so subsequent requests see fresh data.
    """
    path = _INDEX_FILES.get(instrument_key)
    if not path:
        logger.warning(f"[CACHE] Unknown instrument key for refresh: {instrument_key}")
        return
    _preload_geojson(instrument_key, path)
    # Clear the footprints_router lru_cache so it picks up fresh data
    try:
        from api.footprints_router import load_geojson_index
        load_geojson_index.cache_clear()
    except ImportError:
        pass
    logger.info(f"[CACHE] Refreshed {instrument_key} cache after download")


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
HIRISE_DTM_DIR = os.path.join(BASE_DIR, "hirise_dtm_data")
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
    return Response(png, media_type="image/png", headers={"Cache-Control": "public, max-age=86400"})


# Overlay cache directory with LRU cleanup
OVERLAY_CACHE_DIR = os.path.join(BASE_DIR, ".overlay_cache")
os.makedirs(OVERLAY_CACHE_DIR, exist_ok=True)

OVERLAY_CACHE_MAX_MB = 500  # Max cache size in MB
OVERLAY_CACHE_TTL_DAYS = 30  # Max age for cache entries


def _cleanup_overlay_cache():
    """Evict oldest files when cache exceeds size limit or TTL."""
    import time
    now = time.time()
    ttl_seconds = OVERLAY_CACHE_TTL_DAYS * 86400
    files = []
    total_size = 0
    for fname in os.listdir(OVERLAY_CACHE_DIR):
        fpath = os.path.join(OVERLAY_CACHE_DIR, fname)
        if not os.path.isfile(fpath):
            continue
        stat = os.stat(fpath)
        # Remove files older than TTL
        if now - stat.st_mtime > ttl_seconds:
            os.remove(fpath)
            continue
        files.append((fpath, stat.st_mtime, stat.st_size))
        total_size += stat.st_size

    # LRU eviction if over size limit
    max_bytes = OVERLAY_CACHE_MAX_MB * 1024 * 1024
    if total_size > max_bytes:
        files.sort(key=lambda x: x[1])  # oldest first
        while total_size > max_bytes * 0.8 and files:  # evict to 80%
            fpath, _, fsize = files.pop(0)
            os.remove(fpath)
            total_size -= fsize


# Run cleanup on startup
_cleanup_overlay_cache()

@app.get("/hirise/overlay/{product_id}.png")
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

        return Response(png_bytes, media_type="image/png", headers={"Cache-Control": "public, max-age=86400"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/hirise_index.geojson")
def get_hirise_index():
    if "hirise" in _geojson_bytes_cache:
        return Response(
            content=_geojson_bytes_cache["hirise"],
            media_type="application/json",
            headers={"Cache-Control": "public, max-age=86400"},
        )
    raise HTTPException(status_code=404, detail="HiRISE index.geojson not found")

@app.get("/crism_index.geojson")
def get_crism_index():
    if "crism" in _geojson_bytes_cache:
        return Response(
            content=_geojson_bytes_cache["crism"],
            media_type="application/json",
            headers={"Cache-Control": "public, max-age=86400"},
        )
    raise HTTPException(status_code=404, detail="CRISM index.geojson not found")

from api.crism.router import router as crism_router
from api.search_router import router as search_router
from api.footprints_router import router as footprints_router
from api.custom_router import router as custom_router
from api.point_search import router as point_search_router
from api.ai_search import router as ai_search_router
from api.proximity_router import router as proximity_router

app.include_router(crism_router, prefix="/crism")
app.include_router(search_router)  # Mounts at /api/*
app.include_router(footprints_router)  # Viewport-based footprint API
app.include_router(custom_router)  # Custom user data upload
app.include_router(point_search_router)  # Point-based coordinate search
app.include_router(ai_search_router)  # AI-assisted natural language search
app.include_router(proximity_router)  # Product proximity + region lookup

app.mount(
    "/hirise_lbl",
    StaticFiles(directory=HIRISE_DATA_DIR),
    name="hirise_lbl",
)

from api.hirise_pixel import router as hirise_pixel_router
from api.terrain_router import router as terrain_router
from api.sharad_highres_router import router as sharad_highres_router
from api.suggestions_router import router as suggestions_router
from api.fieldnotes_router import router as fieldnotes_router
from api.ai_analysis_router import router as ai_analysis_router
from api.agentic_router import router as agentic_router
from api.llama_search import router as smart_search_router
from api.report_router import router as report_router
from api.mineral_cnn.router import router as mineral_cnn_router
from api.crism_trr3.router import router as crism_trr3_router
from api.sharad_report_router import router as sharad_report_router
from api.multi_report_router import router as multi_report_router
from api.region_scores import router as region_scores_router

app.include_router(
    hirise_pixel_router,
    prefix="/hirise",
    tags=["HiRISE"]
)
app.include_router(terrain_router)  # /api/terrain/slope_stats
app.include_router(sharad_highres_router)  # /api/sharad_highres/*
app.include_router(suggestions_router)  # /api/feature_suggestions
app.include_router(fieldnotes_router)  # /api/fieldnotes
app.include_router(ai_analysis_router)  # /api/ai_analysis
app.include_router(agentic_router)  # /api/agent/* — Agentic AI
app.include_router(smart_search_router)  # /api/ai/smart/* — Llama smart search
app.include_router(report_router)  # /api/report/* — Landing Site Report Generator
app.include_router(mineral_cnn_router)  # /api/mineral-cnn/* — CNN Mineral Classification
app.include_router(crism_trr3_router)  # /api/crism-trr3/* — TRR3 Spectrum + RGB
app.include_router(sharad_report_router)  # /api/sharad-report/* — Subsurface Interface Report
app.include_router(multi_report_router)  # /api/multi-report/* — Multi-Instrument Report
app.include_router(region_scores_router)  # /api/regions/scores — Real region scoring

from api.temporal_router import router as temporal_router
app.include_router(temporal_router)  # /api/temporal/* — Temporal Change Detection

from analysis.ice_evidence.router import router as ice_evidence_router
app.include_router(ice_evidence_router)  # /api/ice/* — Ice Evidence Synthesizer

from api.mola_detect_router import router as mola_detect_router
app.include_router(mola_detect_router)  # /api/mola-detect/* — MOLA Landform Detection

from api.terrain_features import router as terrain_features_router
app.include_router(terrain_features_router)  # /api/terrain/features_in_view

from agent.workflow_router import router as workflow_router
app.include_router(workflow_router)  # /api/workflow/* — Workflow Research Assistant

from api.marvis_chat import router as marvis_chat_router
app.include_router(marvis_chat_router)  # /api/marvis/chat — MARVIS lightweight chat

from api.regolith_router import router as regolith_router
app.include_router(regolith_router)  # /api/regolith/* — Regolith Thickness Estimator

from api.epsilon_router import router as epsilon_router
app.include_router(epsilon_router)  # /api/epsilon/* — εr Inversion (near-crater, hyperbola)

from api.stratigraphy_router import router as stratigraphy_router
app.include_router(stratigraphy_router)  # /api/stratigraphy/* — Crater Stratigraphy

from api.attenuation_router import router as attenuation_router
app.include_router(attenuation_router)  # /api/attenuation/* — Radar Attenuation

from api.mineral_sequence_router import router as mineral_sequence_router
app.include_router(mineral_sequence_router)  # /api/mineral-sequence/* — Mineral Sequence

from api.strat_column_router import router as strat_column_router
app.include_router(strat_column_router)  # /api/strat-column/* — Stratigraphic Column

from api.product_urls import router as product_urls_router
app.include_router(product_urls_router)  # /api/product-urls/* — PDS Download URL Resolver

from api.discussions_router import router as discussions_router
app.include_router(discussions_router)  # /api/discussions — Daily AI Discussions

from api.swim_router import router as swim_router
app.include_router(swim_router)  # /api/swim/* — SWIM Ice Data

from api.swim_ice_router import router as swim_ice_router
app.include_router(swim_ice_router)  # /api/swim-ice/* — SWIM Ice Detection Methods

from api.hirise_landforms_router import router as hirise_landforms_router
app.include_router(hirise_landforms_router)  # /api/hirise-landforms/* — HiRISE Landform Classification

from api.mars_news_router import router as mars_news_router
app.include_router(mars_news_router)  # /api/mars-news — Mars News Digest

from api.mars_research_router import router as mars_research_router
app.include_router(mars_research_router)  # /api/mars-research — Mars Research Digest

from api.accessibility_router import router as accessibility_router
app.include_router(accessibility_router)  # /api/accessibility/* — Ice Accessibility Algorithm

from api.pathfinder_router import router as pathfinder_router
app.include_router(pathfinder_router)  # /api/pathfinder/* — AI Rover Route Planning

from rag.rag_router import router as rag_router
app.include_router(rag_router)  # /api/rag/* — Mars Science RAG

from neural_climate.climate_router import router as neural_climate_router
app.include_router(neural_climate_router)  # /api/climate/neural/* — Mars GCM Neural Emulator
from pinns_interior.pinns_router import router as pinns_router
app.include_router(pinns_router)

from analysis.integration.integration_router import router as integration_router
app.include_router(integration_router)  # /api/integration/* — Cross-system Integration Modules

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


def _reproject_polar_browse(gray: np.ndarray, feature: dict) -> np.ndarray:
    """Reproject a polar stereographic browse image to equirectangular (lat/lon).
    Returns the reprojected image (grayscale) that maps to the polygon's bounding box."""
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.warp import reproject, Resampling
    from pyproj import Transformer

    props = feature["properties"]
    coords = feature["geometry"]["coordinates"][0]
    proj_center_lat = props.get("proj_center_lat", 90.0)

    # Polygon bounding box in geographic coords
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    west, east = min(lons), max(lons)
    south, north = min(lats), max(lats)

    # Mars polar stereographic CRS
    sign = 1 if proj_center_lat > 0 else -1
    mars_a, mars_b = 3396190.0, 3376200.0
    src_crs = (
        f"+proj=stere +lat_0={sign * 90} +lon_0=0 "
        f"+k=1 +x_0=0 +y_0=0 +a={mars_a} +b={mars_b} +units=m +no_defs"
    )
    dst_crs = f"+proj=longlat +a={mars_a} +b={mars_b} +no_defs"

    # Convert polygon bbox to polar stereographic to get image extent
    geo_to_stereo = Transformer.from_crs(dst_crs, src_crs, always_xy=True)
    x_min, y_min = geo_to_stereo.transform(west, south)
    x_max, y_max = geo_to_stereo.transform(east, north)
    # Also transform all corners to get proper extent
    xs, ys = [], []
    for lon, lat in coords:
        x, y = geo_to_stereo.transform(lon, lat)
        xs.append(x)
        ys.append(y)
    src_left, src_right = min(xs), max(xs)
    src_bottom, src_top = min(ys), max(ys)

    h, w = gray.shape
    src_transform = from_bounds(src_left, src_bottom, src_right, src_top, w, h)

    # Target: equirectangular image covering the same geographic bbox
    # Use enough pixels to maintain resolution
    dst_h = max(h, 512)
    dst_w = max(w, 512)
    dst_transform = from_bounds(west, south, east, north, dst_w, dst_h)

    dst = np.zeros((dst_h, dst_w), dtype=gray.dtype)
    reproject(
        source=gray,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.bilinear,
    )
    return dst


@app.get("/hirise/quickview/{product_id}.png")
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
            import rasterio
            with rasterio.open(jp2_path) as ds:
                full_w, full_h = ds.width, ds.height
                # Read a heavily downsampled version (target ~800px wide)
                scale = max(1, full_w // 800)
                out_w, out_h = full_w // scale, full_h // scale
                arr = ds.read(1, out_shape=(out_h, out_w))
            # Normalize to 0-255
            nonzero = arr[arr > 0]
            if nonzero.size > 0:
                vmin, vmax = float(np.percentile(nonzero, 1)), float(np.percentile(nonzero, 99))
                if vmax > vmin:
                    arr = np.clip((arr.astype(float) - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)
                else:
                    arr = arr.astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)
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
        # Browse images (abrowse.jpg) are non-uniformly downscaled from the
        # full RDR product and may include annotation padding, causing ~8-9%
        # aspect ratio mismatch that leads to misaligned quickview overlays.
        h, w = gray.shape
        rdr_props = _get_hirise_rdr_props(product_id)
        if rdr_props:
            rdr_lines, rdr_samples = rdr_props
            target_ratio = rdr_samples / rdr_lines  # width/height
            current_ratio = w / h
            if abs(current_ratio - target_ratio) / target_ratio > 0.01:
                if current_ratio < target_ratio:
                    # Image is too tall — crop top/bottom
                    new_h = int(w / target_ratio)
                    crop = (h - new_h) // 2
                    gray = gray[crop:crop + new_h, :]
                else:
                    # Image is too wide — crop left/right
                    new_w = int(h * target_ratio)
                    crop = (w - new_w) // 2
                    gray = gray[:, crop:crop + new_w]
                h, w = gray.shape

        # Reproject polar stereographic browse images to equirectangular
        # so the overlay aligns correctly on the geographic map
        feat = _get_hirise_feature(product_id)
        if feat:
            proj_lat = feat.get("properties", {}).get("proj_center_lat")
            if proj_lat is not None and abs(proj_lat) >= 85:
                try:
                    gray = _reproject_polar_browse(gray, feat)
                    h, w = gray.shape
                except Exception as e:
                    logger.warning(f"[QuickView] Polar reprojection failed for {product_id}: {e}")

        # Downscale large images to ~1024px wide for faster overlay rendering
        if w > 1024:
            scale = 1024 / w
            new_w, new_h = 1024, int(h * scale)
            gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
            h, w = new_h, new_w

        # Create RGBA with transparent black pixels
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, 0] = gray  # R
        rgba[:, :, 1] = gray  # G
        rgba[:, :, 2] = gray  # B
        # Alpha: 255 for non-black pixels, 0 for black (threshold at 5)
        rgba[:, :, 3] = np.where(gray > 5, 255, 0)

        # Encode as PNG with compression
        ok, png_bytes = cv2.imencode(".png", cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA),
                                     [cv2.IMWRITE_PNG_COMPRESSION, 6])
        if not ok:
            raise HTTPException(status_code=500, detail="Failed to encode PNG")

        # Cache to disk for future requests
        os.makedirs(cache_dir, exist_ok=True)
        with open(cached_png, "wb") as f:
            f.write(png_bytes.tobytes())

        return Response(png_bytes.tobytes(), media_type="image/png")
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
    rows = int(meta["LINES"])
    cols = int(meta["LINE_SAMPLES"])
    bands = int(meta["BANDS"])

    # Use memmap to avoid loading full cube into RAM
    mm = np.memmap(trr_img_path, dtype=np.float32, mode="r",
                   shape=(rows, bands, cols))

    # Pick a VNIR band (~60% through bands for good contrast)
    band_idx = int(bands * 0.6)
    single = np.array(mm[:, band_idx, :], dtype=np.float64)

    valid = (single > 0) & (single < 1.5) & np.isfinite(single)
    if valid.sum() < 10:
        raise HTTPException(status_code=500, detail="No valid data in selected band")

    p2 = float(np.percentile(single[valid], 2))
    p98 = float(np.percentile(single[valid], 98))
    if p98 <= p2:
        p98 = p2 + 0.01

    stretched = np.clip((single - p2) / (p98 - p2) * 255, 0, 255).astype(np.uint8)
    stretched[~valid] = 0

    # RGBA with transparency for invalid pixels
    h, w = stretched.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, 0] = stretched
    rgba[:, :, 1] = stretched
    rgba[:, :, 2] = stretched
    rgba[:, :, 3] = np.where(valid, 255, 0).astype(np.uint8)

    ok, png = cv2.imencode(".png", cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode PNG")

    png_bytes = png.tobytes()

    # Cache the quickview
    cache_path = os.path.join(cache_dir, "quickview.png")
    try:
        with open(cache_path, "wb") as f:
            f.write(png_bytes)
    except OSError:
        pass

    return Response(png_bytes, media_type="image/png",
                    headers={"Cache-Control": "public, max-age=86400"})


@app.get("/crism/quickview/{product_id}.png")
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
    StaticFiles(directory="hirise_quickview", follow_symlink=True),
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

# HiRISE landform heatmap cache
app.mount(
    "/cache/hirise_landforms/heatmaps",
    StaticFiles(directory=os.path.join(BASE_DIR, "data", "hirise_landforms", "cache", "heatmaps")),
    name="hirise_landform_heatmaps",
)

# ======================================================
# SHARAD endpoints and static mounts
# ======================================================
@app.get("/sharad_index.geojson")
def get_sharad_index():
    if "sharad" in _geojson_bytes_cache:
        return Response(
            content=_geojson_bytes_cache["sharad"],
            media_type="application/json",
            headers={"Cache-Control": "public, max-age=86400"},
        )
    raise HTTPException(status_code=404, detail="SHARAD index.geojson not found")

@app.get("/sharad/quickview/{product_id}.jpg")
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
# HiRISE DTM endpoints
# ======================================================
@app.get("/hirise_dtm_index.geojson")
def get_hirise_dtm_index():
    """Serve HiRISE DTM index.geojson."""
    if "hirise_dtm" in _geojson_bytes_cache:
        return Response(
            content=_geojson_bytes_cache["hirise_dtm"],
            media_type="application/json",
            headers={"Cache-Control": "public, max-age=86400"},
        )
    raise HTTPException(status_code=404, detail="HiRISE DTM index.geojson not found")


@app.get("/crism_trr3_index.geojson")
def get_crism_trr3_index():
    """Serve CRISM TRR3 index.geojson."""
    if "crism_trr3" in _geojson_bytes_cache:
        return Response(
            content=_geojson_bytes_cache["crism_trr3"],
            media_type="application/json",
            headers={"Cache-Control": "public, max-age=86400"},
        )
    raise HTTPException(status_code=404, detail="CRISM TRR3 index.geojson not found")


# HiRISE DTM overlay cache
HIRISE_DTM_OVERLAY_CACHE_DIR = os.path.join(OVERLAY_CACHE_DIR, "hirise_dtm")
os.makedirs(HIRISE_DTM_OVERLAY_CACHE_DIR, exist_ok=True)


@app.get("/hirise_dtm/overlay/{product_id}.png")
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

    from pyproj import CRS, Transformer

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
        ds = rasterio.open(source_path)

        # Calculate downsampling factor
        scale = max(ds.width, ds.height) / max_size
        if scale < 1:
            scale = 1

        out_width = int(ds.width / scale)
        out_height = int(ds.height / scale)

        # Capture nodata before closing
        nodata_val = ds.nodata

        # Read with resampling
        data = ds.read(
            1,
            out_shape=(out_height, out_width),
            resampling=Resampling.bilinear
        )
        ds.close()

        if use_hillshade:
            # Generate hillshade from elevation data
            nodata_mask = (data == 0) | np.isnan(data.astype(float))
            if nodata_val is not None:
                nodata_mask = nodata_mask | (data == nodata_val)
            else:
                nodata_mask = nodata_mask | (data == -32768)
            # Compute gradient for hillshade
            dy, dx = np.gradient(data.astype(float))
            azimuth_rad = np.radians(315)
            altitude_rad = np.radians(45)
            slope = np.sqrt(dx**2 + dy**2)
            aspect = np.arctan2(-dy, dx)
            shaded = (np.sin(altitude_rad) * np.cos(np.arctan(slope)) +
                      np.cos(altitude_rad) * np.sin(np.arctan(slope)) *
                      np.cos(azimuth_rad - aspect))
            shaded = np.clip(shaded * 255, 0, 255).astype(np.uint8)
            # Create RGBA with transparency for nodata
            rgba = np.zeros((out_height, out_width, 4), dtype=np.uint8)
            rgba[:, :, 0] = shaded
            rgba[:, :, 1] = shaded
            rgba[:, :, 2] = shaded
            rgba[:, :, 3] = np.where(nodata_mask, 0, 255)
        else:
            # Normalize orthoimage to 0-255
            p2, p98 = np.percentile(data[data > 0], [2, 98]) if np.any(data > 0) else (0, 255)
            if p98 > p2:
                data_norm = np.clip((data - p2) / (p98 - p2) * 255, 0, 255).astype(np.uint8)
            else:
                data_norm = np.clip(data, 0, 255).astype(np.uint8)

            # Create RGBA image with transparency only for nodata (original==0)
            rgba = np.zeros((out_height, out_width, 4), dtype=np.uint8)
            rgba[:, :, 0] = data_norm  # R
            rgba[:, :, 1] = data_norm  # G
            rgba[:, :, 2] = data_norm  # B
            rgba[:, :, 3] = np.where(data > 0, 255, 0)

        # Encode as PNG
        ok, png = cv2.imencode(".png", cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
        if not ok:
            raise HTTPException(status_code=500, detail="Failed to encode PNG")

        png_bytes = png.tobytes()

        # Save to cache
        with open(cache_file, "wb") as f:
            f.write(png_bytes)

        return Response(png_bytes, media_type="image/png", headers={"Cache-Control": "public, max-age=86400"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/hirise_dtm/elevation/{product_id}")
@limiter.limit("20/minute")
def get_hirise_dtm_elevation(
    request: Request,
    product_id: str,
    lat: float,
    lon: float,
    radius: float = 0.01
):
    """
    Get elevation data from HiRISE DTM at a specific location.
    Returns elevation value and statistics within radius.
    """
    try:
        product_id = validate_product_id(product_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid product ID format")

    from pyproj import CRS, Transformer

    # Find DTM file
    index_path = os.path.join(HIRISE_DTM_DIR, "index.geojson")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="HiRISE DTM index not found")

    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    dtm_file = None
    for feature in index.get("features", []):
        props = feature.get("properties", {})
        if props.get("product_id") == product_id:
            dtm_file = props.get("dtm_file")
            break

    if not dtm_file:
        raise HTTPException(status_code=404, detail=f"DTM not found for: {product_id}")

    dtm_path = os.path.join(HIRISE_DTM_DIR, dtm_file)
    if not os.path.exists(dtm_path):
        raise HTTPException(status_code=404, detail=f"DTM file not found: {dtm_file}")

    try:
        ds = rasterio.open(dtm_path)

        # Transform lat/lon to DTM CRS
        mars_lonlat = CRS.from_proj4("+proj=longlat +a=3396190 +b=3376200 +no_defs")
        transformer = Transformer.from_crs(mars_lonlat, ds.crs.to_wkt(), always_xy=True)

        x, y = transformer.transform(lon, lat)

        # Get pixel coordinates
        row, col = ds.index(x, y)

        # Check bounds
        if row < 0 or row >= ds.height or col < 0 or col >= ds.width:
            ds.close()
            raise HTTPException(status_code=400, detail="Location outside DTM bounds")

        # Read single value
        elevation = ds.read(1, window=rasterio.windows.Window(col, row, 1, 1))[0, 0]

        # Read a small patch for stats
        patch_size = 50
        row_start = max(0, row - patch_size // 2)
        col_start = max(0, col - patch_size // 2)
        row_end = min(ds.height, row + patch_size // 2)
        col_end = min(ds.width, col + patch_size // 2)

        patch = ds.read(1, window=rasterio.windows.Window(
            col_start, row_start,
            col_end - col_start, row_end - row_start
        ))

        # Filter out nodata
        nodata = ds.nodata
        if nodata is not None:
            valid = patch[patch != nodata]
        else:
            valid = patch[np.isfinite(patch)]

        ds.close()

        return JSONResponse(content={
            "product_id": product_id,
            "lat": lat,
            "lon": lon,
            "elevation_m": float(elevation) if np.isfinite(elevation) else None,
            "patch_stats": {
                "min_m": float(np.min(valid)) if len(valid) > 0 else None,
                "max_m": float(np.max(valid)) if len(valid) > 0 else None,
                "mean_m": float(np.mean(valid)) if len(valid) > 0 else None,
                "std_m": float(np.std(valid)) if len(valid) > 0 else None,
            }
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
# Index Repair API (manual trigger)
# ======================================================
@app.post("/api/repair-index")
async def repair_index_endpoint():
    """
    Manually trigger index repair: scan all data directories for orphaned
    downloads and add them to their respective index.geojson files.
    Refreshes in-memory caches if any orphans are found.
    """
    from api.index_repair import repair_all_indices

    session = getattr(app.state, "http_session", None)
    results = await repair_all_indices(session)

    total_added = sum(r.get("added", 0) for r in results.values())
    if total_added > 0:
        _preload_indices_parallel()  # Refresh in-memory caches

    return JSONResponse(content={
        "total_added": total_added,
        "instruments": results,
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

# ======================================================
# Health check endpoint
# ======================================================
@app.get("/health")
async def health_check():
    return JSONResponse(content={
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "indices_loaded": len(_geojson_cache),
    })
