# app.py — application bootstrap only: logging, middleware, startup/shutdown,
# static mounts and router registration. Endpoint logic lives in api/*.
#
# Registration order matters (FastAPI/Starlette match routes in order):
# explicit routes must stay ahead of the StaticFiles mounts that share their
# path prefixes (e.g. /hirise/quickview/{id}.png before the /hirise/quickview mount).
import logging
from contextlib import asynccontextmanager
from datetime import datetime

import aiohttp
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from starlette.middleware.gzip import GZipMiddleware
import os

from api.crism.processing import make_rgb  # noqa: F401  (kept: import-time side effects/order)
from api.rate_limit import limiter
# ======================================================
# Logging configuration
# ======================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("marslab")


# Shared state (defined once in api.geojson_cache / api.paths). Re-exported here
# because other modules lazily do `from app import _geojson_cache` and
# `from app import refresh_geojson_cache`.
from api.paths import BASE_DIR, HIRISE_DATA_DIR, HIRISE_VIEWER_DIR  # noqa: E402
from api.geojson_cache import (  # noqa: E402,F401
    _INDEX_FILES,
    _geojson_bytes_cache,
    _geojson_cache,
    _geojson_gz_cache,
    _preload_indices_parallel,
    refresh_geojson_cache,
)

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

    # Pick up index files rewritten by the nightly background fetcher.
    async def _watch_indices():
        from api.geojson_cache import refresh_changed_indices
        while True:
            await asyncio.sleep(120)
            changed = await asyncio.to_thread(refresh_changed_indices)
            if changed:
                logger.info(f"[CACHE] Reloaded changed indices: {changed}")
    watch_task = asyncio.create_task(_watch_indices())

    yield

    # --- Shutdown ---
    watch_task.cancel()
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


def _preload_analysis_pipelines():
    """Warm up SWIM and Accessibility pipelines in background threads at startup.
    
    Prevents cold-start latency on first user request by pre-loading pipeline models.
    Phase 1: SWIM and Accessibility pipelines were cut, so this is now a no-op.
    """
    logger.info("[PRELOAD] No analysis pipelines to warm up (Phase 1 cut SWIM/Accessibility)")


# ======================================================
# Static mounts
# ======================================================

# (1) HiRISE viewer (iframe 용)
app.mount(
    "/hirise_viewer",
    StaticFiles(directory=HIRISE_VIEWER_DIR, html=True),
    name="hirise_viewer",
)

# ======================================================
# Image endpoints: /meta, /world_meta, /world_tiles, HiRISE/CRISM/SHARAD
# quickviews, HiRISE + HiRISE-DTM overlays (all before the quickview mounts)
# ======================================================
from api.overlay_router import router as overlay_router  # noqa: E402
app.include_router(overlay_router)

# GeoJSON indices (/hirise_index.geojson, ...) + POST /api/repair-index
from api.index_router import router as index_router  # noqa: E402
app.include_router(index_router)

from api.crism.router import router as crism_router
from api.search_router import router as search_router
from api.footprints_router import router as footprints_router
from api.custom_router import router as custom_router
from api.point_search import router as point_search_router
from api.proximity_router import router as proximity_router

app.include_router(crism_router, prefix="/crism")
app.include_router(search_router)  # Mounts at /api/*
app.include_router(footprints_router)  # Viewport-based footprint API
app.include_router(custom_router)  # Custom user data upload
app.include_router(point_search_router)  # Point-based coordinate search
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
from api.mineral_cnn.router import router as mineral_cnn_router
from api.crism_trr3.router import router as crism_trr3_router

app.include_router(
    hirise_pixel_router,
    prefix="/hirise",
    tags=["HiRISE"]
)
app.include_router(terrain_router)  # /api/terrain/slope_stats
app.include_router(sharad_highres_router)  # /api/sharad_highres/*
app.include_router(suggestions_router)  # /api/feature_suggestions
app.include_router(fieldnotes_router)  # /api/fieldnotes
app.include_router(mineral_cnn_router)  # /api/mineral-cnn/* — CNN Mineral Classification
app.include_router(crism_trr3_router)  # /api/crism-trr3/* — TRR3 Spectrum + RGB

from api.temporal_router import router as temporal_router
app.include_router(temporal_router)  # /api/temporal/* — Temporal Change Detection

from api.mola_detect_router import router as mola_detect_router
app.include_router(mola_detect_router)  # /api/mola-detect/* — MOLA Landform Detection

from api.terrain_features import router as terrain_features_router
app.include_router(terrain_features_router)  # /api/terrain/features_in_view

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

from api.hirise_landforms_router import router as hirise_landforms_router
app.include_router(hirise_landforms_router)  # /api/hirise-landforms/* — HiRISE Landform Classification

from api.ctx_tile_router import router as ctx_tile_router
app.include_router(ctx_tile_router)  # /api/ctx-mosaic/* — Murray Lab CTX Mosaic Tiles

from rag.rag_router import router as rag_router
app.include_router(rag_router)  # /api/rag/* — Mars Science RAG

# Download page "Smart Search" (natural language → products). Its mount was
# dropped in the April phase-1 cut while the UI stayed, so it 404'd since.
from api.llama_search import router as smart_search_router
app.include_router(smart_search_router)  # /api/ai/smart/*

from api.inspector_router import router as inspector_router
app.include_router(inspector_router)  # /api/inspector/at-point — 4-lane aggregator (Phase 3)

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
# SHARAD static mounts
# ======================================================
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
# HiRISE DTM elevation, score filtering, feature memo
# ======================================================
from api.dtm_router import router as dtm_router  # noqa: E402
app.include_router(dtm_router)  # /hirise_dtm/elevation/{product_id}

from api.scores_router import router as scores_router  # noqa: E402
app.include_router(scores_router)  # /api/filter/ice, /api/filter/hyd, /api/score/stats

from api.feature_memo_router import router as feature_memo_router  # noqa: E402
app.include_router(feature_memo_router)  # /api/feature_memo

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

from api.crossings_router import router as crossings_router  # noqa: E402
app.include_router(crossings_router)  # /api/crossings — SHARAD x HiRISE
