"""In-memory GeoJSON index cache (loaded once at startup).

This module is the single owner of the cache dicts. ``app.py`` re-exports
``_geojson_cache`` and ``refresh_geojson_cache`` so existing lazy imports
(``from app import _geojson_cache``) keep resolving to the same objects.
"""
import gzip
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor

from api.paths import BASE_DIR

logger = logging.getLogger("marslab")

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
    "ctx_mosaic": os.path.join(BASE_DIR, "ctx_mosaic_data", "index.geojson"),
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
