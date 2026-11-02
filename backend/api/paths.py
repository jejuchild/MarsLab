"""Shared filesystem locations for the backend (single source of truth).

``BASE_DIR`` is the ``backend/`` directory (where ``app.py`` lives).
"""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HIRISE_DATA_DIR = os.path.join(BASE_DIR, "hirise_data")
CRISM_DATA_DIR = os.path.join(BASE_DIR, "crism_data")
SHARAD_DATA_DIR = os.path.join(BASE_DIR, "sharad_data")
HIRISE_DTM_DIR = os.path.join(BASE_DIR, "hirise_dtm_data")
HIRISE_VIEWER_DIR = os.path.join(BASE_DIR, "hirise_viewer")

# On-disk cache for generated overlay PNGs (TTL/LRU-evicted)
OVERLAY_CACHE_DIR = os.path.join(BASE_DIR, ".overlay_cache")
