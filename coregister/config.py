"""Paths, URLs, and constants for the coregistration pipeline."""

from pathlib import Path

# ── Directories ───────────────────────────────────────────────────────
BASE_DIR = Path("/disk1/cspark/mastcam")
DATA_DIR = BASE_DIR / "coregister_data"
SPICE_DIR = DATA_DIR / "spice_kernels"
PDS_CACHE = DATA_DIR / "pds_cache"
OUTPUT_DIR = DATA_DIR / "output"

# ── NAIF SPICE Kernels ───────────────────────────────────────────────
NAIF_BASE = "https://naif.jpl.nasa.gov/pub/naif"
M2020_KERNEL_BASE = f"{NAIF_BASE}/M2020/kernels"

# ── Mars constants ────────────────────────────────────────────────────
MARS_BODY_ID = 499
MARS_RADII_KM = (3396.19, 3396.19, 3376.20)  # equatorial, equatorial, polar

# Jezero crater approximate center (for HiRISE DTM search)
JEZERO_LON = 77.45
JEZERO_LAT = 18.44
JEZERO_SEARCH_RADIUS_KM = 15.0

# ── Processing ────────────────────────────────────────────────────────
MAX_RETRIES = 3
DOWNLOAD_TIMEOUT = 120
CHUNK_SIZE = 8192
