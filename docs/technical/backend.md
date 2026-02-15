# Backend Documentation

This document provides a deep dive into the MarsLab Python backend, including tech stack, module breakdown, services, caching, and configuration.

---

## Tech Stack

| Component | Version | Purpose |
|-----------|---------|---------|
| **FastAPI** | >= 0.100.0 | Async web framework |
| **Uvicorn** | >= 0.22.0 | ASGI server |
| **Pydantic** | >= 2.0.0 | Data validation |
| **aiohttp** | >= 3.8.0 | Async HTTP client (ODE queries) |
| **aiofiles** | >= 23.0.0 | Async file I/O |
| **NumPy** | >= 1.24.0 | Numerical operations |
| **OpenCV** | >= 4.8.0 | Image encoding/decoding |
| **Pillow** | >= 9.5.0 | Image processing |
| **Rasterio** | >= 1.3.0 | GeoTIFF processing |
| **Shapely** | (optional) | Geometry operations |
| **GDAL** | (system) | JP2 to GeoTIFF conversion |

**Code Reference:** `backend/requirements.txt`

---

## Directory & Module Breakdown

### Main Application

**File:** `backend/app.py` (576 lines)

The main FastAPI application that:
- Configures CORS middleware
- Mounts static file directories
- Defines core endpoints
- Includes routers from submodules

```python
# Core initialization
app = FastAPI()
app.add_middleware(CORSMiddleware, ...)

# Router includes
app.include_router(crism_router, prefix="/crism")
app.include_router(search_router)
app.include_router(footprints_router)
app.include_router(hirise_pixel_router, prefix="/hirise")
```

### API Routers

#### Search Router
**File:** `backend/api/search_router.py`

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/search` | GET | Search ODE by product ID |
| `/api/search/spatial` | GET | Search by bounding box |
| `/api/exists/{instrument}/{product_id}` | GET | Check local existence |
| `/api/download` | POST | Start download task |
| `/api/download` | GET | List all downloads |
| `/api/download/{task_id}` | GET | Get download status |

#### Footprints Router
**File:** `backend/api/footprints_router.py`

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/footprints` | GET | Get footprints in viewport |
| `/api/footprints/stats` | GET | Get footprint statistics |

#### CRISM Router
**File:** `backend/api/crism/router.py`

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/crism/{product_id}/rgb` | POST | Generate RGB image |
| `/crism/{product_id}/spectrum` | POST | Extract pixel spectrum |

#### HiRISE Pixel Router
**File:** `backend/api/hirise_pixel.py`

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/hirise/pixel_xy` | GET | Get pixel value at coordinates |
| `/hirise/window_xy` | GET | Get statistics for window |

### Service Modules

#### ODE Client
**File:** `backend/api/ode_client.py` (895 lines)

Handles communication with NASA's Orbital Data Explorer REST API.

**Key Classes:**
```python
@dataclass
class ODEProduct:
    product_id: str
    instrument: Instrument
    lat: Optional[float]
    lon: Optional[float]
    pdsid: Optional[str]
    ihid: Optional[str]
    iid: Optional[str]
    pt: Optional[str]

@dataclass
class CRISMBundle:
    base_key: str
    img_file: Optional[ODEFile]
    lbl_file: Optional[ODEFile]
    hdr_file: Optional[ODEFile]
    tab_file: Optional[ODEFile]
    browse_files: List[ODEFile]
    product_type: Optional[str]
```

**Key Functions:**
- `search_ode_products()` - Search by product ID
- `search_ode_spatial()` - Search by coordinates
- `resolve_crism_bundle()` - Resolve file URLs for CRISM
- `resolve_hirise_bundle()` - Resolve file URLs for HiRISE
- `parse_crism_base_key()` - Extract base key from product ID

#### Download Manager
**File:** `backend/api/download_manager.py` (600+ lines)

Manages async download tasks with progress tracking.

**Key Class:**
```python
@dataclass
class DownloadTask:
    task_id: str
    product_id: str
    base_key: str
    instrument: Instrument
    status: DownloadStatus  # pending, queued, downloading, processing, completed, failed
    files: List[FileProgress]
    target_dir: str
    created_at: datetime
    completed_at: Optional[datetime]
    error: Optional[str]
```

**Singleton Pattern:**
```python
# Global singleton instance
download_manager = DownloadManager()

def get_download_manager() -> DownloadManager:
    return download_manager
```

#### Instrument Registry
**File:** `backend/api/registry.py`

Centralized instrument configuration management.

**Key Class:**
```python
@dataclass
class InstrumentConfig:
    id: str
    name: str
    display_name: str
    geometry_type: str
    color: str
    data_directory: str
    index_file: str
    product_id_pattern: str
    quickview_path: str
    supports_spectrum: bool
    supports_rgb: bool
    file_types: Dict[str, List[str]]
```

**Methods:**
- `get(instrument_id)` - Get config by ID
- `detect_instrument(product_id)` - Auto-detect from product ID
- `load_index(instrument_id)` - Load cached GeoJSON index

### CRISM Processing Modules

**Directory:** `backend/api/crism/`

| Module | Purpose |
|--------|---------|
| `loader.py` | Load ENVI hyperspectral cubes |
| `processing.py` | RGB generation, normalization |
| `resolver.py` | File path resolution |
| `rgb.py` | RGB image generation from wavelengths |
| `spectrum.py` | Extract spectral data for pixels |

---

## Data Storage & Loading

### Storage Layout

```
backend/
├── crism_data/
│   ├── index.geojson         # Footprint index (1.6 MB)
│   ├── ice_score_stats.json  # Precomputed statistics
│   └── {base_key}/           # Per-observation directories
│       ├── {product_id}.img  # ENVI image data
│       ├── {product_id}.lbl  # PDS label
│       ├── {product_id}.hdr  # ENVI header
│       └── {product_id}.tab  # Wavelength table
├── hirise_data/
│   ├── index.geojson         # Footprint index
│   └── {product_id}_RED.tif  # GeoTIFF images
├── sharad_data/
│   └── index.geojson         # Track index
├── crism_quickview/          # Preview thumbnails
├── crism_browse/             # Browse products (HYD, ICE, etc.)
├── hirise_quickview/         # HiRISE thumbnails
├── sharad_quickview/         # SHARAD thumbnails
└── sharad_highres/           # Full-res SHARAD
```

### Index File Format

GeoJSON FeatureCollection with instrument-specific properties:

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {
        "product_id": "frt0001fd76_07_if166j_mtr3",
        "instrument": "CRISM",
        "mtr3_img": "frt0001fd76_07_if166j_mtr3.img",
        "quicklook": "/crism/quickview/..."
      },
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[lon, lat], ...]]
      }
    }
  ]
}
```

### Loading Patterns

**GeoTIFF Loading (Rasterio):**
```python
@lru_cache(maxsize=32)
def open_ds(path: str):
    return rasterio.open(path)

# Read with resampling
data = ds.read(
    1,
    out_shape=(out_height, out_width),
    resampling=Resampling.bilinear
)
```

**ENVI Cube Loading:**
```python
import spectral.io.envi as envi

cube = envi.open(hdr_path, img_path)
data = cube.load()  # numpy array [lines, samples, bands]
```

---

## Core Services

### Product Registry Service

**Location:** `backend/api/registry.py`

The registry provides centralized instrument configuration:

```python
from api.registry import get_registry

registry = get_registry()
config = registry.get("crism")

# Access paths
index_path = config.get_index_path(registry.base_dir)
quickview_path = config.get_quickview_path(registry.base_dir)

# Auto-detect instrument
config = registry.detect_instrument("frt00009312_07_if165l_trr3")
# Returns: InstrumentConfig for CRISM
```

### Footprint Query Service

**Location:** `backend/api/footprints_router.py`

Provides viewport-based footprint queries with LOD support:

```python
# LOD thresholds
LOD_THRESHOLDS_KM = {
    "FAR": 15000,   # > 15,000 km: no footprints
    "MID": 5000,    # 5,000-15,000 km: points only
}

# Simplification tolerances (degrees)
SIMPLIFY_TOLERANCES = {
    "low": 0.01,    # ~1km at equator
    "mid": 0.005,   # ~500m at equator
    "high": 0.001,  # ~100m at equator
}
```

**Key Features:**
- Bounding box filtering
- LOD enforcement based on camera height
- Douglas-Peucker geometry simplification
- Antimeridian crossing support
- Result limiting with truncation metadata

### Overlay Serving Service

**Location:** `backend/app.py`

Serves overlay images with transparency:

```python
@app.get("/hirise/overlay/{product_id}.png")
def get_hirise_overlay(product_id: str, max_size: int = 2048):
    # Check disk cache
    cache_file = os.path.join(OVERLAY_CACHE_DIR, f"{product_id}_{max_size}.png")
    if os.path.exists(cache_file):
        return cached_response

    # Process GeoTIFF
    ds = open_ds(path)
    data = ds.read(1, out_shape=(out_height, out_width), ...)

    # Create RGBA with transparency
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, :3] = data_norm
    rgba[:, :, 3] = np.where(data_norm > 5, 255, 0)

    # Cache and return
    ...
```

---

## Caching Strategy

### LRU Cache Configuration

| Function | File | Max Size | Purpose |
|----------|------|----------|---------|
| `open_ds()` | `app.py:72` | 32 | Rasterio dataset handles |
| `world_union_extent()` | `app.py:80` | 1 | Global extent calculation |
| `load_world_tile()` | `app.py:136` | 8192 | Tile PNG bytes |
| `load_geojson_index()` | `footprints_router.py:53` | 8 | GeoJSON indexes |
| `load_index()` | `registry.py` | 8 | Instrument indexes |
| `open_dataset()` | `hirise_pixel.py` | 16 | HiRISE datasets |

### Disk Cache

**Overlay Cache (`backend/.overlay_cache/`):**
- Stores processed HiRISE overlay PNGs
- Key format: `{product_id}_{max_size}.png`
- No TTL (persistent until manually cleared)
- Created automatically on first request

### Global State Cache

**Score Statistics (`backend/app.py:448-459`):**
```python
_score_stats_cache = None

def _load_score_stats():
    global _score_stats_cache
    if _score_stats_cache is None:
        if os.path.exists(SCORE_STATS_FILE):
            with open(SCORE_STATS_FILE, "r") as f:
                _score_stats_cache = json.load(f)
        else:
            _score_stats_cache = {}
    return _score_stats_cache
```

### Cache Invalidation

**Manual Invalidation:**
```python
# Clear index cache
from api.registry import get_registry
get_registry().clear_index_cache()

# Clear LRU caches (restart required)
# LRU caches have no programmatic clear method
```

**Automatic Invalidation:**
- Disk cache: Never (manual cleanup required)
- LRU caches: LRU eviction when maxsize exceeded
- Global caches: Never (restart required)

---

## Error Handling Patterns

### HTTP Exceptions

```python
from fastapi import HTTPException

# 400 Bad Request
raise HTTPException(status_code=400, detail="Invalid bbox format")

# 404 Not Found
raise HTTPException(status_code=404, detail="Product not found")

# 409 Conflict
raise HTTPException(status_code=409, detail="Product already downloaded")

# 500 Internal Server Error
raise HTTPException(status_code=500, detail=str(e))
```

### Graceful Degradation

```python
# Optional dependencies
try:
    from shapely.geometry import shape
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False
    print("[WARNING] shapely not installed - geometry simplification disabled")

# Check before use
if simplify and SHAPELY_AVAILABLE:
    geom = simplify_geometry(geom, tolerance)
```

### Debug Error Responses

```python
@app.get("/world_meta")
def get_world_meta():
    try:
        extent = world_union_extent()
    except Exception as e:
        # Return debug info instead of 500
        return {
            "error": str(e),
            "data_dir": HIRISE_DATA_DIR,
            "exists": os.path.exists(HIRISE_DATA_DIR),
            "files": os.listdir(HIRISE_DATA_DIR) if exists else None,
        }
```

---

## Configuration & Environment Variables

### Hard-Coded Paths

All paths are relative to the backend directory:

```python
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

HIRISE_DATA_DIR = os.path.join(BASE_DIR, "hirise_data")
CRISM_DATA_DIR = os.path.join(BASE_DIR, "crism_data")
SHARAD_DATA_DIR = os.path.join(BASE_DIR, "sharad_data")
OVERLAY_CACHE_DIR = os.path.join(BASE_DIR, ".overlay_cache")
CRISM_SCORE_DIR = os.path.join(BASE_DIR, "crism_score")
```

### Constants

| Constant | Value | Location | Purpose |
|----------|-------|----------|---------|
| `TILE_SIZE` | 256 | `app.py:58` | Tile dimensions |
| `MAX_ZOOM` | 8 | `app.py:59` | Maximum zoom level |
| `DEFAULT_LIMIT` | 2000 | `footprints_router.py:49` | Default footprint limit |
| `MAX_LIMIT` | 5000 | `footprints_router.py:50` | Maximum footprint limit |
| `LOD_THRESHOLDS_KM` | `{FAR: 15000, MID: 5000}` | `footprints_router.py:211-216` | LOD enforcement |

### Score Thresholds

```python
SCORE_THRESHOLDS = [
    0.05, 0.1, 0.15, 0.2, 0.25, 0.3,
    0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
    1.0, 1.2, 1.5
]
```

### External API Endpoints

```python
ODE_REST_BASE = "https://oderest.rsl.wustl.edu/live2"
ODE_PRODUCTFILES_BASE = "https://ode.rsl.wustl.edu/mars/productfiles.aspx"
```

---

## Configuration Table

| Setting | Default | Description | How to Change |
|---------|---------|-------------|---------------|
| Port | 8000 | Server port | `uvicorn app:app --port N` |
| Workers | 1 | Uvicorn workers | `uvicorn app:app --workers N` |
| CORS Origins | `["*"]` | Allowed origins | Edit `app.py:31` |
| Overlay Max Size | 2048 | Max overlay dimension | Query param `max_size` |
| Footprint Limit | 2000 | Default result limit | Query param `limit` |
| LOD Far Threshold | 15000 km | No footprints above | Edit `footprints_router.py:212` |
| LOD Mid Threshold | 5000 km | Points only above | Edit `footprints_router.py:213` |

---

## Static File Mounts

| Path | Directory | Purpose |
|------|-----------|---------|
| `/hirise_viewer` | `hirise_viewer/` | HiRISE viewer iframe |
| `/hirise_lbl` | `hirise_data/` | HiRISE label files |
| `/hirise/quickview` | `hirise_quickview/` | HiRISE thumbnails |
| `/crism/quickview` | `crism_quickview/` | CRISM thumbnails |
| `/crism/browse` | `crism_browse/` | CRISM browse products |
| `/crism_lbl` | `crism_data/` | CRISM label files |
| `/sharad/quickview` | `sharad_quickview/` | SHARAD thumbnails |
| `/sharad/highres` | `sharad_highres/` | SHARAD high-res |

**Code Reference:** `backend/app.py:44-439`
