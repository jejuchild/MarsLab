# API Reference

This document provides complete documentation for all MarsLab backend REST API endpoints.

---

## Base URL

- **Development:** `http://localhost:8000`
- **Via Vite proxy:** `http://localhost:5173` (proxied to backend)

---

## Endpoints Overview

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/footprints` | GET | Get footprints in viewport |
| `/api/footprints/stats` | GET | Get footprint statistics |
| `/api/search` | GET | Search products by ID |
| `/api/search/spatial` | GET | Search by bounding box |
| `/api/exists/{instrument}/{product_id}` | GET | Check local existence |
| `/api/download` | POST | Start download task |
| `/api/download` | GET | List all downloads |
| `/api/download/{task_id}` | GET | Get download status |
| `/api/filter/ice` | GET | Filter by ice score |
| `/api/filter/hyd` | GET | Filter by hydration score |
| `/api/score/stats` | GET | Get score statistics |
| `/crism/{product_id}/rgb` | POST | Generate RGB image |
| `/crism/{product_id}/spectrum` | POST | Extract spectrum |
| `/hirise/pixel_xy` | GET | Get pixel value |
| `/hirise/window_xy` | GET | Get window statistics |
| `/hirise/overlay/{product_id}.png` | GET | Get HiRISE overlay |
| `/hirise/quickview/{product_id}.png` | GET | Get HiRISE quickview |
| `/crism/quickview/{product_id}.png` | GET | Get CRISM quickview |
| `/sharad/quickview/{product_id}.jpg` | GET | Get SHARAD quickview |

---

## Footprints API

### GET /api/footprints

Get footprints within a bounding box with LOD support.

**Query Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `instrument` | string | Yes | - | `CRISM`, `HIRISE`, or `SHARAD` |
| `bbox` | string | Yes | - | `minLon,minLat,maxLon,maxLat` (degrees) |
| `lod` | string | No | `poly` | Level of detail: `none`, `point`, `poly` |
| `simplify` | string | No | - | Simplification: `low`, `mid`, `high` |
| `limit` | int | No | 2000 | Max features (1-5000) |
| `camera_height_km` | float | No | - | Camera height for server LOD enforcement |

**Response:**

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {
        "product_id": "frt0001fd76_07_if166j_mtr3",
        "instrument": "CRISM"
      },
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[74.12, 22.38], [74.38, 22.38], [74.38, 22.65], [74.12, 22.65], [74.12, 22.38]]]
      }
    }
  ],
  "metadata": {
    "truncated": false,
    "returned": 150,
    "total_estimate": 150,
    "lod": "poly",
    "original_lod": "poly",
    "lod_enforced": false,
    "simplify": null,
    "bbox": [-10, -5, 10, 5],
    "instrument": "CRISM"
  }
}
```

**Example Request:**

```bash
curl "http://localhost:8000/api/footprints?instrument=CRISM&bbox=-10,-5,10,5&lod=poly&limit=100"
```

**LOD Enforcement:**

When `camera_height_km` is provided:
- `> 15,000 km`: Forces `lod=none` (no footprints)
- `5,000-15,000 km`: Downgrades `poly` to `point`
- `< 5,000 km`: Full polygons allowed

---

### GET /api/footprints/stats

Get statistics about footprint data for an instrument.

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `instrument` | string | Yes | `CRISM`, `HIRISE`, or `SHARAD` |

**Response:**

```json
{
  "instrument": "CRISM",
  "total_features": 5700,
  "bounds": {
    "min_lon": -180,
    "max_lon": 180,
    "min_lat": -90,
    "max_lat": 90
  },
  "shapely_available": true
}
```

**Example Request:**

```bash
curl "http://localhost:8000/api/footprints/stats?instrument=CRISM"
```

---

## Search API

### GET /api/search

Search ODE for products by product ID (typeahead).

**Query Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `q` | string | Yes | - | Search query (partial product ID) |
| `instrument` | string | No | auto-detect | Filter by `crism`, `hirise`, or `sharad` |
| `limit` | int | No | 10 | Max results (1-50) |

**Response:**

```json
{
  "query": "frt00009",
  "results": [
    {
      "product_id": "frt00009312_07_if165l_trr3",
      "instrument": "crism",
      "base_key": "frt00009312_07",
      "lat": 18.5,
      "lon": -77.0,
      "exists": false,
      "has_core": false,
      "has_browse": false,
      "missing_files": ["img", "lbl", "hdr", "tab"]
    }
  ],
  "count": 1
}
```

**Example Request:**

```bash
curl "http://localhost:8000/api/search?q=frt00009&limit=10"
```

---

### GET /api/search/spatial

Search ODE for products within a bounding box.

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `minlat` | float | Yes | Southern latitude (-90 to 90) |
| `maxlat` | float | Yes | Northern latitude (-90 to 90) |
| `westernlon` | float | Yes | Western longitude (-180 to 360) |
| `easternlon` | float | Yes | Eastern longitude (-180 to 360) |
| `instrument` | string | No | Filter by instrument |
| `limit` | int | No | Max results (default 10) |

**Response:** Same format as `/api/search`

**Example Request:**

```bash
curl "http://localhost:8000/api/search/spatial?minlat=35&maxlat=70&westernlon=-130&easternlon=150"
```

---

### GET /api/exists/{instrument}/{product_id}

Check if a product exists locally with detailed file status.

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `instrument` | string | `crism`, `hirise`, or `sharad` |
| `product_id` | string | Product identifier |

**Response:**

```json
{
  "exists": true,
  "product_id": "frt00009312_07_if165l_trr3",
  "base_key": "frt00009312_07",
  "instrument": "crism",
  "has_core": true,
  "has_header": true,
  "has_wavelength": true,
  "has_browse": true,
  "missing_files": [],
  "existing_files": ["img", "lbl", "hdr", "tab", "browse_HYD", "browse_ICE"]
}
```

**Example Request:**

```bash
curl "http://localhost:8000/api/exists/crism/frt00009312_07_if165l_trr3"
```

---

## Download API

### POST /api/download

Start downloading a product bundle.

**Request Body:**

```json
{
  "product_id": "frt00009312_07_if165l_trr3",
  "instrument": "crism",
  "lat": 18.5,
  "lon": -77.0,
  "file_types": ["core", "browse"]
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `product_id` | string | Yes | Product identifier |
| `instrument` | string | Yes | `crism` or `hirise` |
| `lat` | float | No | Center latitude |
| `lon` | float | No | Center longitude |
| `file_types` | string[] | No | Filter: `core`, `header`, `wavelength`, `browse` |

**Response:**

```json
{
  "task_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "product_id": "frt00009312_07_if165l_trr3",
  "base_key": "frt00009312_07",
  "instrument": "crism",
  "status": "pending",
  "files": [],
  "target_dir": "/path/to/crism_data/frt00009312_07",
  "progress_percent": 0,
  "total_bytes": 0,
  "downloaded_bytes": 0,
  "created_at": "2024-01-15T10:30:00Z",
  "completed_at": null,
  "error": null
}
```

**Example Request:**

```bash
curl -X POST "http://localhost:8000/api/download" \
  -H "Content-Type: application/json" \
  -d '{"product_id": "frt00009312_07_if165l_trr3", "instrument": "crism"}'
```

---

### GET /api/download

List all download tasks.

**Response:**

```json
[
  {
    "task_id": "a1b2c3d4...",
    "product_id": "frt00009312_07_if165l_trr3",
    "status": "completed",
    "progress_percent": 100,
    ...
  }
]
```

---

### GET /api/download/{task_id}

Get status of a specific download task.

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `task_id` | string | Task UUID |

**Response:** Same format as POST response

**Status Values:**
- `pending` - Task created, not started
- `queued` - Waiting for download slot
- `downloading` - Actively downloading files
- `processing` - Post-processing (e.g., JP2 conversion)
- `completed` - All files downloaded successfully
- `failed` - Download failed (check `error` field)

---

## Score Filtering API

### GET /api/filter/ice

Filter CRISM observations by ice score.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_score` | float | 0.3 | Minimum ice score threshold |
| `min_percent` | float | 5.0 | Minimum % of pixels meeting threshold |

**Response:**

```json
{
  "passing_ids": ["frt00003156", "frt000033f8", ...],
  "total": 1505,
  "passing_count": 42,
  "params": {
    "min_score": 0.3,
    "min_percent": 5.0,
    "used_threshold": 0.3
  }
}
```

**Example Request:**

```bash
curl "http://localhost:8000/api/filter/ice?min_score=0.5&min_percent=10"
```

---

### GET /api/filter/hyd

Filter CRISM observations by hydration score. Same parameters and response format as `/api/filter/ice`.

---

### GET /api/score/stats

Get score statistics summary.

**Response:**

```json
{
  "total_observations": 1505,
  "available_thresholds": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5]
}
```

---

## CRISM Processing API

### POST /crism/{product_id}/rgb

Generate RGB image from CRISM hyperspectral data.

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `product_id` | string | CRISM product ID |

**Request Body:**

```json
{
  "r": 2.53,
  "g": 1.51,
  "b": 1.08
}
```

| Field | Type | Description |
|-------|------|-------------|
| `r` | float | Red channel wavelength (micrometers) |
| `g` | float | Green channel wavelength (micrometers) |
| `b` | float | Blue channel wavelength (micrometers) |

**Response:** PNG image (binary)

**Example Request:**

```bash
curl -X POST "http://localhost:8000/crism/frt0001fd76_07_if166j_mtr3/rgb" \
  -H "Content-Type: application/json" \
  -d '{"r": 2.53, "g": 1.51, "b": 1.08}' \
  --output rgb.png
```

---

### POST /crism/{product_id}/spectrum

Extract spectrum for a single pixel.

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `product_id` | string | CRISM product ID |

**Request Body:**

```json
{
  "line": 100,
  "sample": 200
}
```

**Response:**

```json
{
  "wavelengths": [1.0, 1.1, 1.2, ...],
  "values": [0.15, 0.16, 0.14, ...],
  "line": 100,
  "sample": 200
}
```

**Example Request:**

```bash
curl -X POST "http://localhost:8000/crism/frt0001fd76_07_if166j_mtr3/spectrum" \
  -H "Content-Type: application/json" \
  -d '{"line": 100, "sample": 200}'
```

---

## HiRISE API

### GET /hirise/pixel_xy

Get pixel value at specific coordinates.

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `product_id` | string | HiRISE product ID |
| `x` | int | X coordinate (sample) |
| `y` | int | Y coordinate (line) |

**Response:**

```json
{
  "value": 1234.5,
  "x": 100,
  "y": 200
}
```

---

### GET /hirise/window_xy

Get statistics for a window around coordinates.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `product_id` | string | - | HiRISE product ID |
| `x` | int | - | Center X coordinate |
| `y` | int | - | Center Y coordinate |
| `size` | int | 32 | Window size (pixels) |

**Response:**

```json
{
  "min": 100.0,
  "max": 2500.0,
  "mean": 1200.5,
  "std": 350.2,
  "histogram": {
    "bins": [100, 200, 300, ...],
    "counts": [10, 25, 42, ...]
  },
  "x": 100,
  "y": 200,
  "size": 32
}
```

---

### GET /hirise/overlay/{product_id}.png

Get processed HiRISE overlay with transparency.

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `product_id` | string | HiRISE product ID |

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_size` | int | 2048 | Maximum dimension (pixels) |

**Response:** PNG image with alpha channel (black pixels transparent)

**Example Request:**

```bash
curl "http://localhost:8000/hirise/overlay/ESP_024943_2345.png?max_size=1024" --output overlay.png
```

---

### GET /hirise/quickview/{product_id}.png

Get HiRISE quickview thumbnail with transparency.

**Response:** PNG image with alpha channel

---

## Static File Endpoints

### GET /crism/quickview/{product_id}.png

CRISM quickview thumbnail with transparency.

### GET /crism/browse/{filename}.png

CRISM browse products (HYD, ICE, IC2, etc.).

### GET /sharad/quickview/{product_id}.jpg

SHARAD radargram thumbnail.

### GET /sharad/highres/{product_id}.tif

SHARAD high-resolution radargram.

### GET /hirise_lbl/{filename}

HiRISE PDS label files.

### GET /crism_lbl/{path}

CRISM PDS label files.

---

## Index Endpoints

### GET /hirise_index.geojson

Complete HiRISE footprint index.

### GET /crism_index.geojson

Complete CRISM footprint index.

### GET /sharad_index.geojson

Complete SHARAD track index.

**Note:** These endpoints return the full index files. For viewport-based queries, use `/api/footprints` instead.

---

## Error Responses

All endpoints may return error responses:

**400 Bad Request:**
```json
{
  "detail": "Invalid bbox format: Expected 4 values"
}
```

**404 Not Found:**
```json
{
  "detail": "Product not found: frt00009999"
}
```

**409 Conflict:**
```json
{
  "detail": "Product already fully downloaded"
}
```

**500 Internal Server Error:**
```json
{
  "detail": "Failed to process GeoTIFF: [error message]"
}
```

---

## Rate Limiting

No rate limiting is currently implemented. For production deployments, consider adding rate limiting at the reverse proxy level.

---

## CORS

CORS is configured to allow all origins:

```python
allow_origins=["*"]
allow_methods=["*"]
allow_headers=["*"]
```

For production, restrict `allow_origins` to your frontend domain.
