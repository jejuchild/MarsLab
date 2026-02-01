# Performance Optimization Notes

## Overview

This document explains the performance optimizations implemented to address the severe performance issue of rendering global footprints in the MarsLab application.

## Problem Statement

**Original Issue**: Loading and displaying all footprints globally (via static GeoJSON files) caused:
- Massive CPU/GPU load (draw calls, entity count, re-renders)
- Memory pressure from thousands of Cesium entities
- UI freezes during panning/zooming
- Slow initial map load times

**Root Cause**: The app was loading entire GeoJSON index files (1.6MB+ for CRISM alone with ~5700 features) and creating Cesium entities for all footprints regardless of whether they were visible in the current viewport.

## Solution: Viewport-Based Loading Pipeline

### Architecture

```
┌─────────────────┐     moveEnd      ┌──────────────────┐
│  Cesium Camera  │ ───────────────► │ FootprintManager │
└─────────────────┘    (debounced)   └──────────────────┘
                                              │
                                              ▼
                                     ┌──────────────────┐
                                     │  Compute Bbox    │
                                     │  Determine LOD   │
                                     └──────────────────┘
                                              │
                              ┌───────────────┼───────────────┐
                              ▼               ▼               ▼
                         ┌─────────┐    ┌─────────┐    ┌─────────┐
                         │  Cache  │    │  Cache  │    │  Fetch  │
                         │   Hit   │    │  Miss   │    │   API   │
                         └─────────┘    └─────────┘    └─────────┘
                              │               │               │
                              └───────────────┼───────────────┘
                                              ▼
                                     ┌──────────────────┐
                                     │   Diff Render    │
                                     │ (add/remove only │
                                     │     changed)     │
                                     └──────────────────┘
```

### Key Components

#### 1. Backend: `/api/footprints` Endpoint

**Location**: `backend/api/footprints_router.py`

**Features**:
- **Spatial Filtering**: Filters features by bounding box intersection
- **LOD Support**: Returns different geometry detail levels
  - `none`: No features (for far zoom)
  - `point`: Centroids only (for mid zoom)
  - `poly`: Full polygons (for close zoom)
- **Geometry Simplification**: Douglas-Peucker algorithm via Shapely
  - `low`: 0.01° tolerance (~1km)
  - `mid`: 0.005° tolerance (~500m)
  - `high`: 0.001° tolerance (~100m)
- **Antimeridian Handling**: Properly splits queries crossing ±180°
- **Result Limiting**: Default 2000, max 5000 features
- **Truncation Metadata**: Returns `truncated`, `returned`, `total_estimate`

**Query Parameters**:
```
GET /api/footprints?instrument=CRISM&bbox=-10,-5,10,5&lod=poly&simplify=mid&limit=2000
```

#### 2. Frontend: FootprintManager

**Location**: `frontend/src/utils/FootprintManager.ts`

**Features**:

- **Debounced Camera Events**: 300ms debounce on `moveEnd` to avoid request spam
- **Automatic LOD Selection**: Based on camera height
  - `> 20,000 km`: `none` (no footprints)
  - `> 2,000 km`: `point` (centroids only)
  - `< 2,000 km`: `poly` (full polygons)
- **LRU Cache**: 100-entry cache for bbox+lod+simplify combinations
- **AbortController**: Cancels in-flight requests when camera moves again
- **Diff-Based Rendering**: Only adds/removes changed features, avoiding full re-render
- **Memory Efficient**: Maintains feature maps per instrument for O(1) lookups

#### 3. LOD Thresholds

```typescript
const LOD_THRESHOLDS = {
  FAR: 20_000_000,   // 20,000 km: no footprints
  MID: 2_000_000,    // 2,000 km: points only
  CLOSE: 2_000_000,  // Below: full polygons
};

const SIMPLIFY_THRESHOLDS = {
  HIGH: 500_000,     // Below 500km: minimal simplification
  MID: 1_000_000,    // Below 1000km: medium simplification
  // Above: aggressive simplification
};
```

## Performance Improvements

| Metric | Before | After |
|--------|--------|-------|
| Initial Load | ~5000 entities | 0 entities |
| Zoomed View | ~5000 entities | ~100-500 entities |
| Request Count | 1 (full GeoJSON) | 1 per viewport change |
| Memory Usage | All features loaded | Only visible features |
| UI Responsiveness | Freezes on pan/zoom | Smooth interaction |

## Tuning Guide

### Adjusting LOD Thresholds

Edit `frontend/src/utils/FootprintManager.ts`:

```typescript
const LOD_THRESHOLDS = {
  FAR: 20_000_000,  // Increase to show points at higher zoom
  MID: 2_000_000,   // Adjust point-to-polygon transition
};
```

### Adjusting Simplification

Edit `backend/api/footprints_router.py`:

```python
SIMPLIFY_TOLERANCES = {
    "low": 0.01,    # Increase for more aggressive simplification
    "mid": 0.005,
    "high": 0.001,  # Decrease for finer detail
}
```

### Adjusting Debounce

Edit `frontend/src/components/MapView.tsx`:

```typescript
const footprintManager = new FootprintManager({
  debounceMs: 300,  // Increase if too many requests
  // ...
});
```

### Adjusting Cache Size

```typescript
const footprintManager = new FootprintManager({
  maxCacheSize: 100,  // Increase if revisiting areas often
  // ...
});
```

### Adjusting Result Limits

Backend default in `footprints_router.py`:
```python
DEFAULT_LIMIT = 2000  # Increase for denser areas
MAX_LIMIT = 5000      # Maximum allowed
```

## Backward Compatibility

The old entity-based system is preserved for:
- Score map overlays (ICE, Hydration)
- High-resolution image overlays
- SHARAD track rendering (uses LineString)

Both systems coexist, with the FootprintManager handling viewport-based polygon/point rendering and the legacy system handling overlays.

## Dependencies

### Backend
- `shapely` (optional but recommended for geometry simplification)
  ```bash
  pip install shapely
  ```

### Frontend
- No additional dependencies (uses existing Cesium)

## API Reference

### GET /api/footprints

Returns footprints within a bounding box.

**Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| instrument | string | Yes | CRISM, HIRISE, or SHARAD |
| bbox | string | Yes | minLon,minLat,maxLon,maxLat (degrees) |
| lod | string | No | none, point, or poly (default: poly) |
| simplify | string | No | low, mid, or high |
| limit | int | No | Max features (default: 2000, max: 5000) |

**Response**:
```json
{
  "type": "FeatureCollection",
  "features": [...],
  "metadata": {
    "truncated": false,
    "returned": 150,
    "total_estimate": 150,
    "lod": "poly",
    "simplify": "mid",
    "bbox": [-10, -5, 10, 5],
    "instrument": "CRISM"
  }
}
```

### GET /api/footprints/stats

Returns statistics about footprint data.

**Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| instrument | string | Yes | CRISM, HIRISE, or SHARAD |

**Response**:
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
