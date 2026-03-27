"""
CTX Mosaic tile server — serves XYZ tiles from Murray Lab CTX GeoTIFF mosaic.

Uses a GDAL VRT built from individual 4°×4° tiles. Tiles are rendered on demand
and cached in an LRU dict to avoid re-reading for repeated requests.

Endpoint:  GET /api/ctx-mosaic/tile/{z}/{x}/{y}.png
"""

from __future__ import annotations

import math
from collections import OrderedDict
from io import BytesIO
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

router = APIRouter(prefix="/api/ctx-mosaic")

# ── Config ──
VRT_PATH = Path("/disk1/cspark/MarsLab/Data/CTX/arcadia_tiles/ctx_arcadia.vrt")
TILE_SIZE = 256
MAX_CACHE = 2000  # ~2000 tiles × ~50KB ≈ 100MB RAM

# Mars radius for geographic tiling
MARS_RADIUS = 3396190.0

# LRU tile cache
_tile_cache: OrderedDict[str, bytes] = OrderedDict()

# Lazy-loaded rasterio dataset
_vrt_ds = None


def _get_vrt():
    """Open the VRT lazily (once per process)."""
    global _vrt_ds
    if _vrt_ds is None:
        import rasterio
        if not VRT_PATH.exists():
            raise RuntimeError(f"CTX VRT not found: {VRT_PATH}")
        _vrt_ds = rasterio.open(str(VRT_PATH))
    return _vrt_ds


def _tile_bounds_geographic(z: int, x: int, y: int):
    """
    Compute geographic bounds (lon/lat in degrees) for a tile in
    Cesium's GeographicTilingScheme (2 tiles at level 0).
    """
    n_x = 2 * (2 ** z)  # number of tiles in x at this level
    n_y = 1 * (2 ** z)  # number of tiles in y at this level

    lon_min = -180.0 + (x / n_x) * 360.0
    lon_max = -180.0 + ((x + 1) / n_x) * 360.0
    lat_max = 90.0 - (y / n_y) * 180.0
    lat_min = 90.0 - ((y + 1) / n_y) * 180.0

    return lon_min, lat_min, lon_max, lat_max


def _lonlat_to_projected(lon_deg: float, lat_deg: float):
    """Convert geographic degrees to Mars equirectangular projected coords."""
    x = lon_deg * (math.pi / 180.0) * MARS_RADIUS
    y = lat_deg * (math.pi / 180.0) * MARS_RADIUS
    return x, y


@router.get("/tile/{z}/{x}/{y}.png")
def get_ctx_tile(z: int, x: int, y: int):
    """Render a 256×256 PNG tile from the CTX mosaic VRT."""
    cache_key = f"ctx_{z}_{x}_{y}"

    # Check cache
    if cache_key in _tile_cache:
        _tile_cache.move_to_end(cache_key)
        return Response(
            content=_tile_cache[cache_key],
            media_type="image/png",
            headers={"Cache-Control": "public, max-age=604800"},
        )

    # Geographic bounds of this tile
    lon_min, lat_min, lon_max, lat_max = _tile_bounds_geographic(z, x, y)

    # Quick reject: CTX Arcadia data is roughly lat 32–60N, lon 140E–216E (-144W)
    # Skip tiles clearly outside this region
    if lat_max < 30 or lat_min > 62:
        raise HTTPException(status_code=404, detail="No CTX data")
    # Longitude check: data spans 140..180 and -180..-144
    in_positive = lon_max > 138 and lon_min < 182
    in_negative = lon_max > -182 and lon_min < -142
    if not in_positive and not in_negative:
        raise HTTPException(status_code=404, detail="No CTX data")

    try:
        from rasterio.windows import from_bounds
        from rasterio.enums import Resampling

        ds = _get_vrt()

        # Convert geographic bounds to projected coordinates
        px_min, py_min = _lonlat_to_projected(lon_min, lat_min)
        px_max, py_max = _lonlat_to_projected(lon_max, lat_max)

        # Get the rasterio window for this extent
        window = from_bounds(px_min, py_min, px_max, py_max, ds.transform)

        # Read data — resampled to TILE_SIZE × TILE_SIZE
        data = ds.read(
            1,
            window=window,
            out_shape=(TILE_SIZE, TILE_SIZE),
            resampling=Resampling.bilinear,
        )

        # If entirely nodata / zero, return 404
        if data is None or np.max(data) == 0:
            raise HTTPException(status_code=404, detail="No CTX data")

        # Encode as PNG
        from PIL import Image
        img = Image.fromarray(data, mode="L")
        buf = BytesIO()
        img.save(buf, format="PNG", optimize=True)
        png_bytes = buf.getvalue()

    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=404, detail="No CTX data")

    # Cache with LRU eviction
    if len(_tile_cache) >= MAX_CACHE:
        _tile_cache.popitem(last=False)
    _tile_cache[cache_key] = png_bytes

    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=604800"},
    )
