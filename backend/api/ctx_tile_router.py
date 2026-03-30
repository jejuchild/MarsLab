"""
CTX Mosaic tile server — serves XYZ tiles from Murray Lab CTX 4°×4° GeoTIFFs.

Instead of a slow VRT over 140 tiles, directly opens only the TIF(s) that
overlap the requested tile bounds. Each TIF covers a 4°×4° geographic region.

Endpoint:  GET /api/ctx-mosaic/tile/{z}/{x}/{y}.png
"""

from __future__ import annotations

import math
from collections import OrderedDict
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

router = APIRouter(prefix="/api/ctx-mosaic")

# ── Config ──
COG_DIR = Path("/disk1/cspark/MarsLab/Data/CTX/arcadia_cog")
TILE_DIR = Path("/disk1/cspark/MarsLab/Data/CTX/arcadia_tiles")
TILE_SIZE = 256
MAX_CACHE = 2000
MARS_RADIUS = 3396190.0

# LRU tile cache: key → PNG bytes
_tile_cache: OrderedDict[str, bytes] = OrderedDict()

# Lazy rasterio dataset cache: filename → open dataset
_ds_cache: dict[str, object] = {}


def _tile_bounds_geographic(z: int, x: int, y: int):
    """Geographic bounds for Cesium GeographicTilingScheme tile."""
    n_x = 2 * (2 ** z)
    n_y = 1 * (2 ** z)
    lon_min = -180.0 + (x / n_x) * 360.0
    lon_max = -180.0 + ((x + 1) / n_x) * 360.0
    lat_max = 90.0 - (y / n_y) * 180.0
    lat_min = 90.0 - ((y + 1) / n_y) * 180.0
    return lon_min, lat_min, lon_max, lat_max


def _find_ctx_tifs(lon_min: float, lat_min: float, lon_max: float, lat_max: float) -> list[str]:
    """
    Find Murray Lab TIF filenames that overlap the given geographic bounds.
    Murray Lab tiles are 4°×4°, named by lower-left corner:
      E{lon}_N{lat}  (lon can be negative like E-180)
    """
    results = []

    # Iterate over all possible 4° grid cells that could overlap
    # Latitude grid: 32, 36, 40, ..., 56
    lat_start = max(32, int(math.floor(lat_min / 4) * 4))
    lat_end = min(56, int(math.floor(lat_max / 4) * 4))

    for grid_lat in range(lat_start, lat_end + 1, 4):
        if grid_lat + 4 < lat_min or grid_lat > lat_max:
            continue

        # Longitude: need to handle both positive (140..176) and negative (-180..-144) ranges
        for grid_lon in _overlapping_lons(lon_min, lon_max):
            tif_name = _tif_filename(grid_lon, grid_lat)
            if (COG_DIR / tif_name).exists() or (TILE_DIR / tif_name).exists():
                results.append(tif_name)

    return results


def _overlapping_lons(lon_min: float, lon_max: float) -> list[int]:
    """Return Murray Lab grid longitudes (4° steps) that overlap [lon_min, lon_max]."""
    lons = []
    # Positive range: 140, 144, ..., 176
    for g in range(140, 180, 4):
        if g + 4 > lon_min and g < lon_max:
            lons.append(g)
    # Negative range: -180, -176, ..., -144
    for g in range(-180, -140, 4):
        if g + 4 > lon_min and g < lon_max:
            lons.append(g)
    return lons


def _tif_filename(grid_lon: int, grid_lat: int) -> str:
    """Build Murray Lab TIF filename from grid coordinates."""
    if grid_lon < 0:
        lon_str = f"E{grid_lon:04d}"  # E-180, E-176, etc.
    else:
        lon_str = f"E{grid_lon:03d}"  # E140, E144, etc.
    lat_str = f"N{grid_lat:02d}"
    return f"MurrayLab_CTX_V01_{lon_str}_{lat_str}_Mosaic.tif"


def _open_tif(filename: str):
    """Open a TIF with lazy caching. Prefer COG (fast) over raw TIF (slow)."""
    if filename not in _ds_cache:
        import rasterio
        cog_path = COG_DIR / filename
        raw_path = TILE_DIR / filename
        # COG is internally tiled with overviews — much faster for partial reads
        path = cog_path if cog_path.exists() else raw_path
        _ds_cache[filename] = rasterio.open(str(path))
    return _ds_cache[filename]


def _lonlat_to_projected(lon_deg: float, lat_deg: float):
    """Convert geographic degrees to Mars equirectangular projected coords."""
    x = lon_deg * (math.pi / 180.0) * MARS_RADIUS
    y = lat_deg * (math.pi / 180.0) * MARS_RADIUS
    return x, y


def _read_tile_from_tif(tif_name: str, lon_min: float, lat_min: float,
                        lon_max: float, lat_max: float) -> Optional[np.ndarray]:
    """Read a region from a single TIF, resampled to TILE_SIZE."""
    try:
        from rasterio.windows import from_bounds
        from rasterio.enums import Resampling

        ds = _open_tif(tif_name)
        px_min, py_min = _lonlat_to_projected(lon_min, lat_min)
        px_max, py_max = _lonlat_to_projected(lon_max, lat_max)

        window = from_bounds(px_min, py_min, px_max, py_max, ds.transform)

        # Clip window to dataset bounds
        row_off = max(0, int(window.row_off))
        col_off = max(0, int(window.col_off))
        row_end = min(ds.height, int(window.row_off + window.height))
        col_end = min(ds.width, int(window.col_off + window.width))

        if row_end <= row_off or col_end <= col_off:
            return None

        from rasterio.windows import Window
        clipped = Window(col_off, row_off, col_end - col_off, row_end - row_off)

        data = ds.read(
            1,
            window=clipped,
            out_shape=(TILE_SIZE, TILE_SIZE),
            resampling=Resampling.bilinear,
        )
        return data
    except Exception:
        return None


@router.get("/tile/{z}/{x}/{y}.png")
def get_ctx_tile(z: int, x: int, y: int):
    """Render a 256x256 PNG tile from CTX mosaic TIFs."""
    cache_key = f"ctx_{z}_{x}_{y}"

    if cache_key in _tile_cache:
        _tile_cache.move_to_end(cache_key)
        return Response(
            content=_tile_cache[cache_key],
            media_type="image/png",
            headers={"Cache-Control": "public, max-age=604800"},
        )

    lon_min, lat_min, lon_max, lat_max = _tile_bounds_geographic(z, x, y)

    # Quick reject
    if lat_max < 30 or lat_min > 62:
        raise HTTPException(status_code=404, detail="No CTX data")
    in_positive = lon_max > 138 and lon_min < 182
    in_negative = lon_max > -182 and lon_min < -142
    if not in_positive and not in_negative:
        raise HTTPException(status_code=404, detail="No CTX data")

    tifs = _find_ctx_tifs(lon_min, lat_min, lon_max, lat_max)
    if not tifs:
        raise HTTPException(status_code=404, detail="No CTX data")

    # Read from each overlapping TIF and composite
    canvas = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.uint8)

    for tif_name in tifs:
        data = _read_tile_from_tif(tif_name, lon_min, lat_min, lon_max, lat_max)
        if data is not None:
            # Overlay non-zero pixels
            mask = data > 0
            canvas[mask] = data[mask]

    if np.max(canvas) == 0:
        raise HTTPException(status_code=404, detail="No CTX data")

    # Encode PNG
    from PIL import Image
    img = Image.fromarray(canvas, mode="L")
    buf = BytesIO()
    img.save(buf, format="PNG", optimize=True)
    png_bytes = buf.getvalue()

    # Cache
    if len(_tile_cache) >= MAX_CACHE:
        _tile_cache.popitem(last=False)
    _tile_cache[cache_key] = png_bytes

    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=604800"},
    )
