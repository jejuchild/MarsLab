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
QUICKVIEW_CACHE_DIR = Path("/disk1/cspark/MarsLab/Data/CTX/quickview_cache")
QUICKVIEW_SIZE = 1024  # px
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

    # Convert ographic lat to ocentric for TIF grid lookup
    # CTX TIFs use ocentric latitude in their grid naming
    lat_min_oc = _ographic_to_ocentric(lat_min)
    lat_max_oc = _ographic_to_ocentric(lat_max)

    # Iterate over all possible 4° grid cells that could overlap
    # Latitude grid: 32, 36, 40, ..., 56 (ocentric)
    lat_start = max(32, int(math.floor(lat_min_oc / 4) * 4))
    lat_end = min(56, int(math.floor(lat_max_oc / 4) * 4))

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


# Mars flattening for ographic ↔ ocentric conversion
_MARS_F = 1.0 / 169.894447223612
_MARS_1MF2 = (1.0 - _MARS_F) ** 2  # (1-f)^2


def _ographic_to_ocentric(lat_deg: float) -> float:
    """Convert planetographic (geodetic) latitude to planetocentric."""
    lat_rad = math.radians(lat_deg)
    lat_c = math.atan(_MARS_1MF2 * math.tan(lat_rad))
    return math.degrees(lat_c)


def _lonlat_to_projected(lon_deg: float, lat_ographic_deg: float):
    """Convert ographic lon/lat to Mars ocentric equirectangular projected coords."""
    lat_ocentric = _ographic_to_ocentric(lat_ographic_deg)
    x = lon_deg * (math.pi / 180.0) * MARS_RADIUS
    y = lat_ocentric * (math.pi / 180.0) * MARS_RADIUS
    return x, y


def _read_tile_from_tif(tif_name: str, lon_min: float, lat_min: float,
                        lon_max: float, lat_max: float) -> Optional[np.ndarray]:
    """Read a region from a single TIF with per-row ographic→ocentric correction.

    Cesium tiles are in ographic (geodetic) latitude, but the TIF is in ocentric
    (geocentric) equirectangular projection. The conversion is non-linear, so we
    compute the correct projected y for each output row and sample accordingly.
    """
    try:
        ds = _open_tif(tif_name)
        inv_transform = ~ds.transform  # projected coords → pixel coords

        # X (longitude) is the same in both systems — compute column range once
        x_min = lon_min * (math.pi / 180.0) * MARS_RADIUS
        x_max = lon_max * (math.pi / 180.0) * MARS_RADIUS

        # Pixel columns for the lon range
        col_min_f, _ = inv_transform * (x_min, 0)
        col_max_f, _ = inv_transform * (x_max, 0)
        col_start = max(0, int(col_min_f))
        col_end = min(ds.width, int(col_max_f) + 1)
        if col_end <= col_start:
            return None
        col_width = col_end - col_start

        # For each output row, compute the ographic lat → ocentric → projected y → pixel row
        output = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.uint8)

        # Pre-compute the ocentric projected y for each output row
        from rasterio.enums import Resampling
        from rasterio.windows import Window

        lat_step = (lat_max - lat_min) / TILE_SIZE
        row_map = np.empty(TILE_SIZE, dtype=np.float64)
        for i in range(TILE_SIZE):
            # Output row 0 = lat_max (top), row TILE_SIZE-1 = lat_min (bottom)
            lat_ographic = lat_max - (i + 0.5) * lat_step
            lat_ocentric = _ographic_to_ocentric(lat_ographic)
            proj_y = lat_ocentric * (math.pi / 180.0) * MARS_RADIUS
            _, pix_row = inv_transform * (0, proj_y)
            row_map[i] = pix_row

        # Read a slightly larger window to cover all needed rows
        src_row_min = max(0, int(np.min(row_map)) - 1)
        src_row_max = min(ds.height, int(np.max(row_map)) + 2)
        if src_row_max <= src_row_min:
            return None

        window = Window(col_start, src_row_min, col_width, src_row_max - src_row_min)
        raw = ds.read(1, window=window)
        if raw.size == 0:
            return None

        # Resample: for each output row, pick the correct source row via linear interp
        from scipy.ndimage import map_coordinates

        # Build coordinate arrays for map_coordinates
        # row coordinates relative to the window
        src_rows = row_map - src_row_min  # fractional rows in the raw array
        src_cols = np.linspace(0, col_width - 1, TILE_SIZE)

        # Create 2D coordinate grids
        rr, cc = np.meshgrid(src_rows, src_cols, indexing="ij")
        output = map_coordinates(raw.astype(np.float32), [rr, cc], order=1, mode="nearest")
        output = np.clip(output, 0, 255).astype(np.uint8)

        return output
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

    # Quick reject (ographic bounds — slightly wider than ocentric 32-60)
    if lat_max < 29 or lat_min > 63:
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


# ── Quickview endpoint: per-tile downsampled PNG ──

@router.get("/quickview/{product_id}.png")
def get_ctx_quickview(product_id: str):
    """Serve a downsampled quickview PNG for a single CTX mosaic tile.

    product_id format: CTX_MOSAIC_E160_N44
    """
    QUICKVIEW_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = QUICKVIEW_CACHE_DIR / f"{product_id}.png"

    # Serve from disk cache
    if cache_path.exists():
        return Response(
            content=cache_path.read_bytes(),
            media_type="image/png",
            headers={"Cache-Control": "public, max-age=604800"},
        )

    # Parse product_id → TIF filename
    # CTX_MOSAIC_E160_N44 → MurrayLab_CTX_V01_E160_N44_Mosaic.tif
    parts = product_id.split("_")
    if len(parts) < 4 or parts[0] != "CTX" or parts[1] != "MOSAIC":
        raise HTTPException(status_code=400, detail="Invalid product_id")

    lon_str = parts[2]  # E160 or E-180
    lat_str = parts[3]  # N44
    tif_name = f"MurrayLab_CTX_V01_{lon_str}_{lat_str}_Mosaic.tif"

    cog_path = COG_DIR / tif_name
    if not cog_path.exists():
        raw_path = TILE_DIR / tif_name
        if not raw_path.exists():
            raise HTTPException(status_code=404, detail="Tile not found")

    try:
        from rasterio.enums import Resampling

        ds = _open_tif(tif_name)
        data = ds.read(
            1,
            out_shape=(QUICKVIEW_SIZE, QUICKVIEW_SIZE),
            resampling=Resampling.bilinear,
        )

        from PIL import Image
        img = Image.fromarray(data, mode="L")
        buf = BytesIO()
        img.save(buf, format="PNG", optimize=True)
        png_bytes = buf.getvalue()

        # Save to disk cache
        cache_path.write_bytes(png_bytes)

        return Response(
            content=png_bytes,
            media_type="image/png",
            headers={"Cache-Control": "public, max-age=604800"},
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate quickview: {e}")
