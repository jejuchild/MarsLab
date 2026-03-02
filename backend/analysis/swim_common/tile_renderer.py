"""
Tile renderer for SWIM consistency layers.

Generates colored PNG tiles for CesiumJS map overlay from SWIM GeoTIFF data.
All SWIM methods share the same -1 to +1 consistency color scale:
  +0.7 to +1.0: Deep blue (#1a237e) — strong ice evidence
  +0.3 to +0.7: Medium blue (#42a5f5) — moderate ice evidence
  -0.3 to +0.3: Gray (#9e9e9e) — ambiguous / no data
  -0.7 to -0.3: Light red (#ef9a9a) — moderate against ice
  -1.0 to -0.7: Deep red (#b71c1c) — strong against ice
"""

import io
import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# SWIM consistency color map (RGBA, 0-255)
# Interpolated from deep red (-1) through gray (0) to deep blue (+1)
SWIM_COLORMAP = {
    "strong_ice": (26, 35, 126, 200),      # #1a237e — deep blue
    "moderate_ice": (66, 165, 245, 180),    # #42a5f5 — medium blue
    "ambiguous": (158, 158, 158, 120),      # #9e9e9e — gray
    "moderate_no_ice": (239, 154, 154, 180),  # #ef9a9a — light red
    "strong_no_ice": (183, 28, 28, 200),    # #b71c1c — deep red
    "no_data": (0, 0, 0, 0),               # transparent
}

# Breakpoints for color interpolation
_COLOR_STOPS = [
    (-1.0, np.array([183, 28, 28, 200], dtype=np.uint8)),
    (-0.7, np.array([239, 154, 154, 180], dtype=np.uint8)),
    (-0.3, np.array([158, 158, 158, 120], dtype=np.uint8)),
    (0.3,  np.array([158, 158, 158, 120], dtype=np.uint8)),
    (0.7,  np.array([66, 165, 245, 180], dtype=np.uint8)),
    (1.0,  np.array([26, 35, 126, 200], dtype=np.uint8)),
]


def _value_to_rgba(value: float) -> Tuple[int, int, int, int]:
    """Map a consistency value (-1 to +1) to RGBA color."""
    if np.isnan(value) or value <= -30:  # no-data
        return (0, 0, 0, 0)

    value = max(-1.0, min(1.0, value))

    # Find the two surrounding color stops
    for i in range(len(_COLOR_STOPS) - 1):
        v0, c0 = _COLOR_STOPS[i]
        v1, c1 = _COLOR_STOPS[i + 1]
        if value <= v1:
            if v1 == v0:
                t = 0.0
            else:
                t = (value - v0) / (v1 - v0)
            color = (c0 * (1 - t) + c1 * t).astype(np.uint8)
            return tuple(color)

    # Above max
    return tuple(_COLOR_STOPS[-1][1])


def colorize_grid(grid: np.ndarray, no_data: float = -30.0) -> np.ndarray:
    """
    Convert a 2D grid of consistency values to RGBA image array.

    Args:
        grid: 2D numpy array of consistency values (-1 to +1)
        no_data: no-data sentinel value

    Returns:
        3D numpy array (H, W, 4) of uint8 RGBA values
    """
    h, w = grid.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    # Vectorized color mapping using the color stops
    valid_mask = (grid > no_data) & np.isfinite(grid)
    clamped = np.clip(grid, -1.0, 1.0)

    for i in range(len(_COLOR_STOPS) - 1):
        v0, c0 = _COLOR_STOPS[i]
        v1, c1 = _COLOR_STOPS[i + 1]

        if i == 0:
            band_mask = valid_mask & (clamped <= v1)
        elif i == len(_COLOR_STOPS) - 2:
            band_mask = valid_mask & (clamped > v0)
        else:
            band_mask = valid_mask & (clamped > v0) & (clamped <= v1)

        if not np.any(band_mask):
            continue

        if v1 == v0:
            t = np.zeros_like(clamped)
        else:
            t = (clamped - v0) / (v1 - v0)

        t = np.clip(t, 0.0, 1.0)

        for ch in range(4):
            rgba[band_mask, ch] = (
                c0[ch] * (1 - t[band_mask]) + c1[ch] * t[band_mask]
            ).astype(np.uint8)

    return rgba


def render_consistency_tile(
    geotiff,
    z: int,
    x: int,
    y: int,
    tile_size: int = 256,
) -> Optional[bytes]:
    """
    Render a single map tile as PNG bytes.

    Uses simple cylindrical projection with SWIM bounds.
    Tile coordinates follow TMS convention.

    Args:
        geotiff: SwimGeoTIFF instance
        z: Zoom level
        x: Tile X coordinate
        y: Tile Y coordinate
        tile_size: Output tile size in pixels (default 256)

    Returns:
        PNG bytes, or None if geotiff not loaded
    """
    if not geotiff.loaded or geotiff.data is None:
        return None

    # Calculate tile bounds in degrees
    n_tiles = 2 ** z
    lon_per_tile = 360.0 / n_tiles
    lat_per_tile = 180.0 / n_tiles  # simple cylindrical

    tile_west = -180.0 + x * lon_per_tile
    tile_east = tile_west + lon_per_tile
    tile_north = 90.0 - y * lat_per_tile
    tile_south = tile_north - lat_per_tile

    # Check if tile overlaps SWIM region
    if (tile_south >= geotiff.lat_max or tile_north <= geotiff.lat_min or
            tile_east <= geotiff.lon_min or tile_west >= geotiff.lon_max):
        return None  # Outside SWIM coverage

    # Extract subgrid for this tile
    subgrid = geotiff.extract_subgrid(tile_north, tile_south, tile_east, tile_west)
    if subgrid is None or subgrid.size == 0:
        return None

    # Resize subgrid to tile_size using simple nearest-neighbor
    from PIL import Image

    # Colorize the subgrid
    rgba = colorize_grid(subgrid, geotiff.no_data)

    # Convert to PIL and resize
    img = Image.fromarray(rgba, "RGBA")
    if img.size != (tile_size, tile_size):
        img = img.resize((tile_size, tile_size), Image.NEAREST)

    # Encode to PNG
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()
