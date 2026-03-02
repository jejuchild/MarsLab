"""
swim_common — Shared utilities for SWIM (Subsurface Water Ice Mapping) modules.

Provides GeoTIFF loading, tile rendering, and coordinate conversion
shared across all SWIM analysis modules (neutron, thermal, radar, geomorphic, fusion).
"""

from .geotiff_loader import SwimGeoTIFF, load_swim_geotiff
from .tile_renderer import render_consistency_tile, SWIM_COLORMAP
from .coord_utils import lon_360_to_180, lon_180_to_360, validate_swim_bounds, clamp_to_swim_region

__all__ = [
    "SwimGeoTIFF",
    "load_swim_geotiff",
    "render_consistency_tile",
    "SWIM_COLORMAP",
    "lon_360_to_180",
    "lon_180_to_360",
    "validate_swim_bounds",
    "clamp_to_swim_region",
]
