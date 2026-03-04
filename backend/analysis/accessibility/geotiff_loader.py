"""Generic Mars GeoTIFF loader.

Wraps SwimGeoTIFF to load any Mars-projected GeoTIFF (MOLA derived products,
SWIM consistency, etc.) with lazy caching.
"""

import os
import logging
from typing import Optional, Dict

import numpy as np

from analysis.swim_common.geotiff_loader import SwimGeoTIFF

logger = logging.getLogger(__name__)

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_cache: Dict[str, SwimGeoTIFF] = {}


def load_geotiff(
    filepath: str,
    name: Optional[str] = None,
    no_data: Optional[float] = None,
) -> SwimGeoTIFF:
    """
    Load any GeoTIFF file projected on Mars sphere.

    Args:
        filepath: Absolute path to GeoTIFF file.
        name: Human-readable layer name for logging.
        no_data: Override no-data value (if None, read from file metadata).

    Returns:
        SwimGeoTIFF instance (check .loaded and .error).
    """
    if filepath in _cache:
        return _cache[filepath]

    display_name = name or os.path.basename(filepath)
    geo = SwimGeoTIFF(name=display_name, filepath=filepath)
    if no_data is not None:
        geo.no_data = no_data

    if not os.path.exists(filepath):
        geo.error = f"File not found: {filepath}"
        logger.warning("Data file not found: %s", filepath)
        _cache[filepath] = geo
        return geo

    try:
        import rasterio

        with rasterio.open(filepath) as src:
            data = src.read(1).astype(np.float32)
            geo.data = data
            geo.rows, geo.cols = data.shape

            # Read no-data from file if not overridden
            if no_data is None and src.nodata is not None:
                geo.no_data = float(src.nodata)

            bounds = src.bounds
            _MARS_RADIUS = 3396190.0

            if abs(bounds.left) > 360 or abs(bounds.right) > 360:
                import math
                _deg_factor = _MARS_RADIUS * math.pi / 180.0
                geo.lon_min = bounds.left / _deg_factor
                geo.lon_max = bounds.right / _deg_factor
                geo.lat_min = bounds.bottom / _deg_factor
                geo.lat_max = bounds.top / _deg_factor
                logger.info(
                    "%s: projected CRS, converted to degrees: "
                    "lon=[%.1f,%.1f] lat=[%.1f,%.1f]",
                    display_name,
                    geo.lon_min, geo.lon_max,
                    geo.lat_min, geo.lat_max,
                )
            else:
                geo.lon_min = bounds.left
                geo.lon_max = bounds.right
                geo.lat_min = bounds.bottom
                geo.lat_max = bounds.top

                # Handle 0-360° longitude → shift to -180/180
                if geo.lon_max > 180:
                    geo.lon_min = -180.0
                    geo.lon_max = 180.0
                    mid_col = geo.cols // 2
                    geo.data = np.roll(data, mid_col, axis=1)

            geo.loaded = True
            logger.info(
                "Loaded %s: %dx%d (%.1f MB), "
                "bounds=[%.1f,%.1f,%.1f,%.1f], nodata=%.1f",
                display_name, geo.rows, geo.cols,
                data.nbytes / 1e6,
                geo.lat_min, geo.lat_max,
                geo.lon_min, geo.lon_max,
                geo.no_data,
            )

    except ImportError:
        geo.error = "rasterio not installed"
        logger.error("Cannot load %s: install rasterio", display_name)
    except Exception as e:
        geo.error = f"Failed to load: {e}"
        logger.error("%s load failed: %s", display_name, e)

    _cache[filepath] = geo
    return geo


def get_data_path(*parts: str) -> str:
    """Get absolute path relative to backend/data/ directory."""
    return os.path.join(_BACKEND_DIR, "data", *parts)
