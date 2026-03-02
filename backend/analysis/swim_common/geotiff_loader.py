"""
GeoTIFF loader for SWIM products.

Loads SWIM GeoTIFF files (simple cylindrical, Mars sphere, 0-360°E or -180-180°E)
and provides efficient point sampling and region extraction.

All SWIM products share:
  - Pixel scale: ~3 km/pixel
  - No-data value: -30
  - Latitude range: -60° to +60°
  - Projection: simple cylindrical on Mars sphere
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# SWIM constants
SWIM_NO_DATA = -30.0
SWIM_LAT_MIN = -60.0
SWIM_LAT_MAX = 60.0
SWIM_LON_MIN = -180.0
SWIM_LON_MAX = 180.0

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SWIM_DATA_DIR = os.path.join(_BACKEND_DIR, "data", "swim")


@dataclass
class SwimGeoTIFF:
    """Loaded SWIM GeoTIFF with lazy numpy array and metadata."""

    name: str
    filepath: str
    data: Optional[np.ndarray] = field(default=None, repr=False)
    rows: int = 0
    cols: int = 0
    lat_min: float = SWIM_LAT_MIN
    lat_max: float = SWIM_LAT_MAX
    lon_min: float = SWIM_LON_MIN
    lon_max: float = SWIM_LON_MAX
    no_data: float = SWIM_NO_DATA
    loaded: bool = False
    error: Optional[str] = None

    @property
    def ppd_lat(self) -> float:
        """Pixels per degree latitude."""
        if self.rows == 0:
            return 0.0
        return self.rows / (self.lat_max - self.lat_min)

    @property
    def ppd_lon(self) -> float:
        """Pixels per degree longitude."""
        if self.cols == 0:
            return 0.0
        return self.cols / (self.lon_max - self.lon_min)

    def sample_point(self, lat: float, lon: float) -> Optional[float]:
        """
        Sample value at (lat, lon). Returns None if no-data or out of bounds.

        Args:
            lat: Latitude in degrees (-90 to 90)
            lon: Longitude in degrees (-180 to 180)
        """
        if self.data is None:
            return None

        if lat < self.lat_min or lat > self.lat_max:
            return None
        if lon < self.lon_min or lon > self.lon_max:
            return None

        # Convert to pixel coordinates
        # Row 0 = lat_max (top), last row = lat_min (bottom)
        row = int((self.lat_max - lat) * self.ppd_lat)
        col = int((lon - self.lon_min) * self.ppd_lon)

        # Clamp
        row = max(0, min(row, self.rows - 1))
        col = max(0, min(col, self.cols - 1))

        val = float(self.data[row, col])

        # Filter no-data
        if val <= self.no_data or np.isnan(val):
            return None

        return val

    def sample_region(
        self, north: float, south: float, east: float, west: float
    ) -> Dict:
        """
        Extract statistics for a bounding box region.

        Returns dict with stats (mean, std, min, max, coverage_pct)
        or error info if unavailable.
        """
        if self.data is None:
            return {"available": False, "error": f"{self.name} data not loaded"}

        # Clamp to SWIM coverage
        eff_north = min(north, self.lat_max)
        eff_south = max(south, self.lat_min)
        eff_west = max(west, self.lon_min)
        eff_east = min(east, self.lon_max)

        if eff_south >= eff_north or eff_west >= eff_east:
            return {"available": False, "error": "Region outside SWIM coverage"}

        # Pixel range
        row_start = int((self.lat_max - eff_north) * self.ppd_lat)
        row_end = int((self.lat_max - eff_south) * self.ppd_lat)
        col_start = int((eff_west - self.lon_min) * self.ppd_lon)
        col_end = int((eff_east - self.lon_min) * self.ppd_lon)

        row_start = max(0, min(row_start, self.rows - 1))
        row_end = max(0, min(row_end, self.rows))
        col_start = max(0, min(col_start, self.cols - 1))
        col_end = max(0, min(col_end, self.cols))

        if row_end <= row_start or col_end <= col_start:
            return {"available": False, "error": "Region too small"}

        sub = self.data[row_start:row_end, col_start:col_end]
        values = sub.flatten()
        valid = values[(values > self.no_data) & np.isfinite(values)]

        if len(valid) == 0:
            return {"available": False, "error": "No valid data in region"}

        total_pixels = len(values)
        valid_pixels = len(valid)

        return {
            "available": True,
            "stats": {
                "mean": round(float(np.mean(valid)), 4),
                "std": round(float(np.std(valid)), 4),
                "min": round(float(np.min(valid)), 4),
                "max": round(float(np.max(valid)), 4),
                "coverage_pct": round(100.0 * valid_pixels / total_pixels, 1),
            },
            "bounds": {
                "north": eff_north,
                "south": eff_south,
                "east": eff_east,
                "west": eff_west,
            },
        }

    def extract_subgrid(
        self, north: float, south: float, east: float, west: float
    ) -> Optional[np.ndarray]:
        """Extract raw numpy subgrid for a region. Returns None if unavailable."""
        if self.data is None:
            return None

        eff_north = min(north, self.lat_max)
        eff_south = max(south, self.lat_min)
        eff_west = max(west, self.lon_min)
        eff_east = min(east, self.lon_max)

        if eff_south >= eff_north or eff_west >= eff_east:
            return None

        row_start = int((self.lat_max - eff_north) * self.ppd_lat)
        row_end = int((self.lat_max - eff_south) * self.ppd_lat)
        col_start = int((eff_west - self.lon_min) * self.ppd_lon)
        col_end = int((eff_east - self.lon_min) * self.ppd_lon)

        row_start = max(0, min(row_start, self.rows - 1))
        row_end = max(0, min(row_end, self.rows))
        col_start = max(0, min(col_start, self.cols - 1))
        col_end = max(0, min(col_end, self.cols))

        return self.data[row_start:row_end, col_start:col_end]


# Cache of loaded GeoTIFFs
_geotiff_cache: Dict[str, SwimGeoTIFF] = {}


def load_swim_geotiff(filename: str, name: Optional[str] = None) -> SwimGeoTIFF:
    """
    Load a SWIM GeoTIFF file from the SWIM data directory.

    Uses rasterio if available, falls back to GDAL, then to raw numpy.
    Results are cached — subsequent calls return the same instance.

    Args:
        filename: GeoTIFF filename (relative to backend/data/swim/)
        name: Human-readable name for logging

    Returns:
        SwimGeoTIFF instance (check .loaded and .error)
    """
    if filename in _geotiff_cache:
        return _geotiff_cache[filename]

    filepath = os.path.join(SWIM_DATA_DIR, filename)
    display_name = name or filename

    geo = SwimGeoTIFF(name=display_name, filepath=filepath)

    if not os.path.exists(filepath):
        geo.error = f"File not found: {filepath}"
        logger.warning("SWIM data not found: %s — run scripts/download_swim_data.py", filepath)
        _geotiff_cache[filename] = geo
        return geo

    try:
        # Try rasterio first (best GeoTIFF support)
        import rasterio

        with rasterio.open(filepath) as src:
            data = src.read(1).astype(np.float32)
            geo.data = data
            geo.rows, geo.cols = data.shape

            # Extract bounds from transform
            bounds = src.bounds
            geo.lon_min = bounds.left
            geo.lon_max = bounds.right
            geo.lat_min = bounds.bottom
            geo.lat_max = bounds.top

            # If 0-360, convert to -180/180
            if geo.lon_max > 180:
                geo.lon_min = -180.0
                geo.lon_max = 180.0
                # Shift data: left half = 180-360, right half = 0-180
                mid_col = geo.cols // 2
                geo.data = np.roll(data, mid_col, axis=1)

            geo.loaded = True
            logger.info(
                "Loaded SWIM %s: %dx%d, bounds=[%.1f,%.1f,%.1f,%.1f]",
                display_name, geo.rows, geo.cols,
                geo.lat_min, geo.lat_max, geo.lon_min, geo.lon_max,
            )

    except ImportError:
        # Fallback: try loading as raw numpy (for pre-converted .npy files)
        npy_path = filepath.replace(".tif", ".npy")
        if os.path.exists(npy_path):
            try:
                geo.data = np.load(npy_path).astype(np.float32)
                geo.rows, geo.cols = geo.data.shape
                geo.loaded = True
                logger.info("Loaded SWIM %s from .npy: %dx%d", display_name, geo.rows, geo.cols)
            except Exception as e:
                geo.error = f"Failed to load .npy fallback: {e}"
                logger.error("SWIM %s .npy load failed: %s", display_name, e)
        else:
            geo.error = "rasterio not installed and no .npy fallback available"
            logger.error("Cannot load SWIM %s: install rasterio or provide .npy", display_name)

    except Exception as e:
        geo.error = f"Failed to load: {e}"
        logger.error("SWIM %s load failed: %s", display_name, e)

    _geotiff_cache[filename] = geo
    return geo
