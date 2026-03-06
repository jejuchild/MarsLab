"""Accessibility pipeline — orchestrates TES + MOLA data layers.

Lazily loads MOLA derived products and TES thermal inertia, then delegates
scoring to *algorithm.py* and tile rendering to *tile_renderer.py*.

No SWIM data — ice presence is handled by the landform classifier.
"""

import logging
from typing import Dict, Optional

import numpy as np

from .algorithm import (
    AccessibilityResult,
    compute_accessibility,
    compute_accessibility_grid,
)
from .geotiff_loader import get_data_path, load_geotiff
from .tile_renderer import render_accessibility_tile

logger = logging.getLogger(__name__)


def _resample_to_target(
    source: np.ndarray,
    src_lat_min: float,
    src_lat_max: float,
    src_lon_min: float,
    src_lon_max: float,
    target_lats: np.ndarray,
    target_lons: np.ndarray,
    no_data: float = -9999.0,
) -> np.ndarray:
    """Nearest-neighbour resample a 2D grid to target lat/lon vectors."""
    sh, sw = source.shape
    ppd_lat = sh / (src_lat_max - src_lat_min)
    ppd_lon = sw / (src_lon_max - src_lon_min)

    rows = ((src_lat_max - target_lats) * ppd_lat).astype(int)
    cols = ((target_lons - src_lon_min) * ppd_lon).astype(int)

    rows = np.clip(rows, 0, sh - 1)
    cols = np.clip(cols, 0, sw - 1)

    result = source[np.ix_(rows, cols)].astype(np.float32)

    # Mark out-of-bounds as NaN
    lat_ok = (target_lats >= src_lat_min) & (target_lats <= src_lat_max)
    lon_ok = (target_lons >= src_lon_min) & (target_lons <= src_lon_max)
    mask = np.outer(lat_ok, lon_ok)
    result[~mask] = np.nan

    # Mark no-data
    if no_data is not None:
        result[result <= no_data] = np.nan

    return result


class AccessibilityPipeline:
    """Load TES + MOLA layers and compute accessibility scores."""

    # Target grid: 16 ppd matching MOLA derived products
    TARGET_H = 2880
    TARGET_W = 5760
    TARGET_LAT_MIN = -90.0
    TARGET_LAT_MAX = 90.0
    TARGET_LON_MIN = -180.0
    TARGET_LON_MAX = 180.0

    def __init__(self) -> None:
        # MOLA derived (GeoTIFF, 5760×2880)
        self._mola_elev = None
        self._mola_slope = None
        self._mola_tri = None

        # TES Thermal Inertia (numpy, 7200×3600, 0-360°E)
        self._tes_ti_grid: Optional[np.ndarray] = None

        # Pre-computed accessibility grid (lazy)
        self._cached_grid: Optional[np.ndarray] = None

        self._loaded = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True

        # MOLA derived products
        self._mola_elev = load_geotiff(
            get_data_path("mola_derived", "mola_elevation_3km.tif"),
            name="MOLA Elevation",
        )
        self._mola_slope = load_geotiff(
            get_data_path("mola_derived", "mola_slope_3km.tif"),
            name="MOLA Slope",
        )
        self._mola_tri = load_geotiff(
            get_data_path("mola_derived", "mola_tri_3km.tif"),
            name="MOLA TRI",
        )

        # TES Thermal Inertia (existing .npy)
        self._load_tes_ti()

        loaded_layers = sum(1 for g in [
            self._mola_elev, self._mola_slope, self._mola_tri,
        ] if g is not None and g.loaded)
        ti_ok = self._tes_ti_grid is not None

        logger.info(
            "Accessibility pipeline loaded: %d MOLA GeoTIFFs + TES TI=%s",
            loaded_layers, "ok" if ti_ok else "missing",
        )

    def _load_tes_ti(self) -> None:
        """Load TES thermal inertia from existing numpy file."""
        import os
        npy_path = get_data_path("tes_thermal_inertia.npy")
        if not os.path.exists(npy_path):
            logger.warning("TES TI numpy not found: %s", npy_path)
            return
        try:
            grid = np.load(npy_path).astype(np.float32)
            self._tes_ti_grid = grid
            logger.info("Loaded TES TI: shape=%s", grid.shape)
        except Exception as e:
            logger.error("Failed to load TES TI: %s", e)

    # ------------------------------------------------------------------
    # Point query
    # ------------------------------------------------------------------

    def query_point(
        self,
        lat: float,
        lon: float,
        weights: Optional[Dict[str, float]] = None,
        landform: Optional[str] = None,
        landform_confidence: float = 1.0,
        water_mineral_score: Optional[float] = None,
        surface_ice_score: Optional[float] = None,
        crism_obs_id: str = "",
        crism_minerals: Optional[Dict[str, float]] = None,
    ) -> AccessibilityResult:
        """Compute ISRU accessibility at a single point."""
        self._ensure_loaded()

        # Sample each layer
        ti = self._sample_tes_ti(lat, lon)
        elev = self._sample_geotiff(self._mola_elev, lat, lon)
        slope = self._sample_geotiff(self._mola_slope, lat, lon)
        tri = self._sample_geotiff(self._mola_tri, lat, lon)

        return compute_accessibility(
            thermal_inertia=ti,
            elevation=elev,
            slope=slope,
            tri=tri,
            lat=lat,
            lon=lon,
            weights=weights,
            landform=landform,
            landform_confidence=landform_confidence,
            water_mineral_score=water_mineral_score,
            surface_ice_score=surface_ice_score,
            crism_obs_id=crism_obs_id,
            crism_minerals=crism_minerals,
        )

    @staticmethod
    def _sample_geotiff(geo, lat: float, lon: float) -> Optional[float]:
        if geo is None or not geo.loaded:
            return None
        return geo.sample_point(lat, lon)

    def _sample_tes_ti(self, lat: float, lon: float) -> Optional[float]:
        """Sample TES TI at (lat, lon). TES grid is 7200×3600, 0-360°E."""
        if self._tes_ti_grid is None:
            return None

        if lat < -60.0 or lat > 60.0:
            return None

        lon360 = lon % 360
        row = int((90.0 - lat) * 20)  # 20 ppd
        col = int(lon360 * 20)
        row = max(0, min(row, self._tes_ti_grid.shape[0] - 1))
        col = max(0, min(col, self._tes_ti_grid.shape[1] - 1))

        val = float(self._tes_ti_grid[row, col])
        if not np.isfinite(val) or val <= 0 or val > 2000:
            return None
        return val

    # ------------------------------------------------------------------
    # Tile rendering
    # ------------------------------------------------------------------

    def get_tile(
        self,
        z: int,
        x: int,
        y: int,
        tile_size: int = 256,
        weights: Optional[Dict[str, float]] = None,
    ) -> Optional[bytes]:
        """Render an accessibility heatmap tile."""
        self._ensure_loaded()

        # If using default weights, use cached global grid
        if weights is None:
            return self._tile_from_cached(z, x, y, tile_size)

        # Custom weights: compute on the fly for tile region
        return self._tile_on_the_fly(z, x, y, tile_size, weights)

    def _tile_from_cached(
        self, z: int, x: int, y: int, tile_size: int,
    ) -> Optional[bytes]:
        """Serve tile from pre-computed global accessibility grid."""
        if self._cached_grid is None:
            self._precompute_grid()
        if self._cached_grid is None:
            return None

        # Tile bounds
        n_tiles = 2 ** z
        lon_per = 360.0 / n_tiles
        lat_per = 180.0 / n_tiles
        tile_west = -180.0 + x * lon_per
        tile_east = tile_west + lon_per
        tile_north = 90.0 - y * lat_per
        tile_south = tile_north - lat_per

        # Pixel range in cached grid
        ppd_lat = self.TARGET_H / 180.0
        ppd_lon = self.TARGET_W / 360.0
        r0 = int((90.0 - tile_north) * ppd_lat)
        r1 = int((90.0 - tile_south) * ppd_lat)
        c0 = int((tile_west + 180.0) * ppd_lon)
        c1 = int((tile_east + 180.0) * ppd_lon)
        r0 = max(0, min(r0, self.TARGET_H - 1))
        r1 = max(r0 + 1, min(r1, self.TARGET_H))
        c0 = max(0, min(c0, self.TARGET_W - 1))
        c1 = max(c0 + 1, min(c1, self.TARGET_W))

        sub = self._cached_grid[r0:r1, c0:c1]
        if sub.size == 0:
            return None

        # Check if all NaN
        if not np.any(np.isfinite(sub)):
            return None

        return render_accessibility_tile(sub, tile_size)

    def _tile_on_the_fly(
        self, z: int, x: int, y: int, tile_size: int,
        weights: Dict[str, float],
    ) -> Optional[bytes]:
        """Compute tile with custom weights on-the-fly."""
        n_tiles = 2 ** z
        lon_per = 360.0 / n_tiles
        lat_per = 180.0 / n_tiles
        tile_west = -180.0 + x * lon_per
        tile_east = tile_west + lon_per
        tile_north = 90.0 - y * lat_per
        tile_south = tile_north - lat_per

        lats = np.linspace(tile_north, tile_south, tile_size)
        lons = np.linspace(tile_west, tile_east, tile_size)

        layers = self._extract_layers(lats, lons, (tile_size, tile_size))
        score = compute_accessibility_grid(**layers, shape=(tile_size, tile_size), weights=weights)

        if not np.any(np.isfinite(score)):
            return None

        return render_accessibility_tile(score, tile_size)

    # ------------------------------------------------------------------
    # Grid pre-computation
    # ------------------------------------------------------------------

    def _precompute_grid(self) -> None:
        """Compute global accessibility at 16 ppd."""
        logger.info("Pre-computing accessibility grid (%dx%d)...", self.TARGET_H, self.TARGET_W)

        lats = np.linspace(self.TARGET_LAT_MAX, self.TARGET_LAT_MIN, self.TARGET_H)
        lons = np.linspace(self.TARGET_LON_MIN, self.TARGET_LON_MAX, self.TARGET_W)

        layers = self._extract_layers(lats, lons, (self.TARGET_H, self.TARGET_W))
        self._cached_grid = compute_accessibility_grid(
            **layers, shape=(self.TARGET_H, self.TARGET_W),
        )

        valid = np.isfinite(self._cached_grid)
        logger.info(
            "Accessibility grid computed: valid=%.1f%%, mean=%.3f",
            100 * valid.sum() / valid.size,
            float(np.nanmean(self._cached_grid)) if valid.any() else 0,
        )

    def _extract_layers(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        shape: tuple[int, int],
    ) -> Dict[str, Optional[np.ndarray]]:
        """Extract TES + MOLA layers resampled to target lat/lon grid."""

        def _from_geo(geo) -> Optional[np.ndarray]:
            if geo is None or not geo.loaded or geo.data is None:
                return None
            return _resample_to_target(
                geo.data, geo.lat_min, geo.lat_max, geo.lon_min, geo.lon_max,
                lats, lons, no_data=geo.no_data,
            )

        # TES TI: 7200×3600, 0-360°E → need to handle lon mapping
        tes_ti = None
        if self._tes_ti_grid is not None:
            # Convert target lons (-180..180) to 0-360 for TES
            lons360 = lons % 360
            tes_ti = _resample_to_target(
                self._tes_ti_grid,
                src_lat_min=-90.0, src_lat_max=90.0,
                src_lon_min=0.0, src_lon_max=360.0,
                target_lats=lats, target_lons=lons360,
                no_data=0.0,
            )
            # Mask values outside ±60° (TES coverage)
            lat_mask = (lats >= -60.0) & (lats <= 60.0)
            tes_ti[~lat_mask[:, None].repeat(shape[1], axis=1)] = np.nan

        return {
            "thermal_inertia": tes_ti,
            "elevation": _from_geo(self._mola_elev),
            "slope": _from_geo(self._mola_slope),
            "tri": _from_geo(self._mola_tri),
        }

    # ------------------------------------------------------------------
    # Individual layer tiles (for frontend toggles)
    # ------------------------------------------------------------------

    def get_layer_tile(
        self,
        layer: str,
        z: int, x: int, y: int,
        tile_size: int = 256,
    ) -> Optional[bytes]:
        """Render a single data layer as a tile (for debugging / layer panel)."""
        self._ensure_loaded()

        from analysis.swim_common.tile_renderer import render_consistency_tile

        geo_map = {
            "mola_elevation": self._mola_elev,
            "mola_slope": self._mola_slope,
            "mola_tri": self._mola_tri,
        }

        geo = geo_map.get(layer)
        if geo is not None and geo.loaded:
            return render_consistency_tile(geo, z=z, x=x, y=y, tile_size=tile_size)

        return None
