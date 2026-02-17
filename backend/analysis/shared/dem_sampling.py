"""DEM elevation sampling along SHARAD tracks.

Consolidates MOLA + HiRISE DTM sampling from RTE pipeline and
sharad_report into a single reusable module.
"""

import logging
import os
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

# __file__ = backend/analysis/shared/dem_sampling.py
# -> analysis/ -> backend/
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def sample_dem_along_track(
    lats_centric: np.ndarray,
    lons_180: np.ndarray,
    dtm_product_id: str = "",
) -> tuple:
    """Sample DEM elevation along a SHARAD track.

    Tries HiRISE DTM first (if dtm_product_id is provided and covers >30%
    of the track), then falls back to MOLA 200m global DEM.

    Args:
        lats_centric: Planetocentric latitudes (degrees)
        lons_180: Longitudes in -180..180 (degrees)
        dtm_product_id: Optional HiRISE DTM product ID

    Returns:
        (source_name, elevations): ("MOLA" or "HiRISE_DTM", ndarray of floats)
    """
    n = len(lats_centric)
    elevations = np.full(n, np.nan, dtype=np.float64)

    # Try HiRISE DTM if specified
    if dtm_product_id:
        try:
            dtm_elevs = _load_dtm_along_track(dtm_product_id, lats_centric, lons_180)
            if dtm_elevs is not None:
                valid = ~np.isnan(dtm_elevs)
                if valid.sum() > n * 0.3:
                    return "HiRISE_DTM", dtm_elevs
        except Exception:
            logger.debug("HiRISE DTM fallback to MOLA: %s", dtm_product_id)

    # MOLA fallback (always available)
    try:
        elevations = _sample_mola(lats_centric, lons_180)
    except Exception:
        logger.warning("MOLA DEM sampling failed; elevations will be NaN")

    return "MOLA", elevations


def _sample_mola(lats_centric: np.ndarray, lons_180: np.ndarray) -> np.ndarray:
    """Sample MOLA DEM elevation at given coordinates."""
    from api.sharad_highres_router import _get_dem
    from rasterio.windows import Window

    n = len(lats_centric)
    elevations = np.full(n, np.nan, dtype=np.float64)

    ds = _get_dem()
    inv_transform = ~ds.transform
    cols, rows = inv_transform * (lons_180, lats_centric)
    cols = np.round(cols).astype(int)
    rows = np.round(rows).astype(int)
    cols = np.clip(cols, 0, ds.width - 1)
    rows = np.clip(rows, 0, ds.height - 1)

    row_min, row_max = int(rows.min()), int(rows.max())
    col_min, col_max = int(cols.min()), int(cols.max())
    window = Window(col_min, row_min,
                    col_max - col_min + 1, row_max - row_min + 1)
    block = ds.read(1, window=window).astype(np.float64)
    if ds.nodata is not None:
        block[block == ds.nodata] = np.nan

    local_rows = rows - row_min
    local_cols = cols - col_min
    elevations = block[local_rows, local_cols]
    return elevations


def _load_dtm_along_track(
    dtm_product_id: str,
    lats: np.ndarray,
    lons: np.ndarray,
) -> Optional[np.ndarray]:
    """Extract HiRISE DTM elevation along track coordinates."""
    import glob
    import rasterio
    from pyproj import Transformer

    dtm_dir = os.path.join(_BACKEND_DIR, "hirise_dtm_data")
    if not os.path.isdir(dtm_dir):
        return None

    pattern = os.path.join(dtm_dir, f"*{dtm_product_id}*")
    matches = [f for f in glob.glob(pattern) if f.upper().endswith(".IMG")]
    if not matches:
        return None

    try:
        with rasterio.open(matches[0]) as ds:
            crs = ds.crs
            transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
            xs, ys = transformer.transform(lons, lats)

            elevations = np.full(len(lats), np.nan)
            for i in range(len(lats)):
                try:
                    row, col = ds.index(xs[i], ys[i])
                    if 0 <= row < ds.height and 0 <= col < ds.width:
                        val = ds.read(1, window=rasterio.windows.Window(col, row, 1, 1))
                        elevations[i] = float(val[0, 0])
                except Exception:
                    continue

            nodata = ds.nodata
            if nodata is not None:
                elevations[elevations == nodata] = np.nan
            return elevations
    except Exception as e:
        logger.warning("DTM loading failed: %s", e)
        return None
