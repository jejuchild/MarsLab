"""
Terrain proxy evidence for ice probability.

Computes slope, roughness, and morphological indicators from
HiRISE DTM or MOLA/HRSC DEM near candidate locations.

Low slopes and low roughness favor mantled/icy terrain.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Optional

import numpy as np

from .models import E3Terrain

logger = logging.getLogger(__name__)

R_MARS = 3_389_500.0  # metres


def evaluate_terrain_evidence(
    candidate_lat: float,
    candidate_lon: float,
    hirise_dtm_products: Optional[list] = None,
    analysis_radius_m: float = 2000.0,
) -> E3Terrain:
    """Evaluate terrain proxy evidence near a candidate location.

    Tries HiRISE DTM first, then MOLA/HRSC DEM fallback.

    Args:
        candidate_lat/lon: Location
        hirise_dtm_products: Specific DTM product IDs; if None, search index
        analysis_radius_m: Radius around candidate for analysis

    Returns:
        E3Terrain evidence score
    """
    # Try HiRISE DTM
    result = _try_hirise_dtm(candidate_lat, candidate_lon, hirise_dtm_products, analysis_radius_m)
    if result is not None:
        return result

    # Fallback to MOLA DEM
    result = _try_mola_dem(candidate_lat, candidate_lon, analysis_radius_m)
    if result is not None:
        return result

    return E3Terrain(
        score=0.5,  # neutral when no data
        notes="No terrain data available — score neutral (0.5)",
    )


def _try_hirise_dtm(
    lat: float, lon: float,
    products: Optional[list],
    radius_m: float,
) -> Optional[E3Terrain]:
    """Compute terrain stats from HiRISE DTM."""
    import json

    backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dtm_dir = os.path.join(backend_dir, "hirise_dtm_data")
    index_path = os.path.join(dtm_dir, "index.geojson")

    if not os.path.exists(index_path):
        return None

    with open(index_path, "r") as f:
        index = json.load(f)

    # Find DTM covering candidate
    dtm_file = None
    for feat in index.get("features", []):
        props = feat.get("properties", {})
        pid = props.get("product_id", "")
        if products and pid not in products:
            continue
        # Check bbox
        geom = feat.get("geometry", {})
        if geom.get("type") == "Polygon":
            coords = geom["coordinates"][0]
            lons = [c[0] for c in coords]
            lats = [c[1] for c in coords]
            if min(lats) <= lat <= max(lats) and min(lons) <= lon <= max(lons):
                df = props.get("dtm_file")
                if df:
                    full_path = os.path.join(dtm_dir, df)
                    if os.path.exists(full_path):
                        dtm_file = full_path
                        break

    if not dtm_file:
        return None

    try:
        import rasterio
        from analysis.epsilon_terrace.terrace_detect import latlon_to_pixel

        ds = rasterio.open(dtm_file)
        elev = ds.read(1).astype(np.float64)
        res_m = abs(ds.res[0])

        if ds.nodata is not None:
            elev[elev == ds.nodata] = np.nan

        # Get pixel center
        row, col = latlon_to_pixel(ds, lat, lon)
        ds.close()

        # Extract patch
        r_px = max(10, int(radius_m / res_m))
        r0 = max(0, row - r_px)
        r1 = min(elev.shape[0], row + r_px)
        c0 = max(0, col - r_px)
        c1 = min(elev.shape[1], col + r_px)

        patch = elev[r0:r1, c0:c1]
        valid = ~np.isnan(patch)
        if valid.sum() < 50:
            return None

        # Compute slopes
        dy, dx = np.gradient(patch, res_m)
        slope_rad = np.arctan(np.sqrt(dx ** 2 + dy ** 2))
        slope_deg = np.degrees(slope_rad)
        slope_valid = slope_deg[valid[:-1, :-1] if slope_deg.shape != patch.shape else valid]

        if len(slope_valid) == 0:
            return None

        mean_slope = float(np.nanmean(slope_deg))
        p95_slope = float(np.nanpercentile(slope_deg, 95))
        roughness = float(np.nanstd(slope_deg))

        return _score_terrain(mean_slope, p95_slope, roughness, "HiRISE DTM")

    except Exception as e:
        logger.warning(f"HiRISE DTM analysis failed: {e}")
        return None


def _try_mola_dem(lat: float, lon: float, radius_m: float) -> Optional[E3Terrain]:
    """Fallback to MOLA/HRSC global DEM if available."""
    backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Check for MOLA DEM file
    mola_candidates = [
        os.path.join(backend_dir, "Mars_HRSC_MOLA_BlendDEM_Global_200mp_v2.tif"),
        os.path.join(os.path.dirname(backend_dir), "Mars_HRSC_MOLA_BlendDEM_Global_200mp_v2.tif"),
    ]

    dem_path = None
    for p in mola_candidates:
        if os.path.exists(p):
            dem_path = p
            break

    if not dem_path:
        return None

    try:
        import rasterio
        from rasterio.windows import from_bounds

        ds = rasterio.open(dem_path)

        # Convert to DEM CRS (likely simple lon/lat)
        deg_per_m = 1.0 / (R_MARS * math.pi / 180)
        buf_deg = radius_m * deg_per_m

        window = from_bounds(
            lon - buf_deg, lat - buf_deg,
            lon + buf_deg, lat + buf_deg,
            transform=ds.transform,
        )

        elev = ds.read(1, window=window, boundless=True, fill_value=0).astype(np.float64)
        res_m = abs(ds.res[0]) * R_MARS * math.pi / 180
        ds.close()

        if elev.size < 25 or np.all(elev == 0):
            return None

        dy, dx = np.gradient(elev, res_m)
        slope_deg = np.degrees(np.arctan(np.sqrt(dx ** 2 + dy ** 2)))
        mean_slope = float(np.nanmean(slope_deg))
        p95_slope = float(np.nanpercentile(slope_deg, 95))
        roughness = float(np.nanstd(slope_deg))

        return _score_terrain(mean_slope, p95_slope, roughness, "MOLA/HRSC DEM")

    except Exception as e:
        logger.warning(f"MOLA DEM analysis failed: {e}")
        return None


def _score_terrain(
    mean_slope: float,
    p95_slope: float,
    roughness: float,
    source: str,
) -> E3Terrain:
    """Convert terrain stats to a 0-1 score.

    Lower slopes and roughness → higher score (favors icy mantled plains).
    """
    # Slope score: 0 deg → 1.0, ≥15 deg → 0.0
    slope_score = max(0.0, 1.0 - mean_slope / 15.0)

    # Roughness score: std < 2 → 1.0, std ≥ 10 → 0.0
    rough_score = max(0.0, 1.0 - roughness / 10.0)

    # Combined (slope dominates)
    score = 0.7 * slope_score + 0.3 * rough_score

    notes = (
        f"Terrain from {source}: mean slope={mean_slope:.1f}°, "
        f"P95={p95_slope:.1f}°, roughness σ={roughness:.2f}°. "
    )

    if mean_slope < 5 and roughness < 3:
        notes += "Smooth, low-slope terrain — consistent with mantled icy plains."
    elif mean_slope < 10:
        notes += "Moderate terrain — plausible for ice deposits."
    else:
        notes += "Steep/rough terrain — less favorable for shallow ice."

    return E3Terrain(
        score=round(score, 3),
        slope_stats={
            "mean_slope_deg": round(mean_slope, 2),
            "p95_slope_deg": round(p95_slope, 2),
        },
        roughness_proxy=round(roughness, 3),
        notes=notes,
    )
