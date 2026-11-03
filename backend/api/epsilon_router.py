"""FastAPI router for epsilon (εr) inversion endpoints."""

import logging
from typing import Optional

from fastapi import APIRouter, Query

from analysis.epsilon_inversion.near_crater import NearCraterEpsilonEstimator
from analysis.epsilon_inversion.hyperbola import HyperbolaFitter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/epsilon", tags=["epsilon_inversion"])


@router.get("/near_crater")
async def near_crater_inversion(
    lat: float = Query(..., description="Crater center latitude (planetocentric)"),
    lon: float = Query(..., description="Crater center longitude (-180..180)"),
    depth_m: float = Query(..., description="Crater depth in meters"),
    max_distance_km: float = Query(10.0, ge=1.0, le=100.0, description="Max distance from crater (km)"),
    sigma_km: float = Query(3.0, ge=0.5, le=20.0, description="Gaussian weight sigma (km)"),
    min_conf: float = Query(0.6, ge=0.0, le=1.0, description="Minimum confidence threshold"),
) -> dict:
    """Estimate εr from SHARAD tracks passing near a crater.

    Uses distance-weighted triangulation from nearby track detections to infer
    the regolith dielectric constant at the crater location.

    Args:
        lat: Crater center latitude
        lon: Crater center longitude
        depth_m: Known crater depth
        max_distance_km: Maximum distance to consider
        sigma_km: Gaussian weight falloff distance
        min_conf: Minimum detection confidence

    Returns:
        epsilon_r_est: Estimated εr (or null if no candidates)
        epsilon_r_std: Standard deviation in estimate
        n_used: Number of contributing traces
        candidates: List of candidate tracks with distances and weights
        quality_note: Quality assessment
    """
    try:
        logger.info(
            "Near-crater εr estimation: lat=%.2f lon=%.2f depth=%.0f m",
            lat, lon, depth_m
        )

        # Placeholder: would load candidate product IDs from spatial index
        candidate_product_ids = []  # TODO: Query spatial index for nearby products

        result = NearCraterEpsilonEstimator.estimate(
            crater_lat=lat,
            crater_lon=lon,
            crater_depth_m=depth_m,
            candidate_product_ids=candidate_product_ids,
            max_distance_km=max_distance_km,
            distance_weight_sigma_km=sigma_km,
            min_confidence=min_conf,
        )

        return {
            "success": True,
            "epsilon_r_est": result.epsilon_r_est,
            "epsilon_r_std": result.epsilon_r_std,
            "n_used": result.n_used,
            "candidates": result.candidates,
            "quality_note": result.quality_note,
        }

    except Exception as e:
        logger.exception("Near-crater inversion failed")
        return {
            "success": False,
            "error": str(e),
            "epsilon_r_est": None,
            "epsilon_r_std": None,
            "n_used": 0,
            "candidates": [],
            "quality_note": f"Error: {str(e)}",
        }


@router.get("/hyperbola_fit")
def hyperbola_fit(
    product_id: str = Query(..., description="SHARAD product ID"),
    trace_idx0: int = Query(..., ge=0, description="Apex trace index (FULL resolution)"),
    bin_idx0: int = Query(..., ge=0, description="Apex range bin"),
    window_traces: int = Query(300, ge=20, le=6000, description="Window width in full-res traces"),
    window_bins: int = Query(120, ge=20, le=600, description="Window height in bins"),
) -> dict:
    """EXPERIMENTAL: estimate εr from a diffraction hyperbola.

    Fits t(x) = √(t0² + (2(x−x0)/v)²) with t measured below the surface echo
    and x from the RDR geometry; εr = (c/v)².  Poorly constrained for focused
    orbital RDRs (see ``flags`` / ``quality_note``).  Trace indices are full
    resolution (display trace × downsample).
    """
    try:
        logger.info(
            "Hyperbola fit: product=%s center=[%d, %d] window=(%d, %d)",
            product_id, trace_idx0, bin_idx0, window_traces, window_bins
        )

        fitter = HyperbolaFitter()
        result = fitter.fit(
            product_id=product_id,
            trace_idx0=trace_idx0,
            bin_idx0=bin_idx0,
            window_traces=window_traces,
            window_bins=window_bins,
        )

        import math as _m
        return {
            "success": result.epsilon_r_est is not None,
            "experimental": True,
            "epsilon_r_est": result.epsilon_r_est,
            "epsilon_r_std": result.epsilon_r_std,
            "epsilon_r_ci95": list(result.epsilon_r_ci95) if result.epsilon_r_ci95 else None,
            "depth_m": result.depth_m,
            "v_ms": result.v_ms,
            "rmse_bins": None if _m.isnan(result.rmse_bins) else round(result.rmse_bins, 3),
            "inlier_ratio": round(result.inlier_ratio, 3),
            "n_inliers": int(result.n_inliers),
            "quality_note": result.quality_note,
            "flags": result.flags,
            "picks": result.picks,
            "overlay": result.overlay,
            "diagnostics": result.diagnostics,
        }

    except Exception as e:
        logger.exception("Hyperbola fitting failed")
        return {
            "success": False,
            "error": str(e),
            "epsilon_r_est": None,
            "epsilon_r_std": None,
            "rmse_bins": None,
            "inlier_ratio": None,
            "n_inliers": 0,
            "quality_note": f"Error: {str(e)}",
            "diagnostics": {},
        }
