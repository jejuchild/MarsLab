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
async def hyperbola_fit(
    product_id: str = Query(..., description="SHARAD product ID"),
    trace_idx0: int = Query(..., ge=0, description="Center trace index"),
    bin_idx0: int = Query(..., ge=0, description="Center bin index"),
    window_traces: int = Query(150, ge=20, le=500, description="Window width in traces"),
    window_bins: int = Query(120, ge=20, le=300, description="Window height in bins"),
) -> dict:
    """Estimate εr from hyperbolic reflector curvature (NMO analysis).

    Fits a hyperbolic travel-time curve to reflector arrivals around a point
    reflector to estimate velocity → εr.

    Args:
        product_id: SHARAD product ID containing reflector
        trace_idx0: Center trace of point reflector
        bin_idx0: Center bin of point reflector
        window_traces: Traces to include in fitting window
        window_bins: Bins to include in fitting window

    Returns:
        epsilon_r_est: Estimated εr from NMO fit
        epsilon_r_std: Uncertainty in estimate
        rmse_bins: RMS error of fit (in bins)
        inlier_ratio: Fraction of picks used (vs outliers rejected)
        n_inliers: Number of inlier reflections
        quality_note: Fit quality assessment
        diagnostics: Advanced metrics (curve params, residuals, etc.)
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

        return {
            "success": True,
            "epsilon_r_est": result.epsilon_r_est,
            "epsilon_r_std": result.epsilon_r_std,
            "rmse_bins": round(result.rmse_bins, 2),
            "inlier_ratio": round(result.inlier_ratio, 3),
            "n_inliers": result.n_inliers,
            "quality_note": result.quality_note,
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
