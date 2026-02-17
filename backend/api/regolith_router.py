"""
FastAPI router for Regolith Thickness Estimator (RTE).

Endpoints:
  GET /api/regolith/thickness_profile — Run RTE and return full result
  GET /api/regolith/export_csv        — Download profile as CSV
"""

import io
import csv
import logging
import re

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/regolith", tags=["Regolith Thickness"])

_SAFE_ID = re.compile(r"^[a-zA-Z0-9_\-]+$")


def _validate_product_id(pid: str):
    if not _SAFE_ID.match(pid):
        raise HTTPException(status_code=400, detail=f"Invalid product_id: {pid}")


def _run_rte(
    product_id: str,
    epsilon_r: float,
    snr_threshold: float,
    search_lo: int,
    search_hi: int,
    dtm_product_id: str,
):
    """Instantiate and run RegolithThicknessEstimator."""
    from analysis.regolith_thickness.pipeline import RegolithThicknessEstimator

    rte = RegolithThicknessEstimator()
    return rte.run(
        product_id=product_id,
        epsilon_r=epsilon_r,
        snr_threshold=snr_threshold,
        search_lo=search_lo,
        search_hi=search_hi,
        dtm_product_id=dtm_product_id,
    )


@router.get("/thickness_profile")
async def thickness_profile(
    product_id: str = Query(..., description="SHARAD_HIGHRES product ID"),
    epsilon_r: float = Query(2.5, ge=1.5, le=8.0, description="Dielectric constant"),
    snr_threshold: float = Query(3.5, ge=1.0, le=20.0, description="Min SNR for detection"),
    search_lo: int = Query(10, ge=1, le=200, description="Search start (bins below surface)"),
    search_hi: int = Query(150, ge=10, le=500, description="Search end (bins below surface)"),
    dtm_product_id: str = Query("", description="Optional HiRISE DTM product ID"),
):
    """Run Regolith Thickness Estimator on a SHARAD track.

    Returns thickness profile, overlay segments, summary statistics,
    and input parameters for reproducibility.
    """
    _validate_product_id(product_id)
    if dtm_product_id:
        _validate_product_id(dtm_product_id)

    if search_lo >= search_hi:
        raise HTTPException(
            status_code=422,
            detail="search_lo must be less than search_hi",
        )

    try:
        result = _run_rte(
            product_id, epsilon_r, snr_threshold,
            search_lo, search_hi, dtm_product_id,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("RTE failed for %s", product_id)
        raise HTTPException(status_code=500, detail=str(e))

    if not result.success:
        raise HTTPException(status_code=500, detail=result.error or "Unknown error")

    return result.model_dump()


@router.get("/export_csv")
async def export_csv(
    product_id: str = Query(..., description="SHARAD_HIGHRES product ID"),
    epsilon_r: float = Query(2.5, ge=1.5, le=8.0),
    snr_threshold: float = Query(3.5, ge=1.0, le=20.0),
    search_lo: int = Query(10, ge=1, le=200),
    search_hi: int = Query(150, ge=10, le=500),
    dtm_product_id: str = Query("", description="Optional HiRISE DTM product ID"),
):
    """Export regolith thickness profile as CSV."""
    _validate_product_id(product_id)
    if dtm_product_id:
        _validate_product_id(dtm_product_id)

    if search_lo >= search_hi:
        raise HTTPException(status_code=422, detail="search_lo must be < search_hi")

    try:
        result = _run_rte(
            product_id, epsilon_r, snr_threshold,
            search_lo, search_hi, dtm_product_id,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("RTE CSV export failed for %s", product_id)
        raise HTTPException(status_code=500, detail=str(e))

    if not result.success:
        raise HTTPException(status_code=500, detail=result.error or "Unknown error")

    # Stream CSV
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "trace_idx", "lat", "lon", "along_track_km", "surface_elev_m",
        "interface_detected", "delta_bins", "twt_us", "thickness_m",
        "snr", "confidence", "epsilon_r",
    ])
    for s in result.profile:
        writer.writerow([
            s.trace_idx, s.lat, s.lon, s.along_track_km, s.surface_elev_m,
            s.interface_detected,
            s.delta_bins if s.delta_bins is not None else "",
            s.twt_us if s.twt_us is not None else "",
            s.thickness_m if s.thickness_m is not None else "",
            s.snr if s.snr is not None else "",
            s.confidence,
            epsilon_r,
        ])

    buf.seek(0)
    filename = f"regolith_{product_id}.csv"
    return StreamingResponse(
        buf,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
