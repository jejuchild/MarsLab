"""
FastAPI router for Radar Attenuation Mapper.

Endpoints:
  GET /api/attenuation/profile    — Run mapper and return full result
  GET /api/attenuation/export_csv — Download profile as CSV
"""

import io
import csv
import logging
import re
from collections import OrderedDict

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/attenuation", tags=["Radar Attenuation"])

_SAFE_ID = re.compile(r"^[a-zA-Z0-9_\-]+$")

# LRU result cache — avoids re-running 4s pipeline for identical params
_mapper_cache: OrderedDict = OrderedDict()
_MAPPER_CACHE_MAX = 32


def _validate_product_id(pid: str):
    if not _SAFE_ID.match(pid):
        raise HTTPException(status_code=400, detail=f"Invalid product_id: {pid}")


def _run_mapper(
    product_id: str,
    epsilon_r: float,
    snr_threshold: float,
    search_lo: int,
    search_hi: int,
    dtm_product_id: str,
):
    """Instantiate and run RadarAttenuationMapper (with LRU cache)."""
    cache_key = (product_id, epsilon_r, snr_threshold, search_lo, search_hi, dtm_product_id)
    if cache_key in _mapper_cache:
        _mapper_cache.move_to_end(cache_key)
        return _mapper_cache[cache_key]

    from analysis.radar_attenuation.pipeline import RadarAttenuationMapper

    mapper = RadarAttenuationMapper()
    result = mapper.run(
        product_id=product_id,
        epsilon_r=epsilon_r,
        snr_threshold=snr_threshold,
        search_lo=search_lo,
        search_hi=search_hi,
        dtm_product_id=dtm_product_id,
    )
    # Only cache successful results
    if result.success:
        _mapper_cache[cache_key] = result
        if len(_mapper_cache) > _MAPPER_CACHE_MAX:
            _mapper_cache.popitem(last=False)
    return result


@router.get("/profile")
async def attenuation_profile(
    product_id: str = Query(..., description="SHARAD_HIGHRES product ID"),
    epsilon_r: float = Query(2.5, ge=1.5, le=8.0, description="Dielectric constant"),
    snr_threshold: float = Query(3.5, ge=1.0, le=20.0, description="Min SNR for detection"),
    search_lo: int = Query(10, ge=1, le=200, description="Search start (bins below surface)"),
    search_hi: int = Query(150, ge=10, le=500, description="Search end (bins below surface)"),
    dtm_product_id: str = Query("", description="Optional HiRISE DTM product ID"),
):
    """Run Radar Attenuation Mapper on a SHARAD track.

    Returns attenuation profile, overlay segments, summary statistics,
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
        result = _run_mapper(
            product_id, epsilon_r, snr_threshold,
            search_lo, search_hi, dtm_product_id,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Attenuation mapper failed for %s", product_id)
        raise HTTPException(status_code=500, detail=str(e))

    if not result.success:
        err = result.error or "Unknown error"
        if "not found" in err.lower() or "no data" in err.lower() or "does not exist" in err.lower():
            raise HTTPException(status_code=404, detail=err)
        raise HTTPException(status_code=500, detail=err)

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
    """Export attenuation profile as CSV."""
    _validate_product_id(product_id)
    if dtm_product_id:
        _validate_product_id(dtm_product_id)

    if search_lo >= search_hi:
        raise HTTPException(status_code=422, detail="search_lo must be < search_hi")

    try:
        result = _run_mapper(
            product_id, epsilon_r, snr_threshold,
            search_lo, search_hi, dtm_product_id,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Attenuation CSV export failed for %s", product_id)
        raise HTTPException(status_code=500, detail=str(e))

    if not result.success:
        err = result.error or "Unknown error"
        if "not found" in err.lower() or "no data" in err.lower() or "does not exist" in err.lower():
            raise HTTPException(status_code=404, detail=err)
        raise HTTPException(status_code=500, detail=err)

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "trace_idx", "lat", "lon", "along_track_km", "surface_elev_m",
        "interface_detected", "surface_power_dB", "subsurface_power_dB",
        "depth_m", "alpha_dBm", "transparency", "snr", "confidence",
        "epsilon_r",
    ])
    for s in result.profile:
        writer.writerow([
            s.trace_idx, s.lat, s.lon, s.along_track_km, s.surface_elev_m,
            s.interface_detected,
            s.surface_power_dB if s.surface_power_dB is not None else "",
            s.subsurface_power_dB if s.subsurface_power_dB is not None else "",
            s.depth_m if s.depth_m is not None else "",
            s.alpha_dBm if s.alpha_dBm is not None else "",
            s.transparency if s.transparency is not None else "",
            s.snr if s.snr is not None else "",
            s.confidence if s.confidence is not None else "",
            epsilon_r,
        ])

    buf.seek(0)
    filename = f"attenuation_{product_id}.csv"
    return StreamingResponse(
        buf,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
