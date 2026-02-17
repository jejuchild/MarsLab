"""
FastAPI router for Crater Stratigraphy Analyzer.

Endpoints:
  GET /api/stratigraphy/analyze    — Run crater stratigraphy and return full result
  GET /api/stratigraphy/export_csv — Download epsilon estimates as CSV
"""

import io
import csv
import logging

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/stratigraphy", tags=["Crater Stratigraphy"])


def _run_analysis(
    crater_lat: float,
    crater_lon: float,
    diameter_km: float,
    buffer_km: float,
    n_azimuths: int,
    max_radius_m: float,
    min_snr: float,
):
    """Instantiate and run CraterStratigraphyAnalyzer."""
    from analysis.epsilon_terrace.pipeline import CraterStratigraphyAnalyzer

    analyzer = CraterStratigraphyAnalyzer()
    return analyzer.run(
        crater_lat=crater_lat,
        crater_lon=crater_lon,
        diameter_km=diameter_km,
        buffer_km=buffer_km,
        n_azimuths=n_azimuths,
        max_radius_m=max_radius_m,
        min_snr=min_snr,
    )


@router.get("/analyze")
async def analyze(
    crater_lat: float = Query(..., ge=-90, le=90, description="Crater center latitude"),
    crater_lon: float = Query(..., ge=-180, le=360, description="Crater center longitude"),
    diameter_km: float = Query(0.0, ge=0, le=500, description="Crater diameter in km (0 = auto)"),
    buffer_km: float = Query(30.0, ge=5, le=100, description="SHARAD search buffer in km"),
    n_azimuths: int = Query(36, ge=8, le=72, description="Number of radial profile azimuths"),
    max_radius_m: float = Query(3000.0, ge=500, le=20000, description="Max radial distance in meters"),
    min_snr: float = Query(3.5, ge=1.0, le=20.0, description="Min SNR for SHARAD picks"),
):
    """Run crater stratigraphy analysis.

    Searches for SHARAD tracks and DTMs near the crater, extracts terrace
    profiles, picks subsurface reflectors, and computes dielectric constant.
    """
    try:
        result = _run_analysis(
            crater_lat, crater_lon, diameter_km,
            buffer_km, n_azimuths, max_radius_m, min_snr,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Stratigraphy failed for (%.3f, %.3f)", crater_lat, crater_lon)
        raise HTTPException(status_code=500, detail=str(e))

    return result.model_dump()


@router.get("/export_csv")
async def export_csv(
    crater_lat: float = Query(..., ge=-90, le=90),
    crater_lon: float = Query(..., ge=-180, le=360),
    diameter_km: float = Query(0.0, ge=0, le=500),
    buffer_km: float = Query(30.0, ge=5, le=100),
    n_azimuths: int = Query(36, ge=8, le=72),
    max_radius_m: float = Query(3000.0, ge=500, le=20000),
    min_snr: float = Query(3.5, ge=1.0, le=20.0),
):
    """Export stratigraphy results as CSV."""
    try:
        result = _run_analysis(
            crater_lat, crater_lon, diameter_km,
            buffer_km, n_azimuths, max_radius_m, min_snr,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Stratigraphy CSV export failed for (%.3f, %.3f)", crater_lat, crater_lon)
        raise HTTPException(status_code=500, detail=str(e))

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "depth_metric", "depth_m", "depth_unc_m",
        "twt_us", "epsilon_r", "epsilon_low", "epsilon_high",
        "interpretation", "quality", "quality_note",
        "sharad_product", "n_picks",
    ])
    for est in result.epsilon_estimates:
        writer.writerow([
            est.depth_metric, est.depth_m, est.depth_unc_m,
            est.twt_us, est.epsilon_r, est.epsilon_low, est.epsilon_high,
            est.interpretation, est.quality, est.quality_note,
            est.sharad_product, est.n_picks,
        ])

    buf.seek(0)
    lat_str = f"{crater_lat:.2f}".replace(".", "_")
    lon_str = f"{crater_lon:.2f}".replace(".", "_")
    filename = f"stratigraphy_{lat_str}_{lon_str}.csv"
    return StreamingResponse(
        buf,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
