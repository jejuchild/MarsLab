"""
FastAPI router for Stratigraphic Column Builder.

Endpoints:
  GET /api/strat-column/build      — Build column and return full result
  GET /api/strat-column/export_csv — Download column as CSV
"""

import io
import csv
import logging

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/strat-column", tags=["Stratigraphic Column"])


def _run_builder(
    crater_lat: float,
    crater_lon: float,
    diameter_km: float,
    buffer_km: float,
    include_crism: bool,
    include_sharad: bool,
):
    """Instantiate and run StratigraphicColumnBuilder."""
    from analysis.strat_column.pipeline import StratigraphicColumnBuilder

    builder = StratigraphicColumnBuilder()
    return builder.run(
        crater_lat=crater_lat,
        crater_lon=crater_lon,
        diameter_km=diameter_km,
        buffer_km=buffer_km,
        include_crism=include_crism,
        include_sharad=include_sharad,
    )


@router.get("/build")
async def build_column(
    crater_lat: float = Query(..., ge=-90, le=90, description="Crater latitude"),
    crater_lon: float = Query(..., ge=-360, le=360, description="Crater longitude"),
    diameter_km: float = Query(0, ge=0, description="Crater diameter (km)"),
    buffer_km: float = Query(30, ge=5, le=200, description="Search buffer (km)"),
    include_crism: bool = Query(True, description="Include CRISM mineral data"),
    include_sharad: bool = Query(True, description="Include SHARAD subsurface data"),
):
    """Build a composite stratigraphic column for a crater.

    Returns vertical column layers, summary statistics,
    and input parameters for reproducibility.
    """
    try:
        result = _run_builder(
            crater_lat, crater_lon, diameter_km,
            buffer_km, include_crism, include_sharad,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Strat column build failed")
        raise HTTPException(status_code=500, detail=str(e))

    if not result.success:
        raise HTTPException(status_code=500, detail=result.error or "Unknown error")

    return result.model_dump()


@router.get("/export_csv")
async def export_csv(
    crater_lat: float = Query(..., ge=-90, le=90),
    crater_lon: float = Query(..., ge=-360, le=360),
    diameter_km: float = Query(0, ge=0),
    buffer_km: float = Query(30, ge=5, le=200),
    include_crism: bool = Query(True),
    include_sharad: bool = Query(True),
):
    """Export stratigraphic column as CSV."""
    try:
        result = _run_builder(
            crater_lat, crater_lon, diameter_km,
            buffer_km, include_crism, include_sharad,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Strat column CSV export failed")
        raise HTTPException(status_code=500, detail=str(e))

    if not result.success:
        raise HTTPException(status_code=500, detail=result.error or "Unknown error")

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "layer_idx", "depth_top_m", "depth_bottom_m", "thickness_m",
        "source", "instrument", "mineral_name", "geochem_group",
        "epsilon_r", "material_class", "confidence",
    ])
    for layer in result.layers:
        writer.writerow([
            layer.layer_idx,
            layer.depth_top_m, layer.depth_bottom_m, layer.thickness_m,
            layer.source, layer.instrument,
            layer.mineral_name or "",
            layer.geochem_group or "",
            layer.epsilon_r if layer.epsilon_r is not None else "",
            layer.material_class or "",
            layer.confidence if layer.confidence is not None else "",
        ])

    buf.seek(0)
    filename = f"strat_column_{crater_lat:.2f}_{crater_lon:.2f}.csv"
    return StreamingResponse(
        buf,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
