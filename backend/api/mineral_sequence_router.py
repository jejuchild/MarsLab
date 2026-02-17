"""
FastAPI router for Aqueous Mineral Sequence Mapper.

Endpoints:
  GET /api/mineral-sequence/analyze   — Run mapper and return full result
  GET /api/mineral-sequence/export_csv — Download transect as CSV
"""

import io
import csv
import logging
import re

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/mineral-sequence", tags=["Mineral Sequence"])

_SAFE_ID = re.compile(r"^[a-zA-Z0-9_]+$")


def _validate_obs_id(oid: str):
    if not _SAFE_ID.match(oid):
        raise HTTPException(status_code=400, detail=f"Invalid obs_id: {oid}")


def _run_mapper(obs_id: str, transect_direction: str, transect_offset: float):
    """Instantiate and run AqueousMineralMapper."""
    from analysis.mineral_sequence.pipeline import AqueousMineralMapper

    mapper = AqueousMineralMapper()
    return mapper.run(
        obs_id=obs_id,
        transect_direction=transect_direction,
        transect_offset=transect_offset,
    )


@router.get("/analyze")
async def analyze_sequence(
    obs_id: str = Query(..., description="CRISM observation ID (CNN must be cached)"),
    transect_direction: str = Query("NS", description="Transect direction: NS or EW"),
    transect_offset: float = Query(0.5, ge=0.0, le=1.0, description="Transect position (0.0–1.0)"),
):
    """Run Aqueous Mineral Sequence Mapper on a CRISM observation.

    Returns mineral transect, transitions, paleo-environment matches,
    and input parameters for reproducibility.
    """
    _validate_obs_id(obs_id)

    if transect_direction.upper() not in ("NS", "EW"):
        raise HTTPException(status_code=422, detail="transect_direction must be NS or EW")

    try:
        result = _run_mapper(obs_id, transect_direction, transect_offset)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Mineral sequence mapper failed for %s", obs_id)
        raise HTTPException(status_code=500, detail=str(e))

    if not result.success:
        status = 404 if "No CNN" in (result.error or "") else 500
        raise HTTPException(status_code=status, detail=result.error or "Unknown error")

    return result.model_dump()


@router.get("/export_csv")
async def export_csv(
    obs_id: str = Query(..., description="CRISM observation ID"),
    transect_direction: str = Query("NS"),
    transect_offset: float = Query(0.5, ge=0.0, le=1.0),
):
    """Export mineral sequence transect as CSV."""
    _validate_obs_id(obs_id)

    if transect_direction.upper() not in ("NS", "EW"):
        raise HTTPException(status_code=422, detail="transect_direction must be NS or EW")

    try:
        result = _run_mapper(obs_id, transect_direction, transect_offset)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Mineral sequence CSV export failed for %s", obs_id)
        raise HTTPException(status_code=500, detail=str(e))

    if not result.success:
        status = 404 if "No CNN" in (result.error or "") else 500
        raise HTTPException(status_code=status, detail=result.error or "Unknown error")

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "position_idx", "row", "col", "mineral_id", "mineral_name",
        "geochem_group", "confidence",
    ])
    for pt in result.transect:
        writer.writerow([
            pt.position_idx, pt.row, pt.col,
            pt.mineral_id if pt.mineral_id is not None else "",
            pt.mineral_name or "",
            pt.geochem_group or "",
            pt.confidence if pt.confidence is not None else "",
        ])

    # Append transitions section
    writer.writerow([])
    writer.writerow(["# Transitions"])
    writer.writerow(["position_idx", "from_group", "to_group", "from_mineral", "to_mineral"])
    for tr in result.transitions:
        writer.writerow([
            tr.position_idx, tr.from_group, tr.to_group,
            tr.from_mineral, tr.to_mineral,
        ])

    buf.seek(0)
    filename = f"mineral_sequence_{obs_id}.csv"
    return StreamingResponse(
        buf,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
