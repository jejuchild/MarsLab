"""
Multi-Instrument Scientific Report Router — SSE streaming + download.

POST /api/multi-report/run       — Run pipeline (SSE stream)
GET  /api/multi-report/list      — List all generated reports
GET  /api/multi-report/{id}/status   — Check report status + files
GET  /api/multi-report/{id}/download/{filename} — Download file
"""

import json
import logging
import os
import re
from typing import Optional

from fastapi import APIRouter, HTTPException, Path, Query
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/multi-report", tags=["Multi-Instrument Report"])

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT_DIR = os.path.join(_BACKEND_DIR, "multi_reports")

_SAFE_RE = re.compile(r"^[a-zA-Z0-9_]+$")
_FILENAME_RE = re.compile(r"^[a-zA-Z0-9_][a-zA-Z0-9_.\-]*$")


class MultiReportRequest(BaseModel):
    region_name: str = Field("Analysis Region", max_length=200)
    sharad_product_id: Optional[str] = None
    crism_obs_id: Optional[str] = None
    hirise_dtm_id: Optional[str] = None
    center_lat: Optional[float] = Field(None, ge=-90, le=90)
    center_lon: Optional[float] = Field(None, ge=-360, le=360)
    analysis_radius_km: float = Field(50.0, gt=0, le=500)


@router.post("/run")
async def run_multi_report(req: MultiReportRequest):
    """Run multi-instrument analysis pipeline. Returns SSE stream."""
    # Validate product IDs
    for pid in [req.sharad_product_id, req.crism_obs_id, req.hirise_dtm_id]:
        if pid and not _SAFE_RE.match(pid):
            raise HTTPException(status_code=400, detail="Invalid product ID")

    from .multi_report import run_multi_report as pipeline, MultiReportConfig

    config = MultiReportConfig(
        region_name=req.region_name,
        sharad_product_id=req.sharad_product_id,
        crism_obs_id=req.crism_obs_id,
        hirise_dtm_id=req.hirise_dtm_id,
        center_lat=req.center_lat,
        center_lon=req.center_lon,
        analysis_radius_km=req.analysis_radius_km,
    )

    async def event_stream():
        try:
            async for event in pipeline(config):
                evt_type = event.get("event", "message")
                data = event.get("data", {})
                yield f"event: {evt_type}\ndata: {json.dumps(data)}\n\n"
        except Exception:
            logger.exception("Multi-report pipeline failed")
            yield f"event: error\ndata: {json.dumps({'message': 'Internal pipeline error'})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive",
                 "X-Accel-Buffering": "no"},
    )


@router.get("/list")
async def list_multi_reports():
    """List all generated reports."""
    if not os.path.isdir(REPORT_DIR):
        return JSONResponse(content={"reports": []})

    reports = []
    for name in sorted(os.listdir(REPORT_DIR)):
        rdir = os.path.join(REPORT_DIR, name)
        if not os.path.isdir(rdir):
            continue
        entries = os.listdir(rdir)
        complete = ".complete" in entries
        has_pdf = any(f.endswith(".pdf") for f in entries)
        file_count = sum(1 for f in entries if not f.startswith(".") and
                         os.path.isfile(os.path.join(rdir, f)))

        # Read config from completion marker
        config_info = {}
        marker_path = os.path.join(rdir, ".complete")
        if os.path.exists(marker_path):
            try:
                with open(marker_path) as f:
                    marker = json.load(f)
                config_info = marker.get("config", {})
            except Exception:
                pass

        reports.append({
            "report_id": name,
            "complete": complete,
            "has_pdf": has_pdf,
            "file_count": file_count,
            "config": config_info,
        })

    return JSONResponse(content={"reports": reports})


@router.get("/{report_id}/status")
async def get_multi_report_status(report_id: str = Path(...)):
    if not _SAFE_RE.match(report_id):
        raise HTTPException(status_code=400, detail="Invalid report ID")

    out_dir = os.path.join(REPORT_DIR, report_id)
    if not os.path.isdir(out_dir):
        return JSONResponse(content={"exists": False, "report_id": report_id})

    files = {}
    complete = False
    for fname in os.listdir(out_dir):
        if fname == ".complete":
            complete = True
            continue
        fpath = os.path.join(out_dir, fname)
        if os.path.isfile(fpath):
            files[fname] = {
                "size_bytes": os.path.getsize(fpath),
                "type": "pdf" if fname.endswith(".pdf") else
                        "image" if fname.endswith(".png") else
                        "csv" if fname.endswith(".csv") else
                        "text" if fname.endswith(".txt") else "other",
            }

    return JSONResponse(content={
        "exists": True,
        "complete": complete,
        "report_id": report_id,
        "files": files,
    })


@router.get("/{report_id}/download/{filename}")
async def download_multi_report_file(
    report_id: str = Path(...),
    filename: str = Path(...),
):
    if not _SAFE_RE.match(report_id):
        raise HTTPException(status_code=400, detail="Invalid report ID")
    if not _FILENAME_RE.match(filename):
        raise HTTPException(status_code=400, detail="Invalid filename")

    fpath = os.path.join(REPORT_DIR, report_id, filename)
    if not os.path.isfile(fpath):
        raise HTTPException(status_code=404, detail="File not found")

    media_types = {
        ".pdf": "application/pdf", ".png": "image/png",
        ".csv": "text/csv", ".txt": "text/plain",
    }
    ext = os.path.splitext(filename)[1].lower()
    return FileResponse(fpath, media_type=media_types.get(ext, "application/octet-stream"),
                        filename=filename)
