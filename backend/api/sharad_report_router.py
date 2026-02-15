"""
SHARAD Subsurface Interface Report Router — SSE streaming endpoint + file download.

POST /api/sharad-report/{product_id}  — Run pipeline and stream SSE events
GET  /api/sharad-report/{product_id}/status — Check if report exists
GET  /api/sharad-report/{product_id}/download/{filename} — Download output file
GET  /api/sharad-report/list — List all generated reports
"""

import json
import logging
import os
import re

from fastapi import APIRouter, HTTPException, Path
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/sharad-report", tags=["SHARAD Report"])

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT_DIR = os.path.join(_BACKEND_DIR, "sharad_reports")

# Product ID validation
_PRODUCT_RE = re.compile(r"^[a-zA-Z0-9_]+$")


def _validate_product_id(product_id: str) -> str:
    if not _PRODUCT_RE.match(product_id):
        raise HTTPException(status_code=400, detail="Invalid product ID")
    return product_id


# =============================================================================
# SSE Pipeline Endpoint
# =============================================================================

@router.post("/{product_id}")
async def run_sharad_report(product_id: str = Path(..., description="SHARAD RDR product ID")):
    """
    Run the full SHARAD subsurface interface analysis pipeline.
    Returns Server-Sent Events (SSE) stream with progress updates.
    """
    _validate_product_id(product_id)

    from .sharad_report import run_report_pipeline

    async def event_stream():
        try:
            async for event in run_report_pipeline(product_id):
                evt_type = event.get("event", "message")
                data = event.get("data", {})
                yield f"event: {evt_type}\ndata: {json.dumps(data)}\n\n"
        except FileNotFoundError:
            yield f"event: error\ndata: {json.dumps({'message': 'SHARAD data not found'})}\n\n"
        except Exception:
            logger.exception(f"SHARAD report pipeline failed for {product_id}")
            yield f"event: error\ndata: {json.dumps({'message': 'Internal pipeline error'})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# =============================================================================
# Status / Download Endpoints
# =============================================================================

@router.get("/{product_id}/status")
async def get_report_status(product_id: str = Path(...)):
    """Check if a report exists and list its output files."""
    _validate_product_id(product_id)

    out_dir = os.path.join(REPORT_DIR, product_id.upper())
    if not os.path.isdir(out_dir):
        return JSONResponse(content={"exists": False, "product_id": product_id.upper()})

    files = {}
    complete = False
    for fname in os.listdir(out_dir):
        fpath = os.path.join(out_dir, fname)
        if fname == ".complete":
            complete = True
            continue
        if os.path.isfile(fpath):
            files[fname] = {
                "size_bytes": os.path.getsize(fpath),
                "type": "pdf" if fname.endswith(".pdf") else
                        "image" if fname.endswith(".png") else
                        "csv" if fname.endswith(".csv") else "other",
            }

    return JSONResponse(content={
        "exists": True,
        "complete": complete,
        "product_id": product_id.upper(),
        "files": files,
    })


@router.get("/{product_id}/download/{filename}")
async def download_report_file(
    product_id: str = Path(...),
    filename: str = Path(...),
):
    """Download a specific output file from the report."""
    _validate_product_id(product_id)

    # Validate filename (must start with alphanumeric/underscore)
    if not re.match(r"^[a-zA-Z0-9_][a-zA-Z0-9_.\-]*$", filename):
        raise HTTPException(status_code=400, detail="Invalid filename")

    fpath = os.path.join(REPORT_DIR, product_id.upper(), filename)
    if not os.path.isfile(fpath):
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")

    media_types = {
        ".pdf": "application/pdf",
        ".png": "image/png",
        ".csv": "text/csv",
        ".json": "application/json",
    }
    ext = os.path.splitext(filename)[1].lower()
    media_type = media_types.get(ext, "application/octet-stream")

    return FileResponse(fpath, media_type=media_type, filename=filename)


@router.get("/list")
async def list_reports():
    """List all generated SHARAD reports."""
    if not os.path.isdir(REPORT_DIR):
        return JSONResponse(content={"reports": []})

    reports = []
    for name in sorted(os.listdir(REPORT_DIR)):
        rdir = os.path.join(REPORT_DIR, name)
        if not os.path.isdir(rdir):
            continue

        entries = os.listdir(rdir)
        file_count = sum(1 for f in entries if os.path.isfile(os.path.join(rdir, f)) and not f.startswith("."))
        has_pdf = any(f.endswith(".pdf") for f in entries)
        complete = ".complete" in entries

        reports.append({
            "product_id": name,
            "file_count": file_count,
            "has_pdf": has_pdf,
            "complete": complete,
        })

    return JSONResponse(content={"reports": reports})
