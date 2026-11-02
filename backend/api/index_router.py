"""GeoJSON index endpoints (/<instrument>_index.geojson) and manual index repair.

Indices are served from the shared in-memory cache in ``api.geojson_cache``.
"""
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from api.geojson_cache import _geojson_bytes_cache, _preload_indices_parallel

router = APIRouter()


@router.get("/hirise_index.geojson")
def get_hirise_index():
    if "hirise" in _geojson_bytes_cache:
        return Response(
            content=_geojson_bytes_cache["hirise"],
            media_type="application/json",
            headers={"Cache-Control": "public, max-age=86400"},
        )
    raise HTTPException(status_code=404, detail="HiRISE index.geojson not found")


@router.get("/crism_index.geojson")
def get_crism_index():
    if "crism" in _geojson_bytes_cache:
        return Response(
            content=_geojson_bytes_cache["crism"],
            media_type="application/json",
            headers={"Cache-Control": "public, max-age=86400"},
        )
    raise HTTPException(status_code=404, detail="CRISM index.geojson not found")


@router.get("/sharad_index.geojson")
def get_sharad_index():
    if "sharad" in _geojson_bytes_cache:
        return Response(
            content=_geojson_bytes_cache["sharad"],
            media_type="application/json",
            headers={"Cache-Control": "public, max-age=86400"},
        )
    raise HTTPException(status_code=404, detail="SHARAD index.geojson not found")


@router.get("/hirise_dtm_index.geojson")
def get_hirise_dtm_index():
    """Serve HiRISE DTM index.geojson."""
    if "hirise_dtm" in _geojson_bytes_cache:
        return Response(
            content=_geojson_bytes_cache["hirise_dtm"],
            media_type="application/json",
            headers={"Cache-Control": "public, max-age=86400"},
        )
    raise HTTPException(status_code=404, detail="HiRISE DTM index.geojson not found")


@router.get("/crism_trr3_index.geojson")
def get_crism_trr3_index():
    """Serve CRISM TRR3 index.geojson."""
    if "crism_trr3" in _geojson_bytes_cache:
        return Response(
            content=_geojson_bytes_cache["crism_trr3"],
            media_type="application/json",
            headers={"Cache-Control": "public, max-age=86400"},
        )
    raise HTTPException(status_code=404, detail="CRISM TRR3 index.geojson not found")


@router.get("/ctx_mosaic_index.geojson")
def get_ctx_mosaic_index():
    """Serve CTX Mosaic (Murray Lab) index.geojson."""
    if "ctx_mosaic" in _geojson_bytes_cache:
        return Response(
            content=_geojson_bytes_cache["ctx_mosaic"],
            media_type="application/json",
            headers={"Cache-Control": "public, max-age=86400"},
        )
    raise HTTPException(status_code=404, detail="CTX Mosaic index.geojson not found")


# ======================================================
# Index Repair API (manual trigger)
# ======================================================
@router.post("/api/repair-index")
async def repair_index_endpoint(request: Request):
    """
    Manually trigger index repair: scan all data directories for orphaned
    downloads and add them to their respective index.geojson files.
    Refreshes in-memory caches if any orphans are found.
    """
    from api.index_repair import repair_all_indices

    session = getattr(request.app.state, "http_session", None)
    results = await repair_all_indices(session)

    total_added = sum(r.get("added", 0) for r in results.values())
    if total_added > 0:
        _preload_indices_parallel()  # Refresh in-memory caches

    return JSONResponse(content={
        "total_added": total_added,
        "instruments": results,
    })
