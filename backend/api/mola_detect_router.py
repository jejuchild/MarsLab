"""
MOLA Landform Detection Router — SSE streaming + pre-computed cache.

POST /api/mola-detect/scan      — Run detection pipeline, stream SSE events
GET  /api/mola-detect/status    — Check if DEM is available
GET  /api/mola-detect/features  — Query pre-computed landform cache by bbox

Performance: DEM extracted once, derivatives pre-computed, detectors run in
parallel via asyncio.gather + to_thread.
"""

import json
import logging
import asyncio
import time
import os
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/mola-detect", tags=["MOLA Detection"])


class ScanRequest(BaseModel):
    lat: float = Field(..., ge=-90, le=90, description="Center latitude")
    lon: float = Field(..., ge=-180, le=360, description="Center longitude")
    radius_km: float = Field(100, ge=10, le=500, description="Scan radius in km")
    min_diameter_km: float = Field(5.0, ge=2, le=100, description="Min crater diameter")
    min_depth_m: float = Field(100, ge=20, le=2000, description="Min depression depth")
    detect_craters: bool = Field(True, description="Detect craters + volcanics + graben")
    detect_ldas: bool = Field(True, description="Detect lobate debris aprons")
    detect_channels: bool = Field(True, description="Detect channels/valleys")
    detect_ridges: bool = Field(True, description="Detect wrinkle ridges")


def _sse(event: str, data: dict) -> str:
    """Format a Server-Sent Event."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _feature_to_dict(f) -> dict:
    """Convert a DetectedFeature dataclass to a JSON-safe dict."""
    d = {
        "id": f.id,
        "type": f.feature_type,
        "lat": round(f.lat, 5),
        "lon": round(f.lon, 5),
        "confidence": round(f.confidence, 2),
        "description": f.description,
        "morphology": f.morphology,
    }
    if f.diameter_km > 0:
        d["diameter_km"] = round(f.diameter_km, 1)
    if f.depth_m > 0:
        d["depth_m"] = round(f.depth_m, 1)
    if f.depth_diameter_ratio > 0:
        d["depth_diameter_ratio"] = round(f.depth_diameter_ratio, 4)
    if f.circularity > 0:
        d["circularity"] = round(f.circularity, 2)
    if f.rim_elevation_m != 0:
        d["rim_elevation_m"] = round(f.rim_elevation_m, 1)
    if f.floor_elevation_m != 0:
        d["floor_elevation_m"] = round(f.floor_elevation_m, 1)
    if f.n_terraces > 0:
        d["n_terraces"] = f.n_terraces
    if f.length_km > 0:
        d["length_km"] = round(f.length_km, 1)
    if f.width_km > 0:
        d["width_km"] = round(f.width_km, 1)
    if f.area_km2 > 0:
        d["area_km2"] = round(f.area_km2, 1)
    if f.sinuosity > 0:
        d["sinuosity"] = round(f.sinuosity, 2)
    if f.path:
        path = f.path
        if len(path) > 80:
            step = max(1, len(path) // 80)
            path = path[::step]
        d["path"] = [[round(p[0], 4), round(p[1], 4)] for p in path]
    if f.boundary:
        boundary = f.boundary
        if len(boundary) > 80:
            step = max(1, len(boundary) // 80)
            boundary = boundary[::step]
        d["boundary"] = [[round(p[0], 4), round(p[1], 4)] for p in boundary]
    if f.terrace_depth_m > 0:
        d["terrace_depth_m"] = round(f.terrace_depth_m, 1)
    if f.terrace_ring_radii_km:
        d["terrace_ring_radii_km"] = [round(r, 2) for r in f.terrace_ring_radii_km]
    return d


@router.post("/scan")
async def run_scan(req: ScanRequest):
    """
    Run MOLA landform detection pipeline with SSE progress streaming.

    DEM is extracted once, derivatives pre-computed, then detectors
    (channels+ridges, LDAs) run in parallel while craters run first
    (heaviest, needs morphological closing).
    """
    async def event_stream():
        t0 = time.time()
        all_features = []
        counts = {}

        try:
            yield _sse("status", {
                "phase": "starting",
                "message": f"Extracting DEM ({req.radius_km}km radius)...",
                "progress": 0,
            })

            # ── Extract DEM once, pre-compute all shared derivatives ──
            from analysis.mola_detect.common import build_shared_context
            ctx = await asyncio.to_thread(
                build_shared_context, req.lat, req.lon, req.radius_km,
            )

            yield _sse("status", {
                "phase": "dem_ready",
                "message": f"DEM ready ({ctx.elev.shape[1]}x{ctx.elev.shape[0]} px), running detectors...",
                "progress": 0.10,
            })

            # ── Phase 1: Craters (heaviest — morphological closing) ──
            # Run first so results stream early while lighter detectors follow.
            if req.detect_craters:
                yield _sse("status", {
                    "phase": "detecting_craters",
                    "message": "Detecting craters, volcanic constructs, and graben...",
                    "progress": 0.12,
                })

                from analysis.mola_detect.crater_detect import detect_craters_and_volcanics

                features = await asyncio.to_thread(
                    detect_craters_and_volcanics,
                    req.lat, req.lon, req.radius_km,
                    min_diameter_km=req.min_diameter_km,
                    min_depth_m=req.min_depth_m,
                    ctx=ctx,
                )

                for f in features:
                    all_features.append(f)
                    yield _sse("feature", _feature_to_dict(f))
                    counts[f.feature_type] = counts.get(f.feature_type, 0) + 1

                yield _sse("status", {
                    "phase": "craters_done",
                    "message": f"Found {len(features)} crater-type features",
                    "progress": 0.55,
                })

            # ── Phase 2: Channels, ridges, LDAs — run in parallel ──
            parallel_tasks = []
            task_labels = []

            if req.detect_channels:
                from analysis.mola_detect.ridge_channel_detect import detect_channels
                parallel_tasks.append(asyncio.to_thread(
                    detect_channels, req.lat, req.lon, req.radius_km, ctx=ctx,
                ))
                task_labels.append("channels")

            if req.detect_ridges:
                from analysis.mola_detect.ridge_channel_detect import detect_ridges
                parallel_tasks.append(asyncio.to_thread(
                    detect_ridges, req.lat, req.lon, req.radius_km, ctx=ctx,
                ))
                task_labels.append("ridges")

            if req.detect_ldas:
                from analysis.mola_detect.lda_detect import detect_ldas
                parallel_tasks.append(asyncio.to_thread(
                    detect_ldas, req.lat, req.lon, req.radius_km, ctx=ctx,
                ))
                task_labels.append("ldas")

            if parallel_tasks:
                active_names = ", ".join(task_labels)
                yield _sse("status", {
                    "phase": "detecting_parallel",
                    "message": f"Detecting {active_names} (parallel)...",
                    "progress": 0.60,
                })

                results = await asyncio.gather(*parallel_tasks)

                for label, result_features in zip(task_labels, results):
                    for f in result_features:
                        all_features.append(f)
                        yield _sse("feature", _feature_to_dict(f))
                        counts[f.feature_type] = counts.get(f.feature_type, 0) + 1

                    yield _sse("status", {
                        "phase": f"{label}_done",
                        "message": f"Found {len(result_features)} {label}",
                        "progress": 0.90,
                    })

            # Done
            elapsed = round(time.time() - t0, 1)
            yield _sse("complete", {
                "total_features": len(all_features),
                "counts": counts,
                "elapsed_s": elapsed,
                "message": f"Scan complete: {len(all_features)} features detected in {elapsed}s",
            })

        except Exception as e:
            logger.exception("MOLA detection pipeline failed")
            yield _sse("error", {"message": str(e)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/status")
async def check_status():
    """Check if the MOLA DEM is available for scanning."""
    from api.terrain_router import DEM_PATH
    available = os.path.exists(DEM_PATH)
    return JSONResponse(content={
        "available": available,
        "max_radius_km": 500,
        "resolution_m": 200,
        "precomputed": _precomputed_cache is not None,
        "precomputed_count": len(_precomputed_cache) if _precomputed_cache else 0,
        "feature_types": [
            "crater", "terraced_crater", "volcanic", "graben",
            "channel", "wrinkle_ridge", "lda",
        ],
    })


# ---------------------------------------------------------------------------
# Pre-computed landform cache
# ---------------------------------------------------------------------------

_CACHE_PATH = Path(__file__).resolve().parent.parent / "cache" / "landforms_precomputed.json"
_precomputed_cache: Optional[list[dict]] = None


def load_precomputed_cache():
    """Load the pre-computed landforms cache into memory. Called at startup."""
    global _precomputed_cache
    if not _CACHE_PATH.exists():
        logger.info("No precomputed landform cache at %s", _CACHE_PATH)
        _precomputed_cache = None
        return
    try:
        with open(_CACHE_PATH) as f:
            data = json.load(f)
        _precomputed_cache = data.get("features", [])
        logger.info(
            "Loaded %d precomputed landforms (generated %s)",
            len(_precomputed_cache),
            data.get("generated_at", "unknown"),
        )
    except Exception as e:
        logger.error("Failed to load precomputed landforms: %s", e)
        _precomputed_cache = None


# Auto-load on module import
load_precomputed_cache()


@router.get("/features")
async def get_precomputed_features(
    west: float = Query(..., ge=-180, le=360, description="West longitude"),
    south: float = Query(..., ge=-90, le=90, description="South latitude"),
    east: float = Query(..., ge=-180, le=360, description="East longitude"),
    north: float = Query(..., ge=-90, le=90, description="North latitude"),
    types: Optional[str] = Query(None, description="Comma-separated feature types to include"),
    min_confidence: float = Query(0.0, ge=0, le=1, description="Minimum confidence threshold"),
):
    """
    Query pre-computed landform features within a bounding box.

    Returns all features whose centroid falls within [west, south, east, north].
    Supports optional type filtering and confidence threshold.
    """
    if _precomputed_cache is None:
        return JSONResponse(
            status_code=404,
            content={"error": "No precomputed landform cache available. Run precompute_landforms.py first."},
        )

    # Parse type filter
    type_filter = None
    if types:
        type_filter = set(t.strip() for t in types.split(","))

    # Handle antimeridian wrapping: if west > east, the bbox wraps around
    wraps = west > east

    results = []
    for f in _precomputed_cache:
        lat = f["lat"]
        lon = f["lon"]

        # Latitude check
        if lat < south or lat > north:
            continue

        # Longitude check (handle wrap-around)
        if wraps:
            if lon < west and lon > east:
                continue
        else:
            if lon < west or lon > east:
                continue

        # Type filter
        if type_filter and f["type"] not in type_filter:
            continue

        # Confidence filter
        if f.get("confidence", 0) < min_confidence:
            continue

        results.append(f)

    return JSONResponse(content={
        "count": len(results),
        "bbox": {"west": west, "south": south, "east": east, "north": north},
        "features": results,
    })
