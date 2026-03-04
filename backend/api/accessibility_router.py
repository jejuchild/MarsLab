"""API router for the Mars Ice Accessibility algorithm.

Endpoints:
  GET /api/accessibility/score    — Point query with optional custom weights
  GET /api/accessibility/tile     — Heatmap tile for map overlay
  GET /api/accessibility/layer-tile — Individual layer tiles
  GET /api/accessibility/weights  — Current default weights
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse, Response

router = APIRouter(prefix="/api/accessibility", tags=["Ice Accessibility"])

_pipeline: object | None = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from analysis.accessibility.pipeline import AccessibilityPipeline
        _pipeline = AccessibilityPipeline()
    return _pipeline


def _parse_weights(
    w_ice: Optional[float],
    w_depth: Optional[float],
    w_excavation: Optional[float],
    w_landing: Optional[float],
) -> Optional[dict[str, float]]:
    """Build weights dict from query params. Returns None if all defaults."""
    if all(v is None for v in [w_ice, w_depth, w_excavation, w_landing]):
        return None
    w: dict[str, float] = {}
    if w_ice is not None:
        w["ice_presence"] = w_ice
    if w_depth is not None:
        w["ice_depth"] = w_depth
    if w_excavation is not None:
        w["excavation"] = w_excavation
    if w_landing is not None:
        w["landing"] = w_landing
    return w


# --------------------------------------------------------------------------
# Point query
# --------------------------------------------------------------------------

@router.get("/score")
def get_accessibility_score(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    w_ice: Optional[float] = Query(None, ge=0, le=1, alias="w_ice"),
    w_depth: Optional[float] = Query(None, ge=0, le=1, alias="w_depth"),
    w_excavation: Optional[float] = Query(None, ge=0, le=1, alias="w_excavation"),
    w_landing: Optional[float] = Query(None, ge=0, le=1, alias="w_landing"),
):
    """Query ice accessibility score at a single point."""
    pipeline = _get_pipeline()
    weights = _parse_weights(w_ice, w_depth, w_excavation, w_landing)

    try:
        result = pipeline.query_point(lat=lat, lon=lon, weights=weights)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Accessibility computation failed: {exc}")

    return JSONResponse(content=asdict(result))


# --------------------------------------------------------------------------
# Tile endpoints
# --------------------------------------------------------------------------

@router.get("/tile/{z}/{x}/{y}.png")
def get_accessibility_tile(
    z: int,
    x: int,
    y: int,
    w_ice: Optional[float] = Query(None, ge=0, le=1),
    w_depth: Optional[float] = Query(None, ge=0, le=1),
    w_excavation: Optional[float] = Query(None, ge=0, le=1),
    w_landing: Optional[float] = Query(None, ge=0, le=1),
):
    """Render accessibility heatmap tile as PNG."""
    pipeline = _get_pipeline()
    weights = _parse_weights(w_ice, w_depth, w_excavation, w_landing)

    try:
        png_bytes = pipeline.get_tile(z=z, x=x, y=y, weights=weights)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Tile render failed: {exc}")

    if png_bytes is None:
        raise HTTPException(status_code=404, detail="No data for this tile")

    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=3600"},
    )


@router.get("/layer-tile/{layer}/{z}/{x}/{y}.png")
def get_layer_tile(
    layer: str,
    z: int,
    x: int,
    y: int,
):
    """Render an individual data layer tile (for debug / layer panel).

    Supported layers: mola_elevation, mola_slope, mola_tri, swim_consistency
    """
    pipeline = _get_pipeline()

    try:
        png_bytes = pipeline.get_layer_tile(layer=layer, z=z, x=x, y=y)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Layer tile render failed: {exc}")

    if png_bytes is None:
        raise HTTPException(status_code=404, detail="No data for this layer tile")

    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=3600"},
    )


# --------------------------------------------------------------------------
# Metadata
# --------------------------------------------------------------------------

@router.get("/weights")
def get_default_weights():
    """Return the default sub-score weights."""
    from analysis.accessibility.algorithm import DEFAULT_WEIGHTS
    return JSONResponse(content={
        "weights": DEFAULT_WEIGHTS,
        "description": {
            "ice_presence": "How likely is ice at this location (SWIM + landform)",
            "ice_depth": "How shallow is the ice (SWIM depth + thermal inertia)",
            "excavation": "How easy is it to dig (thermal inertia + slope)",
            "landing": "How safe for landing/traversal (elevation + slope + roughness)",
        },
    })


@router.get("/layers")
def get_available_layers():
    """Return status of all data layers used by the algorithm."""
    pipeline = _get_pipeline()
    pipeline._ensure_loaded()

    def _status(geo) -> dict[str, object]:
        if geo is None:
            return {"loaded": False, "error": "not initialised"}
        if not getattr(geo, "loaded", False):
            return {"loaded": False, "error": getattr(geo, "error", "unknown")}
        return {
            "loaded": True,
            "rows": getattr(geo, "rows", 0),
            "cols": getattr(geo, "cols", 0),
        }

    return JSONResponse(content={
        "mola_elevation": _status(pipeline._mola_elev),
        "mola_slope": _status(pipeline._mola_slope),
        "mola_tri": _status(pipeline._mola_tri),
        "swim_consistency_0_1m": _status(pipeline._swim_consistency),
        "swim_depth_0_1m": _status(pipeline._swim_0_1m),
        "swim_depth_1_5m": _status(pipeline._swim_1_5m),
        "swim_depth_5m_plus": _status(pipeline._swim_5m_plus),
        "tes_thermal_inertia": {
            "loaded": pipeline._tes_ti_grid is not None,
            "shape": list(pipeline._tes_ti_grid.shape) if pipeline._tes_ti_grid is not None else None,
        },
    })
