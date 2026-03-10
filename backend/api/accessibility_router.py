"""API router for Mars ISRU Accessibility algorithm.

Endpoints:
  GET /api/accessibility/score    — Point query with optional custom weights
  GET /api/accessibility/explain  — Score + LLM natural-language explanation
  GET /api/accessibility/tile     — Heatmap tile for map overlay
  GET /api/accessibility/layer-tile — Individual layer tiles
  GET /api/accessibility/weights  — Current default weights
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict
from typing import Optional

import requests as http_requests

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse, Response

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/accessibility", tags=["ISRU Accessibility"])

_pipeline: object | None = None
_tile_cache: dict[str, bytes] = {}  # LRU-like cache, max 500 entries


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from analysis.accessibility.pipeline import AccessibilityPipeline
        _pipeline = AccessibilityPipeline()
    return _pipeline


def _parse_weights(
    w_ice_landform: Optional[float],
    w_water_mineral: Optional[float],
    w_surface_ice: Optional[float],
    w_excavation: Optional[float],
    w_landing: Optional[float],
) -> Optional[dict[str, float]]:
    """Build weights dict from query params. Returns None if all defaults."""
    vals = [w_ice_landform, w_water_mineral, w_surface_ice, w_excavation, w_landing]
    if all(v is None for v in vals):
        return None
    w: dict[str, float] = {}
    if w_ice_landform is not None:
        w["ice_landform"] = w_ice_landform
    if w_water_mineral is not None:
        w["water_mineral"] = w_water_mineral
    if w_surface_ice is not None:
        w["surface_ice"] = w_surface_ice
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
    landform: Optional[str] = Query(None, description="Landform class: LDA, LVF, CCF, SCT, OTHER"),
    landform_confidence: float = Query(1.0, ge=0, le=1),
    w_ice_landform: Optional[float] = Query(None, ge=0, le=1),
    w_water_mineral: Optional[float] = Query(None, ge=0, le=1),
    w_surface_ice: Optional[float] = Query(None, ge=0, le=1),
    w_excavation: Optional[float] = Query(None, ge=0, le=1),
    w_landing: Optional[float] = Query(None, ge=0, le=1),
):
    """Query ISRU accessibility score at a single point."""
    pipeline = _get_pipeline()
    weights = _parse_weights(w_ice_landform, w_water_mineral, w_surface_ice, w_excavation, w_landing)
    valid_landforms = {"LDA", "LVF", "CCF", "SCT", "OTHER"}
    lf = landform.upper() if landform else None
    if lf and lf not in valid_landforms:
        lf = None

    try:
        result = pipeline.query_point(
            lat=lat, lon=lon, weights=weights,
            landform=lf, landform_confidence=landform_confidence,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Accessibility computation failed: {exc}")

    return JSONResponse(content=asdict(result), headers={"Cache-Control": "public, max-age=86400"})

# --------------------------------------------------------------------------
# Tile endpoints
# --------------------------------------------------------------------------

@router.get("/tile/{z}/{x}/{y}.png")
def get_accessibility_tile(
    z: int,
    x: int,
    y: int,
    w_ice_landform: Optional[float] = Query(None, ge=0, le=1),
    w_water_mineral: Optional[float] = Query(None, ge=0, le=1),
    w_surface_ice: Optional[float] = Query(None, ge=0, le=1),
    w_excavation: Optional[float] = Query(None, ge=0, le=1),
    w_landing: Optional[float] = Query(None, ge=0, le=1),
):
    """Render accessibility heatmap tile as PNG."""
    # Build cache key from tile coordinates and weights
    weights = _parse_weights(w_ice_landform, w_water_mineral, w_surface_ice, w_excavation, w_landing)
    weights_str = str(sorted(weights.items())) if weights else "default"
    cache_key = f"accessibility_{z}_{x}_{y}_{weights_str}"

    # Check cache first
    if cache_key in _tile_cache:
        return Response(content=_tile_cache[cache_key], media_type="image/png", headers={"Cache-Control": "public, max-age=86400"})

    pipeline = _get_pipeline()

    try:
        png_bytes = pipeline.get_tile(z=z, x=x, y=y, weights=weights)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Tile render failed: {exc}")

    if png_bytes is None:
        raise HTTPException(status_code=404, detail="No data for this tile")

    # Store in cache (simple LRU: remove oldest if cache exceeds 500 entries)
    png_bytes = bytes(png_bytes)
    if len(_tile_cache) >= 500:
        oldest_key = next(iter(_tile_cache))
        del _tile_cache[oldest_key]
    _tile_cache[cache_key] = png_bytes

    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@router.get("/layer-tile/{layer}/{z}/{x}/{y}.png")
def get_layer_tile(
    layer: str,
    z: int,
    x: int,
    y: int,
):
    """Render an individual data layer tile (for debug / layer panel).

    Supported layers: mola_elevation, mola_slope, mola_tri
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
        headers={"Cache-Control": "public, max-age=86400"},
    )


# --------------------------------------------------------------------------
# Metadata
# --------------------------------------------------------------------------

@router.get("/weights")
def get_default_weights():
    """Return the default ISRU sub-score weights."""
    from analysis.accessibility.algorithm import DEFAULT_WEIGHTS
    return JSONResponse(content={
        "weights": DEFAULT_WEIGHTS,
        "description": {
            "ice_landform": "Ice-rich landform indicator from HiRISE classification (LDA/LVF/CCF/SCT)",
            "water_mineral": "Water-related mineral signal from CRISM classification",
            "surface_ice": "Surface H2O ice detection from CRISM",
            "excavation": "How easy to dig (based on TES thermal inertia)",
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
        "tes_thermal_inertia": {
            "loaded": pipeline._tes_ti_grid is not None,
            "shape": list(pipeline._tes_ti_grid.shape) if pipeline._tes_ti_grid is not None else None,
        },
    })


# --------------------------------------------------------------------------
# Explain endpoint (LLM-powered natural-language explanation)
# --------------------------------------------------------------------------

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.1-8b-instant"


def _build_explain_prompt(result: dict) -> str:
    """Build a concise prompt for the LLM from ISRU accessibility result data."""
    inputs = result.get("inputs", {})
    score = result.get("score", 0)
    il = result.get("ice_landform", 0)
    wm = result.get("water_mineral", 0)
    si = result.get("surface_ice", 0)
    ex = result.get("excavation", 0)
    la = result.get("landing", 0)
    lat = result.get("lat", 0)
    lon = result.get("lon", 0)

    return f"""You are a Mars exploration scientist explaining an ISRU accessibility score to a researcher.

Location: ({lat:.2f}°, {lon:.2f}°)
Overall ISRU Score: {score:.0%}

Sub-scores:
- Ice-Related Landform: {il:.0%}
- Water-Related Mineral: {wm:.0%}
- Surface Ice Signal: {si:.0%}
- Excavation Feasibility: {ex:.0%}
- Landing Safety: {la:.0%}

Raw sensor data:
- TES Thermal Inertia: {inputs.get('thermal_inertia', 'N/A')} TIU
- MOLA Elevation: {inputs.get('elevation', 'N/A')} m
- MOLA Slope: {inputs.get('slope', 'N/A')}°
- Terrain Roughness Index: {inputs.get('tri', 'N/A')} m

Reference thresholds:
- TI < 150 = fine regolith (easy dig), > 300 = consolidated rock (hard dig)
- Slope < 5° = flat (safe), > 15° = steep (risky)
- Elevation < 0m = thicker atmosphere (better for landing)
- TRI < 50m = smooth, > 500m = very rough

Write 2-3 concise sentences explaining WHY this location got this score. Be specific about which factors help and which hurt. Use plain language a researcher would appreciate — no jargon dumps, no bullet points. If a value is N/A, skip it."""

def _call_groq(prompt: str) -> str | None:
    """Call Groq LLM for explanation. Returns None on failure."""
    if not GROQ_API_KEY:
        return None
    try:
        resp = http_requests.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.4,
                "max_tokens": 256,
            },
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        logger.warning("Groq explanation failed: %s", exc)
        return None


def _fallback_explanation(result: dict) -> str:
    """Template-based fallback when LLM is unavailable."""
    inputs = result.get("inputs", {})
    score = result.get("score", 0)
    parts: list[str] = []

    ti = inputs.get("thermal_inertia")
    if isinstance(ti, (int, float)):
        if ti < 150:
            parts.append(f"Low thermal inertia ({ti:.0f} TIU) suggests fine regolith — easy to excavate.")
        elif ti > 300:
            parts.append(f"High thermal inertia ({ti:.0f} TIU) indicates consolidated rock — excavation would be difficult.")
        else:
            parts.append(f"Moderate thermal inertia ({ti:.0f} TIU) — mixed soil conditions.")

    slope = inputs.get("slope")
    if isinstance(slope, (int, float)):
        if slope > 15:
            parts.append(f"Steep terrain ({slope:.1f}°) makes landing risky.")
        elif slope > 5:
            parts.append(f"Moderate slope ({slope:.1f}°) — landing is feasible but not ideal.")

    elev = inputs.get("elevation")
    if isinstance(elev, (int, float)) and elev > 2000:
        parts.append(f"High elevation ({elev:.0f}m) reduces atmospheric braking for landing.")

    if not parts:
        if score >= 0.6:
            return "This location shows generally favorable conditions for surface operations."
        return "This location has limited accessibility for surface operations."

    return " ".join(parts)


@router.get("/explain")
def get_accessibility_explanation(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    landform: Optional[str] = Query(None, description="Landform class: LDA, LVF, CCF, SCT, OTHER"),
    landform_confidence: float = Query(1.0, ge=0, le=1),
    w_ice_landform: Optional[float] = Query(None, ge=0, le=1),
    w_water_mineral: Optional[float] = Query(None, ge=0, le=1),
    w_surface_ice: Optional[float] = Query(None, ge=0, le=1),
    w_excavation: Optional[float] = Query(None, ge=0, le=1),
    w_landing: Optional[float] = Query(None, ge=0, le=1),
):
    """ISRU score + LLM natural-language explanation for a point."""
    pipeline = _get_pipeline()
    weights = _parse_weights(w_ice_landform, w_water_mineral, w_surface_ice, w_excavation, w_landing)
    valid_landforms = {"LDA", "LVF", "CCF", "SCT", "OTHER"}
    lf = landform.upper() if landform else None
    if lf and lf not in valid_landforms:
        lf = None

    try:
        result = pipeline.query_point(
            lat=lat, lon=lon, weights=weights,
            landform=lf, landform_confidence=landform_confidence,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Computation failed: {exc}")

    result_dict = asdict(result)

    # Try LLM, fall back to template
    prompt = _build_explain_prompt(result_dict)
    explanation = _call_groq(prompt)
    if explanation is None:
        explanation = _fallback_explanation(result_dict)

    result_dict["explanation"] = explanation
    return JSONResponse(content=result_dict)

# ── Landform cache search (class-based) ─────────────────────────────


@router.get("/landform-cache/search")
async def search_landform_cache(
    dominant_class: Optional[str] = Query(None, description="Filter by class: LDA, LVF, CCF, SCT, OTHER"),
    lat_min: Optional[float] = Query(None, description="Southern bound"),
    lat_max: Optional[float] = Query(None, description="Northern bound"),
    lon_min: Optional[float] = Query(None, description="Western bound"),
    lon_max: Optional[float] = Query(None, description="Eastern bound"),
    limit: int = Query(100, ge=1, le=1000),
):
    """Search the HiRISE landform classification cache by class and/or bounds."""
    pipeline = _get_pipeline()
    cache = pipeline.landform_cache
    if cache is None:
        return JSONResponse(content={"count": 0, "entries": [], "error": "Landform cache not initialized"})

    # Get entries — filter by bounds if provided
    if lat_min is not None and lat_max is not None and lon_min is not None and lon_max is not None:
        entries = cache.get_entries_in_bounds(lat_min, lat_max, lon_min, lon_max)
    else:
        entries = cache.all_entries()

    # Filter by class
    if dominant_class:
        target = dominant_class.upper()
        entries = [e for e in entries if e.dominant_class == target]

    # Sort by confidence descending
    entries.sort(key=lambda e: e.confidence, reverse=True)
    entries = entries[:limit]

    return JSONResponse(content={
        "count": len(entries),
        "entries": [
            {
                "product_id": e.product_id,
                "lat": round(e.lat, 4),
                "lon": round(e.lon, 4),
                "dominant_class": e.dominant_class,
                "confidence": round(e.confidence, 3),
                "model_version": e.model_version,
                "classified_at": e.classified_at,
            }
            for e in entries
        ],
    })
