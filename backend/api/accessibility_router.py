"""API router for the Mars Ice Accessibility algorithm.

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


# --------------------------------------------------------------------------
# Explain endpoint (LLM-powered natural-language explanation)
# --------------------------------------------------------------------------

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.1-8b-instant"


def _build_explain_prompt(result: dict) -> str:
    """Build a concise prompt for the LLM from accessibility result data."""
    inputs = result.get("inputs", {})
    score = result.get("score", 0)
    ip = result.get("ice_presence", 0)
    id_ = result.get("ice_depth", 0)
    ex = result.get("excavation", 0)
    la = result.get("landing", 0)
    lat = result.get("lat", 0)
    lon = result.get("lon", 0)

    return f"""You are a Mars exploration scientist explaining an ice accessibility score to a researcher.

Location: ({lat:.2f}°, {lon:.2f}°)
Overall Score: {score:.0%}

Sub-scores:
- Ice Presence: {ip:.0%}
- Ice Depth: {id_:.0%}
- Excavation Feasibility: {ex:.0%}
- Landing Safety: {la:.0%}

Raw sensor data:
- SWIM ice consistency: {inputs.get('swim_consistency', 'N/A')}
- SWIM depth 0-1m: {inputs.get('swim_0_1m', 'N/A')}
- SWIM depth 1-5m: {inputs.get('swim_1_5m', 'N/A')}
- SWIM depth >5m: {inputs.get('swim_5m_plus', 'N/A')}
- TES Thermal Inertia: {inputs.get('thermal_inertia', 'N/A')} TIU
- MOLA Elevation: {inputs.get('elevation', 'N/A')} m
- MOLA Slope: {inputs.get('slope', 'N/A')}°
- Terrain Roughness Index: {inputs.get('tri', 'N/A')} m

Reference thresholds:
- TI < 150 = fine regolith (easy dig), > 300 = consolidated rock (hard dig)
- Slope < 5° = flat (safe), > 15° = steep (risky)
- Elevation < 0m = thicker atmosphere (better for landing)
- TRI < 50m = smooth, > 500m = very rough
- SWIM values near +1 = strong ice signal, near -1 = no ice

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

    swim = inputs.get("swim_consistency")
    if isinstance(swim, (int, float)):
        if swim > 0.5:
            parts.append("Strong SWIM ice signal — high confidence ice is present.")
        elif swim > 0:
            parts.append("Weak but positive ice signal from SWIM data.")
        elif swim <= 0:
            parts.append("No significant ice signal from SWIM.")

    elev = inputs.get("elevation")
    if isinstance(elev, (int, float)) and elev > 2000:
        parts.append(f"High elevation ({elev:.0f}m) reduces atmospheric braking for landing.")

    if not parts:
        if score >= 0.6:
            return "This location shows generally favorable conditions for ice access."
        return "This location has limited accessibility for ice extraction."

    return " ".join(parts)


@router.get("/explain")
def get_accessibility_explanation(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    w_ice: Optional[float] = Query(None, ge=0, le=1),
    w_depth: Optional[float] = Query(None, ge=0, le=1),
    w_excavation: Optional[float] = Query(None, ge=0, le=1),
    w_landing: Optional[float] = Query(None, ge=0, le=1),
):
    """Score + LLM natural-language explanation for a point."""
    pipeline = _get_pipeline()
    weights = _parse_weights(w_ice, w_depth, w_excavation, w_landing)

    try:
        result = pipeline.query_point(lat=lat, lon=lon, weights=weights)
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
