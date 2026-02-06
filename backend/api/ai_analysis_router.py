"""
AI Analysis Mode – evidence gathering + Gemini reasoning.

Phase 1 (Lite): aggregates existing instrument data around a pin location,
then sends the compiled evidence stack to Gemini for scientific analysis.
"""

import json
import logging
import math
import os
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .footprints_router import (
    load_geojson_index,
    feature_intersects_bbox,
    compute_centroid,
    normalize_lon,
)
from .terrain_router import compute_slope_stats, MARS_MEAN_RADIUS
from .mars_regions import find_nearest_region
from .fieldnotes_router import _load as load_field_notes
from .gemini_parser import get_gemini_client, GEMINI_MODELS

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ai_analysis", tags=["AI Analysis"])

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(BASE_DIR, "cache", "ai_evidence")
os.makedirs(CACHE_DIR, exist_ok=True)

# Instruments to scan for evidence
EVIDENCE_INSTRUMENTS = ["crism", "hirise_dtm", "sharad", "sharad_highres", "ctx"]

# =========================================================================
# Gemini Prompt Templates (hard-coded)
# =========================================================================

GEMINI_SYSTEM_PROMPT = """\
You are a planetary science research assistant specializing in
Mars subsurface radar (SHARAD), spectroscopy (CRISM),
and topographic analysis (MOLA / HiRISE DTM).

Your role is to:
- Reason ONLY from the provided evidence.
- Synthesize multiple datasets into a coherent scientific assessment.
- Clearly separate observations, interpretations, and uncertainties.

CRITICAL RULES:
- Do NOT fabricate data, products, measurements, or coverage.
- If evidence is missing or insufficient, explicitly say so.
- Do NOT present speculative interpretations as facts.
- Do NOT output numerical certainty or probabilities.
- Use cautious, scientific language suitable for a discussion section of a paper."""

GEMINI_USER_TEMPLATE = """\
We are analyzing a specific location on Mars.

The user's question is:

"{question}"

Below is a structured Evidence Stack automatically compiled from MarsLab.
This includes only data that actually exist within the specified radius.

You MUST base your reasoning strictly on this evidence.

========================
EVIDENCE STACK (JSON)
========================
{evidence_json}
========================

Analyze the question using the evidence above."""

GEMINI_OUTPUT_INSTRUCTIONS = """

Follow this exact structure in your response:

### Summary
(3-5 sentences)

### Supporting Evidence
(bulleted list, explicitly referencing datasets)

### Ambiguities and Uncertainties
(bulleted list)

### Interpretation Range
(what is plausible vs not supported)

### Recommended Next Checks
(optional; MarsLab-relevant only)

Additional constraints:
- If SHARAD boundary depth is mentioned, state explicitly that
  depth depends on assumed dielectric constant.
- If CRISM coverage is absent or sparse, do NOT infer mineralogy.
- If evidence is insufficient, say "insufficient evidence" and stop."""


# =========================================================================
# Request Models
# =========================================================================

class EvidenceRequest(BaseModel):
    lat: float
    lon: float
    radius_km: float = 10.0


class AskRequest(BaseModel):
    question: str
    evidence: dict


# =========================================================================
# Helpers
# =========================================================================

def _haversine_scalar(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance on Mars (meters) between two points."""
    lat1_r, lon1_r = math.radians(lat1), math.radians(lon1)
    lat2_r, lon2_r = math.radians(lat2), math.radians(lon2)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    return MARS_MEAN_RADIUS * 2 * math.asin(min(1.0, math.sqrt(a)))


def _bbox_from_radius(lat: float, lon: float, radius_km: float):
    """Compute a -180..180 bbox from a center point and radius in km."""
    radius_deg = radius_km / (math.pi * MARS_MEAN_RADIUS / 180 / 1000)
    cos_lat = max(math.cos(math.radians(lat)), 0.01)
    min_lat = max(lat - radius_deg, -90)
    max_lat = min(lat + radius_deg, 90)
    min_lon = normalize_lon(lon - radius_deg / cos_lat)
    max_lon = normalize_lon(lon + radius_deg / cos_lat)
    return min_lon, min_lat, max_lon, max_lat


def _cache_key(lat: float, lon: float, radius_km: float) -> str:
    lat_r = round(lat, 2)
    lon_r = round(lon, 2)
    return f"{lat_r}_{lon_r}_{radius_km}"


# =========================================================================
# Endpoint A: Gather Evidence
# =========================================================================

@router.post("/evidence")
async def gather_evidence(req: EvidenceRequest):
    """Compile an evidence stack around a pin location."""
    lat, lon_180 = req.lat, normalize_lon(req.lon)
    lon_360 = lon_180 if lon_180 >= 0 else lon_180 + 360
    radius_km = max(0.5, min(req.radius_km, 100))

    # Check cache
    key = _cache_key(lat, lon_180, radius_km)
    cache_path = os.path.join(CACHE_DIR, f"{key}.json")
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            return JSONResponse(content=json.load(f))

    # Compute bbox
    min_lon, min_lat, max_lon, max_lat = _bbox_from_radius(lat, lon_180, radius_km)
    crosses_am = min_lon > max_lon

    # --- Instrument footprints ---
    instruments_evidence: dict = {}
    all_products: list = []

    for instrument in EVIDENCE_INSTRUMENTS:
        try:
            geojson = load_geojson_index(instrument)
        except Exception:
            instruments_evidence[instrument] = {"count": 0, "products": []}
            continue

        features = geojson.get("features", [])
        matched = []
        for feat in features:
            if feature_intersects_bbox(feat, min_lon, min_lat, max_lon, max_lat, crosses_am):
                props = feat.get("properties", {})
                product_id = props.get("product_id", "")
                centroid = compute_centroid(feat)
                center = None
                if centroid and centroid.get("coordinates"):
                    c = centroid["coordinates"]
                    center = {"lon": c[0], "lat": c[1]}
                entry = {
                    "product_id": product_id,
                    "instrument": instrument.upper(),
                    "center": center,
                }
                matched.append(entry)
                all_products.append(entry)

        instruments_evidence[instrument] = {
            "count": len(matched),
            "products": matched[:50],  # cap per instrument
        }

    # --- MOLA ---
    mola_data = None
    try:
        mola_data = compute_slope_stats(lat, lon_180, radius_km * 1000)
    except Exception as e:
        logger.warning(f"MOLA slope stats failed: {e}")

    # --- Region ---
    region_info = None
    region = find_nearest_region(lat, lon_360)
    if region:
        region_info = {
            "name": region.name,
            "description": region.description,
            "distance_deg": round(
                math.sqrt((lat - region.lat) ** 2 + (lon_360 - region.lon) ** 2), 2
            ),
        }

    # --- Field Notes ---
    nearby_notes = []
    try:
        notes = load_field_notes()
        for n in notes:
            dist_m = _haversine_scalar(lat, lon_180, n.get("lat", 0), n.get("lon", 0))
            if dist_m <= radius_km * 1000:
                nearby_notes.append({
                    "product_id": n.get("product_id"),
                    "instrument": n.get("instrument"),
                    "tags": n.get("tags", []),
                    "memo": n.get("memo", ""),
                    "distance_km": round(dist_m / 1000, 2),
                })
    except Exception as e:
        logger.warning(f"Field notes load failed: {e}")

    # --- Data gaps ---
    data_gaps = [inst for inst, ev in instruments_evidence.items() if ev["count"] == 0]

    # Build evidence dict
    evidence = {
        "pin": {"lat": lat, "lon": lon_180},
        "radius_km": radius_km,
        "region": region_info,
        "mola": mola_data,
        "instruments": instruments_evidence,
        "field_notes": nearby_notes,
        "data_gaps": data_gaps,
        "total_products": len(all_products),
    }

    # Write cache
    try:
        with open(cache_path, "w") as f:
            json.dump(evidence, f)
    except Exception:
        pass

    return JSONResponse(content=evidence)


# =========================================================================
# Endpoint B: Ask Gemini
# =========================================================================

@router.post("/ask")
async def ask_gemini(req: AskRequest):
    """Send question + evidence to Gemini and return analysis."""
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required")

    evidence = req.evidence
    if not evidence:
        raise HTTPException(status_code=400, detail="Evidence is required")

    client = get_gemini_client()
    if client is None:
        raise HTTPException(
            status_code=503,
            detail="Gemini API not available. Please set GEMINI_API_KEY in backend/.env",
        )

    # Build prompt
    evidence_json = json.dumps(evidence, indent=2)
    user_prompt = GEMINI_USER_TEMPLATE.format(
        question=question,
        evidence_json=evidence_json,
    )
    full_prompt = GEMINI_SYSTEM_PROMPT + "\n\n" + user_prompt + GEMINI_OUTPUT_INSTRUCTIONS

    # Call Gemini with model fallback
    last_error = None
    for model_name in GEMINI_MODELS:
        try:
            logger.info(f"AI Analysis calling Gemini model: {model_name}")
            response = client.models.generate_content(
                model=model_name,
                contents=[{"role": "user", "parts": [{"text": full_prompt}]}],
                config={
                    "temperature": 0.3,
                    "max_output_tokens": 4096,
                },
            )

            answer_text = response.text.strip()

            # Extract highlights from markdown sections
            highlights = _extract_highlights(answer_text)

            return JSONResponse(content={
                "answer_markdown": answer_text,
                "highlights": highlights,
                "model_used": model_name,
            })

        except Exception as e:
            last_error = str(e)
            logger.warning(f"Gemini model {model_name} failed for analysis: {e}")
            continue

    raise HTTPException(
        status_code=502,
        detail=f"All Gemini models failed. Last error: {last_error}",
    )


def _extract_highlights(markdown: str) -> dict:
    """Extract key highlights from the structured markdown response."""
    supports = []
    uncertainties = []
    actions = []

    current_section = None
    for line in markdown.split("\n"):
        stripped = line.strip()
        lower = stripped.lower()

        if lower.startswith("### supporting evidence"):
            current_section = "supports"
        elif lower.startswith("### ambiguities") or lower.startswith("### uncertainties"):
            current_section = "uncertainties"
        elif lower.startswith("### recommended next"):
            current_section = "actions"
        elif lower.startswith("### "):
            current_section = None
        elif stripped.startswith("- ") and current_section:
            text = stripped[2:].strip()
            if current_section == "supports":
                supports.append(text)
            elif current_section == "uncertainties":
                uncertainties.append(text)
            elif current_section == "actions":
                actions.append(text)

    return {
        "supports": supports,
        "uncertainties": uncertainties,
        "actions": actions,
    }
