"""
MARVIS Chat — Grounded, tool-calling chat endpoint (Llama 3.1 8B only).

Cascade: Groq (llama-3.1-8b-instant) → LLaMA local fallback.

POST /api/marvis/chat  →  SSE stream of response tokens + tool_call events.
"""

import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from api.agent_orchestrator import _sanitize_respond

_backend_dir = Path(__file__).parent.parent
load_dotenv(_backend_dir / ".env")

# ── Mineral score data (lazy-loaded) ─────────────────────────────────

_score_stats: dict | None = None
_crism_coords: dict | None = None  # obs_id -> (lat, lon, product_id)


def _load_score_stats() -> dict:
    global _score_stats
    if _score_stats is None:
        p = _backend_dir / "crism_score" / "score_stats.json"
        if p.exists():
            with open(p) as fp:
                _score_stats = json.load(fp)
        else:
            _score_stats = {}
    return _score_stats


def _load_crism_coords() -> dict:
    """Build obs_id -> (lat, lon, product_id) lookup from CRISM GeoJSON index."""
    global _crism_coords
    if _crism_coords is not None:
        return _crism_coords
    _crism_coords = {}
    try:
        from api.registry import get_registry
        idx = get_registry().load_index("crism")
        for f in idx.get("features", []):
            pid = (f.get("properties") or {}).get("product_id", "")
            m = re.match(r"^([a-z]{3}[0-9a-f]+)", pid.lower())
            if not m:
                continue
            obs_id = m.group(1)
            geom = f.get("geometry") or {}
            if geom.get("type") == "Polygon":
                ring = geom["coordinates"][0]
                lat = sum(c[1] for c in ring) / len(ring)
                lon = sum(c[0] for c in ring) / len(ring)
            elif geom.get("type") == "Point":
                lon, lat = geom["coordinates"][:2]
            else:
                continue
            _crism_coords[obs_id] = (lat, lon, pid)
    except Exception as exc:
        logger.error(f"Failed to load CRISM coords: {exc}")
    return _crism_coords


def search_mineral_products(
    mineral_type: str = "ice",
    min_percent: float = 10.0,
    limit: int = 10,
) -> list[dict]:
    """Search score_stats.json for CRISM observations exceeding a mineral threshold.

    Returns list sorted by percentage descending:
        [{product_id, obs_id, lat, lon, ice_percent, hyd_percent}]
    """
    stats = _load_score_stats()
    coords = _load_crism_coords()

    # Normalize mineral_type
    mt = mineral_type.lower()
    if mt in ("h2o", "water", "water ice", "ice"):
        key = "ice"
    else:
        key = "hyd"

    results = []
    for obs_id, entry in stats.items():
        section = entry.get(key)
        if not section:
            continue
        vp = section.get("valid_pixels", 0)
        if vp == 0:
            continue
        above = section.get("threshold_counts", {}).get("0.3", 0)
        pct = above / vp * 100.0

        if pct < min_percent:
            continue

        # Get coordinates
        coord = coords.get(obs_id)
        if not coord:
            continue
        lat, lon, pid = coord

        # Also compute the other type
        other_key = "hyd" if key == "ice" else "ice"
        other_section = entry.get(other_key, {})
        other_vp = other_section.get("valid_pixels", 0)
        other_above = other_section.get("threshold_counts", {}).get("0.3", 0)
        other_pct = (other_above / other_vp * 100.0) if other_vp else 0.0

        results.append({
            "product_id": pid,
            "obs_id": obs_id,
            "lat": round(lat, 3),
            "lon": round(lon, 3),
            "ice_percent": round(pct if key == "ice" else other_pct, 2),
            "hyd_percent": round(pct if key == "hyd" else other_pct, 2),
        })

    results.sort(key=lambda r: r["ice_percent" if key == "ice" else "hyd_percent"], reverse=True)
    return results[:limit]


# ── Landform cache (lazy-loaded, indexed by type + latitude band) ────

_landform_by_type_band: dict[str, dict[int, list]] | None = None
LANDFORM_PROXIMITY_KM = 50.0

LANDFORM_TYPES = ["crater", "terraced_crater", "volcanic", "graben", "channel", "wrinkle_ridge", "lda"]


def _load_landform_index() -> dict[str, dict[int, list]]:
    """Lazy-load landform features and build type + 5-deg latitude-band index."""
    global _landform_by_type_band
    if _landform_by_type_band is not None:
        return _landform_by_type_band

    _landform_by_type_band = {}
    cache_dir = _backend_dir / "cache"
    for fname in ("landforms_precomputed.json", "landforms_progress.json"):
        p = cache_dir / fname
        if p.exists():
            try:
                with open(p) as fp:
                    data = json.load(fp)
                features = data.get("features", [])
                for lf in features:
                    t = lf.get("type", "")
                    lat = lf.get("lat", 0.0)
                    band = int(lat // 5)
                    _landform_by_type_band.setdefault(t, {}).setdefault(band, []).append(lf)
                logging.getLogger(__name__).info(
                    "Loaded %d landforms from %s (%d types)",
                    len(features), fname, len(_landform_by_type_band),
                )
            except Exception as exc:
                logging.getLogger(__name__).error("Failed to load landform cache: %s", exc)
            break
    return _landform_by_type_band


def _find_nearest_landform(
    lat: float, lon: float, landform_type: str, radius_km: float = LANDFORM_PROXIMITY_KM,
) -> tuple[dict | None, float | None]:
    """Find the nearest landform of given type within radius_km. Returns (feature, dist_km)."""
    from api.proximity_router import haversine_km

    index = _load_landform_index()
    type_bands = index.get(landform_type, {})
    if not type_bands:
        return None, None

    center_band = int(lat // 5)
    candidates = []
    for offset in (-1, 0, 1):
        candidates.extend(type_bands.get(center_band + offset, []))

    best, best_dist = None, radius_km
    for lf in candidates:
        # Quick degree pre-filter (~60 km per degree on Mars)
        if abs(lf["lat"] - lat) > 1.0 or abs(lf.get("lon", 0) - lon) > 1.5:
            continue
        d = haversine_km(lat, lon, lf["lat"], lf.get("lon", 0))
        if d < best_dist:
            best_dist = d
            best = lf
    return (best, round(best_dist, 1)) if best else (None, None)


# ── Compound search pipeline ─────────────────────────────────────────

def compound_search(
    instrument: str,
    intersect_with: str | None = None,
    mineral_type: str | None = None,
    min_percent: float | None = None,
    near_landform: str | None = None,
    viewport_lat: float | None = None,
    viewport_lon: float | None = None,
    limit: int = 10,
) -> list[dict]:
    """Unified search: instrument + optional mineral/landform/intersection filters."""
    from api.registry import get_registry
    from api.proximity_router import (
        haversine_km,
        _bbox_from_geometry,
        _centroid_from_geometry,
        _geometries_overlap,
    )

    _logger = logging.getLogger(__name__)
    registry = get_registry()

    # Normalize mineral_type
    mineral_key = None
    if mineral_type:
        mt = mineral_type.lower()
        mineral_key = "ice" if mt in ("h2o", "water", "water ice", "ice") else "hyd"
        if min_percent is None:
            min_percent = 10.0

    # Determine which instrument gets the mineral filter
    crism_family = {"CRISM", "CRISM_TRR3"}
    mineral_on_primary = mineral_key and instrument.upper() in crism_family
    mineral_on_secondary = mineral_key and intersect_with and intersect_with.upper() in crism_family

    # ── Step 1: Load primary instrument features ──
    inst_key = instrument.lower().replace("-", "_").replace(" ", "_")
    try:
        idx = registry.load_index(inst_key)
    except Exception:
        _logger.error("Cannot load index for %s", inst_key)
        return []

    features = []
    for f in idx.get("features", []):
        props = f.get("properties") or {}
        geom = f.get("geometry")
        if not geom or not props.get("product_id"):
            continue
        bbox = _bbox_from_geometry(geom)
        centroid = _centroid_from_geometry(geom)
        if not bbox or not centroid:
            continue

        # Viewport pre-filter (±15 deg latitude)
        if viewport_lat is not None:
            if abs(centroid["lat"] - viewport_lat) > 15:
                continue

        features.append({
            "product_id": props["product_id"],
            "instrument": instrument.upper(),
            "lat": round(centroid["lat"], 3),
            "lon": round(centroid["lon"], 3),
            "geom": geom,
            "bbox": bbox,
        })

    # ── Step 2: Mineral filter on primary ──
    if mineral_on_primary:
        passing_obs = set()
        mineral_data = {}
        stats = _load_score_stats()
        coords = _load_crism_coords()
        for obs_id, entry in stats.items():
            section = entry.get(mineral_key, {})
            vp = section.get("valid_pixels", 0)
            if vp == 0:
                continue
            above = section.get("threshold_counts", {}).get("0.3", 0)
            pct = above / vp * 100.0
            if pct >= min_percent:
                passing_obs.add(obs_id)
                other = "hyd" if mineral_key == "ice" else "ice"
                os_ = entry.get(other, {})
                ovp = os_.get("valid_pixels", 0)
                oab = os_.get("threshold_counts", {}).get("0.3", 0)
                opct = (oab / ovp * 100.0) if ovp else 0.0
                mineral_data[obs_id] = {
                    "ice_percent": round(pct if mineral_key == "ice" else opct, 2),
                    "hyd_percent": round(pct if mineral_key == "hyd" else opct, 2),
                }

        filtered = []
        for feat in features:
            pid = feat["product_id"]
            m = re.match(r"^([a-z]{3}[0-9a-f]+)", pid.lower())
            if m and m.group(1) in passing_obs:
                feat.update(mineral_data[m.group(1)])
                filtered.append(feat)
        features = filtered

    # ── Step 3: Landform proximity filter ──
    if near_landform:
        filtered = []
        for feat in features:
            lf, dist = _find_nearest_landform(feat["lat"], feat["lon"], near_landform)
            if lf and dist is not None:
                feat["near_landform_type"] = near_landform
                feat["near_landform_distance_km"] = dist
                filtered.append(feat)
        features = filtered

    # ── Step 4: Intersection filter ──
    if intersect_with:
        sec_key = intersect_with.lower().replace("-", "_").replace(" ", "_")
        try:
            sec_idx = registry.load_index(sec_key)
        except Exception:
            _logger.error("Cannot load index for %s", sec_key)
            return []

        # Build secondary features (with optional mineral pre-filter)
        sec_passing_obs = None
        sec_mineral_data = {}
        if mineral_on_secondary:
            sec_passing_obs = set()
            stats = _load_score_stats()
            for obs_id, entry in stats.items():
                section = entry.get(mineral_key, {})
                vp = section.get("valid_pixels", 0)
                if vp == 0:
                    continue
                above = section.get("threshold_counts", {}).get("0.3", 0)
                pct = above / vp * 100.0
                if pct >= min_percent:
                    sec_passing_obs.add(obs_id)
                    other = "hyd" if mineral_key == "ice" else "ice"
                    os_ = entry.get(other, {})
                    ovp = os_.get("valid_pixels", 0)
                    oab = os_.get("threshold_counts", {}).get("0.3", 0)
                    opct = (oab / ovp * 100.0) if ovp else 0.0
                    sec_mineral_data[obs_id] = {
                        "ice_percent": round(pct if mineral_key == "ice" else opct, 2),
                        "hyd_percent": round(pct if mineral_key == "hyd" else opct, 2),
                    }

        sec_features = []
        for f in sec_idx.get("features", []):
            props = f.get("properties") or {}
            geom = f.get("geometry")
            if not geom or not props.get("product_id"):
                continue
            pid = props["product_id"]

            # Mineral pre-filter on secondary
            if sec_passing_obs is not None:
                m = re.match(r"^([a-z]{3}[0-9a-f]+)", pid.lower())
                if not m or m.group(1) not in sec_passing_obs:
                    continue

            bbox = _bbox_from_geometry(geom)
            centroid = _centroid_from_geometry(geom)
            if not bbox or not centroid:
                continue

            sf = {"product_id": pid, "geom": geom, "bbox": bbox, "centroid": centroid}
            if sec_passing_obs is not None:
                obs = re.match(r"^([a-z]{3}[0-9a-f]+)", pid.lower())
                if obs and obs.group(1) in sec_mineral_data:
                    sf.update(sec_mineral_data[obs.group(1)])
            sec_features.append(sf)

        # Find overlapping pairs
        paired = []
        for feat in features:
            for sf in sec_features:
                if _geometries_overlap(feat["geom"], feat["bbox"], sf["geom"], sf["bbox"]):
                    result = {**feat}
                    result["paired_product"] = sf["product_id"]
                    result["paired_instrument"] = intersect_with.upper()
                    if "ice_percent" in sf:
                        result["ice_percent"] = sf["ice_percent"]
                    if "hyd_percent" in sf:
                        result["hyd_percent"] = sf["hyd_percent"]
                    paired.append(result)
                    break  # One match per primary product
        features = paired

    # ── Step 5: Rank and return ──
    for feat in features:
        feat.pop("geom", None)
        feat.pop("bbox", None)

    if mineral_key and ("ice_percent" in (features[0] if features else {})):
        sort_key = "ice_percent" if mineral_key == "ice" else "hyd_percent"
        features.sort(key=lambda r: r.get(sort_key, 0), reverse=True)
    elif near_landform:
        features.sort(key=lambda r: r.get("near_landform_distance_km", 999))
    elif viewport_lat is not None and viewport_lon is not None:
        features.sort(key=lambda r: abs(r["lat"] - viewport_lat) + abs(r["lon"] - viewport_lon))

    return features[:limit]


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/marvis", tags=["marvis-chat"])

# ── Groq config (primary — llama 3.1 8B, fast + cheap) ──────────────

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_BASE_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.1-8b-instant"


# ── Region lookup ────────────────────────────────────────────────────

def _resolve_region(name: str):
    """Resolve a region name to (lat, lon, display_name) or None."""
    try:
        from api.mars_regions import MARS_REGIONS
        q = name.lower().replace("_", " ")
        for r in MARS_REGIONS.values():
            if q in r.display_name.lower() or q in r.region_id.replace("_", " "):
                return (r.center_lat, r.center_lon, r.display_name)
    except Exception:
        pass
    return None


def _resolve_region_from_coords(lat: float, lon: float) -> str | None:
    """Reverse-lookup: find which Mars region contains (lat, lon)."""
    try:
        from api.mars_regions import MARS_REGIONS
        for r in MARS_REGIONS.values():
            lat_min = min(r.lat_min, r.lat_max)
            lat_max = max(r.lat_min, r.lat_max)
            if lat < lat_min or lat > lat_max:
                continue
            # Handle antimeridian-crossing regions (lon_min > lon_max)
            if r.lon_min <= r.lon_max:
                if r.lon_min <= lon <= r.lon_max:
                    return r.display_name
            else:
                # Wraps antimeridian: lon_min..180 or -180..lon_max
                if lon >= r.lon_min or lon <= r.lon_max:
                    return r.display_name
    except Exception:
        pass
    return None


INSTRUMENT_LIST = ["CRISM", "HIRISE", "SHARAD", "CTX", "SHARAD_HIGHRES", "HIRISE_DTM", "CRISM_TRR3"]


# ── System prompt (context-aware) ────────────────────────────────────

_BASE_SYSTEM = """You are MARVIS, a terse Mars science research assistant embedded in a map-based analysis tool.

WHEN TO USE TOOLS vs TEXT:
- ONLY call fly_to_location when the user explicitly asks to GO TO, NAVIGATE, ZOOM, or CENTER ON a NEW, DIFFERENT location.
- ONLY call load_instrument when the user explicitly asks to LOAD, SHOW, or ENABLE a specific instrument dataset.
- ONLY call select_product when the user asks to PICK, SELECT, OPEN, INSPECT, or SHOW a specific product or a product from a specific instrument (e.g. "pick one SHARAD", "open a HiRISE product", "select that product").
- ONLY call search when the user asks to FIND, SEARCH, or LOCATE products with any combination of: (a) a specific instrument, (b) mineral/ice/hydration content threshold, (c) proximity to landforms (crater, terraced crater, volcano, channel, graben, ridge, LDA), (d) intersection/overlap with another instrument. Do NOT call search for general science questions or selecting visible products.
- For ALL other messages (general science questions, "what products", etc.) → answer with TEXT only. Do NOT call any tool.
- If unsure whether the user wants navigation or information, answer with text.

CRITICAL — DO NOT RE-NAVIGATE:
- The user's CURRENT VIEWPORT is shown below. If the user is already viewing a region, do NOT call fly_to_location to that same region again.
- "find ice", "what do you see", "what products" → these are analysis questions, NOT navigation requests.
- Only navigate when the user names a DIFFERENT destination than where they already are.

STRICT RULES:
1. When you do call a tool, NEVER also narrate the action in text. No "Zooming into...", "Let me navigate...", "Loading CRISM...".
2. Only cite specific SHARAD/CRISM/HiRISE observations if those instruments are listed as LOADED below. Otherwise note it as general background and suggest loading the data.
3. Answer text questions in 1-3 sentences. Be brief, precise, and scientific.
4. Never say "feel free to ask" or similar filler. Never reintroduce yourself.
5. Always respond in English.

SESSION CONTEXT:
{context_block}"""


def _build_system(context: Optional[dict]) -> str:
    if not context:
        return _BASE_SYSTEM.format(context_block="- Current viewport: unknown\n- No instruments loaded.\n- No products visible.")

    loaded = context.get("loaded_instruments") or []
    product_count = context.get("visible_product_count", 0)
    cur_lat = context.get("current_lat")
    cur_lon = context.get("current_lon")

    lines = []

    # Tell the LLM where the user currently is
    if cur_lat is not None and cur_lon is not None:
        region = _resolve_region_from_coords(cur_lat, cur_lon)
        if region:
            lines.append(f"- Current viewport: {region} ({cur_lat:.1f}°, {cur_lon:.1f}°)")
        else:
            lines.append(f"- Current viewport: ({cur_lat:.1f}°, {cur_lon:.1f}°)")
    else:
        lines.append("- Current viewport: unknown (user hasn't navigated yet)")

    if loaded:
        lines.append(f"- Loaded instruments: {', '.join(loaded)}")
    else:
        lines.append("- No instruments loaded yet.")
    lines.append(f"- Visible products in viewport: {product_count}")

    return _BASE_SYSTEM.format(context_block="\n".join(lines))


# ── Anti-hallucination: detect pretend-action phrases ────────────────

_PRETEND_ACTION_RE = re.compile(
    r"(?i)\b(?:zooming|navigating|panning|centering|moving|flying|scrolling)\s+(?:to|into|over|toward|across)"
    r"|\b(?:loading|activating|enabling|opening|pulling up)\s+\w+\s+(?:data|footprints|layer|imagery)"
    r"|\blet me (?:zoom|navigate|pan|center|move|fly|load|activate|pull up|bring up)"
    r"|\bi(?:'m| am) (?:zooming|navigating|loading|pulling|bringing)"
)


def _fix_pretend_actions(text: str, had_tool_call: bool) -> str:
    """If the text contains action narration without a tool call, strip it."""
    if had_tool_call:
        return text  # Tool was actually called, narration is a confirmation

    if _PRETEND_ACTION_RE.search(text):
        # Replace the entire response — the model hallucinated an action
        return "I can't perform that action directly through chat. Could you be more specific about what you'd like? For navigation, tell me a region name. For data, tell me which instrument."

    return text


# ── Request models ───────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: str


class SessionContext(BaseModel):
    loaded_instruments: List[str] = []
    visible_product_count: int = 0
    current_lat: Optional[float] = None
    current_lon: Optional[float] = None


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatMessage]] = None
    context: Optional[SessionContext] = None


MAX_HISTORY_TURNS = 6


def _tool_confirmation(tool: str, params: dict, context: Optional[SessionContext]) -> str:
    """Build a grounded confirmation message after a tool call."""
    loaded = (context.loaded_instruments if context else []) or []

    if tool == "fly_to_location":
        name = params.get("region_name", f"{params.get('lat', '?')}°, {params.get('lon', '?')}°")
        msg = f"Navigating to {name}."
        if not loaded:
            msg += " No instrument layers are loaded yet — want me to load SHARAD, CRISM, or HiRISE for this area?"
        else:
            msg += f" {', '.join(loaded)} footprints will update for this view."
        return msg

    if tool == "load_instrument":
        inst = params.get("instrument", "?")
        msg = f"Loading {inst} footprints."
        if inst in loaded:
            msg = f"{inst} is already loaded — footprints should be visible in the current view."
        return msg

    if tool == "select_product":
        inst = params.get("instrument", "?")
        return f"Opening {inst} product in the inspector panel."

    if tool == "search":
        results = params.get("results", [])
        inst = params.get("instrument", "?")
        intersect = params.get("intersect_with")
        mt = params.get("mineral_type")
        min_pct = params.get("min_percent", 10)
        landform = params.get("near_landform")

        filters = []
        if mt:
            label = "water ice" if mt == "ice" else "hydration"
            filters.append(f">{min_pct:.0f}% {label}")
        if landform:
            filters.append(f"near {landform.replace('_', ' ')}s")
        if intersect:
            filters.append(f"intersecting {intersect}")
        filter_str = " " + ", ".join(filters) if filters else ""

        if results:
            best = results[0]
            msg = f"Found {len(results)} {inst} products{filter_str}. Top: {best['product_id']}"
            if best.get("ice_percent") is not None:
                msg += f" ({best['ice_percent']:.1f}% ice)"
            if best.get("near_landform_distance_km") is not None:
                msg += f" ({best['near_landform_distance_km']:.0f} km from {landform.replace('_', ' ')})"
            msg += f" at {best['lat']:.2f}\u00b0, {best['lon']:.2f}\u00b0. Flying to top result."
            return msg
        return f"No {inst} products found{filter_str}. Try broadening your search."

    return "Done."


# ── Dedup guard: suppress redundant navigation ───────────────────────

_NEAR_THRESHOLD_DEG = 10.0  # ~600 km on Mars — "same region" tolerance


def _is_near_current(params: dict, context: Optional[SessionContext]) -> bool:
    """Return True if fly_to_location target is within threshold of current viewport."""
    if not context or context.current_lat is None or context.current_lon is None:
        return False
    target_lat = params.get("lat")
    target_lon = params.get("lon")
    if target_lat is None or target_lon is None:
        return False
    dlat = abs(target_lat - context.current_lat)
    dlon = abs(target_lon - context.current_lon)
    # Handle antimeridian wrap for longitude
    if dlon > 180:
        dlon = 360 - dlon
    return dlat < _NEAR_THRESHOLD_DEG and dlon < _NEAR_THRESHOLD_DEG


# ── Groq: OpenAI-compatible tool calling (primary) ───────────────────

_GROQ_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "fly_to_location",
            "description": "Navigate/zoom/pan the map to a Mars region or coordinates. Use this for ANY request to go to, zoom into, center on, explore, or look at a location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "region_name": {"type": "string", "description": "Mars region or feature name, e.g. 'Arcadia Planitia', 'Jezero Crater'"},
                    "lat": {"type": "number", "description": "Target latitude (if known)"},
                    "lon": {"type": "number", "description": "Target longitude (if known)"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "load_instrument",
            "description": "Load footprint data for an orbital instrument onto the map. Use this when the user asks to see, load, show, or check data from an instrument.",
            "parameters": {
                "type": "object",
                "properties": {
                    "instrument": {
                        "type": "string",
                        "description": "Instrument to load",
                        "enum": INSTRUMENT_LIST,
                    },
                },
                "required": ["instrument"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "select_product",
            "description": "Pick and open a product from the visible footprints on the map. Use when the user asks to pick, select, open, or inspect a product or a product from a specific instrument.",
            "parameters": {
                "type": "object",
                "properties": {
                    "instrument": {
                        "type": "string",
                        "description": "Instrument to pick a product from",
                        "enum": INSTRUMENT_LIST,
                    },
                },
                "required": ["instrument"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "Search for Mars orbital products with optional filters. "
                "Use when the user asks to FIND, SEARCH, or LOCATE products from an instrument, "
                "optionally filtered by mineral content (ice/h2o/hydration percentage), "
                "by proximity to landforms (crater, terraced crater, volcano, channel, graben, ridge, LDA), "
                "or by intersection/overlap with another instrument. "
                "Examples: 'find CRISM with ice > 10%', 'find HiRISE DTM near terraced craters', "
                "'find SHARAD intersecting CRISM with h2o > 10%'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "instrument": {
                        "type": "string",
                        "description": "Primary instrument to search",
                        "enum": INSTRUMENT_LIST,
                    },
                    "intersect_with": {
                        "type": "string",
                        "description": "Find products overlapping this second instrument. Only use when user says 'intersect', 'overlap', or 'cross'.",
                        "enum": INSTRUMENT_LIST,
                    },
                    "mineral_type": {
                        "type": "string",
                        "description": "Filter by mineral content: 'ice' for water ice/h2o, 'hyd' for hydration/clay. Applies to CRISM/CRISM_TRR3.",
                        "enum": ["ice", "hyd"],
                    },
                    "min_percent": {
                        "type": "number",
                        "description": "Minimum mineral percentage (0-100). Default 10 if mineral_type set but this is missing.",
                    },
                    "near_landform": {
                        "type": "string",
                        "description": "Filter to products near a landform type detected from MOLA topography.",
                        "enum": LANDFORM_TYPES,
                    },
                },
                "required": ["instrument"],
            },
        },
    },
]


async def _stream_groq(
    message: str,
    history: List[ChatMessage],
    context: Optional[SessionContext],
    queue: asyncio.Queue,
) -> bool:
    """Call Groq API with tool calling. Returns True if successful, False to cascade."""
    if not GROQ_API_KEY:
        return False

    import aiohttp

    system = _build_system(context.model_dump() if context else None)

    messages = [{"role": "system", "content": system}]
    for entry in history[-MAX_HISTORY_TURNS:]:
        role = "user" if entry.role == "user" else "assistant"
        messages.append({"role": role, "content": entry.content})
    messages.append({"role": "user", "content": message})

    payload = {
        "model": GROQ_MODEL,
        "messages": messages,
        "tools": _GROQ_TOOLS,
        "tool_choice": "auto",
        "temperature": 0.3,
        "max_tokens": 400,
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                GROQ_BASE_URL,
                json=payload,
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status == 429:
                    logger.warning("Groq rate limited, cascading to local LLaMA")
                    return False
                if resp.status != 200:
                    body = await resp.text()
                    logger.error(f"Groq error {resp.status}: {body[:200]}")
                    return False

                data = await resp.json()

        choice = data.get("choices", [{}])[0]
        msg = choice.get("message", {})

        had_tool_call = False

        # Process tool calls
        tool_calls = msg.get("tool_calls") or []
        for tc in tool_calls:
            fn = tc.get("function", {})
            tool_name = fn.get("name", "")
            try:
                params = json.loads(fn.get("arguments", "{}"))
            except json.JSONDecodeError:
                params = {}

            # Resolve region
            if tool_name == "fly_to_location":
                region_name = params.get("region_name", "")
                # Resolve if region_name provided and lat/lon are missing or zero
                lat_val = params.get("lat")
                needs_resolve = region_name and (lat_val is None or lat_val == 0)
                if needs_resolve:
                    resolved = _resolve_region(region_name)
                    if resolved:
                        params["lat"] = resolved[0]
                        params["lon"] = resolved[1]
                        params["region_name"] = resolved[2]
                    else:
                        msg_text = f"I don't have coordinates for \"{region_name}\". Could you provide a lat/lon or try a more specific name?"
                        await queue.put({"event": "chunk", "data": {"text": msg_text}})
                        await queue.put({"event": "done", "data": {"full_text": msg_text}})
                        await queue.put(None)
                        return True

                # Dedup guard: suppress navigation if already near target
                if _is_near_current(params, context):
                    logger.info(f"Suppressed redundant fly_to_location (already near target)")
                    continue

            # Run compound search server-side and attach results
            if tool_name == "search":
                if params.get("mineral_type") and params.get("min_percent") is None:
                    params["min_percent"] = 10.0
                try:
                    results = compound_search(
                        instrument=params.get("instrument", ""),
                        intersect_with=params.get("intersect_with"),
                        mineral_type=params.get("mineral_type"),
                        min_percent=params.get("min_percent"),
                        near_landform=params.get("near_landform"),
                        viewport_lat=context.current_lat if context else None,
                        viewport_lon=context.current_lon if context else None,
                        limit=10,
                    )
                except Exception as exc:
                    logger.error(f"Compound search error: {exc}")
                    results = []
                params["results"] = results

            had_tool_call = True
            confirm = _tool_confirmation(tool_name, params, context)
            await queue.put({
                "event": "tool_call",
                "data": {"tool": tool_name, "params": params, "message": confirm},
            })

        # Process text content
        text_content = msg.get("content") or ""
        if text_content:
            text_content = _fix_pretend_actions(text_content, had_tool_call)
            clean = _sanitize_respond(text_content) if text_content else text_content
            final = clean or text_content
            await queue.put({"event": "chunk", "data": {"text": final}})
            await queue.put({"event": "done", "data": {"full_text": final}})
        elif had_tool_call:
            await queue.put({"event": "done", "data": {"full_text": ""}})

        await queue.put(None)
        return True

    except Exception as exc:
        logger.error(f"Groq error: {exc}")
        return False


# ── LLaMA fallback (with prompt-based tool calling) ──────────────────

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.1:8b"

# Region keywords for LLaMA prompt-based tool routing
_NAV_RE = re.compile(
    r"(?i)\b(?:go to|fly to|zoom (?:to|into)|navigate to|show me|explore|take me to|center on|look at|move to)\b"
)
_LOAD_RE = re.compile(
    r"(?i)\b(?:load|show|enable|activate|turn on)\s+(?:the\s+)?("
    + "|".join(INSTRUMENT_LIST)
    + r")\b",
)
_SELECT_RE = re.compile(
    r"(?i)\b(?:pick|select|open|inspect|choose)\b.*?\b("
    + "|".join(INSTRUMENT_LIST)
    + r")\b"
    r"|\b(" + "|".join(INSTRUMENT_LIST) + r")\b.*?\b(?:pick|select|open|inspect|choose|panel)\b",
)
# Unified search regex (replaces _INTERSECT_RE and _MINERAL_RE)
_SEARCH_RE = re.compile(
    r"(?i)\b(?:find|search|where|locate)\b"
    r".*?\b(" + "|".join(INSTRUMENT_LIST) + r")\b",
)
_SEARCH_INTERSECT_RE = re.compile(
    r"(?i)\b(?:intersect\w*|overlap\w*|cross\w*)\b"
    r".*?\b(" + "|".join(INSTRUMENT_LIST) + r")\b"
    r"|\b(" + "|".join(INSTRUMENT_LIST) + r")\b"
    r".*?\b(?:intersect\w*|overlap\w*|cross\w*)\b",
)
_SEARCH_LANDFORM_RE = re.compile(
    r"(?i)\b(terraced[_ ]?crater|crater|volcanic|volcano|graben|channel|wrinkle[_ ]?ridge|ridge|lda|lobate[_ ]?debris)\b",
)
_MINERAL_PCT_RE = re.compile(
    r"(?:more\s+than|greater\s+than|above|over|>|>=)\s*(\d+(?:\.\d+)?)\s*%?"
    r"|(\d+(?:\.\d+)?)\s*%",
)
_MINERAL_TYPE_RE = re.compile(
    r"(?i)\b(h2o|ice|water)\b",
)


async def _stream_llama(
    message: str,
    history: List[ChatMessage],
    context: Optional[SessionContext],
    queue: asyncio.Queue,
):
    """LLaMA fallback with regex-based tool dispatch for navigation/instrument loading."""
    import aiohttp

    # ── Check for tool-like intents via regex (since LLaMA lacks function calling)
    nav_match = _NAV_RE.search(message)
    load_match = _LOAD_RE.search(message)
    search_match = _SEARCH_RE.search(message)

    # Unified search: handles mineral, landform, intersection, and compound queries
    if search_match:
        instrument = search_match.group(1).upper()
        params: dict = {"instrument": instrument}

        # Check for intersection with second instrument
        intersect_m = _SEARCH_INTERSECT_RE.search(message)
        if intersect_m:
            second = (intersect_m.group(1) or intersect_m.group(2) or "").upper()
            if second and second != instrument:
                params["intersect_with"] = second

        # Check for landform filter
        landform_m = _SEARCH_LANDFORM_RE.search(message)
        if landform_m:
            lf = landform_m.group(1).lower().replace(" ", "_")
            if lf == "volcano":
                lf = "volcanic"
            elif lf == "ridge":
                lf = "wrinkle_ridge"
            elif lf == "lobate_debris":
                lf = "lda"
            params["near_landform"] = lf

        # Check for mineral filter
        mineral_m = _MINERAL_TYPE_RE.search(message)
        if mineral_m:
            params["mineral_type"] = "ice"
        pct_m = _MINERAL_PCT_RE.search(message)
        if pct_m:
            params["min_percent"] = float(pct_m.group(1) or pct_m.group(2))

        # Only dispatch if there's at least one filter beyond just the instrument
        has_filter = any(k in params for k in ("intersect_with", "mineral_type", "near_landform", "min_percent"))
        if has_filter:
            if params.get("mineral_type") and "min_percent" not in params:
                params["min_percent"] = 10.0
            try:
                results = compound_search(
                    instrument=instrument,
                    intersect_with=params.get("intersect_with"),
                    mineral_type=params.get("mineral_type"),
                    min_percent=params.get("min_percent"),
                    near_landform=params.get("near_landform"),
                    viewport_lat=context.current_lat if context else None,
                    viewport_lon=context.current_lon if context else None,
                    limit=10,
                )
            except Exception as exc:
                logger.error(f"LLaMA compound search error: {exc}")
                results = []
            params["results"] = results
            confirm = _tool_confirmation("search", params, context)
            await queue.put({"event": "tool_call", "data": {
                "tool": "search", "params": params, "message": confirm,
            }})
            await queue.put({"event": "done", "data": {"full_text": ""}})
            await queue.put(None)
            return

    if nav_match:
        # Extract region name from the message (everything after the nav keyword)
        after = message[nav_match.end():].strip().rstrip("?.!,")
        resolved = _resolve_region(after) if after else None
        if resolved:
            params = {"lat": resolved[0], "lon": resolved[1], "region_name": resolved[2]}
            confirm = _tool_confirmation("fly_to_location", params, context)
            await queue.put({"event": "tool_call", "data": {
                "tool": "fly_to_location", "params": params, "message": confirm,
            }})
            await queue.put({"event": "done", "data": {"full_text": ""}})
            await queue.put(None)
            return
        # Can't resolve — let LLaMA respond naturally (it'll say it doesn't know)

    if load_match:
        inst = load_match.group(1).upper()
        params = {"instrument": inst}
        confirm = _tool_confirmation("load_instrument", params, context)
        await queue.put({"event": "tool_call", "data": {
            "tool": "load_instrument", "params": params, "message": confirm,
        }})
        await queue.put({"event": "done", "data": {"full_text": ""}})
        await queue.put(None)
        return

    select_match = _SELECT_RE.search(message)
    if select_match:
        inst = (select_match.group(1) or select_match.group(2) or "").upper()
        if inst:
            params = {"instrument": inst}
            confirm = _tool_confirmation("select_product", params, context)
            await queue.put({"event": "tool_call", "data": {
                "tool": "select_product", "params": params, "message": confirm,
            }})
            await queue.put({"event": "done", "data": {"full_text": ""}})
            await queue.put(None)
            return

    # ── Regular text response via LLaMA streaming
    system = _build_system(context.model_dump() if context else None)
    conversation = ""
    for entry in history[-MAX_HISTORY_TURNS:]:
        tag = "User" if entry.role == "user" else "MARVIS"
        conversation += f"{tag}: {entry.content}\n"
    prompt = f"{conversation}User: {message}\nMARVIS:"

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "system": system,
        "stream": True,
        "keep_alive": -1,
        "options": {"temperature": 0.3, "num_predict": 300, "num_ctx": 4096},
    }

    full_text = ""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                if resp.status != 200:
                    await queue.put({"event": "error", "data": {"error": f"LLaMA error {resp.status}"}})
                    await queue.put(None)
                    return
                async for line in resp.content:
                    line_str = line.decode("utf-8").strip()
                    if not line_str:
                        continue
                    try:
                        chunk = json.loads(line_str)
                        token = chunk.get("response", "")
                        if token:
                            full_text += token
                            await queue.put({"event": "chunk", "data": {"text": token}})
                    except json.JSONDecodeError:
                        continue

        # Anti-hallucination + sanitize
        full_text = _fix_pretend_actions(full_text, False)
        clean = _sanitize_respond(full_text) if full_text else full_text
        await queue.put({"event": "done", "data": {"full_text": clean or full_text}})
    except Exception as exc:
        logger.error(f"LLaMA chat error: {exc}")
        await queue.put({"event": "error", "data": {"error": str(exc)}})
    finally:
        await queue.put(None)


# ── Endpoint ─────────────────────────────────────────────────────────

async def _cascade(
    message: str,
    history: List[ChatMessage],
    context: Optional[SessionContext],
    queue: asyncio.Queue,
):
    """Try providers: Groq (llama-3.1-8b) → LLaMA local fallback."""
    ok = await _stream_groq(message, history, context, queue)
    if ok:
        return

    # Groq unavailable — fall back to local LLaMA
    logger.warning("Groq unavailable, falling back to local LLaMA")
    await _stream_llama(message, history, context, queue)


@router.post("/chat")
async def marvis_chat(request: ChatRequest):
    """Stream a grounded MARVIS chat response via SSE with tool calling."""
    queue: asyncio.Queue = asyncio.Queue()
    asyncio.create_task(
        _cascade(request.message, request.history or [], request.context, queue)
    )

    async def event_generator():
        try:
            while True:
                event = await asyncio.wait_for(queue.get(), timeout=60)
                if event is None:
                    break
                yield f"data: {json.dumps(event)}\n\n"
        except asyncio.TimeoutError:
            yield f"data: {json.dumps({'event': 'error', 'data': {'error': 'timeout'}})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
