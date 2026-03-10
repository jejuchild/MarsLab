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

LANDFORM_TYPES = ["crater", "terraced_crater", "volcanic", "graben", "channel", "wrinkle_ridge", "lda", "lvf", "ccf"]


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
    """Find the nearest landform of given type within radius_km.

    Checks both MOLA-precomputed landforms AND HiRISE classification cache.
    Returns (feature, dist_km).
    """
    from api.proximity_router import haversine_km

    best, best_dist = None, radius_km

    # 1) Check MOLA-precomputed landforms
    index = _load_landform_index()
    type_bands = index.get(landform_type, {})
    if type_bands:
        center_band = int(lat // 5)
        candidates = []
        for offset in (-1, 0, 1):
            candidates.extend(type_bands.get(center_band + offset, []))

        import math
        lat_threshold = radius_km / 60.0 + 0.5
        cos_lat = max(math.cos(math.radians(lat)), 0.1)
        lon_threshold = lat_threshold / cos_lat

        for lf in candidates:
            if abs(lf["lat"] - lat) > lat_threshold:
                continue
            dlon = abs(lf.get("lon", 0) - lon)
            if dlon > 180:
                dlon = 360 - dlon
            if dlon > lon_threshold:
                continue
            d = haversine_km(lat, lon, lf["lat"], lf.get("lon", 0))
            if d < best_dist:
                best_dist = d
                best = lf

    # 2) Also check HiRISE classification cache for LDA/LVF/CCF
    hirise_results = _find_hirise_classified_landforms(lat, lon, landform_type, radius_km)
    for hr in hirise_results:
        if hr["distance_km"] < best_dist:
            best_dist = hr["distance_km"]
            best = {
                "type": landform_type,
                "lat": hr["lat"],
                "lon": hr["lon"],
                "product_id": hr["product_id"],
                "confidence": hr["confidence"],
                "source": "hirise_classification",
            }

    return (best, round(best_dist, 1)) if best else (None, None)


# ── HiRISE classification cache integration ─────────────────────────

_hirise_landform_cache = None


def _get_hirise_landform_cache():
    """Lazy-load the HiRISE landform classification cache (fusion)."""
    global _hirise_landform_cache
    if _hirise_landform_cache is not None:
        return _hirise_landform_cache
    try:
        from analysis.fusion.landform_cache import LandformCache
        _hirise_landform_cache = LandformCache()
        logging.getLogger(__name__).info(
            "Loaded HiRISE landform cache (%d entries)", _hirise_landform_cache.size
        )
    except Exception as exc:
        logging.getLogger(__name__).warning("HiRISE landform cache unavailable: %s", exc)
        _hirise_landform_cache = False  # sentinel: tried and failed
    return _hirise_landform_cache


def _find_hirise_classified_landforms(
    lat: float, lon: float, landform_type: str, radius_km: float = LANDFORM_PROXIMITY_KM,
) -> list[dict]:
    """Find HiRISE-classified landforms (LDA/LVF/CCF/SCT) near (lat, lon).

    Returns list of dicts compatible with compound_search result format.
    """
    cache = _get_hirise_landform_cache()
    if not cache:
        return []

    class_map = {"lda": "LDA", "lvf": "LVF", "ccf": "CCF",
                 "sct": "SCT", "scalloped": "SCT", "lobate_debris": "LDA"}
    target_class = class_map.get(landform_type.lower())
    if not target_class:
        return []

    from api.proximity_router import haversine_km
    import math

    # Search radius in degrees (rough)
    deg_lat = radius_km / 59.2  # ~1 deg lat = 59.2 km on Mars
    cos_lat = max(math.cos(math.radians(lat)), 0.01)
    deg_lon = deg_lat / cos_lat

    entries = cache.get_entries_in_bounds(
        lat - deg_lat, lat + deg_lat,
        lon - deg_lon, lon + deg_lon,
    )

    results = []
    for entry in entries:
        if entry.dominant_class != target_class:
            continue
        dist = haversine_km(lat, lon, entry.lat, entry.lon)
        if dist <= radius_km:
            results.append({
                "product_id": entry.product_id,
                "lat": round(entry.lat, 3),
                "lon": round(entry.lon, 3),
                "dominant_class": entry.dominant_class,
                "confidence": round(entry.confidence, 3),
                "model_version": entry.model_version,
                "distance_km": round(dist, 1),
                "source": "hirise_classification",
            })

    results.sort(key=lambda r: r["distance_km"])
    return results


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
    limit = max(1, min(limit, 100))  # clamp to [1, 100]

    # Normalize instrument aliases (LLM often sends variants)
    _INST_ALIASES = {
        "SHARAD_HIGH_RES": "SHARAD_HIGHRES", "SHARAD HIGH-RES": "SHARAD_HIGHRES",
        "SHARAD_HIRES": "SHARAD_HIGHRES", "SHARAD HIGHRES": "SHARAD_HIGHRES",
        "CRISM_TRDR": "CRISM_TRR3", "CRISM_MTRDR": "CRISM_TRR3",
        "MTRDR": "CRISM_TRR3", "TRDR": "CRISM_TRR3", "TRR3": "CRISM_TRR3",
        "HIRISE_DEM": "HIRISE_DTM",
    }
    instrument = _INST_ALIASES.get(instrument.upper().replace("-", "_").replace(" ", "_"), instrument).upper()
    if intersect_with:
        intersect_with = _INST_ALIASES.get(intersect_with.upper().replace("-", "_").replace(" ", "_"), intersect_with).upper()

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
            return [{"error": f"Cannot load {intersect_with} index"}]

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

        # Compute bounding extent of remaining primary features for spatial pre-filter
        pri_lat_min = min(f["lat"] for f in features) - 2.0 if features else -90
        pri_lat_max = max(f["lat"] for f in features) + 2.0 if features else 90
        pri_lon_min = min(f["lon"] for f in features) - 5.0 if features else -180
        pri_lon_max = max(f["lon"] for f in features) + 5.0 if features else 180

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

            # Spatial pre-filter: skip secondary features far from any primary feature
            if centroid["lat"] < pri_lat_min or centroid["lat"] > pri_lat_max:
                continue
            if centroid["lon"] < pri_lon_min or centroid["lon"] > pri_lon_max:
                # Check antimeridian wrap
                if not (pri_lon_min < -170 and centroid["lon"] > 170) and \
                   not (pri_lon_max > 170 and centroid["lon"] < -170):
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

CRITICAL — INTENT CLASSIFICATION (follow this EXACTLY):

STEP 1: Classify the user's intent into one of these categories:
  (A) KNOWLEDGE QUESTION — user wants to LEARN something. Phrases like: "what is", "tell me about",
      "explain", "how does", "info", "information", "wavelength", "specs", "details",
      "characteristics", "capabilities", "resolution", "how many", "what does X measure",
      "describe", "compare", "difference between", "why", "when was", "history of",
      or any question about an instrument's properties, science, or design.
      → Answer with TEXT only. Do NOT call any tool. Even if an instrument name appears.

  (B) ACTION REQUEST — user wants you to DO something on the map. Phrases like:
      "go to", "fly to", "zoom", "navigate", "load", "enable", "activate",
      "turn on", "pick", "select", "open", "find", "search", "locate".
      → Use the appropriate tool.

  (C) ANALYSIS QUESTION — user asks about what's visible, what products exist, what do you see.
      → Answer with TEXT only. Do NOT call any tool.

EXAMPLES OF KNOWLEDGE QUESTIONS (NEVER use tools for these):
  - "CRISM wavelength info" → text answer about CRISM wavelengths
  - "tell me about SHARAD" → text answer about SHARAD instrument
  - "what resolution does HiRISE have" → text answer
  - "how does CRISM detect minerals" → text answer
  - "SHARAD depth penetration" → text answer
  - "what instruments are on MRO" → text answer

EXAMPLES OF ACTION REQUESTS (use tools):
  - "load CRISM" → call load_instrument
  - "show me CRISM footprints" → call load_instrument
  - "go to Jezero Crater" → call fly_to_location
  - "find SHARAD near craters" → call search

WHEN TO USE EACH TOOL:
- fly_to_location: ONLY when user explicitly asks to GO TO, NAVIGATE, ZOOM, or CENTER ON a NEW, DIFFERENT location.
- load_instrument: ONLY when user explicitly asks to LOAD, ENABLE, ACTIVATE, or TURN ON an instrument's footprint data on the map, or says "show me [instrument] footprints/data".
- select_product: ONLY when user asks to PICK, SELECT, OPEN, INSPECT a specific product.
- search: ONLY when user asks to FIND, SEARCH, or LOCATE products with filters (instrument, mineral, landform, intersection).

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
6. If the user's message contains an instrument name but is asking ABOUT the instrument (not asking to load/enable it), respond with TEXT. The presence of an instrument name does NOT mean you should call load_instrument.

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
            "description": "Navigate/zoom/pan the map to a Mars region or coordinates. Use ONLY when user explicitly says 'go to', 'fly to', 'zoom to', 'navigate to', 'center on', 'take me to', or 'move to'. Do NOT use for questions about a location.",
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
            "description": "Load and display footprint outlines for an orbital instrument on the map. Use ONLY when user explicitly asks to LOAD, ENABLE, ACTIVATE, or TURN ON an instrument's data layer. Do NOT use for questions ABOUT an instrument (e.g. 'what is CRISM', 'CRISM wavelength info', 'tell me about SHARAD'). The presence of an instrument name alone does NOT mean you should call this tool.",
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
                "Search for Mars orbital products. Supports compound queries with multiple filters. "
                "IMPORTANT: If the user mentions TWO instruments (e.g. 'SHARAD intersects CRISM'), "
                "set 'instrument' to the first and 'intersect_with' to the second. "
                "IMPORTANT: 'sharad high-res' or 'sharad highres' = SHARAD_HIGHRES, 'crism trdr' or 'crism trr3' or 'mtrdr' = CRISM_TRR3, "
                "'hirise dtm' = HIRISE_DTM. "
                "Examples: "
                "search(instrument='CRISM_TRR3', mineral_type='ice', min_percent=10) — find CRISM with ice > 10%. "
                "search(instrument='SHARAD_HIGHRES', intersect_with='CRISM_TRR3', mineral_type='ice', min_percent=10) — SHARAD tracks crossing high-ice CRISM. "
                "search(instrument='HIRISE_DTM', near_landform='terraced_crater') — DTMs near terraced craters."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "instrument": {
                        "type": "string",
                        "description": "Primary instrument. 'sharad high-res'=SHARAD_HIGHRES, 'hirise dtm'=HIRISE_DTM, 'crism trdr/mtrdr'=CRISM_TRR3.",
                        "enum": INSTRUMENT_LIST,
                    },
                    "intersect_with": {
                        "type": "string",
                        "description": "Second instrument to intersect/overlap with. MUST set this when user says 'intersect', 'overlap', 'cross', or names TWO instruments.",
                        "enum": INSTRUMENT_LIST,
                    },
                    "mineral_type": {
                        "type": "string",
                        "description": "'ice' for water ice/h2o/water, 'hyd' for hydration/clay. Mineral filter applies to CRISM/CRISM_TRR3.",
                        "enum": ["ice", "hyd"],
                    },
                    "min_percent": {
                        "type": "number",
                        "description": "Minimum mineral percentage (0-100). Default 10 if mineral_type set.",
                    },
                    "near_landform": {
                        "type": "string",
                        "description": "Only products near this landform type.",
                        "enum": LANDFORM_TYPES,
                    },
                },
                "required": ["instrument"],
            },
        },
    },
]


def _normalize_instrument_names(text: str) -> str:
    """Replace common instrument aliases with canonical names to help the LLM."""
    import re as _re
    text = _re.sub(r"(?i)\bsharad[\s_-]*high[\s_-]*res\b", "SHARAD_HIGHRES", text)
    text = _re.sub(r"(?i)\bsharad[\s_-]*hires\b", "SHARAD_HIGHRES", text)
    text = _re.sub(r"(?i)\bhirise[\s_-]*dtm\b", "HIRISE_DTM", text)
    text = _re.sub(r"(?i)\bcrism[\s_-]*tr(?:d?r|r3)\b", "CRISM_TRR3", text)
    text = _re.sub(r"(?i)\bmtrdr\b", "CRISM_TRR3", text)
    return text

# ── Intent pre-classifier (server-side guard against spurious tool calls) ────

# Patterns that indicate INFORMATIONAL intent (user asking ABOUT something, not requesting action)
_INFO_PATTERNS = re.compile(
    r"(?i)"
    r"(?:"
    r"\b(?:what|how|why|when|where|which|who|describe|explain|tell\s+me|info(?:rmation)?|detail|spec|characteristics?)\b"
    r"|\b(?:wavelength|resolution|bandwidth|frequency|penetration|depth|accuracy|precision|spectral|spatial|temporal)\b"
    r"|\b(?:history|design|purpose|capability|capabilities|sensor|detector|optics|antenna|instrument)\s+(?:of|about|info|details|specs|description)"
    r"|(?:about|regarding)\s+(?:" + "|".join(INSTRUMENT_LIST) + r")"
    r"|(?:" + "|".join(INSTRUMENT_LIST) + r")\s+(?:info|information|details|specs|specification|wavelength|resolution|bands?|channels?|description|overview|capabilities|instrument|sensor|science|measure|detect|work)"
    r")"
)

# Patterns that indicate ACTION intent (user wants to DO something)
_ACTION_PATTERNS = re.compile(
    r"(?i)"
    r"\b(?:load|enable|activate|turn\s+on|go\s+to|fly\s+to|zoom|navigate|pick|select|open|find|search|locate)\b"
)


def _get_rag_context(query: str, n_results: int = 3) -> str:
    """Retrieve RAG context for an informational query. Returns formatted string or empty."""
    try:
        from rag.retriever import retrieve, format_context
        chunks = retrieve(query, n_results=n_results, collection="mars_science", min_score=0.2)
        if not chunks:
            return ""
        ctx = format_context(chunks, max_chars=3000)
        return (
            "\n\n--- RETRIEVED KNOWLEDGE BASE CONTEXT ---\n"
            "Use the following retrieved context to enhance your answer. "
            "Reference [Source N] when citing.\n\n"
            f"{ctx}\n"
            "--- END CONTEXT ---"
        )
    except Exception as e:
        logging.getLogger(__name__).debug(f"RAG retrieval skipped: {e}")
        return ""


def _is_informational(message: str) -> bool:
    """Return True if the message is a knowledge/informational question, not an action request.
    
    This is a server-side guard: if the user asks 'CRISM wavelength info',
    we force tool_choice='none' so the LLM can only respond with text.
    """
    has_info = bool(_INFO_PATTERNS.search(message))
    has_action = bool(_ACTION_PATTERNS.search(message))
    # If it has info markers and no action verbs, it's informational
    if has_info and not has_action:
        return True
    # Special case: bare "<INSTRUMENT> <info-word>" without action verb
    # e.g. "CRISM wavelength info", "SHARAD depth penetration"
    for inst in INSTRUMENT_LIST:
        pattern = re.compile(
            rf"(?i)^\s*{re.escape(inst)}\s+(?:wavelength|resolution|info|details|specs?|bands?|science|overview|description|capabilities)\b"
        )
        if pattern.search(message):
            return True
    return False


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

    # Normalize instrument aliases in user message to help the LLM
    normalized_msg = _normalize_instrument_names(message)

    # Server-side intent guard: force text-only response for informational queries
    info_mode = _is_informational(normalized_msg)
    if info_mode:
        logger.info(f"Detected informational query, forcing text-only: {normalized_msg[:80]}")
        # RAG augmentation: inject retrieved context for knowledge queries
        rag_context = _get_rag_context(normalized_msg)
        if rag_context:
            system += rag_context

    messages = [{"role": "system", "content": system}]
    for entry in history[-MAX_HISTORY_TURNS:]:
        role = "user" if entry.role == "user" else "assistant"
        messages.append({"role": role, "content": entry.content})
    messages.append({"role": "user", "content": normalized_msg})

    payload = {
        "model": GROQ_MODEL,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 400,
    }
    if not info_mode:
        payload["tools"] = _GROQ_TOOLS
        payload["tool_choice"] = "auto"

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
    r"(?i)\b(?:load|enable|activate|turn on)\s+(?:the\s+)?("
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
    r"(?i)\b(terraced[_ ]?crater|crater|volcanic|volcano|graben|channel|wrinkle[_ ]?ridge|ridge|lda|lobate[_ ]?debris|lvf|lobate[_ ]?viscous|ccf|concentric[_ ]?crater|brain[_ ]?terrain)\b",
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

    # ── Server-side intent guard: skip regex tool dispatch for informational queries
    if _is_informational(message):
        logger.info(f"LLaMA: informational query detected, skipping tool dispatch: {message[:80]}")
        nav_match = None
        load_match = None
        search_match = None
    else:
        # Check for tool-like intents via regex (since LLaMA lacks function calling)
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
            # Normalize aliases
            _LF_ALIASES = {
                "volcano": "volcanic", "ridge": "wrinkle_ridge",
                "lobate_debris": "lda", "lobate_viscous": "lvf",
                "concentric_crater": "ccf", "brain_terrain": "ccf",
            }
            lf = _LF_ALIASES.get(lf, lf)
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
