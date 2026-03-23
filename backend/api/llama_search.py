"""
Llama-powered Smart AI Search.

Single-step: User query → Llama parses → search executes → Llama picks best
products → auto-download → return results with reasoning.

Uses Groq API (cloud Llama 3.1) for fast inference.
"""

import json
import logging
import asyncio
import os
import uuid
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

import aiohttp
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .mars_regions import find_region, MARS_REGIONS
from .ode_client import Instrument
from .download_manager import (
    download_manager,
    check_local_existence,
    check_local_existence_detailed,
)
from .ai_search import (
    _execute_gemini_plan,
    _execute_cross_instrument,
    _resolve_region_bbox,
    _auto_detect_cross_instrument,
    _search_local_index,
    haversine_distance_km,
    GeminiSearchResult,
)
from .gemini_parser import (
    GeminiPlan,
    GeminiRegion,
    GeminiFilters,
    GeminiCrossFilter,
    normalize_plan_coordinates,
    validate_plan,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ai/smart", tags=["smart-search"])

# Groq config (fast cloud inference — primary)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.1-8b-instant"
GROQ_BASE_URL = "https://api.groq.com/openai/v1/chat/completions"

# Ollama config (unused — kept for reference)
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.1:8b"


# =============================================================================
# Data Models
# =============================================================================

class SmartSearchRequest(BaseModel):
    """Request for smart AI search."""
    query: str
    max_results: int = 20


class SmartProductSelection(BaseModel):
    """A product selected by Llama with reasoning."""
    product_id: str
    instrument: str
    lat: Optional[float] = None
    lon: Optional[float] = None
    distance_km: Optional[float] = None
    reason: str = ""
    already_local: bool = False
    download_task_id: Optional[str] = None


class SmartSearchResponse(BaseModel):
    """Full response from smart search."""
    session_id: str
    query: str
    reasoning: str
    selected_products: List[SmartProductSelection]
    download_tasks: List[str]
    search_summary: str
    total_found: int
    total_selected: int
    total_downloading: int
    total_already_local: int


# =============================================================================
# SSE Session Tracking
# =============================================================================

@dataclass
class SmartSession:
    session_id: str
    query: str
    status: str = "starting"  # starting, parsing, searching, analyzing, downloading, done, error
    stage_message: str = ""
    reasoning: str = ""
    selected: List[Dict[str, Any]] = field(default_factory=list)
    download_tasks: List[str] = field(default_factory=list)
    search_summary: str = ""
    total_found: int = 0
    error: Optional[str] = None
    cancelled: bool = False


_sessions: Dict[str, SmartSession] = {}
_event_queues: Dict[str, asyncio.Queue] = {}
_MAX_SESSIONS = 50


def _cleanup_old_sessions():
    """Keep only the most recent sessions to prevent memory leak."""
    if len(_sessions) > _MAX_SESSIONS:
        # Remove oldest sessions (keep last _MAX_SESSIONS)
        session_ids = list(_sessions.keys())
        for sid in session_ids[:-_MAX_SESSIONS]:
            _sessions.pop(sid, None)
            _event_queues.pop(sid, None)


# =============================================================================
# Groq API Integration
# =============================================================================

async def _check_groq() -> bool:
    """Check if Groq API key is configured."""
    return bool(GROQ_API_KEY)


async def _call_groq(
    prompt: str,
    system: str = "",
    temperature: float = 0.2,
    max_tokens: int = 4096,
) -> str:
    """Call Groq API for fast Llama inference."""
    if not GROQ_API_KEY:
        logger.error("GROQ_API_KEY not set")
        return ""

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        payload = {
            "model": GROQ_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                GROQ_BASE_URL,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    logger.error(f"Groq error {resp.status}: {text}")
                    return ""
                data = await resp.json()
                return data["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"Groq call failed: {e}")
        return ""


# =============================================================================
# JSON Extraction Helper
# =============================================================================

def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Extract the first valid JSON object from text using bracket counting."""
    # Try parsing the whole text first
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # Find first '{' and use bracket counting to find matching '}'
    start = stripped.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False

    for i in range(start, len(stripped)):
        c = stripped[i]
        if escape_next:
            escape_next = False
            continue
        if c == "\\":
            escape_next = True
            continue
        if c == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                candidate = stripped[start : i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    return None
    return None


# =============================================================================
# Llama Prompts
# =============================================================================

PARSE_SYSTEM_PROMPT = """You are a Mars science data search planner. Parse the user's request into a JSON plan.

Output ONLY valid JSON, no other text or markdown.

Instruments:
- CRISM: spectrometer (mineral composition, spectral data)
- HIRISE: high-resolution camera (imagery, surface detail)
- SHARAD: subsurface radar (ice detection, radargrams)
- SHARAD_HIGHRES: high-res processed SHARAD radar (full RDR products)
- CTX: context camera (wide-angle surface imagery)
- HIRISE_DTM: digital terrain model (elevation, topography)

Known Mars regions with bounding boxes:
- Arcadia Planitia: [38, 60, -170, -140]
- Amazonis Planitia: [15, 40, -170, -145]
- Utopia Planitia: [30, 60, 100, 140]
- Elysium Planitia: [-10, 15, 140, 170]
- Isidis Planitia: [5, 22, 80, 100]
- Hellas Planitia: [-55, -30, 55, 85]
- Chryse Planitia: [15, 30, -45, -25]
- Valles Marineris: [-20, -5, -80, -30]
- Jezero Crater: [17.8, 19.0, 77.0, 78.2]
- Gale Crater: [-7, -4, 136, 139]
- Olympus Mons: [15, 23, -137, -128]
- Syrtis Major: [0, 16, 62, 78]
- Meridiani Planum: [-5, 2, -8, 5]
- Arabia Terra: [5, 35, -15, 25]
- Nili Fossae: [18, 24, 72, 78]
- Mawrth Vallis: [19, 25, -22, -16]
- Tharsis: [-15, 15, -115, -85]
- North Pole: [78, 90, -180, 180]
- South Pole: [-90, -78, -180, 180]

JSON schema:
{
  "intent": "download",
  "targets": ["CRISM"],
  "region": {
    "type": "named_region",
    "name": "Jezero Crater",
    "bbox": {"minLat": 17.8, "maxLat": 19.0, "minLon": 77.0, "maxLon": 78.2},
    "point": null,
    "radiusKm": 100
  },
  "filters": {
    "maxResults": 20,
    "spatialPredicate": "any",
    "distribution": "any",
    "terrainHint": null,
    "crossInstrument": null
  }
}

region.type MUST be exactly one of: "named_region", "bbox", "point", "global".
- "named_region": Use when user mentions a known Mars location. Set name and bbox.
- "bbox": Use when user gives explicit coordinates. Set bbox only.
- "point": Use for a single coordinate. Set point only.
- "global": Use when no location is specified.

spatialPredicate MUST be one of: "any", "intersects", "crosses", "within", "nearest".
crossInstrument: Use {"instrument": "HIRISE_DTM", "titleContains": null} when user wants cross-instrument overlap, otherwise null.

crossInstrument: Use when user wants products from instrument A that overlap/intersect with instrument B.
Put A in "targets", B in crossInstrument.instrument.

Examples:
- "CRISM that intersects SHARAD_HIGHRES and HIRISE DTM" → targets=["CRISM"], crossInstrument={"instrument":"SHARAD_HIGHRES"} (pick primary reference instrument)
- "download CRISM over Jezero" → targets=["CRISM"], region.type="named_region", region.name="Jezero Crater"
- "SHARAD_HIGHRES intersecting HiRISE DTM" → targets=["SHARAD_HIGHRES"], crossInstrument={"instrument":"HIRISE_DTM"}"""

SELECTION_SYSTEM_PROMPT = """You are a Mars science data analyst. Given search results for a user's query, select the BEST products to download and explain why.

Think about:
1. What the user actually wants (their intent)
2. Spatial relevance (closer to region of interest = better)
3. Instrument coverage (if they want cross-instrument data, ensure overlap)
4. Avoid duplicates — pick diverse, complementary products
5. Already-downloaded products don't need downloading again

Output ONLY valid JSON:
{
  "reasoning": "2-4 sentence explanation of your overall approach and why these products were chosen",
  "selections": [
    {
      "product_id": "...",
      "reason": "Short reason why this specific product was selected"
    }
  ]
}

Be concise but specific in your reasoning. Reference spatial relationships, distances, and instrument purposes."""


# =============================================================================
# Core Smart Search Logic
# =============================================================================

async def _parse_with_llama(query: str, max_results: int) -> GeminiPlan:
    """Use Llama to parse user query into a structured plan."""
    prompt = f"""Parse this Mars data request into JSON:
"{query}"

User wants max {max_results} results.
Output ONLY the JSON object."""

    response = await _call_groq(prompt, PARSE_SYSTEM_PROMPT)

    if not response:
        return GeminiPlan(
            intent="error",
            targets=[],
            region=GeminiRegion(type="global"),
            filters=GeminiFilters(maxResults=max_results),
            error="Llama returned empty response",
        )

    # Extract JSON from response using bracket-counting
    data = _extract_json(response)
    if data is None:
        return GeminiPlan(
            intent="error",
            targets=[],
            region=GeminiRegion(type="global"),
            filters=GeminiFilters(maxResults=max_results),
            error="Could not extract valid JSON from Llama response",
        )

    # Build GeminiPlan from parsed data
    region_data = data.get("region", {})

    # Sanitize region type — Llama sometimes returns the schema description literally
    raw_region_type = region_data.get("type", "global")
    VALID_REGION_TYPES = {"named_region", "bbox", "point", "global"}
    if raw_region_type not in VALID_REGION_TYPES:
        # Try to infer from available data
        if region_data.get("name"):
            raw_region_type = "named_region"
        elif region_data.get("point"):
            raw_region_type = "point"
        elif region_data.get("bbox"):
            raw_region_type = "bbox"
        else:
            raw_region_type = "global"
        logger.info(f"Sanitized invalid region type to: {raw_region_type}")

    region = GeminiRegion(
        type=raw_region_type,
        name=region_data.get("name"),
        bbox=region_data.get("bbox"),
        point=region_data.get("point"),
        radiusKm=region_data.get("radiusKm", 100.0),
    )

    filters_data = data.get("filters", {})
    cross_data = filters_data.get("crossInstrument")
    cross_filter = None
    if cross_data and isinstance(cross_data, dict) and cross_data.get("instrument"):
        cross_filter = GeminiCrossFilter(
            instrument=cross_data["instrument"],
            titleContains=cross_data.get("titleContains"),
        )

    # Sanitize spatialPredicate
    raw_predicate = filters_data.get("spatialPredicate", "any")
    VALID_PREDICATES = {"any", "intersects", "crosses", "within", "nearest"}
    if raw_predicate not in VALID_PREDICATES:
        raw_predicate = "any"

    filters = GeminiFilters(
        maxResults=min(filters_data.get("maxResults", max_results), max_results),
        spatialPredicate=raw_predicate,
        distribution=filters_data.get("distribution", "any"),
        terrainHint=filters_data.get("terrainHint"),
        crossInstrument=cross_filter,
    )

    plan = GeminiPlan(
        intent=data.get("intent", "download"),
        targets=data.get("targets", []),
        region=region,
        filters=filters,
        raw_response=response,
    )

    # Validate and normalize
    normalize_plan_coordinates(plan)
    is_valid, error_msg = validate_plan(plan, max_results)
    if not is_valid:
        plan.error = error_msg

    return plan


async def _select_with_llama(
    query: str,
    results: List[GeminiSearchResult],
    plan: GeminiPlan,
    max_select: int = 20,
) -> Dict[str, Any]:
    """Use Llama to analyze results and pick the best products."""
    # Build a summary of results for Llama
    result_summary = []
    for r in results[:50]:  # Cap at 50 to avoid token overflow
        entry = {
            "product_id": r.product_id,
            "instrument": r.instrument,
            "distance_km": r.distance_km,
            "already_downloaded": r.exists,
        }
        if r.lat is not None:
            entry["lat"] = round(r.lat, 2)
            entry["lon"] = round(r.lon, 2) if r.lon else None
        result_summary.append(entry)

    # Build context about what we searched for
    context_parts = []
    context_parts.append(f"User query: \"{query}\"")
    context_parts.append(f"Instruments searched: {', '.join(plan.targets)}")
    if plan.region.name:
        context_parts.append(f"Region: {plan.region.name}")
    if plan.filters.crossInstrument:
        cross = plan.filters.crossInstrument
        context_parts.append(f"Cross-instrument filter: intersect with {cross.instrument}")
        if cross.titleContains:
            context_parts.append(f"Reference title filter: {cross.titleContains}")
    context_parts.append(f"Total results found: {len(results)}")
    context_parts.append(f"Select up to {max_select} best products to download")

    prompt = f"""{chr(10).join(context_parts)}

Search results (JSON array):
{json.dumps(result_summary, indent=1)}

Select the best products to download. Prefer products that are NOT already downloaded.
If all relevant products are already downloaded, still list them and note they're local.
Output ONLY JSON."""

    response = await _call_groq(prompt, SELECTION_SYSTEM_PROMPT, temperature=0.3)

    if not response:
        # Fallback: select all non-downloaded products
        return _fallback_selection(results, query, max_select)

    data = _extract_json(response)
    if data and "selections" in data:
        return data

    return _fallback_selection(results, query, max_select)


def _fallback_selection(
    results: List[GeminiSearchResult],
    query: str,
    max_select: int,
) -> Dict[str, Any]:
    """Fallback selection when Llama is unavailable — pick closest non-downloaded."""
    selections = []
    for r in results[:max_select]:
        reason = "Closest to search area"
        if r.exists:
            reason = "Already downloaded locally"
        elif r.distance_km is not None:
            reason = f"Within {r.distance_km:.1f} km of target region"
        selections.append({
            "product_id": r.product_id,
            "reason": reason,
        })

    return {
        "reasoning": f"Selected the {len(selections)} closest products to the target region. Products are sorted by distance — closer products are more relevant to the search area.",
        "selections": selections,
    }


# =============================================================================
# Main Smart Search Pipeline
# =============================================================================

async def _run_smart_search(
    session: SmartSession,
    queue: asyncio.Queue,
    max_results: int,
):
    """Run the full smart search pipeline with SSE events."""

    async def emit(event: str, data: Dict[str, Any]):
        if not session.cancelled:
            await queue.put({"event": event, "data": data})

    try:
        # ── Step 1: Check Groq ─────────────────────
        session.status = "parsing"
        session.stage_message = "Connecting to Groq..."
        await emit("stage", {"status": "parsing", "message": "Connecting to Groq..."})

        groq_ok = await _check_groq()
        if not groq_ok:
            session.error = "GROQ_API_KEY is not set. Configure it in your environment."
            session.status = "error"
            await emit("error", {"error": session.error})
            return

        # ── Step 2: Parse with Llama ─────────────────
        session.stage_message = "Groq Llama is analyzing your request..."
        await emit("stage", {"status": "parsing", "message": "Groq Llama is analyzing your request..."})

        plan = await _parse_with_llama(session.query, max_results)

        if plan.error:
            session.error = f"Parse error: {plan.error}"
            session.status = "error"
            await emit("error", {"error": session.error})
            return

        # Resolve region bbox from lookup table
        _resolve_region_bbox(plan)
        # Auto-detect cross-instrument patterns
        _auto_detect_cross_instrument(plan)

        # Build human-readable parse summary
        parse_info = {
            "instruments": plan.targets,
            "region": plan.region.name or plan.region.type,
        }
        if plan.filters.crossInstrument:
            parse_info["cross_instrument"] = plan.filters.crossInstrument.instrument
        if plan.region.bbox:
            parse_info["bbox"] = plan.region.bbox

        await emit("parsed", {"plan": parse_info})

        # ── Check cancellation ────────────────────────
        if session.cancelled:
            return

        # ── Step 3: Execute Search ───────────────────
        session.status = "searching"
        session.stage_message = f"Searching {', '.join(plan.targets)}..."
        await emit("stage", {"status": "searching", "message": session.stage_message})

        results = await _execute_gemini_plan(plan)

        # Sort by distance
        results.sort(key=lambda r: r.distance_km if r.distance_km is not None else float("inf"))

        session.total_found = len(results)
        await emit("search_done", {
            "total_found": len(results),
            "instruments": list(set(r.instrument for r in results)),
        })

        if not results:
            session.reasoning = "No products were found matching your search criteria. Try broadening the region or using different instruments."
            session.search_summary = "No results found"
            session.status = "done"
            await emit("done", _build_response(session))
            return

        # ── Check cancellation ────────────────────────
        if session.cancelled:
            return

        # ── Step 4: Llama Analyzes & Selects ─────────
        session.status = "analyzing"
        session.stage_message = "Llama is choosing the best products..."
        await emit("stage", {"status": "analyzing", "message": "Llama is choosing the best products..."})

        selection = await _select_with_llama(
            session.query, results, plan, max_select=max_results,
        )

        reasoning = selection.get("reasoning", "Products selected by proximity to search area.")
        selections = selection.get("selections", [])

        session.reasoning = reasoning
        await emit("reasoning", {"reasoning": reasoning})

        # Map selections back to full result data
        result_map = {r.product_id: r for r in results}
        selected_products: List[SmartProductSelection] = []

        for sel in selections:
            pid = sel.get("product_id", "")
            r = result_map.get(pid)
            if not r:
                continue
            selected_products.append(SmartProductSelection(
                product_id=r.product_id,
                instrument=r.instrument,
                lat=r.lat,
                lon=r.lon,
                distance_km=r.distance_km,
                reason=sel.get("reason", ""),
                already_local=r.exists,
            ))

        # If Llama returned nothing useful, fall back to top results
        if not selected_products:
            for r in results[:max_results]:
                selected_products.append(SmartProductSelection(
                    product_id=r.product_id,
                    instrument=r.instrument,
                    lat=r.lat,
                    lon=r.lon,
                    distance_km=r.distance_km,
                    reason="Closest to search area" if not r.exists else "Already downloaded locally",
                    already_local=r.exists,
                ))

        session.selected = [sp.model_dump() for sp in selected_products]
        await emit("selected", {
            "count": len(selected_products),
            "products": [
                {"product_id": sp.product_id, "instrument": sp.instrument, "reason": sp.reason, "already_local": sp.already_local}
                for sp in selected_products
            ],
        })

        # ── Check cancellation ────────────────────────
        if session.cancelled:
            return

        # ── Step 5: Auto-Download ────────────────────
        to_download = [sp for sp in selected_products if not sp.already_local]
        # Skip instruments that are local-only (CTX, HIRISE_DTM — already in local index)
        to_download = [
            sp for sp in to_download
            if sp.instrument.upper() not in ("CTX", "HIRISE_DTM")
        ]

        if to_download:
            session.status = "downloading"
            session.stage_message = f"Starting {len(to_download)} downloads..."
            await emit("stage", {"status": "downloading", "message": session.stage_message})

            for sp in to_download:
                if session.cancelled:
                    break
                try:
                    inst_lower = sp.instrument.lower()
                    try:
                        inst_enum = Instrument(inst_lower)
                    except ValueError:
                        continue

                    # Skip if already exists (double-check)
                    if check_local_existence(sp.product_id, inst_enum):
                        sp.already_local = True
                        continue

                    task = await download_manager.start_download(
                        product_id=sp.product_id,
                        instrument=inst_enum,
                        lat=sp.lat,
                        lon=sp.lon,
                    )
                    sp.download_task_id = task.task_id
                    session.download_tasks.append(task.task_id)

                    await emit("download_started", {
                        "product_id": sp.product_id,
                        "instrument": sp.instrument,
                        "task_id": task.task_id,
                    })
                except Exception as e:
                    logger.error(f"Download failed for {sp.product_id}: {e}")
                    await emit("download_error", {
                        "product_id": sp.product_id,
                        "error": str(e),
                    })

        # ── Done ─────────────────────────────────────
        already_local = sum(1 for sp in selected_products if sp.already_local)
        downloading = len(session.download_tasks)

        session.search_summary = (
            f"Found {session.total_found} products. "
            f"Selected {len(selected_products)}. "
            f"{already_local} already local, {downloading} downloading."
        )
        session.selected = [sp.model_dump() for sp in selected_products]
        session.status = "done"

        await emit("done", _build_response(session))

    except Exception as e:
        logger.error(f"Smart search error: {e}", exc_info=True)
        session.status = "error"
        session.error = str(e)
        await emit("error", {"error": str(e), "session_id": session.session_id})
    finally:
        await queue.put(None)  # End sentinel


def _build_response(session: SmartSession) -> Dict[str, Any]:
    """Build the final response dict."""
    selected = session.selected or []
    already_local = sum(1 for s in selected if s.get("already_local"))
    downloading = len(session.download_tasks)

    return {
        "session_id": session.session_id,
        "query": session.query,
        "reasoning": session.reasoning,
        "selected_products": selected,
        "download_tasks": session.download_tasks,
        "search_summary": session.search_summary,
        "total_found": session.total_found,
        "total_selected": len(selected),
        "total_downloading": downloading,
        "total_already_local": already_local,
    }


# =============================================================================
# API Endpoints
# =============================================================================

@router.get("/status")
async def smart_search_status():
    """Check Groq/Llama availability for smart search."""
    available = await _check_groq()
    return {
        "ollama_available": available,
        "model": GROQ_MODEL,
        "message": "Groq Llama ready for smart search" if available else "GROQ_API_KEY not set",
    }


@router.post("/search")
async def smart_search(request: SmartSearchRequest):
    """
    Start a smart search session. Returns session_id immediately.
    Connect to /api/ai/smart/stream/{session_id} for real-time events.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    max_results = min(max(1, request.max_results), 50)

    _cleanup_old_sessions()

    session_id = str(uuid.uuid4())[:8]
    session = SmartSession(session_id=session_id, query=request.query)
    _sessions[session_id] = session

    queue: asyncio.Queue = asyncio.Queue()
    _event_queues[session_id] = queue

    asyncio.create_task(_run_smart_search(session, queue, max_results))

    return {"session_id": session_id, "status": "started"}


@router.get("/stream/{session_id}")
async def smart_search_stream(session_id: str):
    """SSE stream of smart search events."""
    queue = _event_queues.get(session_id)
    if not queue:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    async def event_generator():
        try:
            while True:
                event = await asyncio.wait_for(queue.get(), timeout=300)
                if event is None:
                    yield f"data: {json.dumps({'event': 'stream_end'})}\n\n"
                    break
                yield f"data: {json.dumps(event)}\n\n"
        except asyncio.TimeoutError:
            yield f"data: {json.dumps({'event': 'timeout'})}\n\n"
        finally:
            _event_queues.pop(session_id, None)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/session/{session_id}")
async def smart_search_session(session_id: str):
    """Get session state (polling fallback)."""
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    return _build_response(session)


@router.delete("/session/{session_id}")
async def cancel_smart_search(session_id: str):
    """Cancel a running smart search session."""
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    session.cancelled = True
    session.status = "cancelled"
    return {"status": "cancelled", "session_id": session_id}
