"""
Agentic AI Orchestrator.

Core agent loop that:
1. Takes a natural language mission objective
2. Uses Llama (via Ollama) to generate an execution plan
3. Executes each task step-by-step
4. Uses Llama to synthesize results into a narrative report

Falls back to rule-based planning if Ollama is unavailable.
"""

import json
import logging
import asyncio
import os
import uuid
import re
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, AsyncGenerator, Callable, Awaitable, Tuple
from dataclasses import dataclass, field
from enum import Enum

import aiohttp

from .agent_tasks import (
    TaskResult,
    RegionBBox,
    resolve_region,
    bbox_from_coords,
    search_region,
    check_local_data,
    download_data,
    slope_analysis,
    subsurface_scan,
    mineral_analysis,
    mineral_cnn_classify,
    dielectric_analysis,
    synthesize_results,
    recommend_site,
)
from .mars_climate import climate_analysis_for_region
from .thermal_inertia import thermal_inertia_analysis_for_region
from .science_context import get_context_for_agent, get_region_context_by_name

logger = logging.getLogger(__name__)

# Ollama config
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.3"


# =============================================================================
# Data Models
# =============================================================================

class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class AgentStep:
    """A single step in the agent's execution plan."""
    id: str
    type: str  # search, check_data, download, slope, subsurface, mineral, synthesize
    description: str
    instrument: Optional[str] = None
    status: StepStatus = StepStatus.PENDING
    result: Optional[TaskResult] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "description": self.description,
            "instrument": self.instrument,
            "status": self.status.value,
            "result_summary": self.result.summary if self.result else None,
            "error": self.error,
        }


@dataclass
class AgentSession:
    """Tracks a full agent execution session."""
    session_id: str
    objective: str
    status: str = "planning"  # planning, executing, synthesizing, done, error
    region_name: Optional[str] = None
    bbox: Optional[RegionBBox] = None
    steps: List[AgentStep] = field(default_factory=list)
    all_results: Dict[str, TaskResult] = field(default_factory=dict)
    narrative: str = ""
    synthesis: Optional[Dict[str, Any]] = None
    figures: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Event buffer for replay/resume — stores every SSE event emitted
    events: List[Dict[str, Any]] = field(default_factory=list)
    # Monotonic counter incremented on each emit; consumers poll this
    _event_count: int = field(default=0, repr=False)
    # Condition var — wakes ALL waiting consumers on emit (multi-consumer safe)
    _condition: asyncio.Condition = field(default_factory=asyncio.Condition, repr=False)
    # Reference to the background asyncio.Task (if running)
    _task: Optional[asyncio.Task] = field(default=None, repr=False)

    async def emit(self, event: Dict[str, Any]):
        """Append an event to the buffer and wake all listeners."""
        self.events.append(event)
        async with self._condition:
            self._event_count += 1
            self._condition.notify_all()

    @property
    def is_terminal(self) -> bool:
        return self.status in ("done", "error")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "objective": self.objective,
            "status": self.status,
            "region_name": self.region_name,
            "steps": [s.to_dict() for s in self.steps],
            "narrative": self.narrative,
            "synthesis": self.synthesis,
            "figures": self.figures,
            "error": self.error,
            "created_at": self.created_at,
        }


# Active sessions store
_sessions: Dict[str, AgentSession] = {}
MAX_SESSIONS = 50

# Persistent storage
_SESSIONS_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
_SESSIONS_FILE = os.path.join(_SESSIONS_DIR, "agent_sessions.json")


def _load_sessions():
    """Load completed sessions from disk into _sessions dict."""
    if not os.path.exists(_SESSIONS_FILE):
        return
    try:
        with open(_SESSIONS_FILE, "r") as f:
            records = json.load(f)
        for rec in records:
            sid = rec.get("session_id")
            if not sid or sid in _sessions:
                continue
            # Reconstruct AgentStep objects
            steps = []
            for s in rec.get("steps", []):
                result = None
                if s.get("result"):
                    result = TaskResult(
                        task_type=s["result"].get("task_type", ""),
                        instrument=s["result"].get("instrument"),
                        success=s["result"].get("success", True),
                        data=s["result"].get("data", {}),
                        error=s["result"].get("error"),
                        summary=s["result"].get("summary", ""),
                    )
                steps.append(AgentStep(
                    id=s["id"],
                    type=s["type"],
                    description=s["description"],
                    instrument=s.get("instrument"),
                    status=StepStatus(s.get("status", "completed")),
                    result=result,
                    error=s.get("error"),
                ))
            bbox = None
            if rec.get("bbox"):
                b = rec["bbox"]
                bbox = RegionBBox(b["min_lat"], b["max_lat"], b["min_lon"], b["max_lon"])
            session = AgentSession(
                session_id=sid,
                objective=rec.get("objective", ""),
                status=rec.get("status", "done"),
                region_name=rec.get("region_name"),
                bbox=bbox,
                steps=steps,
                narrative=rec.get("narrative", ""),
                synthesis=rec.get("synthesis"),
                error=rec.get("error"),
                created_at=rec.get("created_at", ""),
            )
            _sessions[sid] = session
        logger.info(f"Loaded {len(records)} agent sessions from disk")
    except Exception as e:
        logger.error(f"Failed to load agent sessions: {e}")


def _save_session(session: AgentSession):
    """Save a terminal session to disk (append/update)."""
    if not session.is_terminal:
        return
    os.makedirs(_SESSIONS_DIR, exist_ok=True)

    # Build serializable record with full step results
    step_records = []
    for s in session.steps:
        sr = s.to_dict()
        if s.result:
            sr["result"] = {
                "task_type": s.result.task_type,
                "instrument": s.result.instrument,
                "success": s.result.success,
                "data": s.result.data,
                "error": s.result.error,
                "summary": s.result.summary,
            }
        step_records.append(sr)

    record = session.to_dict()
    record["steps"] = step_records
    # Strip base64 image data from figures to keep JSON file small
    if record.get("figures"):
        record["figures"] = [
            {k: v for k, v in fig.items() if k != "base64"}
            for fig in record["figures"]
        ]
    if session.bbox:
        record["bbox"] = {
            "min_lat": session.bbox.min_lat,
            "max_lat": session.bbox.max_lat,
            "min_lon": session.bbox.min_lon,
            "max_lon": session.bbox.max_lon,
        }

    # Load existing, replace or append, write back
    existing: list = []
    if os.path.exists(_SESSIONS_FILE):
        try:
            with open(_SESSIONS_FILE, "r") as f:
                existing = json.load(f)
        except Exception:
            existing = []

    # Update in place or append
    found = False
    for i, rec in enumerate(existing):
        if rec.get("session_id") == session.session_id:
            existing[i] = record
            found = True
            break
    if not found:
        existing.append(record)

    # Keep only latest MAX_SESSIONS
    if len(existing) > MAX_SESSIONS:
        existing.sort(key=lambda r: r.get("created_at", ""))
        existing = existing[-MAX_SESSIONS:]

    try:
        with open(_SESSIONS_FILE, "w") as f:
            json.dump(existing, f, indent=2, default=str)
        logger.info(f"Saved agent session {session.session_id} to disk")
    except Exception as e:
        logger.error(f"Failed to save agent session: {e}")


# Load persisted sessions on module import
_load_sessions()


def _evict_old_sessions():
    """Remove oldest terminal sessions when we exceed MAX_SESSIONS."""
    if len(_sessions) <= MAX_SESSIONS:
        return
    terminal = sorted(
        (s for s in _sessions.values() if s.is_terminal),
        key=lambda s: s.created_at,
    )
    to_remove = len(_sessions) - MAX_SESSIONS
    for s in terminal[:to_remove]:
        _sessions.pop(s.session_id, None)


def get_session(session_id: str) -> Optional[AgentSession]:
    return _sessions.get(session_id)


def list_sessions() -> List[AgentSession]:
    """Return all sessions, newest first."""
    sessions = list(_sessions.values())
    sessions.sort(key=lambda s: s.created_at, reverse=True)
    return sessions


async def cancel_session(session_id: str) -> bool:
    """Cancel a running agent session. Returns True if cancelled."""
    session = _sessions.get(session_id)
    if not session or session.is_terminal:
        return False
    # Cancel the background task
    if session._task and not session._task.done():
        session._task.cancel()
    # Mark any running steps as failed
    for step in session.steps:
        if step.status == StepStatus.RUNNING:
            step.status = StepStatus.FAILED
            step.error = "Cancelled by user"
    session.status = "error"
    session.error = "Cancelled by user"
    await session.emit({"event": "error", "data": {"error": "Cancelled by user"}})
    await session.emit({"event": "stream_end"})
    _save_session(session)
    return True


# =============================================================================
# Ollama LLM Integration
# =============================================================================

async def _check_ollama() -> bool:
    """Check if Ollama is running."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=aiohttp.ClientTimeout(total=3)) as resp:
                return resp.status == 200
    except Exception:
        return False


async def _call_ollama(prompt: str, system: str = "", temperature: float = 0.3) -> str:
    """Call Ollama API for text generation."""
    try:
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "keep_alive": -1,
            "options": {
                "temperature": temperature,
                "num_predict": 2048,
            },
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=300),
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    logger.error(f"Ollama error {resp.status}: {text}")
                    return ""
                data = await resp.json()
                return data.get("response", "")

    except Exception as e:
        logger.error(f"Ollama call failed: {e}")
        return ""


async def _call_ollama_streaming(
    prompt: str,
    system: str = "",
    temperature: float = 0.3,
    on_chunk: Optional[Callable[[str], Awaitable[None]]] = None,
) -> str:
    """Call Ollama with stream=true, yielding text chunks via on_chunk callback.
    Returns the full accumulated response text."""
    try:
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "system": system,
            "stream": True,
            "keep_alive": -1,
            "options": {
                "temperature": temperature,
                "num_predict": 2048,
            },
        }

        full_text = ""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=300),
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    logger.error(f"Ollama streaming error {resp.status}: {text}")
                    return ""

                async for line in resp.content:
                    line_str = line.decode("utf-8").strip()
                    if not line_str:
                        continue
                    try:
                        chunk_data = json.loads(line_str)
                        token = chunk_data.get("response", "")
                        if token:
                            full_text += token
                            if on_chunk:
                                await on_chunk(token)
                    except json.JSONDecodeError:
                        continue

        return full_text

    except Exception as e:
        logger.error(f"Ollama streaming call failed: {e}")
        return ""


# =============================================================================
# ReAct Agent — Tool Registry
# =============================================================================

MAX_ITERATIONS = 15


def _format_tool_descriptions() -> str:
    """Build a human-readable tool list for the system prompt."""
    lines = []
    for name, spec in AGENT_TOOLS.items():
        params_str = ", ".join(f"{k}: {v}" for k, v in spec["params"].items()) if spec["params"] else "none"
        lines.append(f"- {name}({params_str}): {spec['description']}")
    return "\n".join(lines)


REACT_SYSTEM_PROMPT = """You are a Mars science mission AI agent. You analyze multi-instrument orbital data to evaluate landing sites, find subsurface ice, and assess engineering feasibility.

You work in a loop: Thought → Action → Observation → Thought → Action → ...

AVAILABLE TOOLS:
{tools}

OUTPUT FORMAT — You MUST output exactly two lines on each turn:

Thought: <your reasoning, 1-3 sentences>
Action: {{"tool": "<tool_name>", "params": {{...}}}}

IMPORTANT: Do NOT wrap the JSON in code blocks. Do NOT add any text after the Action JSON. Just output Thought: then Action: with raw JSON.

EXAMPLE 1 — resolving a region:
Thought: I need to resolve Jezero Crater to get its bounding box coordinates before I can search for data.
Action: {{"tool": "resolve_region", "params": {{"region_name": "Jezero Crater"}}}}

EXAMPLE 2 — searching for data:
Thought: Now I should search for SHARAD high-resolution data to look for subsurface ice reflectors in this region.
Action: {{"tool": "search_products", "params": {{"instrument": "SHARAD_HIGHRES"}}}}

EXAMPLE 3 — finishing:
Thought: I have gathered enough data. The region shows strong subsurface ice signatures and favorable terrain. I should provide my final assessment.
Action: {{"tool": "finish", "params": {{"summary": "Jezero Crater shows strong evidence of subsurface ice at 15-30m depth with favorable landing terrain.", "recommendation": "STRONG_CANDIDATE"}}}}

STRATEGY:
1. Resolve the region first if a named location is mentioned.
2. Search ALL relevant instruments one by one: CRISM, HIRISE, SHARAD, SHARAD_HIGHRES, CTX, HIRISE_DTM. Each search_products call handles one instrument.
3. Call check_local_data to see what is available locally, then download_products for missing data.
4. Run analyses: analyze_subsurface, analyze_minerals, analyze_slope, classify_minerals_cnn, estimate_dielectric.
5. Call recommend_site to cross-reference everything into a landing site recommendation.
6. Call finish with your final summary and recommendation.

RULES:
- Output Thought: and Action: on EVERY turn. Nothing else.
- One action per turn.
- Call check_local_data before download_products.
- Be adaptive: if something returns 0 results, try a different approach."""


async def _tool_resolve_region(session: "AgentSession", params: dict):
    """Look up a named Mars region."""
    name = params.get("region_name", "")
    bbox = resolve_region(name)
    if bbox:
        session.bbox = bbox
        session.region_name = name
        return TaskResult(task_type="resolve_region", success=True, summary=f"Resolved {name}",
                          data={"bbox": {"min_lat": bbox.min_lat, "max_lat": bbox.max_lat,
                                         "min_lon": bbox.min_lon, "max_lon": bbox.max_lon}}), \
            f"Resolved '{name}': lat {bbox.min_lat:.1f} to {bbox.max_lat:.1f}, lon {bbox.min_lon:.1f} to {bbox.max_lon:.1f}."
    return TaskResult(task_type="resolve_region", success=False, summary=f"Could not resolve '{name}'",
                      data={}), \
        f"Could not resolve region '{name}'. Try a different name or provide coordinates."


async def _tool_search_products(session: "AgentSession", params: dict):
    """Search for instrument products."""
    instrument = params.get("instrument", "").upper()
    if not session.bbox:
        return TaskResult(task_type="search", success=False, summary="No region set", data={}), \
            "Error: No region resolved yet. Call resolve_region first."
    result = await search_region(instrument, session.bbox)
    session.all_results[f"search_{instrument}"] = result
    if result.success:
        products = result.data.get("products", [])
        if not hasattr(session, "_all_products"):
            session._all_products = []
        session._all_products.extend(products)
        count = len(products)
        return result, f"Found {count} {instrument} products in {session.region_name}."
    return result, f"Search for {instrument} failed: {result.error or 'unknown error'}."


async def _tool_check_local(session: "AgentSession", params: dict):
    """Check local data availability."""
    products = getattr(session, "_all_products", [])
    if not products:
        return TaskResult(task_type="check_data", success=True, summary="No products to check",
                          data={"available_count": 0, "missing_count": 0}), \
            "No products have been searched yet. Search for instruments first."
    result = check_local_data(products)
    session.all_results["check_local"] = result
    avail = result.data.get("available_count", 0)
    missing = result.data.get("missing_count", 0)
    return result, f"{avail} products available locally, {missing} need downloading."


async def _tool_download(session: "AgentSession", params: dict):
    """Download missing products."""
    check_result = session.all_results.get("check_local")
    if not check_result:
        return TaskResult(task_type="download", success=False, summary="Run check_local_data first",
                          data={}), \
            "Error: Run check_local_data first to identify missing products."
    missing = check_result.data.get("missing", [])
    max_count = params.get("max_count", MAX_DOWNLOADS_PER_RUN)
    to_download, strategy = _select_downloads(missing, session.bbox, max_total=min(max_count, MAX_DOWNLOADS_PER_RUN))
    result = await download_data(to_download)
    session.all_results["download"] = result
    downloaded = result.data.get("downloaded_count", 0)
    failed = result.data.get("failed_count", 0)
    return result, f"Downloaded {downloaded} products ({strategy} strategy). {failed} failed. {len(missing) - len(to_download)} skipped (over limit)."


async def _tool_download_with_progress(session: "AgentSession", params: dict, on_progress=None):
    """Download tool wrapper that supports progress callbacks for SSE streaming."""
    if not getattr(session, "_auto_download", True):
        return TaskResult(task_type="download", success=True, summary="Auto-download disabled",
                          data={"downloaded_count": 0}), \
            "Auto-download is disabled by user settings. Proceeding with locally available data only."
    check_result = session.all_results.get("check_local")
    if not check_result:
        return TaskResult(task_type="download", success=False, summary="Run check_local_data first",
                          data={}), \
            "Error: Run check_local_data first to identify missing products."
    missing = check_result.data.get("missing", [])
    max_count = params.get("max_count", MAX_DOWNLOADS_PER_RUN)
    to_download, strategy = _select_downloads(
        missing, session.bbox, max_total=min(max_count, MAX_DOWNLOADS_PER_RUN)
    )
    result = await download_data(to_download, on_progress=on_progress)
    session.all_results["download"] = result
    downloaded = result.data.get("downloaded_count", 0)
    failed = result.data.get("failed_count", 0)
    return result, f"Downloaded {downloaded} products ({strategy} strategy). {failed} failed. {len(missing) - len(to_download)} skipped (over limit)."


async def _tool_slope(session: "AgentSession", params: dict):
    """Analyze terrain slope."""
    if not session.bbox:
        return TaskResult(task_type="slope", success=False, summary="No region", data={}), \
            "Error: No region resolved."
    radius_m = params.get("radius_m", 5000)
    result = slope_analysis(session.bbox.center_lat, session.bbox.center_lon, radius_m=radius_m, bbox=session.bbox)
    session.all_results["slope"] = result
    data = result.data
    safety = data.get("safety", "UNKNOWN")
    mean_sl = data.get("best_point", {}).get("mean_slope", data.get("distribution", {}).get("mean_slope", "?"))
    return result, f"Slope analysis: safety={safety}, mean slope={mean_sl} deg. {data.get('favorable_zones', 0)} favorable zones found."


async def _tool_subsurface(session: "AgentSession", params: dict):
    """Analyze SHARAD subsurface radar."""
    products = getattr(session, "_all_products", [])
    result = subsurface_scan(products)
    session.all_results["subsurface"] = result
    sub = result.data
    analyzed = sub.get("analyzed_count", 0)
    detections = sub.get("subsurface_detections", 0)
    if detections > 0:
        depth = sub.get("depth_summary", {})
        return result, (
            f"Analyzed {analyzed} SHARAD tracks. Found {detections} subsurface reflector(s). "
            f"Estimated ice depth: {depth.get('min_depth_m', '?')}-{depth.get('max_depth_m', '?')} m "
            f"(median {depth.get('median_depth_m', '?')} m, assuming er={depth.get('epsilon_r_assumed', 3.15)})."
        )
    return result, f"Analyzed {analyzed} SHARAD tracks. No subsurface reflectors detected (SNR below threshold)."


async def _tool_minerals(session: "AgentSession", params: dict):
    """Analyze CRISM mineral signatures."""
    products = getattr(session, "_all_products", [])
    result = mineral_analysis(products)
    session.all_results["mineral"] = result
    data = result.data
    crism_count = data.get("crism_count", 0)
    ice_count = data.get("high_ice_count", 0)
    hyd_count = data.get("high_hyd_count", 0)
    hotspot = data.get("ice_hotspot")
    obs = f"Analyzed {crism_count} CRISM products. {ice_count} with significant ice signatures, {hyd_count} with hydration."
    if hotspot:
        obs += f" Ice cluster centroid: ({hotspot['center_lat']:.2f}, {hotspot['center_lon']:.2f})."
    return result, obs


async def _tool_mineral_cnn(session: "AgentSession", params: dict):
    """Run CNN mineral classification."""
    products = getattr(session, "_all_products", [])
    result = await mineral_cnn_classify(products)
    session.all_results["mineral_cnn"] = result
    data = result.data
    classified = data.get("classified_count", 0)
    ice_detected = data.get("ice_detected", False)
    top_minerals = data.get("top_minerals", [])[:3]
    top_str = ", ".join(f"{m['name']} ({m['total_pixels']} px)" for m in top_minerals)
    obs = f"CNN classified {classified} observations. Ice detected: {ice_detected}."
    if top_str:
        obs += f" Top minerals: {top_str}."
    return result, obs


async def _tool_dielectric(session: "AgentSession", params: dict):
    """Estimate dielectric constant."""
    sub_result = session.all_results.get("subsurface")
    products = getattr(session, "_all_products", [])
    if not session.bbox:
        return TaskResult(task_type="dielectric", success=False, summary="No region", data={}), \
            "Error: No region resolved."
    result = dielectric_analysis(sub_result, products, session.bbox)
    session.all_results["dielectric"] = result
    data = result.data
    count = data.get("estimates_count", 0)
    if count > 0:
        return result, (
            f"Estimated dielectric constant from {count} SHARAD-DTM pairs. "
            f"Mean er={data.get('mean_epsilon_r', '?'):.2f}, interpretation: {data.get('interpretation', '?')}."
        )
    return result, "No dielectric estimates possible (need both SHARAD reflectors and nearby DTM data)."


async def _tool_recommend(session: "AgentSession", params: dict):
    """Cross-reference all data for site recommendation."""
    result = recommend_site(session.all_results)
    session.all_results["recommend"] = result
    data = result.data
    primary = data.get("primary_site")
    if primary:
        return result, (
            f"Primary landing site: ({primary['lat']:.2f}, {primary['lon']:.2f}), "
            f"score={primary.get('score', '?')}, reasons: {', '.join(primary.get('reasons', [])[:3])}."
        )
    candidates = data.get("candidates", [])
    return result, f"Found {len(candidates)} candidate sites but no clear primary recommendation."


async def _tool_climate(session: "AgentSession", params: dict):
    """Analyze Mars climate conditions for the target region."""
    if not session.bbox:
        return TaskResult(task_type="climate", success=False, error="No bbox"), "No region defined"
    data = climate_analysis_for_region(
        session.bbox.min_lat, session.bbox.max_lat,
        session.bbox.min_lon, session.bbox.max_lon,
    )
    result = TaskResult(
        task_type="climate",
        success=data.get("success", True),
        data=data,
        summary=data.get("climate_summary", "Climate analysis complete"),
    )
    session.all_results["climate"] = result
    return result, data.get("climate_summary", "Climate analysis complete")


async def _tool_thermal_inertia(session: "AgentSession", params: dict):
    """Analyze TES thermal inertia for the target region."""
    if not session.bbox:
        return TaskResult(task_type="thermal_inertia", success=False, error="No bbox"), "No region defined"
    # Determine if ice signal present from previous results
    has_ice = False
    mineral_r = session.all_results.get("mineral")
    if mineral_r and mineral_r.success:
        has_ice = mineral_r.data.get("high_ice_count", 0) > 0
    sub_r = session.all_results.get("subsurface")
    if sub_r and sub_r.success:
        has_ice = has_ice or sub_r.data.get("subsurface_detections", 0) > 0

    data = thermal_inertia_analysis_for_region(
        session.bbox.min_lat, session.bbox.max_lat,
        session.bbox.min_lon, session.bbox.max_lon,
        has_ice_signal=has_ice,
    )
    result = TaskResult(
        task_type="thermal_inertia",
        success=data.get("available", False),
        data=data,
        summary=data.get("ti_explanation", "Thermal inertia analysis complete"),
    )
    session.all_results["thermal_inertia"] = result
    return result, data.get("ti_explanation", "TI analysis complete")


# Tool registry — maps name → {description, params, executor}
AGENT_TOOLS: Dict[str, Dict[str, Any]] = {
    "resolve_region": {
        "description": "Look up a named Mars region and get its bounding box coordinates.",
        "params": {"region_name": "str — e.g. 'Jezero Crater', 'Arcadia Planitia'"},
        "executor": _tool_resolve_region,
    },
    "search_products": {
        "description": "Search for instrument products in the current region.",
        "params": {"instrument": "str — CRISM, HIRISE, SHARAD, SHARAD_HIGHRES, CTX, or HIRISE_DTM"},
        "executor": _tool_search_products,
    },
    "check_local_data": {
        "description": "Check which found products are available locally vs need downloading.",
        "params": {},
        "executor": _tool_check_local,
    },
    "download_products": {
        "description": "Download missing products (max 30, smart selection strategy).",
        "params": {"max_count": "int, optional — default 30"},
        "executor": _tool_download,
    },
    "analyze_slope": {
        "description": "Analyze terrain slope for rover/landing feasibility.",
        "params": {"radius_m": "int, optional — analysis radius in meters (default 5000)"},
        "executor": _tool_slope,
    },
    "analyze_subsurface": {
        "description": "Analyze SHARAD radar data for subsurface ice reflectors and depth estimates.",
        "params": {},
        "executor": _tool_subsurface,
    },
    "analyze_minerals": {
        "description": "Analyze CRISM spectral data for ice and hydration signatures.",
        "params": {},
        "executor": _tool_minerals,
    },
    "classify_minerals_cnn": {
        "description": "Run 1D CNN-Attention mineral classifier on CRISM TRR3 data (24 mineral classes).",
        "params": {},
        "executor": _tool_mineral_cnn,
    },
    "estimate_dielectric": {
        "description": "Estimate dielectric constant from SHARAD + HiRISE DTM. Indicates ice vs rock.",
        "params": {},
        "executor": _tool_dielectric,
    },
    "recommend_site": {
        "description": "Cross-reference all collected data to identify optimal landing/rover site.",
        "params": {},
        "executor": _tool_recommend,
    },
    "analyze_climate": {
        "description": "Analyze Mars climate conditions (temperature, dust, wind, frost) for the target region.",
        "params": {},
        "executor": _tool_climate,
    },
    "analyze_thermal_inertia": {
        "description": "Analyze TES thermal inertia to assess surface consolidation and ice-cemented regolith.",
        "params": {},
        "executor": _tool_thermal_inertia,
    },
    "finish": {
        "description": "Call when analysis is complete. Provide summary and recommendation.",
        "params": {"summary": "str — brief findings summary", "recommendation": "str — STRONG_CANDIDATE, PROMISING_WITH_CAVEATS, REQUIRES_FURTHER_INVESTIGATION, or LOW_PRIORITY"},
        "executor": None,
    },
}


def _build_react_prompt(objective: str, session: "AgentSession", history: list) -> str:
    """Build the per-turn prompt with full history.

    Ends with 'Thought:' prefix to force Llama into the correct format.
    """
    parts = [f"Objective: {objective}"]
    if session.region_name and session.bbox:
        b = session.bbox
        parts.append(f"Region: {session.region_name} (lat {b.min_lat:.1f}–{b.max_lat:.1f}, lon {b.min_lon:.1f}–{b.max_lon:.1f})")
    parts.append("")

    for i, entry in enumerate(history, 1):
        parts.append(f"--- Turn {i} ---")
        parts.append(f"Thought: {entry['thought']}")
        parts.append(f"Action: {json.dumps(entry['action'])}")
        parts.append(f"Observation: {entry['observation']}")
        parts.append("")

    parts.append(f"--- Turn {len(history) + 1} ---")
    # End with "Thought:" to prime Llama to continue in the correct format
    parts.append("Thought:")
    return "\n".join(parts)


def _parse_react_output(text: str, session: "AgentSession" = None) -> tuple:
    """Extract Thought and Action from Llama ReAct output.

    Handles common LLM format variations:
    - Markdown bold: **Thought:**
    - Code blocks: ```json { ... } ```
    - Mixed casing: thought:, THOUGHT:, Thought:
    - Single-quoted JSON
    - Missing Action: label (fallback: any JSON with "tool" key)
    - Completely unstructured text (fallback: NL intent inference)
    """
    thought = ""
    action = {}

    logger.info(f"Parsing ReAct output ({len(text)} chars): {text[:500]}...")

    if not text or not text.strip():
        logger.warning("Empty Llama response — nothing to parse")
        return "", {}

    # Normalize: strip markdown bold markers
    normalized = re.sub(r'\*\*', '', text)
    # Strip code block markers
    normalized = re.sub(r'```(?:json|JSON)?\s*', '', normalized)

    # Extract Thought — case-insensitive, multiple label variants
    for pattern in [
        r'[Tt]hought:\s*(.*?)(?=[Aa]ction\s*:|$)',
        r'[Rr]easoning:\s*(.*?)(?=[Aa]ction\s*:|$)',
        r'[Tt]hinking:\s*(.*?)(?=[Aa]ction\s*:|$)',
    ]:
        m = re.search(pattern, normalized, re.DOTALL)
        if m and m.group(1).strip():
            thought = m.group(1).strip()
            break

    # The prompt now ends with "Thought:" so Llama continues directly.
    # If no Thought:/Action: labels found, the text before the first JSON
    # IS the thought (Llama continued after our "Thought:" prefix).
    if not thought:
        first_brace = normalized.find('{')
        if first_brace > 0:
            thought = normalized[:first_brace].strip()
            thought = re.sub(r'[Aa]ction\s*:\s*$', '', thought).strip()
        elif '{' not in normalized:
            # No JSON at all — entire text is the thought (conversational response)
            thought = normalized.strip()[:300]

    # Extract Action JSON
    # 1. Try standard: Action: { ... }
    action_match = re.search(r'[Aa]ction\s*:\s*', normalized)
    if action_match:
        remaining = normalized[action_match.end():].strip()
        action = _extract_json_object(remaining)

    # 2. Fallback: find ANY JSON with a "tool" key anywhere in the output
    if not action or "tool" not in action:
        action = _find_tool_json(normalized)

    # 3. Last resort: infer tool from natural language
    if (not action or "tool" not in action) and session:
        logger.info("No structured action found — attempting NL inference")
        action = _infer_tool_from_text(normalized, session)
        if action and "tool" in action:
            logger.info(f"NL inference produced tool: {action['tool']}")

    if not action or "tool" not in action:
        logger.warning(f"Failed to parse action from Llama output: {text[:500]}")

    return thought, action


def _extract_json_object(text: str) -> dict:
    """Extract the first JSON object from text using bracket-counting.
    Handles strings properly (ignores braces inside quoted strings)."""
    start = text.find('{')
    if start < 0:
        return {}

    depth = 0
    in_string = False
    escape_next = False

    for i in range(start, len(text)):
        c = text[i]
        if escape_next:
            escape_next = False
            continue
        if c == '\\':
            escape_next = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                json_str = text[start:i + 1]
                # Try parsing as-is
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    pass
                # Try fixing single quotes → double quotes
                try:
                    fixed = json_str.replace("'", '"')
                    return json.loads(fixed)
                except json.JSONDecodeError:
                    pass
                return {}
    return {}


def _find_tool_json(text: str) -> dict:
    """Find any JSON object in text that contains a 'tool' key."""
    for m in re.finditer(r'\{', text):
        result = _extract_json_object(text[m.start():])
        if result and "tool" in result:
            return result
    return {}


def _infer_tool_from_text(text: str, session: "AgentSession") -> dict:
    """Last-resort: infer a tool call from natural language when Llama
    ignores the structured format entirely.

    This handles cases where Llama responds conversationally like:
    'I'll start by looking up the region...' or
    'Let me search for CRISM data in this area.'
    """
    t = text.lower()

    # Detect region resolution intent
    if any(kw in t for kw in ["resolve", "look up", "find the region", "locate", "identify the region", "bounding box"]):
        region = session.region_name or "target region"
        return {"tool": "resolve_region", "params": {"region_name": region}}

    # Detect instrument search intent
    for inst in ["crism", "hirise", "sharad_highres", "sharad", "ctx", "hirise_dtm"]:
        if inst.replace("_", " ") in t or inst in t:
            if any(kw in t for kw in ["search", "find", "look for", "check", "query", "available", "data"]):
                return {"tool": "search_products", "params": {"instrument": inst.upper()}}

    # Detect download intent
    if any(kw in t for kw in ["download", "fetch", "retrieve", "get the data", "acquire"]):
        return {"tool": "download_products", "params": {}}

    # Detect analysis intents
    if any(kw in t for kw in ["slope", "terrain", "landing", "rover feasibility", "engineering"]):
        return {"tool": "analyze_slope", "params": {}}
    if any(kw in t for kw in ["subsurface", "radar", "sharad analysis", "ice depth", "reflector"]):
        return {"tool": "analyze_subsurface", "params": {}}
    if any(kw in t for kw in ["mineral", "spectral", "crism analysis", "hydration", "ice signature"]):
        return {"tool": "analyze_minerals", "params": {}}
    if any(kw in t for kw in ["cnn", "classifier", "classification", "24 mineral"]):
        return {"tool": "classify_minerals_cnn", "params": {}}
    if any(kw in t for kw in ["dielectric", "permittivity", "ice vs rock"]):
        return {"tool": "estimate_dielectric", "params": {}}
    if any(kw in t for kw in ["climate", "temperature", "dust", "wind", "frost"]):
        return {"tool": "analyze_climate", "params": {}}
    if any(kw in t for kw in ["thermal inertia", "surface consolidation", "tes"]):
        return {"tool": "analyze_thermal_inertia", "params": {}}
    if any(kw in t for kw in ["recommend", "optimal site", "best location", "landing site"]):
        return {"tool": "recommend_site", "params": {}}
    if any(kw in t for kw in ["local data", "locally available", "check what we have", "check_local"]):
        return {"tool": "check_local_data", "params": {}}
    if any(kw in t for kw in ["finish", "conclude", "final assessment", "done", "complete"]):
        return {"tool": "finish", "params": {"summary": text[:200], "recommendation": "REQUIRES_FURTHER_INVESTIGATION"}}

    # If session has no bbox yet, default to resolving region
    if not session.bbox:
        region = session.region_name or "target region"
        return {"tool": "resolve_region", "params": {"region_name": region}}

    # If no searches done yet, default to CRISM search
    if not any(k.startswith("search_") for k in session.all_results):
        return {"tool": "search_products", "params": {"instrument": "CRISM"}}

    return {}


# =============================================================================
# Plan Generation (used by rule-based fallback)
# =============================================================================

PLAN_SYSTEM_PROMPT = """You are a Mars science mission planner AI. Given a user's mission objective,
generate an execution plan as a JSON array of steps.

Each step must have:
- "type": one of "search", "check_data", "download", "slope", "subsurface", "mineral", "mineral_cnn", "dielectric", "climate", "thermal_inertia", "synthesize"
- "description": human-readable description
- "instrument": instrument name if applicable (CRISM, HIRISE, SHARAD, SHARAD_HIGHRES, CTX, HIRISE_DTM)

Also extract:
- "region": the Mars region name or coordinates mentioned
- "instruments": list of all instruments to search

IMPORTANT: Always include SHARAD_HIGHRES for subsurface ice analysis (high-res radargrams from local index).
Always include HIRISE_DTM for slope/terrain analysis (local index, instant search).
These are locally-indexed instruments and do not require remote API calls.
Include mineral_cnn step after mineral step if CRISM is present (runs 1D CNN-Attention mineral classification on TRR3 data).
Include dielectric step after subsurface step if SHARAD_HIGHRES and HIRISE_DTM are present (estimates dielectric constant from radar + elevation).

Return ONLY valid JSON with this structure:
{
  "region": "region name or coordinates",
  "instruments": ["CRISM", "SHARAD", ...],
  "steps": [...]
}"""


async def _generate_plan_llm(
    objective: str,
    science_context: str = "",
    on_chunk: Optional[Callable[[str], Awaitable[None]]] = None,
) -> Dict[str, Any]:
    """Use Llama to generate an execution plan from the objective."""
    context_block = ""
    if science_context:
        context_block = f"""
--- Science Context ---
{science_context}
--- End Context ---

Use the above context to inform your plan. Prioritize instruments and analyses
relevant to the region's known characteristics (e.g., if ice confidence is high,
include SHARAD subsurface scan; if minerals are known, include CRISM analysis).

"""

    prompt = f"""Mission Objective: {objective}

{context_block}Generate an execution plan as JSON. Include search steps for relevant instruments,
data availability check, downloads if needed, analysis steps (slope for engineering,
SHARAD for subsurface ice, CRISM for mineral signatures), and a final synthesis step.

Return ONLY the JSON object, no other text."""

    if on_chunk:
        response = await _call_ollama_streaming(prompt, PLAN_SYSTEM_PROMPT, on_chunk=on_chunk)
    else:
        response = await _call_ollama(prompt, PLAN_SYSTEM_PROMPT)

    # Parse JSON from response
    try:
        # Try to extract JSON from the response
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            return json.loads(json_match.group())
    except json.JSONDecodeError:
        pass

    return {}


def _generate_plan_rules(objective: str) -> Dict[str, Any]:
    """
    Rule-based fallback plan generator.

    Parses the objective for keywords and generates appropriate steps.
    """
    obj_lower = objective.lower()

    # Detect region
    region_name = None
    # Try common Mars regions
    from .mars_regions import MARS_REGIONS
    for r in MARS_REGIONS.values():
        if r.display_name.lower() in obj_lower or r.region_id.lower() in obj_lower:
            region_name = r.display_name
            break

    # Detect instruments from keywords
    instruments = []
    instrument_map = {
        "CRISM": ["crism", "mineral", "spectral", "composition", "ice score", "hydration"],
        "HIRISE": ["hirise", "high resolution", "surface detail", "imagery"],
        "SHARAD": ["sharad", "radar", "subsurface", "ice detect"],
        "SHARAD_HIGHRES": ["sharad high-res", "sharad highres", "high-res radar"],
        "CTX": ["ctx", "context"],
        "HIRISE_DTM": ["dtm", "elevation", "terrain model", "topography"],
    }

    for inst, keywords in instrument_map.items():
        for kw in keywords:
            if kw in obj_lower:
                if inst not in instruments:
                    instruments.append(inst)
                break

    # If objective mentions general analysis needs, include standard instruments
    general_keywords = ["ice", "rover", "traverse", "landing", "feasibility", "shallow ice",
                        "engineering", "mission", "explore", "survey"]
    if any(kw in obj_lower for kw in general_keywords) and len(instruments) < 3:
        for default in ["CRISM", "SHARAD", "SHARAD_HIGHRES", "CTX", "HIRISE_DTM"]:
            if default not in instruments:
                instruments.append(default)

    if not instruments:
        instruments = ["CRISM", "SHARAD", "SHARAD_HIGHRES", "CTX", "HIRISE_DTM"]

    # Build steps
    steps = []

    # 1. Search each instrument
    for inst in instruments:
        steps.append({
            "type": "search",
            "description": f"Search for {inst} products in the target region",
            "instrument": inst,
        })

    # 2. Check local data
    steps.append({
        "type": "check_data",
        "description": "Check which products are already available locally",
    })

    # 3. Download missing data
    steps.append({
        "type": "download",
        "description": "Download missing products needed for analysis",
    })

    # 4. Analysis steps based on objective
    if any(kw in obj_lower for kw in ["slope", "terrain", "engineering", "feasibility", "rover", "landing", "traverse"]):
        steps.append({
            "type": "slope",
            "description": "Analyze terrain slope for engineering feasibility",
        })

    if any(kw in obj_lower for kw in ["subsurface", "radar", "sharad", "ice"]) or "SHARAD" in instruments:
        steps.append({
            "type": "subsurface",
            "description": "Assess SHARAD subsurface radar coverage for ice detection",
        })
        # Dielectric estimation if both SHARAD and DTM available
        if "HIRISE_DTM" in instruments:
            steps.append({
                "type": "dielectric",
                "description": "Estimate dielectric constant using SHARAD radar + HiRISE DTM elevation",
            })

    if "CRISM" in instruments or any(kw in obj_lower for kw in ["mineral", "ice", "hydration", "composition"]):
        steps.append({
            "type": "mineral",
            "description": "Analyze CRISM mineral signatures for ice/hydration indicators",
        })
        # CNN mineral classification on TRR3 data
        steps.append({
            "type": "mineral_cnn",
            "description": "Run 1D CNN-Attention mineral classification on CRISM TRR3 data",
        })

    # 5. Climate + Thermal Inertia (always run for comprehensive assessment)
    steps.append({
        "type": "climate",
        "description": "Analyze Mars climate conditions (temperature, dust, wind, frost)",
    })
    steps.append({
        "type": "thermal_inertia",
        "description": "Analyze TES thermal inertia for surface consolidation and ice indicators",
    })

    # 6. Recommend best site
    if any(kw in obj_lower for kw in ["rover", "landing", "site", "traverse", "location", "where", "best"]):
        steps.append({
            "type": "recommend",
            "description": "Cross-reference all data to identify the optimal rover site",
        })

    # 7. Synthesize
    steps.append({
        "type": "synthesize",
        "description": "Combine all analyses into a comprehensive assessment",
    })

    return {
        "region": region_name,
        "instruments": instruments,
        "steps": steps,
    }


# =============================================================================
# Download Strategy
# =============================================================================

MAX_DOWNLOADS_PER_RUN = 30


def _select_downloads(
    missing: List[Dict[str, Any]],
    bbox: Optional["RegionBBox"],
    max_total: int = MAX_DOWNLOADS_PER_RUN,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Select up to *max_total* products to download using sparse or dense strategy.

    Strategy:
    - **dense** (region area ≤ 10 deg²): prioritize products closest to center
    - **sparse** (region area > 10 deg²): prioritize spatial spread for broad coverage

    Priority tiers (science-critical first):
      SHARAD_HIGHRES > CRISM > SHARAD > HIRISE > CTX/HIRISE_DTM
    """
    if len(missing) <= max_total:
        return missing, "all"

    # Determine strategy from region size
    if bbox:
        lat_span = bbox.max_lat - bbox.min_lat
        lon_span = bbox.max_lon - bbox.min_lon
        area = lat_span * lon_span
        center_lat, center_lon = bbox.center_lat, bbox.center_lon
    else:
        area = 0
        center_lat, center_lon = 0.0, 0.0

    strategy = "sparse" if area > 10 else "dense"

    PRIORITY = {
        "SHARAD_HIGHRES": 0, "CRISM": 1, "SHARAD": 2,
        "HIRISE": 3, "CTX": 4, "HIRISE_DTM": 4,
    }

    def _dist(p: Dict) -> float:
        lat = p.get("lat") or center_lat
        lon = p.get("lon") or center_lon
        return ((lat - center_lat) ** 2 + (lon - center_lon) ** 2) ** 0.5

    # Group by priority tier
    by_tier: Dict[int, List[Dict]] = {}
    for p in missing:
        tier = PRIORITY.get(p.get("instrument", "").upper(), 5)
        by_tier.setdefault(tier, []).append(p)

    # Sort within each tier by distance (ascending for dense, descending for sparse)
    for tier_key in by_tier:
        by_tier[tier_key].sort(key=_dist, reverse=(strategy == "sparse"))

    # Allocate budget across tiers proportionally, minimum 1 per tier
    selected: List[Dict] = []
    remaining = max_total
    total_missing = len(missing)

    for tier_key in sorted(by_tier.keys()):
        if remaining <= 0:
            break
        tier_products = by_tier[tier_key]
        # Proportional allocation
        alloc = max(1, round(remaining * len(tier_products) / max(1, total_missing)))
        alloc = min(alloc, len(tier_products), remaining)

        if strategy == "sparse" and alloc < len(tier_products):
            # Evenly sample across the spatial spread
            step = len(tier_products) / alloc
            tier_selected = [tier_products[int(i * step)] for i in range(alloc)]
        else:
            tier_selected = tier_products[:alloc]

        selected.extend(tier_selected)
        remaining -= len(tier_selected)
        total_missing -= len(tier_products)

    return selected[:max_total], strategy


# =============================================================================
# Narrative Generation
# =============================================================================

NARRATIVE_SYSTEM_PROMPT = """You are a Mars mission assessment analyst producing a decision-oriented report.
Write like a mission assessment document, not a summary or blog post.

STRUCTURE (follow this order strictly):
1. **Objective** — Paraphrase the user's request in scientific language. Do NOT copy verbatim.
2. **Subsurface Potential** — SHARAD analysis first. Cite εr, depth in meters, detection confidence.
3. **Surface Composition** — CRISM ice/hydration. Spatial distribution, ranked candidates.
4. **Cross-Instrument Consistency** — Where SHARAD + CRISM agree or disagree spatially.
   State distances, whether evidence is direct (SHARAD) or indirect (CRISM proxy).
5. **Engineering Feasibility** — Slope/terrain as a FINAL FILTER, not an initial driver.
6. **Landing Site Decision** — State a single Primary Landing Site coordinate.
   Optionally a Secondary/Backup site. For each: location, why chosen, what was sacrificed.

RULES:
- Be explicit about uncertainty. Use trade-off language ("we prioritize X over Y").
- Do NOT emphasize coverage counts (product numbers) as primary results.
- Do NOT default to maximum optimism. Acknowledge gaps and limitations.
- Include specific numbers: coordinates, slope degrees, depth meters, ice percentages.
- End with a clear, concrete recommendation tied to a single coordinate."""


async def _generate_narrative(
    session: AgentSession,
    on_chunk: Optional[Callable[[str], Awaitable[None]]] = None,
) -> str:
    """Use Llama to generate a narrative summary from the synthesis results."""
    synthesis = session.synthesis or {}

    # Inject region science context into narrative prompt
    region_ctx = ""
    if session.region_name:
        ctx = get_region_context_by_name(session.region_name)
        if ctx:
            region_ctx = f"\nRegion Science Context:\n{ctx}\n"

    # Build structured analysis data for the LLM
    sub = synthesis.get("subsurface_coverage", {})
    mineral = synthesis.get("mineral_signatures", {})
    eng = synthesis.get("engineering_feasibility", {})
    cross = synthesis.get("cross_instrument", {})
    rec_data = synthesis.get("recommended_site", {})
    primary = rec_data.get("primary_site") or rec_data.get("best_site")
    secondary = rec_data.get("secondary_site")
    science_targets = rec_data.get("science_targets", [])
    trade_offs = rec_data.get("trade_offs", [])
    score_range = synthesis.get("score_range", {})

    prompt = f"""Write a decision-oriented mission assessment for {session.region_name or 'Unknown'}.

Objective to paraphrase (DO NOT copy verbatim): {session.objective}
{region_ctx}
--- SHARAD Subsurface Analysis ---
{json.dumps(sub, indent=2)}

--- CRISM Mineral/Ice Analysis ---
{json.dumps(mineral, indent=2)}

--- Cross-Instrument Consistency ---
{json.dumps(cross, indent=2)}

--- Engineering / Slope (final filter) ---
{json.dumps(eng, indent=2)}

--- Landing Site Decision ---
Primary site: {json.dumps(primary) if primary else 'None identified'}
Secondary site: {json.dumps(secondary) if secondary else 'None'}
Science targets: {json.dumps(science_targets) if science_targets else 'None'}
Trade-offs: {json.dumps(trade_offs) if trade_offs else 'None'}

Score: {score_range.get('low', 0)}-{score_range.get('high', 0)}/100
Strengths: {json.dumps(synthesis.get('strengths', []))}
Uncertainties: {json.dumps(synthesis.get('uncertainties', []))}

Follow the structure: Subsurface first → CRISM → Cross-instrument → Engineering → Landing Decision.
Slope does NOT drive site selection; it is a final feasibility filter.
State a concrete Primary Landing Site with coordinates and trade-off reasoning."""

    if on_chunk:
        response = await _call_ollama_streaming(prompt, NARRATIVE_SYSTEM_PROMPT, temperature=0.5, on_chunk=on_chunk)
    else:
        response = await _call_ollama(prompt, NARRATIVE_SYSTEM_PROMPT, temperature=0.5)

    if not response:
        return _generate_narrative_fallback(session)

    return response


def _generate_narrative_fallback(session: AgentSession) -> str:
    """Generate a decision-oriented narrative without LLM.

    Follows the mandated order: SHARAD → CRISM → Cross-Instrument → Slope → Decision.
    """
    synthesis = session.synthesis or {}
    parts = []
    region = session.region_name or "Target Region"

    parts.append(f"## Mission Assessment: {region}\n")
    parts.append(
        f"This assessment evaluates the suitability of {region} for a surface mission, "
        f"prioritizing subsurface ice potential and surface composition evidence before "
        f"applying engineering terrain constraints as a final filter.\n"
    )

    # ── 1. Subsurface Potential (SHARAD) ──
    sub = synthesis.get("subsurface_coverage", {})
    if sub:
        parts.append("### Subsurface Potential (SHARAD Radar)")
        n_analyzed = sub.get("analyzed_count", 0)
        n_detect = sub.get("subsurface_detections", 0)
        depth = sub.get("depth_summary")

        if n_analyzed > 0 and n_detect > 0:
            parts.append(
                f"SHARAD radargram analysis of {n_analyzed} tracks yielded "
                f"**{n_detect} subsurface reflector detection(s)**, providing direct "
                f"evidence of a dielectric interface beneath the surface."
            )
            if depth:
                try:
                    med_d = float(depth.get('median_depth_m', 50))
                except (TypeError, ValueError):
                    med_d = 50.0
                if med_d <= 10:
                    depth_qual = "This is **shallow ice** — potentially accessible for in-situ resource extraction without deep drilling."
                elif med_d <= 30:
                    depth_qual = "This is a moderate depth — drilling infrastructure would be required for extraction."
                elif med_d <= 80:
                    depth_qual = "This depth is significant and would require substantial drilling capability."
                else:
                    depth_qual = "This depth is **too deep for practical ice extraction** and reduces the site's utility for ISRU."
                parts.append(
                    f"Assuming a water-ice dielectric constant (εr = {depth.get('epsilon_r_assumed', 3.15)}), "
                    f"the estimated ice-table depth ranges from **{depth['min_depth_m']} to "
                    f"{depth['max_depth_m']} m** (median {depth['median_depth_m']} m). "
                    f"{depth_qual}"
                )
        elif n_analyzed > 0:
            parts.append(
                f"{n_analyzed} SHARAD radargrams were analyzed but no subsurface "
                f"reflectors met the detection threshold (SNR >= 2.5). This does not "
                f"rule out subsurface ice — it may be below SHARAD vertical resolution "
                f"(~15 m) or obscured by surface clutter."
            )
        else:
            parts.append(
                f"SHARAD coverage: {sub.get('coverage', 'NONE')} ({sub.get('total_tracks', 0)} tracks). "
                f"No high-resolution radargram data was available for quantitative analysis."
            )
        parts.append("")

    # ── 2. Surface Composition (CRISM) ──
    mineral = synthesis.get("mineral_signatures", {})
    if mineral:
        parts.append("### Surface / Near-Surface Composition (CRISM)")
        high_ice = mineral.get("high_ice_count", 0)
        high_hyd = mineral.get("high_hyd_count", 0)

        if high_ice > 0 or high_hyd > 0:
            parts.append(
                f"Of {mineral.get('crism_count', 0)} CRISM products analyzed, "
                f"**{high_ice}** exhibit significant ice spectral signatures "
                f"(>5% of pixels above 0.3 threshold) and {high_hyd} show "
                f"hydration indicators."
            )
        else:
            parts.append(
                f"{mineral.get('crism_count', 0)} CRISM products were analyzed. "
                f"No significant ice or hydration signatures exceeded the detection threshold."
            )

        hotspot = mineral.get("ice_hotspot")
        if hotspot:
            parts.append(
                f"Ice signatures cluster near **({hotspot['center_lat']:.2f}, "
                f"{hotspot['center_lon']:.2f})**, representing the centroid of "
                f"{hotspot['n_products']} high-ice observations "
                f"(maximum {hotspot['max_ice_pct']}% ice coverage). "
                f"CRISM detections represent indirect (spectral proxy) evidence "
                f"of surface or near-surface ice/hydration."
            )
        top = mineral.get("top_ice_candidates", [])
        if top:
            best_c = top[0]
            parts.append(
                f"Strongest ice candidate: **{best_c.get('obs_id', '?')}** at "
                f"({best_c.get('lat', 0):.2f}, {best_c.get('lon', 0):.2f}) with "
                f"{best_c.get('ice_percent', 0)}% ice-indicative pixels."
            )
        parts.append("")

    # ── 3. Cross-Instrument Consistency ──
    cross = synthesis.get("cross_instrument", {})
    if cross and cross.get("notes"):
        parts.append("### Cross-Instrument Consistency Analysis")
        for note in cross["notes"]:
            parts.append(note)

        dist = cross.get("sharad_crism_min_distance_km")
        if dist is not None:
            parts.append(
                f"\nMinimum SHARAD–CRISM separation: **{dist:.0f} km**. "
                f"Coincident detections (SHARAD track within 100 km of CRISM ice): "
                f"**{cross.get('coincident_detections', 0)}**."
            )

        n_direct = len(cross.get("direct_ice_evidence", []))
        n_indirect = len(cross.get("indirect_ice_evidence", []))
        if n_direct or n_indirect:
            parts.append(
                f"\nEvidence summary: {n_direct} direct (SHARAD reflector) and "
                f"{n_indirect} indirect (CRISM spectral) ice indicators."
            )
        parts.append("")

    # ── 4. Engineering Feasibility (slope — final filter) ──
    eng = synthesis.get("engineering_feasibility", {})
    if eng:
        parts.append("### Engineering Feasibility (Terrain — Final Filter)")
        parts.append(
            f"Terrain assessment is applied as a final constraint, not as the "
            f"primary site-selection driver."
        )
        parts.append(
            f"Overall safety: **{eng.get('safety', 'UNKNOWN')}** — "
            f"mean slope {eng.get('mean_slope', 'N/A')} deg, "
            f"max {eng.get('max_slope', 'N/A')} deg."
        )
        grid_n = eng.get("grid_size", 0)
        if grid_n:
            parts.append(
                f"Grid analysis ({grid_n} points): {eng.get('favorable_zones', 0)} FAVORABLE, "
                f"{eng.get('grid_size', 0) - eng.get('favorable_zones', 0)} marginal or unfavorable."
            )
        parts.append("")

    # ── 5. Climate Constraints ──
    clim = synthesis.get("climate", {})
    if clim:
        parts.append("### Climate Constraints")
        summary = clim.get("climate_summary", "")
        clim_score = clim.get("climate_score", 0)
        if summary:
            parts.append(summary)
        parts.append(f"Climate score: **{clim_score}/10**.")
        annual = clim.get("annual_stats", {})
        if annual:
            frost = annual.get("frost_max_probability", 0)
            if frost > 0.5:
                parts.append(
                    f"⚠ Significant CO2 frost risk ({frost:.0%}) — "
                    f"seasonal operations may be limited."
                )
        parts.append("")

    # ── 6. Thermal Inertia ──
    ti = synthesis.get("thermal_inertia", {})
    if ti:
        parts.append("### Thermal Inertia (TES)")
        explanation = ti.get("ti_explanation", "")
        if explanation:
            parts.append(explanation)
        ti_score = ti.get("ti_score", 0)
        parts.append(f"Thermal inertia score: **{ti_score}/10**.")
        parts.append("")

    # ── 7. Landing Site Decision ──
    rec_data = synthesis.get("recommended_site", {})
    primary = rec_data.get("primary_site") or rec_data.get("best_site")
    secondary = rec_data.get("secondary_site")
    science_targets = rec_data.get("science_targets", [])
    trade_offs = rec_data.get("trade_offs", [])

    if primary and primary.get("lat") is not None:
        parts.append("### Landing Site Decision")
        parts.append(
            f"**Primary Landing Site:** ({primary['lat']:.3f}, {primary['lon']:.3f}) — "
            f"composite score {primary.get('score', 'N/A')}/100, mean slope {primary.get('mean_slope', 'N/A')} deg"
        )
        for r in primary.get("reasons", []):
            parts.append(f"  - {r}")

        if secondary and secondary.get("lat") is not None:
            parts.append(
                f"\n**Secondary (Backup) Site:** ({secondary['lat']:.3f}, {secondary['lon']:.3f}) — "
                f"score {secondary.get('score', 'N/A')}/100, slope {secondary.get('mean_slope', 'N/A')} deg"
            )

        if science_targets:
            parts.append("\n**Science Targets** (non-landing, traverse candidates):")
            for st in science_targets:
                lat_s = f"{st['lat']:.2f}" if st.get('lat') is not None else "N/A"
                lon_s = f"{st['lon']:.2f}" if st.get('lon') is not None else "N/A"
                parts.append(f"  - ({lat_s}, {lon_s}): {st.get('reason', 'High science value')}")

        if trade_offs:
            parts.append("\n**Trade-offs:**")
            for t in trade_offs:
                parts.append(f"  - {t}")
        parts.append("")

    # ── Score + strengths/uncertainties ──
    score_range = synthesis.get("score_range", {})
    strengths = synthesis.get("strengths", [])
    uncertainties_list = synthesis.get("uncertainties", [])

    parts.append("### Assessment Summary")
    lo = score_range.get("low", synthesis.get("overall_score", 0))
    hi = score_range.get("high", synthesis.get("overall_score", 0))
    rec = synthesis.get("recommendation", "N/A").replace("_", " ")
    parts.append(f"**Score: {lo}–{hi} / 100** | Recommendation: {rec}")
    if strengths:
        parts.append("\n**Strengths:**")
        for s in strengths:
            parts.append(f"  - {s}")
    if uncertainties_list:
        parts.append("\n**Uncertainties:**")
        for u in uncertainties_list:
            parts.append(f"  - {u}")

    return "\n".join(parts)


# =============================================================================
# Main Agent Loop — ReAct (Reason + Act)
# =============================================================================

async def run_agent(
    objective: str,
    auto_download: bool = True,
    _session: Optional[AgentSession] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Main agent entry point. Yields SSE events.

    Uses ReAct (Reason+Act) loop when Ollama is available,
    falls back to rule-based pipeline otherwise.
    """
    # Session setup
    if _session is not None:
        session = _session
        session_id = session.session_id
    else:
        session_id = str(uuid.uuid4())[:8]
        session = AgentSession(session_id=session_id, objective=objective)
        _sessions[session_id] = session

    # Check Ollama availability
    ollama_available = await _check_ollama()

    if not ollama_available:
        # Fallback to rule-based pipeline (LLM unavailable)
        logger.info("Ollama unavailable — falling back to rule-based pipeline")
        async for event in _run_agent_rules(objective, auto_download, _session=session):
            yield event
        return

    # ── ReAct Agent Loop ─────────────────────────────
    logger.info("Starting ReAct agent loop with Ollama/Llama")
    yield {"event": "session_start", "data": {
        "session_id": session_id, "objective": objective, "mode": "react",
    }}

    reasoning_queue: asyncio.Queue[Optional[Dict[str, Any]]] = asyncio.Queue()

    async def _emit_chunk(text: str):
        await reasoning_queue.put({"event": "thought_chunk", "data": {"text": text}})

    try:
        session.status = "executing"
        session._all_products = []
        session._auto_download = auto_download

        # Early region resolution from objective text
        from .mars_regions import MARS_REGIONS
        obj_lower = objective.lower()
        for r in MARS_REGIONS.values():
            if r.display_name.lower() in obj_lower or r.region_id.lower() in obj_lower:
                session.region_name = r.display_name
                session.bbox = resolve_region(r.display_name)
                break

        # If no named region, try coordinates
        if not session.bbox:
            coord_match = re.search(
                r'(-?\d+\.?\d*)\s*[,\s]\s*(-?\d+\.?\d*)', objective
            )
            if coord_match:
                lat, lon = float(coord_match.group(1)), float(coord_match.group(2))
                session.bbox = bbox_from_coords(lat, lon, radius_deg=3.0)
                session.region_name = f"({lat}, {lon})"

        history: List[Dict[str, Any]] = []
        iteration = 0
        consecutive_errors = 0

        while iteration < MAX_ITERATIONS:
            iteration += 1

            # 1. Build prompt with full history
            system_prompt = REACT_SYSTEM_PROMPT.format(
                tools=_format_tool_descriptions()
            )
            prompt = _build_react_prompt(objective, session, history)

            # 2. Stream Llama thinking
            yield {"event": "thought_start", "data": {"iteration": iteration}}

            full_response = ""

            async def _do_think():
                nonlocal full_response
                full_response = await _call_ollama_streaming(
                    prompt, system_prompt, on_chunk=_emit_chunk
                )
                await reasoning_queue.put(None)  # sentinel

            think_task = asyncio.create_task(_do_think())
            while True:
                item = await reasoning_queue.get()
                if item is None:
                    break
                yield item
            await think_task

            # 3. Parse output → thought + action
            thought, action = _parse_react_output(full_response, session=session)
            yield {"event": "thought_end", "data": {
                "thought": thought, "iteration": iteration,
            }}

            # Handle empty/malformed output
            if not action or "tool" not in action:
                consecutive_errors += 1
                # Give progressively more specific hints
                if consecutive_errors == 1:
                    observation = (
                        "Error: Could not parse your action. Remember the format:\n"
                        "Thought: your reasoning here\n"
                        'Action: {"tool": "resolve_region", "params": {"region_name": "Jezero Crater"}}\n'
                        "Try again with this exact format."
                    )
                elif consecutive_errors == 2:
                    # Suggest a specific action based on context
                    if not session.bbox:
                        suggestion = '{"tool": "resolve_region", "params": {"region_name": "' + (session.region_name or "the target region") + '"}}'
                    elif not any(k.startswith("search_") for k in session.all_results):
                        suggestion = '{"tool": "search_products", "params": {"instrument": "CRISM"}}'
                    else:
                        suggestion = '{"tool": "analyze_subsurface", "params": {}}'
                    observation = (
                        f"Error: Still could not parse. Output ONLY this:\n"
                        f"Thought: Proceeding with analysis.\n"
                        f"Action: {suggestion}"
                    )
                else:
                    observation = (
                        "Error: Parse failed 3 times. The agent will auto-recover "
                        "by running standard searches and analyses."
                    )
                history.append({
                    "thought": thought or "(no thought parsed)",
                    "action": {"tool": "parse_error"},
                    "observation": observation,
                })
                yield {"event": "action_complete", "data": {
                    "tool": "parse_error", "observation": observation,
                    "success": False, "iteration": iteration,
                }}
                if consecutive_errors >= 3:
                    logger.warning(
                        "Too many consecutive parse errors — "
                        "will auto-recover with fallback searches"
                    )
                    break
                continue

            consecutive_errors = 0
            tool_name = action.get("tool", "")
            params = action.get("params", {})

            # 4. Handle finish action
            if tool_name == "finish":
                yield {"event": "action", "data": {
                    "tool": "finish", "params": params,
                    "iteration": iteration,
                }}
                break

            # 5. Validate tool name
            if tool_name not in AGENT_TOOLS:
                observation = (
                    f"Error: Unknown tool '{tool_name}'. "
                    f"Available: {', '.join(AGENT_TOOLS.keys())}"
                )
                history.append({
                    "thought": thought, "action": action,
                    "observation": observation,
                })
                yield {"event": "action_complete", "data": {
                    "tool": tool_name, "observation": observation,
                    "success": False, "iteration": iteration,
                }}
                continue

            # 6. Execute tool
            tool_def = AGENT_TOOLS[tool_name]
            yield {"event": "action_start", "data": {
                "tool": tool_name, "params": params,
                "iteration": iteration,
            }}

            try:
                if tool_name == "download_products":
                    # Special handling: stream download progress events
                    progress_q: asyncio.Queue = asyncio.Queue()

                    async def _on_dl_progress(completed, failed_n, skipped_n, total):
                        await progress_q.put({"event": "download_progress", "data": {
                            "completed": completed, "failed": failed_n,
                            "skipped": skipped_n, "total": total,
                        }})

                    async def _do_download():
                        res, obs = await _tool_download_with_progress(
                            session, params, _on_dl_progress
                        )
                        await progress_q.put(None)  # sentinel
                        return res, obs

                    dl_task = asyncio.create_task(_do_download())
                    while True:
                        try:
                            item = await asyncio.wait_for(
                                progress_q.get(), timeout=2.0
                            )
                        except asyncio.TimeoutError:
                            if dl_task.done():
                                break
                            continue
                        if item is None:
                            break
                        yield item
                    result, observation = await dl_task
                else:
                    result, observation = await tool_def["executor"](
                        session, params
                    )

                yield {"event": "action_complete", "data": {
                    "tool": tool_name, "observation": observation,
                    "success": result.success, "iteration": iteration,
                }}

                # Record as session step for history/persistence
                step = AgentStep(
                    id=str(uuid.uuid4())[:6],
                    type=tool_name,
                    description=thought[:120] if thought else tool_name,
                    status=(StepStatus.COMPLETED if result.success
                            else StepStatus.FAILED),
                    result=result,
                )
                session.steps.append(step)

            except Exception as e:
                observation = f"Error executing {tool_name}: {e}"
                logger.error(f"Tool {tool_name} failed: {e}")
                yield {"event": "action_complete", "data": {
                    "tool": tool_name, "observation": observation,
                    "success": False, "iteration": iteration,
                }}
                step = AgentStep(
                    id=str(uuid.uuid4())[:6],
                    type=tool_name,
                    description=thought[:120] if thought else tool_name,
                    status=StepStatus.FAILED,
                    error=str(e),
                )
                session.steps.append(step)

            # 7. Append to history for next Llama turn
            history.append({
                "thought": thought, "action": action,
                "observation": observation,
            })

        # ── Safety net: if agent loop ended without any searches, auto-run ──
        has_searches = any(
            k.startswith("search_") for k in session.all_results
        )
        if not has_searches and session.bbox:
            logger.warning(
                "ReAct loop ended without any searches — "
                "running fallback data collection"
            )
            yield {"event": "action_start", "data": {
                "tool": "_auto_recovery", "params": {},
                "iteration": iteration + 1,
            }}

            # Run all instrument searches
            fallback_instruments = [
                "CRISM", "HIRISE", "SHARAD", "SHARAD_HIGHRES",
                "CTX", "HIRISE_DTM",
            ]
            for inst in fallback_instruments:
                try:
                    result = await search_region(inst, session.bbox)
                    session.all_results[f"search_{inst}"] = result
                    if result.success:
                        products = result.data.get("products", [])
                        if not hasattr(session, "_all_products"):
                            session._all_products = []
                        session._all_products.extend(products)
                except Exception as e:
                    logger.error(f"Fallback search {inst} failed: {e}")

            # Check local data + run basic analyses
            products = getattr(session, "_all_products", [])
            if products:
                check_result = check_local_data(products)
                session.all_results["check_local"] = check_result

                try:
                    sub_result = subsurface_scan(products)
                    session.all_results["subsurface"] = sub_result
                except Exception:
                    pass
                try:
                    min_result = mineral_analysis(products)
                    session.all_results["mineral"] = min_result
                except Exception:
                    pass
                try:
                    slope_result = slope_analysis(
                        session.bbox.center_lat,
                        session.bbox.center_lon,
                        radius_m=5000,
                        bbox=session.bbox,
                    )
                    session.all_results["slope"] = slope_result
                except Exception:
                    pass

            count = len(products)
            yield {"event": "action_complete", "data": {
                "tool": "_auto_recovery",
                "observation": (
                    f"Auto-recovery: searched {len(fallback_instruments)} "
                    f"instruments, found {count} products, ran basic analyses."
                ),
                "success": count > 0,
                "iteration": iteration + 1,
            }}

        # ── Synthesize results ──
        if "synthesize" not in session.all_results:
            region_ctx = None
            if session.region_name:
                region_ctx = get_region_context_by_name(session.region_name)
            synth_result = synthesize_results(
                session.region_name or "Unknown",
                session.all_results,
                region_context=region_ctx,
            )
            session.synthesis = synth_result.data
            session.all_results["synthesize"] = synth_result

        # ── Generate evidence figures ──
        try:
            from .agentic_router import generate_evidence_figures
            figures_data = generate_evidence_figures(session)
            if figures_data.get("figures"):
                session.figures = figures_data["figures"]
                yield {"event": "figures", "data": figures_data}
        except Exception as e:
            logger.warning(f"Evidence figure generation failed (react): {e}")

        # ── Generate narrative report ──
        session.status = "synthesizing"
        yield {"event": "thought_start", "data": {
            "iteration": iteration + 1, "phase": "narrative",
        }}

        async def _do_narrative():
            session.narrative = await _generate_narrative(
                session, on_chunk=_emit_chunk
            )
            await reasoning_queue.put(None)  # sentinel

        narr_task = asyncio.create_task(_do_narrative())
        while True:
            item = await reasoning_queue.get()
            if item is None:
                break
            yield item
        await narr_task

        yield {"event": "thought_end", "data": {"phase": "narrative"}}
        yield {"event": "narrative", "data": {"narrative": session.narrative}}

        # ── Done ──
        session.status = "done"
        yield {"event": "done", "data": session.to_dict()}

    except Exception as e:
        session.status = "error"
        session.error = str(e)
        logger.error(f"Agent session {session_id} error: {e}")
        yield {"event": "error", "data": {
            "error": str(e), "session_id": session_id,
        }}


# =============================================================================
# Rule-Based Fallback Pipeline
# =============================================================================

async def _run_agent_rules(
    objective: str,
    auto_download: bool = True,
    _session: Optional[AgentSession] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Rule-based agent pipeline (fallback when Ollama is unavailable).

    Plans all steps upfront, executes sequentially, synthesizes at the end.
    """
    if _session is not None:
        session = _session
        session_id = session.session_id
    else:
        session_id = str(uuid.uuid4())[:8]
        session = AgentSession(session_id=session_id, objective=objective)
        _sessions[session_id] = session

    yield {"event": "session_start", "data": {"session_id": session_id, "objective": objective}}

    # Queue to shuttle reasoning chunks from callback to the main generator
    reasoning_queue: asyncio.Queue[Optional[Dict[str, Any]]] = asyncio.Queue()

    async def _emit_chunk(text: str):
        await reasoning_queue.put({"event": "reasoning_chunk", "data": {"text": text}})

    try:
        # ── Phase 0: Early region resolution for science context ──
        # Try to resolve region from objective BEFORE plan generation
        # so Llama gets enriched science context in its prompt.
        early_region_name = None
        from .mars_regions import MARS_REGIONS
        obj_lower = objective.lower()
        for r in MARS_REGIONS.values():
            if r.display_name.lower() in obj_lower or r.region_id.lower() in obj_lower:
                early_region_name = r.display_name
                break

        # Build science context for plan generation
        science_context = get_context_for_agent(
            region_name=early_region_name,
            instruments=None,  # will be determined by plan
        )

        # ── Phase 1: Generate Plan ──────────────────────
        session.status = "planning"
        ollama_available = await _check_ollama()

        if ollama_available:
            logger.info("Using Llama for plan generation")
            yield {"event": "reasoning_start", "data": {"phase": "planning"}}

            # Run plan generation in background task so we can drain the queue
            plan_data: Dict[str, Any] = {}

            async def _do_plan():
                nonlocal plan_data
                plan_data = await _generate_plan_llm(
                    objective, science_context=science_context, on_chunk=_emit_chunk
                )
                await reasoning_queue.put(None)  # sentinel

            plan_task = asyncio.create_task(_do_plan())
            # Drain reasoning chunks while plan generation runs
            while True:
                item = await reasoning_queue.get()
                if item is None:
                    break
                yield item
            await plan_task  # ensure completion

            yield {"event": "reasoning_end", "data": {"phase": "planning"}}
        else:
            logger.info("Ollama unavailable, using rule-based planning")
            plan_data = {}

        # Fallback to rules if LLM plan is empty/invalid
        if not plan_data or "steps" not in plan_data:
            plan_data = _generate_plan_rules(objective)

        # Resolve region (use early resolution if available, else from plan)
        region_name = early_region_name or plan_data.get("region")
        if region_name:
            session.bbox = resolve_region(region_name)
            session.region_name = region_name

        # If no region resolved, try to extract coordinates from objective
        if not session.bbox:
            import re as _re
            coord_match = _re.search(r'(-?\d+\.?\d*)\s*[,\s]\s*(-?\d+\.?\d*)', objective)
            if coord_match:
                lat = float(coord_match.group(1))
                lon = float(coord_match.group(2))
                session.bbox = bbox_from_coords(lat, lon, radius_deg=3.0)
                session.region_name = f"({lat}, {lon})"

        # If still no bbox, use a default large area
        if not session.bbox:
            # Default to a broad search
            session.bbox = bbox_from_coords(0, 0, radius_deg=30)
            session.region_name = region_name or "Unresolved Region"

        # Build agent steps
        instruments = plan_data.get("instruments", [])

        # Enforce essential local-index instruments that LLM may omit
        REQUIRED_INSTRUMENTS = {
            "SHARAD_HIGHRES": "SHARAD",    # add if SHARAD present
            "HIRISE_DTM": None,            # always add (slope analysis)
        }
        instruments_upper = [i.upper() for i in instruments]
        for required, trigger in REQUIRED_INSTRUMENTS.items():
            if required not in instruments_upper:
                if trigger is None or trigger in instruments_upper:
                    instruments.append(required)

        # Inject search steps for any required instruments missing from plan steps
        plan_steps = plan_data.get("steps", [])
        planned_search_instruments = {
            s.get("instrument", "").upper()
            for s in plan_steps if s.get("type") == "search"
        }
        insert_pos = 0
        for i, s in enumerate(plan_steps):
            if s.get("type") == "search":
                insert_pos = i + 1
        for required in instruments:
            if required.upper() not in planned_search_instruments:
                plan_steps.insert(insert_pos, {
                    "type": "search",
                    "description": f"Search for {required} products in the target region",
                    "instrument": required,
                })
                insert_pos += 1

        # Enforce analysis steps: mineral_cnn after mineral, dielectric after subsurface
        planned_types = {s.get("type") for s in plan_steps}
        if "mineral" in planned_types and "mineral_cnn" not in planned_types:
            # Insert mineral_cnn right after the mineral step
            for i, s in enumerate(plan_steps):
                if s.get("type") == "mineral":
                    plan_steps.insert(i + 1, {
                        "type": "mineral_cnn",
                        "description": "Run 1D CNN-Attention mineral classification on CRISM TRR3 data",
                    })
                    break
        if "subsurface" in planned_types and "dielectric" not in planned_types:
            if "HIRISE_DTM" in instruments_upper:
                for i, s in enumerate(plan_steps):
                    if s.get("type") == "subsurface":
                        plan_steps.insert(i + 1, {
                            "type": "dielectric",
                            "description": "Estimate dielectric constant using SHARAD radar + HiRISE DTM elevation",
                        })
                        break

        for step_data in plan_steps:
            step = AgentStep(
                id=str(uuid.uuid4())[:6],
                type=step_data["type"],
                description=step_data["description"],
                instrument=step_data.get("instrument"),
            )
            session.steps.append(step)

        yield {
            "event": "plan",
            "data": {
                "session_id": session_id,
                "region": session.region_name,
                "instruments": instruments,
                "steps": [s.to_dict() for s in session.steps],
                "planned_by": "llama" if ollama_available and plan_data else "rules",
            },
        }

        # ── Phase 2: Execute Steps ─────────────────────
        session.status = "executing"
        all_products: List[Dict[str, Any]] = []

        step_count = len(session.steps)
        for step_index, step in enumerate(session.steps):
            step.status = StepStatus.RUNNING
            yield {"event": "step_start", "data": {**step.to_dict(), "step_index": step_index, "step_count": step_count}}

            try:
                if step.type == "search" and step.instrument:
                    result = await search_region(step.instrument, session.bbox)
                    step.result = result
                    if result.success:
                        all_products.extend(result.data.get("products", []))
                    session.all_results[f"search_{step.instrument}"] = result

                elif step.type == "check_data":
                    result = check_local_data(all_products)
                    step.result = result
                    session.all_results["check_local"] = result

                elif step.type == "download":
                    if auto_download:
                        check_result = session.all_results.get("check_local")
                        missing = check_result.data.get("missing", []) if check_result else []

                        # Smart download selection: cap at 30, sparse vs dense
                        to_download, dl_strategy = _select_downloads(missing, session.bbox)
                        if len(missing) > len(to_download):
                            step.description = (
                                f"Download {len(to_download)}/{len(missing)} "
                                f"missing products ({dl_strategy} strategy)"
                            )
                            yield {"event": "step_update", "data": {
                                **step.to_dict(),
                                "download_strategy": dl_strategy,
                                "total_missing": len(missing),
                            }}

                        # Stream per-file progress via queue
                        progress_q: asyncio.Queue = asyncio.Queue()

                        async def _on_dl_progress(completed, failed_n, skipped_n, total):
                            await progress_q.put({"completed": completed, "failed": failed_n, "skipped": skipped_n, "total": total})

                        dl_task = asyncio.create_task(download_data(to_download, on_progress=_on_dl_progress))

                        while not dl_task.done():
                            try:
                                prog = await asyncio.wait_for(progress_q.get(), timeout=2.0)
                                yield {"event": "download_progress", "data": prog}
                            except asyncio.TimeoutError:
                                pass

                        result = await dl_task
                        # Drain remaining progress events
                        while not progress_q.empty():
                            prog = progress_q.get_nowait()
                            yield {"event": "download_progress", "data": prog}

                        step.result = result
                        session.all_results["download"] = result
                    else:
                        step.result = TaskResult(
                            task_type="download",
                            success=True,
                            summary="Auto-download disabled",
                        )
                        step.status = StepStatus.SKIPPED
                        session.all_results["download"] = step.result

                elif step.type == "slope":
                    result = slope_analysis(
                        session.bbox.center_lat,
                        session.bbox.center_lon,
                        radius_m=5000,
                        bbox=session.bbox,
                    )
                    step.result = result
                    session.all_results["slope"] = result

                elif step.type == "subsurface":
                    result = subsurface_scan(all_products)
                    step.result = result
                    session.all_results["subsurface"] = result

                elif step.type == "mineral":
                    result = mineral_analysis(all_products)
                    step.result = result
                    session.all_results["mineral"] = result

                elif step.type == "mineral_cnn":
                    result = await mineral_cnn_classify(all_products)
                    step.result = result
                    session.all_results["mineral_cnn"] = result

                elif step.type == "dielectric":
                    sub_result = session.all_results.get("subsurface")
                    result = dielectric_analysis(sub_result, all_products, session.bbox)
                    step.result = result
                    session.all_results["dielectric"] = result

                elif step.type == "recommend":
                    result = recommend_site(session.all_results)
                    step.result = result
                    session.all_results["recommend"] = result

                elif step.type == "climate":
                    _, _ = await _tool_climate(session, {})
                    step.result = session.all_results.get("climate")

                elif step.type == "thermal_inertia":
                    _, _ = await _tool_thermal_inertia(session, {})
                    step.result = session.all_results.get("thermal_inertia")

                elif step.type == "synthesize":
                    region_ctx = None
                    if session.region_name:
                        region_ctx = get_region_context_by_name(session.region_name)
                    result = synthesize_results(
                        session.region_name or "Unknown",
                        session.all_results,
                        region_context=region_ctx,
                    )
                    step.result = result
                    session.synthesis = result.data
                    session.all_results["synthesize"] = result

                if step.status != StepStatus.SKIPPED:
                    step.status = StepStatus.COMPLETED if (step.result and step.result.success) else StepStatus.FAILED

                yield {"event": "step_complete", "data": step.to_dict()}

            except Exception as e:
                step.status = StepStatus.FAILED
                step.error = str(e)
                step.result = TaskResult(
                    task_type=step.type,
                    success=False,
                    error=str(e),
                    summary=f"Failed: {e}",
                )
                logger.error(f"Step {step.type} failed: {e}")
                yield {"event": "step_failed", "data": step.to_dict()}

        # ── Phase 2.5: Generate evidence figures ──────────
        try:
            from .agentic_router import generate_evidence_figures
            figures_data = generate_evidence_figures(session)
            if figures_data.get("figures"):
                session.figures = figures_data["figures"]
                yield {"event": "figures", "data": figures_data}
        except Exception as e:
            logger.warning(f"Evidence figure generation failed (rules): {e}")

        # ── Phase 3: Generate Narrative ────────────────
        session.status = "synthesizing"

        if ollama_available:
            yield {"event": "reasoning_start", "data": {"phase": "narrative"}}

            async def _do_narrative():
                session.narrative = await _generate_narrative(session, on_chunk=_emit_chunk)
                await reasoning_queue.put(None)  # sentinel

            narr_task = asyncio.create_task(_do_narrative())
            while True:
                item = await reasoning_queue.get()
                if item is None:
                    break
                yield item
            await narr_task

            yield {"event": "reasoning_end", "data": {"phase": "narrative"}}
        else:
            session.narrative = _generate_narrative_fallback(session)

        yield {"event": "narrative", "data": {"narrative": session.narrative}}

        # ── Done ───────────────────────────────────────
        session.status = "done"
        yield {"event": "done", "data": session.to_dict()}

    except Exception as e:
        session.status = "error"
        session.error = str(e)
        logger.error(f"Agent session {session_id} error: {e}")
        yield {"event": "error", "data": {"error": str(e), "session_id": session_id}}


# =============================================================================
# Background Execution + Resumable Streaming
# =============================================================================

def start_agent_background(objective: str, auto_download: bool = True) -> AgentSession:
    """
    Start the agent as a background asyncio.Task.

    Events are stored in session.events so any number of clients can
    connect / reconnect and replay + follow the live stream.
    """
    _evict_old_sessions()
    session_id = str(uuid.uuid4())[:8]
    session = AgentSession(session_id=session_id, objective=objective)
    _sessions[session_id] = session

    async def _run():
        try:
            async for event in run_agent(objective, auto_download, _session=session):
                await session.emit(event)
        except Exception as e:
            logger.error(f"Background agent {session_id} crashed: {e}")
            session.status = "error"
            session.error = str(e)
            await session.emit({"event": "error", "data": {"error": str(e), "session_id": session_id}})
        finally:
            # Persist completed session to disk
            _save_session(session)
            # Ensure a terminal stream_end is always present
            await session.emit({"event": "stream_end"})
            session._task = None  # Allow GC of completed task

    session._task = asyncio.create_task(_run())
    return session


async def stream_session_events(
    session: AgentSession,
    from_index: int = 0,
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Yield events from a session's buffer starting at *from_index*.

    First replays all buffered events, then waits for new ones until the
    session reaches a terminal state (done / error) and the buffer is drained.
    Uses asyncio.Condition for multi-consumer safety.
    """
    cursor = from_index

    while True:
        # Yield any buffered events we haven't sent yet
        while cursor < len(session.events):
            event = session.events[cursor]
            cursor += 1
            yield event
            # Stop once we've delivered the stream_end sentinel
            if event.get("event") == "stream_end":
                return

        # If session finished and we've drained everything, stop
        if session.is_terminal and cursor >= len(session.events):
            return

        # Wait for new events (multi-consumer safe via Condition)
        try:
            async with session._condition:
                # Re-check inside the lock to avoid missed signals
                if cursor < len(session.events):
                    continue
                await asyncio.wait_for(
                    session._condition.wait(), timeout=2.0
                )
        except asyncio.TimeoutError:
            # Keep looping — check for new events or terminal state
            pass
