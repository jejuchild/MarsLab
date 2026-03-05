"""
Landing Site Report Orchestrator.

Multi-region comparison report generator with ground-rules filtering.
Runs the existing agent analysis pipeline for each candidate region,
compares results, and generates a structured comparison report.

Reuses atomic tasks from agent_tasks.py and SSE patterns from agent_orchestrator.py.
"""

import asyncio
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Awaitable, Callable, Dict, List, Optional

import aiohttp

from .agent_orchestrator import (
    AgentStep,
    StepStatus,
    _call_groq,
    _call_groq_streaming,
    _check_groq,
)
from .agent_tasks import (
    RegionBBox,
    TaskResult,
    check_local_data,
    download_data,
    mineral_analysis,
    resolve_region,
    search_region,
    slope_analysis,
    subsurface_scan,
    synthesize_results,
)
from .mars_regions import MARS_REGIONS, MarsRegion, list_all_regions
from .science_context import get_context_for_agent, get_region_context_by_name
from .scoring_methodology import (
    compute_composite_score,
    classify_evidence_strength,
    classify_recommendation,
    sensitivity_analysis,
    full_assessment,
    DEFAULT_WEIGHTS,
    WEIGHT_JUSTIFICATION,
)
from .mars_climate import climate_analysis_for_region

logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class GroundRules:
    """User-defined constraints for filtering candidate regions."""

    lat_min: Optional[float] = None  # e.g., -50
    lat_max: Optional[float] = None  # e.g., 50
    lon_min: Optional[float] = None
    lon_max: Optional[float] = None
    include_regions: List[str] = field(default_factory=list)  # only these IDs
    exclude_regions: List[str] = field(default_factory=list)  # skip these IDs
    include_tags: List[str] = field(default_factory=list)  # only with these tags
    exclude_tags: List[str] = field(default_factory=list)  # skip with these tags
    min_slope_safety: Optional[str] = None  # "FAVORABLE" or "MARGINAL"
    custom_notes: str = ""  # free-text for LLM narrative

    def summary(self) -> str:
        """Human-readable summary of active ground rules."""
        parts = []
        if self.lat_min is not None or self.lat_max is not None:
            lo = self.lat_min if self.lat_min is not None else -90
            hi = self.lat_max if self.lat_max is not None else 90
            parts.append(f"Latitude: {lo}° to {hi}°")
        if self.lon_min is not None or self.lon_max is not None:
            lo = self.lon_min if self.lon_min is not None else -180
            hi = self.lon_max if self.lon_max is not None else 180
            parts.append(f"Longitude: {lo}° to {hi}°")
        if self.include_regions:
            parts.append(f"Include only: {', '.join(self.include_regions)}")
        if self.exclude_regions:
            parts.append(f"Exclude: {', '.join(self.exclude_regions)}")
        if self.include_tags:
            parts.append(f"Tags required: {', '.join(self.include_tags)}")
        if self.exclude_tags:
            parts.append(f"Tags excluded: {', '.join(self.exclude_tags)}")
        if self.min_slope_safety:
            parts.append(f"Min slope safety: {self.min_slope_safety}")
        if self.custom_notes:
            parts.append(f"Notes: {self.custom_notes}")
        return "; ".join(parts) if parts else "No constraints"


@dataclass
class ReportConfig:
    """Configuration for a comparison report."""

    ground_rules: GroundRules = field(default_factory=GroundRules)
    analyses: List[str] = field(
        default_factory=lambda: ["slope", "subsurface", "mineral"]
    )
    auto_download: bool = True
    instruments: List[str] = field(
        default_factory=lambda: [
            "CRISM",
            "SHARAD",
            "SHARAD_HIGHRES",
            "CTX",
            "HIRISE_DTM",
        ]
    )
    max_regions: int = 5


@dataclass
class RegionAnalysis:
    """Results for a single region within a comparison report."""

    region_id: str
    region_name: str
    bbox: RegionBBox
    status: str = "pending"  # pending | running | completed | failed
    steps: List[AgentStep] = field(default_factory=list)
    all_results: Dict[str, TaskResult] = field(default_factory=dict)
    synthesis: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    passes_ground_rules: bool = True
    ground_rule_violations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "region_id": self.region_id,
            "region_name": self.region_name,
            "status": self.status,
            "steps": [s.to_dict() for s in self.steps],
            "synthesis": self.synthesis,
            "error": self.error,
            "passes_ground_rules": self.passes_ground_rules,
            "ground_rule_violations": self.ground_rule_violations,
        }


@dataclass
class ReportSession:
    """Tracks a multi-region comparison report generation."""

    session_id: str
    status: str = "filtering"  # filtering|analyzing|comparing|generating|done|error
    config: ReportConfig = field(default_factory=ReportConfig)
    candidate_regions: List[Dict[str, Any]] = field(default_factory=list)
    regions: List[RegionAnalysis] = field(default_factory=list)
    comparison: Optional[Dict[str, Any]] = None
    rankings: Optional[List[Dict[str, Any]]] = None
    executive_summary: str = ""
    report_markdown: str = ""
    error: Optional[str] = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # SSE event buffer (same pattern as AgentSession)
    events: List[Dict[str, Any]] = field(default_factory=list)
    _event_count: int = field(default=0, repr=False)
    _condition: asyncio.Condition = field(
        default_factory=asyncio.Condition, repr=False
    )
    _task: Optional[asyncio.Task] = field(default=None, repr=False)

    async def emit(self, event: Dict[str, Any]):
        """Append an event and wake all listeners."""
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
            "status": self.status,
            "candidate_regions": self.candidate_regions,
            "regions": [r.to_dict() for r in self.regions],
            "comparison": self.comparison,
            "rankings": self.rankings,
            "executive_summary": self.executive_summary,
            "report_markdown": self.report_markdown,
            "error": self.error,
            "created_at": self.created_at,
        }


# Session store
_report_sessions: Dict[str, ReportSession] = {}
MAX_REPORT_SESSIONS = 20


def _evict_old_report_sessions():
    if len(_report_sessions) <= MAX_REPORT_SESSIONS:
        return
    terminal = sorted(
        (s for s in _report_sessions.values() if s.is_terminal),
        key=lambda s: s.created_at,
    )
    to_remove = len(_report_sessions) - MAX_REPORT_SESSIONS
    for s in terminal[:to_remove]:
        _report_sessions.pop(s.session_id, None)


def get_report_session(session_id: str) -> Optional[ReportSession]:
    return _report_sessions.get(session_id)


def list_report_sessions() -> List[ReportSession]:
    sessions = list(_report_sessions.values())
    sessions.sort(key=lambda s: s.created_at, reverse=True)
    return sessions


async def cancel_report_session(session_id: str) -> bool:
    """Cancel a running report session. Returns True if cancelled."""
    session = _report_sessions.get(session_id)
    if not session or session.is_terminal:
        return False
    if session._task and not session._task.done():
        session._task.cancel()
    # Mark running regions as failed
    for ra in session.regions:
        if ra.status == "running":
            ra.status = "failed"
            ra.error = "Cancelled by user"
            for step in ra.steps:
                if step.status == StepStatus.RUNNING:
                    step.status = StepStatus.FAILED
                    step.error = "Cancelled by user"
    session.status = "error"
    session.error = "Cancelled by user"
    await session.emit({"event": "error", "data": {"error": "Cancelled by user"}})
    await session.emit({"event": "stream_end"})
    _save_report_session(session)
    return True


# Persistent storage
_REPORT_SESSIONS_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
_REPORT_SESSIONS_FILE = os.path.join(_REPORT_SESSIONS_DIR, "report_sessions.json")


def _load_report_sessions():
    """Load completed report sessions from disk."""
    if not os.path.exists(_REPORT_SESSIONS_FILE):
        return
    try:
        with open(_REPORT_SESSIONS_FILE, "r") as f:
            records = json.load(f)
        for rec in records:
            sid = rec.get("session_id")
            if not sid or sid in _report_sessions:
                continue
            # Reconstruct RegionAnalysis objects
            regions = []
            for ra in rec.get("regions", []):
                steps = []
                for s in ra.get("steps", []):
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
                bbox_data = ra.get("bbox")
                bbox = RegionBBox(
                    bbox_data["min_lat"], bbox_data["max_lat"],
                    bbox_data["min_lon"], bbox_data["max_lon"],
                ) if bbox_data else RegionBBox(0, 0, 0, 0)
                regions.append(RegionAnalysis(
                    region_id=ra["region_id"],
                    region_name=ra["region_name"],
                    bbox=bbox,
                    status=ra.get("status", "completed"),
                    steps=steps,
                    synthesis=ra.get("synthesis"),
                    error=ra.get("error"),
                    passes_ground_rules=ra.get("passes_ground_rules", True),
                    ground_rule_violations=ra.get("ground_rule_violations", []),
                ))
            session = ReportSession(
                session_id=sid,
                status=rec.get("status", "done"),
                candidate_regions=rec.get("candidate_regions", []),
                regions=regions,
                comparison=rec.get("comparison"),
                rankings=rec.get("rankings"),
                executive_summary=rec.get("executive_summary", ""),
                report_markdown=rec.get("report_markdown", ""),
                error=rec.get("error"),
                created_at=rec.get("created_at", ""),
            )
            _report_sessions[sid] = session
        logger.info(f"Loaded {len(records)} report sessions from disk")
    except Exception as e:
        logger.error(f"Failed to load report sessions: {e}")


def _save_report_session(session: ReportSession):
    """Save a terminal report session to disk."""
    if not session.is_terminal:
        return
    os.makedirs(_REPORT_SESSIONS_DIR, exist_ok=True)

    # Build serializable record with full step results
    region_records = []
    for ra in session.regions:
        step_records = []
        for s in ra.steps:
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
        region_records.append({
            "region_id": ra.region_id,
            "region_name": ra.region_name,
            "bbox": {
                "min_lat": ra.bbox.min_lat,
                "max_lat": ra.bbox.max_lat,
                "min_lon": ra.bbox.min_lon,
                "max_lon": ra.bbox.max_lon,
            },
            "status": ra.status,
            "steps": step_records,
            "synthesis": ra.synthesis,
            "error": ra.error,
            "passes_ground_rules": ra.passes_ground_rules,
            "ground_rule_violations": ra.ground_rule_violations,
        })

    record = {
        "session_id": session.session_id,
        "status": session.status,
        "candidate_regions": session.candidate_regions,
        "regions": region_records,
        "comparison": session.comparison,
        "rankings": session.rankings,
        "executive_summary": session.executive_summary,
        "report_markdown": session.report_markdown,
        "error": session.error,
        "created_at": session.created_at,
    }

    # Load existing, replace or append, write back
    existing: list = []
    if os.path.exists(_REPORT_SESSIONS_FILE):
        try:
            with open(_REPORT_SESSIONS_FILE, "r") as f:
                existing = json.load(f)
        except Exception:
            existing = []

    found = False
    for i, rec in enumerate(existing):
        if rec.get("session_id") == session.session_id:
            existing[i] = record
            found = True
            break
    if not found:
        existing.append(record)

    # Keep only the most recent sessions
    if len(existing) > MAX_REPORT_SESSIONS:
        existing = existing[-MAX_REPORT_SESSIONS:]

    with open(_REPORT_SESSIONS_FILE, "w") as f:
        json.dump(existing, f, indent=2, default=str)

    logger.info(f"Saved report session {session.session_id} to disk")


# Load persisted sessions on module import
_load_report_sessions()


# =============================================================================
# Ground Rules Filtering
# =============================================================================


def filter_regions(ground_rules: GroundRules) -> List[MarsRegion]:
    """Apply ground rules to filter candidate regions from MARS_REGIONS."""
    all_regions = list_all_regions()
    candidates = []

    for r in all_regions:
        # Include-only filter — when explicit regions are selected,
        # bypass lat/lon/tag filters (user explicitly chose these).
        if ground_rules.include_regions:
            if r.region_id in ground_rules.include_regions:
                # Still respect explicit exclusions
                if r.region_id not in ground_rules.exclude_regions:
                    candidates.append(r)
            continue

        # Exclude filter
        if r.region_id in ground_rules.exclude_regions:
            continue

        # Latitude bounds (check center lat)
        if ground_rules.lat_min is not None and r.center_lat < ground_rules.lat_min:
            continue
        if ground_rules.lat_max is not None and r.center_lat > ground_rules.lat_max:
            continue

        # Longitude bounds (check center lon)
        if ground_rules.lon_min is not None and r.center_lon < ground_rules.lon_min:
            continue
        if ground_rules.lon_max is not None and r.center_lon > ground_rules.lon_max:
            continue

        # Tag inclusion (region must have at least one matching tag)
        if ground_rules.include_tags:
            region_tags = [t.lower() for t in r.tags]
            if not any(t.lower() in region_tags for t in ground_rules.include_tags):
                continue

        # Tag exclusion (skip if region has any excluded tag)
        if ground_rules.exclude_tags:
            region_tags = [t.lower() for t in r.tags]
            if any(t.lower() in region_tags for t in ground_rules.exclude_tags):
                continue

        candidates.append(r)

    return candidates


# =============================================================================
# Per-Region Analysis Pipeline
# =============================================================================


async def _analyze_region(
    region: MarsRegion,
    config: ReportConfig,
    region_index: int,
    emit: Callable[[Dict[str, Any]], Awaitable[None]],
) -> RegionAnalysis:
    """Run the full analysis pipeline for a single region."""
    bbox = RegionBBox(
        min_lat=region.lat_min,
        max_lat=region.lat_max,
        min_lon=region.lon_min,
        max_lon=region.lon_max,
    )

    analysis = RegionAnalysis(
        region_id=region.region_id,
        region_name=region.display_name,
        bbox=bbox,
        status="running",
    )

    # Build steps based on config
    step_defs = []

    # Search each instrument
    for inst in config.instruments:
        step_defs.append(
            {"type": "search", "description": f"Search {inst}", "instrument": inst}
        )

    step_defs.append(
        {"type": "check_data", "description": "Check local data availability"}
    )

    if config.auto_download:
        step_defs.append(
            {"type": "download", "description": "Download missing products"}
        )

    if "slope" in config.analyses:
        step_defs.append({"type": "slope", "description": "Slope analysis"})
    if "subsurface" in config.analyses:
        step_defs.append({"type": "subsurface", "description": "Subsurface scan"})
    if "mineral" in config.analyses:
        step_defs.append({"type": "mineral", "description": "Mineral analysis"})

    # Climate analysis step
    step_defs.append({"type": "climate", "description": "Climate analysis"})
    # Landform classification query
    step_defs.append({"type": "landform_classification", "description": "HiRISE landform classification"})

    step_defs.append({"type": "synthesize", "description": "Synthesize results"})

    # Create AgentStep objects
    for sd in step_defs:
        analysis.steps.append(
            AgentStep(
                id=str(uuid.uuid4())[:6],
                type=sd["type"],
                description=sd["description"],
                instrument=sd.get("instrument"),
            )
        )

    await emit(
        {
            "event": "region_start",
            "data": {
                "region_index": region_index,
                "region_id": region.region_id,
                "region_name": region.display_name,
                "step_count": len(analysis.steps),
            },
        }
    )

    all_products: List[Dict[str, Any]] = []

    for step in analysis.steps:
        step.status = StepStatus.RUNNING
        await emit(
            {
                "event": "region_step_start",
                "data": {"region_index": region_index, "step": step.to_dict()},
            }
        )

        try:
            if step.type == "search" and step.instrument:
                result = await search_region(step.instrument, bbox)
                step.result = result
                if result.success:
                    all_products.extend(result.data.get("products", []))
                analysis.all_results[f"search_{step.instrument}"] = result

            elif step.type == "check_data":
                result = check_local_data(all_products)
                step.result = result
                analysis.all_results["check_local"] = result

            elif step.type == "download":
                if config.auto_download:
                    check_result = analysis.all_results.get("check_local")
                    missing = (
                        check_result.data.get("missing", []) if check_result else []
                    )
                    to_download = missing[:10]

                    progress_q: asyncio.Queue = asyncio.Queue()

                    async def _on_dl_progress(completed, failed_n, skipped_n, total):
                        await progress_q.put(
                            {
                                "completed": completed,
                                "failed": failed_n,
                                "skipped": skipped_n,
                                "total": total,
                            }
                        )

                    dl_task = asyncio.create_task(
                        download_data(to_download, on_progress=_on_dl_progress)
                    )

                    while not dl_task.done():
                        try:
                            prog = await asyncio.wait_for(
                                progress_q.get(), timeout=2.0
                            )
                            await emit(
                                {
                                    "event": "download_progress",
                                    "data": {**prog, "region_index": region_index},
                                }
                            )
                        except asyncio.TimeoutError:
                            pass

                    result = await dl_task
                    while not progress_q.empty():
                        prog = progress_q.get_nowait()
                        await emit(
                            {
                                "event": "download_progress",
                                "data": {**prog, "region_index": region_index},
                            }
                        )

                    step.result = result
                    analysis.all_results["download"] = result
                else:
                    step.result = TaskResult(
                        task_type="download",
                        success=True,
                        summary="Auto-download disabled",
                    )
                    step.status = StepStatus.SKIPPED
                    analysis.all_results["download"] = step.result

            elif step.type == "slope":
                result = slope_analysis(
                    bbox.center_lat, bbox.center_lon, radius_m=5000, bbox=bbox
                )
                step.result = result
                analysis.all_results["slope"] = result

            elif step.type == "subsurface":
                result = subsurface_scan(all_products)
                step.result = result
                analysis.all_results["subsurface"] = result

            elif step.type == "mineral":
                result = mineral_analysis(all_products)
                step.result = result
                analysis.all_results["mineral"] = result

            elif step.type == "climate":
                bbox_val = analysis.bbox
                clim_result_data = climate_analysis_for_region(
                    bbox_val.min_lat, bbox_val.max_lat,
                    bbox_val.min_lon, bbox_val.max_lon,
                )
                result = TaskResult(
                    task_type="climate",
                    success=clim_result_data.get("success", False),
                    data=clim_result_data,
                    summary=clim_result_data.get("climate_summary", "Climate analysis complete"),
                )
                step.result = result
                analysis.all_results["climate"] = result

            elif step.type == "landform_classification":
                from .agent_tasks import query_landform_classifications
                result = query_landform_classifications(bbox)
                step.result = result
                analysis.all_results["landform_classification"] = result

            elif step.type == "synthesize":
                region_ctx = get_region_context_by_name(region.display_name)
                result = synthesize_results(
                    region.display_name,
                    analysis.all_results,
                    region_context=region_ctx,
                )
                step.result = result
                analysis.synthesis = result.data
                analysis.all_results["synthesize"] = result

            if step.status != StepStatus.SKIPPED:
                step.status = (
                    StepStatus.COMPLETED
                    if (step.result and step.result.success)
                    else StepStatus.FAILED
                )

            await emit(
                {
                    "event": "region_step_complete",
                    "data": {"region_index": region_index, "step": step.to_dict()},
                }
            )

        except Exception as e:
            step.status = StepStatus.FAILED
            step.error = str(e)
            step.result = TaskResult(
                task_type=step.type,
                success=False,
                error=str(e),
                summary=f"Failed: {e}",
            )
            logger.error(
                f"Report step {step.type} failed for {region.display_name}: {e}"
            )
            await emit(
                {
                    "event": "region_step_failed",
                    "data": {"region_index": region_index, "step": step.to_dict()},
                }
            )

    analysis.status = "completed" if analysis.synthesis else "failed"
    return analysis


# =============================================================================
# Post-Analysis Ground Rule Checks
# =============================================================================


def _check_post_analysis_rules(
    analysis: RegionAnalysis, ground_rules: GroundRules
) -> None:
    """Check hard constraints after analysis. Modifies analysis in place."""
    if not ground_rules.min_slope_safety:
        return

    synthesis = analysis.synthesis or {}
    eng = synthesis.get("engineering_feasibility", {})
    safety = eng.get("safety", "UNKNOWN")

    # Rank safety: FAVORABLE > MARGINAL > UNFAVORABLE > UNKNOWN
    safety_rank = {"FAVORABLE": 3, "MARGINAL": 2, "UNFAVORABLE": 1, "UNKNOWN": 0}
    required = safety_rank.get(ground_rules.min_slope_safety, 0)
    actual = safety_rank.get(safety, 0)

    if actual < required:
        analysis.passes_ground_rules = False
        analysis.ground_rule_violations.append(
            f"Slope safety {safety} below required {ground_rules.min_slope_safety}"
        )


# =============================================================================
# Cross-Region Comparison
# =============================================================================


def compare_regions(regions: List[RegionAnalysis]) -> Dict[str, Any]:
    """Compare analysis results using the unified scoring methodology."""
    entries = []

    for r in regions:
        if not r.synthesis:
            continue

        s = r.synthesis

        # Use the scoring_model if already computed by synthesize_results
        scoring_model = s.get("scoring_model", {})
        sub_scores = scoring_model.get("sub_scores", {})

        # If scoring_model not present (backward compat), compute it here
        if not scoring_model:
            scoring_model = compute_composite_score(
                subsurface_data=s.get("subsurface_coverage", {}),
                dielectric_data=s.get("dielectric_analysis", {}),
                cross_instrument_data=s.get("cross_instrument", {}),
                mineral_data=s.get("mineral_signatures", {}),
                cnn_data=s.get("cnn_mineral_classification", {}),
                engineering_data=s.get("engineering_feasibility", {}),
                climate_data=s.get("climate", {}),
            )
            sub_scores = scoring_model.get("sub_scores", {})

        final_score = scoring_model.get("final_score", 0)
        final_score_100 = scoring_model.get("final_score_100", 0)

        # Evidence strength
        evidence = s.get("evidence_strength", {})
        if not evidence:
            evidence = classify_evidence_strength(sub_scores, s.get("cross_instrument", {}))

        # Recommendation
        rec_v2 = s.get("recommendation_v2", {})
        if not rec_v2:
            rec_v2 = classify_recommendation(
                final_score,
                evidence,
                s.get("cross_instrument", {}).get("evidence_consistency", "insufficient_data"),
            )

        # Extract sub-score values for ranking table
        sub_subsurface = sub_scores.get("subsurface_potential", {}).get("score", 0)
        sub_surface_ice = sub_scores.get("surface_ice", {}).get("score", 0)
        sub_terrain = sub_scores.get("terrain_safety", {}).get("score", 0)
        sub_climate = sub_scores.get("climate", {}).get("score", 0)

        # Highlights
        eng = s.get("engineering_feasibility", {})
        sub = s.get("subsurface_coverage", {})
        mineral = s.get("mineral_signatures", {})
        cross = s.get("cross_instrument", {})

        highlights = []
        if eng.get("safety") in ("FAVORABLE", "MARGINAL"):
            highlights.append(f"{eng['safety']} terrain ({eng.get('mean_slope', '?')}deg mean)")
        n_detect = sub.get("subsurface_detections", 0)
        if n_detect > 0:
            highlights.append(f"{n_detect} SHARAD reflector(s)")
        ice_count = mineral.get("high_ice_count", 0)
        if ice_count > 0:
            highlights.append(f"{ice_count} high-ice CRISM")
        consistency = cross.get("evidence_consistency", "")
        if consistency == "consistent":
            highlights.append("Cross-instrument corroboration")

        entries.append({
            "region_id": r.region_id,
            "region_name": r.region_name,
            "overall_score": final_score_100,
            "final_score": final_score,
            "sub_scores": {
                "subsurface": round(sub_subsurface, 3),
                "surface_ice": round(sub_surface_ice, 3),
                "terrain": round(sub_terrain, 3),
                "climate": round(sub_climate, 3),
            },
            "evidence_strength": evidence,
            "recommendation_v2": rec_v2,
            "recommendation": rec_v2.get("classification", "N/A"),
            "highlights": highlights,
            "passes_ground_rules": r.passes_ground_rules,
            "ground_rule_violations": r.ground_rule_violations,
            "cross_instrument_consistency": cross.get("evidence_consistency", "N/A"),
            "consistency_score": cross.get("consistency_score", 0),
            "epsilon_r_source": s.get("epsilon_r_source", "assumed"),
            "is_fallback": s.get("is_fallback", True),
            # Backward compat
            "engineering_safety": eng.get("safety", "UNKNOWN"),
            "subsurface_coverage": sub.get("coverage", "NONE"),
            "subsurface_detections": n_detect,
            "ice_count": ice_count,
            "total_products": s.get("total_products_found", 0),
        })

    # Sort by final_score (highest first)
    entries.sort(key=lambda e: e["final_score"], reverse=True)
    for i, e in enumerate(entries):
        e["rank"] = i + 1

    # Category winners
    category_winners = {}
    if entries:
        category_winners["best_subsurface"] = _pick_winner(entries, lambda e: e["sub_scores"]["subsurface"], "subsurface")
        category_winners["best_surface_ice"] = _pick_winner(entries, lambda e: e["sub_scores"]["surface_ice"], "surface_ice")
        category_winners["best_terrain"] = _pick_winner(entries, lambda e: e["sub_scores"]["terrain"], "terrain")
        category_winners["best_climate"] = _pick_winner(entries, lambda e: e["sub_scores"]["climate"], "climate")

    recommended = entries[0] if entries else None

    return {
        "rankings": entries,
        "category_winners": category_winners,
        "recommended": {
            "region_id": recommended["region_id"],
            "region_name": recommended["region_name"],
            "score": recommended["overall_score"],
            "final_score": recommended["final_score"],
            "recommendation": recommended["recommendation"],
            "evidence_strength": recommended["evidence_strength"],
            "highlights": recommended["highlights"],
        } if recommended else None,
    }


def _pick_winner(entries, key_fn, category_name):
    """Pick the winner for a category."""
    best = max(entries, key=key_fn)
    return {
        "region_id": best["region_id"],
        "region_name": best["region_name"],
        "value": f"{key_fn(best):.3f}",
    }


# =============================================================================
# Report Markdown Generation
# =============================================================================


def _md_table(headers: list, rows: list) -> str:
    """Generate a markdown table from headers and rows."""
    lines = []
    lines.append("| " + " | ".join(str(h) for h in headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def build_report_markdown(session: ReportSession) -> str:
    """Build a scientifically rigorous v2 mission assessment report in Markdown."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines: List[str] = []

    region_names = [r.region_name for r in session.regions]
    region_label = ", ".join(region_names) if region_names else "Unknown"

    # ── Title ──
    lines.append(f"# Mission Assessment Report v2 -- {region_label}")
    lines.append("")
    lines.append(f"**Generated:** {now}")
    lines.append(f"**Session:** `{session.session_id}`")
    lines.append(f"**Pipeline Version:** v2.0 (Scientifically Rigorous)")
    lines.append("")

    # Executive summary (if available, inject early)
    if session.executive_summary:
        lines.append("> " + session.executive_summary.replace("\n", "\n> "))
        lines.append("")

    lines.append("---")
    lines.append("")

    # ══════════════════════════════════════════════════════════════
    # Section 1: Assumptions
    # ══════════════════════════════════════════════════════════════
    lines.append("## 1. Assumptions")
    lines.append("")

    # 1.1 Dielectric Constants
    lines.append("### 1.1 Dielectric Constants")
    lines.append("")
    lines.append(
        "Depth estimates require a dielectric constant (εr). Physics-based inversion "
        "(terraced crater morphology or hyperbola curvature fitting) is preferred. "
        "Assumed εr is used ONLY as a fallback and is explicitly labeled."
    )
    lines.append("")
    lines.append(_md_table(
        ["Scenario", "er", "Material", "Depth Impact"],
        [
            ["Porous ice/regolith", "2.8", "Porous ice/regolith mix", "Shallowest depth estimate"],
            ["Pure water ice", "3.15", "Pure water ice", "Nominal (fallback) depth estimate"],
            ["Basaltic contrast", "4.0", "Basaltic contrast interface", "Deepest depth estimate"],
        ],
    ))
    lines.append("")

    # Per-region εr methods summary
    eps_method_rows = []
    for region_analysis in session.regions:
        syn = region_analysis.synthesis or {}
        eps_src = syn.get("epsilon_r_source", "assumed")
        is_fb = syn.get("is_fallback", True)
        method_label = eps_src.replace("_", " ").title()
        status = "Fallback" if is_fb else "Physics-Based"
        eps_method_rows.append([
            region_analysis.region_name,
            method_label,
            status,
        ])
    if eps_method_rows:
        lines.append("**Per-Region εr Method:**")
        lines.append("")
        lines.append(_md_table(
            ["Region", "εr Method", "Status"],
            eps_method_rows,
        ))
        lines.append("")

    # 1.2 Detection Thresholds
    lines.append("### 1.2 Detection Thresholds")
    lines.append("")
    lines.append("- SHARAD subsurface reflector: SNR >= 4.0")
    lines.append("- CRISM ice score threshold: 0.3 (justification: ~3 sigma above background)")
    lines.append("- CNN mineral classification: softmax >= 95%")
    lines.append("- Slope safety threshold: <5 deg for favorable terrain")
    lines.append("")

    # 1.3 Climate Model
    lines.append("### 1.3 Climate Model")
    lines.append("")
    lines.append("- Parametric approximation (NOT Mars Climate Database)")
    lines.append("- 12 Ls-bin seasonal resolution (30 deg increments)")
    lines.append("- See Climate Assessment section for formula details")
    lines.append("")

    # 1.4 Scoring Model
    w = DEFAULT_WEIGHTS
    lines.append("### 1.4 Scoring Model")
    lines.append("")
    lines.append(
        f"Score = {w['w1']} x Subsurface + {w['w2']} x Surface Ice "
        f"+ {w['w3']} x Terrain + {w['w4']} x Climate"
    )
    lines.append("")
    for wline in WEIGHT_JUSTIFICATION.split("\n"):
        lines.append(f"- {wline}" if wline.strip() else "")
    lines.append("")

    lines.append("---")
    lines.append("")

    # ══════════════════════════════════════════════════════════════
    # Section 2: Rankings
    # ══════════════════════════════════════════════════════════════
    comparison = session.comparison or {}
    rankings = comparison.get("rankings", [])

    lines.append("## 2. Rankings")
    lines.append("")

    if rankings:
        rank_headers = ["Rank", "Region", "Score", "Subsurface", "Surface Ice", "Terrain", "Climate", "εr Method", "Evidence", "Recommendation"]
        rank_rows = []
        for r in rankings:
            violations_mark = " (!)" if not r.get("passes_ground_rules", True) else ""
            ev = r.get("evidence_strength", {})
            overall_ev = ev.get("overall", "N/A")
            ss = r.get("sub_scores", {})
            # Determine εr method for this region
            eps_method = r.get("epsilon_r_source", "assumed")
            if eps_method == "assumed":
                eps_label = "Assumed"
            else:
                eps_label = eps_method.replace("_", " ").title()
            rank_rows.append([
                r["rank"],
                f"{r['region_name']}{violations_mark}",
                f"**{r['overall_score']}/100**",
                f"{ss.get('subsurface', 0):.3f}",
                f"{ss.get('surface_ice', 0):.3f}",
                f"{ss.get('terrain', 0):.3f}",
                f"{ss.get('climate', 0):.3f}",
                eps_label,
                overall_ev,
                r.get("recommendation", "N/A"),
            ])
        lines.append(_md_table(rank_headers, rank_rows))
        lines.append("")
    else:
        lines.append("No regions were successfully ranked.")
        lines.append("")

    lines.append("---")
    lines.append("")

    # ══════════════════════════════════════════════════════════════
    # Section 3: Per-Region Analysis
    # ══════════════════════════════════════════════════════════════
    lines.append("## 3. Per-Region Analysis")
    lines.append("")

    for i, region_analysis in enumerate(session.regions):
        ri = i + 1
        syn = region_analysis.synthesis or {}
        scoring_model = syn.get("scoring_model", {})
        sub_scores = scoring_model.get("sub_scores", {})
        evidence = syn.get("evidence_strength", {})
        rec_v2 = syn.get("recommendation_v2", {})

        lines.append(f"### 3.{ri}: {region_analysis.region_name}")
        lines.append("")

        if not region_analysis.passes_ground_rules:
            lines.append(
                f"> **Ground Rule Violations:** {', '.join(region_analysis.ground_rule_violations)}"
            )
            lines.append("")

        score_100 = scoring_model.get("final_score_100", syn.get("overall_score", "N/A"))
        classification = rec_v2.get("classification", "N/A")
        confidence = rec_v2.get("confidence", "N/A")
        lines.append(f"**Score:** {score_100}/100 | **Classification:** {classification} | **Confidence:** {confidence}")
        lines.append("")

        # ── 3.i.1 SHARAD Subsurface Analysis ──
        lines.append(f"#### 3.{ri}.1 SHARAD Subsurface Analysis")
        lines.append("")
        sub_data = syn.get("subsurface_coverage", {})
        sub_ev = evidence.get("subsurface", "INCONCLUSIVE")
        lines.append(f"**Evidence Strength:** {sub_ev}")
        lines.append("")

        n_analyzed = sub_data.get("analyzed_count", 0)
        n_detect = sub_data.get("subsurface_detections", 0)
        total_tracks = sub_data.get("total_tracks", 0)

        if n_detect > 0:
            depth_summary = sub_data.get("depth_summary") or {}

            # Dielectric Sensitivity table
            depth_ranges = depth_summary.get("depth_ranges", {})
            if depth_ranges:
                lines.append("##### Dielectric Sensitivity")
                lines.append("")
                ds_headers = ["Scenario", "er", "Min Depth (m)", "Max Depth (m)", "Median Depth (m)"]
                ds_rows = []
                scenario_map = {
                    "porous_ice": ("Porous ice/regolith", 2.8),
                    "pure_ice": ("Pure water ice", 3.15),
                    "basaltic": ("Basaltic contrast", 4.0),
                }
                for label, (desc, eps) in scenario_map.items():
                    dr = depth_ranges.get(label, {})
                    if dr:
                        ds_rows.append([desc, eps, dr.get("min", "N/A"), dr.get("max", "N/A"), dr.get("median", "N/A")])
                if ds_rows:
                    lines.append(_md_table(ds_headers, ds_rows))
                    lines.append("")

            # SNR Distribution
            snr_dist = sub_data.get("snr_distribution")
            if snr_dist:
                lines.append("##### SNR Distribution")
                lines.append("")
                lines.append(f"- Mean: {snr_dist.get('mean', 'N/A')}, Median: {snr_dist.get('median', 'N/A')}, Std: {snr_dist.get('std', 'N/A')}")
                lines.append(f"- Range: {snr_dist.get('min', 'N/A')} - {snr_dist.get('max', 'N/A')}")
                lines.append("")

            # Clutter rejection
            clutter = sub_data.get("clutter_rejection")
            if clutter:
                lines.append("##### Clutter Rejection Methodology")
                lines.append("")
                lines.append(f"{clutter}")
                lines.append("")

            # Reflector spatial density
            density = sub_data.get("reflector_density_per_km")
            if density is not None:
                lines.append("##### Reflector Spatial Density")
                lines.append("")
                lines.append(f"- {density} detections per km of track")
                lines.append(f"- {n_analyzed} tracks analyzed, {n_detect} with subsurface reflectors")
                lines.append("")
            else:
                lines.append(f"- {n_analyzed} tracks analyzed, {n_detect} with subsurface reflectors")
                lines.append("")
        else:
            lines.append(f"- {n_analyzed} tracks analyzed (of {total_tracks} total), no subsurface reflectors detected")
            lines.append(f"- Evidence: INCONCLUSIVE")
            lines.append("")

        # Science distance note
        sci_penalty = scoring_model.get("science_distance_penalty")
        if sci_penalty and sci_penalty.get("applied") and sci_penalty.get("distance_km", 0) > 50:
            lines.append(f"> Note: Science target is {sci_penalty['distance_km']:.0f} km from best landing site (penalty factor: {sci_penalty['factor']:.2f})")
            lines.append("")

        # ── 3.i.2 CRISM Ice Detection ──
        lines.append(f"#### 3.{ri}.2 CRISM Ice Detection")
        lines.append("")
        mineral_data = syn.get("mineral_signatures", {})
        ice_ev = evidence.get("surface_ice", "INCONCLUSIVE")
        lines.append(f"**Evidence Strength:** {ice_ev}")
        lines.append("")

        crism_count = mineral_data.get("crism_count", 0)
        high_ice_count = mineral_data.get("high_ice_count", 0)
        high_hyd_count = mineral_data.get("high_hyd_count", 0)

        # Threshold Justification
        lines.append("##### Threshold Justification")
        lines.append("")
        lines.append("- Threshold: 0.3 (~3 sigma above global background mean)")
        lines.append("")

        # Evidence Separation
        lines.append("##### Evidence Separation")
        lines.append("")
        ev_sep_headers = ["Category", "Products", "Description"]
        ev_sep_rows = [
            ["Water Ice", high_ice_count, "Products with ice score >= threshold"],
            ["Spectral Hydration", high_hyd_count, "Hydration but below ice threshold"],
            ["Total CRISM", crism_count, "All CRISM products in region"],
        ]
        lines.append(_md_table(ev_sep_headers, ev_sep_rows))
        lines.append("")

        # Ice hotspot
        hotspot = mineral_data.get("ice_hotspot")
        if hotspot:
            lines.append(f"- Ice hotspot detected at ({hotspot['center_lat']:.2f} deg, {hotspot['center_lon']:.2f} deg)")
            lines.append("")

        # CNN H2O data
        cnn_data = syn.get("cnn_mineral_classification", {})
        if cnn_data:
            h2o_obs = cnn_data.get("h2o_observations", 0)
            h2o_rich = cnn_data.get("h2o_rich_observations", 0)
            obs_classified = cnn_data.get("observations_classified", 0)
            if obs_classified > 0:
                lines.append(f"- CNN classification: {obs_classified} observations classified, {h2o_obs} with H2O detection, {h2o_rich} H2O-rich (>= 1% pixels)")
                lines.append("")

        # ── 3.i.3 Cross-Instrument Consistency ──
        lines.append(f"#### 3.{ri}.3 Cross-Instrument Consistency")
        lines.append("")
        cross = syn.get("cross_instrument", {})
        consistency_score = cross.get("consistency_score", 0)
        consistency_class = cross.get("evidence_consistency", "insufficient_data")
        lines.append(f"**Probabilistic Consistency Score:** {consistency_score} (0-1)")
        lines.append("")

        # Multi-Scale Proximity
        multi_scale = cross.get("multi_scale_proximity")
        if multi_scale and isinstance(multi_scale, list):
            lines.append("##### Multi-Scale Proximity")
            lines.append("")
            ms_headers = ["Scale (km)", "Coincidences", "Possible Pairs", "Fraction", "Weight"]
            ms_rows = []
            for ms in multi_scale:
                ms_rows.append([
                    ms.get("scale_km", "N/A"),
                    ms.get("coincidences", "N/A"),
                    ms.get("possible_pairs", "N/A"),
                    ms.get("fraction", "N/A"),
                    ms.get("weight", "N/A"),
                ])
            if ms_rows:
                lines.append(_md_table(ms_headers, ms_rows))
                lines.append("")

        # Instrument Consistency Matrix
        inst_matrix = cross.get("instrument_matrix")
        if inst_matrix and isinstance(inst_matrix, dict):
            lines.append("##### Instrument Consistency Matrix")
            lines.append("")
            # Render the matrix as available
            for pair_key, pair_val in inst_matrix.items():
                if isinstance(pair_val, dict):
                    lines.append(f"- **{pair_key}:** distance={pair_val.get('distance_km', 'N/A')} km, coincident={pair_val.get('coincident', 'N/A')}")
                else:
                    lines.append(f"- **{pair_key}:** {pair_val}")
            lines.append("")

        lines.append(f"- Assessment: {consistency_class.replace('_', ' ')}")
        assessment_notes = cross.get("notes", cross.get("assessment_notes", ""))
        if assessment_notes:
            lines.append(f"- {assessment_notes}")
        lines.append("")

        # ── 3.i.4 Terrain Analysis ──
        lines.append(f"#### 3.{ri}.4 Terrain Analysis")
        lines.append("")
        eng = syn.get("engineering_feasibility", {})
        terrain_ev = evidence.get("terrain", "INCONCLUSIVE")
        lines.append(f"**Evidence Strength:** {terrain_ev}")
        lines.append("")

        if eng:
            grid_size = eng.get("grid_size", 0)
            fav_zones = eng.get("favorable_zones", 0)
            safety = eng.get("safety", "UNKNOWN")
            mean_slope = eng.get("mean_slope", "N/A")
            max_slope = eng.get("max_slope", "N/A")
            hazard_density = eng.get("hazard_density")

            lines.append(f"- Grid: {grid_size} sample points")
            lines.append(f"- Safety: {safety}")
            lines.append(f"- Mean slope: {mean_slope} deg, Max: {max_slope} deg")
            if hazard_density is not None:
                lines.append(f"- Hazard density: {hazard_density}% of points >5 deg")

            roughness_stats = eng.get("roughness_stats")
            if roughness_stats and isinstance(roughness_stats, dict):
                lines.append(f"- Roughness: mean {roughness_stats.get('mean', 'N/A')} deg, max {roughness_stats.get('max', 'N/A')} deg")
            lines.append("")

            # Zone breakdown
            if grid_size > 0:
                unfav_zones = grid_size - fav_zones
                # Estimate marginal as the difference (simplified: favorable vs unfavorable)
                fav_pct = round(100 * fav_zones / grid_size, 1) if grid_size else 0
                unfav_pct = round(100 * unfav_zones / grid_size, 1) if grid_size else 0
                zone_headers = ["Zone Type", "Count", "%"]
                zone_rows = [
                    ["Favorable (<5 deg)", fav_zones, f"{fav_pct}%"],
                    ["Unfavorable (>=5 deg)", unfav_zones, f"{unfav_pct}%"],
                ]
                lines.append(_md_table(zone_headers, zone_rows))
                lines.append("")
        else:
            lines.append("- No terrain data available")
            lines.append("")

        # ── 3.i.5 Climate Assessment ──
        lines.append(f"#### 3.{ri}.5 Climate Assessment")
        lines.append("")
        climate_data = syn.get("climate", {})

        if climate_data:
            seasonal = climate_data.get("seasonal_profile") or climate_data.get("annual_stats", {}).get("seasonal_profile", [])
            annual = climate_data.get("annual_stats", {})

            if seasonal and isinstance(seasonal, list):
                lines.append("##### Seasonal Profile (12 Ls bins)")
                lines.append("")
                sp_headers = ["Ls", "Season", "T Mean (K)", "T Min (K)", "Dust tau", "Wind (m/s)", "Frost Prob"]
                sp_rows = []
                for sp in seasonal:
                    temp = sp.get("temperature", {})
                    t_mean = temp.get("mean_k", sp.get("temp_mean_k", "N/A"))
                    t_min = temp.get("min_k", sp.get("temp_min_k", "N/A"))
                    sp_rows.append([
                        sp.get("ls", "N/A"),
                        sp.get("ls_label", ""),
                        f"{t_mean}" if t_mean != "N/A" else "N/A",
                        f"{t_min}" if t_min != "N/A" else "N/A",
                        sp.get("dust_tau", "N/A"),
                        sp.get("wind_mean_ms", "N/A"),
                        sp.get("frost_probability", "N/A"),
                    ])
                lines.append(_md_table(sp_headers, sp_rows))
                lines.append("")

            # Derived Metrics
            if annual:
                lines.append("##### Derived Metrics")
                lines.append("")
                temp_range = annual.get("temperature_amplitude_k")
                if temp_range is not None:
                    lines.append(f"- Annual temperature range: {temp_range} K")
                frost_bins = annual.get("frost_bins", 0)
                if frost_bins:
                    lines.append(f"- Frost duration: {frost_bins}/12 Ls bins")
                dust_var = annual.get("dust_variability")
                if dust_var is not None:
                    lines.append(f"- Dust variability: {dust_var}")
                pressure = annual.get("pressure_pa")
                if pressure is not None:
                    lines.append(f"- Surface pressure: {pressure} Pa")

                # Climate sub-score from scoring model
                climate_ss = sub_scores.get("climate", {})
                climate_score_val = climate_ss.get("score", 0)
                climate_bd = climate_ss.get("breakdown", {})
                lines.append(f"- Climate sub-score: {climate_score_val} (formula: 0.30xT + 0.25xD + 0.20xW + 0.25xF)")
                lines.append("")
            else:
                lines.append(f"- Climate score: {climate_data.get('climate_score', 'N/A')}")
                lines.append(f"- Summary: {climate_data.get('climate_summary', 'N/A')}")
                lines.append("")
        else:
            lines.append("- No climate data available")
            lines.append("")

        # ── 3.i.6 Scoring Breakdown ──
        lines.append(f"#### 3.{ri}.6 Scoring Breakdown")
        lines.append("")
        if scoring_model:
            weights = scoring_model.get("weights", DEFAULT_WEIGHTS)
            sb_headers = ["Component", "Weight", "Raw Score", "Weighted"]
            sb_rows = []
            for comp_key, comp_label, w_key in [
                ("subsurface_potential", "Subsurface Potential", "w1"),
                ("surface_ice", "Surface Ice", "w2"),
                ("terrain_safety", "Terrain Safety", "w3"),
                ("climate", "Climate", "w4"),
            ]:
                ss_entry = sub_scores.get(comp_key, {})
                raw = ss_entry.get("score", 0)
                wt = weights.get(w_key, 0)
                weighted = round(raw * wt, 4)
                sb_rows.append([comp_label, wt, f"{raw:.4f}", f"{weighted:.4f}"])

            composite = scoring_model.get("composite_score", scoring_model.get("final_score", 0))
            sb_rows.append(["**Composite**", "", "", f"**{composite:.4f}**"])
            lines.append(_md_table(sb_headers, sb_rows))
            lines.append("")

            # Science distance penalty
            if sci_penalty and sci_penalty.get("applied"):
                lines.append(f"- Science distance: {sci_penalty['distance_km']:.0f} km")
                lines.append(f"- Penalty factor: {sci_penalty['factor']:.2f}")
                lines.append(f"- Adjusted score: {scoring_model.get('final_score', 0):.4f}")
                lines.append("")
        else:
            lines.append(f"- Overall score: {syn.get('overall_score', 'N/A')}/100")
            lines.append("")

        lines.append("")

    lines.append("---")
    lines.append("")

    # ══════════════════════════════════════════════════════════════
    # Section 4: Uncertainty & Sensitivity Analysis
    # ══════════════════════════════════════════════════════════════
    lines.append("## 4. Uncertainty & Sensitivity Analysis")
    lines.append("")

    # Compute or retrieve sensitivity for the top-ranked region
    sensitivity_data = None
    for region_analysis in session.regions:
        syn = region_analysis.synthesis or {}
        if syn.get("scoring_model"):
            # Compute sensitivity analysis from the synthesis data
            try:
                sensitivity_data = sensitivity_analysis(
                    subsurface_data=syn.get("subsurface_coverage", {}),
                    dielectric_data=syn.get("dielectric_analysis", {}),
                    cross_instrument_data=syn.get("cross_instrument", {}),
                    mineral_data=syn.get("mineral_signatures", {}),
                    cnn_data=syn.get("cnn_mineral_classification", {}),
                    engineering_data=syn.get("engineering_feasibility", {}),
                    climate_data=syn.get("climate", {}),
                )
            except Exception as exc:
                logger.warning(f"Sensitivity analysis failed for {region_analysis.region_name}: {exc}")
            break  # Only compute for top-ranked

    if sensitivity_data:
        # 4.1 Weight Sensitivity
        weight_sens = sensitivity_data.get("weight_sensitivity", [])
        if weight_sens:
            lines.append("### 4.1 Weight Sensitivity")
            lines.append("")
            ws_headers = ["Parameter", "Low", "High", "Delta"]
            ws_rows = [[ws["parameter"], f"{ws['low']:.4f}", f"{ws['high']:.4f}", f"{ws['delta']:.4f}"] for ws in weight_sens]
            lines.append(_md_table(ws_headers, ws_rows))
            lines.append("")

        # 4.2 Sub-score Sensitivity
        subscore_sens = sensitivity_data.get("subscore_sensitivity", [])
        if subscore_sens:
            lines.append("### 4.2 Sub-score Sensitivity")
            lines.append("")
            ss_headers = ["Parameter", "Low", "High", "Delta"]
            ss_rows = [[ss["parameter"], f"{ss['low']:.4f}", f"{ss['high']:.4f}", f"{ss['delta']:.4f}"] for ss in subscore_sens]
            lines.append(_md_table(ss_headers, ss_rows))
            lines.append("")

        # 4.3 Overall Score Range
        overall_range = sensitivity_data.get("overall_range", {})
        lines.append("### 4.3 Overall Score Range")
        lines.append("")
        lines.append(f"- Minimum: {overall_range.get('min', 'N/A')}")
        lines.append(f"- Maximum: {overall_range.get('max', 'N/A')}")
        lines.append("")
    else:
        lines.append("Sensitivity analysis data not available.")
        lines.append("")

    lines.append("---")
    lines.append("")

    # ══════════════════════════════════════════════════════════════
    # Section 5: Recommendation
    # ══════════════════════════════════════════════════════════════
    lines.append("## 5. Recommendation")
    lines.append("")

    recommended = comparison.get("recommended")
    if recommended:
        rec_ev = recommended.get("evidence_strength", {})
        rec_classification = recommended.get("recommendation", "N/A")
        rec_confidence = rec_ev.get("overall", "N/A") if isinstance(rec_ev, dict) else "N/A"

        lines.append(f"### Classification: {rec_classification}")
        lines.append(f"### Confidence: {rec_confidence}")
        lines.append("")
        lines.append(f"**Region:** {recommended['region_name']}")
        lines.append(f"**Score:** {recommended['score']}/100")
        lines.append("")
        if recommended.get("highlights"):
            for h in recommended["highlights"]:
                lines.append(f"- {h}")
            lines.append("")

        # Retrieve recommendation_v2 rationale from the top region
        for region_analysis in session.regions:
            if region_analysis.region_id == recommended.get("region_id"):
                syn = region_analysis.synthesis or {}
                rv2 = syn.get("recommendation_v2", {})
                rationale = rv2.get("rationale", "")
                if rationale:
                    lines.append(rationale)
                    lines.append("")
                critical_flags = rv2.get("critical_flags", [])
                if critical_flags:
                    lines.append("**Critical Flags:**")
                    for flag in critical_flags:
                        lines.append(f"- {flag}")
                    lines.append("")
                break

        lines.append("**CRITICAL RULE:** If cross-instrument corroboration is zero, this site is NOT recommended as a primary science target. It may be engineering-safe, but science confidence is LOW.")
        lines.append("")
    else:
        lines.append("No recommendation available -- no regions were successfully analyzed.")
        lines.append("")

    lines.append("---")
    lines.append("")

    # ══════════════════════════════════════════════════════════════
    # Appendix A: Scoring Methodology
    # ══════════════════════════════════════════════════════════════
    lines.append("## Appendix A: Scoring Methodology")
    lines.append("")
    for wline in WEIGHT_JUSTIFICATION.split("\n"):
        lines.append(wline if wline.strip() else "")
    lines.append("")

    lines.append("### Score Formula")
    lines.append("")
    lines.append("Score = w1 x subsurface_potential + w2 x surface_ice_score + w3 x terrain_safety + w4 x climate_score")
    lines.append("")

    lines.append("### Normalization Details")
    lines.append("")
    lines.append("- **Subsurface (0-1):** 0.0=no data, 0.2=analyzed/no reflectors, 0.4=deep(>80m), 0.6=moderate(30-80m), 0.8=shallow(10-30m), 1.0=very shallow(<10m) + multi-track + ice-like dielectric")
    lines.append("- **Surface Ice (0-1):** Sum of CRISM spectral component (0-0.5) and CNN H2O component (0-0.5)")
    lines.append("- **Terrain (0-1):** 1.0=FAVORABLE+>60% favorable zones, 0.7=FAVORABLE, 0.5=MARGINAL, 0.2=UNFAVORABLE, 0.0=unknown; -0.1 roughness penalty")
    lines.append("- **Climate (0-1):** 0.30xTemperature + 0.25xDust + 0.20xWind + 0.25xFrost (each 0 or 0.5 or 1.0 raw)")
    lines.append("")

    lines.append("### Science-Distance Penalty")
    lines.append("")
    lines.append("penalty_factor = max(0.5, 1.0 - distance_km / 200.0)")
    lines.append("")
    lines.append("Applied when science target >50 km from landing site.")
    lines.append("")

    lines.append("---")
    lines.append("")

    # ══════════════════════════════════════════════════════════════
    # Appendix B: Sensitivity Analysis
    # ══════════════════════════════════════════════════════════════
    lines.append("## Appendix B: Sensitivity Analysis")
    lines.append("")
    if sensitivity_data:
        lines.append(f"Baseline score: {sensitivity_data.get('baseline_score', 'N/A')}")
        lines.append("")

        weight_sens = sensitivity_data.get("weight_sensitivity", [])
        if weight_sens:
            lines.append("### Weight Perturbation (+/-0.05, renormalized)")
            lines.append("")
            ws_headers = ["Parameter", "Low", "High", "Delta"]
            ws_rows = [[ws["parameter"], f"{ws['low']:.4f}", f"{ws['high']:.4f}", f"{ws['delta']:.4f}"] for ws in weight_sens]
            lines.append(_md_table(ws_headers, ws_rows))
            lines.append("")

        subscore_sens = sensitivity_data.get("subscore_sensitivity", [])
        if subscore_sens:
            lines.append("### Sub-score Perturbation (+/-0.1)")
            lines.append("")
            ss_headers = ["Parameter", "Low", "High", "Delta"]
            ss_rows = [[ss["parameter"], f"{ss['low']:.4f}", f"{ss['high']:.4f}", f"{ss['delta']:.4f}"] for ss in subscore_sens]
            lines.append(_md_table(ss_headers, ss_rows))
            lines.append("")

        overall_range = sensitivity_data.get("overall_range", {})
        lines.append(f"Overall range: [{overall_range.get('min', 'N/A')}, {overall_range.get('max', 'N/A')}]")
        lines.append("")
    else:
        lines.append("Sensitivity analysis data not available for this session.")
        lines.append("")

    lines.append("---")
    lines.append("")

    # ══════════════════════════════════════════════════════════════
    # Appendix C: Instrument Consistency Matrix
    # ══════════════════════════════════════════════════════════════
    lines.append("## Appendix C: Instrument Consistency Matrix")
    lines.append("")
    for region_analysis in session.regions:
        syn = region_analysis.synthesis or {}
        cross = syn.get("cross_instrument", {})
        lines.append(f"### {region_analysis.region_name}")
        lines.append("")
        consistency_class = cross.get("evidence_consistency", "insufficient_data")
        consistency_score = cross.get("consistency_score", 0)
        lines.append(f"- Consistency: {consistency_class.replace('_', ' ')}")
        lines.append(f"- Score: {consistency_score}")
        lines.append("")

        inst_matrix = cross.get("instrument_matrix")
        if inst_matrix and isinstance(inst_matrix, dict):
            for pair_key, pair_val in inst_matrix.items():
                if isinstance(pair_val, dict):
                    lines.append(f"  - {pair_key}: distance={pair_val.get('distance_km', 'N/A')} km, coincident={pair_val.get('coincident', 'N/A')}")
                else:
                    lines.append(f"  - {pair_key}: {pair_val}")
            lines.append("")

        assessment_notes = cross.get("notes", cross.get("assessment_notes", ""))
        if assessment_notes:
            lines.append(f"Assessment: {assessment_notes}")
            lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("*Report generated by MarsLab Mission Assessment Pipeline v2.0*")

    return "\n".join(lines)


# =============================================================================
# Executive Summary Generation
# =============================================================================

EXECUTIVE_SYSTEM_PROMPT = """You are a Mars mission planning analyst writing an executive summary
for a v2 mission assessment report. Be concise, data-driven, and professional.
Cite specific numbers (composite scores, sub-scores, evidence strength classifications).
The scoring model uses weighted components: 0.30 Subsurface + 0.25 Surface Ice + 0.25 Terrain + 0.20 Climate.
Mention evidence strength (STRONG/MODERATE/WEAK/INCONCLUSIVE) and recommendation classification
(Science-ready candidate / Screening-level / Engineering-safe only / Low priority).
Write 2-3 paragraphs maximum."""


async def _generate_executive_summary(
    session: ReportSession,
    on_chunk: Optional[Callable[[str], Awaitable[None]]] = None,
) -> str:
    """Generate executive summary using Llama or fallback."""
    comparison = session.comparison or {}
    rankings = comparison.get("rankings", [])
    recommended = comparison.get("recommended", {})
    rules = session.config.ground_rules

    prompt = f"""Write an executive summary for a Mars mission assessment report (v2 pipeline).

Scoring Model: Score = 0.30 x Subsurface + 0.25 x Surface Ice + 0.25 x Terrain + 0.20 x Climate

Ground Rules: {rules.summary()}
{f"User Notes: {rules.custom_notes}" if rules.custom_notes else ""}

Regions Analyzed: {len(rankings)}
Rankings (best to worst):
"""
    for r in rankings:
        violations = f" [FAILS: {', '.join(r.get('ground_rule_violations', []))}]" if not r.get("passes_ground_rules", True) else ""
        ev = r.get("evidence_strength", {})
        overall_ev = ev.get("overall", "N/A") if isinstance(ev, dict) else "N/A"
        ss = r.get("sub_scores", {})
        prompt += f"  {r['rank']}. {r['region_name']}: {r['overall_score']}/100 ({r['recommendation']}){violations}\n"
        prompt += f"     Sub-scores: subsurface={ss.get('subsurface', 0):.3f}, surface_ice={ss.get('surface_ice', 0):.3f}, terrain={ss.get('terrain', 0):.3f}, climate={ss.get('climate', 0):.3f}\n"
        prompt += f"     Evidence: {overall_ev}, Cross-instrument: {r.get('cross_instrument_consistency', 'N/A')}\n"
        prompt += f"     Engineering: {r['engineering_safety']}, SHARAD detections: {r['subsurface_detections']}, Ice: {r['ice_count']} high-ice products\n"

    if recommended:
        rec_ev = recommended.get("evidence_strength", {})
        rec_overall_ev = rec_ev.get("overall", "N/A") if isinstance(rec_ev, dict) else "N/A"
        prompt += f"\nRecommended: {recommended['region_name']} (score {recommended['score']}/100, evidence: {rec_overall_ev})\n"

    prompt += "\nWrite a 2-3 paragraph executive summary. Reference the composite scoring model, evidence strength classifications, and key differentiators between regions."

    if on_chunk:
        response = await _call_groq_streaming(
            prompt, EXECUTIVE_SYSTEM_PROMPT, temperature=0.4, on_chunk=on_chunk
        )
    else:
        response = await _call_groq(
            prompt, EXECUTIVE_SYSTEM_PROMPT, temperature=0.4
        )

    if not response:
        return _generate_executive_summary_fallback(session)

    return response


def _generate_executive_summary_fallback(session: ReportSession) -> str:
    """Template-based executive summary when LLM is unavailable."""
    comparison = session.comparison or {}
    rankings = comparison.get("rankings", [])
    recommended = comparison.get("recommended")

    if not rankings:
        return "No regions were successfully analyzed."

    parts = []
    parts.append(
        f"This report compares {len(rankings)} candidate Mars landing sites "
        f"using the v2 weighted composite scoring model (0.30 Subsurface + "
        f"0.25 Surface Ice + 0.25 Terrain + 0.20 Climate) with evidence "
        f"strength classification and cross-instrument consistency analysis."
    )

    if recommended:
        rec_ev = recommended.get("evidence_strength", {})
        overall_ev = rec_ev.get("overall", "N/A") if isinstance(rec_ev, dict) else "N/A"
        rec_classification = recommended.get("recommendation", "N/A")
        parts.append(
            f"\n\n**{recommended['region_name']}** ranks highest with a composite score of "
            f"**{recommended['score']}/100** (classification: {rec_classification}, "
            f"evidence strength: {overall_ev}). "
        )
        if recommended.get("highlights"):
            parts.append("Key factors: " + "; ".join(recommended["highlights"]) + ".")

    # Compare top 2 if available
    if len(rankings) >= 2:
        r1, r2 = rankings[0], rankings[1]
        diff = r1["overall_score"] - r2["overall_score"]
        r1_ev = r1.get("evidence_strength", {})
        r1_overall = r1_ev.get("overall", "N/A") if isinstance(r1_ev, dict) else "N/A"
        r2_ev = r2.get("evidence_strength", {})
        r2_overall = r2_ev.get("overall", "N/A") if isinstance(r2_ev, dict) else "N/A"
        parts.append(
            f"\n\nThe margin between the top choice ({r1['region_name']}, "
            f"{r1['overall_score']}/100, evidence: {r1_overall}) and runner-up ({r2['region_name']}, "
            f"{r2['overall_score']}/100, evidence: {r2_overall}) is {diff} points."
        )

    # Note any ground-rule violations
    failed = [r for r in rankings if not r.get("passes_ground_rules", True)]
    if failed:
        names = ", ".join(r["region_name"] for r in failed)
        parts.append(
            f" Note: {names} failed one or more ground-rule constraints."
        )

    # Note zero corroboration warning
    for r in rankings[:1]:
        rec_v2 = r.get("recommendation_v2", {})
        flags = rec_v2.get("critical_flags", [])
        if "Zero cross-instrument corroboration" in flags:
            parts.append(
                f"\n\n**Warning:** The top-ranked site has zero cross-instrument "
                f"corroboration and cannot be recommended as a primary science "
                f"target without independent confirmation."
            )

    return "".join(parts)


# =============================================================================
# Main Report Loop
# =============================================================================


async def run_report(
    config: ReportConfig,
    session: ReportSession,
) -> AsyncGenerator[Dict[str, Any], None]:
    """Main report generation loop. Yields SSE events as it progresses."""
    try:
        # ── Phase 0: Filter regions ──
        session.status = "filtering"
        candidates = filter_regions(config.ground_rules)

        # Cap to max_regions (pick first N — they're in dict order which is insertion order)
        if len(candidates) > config.max_regions:
            candidates = candidates[: config.max_regions]

        if not candidates:
            raise ValueError(
                "No regions match the specified ground rules. "
                "Try relaxing your constraints."
            )

        session.candidate_regions = [
            {
                "region_id": r.region_id,
                "region_name": r.display_name,
                "center_lat": r.center_lat,
                "center_lon": r.center_lon,
                "tags": r.tags,
            }
            for r in candidates
        ]

        yield {
            "event": "session_start",
            "data": {
                "session_id": session.session_id,
                "region_count": len(candidates),
                "regions": session.candidate_regions,
                "ground_rules_summary": config.ground_rules.summary(),
            },
        }

        yield {
            "event": "filter_result",
            "data": {
                "total_regions": len(list_all_regions()),
                "candidates": len(candidates),
                "filtered_out": len(list_all_regions()) - len(candidates),
                "reason_summary": config.ground_rules.summary(),
            },
        }

        # ── Phase 1: Per-region analysis ──
        session.status = "analyzing"

        for i, region in enumerate(candidates):
            analysis = await _analyze_region(region, config, i, session.emit)

            # Post-analysis ground rule checks
            _check_post_analysis_rules(analysis, config.ground_rules)

            session.regions.append(analysis)

            score = analysis.synthesis.get("overall_score", 0) if analysis.synthesis else 0
            rec = analysis.synthesis.get("recommendation", "N/A") if analysis.synthesis else "N/A"

            yield {
                "event": "region_complete",
                "data": {
                    "region_index": i,
                    "region_id": region.region_id,
                    "region_name": region.display_name,
                    "score": score,
                    "recommendation": rec,
                    "passes_ground_rules": analysis.passes_ground_rules,
                    "ground_rule_violations": analysis.ground_rule_violations,
                },
            }

        # ── Phase 2: Compare regions ──
        session.status = "comparing"
        comparison = compare_regions(session.regions)
        session.comparison = comparison
        session.rankings = comparison.get("rankings", [])

        yield {
            "event": "comparison_complete",
            "data": {
                "rankings": comparison["rankings"],
                "category_winners": comparison.get("category_winners", {}),
                "recommended": comparison.get("recommended"),
            },
        }

        # ── Phase 3: Generate report ──
        session.status = "generating"

        # Executive summary (LLM or fallback)
        reasoning_queue: asyncio.Queue[Optional[Dict[str, Any]]] = asyncio.Queue()

        async def _emit_chunk(text: str):
            await reasoning_queue.put(
                {"event": "reasoning_chunk", "data": {"text": text}}
            )

        ollama_available = await _check_groq()

        if ollama_available:
            yield {"event": "reasoning_start", "data": {"phase": "executive_summary"}}

            summary_result = ""

            async def _do_summary():
                nonlocal summary_result
                summary_result = await _generate_executive_summary(
                    session, on_chunk=_emit_chunk
                )
                await reasoning_queue.put(None)

            summary_task = asyncio.create_task(_do_summary())
            while True:
                item = await reasoning_queue.get()
                if item is None:
                    break
                yield item
            await summary_task

            session.executive_summary = summary_result
            yield {"event": "reasoning_end", "data": {"phase": "executive_summary"}}
        else:
            session.executive_summary = _generate_executive_summary_fallback(session)

        # Build the full markdown report
        session.report_markdown = build_report_markdown(session)

        yield {
            "event": "report_ready",
            "data": {
                "session_id": session.session_id,
                "report_preview": session.report_markdown[:500],
            },
        }

        # ── Done ──
        session.status = "done"
        yield {
            "event": "done",
            "data": {
                "session_id": session.session_id,
                "rankings": session.rankings,
                "recommended": comparison.get("recommended"),
            },
        }

    except Exception as e:
        session.status = "error"
        session.error = str(e)
        logger.error(f"Report session {session.session_id} error: {e}")
        yield {
            "event": "error",
            "data": {"error": str(e), "session_id": session.session_id},
        }


# =============================================================================
# Background Execution + Resumable Streaming
# =============================================================================


def start_report_background(config: ReportConfig) -> ReportSession:
    """Start the report as a background asyncio.Task."""
    _evict_old_report_sessions()
    session_id = str(uuid.uuid4())[:8]
    session = ReportSession(session_id=session_id, config=config)
    _report_sessions[session_id] = session

    async def _run():
        try:
            async for event in run_report(config, session):
                await session.emit(event)
        except Exception as e:
            logger.error(f"Background report {session_id} crashed: {e}")
            session.status = "error"
            session.error = str(e)
            await session.emit(
                {
                    "event": "error",
                    "data": {"error": str(e), "session_id": session_id},
                }
            )
        finally:
            await session.emit({"event": "stream_end"})
            session._task = None
            _save_report_session(session)

    session._task = asyncio.create_task(_run())
    return session


async def stream_report_events(
    session: ReportSession,
    from_index: int = 0,
) -> AsyncGenerator[Dict[str, Any], None]:
    """Yield events from a report session's buffer, then follow live."""
    cursor = from_index

    while True:
        while cursor < len(session.events):
            event = session.events[cursor]
            cursor += 1
            yield event
            if event.get("event") == "stream_end":
                return

        if session.is_terminal and cursor >= len(session.events):
            return

        try:
            async with session._condition:
                if cursor < len(session.events):
                    continue
                await asyncio.wait_for(session._condition.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            pass
