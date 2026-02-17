"""
Tool registry for the MarsLab Workflow Research Assistant.

Wraps existing analysis functions from agent_tasks.py and analysis modules
as structured tools with Pydantic-validated specs.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Coroutine, Dict, List, Optional

from agent.schemas import ToolSpec

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Registry of executable tools available to the workflow engine."""

    def __init__(self) -> None:
        self._tools: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        name: str,
        executor: Callable[..., Any],
        description: str,
        params_schema: Dict[str, Any],
        requires_approval: bool = False,
        approval_template: Optional[str] = None,
    ) -> None:
        self._tools[name] = {
            "spec": ToolSpec(
                name=name,
                description=description,
                params_schema=params_schema,
                requires_approval=requires_approval,
                approval_template=approval_template,
            ),
            "executor": executor,
        }

    def get_tool(self, name: str) -> Optional[Dict[str, Any]]:
        return self._tools.get(name)

    def get_spec(self, name: str) -> Optional[ToolSpec]:
        tool = self._tools.get(name)
        return tool["spec"] if tool else None

    def get_all_specs(self) -> List[ToolSpec]:
        return [t["spec"] for t in self._tools.values()]

    def tool_names(self) -> List[str]:
        return list(self._tools.keys())

    async def execute(self, name: str, context: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool and return its result dict.

        Each tool executor is called as: executor(context, params) -> dict
        """
        tool = self._tools.get(name)
        if not tool:
            raise ValueError(f"Unknown tool: {name}")

        executor = tool["executor"]
        try:
            import asyncio
            if asyncio.iscoroutinefunction(executor):
                result = await executor(context, params)
            else:
                result = executor(context, params)
            return result
        except Exception as e:
            logger.exception(f"Tool {name} failed: {e}")
            raise


# ===========================================================================
# Singleton instance
# ===========================================================================

workflow_tools = ToolRegistry()


# ===========================================================================
# Tool executor wrappers
# Each wraps an existing function from agent_tasks.py or analysis modules.
# Signature: (context: dict, params: dict) -> dict
# ===========================================================================

def _tool_resolve_region(context: Dict, params: Dict) -> Dict:
    from api.agent_tasks import resolve_region, bbox_from_coords

    query = params.get("query", "")
    bbox = resolve_region(query)
    if bbox is None:
        # Try as coordinates: "lat, lon"
        parts = query.replace(",", " ").split()
        if len(parts) >= 2:
            try:
                lat, lon = float(parts[0]), float(parts[1])
                radius = float(parts[2]) if len(parts) > 2 else 2.0
                bbox = bbox_from_coords(lat, lon, radius)
            except ValueError:
                pass

    if bbox is None:
        return {"success": False, "error": f"Could not resolve region: {query}"}

    bbox_dict = {
        "min_lat": bbox.min_lat, "max_lat": bbox.max_lat,
        "min_lon": bbox.min_lon, "max_lon": bbox.max_lon,
    }
    context["region_bbox"] = bbox_dict
    context["region_name"] = query
    return {"success": True, "bbox": bbox_dict, "region_name": query}


async def _tool_search_products(context: Dict, params: Dict) -> Dict:
    from api.agent_tasks import search_region, RegionBBox

    instrument = params.get("instrument", "SHARAD_HIGHRES")
    bbox_dict = params.get("bbox") or context.get("region_bbox")
    if not bbox_dict:
        return {"success": False, "error": "No region bbox available. Run resolve_region first."}

    bbox = RegionBBox(**bbox_dict)
    result = await search_region(instrument, bbox)

    products = result.data.get("products", []) if result.success else []

    # Store in context keyed by instrument
    key = f"products_{instrument.lower()}"
    existing = context.get(key, [])
    existing.extend(products)
    context[key] = existing

    # Also maintain a flat list for cross-instrument tools
    all_products = context.get("all_products", [])
    all_products.extend(products)
    context["all_products"] = all_products

    return {
        "success": result.success,
        "instrument": instrument,
        "count": len(products),
        "summary": result.summary,
    }


def _tool_check_local_data(context: Dict, params: Dict) -> Dict:
    from api.agent_tasks import check_local_data

    products = params.get("products") or context.get("all_products", [])
    result = check_local_data(products)
    if result.success:
        context["local_data_status"] = result.data
    return {
        "success": result.success,
        "data": result.data,
        "summary": result.summary,
    }


async def _tool_download_products(context: Dict, params: Dict) -> Dict:
    from api.agent_tasks import download_data

    products = params.get("products") or context.get("all_products", [])
    instruments = params.get("instruments")
    if instruments:
        products = [p for p in products if p.get("instrument") in instruments]

    result = await download_data(products)
    if result.success:
        context["download_result"] = result.data
    return {
        "success": result.success,
        "data": result.data,
        "summary": result.summary,
    }


def _tool_analyze_slope(context: Dict, params: Dict) -> Dict:
    from api.agent_tasks import slope_analysis, RegionBBox

    bbox_dict = params.get("bbox") or context.get("region_bbox")
    if not bbox_dict:
        return {"success": False, "error": "No region bbox available."}

    bbox = RegionBBox(**bbox_dict)
    result = slope_analysis(bbox)
    if result.success:
        context["slope_result"] = result.data
    return {
        "success": result.success,
        "data": result.data,
        "summary": result.summary,
    }


def _tool_analyze_subsurface(context: Dict, params: Dict) -> Dict:
    from api.agent_tasks import subsurface_scan

    products = params.get("products") or context.get("all_products", [])
    sharad_products = [p for p in products if p.get("instrument") in ("SHARAD", "SHARAD_HIGHRES")]
    result = subsurface_scan(sharad_products)
    if result.success:
        context["subsurface_result"] = result.data
    return {
        "success": result.success,
        "data": result.data,
        "summary": result.summary,
    }


def _tool_analyze_minerals(context: Dict, params: Dict) -> Dict:
    from api.agent_tasks import mineral_analysis

    products = params.get("products") or context.get("all_products", [])
    crism_products = [p for p in products if "CRISM" in (p.get("instrument") or "")]
    result = mineral_analysis(crism_products)
    if result.success:
        context["mineral_result"] = result.data
    return {
        "success": result.success,
        "data": result.data,
        "summary": result.summary,
    }


async def _tool_classify_minerals_cnn(context: Dict, params: Dict) -> Dict:
    from api.agent_tasks import mineral_cnn_classify

    products = params.get("products") or context.get("all_products", [])
    crism_products = [p for p in products if "CRISM" in (p.get("instrument") or "")]
    result = await mineral_cnn_classify(crism_products)
    if result.success:
        context["mineral_cnn_result"] = result.data
    return {
        "success": result.success,
        "data": result.data,
        "summary": result.summary,
    }


def _tool_find_sharad_intersections(context: Dict, params: Dict) -> Dict:
    from api.agent_tasks import find_sharad_intersections

    products = params.get("products") or context.get("all_products", [])
    result = find_sharad_intersections(products)
    context["sharad_intersections"] = result
    return {"success": True, "data": result}


def _tool_targeted_subsurface_at_ice(context: Dict, params: Dict) -> Dict:
    from api.agent_tasks import targeted_subsurface_at_ice, TaskResult

    products = params.get("products") or context.get("all_products", [])
    # Build all_results dict from context
    all_results: Dict[str, Any] = {}
    if context.get("mineral_result"):
        all_results["mineral"] = TaskResult(
            task_type="mineral_analysis", success=True,
            data=context["mineral_result"],
        )
    if context.get("mineral_cnn_result"):
        all_results["mineral_cnn"] = TaskResult(
            task_type="mineral_cnn", success=True,
            data=context["mineral_cnn_result"],
        )

    result = targeted_subsurface_at_ice(products, all_results)
    if result.success:
        context["targeted_subsurface"] = result.data
    return {
        "success": result.success,
        "data": result.data,
        "summary": result.summary,
    }


def _tool_run_sharad_inversion(context: Dict, params: Dict) -> Dict:
    from api.agent_tasks import sharad_physics_inversion

    products = params.get("products") or context.get("all_products", [])
    sharad_products = [p for p in products if p.get("instrument") in ("SHARAD", "SHARAD_HIGHRES")]
    result = sharad_physics_inversion(sharad_products)
    if result.success:
        context["sharad_inversion"] = result.data
    return {
        "success": result.success,
        "data": result.data,
        "summary": result.summary,
    }


def _tool_detect_terraces(context: Dict, params: Dict) -> Dict:
    try:
        from analysis.epsilon_terrace.terrace_detect import extract_radial_profiles
    except ImportError:
        from analysis.epsilon_terrace.terrace_detect import extract_radial_profiles

    dtm_path = params.get("dtm_path")
    center_lat = params.get("lat") or params.get("center_lat")
    center_lon = params.get("lon") or params.get("center_lon")
    n_azimuths = params.get("n_azimuths", 36)
    max_radius_m = params.get("max_radius_m", 3000.0)

    # Fallback: find a locally available DTM from search results in context
    if not dtm_path:
        for p in context.get("products_hirise_dtm", []):
            lp = p.get("local_path")
            if lp:
                dtm_path = lp
                break

    if not dtm_path:
        return {"success": False, "error": "dtm_path required for terrace detection (no local DTM found in context)"}

    try:
        terrace_result = extract_radial_profiles(
            dtm_path,
            center_lat=center_lat,
            center_lon=center_lon,
            n_azimuths=n_azimuths,
            max_radius_m=max_radius_m,
        )
        data = {
            "is_terraced": terrace_result.is_terraced,
            "n_terraces": len(terrace_result.terraces),
            "terrace_depths": terrace_result.terrace_depths,
            "crater_center": {"lat": terrace_result.crater_lat, "lon": terrace_result.crater_lon},
        }
        context["terrace_result"] = data
        return {"success": True, "data": data}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def _tool_epsilon_inversion(context: Dict, params: Dict) -> Dict:
    try:
        from analysis.epsilon_terrace.run import run_pipeline_async
    except ImportError:
        from analysis.epsilon_terrace.run import run_pipeline_async

    sharad_pid = params.get("sharad_product_id")
    if not sharad_pid:
        # Pick first SHARAD product from context
        sharad_products = context.get("products_sharad_highres", [])
        if sharad_products:
            sharad_pid = sharad_products[0].get("product_id")
    if not sharad_pid:
        return {"success": False, "error": "No SHARAD product ID available."}

    try:
        result = await run_pipeline_async(sharad_pid)
        context["epsilon_result"] = result
        return {"success": True, "data": result}
    except Exception as e:
        return {"success": False, "error": str(e)}


def _tool_evaluate_ice_evidence(context: Dict, params: Dict) -> Dict:
    # Synthesize multi-criteria ice evidence from context
    subsurface = context.get("subsurface_result", {})
    mineral = context.get("mineral_result", {})
    targeted = context.get("targeted_subsurface", {})
    intersections = context.get("sharad_intersections", {})

    ice_score = 0.0
    notes = []

    # Subsurface reflectors
    tracks = subsurface.get("tracks", [])
    detected = [t for t in tracks if t.get("subsurface_detected")]
    if detected:
        ice_score += 0.3
        notes.append(f"{len(detected)} SHARAD subsurface reflectors detected")

    # CRISM ice
    ice_count = mineral.get("high_ice_count", 0)
    if ice_count > 0:
        ice_score += 0.3
        notes.append(f"CRISM ice signals in {ice_count} observations")

    # Co-located evidence
    reflectors_at_ice = targeted.get("reflectors_at_ice", 0)
    if reflectors_at_ice > 0:
        ice_score += 0.3
        notes.append(f"Co-located surface+subsurface ice at {reflectors_at_ice} locations")

    # Geometric intersections
    geo_crism = intersections.get("sharad_crism_intersections", 0)
    if geo_crism > 0:
        ice_score += 0.1
        notes.append(f"{geo_crism} SHARAD tracks cross CRISM footprints")

    ice_score = min(ice_score, 1.0)
    data = {
        "ice_probability": round(ice_score, 2),
        "evidence_notes": notes,
        "subsurface_reflectors": len(detected),
        "crism_ice_detections": ice_count,
        "colocated_evidence": reflectors_at_ice,
    }
    context["ice_evidence"] = data
    return {"success": True, "data": data}


def _tool_generate_report(context: Dict, params: Dict) -> Dict:
    """Generate a markdown summary report from workflow context."""
    lines = []
    title = params.get("title", "MarsLab Workflow Report")
    lines.append(f"# {title}")
    lines.append("")

    region = context.get("region_name", "Unknown Region")
    lines.append(f"**Region:** {region}")
    lines.append("")

    # Products summary
    all_products = context.get("all_products", [])
    if all_products:
        by_inst: Dict[str, int] = {}
        for p in all_products:
            inst = p.get("instrument", "unknown")
            by_inst[inst] = by_inst.get(inst, 0) + 1
        lines.append("## Data Products")
        lines.append("| Instrument | Count |")
        lines.append("|------------|-------|")
        for inst, count in sorted(by_inst.items()):
            lines.append(f"| {inst} | {count} |")
        lines.append("")

    # SHARAD intersections
    intersections = context.get("sharad_intersections", {})
    if intersections:
        lines.append("## SHARAD Track Intersections")
        lines.append(f"- CRISM: {intersections.get('sharad_crism_intersections', 0)}")
        lines.append(f"- HiRISE DTM: {intersections.get('sharad_dtm_intersections', 0)}")
        lines.append("")

    # Ice evidence
    ice = context.get("ice_evidence", {})
    if ice:
        lines.append("## Ice Evidence Summary")
        lines.append(f"**Ice probability:** {ice.get('ice_probability', 0):.0%}")
        for note in ice.get("evidence_notes", []):
            lines.append(f"- {note}")
        lines.append("")

    # Epsilon inversion
    eps = context.get("epsilon_result", {})
    if eps:
        lines.append("## Dielectric Inversion")
        best_eps = eps.get("best_epsilon_r")
        if best_eps is not None:
            lines.append(f"**Best εr:** {best_eps:.2f}")
        lines.append("")

    # Targeted subsurface
    targeted = context.get("targeted_subsurface", {})
    if targeted and targeted.get("targeted_picks"):
        lines.append("## Targeted Subsurface at Ice Locations")
        for pick in targeted["targeted_picks"]:
            loc = f"{abs(pick['ice_lat']):.1f}°{'N' if pick['ice_lat'] >= 0 else 'S'}"
            if pick.get("reflector_detected"):
                lines.append(f"- {loc}: reflector at ~{pick.get('depth_m_assumed', '?')}m (SNR={pick.get('median_snr', '?')})")
            else:
                lines.append(f"- {loc}: no reflector detected")
        lines.append("")

    report_md = "\n".join(lines)
    context["report_markdown"] = report_md
    return {"success": True, "report_markdown": report_md}


# ===========================================================================
# Register all tools
# ===========================================================================

def _register_all_tools() -> None:
    """Register all available tools in the workflow registry."""

    workflow_tools.register(
        name="resolve_region",
        executor=_tool_resolve_region,
        description="Look up a named Mars region (e.g., 'Arcadia Planitia', 'Jezero') and return its bounding box. Also accepts 'lat, lon' coordinates.",
        params_schema={
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Region name or 'lat, lon' coordinates"}},
            "required": ["query"],
        },
    )

    workflow_tools.register(
        name="search_products",
        executor=_tool_search_products,
        description="Search for instrument products (SHARAD_HIGHRES, CRISM, HIRISE_DTM, CTX, etc.) in a region. Uses the resolved region bbox from context.",
        params_schema={
            "type": "object",
            "properties": {
                "instrument": {"type": "string", "enum": ["SHARAD_HIGHRES", "CRISM", "CRISM_TRR3", "HIRISE_DTM", "HIRISE", "CTX"]},
                "bbox": {"type": "object", "description": "Optional override bbox {min_lat, max_lat, min_lon, max_lon}"},
            },
            "required": ["instrument"],
        },
    )

    workflow_tools.register(
        name="check_local_data",
        executor=_tool_check_local_data,
        description="Check which products are already downloaded locally. Reports available/missing counts.",
        params_schema={"type": "object", "properties": {}},
    )

    workflow_tools.register(
        name="download_products",
        executor=_tool_download_products,
        description="Download missing product files for analysis. May download SHARAD radargrams, CRISM cubes, HiRISE DTMs, etc.",
        params_schema={
            "type": "object",
            "properties": {
                "instruments": {"type": "array", "items": {"type": "string"}, "description": "Filter downloads to specific instruments"},
            },
        },
        requires_approval=True,
        approval_template="Download data files for analysis?",
    )

    workflow_tools.register(
        name="analyze_slope",
        executor=_tool_analyze_slope,
        description="Analyze terrain slope using MOLA/HRSC DEM on a 15x15 grid. Reports mean slope, safety rating, favorable zones.",
        params_schema={"type": "object", "properties": {}},
    )

    workflow_tools.register(
        name="analyze_subsurface",
        executor=_tool_analyze_subsurface,
        description="Scan SHARAD radargrams for subsurface reflectors. Reports detection confidence, SNR, and two-way travel time.",
        params_schema={"type": "object", "properties": {}},
    )

    workflow_tools.register(
        name="analyze_minerals",
        executor=_tool_analyze_minerals,
        description="Analyze CRISM observations for ice/hydration mineral signatures using band parameters (BD1500, BD1900). Reports ice hotspot locations.",
        params_schema={"type": "object", "properties": {}},
    )

    workflow_tools.register(
        name="classify_minerals_cnn",
        executor=_tool_classify_minerals_cnn,
        description="Run 1D CNN-Attention mineral classification on CRISM TRR3 data. Identifies H2O ice, phyllosilicates, sulfates, etc.",
        params_schema={"type": "object", "properties": {}},
    )

    workflow_tools.register(
        name="find_sharad_intersections",
        executor=_tool_find_sharad_intersections,
        description="Find which SHARAD tracks geometrically intersect CRISM/HiRISE/DTM footprints using Liang-Barsky line clipping.",
        params_schema={"type": "object", "properties": {}},
    )

    workflow_tools.register(
        name="targeted_subsurface_at_ice",
        executor=_tool_targeted_subsurface_at_ice,
        description="Check SHARAD subsurface reflectors at exact locations where CRISM/CNN detected surface ice. Must run after analyze_minerals.",
        params_schema={"type": "object", "properties": {}},
    )

    workflow_tools.register(
        name="run_sharad_inversion",
        executor=_tool_run_sharad_inversion,
        description="Run physics-based SHARAD dielectric inversion using DTM depth constraints. Returns measured epsilon_r (not assumed).",
        params_schema={"type": "object", "properties": {}},
    )

    workflow_tools.register(
        name="detect_terraces",
        executor=_tool_detect_terraces,
        description="Detect concentric terraces in a crater using HiRISE DTM radial elevation profiles. Returns terrace depths and confidence.",
        params_schema={
            "type": "object",
            "properties": {
                "dtm_path": {"type": "string"},
                "lat": {"type": "number"},
                "lon": {"type": "number"},
                "n_azimuths": {"type": "integer", "default": 36},
                "max_radius_m": {"type": "number", "default": 3000},
            },
            "required": ["dtm_path"],
        },
    )

    workflow_tools.register(
        name="epsilon_inversion",
        executor=_tool_epsilon_inversion,
        description="Calculate dielectric constant (epsilon_r) using terraced crater depth + SHARAD two-way travel time. No assumed epsilon.",
        params_schema={
            "type": "object",
            "properties": {
                "sharad_product_id": {"type": "string"},
            },
        },
    )

    workflow_tools.register(
        name="evaluate_ice_evidence",
        executor=_tool_evaluate_ice_evidence,
        description="Synthesize multi-criteria ice evidence from SHARAD reflectors, CRISM minerals, and co-located detections into an ice probability score.",
        params_schema={"type": "object", "properties": {}},
    )

    workflow_tools.register(
        name="generate_report",
        executor=_tool_generate_report,
        description="Generate a markdown summary report from all workflow results.",
        params_schema={
            "type": "object",
            "properties": {
                "title": {"type": "string", "default": "MarsLab Workflow Report"},
            },
        },
    )


# Auto-register on import
_register_all_tools()
