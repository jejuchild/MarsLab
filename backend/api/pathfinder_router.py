"""API router for Mars Rover Pathfinder — AI-powered route planning.

Endpoints:
  POST /api/pathfinder/plan        — Plan optimal route between two points
  GET  /api/pathfinder/cost-tile   — Cost map heatmap tile for map overlay
  POST /api/pathfinder/analyze     — Analyze a single route segment
  GET  /api/pathfinder/rovers      — List available rover profiles
  GET  /api/pathfinder/status      — Check DEM / system readiness

Inspired by NASA JPL's Perseverance AI-planned drive system (Dec 2025).

References:
    [1] NASA JPL, "Perseverance Rover Completes First AI-Planned Drive," 2026
    [2] Ferguson & Stentz, "Field D*," J. Field Robotics, 2006
    [3] Anthropic, "Claude AI Powers NASA's First AI-Planned Mars Rover Drive"
"""

from __future__ import annotations

import json
import logging
import time
import asyncio
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Body
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/pathfinder", tags=["Pathfinder"])

# Lazy-loaded modules (heavy imports deferred)
_cost_map_mod = None
_planner_mod = None
_waypoints_mod = None


def _get_cost_map():
    global _cost_map_mod
    if _cost_map_mod is None:
        from analysis.pathfinder import cost_map as cm
        _cost_map_mod = cm
    return _cost_map_mod


def _get_planner():
    global _planner_mod
    if _planner_mod is None:
        from analysis.pathfinder import planner as pl
        _planner_mod = pl
    return _planner_mod


def _get_waypoints():
    global _waypoints_mod
    if _waypoints_mod is None:
        from analysis.pathfinder import waypoints as wp
        _waypoints_mod = wp
    return _waypoints_mod


# ── Request / Response Models ──────────────────────────────────

class PlanRequest(BaseModel):
    """Route planning request."""
    start_lat: float = Field(..., ge=-90, le=90, description="Start latitude")
    start_lon: float = Field(..., ge=-180, le=360, description="Start longitude")
    goal_lat: float = Field(..., ge=-90, le=90, description="Goal latitude")
    goal_lon: float = Field(..., ge=-180, le=360, description="Goal longitude")
    rover_type: str = Field("perseverance", description="Rover profile: perseverance, curiosity, generic_small")
    margin_km: float = Field(5.0, ge=1.0, le=50.0, description="Bounding box margin around route (km)")
    waypoint_spacing_m: float = Field(10.0, ge=1.0, le=500.0, description="Waypoint spacing (m). NASA uses 10m.")
    # Cost map weights (optional overrides)
    w_slope: Optional[float] = Field(None, ge=0, le=1, description="Slope cost weight")
    w_roughness: Optional[float] = Field(None, ge=0, le=1, description="Roughness cost weight")
    w_hazard: Optional[float] = Field(None, ge=0, le=1, description="Hazard cost weight")
    w_elevation: Optional[float] = Field(None, ge=0, le=1, description="Elevation cost weight")


class SegmentAnalysisRequest(BaseModel):
    """Request for analyzing a route segment."""
    lat_a: float = Field(..., ge=-90, le=90)
    lon_a: float = Field(..., ge=-180, le=360)
    lat_b: float = Field(..., ge=-90, le=90)
    lon_b: float = Field(..., ge=-180, le=360)


# ── SSE Helpers ────────────────────────────────────────────────

def _sse(event: str, data: dict[str, object]) -> str:
    """Format a Server-Sent Event."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# ── Route Planning (SSE streaming) ─────────────────────────────

@router.post("/plan")
async def plan_route(req: PlanRequest):
    """Plan optimal route between two points on Mars.

    Returns an SSE stream with real-time progress updates, then the
    final route with waypoints, elevation profile, and statistics.
    """

    async def stream():
        t0 = time.perf_counter()

        # ── Step 0: Load modules & rover ──────────────────────
        yield _sse("progress", {"stage": "init", "message": "Initializing pathfinder...", "pct": 0})

        from analysis.pathfinder.rover_models import get_rover, RoverType
        try:
            rover = get_rover(req.rover_type)
        except ValueError:
            yield _sse("error", {"message": f"Unknown rover type: {req.rover_type}"})
            return

        # ── Step 1: Generate cost map ─────────────────────────
        yield _sse("progress", {"stage": "cost_map", "message": "Generating traversability cost map...", "pct": 10})

        cm = _get_cost_map()
        weights = {}
        if req.w_slope is not None:
            weights["slope"] = req.w_slope
        if req.w_roughness is not None:
            weights["roughness"] = req.w_roughness
        if req.w_hazard is not None:
            weights["hazard"] = req.w_hazard
        if req.w_elevation is not None:
            weights["elevation"] = req.w_elevation

        try:
            cost_result = await asyncio.to_thread(
                cm.compute_cost_map_for_route,
                req.start_lat, req.start_lon,
                req.goal_lat, req.goal_lon,
                margin_km=req.margin_km,
                rover=rover,
                weights=weights if weights else None,
            )
        except Exception as e:
            logger.exception("Cost map generation failed")
            yield _sse("error", {"message": f"Cost map failed: {e}"})
            return

        cost_time = time.perf_counter() - t0
        dem_source = cost_result.meta.get('dem_source', 'MOLA')
        dem_res = cost_result.meta.get('dem_resolution_m', 200)
        dem_pid = cost_result.meta.get('dem_product_id', '')
        dem_label = f"{dem_source} ({dem_res:.0f} m/px)"
        if dem_pid:
            dem_label += f" [{dem_pid}]"
        yield _sse("progress", {
            "stage": "cost_map_done",
            "message": f"Cost map ready ({cost_result.meta['rows']}×{cost_result.meta['cols']} grid, DEM: {dem_label})",
            "pct": 40,
            "elapsed_s": round(cost_time, 2),
            "dem_source": dem_source,
            "dem_resolution_m": dem_res,
        })

        # ── Step 2: VLM Terrain Analysis (optional) ───────────
        vlm_result = None
        try:
            from analysis.pathfinder.terrain_analyzer import (
                analyze_terrain_vlm, augment_cost_grid,
            )
            yield _sse("progress", {
                "stage": "vlm_analysis",
                "message": "Analyzing terrain with AI vision (Llama Vision)...",
                "pct": 45,
            })
            vlm_result = await analyze_terrain_vlm(
                cost_result, rover_name=rover.name, max_slope=rover.max_slope_deg,
            )
            if vlm_result:
                # Augment cost grid with VLM terrain intelligence
                augmented_cost = augment_cost_grid(cost_result, vlm_result)
                cost_result.cost_grid = augmented_cost
                yield _sse("progress", {
                    "stage": "vlm_done",
                    "message": f"AI terrain analysis complete — {len(vlm_result.zones)} zones, risk: {vlm_result.risk_level}",
                    "pct": 50,
                })
            else:
                yield _sse("progress", {
                    "stage": "vlm_skipped",
                    "message": "VLM unavailable — using slope-based analysis only",
                    "pct": 50,
                })
        except Exception as e:
            logger.warning(f"VLM terrain analysis failed (non-fatal): {e}")
            yield _sse("progress", {
                "stage": "vlm_error",
                "message": "AI terrain analysis failed — continuing with slope-based analysis",
                "pct": 50,
            })

        # ── Step 3: Plan path (Field D* / A*) ─────────────────
        yield _sse("progress", {"stage": "planning", "message": "Finding optimal path (Field D*)...", "pct": 55})

        pl = _get_planner()
        try:
            plan_result = await asyncio.to_thread(
                pl.plan_geo,
                req.start_lat, req.start_lon,
                req.goal_lat, req.goal_lon,
                cost_result, rover,
            )
        except Exception as e:
            logger.exception("Path planning failed")
            yield _sse("error", {"message": f"Planning failed: {e}"})
            return

        if not plan_result.success:
            yield _sse("error", {
                "message": plan_result.message,
                "cells_explored": plan_result.cells_explored,
            })
            return

        plan_time = time.perf_counter() - t0
        yield _sse("progress", {
            "stage": "planning_done",
            "message": f"Path found ({plan_result.total_distance_m:.0f} m, {plan_result.cells_explored} cells explored)",
            "pct": 75,
            "elapsed_s": round(plan_time, 2),
        })

        # ── Step 3: Generate waypoints ────────────────────────
        yield _sse("progress", {"stage": "waypoints", "message": "Generating waypoint sequence...", "pct": 80})

        wp_mod = _get_waypoints()
        try:
            waypoint_seq = wp_mod.generate_waypoints(
                path_geo=plan_result.path_geo,
                elevation_profile=plan_result.elevation_profile,
                slope_profile=plan_result.slope_profile,
                cost_profile=getattr(plan_result, "cost_profile", [1.0] * len(plan_result.path_geo)),
                rover=rover,
                target_spacing_m=req.waypoint_spacing_m,
            )
        except Exception as e:
            logger.exception("Waypoint generation failed")
            yield _sse("error", {"message": f"Waypoint generation failed: {e}"})
            return

        # ── Step 4: Sol planning ──────────────────────────────
        yield _sse("progress", {"stage": "sol_plan", "message": "Computing sol drive plan...", "pct": 92})
        sol_plan = wp_mod.estimate_sol_plan(waypoint_seq, rover)

        total_time = time.perf_counter() - t0

        # ── Final result ──────────────────────────────────────
        result = waypoint_seq.to_dict()
        result["sol_plan"] = sol_plan
        result["planning_meta"] = {
            "total_computation_s": round(total_time, 2),
            "cells_explored": plan_result.cells_explored,
            "path_length_cells": plan_result.path_length_cells,
            "grid_size": f"{cost_result.meta['rows']}x{cost_result.meta['cols']}",
            "rover_type": req.rover_type,
            "algorithm": "field_d_star",
            "dem_source": cost_result.meta.get('dem_source', 'MOLA'),
            "dem_resolution_m": cost_result.meta.get('dem_resolution_m', 200),
            "dem_product_id": cost_result.meta.get('dem_product_id', ''),
        }
        result["route_geo"] = [
            {"lat": round(p[0], 6), "lon": round(p[1], 6)}
            for p in plan_result.path_geo
        ]

        # Include VLM analysis if available
        if vlm_result:
            result["vlm_analysis"] = vlm_result.to_dict()
        yield _sse("progress", {"stage": "complete", "message": "Route planning complete", "pct": 100})
        yield _sse("result", result)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── Cost Map Tile ──────────────────────────────────────────────

@router.get("/cost-tile")
async def get_cost_tile(
    lat_min: float = Query(..., ge=-90, le=90),
    lat_max: float = Query(..., ge=-90, le=90),
    lon_min: float = Query(..., ge=-180, le=360),
    lon_max: float = Query(..., ge=-180, le=360),
    rover_type: str = Query("perseverance"),
    size: int = Query(256, ge=64, le=1024, description="Tile size in pixels"),
):
    """Generate a cost map tile for the given bounding box.

    Returns a PNG image (green=easy, yellow=moderate, red=hard, black=impassable).
    """
    from analysis.pathfinder.rover_models import get_rover

    try:
        rover = get_rover(rover_type)
    except ValueError:
        raise HTTPException(400, f"Unknown rover: {rover_type}")

    cm = _get_cost_map()
    try:
        cost_result = await asyncio.to_thread(
            cm.compute_cost_map,
            lat_min, lat_max, lon_min, lon_max,
            rover=rover,
        )
        tile_bytes = cm.render_cost_map_tile(cost_result, tile_size=size)
    except Exception as e:
        logger.exception("Cost tile generation failed")
        raise HTTPException(500, str(e))

    return Response(content=tile_bytes, media_type="image/png")


# ── Segment Analysis ──────────────────────────────────────────

@router.post("/analyze")
async def analyze_segment(req: SegmentAnalysisRequest):
    """Analyze terrain along a single route segment (A → B).

    Returns elevation profile, slope profile, hazard count, and
    traversability assessment.
    """
    from analysis.pathfinder.mars_constants import haversine_mars
    from api.terrain_router import compute_slope_stats

    distance_m = haversine_mars(req.lat_a, req.lon_a, req.lat_b, req.lon_b)

    # Sample terrain at both endpoints
    try:
        stats_a = await asyncio.to_thread(
            compute_slope_stats, req.lat_a, req.lon_a, 500,
        )
        stats_b = await asyncio.to_thread(
            compute_slope_stats, req.lat_b, req.lon_b, 500,
        )
    except Exception as e:
        logger.exception("Segment analysis failed")
        raise HTTPException(500, str(e))

    # Safety assessment
    max_slope = max(stats_a.get("max_slope", 0), stats_b.get("max_slope", 0))
    if max_slope > 15:
        safety = "UNFAVORABLE"
    elif max_slope > 5:
        safety = "MARGINAL"
    else:
        safety = "FAVORABLE"

    return JSONResponse(content={
        "distance_m": round(distance_m, 1),
        "start": {"lat": req.lat_a, "lon": req.lon_a, **stats_a},
        "goal": {"lat": req.lat_b, "lon": req.lon_b, **stats_b},
        "max_slope_deg": round(max_slope, 2),
        "safety": safety,
    })


# ── Rover Profiles ────────────────────────────────────────────

@router.get("/rovers")
async def list_rovers():
    """List available rover profiles with their specifications."""
    from analysis.pathfinder.rover_models import ROVER_PROFILES
    from dataclasses import asdict

    rovers = {}
    for rtype, model in ROVER_PROFILES.items():
        d = asdict(model)
        d["rover_type"] = rtype.value
        rovers[rtype.value] = d

    return JSONResponse(content={"rovers": rovers})


# ── Status Check ──────────────────────────────────────────────

@router.get("/status")
async def pathfinder_status():
    """Check if the pathfinder system is ready (DEM loaded, etc.)."""
    import os

    _BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _PROJECT_ROOT = os.path.dirname(_BACKEND_DIR)
    dem_path = os.path.join(
        _PROJECT_ROOT,
        "Mars_HRSC_MOLA_BlendDEM_Global_200mp_v2.tif",
    )
    dem_exists = os.path.exists(dem_path)

    return JSONResponse(content={
        "ready": dem_exists,
        "dem_available": dem_exists,
        "dem_path": dem_path if dem_exists else None,
        "algorithms": ["a_star", "field_d_star"],
        "rovers": ["perseverance", "curiosity", "generic_small"],
    })


# ── Suggested Routes ───────────────────────────────────────────

# Curated routes near Mars landing sites and geologically interesting areas.
# Each route is designed to showcase varied terrain and scientific interest.
SUGGESTED_ROUTES = [
    {
        "id": "jezero_delta",
        "name": "Jezero Crater Delta Traverse",
        "description": "Explore the ancient river delta in Jezero Crater where Perseverance searches for biosignatures. Features layered sedimentary rocks, fan deposits, and varied terrain.",
        "start": {"lat": 18.4447, "lon": 77.4508},
        "goal": {"lat": 18.4700, "lon": 77.4200},
        "tags": ["perseverance", "astrobiology", "delta", "sedimentary"],
        "difficulty": "moderate",
        "estimated_distance_km": 3.2,
        "science_interest": "high",
    },
    {
        "id": "gale_msl",
        "name": "Gale Crater — Mt. Sharp Foothills",
        "description": "Follow Curiosity's path toward the foothills of Mt. Sharp (Aeolis Mons). Terrain transitions from smooth crater floor to layered sedimentary outcrops.",
        "start": {"lat": -4.5895, "lon": 137.4417},
        "goal": {"lat": -4.6300, "lon": 137.3900},
        "tags": ["curiosity", "mt_sharp", "clay_minerals", "stratigraphy"],
        "difficulty": "moderate",
        "estimated_distance_km": 5.8,
        "science_interest": "high",
    },
    {
        "id": "elysium_smooth",
        "name": "Elysium Planitia — InSight Landing Zone",
        "description": "Flat, smooth terrain ideal for testing route planning on low-obstacle ground. Near the InSight lander site with minimal slope variation.",
        "start": {"lat": 4.5024, "lon": 135.6234},
        "goal": {"lat": 4.5500, "lon": 135.5700},
        "tags": ["insight", "flat_terrain", "easy", "testing"],
        "difficulty": "easy",
        "estimated_distance_km": 6.1,
        "science_interest": "low",
    },
    {
        "id": "valles_marineris",
        "name": "Valles Marineris Canyon Rim",
        "description": "Traverse along the rim of the solar system's largest canyon. Extreme slopes and dramatic terrain with layered geological exposures.",
        "start": {"lat": -13.9, "lon": -59.2},
        "goal": {"lat": -14.1, "lon": -59.5},
        "tags": ["canyon", "extreme_terrain", "geology", "layered_deposits"],
        "difficulty": "hard",
        "estimated_distance_km": 35.0,
        "science_interest": "very_high",
    },
    {
        "id": "olympus_mons_base",
        "name": "Olympus Mons — Shield Volcano Base",
        "description": "Explore the base of the tallest volcano in the solar system. Lava flow textures, volcanic plains, and gradual slope increase toward the caldera.",
        "start": {"lat": 17.5, "lon": -134.5},
        "goal": {"lat": 18.0, "lon": -134.0},
        "tags": ["volcano", "olympus_mons", "lava_flows", "volcanic"],
        "difficulty": "moderate",
        "estimated_distance_km": 55.0,
        "science_interest": "high",
    },
    {
        "id": "arcadia_ice_traverse",
        "name": "Arcadia Planitia — Ice-Rich Traverse (HiRISE)",
        "description": "High-resolution traverse through Arcadia Planitia's ice-rich terrain using 1m/px HiRISE DTM (DTEEC_007736). Features periglacial polygonal terrain, sublimation pits, and potential subsurface ice deposits for ISRU.",
        "start": {"lat": 54.26, "lon": -169.04},
        "goal": {"lat": 54.42, "lon": -169.07},
        "tags": ["ice", "hirise_dtm", "high_resolution", "isru", "arcadia"],
        "difficulty": "moderate",
        "estimated_distance_km": 9.5,
        "science_interest": "very_high",
    },
    {
        "id": "utopia_ice",
        "name": "Utopia Planitia — Ice-Rich Terrain",
        "description": "Cross subsurface ice-rich terrain detected by radar. Important for future ISRU (In-Situ Resource Utilization) and human exploration water supply.",
        "start": {"lat": 46.7, "lon": 110.0},
        "goal": {"lat": 46.9, "lon": 110.4},
        "tags": ["ice", "isru", "human_exploration", "water"],
        "difficulty": "easy",
        "estimated_distance_km": 22.0,
        "science_interest": "high",
    },
]


@router.get("/suggest-routes")
async def suggest_routes():
    """Return curated suggested routes near interesting Mars locations.

    These are pre-defined routes near landing sites and geologically significant
    areas, designed to showcase different terrain types and difficulty levels.
    """
    return JSONResponse(content={"routes": SUGGESTED_ROUTES})
