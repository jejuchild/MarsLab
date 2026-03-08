"""Mars rover path planning with A* and Field D* interpolation.

Implements A* search on a 2D cost grid with Field D* edge-cost
interpolation for smooth, continuous-heading paths.  Post-processes
with line-of-sight path smoothing.

References:
    [1] Ferguson & Stentz, "Using Interpolation to Improve Path Planning:
        The Field D* Algorithm," J. Field Robotics 23(2), 2006.
        DOI: 10.1002/rob.20109
    [2] Carsten et al., "Global Path Planning on Board the Mars Rovers,"
        IEEE Aerospace Conference, 2007
"""

from __future__ import annotations

import heapq
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING, List, Optional, Tuple

import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from .rover_models import RoverModel, PERSEVERANCE
from .mars_constants import haversine_mars

if TYPE_CHECKING:
    from .cost_map import CostMapResult

logger = logging.getLogger(__name__)

# ── Planning timeout ────────────────────────────────────────────
PLANNING_TIMEOUT_S = 60.0  # increased for high-res HiRISE DTM grids

# ── 8-connected neighbor offsets ────────────────────────────────
_NEIGHBORS = [
    (-1, -1), (-1, 0), (-1, 1),
    ( 0, -1),          ( 0, 1),
    ( 1, -1), ( 1, 0), ( 1, 1),
]
_SQRT2 = math.sqrt(2.0)


# ── Data Classes ────────────────────────────────────────────────

@dataclass
class PlanResult:
    """Result from grid-space path planning."""
    path: List[Tuple[int, int]]       # (row, col) tuples
    total_cost: float
    path_length_cells: int
    computation_time_ms: float
    cells_explored: int
    success: bool
    message: str


@dataclass
class GeoPlanResult:
    """Result from geo-referenced path planning."""
    path_geo: List[Tuple[float, float]]   # (lat, lon) tuples
    elevation_profile: List[float]
    slope_profile: List[float]
    cost_profile: List[float]
    total_distance_m: float
    cells_explored: int
    path_length_cells: int
    success: bool
    message: str


# ── Field D* interpolation ─────────────────────────────────────

def _field_d_star_edge_cost(
    cost_grid: np.ndarray,
    r: int, c: int,
    dr: int, dc: int,
) -> float:
    """Compute edge traversal cost using Field D* interpolation.

    Instead of using just the destination cell's cost, interpolate
    between the two cells adjacent to the edge being crossed.
    This allows paths with arbitrary exit angles, producing smoother
    trajectories than standard grid-constrained A*.

    Reference: Ferguson & Stentz 2006, Section 3.
    """
    rows, cols = cost_grid.shape
    nr, nc = r + dr, c + dc

    if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
        return np.inf

    dest_cost = cost_grid[nr, nc]
    if not np.isfinite(dest_cost):
        return np.inf

    # For cardinal moves (dr or dc is 0), basic cost
    if dr == 0 or dc == 0:
        # Cardinal: interpolate with the cell perpendicular to movement
        # For horizontal move: interpolate vertically
        # For vertical move: interpolate horizontally
        if dr == 0:
            # Moving horizontally — interpolate with row neighbors
            above = cost_grid[nr - 1, nc] if nr > 0 else dest_cost
            below = cost_grid[nr + 1, nc] if nr < rows - 1 else dest_cost
            above = above if np.isfinite(above) else dest_cost
            below = below if np.isfinite(below) else dest_cost
            interp_cost = 0.5 * dest_cost + 0.25 * above + 0.25 * below
        else:
            # Moving vertically — interpolate with col neighbors
            left = cost_grid[nr, nc - 1] if nc > 0 else dest_cost
            right = cost_grid[nr, nc + 1] if nc < cols - 1 else dest_cost
            left = left if np.isfinite(left) else dest_cost
            right = right if np.isfinite(right) else dest_cost
            interp_cost = 0.5 * dest_cost + 0.25 * left + 0.25 * right

        return max(interp_cost, 0.01)

    # Diagonal: interpolate between the two cells sharing the diagonal edge
    # The two "bridge" cells for diagonal (dr, dc) are (r+dr, c) and (r, c+dc)
    bridge_a = cost_grid[r + dr, c] if 0 <= r + dr < rows else np.inf
    bridge_b = cost_grid[r, c + dc] if 0 <= c + dc < cols else np.inf

    if not np.isfinite(bridge_a):
        bridge_a = dest_cost
    if not np.isfinite(bridge_b):
        bridge_b = dest_cost

    # Weighted interpolation: destination gets 50%, bridges share 50%
    interp_cost = 0.5 * dest_cost + 0.25 * bridge_a + 0.25 * bridge_b
    return max(interp_cost, 0.01) * _SQRT2


# ── A* with Field D* ───────────────────────────────────────────

def _a_star_field_d(
    cost_grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    timeout_s: float = PLANNING_TIMEOUT_S,
) -> PlanResult:
    """A* search with Field D* edge-cost interpolation.

    Uses 8-connected grid with interpolated edge costs for smooth paths.
    """
    t0 = time.perf_counter()
    rows, cols = cost_grid.shape
    sr, sc = start
    gr, gc = goal

    # Validate
    if not (0 <= sr < rows and 0 <= sc < cols):
        return PlanResult([], 0, 0, 0, 0, False, f"Start ({sr},{sc}) out of bounds")
    if not (0 <= gr < rows and 0 <= gc < cols):
        return PlanResult([], 0, 0, 0, 0, False, f"Goal ({gr},{gc}) out of bounds")
    if not np.isfinite(cost_grid[sr, sc]):
        return PlanResult([], 0, 0, 0, 0, False, "Start position is on impassable terrain")
    if not np.isfinite(cost_grid[gr, gc]):
        return PlanResult([], 0, 0, 0, 0, False, "Goal position is on impassable terrain")
    if sr == gr and sc == gc:
        elapsed = (time.perf_counter() - t0) * 1000
        return PlanResult([(sr, sc)], 0.0, 1, elapsed, 0, True, "Start equals goal")

    # Minimum finite cost for admissible heuristic
    finite_costs = cost_grid[np.isfinite(cost_grid)]
    min_cost = float(np.min(finite_costs)) if len(finite_costs) > 0 else 0.01

    def heuristic(r: int, c: int) -> float:
        return min_cost * math.sqrt((r - gr) ** 2 + (c - gc) ** 2)

    # Priority queue: (f_score, counter, row, col)
    counter = 0
    open_set = [(heuristic(sr, sc), counter, sr, sc)]
    g_score = {(sr, sc): 0.0}
    came_from: dict[Tuple[int, int], Tuple[int, int]] = {}
    closed = set()
    cells_explored = 0

    while open_set:
        # Timeout check
        if (time.perf_counter() - t0) > timeout_s:
            elapsed = (time.perf_counter() - t0) * 1000
            return PlanResult(
                [], 0, 0, elapsed, cells_explored, False,
                f"Planning timed out after {timeout_s:.0f}s ({cells_explored} cells explored)",
            )

        _, _, cr, cc = heapq.heappop(open_set)

        if (cr, cc) in closed:
            continue
        closed.add((cr, cc))
        cells_explored += 1

        # Goal reached
        if cr == gr and cc == gc:
            # Reconstruct path
            path = [(cr, cc)]
            while (cr, cc) in came_from:
                cr, cc = came_from[(cr, cc)]
                path.append((cr, cc))
            path.reverse()

            elapsed = (time.perf_counter() - t0) * 1000
            return PlanResult(
                path=path,
                total_cost=g_score[(gr, gc)],
                path_length_cells=len(path),
                computation_time_ms=elapsed,
                cells_explored=cells_explored,
                success=True,
                message=f"Path found: {len(path)} cells, {cells_explored} explored",
            )

        current_g = g_score[(cr, cc)]

        for dr, dc in _NEIGHBORS:
            nr, nc = cr + dr, cc + dc

            if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                continue
            if (nr, nc) in closed:
                continue

            edge_cost = _field_d_star_edge_cost(cost_grid, cr, cc, dr, dc)
            if not np.isfinite(edge_cost):
                continue

            tentative_g = current_g + edge_cost

            if tentative_g < g_score.get((nr, nc), np.inf):
                g_score[(nr, nc)] = tentative_g
                came_from[(nr, nc)] = (cr, cc)
                f_score = tentative_g + heuristic(nr, nc)
                counter += 1
                heapq.heappush(open_set, (f_score, counter, nr, nc))

    elapsed = (time.perf_counter() - t0) * 1000
    return PlanResult(
        [], 0, 0, elapsed, cells_explored, False,
        f"No path found ({cells_explored} cells explored)",
    )


# ── Path Smoothing ──────────────────────────────────────────────

def _bresenham_line(r0: int, c0: int, r1: int, c1: int) -> List[Tuple[int, int]]:
    """Bresenham's line algorithm for grid cells."""
    cells = []
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc

    while True:
        cells.append((r0, c0))
        if r0 == r1 and c0 == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r0 += sr
        if e2 < dr:
            err += dr
            c0 += sc

    return cells


def _line_of_sight(
    cost_grid: np.ndarray,
    r0: int, c0: int,
    r1: int, c1: int,
) -> bool:
    """Check if straight line between two cells crosses only traversable terrain."""
    rows, cols = cost_grid.shape
    for r, c in _bresenham_line(r0, c0, r1, c1):
        if r < 0 or r >= rows or c < 0 or c >= cols:
            return False
        if not np.isfinite(cost_grid[r, c]):
            return False
    return True


def _smooth_path(
    path: List[Tuple[int, int]],
    cost_grid: np.ndarray,
) -> List[Tuple[int, int]]:
    """Remove redundant waypoints where line-of-sight exists.

    Greedy approach: from each waypoint, skip as many intermediate
    points as possible while maintaining line-of-sight.
    """
    if len(path) <= 2:
        return path

    smoothed = [path[0]]
    i = 0

    while i < len(path) - 1:
        # Try to skip as far ahead as possible
        farthest = i + 1
        for j in range(len(path) - 1, i + 1, -1):
            if _line_of_sight(
                cost_grid,
                path[i][0], path[i][1],
                path[j][0], path[j][1],
            ):
                farthest = j
                break
        smoothed.append(path[farthest])
        i = farthest

    return smoothed


# ── Coordinate Conversions ──────────────────────────────────────

def _geo_to_grid(lat: float, lon: float, meta: dict[str, Any]) -> Tuple[int, int]:
    """Convert (lat, lon) to (row, col) in cost grid."""
    lat_range = meta["lat_max"] - meta["lat_min"]
    lon_range = meta["lon_max"] - meta["lon_min"]

    if lat_range < 1e-10 or lon_range < 1e-10:
        return (0, 0)

    row = int((meta["lat_max"] - lat) / lat_range * (meta["rows"] - 1))
    col = int((lon - meta["lon_min"]) / lon_range * (meta["cols"] - 1))

    row = max(0, min(meta["rows"] - 1, row))
    col = max(0, min(meta["cols"] - 1, col))

    return (row, col)


def _grid_to_geo(row: int, col: int, meta: dict[str, Any]) -> Tuple[float, float]:
    """Convert (row, col) in cost grid to (lat, lon)."""
    rows = meta["rows"]
    cols = meta["cols"]

    lat = meta["lat_max"] - (row / max(1, rows - 1)) * (meta["lat_max"] - meta["lat_min"])
    lon = meta["lon_min"] + (col / max(1, cols - 1)) * (meta["lon_max"] - meta["lon_min"])

    return (lat, lon)


# ── MarsPathPlanner Class ──────────────────────────────────────

class MarsPathPlanner:
    """Path planner for Mars rover navigation.

    Wraps A* with Field D* interpolation and post-processing.
    """

    def plan(
        self,
        start_rc: Tuple[int, int],
        goal_rc: Tuple[int, int],
        cost_grid: np.ndarray,
        rover: Optional[RoverModel] = None,
    ) -> PlanResult:
        """Plan path in grid coordinates."""
        result = _a_star_field_d(cost_grid, start_rc, goal_rc)

        if result.success and len(result.path) > 2:
            smoothed = _smooth_path(result.path, cost_grid)
            result = PlanResult(
                path=smoothed,
                total_cost=result.total_cost,
                path_length_cells=len(smoothed),
                computation_time_ms=result.computation_time_ms,
                cells_explored=result.cells_explored,
                success=True,
                message=f"Path found and smoothed: {len(smoothed)} cells "
                        f"(from {result.path_length_cells}), "
                        f"{result.cells_explored} explored",
            )

        return result

    def plan_geo(
        self,
        start_lat: float, start_lon: float,
        goal_lat: float, goal_lon: float,
        cost_result: CostMapResult,
        rover: Optional[RoverModel] = None,
    ) -> GeoPlanResult:
        """Plan path in geographic coordinates.

        Converts lat/lon to grid, plans, then converts back.
        Extracts elevation and slope profiles along the path.
        """
        meta = cost_result.meta

        # Convert to grid
        start_rc = _geo_to_grid(start_lat, start_lon, meta)
        goal_rc = _geo_to_grid(goal_lat, goal_lon, meta)

        logger.info(
            "Planning: (%.4f, %.4f) → (%.4f, %.4f) | grid (%d,%d) → (%d,%d) | %dx%d grid",
            start_lat, start_lon, goal_lat, goal_lon,
            start_rc[0], start_rc[1], goal_rc[0], goal_rc[1],
            meta["rows"], meta["cols"],
        )

        # Plan
        result = self.plan(start_rc, goal_rc, cost_result.cost_grid, rover)

        if not result.success:
            return GeoPlanResult(
                path_geo=[],
                elevation_profile=[],
                slope_profile=[],
                cost_profile=[],
                total_distance_m=0.0,
                cells_explored=result.cells_explored,
                path_length_cells=0,
                success=False,
                message=result.message,
            )

        # Convert path to geo coordinates and extract profiles
        path_geo = []
        elevation_profile = []
        slope_profile = []
        cost_profile = []

        for r, c in result.path:
            lat, lon = _grid_to_geo(r, c, meta)
            path_geo.append((lat, lon))

            # Sample elevation and slope from grids
            r_clamped = max(0, min(cost_result.elevation_grid.shape[0] - 1, r))
            c_clamped = max(0, min(cost_result.elevation_grid.shape[1] - 1, c))

            elevation_profile.append(float(cost_result.elevation_grid[r_clamped, c_clamped]))
            slope_profile.append(float(cost_result.slope_grid[r_clamped, c_clamped]))
            cost_val = cost_result.cost_grid[r_clamped, c_clamped]
            cost_profile.append(float(cost_val) if np.isfinite(cost_val) else 1.0)

        # Compute total distance
        total_distance = 0.0
        for i in range(1, len(path_geo)):
            total_distance += haversine_mars(
                path_geo[i - 1][0], path_geo[i - 1][1],
                path_geo[i][0], path_geo[i][1],
            )

        logger.info(
            "Plan complete: %.0f m, %d waypoints, %d cells explored in %.1f ms",
            total_distance, len(path_geo), result.cells_explored,
            result.computation_time_ms,
        )

        return GeoPlanResult(
            path_geo=path_geo,
            elevation_profile=elevation_profile,
            slope_profile=slope_profile,
            cost_profile=cost_profile,
            total_distance_m=total_distance,
            cells_explored=result.cells_explored,
            path_length_cells=len(path_geo),
            success=True,
            message=result.message,
        )


# ── Module-Level Convenience ────────────────────────────────────

_planner = MarsPathPlanner()


def plan_geo(
    start_lat: float, start_lon: float,
    goal_lat: float, goal_lon: float,
    cost_result: CostMapResult,
    rover: Optional[RoverModel] = None,
) -> GeoPlanResult:
    """Module-level convenience for plan_geo (used by pathfinder_router)."""
    return _planner.plan_geo(start_lat, start_lon, goal_lat, goal_lon, cost_result, rover)
