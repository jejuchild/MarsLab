"""Waypoint generation from planned paths.

Converts raw grid-cell paths from the planner into rover-executable
waypoint sequences with heading, distance, elevation, slope, and
terrain metadata at each point.

Mimics the 10m-segment waypoint approach used by NASA JPL for the
Perseverance AI-planned drives (Dec 2025).

References:
    [1] Anthropic, "Claude AI Powers NASA's First AI-Planned Mars Rover Drive"
    [2] NASA JPL, "Perseverance Rover Completes First AI-Planned Drive," 2026
"""

import logging
import math
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .mars_constants import (
    bearing_deg,
    haversine_mars,
    MARS_GRAVITY_M_S2,
)
from .rover_models import RoverModel, PERSEVERANCE

logger = logging.getLogger(__name__)


# ── Data Classes ───────────────────────────────────────────────

@dataclass
class Waypoint:
    """A single waypoint in a rover traverse plan."""

    id: int
    lat: float
    lon: float
    elevation_m: float
    heading_deg: float                # bearing to next waypoint (0=N, 90=E)
    slope_deg: float                  # local terrain slope
    segment_distance_m: float         # distance from previous waypoint
    cumulative_distance_m: float      # total distance from start
    terrain_cost: float               # cost map value at this point
    estimated_time_s: float           # cumulative estimated drive time
    science_interest: float = 0.0     # 0-1, science value (future use)
    terrain_type: str = "unknown"     # terrain classification label
    is_stop: bool = False             # whether rover should stop here
    notes: str = ""                   # human-readable annotation


@dataclass
class WaypointSequence:
    """Complete waypoint sequence for a rover traverse."""

    waypoints: List[Waypoint]
    rover_name: str
    total_distance_m: float
    total_time_hours: float
    total_elevation_gain_m: float
    total_elevation_loss_m: float
    max_slope_deg: float
    mean_slope_deg: float
    n_waypoints: int
    waypoint_spacing_m: float
    generation_time_ms: float

    # Elevation profile along the route
    elevation_profile: List[float] = field(default_factory=list)
    slope_profile: List[float] = field(default_factory=list)
    distance_profile: List[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Serialize for JSON API response."""
        return {
            "waypoints": [
                {
                    "id": w.id,
                    "lat": round(w.lat, 6),
                    "lon": round(w.lon, 6),
                    "elevation_m": round(w.elevation_m, 1),
                    "heading_deg": round(w.heading_deg, 1),
                    "slope_deg": round(w.slope_deg, 2),
                    "segment_distance_m": round(w.segment_distance_m, 1),
                    "cumulative_distance_m": round(w.cumulative_distance_m, 1),
                    "terrain_cost": round(w.terrain_cost, 3),
                    "estimated_time_s": round(w.estimated_time_s, 0),
                    "science_interest": round(w.science_interest, 2),
                    "terrain_type": w.terrain_type,
                    "is_stop": w.is_stop,
                    "notes": w.notes,
                }
                for w in self.waypoints
            ],
            "summary": {
                "rover": self.rover_name,
                "total_distance_m": round(self.total_distance_m, 1),
                "total_time_hours": round(self.total_time_hours, 2),
                "total_elevation_gain_m": round(self.total_elevation_gain_m, 1),
                "total_elevation_loss_m": round(self.total_elevation_loss_m, 1),
                "max_slope_deg": round(self.max_slope_deg, 2),
                "mean_slope_deg": round(self.mean_slope_deg, 2),
                "n_waypoints": self.n_waypoints,
                "waypoint_spacing_m": self.waypoint_spacing_m,
                "generation_time_ms": round(self.generation_time_ms, 1),
            },
            "profiles": {
                "elevation": [round(e, 1) for e in self.elevation_profile],
                "slope": [round(s, 2) for s in self.slope_profile],
                "distance": [round(d, 1) for d in self.distance_profile],
            },
        }


# ── Waypoint Generation ───────────────────────────────────────

def generate_waypoints(
    path_geo: List[Tuple[float, float]],
    elevation_profile: List[float],
    slope_profile: List[float],
    cost_profile: List[float],
    rover: RoverModel = PERSEVERANCE,
    target_spacing_m: float = 10.0,
) -> WaypointSequence:
    """Generate a waypoint sequence from a geo-referenced path.

    Takes the raw path (dense lat/lon points from planner) and resamples
    it into evenly-spaced waypoints at the target spacing distance.

    This mimics the 10m-segment approach used by NASA JPL for
    Perseverance's AI-planned drives.

    Args:
        path_geo: List of (lat, lon) tuples along the planned path
        elevation_profile: Elevation (m) at each path point
        slope_profile: Slope (deg) at each path point
        cost_profile: Cost map value at each path point
        rover: Rover model for speed/constraint calculations
        target_spacing_m: Distance between waypoints (default 10m, per NASA)

    Returns:
        WaypointSequence with evenly-spaced waypoints
    """
    t0 = time.perf_counter()

    if len(path_geo) < 2:
        return _empty_sequence(rover, target_spacing_m, 0.0)

    n_raw = len(path_geo)

    # ── Step 1: Compute cumulative distances along the raw path ──
    cum_dist = [0.0]
    for i in range(1, n_raw):
        d = haversine_mars(
            path_geo[i - 1][0], path_geo[i - 1][1],
            path_geo[i][0], path_geo[i][1],
        )
        cum_dist.append(cum_dist[-1] + d)

    total_distance = cum_dist[-1]
    if total_distance < 0.1:
        return _empty_sequence(rover, target_spacing_m, 0.0)

    # ── Step 2: Resample at target_spacing_m intervals ───────────
    # Use linear interpolation along the cumulative distance
    n_waypoints = max(2, int(total_distance / target_spacing_m) + 1)
    wp_distances = np.linspace(0, total_distance, n_waypoints)

    cum_dist_arr = np.array(cum_dist)
    lats_raw = np.array([p[0] for p in path_geo])
    lons_raw = np.array([p[1] for p in path_geo])
    elev_raw = np.array(elevation_profile[:n_raw])
    slope_raw = np.array(slope_profile[:n_raw])
    cost_raw = np.array(cost_profile[:n_raw])

    # Pad arrays if shorter than path (defensive)
    if len(elev_raw) < n_raw:
        elev_raw = np.pad(elev_raw, (0, n_raw - len(elev_raw)),
                          constant_values=np.nan)
    if len(slope_raw) < n_raw:
        slope_raw = np.pad(slope_raw, (0, n_raw - len(slope_raw)),
                           constant_values=0.0)
    if len(cost_raw) < n_raw:
        cost_raw = np.pad(cost_raw, (0, n_raw - len(cost_raw)),
                          constant_values=1.0)

    # Interpolate all profiles to waypoint distances
    wp_lats = np.interp(wp_distances, cum_dist_arr, lats_raw)
    wp_lons = np.interp(wp_distances, cum_dist_arr, lons_raw)
    wp_elevs = np.interp(wp_distances, cum_dist_arr, elev_raw)
    wp_slopes = np.interp(wp_distances, cum_dist_arr, slope_raw)
    wp_costs = np.interp(wp_distances, cum_dist_arr, cost_raw)

    # ── Step 3: Build waypoint objects ───────────────────────────
    waypoints: List[Waypoint] = []
    cum_time_s = 0.0
    total_gain = 0.0
    total_loss = 0.0
    max_slope = 0.0

    for i in range(n_waypoints):
        # Heading to next waypoint
        if i < n_waypoints - 1:
            heading = bearing_deg(
                wp_lats[i], wp_lons[i],
                wp_lats[i + 1], wp_lons[i + 1],
            )
        else:
            # Last waypoint: keep heading from previous segment
            heading = waypoints[-1].heading_deg if waypoints else 0.0

        # Segment distance
        if i == 0:
            seg_dist = 0.0
        else:
            seg_dist = float(wp_distances[i] - wp_distances[i - 1])

        # Elevation gain/loss tracking
        if i > 0:
            dz = float(wp_elevs[i] - wp_elevs[i - 1])
            if dz > 0:
                total_gain += dz
            else:
                total_loss += abs(dz)

        # Time estimation: adjust speed for slope
        slope_here = float(wp_slopes[i])
        max_slope = max(max_slope, slope_here)
        effective_speed = _slope_adjusted_speed(slope_here, rover)
        if seg_dist > 0 and effective_speed > 0:
            cum_time_s += seg_dist / effective_speed

        # Mark start and goal as stops
        is_stop = (i == 0 or i == n_waypoints - 1)

        wp = Waypoint(
            id=i,
            lat=float(wp_lats[i]),
            lon=float(wp_lons[i]),
            elevation_m=float(wp_elevs[i]),
            heading_deg=float(heading),
            slope_deg=slope_here,
            segment_distance_m=round(seg_dist, 2),
            cumulative_distance_m=float(wp_distances[i]),
            terrain_cost=float(wp_costs[i]),
            estimated_time_s=cum_time_s,
            is_stop=is_stop,
        )
        waypoints.append(wp)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    valid_slopes = wp_slopes[~np.isnan(wp_slopes)]
    mean_slope = float(np.mean(valid_slopes)) if len(valid_slopes) > 0 else 0.0

    seq = WaypointSequence(
        waypoints=waypoints,
        rover_name=rover.name,
        total_distance_m=total_distance,
        total_time_hours=cum_time_s / 3600.0,
        total_elevation_gain_m=total_gain,
        total_elevation_loss_m=total_loss,
        max_slope_deg=max_slope,
        mean_slope_deg=mean_slope,
        n_waypoints=n_waypoints,
        waypoint_spacing_m=target_spacing_m,
        generation_time_ms=elapsed_ms,
        elevation_profile=[float(e) for e in wp_elevs],
        slope_profile=[float(s) for s in wp_slopes],
        distance_profile=[float(d) for d in wp_distances],
    )

    logger.info(
        "Generated %d waypoints over %.0f m (%.1f hrs, spacing=%.0f m) in %.1f ms",
        n_waypoints, total_distance, cum_time_s / 3600, target_spacing_m, elapsed_ms,
    )
    return seq


# ── Helpers ────────────────────────────────────────────────────

def _slope_adjusted_speed(slope_deg: float, rover: RoverModel) -> float:
    """Compute effective speed (m/s) adjusted for slope.

    - Flat terrain: cruise_speed
    - 0-10°: linear decrease from cruise to 50% cruise
    - 10°-safe_limit: linear decrease from 50% to 20% cruise
    - Above safe_limit: 10% cruise (creep mode)
    """
    s = abs(slope_deg)
    cruise = rover.cruise_speed_m_s

    if s <= 0.5:
        return cruise
    elif s <= 10.0:
        # Linear decay to 50% at 10°
        factor = 1.0 - 0.5 * (s / 10.0)
        return cruise * factor
    elif s <= rover.safe_slope_deg:
        # Linear decay from 50% to 20%
        t = (s - 10.0) / (rover.safe_slope_deg - 10.0)
        factor = 0.5 - 0.3 * t
        return cruise * factor
    else:
        # Creep mode
        return cruise * 0.1


def _empty_sequence(
    rover: RoverModel,
    spacing: float,
    elapsed_ms: float,
) -> WaypointSequence:
    """Return an empty waypoint sequence (no path or trivial path)."""
    return WaypointSequence(
        waypoints=[],
        rover_name=rover.name,
        total_distance_m=0.0,
        total_time_hours=0.0,
        total_elevation_gain_m=0.0,
        total_elevation_loss_m=0.0,
        max_slope_deg=0.0,
        mean_slope_deg=0.0,
        n_waypoints=0,
        waypoint_spacing_m=spacing,
        generation_time_ms=elapsed_ms,
    )


def estimate_sol_plan(
    sequence: WaypointSequence,
    rover: RoverModel = PERSEVERANCE,
) -> List[dict[str, object]]:
    """Break a waypoint sequence into per-sol drive plans.

    Each sol has a limited drive window (max_drive_hours_per_sol) and
    max distance (max_drive_per_sol_m). This function segments the
    route into achievable daily drives.

    Returns:
        List of dicts, each with: sol_number, start_wp_id, end_wp_id,
        distance_m, time_hours, n_waypoints
    """
    if not sequence.waypoints:
        return []

    sol_plans = []
    sol = 1
    sol_start_idx = 0
    sol_distance = 0.0
    sol_time = 0.0

    for i, wp in enumerate(sequence.waypoints):
        if i == 0:
            continue

        seg_dist = wp.segment_distance_m
        prev_wp = sequence.waypoints[i - 1]
        seg_time = (wp.estimated_time_s - prev_wp.estimated_time_s)

        # Check if adding this segment exceeds sol limits
        if (sol_distance + seg_dist > rover.max_drive_per_sol_m
                or sol_time + seg_time > rover.max_drive_hours_per_sol * 3600):
            # Close current sol
            sol_plans.append({
                "sol_number": sol,
                "start_wp_id": sequence.waypoints[sol_start_idx].id,
                "end_wp_id": sequence.waypoints[i - 1].id,
                "distance_m": round(sol_distance, 1),
                "time_hours": round(sol_time / 3600, 2),
                "n_waypoints": i - sol_start_idx,
            })
            sol += 1
            sol_start_idx = i
            sol_distance = 0.0
            sol_time = 0.0

        sol_distance += seg_dist
        sol_time += seg_time

    # Close final sol
    if sol_start_idx < len(sequence.waypoints):
        sol_plans.append({
            "sol_number": sol,
            "start_wp_id": sequence.waypoints[sol_start_idx].id,
            "end_wp_id": sequence.waypoints[-1].id,
            "distance_m": round(sol_distance, 1),
            "time_hours": round(sol_time / 3600, 2),
            "n_waypoints": len(sequence.waypoints) - sol_start_idx,
        })

    return sol_plans
