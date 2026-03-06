"""Mars rover specifications and physical constraint models.

Provides pre-defined rover profiles (Perseverance, Curiosity, generic)
and a constraint checker used by the path planner to validate segments.

Rover specs sourced from:
    - NASA Mars 2020 Mission: https://mars.nasa.gov/mars2020/
    - NASA MSL Mission: https://mars.nasa.gov/msl/
    - MarsLab knowledge/mars_instruments_mars2020.md
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class RoverType(str, Enum):
    """Supported rover profiles."""
    PERSEVERANCE = "perseverance"
    CURIOSITY = "curiosity"
    GENERIC_SMALL = "generic_small"
    CUSTOM = "custom"


@dataclass(frozen=True)
class RoverModel:
    """Physical specifications and operational constraints for a rover."""

    name: str
    rover_type: RoverType

    # ── Dimensions (meters) ────────────────────────────────────
    length_m: float = 3.0
    width_m: float = 2.7
    height_m: float = 2.2
    wheel_diameter_m: float = 0.526
    wheel_width_m: float = 0.165
    ground_clearance_m: float = 0.43
    wheelbase_m: float = 2.0          # distance between front and rear axle

    # ── Mass ───────────────────────────────────────────────────
    mass_kg: float = 1025.0

    # ── Mobility Limits ────────────────────────────────────────
    max_slope_deg: float = 30.0       # absolute physical limit
    safe_slope_deg: float = 15.0      # recommended operational limit
    max_step_height_m: float = 0.40   # max obstacle height traversable
    min_turn_radius_m: float = 0.0    # 0 = point turn capable
    max_speed_m_s: float = 0.042      # top speed (Perseverance: ~152 m/hr)
    cruise_speed_m_s: float = 0.030   # typical cruising speed

    # ── Traversability Factors ─────────────────────────────────
    max_roughness_tri: float = 500.0  # max TRI (terrain ruggedness index)
    sand_slip_threshold: float = 0.3  # wheel slip ratio triggering caution
    tilt_limit_deg: float = 45.0      # absolute tip-over limit

    # ── Power & Endurance ──────────────────────────────────────
    power_source: str = "RTG"         # "RTG" or "SOLAR"
    nominal_power_w: float = 110.0    # available power for driving
    energy_per_meter_wh: float = 8.0  # approximate energy cost per meter
    max_drive_per_sol_m: float = 200.0  # max distance in a single sol
    max_drive_hours_per_sol: float = 4.0  # typical drive window per sol

    # ── Waypoint Constraints ───────────────────────────────────
    max_segment_m: float = 100.0      # max distance between waypoints
    min_segment_m: float = 1.0        # min distance between waypoints
    waypoint_spacing_m: float = 10.0  # default spacing (NASA used 10m)


# ── Pre-defined Rover Profiles ─────────────────────────────────

PERSEVERANCE = RoverModel(
    name="Perseverance (Mars 2020)",
    rover_type=RoverType.PERSEVERANCE,
    length_m=3.0,
    width_m=2.7,
    height_m=2.2,
    wheel_diameter_m=0.526,
    wheel_width_m=0.165,
    ground_clearance_m=0.43,
    wheelbase_m=2.0,
    mass_kg=1025.0,
    max_slope_deg=30.0,
    safe_slope_deg=15.0,
    max_step_height_m=0.40,
    min_turn_radius_m=0.0,          # point turn capable
    max_speed_m_s=0.042,            # 152 m/hr
    cruise_speed_m_s=0.030,
    max_roughness_tri=500.0,
    sand_slip_threshold=0.3,
    tilt_limit_deg=45.0,
    power_source="RTG",
    nominal_power_w=110.0,          # MMRTG
    energy_per_meter_wh=8.0,
    max_drive_per_sol_m=200.0,
    max_drive_hours_per_sol=4.0,
    max_segment_m=100.0,
    min_segment_m=1.0,
    waypoint_spacing_m=10.0,
)

CURIOSITY = RoverModel(
    name="Curiosity (MSL)",
    rover_type=RoverType.CURIOSITY,
    length_m=3.0,
    width_m=2.7,
    height_m=2.1,
    wheel_diameter_m=0.508,
    wheel_width_m=0.406,
    ground_clearance_m=0.66,
    wheelbase_m=1.9,
    mass_kg=899.0,
    max_slope_deg=30.0,
    safe_slope_deg=15.0,
    max_step_height_m=0.38,
    min_turn_radius_m=0.0,
    max_speed_m_s=0.038,            # ~140 m/hr
    cruise_speed_m_s=0.025,
    max_roughness_tri=400.0,
    sand_slip_threshold=0.35,       # slightly more sensitive (Spirit incident)
    tilt_limit_deg=45.0,
    power_source="RTG",
    nominal_power_w=110.0,
    energy_per_meter_wh=9.0,        # slightly less efficient
    max_drive_per_sol_m=150.0,
    max_drive_hours_per_sol=3.5,
    max_segment_m=100.0,
    min_segment_m=1.0,
    waypoint_spacing_m=10.0,
)

GENERIC_SMALL = RoverModel(
    name="Generic Small Rover",
    rover_type=RoverType.GENERIC_SMALL,
    length_m=0.6,
    width_m=0.5,
    height_m=0.3,
    wheel_diameter_m=0.20,
    wheel_width_m=0.10,
    ground_clearance_m=0.15,
    wheelbase_m=0.4,
    mass_kg=30.0,
    max_slope_deg=20.0,
    safe_slope_deg=10.0,
    max_step_height_m=0.15,
    min_turn_radius_m=0.0,
    max_speed_m_s=0.10,
    cruise_speed_m_s=0.05,
    max_roughness_tri=200.0,
    sand_slip_threshold=0.4,
    tilt_limit_deg=35.0,
    power_source="SOLAR",
    nominal_power_w=30.0,
    energy_per_meter_wh=3.0,
    max_drive_per_sol_m=500.0,
    max_drive_hours_per_sol=6.0,
    max_segment_m=50.0,
    min_segment_m=0.5,
    waypoint_spacing_m=5.0,
)

# Lookup table for quick access
ROVER_PROFILES = {
    RoverType.PERSEVERANCE: PERSEVERANCE,
    RoverType.CURIOSITY: CURIOSITY,
    RoverType.GENERIC_SMALL: GENERIC_SMALL,
}


def get_rover(rover_type: RoverType | str) -> RoverModel:
    """Get a pre-defined rover model by type."""
    if isinstance(rover_type, str):
        rover_type = RoverType(rover_type)
    model = ROVER_PROFILES.get(rover_type)
    if model is None:
        raise ValueError(f"Unknown rover type: {rover_type}")
    return model
