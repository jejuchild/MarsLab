#!/usr/bin/env python3
"""
Arcadia Planitia Comprehensive Mission Planner
===============================================
Cross-analyzes MOLA, SWIM, SHARAD, CRISM, HiRISE DTM data to:
  1. Score and rank candidate landing sites
  2. Select optimal landing site with EDL constraints
  3. Design rover traverse to science waypoints
  4. Generate 3D terrain animation (MP4)
  5. Produce HTML mission report

Usage:
  cd backend && python -m scripts.arcadia_mission_planner

References:
  Golombek et al. 2021, LPSC 2420 (SpaceX Starship site selection)
  Bramson et al. 2015, GRL 42:6566 (Arcadia subsurface ice)
  Morgan et al. 2021 (SWIM v4)
  Ferguson & Stentz 2006, J. Field Robotics (Field D*)
"""

from __future__ import annotations

import base64
import io
import json
import logging
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.windows import Window
from scipy.ndimage import gaussian_filter

# ── Paths ────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
PROJECT_DIR = BACKEND_DIR.parent
OUTPUT_DIR = SCRIPT_DIR / "mission_output"

MOLA_PATH = PROJECT_DIR / "Mars_HRSC_MOLA_BlendDEM_Global_200mp_v2.tif"
SWIM_DIR = BACKEND_DIR / "data" / "swim"
HIRISE_DIR = BACKEND_DIR / "hirise_dtm_data"
SHARAD_INDEX = BACKEND_DIR / "sharad_data" / "index.geojson"
CRISM_DIR = BACKEND_DIR / "crism_data"
TES_PATH = BACKEND_DIR / "data" / "tes_thermal_inertia.npy"

# Pathfinder
sys.path.insert(0, str(BACKEND_DIR))

# ── Constants ────────────────────────────────────────────────────
MARS_RADIUS_KM = 3389.5
ARCADIA_LAT = (35.0, 55.0)
ARCADIA_LON = (-175.0, -145.0)
GRID_STEP = 0.25  # degrees

# EDL constraints (Golombek et al. 2021)
MAX_ELEV_M = -2000
IDEAL_ELEV_M = -3000
MAX_SLOPE_DEG = 15.0
IDEAL_SLOPE_DEG = 5.0
MIN_THERMAL_INERTIA = 100
MIN_SWIM_SHALLOW = 0.3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("mission_planner")


# ═══════════════════════════════════════════════════════════════
# Data Classes
# ═══════════════════════════════════════════════════════════════

@dataclass
class SiteCandidate:
    lat: float
    lon: float
    elevation_m: float = 0.0
    slope_deg: float = 0.0
    swim_0_1: Optional[float] = None
    swim_1_5: Optional[float] = None
    swim_5p: Optional[float] = None
    thermal_inertia: Optional[float] = None
    hirise_coverage: bool = False
    hirise_dtm_id: str = ""
    sharad_count: int = 0
    crism_count: int = 0
    elev_score: float = 0.0
    slope_score: float = 0.0
    ice_score: float = 0.0
    thermal_score: float = 0.0
    data_score: float = 0.0
    final_score: float = 0.0

@dataclass
class ScienceWaypoint:
    lat: float
    lon: float
    name: str
    source: str  # "SHARAD" | "CRISM" | "HiRISE_DTM"
    distance_km: float = 0.0
    product_id: str = ""


# ═══════════════════════════════════════════════════════════════
# Utility Functions
# ═══════════════════════════════════════════════════════════════

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance on Mars in km."""
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return 2 * MARS_RADIUS_KM * math.asin(math.sqrt(min(1.0, a)))


def score_linear(val: float, ideal: float, worst: float) -> float:
    """Linear score: 1.0 at ideal, 0.0 at worst, clamped [0,1]."""
    if ideal == worst:
        return 1.0 if val == ideal else 0.0
    t = (val - worst) / (ideal - worst)
    return max(0.0, min(1.0, t))


# ═══════════════════════════════════════════════════════════════
# Phase 1: Data Loading
# ═══════════════════════════════════════════════════════════════

class MarsDataLoader:
    def __init__(self):
        self.mola_ds = None
        self.swim_layers: Dict[str, np.ndarray] = {}
        self.hirise_index: List[Dict] = []
        self.sharad_tracks: List[Dict] = []
        self.crism_obs: List[Dict] = []
        self.tes_data: Optional[np.ndarray] = None

    def load_all(self):
        log.info("=" * 60)
        log.info("PHASE 1: Loading Data")
        log.info("=" * 60)
        self._load_mola()
        self._load_swim()
        self._load_hirise_index()
        self._load_sharad_index()
        self._load_crism_index()
        self._load_tes()
        log.info("Data loading complete.")

    def _load_mola(self):
        log.info("  Loading MOLA DEM...")
        try:
            self.mola_ds = rasterio.open(str(MOLA_PATH))
            log.info("    MOLA: %dx%d, bounds=%s", self.mola_ds.width, self.mola_ds.height, self.mola_ds.bounds)
        except Exception as e:
            log.error("    MOLA failed: %s", e)

    def _load_swim(self):
        log.info("  Loading SWIM ice consistency layers...")
        swim_files = {
            "0-1m": "SWIM4MIM_Ci_0_1.tif",
            "1-5m": "SWIM4MIM_Ci_1_5.tif",
            ">5m": "SWIM4MIM_Ci_5.tif",
        }
        for depth, fname in swim_files.items():
            path = SWIM_DIR / fname
            if not path.exists():
                log.warning("    SWIM %s not found: %s", depth, path)
                continue
            try:
                ds = rasterio.open(str(path))
                data = ds.read(1)
                self.swim_layers[depth] = data
                ds.close()
                log.info("    SWIM %s: %dx%d loaded", depth, data.shape[1], data.shape[0])
            except Exception as e:
                log.warning("    SWIM %s failed: %s", depth, e)

    def _load_hirise_index(self):
        log.info("  Loading HiRISE DTM index...")
        idx_path = HIRISE_DIR / "index.geojson"
        if not idx_path.exists():
            log.warning("    HiRISE index not found")
            return
        with open(idx_path) as f:
            data = json.load(f)
        for feat in data.get("features", []):
            p = feat["properties"]
            # Only keep Arcadia DTMs
            mid_lat = (p.get("south", 0) + p.get("north", 0)) / 2
            mid_lon = (p.get("west", 0) + p.get("east", 0)) / 2
            if ARCADIA_LAT[0] - 5 <= mid_lat <= ARCADIA_LAT[1] + 5:
                # Skip known corrupt file
                if "007947" in p.get("product_id", ""):
                    continue
                dtm_path = HIRISE_DIR / p.get("dtm_file", "")
                if dtm_path.exists():
                    self.hirise_index.append(p)
        log.info("    HiRISE DTMs in Arcadia: %d", len(self.hirise_index))

    def _load_sharad_index(self):
        log.info("  Loading SHARAD index...")
        if not SHARAD_INDEX.exists():
            log.warning("    SHARAD index not found")
            return
        with open(SHARAD_INDEX) as f:
            data = json.load(f)
        for feat in data.get("features", []):
            p = feat["properties"]
            g = feat.get("geometry", {})
            coords = g.get("coordinates", [])
            lats, lons = [], []
            for c in coords:
                if isinstance(c, list) and len(c) >= 2:
                    lons.append(c[0])
                    lats.append(c[1])
            if not lats:
                lats = [p.get("start_lat", 0)]
                lons = [p.get("start_lon", 0)]
            # Check Arcadia overlap
            in_arcadia = any(
                ARCADIA_LAT[0] <= lat <= ARCADIA_LAT[1] and ARCADIA_LON[0] <= lon <= ARCADIA_LON[1]
                for lat, lon in zip(lats, lons)
            )
            if in_arcadia:
                self.sharad_tracks.append({
                    "product_id": p["product_id"],
                    "lats": lats,
                    "lons": lons,
                })
        log.info("    SHARAD tracks in Arcadia: %d", len(self.sharad_tracks))

    def _load_crism_index(self):
        log.info("  Loading CRISM observations...")
        if not CRISM_DIR.exists():
            log.warning("    CRISM dir not found")
            return
        lbl_files = list(CRISM_DIR.glob("*_brcarj_mtr3.lbl"))
        for lbl in lbl_files:
            try:
                content = lbl.read_text()
                min_lat = max_lat = w_lon = e_lon = None
                for line in content.split("\n"):
                    l = line.strip()
                    if "MINIMUM_LATITUDE" in l and "=" in l:
                        min_lat = float(l.split("=")[1].strip().split()[0])
                    elif "MAXIMUM_LATITUDE" in l and "=" in l:
                        max_lat = float(l.split("=")[1].strip().split()[0])
                    elif "WESTERNMOST_LONGITUDE" in l and "=" in l:
                        w_lon = float(l.split("=")[1].strip().split()[0])
                    elif "EASTERNMOST_LONGITUDE" in l and "=" in l:
                        e_lon = float(l.split("=")[1].strip().split()[0])
                if all(v is not None for v in [min_lat, max_lat, w_lon, e_lon]):
                    mid_lat = (min_lat + max_lat) / 2
                    mid_lon = (w_lon + e_lon) / 2
                    if mid_lon > 180:
                        mid_lon -= 360
                    if ARCADIA_LAT[0] <= mid_lat <= ARCADIA_LAT[1] and ARCADIA_LON[0] <= mid_lon <= ARCADIA_LON[1]:
                        self.crism_obs.append({
                            "product_id": lbl.stem.replace("_brcarj_mtr3", ""),
                            "lat": mid_lat,
                            "lon": mid_lon,
                        })
            except Exception:
                pass
        log.info("    CRISM observations in Arcadia: %d", len(self.crism_obs))

    def _load_tes(self):
        log.info("  Loading TES thermal inertia...")
        if not TES_PATH.exists():
            log.warning("    TES data not found")
            return
        try:
            self.tes_data = np.load(str(TES_PATH))
            log.info("    TES: %s loaded", self.tes_data.shape)
        except Exception as e:
            log.warning("    TES failed: %s", e)

    # ── Sampling functions ──

    def sample_mola(self, lat: float, lon: float) -> Tuple[float, float]:
        """Sample elevation and slope from MOLA DEM. Returns (elev_m, slope_deg)."""
        if self.mola_ds is None:
            return 0.0, 0.0
        ds = self.mola_ds
        lon_n = ((lon + 180) % 360) - 180
        col = int((lon_n - ds.transform.c) / ds.transform.a)
        row = int((lat - ds.transform.f) / ds.transform.e)
        col = max(0, min(ds.width - 1, col))
        row = max(0, min(ds.height - 1, row))
        # Read 3x3 window for slope
        r0 = max(0, row - 1)
        c0 = max(0, col - 1)
        r1 = min(ds.height, row + 2)
        c1 = min(ds.width, col + 2)
        win = Window(c0, r0, c1 - c0, r1 - r0)
        patch = ds.read(1, window=win).astype(np.float32)
        elev = float(patch[min(1, row - r0), min(1, col - c0)])
        # Compute slope from gradient
        if patch.shape[0] >= 2 and patch.shape[1] >= 2:
            px_m = abs(ds.transform.a) * (math.pi / 180) * MARS_RADIUS_KM * 1000 * math.cos(math.radians(lat))
            py_m = abs(ds.transform.e) * (math.pi / 180) * MARS_RADIUS_KM * 1000
            dy, dx = np.gradient(patch, py_m, px_m)
            slope = float(np.degrees(np.arctan(np.sqrt(dx ** 2 + dy ** 2)))[min(1, row - r0), min(1, col - c0)])
        else:
            slope = 0.0
        return elev, slope

    def sample_swim(self, lat: float, lon: float, depth: str) -> Optional[float]:
        """Sample SWIM ice consistency at a point."""
        data = self.swim_layers.get(depth)
        if data is None:
            return None
        ppd_lat = data.shape[0] / 120.0  # -60 to 60
        ppd_lon = data.shape[1] / 360.0  # -180 to 180
        row = int((60.0 - lat) * ppd_lat)
        col = int((lon + 180.0) * ppd_lon)
        row = max(0, min(data.shape[0] - 1, row))
        col = max(0, min(data.shape[1] - 1, col))
        val = float(data[row, col])
        return val if val > -29 else None

    def sample_tes(self, lat: float, lon: float) -> Optional[float]:
        """Sample TES thermal inertia."""
        if self.tes_data is None:
            return None
        rows, cols = self.tes_data.shape
        ppd_lat = rows / 180.0
        ppd_lon = cols / 360.0
        row = int((90.0 - lat) * ppd_lat)
        col = int((lon + 180.0) * ppd_lon)
        row = max(0, min(rows - 1, row))
        col = max(0, min(cols - 1, col))
        val = float(self.tes_data[row, col])
        return val if val > 0 else None

    def check_hirise_coverage(self, lat: float, lon: float) -> Tuple[bool, str]:
        """Check if any HiRISE DTM covers this point."""
        for dtm in self.hirise_index:
            if (dtm.get("south", 0) <= lat <= dtm.get("north", 0) and
                    dtm.get("west", 0) <= lon <= dtm.get("east", 0)):
                return True, dtm.get("product_id", "")
        return False, ""

    def count_sharad_nearby(self, lat: float, lon: float, radius_km: float = 50) -> int:
        """Count SHARAD tracks within radius of a point."""
        count = 0
        for track in self.sharad_tracks:
            for tlat, tlon in zip(track["lats"], track["lons"]):
                if haversine_km(lat, lon, tlat, tlon) < radius_km:
                    count += 1
                    break
        return count

    def count_crism_nearby(self, lat: float, lon: float, radius_km: float = 50) -> int:
        """Count CRISM observations within radius."""
        count = 0
        for obs in self.crism_obs:
            if haversine_km(lat, lon, obs["lat"], obs["lon"]) < radius_km:
                count += 1
        return count


# ═══════════════════════════════════════════════════════════════
# Phase 2: Grid-Based Site Scoring
# ═══════════════════════════════════════════════════════════════

def score_sites(loader: MarsDataLoader) -> List[SiteCandidate]:
    log.info("=" * 60)
    log.info("PHASE 2: Grid-Based Site Scoring")
    log.info("=" * 60)

    lats = np.arange(ARCADIA_LAT[0], ARCADIA_LAT[1] + 0.01, GRID_STEP)
    lons = np.arange(ARCADIA_LON[0], ARCADIA_LON[1] + 0.01, GRID_STEP)
    total = len(lats) * len(lons)
    log.info("  Grid: %d lat × %d lon = %d points (%.2f° step)", len(lats), len(lons), total, GRID_STEP)

    candidates = []
    t0 = time.time()

    for i, lat in enumerate(lats):
        if i % 10 == 0:
            elapsed = time.time() - t0
            pct = (i * len(lons)) / total * 100
            log.info("  Progress: %.0f%% (%d/%d rows, %.1fs)", pct, i, len(lats), elapsed)

        for lon in lons:
            c = SiteCandidate(lat=lat, lon=lon)

            # Elevation & slope
            c.elevation_m, c.slope_deg = loader.sample_mola(lat, lon)

            # SWIM ice
            c.swim_0_1 = loader.sample_swim(lat, lon, "0-1m")
            c.swim_1_5 = loader.sample_swim(lat, lon, "1-5m")
            c.swim_5p = loader.sample_swim(lat, lon, ">5m")

            # TES thermal inertia
            c.thermal_inertia = loader.sample_tes(lat, lon)

            # HiRISE DTM coverage
            c.hirise_coverage, c.hirise_dtm_id = loader.check_hirise_coverage(lat, lon)

            # SHARAD & CRISM proximity
            c.sharad_count = loader.count_sharad_nearby(lat, lon, 50)
            c.crism_count = loader.count_crism_nearby(lat, lon, 50)

            # ── Scoring ──
            # Elevation: 1.0 if < -3km, decay to 0 at -1km, fail > -1km
            c.elev_score = score_linear(c.elevation_m, IDEAL_ELEV_M, -1000)

            # Slope: 1.0 if < 2°, decay to 0 at 15°
            c.slope_score = score_linear(c.slope_deg, 2.0, MAX_SLOPE_DEG)

            # Terrain gate (must pass both)
            terrain_gate = c.elev_score * c.slope_score

            # Ice score (weighted average of 3 depths)
            ice_vals = []
            if c.swim_0_1 is not None:
                ice_vals.append(("0-1m", max(0, c.swim_0_1), 0.40))
            if c.swim_1_5 is not None:
                ice_vals.append(("1-5m", max(0, c.swim_1_5), 0.35))
            if c.swim_5p is not None:
                ice_vals.append((">5m", max(0, c.swim_5p), 0.25))
            if ice_vals:
                total_w = sum(w for _, _, w in ice_vals)
                c.ice_score = sum(v * w for _, v, w in ice_vals) / total_w if total_w > 0 else 0
            else:
                c.ice_score = 0.0

            # Thermal score
            ti = c.thermal_inertia or 0
            c.thermal_score = score_linear(ti, 150, 50)

            # Data coverage bonus
            hirise_bonus = 0.2 if c.hirise_coverage else 0.0
            sharad_bonus = min(1.0, c.sharad_count / 5.0) * 0.15
            crism_bonus = min(1.0, c.crism_count / 3.0) * 0.10
            c.data_score = hirise_bonus + sharad_bonus + crism_bonus

            # Final composite
            science = c.ice_score * 0.55 + c.data_score * 0.15 + c.thermal_score * 0.30
            c.final_score = terrain_gate * science

            candidates.append(c)

    elapsed = time.time() - t0
    log.info("  Scoring complete: %d sites in %.1fs", len(candidates), elapsed)
    return candidates


# ═══════════════════════════════════════════════════════════════
# Phase 3: Select Landing Site
# ═══════════════════════════════════════════════════════════════

def select_landing_site(candidates: List[SiteCandidate]) -> Tuple[SiteCandidate, List[SiteCandidate]]:
    log.info("=" * 60)
    log.info("PHASE 3: Landing Site Selection")
    log.info("=" * 60)

    # Sort by score
    ranked = sorted(candidates, key=lambda c: c.final_score, reverse=True)

    # Apply hard constraints
    viable = []
    for c in ranked:
        if c.elevation_m > MAX_ELEV_M:
            continue
        if c.slope_deg > MAX_SLOPE_DEG:
            continue
        if c.swim_0_1 is not None and c.swim_0_1 < MIN_SWIM_SHALLOW:
            continue
        ti = c.thermal_inertia or 0
        if ti < MIN_THERMAL_INERTIA and ti > 0:
            continue
        viable.append(c)
        if len(viable) >= 10:
            break

    if not viable:
        log.warning("  No sites pass hard constraints! Using top-scored site.")
        viable = ranked[:5]

    top5 = viable[:5]
    winner = top5[0]

    log.info("  TOP 5 CANDIDATES:")
    for i, c in enumerate(top5):
        log.info(
            "    #%d: (%.2f°N, %.2f°E) score=%.3f elev=%.0fm slope=%.1f° ice=%.2f TI=%.0f %s",
            i + 1, c.lat, c.lon, c.final_score, c.elevation_m, c.slope_deg,
            c.ice_score, c.thermal_inertia or 0,
            "[HiRISE]" if c.hirise_coverage else "",
        )

    log.info("  ★ SELECTED: (%.2f°N, %.2f°E) — score=%.3f", winner.lat, winner.lon, winner.final_score)
    return winner, top5


# ═══════════════════════════════════════════════════════════════
# Phase 4: Science Waypoints
# ═══════════════════════════════════════════════════════════════

def find_science_waypoints(
    landing: SiteCandidate, loader: MarsDataLoader, max_dist_km: float = 100
) -> List[ScienceWaypoint]:
    log.info("=" * 60)
    log.info("PHASE 4: Science Waypoints")
    log.info("=" * 60)

    waypoints = []

    # Nearest SHARAD tracks
    sharad_pts = []
    for track in loader.sharad_tracks:
        for tlat, tlon in zip(track["lats"], track["lons"]):
            d = haversine_km(landing.lat, landing.lon, tlat, tlon)
            if d < max_dist_km and d > 1:
                sharad_pts.append((d, tlat, tlon, track["product_id"]))
    sharad_pts.sort()
    if sharad_pts:
        d, lat, lon, pid = sharad_pts[0]
        waypoints.append(ScienceWaypoint(
            lat=lat, lon=lon, name=f"SHARAD Track {pid[:15]}",
            source="SHARAD", distance_km=d, product_id=pid,
        ))
        log.info("  SHARAD waypoint: (%.2f, %.2f) — %s @ %.1f km", lat, lon, pid, d)

    # Nearest CRISM
    crism_pts = []
    for obs in loader.crism_obs:
        d = haversine_km(landing.lat, landing.lon, obs["lat"], obs["lon"])
        if d < max_dist_km and d > 1:
            crism_pts.append((d, obs["lat"], obs["lon"], obs["product_id"]))
    crism_pts.sort()
    if crism_pts:
        d, lat, lon, pid = crism_pts[0]
        waypoints.append(ScienceWaypoint(
            lat=lat, lon=lon, name=f"CRISM {pid[:15]}",
            source="CRISM", distance_km=d, product_id=pid,
        ))
        log.info("  CRISM waypoint: (%.2f, %.2f) — %s @ %.1f km", lat, lon, pid, d)

    # Nearest HiRISE DTM center
    hirise_pts = []
    for dtm in loader.hirise_index:
        mid_lat = (dtm["south"] + dtm["north"]) / 2
        mid_lon = (dtm["west"] + dtm["east"]) / 2
        d = haversine_km(landing.lat, landing.lon, mid_lat, mid_lon)
        if d < max_dist_km and d > 1:
            hirise_pts.append((d, mid_lat, mid_lon, dtm["product_id"]))
    hirise_pts.sort()
    if hirise_pts:
        d, lat, lon, pid = hirise_pts[0]
        waypoints.append(ScienceWaypoint(
            lat=lat, lon=lon, name=f"HiRISE DTM {pid[:20]}",
            source="HiRISE_DTM", distance_km=d, product_id=pid,
        ))
        log.info("  HiRISE waypoint: (%.2f, %.2f) — %s @ %.1f km", lat, lon, pid, d)

    if not waypoints:
        # Create a default waypoint 10km north
        waypoints.append(ScienceWaypoint(
            lat=landing.lat + 0.1, lon=landing.lon,
            name="Exploration Target", source="DEFAULT", distance_km=10,
        ))
        log.info("  No nearby science targets — using default 10km traverse")

    log.info("  Total science waypoints: %d", len(waypoints))
    return waypoints


# ═══════════════════════════════════════════════════════════════
# Phase 5: Rover Traverse
# ═══════════════════════════════════════════════════════════════

def _snap_to_traversable(cost_grid: np.ndarray, row: int, col: int, max_radius: int = 50) -> Tuple[int, int]:
    """Find nearest traversable cell via spiral search from (row, col)."""
    rows, cols = cost_grid.shape
    if 0 <= row < rows and 0 <= col < cols and np.isfinite(cost_grid[row, col]):
        return (row, col)
    for radius in range(1, max_radius + 1):
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if abs(dr) != radius and abs(dc) != radius:
                    continue  # only check perimeter
                r2, c2 = row + dr, col + dc
                if 0 <= r2 < rows and 0 <= c2 < cols and np.isfinite(cost_grid[r2, c2]):
                    return (r2, c2)
    return (row, col)  # fallback


def plan_traverse(
    landing: SiteCandidate, waypoints: List[ScienceWaypoint]
) -> Optional[Dict[str, Any]]:
    log.info("=" * 60)
    log.info("PHASE 5: Rover Traverse Planning")
    log.info("=" * 60)

    try:
        from analysis.pathfinder.cost_map import compute_cost_map_for_route
        from analysis.pathfinder.planner import plan_geo, _geo_to_grid, _grid_to_geo, MarsPathPlanner
        from analysis.pathfinder.waypoints import generate_waypoints, estimate_sol_plan
        from analysis.pathfinder.rover_models import PERSEVERANCE
    except ImportError as e:
        log.error("  Pathfinder import failed: %s", e)
        return None

    # Pick the nearest science waypoint as destination
    waypoints_sorted = sorted(waypoints, key=lambda w: w.distance_km)
    target = waypoints_sorted[0]

    # If target is too far (>30km), pick a point 15km along the bearing
    if target.distance_km > 30:
        frac = 15.0 / target.distance_km
        target_lat = landing.lat + frac * (target.lat - landing.lat)
        target_lon = landing.lon + frac * (target.lon - landing.lon)
        log.info("  Target too far (%.1f km), shortening to 15 km", target.distance_km)
    else:
        target_lat = target.lat
        target_lon = target.lon

    log.info("  Route: (%.4f, %.4f) → (%.4f, %.4f)", landing.lat, landing.lon, target_lat, target_lon)

    # Try planning — first compute cost map, then snap start/goal to traversable cells
    for attempt, (dlat, dlon) in enumerate([(0, 0), (0.01, 0.01), (-0.01, -0.01)]):
        sl = landing.lat + dlat
        slo = landing.lon + dlon
        gl = target_lat + dlat
        glo = target_lon + dlon

        try:
            cost_result = compute_cost_map_for_route(sl, slo, gl, glo, margin_km=5.0, rover=PERSEVERANCE)
            meta = cost_result.meta
            cost_grid = cost_result.cost_grid

            # Convert to grid and snap to nearest traversable cell
            sr, sc = _geo_to_grid(sl, slo, meta)
            gr, gc = _geo_to_grid(gl, glo, meta)

            sr2, sc2 = _snap_to_traversable(cost_grid, sr, sc, max_radius=80)
            gr2, gc2 = _snap_to_traversable(cost_grid, gr, gc, max_radius=80)

            if not np.isfinite(cost_grid[sr2, sc2]) or not np.isfinite(cost_grid[gr2, gc2]):
                log.info("  Attempt %d: no traversable cell near start/goal within radius", attempt + 1)
                continue

            # Convert snapped grid cells back to geo
            snap_start_lat, snap_start_lon = _grid_to_geo(sr2, sc2, meta)
            snap_goal_lat, snap_goal_lon = _grid_to_geo(gr2, gc2, meta)

            log.info("  Snapped start: (%.4f, %.4f) → (%.4f, %.4f) [shift %d cells]",
                     sl, slo, snap_start_lat, snap_start_lon, abs(sr2 - sr) + abs(sc2 - sc))
            log.info("  Snapped goal:  (%.4f, %.4f) → (%.4f, %.4f) [shift %d cells]",
                     gl, glo, snap_goal_lat, snap_goal_lon, abs(gr2 - gr) + abs(gc2 - gc))

            # Plan with snapped coordinates
            plan = plan_geo(snap_start_lat, snap_start_lon, snap_goal_lat, snap_goal_lon, cost_result, PERSEVERANCE)

            if plan.success:
                log.info("  ✓ Route found (attempt %d): %.0f m, %d cells explored", attempt + 1, plan.total_distance_m, plan.cells_explored)

                wp_seq = generate_waypoints(
                    path_geo=plan.path_geo,
                    elevation_profile=plan.elevation_profile,
                    slope_profile=plan.slope_profile,
                    cost_profile=getattr(plan, "cost_profile", [1.0] * len(plan.path_geo)),
                    rover=PERSEVERANCE,
                )
                sol_plan = estimate_sol_plan(wp_seq, PERSEVERANCE)

                return {
                    "plan": plan,
                    "cost_result": cost_result,
                    "waypoint_seq": wp_seq,
                    "sol_plan": sol_plan,
                    "start": (snap_start_lat, snap_start_lon),
                    "goal": (snap_goal_lat, snap_goal_lon),
                    "target_name": target.name,
                    "dem_source": cost_result.meta.get("dem_source", "MOLA"),
                    "dem_resolution_m": cost_result.meta.get("dem_resolution_m", 200),
                }
            else:
                log.info("  Attempt %d failed: %s", attempt + 1, plan.message)
        except Exception as e:
            log.info("  Attempt %d error: %s", attempt + 1, e)
            import traceback
            traceback.print_exc()

    # Last resort: try with MOLA (lower resolution, more likely to succeed)
    log.info("  Trying MOLA fallback (no HiRISE)...")
    try:
        from analysis.pathfinder.cost_map import compute_cost_map
        from analysis.pathfinder.mars_constants import meters_per_degree_lat, meters_per_degree_lon
        mid_lat = (landing.lat + target_lat) / 2.0
        margin_deg_lat = 5.0 / (meters_per_degree_lat(mid_lat) / 1000.0)
        margin_deg_lon = 5.0 / (meters_per_degree_lon(mid_lat) / 1000.0)
        lat_min = min(landing.lat, target_lat) - margin_deg_lat
        lat_max = max(landing.lat, target_lat) + margin_deg_lat
        lon_min = min(landing.lon, target_lon) - margin_deg_lon
        lon_max = max(landing.lon, target_lon) + margin_deg_lon

        # Force MOLA by temporarily hiding HiRISE index
        import analysis.pathfinder.cost_map as cm_mod
        orig_find = cm_mod._find_best_hirise_dtm
        cm_mod._find_best_hirise_dtm = lambda *a, **k: None
        try:
            cost_result = compute_cost_map(lat_min, lat_max, lon_min, lon_max, PERSEVERANCE)
        finally:
            cm_mod._find_best_hirise_dtm = orig_find

        meta = cost_result.meta
        cost_grid = cost_result.cost_grid
        sr, sc = _geo_to_grid(landing.lat, landing.lon, meta)
        gr, gc = _geo_to_grid(target_lat, target_lon, meta)
        sr2, sc2 = _snap_to_traversable(cost_grid, sr, sc)
        gr2, gc2 = _snap_to_traversable(cost_grid, gr, gc)
        snap_start_lat, snap_start_lon = _grid_to_geo(sr2, sc2, meta)
        snap_goal_lat, snap_goal_lon = _grid_to_geo(gr2, gc2, meta)

        plan = plan_geo(snap_start_lat, snap_start_lon, snap_goal_lat, snap_goal_lon, cost_result, PERSEVERANCE)
        if plan.success:
            log.info("  ✓ MOLA route found: %.0f m, %d cells", plan.total_distance_m, plan.cells_explored)
            wp_seq = generate_waypoints(
                path_geo=plan.path_geo,
                elevation_profile=plan.elevation_profile,
                slope_profile=plan.slope_profile,
                cost_profile=getattr(plan, "cost_profile", [1.0] * len(plan.path_geo)),
                rover=PERSEVERANCE,
            )
            sol_plan = estimate_sol_plan(wp_seq, PERSEVERANCE)
            return {
                "plan": plan,
                "cost_result": cost_result,
                "waypoint_seq": wp_seq,
                "sol_plan": sol_plan,
                "start": (snap_start_lat, snap_start_lon),
                "goal": (snap_goal_lat, snap_goal_lon),
                "target_name": target.name,
                "dem_source": "MOLA",
                "dem_resolution_m": 200,
            }
    except Exception as e:
        log.error("  MOLA fallback error: %s", e)
        import traceback
        traceback.print_exc()

    log.warning("  All route planning attempts failed.")
    return None

# ═══════════════════════════════════════════════════════════════
# Phase 6: Generate Figures
# ═══════════════════════════════════════════════════════════════

def generate_figures(
    candidates: List[SiteCandidate],
    top5: List[SiteCandidate],
    landing: SiteCandidate,
    waypoints: List[ScienceWaypoint],
    traverse: Optional[Dict],
    loader: MarsDataLoader,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    log.info("=" * 60)
    log.info("PHASE 6: Generating Figures")
    log.info("=" * 60)

    plt.style.use("dark_background")
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150})

    # Build score grid for heatmap
    lats = sorted(set(c.lat for c in candidates))
    lons = sorted(set(c.lon for c in candidates))
    lat_idx = {v: i for i, v in enumerate(lats)}
    lon_idx = {v: i for i, v in enumerate(lons)}
    grid = np.full((len(lats), len(lons)), np.nan)
    for c in candidates:
        grid[lat_idx[c.lat], lon_idx[c.lon]] = c.final_score

    # ── Fig 1: Site Selection Heatmap ──
    fig, ax = plt.subplots(figsize=(14, 8))
    extent = [ARCADIA_LON[0], ARCADIA_LON[1], ARCADIA_LAT[0], ARCADIA_LAT[1]]
    im = ax.imshow(grid, origin="lower", extent=extent, cmap="RdYlGn", aspect="auto", interpolation="bilinear")
    plt.colorbar(im, ax=ax, label="Composite Score", shrink=0.8)
    for i, c in enumerate(top5):
        marker = "★" if i == 0 else "☆"
        ax.plot(c.lon, c.lat, "*", color="cyan" if i == 0 else "white", markersize=20 if i == 0 else 12, zorder=10)
        ax.annotate(f"#{i + 1}", (c.lon, c.lat), fontsize=8, color="white", xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title("Arcadia Planitia — Landing Site Composite Score\n(MOLA + SWIM + SHARAD + CRISM + HiRISE + TES)", fontsize=13)
    ax.grid(True, alpha=0.2)
    fig.savefig(OUTPUT_DIR / "fig1_site_selection_map.png", bbox_inches="tight")
    plt.close(fig)
    log.info("  ✓ fig1_site_selection_map.png")

    # ── Fig 2: SWIM Ice Layers ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for idx, (depth, label) in enumerate([("0-1m", "0–1 m"), ("1-5m", "1–5 m"), (">5m", "> 5 m")]):
        ax = axes[idx]
        data = loader.swim_layers.get(depth)
        if data is not None:
            ppd = data.shape[0] / 120.0
            r0 = int((60 - ARCADIA_LAT[1]) * ppd)
            r1 = int((60 - ARCADIA_LAT[0]) * ppd)
            c0 = int((ARCADIA_LON[0] + 180) * data.shape[1] / 360.0)
            c1 = int((ARCADIA_LON[1] + 180) * data.shape[1] / 360.0)
            patch = data[max(0, r0):min(data.shape[0], r1), max(0, c0):min(data.shape[1], c1)].astype(float)
            patch[patch < -29] = np.nan
            im = ax.imshow(patch, origin="upper", extent=extent, cmap="Blues", vmin=-0.5, vmax=1.0, aspect="auto")
            plt.colorbar(im, ax=ax, label="Ice Consistency", shrink=0.7)
        ax.plot(landing.lon, landing.lat, "*", color="red", markersize=15, zorder=10)
        ax.set_title(f"SWIM Ice Consistency — Depth {label}")
        ax.set_xlabel("Longitude (°E)")
        if idx == 0:
            ax.set_ylabel("Latitude (°N)")
        ax.grid(True, alpha=0.2)
    fig.suptitle("Subsurface Water Ice Mapping (SWIM) — Arcadia Planitia", fontsize=14, y=1.02)
    fig.savefig(OUTPUT_DIR / "fig2_swim_ice_layers.png", bbox_inches="tight")
    plt.close(fig)
    log.info("  ✓ fig2_swim_ice_layers.png")

    # ── Fig 3: Data Coverage ──
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(ARCADIA_LON)
    ax.set_ylim(ARCADIA_LAT)
    # HiRISE DTM footprints
    for dtm in loader.hirise_index:
        s, n, w, e = dtm["south"], dtm["north"], dtm["west"], dtm["east"]
        rect = plt.Rectangle((w, s), e - w, n - s, fill=False, edgecolor="cyan", linewidth=1.5, alpha=0.7)
        ax.add_patch(rect)
    # SHARAD tracks
    for track in loader.sharad_tracks:
        ax.plot(track["lons"], track["lats"], "-", color="red", alpha=0.4, linewidth=0.8)
    # CRISM dots
    for obs in loader.crism_obs:
        ax.plot(obs["lon"], obs["lat"], "o", color="lime", markersize=4, alpha=0.6)
    ax.plot(landing.lon, landing.lat, "*", color="yellow", markersize=20, zorder=10)
    ax.legend(["HiRISE DTM", "SHARAD Track", "CRISM Obs", "Landing Site"], loc="upper right")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title("Data Coverage — HiRISE DTM / SHARAD / CRISM")
    ax.grid(True, alpha=0.2)
    fig.savefig(OUTPUT_DIR / "fig3_data_coverage.png", bbox_inches="tight")
    plt.close(fig)
    log.info("  ✓ fig3_data_coverage.png")

    # ── Fig 4: Landing Site Detail ──
    fig, ax = plt.subplots(figsize=(10, 10))
    zoom = 2.0  # degrees around site
    ax.set_xlim(landing.lon - zoom, landing.lon + zoom)
    ax.set_ylim(landing.lat - zoom, landing.lat + zoom)
    # Elevation contour from MOLA
    detail_lats = np.arange(landing.lat - zoom, landing.lat + zoom, 0.05)
    detail_lons = np.arange(landing.lon - zoom, landing.lon + zoom, 0.05)
    elev_grid = np.zeros((len(detail_lats), len(detail_lons)))
    for ri, la in enumerate(detail_lats):
        for ci, lo in enumerate(detail_lons):
            elev_grid[ri, ci], _ = loader.sample_mola(la, lo)
    X, Y = np.meshgrid(detail_lons, detail_lats)
    cs = ax.contourf(X, Y, elev_grid, levels=20, cmap="gist_earth", alpha=0.8)
    plt.colorbar(cs, ax=ax, label="Elevation (m)", shrink=0.7)
    ax.contour(X, Y, elev_grid, levels=10, colors="white", linewidths=0.3, alpha=0.5)
    # Landing site
    ax.plot(landing.lon, landing.lat, "*", color="red", markersize=20, zorder=10)
    ax.annotate("LANDING SITE", (landing.lon, landing.lat), fontsize=10, color="red",
                fontweight="bold", xytext=(10, 10), textcoords="offset points")
    # Science waypoints
    for wp in waypoints:
        color = {"SHARAD": "red", "CRISM": "lime", "HiRISE_DTM": "cyan"}.get(wp.source, "white")
        ax.plot(wp.lon, wp.lat, "D", color=color, markersize=10, zorder=9)
        ax.annotate(wp.name[:20], (wp.lon, wp.lat), fontsize=7, color=color, xytext=(5, 5), textcoords="offset points")
    # Traverse route
    if traverse:
        path_geo = traverse["plan"].path_geo
        if path_geo:
            rlats = [p[0] for p in path_geo]
            rlons = [p[1] for p in path_geo]
            ax.plot(rlons, rlats, "-", color="yellow", linewidth=2, zorder=8)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title(f"Landing Site Detail — ({landing.lat:.2f}°N, {landing.lon:.2f}°E)\nElev: {landing.elevation_m:.0f}m | Slope: {landing.slope_deg:.1f}° | SWIM: {landing.ice_score:.2f}")
    ax.grid(True, alpha=0.2)
    fig.savefig(OUTPUT_DIR / "fig4_landing_site_detail.png", bbox_inches="tight")
    plt.close(fig)
    log.info("  ✓ fig4_landing_site_detail.png")

    # ── Fig 5: Elevation Profile ──
    if traverse:
        plan = traverse["plan"]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        n = len(plan.elevation_profile)
        dist = np.linspace(0, plan.total_distance_m / 1000, n)
        ax1.fill_between(dist, plan.elevation_profile, alpha=0.3, color="steelblue")
        ax1.plot(dist, plan.elevation_profile, color="steelblue", linewidth=1.5)
        ax1.set_ylabel("Elevation (m)")
        ax1.set_title(f"Rover Traverse Elevation Profile — {plan.total_distance_m / 1000:.1f} km")
        ax1.grid(True, alpha=0.2)
        ax2.fill_between(dist, plan.slope_profile, alpha=0.3, color="coral")
        ax2.plot(dist, plan.slope_profile, color="coral", linewidth=1.5)
        ax2.axhline(y=15, color="red", linestyle="--", alpha=0.5, label="Max safe slope (15°)")
        ax2.set_ylabel("Slope (°)")
        ax2.set_xlabel("Distance (km)")
        ax2.legend()
        ax2.grid(True, alpha=0.2)
        fig.savefig(OUTPUT_DIR / "fig5_elevation_profile.png", bbox_inches="tight")
        plt.close(fig)
        log.info("  ✓ fig5_elevation_profile.png")
    else:
        log.info("  ⊘ fig5 skipped (no traverse)")

    # ── Fig 6: Sol Plan ──
    if traverse and traverse.get("sol_plan"):
        sol_plan = traverse["sol_plan"]
        fig, ax = plt.subplots(figsize=(12, 5))
        sols = [s["sol_number"] for s in sol_plan]
        dists = [s["distance_m"] / 1000 for s in sol_plan]
        bars = ax.bar(sols, dists, color="goldenrod", edgecolor="orange", alpha=0.8)
        ax.set_xlabel("Sol Number")
        ax.set_ylabel("Distance (km)")
        ax.set_title(f"Sol-by-Sol Traverse Plan — {len(sol_plan)} sols, {sum(dists):.1f} km total")
        ax.grid(True, alpha=0.2, axis="y")
        fig.savefig(OUTPUT_DIR / "fig6_traverse_plan.png", bbox_inches="tight")
        plt.close(fig)
        log.info("  ✓ fig6_traverse_plan.png")
    else:
        log.info("  ⊘ fig6 skipped (no sol plan)")


# ═══════════════════════════════════════════════════════════════
# Phase 7: 3D Animation
# ═══════════════════════════════════════════════════════════════

def generate_animation(
    landing: SiteCandidate, traverse: Optional[Dict], loader: MarsDataLoader
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    log.info("=" * 60)
    log.info("PHASE 7: 3D Animation")
    log.info("=" * 60)

    if traverse is None:
        log.warning("  No traverse data — skipping animation")
        return

    plan = traverse["plan"]
    if not plan.path_geo or len(plan.path_geo) < 2:
        log.warning("  No path geometry — skipping animation")
        return

    # Determine terrain source
    path_lats = [p[0] for p in plan.path_geo]
    path_lons = [p[1] for p in plan.path_geo]
    lat_min = min(path_lats) - 0.02
    lat_max = max(path_lats) + 0.02
    lon_min = min(path_lons) - 0.02
    lon_max = max(path_lons) + 0.02

    # Try to load HiRISE DTM for the area
    terrain_z = None
    terrain_x = None
    terrain_y = None

    # Check if HiRISE covers the route
    best_dtm = None
    for dtm in loader.hirise_index:
        if (dtm["south"] <= landing.lat <= dtm["north"] and
                dtm["west"] <= landing.lon <= dtm["east"]):
            best_dtm = dtm
            break

    if best_dtm:
        log.info("  Using HiRISE DTM: %s", best_dtm["product_id"])
        try:
            dtm_path = HIRISE_DIR / best_dtm["dtm_file"]
            ds = rasterio.open(str(dtm_path))
            # Read the full DTM and subsample
            data = ds.read(1)
            rows, cols = data.shape
            # Subsample to ~300x300
            step = max(1, max(rows, cols) // 300)
            z = data[::step, ::step].astype(np.float32)
            z[z == ds.nodata] = np.nan if ds.nodata else z[z == ds.nodata]
            # Create coordinate arrays (approximate)
            mid_lat = (best_dtm["south"] + best_dtm["north"]) / 2
            mid_lon = (best_dtm["west"] + best_dtm["east"]) / 2
            y_arr = np.linspace(best_dtm["north"], best_dtm["south"], z.shape[0])
            x_arr = np.linspace(best_dtm["west"], best_dtm["east"], z.shape[1])
            terrain_x, terrain_y = np.meshgrid(x_arr, y_arr)
            terrain_z = z
            ds.close()
        except Exception as e:
            log.warning("  HiRISE DTM read failed: %s — falling back to MOLA", e)
            best_dtm = None

    if terrain_z is None:
        log.info("  Using MOLA DEM for terrain")
        # Sample MOLA on grid
        n_pts = 200
        y_arr = np.linspace(lat_max, lat_min, n_pts)
        x_arr = np.linspace(lon_min, lon_max, n_pts)
        terrain_x, terrain_y = np.meshgrid(x_arr, y_arr)
        terrain_z = np.zeros((n_pts, n_pts), dtype=np.float32)
        for ri, la in enumerate(y_arr):
            for ci, lo in enumerate(x_arr):
                terrain_z[ri, ci], _ = loader.sample_mola(la, lo)

    # Fill NaN
    nan_mask = np.isnan(terrain_z)
    if np.any(nan_mask):
        fill = np.nanmean(terrain_z) if not np.all(nan_mask) else 0
        terrain_z = np.where(nan_mask, fill, terrain_z)

    # Smooth for visual quality
    terrain_z = gaussian_filter(terrain_z, sigma=1.0)

    # Compute path elevation on terrain
    path_z = []
    for lat, lon in plan.path_geo:
        # Find nearest terrain grid point
        ri = int((terrain_y[0, 0] - lat) / (terrain_y[0, 0] - terrain_y[-1, 0]) * (terrain_z.shape[0] - 1))
        ci = int((lon - terrain_x[0, 0]) / (terrain_x[0, -1] - terrain_x[0, 0]) * (terrain_z.shape[1] - 1))
        ri = max(0, min(terrain_z.shape[0] - 1, ri))
        ci = max(0, min(terrain_z.shape[1] - 1, ci))
        path_z.append(float(terrain_z[ri, ci]) + 5)  # 5m above surface

    # Create animation
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor("#0a0a0a")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#0a0a0a")

    # Plot terrain surface (subsample further for rendering)
    step_render = max(1, max(terrain_z.shape) // 200)
    zr = terrain_z[::step_render, ::step_render]
    xr = terrain_x[::step_render, ::step_render]
    yr = terrain_y[::step_render, ::step_render]

    surf = ax.plot_surface(xr, yr, zr, cmap="gist_earth", alpha=0.85, antialiased=False,
                           rstride=1, cstride=1, linewidth=0)

    # Plot full path
    ax.plot(path_lons, path_lats, path_z, "-", color="yellow", linewidth=2, alpha=0.7, zorder=5)

    # Rover dot (will be animated)
    rover_dot, = ax.plot([path_lons[0]], [path_lats[0]], [path_z[0]], "o",
                         color="red", markersize=8, zorder=10)

    ax.set_xlabel("Longitude (°E)", fontsize=8)
    ax.set_ylabel("Latitude (°N)", fontsize=8)
    ax.set_zlabel("Elevation (m)", fontsize=8)
    ax.set_title("Arcadia Planitia — Rover Traverse (3D)", color="white", fontsize=14)

    # Interpolate path to N frames
    n_frames = 120
    path_indices = np.linspace(0, len(path_lons) - 1, n_frames).astype(int)

    def update(frame):
        idx = path_indices[frame]
        rover_dot.set_data_3d([path_lons[idx]], [path_lats[idx]], [path_z[idx]])
        # Rotate camera slowly
        azim = 220 + frame * 0.5
        ax.view_init(elev=35, azim=azim)
        return rover_dot,

    from matplotlib.animation import FuncAnimation
    anim = FuncAnimation(fig, update, frames=n_frames, interval=100, blit=False)

    # Save as MP4
    mp4_path = OUTPUT_DIR / "rover_traverse_3d.mp4"
    try:
        import imageio_ffmpeg
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        plt.rcParams["animation.ffmpeg_path"] = ffmpeg_path
        from matplotlib.animation import FFMpegWriter
        writer = FFMpegWriter(fps=12, bitrate=2000)
        anim.save(str(mp4_path), writer=writer)
        log.info("  ✓ rover_traverse_3d.mp4 saved (%d frames)", n_frames)
    except Exception as e:
        log.warning("  MP4 save failed: %s — trying GIF fallback", e)
        try:
            gif_path = OUTPUT_DIR / "rover_traverse_3d.gif"
            anim.save(str(gif_path), writer="pillow", fps=12)
            log.info("  ✓ rover_traverse_3d.gif saved (fallback)")
        except Exception as e2:
            log.error("  Animation save failed: %s", e2)

    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
# Phase 8: HTML Report
# ═══════════════════════════════════════════════════════════════

def generate_report(
    landing: SiteCandidate,
    top5: List[SiteCandidate],
    waypoints: List[ScienceWaypoint],
    traverse: Optional[Dict],
    loader: MarsDataLoader,
):
    log.info("=" * 60)
    log.info("PHASE 8: HTML Report")
    log.info("=" * 60)

    def img_to_b64(filename: str) -> str:
        path = OUTPUT_DIR / filename
        if not path.exists():
            return ""
        data = path.read_bytes()
        return f"data:image/png;base64,{base64.b64encode(data).decode()}"

    # Build top5 table rows
    top5_rows = ""
    for i, c in enumerate(top5):
        top5_rows += f"""<tr>
            <td>{'★' if i == 0 else f'#{i + 1}'}</td>
            <td>{c.lat:.2f}°N</td><td>{c.lon:.2f}°E</td>
            <td>{c.final_score:.3f}</td><td>{c.elevation_m:.0f}</td>
            <td>{c.slope_deg:.1f}°</td><td>{c.ice_score:.2f}</td>
            <td>{c.thermal_inertia or 0:.0f}</td>
            <td>{'✓' if c.hirise_coverage else '—'}</td>
            <td>{c.sharad_count}</td><td>{c.crism_count}</td>
        </tr>"""

    # Waypoints table
    wp_rows = ""
    for wp in waypoints:
        wp_rows += f"""<tr>
            <td>{wp.name}</td><td>{wp.source}</td>
            <td>{wp.lat:.2f}°N, {wp.lon:.2f}°E</td>
            <td>{wp.distance_km:.1f} km</td>
            <td>{wp.product_id}</td>
        </tr>"""

    # Traverse stats
    if traverse:
        plan = traverse["plan"]
        sol_plan = traverse.get("sol_plan", [])
        traverse_stats = f"""
        <li><b>Total Distance:</b> {plan.total_distance_m / 1000:.1f} km ({plan.total_distance_m:.0f} m)</li>
        <li><b>DEM Source:</b> {traverse['dem_source']} ({traverse['dem_resolution_m']:.1f} m/px)</li>
        <li><b>Sols Required:</b> {len(sol_plan)}</li>
        <li><b>Waypoints Generated:</b> {traverse['waypoint_seq'].summary.get('n_waypoints', 0) if hasattr(traverse.get('waypoint_seq', {}), 'summary') else 'N/A'}</li>
        <li><b>Destination:</b> {traverse['target_name']}</li>
        """
    else:
        traverse_stats = "<li>Route planning was not successful for this site.</li>"

    # Video embed
    mp4_exists = (OUTPUT_DIR / "rover_traverse_3d.mp4").exists()
    gif_exists = (OUTPUT_DIR / "rover_traverse_3d.gif").exists()
    if mp4_exists:
        video_html = '<video controls width="100%" autoplay loop muted><source src="rover_traverse_3d.mp4" type="video/mp4"></video>'
    elif gif_exists:
        video_html = '<img src="rover_traverse_3d.gif" style="width:100%">'
    else:
        video_html = '<p style="color:#888">Animation not available.</p>'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Arcadia Planitia Mission Plan</title>
<style>
  body {{ background: #0a0f18; color: #d0d8e8; font-family: 'Segoe UI', system-ui, sans-serif; margin: 0; padding: 20px; line-height: 1.6; }}
  .container {{ max-width: 1200px; margin: 0 auto; }}
  h1 {{ color: #f0a050; border-bottom: 2px solid #f0a050; padding-bottom: 10px; }}
  h2 {{ color: #80b0e0; margin-top: 40px; }}
  h3 {{ color: #90c090; }}
  table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
  th, td {{ border: 1px solid #2a3a5a; padding: 8px 12px; text-align: left; }}
  th {{ background: #1a2a4a; color: #80b0e0; }}
  tr:nth-child(even) {{ background: #0f1928; }}
  img {{ max-width: 100%; border-radius: 8px; margin: 10px 0; border: 1px solid #2a3a5a; }}
  .stat-box {{ background: #0f1928; border: 1px solid #2a3a5a; border-radius: 8px; padding: 15px; margin: 10px 0; }}
  .hero {{ background: linear-gradient(135deg, #1a0a0a, #0a1a2a); padding: 30px; border-radius: 12px; margin-bottom: 30px; border: 1px solid #3a2a1a; }}
  .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  .highlight {{ color: #f0c050; font-weight: bold; }}
  video {{ border-radius: 8px; border: 1px solid #2a3a5a; }}
</style>
</head>
<body>
<div class="container">

<div class="hero">
<h1>🔴 Arcadia Planitia Mission Plan</h1>
<p style="font-size:1.2em">Comprehensive cross-data analysis for rover landing site selection and traverse planning</p>
<div class="grid-2">
<div class="stat-box">
  <h3>Selected Landing Site</h3>
  <p class="highlight" style="font-size:1.5em">{landing.lat:.2f}°N, {landing.lon:.2f}°E</p>
  <ul>
    <li>Elevation: {landing.elevation_m:.0f} m (MOLA)</li>
    <li>Slope: {landing.slope_deg:.1f}°</li>
    <li>SWIM Ice Score: {landing.ice_score:.2f}</li>
    <li>Composite Score: {landing.final_score:.3f}</li>
  </ul>
</div>
<div class="stat-box">
  <h3>Data Sources Analyzed</h3>
  <ul>
    <li>MOLA DEM: Global 200 m/px</li>
    <li>SWIM: 3 depth layers (0–1m, 1–5m, >5m)</li>
    <li>SHARAD: {len(loader.sharad_tracks)} radar tracks</li>
    <li>CRISM: {len(loader.crism_obs)} spectral observations</li>
    <li>HiRISE DTM: {len(loader.hirise_index)} high-res DEMs</li>
    <li>TES: Thermal inertia (global)</li>
  </ul>
</div>
</div>
</div>

<h2>1. Site Selection Analysis</h2>
<p>Grid-based scoring at {GRID_STEP}° resolution across Arcadia Planitia ({ARCADIA_LAT[0]}–{ARCADIA_LAT[1]}°N, {ARCADIA_LON[0]}–{ARCADIA_LON[1]}°E). Each point scored on terrain safety (elevation, slope), subsurface ice potential (SWIM 3-layer), thermal inertia (TES), and data coverage (SHARAD, CRISM, HiRISE).</p>
<img src="{img_to_b64('fig1_site_selection_map.png')}" alt="Site Selection Map">

<h2>2. Subsurface Water Ice (SWIM)</h2>
<p>SWIM ice consistency maps (Morgan et al. 2021) at three depth ranges. Values range from −1 (no ice) to +1 (strong ice evidence). Arcadia shows consistently positive signals at 1–5m depth, indicating accessible subsurface ice for ISRU.</p>
<img src="{img_to_b64('fig2_swim_ice_layers.png')}" alt="SWIM Ice Layers">

<h2>3. Multi-Instrument Data Coverage</h2>
<p>Spatial coverage of orbital instruments across Arcadia. HiRISE DTMs (cyan rectangles) provide ~1 m/px terrain data. SHARAD tracks (red) show radar sounding lines. CRISM observations (green) provide mineral/spectral data.</p>
<img src="{img_to_b64('fig3_data_coverage.png')}" alt="Data Coverage">

<h2>4. Top 5 Landing Site Candidates</h2>
<table>
<tr><th>Rank</th><th>Lat</th><th>Lon</th><th>Score</th><th>Elev (m)</th><th>Slope</th><th>Ice</th><th>TI</th><th>HiRISE</th><th>SHARAD</th><th>CRISM</th></tr>
{top5_rows}
</table>

<h2>5. Landing Site Detail</h2>
<img src="{img_to_b64('fig4_landing_site_detail.png')}" alt="Landing Site Detail">

<h2>6. Science Waypoints</h2>
<table>
<tr><th>Name</th><th>Source</th><th>Coordinates</th><th>Distance from Landing</th><th>Product ID</th></tr>
{wp_rows}
</table>

<h2>7. Rover Traverse</h2>
<div class="stat-box">
<h3>Traverse Statistics</h3>
<ul>{traverse_stats}</ul>
</div>
<img src="{img_to_b64('fig5_elevation_profile.png')}" alt="Elevation Profile">
<img src="{img_to_b64('fig6_traverse_plan.png')}" alt="Sol Plan">

<h2>8. 3D Terrain Animation</h2>
{video_html}

<h2>Appendix: Scoring Methodology</h2>
<div class="stat-box">
<h3>Hard Constraints (EDL — Golombek et al. 2021)</h3>
<ul>
  <li>MOLA elevation: < {MAX_ELEV_M} m</li>
  <li>Slope (100m scale): < {MAX_SLOPE_DEG}° (ideal < {IDEAL_SLOPE_DEG}°)</li>
  <li>SWIM shallow ice: > {MIN_SWIM_SHALLOW}</li>
  <li>Thermal inertia: > {MIN_THERMAL_INERTIA} TIU</li>
</ul>
<h3>Composite Score Formula</h3>
<pre>
terrain_gate = elevation_score × slope_score  (0 if either fails)
ice_score = 0.40 × SWIM(0-1m) + 0.35 × SWIM(1-5m) + 0.25 × SWIM(>5m)
data_bonus = 0.20 × HiRISE + 0.15 × SHARAD_proximity + 0.10 × CRISM_proximity
science = 0.55 × ice_score + 0.15 × data_bonus + 0.30 × thermal_score
final = terrain_gate × science
</pre>
</div>

<p style="color:#666; margin-top:40px; font-size:0.85em">
Generated by MarsLab Arcadia Mission Planner | References: Golombek et al. 2021 (LPSC 2420),
Bramson et al. 2015 (GRL 42:6566), Morgan et al. 2021 (SWIM v4), Ferguson & Stentz 2006 (Field D*)
</p>

</div>
</body>
</html>"""

    report_path = OUTPUT_DIR / "arcadia_mission_report.html"
    report_path.write_text(html)
    log.info("  ✓ arcadia_mission_report.html saved")


# ═══════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    log.info("╔══════════════════════════════════════════════════════════╗")
    log.info("║  ARCADIA PLANITIA MISSION PLANNER                      ║")
    log.info("║  Cross-Data Landing Site Selection & Traverse Planning  ║")
    log.info("╚══════════════════════════════════════════════════════════╝")

    # Phase 1: Load data
    loader = MarsDataLoader()
    loader.load_all()

    # Phase 2: Score sites
    candidates = score_sites(loader)

    # Phase 3: Select landing site
    landing, top5 = select_landing_site(candidates)

    # Phase 4: Science waypoints
    waypoints = find_science_waypoints(landing, loader)

    # Phase 5: Rover traverse
    traverse = plan_traverse(landing, waypoints)

    # Phase 6: Figures
    generate_figures(candidates, top5, landing, waypoints, traverse, loader)

    # Phase 7: Animation
    generate_animation(landing, traverse, loader)

    # Phase 8: Report
    generate_report(landing, top5, waypoints, traverse, loader)

    elapsed = time.time() - t0
    log.info("=" * 60)
    log.info("MISSION PLANNING COMPLETE — %.1f seconds", elapsed)
    log.info("Output: %s", OUTPUT_DIR)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
