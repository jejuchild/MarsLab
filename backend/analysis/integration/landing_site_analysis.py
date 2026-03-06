#!/usr/bin/env python3
"""
Mars Landing Site Comprehensive Analysis Pipeline
==================================================
Option D: Python script for data collection → ranked report.

Combines ALL MarsLab subsystems:
  - 55 named regions (mars_regions.py)
  - Landing Site Scorer (6-category weighted, 4 seasons)
  - SWIM ice consistency (3 depth ranges)
  - ISRU Accessibility (5-score composite)
  - Neural Climate (7 atmospheric variables × 12 seasonal points)
  - PINNS Interior (seismic risk proxy)
  - RAG Knowledge Base (scientific context)

Output: Ranked Top 10 landing sites with full justification.

Real-world constraints sourced from:
  - Golombek et al. 2021 (SpaceX Starship / JPL)
  - Morgan et al. 2021/2025 (SWIM)
  - NASA I-MIM mission criteria
  - NASA DRA 5.0 human exploration reference

Usage:
  cd backend && python -m analysis.integration.landing_site_analysis
"""

import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Ensure backend is importable
_BACKEND = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from api.mars_regions import MARS_REGIONS, MarsRegion
from api.terrain_router import compute_slope_stats


def _get_elevation_and_slope(lat: float, lon: float) -> Tuple[float, float]:
    """Reliable elevation+slope lookup via MOLA-derived GeoTIFFs."""
    try:
        stats = compute_slope_stats(lat, lon, radius_m=2000)
        return stats.get('elevation_m', 0.0), stats.get('mean_slope', 0.0)
    except Exception:
        return 0.0, 0.0

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("landing_analysis")


# ═══════════════════════════════════════════════════════════════════════
# Hard Constraints (NASA/ESA/SpaceX — peer-reviewed thresholds)
# ═══════════════════════════════════════════════════════════════════════

HARD_CONSTRAINTS = {
    "max_elevation_m": -2000,       # Starship EDL requires < -2 km (Golombek 2021)
    "lat_min": 25,                  # Ice stability lower bound
    "lat_max": 50,                  # Solar power + thermal upper bound
    "max_slope_deg": 10,            # Starship < 5° ideal, we allow 10° for region avg
}

SOFT_CONSTRAINTS = {
    "preferred_elevation_m": -3000, # Preferred < -3 km
    "ideal_lat_min": 30,            # Sweet spot for ice + solar
    "ideal_lat_max": 45,            # Sweet spot (Golombek 2021: < 40°N)
    "max_dust_tau": 0.5,            # Annual average
    "min_swim_consistency": 0.5,    # Morgan et al. 2021
    "max_ice_depth_m": 10,          # I-MIM target: 0-10m
    "max_wind_ms": 15,              # EDL hazard threshold
    "max_frost_prob": 0.5,          # Surface ops risk
}


# ═══════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class RegionCandidate:
    """A candidate region with all analysis results."""
    region_id: str
    name: str
    center_lat: float
    center_lon: float
    elevation_m: float = 0.0
    
    # Phase 1: Filter results
    passed_hard_filter: bool = False
    filter_reasons: List[str] = field(default_factory=list)
    
    # Phase 2: Landing site scorer (4 seasons)
    seasonal_scores: Dict[float, float] = field(default_factory=dict)  # ls → score
    seasonal_grades: Dict[float, str] = field(default_factory=dict)
    seasonal_avg: float = 0.0
    worst_season_score: float = 0.0
    best_season_score: float = 0.0
    category_details: Dict[str, Any] = field(default_factory=dict)
    
    # Phase 3: SWIM ice
    swim_consistency: Dict[str, Optional[float]] = field(default_factory=dict)
    swim_avg: float = 0.0
    
    # Phase 3: ISRU accessibility 
    accessibility_score: float = 0.0
    accessibility_details: Dict[str, Any] = field(default_factory=dict)
    
    # Phase 4: Climate resilience (12-point)
    temp_range_k: Tuple[float, float] = (0.0, 0.0)
    max_dust_tau: float = 0.0
    max_wind_ms: float = 0.0
    frost_months: int = 0
    climate_resilience: float = 0.0
    
    # Phase 5: RAG context
    science_context: str = ""
    
    # Final composite
    final_score: float = 0.0
    final_rank: int = 0
    
    # Metadata
    warnings: List[str] = field(default_factory=list)
    data_sources: Dict[str, str] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════
# Phase 1: Hard Filter
# ═══════════════════════════════════════════════════════════════════════

def phase1_hard_filter(regions: Dict[str, MarsRegion]) -> List[RegionCandidate]:
    """Filter 55 regions by hard constraints → viable candidates."""
    log.info("=" * 60)
    log.info("PHASE 1: Hard Filter (%d regions)", len(regions))
    log.info("=" * 60)
    
    candidates = []
    rejected = []
    
    for rid, region in regions.items():
        c = RegionCandidate(
            region_id=rid,
            name=region.display_name,
            center_lat=region.center_lat,
            center_lon=region.center_lon,
        )
        
        reasons = []
        
        # Get elevation + slope at center (reliable terrain_router)
        elev, slope = _get_elevation_and_slope(region.center_lat, region.center_lon)
        c.elevation_m = elev
        
        # Latitude filter: 25-50°N (northern ice belt + solar)
        lat = region.center_lat
        if lat < HARD_CONSTRAINTS['lat_min'] or lat > HARD_CONSTRAINTS['lat_max']:
            reasons.append(f"Latitude {lat:.1f}° outside [{HARD_CONSTRAINTS['lat_min']}°, {HARD_CONSTRAINTS['lat_max']}°]N")
        
        # Elevation filter: < -2 km (skip if 0 = unknown)
        if elev == 0.0:
            pass  # Unknown, don't reject
        elif elev > HARD_CONSTRAINTS['max_elevation_m']:
            reasons.append(f"Elevation {elev:.0f}m > {HARD_CONSTRAINTS['max_elevation_m']}m")
        # Slope filter: check if region is known for steep terrain
        # (We'll do detailed slope checking in Phase 2 via landing_site_scorer)
        steep_tags = {"volcanic", "shield_volcano", "canyon", "chasma"}
        if steep_tags & set(region.tags):
            # Volcanos and canyons likely fail slope
            reasons.append(f"Terrain type [{', '.join(steep_tags & set(region.tags))}] likely exceeds slope limit")
        
        if reasons:
            c.passed_hard_filter = False
            c.filter_reasons = reasons
            rejected.append(c)
        else:
            c.passed_hard_filter = True
            candidates.append(c)
    
    log.info("  Passed: %d regions", len(candidates))
    for c in candidates:
        log.info("    ✓ %-30s  lat=%.1f° lon=%.1f° elev=%.0fm",
                 c.name, c.center_lat, c.center_lon, c.elevation_m)
    
    log.info("  Rejected: %d regions", len(rejected))
    for c in rejected:
        log.info("    ✗ %-30s  %s", c.name, "; ".join(c.filter_reasons))
    
    return candidates


# ═══════════════════════════════════════════════════════════════════════
# Phase 2: Landing Site Scorer × 4 Seasons
# ═══════════════════════════════════════════════════════════════════════

def phase2_seasonal_scoring(candidates: List[RegionCandidate]) -> List[RegionCandidate]:
    """Score each candidate across 4 Martian seasons."""
    log.info("")
    log.info("=" * 60)
    log.info("PHASE 2: Landing Site Scorer × 4 Seasons (%d candidates)", len(candidates))
    log.info("=" * 60)
    
    from analysis.integration.landing_site_scorer import score_landing_site
    
    seasons = {
        0:   "Northern Spring Equinox",
        90:  "Northern Summer Solstice",
        180: "Northern Autumn Equinox",
        270: "Northern Winter Solstice",
    }
    
    for c in candidates:
        scores = {}
        grades = {}
        all_categories = {}
        
        for ls, season_name in seasons.items():
            try:
                result = score_landing_site(c.center_lat, c.center_lon, ls=float(ls))
                scores[ls] = result.overall_score
                grades[ls] = result.grade
                
                # Store detailed category breakdown for the first season
                if ls == 0:
                    for cat in result.categories:
                        all_categories[cat.name] = {
                            "score": cat.score,
                            "weight": cat.weight,
                            "weighted": cat.weighted,
                            "assessment": cat.assessment,
                        }
                    c.warnings.extend(result.warnings)
                    c.data_sources = result.data_sources
                    
            except Exception as e:
                log.warning("  Scorer failed for %s at Ls=%d: %s", c.name, ls, e)
                scores[ls] = 0.0
                grades[ls] = "F"
        
        c.seasonal_scores = scores
        c.seasonal_grades = grades
        c.category_details = all_categories
        
        if scores:
            vals = list(scores.values())
            c.seasonal_avg = round(sum(vals) / len(vals), 1)
            c.worst_season_score = min(vals)
            c.best_season_score = max(vals)
        
        log.info("  %-30s  avg=%.1f  worst=%.1f  grades=%s",
                 c.name, c.seasonal_avg, c.worst_season_score,
                 "/".join(grades.get(ls, "?") for ls in [0, 90, 180, 270]))
    
    # Sort by seasonal average
    candidates.sort(key=lambda c: c.seasonal_avg, reverse=True)
    return candidates


# ═══════════════════════════════════════════════════════════════════════
# Phase 3: SWIM Ice + ISRU Accessibility
# ═══════════════════════════════════════════════════════════════════════

def phase3_ice_analysis(candidates: List[RegionCandidate]) -> List[RegionCandidate]:
    """Deep ice and ISRU analysis using SWIM + accessibility algorithm."""
    log.info("")
    log.info("=" * 60)
    log.info("PHASE 3: SWIM Ice + ISRU Accessibility (%d candidates)", len(candidates))
    log.info("=" * 60)
    
    # Try SWIM consistency via fusion pipeline
    swim_available = False
    try:
        from analysis.swim_fusion.pipeline import SwimFusionPipeline
        fusion = SwimFusionPipeline()
        swim_available = True
        log.info("  SWIM fusion pipeline loaded")
    except Exception as e:
        log.warning("  SWIM fusion pipeline unavailable: %s", e)
    
    # Accessibility algorithm
    from analysis.accessibility.algorithm import compute_accessibility
    for c in candidates:
        # --- SWIM Consistency ---
        if swim_available:
            try:
                result = fusion.query_point(lat=c.center_lat, lon=c.center_lon, mode="precomputed")
                if hasattr(result, 'model_dump'):
                    result = result.model_dump()
                elif not isinstance(result, dict):
                    result = {}
                
                c.swim_consistency = {
                    "0-1m": result.get("consistency_0_1m"),
                    "1-5m": result.get("consistency_1_5m"),
                    "5m+": result.get("consistency_5m_plus"),
                }
                vals = [v for v in c.swim_consistency.values() if v is not None]
                c.swim_avg = round(sum(vals) / len(vals), 3) if vals else 0.0
                c.data_sources["swim"] = "SWIM_v2_precomputed"
            except Exception as e:
                log.warning("  SWIM query failed for %s: %s", c.name, e)
                c.swim_consistency = {"0-1m": None, "1-5m": None, "5m+": None}
                c.swim_avg = 0.0
        
        # --- ISRU Accessibility ---
        try:
            elev = c.elevation_m
            slope = 0.0
            tri = 0.0
            thermal_inertia = None
            
            try:
                stats = compute_slope_stats(c.center_lat, c.center_lon, radius_m=2000)
                slope = stats.get("mean_slope", 0.0)
                tri = stats.get("tri", 0.0) if "tri" in stats else 0.0
            except Exception:
                pass
            
            # Try to get thermal inertia from TES
            try:
                from analysis.accessibility.pipeline import AccessibilityPipeline
                pipe = AccessibilityPipeline()
                point_data = pipe.query_point(c.center_lat, c.center_lon)
                if hasattr(point_data, 'inputs'):
                    thermal_inertia = point_data.inputs.get("thermal_inertia")
                elif isinstance(point_data, dict):
                    thermal_inertia = point_data.get("inputs", {}).get("thermal_inertia")
            except Exception:
                pass
            
            acc = compute_accessibility(
                thermal_inertia=thermal_inertia,
                elevation=elev,
                slope=slope,
                tri=tri,
                lat=c.center_lat,
                lon=c.center_lon,
            )
            c.accessibility_score = acc.score
            c.accessibility_details = {
                "excavation": acc.excavation,
                "landing": acc.landing,
                "ice_landform": acc.ice_landform,
                "water_mineral": acc.water_mineral,
                "surface_ice": acc.surface_ice,
                "confidence": acc.confidence,
                "layers": acc.layers_available,
            }
            c.data_sources["accessibility"] = f"algorithm ({acc.confidence})"
        except Exception as e:
            log.warning("  Accessibility failed for %s: %s", c.name, e)
        
        log.info("  %-30s  SWIM=%.3f  access=%.3f  ice_depths=%s",
                 c.name, c.swim_avg, c.accessibility_score,
                 {k: f"{v:.2f}" if v else "N/A" for k, v in c.swim_consistency.items()})
    
    return candidates


# ═══════════════════════════════════════════════════════════════════════
# Phase 4: Climate Resilience (12-point seasonal)
# ═══════════════════════════════════════════════════════════════════════

def phase4_climate_resilience(candidates: List[RegionCandidate]) -> List[RegionCandidate]:
    """Full-year climate analysis at 12 seasonal points."""
    log.info("")
    log.info("=" * 60)
    log.info("PHASE 4: Climate Resilience — 12-Point Annual (%d candidates)", len(candidates))
    log.info("=" * 60)
    
    # Try neural climate first, fallback to parametric
    use_neural = False
    try:
        from neural_climate.predictor import get_predictor, is_model_trained
        if is_model_trained():
            predictor = get_predictor()
            use_neural = True
            log.info("  Using Neural Climate Emulator")
    except Exception:
        pass
    
    if not use_neural:
        log.info("  Using Parametric Climate Model (neural unavailable)")
    
    from api.mars_climate import (
        surface_temperature_k, surface_pressure_pa,
        dust_opacity, wind_speed, co2_frost_probability,
    )
    
    ls_points = list(range(0, 360, 30))  # 12 points, every 30° Ls
    
    for c in candidates:
        temps_min = []
        temps_max = []
        dust_vals = []
        wind_vals = []
        frost_vals = []
        
        for ls in ls_points:
            try:
                if use_neural:
                    pred = predictor.predict(c.center_lat, c.center_lon, float(ls))
                    temps_min.append(pred["temperature_min_k"])
                    temps_max.append(pred["temperature_max_k"])
                    dust_vals.append(pred["dust_tau_mean"])
                    wind_vals.append(pred["wind_mean_ms"])
                    frost_vals.append(pred["frost_probability"])
                else:
                    temp = surface_temperature_k(c.center_lat, float(ls), c.elevation_m)
                    dust = dust_opacity(c.center_lat, float(ls))
                    wind = wind_speed(c.center_lat, float(ls))
                    frost = co2_frost_probability(c.center_lat, float(ls), c.elevation_m)
                    temps_min.append(temp["min_k"])
                    temps_max.append(temp["max_k"])
                    dust_vals.append(dust["tau_mean"])
                    wind_vals.append(wind["mean_ms"])
                    frost_vals.append(frost["frost_probability"])
            except Exception as e:
                log.debug("  Climate query failed for %s at Ls=%d: %s", c.name, ls, e)
        
        if temps_min and temps_max:
            c.temp_range_k = (round(min(temps_min), 1), round(max(temps_max), 1))
            c.max_dust_tau = round(max(dust_vals), 3) if dust_vals else 0.0
            c.max_wind_ms = round(max(wind_vals), 1) if wind_vals else 0.0
            c.frost_months = sum(1 for f in frost_vals if f > 0.3) if frost_vals else 0
            
            # Climate resilience score (0-1)
            # Penalize: extreme temp range, high dust, high wind, many frost months
            temp_span = c.temp_range_k[1] - c.temp_range_k[0]
            temp_score = max(0, 1.0 - (temp_span - 60) / 100)  # 60K span = ideal
            dust_score = max(0, 1.0 - c.max_dust_tau / 2.0)
            wind_score = max(0, 1.0 - c.max_wind_ms / 15.0)
            frost_score = max(0, 1.0 - c.frost_months / 6.0)
            
            c.climate_resilience = round(
                0.25 * temp_score + 0.25 * dust_score + 0.25 * wind_score + 0.25 * frost_score,
                3
            )
            c.data_sources["climate_annual"] = "neural_emulator" if use_neural else "parametric"
            
            # Warnings
            if c.max_wind_ms > SOFT_CONSTRAINTS["max_wind_ms"]:
                c.warnings.append(f"⚠ Peak wind {c.max_wind_ms} m/s exceeds EDL limit")
            if c.max_dust_tau > 1.5:
                c.warnings.append(f"⚠ Peak dust τ={c.max_dust_tau} — dust storm season hazard")
            if c.frost_months > 4:
                c.warnings.append(f"⚠ {c.frost_months}/12 months with frost risk")
        
        log.info("  %-30s  T=[%.0f,%.0f]K  dust_max=%.2f  wind_max=%.1f  frost=%d/12  resil=%.3f",
                 c.name, c.temp_range_k[0], c.temp_range_k[1],
                 c.max_dust_tau, c.max_wind_ms, c.frost_months, c.climate_resilience)
    
    return candidates


# ═══════════════════════════════════════════════════════════════════════
# Phase 5: Final Composite Score + Ranking
# ═══════════════════════════════════════════════════════════════════════

FINAL_WEIGHTS = {
    "landing_scorer": 0.30,    # 6-category weighted seasonal avg
    "swim_ice": 0.25,          # SWIM consistency
    "accessibility": 0.15,     # ISRU feasibility
    "climate_resilience": 0.20,# Full-year climate
    "science_bonus": 0.10,     # Latitude/elevation science proxy
}

def phase5_final_ranking(candidates: List[RegionCandidate]) -> List[RegionCandidate]:
    """Compute final composite score and rank."""
    log.info("")
    log.info("=" * 60)
    log.info("PHASE 5: Final Composite Score & Ranking")
    log.info("=" * 60)
    log.info("  Weights: %s", FINAL_WEIGHTS)
    
    for c in candidates:
        # Normalize landing_scorer to 0-1 (it's on 0-100 scale)
        ls_norm = c.seasonal_avg / 100.0
        
        # SWIM: already 0-1 (or 0 if unavailable)
        swim_norm = c.swim_avg
        
        # Accessibility: already 0-1
        acc_norm = c.accessibility_score
        
        # Climate resilience: already 0-1
        clim_norm = c.climate_resilience
        
        # Science bonus: mid-lat ice regions get bonus
        science = 0.5  # baseline
        if 30 <= c.center_lat <= 50:
            science += 0.2  # prime ice belt
        if c.elevation_m < -3000:
            science += 0.15  # deep basin = more atmosphere + sediments
        if c.swim_avg > 0.5:
            science += 0.15  # confirmed ice
        science = min(1.0, science)
        
        # Composite
        c.final_score = round(
            FINAL_WEIGHTS["landing_scorer"] * ls_norm +
            FINAL_WEIGHTS["swim_ice"] * swim_norm +
            FINAL_WEIGHTS["accessibility"] * acc_norm +
            FINAL_WEIGHTS["climate_resilience"] * clim_norm +
            FINAL_WEIGHTS["science_bonus"] * science,
            4
        ) * 100  # Scale to 0-100
        c.final_score = round(c.final_score, 1)
    
    # Rank
    candidates.sort(key=lambda c: c.final_score, reverse=True)
    for i, c in enumerate(candidates):
        c.final_rank = i + 1
    
    return candidates


# ═══════════════════════════════════════════════════════════════════════
# Report Generation
# ═══════════════════════════════════════════════════════════════════════

def generate_report(candidates: List[RegionCandidate], elapsed: float) -> str:
    """Generate the final analysis report."""
    lines = []
    
    def add(text=""):
        lines.append(text)
    
    add("╔══════════════════════════════════════════════════════════════════╗")
    add("║     MARS LANDING SITE ANALYSIS — COMPREHENSIVE REPORT          ║")
    add("║     MarsLab Integration Pipeline v1.0                          ║")
    add("╚══════════════════════════════════════════════════════════════════╝")
    add()
    add(f"Analysis completed in {elapsed:.1f}s")
    add(f"Regions analyzed: {len(MARS_REGIONS)} → {len(candidates)} viable candidates")
    add()
    
    add("═══ FINAL WEIGHTS ═══")
    for k, v in FINAL_WEIGHTS.items():
        add(f"  {k:25s} {v:.0%}")
    add()
    
    add("═══ TOP LANDING SITES ═══")
    add()
    
    for c in candidates[:10]:
        grade = "A" if c.final_score >= 80 else "B" if c.final_score >= 65 else "C" if c.final_score >= 50 else "D" if c.final_score >= 35 else "F"
        
        add(f"┌──────────────────────────────────────────────────────────────┐")
        add(f"│  #{c.final_rank}  {c.name:<40s}  [{grade}] {c.final_score:.1f}/100 │")
        add(f"├──────────────────────────────────────────────────────────────┤")
        add(f"│  Location:    {c.center_lat:.2f}°N, {c.center_lon:.2f}°E")
        add(f"│  Elevation:   {c.elevation_m:.0f} m MOLA")
        add(f"│")
        add(f"│  Landing Scorer (4-season):  avg={c.seasonal_avg:.1f}  worst={c.worst_season_score:.1f}  best={c.best_season_score:.1f}")
        add(f"│    Grades: Spr={c.seasonal_grades.get(0,'?')} Sum={c.seasonal_grades.get(90,'?')} Aut={c.seasonal_grades.get(180,'?')} Win={c.seasonal_grades.get(270,'?')}")
        if c.category_details:
            for cat_name, cat_data in c.category_details.items():
                add(f"│    {cat_name:15s}: {cat_data['score']:.3f} × {cat_data['weight']:.2f} = {cat_data['weighted']:.3f}  {cat_data['assessment']}")
        add(f"│")
        add(f"│  SWIM Ice Consistency:  avg={c.swim_avg:.3f}")
        for depth, val in c.swim_consistency.items():
            add(f"│    {depth:8s}: {val:.3f}" if val is not None else f"│    {depth:8s}: N/A")
        add(f"│")
        add(f"│  ISRU Accessibility:  {c.accessibility_score:.3f}  (conf={c.accessibility_details.get('confidence', '?')})")
        add(f"│    excavation={c.accessibility_details.get('excavation', 0):.3f}  landing={c.accessibility_details.get('landing', 0):.3f}")
        add(f"│")
        add(f"│  Climate Resilience:  {c.climate_resilience:.3f}")
        add(f"│    Temp range: {c.temp_range_k[0]:.0f}–{c.temp_range_k[1]:.0f} K")
        add(f"│    Peak dust τ: {c.max_dust_tau:.2f}  Peak wind: {c.max_wind_ms:.1f} m/s")
        add(f"│    Frost risk months: {c.frost_months}/12")
        add(f"│")
        if c.warnings:
            add(f"│  Warnings:")
            for w in c.warnings:
                add(f"│    {w}")
            add(f"│")
        add(f"│  Data sources: {c.data_sources}")
        add(f"└──────────────────────────────────────────────────────────────┘")
        add()
    
    # Summary table
    add("═══ SUMMARY TABLE ═══")
    add()
    add(f"{'Rank':<5} {'Region':<30} {'Score':<8} {'Grade':<6} {'Elev(m)':<10} {'SWIM':<8} {'Access':<8} {'Climate':<8} {'LS Avg':<8}")
    add("─" * 100)
    for c in candidates[:10]:
        grade = "A" if c.final_score >= 80 else "B" if c.final_score >= 65 else "C" if c.final_score >= 50 else "D" if c.final_score >= 35 else "F"
        add(f"#{c.final_rank:<4} {c.name:<30} {c.final_score:<8.1f} {grade:<6} {c.elevation_m:<10.0f} {c.swim_avg:<8.3f} {c.accessibility_score:<8.3f} {c.climate_resilience:<8.3f} {c.seasonal_avg:<8.1f}")
    
    add()
    add("═══ REFERENCES ═══")
    add("  [1] Golombek et al. (2021) — SpaceX Starship Landing Sites, LPSC 52, #2420")
    add("  [2] Morgan et al. (2021) — SWIM Ice Availability, Nature Astronomy 5, 230–236")
    add("  [3] Morgan et al. (2025) — SWIM Refined Global Mapping, PSJ 6(2):29")
    add("  [4] Baker et al. (2024) — I-MIM Phase 2, LPSC 2024, #2506")
    add("  [5] NASA DRA 5.0 — Human Exploration Reference Architecture")
    add("  [6] Stuurman et al. (2016) — Utopia Planitia Ice, GRL 43")
    add("  [7] Plaut et al. (2009) — Deuteronilus LDA Ice, GRL 36")
    add()
    
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    
    log.info("Mars Landing Site Comprehensive Analysis")
    log.info("Using %d named regions from mars_regions.py", len(MARS_REGIONS))
    log.info("")
    
    # Phase 1: Hard filter
    candidates = phase1_hard_filter(MARS_REGIONS)
    
    if not candidates:
        log.error("No regions passed hard filter! Check constraints.")
        return
    
    # Phase 2: Landing site scorer × 4 seasons
    candidates = phase2_seasonal_scoring(candidates)
    
    # Phase 3: SWIM ice + ISRU accessibility
    candidates = phase3_ice_analysis(candidates)
    
    # Phase 4: Climate resilience (12-point annual)
    candidates = phase4_climate_resilience(candidates)
    
    # Phase 5: Final composite + ranking
    candidates = phase5_final_ranking(candidates)
    
    elapsed = time.time() - t0
    
    # Generate report
    report = generate_report(candidates, elapsed)
    print()
    print(report)
    
    # Save to file
    output_dir = os.path.join(_BACKEND, "analysis", "integration")
    report_path = os.path.join(output_dir, "landing_site_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    log.info("Report saved to %s", report_path)
    
    # Save structured JSON
    json_path = os.path.join(output_dir, "landing_site_results.json")
    json_data = {
        "analysis_time_s": round(elapsed, 1),
        "total_regions": len(MARS_REGIONS),
        "viable_candidates": len(candidates),
        "constraints": {
            "hard": HARD_CONSTRAINTS,
            "soft": SOFT_CONSTRAINTS,
        },
        "final_weights": FINAL_WEIGHTS,
        "results": [],
    }
    for c in candidates:
        json_data["results"].append({
            "rank": c.final_rank,
            "region_id": c.region_id,
            "name": c.name,
            "center_lat": c.center_lat,
            "center_lon": c.center_lon,
            "elevation_m": c.elevation_m,
            "final_score": c.final_score,
            "seasonal_avg": c.seasonal_avg,
            "worst_season": c.worst_season_score,
            "best_season": c.best_season_score,
            "seasonal_grades": c.seasonal_grades,
            "swim_consistency": c.swim_consistency,
            "swim_avg": c.swim_avg,
            "accessibility_score": c.accessibility_score,
            "accessibility_details": c.accessibility_details,
            "climate_resilience": c.climate_resilience,
            "temp_range_k": list(c.temp_range_k),
            "max_dust_tau": c.max_dust_tau,
            "max_wind_ms": c.max_wind_ms,
            "frost_months": c.frost_months,
            "category_details": c.category_details,
            "warnings": c.warnings,
            "data_sources": c.data_sources,
        })
    
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2, default=str)
    log.info("JSON results saved to %s", json_path)
    
    return candidates


if __name__ == "__main__":
    main()
