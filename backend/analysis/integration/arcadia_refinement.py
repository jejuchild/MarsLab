#!/usr/bin/env python3
"""
Arcadia Planitia High-Resolution Landing Site Refinement
=========================================================
0.5° grid search around the top zone identified by the coarse analysis:
  Zone #1: 42°N, 176°E — combined score 0.858

Scans: 38-46°N × 170-184°E at 0.5° resolution = ~17×29 = ~493 points
Each point evaluated with:
  - SWIM ice consistency (3 depth ranges)
  - Landing Site Scorer (Ls=0 spring equinox)
  - Terrain quality (elevation, slope, TRI)

Output: Top 20 candidate sites ranked by composite score.

Usage:
  cd backend && python -m analysis.integration.arcadia_refinement
"""

import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# Ensure backend is importable
_BACKEND = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)


# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════

# Refinement grid: focused on Arcadia Planitia top zone
LAT_MIN, LAT_MAX = 38.0, 46.0   # 8° range (was 34-48 in coarse)
LON_MIN, LON_MAX = 170.0, 184.0  # 14° range (centered on 176°E)
STEP = 0.5  # 0.5° resolution

# Hard constraints (same as main analysis)
HARD = {
    "max_elevation_m": -2000,
    "max_slope_deg": 5.0,  # Tighter for final pinpoint (Starship ideal < 5°)
}

# Composite weights for high-res scoring
WEIGHTS = {
    "swim_avg": 0.30,        # Ice availability (primary mission driver)
    "swim_shallow": 0.10,    # 0-1m shallow ice bonus (easier excavation)
    "landing_score": 0.25,   # Landing site scorer (terrain+climate+dust+wind)
    "terrain": 0.20,         # Terrain quality (elev+slope+TRI)
    "climate": 0.15,         # Climate resilience 
}


@dataclass
class GridPoint:
    lat: float
    lon: float
    elevation_m: float = 0.0
    slope_deg: float = 0.0
    tri: float = 0.0
    
    # SWIM
    swim_0_1m: Optional[float] = None
    swim_1_5m: Optional[float] = None
    swim_5m_plus: Optional[float] = None
    swim_avg: float = 0.0
    
    # Landing scorer
    landing_score: float = 0.0
    landing_grade: str = ""
    
    # Climate
    climate_resilience: float = 0.0
    temp_min_k: float = 0.0
    temp_max_k: float = 0.0
    dust_tau: float = 0.0
    wind_ms: float = 0.0
    frost_prob: float = 0.0
    
    # Composite
    composite: float = 0.0
    passed_hard: bool = True
    reject_reason: str = ""


def run_refinement():
    """Execute the high-resolution grid search."""
    t0 = time.time()
    
    # Build grid
    lats = np.arange(LAT_MIN, LAT_MAX + STEP/2, STEP)
    lons = np.arange(LON_MIN, LON_MAX + STEP/2, STEP)
    total = len(lats) * len(lons)
    
    print(f"╔══════════════════════════════════════════════════════════════╗")
    print(f"║  ARCADIA PLANITIA — HIGH-RESOLUTION REFINEMENT             ║")
    print(f"║  Grid: {LAT_MIN}–{LAT_MAX}°N × {LON_MIN}–{LON_MAX}°E     ║")
    print(f"║  Step: {STEP}°  →  {total} points                         ║")
    print(f"╚══════════════════════════════════════════════════════════════╝")
    print()
    
    # Load subsystems
    print("Loading subsystems...")
    
    from api.terrain_router import compute_slope_stats
    
    swim_ok = False
    try:
        from analysis.swim_fusion.pipeline import SwimFusionPipeline
        fusion = SwimFusionPipeline()
        swim_ok = True
        print("  ✓ SWIM fusion pipeline")
    except Exception as e:
        print(f"  ✗ SWIM unavailable: {e}")
    
    scorer_ok = False
    try:
        from analysis.integration.landing_site_scorer import score_landing_site
        scorer_ok = True
        print("  ✓ Landing site scorer")
    except Exception as e:
        print(f"  ✗ Scorer unavailable: {e}")
    
    climate_ok = False
    try:
        from neural_climate.predictor import get_predictor, is_model_trained
        if is_model_trained():
            predictor = get_predictor()
            climate_ok = True
            print("  ✓ Neural climate emulator")
    except Exception:
        pass
    if not climate_ok:
        from api.mars_climate import (
            surface_temperature_k, dust_opacity, wind_speed, co2_frost_probability,
        )
        print("  ✓ Parametric climate model (fallback)")
    
    print()
    
    # Scan grid
    points: List[GridPoint] = []
    passed = 0
    rejected = 0
    
    for i, lat in enumerate(lats):
        row_points = 0
        for lon in lons:
            gp = GridPoint(lat=float(lat), lon=float(lon))
            
            # --- Terrain ---
            try:
                stats = compute_slope_stats(lat, lon, radius_m=2000)
                gp.elevation_m = stats.get('elevation_m', 0.0)
                gp.slope_deg = stats.get('mean_slope', 0.0)
                gp.tri = stats.get('tri', 0.0) if 'tri' in stats else 0.0
            except Exception:
                gp.elevation_m = 0.0
                gp.slope_deg = 0.0
            
            # --- Hard filter ---
            if gp.elevation_m != 0.0 and gp.elevation_m > HARD["max_elevation_m"]:
                gp.passed_hard = False
                gp.reject_reason = f"elev={gp.elevation_m:.0f}m"
                rejected += 1
                points.append(gp)
                continue
            if gp.slope_deg > HARD["max_slope_deg"]:
                gp.passed_hard = False
                gp.reject_reason = f"slope={gp.slope_deg:.1f}°"
                rejected += 1
                points.append(gp)
                continue
            
            # --- SWIM ---
            if swim_ok:
                try:
                    result = fusion.query_point(lat=float(lat), lon=float(lon), mode="precomputed")
                    if hasattr(result, 'model_dump'):
                        result = result.model_dump()
                    elif not isinstance(result, dict):
                        result = {}
                    gp.swim_0_1m = result.get("consistency_0_1m")
                    gp.swim_1_5m = result.get("consistency_1_5m")
                    gp.swim_5m_plus = result.get("consistency_5m_plus")
                    vals = [v for v in [gp.swim_0_1m, gp.swim_1_5m, gp.swim_5m_plus] if v is not None]
                    gp.swim_avg = sum(vals) / len(vals) if vals else 0.0
                except Exception:
                    pass
            
            # --- Landing Scorer (Ls=0, spring equinox) ---
            if scorer_ok:
                try:
                    result = score_landing_site(float(lat), float(lon), ls=0.0)
                    gp.landing_score = result.overall_score
                    gp.landing_grade = result.grade
                except Exception:
                    pass
            
            # --- Climate (4-season check) ---
            ls_points = [0, 90, 180, 270]
            temps_min, temps_max, dusts, winds, frosts = [], [], [], [], []
            for ls in ls_points:
                try:
                    if climate_ok:
                        pred = predictor.predict(float(lat), float(lon), float(ls))
                        temps_min.append(pred["temperature_min_k"])
                        temps_max.append(pred["temperature_max_k"])
                        dusts.append(pred["dust_tau_mean"])
                        winds.append(pred["wind_mean_ms"])
                        frosts.append(pred["frost_probability"])
                    else:
                        temp = surface_temperature_k(float(lat), float(ls), gp.elevation_m)
                        dust = dust_opacity(float(lat), float(ls))
                        wind = wind_speed(float(lat), float(ls))
                        frost = co2_frost_probability(float(lat), float(ls), gp.elevation_m)
                        temps_min.append(temp["min_k"])
                        temps_max.append(temp["max_k"])
                        dusts.append(dust["tau_mean"])
                        winds.append(wind["mean_ms"])
                        frosts.append(frost["frost_probability"])
                except Exception:
                    pass
            
            if temps_min:
                gp.temp_min_k = min(temps_min)
                gp.temp_max_k = max(temps_max)
                gp.dust_tau = max(dusts) if dusts else 0.0
                gp.wind_ms = max(winds) if winds else 0.0
                gp.frost_prob = max(frosts) if frosts else 0.0
                
                temp_span = gp.temp_max_k - gp.temp_min_k
                t_score = max(0, 1.0 - (temp_span - 60) / 100)
                d_score = max(0, 1.0 - gp.dust_tau / 2.0)
                w_score = max(0, 1.0 - gp.wind_ms / 15.0)
                f_score = max(0, 1.0 - gp.frost_prob)
                gp.climate_resilience = 0.25 * t_score + 0.25 * d_score + 0.25 * w_score + 0.25 * f_score
            
            passed += 1
            row_points += 1
            points.append(gp)
        
        # Progress
        pct = (i + 1) / len(lats) * 100
        print(f"  Row {i+1}/{len(lats)} (lat={lat:.1f}°N) — {row_points} points passed | {pct:.0f}%")
    
    print()
    print(f"Grid complete: {passed} passed, {rejected} rejected out of {total}")
    print()
    
    # --- Compute composite scores ---
    viable = [p for p in points if p.passed_hard]
    
    if not viable:
        print("ERROR: No viable points! Check constraints.")
        return
    
    # Normalize each dimension to 0-1 within viable set
    ls_scores = [p.landing_score for p in viable]
    ls_min, ls_max = min(ls_scores), max(ls_scores)
    
    elevs = [p.elevation_m for p in viable if p.elevation_m != 0.0]
    elev_min = min(elevs) if elevs else -5000
    elev_max = max(elevs) if elevs else -2000
    
    slopes = [p.slope_deg for p in viable]
    slope_max_val = max(slopes) if slopes else 5.0
    
    for p in viable:
        # Normalize landing score (0-100 → 0-1)
        ls_norm = p.landing_score / 100.0
        
        # SWIM avg (already 0-1)
        swim_norm = p.swim_avg
        swim_shallow = p.swim_0_1m if p.swim_0_1m is not None else 0.0
        
        # Terrain quality: lower elevation = better (more atmosphere), lower slope = better
        if p.elevation_m != 0.0:
            # Lower elevation → higher score. Range roughly -5000 to -2000
            elev_range = elev_max - elev_min if elev_max != elev_min else 1.0
            elev_score = (elev_max - p.elevation_m) / elev_range
        else:
            elev_score = 0.5
        slope_score = max(0, 1.0 - p.slope_deg / 5.0)
        terrain_norm = 0.6 * elev_score + 0.4 * slope_score
        
        # Climate (already 0-1)
        clim_norm = p.climate_resilience
        
        # Composite
        p.composite = (
            WEIGHTS["swim_avg"] * swim_norm +
            WEIGHTS["swim_shallow"] * swim_shallow +
            WEIGHTS["landing_score"] * ls_norm +
            WEIGHTS["terrain"] * terrain_norm +
            WEIGHTS["climate"] * clim_norm
        )
    
    # Sort by composite
    viable.sort(key=lambda p: p.composite, reverse=True)
    
    # Print Top 20
    print("═══ TOP 20 CANDIDATE SITES ═══")
    print()
    print(f"{'Rank':<5} {'Lat°N':<7} {'Lon°E':<8} {'Composite':<10} {'SWIM_avg':<10} {'SWIM_0-1m':<10} {'SWIM_1-5m':<10} {'SWIM_5m+':<10} {'Elev(m)':<10} {'Slope°':<8} {'LandScore':<10} {'Climate':<8} {'Grade':<6}")
    print("─" * 135)
    
    for i, p in enumerate(viable[:20]):
        swim_01 = f"{p.swim_0_1m:.3f}" if p.swim_0_1m is not None else "N/A"
        swim_15 = f"{p.swim_1_5m:.3f}" if p.swim_1_5m is not None else "N/A"
        swim_5p = f"{p.swim_5m_plus:.3f}" if p.swim_5m_plus is not None else "N/A"
        print(f"#{i+1:<4} {p.lat:<7.1f} {p.lon:<8.1f} {p.composite:<10.4f} {p.swim_avg:<10.3f} {swim_01:<10} {swim_15:<10} {swim_5p:<10} {p.elevation_m:<10.0f} {p.slope_deg:<8.1f} {p.landing_score:<10.1f} {p.climate_resilience:<8.3f} {p.landing_grade:<6}")
    
    elapsed = time.time() - t0
    print()
    print(f"Analysis completed in {elapsed:.1f}s")
    
    # ═══ Zone Analysis ═══
    # Identify clusters among top 20
    print()
    print("═══ ZONE ANALYSIS (Top 20 clustering) ═══")
    print()
    
    top20 = viable[:20]
    # Simple grid-based clustering: group by 1° lat/lon bins
    clusters = {}
    for p in top20:
        key = (round(p.lat), round(p.lon))
        if key not in clusters:
            clusters[key] = []
        clusters[key].append(p)
    
    print(f"{'Zone Center':<20} {'Points':<8} {'Best Composite':<15} {'Avg SWIM':<10} {'Avg Elev':<10} {'Avg Slope':<10}")
    print("─" * 80)
    for key in sorted(clusters.keys(), key=lambda k: max(p.composite for p in clusters[k]), reverse=True):
        pts = clusters[key]
        best = max(p.composite for p in pts)
        avg_swim = sum(p.swim_avg for p in pts) / len(pts)
        avg_elev = sum(p.elevation_m for p in pts) / len(pts)
        avg_slope = sum(p.slope_deg for p in pts) / len(pts)
        print(f"{key[0]}°N, {key[1]}°E       {len(pts):<8} {best:<15.4f} {avg_swim:<10.3f} {avg_elev:<10.0f} {avg_slope:<10.1f}")
    
    # ═══ Final Recommendation ═══
    best = viable[0]
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print(f"║  OPTIMAL LANDING SITE: {best.lat:.1f}°N, {best.lon:.1f}°E             ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print(f"║  Composite Score:   {best.composite:.4f}                            ║")
    print(f"║  Elevation:         {best.elevation_m:.0f} m MOLA                       ║")
    print(f"║  Slope:             {best.slope_deg:.1f}°                               ║")
    print(f"║  Landing Score:     {best.landing_score:.1f}/100 ({best.landing_grade})                      ║")
    print(f"║  SWIM Ice:          avg={best.swim_avg:.3f}                         ║")
    swim_01 = best.swim_0_1m if best.swim_0_1m is not None else 0
    swim_15 = best.swim_1_5m if best.swim_1_5m is not None else 0
    swim_5p = best.swim_5m_plus if best.swim_5m_plus is not None else 0
    print(f"║    0-1m: {swim_01:.3f}  1-5m: {swim_15:.3f}  5m+: {swim_5p:.3f}        ║")
    print(f"║  Climate:           resilience={best.climate_resilience:.3f}            ║")
    print(f"║    Temp: {best.temp_min_k:.0f}–{best.temp_max_k:.0f} K                             ║")
    print(f"║    Dust τ_max: {best.dust_tau:.3f}  Wind_max: {best.wind_ms:.1f} m/s           ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    
    # Save JSON results
    output_dir = os.path.join(_BACKEND, "analysis", "integration")
    json_path = os.path.join(output_dir, "arcadia_refinement_results.json")
    
    results = {
        "grid": {
            "lat_range": [LAT_MIN, LAT_MAX],
            "lon_range": [LON_MIN, LON_MAX],
            "step_deg": STEP,
            "total_points": total,
            "passed_points": passed,
            "rejected_points": rejected,
        },
        "weights": WEIGHTS,
        "hard_constraints": HARD,
        "analysis_time_s": round(elapsed, 1),
        "optimal_site": {
            "lat": best.lat,
            "lon": best.lon,
            "lon_west": best.lon - 360 if best.lon > 180 else best.lon,
            "composite_score": round(best.composite, 4),
            "elevation_m": best.elevation_m,
            "slope_deg": best.slope_deg,
            "landing_score": best.landing_score,
            "landing_grade": best.landing_grade,
            "swim_avg": round(best.swim_avg, 3),
            "swim_0_1m": best.swim_0_1m,
            "swim_1_5m": best.swim_1_5m,
            "swim_5m_plus": best.swim_5m_plus,
            "climate_resilience": round(best.climate_resilience, 3),
            "temp_range_k": [best.temp_min_k, best.temp_max_k],
            "dust_tau_max": best.dust_tau,
            "wind_max_ms": best.wind_ms,
        },
        "top_20": [],
    }
    
    for i, p in enumerate(viable[:20]):
        results["top_20"].append({
            "rank": i + 1,
            "lat": p.lat,
            "lon": p.lon,
            "lon_west": p.lon - 360 if p.lon > 180 else p.lon,
            "composite": round(p.composite, 4),
            "swim_avg": round(p.swim_avg, 3),
            "swim_0_1m": p.swim_0_1m,
            "swim_1_5m": p.swim_1_5m,
            "swim_5m_plus": p.swim_5m_plus,
            "elevation_m": p.elevation_m,
            "slope_deg": p.slope_deg,
            "landing_score": p.landing_score,
            "landing_grade": p.landing_grade,
            "climate_resilience": round(p.climate_resilience, 3),
        })
    
    # Also save all viable points for potential visualization
    results["all_viable"] = []
    for p in viable:
        results["all_viable"].append({
            "lat": p.lat,
            "lon": p.lon,
            "composite": round(p.composite, 4),
            "swim_avg": round(p.swim_avg, 3),
            "elevation_m": p.elevation_m,
            "slope_deg": round(p.slope_deg, 2),
        })
    
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {json_path}")
    
    return results


if __name__ == "__main__":
    run_refinement()
