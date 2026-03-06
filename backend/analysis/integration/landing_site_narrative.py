#!/usr/bin/env python3
"""
Mars Landing Site Narrative Report — RAG-Enriched Analysis
==========================================================
Option D Phase 2: Uses RAG knowledge base (5,193 vectors) + Groq LLM
to generate scientifically-grounded narrative for each candidate site.

Reads landing_site_results.json from Phase 1, enriches with RAG context,
produces final recommendation document.

Usage:
  cd backend && python -m analysis.integration.landing_site_narrative
"""

import json
import logging
import os
import sys
import time

_BACKEND = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("landing_narrative")

_RESULTS_PATH = os.path.join(os.path.dirname(__file__), "landing_site_results.json")
_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "landing_site_final_report.md")


def load_results():
    with open(_RESULTS_PATH) as f:
        return json.load(f)


def query_rag(query: str, n_results: int = 8) -> dict:
    """Query RAG knowledge base directly (no HTTP server needed)."""
    try:
        from rag.generator import generate_answer
        return generate_answer(query, n_results=n_results, collection="mars_science")
    except Exception as e:
        log.warning("RAG query failed: %s", e)
        return {"answer": f"[RAG unavailable: {e}]", "citations": [], "grounded": False}


def generate_site_narrative(site: dict) -> dict:
    """Query RAG for scientific context on a candidate site."""
    name = site["name"]
    lat = site["center_lat"]
    swim = site["swim_avg"]
    elev = site["elevation_m"]
    
    # Query 1: Ice & ISRU potential
    q1 = (f"What is known about subsurface water ice at {name}? "
          f"Include SWIM data, SHARAD radar detections, ice depth estimates, "
          f"and ISRU implications for human exploration.")
    
    # Query 2: Landing site suitability  
    q2 = (f"Analyze {name} (lat {lat:.1f}°N, elev {elev:.0f}m) as a potential "
          f"Mars landing site for human missions. Consider terrain, EDL constraints, "
          f"geological hazards, and scientific value.")
    
    log.info("  Querying RAG for %s...", name)
    
    ice_context = query_rag(q1)
    landing_context = query_rag(q2)
    
    return {
        "ice_narrative": ice_context["answer"],
        "ice_grounded": ice_context["grounded"],
        "ice_citations": ice_context["citations"],
        "landing_narrative": landing_context["answer"],
        "landing_grounded": landing_context["grounded"],
        "landing_citations": landing_context["citations"],
    }


def generate_final_recommendation(results: dict) -> str:
    """Generate the grand synthesis using Groq 70B."""
    sites = results["results"][:5]
    
    summary_data = "\n".join([
        f"#{s['rank']} {s['name']}: score={s['final_score']}, "
        f"SWIM={s['swim_avg']:.3f}, accessibility={s['accessibility_score']:.3f}, "
        f"climate_resilience={s.get('climate_resilience', 0):.3f}, "
        f"elev={s['elevation_m']:.0f}m, seasonal_avg={s['seasonal_avg']:.1f}"
        for s in sites
    ])
    
    prompt = f"""You are a planetary scientist advising NASA/SpaceX on Mars landing site selection 
for the first human ISRU mission. Based on the following quantitative analysis of 5 candidate sites 
(scored from 55 named Mars regions using MOLA terrain, Neural Climate predictions, SWIM ice consistency, 
ISRU accessibility algorithms, and 4-season landing site scoring), provide your expert recommendation.

QUANTITATIVE RESULTS:
{summary_data}

KEY CONSTRAINTS (Golombek et al. 2021, Morgan et al. 2021/2025):
- Elevation < -2 km MOLA (Starship EDL)
- Latitude 30-45°N optimal (ice + solar trade-off)
- SWIM ice consistency > 0.5 preferred
- Ice depth < 10m for accessible ISRU
- Slope < 5° at 10m scale

Provide:
1. Your top recommendation with scientific justification (3-4 paragraphs)
2. Why the runner-up sites are also viable but less optimal
3. Key risks and uncertainties for the top site
4. What precursor missions should investigate before committing

Write in a tone suitable for a mission architecture review. Cite the scoring data."""
    
    log.info("  Generating grand synthesis (Groq 70B)...")
    
    try:
        from rag.generator import _call_groq
        answer = _call_groq(
            messages=[
                {"role": "system", "content": "You are a senior planetary scientist specializing in Mars exploration architecture and ISRU site selection."},
                {"role": "user", "content": prompt},
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.4,
            max_tokens=3000,
        )
        return answer or "[Synthesis generation failed]"
    except Exception as e:
        log.warning("Synthesis generation failed: %s", e)
        return f"[Synthesis unavailable: {e}]"


def build_markdown_report(results: dict, narratives: dict, synthesis: str) -> str:
    """Build the final Markdown report."""
    lines = []
    
    def add(text=""):
        lines.append(text)
    
    add("# 🔴 Mars Landing Site Selection — Final Report")
    add("")
    add(f"> MarsLab Integration Pipeline v1.0 — {time.strftime('%Y-%m-%d %H:%M UTC')}")
    add(f"> Analyzed {results['total_regions']} named regions → {results['viable_candidates']} viable candidates")
    add(f"> Data sources: MOLA terrain, Neural Climate Emulator, SWIM v2.1, ISRU Accessibility, RAG (5,193 vectors)")
    add("")
    
    # Executive Summary
    add("## Executive Summary")
    add("")
    add(synthesis)
    add("")
    
    # Rankings table
    add("## Quantitative Rankings")
    add("")
    add("| Rank | Region | Score | Grade | Elev (m) | SWIM Ice | Access | Climate | LS Avg |")
    add("|------|--------|-------|-------|----------|----------|--------|---------|--------|")
    for s in results["results"][:5]:
        grade = "A" if s["final_score"] >= 80 else "B" if s["final_score"] >= 65 else "C" if s["final_score"] >= 50 else "D"
        add(f"| #{s['rank']} | **{s['name']}** | {s['final_score']} | {grade} | {s['elevation_m']:.0f} | {s['swim_avg']:.3f} | {s['accessibility_score']:.3f} | {s.get('climate_resilience', 0):.3f} | {s['seasonal_avg']:.1f} |")
    add("")
    
    # Scoring methodology
    add("## Scoring Methodology")
    add("")
    add("### Hard Constraints (Elimination)")
    add(f"- **Elevation**: < {results['constraints']['hard']['max_elevation_m']}m MOLA (Starship EDL)")
    add(f"- **Latitude**: {results['constraints']['hard']['lat_min']}–{results['constraints']['hard']['lat_max']}°N (ice belt + solar)")
    add(f"- **Terrain**: Exclude volcanic/canyon regions (slope hazard)")
    add("")
    add("### Final Composite Weights")
    for k, v in results["final_weights"].items():
        add(f"- **{k.replace('_', ' ').title()}**: {v:.0%}")
    add("")
    
    # Detailed site profiles
    add("## Detailed Site Profiles")
    add("")
    
    for s in results["results"][:5]:
        rid = s["region_id"]
        narr = narratives.get(rid, {})
        grade = "A" if s["final_score"] >= 80 else "B" if s["final_score"] >= 65 else "C" if s["final_score"] >= 50 else "D"
        
        add(f"### #{s['rank']} — {s['name']} ({grade}, {s['final_score']}/100)")
        add("")
        add(f"**Location**: {s['center_lat']:.2f}°N, {s['center_lon']:.2f}°E | **Elevation**: {s['elevation_m']:.0f}m MOLA")
        add("")
        
        # Seasonal scoring
        add("#### Landing Site Scorer (4 Seasons)")
        add(f"- Average: **{s['seasonal_avg']:.1f}/100** | Worst: {s['worst_season']:.1f} | Best: {s['best_season']:.1f}")
        grades_str = ", ".join(f"Ls={ls}°: {g}" for ls, g in sorted(s.get("seasonal_grades", {}).items(), key=lambda x: float(x[0])))
        add(f"- Grades: {grades_str}")
        if s.get("category_details"):
            add("")
            add("| Category | Score | Weight | Assessment |")
            add("|----------|-------|--------|------------|")
            for cat_name, cat in s["category_details"].items():
                add(f"| {cat_name} | {cat['score']:.3f} | {cat['weight']:.2f} | {cat['assessment']} |")
        add("")
        
        # SWIM ice
        add("#### SWIM Ice Consistency")
        add(f"- **Average**: {s['swim_avg']:.3f}")
        swim = s.get("swim_consistency", {})
        for depth, val in swim.items():
            val_str = f"{val:.3f}" if val is not None else "N/A"
            add(f"- {depth}: {val_str}")
        add("")
        
        # ISRU
        acc = s.get("accessibility_details", {})
        add("#### ISRU Accessibility")
        add(f"- **Composite**: {s['accessibility_score']:.3f} (confidence: {acc.get('confidence', '?')})")
        add(f"- Excavation: {acc.get('excavation', 0):.3f} | Landing: {acc.get('landing', 0):.3f}")
        add("")
        
        # Climate
        add("#### Climate Resilience (12-Point Annual)")
        add(f"- **Score**: {s.get('climate_resilience', 0):.3f}")
        tr = s.get("temp_range_k", [0, 0])
        add(f"- Temperature: {tr[0]:.0f}–{tr[1]:.0f} K ({tr[0]-273.15:.0f}–{tr[1]-273.15:.0f} °C)")
        add(f"- Peak dust τ: {s.get('max_dust_tau', 0):.2f} | Peak wind: {s.get('max_wind_ms', 0):.1f} m/s | Frost months: {s.get('frost_months', 0)}/12")
        add("")
        
        # RAG narratives
        if narr.get("ice_narrative"):
            add("#### Scientific Context — Ice & ISRU")
            add(narr["ice_narrative"])
            if narr.get("ice_citations"):
                add("")
                add("*Sources: " + ", ".join(c.get("source", c.get("title", "?"))[:60] for c in narr["ice_citations"][:3]) + "*")
            add("")
        
        if narr.get("landing_narrative"):
            add("#### Scientific Context — Landing Suitability")
            add(narr["landing_narrative"])
            if narr.get("landing_citations"):
                add("")
                add("*Sources: " + ", ".join(c.get("source", c.get("title", "?"))[:60] for c in narr["landing_citations"][:3]) + "*")
            add("")
        
        # Warnings
        if s.get("warnings"):
            add("#### ⚠ Warnings")
            for w in s["warnings"]:
                add(f"- {w}")
            add("")
        
        add("---")
        add("")
    
    # References
    add("## References")
    add("")
    add("1. Golombek et al. (2021) — *SpaceX Starship Landing Sites on Mars*, LPSC 52, Abstract 2420")
    add("2. Morgan et al. (2021) — *Availability of subsurface water-ice resources in the northern mid-latitudes of Mars*, Nature Astronomy 5, 230–236")
    add("3. Morgan et al. (2025) — *Refined Mapping of Subsurface Water Ice on Mars*, PSJ 6(2):29")
    add("4. Baker et al. (2024) — *International Mars Ice Mapper Phase 2*, LPSC 2024, Abstract 2506")
    add("5. Stuurman et al. (2016) — *SHARAD detection of widespread subsurface ice in Utopia Planitia*, GRL 43")
    add("6. Plaut et al. (2009) — *Radar evidence for ice in lobate debris aprons in the mid-northern latitudes of Mars*, GRL 36")
    add("7. Bramson et al. (2015) — *Widespread excess ice in Arcadia Planitia*, GRL 42")
    add("8. NASA DRA 5.0 — *Human Exploration of Mars Design Reference Architecture*")
    add("9. Bussey & Hoffman (2016) — *Human Mars Landing Site and Impacts on Mars Surface Operations*, NASA NTRS")
    add("10. Luzzi et al. (2025) — *Geomorphological evidence of near-surface ice at Arcadia/Amazonis boundary*, JGR Planets")
    add("")
    
    return "\n".join(lines)


def main():
    t0 = time.time()
    
    log.info("Mars Landing Site Narrative Report — RAG Enrichment")
    log.info("Loading quantitative results...")
    
    results = load_results()
    sites = results["results"][:5]
    log.info("Loaded %d candidate sites", len(sites))
    
    # Query RAG for each site
    narratives = {}
    for site in sites:
        narr = generate_site_narrative(site)
        narratives[site["region_id"]] = narr
    
    # Generate grand synthesis
    synthesis = generate_final_recommendation(results)
    
    # Build final report
    report = build_markdown_report(results, narratives, synthesis)
    
    # Save
    with open(_OUTPUT_PATH, "w") as f:
        f.write(report)
    
    elapsed = time.time() - t0
    log.info("Final report saved to %s (%.1fs)", _OUTPUT_PATH, elapsed)
    
    # Print to stdout
    print()
    print(report)


if __name__ == "__main__":
    main()
