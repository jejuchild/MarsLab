#!/usr/bin/env python3
"""
Test script for ice evidence synthesis.

Defines candidate locations in Arcadia Planitia (known ice-rich region)
and runs the full multi-criteria ice evidence pipeline.

Usage:
    cd backend
    python -m scripts.test_ice_evidence_run
"""

import json
import os
import sys

# Add backend to path
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_DIR)


def main():
    print(f"\n{'='*60}")
    print(f"  Ice Evidence Synthesis Test — Arcadia Planitia")
    print(f"{'='*60}\n")

    from analysis.ice_evidence.models import (
        IceEvidenceRequest,
        CandidateLocation,
        RegionSpec,
        SharadSpec,
        CrismSpec,
        DtmSpec,
        EvidenceParams,
    )
    from analysis.ice_evidence.sharad_reflectors import evaluate_reflector_evidence
    from analysis.ice_evidence.terrain_proxy import evaluate_terrain_evidence
    from analysis.ice_evidence.crism_proxy import evaluate_crism_evidence
    from analysis.ice_evidence.fusion import fuse_evidence
    from analysis.ice_evidence.io import save_evidence_result
    from analysis.ice_evidence.models import E1Hyperbola

    # Arcadia Planitia candidates (known ice-rich area, ~40-50°N, ~180-220°E)
    candidates = [
        CandidateLocation(lat=46.7, lon=-176.2, id="arcadia_north"),
        CandidateLocation(lat=44.5, lon=-170.8, id="arcadia_central"),
        CandidateLocation(lat=42.3, lon=-165.5, id="arcadia_south"),
    ]

    params = EvidenceParams()

    print(f"Weights: E1={params.weights.E1}, E2={params.weights.E2}, "
          f"E3={params.weights.E3}, E4={params.weights.E4}")
    print(f"Ice εr range: {params.epsr_ice_range}")
    print(f"Distance penalty: {params.distance_penalty_km} km\n")

    results = []

    for cand in candidates:
        print(f"── Candidate: {cand.id} ({cand.lat}°N, {cand.lon}°E) ──")

        # E1: Hyperbola (check for stored fits)
        print("  E1: Checking hyperbola fits...")
        e1 = E1Hyperbola(score=0.0, notes="No fits available for test")
        # Check if any fits exist
        results_dir = os.path.join(BACKEND_DIR, "results", "ice_evidence")
        if os.path.exists(results_dir):
            for fname in os.listdir(results_dir):
                if fname.startswith("hyperbola_") and fname.endswith(".json"):
                    with open(os.path.join(results_dir, fname)) as f:
                        fit = json.load(f)
                    epsr = fit.get("epsr", 0)
                    if 2.7 <= epsr <= 3.4:
                        e1 = E1Hyperbola(score=0.85, epsr=epsr,
                                         ci=fit.get("epsr_ci95"),
                                         flags=fit.get("flags", []),
                                         notes=f"Using stored fit εr={epsr:.2f}")
                    elif epsr > 0:
                        score = max(0, 0.9 - abs(epsr - 3.05) / 3.0)
                        e1 = E1Hyperbola(score=round(score, 3), epsr=epsr,
                                         ci=fit.get("epsr_ci95"),
                                         flags=fit.get("flags", []),
                                         notes=f"Using stored fit εr={epsr:.2f}")
                    break
        print(f"      Score={e1.score:.2f} — {e1.notes}")

        # E2: Reflectors
        print("  E2: Evaluating SHARAD reflectors...")
        e2 = evaluate_reflector_evidence(cand.lat, cand.lon)
        print(f"      Score={e2.score:.2f} — {e2.notes[:80]}")

        # E3: Terrain
        print("  E3: Evaluating terrain proxy...")
        e3 = evaluate_terrain_evidence(cand.lat, cand.lon)
        print(f"      Score={e3.score:.2f} — {e3.notes[:80]}")

        # E4: CRISM
        print("  E4: Evaluating CRISM evidence...")
        e4 = evaluate_crism_evidence(cand.lat, cand.lon)
        print(f"      Score={e4.score:.2f} — {e4.notes[:80]}")

        # Fuse
        result = fuse_evidence(cand, e1, e2, e3, e4, params)

        # Save
        json_path = save_evidence_result(result)
        result.artifacts.json_path = json_path

        results.append(result)

        print(f"\n  ╔══════════════════════════════════════════╗")
        print(f"  ║  Ice Probability: {result.ice_probability:.0%}".ljust(45) + "║")
        print(f"  ║  Confidence:      {result.confidence:.0%}".ljust(45) + "║")
        print(f"  ╚══════════════════════════════════════════╝")
        print(f"  Agreement: {result.consistency.agreement_score:.2f}")
        if result.consistency.conflicts:
            for c in result.consistency.conflicts:
                print(f"  ⚠ CONFLICT: {c}")
        print(f"  Saved: {json_path}")
        print()

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY — {len(results)} Candidates Evaluated")
    print(f"{'='*60}")
    print(f"{'ID':<20} {'Ice Prob':>10} {'Conf':>8} {'E1':>6} {'E2':>6} {'E3':>6} {'E4':>6}")
    print(f"{'─'*20} {'─'*10} {'─'*8} {'─'*6} {'─'*6} {'─'*6} {'─'*6}")
    for r in results:
        e = r.evidence
        print(
            f"{r.candidate_id:<20} "
            f"{r.ice_probability:>9.1%} "
            f"{r.confidence:>7.1%} "
            f"{e.E1_hyperbola.score:>5.2f} "
            f"{e.E2_reflector.score:>5.2f} "
            f"{e.E3_terrain.score:>5.2f} "
            f"{e.E4_crism.score:>5.2f}"
        )

    # Best candidate
    best = max(results, key=lambda r: r.ice_probability)
    print(f"\n  Best candidate: {best.candidate_id}")
    print(f"  Ice probability: {best.ice_probability:.0%}")
    print(f"  NOTE: These are evidence-based estimates, NOT direct proof of ice.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
