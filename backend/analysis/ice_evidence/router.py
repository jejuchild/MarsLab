"""
FastAPI router for Ice Evidence Synthesizer.

Endpoints:
  POST /api/ice/hyperbola/fit    — Fit hyperbola to SHARAD diffraction
  POST /api/ice/evidence/run     — Run full ice evidence synthesis
  GET  /api/ice/evidence/latest  — Get latest result for a candidate
  GET  /api/ice/evidence/list    — List recent results
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from .models import (
    HyperbolaFitRequest,
    HyperbolaFitResult,
    IceEvidenceRequest,
    IceEvidenceResult,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ice", tags=["Ice Evidence"])

_SAFE_ID = re.compile(r"^[a-zA-Z0-9_\-]+$")


def _validate_id(val: str, name: str = "id"):
    if not _SAFE_ID.match(val):
        raise HTTPException(status_code=400, detail=f"Invalid {name}: {val}")


# ── Hyperbola Fit ────────────────────────────────────────

@router.post("/hyperbola/fit", response_model=HyperbolaFitResult)
async def fit_hyperbola(req: HyperbolaFitRequest):
    """Fit a hyperbola to a SHARAD diffraction pattern.

    Returns velocity, εr, depth, confidence intervals, overlay curve,
    and preset comparison curves.
    """
    _validate_id(req.product_id, "product_id")

    try:
        from .hyperbola_fit import fit_hyperbola as _fit
        result = _fit(req)

        # Persist result
        from .io import save_hyperbola_fit
        save_hyperbola_fit(req.product_id, result)

        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception(f"Hyperbola fit failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/hyperbola/auto-detect")
async def auto_detect_apexes(
    product_id: str = Query(..., description="SHARAD product ID"),
    n: int = Query(3, ge=1, le=10, description="Number of candidates"),
):
    """Auto-detect top diffraction apex candidates in a SHARAD product."""
    _validate_id(product_id, "product_id")

    try:
        from .hyperbola_fit import auto_detect_apexes as _detect
        candidates = _detect(product_id, n_candidates=n)
        return JSONResponse(content={"candidates": candidates})
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception(f"Auto-detect failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Ice Evidence Synthesis ───────────────────────────────

@router.post("/evidence/run", response_model=List[IceEvidenceResult])
async def run_ice_evidence(req: IceEvidenceRequest):
    """Run full multi-criteria ice evidence synthesis for candidate locations.

    Combines:
      E1: SHARAD hyperbola-based εr
      E2: SHARAD subsurface reflector confidence
      E3: Terrain proxy from DTM/DEM
      E4: CRISM ice/hydration evidence

    Returns per-candidate ice probability with uncertainty and explainability.
    """
    if not req.candidates:
        raise HTTPException(status_code=400, detail="No candidates provided")

    try:
        from .sharad_reflectors import evaluate_reflector_evidence
        from .terrain_proxy import evaluate_terrain_evidence
        from .crism_proxy import evaluate_crism_evidence
        from .fusion import fuse_evidence
        from .io import save_evidence_result
        from .models import E1Hyperbola

        results = []

        for cand in req.candidates:
            logger.info(f"Processing candidate {cand.id} at ({cand.lat}, {cand.lon})")

            # E1: Hyperbola εr — use stored fits if available, else score=0
            e1 = _get_hyperbola_evidence(cand.lat, cand.lon, req.params)

            # E2: Reflector evidence
            e2 = evaluate_reflector_evidence(
                cand.lat, cand.lon,
                sharad_tracks=req.sharad.tracks if req.sharad else None,
            )

            # E3: Terrain proxy
            e3 = evaluate_terrain_evidence(
                cand.lat, cand.lon,
                hirise_dtm_products=req.dtm.hirise_dtm_products if req.dtm else None,
            )

            # E4: CRISM evidence
            e4 = evaluate_crism_evidence(
                cand.lat, cand.lon,
                crism_products=req.crism.products if req.crism else None,
                distance_scale_km=req.params.distance_penalty_km,
            )

            # Fuse
            result = fuse_evidence(cand, e1, e2, e3, e4, req.params)

            # Save
            json_path = save_evidence_result(result)
            result.artifacts.json_path = json_path

            results.append(result)
            logger.info(
                f"Candidate {cand.id}: ice_prob={result.ice_probability:.3f}, "
                f"confidence={result.confidence:.3f}"
            )

        return results

    except Exception as e:
        logger.exception(f"Ice evidence run failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/evidence/latest")
async def get_latest_evidence(
    candidate_id: str = Query(..., description="Candidate ID"),
):
    """Return most recent stored ice evidence result for a candidate."""
    _validate_id(candidate_id, "candidate_id")

    from .io import load_latest_result
    result = load_latest_result(candidate_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"No results for {candidate_id}")
    return JSONResponse(content=result)


@router.get("/evidence/list")
async def list_evidence_results(
    limit: int = Query(50, ge=1, le=500),
):
    """List recent ice evidence results."""
    from .io import list_results
    return JSONResponse(content={"results": list_results(limit)})


# ── Internal helpers ─────────────────────────────────────

def _get_hyperbola_evidence(lat: float, lon: float, params) -> "E1Hyperbola":
    """Check for stored hyperbola fits near candidate location.

    If fits exist, convert the best εr to an E1 score.
    """
    from .models import E1Hyperbola
    import os
    import json

    from .io import RESULTS_DIR

    if not os.path.exists(RESULTS_DIR):
        return E1Hyperbola(score=0.0, notes="No hyperbola fits available")

    # Look for hyperbola fit files
    epsr_values = []
    best_fit = None

    for fname in os.listdir(RESULTS_DIR):
        if fname.startswith("hyperbola_") and fname.endswith(".json"):
            try:
                filepath = os.path.join(RESULTS_DIR, fname)
                with open(filepath, "r") as f:
                    fit_data = json.load(f)
                epsr = fit_data.get("epsr", 0)
                if epsr > 0:
                    epsr_values.append(epsr)
                    if best_fit is None or abs(epsr - 3.15) < abs(best_fit.get("epsr", 999) - 3.15):
                        best_fit = fit_data
            except Exception:
                continue

    if not epsr_values or not best_fit:
        return E1Hyperbola(score=0.0, notes="No hyperbola fits available for scoring")

    epsr = best_fit["epsr"]
    ci = best_fit.get("epsr_ci95", [epsr * 0.8, epsr * 1.2])
    flags = best_fit.get("flags", [])

    # Score: how close is εr to ice range?
    ice_lo, ice_hi = params.epsr_ice_range
    if ice_lo <= epsr <= ice_hi:
        score = 0.9  # Strong ice-consistent
    elif epsr < ice_lo:
        # Below ice range — could be very porous ice or vacuum
        score = max(0.0, 0.9 - (ice_lo - epsr) / 2.0)
    else:
        # Above ice range — regolith or rock
        score = max(0.0, 0.9 - (epsr - ice_hi) / 3.0)

    # Reduce score for high clutter risk
    if "CLUTTER_RISK_HIGH" in flags:
        score *= 0.5
    elif "CLUTTER_RISK_MED" in flags:
        score *= 0.75

    notes = f"Hyperbola fit εr={epsr:.2f} (CI95: [{ci[0]:.2f}, {ci[1]:.2f}]). "
    if ice_lo <= epsr <= ice_hi:
        notes += "Ice-consistent dielectric — evidence supports subsurface water ice."
    elif epsr < ice_lo:
        notes += "Below typical ice range — possibly very porous ice or low-density regolith."
    else:
        notes += "Above ice range — suggests denser material (ice-cemented regolith or basalt)."

    return E1Hyperbola(
        score=round(score, 3),
        epsr=round(epsr, 3),
        ci=ci,
        flags=flags,
        notes=notes,
    )
