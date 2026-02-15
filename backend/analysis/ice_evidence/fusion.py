"""
Evidence fusion for multi-criteria ice probability.

Combines E1 (hyperbola εr), E2 (reflector), E3 (terrain), E4 (CRISM)
into a single ice_probability with uncertainty and explainability.
"""

from __future__ import annotations

import logging
import math
from typing import List

from .models import (
    ConsistencyInfo,
    EvidenceBreakdown,
    EvidenceParams,
    IceEvidenceResult,
    CandidateLocation,
    ArtifactPaths,
    E1Hyperbola,
    E2Reflector,
    E3Terrain,
    E4Crism,
)

logger = logging.getLogger(__name__)


def fuse_evidence(
    candidate: CandidateLocation,
    e1: E1Hyperbola,
    e2: E2Reflector,
    e3: E3Terrain,
    e4: E4Crism,
    params: EvidenceParams,
    json_path: str = "",
    png_paths: List[str] = None,
) -> IceEvidenceResult:
    """Fuse all evidence sources into a final ice probability score.

    Algorithm:
    1. Compute weighted sum S = Σ w_i * E_i
    2. Compute agreement score and detect conflicts
    3. Apply conflict penalty
    4. Apply distance penalty if science data is far
    5. Compute confidence from quality metrics

    SCIENCE SAFETY: Never claims "direct proof of ice".
    """
    w = params.weights
    scores = {
        "E1": e1.score,
        "E2": e2.score,
        "E3": e3.score,
        "E4": e4.score,
    }
    weights = {
        "E1": w.E1,
        "E2": w.E2,
        "E3": w.E3,
        "E4": w.E4,
    }

    # ── Step 1: Weighted sum ──
    weighted_sum = sum(scores[k] * weights[k] for k in scores)

    # ── Step 2: Agreement & conflicts ──
    agreement, conflicts = _compute_agreement(scores)

    # ── Step 3: Conflict penalty ──
    conflict_penalty = 0.0
    if conflicts:
        # Reduce by up to 0.2 depending on severity
        conflict_penalty = min(0.2, 0.05 * len(conflicts))

    # ── Step 4: Distance penalty ──
    dist_penalty = 0.0
    if e4.distance_km > params.distance_penalty_km and e4.score > 0:
        # Penalize if CRISM data is far from candidate
        excess_km = e4.distance_km - params.distance_penalty_km
        dist_penalty = min(0.1, excess_km / 500.0)

    # ── Final ice probability ──
    ice_prob = max(0.0, min(1.0, weighted_sum - conflict_penalty - dist_penalty))

    # ── Step 5: Confidence ──
    confidence = _compute_confidence(scores, e1, e2, e3, e4, agreement)

    return IceEvidenceResult(
        candidate_id=candidate.id,
        lat=candidate.lat,
        lon=candidate.lon,
        ice_probability=round(ice_prob, 3),
        confidence=round(confidence, 3),
        evidence=EvidenceBreakdown(
            E1_hyperbola=e1,
            E2_reflector=e2,
            E3_terrain=e3,
            E4_crism=e4,
        ),
        consistency=ConsistencyInfo(
            agreement_score=round(agreement, 3),
            conflicts=conflicts,
        ),
        artifacts=ArtifactPaths(
            json_path=json_path,
            optional_png_paths=png_paths or [],
        ),
    )


def _compute_agreement(scores: dict) -> tuple:
    """Compute agreement score and identify conflicts.

    High agreement: multiple strong signals agree.
    Conflicts: one strong high and another strong low.
    """
    vals = list(scores.values())
    keys = list(scores.keys())
    conflicts = []

    # Check pairwise conflicts (one >0.6, other <0.2)
    important_pairs = [
        ("E1", "E4"),  # Hyperbola vs CRISM
        ("E1", "E2"),  # Hyperbola vs Reflector
        ("E2", "E4"),  # Reflector vs CRISM
    ]

    for k1, k2 in important_pairs:
        s1, s2 = scores.get(k1, 0), scores.get(k2, 0)
        if s1 > 0.6 and s2 < 0.15:
            conflicts.append(f"{k1} strong ({s1:.2f}) but {k2} weak ({s2:.2f}) — inconsistent")
        elif s2 > 0.6 and s1 < 0.15:
            conflicts.append(f"{k2} strong ({s2:.2f}) but {k1} weak ({s1:.2f}) — inconsistent")

    # Agreement: measure concordance among non-zero sources
    active = [v for v in vals if v > 0.05]
    if len(active) >= 2:
        # Low variance among active sources = high agreement
        std = float(max(0, 1.0 - 2.0 * (max(active) - min(active))))
        # Bonus if multiple sources are high
        high_count = sum(1 for v in active if v > 0.4)
        bonus = min(0.3, high_count * 0.1)
        agreement = min(1.0, std + bonus)
    elif len(active) == 1:
        agreement = 0.3  # single source = low agreement
    else:
        agreement = 0.0

    # Penalize agreement for each conflict
    agreement = max(0.0, agreement - 0.15 * len(conflicts))

    return agreement, conflicts


def _compute_confidence(
    scores: dict,
    e1: E1Hyperbola, e2: E2Reflector,
    e3: E3Terrain, e4: E4Crism,
    agreement: float,
) -> float:
    """Compute confidence (0-1) based on data quality and agreement.

    Higher when: more evidence sources, higher SNR, tighter CI, better agreement.
    """
    factors = []

    # Agreement contributes
    factors.append(agreement)

    # Number of active evidence sources
    active = sum(1 for v in scores.values() if v > 0.05)
    factors.append(min(1.0, active / 3.0))

    # E1 quality: tighter CI = more confident
    if e1.ci and len(e1.ci) == 2 and e1.ci[1] > e1.ci[0]:
        ci_width = e1.ci[1] - e1.ci[0]
        ci_conf = max(0.0, 1.0 - ci_width / 3.0)  # width of 3 = 0 confidence
        factors.append(ci_conf)

    # E2 quality: SNR
    if e2.snr_stats:
        snr = e2.snr_stats.get("best_snr", 0)
        snr_conf = min(1.0, snr / 10.0)
        factors.append(snr_conf)

    if not factors:
        return 0.1

    return float(sum(factors) / len(factors))
