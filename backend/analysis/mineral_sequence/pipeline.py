"""
AqueousMineralMapper — transect-based mineral sequence analysis from CRISM CNN.

Algorithm:
  1. Load cached CNN classification result (mineral_map, confidence_map)
  2. Extract mineral class along a spatial transect (NS or EW) through the observation
  3. Map mineral class IDs → geochemical groups
  4. Detect group transitions along the transect
  5. Match observed group sequence against canonical paleo-environment patterns
  6. Compute histogram and summary statistics
"""

import logging
import numpy as np
from typing import Dict, List, Optional

from analysis.shared.base import AnalysisModule
from .models import (
    TransectPoint,
    MineralTransition,
    SequenceMatch,
    MineralSequenceSummary,
    MineralSequenceResult,
    MineralSequenceParameters,
)
from .taxonomy import group_for_class, match_sequence

logger = logging.getLogger(__name__)


class AqueousMineralMapper(AnalysisModule):
    """Analyze mineral sequences along a transect through a CRISM observation."""

    def __init__(self):
        self._result: Optional[MineralSequenceResult] = None

    # ────────────────────────────────────────────────────────────────
    # AnalysisModule interface
    # ────────────────────────────────────────────────────────────────

    def run(
        self,
        obs_id: str,
        transect_direction: str = "NS",
        transect_offset: float = 0.5,
    ) -> MineralSequenceResult:
        """Execute the mineral sequence analysis pipeline."""
        try:
            self._result = self._run_impl(obs_id, transect_direction, transect_offset)
        except Exception as exc:
            logger.exception("Mineral sequence pipeline failed for %s", obs_id)
            self._result = MineralSequenceResult(success=False, error=str(exc))
        return self._result

    def generate_profile(self) -> List[Dict]:
        if not self._result or not self._result.transect:
            return []
        return [t.model_dump() for t in self._result.transect]

    def generate_overlay(self) -> List[Dict]:
        # Mineral sequence doesn't have a map overlay; return empty
        return []

    def generate_summary(self) -> Dict:
        if not self._result:
            return {"success": False, "error": "Not run yet"}
        d: Dict = {"success": self._result.success, "error": self._result.error}
        if self._result.summary:
            d.update(self._result.summary.model_dump())
        return d

    # ────────────────────────────────────────────────────────────────
    # Core implementation
    # ────────────────────────────────────────────────────────────────

    def _run_impl(
        self,
        obs_id: str,
        transect_direction: str,
        transect_offset: float,
    ) -> MineralSequenceResult:
        from api.mineral_cnn.pipeline import load_cached_result, has_cached_result
        from api.mineral_cnn.constants import CLASS_NAME

        logger.info(
            "Mineral sequence: obs=%s direction=%s offset=%.2f",
            obs_id, transect_direction, transect_offset,
        )

        # ── Step 1: Load cached CNN result ────────────────────────
        if not has_cached_result(obs_id):
            return MineralSequenceResult(
                success=False,
                error=f"No CNN classification result cached for {obs_id}. Run classification first.",
            )

        cnn_result = load_cached_result(obs_id)
        mineral_map = cnn_result.mineral_map     # (rows, cols) int32
        confidence_map = cnn_result.confidence_map  # (rows, cols) float32
        valid_mask = cnn_result.valid_mask        # (rows, cols) bool
        rows, cols = cnn_result.rows, cnn_result.cols

        logger.info("Mineral sequence: loaded %dx%d mineral map", rows, cols)

        # ── Step 2: Extract transect ──────────────────────────────
        offset = max(0.0, min(1.0, transect_offset))

        if transect_direction.upper() == "EW":
            # East-West transect: fixed row, sweep columns
            fixed_row = int(offset * (rows - 1))
            transect_indices = [(fixed_row, c) for c in range(cols)]
        else:
            # North-South transect: fixed column, sweep rows
            fixed_col = int(offset * (cols - 1))
            transect_indices = [(r, fixed_col) for r in range(rows)]

        # ── Step 3: Sample transect ───────────────────────────────
        transect: List[TransectPoint] = []
        for idx, (r, c) in enumerate(transect_indices):
            if not valid_mask[r, c]:
                transect.append(TransectPoint(
                    position_idx=idx, row=r, col=c,
                ))
                continue

            mid = int(mineral_map[r, c])
            conf = float(confidence_map[r, c])

            # Skip unclassified (-1) and water-unrelated (100)
            if mid < 0 or mid == 100:
                transect.append(TransectPoint(
                    position_idx=idx, row=r, col=c,
                    mineral_id=mid if mid >= 0 else None,
                    confidence=round(conf, 3) if mid >= 0 else None,
                ))
                continue

            mineral_name = CLASS_NAME.get(mid, f"Class {mid}")
            geochem_group = group_for_class(mid)

            transect.append(TransectPoint(
                position_idx=idx, row=r, col=c,
                mineral_id=mid,
                mineral_name=mineral_name,
                geochem_group=geochem_group,
                confidence=round(conf, 3),
            ))

        # ── Step 4: Detect transitions ────────────────────────────
        transitions: List[MineralTransition] = []
        prev_group: Optional[str] = None
        prev_mineral: Optional[str] = None

        for pt in transect:
            if pt.geochem_group is not None:
                if prev_group is not None and pt.geochem_group != prev_group:
                    transitions.append(MineralTransition(
                        position_idx=pt.position_idx,
                        from_group=prev_group,
                        to_group=pt.geochem_group,
                        from_mineral=prev_mineral or "?",
                        to_mineral=pt.mineral_name or "?",
                    ))
                prev_group = pt.geochem_group
                prev_mineral = pt.mineral_name

        # ── Step 5: Build deduplicated group sequence ─────────────
        group_sequence: List[str] = []
        for pt in transect:
            if pt.geochem_group is not None:
                if not group_sequence or group_sequence[-1] != pt.geochem_group:
                    group_sequence.append(pt.geochem_group)

        # ── Step 6: Match against paleo-environments ──────────────
        matched_envs = match_sequence(group_sequence)

        # Compute confidence for sequence matches
        classified = [pt for pt in transect if pt.geochem_group is not None]
        n_classified = len(classified)
        n_total = len(transect)
        classification_rate = n_classified / max(n_total, 1)
        mean_conf = (
            float(np.mean([pt.confidence for pt in classified if pt.confidence is not None]))
            if classified else None
        )

        # Composite confidence: classification_rate * mean_conf
        composite_conf = (
            classification_rate * mean_conf
            if mean_conf is not None else 0.0
        )

        sequence_matches: List[SequenceMatch] = []
        for env_name in matched_envs:
            sequence_matches.append(SequenceMatch(
                environment=env_name,
                matched_groups=group_sequence,
                confidence=round(composite_conf, 3),
            ))

        # ── Step 7: Group histogram ──────────────────────────────
        group_histogram: Dict[str, int] = {}
        for pt in transect:
            if pt.geochem_group is not None:
                group_histogram[pt.geochem_group] = group_histogram.get(pt.geochem_group, 0) + 1

        # Dominant group
        dominant_group = max(group_histogram, key=group_histogram.get) if group_histogram else None

        # ── Step 8: Summary ──────────────────────────────────────
        summary = MineralSequenceSummary(
            obs_id=obs_id,
            total_transect_points=n_total,
            classified_points=n_classified,
            classification_rate=round(classification_rate, 4),
            n_transitions=len(transitions),
            dominant_group=dominant_group,
            n_groups_present=len(group_histogram),
            matched_environments=matched_envs,
            mean_confidence=round(mean_conf, 3) if mean_conf is not None else None,
        )

        params = MineralSequenceParameters(
            obs_id=obs_id,
            transect_direction=transect_direction.upper(),
            transect_offset=round(offset, 2),
        )

        logger.info(
            "Mineral sequence: done — %d classified, %d transitions, %d environments matched",
            n_classified, len(transitions), len(matched_envs),
        )

        return MineralSequenceResult(
            success=True,
            summary=summary,
            transect=transect,
            transitions=transitions,
            sequence_matches=sequence_matches,
            group_histogram=group_histogram,
            parameters=params,
        )
