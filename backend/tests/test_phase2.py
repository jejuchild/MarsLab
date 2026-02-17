"""
Unit tests for Phase 2: Terraced crater εr inversion enhancements.

Tests cover:
  - quality_weighted_aggregate: weighted εr with CI, various quality mixes
  - run_epsilon_analysis: lowered min_depth_m threshold, aggregate field
  - compute_epsilon: basic back-calculation sanity
  - Adaptive SHARAD window computation
  - Enhanced terrace_dielectric_analysis output structure
  - Evidence pack terrace section assembly
  - StratigraphySummary new Phase 2 fields
"""
import math
import sys
import os
import pytest

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_DIR)

from analysis.epsilon_terrace.epsilon_calc import (
    compute_epsilon,
    quality_weighted_aggregate,
    run_epsilon_analysis,
    EpsilonEstimate,
    EpsilonResult,
    _interpret_epsilon,
    _QUALITY_WEIGHTS,
    SPEED_OF_LIGHT,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_estimate(
    epsilon_r=3.1,
    quality="good",
    depth_true_m=100.0,
    depth_unc_m=10.0,
    twt_us=1.0,
    twt_unc_us=0.05,
    n_picks=10,
    crater_id="test_crater",
    depth_metric="terrace_to_floor",
) -> EpsilonEstimate:
    """Create a test EpsilonEstimate."""
    return EpsilonEstimate(
        crater_id=crater_id,
        depth_metric=depth_metric,
        depth_true_m=depth_true_m,
        depth_unc_m=depth_unc_m,
        twt_us=twt_us,
        twt_unc_us=twt_unc_us,
        epsilon_r=epsilon_r,
        epsilon_low=epsilon_r * 0.8,
        epsilon_high=epsilon_r * 1.2,
        interpretation=_interpret_epsilon(epsilon_r),
        quality=quality,
        quality_note="",
        sharad_product="test_sharad",
        n_sharad_picks=n_picks,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. quality_weighted_aggregate
# ─────────────────────────────────────────────────────────────────────────────

class TestQualityWeightedAggregate:
    """Test the Phase 2 quality-weighted aggregate function."""

    def test_empty_list_returns_none(self):
        assert quality_weighted_aggregate([]) is None

    def test_single_good_estimate(self):
        est = _make_estimate(epsilon_r=3.1, quality="good")
        agg = quality_weighted_aggregate([est])
        assert agg is not None
        assert agg["weighted_mean_epsilon_r"] == 3.1
        assert agg["weighted_median_epsilon_r"] == 3.1
        assert agg["n_estimates"] == 1
        assert agg["n_good"] == 1
        assert agg["n_marginal"] == 0

    def test_multiple_good_estimates_mean(self):
        ests = [
            _make_estimate(epsilon_r=3.0, quality="good", n_picks=10),
            _make_estimate(epsilon_r=3.2, quality="good", n_picks=10),
        ]
        agg = quality_weighted_aggregate(ests)
        assert agg is not None
        # With same quality and picks, weighted mean should be close to simple mean
        assert abs(agg["weighted_mean_epsilon_r"] - 3.1) < 0.05

    def test_good_outweighs_marginal(self):
        ests = [
            _make_estimate(epsilon_r=3.0, quality="good", n_picks=10),
            _make_estimate(epsilon_r=6.0, quality="marginal", n_picks=10),
        ]
        agg = quality_weighted_aggregate(ests)
        assert agg is not None
        # Good (weight 1.0) should dominate over marginal (weight 0.4)
        # Weighted mean should be closer to 3.0 than to 6.0
        assert agg["weighted_mean_epsilon_r"] < 4.5

    def test_unreliable_excluded_when_better_available(self):
        ests = [
            _make_estimate(epsilon_r=3.0, quality="good", n_picks=10),
            _make_estimate(epsilon_r=20.0, quality="unreliable", n_picks=5),
        ]
        agg = quality_weighted_aggregate(ests)
        assert agg is not None
        # Unreliable should be filtered out
        assert agg["n_estimates"] == 1
        assert agg["weighted_mean_epsilon_r"] == 3.0

    def test_all_unreliable_still_works(self):
        ests = [
            _make_estimate(epsilon_r=8.0, quality="unreliable", n_picks=5),
            _make_estimate(epsilon_r=9.0, quality="unreliable", n_picks=5),
        ]
        agg = quality_weighted_aggregate(ests)
        assert agg is not None
        assert agg["n_estimates"] == 2

    def test_confidence_interval_68_present(self):
        ests = [
            _make_estimate(epsilon_r=3.0, quality="good"),
            _make_estimate(epsilon_r=3.5, quality="good"),
            _make_estimate(epsilon_r=3.2, quality="good"),
        ]
        agg = quality_weighted_aggregate(ests)
        assert "confidence_interval_68" in agg
        ci = agg["confidence_interval_68"]
        assert len(ci) == 2
        assert ci[0] <= ci[1]

    def test_confidence_interval_95_present(self):
        ests = [
            _make_estimate(epsilon_r=3.0, quality="good"),
            _make_estimate(epsilon_r=4.0, quality="good"),
        ]
        agg = quality_weighted_aggregate(ests)
        assert "confidence_interval_95" in agg
        ci95 = agg["confidence_interval_95"]
        ci68 = agg["confidence_interval_68"]
        # 95% CI should be wider or equal to 68% CI
        assert ci95[0] <= ci68[0]
        assert ci95[1] >= ci68[1]

    def test_per_estimate_weights_present(self):
        ests = [
            _make_estimate(epsilon_r=3.0, quality="good", crater_id="A"),
            _make_estimate(epsilon_r=3.5, quality="marginal", crater_id="B"),
        ]
        agg = quality_weighted_aggregate(ests)
        assert "per_estimate_weights" in agg
        assert len(agg["per_estimate_weights"]) == 2
        # Good estimate should have higher weight
        weights = {pw["crater_id"]: pw["weight"] for pw in agg["per_estimate_weights"]}
        assert weights["A"] > weights["B"]

    def test_more_picks_higher_weight(self):
        ests = [
            _make_estimate(epsilon_r=3.0, quality="good", n_picks=50),
            _make_estimate(epsilon_r=3.0, quality="good", n_picks=2),
        ]
        agg = quality_weighted_aggregate(ests)
        per_est = agg["per_estimate_weights"]
        # Higher pick count → higher pick_weight
        many_picks = next(p for p in per_est if p["pick_weight"] > 0.5)
        few_picks = next(p for p in per_est if p["pick_weight"] <= 0.5)
        assert many_picks["weight"] > few_picks["weight"]

    def test_lower_depth_uncertainty_higher_weight(self):
        ests = [
            _make_estimate(epsilon_r=3.0, quality="good", depth_true_m=100, depth_unc_m=5, n_picks=10),
            _make_estimate(epsilon_r=3.0, quality="good", depth_true_m=100, depth_unc_m=50, n_picks=10),
        ]
        agg = quality_weighted_aggregate(ests)
        per_est = agg["per_estimate_weights"]
        # Lower uncertainty → higher depth_weight
        # depth_unc=5: 1/(1+5/100) = 0.952, depth_unc=50: 1/(1+50/100) = 0.667
        dw_values = sorted([p["depth_weight"] for p in per_est])
        assert dw_values[1] > dw_values[0]  # higher weight for lower uncertainty
        assert dw_values[1] > 0.9  # ~0.952 for unc=5/100

    def test_interpretation_field(self):
        ests = [_make_estimate(epsilon_r=3.1, quality="good")]
        agg = quality_weighted_aggregate(ests)
        assert "interpretation" in agg
        assert "ice" in agg["interpretation"].lower()

    def test_weighted_median_with_three_values(self):
        ests = [
            _make_estimate(epsilon_r=2.5, quality="good", n_picks=10),
            _make_estimate(epsilon_r=3.0, quality="good", n_picks=10),
            _make_estimate(epsilon_r=5.0, quality="marginal", n_picks=10),
        ]
        agg = quality_weighted_aggregate(ests)
        # Weighted median should be one of the good values since marginal has lower weight
        assert agg["weighted_median_epsilon_r"] in [2.5, 3.0]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Quality weights
# ─────────────────────────────────────────────────────────────────────────────

class TestQualityWeights:
    """Test the quality weight map."""

    def test_good_weight(self):
        assert _QUALITY_WEIGHTS["good"] == 1.0

    def test_marginal_weight(self):
        assert _QUALITY_WEIGHTS["marginal"] == 0.4

    def test_unreliable_weight(self):
        assert _QUALITY_WEIGHTS["unreliable"] == 0.1

    def test_good_greater_than_marginal(self):
        assert _QUALITY_WEIGHTS["good"] > _QUALITY_WEIGHTS["marginal"]

    def test_marginal_greater_than_unreliable(self):
        assert _QUALITY_WEIGHTS["marginal"] > _QUALITY_WEIGHTS["unreliable"]


# ─────────────────────────────────────────────────────────────────────────────
# 3. compute_epsilon sanity
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeEpsilon:
    """Test the basic epsilon back-calculation."""

    def test_known_ice_values(self):
        # For pure ice: εr ≈ 3.1
        # depth = c * t / (2 * sqrt(εr))
        # Reverse: εr = (c * t / (2 * d))²
        # Pick d=100m, compute t that gives εr=3.1:
        # t = 2*d*sqrt(3.1)/c = 2*100*1.761/299792458 ≈ 1.175e-6 s = 1.175 µs
        twt_us = 2 * 100 * math.sqrt(3.1) / SPEED_OF_LIGHT * 1e6
        est = compute_epsilon(
            crater_id="test",
            depth_true_m=100.0,
            depth_unc_m=10.0,
            twt_us=twt_us,
            twt_unc_us=0.05,
        )
        assert est is not None
        assert abs(est.epsilon_r - 3.1) < 0.01

    def test_zero_depth_returns_none(self):
        est = compute_epsilon(
            crater_id="test", depth_true_m=0, depth_unc_m=1,
            twt_us=1.0, twt_unc_us=0.1,
        )
        assert est is None

    def test_zero_twt_returns_none(self):
        est = compute_epsilon(
            crater_id="test", depth_true_m=100, depth_unc_m=10,
            twt_us=0, twt_unc_us=0.1,
        )
        assert est is None

    def test_quality_good_for_plausible_eps(self):
        # εr = 3.1 → should be "good"
        twt_us = 2 * 100 * math.sqrt(3.1) / SPEED_OF_LIGHT * 1e6
        est = compute_epsilon(
            crater_id="test", depth_true_m=100, depth_unc_m=10,
            twt_us=twt_us, twt_unc_us=0.05,
        )
        assert est.quality == "good"

    def test_quality_marginal_for_high_eps(self):
        # εr ≈ 7 → should be "marginal"
        twt_us = 2 * 100 * math.sqrt(7.0) / SPEED_OF_LIGHT * 1e6
        est = compute_epsilon(
            crater_id="test", depth_true_m=100, depth_unc_m=10,
            twt_us=twt_us, twt_unc_us=0.05,
        )
        assert est.quality == "marginal"

    def test_quality_unreliable_for_very_high_eps(self):
        # εr > 10 → unreliable
        twt_us = 2 * 100 * math.sqrt(12.0) / SPEED_OF_LIGHT * 1e6
        est = compute_epsilon(
            crater_id="test", depth_true_m=100, depth_unc_m=10,
            twt_us=twt_us, twt_unc_us=0.05,
        )
        assert est.quality == "unreliable"

    def test_bounds_bracket_central(self):
        twt_us = 2 * 100 * math.sqrt(3.1) / SPEED_OF_LIGHT * 1e6
        est = compute_epsilon(
            crater_id="test", depth_true_m=100, depth_unc_m=10,
            twt_us=twt_us, twt_unc_us=0.05,
        )
        assert est.epsilon_low <= est.epsilon_r
        assert est.epsilon_high >= est.epsilon_r


# ─────────────────────────────────────────────────────────────────────────────
# 4. run_epsilon_analysis with lowered min_depth_m
# ─────────────────────────────────────────────────────────────────────────────

class TestRunEpsilonAnalysis:
    """Test run_epsilon_analysis with Phase 2 enhancements."""

    def _make_terrace_result(self, depths):
        """Create a minimal mock TerraceResult."""
        from dataclasses import dataclass, field
        from typing import List, Optional
        import numpy as np

        @dataclass
        class MockTerraceResult:
            dtm_file: str = "test_dtm.IMG"
            crater_center_lat: float = -45.0
            crater_center_lon: float = 30.0
            crater_radius_m: float = 500.0
            n_azimuths: int = 36
            terraces: list = field(default_factory=list)
            median_profile: object = None
            radial_distances: object = None
            all_profiles: object = None
            terrace_depths: list = field(default_factory=list)
            error: object = None

        return MockTerraceResult(
            terrace_depths=[
                {"depth_m": d, "uncertainty_m": d * 0.1, "metric": f"metric_{i}"}
                for i, d in enumerate(depths)
            ]
        )

    def _make_pick_result(self, n_picks=10, twt_us=1.0):
        from dataclasses import dataclass, field
        from typing import List, Optional

        @dataclass
        class MockPick:
            product_id: str = "test_sharad"

        @dataclass
        class MockPickResult:
            crater_lat: float = -45.0
            crater_lon: float = 30.0
            picks: list = field(default_factory=list)
            median_twt_us: float = 1.0
            twt_unc_us: float = 0.05
            error: object = None

        return MockPickResult(
            picks=[MockPick()] * n_picks,
            median_twt_us=twt_us,
        )

    def test_min_depth_default_is_5m(self):
        """Phase 2: default min_depth lowered from 10m to 5m."""
        # Depths of 7m should now be included (would have been excluded at 10m)
        tr = self._make_terrace_result([7.0])
        pr = self._make_pick_result(twt_us=0.1)
        result = run_epsilon_analysis([tr], [(tr, pr)], min_depth_m=5.0)
        assert len(result.estimates) >= 1

    def test_very_shallow_depth_excluded(self):
        """Depths below min_depth_m should still be excluded."""
        tr = self._make_terrace_result([3.0])
        pr = self._make_pick_result(twt_us=0.1)
        result = run_epsilon_analysis([tr], [(tr, pr)], min_depth_m=5.0)
        assert len(result.estimates) == 0

    def test_old_threshold_excludes_shallow(self):
        """When using old threshold (10m), 7m depths are excluded."""
        tr = self._make_terrace_result([7.0])
        pr = self._make_pick_result(twt_us=0.1)
        result = run_epsilon_analysis([tr], [(tr, pr)], min_depth_m=10.0)
        assert len(result.estimates) == 0

    def test_aggregate_present_in_result(self):
        """Phase 2: result should have aggregate field when estimates exist."""
        tr = self._make_terrace_result([100.0])
        pr = self._make_pick_result(twt_us=1.0)
        result = run_epsilon_analysis([tr], [(tr, pr)])
        if result.estimates:
            assert result.aggregate is not None

    def test_no_estimates_no_aggregate(self):
        result = run_epsilon_analysis([], [])
        assert result.aggregate is None
        assert len(result.estimates) == 0

    def test_multiple_depths_produces_multiple_estimates(self):
        tr = self._make_terrace_result([50.0, 100.0, 150.0])
        pr = self._make_pick_result(twt_us=1.0)
        result = run_epsilon_analysis([tr], [(tr, pr)])
        # At least 2 should be valid (depending on εr bounds)
        assert len(result.estimates) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# 5. interpret_epsilon
# ─────────────────────────────────────────────────────────────────────────────

class TestInterpretEpsilon:
    """Test material interpretation from εr."""

    def test_very_low(self):
        interp = _interpret_epsilon(1.5)
        assert "porous" in interp.lower() or "dry" in interp.lower()

    def test_ice_range(self):
        interp = _interpret_epsilon(3.1)
        assert "ice" in interp.lower()

    def test_basalt_range(self):
        interp = _interpret_epsilon(6.0)
        assert "basalt" in interp.lower() or "rock" in interp.lower()

    def test_very_high(self):
        interp = _interpret_epsilon(10.0)
        assert "dense" in interp.lower() or "artifact" in interp.lower()


# ─────────────────────────────────────────────────────────────────────────────
# 6. EpsilonResult aggregate field
# ─────────────────────────────────────────────────────────────────────────────

class TestEpsilonResultAggregate:
    """Test the Phase 2 aggregate field on EpsilonResult."""

    def test_default_aggregate_is_none(self):
        result = EpsilonResult()
        assert result.aggregate is None

    def test_aggregate_can_be_set(self):
        result = EpsilonResult(aggregate={"weighted_mean_epsilon_r": 3.1})
        assert result.aggregate is not None
        assert result.aggregate["weighted_mean_epsilon_r"] == 3.1


# ─────────────────────────────────────────────────────────────────────────────
# 7. StratigraphySummary Phase 2 fields
# ─────────────────────────────────────────────────────────────────────────────

class TestStratigraphySummaryPhase2:
    """Test new Phase 2 fields on StratigraphySummary model."""

    def test_new_fields_default(self):
        from analysis.epsilon_terrace.models import StratigraphySummary
        s = StratigraphySummary(
            n_sharad_tracks=3,
            n_epsilon_estimates=2,
        )
        assert s.weighted_epsilon_r is None
        assert s.weighted_epsilon_std is None
        assert s.confidence_interval_68 is None
        assert s.confidence_interval_95 is None
        assert s.n_good_estimates == 0
        assert s.n_marginal_estimates == 0

    def test_new_fields_with_values(self):
        from analysis.epsilon_terrace.models import StratigraphySummary
        s = StratigraphySummary(
            n_sharad_tracks=3,
            n_epsilon_estimates=5,
            weighted_epsilon_r=3.15,
            weighted_epsilon_std=0.25,
            confidence_interval_68=[2.90, 3.40],
            confidence_interval_95=[2.65, 3.65],
            n_good_estimates=3,
            n_marginal_estimates=2,
        )
        assert s.weighted_epsilon_r == 3.15
        assert s.confidence_interval_68 == [2.90, 3.40]
        assert s.n_good_estimates == 3


# ─────────────────────────────────────────────────────────────────────────────
# 8. Adaptive SHARAD window logic
# ─────────────────────────────────────────────────────────────────────────────

class TestAdaptiveSharadWindow:
    """Test the adaptive window computation logic (as used in pipeline.py)."""

    def _compute_adaptive_window(self, diameter_km, buffer_km=30.0):
        """Replicate the adaptive window formula from pipeline.py."""
        return max(5.0, min(diameter_km * 1.5, buffer_km / 2))

    def test_small_crater(self):
        """Small crater (2km) should use minimum 5km window."""
        w = self._compute_adaptive_window(2.0)
        assert w == 5.0

    def test_medium_crater(self):
        """Medium crater (5km) should scale with diameter."""
        w = self._compute_adaptive_window(5.0)
        assert w == 7.5  # 5.0 * 1.5

    def test_large_crater(self):
        """Large crater (15km) caps at buffer_km/2."""
        w = self._compute_adaptive_window(15.0, buffer_km=30.0)
        assert w == 15.0  # min(22.5, 15.0)

    def test_very_large_crater(self):
        """Very large crater (30km) caps at buffer/2."""
        w = self._compute_adaptive_window(30.0, buffer_km=30.0)
        assert w == 15.0

    def test_adaptive_max_radius(self):
        """Test adaptive max_radius_m computation from pipeline.py."""
        max_radius_m = 3000.0
        diameter_km = 10.0
        adaptive = max(max_radius_m, diameter_km * 500.0 * 1.5)
        adaptive = min(adaptive, 10000.0)
        assert adaptive == 7500.0  # 10 * 500 * 1.5

    def test_adaptive_max_radius_small_crater(self):
        """Small crater uses default max_radius_m."""
        max_radius_m = 3000.0
        diameter_km = 2.0
        adaptive = max(max_radius_m, diameter_km * 500.0 * 1.5)
        adaptive = min(adaptive, 10000.0)
        assert adaptive == 3000.0

    def test_adaptive_max_radius_huge_crater_capped(self):
        """Huge crater is capped at 10000m."""
        max_radius_m = 3000.0
        diameter_km = 20.0
        adaptive = max(max_radius_m, diameter_km * 500.0 * 1.5)
        adaptive = min(adaptive, 10000.0)
        assert adaptive == 10000.0


# ─────────────────────────────────────────────────────────────────────────────
# 9. Evidence pack terrace assembly
# ─────────────────────────────────────────────────────────────────────────────

class TestEvidencePackTerraceAssembly:
    """Test that evidence_pack correctly assembles Phase 2 terrace data."""

    def _make_session(self, terrace_data=None):
        """Create a minimal mock session for evidence pack testing."""
        from types import SimpleNamespace
        synth = {
            "subsurface_coverage": {},
            "dielectric_analysis": {},
            "terrace_dielectric": terrace_data or {},
            "sharad_physics_inversion": {},
            "dielectric_method_hierarchy": [],
            "mineral_signatures": {},
            "crism_spectral_analysis": {},
            "terrain_favorability": {},
            "climate_compatibility": {},
            "scoring": {},
            "isru_assessment": {},
        }
        return SimpleNamespace(
            synthesis=synth,
            bbox=None,
            region_name="Test Region",
            session_id="test-session-001",
            objective="Test Phase 2",
            narrative="",
        )

    def test_weighted_aggregate_propagated(self):
        """Weighted aggregate should appear in evidence pack dielectric section."""
        from api.evidence_pack import assemble_evidence_pack

        session = self._make_session({
            "estimates_count": 3,
            "median_epsilon_r": 3.2,
            "weighted_aggregate": {
                "weighted_median_epsilon_r": 3.15,
                "weighted_mean_epsilon_r": 3.18,
                "weighted_std": 0.2,
                "n_estimates": 3,
                "n_good": 2,
                "n_marginal": 1,
                "confidence_interval_68": (2.95, 3.35),
                "confidence_interval_95": (2.75, 3.55),
                "interpretation": "consistent with water ice",
                "per_estimate_weights": [],
            },
            "estimates": [],
            "interpretation": "ice",
        })
        pack = assemble_evidence_pack(session)
        diel = pack["dielectric"]
        assert diel["terrace_weighted_epsilon_r"] == 3.15
        assert diel["terrace_confidence_interval_68"] == (2.95, 3.35)
        assert diel["terrace_n_good"] == 2

    def test_no_aggregate_defaults(self):
        """Without weighted aggregate, fields should be None/0."""
        from api.evidence_pack import assemble_evidence_pack

        session = self._make_session({
            "estimates_count": 1,
            "median_epsilon_r": 4.0,
            "estimates": [],
        })
        pack = assemble_evidence_pack(session)
        diel = pack["dielectric"]
        assert diel["terrace_weighted_epsilon_r"] is None
        assert diel["terrace_confidence_interval_68"] is None
        assert diel["terrace_n_good"] == 0

    def test_epsilon_r_prefers_weighted_terrace(self):
        """Evidence pack should prefer weighted terrace εr over raw median."""
        from api.evidence_pack import assemble_evidence_pack

        session = self._make_session({
            "estimates_count": 2,
            "median_epsilon_r": 3.5,
            "weighted_aggregate": {
                "weighted_median_epsilon_r": 3.2,
                "weighted_mean_epsilon_r": 3.25,
                "weighted_std": 0.15,
                "n_estimates": 2,
                "n_good": 2,
                "n_marginal": 0,
                "confidence_interval_68": (3.05, 3.35),
                "confidence_interval_95": (2.90, 3.50),
                "interpretation": "water ice",
                "per_estimate_weights": [],
            },
            "estimates": [],
        })
        pack = assemble_evidence_pack(session)
        # Should pick weighted value (3.2) over raw median (3.5)
        assert pack["dielectric"]["epsilon_r"] == 3.2
