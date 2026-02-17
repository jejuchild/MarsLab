"""
Unit tests for Phase 1: ISRU-relevant depth logic.

Tests cover:
  - compute_depth_from_twt: TWT → depth conversion with physics-based εr
  - compute_isru_accessibility: ISRU tier classification, slope penalty
  - depth_unknown when no physics εr
  - Scoring integration: normalize_subsurface with isru_data
  - Ice purity classification
"""
import math
import sys
import os
import pytest

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_DIR)

from analysis.isru.accessibility import (
    compute_depth_from_twt,
    compute_isru_accessibility,
    _classify_ice_purity,
    _compute_slope_penalty,
    IsruAccessibility,
    _SPEED_OF_LIGHT,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. compute_depth_from_twt
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeDepthFromTwt:
    """Test TWT → depth conversion using physics-based εr."""

    def test_basic_depth(self):
        """Known case: εr=3.15, TWT=1.0 µs → depth ≈ 84.5 m."""
        depth, unc = compute_depth_from_twt(twt_us=1.0, epsilon_r=3.15)
        # velocity = c / sqrt(3.15) = 168,944,... m/s
        # depth = velocity * 1e-6 / 2 = ~84.5 m
        expected = _SPEED_OF_LIGHT * 1e-6 / (2.0 * math.sqrt(3.15))
        assert depth == pytest.approx(expected, rel=0.001)
        assert unc is None  # No sigma provided

    def test_depth_with_uncertainty(self):
        """Uncertainty propagation from εr sigma."""
        depth, unc = compute_depth_from_twt(
            twt_us=1.0, epsilon_r=3.0, epsilon_r_sigma=0.5,
        )
        assert depth > 0
        assert unc is not None
        assert unc > 0

    def test_depth_increases_with_twt(self):
        """Longer TWT → deeper reflector."""
        d1, _ = compute_depth_from_twt(1.0, 3.0)
        d2, _ = compute_depth_from_twt(2.0, 3.0)
        assert d2 > d1
        assert d2 == pytest.approx(2 * d1, rel=0.001)

    def test_depth_decreases_with_higher_epsilon(self):
        """Higher εr → lower velocity → shallower depth for same TWT."""
        d1, _ = compute_depth_from_twt(1.0, 2.0)  # Low εr (fast)
        d2, _ = compute_depth_from_twt(1.0, 5.0)  # High εr (slow)
        assert d1 > d2

    def test_epsilon_must_be_gt_1(self):
        with pytest.raises(ValueError, match="εr must be > 1.0"):
            compute_depth_from_twt(1.0, 0.5)

    def test_epsilon_exactly_1_raises(self):
        with pytest.raises(ValueError, match="εr must be > 1.0"):
            compute_depth_from_twt(1.0, 1.0)

    def test_twt_must_be_positive(self):
        with pytest.raises(ValueError, match="TWT must be > 0"):
            compute_depth_from_twt(0.0, 3.0)
        with pytest.raises(ValueError, match="TWT must be > 0"):
            compute_depth_from_twt(-1.0, 3.0)

    def test_uncertainty_zero_sigma(self):
        """Zero sigma → no uncertainty."""
        _, unc = compute_depth_from_twt(1.0, 3.0, epsilon_r_sigma=0.0)
        assert unc is None

    def test_uncertainty_proportional_to_sigma(self):
        """Larger εr sigma → larger depth uncertainty."""
        _, unc1 = compute_depth_from_twt(1.0, 3.0, epsilon_r_sigma=0.1)
        _, unc2 = compute_depth_from_twt(1.0, 3.0, epsilon_r_sigma=0.5)
        assert unc2 > unc1

    def test_formula_consistency(self):
        """Verify: depth = c * twt_s / (2 * sqrt(εr))."""
        twt_us = 2.5
        eps = 3.0
        depth, _ = compute_depth_from_twt(twt_us, eps)
        expected = _SPEED_OF_LIGHT * twt_us * 1e-6 / (2.0 * math.sqrt(eps))
        assert depth == pytest.approx(expected, abs=0.01)


# ─────────────────────────────────────────────────────────────────────────────
# 2. ISRU Accessibility Classification
# ─────────────────────────────────────────────────────────────────────────────

class TestIsruAccessibility:
    """Test compute_isru_accessibility() ISRU tiers and scoring."""

    def test_no_data_returns_depth_unknown(self):
        """No TWT and no εr → depth_unknown."""
        result = compute_isru_accessibility()
        assert result.accessibility_category == "depth_unknown"
        assert result.isru_tier == "unknown"
        assert result.depth_m is None
        assert result.accessibility_score == 0.0

    def test_twt_without_physics_epsilon(self):
        """TWT present but εr from assumed source → still depth_unknown."""
        result = compute_isru_accessibility(
            twt_us=1.0, epsilon_r=3.15,
            epsilon_r_source="assumed",
        )
        assert result.accessibility_category == "depth_unknown"
        assert result.depth_m is None

    def test_twt_with_physics_epsilon_shallow(self):
        """TWT + physics εr yielding depth < 5 m → tier_1, easy_excavation."""
        # Need TWT that gives depth < 5m with εr=3.0
        # depth = c * twt * 1e-6 / (2 * sqrt(3.0))
        # For depth=4m: twt = 4 * 2 * sqrt(3.0) / (c * 1e-6) ≈ 0.0462 µs
        twt = 4.0 * 2.0 * math.sqrt(3.0) / (_SPEED_OF_LIGHT * 1e-6)
        result = compute_isru_accessibility(
            twt_us=twt, epsilon_r=3.0,
            epsilon_r_source="physics_inversion",
        )
        assert result.depth_m is not None
        assert result.depth_m < 5.0
        assert result.accessibility_category == "easy_excavation"
        assert result.isru_tier == "tier_1"
        assert result.accessibility_score == pytest.approx(1.0, abs=0.01)

    def test_moderate_excavation(self):
        """Depth 5-10 m → tier_1, moderate_excavation, score 0.9."""
        twt = 7.0 * 2.0 * math.sqrt(3.0) / (_SPEED_OF_LIGHT * 1e-6)
        result = compute_isru_accessibility(
            twt_us=twt, epsilon_r=3.0,
            epsilon_r_source="physics_inversion",
        )
        assert 5.0 <= result.depth_m <= 10.0
        assert result.accessibility_category == "moderate_excavation"
        assert result.isru_tier == "tier_1"
        assert result.accessibility_score == pytest.approx(0.9, abs=0.01)

    def test_significant_excavation(self):
        """Depth 10-20 m → tier_2, significant_excavation."""
        twt = 15.0 * 2.0 * math.sqrt(3.0) / (_SPEED_OF_LIGHT * 1e-6)
        result = compute_isru_accessibility(
            twt_us=twt, epsilon_r=3.0,
            epsilon_r_source="terrace_dielectric",
        )
        assert 10.0 <= result.depth_m <= 20.0
        assert result.accessibility_category == "significant_excavation"
        assert result.isru_tier == "tier_2"
        assert 0.5 <= result.accessibility_score <= 0.8

    def test_major_infrastructure(self):
        """Depth 20-30 m → tier_3, major_infrastructure."""
        twt = 25.0 * 2.0 * math.sqrt(3.0) / (_SPEED_OF_LIGHT * 1e-6)
        result = compute_isru_accessibility(
            twt_us=twt, epsilon_r=3.0,
            epsilon_r_source="physics_inversion",
        )
        assert 20.0 <= result.depth_m <= 30.0
        assert result.accessibility_category == "major_infrastructure"
        assert result.isru_tier == "tier_3"

    def test_not_practical(self):
        """Depth > 30 m → not_suitable."""
        twt = 50.0 * 2.0 * math.sqrt(3.0) / (_SPEED_OF_LIGHT * 1e-6)
        result = compute_isru_accessibility(
            twt_us=twt, epsilon_r=3.0,
            epsilon_r_source="physics_inversion",
        )
        assert result.depth_m > 30.0
        assert result.accessibility_category == "not_practical"
        assert result.isru_tier == "not_suitable"

    def test_score_decreases_with_depth(self):
        """Accessibility score strictly decreases as depth increases."""
        scores = []
        for depth_target in [3, 7, 15, 25, 50]:
            twt = depth_target * 2.0 * math.sqrt(3.0) / (_SPEED_OF_LIGHT * 1e-6)
            result = compute_isru_accessibility(
                twt_us=twt, epsilon_r=3.0,
                epsilon_r_source="physics_inversion",
            )
            scores.append(result.accessibility_score)
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], \
                f"Score should decrease: {scores[i]} >= {scores[i+1]} at idx {i}"

    def test_crism_ice_note_when_no_depth(self):
        """If CRISM shows ice but no depth, note is included."""
        result = compute_isru_accessibility(
            ice_fraction_crism=0.05,
        )
        assert any("CRISM" in n for n in result.notes)

    def test_notes_populated(self):
        """Notes list is always populated."""
        result = compute_isru_accessibility()
        assert len(result.notes) > 0


# ─────────────────────────────────────────────────────────────────────────────
# 3. Slope Penalty
# ─────────────────────────────────────────────────────────────────────────────

class TestSlopePenalty:
    """Test slope-excavation coupling."""

    def test_flat_no_penalty(self):
        factor, stability = _compute_slope_penalty(3.0)
        assert factor == 1.0
        assert stability == "stable"

    def test_moderate_slope(self):
        factor, stability = _compute_slope_penalty(7.0)
        assert factor == 1.1
        assert stability == "stable"

    def test_marginal_slope_10_15(self):
        factor, stability = _compute_slope_penalty(12.0)
        assert factor == 1.3
        assert stability == "marginal"

    def test_marginal_slope_15_20(self):
        factor, stability = _compute_slope_penalty(17.0)
        assert factor == 1.6
        assert stability == "marginal"

    def test_unstable_slope(self):
        factor, stability = _compute_slope_penalty(25.0)
        assert factor == 2.5
        assert stability == "unstable"

    def test_slope_penalty_reduces_score(self):
        """Steep slopes should reduce ISRU accessibility score."""
        twt = 4.0 * 2.0 * math.sqrt(3.0) / (_SPEED_OF_LIGHT * 1e-6)
        r_flat = compute_isru_accessibility(
            twt_us=twt, epsilon_r=3.0,
            epsilon_r_source="physics_inversion",
            mean_slope_deg=2.0,
        )
        r_steep = compute_isru_accessibility(
            twt_us=twt, epsilon_r=3.0,
            epsilon_r_source="physics_inversion",
            mean_slope_deg=25.0,
        )
        assert r_flat.accessibility_score > r_steep.accessibility_score


# ─────────────────────────────────────────────────────────────────────────────
# 4. Ice Purity Classification
# ─────────────────────────────────────────────────────────────────────────────

class TestIcePurity:
    """Test εr → ice purity classification."""

    def test_none_unknown(self):
        assert _classify_ice_purity(None) == "unknown"

    def test_pure_ice(self):
        assert _classify_ice_purity(3.15) == "high_purity_ice"

    def test_low_epsilon_ice(self):
        assert _classify_ice_purity(2.0) == "high_purity_ice"

    def test_icy_regolith(self):
        assert _classify_ice_purity(4.0) == "icy_regolith"

    def test_rocky(self):
        assert _classify_ice_purity(7.0) == "rocky"

    def test_boundary_3_5(self):
        """3.5 is still high_purity_ice."""
        assert _classify_ice_purity(3.5) == "high_purity_ice"

    def test_boundary_5_0(self):
        """5.0 is still icy_regolith."""
        assert _classify_ice_purity(5.0) == "icy_regolith"


# ─────────────────────────────────────────────────────────────────────────────
# 5. Scoring Integration (normalize_subsurface with isru_data)
# ─────────────────────────────────────────────────────────────────────────────

from api.scoring_methodology import normalize_subsurface


class TestScoringIsruIntegration:
    """Test that normalize_subsurface uses ISRU data when available."""

    def test_no_isru_data_fallback(self):
        """Without isru_data, falls back to generic depth ladder."""
        score, bd = normalize_subsurface(
            subsurface_data={"analyzed_count": 5, "subsurface_detections": 3},
            dielectric_data={},
            cross_instrument_data={},
            isru_data=None,
        )
        assert bd["depth_category"] == "depth_unknown"
        assert bd["isru_tier"] is None

    def test_isru_tier_1_in_breakdown(self):
        """With physics-based ISRU data, breakdown shows tier."""
        isru = {
            "depth_m": 6.0,
            "depth_source": "physics_inversion",
            "accessibility_score": 0.9,
            "isru_tier": "tier_1",
            "accessibility_category": "moderate_excavation",
        }
        score, bd = normalize_subsurface(
            subsurface_data={"analyzed_count": 5, "subsurface_detections": 3},
            dielectric_data={},
            cross_instrument_data={},
            isru_data=isru,
        )
        assert bd["isru_tier"] == "tier_1"
        assert bd["isru_accessibility_score"] == 0.9
        assert "isru_" in bd["depth_category"]
        assert score >= 0.4  # Floor at 0.4 since reflectors detected

    def test_isru_score_used_as_base(self):
        """ISRU accessibility_score becomes the base score when physics-based."""
        isru = {
            "depth_m": 25.0,
            "depth_source": "physics_inversion",
            "accessibility_score": 0.3,
            "isru_tier": "tier_3",
            "accessibility_category": "major_infrastructure",
        }
        score, bd = normalize_subsurface(
            subsurface_data={"analyzed_count": 5, "subsurface_detections": 3},
            dielectric_data={},
            cross_instrument_data={},
            isru_data=isru,
        )
        # Floor is 0.4, so score should be 0.4 (not 0.3)
        assert bd["base_score"] == pytest.approx(0.4, abs=0.01)

    def test_assumed_epsilon_not_used_for_isru(self):
        """ISRU data with assumed source should not activate ISRU scoring."""
        isru = {
            "depth_m": None,
            "depth_source": "not_available",
            "accessibility_score": 0.0,
            "isru_tier": "unknown",
            "accessibility_category": "depth_unknown",
        }
        score, bd = normalize_subsurface(
            subsurface_data={"analyzed_count": 5, "subsurface_detections": 3},
            dielectric_data={},
            cross_instrument_data={},
            isru_data=isru,
        )
        assert bd["isru_tier"] is None
        assert bd["depth_category"] == "depth_unknown"


# ─────────────────────────────────────────────────────────────────────────────
# 6. IsruAccessibility dataclass
# ─────────────────────────────────────────────────────────────────────────────

class TestIsruAccessibilityDataclass:
    """Test the IsruAccessibility dataclass defaults."""

    def test_defaults(self):
        result = IsruAccessibility()
        assert result.depth_m is None
        assert result.accessibility_category == "depth_unknown"
        assert result.accessibility_score == 0.0
        assert result.isru_tier == "unknown"
        assert result.slope_penalty_factor == 1.0
        assert result.slope_stability == "stable"
        assert result.ice_purity_estimate == "unknown"
        assert result.notes == []

    def test_round_trip(self):
        """Verify all fields can be set."""
        result = IsruAccessibility(
            depth_m=5.5,
            depth_uncertainty_m=1.2,
            depth_source="physics_inversion",
            accessibility_category="moderate_excavation",
            accessibility_score=0.9,
            isru_tier="tier_1",
            slope_penalty_factor=1.1,
            slope_stability="stable",
            epsilon_r=3.0,
            ice_purity_estimate="high_purity_ice",
            notes=["Test note"],
        )
        assert result.depth_m == 5.5
        assert result.isru_tier == "tier_1"
        assert result.notes == ["Test note"]
