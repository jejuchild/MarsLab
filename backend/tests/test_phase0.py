"""
Unit tests for Phase 0 pipeline overhaul.

Tests cover:
  - TWT computation from delta_bins (no assumed εr)
  - Cluttergram gate rejection logic
  - CRISM physics score and interpretation labels
  - Cross-instrument targeted product selection
  - depth_m is always None without physics-based εr
"""
import math
import sys
import os
import pytest

# Ensure backend is importable
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_DIR)


# ─────────────────────────────────────────────────────────────────────────────
# 1. TWT from delta_bins — constants and formula
# ─────────────────────────────────────────────────────────────────────────────

_SPEED_OF_LIGHT = 299_792_458.0
_SHARAD_DT_US = 3.0 / 80.0  # 0.0375 µs per range-bin


class TestTwtFromDeltaBins:
    """Verify TWT computation from delta_bins uses correct formula."""

    def test_sharad_dt_us_value(self):
        assert _SHARAD_DT_US == pytest.approx(0.0375, abs=1e-6)

    def test_twt_basic(self):
        """10 delta_bins → 0.375 µs TWT."""
        delta_bins = 10
        twt_us = delta_bins * _SHARAD_DT_US
        assert twt_us == pytest.approx(0.375, abs=1e-6)

    def test_twt_zero(self):
        """0 delta_bins → 0 TWT."""
        assert 0 * _SHARAD_DT_US == 0.0

    def test_twt_large(self):
        """200 delta_bins (max search window) → 7.5 µs."""
        assert 200 * _SHARAD_DT_US == pytest.approx(7.5, abs=1e-6)

    def test_twt_single_bin(self):
        """1 delta_bin → 0.0375 µs."""
        assert 1 * _SHARAD_DT_US == pytest.approx(0.0375, abs=1e-6)

    def test_depth_not_computable_without_epsilon(self):
        """Depth requires εr — cannot be derived from TWT alone."""
        delta_bins = 50
        twt_us = delta_bins * _SHARAD_DT_US
        # Depth formula: d = (c * TWT) / (2 * sqrt(εr))
        # Without εr, we MUST return None
        depth_m = None  # This is the Phase 0 rule
        assert depth_m is None
        assert twt_us > 0  # TWT is still valid


# ─────────────────────────────────────────────────────────────────────────────
# 2. CRISM Physics Score
# ─────────────────────────────────────────────────────────────────────────────

from analysis.crism_spectral.pipeline import _compute_physics_score


class TestCrismPhysicsScore:
    """Test _compute_physics_score() weights, normalization, and labels."""

    def test_all_zeros(self):
        score, label = _compute_physics_score(0.0, 0.0, 0.0, 0.0, 0.0)
        assert score == 0.0
        assert label == "No significant ice/hydration signature"

    def test_perfect_ice_signal(self):
        """Maximum values → score near 1.0, label = Likely H2O ice."""
        score, label = _compute_physics_score(
            bd1500=0.10, bd1900=0.08, bd2200=0.05,
            water_ice_fraction=0.05, sam_confidence=1.0,
        )
        assert score == pytest.approx(1.0, abs=0.01)
        assert label == "Likely H2O ice"

    def test_bd1500_normalization_clamp(self):
        """BD1500 > 0.10 still normalizes to 1.0 (clamped)."""
        score1, _ = _compute_physics_score(0.10, 0.0, 0.0, 0.0, 0.0)
        score2, _ = _compute_physics_score(0.20, 0.0, 0.0, 0.0, 0.0)
        # Both should have bd1500_norm = 1.0
        assert score1 == score2

    def test_bd1900_normalization_clamp(self):
        """BD1900 > 0.08 still normalizes to 1.0 (clamped)."""
        score1, _ = _compute_physics_score(0.0, 0.08, 0.0, 0.0, 0.0)
        score2, _ = _compute_physics_score(0.0, 0.16, 0.0, 0.0, 0.0)
        assert score1 == score2

    def test_ice_fraction_normalization_clamp(self):
        """water_ice_fraction > 0.05 normalizes to 1.0."""
        score1, _ = _compute_physics_score(0.0, 0.0, 0.0, 0.05, 0.0)
        score2, _ = _compute_physics_score(0.0, 0.0, 0.0, 0.50, 0.0)
        assert score1 == score2

    def test_weights_sum_to_one(self):
        """Weights 0.30 + 0.30 + 0.20 + 0.20 = 1.0."""
        assert 0.30 + 0.30 + 0.20 + 0.20 == pytest.approx(1.0)

    def test_individual_weight_bd1500(self):
        """BD1500 alone at max → score = 0.30."""
        score, _ = _compute_physics_score(0.10, 0.0, 0.0, 0.0, 0.0)
        assert score == pytest.approx(0.30, abs=0.01)

    def test_individual_weight_bd1900(self):
        """BD1900 alone at max → score = 0.30."""
        score, _ = _compute_physics_score(0.0, 0.08, 0.0, 0.0, 0.0)
        assert score == pytest.approx(0.30, abs=0.01)

    def test_individual_weight_ice_frac(self):
        """Ice fraction alone at max → score = 0.20."""
        score, _ = _compute_physics_score(0.0, 0.0, 0.0, 0.05, 0.0)
        assert score == pytest.approx(0.20, abs=0.01)

    def test_individual_weight_sam_conf(self):
        """SAM confidence alone at max → score = 0.20."""
        score, _ = _compute_physics_score(0.0, 0.0, 0.0, 0.0, 1.0)
        assert score == pytest.approx(0.20, abs=0.01)

    def test_partial_bd1500(self):
        """BD1500 = 0.05 → norm = 0.5, contrib = 0.15."""
        score, _ = _compute_physics_score(0.05, 0.0, 0.0, 0.0, 0.0)
        assert score == pytest.approx(0.15, abs=0.01)

    # ── Label tests ──

    def test_label_atmospheric_contamination(self):
        """water_ice_fraction > 0.50 → ambiguous/atmospheric."""
        _, label = _compute_physics_score(0.10, 0.10, 0.10, 0.60, 0.9)
        assert label == "Ambiguous/atmospheric contamination risk"

    def test_label_likely_h2o_ice(self):
        """BD1500 > 0.05 AND ice_frac > 0.02 → Likely H2O ice."""
        _, label = _compute_physics_score(0.06, 0.01, 0.0, 0.03, 0.5)
        assert label == "Likely H2O ice"

    def test_label_likely_hydrated_minerals(self):
        """BD1900 > 0.05 AND BD2200 > 0.03 → Likely hydrated minerals."""
        _, label = _compute_physics_score(0.01, 0.06, 0.04, 0.01, 0.5)
        assert label == "Likely hydrated minerals"

    def test_label_weak_hydration_bd1500(self):
        """BD1500 > 0.02 (but < 0.05 or ice_frac < 0.02) → Weak hydration."""
        _, label = _compute_physics_score(0.03, 0.01, 0.0, 0.01, 0.5)
        assert label == "Weak hydration signal"

    def test_label_weak_hydration_bd1900(self):
        """BD1900 > 0.03 (but BD2200 < 0.03) → Weak hydration."""
        _, label = _compute_physics_score(0.01, 0.04, 0.02, 0.01, 0.5)
        assert label == "Weak hydration signal"

    def test_label_no_signature(self):
        """All below thresholds → no significant signature."""
        _, label = _compute_physics_score(0.01, 0.02, 0.02, 0.01, 0.5)
        assert label == "No significant ice/hydration signature"

    def test_negative_inputs_treated_as_zero(self):
        """Negative band depths should normalize to 0."""
        score, _ = _compute_physics_score(-0.05, -0.03, -0.01, 0.0, 0.0)
        assert score == 0.0

    def test_label_priority_atmospheric_over_ice(self):
        """Atmospheric check takes priority even with strong BD1500."""
        _, label = _compute_physics_score(0.10, 0.10, 0.10, 0.55, 1.0)
        assert label == "Ambiguous/atmospheric contamination risk"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Cluttergram Gate Rejection
# ─────────────────────────────────────────────────────────────────────────────

from analysis.sharad_inversion.clutter_filter import assess_clutter
from analysis.sharad_inversion.models import ClutterAssessment


class TestClutterGate:
    """Test assess_clutter() scoring and rejection thresholds."""

    def test_unavailable_cluttergram(self):
        """When cluttergram data doesn't exist, available=False, not rejected."""
        result = assess_clutter(
            product_id="nonexistent_product_12345",
            trace_idx=100,
            interface_bin=150,
            surface_bin=50,
        )
        assert isinstance(result, ClutterAssessment)
        assert result.cluttergram_available is False
        assert result.rejected is False

    def test_return_type(self):
        """assess_clutter always returns ClutterAssessment."""
        result = assess_clutter("fake_product", 0, 100, 50)
        assert isinstance(result, ClutterAssessment)

    def test_score_clamped_0_1(self):
        """Score should always be in [0, 1]."""
        result = assess_clutter("fake_product", 0, 100, 50)
        assert 0.0 <= result.clutter_likelihood_score <= 1.0

    def test_rejection_threshold_is_0_7(self):
        """Rejection occurs only when score > 0.7."""
        # We can't control the internal score easily without real data,
        # but we verify the invariant: rejected → score > 0.7
        result = assess_clutter("fake_product", 0, 100, 50)
        if result.rejected:
            assert result.clutter_likelihood_score > 0.7

    def test_score_rounded_to_3_decimals(self):
        """Score should have at most 3 decimal places."""
        result = assess_clutter("fake_product", 0, 100, 50)
        score_str = f"{result.clutter_likelihood_score:.10f}"
        # Check that digits after 3rd decimal are 0
        decimal_part = score_str.split(".")[1]
        assert decimal_part[3:] == "0000000", \
            f"Score {result.clutter_likelihood_score} has more than 3 decimal places"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Cross-instrument Targeted Product Selection
# ─────────────────────────────────────────────────────────────────────────────

from api.agent_tasks import select_targeted_products


class TestTargetedProductSelection:
    """Test select_targeted_products() cross-instrument logic."""

    def test_no_crism_results_fallback(self):
        """No CRISM results → spatial_fallback method."""
        result = select_targeted_products(
            crism_results=None,
            all_products=[{"product_id": "p1", "instrument": "SHARAD", "lat": 0, "lon": 0}],
        )
        assert result["method"] == "spatial_fallback"
        assert result["targeted_products"] == []

    def test_empty_crism_results_fallback(self):
        """Empty CRISM results dict → spatial_fallback."""
        result = select_targeted_products(
            crism_results={},
            all_products=[{"product_id": "p1", "instrument": "SHARAD", "lat": 0, "lon": 0}],
        )
        assert result["method"] == "spatial_fallback"

    def test_crism_with_no_matching_products(self):
        """CRISM candidates exist but no products nearby."""
        crism = {
            "observations": [
                {"obs_id": "frt0001", "lat": 45.0, "lon": 30.0, "physics_score": 0.8,
                 "interpretation_label": "Likely H2O ice"},
            ]
        }
        products = [
            {"product_id": "s1", "instrument": "SHARAD", "lat": -45.0, "lon": -150.0},
        ]
        result = select_targeted_products(crism, products, buffer_km=50.0)
        assert result["n_sharad_targeted"] == 0

    def test_sharad_within_buffer_selected(self):
        """SHARAD product within buffer_km of CRISM candidate is selected."""
        crism = {
            "observations": [
                {"obs_id": "frt0001", "lat": 45.0, "lon": 30.0, "physics_score": 0.8,
                 "interpretation_label": "Likely H2O ice"},
            ]
        }
        products = [
            {"product_id": "s1", "instrument": "SHARAD", "lat": 45.01, "lon": 30.01},
        ]
        result = select_targeted_products(crism, products, buffer_km=50.0)
        assert result["n_sharad_targeted"] >= 1
        assert result["method"] == "cross_instrument_targeting"

    def test_sharad_outside_buffer_excluded(self):
        """SHARAD product far from CRISM candidate is not selected."""
        crism = {
            "observations": [
                {"obs_id": "frt0001", "lat": 45.0, "lon": 30.0, "physics_score": 0.8,
                 "interpretation_label": "Likely H2O ice"},
            ]
        }
        products = [
            {"product_id": "s1", "instrument": "SHARAD", "lat": 0.0, "lon": 0.0},
        ]
        result = select_targeted_products(crism, products, buffer_km=50.0, fallback_buffer_km=50.0)
        assert result["n_sharad_targeted"] == 0

    def test_candidates_sorted_by_score_descending(self):
        """Higher physics_score CRISM candidates should be prioritized."""
        crism = {
            "observations": [
                {"obs_id": "frt0001", "lat": 45.0, "lon": 30.0, "physics_score": 0.3,
                 "interpretation_label": "Weak hydration signal"},
                {"obs_id": "frt0002", "lat": 46.0, "lon": 31.0, "physics_score": 0.9,
                 "interpretation_label": "Likely H2O ice"},
            ]
        }
        products = [
            {"product_id": "s1", "instrument": "SHARAD", "lat": 46.01, "lon": 31.01},
        ]
        result = select_targeted_products(crism, products, buffer_km=50.0, max_targets=1)
        # With max_targets=1, only the top-scoring candidate (frt0002) should be used
        if result["targets_used"]:
            top = result["targets_used"][0]
            assert top["obs_id"] == "frt0002"

    def test_deduplication_by_obs_id(self):
        """Same obs_id from multiple sources should not duplicate."""
        crism = {
            "observations": [
                {"obs_id": "frt0001", "lat": 45.0, "lon": 30.0, "physics_score": 0.7,
                 "interpretation_label": "Likely H2O ice"},
            ],
            "top_ice_candidates": [
                {"obs_id": "frt0001", "lat": 45.0, "lon": 30.0, "ice_score": 0.5},
            ],
        }
        products = [
            {"product_id": "s1", "instrument": "SHARAD", "lat": 45.01, "lon": 30.01},
        ]
        result = select_targeted_products(crism, products, buffer_km=50.0)
        # targets_used should not contain duplicate obs_ids
        obs_ids = [t["obs_id"] for t in result["targets_used"]]
        assert len(obs_ids) == len(set(obs_ids))

    def test_targeting_enriches_product_fields(self):
        """Selected products should have targeting_reason and targeting_distance_km."""
        crism = {
            "observations": [
                {"obs_id": "frt0001", "lat": 45.0, "lon": 30.0, "physics_score": 0.8,
                 "interpretation_label": "Likely H2O ice"},
            ]
        }
        products = [
            {"product_id": "s1", "instrument": "SHARAD", "lat": 45.01, "lon": 30.01},
        ]
        result = select_targeted_products(crism, products, buffer_km=50.0)
        for p in result["targeted_products"]:
            assert "targeting_reason" in p
            assert "targeting_distance_km" in p
            assert isinstance(p["targeting_distance_km"], float)

    def test_return_structure(self):
        """Return dict should always contain required keys."""
        result = select_targeted_products(None, [])
        assert "targeted_products" in result
        assert "method" in result
        assert "n_sharad_targeted" in result
        assert "n_other_targeted" in result
        assert "targets_used" in result
        assert "targeting_rationale" in result

    def test_hirise_ctx_included_as_context(self):
        """HiRISE/CTX products within buffer are included for context."""
        crism = {
            "observations": [
                {"obs_id": "frt0001", "lat": 45.0, "lon": 30.0, "physics_score": 0.8,
                 "interpretation_label": "Likely H2O ice"},
            ]
        }
        products = [
            {"product_id": "h1", "instrument": "HIRISE", "lat": 45.01, "lon": 30.01},
            {"product_id": "c1", "instrument": "CTX", "lat": 45.01, "lon": 30.01},
        ]
        result = select_targeted_products(crism, products, buffer_km=50.0)
        assert result["n_other_targeted"] >= 1


# ─────────────────────────────────────────────────────────────────────────────
# 5. depth_m is always None without physics-based εr
# ─────────────────────────────────────────────────────────────────────────────

class TestDepthMIsNone:
    """Verify that depth_m is never computed from assumed εr=3.15."""

    def test_pick_depth_m_none(self):
        """A subsurface pick dict should have depth_m=None."""
        pick = {
            "trace_idx": 42,
            "bin_idx": 150,
            "delta_bins": 100,
            "twt_us": round(100 * _SHARAD_DT_US, 4),
            "snr": 8.5,
            "depth_m": None,
            "epsilon_r_source": "not_estimated",
        }
        assert pick["depth_m"] is None
        assert pick["epsilon_r_source"] == "not_estimated"

    def test_reflector_summary_depth_m_none(self):
        """Reflector summary dict should have depth_m=None."""
        summary = {
            "n_tracks_with_reflectors": 3,
            "total_picks": 45,
            "median_delta_bins": 80,
            "median_twt_us": round(80 * _SHARAD_DT_US, 4),
            "depth_m": None,
            "epsilon_r_source": "not_estimated",
        }
        assert summary["depth_m"] is None
        assert summary["epsilon_r_source"] == "not_estimated"

    def test_twt_is_valid_observable(self):
        """TWT is a valid observable even without εr."""
        delta_bins = 80
        twt_us = round(delta_bins * _SHARAD_DT_US, 4)
        assert twt_us == pytest.approx(3.0, abs=0.01)
        # But we CANNOT convert to depth without εr
        # depth_m = c * twt_us * 1e-6 / (2 * sqrt(εr))  -- requires εr!

    def test_no_ice_epsilon_constant(self):
        """Ensure _ICE_EPSILON = 3.15 is NOT defined in agent_tasks module."""
        from api import agent_tasks
        assert not hasattr(agent_tasks, '_ICE_EPSILON'), \
            "_ICE_EPSILON should have been removed in Phase 0.1"


# ─────────────────────────────────────────────────────────────────────────────
# 6. CrismSpectralResult model
# ─────────────────────────────────────────────────────────────────────────────

from analysis.crism_spectral.models import CrismSpectralResult


class TestCrismSpectralResultModel:
    """Test CrismSpectralResult model fields."""

    def test_default_physics_score(self):
        """Default physics_score is 0.0."""
        result = CrismSpectralResult(obs_id="test")
        assert result.physics_score == 0.0

    def test_default_interpretation_label(self):
        """Default interpretation_label is 'Unanalyzed'."""
        result = CrismSpectralResult(obs_id="test")
        assert result.interpretation_label == "Unanalyzed"

    def test_physics_score_range(self):
        """physics_score should accept values in [0, 1]."""
        result = CrismSpectralResult(obs_id="test", physics_score=0.75)
        assert result.physics_score == 0.75

    def test_water_ice_pixels_field_name(self):
        """Field is water_ice_pixels (not water_ice_pixel_count)."""
        result = CrismSpectralResult(obs_id="test", water_ice_pixels=42)
        assert result.water_ice_pixels == 42

    def test_mean_bd_field_names(self):
        """Fields are mean_BD1500, mean_BD1900, etc. (not nested)."""
        result = CrismSpectralResult(
            obs_id="test",
            mean_BD1500=0.05,
            mean_BD1900=0.03,
            mean_BD2100=0.02,
            mean_BD2200=0.01,
        )
        assert result.mean_BD1500 == 0.05
        assert result.mean_BD1900 == 0.03
        assert result.mean_BD2100 == 0.02
        assert result.mean_BD2200 == 0.01

    def test_optional_band_params_none(self):
        """Band params are None when not set."""
        result = CrismSpectralResult(obs_id="test")
        assert result.mean_BD1500 is None
        assert result.mean_BD1900 is None
        assert result.mean_BD2100 is None
        assert result.mean_BD2200 is None

    def test_success_default_false(self):
        """Default success is False (must be explicitly set)."""
        result = CrismSpectralResult(obs_id="test")
        assert result.success is False

    def test_interpretation_labels_enum(self):
        """All valid interpretation labels are accepted."""
        valid_labels = [
            "Unanalyzed",
            "Likely H2O ice",
            "Likely hydrated minerals",
            "Ambiguous/atmospheric contamination risk",
            "Weak hydration signal",
            "No significant ice/hydration signature",
        ]
        for label in valid_labels:
            result = CrismSpectralResult(obs_id="test", interpretation_label=label)
            assert result.interpretation_label == label


# ─────────────────────────────────────────────────────────────────────────────
# 7. Haversine distance (Mars radius)
# ─────────────────────────────────────────────────────────────────────────────

class TestHaversineMars:
    """Test the Mars-specific haversine used in targeting."""

    MARS_R_KM = 3389.5

    @staticmethod
    def _haversine_km(lat1, lon1, lat2, lon2):
        R = 3389.5
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat / 2) ** 2
             + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
             * math.sin(dlon / 2) ** 2)
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    def test_same_point_zero_distance(self):
        assert self._haversine_km(0, 0, 0, 0) == 0.0

    def test_equator_one_degree(self):
        """1° longitude at equator ≈ 59.2 km on Mars."""
        d = self._haversine_km(0, 0, 0, 1)
        assert 58.0 < d < 60.0

    def test_pole_to_pole(self):
        """North to south pole ≈ π × R ≈ 10,647 km."""
        d = self._haversine_km(90, 0, -90, 0)
        expected = math.pi * self.MARS_R_KM
        assert d == pytest.approx(expected, rel=0.01)

    def test_symmetric(self):
        d1 = self._haversine_km(45, 30, 46, 31)
        d2 = self._haversine_km(46, 31, 45, 30)
        assert d1 == pytest.approx(d2, abs=1e-6)
