"""
Unit tests for Phase 4: Climate + Thermal Inertia integration.

Tests cover:
  - Ice thermodynamic stability (Clausius-Clapeyron, vapor pressure)
  - Seasonal operation window computation
  - TI-εr cross-correlation classification
  - Climate-ice compatibility assessment
  - TI modifier in climate scoring
  - Evidence pack Phase 4 fields
  - Climate region analysis with Phase 4 outputs
"""
import math
import sys
import os
import pytest

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_DIR)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Ice Stability (Clausius-Clapeyron)
# ─────────────────────────────────────────────────────────────────────────────

class TestIceStability:
    """Test compute_ice_stability from mars_climate.py."""

    def test_polar_latitude_stable(self):
        """Ice should be stable at polar latitudes."""
        from api.mars_climate import compute_ice_stability
        result = compute_ice_stability(lat_deg=70.0)
        assert result["ice_table_stable"] is True
        assert result["sublimation_regime"] == "stable"
        assert result["stability_margin"] >= 1.0

    def test_equatorial_latitude_unstable(self):
        """Ice should be unstable near the equator."""
        from api.mars_climate import compute_ice_stability
        result = compute_ice_stability(lat_deg=5.0)
        assert result["ice_table_stable"] is False
        assert result["sublimation_regime"] == "sublimating"

    def test_mid_latitude_marginal_or_stable(self):
        """Mid-latitudes (~45°) may be marginal or stable."""
        from api.mars_climate import compute_ice_stability
        result = compute_ice_stability(lat_deg=45.0)
        assert result["sublimation_regime"] in ("stable", "marginal")

    def test_annual_mean_temp_reasonable(self):
        """Annual mean temperature should be in plausible range."""
        from api.mars_climate import compute_ice_stability
        result = compute_ice_stability(lat_deg=30.0)
        assert 150 < result["annual_mean_temp_k"] < 230

    def test_vapor_pressure_positive(self):
        """Equilibrium vapor pressure should always be positive."""
        from api.mars_climate import compute_ice_stability
        result = compute_ice_stability(lat_deg=50.0)
        assert result["equilibrium_vapor_pressure_pa"] > 0

    def test_atmospheric_h2o_positive(self):
        """Atmospheric H2O partial pressure should be positive."""
        from api.mars_climate import compute_ice_stability
        result = compute_ice_stability(lat_deg=50.0)
        assert result["atmospheric_h2o_pa"] > 0

    def test_ice_table_depth_polar_shallow(self):
        """Ice table at polar latitudes should be very shallow."""
        from api.mars_climate import compute_ice_stability
        result = compute_ice_stability(lat_deg=75.0)
        if result["estimated_ice_table_depth_m"] is not None:
            assert result["estimated_ice_table_depth_m"] < 1.0

    def test_ice_table_depth_none_when_sublimating(self):
        """Ice table depth should be None when ice is sublimating."""
        from api.mars_climate import compute_ice_stability
        result = compute_ice_stability(lat_deg=0.0)
        if result["sublimation_regime"] == "sublimating":
            assert result["estimated_ice_table_depth_m"] is None

    def test_output_fields_present(self):
        """All expected output fields should be present."""
        from api.mars_climate import compute_ice_stability
        result = compute_ice_stability(lat_deg=50.0)
        assert "ice_table_stable" in result
        assert "annual_mean_temp_k" in result
        assert "equilibrium_vapor_pressure_pa" in result
        assert "atmospheric_h2o_pa" in result
        assert "stability_margin" in result
        assert "estimated_ice_table_depth_m" in result
        assert "sublimation_regime" in result

    def test_elevation_effect(self):
        """Higher elevation should be colder, potentially more stable."""
        from api.mars_climate import compute_ice_stability
        low = compute_ice_stability(lat_deg=50.0, elevation_m=0)
        high = compute_ice_stability(lat_deg=50.0, elevation_m=-3000)
        # Lower elevation (less negative) = warmer = less stable
        assert high["annual_mean_temp_k"] >= low["annual_mean_temp_k"]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Water Vapor Pressure Function
# ─────────────────────────────────────────────────────────────────────────────

class TestWaterVaporPressure:
    """Test the Clausius-Clapeyron vapor pressure function."""

    def test_triple_point(self):
        """At triple point temperature, VP should equal triple point pressure."""
        from api.mars_climate import _water_vapor_pressure_pa, _TRIPLE_POINT_T_K, _TRIPLE_POINT_P_PA
        vp = _water_vapor_pressure_pa(_TRIPLE_POINT_T_K)
        assert abs(vp - _TRIPLE_POINT_P_PA) < 0.01

    def test_zero_temp(self):
        """VP at 0 K should be 0."""
        from api.mars_climate import _water_vapor_pressure_pa
        assert _water_vapor_pressure_pa(0.0) == 0.0

    def test_increases_with_temperature(self):
        """VP should increase with temperature."""
        from api.mars_climate import _water_vapor_pressure_pa
        vp_low = _water_vapor_pressure_pa(180.0)
        vp_high = _water_vapor_pressure_pa(220.0)
        assert vp_high > vp_low

    def test_mars_range_very_low(self):
        """VP at Mars temperatures (~180-220 K) should be very small."""
        from api.mars_climate import _water_vapor_pressure_pa
        vp = _water_vapor_pressure_pa(200.0)
        assert vp < 1.0  # Much less than 1 Pa at Mars temps


# ─────────────────────────────────────────────────────────────────────────────
# 3. Seasonal Operation Window
# ─────────────────────────────────────────────────────────────────────────────

class TestSeasonalOperationWindow:
    """Test compute_seasonal_operation_window."""

    def test_equatorial_many_safe_bins(self):
        """Equatorial regions should have many safe operation bins."""
        from api.mars_climate import compute_seasonal_operation_window
        result = compute_seasonal_operation_window(lat_deg=10.0)
        assert result["n_safe_bins"] >= 6  # At least half the year

    def test_polar_fewer_safe_bins(self):
        """High polar latitudes should have fewer safe bins (frost, cold)."""
        from api.mars_climate import compute_seasonal_operation_window
        result = compute_seasonal_operation_window(lat_deg=75.0)
        assert result["n_safe_bins"] < 12  # Not all bins safe

    def test_operational_fraction_range(self):
        """Operational fraction should be between 0 and 1."""
        from api.mars_climate import compute_seasonal_operation_window
        result = compute_seasonal_operation_window(lat_deg=40.0)
        assert 0 <= result["operational_fraction"] <= 1.0

    def test_output_fields_present(self):
        """All expected output fields should be present."""
        from api.mars_climate import compute_seasonal_operation_window
        result = compute_seasonal_operation_window(lat_deg=40.0)
        assert "safe_bins" in result
        assert "n_safe_bins" in result
        assert "total_bins" in result
        assert "operational_fraction" in result
        assert "best_season_ls" in result
        assert "worst_season_ls" in result
        assert "constraints" in result
        assert "bin_scores" in result

    def test_bin_scores_length(self):
        """Should have 12 bin scores (one per 30° Ls)."""
        from api.mars_climate import compute_seasonal_operation_window
        result = compute_seasonal_operation_window(lat_deg=30.0)
        assert len(result["bin_scores"]) == 12

    def test_best_season_score_gte_worst(self):
        """Best season score should be >= worst season score."""
        from api.mars_climate import compute_seasonal_operation_window
        result = compute_seasonal_operation_window(lat_deg=30.0)
        assert result["best_season_score"] >= result["worst_season_score"]


# ─────────────────────────────────────────────────────────────────────────────
# 4. TI-εr Cross-Correlation
# ─────────────────────────────────────────────────────────────────────────────

class TestTiEpsilonCorrelation:
    """Test the TI-εr cross-correlation in _assess_climate_ice_compatibility."""

    def _assess(self, ti_median=None, epsilon_r=None, **kwargs):
        from api.agent_tasks import _assess_climate_ice_compatibility
        climate = kwargs.get("climate", {
            "ice_stability": {"sublimation_regime": "stable", "stability_margin": 5.0},
            "seasonal_operation_window": {"n_safe_bins": 10, "total_bins": 12,
                                          "operational_fraction": 0.83, "constraints": []},
        })
        ti = {"ti_median": ti_median} if ti_median is not None else {}
        dielectric = {"best_epsilon_r": epsilon_r} if epsilon_r is not None else {}
        return _assess_climate_ice_compatibility(
            climate=climate,
            thermal_inertia=ti,
            dielectric=dielectric,
            terrace_diel={},
            cross_val={},
            mineral=kwargs.get("mineral", {}),
        )

    def test_high_ti_low_eps_strongly_corroborated(self):
        result = self._assess(ti_median=350, epsilon_r=3.0)
        assert result["ti_epsilon_correlation"]["correlation"] == "strongly_corroborated"

    def test_high_ti_high_eps_consolidated_rock(self):
        result = self._assess(ti_median=400, epsilon_r=7.0)
        assert result["ti_epsilon_correlation"]["correlation"] == "consolidated_rock"

    def test_mid_ti_low_eps_moderate_support(self):
        result = self._assess(ti_median=200, epsilon_r=3.5)
        assert result["ti_epsilon_correlation"]["correlation"] == "moderate_support"

    def test_low_ti_low_eps_uncertain(self):
        result = self._assess(ti_median=100, epsilon_r=3.0)
        assert result["ti_epsilon_correlation"]["correlation"] == "uncertain_dust_covered"

    def test_low_ti_high_eps_no_ice(self):
        result = self._assess(ti_median=80, epsilon_r=8.0)
        assert result["ti_epsilon_correlation"]["correlation"] == "no_ice_indication"

    def test_no_ti_unavailable(self):
        result = self._assess(ti_median=None, epsilon_r=3.0)
        assert result["ti_epsilon_correlation"]["correlation"] == "unavailable"

    def test_no_eps_unavailable(self):
        result = self._assess(ti_median=300, epsilon_r=None)
        assert result["ti_epsilon_correlation"]["correlation"] == "unavailable"

    def test_interpretation_populated(self):
        result = self._assess(ti_median=350, epsilon_r=3.0)
        assert len(result["ti_epsilon_correlation"]["interpretation"]) > 10


# ─────────────────────────────────────────────────────────────────────────────
# 5. Climate-Ice Compatibility Overall
# ─────────────────────────────────────────────────────────────────────────────

class TestClimateIceCompatibility:
    """Test the overall climate-ice compatibility assessment."""

    def _assess(self, regime="stable", margin=5.0, ti_median=350, eps=3.0,
                op_frac=0.83, ice_pixels=5):
        from api.agent_tasks import _assess_climate_ice_compatibility
        return _assess_climate_ice_compatibility(
            climate={
                "ice_stability": {
                    "sublimation_regime": regime,
                    "stability_margin": margin,
                    "estimated_ice_table_depth_m": 0.5,
                },
                "seasonal_operation_window": {
                    "n_safe_bins": int(op_frac * 12),
                    "total_bins": 12,
                    "operational_fraction": op_frac,
                    "constraints": [],
                },
            },
            thermal_inertia={"ti_median": ti_median},
            dielectric={"best_epsilon_r": eps},
            terrace_diel={},
            cross_val={},
            mineral={"ice_pixel_count": ice_pixels},
        )

    def test_ideal_conditions_highly_compatible(self):
        result = self._assess(regime="stable", ti_median=400, eps=3.0, op_frac=0.9)
        assert result["overall_compatibility"] == "highly_compatible"
        assert result["compatibility_score"] >= 0.7

    def test_no_ice_no_stability_incompatible(self):
        result = self._assess(regime="sublimating", margin=0.1, ti_median=80,
                              eps=8.0, op_frac=0.3, ice_pixels=0)
        assert result["overall_compatibility"] in ("incompatible", "weakly_compatible")
        assert result["compatibility_score"] < 0.4

    def test_marginal_stability_moderate(self):
        result = self._assess(regime="marginal", margin=0.7, ti_median=200,
                              eps=4.0, op_frac=0.6)
        assert result["overall_compatibility"] in ("moderately_compatible", "weakly_compatible")

    def test_notes_populated(self):
        result = self._assess()
        assert len(result["notes"]) > 0

    def test_assessed_flag(self):
        result = self._assess()
        assert result["assessed"] is True

    def test_empty_inputs_not_assessed(self):
        from api.agent_tasks import _assess_climate_ice_compatibility
        result = _assess_climate_ice_compatibility(
            climate={}, thermal_inertia={}, dielectric={},
            terrace_diel={}, cross_val={}, mineral={},
        )
        assert result["assessed"] is False


# ─────────────────────────────────────────────────────────────────────────────
# 6. TI Modifier in Climate Scoring
# ─────────────────────────────────────────────────────────────────────────────

class TestTiScoringModifier:
    """Test Phase 4 TI modifier in normalize_climate."""

    def _score(self, ti_median=None, has_ice=False):
        from api.scoring_methodology import normalize_climate
        climate_data = {
            "annual_stats": {
                "temp_mean_k": 200,  # Good temperature
                "dust_tau_mean": 0.3,  # Low dust
                "wind_mean_ms": 5.0,  # Low wind
                "frost_max_probability": 0.02,  # No frost
            }
        }
        ti_data = {"ti_median": ti_median} if ti_median is not None else None
        score, bd = normalize_climate(climate_data, thermal_inertia_data=ti_data,
                                       has_ice_signal=has_ice)
        return score, bd

    def test_no_ti_no_modifier(self):
        score, bd = self._score(ti_median=None)
        assert bd["ti_modifier"] == 0.0

    def test_high_ti_with_ice_bonus(self):
        # Use moderate climate (not perfect) so bonus has room to add
        from api.scoring_methodology import normalize_climate
        climate_data = {
            "annual_stats": {
                "temp_mean_k": 200,
                "dust_tau_mean": 0.5,  # moderate dust → d_raw=0.5
                "wind_mean_ms": 5.0,
                "frost_max_probability": 0.02,
            }
        }
        score_base, _ = normalize_climate(climate_data)
        score_ti, bd = normalize_climate(
            climate_data,
            thermal_inertia_data={"ti_median": 350},
            has_ice_signal=True,
        )
        assert bd["ti_modifier"] == 0.10
        assert score_ti > score_base

    def test_high_ti_no_ice_smaller_bonus(self):
        _, bd = self._score(ti_median=350, has_ice=False)
        assert bd["ti_modifier"] == 0.05

    def test_low_ti_with_ice_penalty(self):
        _, bd = self._score(ti_median=100, has_ice=True)
        assert bd["ti_modifier"] == -0.05

    def test_low_ti_no_ice_no_modifier(self):
        _, bd = self._score(ti_median=100, has_ice=False)
        assert bd["ti_modifier"] == 0.0

    def test_mid_ti_no_modifier(self):
        _, bd = self._score(ti_median=200, has_ice=True)
        assert bd["ti_modifier"] == 0.0

    def test_score_clamped_at_1(self):
        score, _ = self._score(ti_median=500, has_ice=True)
        assert score <= 1.0

    def test_ti_notes_present(self):
        _, bd = self._score(ti_median=350, has_ice=True)
        ti_notes = [n for n in bd["notes"] if "TI modifier" in n]
        assert len(ti_notes) == 1


# ─────────────────────────────────────────────────────────────────────────────
# 7. Evidence Pack Phase 4 Fields
# ─────────────────────────────────────────────────────────────────────────────

class TestEvidencePackPhase4:
    """Test evidence pack assembly with Phase 4 climate-ice fields."""

    def _make_session(self, ice_stability=None, seasonal_window=None,
                       climate_ice_compat=None):
        from types import SimpleNamespace
        synth = {
            "subsurface_coverage": {},
            "dielectric_analysis": {},
            "terrace_dielectric": {},
            "sharad_physics_inversion": {},
            "hyperbola_epsilon": {},
            "epsilon_cross_validation": {},
            "dielectric_method_hierarchy": [],
            "mineral_signatures": {},
            "crism_spectral_analysis": {},
            "terrain_favorability": {},
            "climate": {
                "climate_score": 7,
                "climate_summary": "Favorable conditions",
                "annual_stats": {"temp_mean_k": 200},
                "ice_stability": ice_stability or {},
                "seasonal_operation_window": seasonal_window or {},
            },
            "thermal_inertia": {
                "ti_score": 5,
                "ti_median": 250,
                "ti_mean": 260,
                "classification": "moderately consolidated",
                "distribution_pct": {},
                "ti_explanation": "Moderate TI",
            },
            "climate_ice_compatibility": climate_ice_compat or {},
            "climate_compatibility": {},
            "scoring": {},
            "isru_assessment": {},
        }
        return SimpleNamespace(
            synthesis=synth,
            bbox=None,
            region_name="Test Region",
            session_id="test-session-004",
            objective="Test Phase 4",
            narrative="",
        )

    def test_ice_stability_in_pack(self):
        from api.evidence_pack import assemble_evidence_pack
        ice_stab = {
            "ice_table_stable": True,
            "sublimation_regime": "stable",
            "stability_margin": 3.5,
            "annual_mean_temp_k": 185.0,
        }
        session = self._make_session(ice_stability=ice_stab)
        pack = assemble_evidence_pack(session)
        assert pack["climate"]["ice_stability"]["ice_table_stable"] is True
        assert pack["climate"]["ice_stability"]["sublimation_regime"] == "stable"

    def test_seasonal_window_in_pack(self):
        from api.evidence_pack import assemble_evidence_pack
        sow = {"n_safe_bins": 10, "total_bins": 12, "operational_fraction": 0.83}
        session = self._make_session(seasonal_window=sow)
        pack = assemble_evidence_pack(session)
        assert pack["climate"]["seasonal_operation_window"]["n_safe_bins"] == 10

    def test_climate_ice_compat_in_pack(self):
        from api.evidence_pack import assemble_evidence_pack
        compat = {
            "assessed": True,
            "overall_compatibility": "highly_compatible",
            "compatibility_score": 0.85,
            "ti_epsilon_correlation": {"correlation": "strongly_corroborated"},
            "notes": ["Ice stable", "High TI"],
        }
        session = self._make_session(climate_ice_compat=compat)
        pack = assemble_evidence_pack(session)
        cic = pack["climate_ice_compatibility"]
        assert cic["assessed"] is True
        assert cic["overall_compatibility"] == "highly_compatible"
        assert cic["compatibility_score"] == 0.85

    def test_no_compat_defaults(self):
        from api.evidence_pack import assemble_evidence_pack
        session = self._make_session()
        pack = assemble_evidence_pack(session)
        assert pack["climate_ice_compatibility"]["assessed"] is False


# ─────────────────────────────────────────────────────────────────────────────
# 8. Climate Region Analysis Integration
# ─────────────────────────────────────────────────────────────────────────────

class TestClimateRegionPhase4:
    """Test that climate_analysis_for_region now includes Phase 4 fields."""

    def test_ice_stability_in_region_output(self):
        from api.mars_climate import climate_analysis_for_region
        result = climate_analysis_for_region(40.0, 42.0, 10.0, 12.0)
        assert "ice_stability" in result
        assert "sublimation_regime" in result["ice_stability"]

    def test_seasonal_window_in_region_output(self):
        from api.mars_climate import climate_analysis_for_region
        result = climate_analysis_for_region(40.0, 42.0, 10.0, 12.0)
        assert "seasonal_operation_window" in result
        assert "n_safe_bins" in result["seasonal_operation_window"]


# ─────────────────────────────────────────────────────────────────────────────
# 9. Composite Score with TI
# ─────────────────────────────────────────────────────────────────────────────

class TestCompositeScoreWithTI:
    """Test that compute_composite_score accepts thermal_inertia_data."""

    def test_composite_accepts_ti(self):
        from api.scoring_methodology import compute_composite_score
        result = compute_composite_score(
            subsurface_data={},
            dielectric_data={},
            cross_instrument_data={},
            mineral_data={"ice_pixel_count": 5},
            cnn_data={},
            engineering_data={},
            climate_data={"annual_stats": {
                "temp_mean_k": 200, "dust_tau_mean": 0.3,
                "wind_mean_ms": 5.0, "frost_max_probability": 0.02,
            }},
            thermal_inertia_data={"ti_median": 350},
        )
        assert "climate" in result["sub_scores"]
        # TI modifier should be reflected
        bd = result["sub_scores"]["climate"]["breakdown"]
        assert bd["ti_modifier"] == 0.10  # ice_pixel_count > 0 = ice signal

    def test_composite_without_ti(self):
        from api.scoring_methodology import compute_composite_score
        result = compute_composite_score(
            subsurface_data={},
            dielectric_data={},
            cross_instrument_data={},
            mineral_data={},
            cnn_data={},
            engineering_data={},
            climate_data={"annual_stats": {
                "temp_mean_k": 200, "dust_tau_mean": 0.3,
                "wind_mean_ms": 5.0, "frost_max_probability": 0.02,
            }},
        )
        bd = result["sub_scores"]["climate"]["breakdown"]
        assert bd["ti_modifier"] == 0.0
