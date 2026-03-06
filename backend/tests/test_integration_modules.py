# pyright: reportPrivateUsage=false, reportUnknownMemberType=false, reportAny=false, reportMissingImports=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportUnusedParameter=false, reportAttributeAccessIssue=false

from unittest.mock import Mock, patch

import pytest

from backend.analysis.integration import landing_site_scorer as scorer


def _climate(
    t_mean: float = 220.0,
    t_max: float = 250.0,
    t_min: float = 180.0,
    pressure: float = 700.0,
    dust: float = 0.3,
    wind: float = 5.0,
    frost: float = 0.1,
) -> dict[str, float]:
    return {
        "temperature_mean_k": t_mean,
        "temperature_max_k": t_max,
        "temperature_min_k": t_min,
        "pressure_pa": pressure,
        "dust_tau": dust,
        "wind_ms": wind,
        "frost_prob": frost,
    }


def test_clamp_below_zero():
    assert scorer._clamp(-0.1) == 0.0


def test_clamp_above_one():
    assert scorer._clamp(1.5) == 1.0


def test_clamp_exact_boundaries():
    assert scorer._clamp(0.0) == 0.0
    assert scorer._clamp(1.0) == 1.0


def test_score_terrain_flat_low_elevation():
    terrain = {"elevation_m": -3000.0, "slope_deg": 1.0, "source": "MOLA"}
    category = scorer._score_terrain(terrain)
    assert category.score == 1.0
    assert category.weighted == 0.2
    assert "Excellent terrain" in category.assessment


def test_score_terrain_high_elevation_still_acceptable():
    terrain = {"elevation_m": 4000.0, "slope_deg": 2.0, "source": "MOLA"}
    category = scorer._score_terrain(terrain)
    assert category.score == pytest.approx(0.6, abs=1e-3)
    assert category.weighted == pytest.approx(0.12, abs=1e-3)
    assert "Acceptable terrain" in category.assessment


def test_score_terrain_extreme_slope_challenging():
    terrain = {"elevation_m": 3000.0, "slope_deg": 25.0, "source": "MOLA"}
    category = scorer._score_terrain(terrain)
    assert category.score == pytest.approx(0.4, abs=1e-3)
    assert category.weighted == pytest.approx(0.08, abs=1e-3)
    assert "Challenging terrain" in category.assessment


def test_score_climate_ideal_temperature_and_pressure():
    category = scorer._score_climate(_climate(t_mean=220.0, pressure=800.0))
    assert category.score == 1.0
    assert category.weighted == 0.25


def test_score_climate_extreme_cold_penalized():
    category = scorer._score_climate(_climate(t_mean=100.0, pressure=636.0))
    assert category.score == pytest.approx(0.238, abs=1e-3)
    assert category.weighted == pytest.approx(0.06, abs=1e-3)


def test_score_climate_extreme_hot_penalized():
    category = scorer._score_climate(_climate(t_mean=320.0, pressure=636.0))
    assert category.score == pytest.approx(0.238, abs=1e-3)
    assert category.weighted == pytest.approx(0.06, abs=1e-3)


def test_score_climate_low_pressure_penalized():
    category = scorer._score_climate(_climate(t_mean=220.0, pressure=100.0))
    assert category.score == pytest.approx(0.737, abs=1e-3)


def test_score_climate_high_pressure_clamped():
    category = scorer._score_climate(_climate(t_mean=220.0, pressure=1200.0))
    assert category.score == 1.0


def test_score_dust_clear_skies():
    category = scorer._score_dust(_climate(dust=0.2))
    assert category.score == 1.0
    assert "Clear skies" in category.assessment


def test_score_dust_moderate_conditions():
    category = scorer._score_dust(_climate(dust=1.0))
    assert category.score == pytest.approx(0.667, abs=1e-3)
    assert "Moderate dust" in category.assessment


def test_score_dust_extreme_dust_storm():
    category = scorer._score_dust(_climate(dust=2.5))
    assert category.score == 0.0
    assert "High dust" in category.assessment


def test_score_wind_calm_conditions():
    category = scorer._score_wind(_climate(wind=3.0))
    assert category.score == 1.0
    assert "calm" in category.assessment


def test_score_wind_moderate_conditions():
    category = scorer._score_wind(_climate(wind=9.0))
    assert category.score == pytest.approx(0.6, abs=1e-3)
    assert "moderate" in category.assessment


def test_score_wind_hazardous_conditions():
    category = scorer._score_wind(_climate(wind=20.0))
    assert category.score == 0.0
    assert "hazardous" in category.assessment


def test_score_frost_no_frost_risk():
    category = scorer._score_frost(_climate(frost=0.0))
    assert category.score == 1.0
    assert category.assessment == "No significant frost risk"


def test_score_frost_moderate_risk():
    category = scorer._score_frost(_climate(frost=0.2))
    assert category.score == pytest.approx(0.6, abs=1e-3)
    assert "Occasional frost" in category.assessment


def test_score_frost_heavy_risk_clamped_to_zero():
    category = scorer._score_frost(_climate(frost=0.6))
    assert category.score == 0.0
    assert "Frequent frost" in category.assessment


def test_score_science_value_equatorial_site():
    terrain = {"elevation_m": -1000.0, "slope_deg": 3.0, "source": "MOLA"}
    climate = _climate(frost=0.1)
    category = scorer._score_science_value(10.0, 20.0, terrain, climate)
    assert category.score == pytest.approx(0.6, abs=1e-3)
    assert category.details["elevation_m"] == -1000.0


def test_score_science_value_mid_latitude_site_high_value():
    terrain = {"elevation_m": -3000.0, "slope_deg": 4.0, "source": "MOLA"}
    climate = _climate(frost=0.1)
    category = scorer._score_science_value(35.0, 120.0, terrain, climate)
    assert category.score == pytest.approx(0.9, abs=1e-3)
    assert "high" in category.assessment


def test_score_science_value_polar_site_moderate_value():
    terrain = {"elevation_m": -3000.0, "slope_deg": 4.0, "source": "MOLA"}
    climate = _climate(frost=0.3)
    category = scorer._score_science_value(70.0, 120.0, terrain, climate)
    assert category.score == pytest.approx(0.65, abs=1e-3)
    assert "moderate" in category.assessment


@pytest.mark.parametrize(
    "score,expected",
    [(80.0, "A"), (65.0, "B"), (50.0, "C"), (35.0, "D"), (34.9, "F")],
)
def test_grade_thresholds(score: float, expected: str):
    assert scorer._grade(score) == expected


def test_score_landing_site_full_pipeline_with_neural_and_pinns():
    neural_predictor = Mock()
    neural_predictor.predict.return_value = {
        "temperature_mean_k": 220.0,
        "temperature_max_k": 252.0,
        "temperature_min_k": 186.0,
        "pressure_pa": 700.0,
        "dust_tau_mean": 0.4,
        "wind_mean_ms": 6.0,
        "frost_probability": 0.1,
    }
    pinns_predictor = Mock()
    pinns_predictor.predict_velocity.return_value = 4.6

    with patch("api.mars_climate.get_elevation_m", return_value=-3500.0) as p_elev, patch(
        "api.terrain_router.compute_slope_stats", return_value={"mean_slope": 4.0}
    ) as p_slope, patch(
        "neural_climate.predictor.is_model_trained", return_value=True
    ) as p_neural_ready, patch(
        "neural_climate.predictor.get_predictor", return_value=neural_predictor
    ) as p_neural_pred, patch(
        "pinns_interior.predictor.is_model_trained", return_value=True
    ) as p_pinns_ready, patch(
        "pinns_interior.predictor.get_predictor", return_value=pinns_predictor
    ) as p_pinns_pred:
        result = scorer.score_landing_site(lat=35.0, lon=140.0, ls=30.0)

    assert result.overall_score == pytest.approx(94.1, abs=0.1)
    assert result.grade == "A"
    assert [c.name for c in result.categories] == [
        "terrain",
        "climate",
        "dust",
        "wind",
        "frost",
        "science_value",
    ]
    assert result.data_sources == {
        "terrain": "MOLA",
        "climate": "neural_climate",
        "seismic": "pinns",
    }
    assert result.warnings == []
    p_elev.assert_called_once_with(35.0, 140.0)
    p_slope.assert_called_once_with(35.0, 140.0, radius_m=1000)
    p_neural_ready.assert_called_once()
    p_neural_pred.assert_called_once()
    p_pinns_ready.assert_called_once()
    p_pinns_pred.assert_called_once()


def test_score_landing_site_graceful_degradation_on_subsystem_exceptions():
    with patch("api.mars_climate.get_elevation_m", side_effect=RuntimeError("elev fail")), patch(
        "api.terrain_router.compute_slope_stats", side_effect=RuntimeError("slope fail")
    ), patch("neural_climate.predictor.is_model_trained", side_effect=RuntimeError("neural fail")), patch(
        "pinns_interior.predictor.is_model_trained", side_effect=RuntimeError("pinns fail")
    ), patch(
        "api.mars_climate.surface_temperature_k", side_effect=RuntimeError("temp fail")
    ), patch("api.mars_climate.surface_pressure_pa", side_effect=RuntimeError("pressure fail")), patch(
        "api.mars_climate.dust_opacity", side_effect=RuntimeError("dust fail")
    ), patch("api.mars_climate.wind_speed", side_effect=RuntimeError("wind fail")), patch(
        "api.mars_climate.co2_frost_probability", side_effect=RuntimeError("frost fail")
    ):
        result = scorer.score_landing_site(lat=0.0, lon=0.0, ls=90.0)

    assert result.overall_score == pytest.approx(90.5, abs=0.1)
    assert result.grade == "A"
    assert result.data_sources == {"terrain": "default", "climate": "fallback", "seismic": "none"}
    assert result.warnings == []


def test_score_landing_site_uses_parametric_climate_when_neural_untrained():
    with patch("api.mars_climate.get_elevation_m", return_value=-1000.0), patch(
        "api.terrain_router.compute_slope_stats", return_value={"mean_slope": 2.0}
    ), patch("neural_climate.predictor.is_model_trained", return_value=False), patch(
        "api.mars_climate.surface_temperature_k",
        return_value={"mean_k": 210.0, "max_k": 240.0, "min_k": 170.0},
    ), patch("api.mars_climate.surface_pressure_pa", return_value=650.0), patch(
        "api.mars_climate.dust_opacity", return_value={"tau_mean": 0.7}
    ), patch("api.mars_climate.wind_speed", return_value={"mean_ms": 8.0}), patch(
        "api.mars_climate.co2_frost_probability", return_value={"frost_probability": 0.2}
    ), patch("pinns_interior.predictor.is_model_trained", return_value=False):
        result = scorer.score_landing_site(lat=20.0, lon=45.0, ls=180.0)

    assert result.data_sources["climate"] == "parametric"
    assert result.data_sources["seismic"] == "none"
    assert 0.0 <= result.overall_score <= 100.0


def test_score_landing_site_generates_all_major_warnings():
    neural_predictor = Mock()
    neural_predictor.predict.return_value = {
        "temperature_mean_k": 215.0,
        "temperature_max_k": 250.0,
        "temperature_min_k": 180.0,
        "pressure_pa": 600.0,
        "dust_tau_mean": 2.5,
        "wind_mean_ms": 12.0,
        "frost_probability": 0.8,
    }
    pinns_predictor = Mock()
    pinns_predictor.predict_velocity.return_value = 6.0

    with patch("api.mars_climate.get_elevation_m", return_value=3000.0), patch(
        "api.terrain_router.compute_slope_stats", return_value={"mean_slope": 22.0}
    ), patch("neural_climate.predictor.is_model_trained", return_value=True), patch(
        "neural_climate.predictor.get_predictor", return_value=neural_predictor
    ), patch("pinns_interior.predictor.is_model_trained", return_value=True), patch(
        "pinns_interior.predictor.get_predictor", return_value=pinns_predictor
    ):
        result = scorer.score_landing_site(lat=45.0, lon=10.0, ls=270.0)

    assert any("Slope" in w for w in result.warnings)
    assert any("Elevation" in w for w in result.warnings)
    assert any("Extreme dust opacity" in w for w in result.warnings)
    assert any("High frost probability" in w for w in result.warnings)
    assert any("Elevated seismic risk" in w for w in result.warnings)
    assert len(result.warnings) == 5


def test_score_landing_site_custom_weights_parameter_is_ignored_by_current_pipeline():
    with patch("api.mars_climate.get_elevation_m", return_value=-2000.0), patch(
        "api.terrain_router.compute_slope_stats", return_value={"mean_slope": 3.0}
    ), patch("neural_climate.predictor.is_model_trained", return_value=False), patch(
        "api.mars_climate.surface_temperature_k",
        return_value={"mean_k": 220.0, "max_k": 250.0, "min_k": 180.0},
    ), patch("api.mars_climate.surface_pressure_pa", return_value=700.0), patch(
        "api.mars_climate.dust_opacity", return_value={"tau_mean": 0.5}
    ), patch("api.mars_climate.wind_speed", return_value={"mean_ms": 5.0}), patch(
        "api.mars_climate.co2_frost_probability", return_value={"frost_probability": 0.1}
    ), patch("pinns_interior.predictor.is_model_trained", return_value=False):
        default_result = scorer.score_landing_site(lat=15.0, lon=30.0, ls=0.0)
        custom_result = scorer.score_landing_site(
            lat=15.0,
            lon=30.0,
            ls=0.0,
            weights={
                "terrain": 0.9,
                "climate": 0.02,
                "dust": 0.02,
                "wind": 0.02,
                "frost": 0.02,
                "science_value": 0.02,
            },
        )

    assert custom_result.overall_score == default_result.overall_score
    assert [c.weight for c in custom_result.categories] == [c.weight for c in default_result.categories]


def test_compare_sites_ranks_results_in_descending_order():
    site_a = scorer.LandingSiteResult(
        lat=0.0,
        lon=0.0,
        ls=0.0,
        overall_score=62.5,
        grade="C",
        categories=[],
        recommendation="",
        warnings=[],
        data_sources={},
    )
    site_b = scorer.LandingSiteResult(
        lat=10.0,
        lon=10.0,
        ls=0.0,
        overall_score=91.0,
        grade="A",
        categories=[],
        recommendation="",
        warnings=[],
        data_sources={},
    )
    site_c = scorer.LandingSiteResult(
        lat=-10.0,
        lon=20.0,
        ls=0.0,
        overall_score=78.2,
        grade="B",
        categories=[],
        recommendation="",
        warnings=[],
        data_sources={},
    )

    with patch.object(scorer, "score_landing_site", side_effect=[site_a, site_b, site_c]) as p_score:
        ranked = scorer.compare_sites(
            [{"lat": 0.0, "lon": 0.0}, {"lat": 10.0, "lon": 10.0}, {"lat": -10.0, "lon": 20.0}],
            ls=45.0,
        )

    assert [r.overall_score for r in ranked] == [91.0, 78.2, 62.5]
    assert p_score.call_count == 3
    assert p_score.call_args_list[0].kwargs == {"lat": 0.0, "lon": 0.0, "ls": 45.0}


from backend.analysis.integration import ice_evolution as ice
from backend.analysis.integration import mineral_stability as mineral
from backend.analysis.integration import seismic_surface as seismic


def test_water_ice_sublimation_pressure_at_triple_point_matches_reference():
    p_sub = mineral.water_ice_sublimation_pressure(273.15)
    assert p_sub == pytest.approx(611.657, rel=1e-6)


@pytest.mark.parametrize(
    "temp_k,expected",
    [
        (200.0, 0.1641534449),
        (250.0, 76.2680380930),
    ],
)
def test_water_ice_sublimation_pressure_known_values(temp_k: float, expected: float):
    assert mineral.water_ice_sublimation_pressure(temp_k) == pytest.approx(expected, rel=1e-6)


def test_water_ice_sublimation_pressure_zero_or_negative_returns_zero():
    assert mineral.water_ice_sublimation_pressure(0.0) == 0.0
    assert mineral.water_ice_sublimation_pressure(-5.0) == 0.0


def test_is_water_ice_stable_cold_high_pressure_is_stable():
    stable, note = mineral.is_water_ice_stable(temp_k=200.0, pressure_pa=1000.0)
    assert stable is True
    assert "Ice stable" in note


def test_is_water_ice_stable_warm_conditions_unstable():
    stable, note = mineral.is_water_ice_stable(temp_k=220.0, pressure_pa=700.0)
    assert stable is False
    assert "sublimation expected" in note


@pytest.mark.parametrize(
    "temp_k,expected_phase",
    [
        (250.0, "kieserite (MgSO4\u00b7H2O)"),
        (265.0, "hexahydrite (MgSO4\u00b76H2O)"),
        (280.0, "epsomite (MgSO4\u00b77H2O)"),
    ],
)
def test_sulfate_phase_transitions(temp_k: float, expected_phase: str):
    phase, _ = mineral.sulfate_phase(temp_k)
    assert phase == expected_phase


def test_perchlorate_brine_check_at_210k_has_subset_of_active_brines():
    results = mineral.perchlorate_brine_check(210.0)
    active = {r["salt"] for r in results if r["brine_possible"]}
    assert active == {"Ca(ClO4)2", "Mg(ClO4)2", "Fe(ClO4)3"}


def test_perchlorate_brine_check_at_190k_has_no_active_brines():
    results = mineral.perchlorate_brine_check(190.0)
    assert all(r["brine_possible"] is False for r in results)
    assert len(results) == len(mineral.PERCHLORATE_EUTECTICS)


def test_perchlorate_brine_check_at_260k_has_all_active_brines():
    results = mineral.perchlorate_brine_check(260.0)
    assert all(r["brine_possible"] is True for r in results)
    assert {r["salt"] for r in results} == set(mineral.PERCHLORATE_EUTECTICS)


def test_carbonate_stability_normal_mars_pressure_and_near_zero_pressure():
    stable_mars, note_mars = mineral.carbonate_stability(636.0)
    stable_low, note_low = mineral.carbonate_stability(1.0)
    assert stable_mars is True
    assert "carbonates stable" in note_mars
    assert stable_low is False
    assert "unstable" in note_low


@pytest.mark.parametrize("temp_k", [140.0, 220.0, 320.0])
def test_phyllosilicate_stability_always_stable_at_mars_temperatures(temp_k: float):
    stable, note = mineral.phyllosilicate_stability(temp_k)
    assert stable is True
    assert "Phyllosilicates stable" in note


def test_assess_mineral_stability_full_pipeline_with_mocks():
    climate_predictor = Mock()
    climate_predictor.predict.return_value = {
        "temperature_mean_k": 220.0,
        "temperature_max_k": 240.0,
        "temperature_min_k": 190.0,
        "pressure_pa": 700.0,
    }
    crism_result = Mock(score=0.42, hyd_score=0.31, distance_km=12.0, notes="hydrated signal")

    with patch("neural_climate.predictor.is_model_trained", return_value=True), patch(
        "neural_climate.predictor.get_predictor", return_value=climate_predictor
    ), patch("api.thermal_inertia.get_thermal_inertia", return_value=340.0), patch(
        "analysis.ice_evidence.crism_proxy.evaluate_crism_evidence", return_value=crism_result
    ):
        result = mineral.assess_mineral_stability(lat=62.0, lon=15.0, ls=45.0)

    assert result.climate_source == "neural_climate"
    assert result.temperature_k == pytest.approx(220.0)
    assert result.pressure_pa == pytest.approx(700.0)
    assert len(result.minerals) == 5
    assert result.water_ice_stable is True  # At 190K min temp, ice IS stable (pH2O > P_sub)
    assert result.dominant_sulfate_phase == "kieserite (MgSO4\u00b7H2O)"
    assert set(result.brine_candidates) == set(mineral.PERCHLORATE_EUTECTICS)
    assert result.thermal_inertia == pytest.approx(340.0)
    assert result.crism_validation == {
        "ice_score": 0.42,
        "hyd_score": 0.31,
        "distance_km": 12.0,
        "notes": "hydrated signal",
    }
    assert "Transient liquid brines possible" in result.summary


def test_brine_habitability_window_with_mocked_climate_has_partial_year_activity():
    climate_predictor = Mock()

    def _predict(*, lat: float, lon: float, ls: float):
        _ = (lat, lon)
        t_max = 210.0 if 90.0 <= ls < 270.0 else 190.0
        return {
            "temperature_mean_k": 200.0,
            "temperature_max_k": t_max,
            "temperature_min_k": 170.0,
            "pressure_pa": 700.0,
        }

    climate_predictor.predict.side_effect = _predict

    with patch("neural_climate.predictor.is_model_trained", return_value=True), patch(
        "neural_climate.predictor.get_predictor", return_value=climate_predictor
    ):
        result = mineral.brine_habitability_window(lat=20.0, lon=20.0, n_seasons=12)

    assert result["brine_fraction"] == pytest.approx(0.5, abs=1e-6)
    assert result["salts_involved"] == ["Ca(ClO4)2", "Fe(ClO4)3", "Mg(ClO4)2"]
    assert result["max_temperature_k"] == pytest.approx(210.0)
    assert result["n_samples"] == 12


def test_sublimation_pressure_known_temperatures_and_zero_edge_case():
    assert ice.sublimation_pressure(200.0) == pytest.approx(0.1641534449, rel=1e-6)
    assert ice.sublimation_pressure(250.0) == pytest.approx(76.2680380930, rel=1e-6)
    assert ice.sublimation_pressure(0.0) == 0.0


def test_pore_pressure_at_depth_increases_monotonically():
    p0 = ice.pore_pressure_at_depth(depth_m=0.0, surface_pressure_pa=700.0)
    p5 = ice.pore_pressure_at_depth(depth_m=5.0, surface_pressure_pa=700.0)
    p20 = ice.pore_pressure_at_depth(depth_m=20.0, surface_pressure_pa=700.0)
    assert p0 == pytest.approx(0.21)
    assert p5 > p0
    assert p20 > p5


def test_annual_thermal_skin_depth_default_regolith_in_reasonable_range():
    skin = ice.annual_thermal_skin_depth()
    assert 0.5 <= skin <= 2.0  # Dry Mars regolith: ~0.8m annual skin depth


def test_subsurface_temperature_deeper_has_smaller_annual_wave_component():
    t0 = ice.subsurface_temperature(
        depth_m=0.0,
        t_mean_surface=210.0,
        t_amplitude=30.0,
        geothermal_gradient=0.0,
        ls=0.0,
        skin_depth=2.0,
    )
    t4 = ice.subsurface_temperature(
        depth_m=4.0,
        t_mean_surface=210.0,
        t_amplitude=30.0,
        geothermal_gradient=0.0,
        ls=0.0,
        skin_depth=2.0,
    )
    assert abs(t0 - 210.0) > abs(t4 - 210.0)


def test_subsurface_temperature_includes_geothermal_gradient_offset():
    t_no_gradient = ice.subsurface_temperature(
        depth_m=5.0,
        t_mean_surface=200.0,
        t_amplitude=10.0,
        geothermal_gradient=0.0,
        ls=180.0,
        skin_depth=2.0,
    )
    t_with_gradient = ice.subsurface_temperature(
        depth_m=5.0,
        t_mean_surface=200.0,
        t_amplitude=10.0,
        geothermal_gradient=0.01,
        ls=180.0,
        skin_depth=2.0,
    )
    assert t_with_gradient - t_no_gradient == pytest.approx(0.05, abs=1e-6)


def test_max_annual_temperature_at_depth_thermal_wave_attenuates_with_depth():
    shallow = ice.max_annual_temperature_at_depth(
        depth_m=0.0,
        t_mean_surface=210.0,
        t_amplitude=20.0,
        geothermal_gradient=0.0,
        skin_depth=2.0,
    )
    deep = ice.max_annual_temperature_at_depth(
        depth_m=6.0,
        t_mean_surface=210.0,
        t_amplitude=20.0,
        geothermal_gradient=0.0,
        skin_depth=2.0,
    )
    assert shallow == pytest.approx(230.0)
    assert deep < shallow


def test_compute_ice_stability_with_mocked_climate_pinns_and_ti():
    climate_predictor = Mock()
    climate_predictor.predict.return_value = {
        "temperature_mean_k": 190.0,
        "temperature_max_k": 220.0,
        "temperature_min_k": 160.0,
        "pressure_pa": 700.0,
    }
    pinns_predictor = Mock()
    pinns_predictor.predict_velocity.return_value = 4.0

    with patch("neural_climate.predictor.is_model_trained", return_value=True), patch(
        "neural_climate.predictor.get_predictor", return_value=climate_predictor
    ), patch("pinns_interior.predictor.is_model_trained", return_value=True), patch(
        "pinns_interior.predictor.get_predictor", return_value=pinns_predictor
    ), patch("api.thermal_inertia.get_thermal_inertia", return_value=100.0):
        result = ice.compute_ice_stability(
            lat=55.0,
            lon=10.0,
            ls=30.0,
            max_depth_m=6.0,
            depth_resolution_m=0.5,
        )

    assert result.climate_source == "neural_climate"
    assert result.geothermal_source == "pinns"
    assert result.ice_stable_surface == False  # noqa: E712 — use == not is for numpy bool
    assert result.ice_stability_depth_m is not None
    assert result.ice_stability_depth_m > 0.0
    assert result.profile[0].depth_m == 0.0
    assert result.profile[-1].depth_m == pytest.approx(6.0)


def test_assess_ice_evolution_full_pipeline_with_mocks():
    mocked_stability = ice.IceStabilityResult(
        lat=60.0,
        lon=30.0,
        ls=90.0,
        ice_stability_depth_m=1.5,
        ice_stable_surface=False,
        surface_temperature_k=195.0,
        annual_max_surface_temp_k=225.0,
        annual_min_surface_temp_k=165.0,
        geothermal_gradient_k_per_m=0.006,
        thermal_skin_depth_m=1.8,
        profile=[
            ice.SubsurfaceProfile(
                depth_m=0.0,
                temperature_k=210.0,
                ice_stable=False,
                sublimation_pressure_pa=1.0,
                pore_pressure_pa=0.2,
            ),
            ice.SubsurfaceProfile(
                depth_m=1.5,
                temperature_k=195.0,
                ice_stable=True,
                sublimation_pressure_pa=0.1,
                pore_pressure_pa=0.21,
            ),
        ],
        geothermal_source="pinns",
        climate_source="neural_climate",
    )

    with patch.object(ice, "compute_ice_stability", return_value=mocked_stability), patch.object(
        ice,
        "_get_sharad_ice_depth",
        return_value={"ice_probability": 0.8, "confidence": 0.7, "crism_ice_score": 0.6, "crism_distance_km": 8.0},
    ), patch.object(ice, "_get_swim_ice_probability", return_value=0.65), patch.object(
        ice, "_get_thermal_inertia", return_value=320.0
    ):
        result = ice.assess_ice_evolution(lat=60.0, lon=30.0, ls=90.0)

    assert result.stability.ice_stability_depth_m == pytest.approx(1.5)
    assert result.sharad_validation == {
        "ice_probability": 0.8,
        "confidence": 0.7,
        "crism_ice_score": 0.6,
        "crism_distance_km": 8.0,
    }
    assert result.swim_probability == pytest.approx(0.65)
    assert result.thermal_inertia == pytest.approx(320.0)
    assert result.consistency_score > 0.8
    assert "Ice stability depth: 1.5 m below surface." in result.summary


def test_get_reference_vp_at_known_depth_returns_exact_reference_value():
    assert seismic._get_reference_vp(30.0) == pytest.approx(4.5)


def test_get_reference_vp_interpolates_between_reference_depths():
    assert seismic._get_reference_vp(40.0) == pytest.approx(5.0)


def test_get_reference_vp_out_of_range_clamps_to_endpoints():
    assert seismic._get_reference_vp(1.0) == pytest.approx(3.5)
    assert seismic._get_reference_vp(900.0) == pytest.approx(8.0)


def test_interpret_anomaly_low_anomaly_reports_normal():
    msg = seismic._interpret_anomaly(depth_km=30.0, vp=4.48, ref=4.5, anom_pct=0.4)
    assert "Normal velocity" in msg


def test_interpret_anomaly_high_anomaly_reports_anomalous():
    msg = seismic._interpret_anomaly(depth_km=20.0, vp=3.5, ref=4.5, anom_pct=22.2)
    assert "Anomalous" in msg
    assert "shallow crust" in msg


@pytest.mark.parametrize(
    "anom_pct,expected",
    [
        (2.0, "low"),
        (7.0, "moderate"),
        (15.0, "elevated"),
        (25.0, "high"),
    ],
)
def test_classify_seismic_risk_thresholds(anom_pct: float, expected: str):
    anomalies = [
        seismic.VelocityAnomaly(
            depth_km=30.0,
            vp_predicted_km_s=4.0,
            vp_reference_km_s=4.5,
            anomaly_pct=anom_pct,
            interpretation="x",
        )
    ]
    risk, max_anom, mean_anom = seismic._classify_seismic_risk(anomalies)
    assert risk == expected
    assert max_anom == pytest.approx(anom_pct)
    assert mean_anom == pytest.approx(anom_pct)


def test_compute_correlation_supporting_when_interior_and_surface_agree():
    seismic_profile = seismic.SeismicProfile(
        anomalies=[
            seismic.VelocityAnomaly(30.0, 3.7, 4.5, 17.8, "low vp"),
            seismic.VelocityAnomaly(100.0, 6.1, 6.5, 6.2, "moderate"),
        ],
        max_anomaly_pct=17.8,
        mean_anomaly_pct=12.0,
        risk_level="elevated",
        profile_source="pinns",
        interpretation="",
    )
    features = [
        seismic.SurfaceFeature("graben", 2, 0.9, "graben nearby"),
        seismic.SurfaceFeature("volcanic", 1, 0.8, "volcanic cone"),
    ]
    terrain = {"mean_slope_deg": 6.0}

    corr = seismic._compute_correlation(seismic_profile, features, terrain)
    assert corr.correlation_score > 0.7
    assert len(corr.supporting_evidence) >= 2
    assert corr.contradicting_evidence == []


def test_compute_correlation_contradicting_when_both_disagree():
    seismic_profile = seismic.SeismicProfile(
        anomalies=[seismic.VelocityAnomaly(30.0, 4.5, 4.5, 0.0, "normal")],
        max_anomaly_pct=0.0,
        mean_anomaly_pct=0.0,
        risk_level="low",
        profile_source="default",
        interpretation="",
    )
    features = [seismic.SurfaceFeature("graben", 1, 0.9, "active-looking fault")]
    terrain = {"mean_slope_deg": 2.0}

    corr = seismic._compute_correlation(seismic_profile, features, terrain)
    assert corr.correlation_score < 0.5
    assert len(corr.contradicting_evidence) == 1


def test_compute_correlation_mixed_signals_returns_midrange_score():
    seismic_profile = seismic.SeismicProfile(
        anomalies=[seismic.VelocityAnomaly(30.0, 3.8, 4.5, 15.6, "anomaly")],
        max_anomaly_pct=15.6,
        mean_anomaly_pct=15.6,
        risk_level="elevated",
        profile_source="pinns",
        interpretation="",
    )
    features = [seismic.SurfaceFeature("channel", 2, 0.3, "fluvial channels")]
    terrain = {"mean_slope_deg": 1.0}

    corr = seismic._compute_correlation(seismic_profile, features, terrain)
    assert 0.1 <= corr.correlation_score <= 0.9
    assert corr.supporting_evidence == []
    assert len(corr.contradicting_evidence) == 0


def test_compute_overall_risk_low_features_low_seismic_returns_low_grade():
    seismic_profile = seismic.SeismicProfile(
        anomalies=[seismic.VelocityAnomaly(30.0, 4.45, 4.5, 1.1, "normal")],
        max_anomaly_pct=1.1,
        mean_anomaly_pct=1.1,
        risk_level="low",
        profile_source="default",
        interpretation="",
    )
    corr = seismic.CorrelationResult(0.2, [], [])
    risk, grade = seismic._compute_overall_risk(
        seismic_profile,
        features=[],
        correlation=corr,
        terrain={"mean_slope_deg": 1.0},
    )
    assert risk < 0.2
    assert grade == "Low"


def test_compute_overall_risk_high_features_and_high_seismic_returns_high_grade():
    seismic_profile = seismic.SeismicProfile(
        anomalies=[seismic.VelocityAnomaly(30.0, 3.2, 4.5, 28.9, "major anomaly")],
        max_anomaly_pct=28.9,
        mean_anomaly_pct=28.9,
        risk_level="high",
        profile_source="pinns",
        interpretation="",
    )
    features = [
        seismic.SurfaceFeature("graben", 5, 0.9, "dense graben network"),
        seismic.SurfaceFeature("volcanic", 4, 0.8, "multiple volcanic edifices"),
    ]
    corr = seismic.CorrelationResult(0.9, ["support"], [])
    risk, grade = seismic._compute_overall_risk(
        seismic_profile,
        features=features,
        correlation=corr,
        terrain={"mean_slope_deg": 18.0},
    )
    assert risk >= 0.7
    assert grade == "High"


def test_assess_seismic_surface_full_pipeline_with_mocks():
    pinns_predictor = Mock()
    # predict_velocity is called with float(depth), so keys must match
    depth_vp_map = {
        10.0: 3.2, 30.0: 3.9, 50.0: 4.9,
        100.0: 6.1, 200.0: 7.1, 500.0: 7.6,
    }
    def _mock_vp(depth_km):
        return depth_vp_map.get(float(depth_km), 4.5)
    pinns_predictor.predict_velocity.side_effect = _mock_vp

    features = [
        seismic.SurfaceFeature("graben", 2, seismic.FEATURE_SEISMIC_WEIGHTS["graben"], "fault set"),
        seismic.SurfaceFeature("volcanic", 1, seismic.FEATURE_SEISMIC_WEIGHTS["volcanic"], "cone"),
    ]

    with patch("pinns_interior.predictor.is_model_trained", return_value=True), patch(
        "pinns_interior.predictor.get_predictor", return_value=pinns_predictor
    ), patch.object(seismic, "_query_surface_features", return_value=features), patch(
        "api.terrain_router.compute_slope_stats",
        return_value={"elevation_m": -1500.0, "mean_slope": 7.5, "max_slope": 14.0, "std_slope": 2.0},
    ):
        result = seismic.assess_seismic_surface(lat=5.0, lon=100.0)

    assert result.seismic_profile.profile_source == "pinns"
    assert result.seismic_profile.max_anomaly_pct > 10.0
    assert result.correlation.correlation_score > 0.6
    assert result.risk_grade in {"Moderate", "Elevated", "High"}
    assert result.terrain_context["mean_slope_deg"] == pytest.approx(7.5)
    assert "Seismic-geological assessment" in result.summary


def test_compare_seismic_risk_ranks_safest_first():
    site_high = seismic.SeismicSurfaceResult(
        lat=0.0,
        lon=0.0,
        seismic_profile=Mock(),
        surface_features=[],
        terrain_context={},
        correlation=Mock(),
        overall_risk_score=0.8,
        risk_grade="High",
        summary="",
    )
    site_low = seismic.SeismicSurfaceResult(
        lat=1.0,
        lon=1.0,
        seismic_profile=Mock(),
        surface_features=[],
        terrain_context={},
        correlation=Mock(),
        overall_risk_score=0.2,
        risk_grade="Moderate",
        summary="",
    )
    site_mid = seismic.SeismicSurfaceResult(
        lat=2.0,
        lon=2.0,
        seismic_profile=Mock(),
        surface_features=[],
        terrain_context={},
        correlation=Mock(),
        overall_risk_score=0.5,
        risk_grade="Elevated",
        summary="",
    )

    with patch.object(seismic, "assess_seismic_surface", side_effect=[site_high, site_low, site_mid]):
        ranked = seismic.compare_seismic_risk(
            [
                {"lat": 0.0, "lon": 0.0},
                {"lat": 1.0, "lon": 1.0},
                {"lat": 2.0, "lon": 2.0},
            ]
        )

    assert [r.overall_risk_score for r in ranked] == [0.2, 0.5, 0.8]
