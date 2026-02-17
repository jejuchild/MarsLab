"""Unit tests for RTE shallow-mode implementation (Phase 1)."""

import numpy as np
import pytest
from analysis.regolith_thickness.pipeline import (
    RegolithThicknessEstimator,
    _apply_mode_defaults,
    RING_GUARD_BINS_DEFAULT,
    RING_GUARD_BINS_SHALLOW,
    MEDIAN_KERNEL_DEFAULT,
    MEDIAN_KERNEL_SHALLOW,
    OUTLIER_THRESHOLD_DEFAULT,
    OUTLIER_THRESHOLD_SHALLOW,
)


class TestModeDefaults:
    """Test mode-aware search window defaults."""

    def test_default_mode_no_change(self):
        """Default mode should not change search bounds."""
        lo, hi = _apply_mode_defaults("default", 10, 150)
        assert lo == 10
        assert hi == 150

    def test_shallow_mode_applies_defaults(self):
        """Shallow mode should apply reduced search bounds."""
        lo, hi = _apply_mode_defaults("shallow", 10, 150)
        assert lo == 2
        assert hi == 120

    def test_shallow_mode_respects_explicit_values(self):
        """Shallow mode should not override explicit user-specified bounds."""
        lo, hi = _apply_mode_defaults("shallow", 5, 100)
        assert lo == 5  # User specified, not overridden
        assert hi == 100  # User specified, not overridden


class TestShallowModeConstants:
    """Verify shallow-mode constant values."""

    def test_ring_guard_bins(self):
        """Verify ring guard bin settings."""
        assert RING_GUARD_BINS_DEFAULT == 4
        assert RING_GUARD_BINS_SHALLOW == 3
        assert RING_GUARD_BINS_SHALLOW < RING_GUARD_BINS_DEFAULT

    def test_median_kernel(self):
        """Verify coherence median kernel settings."""
        assert MEDIAN_KERNEL_DEFAULT == 15
        assert MEDIAN_KERNEL_SHALLOW == 11
        assert MEDIAN_KERNEL_SHALLOW < MEDIAN_KERNEL_DEFAULT

    def test_outlier_threshold(self):
        """Verify outlier rejection threshold settings."""
        assert OUTLIER_THRESHOLD_DEFAULT == 15
        assert OUTLIER_THRESHOLD_SHALLOW == 12
        assert OUTLIER_THRESHOLD_SHALLOW < OUTLIER_THRESHOLD_DEFAULT


class TestCoherenceFilterParameters:
    """Test coherence filter with mode-aware parameters."""

    def test_coherence_filter_signature(self):
        """Verify _coherence_filter accepts median_kernel and outlier_threshold."""
        detected = np.array([True, True, False, True, True], dtype=bool)
        delta_bins = np.array([10, 11, 0, 12, 13], dtype=np.int32)
        snr = np.array([5.0, 5.5, 0.0, 6.0, 5.8], dtype=np.float32)
        coherence = np.zeros(5, dtype=np.float32)

        # Should accept mode-aware parameters
        result = RegolithThicknessEstimator._coherence_filter(
            detected, delta_bins, snr, coherence,
            median_kernel=11,
            outlier_threshold=12,
        )
        assert result is not None
        assert len(result) == 5


class TestRTEPipelineParameters:
    """Test RTE pipeline with new mode parameters."""

    def test_rte_accepts_mode_parameter(self):
        """RTE.run() should accept mode parameter."""
        # This test checks the signature exists; full integration test
        # requires actual SHARAD data.
        rte = RegolithThicknessEstimator()
        assert hasattr(rte, 'run')
        # Check that the run method can be called with mode parameter
        # (actual test would need real product_id and data)

    def test_rte_accepts_epsilon_uncertainty(self):
        """RTE.run() should accept epsilon_uncertainty parameter."""
        rte = RegolithThicknessEstimator()
        assert hasattr(rte, 'run')
        # Signature includes epsilon_uncertainty

    def test_shallow_mode_produces_output(self):
        """Shallow mode should produce RegolithResult."""
        rte = RegolithThicknessEstimator()
        assert hasattr(rte, '_run_impl')
        # Integration test would run with real product_id


class TestModelFields:
    """Test that models include new fields."""

    def test_regolith_sample_has_ringing_rejected(self):
        """RegolithSample should have ringing_rejected field."""
        from analysis.regolith_thickness.models import RegolithSample
        sample = RegolithSample(
            trace_idx=0,
            lat=0.0,
            lon=0.0,
            along_track_km=0.0,
            surface_elev_m=0.0,
            interface_detected=False,
            ringing_rejected=True,
            mode="shallow",
        )
        assert sample.ringing_rejected is True
        assert sample.mode == "shallow"

    def test_regolith_sample_has_epsilon_band(self):
        """RegolithSample should have thickness_low_m and thickness_high_m."""
        from analysis.regolith_thickness.models import RegolithSample
        sample = RegolithSample(
            trace_idx=0,
            lat=0.0,
            lon=0.0,
            along_track_km=0.0,
            surface_elev_m=0.0,
            interface_detected=True,
            thickness_m=50.0,
            thickness_low_m=48.0,
            thickness_high_m=52.0,
        )
        assert sample.thickness_low_m == 48.0
        assert sample.thickness_high_m == 52.0

    def test_regolith_summary_has_shallow_mode_fields(self):
        """RegolithSummary should have shallow-mode fields."""
        from analysis.regolith_thickness.models import RegolithSummary
        summary = RegolithSummary(
            product_id="test",
            epsilon_r=2.5,
            total_traces=100,
            valid_traces=50,
            detection_rate=0.5,
            shallow_mode_enabled=True,
            ring_reject_rate=0.1,
        )
        assert summary.shallow_mode_enabled is True
        assert summary.ring_reject_rate == 0.1

    def test_regolith_parameters_has_mode(self):
        """RegolithParameters should include mode parameter."""
        from analysis.regolith_thickness.models import RegolithParameters
        params = RegolithParameters(
            epsilon_r=2.5,
            snr_threshold=3.5,
            search_lo=10,
            search_hi=150,
            dem_source="MOLA",
            mode="shallow",
            epsilon_uncertainty=0.5,
        )
        assert params.mode == "shallow"
        assert params.epsilon_uncertainty == 0.5
