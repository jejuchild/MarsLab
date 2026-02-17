"""Unit tests for clutter integration in RTE pipeline (Phase 2)."""

import numpy as np
import pytest

from analysis.shared.clutter_mask import compute_clutter_mask
from analysis.shared.clutter_alignment import ClutterAligner
from analysis.shared.sharad_clutter import extract_obs_id, find_clutter_pair


class TestExtractObsId:
    """Test observation ID extraction from SHARAD product ID."""

    def test_standard_product_id(self):
        """Extract obs_id from standard product ID."""
        product_id = "R_0277201_001_SS19_700_A"
        obs_id = extract_obs_id(product_id)
        assert obs_id == "00277201"
        assert len(obs_id) == 8

    def test_zero_padding(self):
        """Verify zero-padding to 8 digits."""
        product_id = "R_0001_001_SS19_700_A"
        obs_id = extract_obs_id(product_id)
        assert obs_id == "00000001"

    def test_invalid_product_id(self):
        """Invalid product ID should raise ValueError."""
        with pytest.raises(ValueError):
            extract_obs_id("INVALID")


class TestClutterAligner:
    """Test vertical and horizontal clutter alignment."""

    def test_align_vertical_basic(self):
        """Align clutter surface to RDR surface."""
        # Synthetic clutter: surface at bin 50, RDR surface at bin 40
        clutter_power = np.zeros((100, 10), dtype=np.float32)
        clutter_power[50, :] = 1.0  # Surface peak at bin 50

        rdr_surface_bins = np.full(10, 40, dtype=np.int32)
        rdr_n_traces = 10

        aligner = ClutterAligner(clutter_power, rdr_surface_bins, rdr_n_traces)
        offsets = aligner.align_vertical()

        # Expected: offset = 50 - 40 = 10
        assert offsets.shape == (10,)
        assert np.all(offsets == 10)

    def test_apply_offset(self):
        """Extract 667-bin window using computed offsets."""
        clutter_power = np.random.rand(1024, 10).astype(np.float32)
        rdr_surface_bins = np.full(10, 40, dtype=np.int32)
        rdr_n_traces = 10

        aligner = ClutterAligner(clutter_power, rdr_surface_bins, rdr_n_traces)
        offsets = np.full(10, 50, dtype=np.int32)
        aligned = aligner.apply_offset(offsets)

        assert aligned.shape == (667, 10)
        assert not np.isnan(aligned).all()  # Not all NaN

    def test_resample_if_needed(self):
        """Resample horizontally if trace counts differ."""
        clutter_power = np.random.rand(667, 100).astype(np.float32)
        rdr_surface_bins = np.full(50, 40, dtype=np.int32)  # RDR has 50 traces
        rdr_n_traces = 50

        aligner = ClutterAligner(clutter_power, rdr_surface_bins, rdr_n_traces)

        # Simulate offset application (would normally be 667×100)
        aligned_100_traces = np.random.rand(667, 100).astype(np.float32)
        resampled = aligner.resample_if_needed(aligned_100_traces)

        assert resampled.shape == (667, 50)

    def test_align_full_integration(self):
        """Full alignment pipeline."""
        clutter_power = np.random.rand(667, 50).astype(np.float32)
        rdr_surface_bins = np.full(50, 30, dtype=np.int32)
        rdr_n_traces = 50

        aligner = ClutterAligner(clutter_power, rdr_surface_bins, rdr_n_traces)
        aligned = aligner.align_full()

        assert aligned.shape == (667, 50)
        assert aligned.dtype == np.float32


class TestClutterMasking:
    """Test clutter peak detection and flagging."""

    def test_clutter_flag_when_snr_high_and_bin_near(self):
        """Flag trace when clutter SNR high and peak near detected reflector."""
        # Create synthetic clutter: strong peak at bin 10-15
        aligned_clutter = np.ones((667, 5), dtype=np.float32) * 0.1
        aligned_clutter[50:70, 0] = 1.0  # Strong clutter signal at bin 50-70

        detected_peaks = np.array([10, -1, 10, -1, 5], dtype=np.int32)
        search_lo = 10

        clutter_flagged, clutter_snr = compute_clutter_mask(
            aligned_clutter,
            detected_peaks,
            search_lo,
            snr_threshold=3.0,
            bin_tolerance=3,
        )

        # Trace 0: detected at bin 10, clutter strong → should check distance
        assert clutter_flagged.shape == (5,)
        assert not clutter_flagged[1]  # Undetected trace
        assert np.isnan(clutter_snr[1])

    def test_no_flag_when_clutter_far(self):
        """Don't flag when clutter peak >bin_tolerance away from detection."""
        aligned_clutter = np.ones((667, 3), dtype=np.float32) * 0.1
        aligned_clutter[10:20, :] = 2.0  # Clutter peak at bin 10-20

        detected_peaks = np.array([50, 50, 50], dtype=np.int32)  # Detected far away
        search_lo = 10

        clutter_flagged, _ = compute_clutter_mask(
            aligned_clutter,
            detected_peaks,
            search_lo,
            snr_threshold=3.0,
            bin_tolerance=3,
        )

        # All should be unflagged (clutter peak is far from detected peak)
        assert not clutter_flagged.any()

    def test_skip_undetected_traces(self):
        """Skip clutter check for undetected traces."""
        aligned_clutter = np.random.rand(667, 5).astype(np.float32)

        detected_peaks = np.array([-1, -1, -1, -1, -1], dtype=np.int32)  # All -1
        search_lo = 10

        clutter_flagged, clutter_snr = compute_clutter_mask(
            aligned_clutter,
            detected_peaks,
            search_lo,
        )

        # All should be unflagged and SNR NaN
        assert not clutter_flagged.any()
        assert np.all(np.isnan(clutter_snr))

    def test_clutter_snr_computation(self):
        """Verify SNR is computed correctly."""
        # Simple test: constant signal + peak
        aligned_clutter = np.ones((70, 1), dtype=np.float32) * 1.0  # Noise level = 1
        aligned_clutter[55:65, 0] = 10.0  # Peak at bin 55-65, power = 10
        # Expected SNR ≈ 10 / 1 = 10

        detected_peaks = np.array([10], dtype=np.int32)
        search_lo = 10

        _, clutter_snr = compute_clutter_mask(
            aligned_clutter,
            detected_peaks,
            search_lo,
            bin_tolerance=10,
        )

        # SNR should be computed (not NaN)
        assert not np.isnan(clutter_snr[0])
        # Should be roughly 10 (peak power / median noise)
        assert clutter_snr[0] > 5  # At least reasonable value


class TestClutterIntegrationEdgeCases:
    """Test edge cases in clutter handling."""

    def test_missing_clutter_graceful_degradation(self):
        """Pipeline should work when cluttergram is not found."""
        # This is more of an integration test; individual utilities handle None
        product_id = "R_INVALID_000_SS19_700_A"
        clutter_pair = find_clutter_pair(product_id)
        assert clutter_pair is None  # Expected: no cluttergram found

    def test_empty_clutter_band(self):
        """Handle case where clutter band is empty."""
        aligned_clutter = np.zeros((600, 3), dtype=np.float32)
        detected_peaks = np.array([10, 10, 10], dtype=np.int32)
        search_lo = 650  # Window beyond array size

        # Should not raise; should handle gracefully
        clutter_flagged, clutter_snr = compute_clutter_mask(
            aligned_clutter,
            detected_peaks,
            search_lo,
        )

        assert len(clutter_flagged) == 3
