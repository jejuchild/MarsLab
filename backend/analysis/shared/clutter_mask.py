"""Clutter masking: detect traces where clutter peak conflicts with detected reflectors."""

import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


def compute_clutter_mask(
    aligned_clutter: np.ndarray,
    detected_peaks: np.ndarray,
    search_lo: int,
    snr_threshold: float = 3.0,
    bin_tolerance: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Identify traces where clutter peak conflicts with detected reflector.

    This function checks if there is a strong clutter peak near the detected
    reflector. If the clutter SNR is above the threshold AND the clutter peak
    is within bin_tolerance of the detected reflector, the trace is flagged.

    Args:
        aligned_clutter: (667, n_traces) float32 aligned clutter power array
        detected_peaks: (n_traces,) int32 delta_bins (= search_lo + peak_idx), 0 if not detected
        search_lo: Offset of search band start from surface (in bins)
        snr_threshold: Clutter SNR threshold for flagging (default 3.0)
        bin_tolerance: Max bin distance to flag as conflict (default 3)

    Returns:
        clutter_flagged: (n_traces,) bool array, True if SNR > threshold AND within tolerance
        clutter_snr: (n_traces,) float array, measured clutter SNR or NaN
    """
    n_traces = aligned_clutter.shape[1]
    clutter_flagged = np.zeros(n_traces, dtype=bool)
    clutter_snr_arr = np.full(n_traces, np.nan, dtype=np.float32)

    # Define search band in clutter array
    clutter_band_start = search_lo
    clutter_band_end = min(search_lo + 120, aligned_clutter.shape[0])

    for i in range(n_traces):
        # Skip undetected traces (pipeline stores 0 for undetected)
        if detected_peaks[i] <= 0:
            clutter_flagged[i] = False
            continue

        # Extract clutter search band
        clutter_band = aligned_clutter[clutter_band_start:clutter_band_end, i]

        if len(clutter_band) == 0:
            logger.debug(f"Clutter band empty for trace {i}")
            continue

        # Find clutter peak and SNR
        clutter_peak_idx = np.argmax(clutter_band)
        clutter_peak_power = clutter_band[clutter_peak_idx]
        clutter_noise = np.median(clutter_band)

        # Compute SNR (avoid division by zero)
        clutter_snr = clutter_peak_power / (clutter_noise + 1e-12)
        clutter_snr_arr[i] = clutter_snr

        # Check: clutter SNR high AND bin proximity to detected peak
        if clutter_snr >= snr_threshold:
            # detected_peaks[i] = search_lo + peak_idx (offset from surface)
            # clutter_peak_idx is offset from search_lo within clutter band
            # Normalize to same reference frame before comparing
            detected_bin_from_search_lo = detected_peaks[i] - search_lo
            clutter_detected_bin = clutter_peak_idx

            # Bin distance in search-band coordinates
            bin_distance = abs(clutter_detected_bin - detected_bin_from_search_lo)

            if bin_distance <= bin_tolerance:
                clutter_flagged[i] = True
                logger.debug(
                    f"Trace {i}: Clutter conflict detected "
                    f"(SNR={clutter_snr:.2f}, bin_dist={bin_distance})"
                )

    return clutter_flagged, clutter_snr_arr
