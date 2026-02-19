"""
Cluttergram-based rejection filter for SHARAD subsurface reflectors.

Compares each reflector pick against the aligned cluttergram simulation.
If the cluttergram shows a similar feature at the same (trace, bin) location,
the reflector is likely surface clutter, not a true subsurface interface.
"""
import logging
import numpy as np
from .models import ClutterAssessment

logger = logging.getLogger(__name__)

SPEED_OF_LIGHT = 299_792_458.0

def assess_clutter(
    product_id: str,
    trace_idx: int,
    interface_bin: int,
    surface_bin: int,
    window_bins: int = 10,
) -> ClutterAssessment:
    """
    Check if a reflector pick matches a cluttergram feature.

    Loads the aligned cluttergram for the product (if available),
    extracts the local window around the pick, and compares
    power levels to determine clutter likelihood.

    Args:
        product_id: SHARAD high-res product ID
        trace_idx: Trace index of the reflector pick
        interface_bin: Range bin of the subsurface reflector
        surface_bin: Range bin of the surface return
        window_bins: Half-width of comparison window (±bins)

    Returns:
        ClutterAssessment with clutter_likelihood_score (0-1)
    """
    try:
        # Import from sharad_highres_router to reuse existing cluttergram loading
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'api'))
        from sharad_highres_router import _get_aligned_cluttergram, _get_power, _pick_surface

        # Try to load aligned cluttergram
        clutter_arr = _get_aligned_cluttergram(product_id)
        if clutter_arr is None:
            return ClutterAssessment(
                cluttergram_available=False,
                notes="No cluttergram available for this product"
            )

        # Router-aligned cluttergram is (range_bins, traces).
        n_bins, n_traces = clutter_arr.shape

        # Bounds check
        if trace_idx < 0 or trace_idx >= n_traces:
            return ClutterAssessment(
                cluttergram_available=True,
                notes=f"Trace {trace_idx} out of cluttergram range (0-{n_traces-1})"
            )

        if interface_bin < 0 or interface_bin >= n_bins:
            return ClutterAssessment(
                cluttergram_available=True,
                notes=f"Bin {interface_bin} out of cluttergram range (0-{n_bins-1})"
            )

        # Extract local window around the pick in cluttergram
        bin_lo = max(0, interface_bin - window_bins)
        bin_hi = min(n_bins, interface_bin + window_bins + 1)
        trace_lo = max(0, trace_idx - 5)
        trace_hi = min(n_traces, trace_idx + 6)

        clutter_patch = clutter_arr[bin_lo:bin_hi, trace_lo:trace_hi].astype(np.float64)

        if clutter_patch.size == 0:
            return ClutterAssessment(cluttergram_available=True, notes="Empty clutter patch")

        # Compute clutter power at pick location relative to background
        clutter_at_pick = float(clutter_arr[interface_bin, trace_idx])

        # Background: median of full subsurface column below surface
        sub_start = max(0, surface_bin + 20)
        sub_end = min(n_bins, surface_bin + 200)
        if sub_end > sub_start:
            bg_col = clutter_arr[sub_start:sub_end, trace_idx].astype(np.float64)
            bg_median = float(np.median(bg_col[bg_col > 0])) if np.any(bg_col > 0) else 1.0
        else:
            bg_median = 1.0

        # Clutter SNR: how bright is the cluttergram at the pick location?
        clutter_snr = clutter_at_pick / max(bg_median, 1e-10)

        # Also check local peak in cluttergram patch
        patch_max = float(np.max(clutter_patch))
        patch_median = float(np.median(clutter_patch[clutter_patch > 0])) if np.any(clutter_patch > 0) else 1.0
        local_contrast = patch_max / max(patch_median, 1e-10)

        # Score: 0 = no clutter evidence, 1 = strong clutter
        # High clutter_snr AND high local_contrast → likely clutter
        if clutter_snr > 5.0 and local_contrast > 3.0:
            score = min(1.0, 0.5 + 0.1 * (clutter_snr - 5.0))
            rejected = score > 0.7
            notes = f"Strong cluttergram feature (SNR={clutter_snr:.1f}, contrast={local_contrast:.1f})"
        elif clutter_snr > 3.0:
            score = 0.3 + 0.1 * (clutter_snr - 3.0)
            rejected = False
            notes = f"Moderate cluttergram feature (SNR={clutter_snr:.1f})"
        elif clutter_snr > 1.5:
            score = 0.1 + 0.1 * (clutter_snr - 1.5)
            rejected = False
            notes = f"Weak cluttergram feature (SNR={clutter_snr:.1f})"
        else:
            score = max(0.0, 0.05 * clutter_snr)
            rejected = False
            notes = f"No significant cluttergram feature (SNR={clutter_snr:.1f}) — likely true subsurface"

        return ClutterAssessment(
            cluttergram_available=True,
            clutter_likelihood_score=round(min(1.0, max(0.0, score)), 3),
            rejected=rejected,
            notes=notes,
        )

    except Exception as e:
        logger.warning(f"Clutter assessment failed for {product_id} trace {trace_idx}: {e}")
        return ClutterAssessment(
            cluttergram_available=False,
            notes=f"Clutter assessment error: {str(e)}"
        )
