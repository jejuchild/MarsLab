"""Clutter alignment utilities to match cluttergram to RDR power arrays."""

import logging
from typing import Optional

import numpy as np
from scipy import ndimage

logger = logging.getLogger(__name__)


class ClutterAligner:
    """Align cluttergram power array vertically and horizontally to RDR window.

    Cluttergrams may have different range/trace dimensions than RDR arrays.
    This class handles surface-aligned vertical offset computation and optional
    resampling to match RDR dimensions.
    """

    def __init__(
        self,
        clutter_power: np.ndarray,
        rdr_surface_bins: np.ndarray,
        rdr_n_traces: int,
    ):
        """Initialize aligner with clutter and RDR surface geometry.

        Args:
            clutter_power: (n_ranges_clutter, n_traces_clutter) float32 array
            rdr_surface_bins: (n_traces_rdr,) int32 array of surface bin indices
            rdr_n_traces: Number of traces in RDR (should match len(rdr_surface_bins))
        """
        self.clutter_power = clutter_power
        self.clutter_n_ranges, self.clutter_n_traces = clutter_power.shape
        self.rdr_surface_bins = rdr_surface_bins
        self.rdr_n_traces = rdr_n_traces

    def align_vertical(self) -> np.ndarray:
        """Compute per-trace vertical offset to align clutter surface to RDR.

        Algorithm:
        1. Find clutter surface peaks: argmax(clutter_power, axis=0) per trace
        2. Smooth peaks with median filter to remove noise
        3. Compute RDR surface median as anchor
        4. Compute per-trace offset = clutter_peaks_smoothed - rdr_surface_median
        5. Validate offsets: ensure 0 < offset and offset + 667 <= clutter_n_ranges
        6. Fallback: use global median_offset for invalid traces

        Returns:
            (n_traces_clutter,) int32 array of per-trace vertical offsets in bins
        """
        # Find clutter surface peaks per trace
        clutter_peaks = np.argmax(self.clutter_power, axis=0).astype(np.float32)

        # Smooth with median filter to reduce noise
        kernel_size = min(31, max(3, self.clutter_n_traces // 20))
        if kernel_size % 2 == 0:
            kernel_size += 1
        clutter_peaks_smooth = ndimage.median_filter(clutter_peaks, size=kernel_size)

        # Anchor: median RDR surface (use only valid traces)
        valid_mask = (self.rdr_surface_bins > 0) & (
            self.rdr_surface_bins < 667
        )  # Sanity check
        if valid_mask.sum() == 0:
            logger.warning(
                "No valid RDR surface bins; using median of all surface bins"
            )
            rdr_surface_median = np.nanmedian(self.rdr_surface_bins)
        else:
            rdr_surface_median = np.median(self.rdr_surface_bins[valid_mask])

        # Compute offsets
        offsets = np.round(clutter_peaks_smooth - rdr_surface_median).astype(np.int32)

        # Validate: ensure window [offset, offset+667] fits in clutter array
        valid_offsets = (offsets >= 0) & (offsets + 667 <= self.clutter_n_ranges)
        invalid_count = (~valid_offsets).sum()

        if invalid_count > 0:
            # Use global median offset as fallback
            valid_subset = offsets[valid_offsets]
            if valid_subset.size > 0:
                median_offset = int(np.median(valid_subset))
            else:
                median_offset = 0
                logger.warning("ClutterAligner: no valid offsets found; using 0")
            offsets[~valid_offsets] = median_offset
            logger.debug(
                f"ClutterAligner: {invalid_count} traces had invalid offsets; "
                f"using median={median_offset}"
            )

        return offsets

    def apply_offset(
        self, offsets: np.ndarray
    ) -> np.ndarray:
        """Extract aligned 667-bin window from clutter array using per-trace offsets.

        Args:
            offsets: (n_traces_clutter,) int32 array from align_vertical()

        Returns:
            (667, n_traces_clutter) float32 aligned clutter window
        """
        n_ranges_window = 667
        aligned = np.zeros(
            (n_ranges_window, len(offsets)), dtype=np.float32
        )

        for i, offset in enumerate(offsets):
            offset = int(offset)
            # Bounds check
            if 0 <= offset and offset + n_ranges_window <= self.clutter_n_ranges:
                aligned[:, i] = self.clutter_power[
                    offset : offset + n_ranges_window, i
                ]
            else:
                # Fallback: copy what we can
                available = min(
                    n_ranges_window,
                    self.clutter_n_ranges - max(0, offset),
                )
                if available > 0:
                    src_start = max(0, offset)
                    aligned[:available, i] = self.clutter_power[
                        src_start : src_start + available, i
                    ]
                logger.debug(
                    f"ClutterAligner.apply_offset: Trace {i} offset {offset} "
                    f"out of bounds; copied {available} bins"
                )

        return aligned

    def resample_if_needed(self, aligned: np.ndarray) -> np.ndarray:
        """Resample horizontally if clutter trace count differs from RDR.

        Args:
            aligned: (667, n_traces_clutter) from apply_offset()

        Returns:
            (667, rdr_n_traces) float32 resampled array
        """
        n_ranges, n_traces_clutter = aligned.shape

        if n_traces_clutter == self.rdr_n_traces:
            # No resampling needed
            return aligned

        # Resample horizontally
        trace_ratio = self.rdr_n_traces / n_traces_clutter
        logger.debug(
            f"ClutterAligner: Resampling {n_traces_clutter} → {self.rdr_n_traces} traces "
            f"(ratio={trace_ratio:.2f})"
        )

        # Use scipy zoom for linear interpolation
        resampled = ndimage.zoom(
            aligned, (1.0, trace_ratio), order=1, mode="nearest"
        )

        # Ensure exact output size
        if resampled.shape[1] != self.rdr_n_traces:
            resampled = resampled[:, : self.rdr_n_traces]

        return resampled

    def align_full(self) -> np.ndarray:
        """Perform full alignment: vertical offset + horizontal resample.

        Returns:
            (667, rdr_n_traces) float32 fully aligned clutter window
        """
        offsets = self.align_vertical()
        aligned = self.apply_offset(offsets)
        resampled = self.resample_if_needed(aligned)
        return resampled
