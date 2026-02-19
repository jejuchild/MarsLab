"""
RegolithThicknessEstimator — along-track regolith depth profiling from SHARAD.

Algorithm:
  1. Load SHARAD power array + surface picks (cached)
  2. For each trace, detect near-surface reflector in [surface+lo, surface+hi]
  3. Apply coherence filtering (median + outlier rejection)
  4. Convert TWT → depth using configurable εr
  5. Sample DEM elevation along track (MOLA or HiRISE DTM)
  6. Compute per-trace confidence score
  7. Generate colored overlay segments for map rendering
"""

import logging
import math
import numpy as np
from typing import Dict, List, Optional

from .base import AnalysisModule
from .models import (
    RegolithSample,
    RegolithSummary,
    RegolithResult,
    RegolithParameters,
    OverlaySegment,
)

from analysis.shared.constants import SPEED_OF_LIGHT, SHARAD_SAMPLE_INTERVAL_US
from analysis.shared.coordinates import lon_to_180, centric_to_graphic, cumulative_distance_m
from analysis.shared.dem_sampling import sample_dem_along_track
from analysis.shared.overlay import interpolate_colormap, CMAP_THICKNESS
from analysis.shared.clutter_mask import compute_clutter_mask
from analysis.shared.subsurface_picker import (
    suppress_clutter_adaptive,
    pick_discontinuous_reflectors,
)

logger = logging.getLogger(__name__)

# ── PHASE 1: RTE SHALLOW-MODE CONSTANTS ─────────────────────────────────
RING_GUARD_BINS_DEFAULT = 4      # reject peaks in first 4 bins (default)
RING_GUARD_BINS_SHALLOW = 3      # reject peaks in first 3 bins (shallow mode)
MEDIAN_KERNEL_DEFAULT = 15       # coherence filter kernel (default)
MEDIAN_KERNEL_SHALLOW = 11       # coherence filter kernel (shallow mode)
OUTLIER_THRESHOLD_DEFAULT = 15   # outlier rejection threshold in bins (default)
OUTLIER_THRESHOLD_SHALLOW = 12   # outlier rejection threshold in bins (shallow mode)


def _apply_mode_defaults(
    mode: str,
    search_lo: int,
    search_hi: int,
) -> tuple:
    """Apply mode-based defaults for search window.

    If user specifies explicit search_lo/hi, those take precedence.
    Only apply defaults if user is using the default values.
    """
    if mode == "shallow":
        # Shallow mode defaults: search_lo=2, search_hi=120
        if search_lo == 10:  # user not overriding
            search_lo = 2
        if search_hi == 150:  # user not overriding
            search_hi = 120
    return search_lo, search_hi


class RegolithThicknessEstimator(AnalysisModule):
    """Estimate along-track regolith thickness from a single SHARAD track."""

    def __init__(self):
        self._result: Optional[RegolithResult] = None

    # ────────────────────────────────────────────────────────────────
    # AnalysisModule interface
    # ────────────────────────────────────────────────────────────────

    def run(
        self,
        product_id: str,
        epsilon_r: float = 2.5,
        snr_threshold: float = 3.5,
        search_lo: int = 10,
        search_hi: int = 150,
        dtm_product_id: str = "",
        mode: str = "default",
        epsilon_uncertainty: float = 0.5,
        clutter_mode: str = "off",
        clutter_snr_threshold: float = 3.0,
        clutter_bin_tolerance: int = 3,
    ) -> RegolithResult:
        """Execute the full RTE pipeline. Returns RegolithResult."""
        try:
            self._result = self._run_impl(
                product_id, epsilon_r, snr_threshold,
                search_lo, search_hi, dtm_product_id,
                mode, epsilon_uncertainty,
                clutter_mode, clutter_snr_threshold, clutter_bin_tolerance,
            )
        except Exception as exc:
            logger.exception("RTE pipeline failed for %s", product_id)
            self._result = RegolithResult(success=False, error=str(exc))
        return self._result

    def generate_profile(self) -> List[Dict]:
        if not self._result or not self._result.profile:
            return []
        return [s.model_dump() for s in self._result.profile]

    def generate_overlay(self) -> List[Dict]:
        if not self._result or not self._result.overlay_segments:
            return []
        return [s.model_dump() for s in self._result.overlay_segments]

    def generate_summary(self) -> Dict:
        if not self._result:
            return {"success": False, "error": "Not run yet"}
        d = {"success": self._result.success, "error": self._result.error}
        if self._result.summary:
            d.update(self._result.summary.model_dump())
        return d

    # ────────────────────────────────────────────────────────────────
    # Core implementation
    # ────────────────────────────────────────────────────────────────

    def _run_impl(
        self,
        product_id: str,
        epsilon_r: float,
        snr_threshold: float,
        search_lo: int,
        search_hi: int,
        dtm_product_id: str,
        mode: str = "default",
        epsilon_uncertainty: float = 0.5,
        clutter_mode: str = "off",
        clutter_snr_threshold: float = 3.0,
        clutter_bin_tolerance: int = 3,
    ) -> RegolithResult:
        # Late imports to avoid circular deps at module load
        from api.sharad_highres_router import (
            _get_power,
            _get_geometry,
            _pick_surface,
            _get_aligned_cluttergram,
        )

        logger.info(
            "RTE: product=%s εr=%.2f snr≥%.1f search=[%d,%d] mode=%s",
            product_id, epsilon_r, snr_threshold, search_lo, search_hi, mode,
        )

        # ── PHASE 1: Apply mode defaults ────────────────────────────────
        search_lo, search_hi = _apply_mode_defaults(mode, search_lo, search_hi)
        logger.debug("RTE: after mode=%s defaults: search=[%d,%d]",
                     mode, search_lo, search_hi)

        # ── Step 1: Load SHARAD data ────────────────────────────────
        power, total_rows = _get_power(product_id)
        geom, _ = _get_geometry(product_id)
        surface = _pick_surface(product_id, power)

        lats_centric = geom["lat"][:total_rows]
        lons_360 = geom["lon"][:total_rows]
        lons_180 = lon_to_180(lons_360)
        lats_graphic = centric_to_graphic(lats_centric)

        logger.debug("RTE: loaded %d traces, %d bins", total_rows, power.shape[1])

        # ── PHASE 6.1: Array truncation (geometry index mismatch fix) ──
        # Handle case where arrays have different lengths (e.g., R_3908602)
        n_traces_min = min(len(lats_centric), len(lons_180), len(surface), power.shape[0])
        if n_traces_min < total_rows:
            logger.warning(
                "RTE: Array size mismatch detected; truncating to %d traces "
                "(lat: %d, lon: %d, surface: %d, power: %d)",
                n_traces_min, len(lats_centric), len(lons_180), len(surface), power.shape[0]
            )
            lats_centric = lats_centric[:n_traces_min]
            lons_180 = lons_180[:n_traces_min]
            lats_graphic = centric_to_graphic(lats_centric)
            surface = surface[:n_traces_min]
            power = power[:n_traces_min]
            total_rows = n_traces_min

        # ── Step 2: Sample DEM along track ──────────────────────────
        dem_source, surface_elevs = sample_dem_along_track(
            lats_centric, lons_180, dtm_product_id,
        )

        # ── Step 3: Along-track distance ────────────────────────────
        distances_m = cumulative_distance_m(lats_centric, lons_180)
        distances_km = distances_m / 1000.0

        # ── PHASE 2: Clutter loading and alignment ──────────────────
        clutter_available = False
        clutter_power_aligned = None
        clutter_flagged_arr = np.zeros(total_rows, dtype=bool)
        clutter_snr_arr = np.full(total_rows, np.nan, dtype=np.float32)

        if clutter_mode != "off":
            # Use canonical alignment from SHARAD router to avoid duplicated
            # parsing/layout assumptions in RTE.
            clutter_power_aligned = _get_aligned_cluttergram(product_id)
            clutter_available = clutter_power_aligned is not None
            if clutter_available:
                logger.info("RTE: aligned cluttergram loaded: shape=%s", clutter_power_aligned.shape)
            else:
                logger.debug("RTE: no aligned cluttergram found for %s", product_id)

        # Optional clutter suppression before interface picking.
        power_for_pick = power
        if clutter_available and clutter_mode != "off":
            power_for_pick, clutter_scales, clutter_corr = suppress_clutter_adaptive(
                power=power,
                surface_bins=surface,
                clutter_aligned=clutter_power_aligned,
                search_lo=search_lo,
                search_hi=search_hi,
            )
            logger.info(
                "RTE: adaptive clutter suppression applied "
                "(scale_median=%.3f, corr_median=%.3f)",
                float(np.median(clutter_scales[np.isfinite(clutter_scales)])) if clutter_scales.size else 0.0,
                float(np.median(clutter_corr[np.isfinite(clutter_corr)])) if clutter_corr.size else 0.0,
            )

        # ── Step 4: Vectorized near-surface reflector detection ─────
        n_traces = total_rows
        n_bins = power.shape[1]

        # Pre-allocate output arrays
        detected = np.zeros(n_traces, dtype=bool)
        delta_bins_arr = np.zeros(n_traces, dtype=np.int32)
        snr_arr = np.zeros(n_traces, dtype=np.float32)
        ringing_rejected = np.zeros(n_traces, dtype=bool)  # PHASE 1: track ringing

        valid_surface = surface >= 0
        valid_indices = np.where(valid_surface)[0]

        logger.debug("RTE: %d/%d traces have valid surface pick", len(valid_indices), n_traces)

        ring_guard_bins = RING_GUARD_BINS_SHALLOW if mode == "shallow" else RING_GUARD_BINS_DEFAULT
        pick_arrays = pick_discontinuous_reflectors(
            power=power_for_pick,
            surface_bins=surface,
            search_lo=search_lo,
            search_hi=search_hi,
            min_snr=snr_threshold,
            clutter_aligned=clutter_power_aligned if clutter_available else None,
            ring_guard_bins=ring_guard_bins,
            neighbor_window=3 if mode == "shallow" else 4,
            neighbor_bin_tol=7 if mode == "shallow" else 9,
        )
        detected[:] = pick_arrays.detected
        delta_bins_arr[:] = pick_arrays.delta_bins
        snr_arr[:] = pick_arrays.snr
        ringing_rejected[:] = pick_arrays.ringing_rejected

        n_valid = int(detected.sum())
        logger.info("RTE: raw detections: %d/%d (%.1f%%)",
                     n_valid, n_traces, 100 * n_valid / max(n_traces, 1))

        # ── Step 5: Coherence filtering ─────────────────────────────
        # Use local neighborhood support (from discontinuous picker) as coherence.
        coherence = np.zeros(n_traces, dtype=np.float32)
        if n_valid > 0:
            coherence[detected] = np.clip(pick_arrays.support[detected] / 3.0, 0.35, 1.0).astype(np.float32)

        # ── Step 6: Convert TWT → depth ─────────────────────────────
        velocity = SPEED_OF_LIGHT / math.sqrt(epsilon_r)
        twt_us = delta_bins_arr.astype(np.float64) * SHARAD_SAMPLE_INTERVAL_US
        thickness_m = np.where(
            detected,
            (velocity * twt_us * 1e-6) / 2.0,
            0.0,
        )

        # PHASE 3: Compute epsilon_uncertainty band
        # thickness_low_m corresponds to higher εr (lower depth)
        # thickness_high_m corresponds to lower εr (higher depth)
        thickness_low_m = np.zeros(n_traces, dtype=np.float64)
        thickness_high_m = np.zeros(n_traces, dtype=np.float64)

        if epsilon_uncertainty > 0:
            velocity_low = SPEED_OF_LIGHT / math.sqrt(epsilon_r + epsilon_uncertainty)
            velocity_high = SPEED_OF_LIGHT / math.sqrt(max(epsilon_r - epsilon_uncertainty, 1.0))
            thickness_low_m = np.where(
                detected,
                (velocity_low * twt_us * 1e-6) / 2.0,
                0.0,
            )
            thickness_high_m = np.where(
                detected,
                (velocity_high * twt_us * 1e-6) / 2.0,
                0.0,
            )

        # ── Step 7: Confidence score ────────────────────────────────
        dem_bonus = 1.0 if dem_source == "HiRISE_DTM" else 0.5

        if n_valid > 0:
            median_db = float(np.median(delta_bins_arr[detected]))
        else:
            median_db = 0.0

        confidence = np.zeros(n_traces, dtype=np.float32)
        for i in range(n_traces):
            if not detected[i]:
                confidence[i] = 0.0
                continue
            c_snr = min(snr_arr[i] / 10.0, 1.0) * 0.40
            c_coh = coherence[i] * 0.30
            c_con = max(1.0 - abs(delta_bins_arr[i] - median_db) / 50.0, 0.0) * 0.20
            c_dem = dem_bonus * 0.10
            confidence[i] = min(c_snr + c_coh + c_con + c_dem, 1.0)

        # ── PHASE 2: Clutter masking and confidence penalty ──────────
        if clutter_available and clutter_power_aligned is not None:
            clutter_flagged_arr, clutter_snr_arr = compute_clutter_mask(
                clutter_power_aligned,
                delta_bins_arr,
                search_lo,
                snr_threshold=clutter_snr_threshold,
                bin_tolerance=clutter_bin_tolerance,
            )
            # Apply confidence penalty to clutter-flagged traces
            for i in range(n_traces):
                if clutter_flagged_arr[i] and detected[i]:
                    confidence[i] *= 0.35  # Multiplicative penalty
                    logger.debug(f"Trace {i}: clutter penalty applied ({confidence[i]:.3f})")

        # ── Step 8: Build profile ───────────────────────────────────
        profile: List[RegolithSample] = []
        for i in range(n_traces):
            sample = RegolithSample(
                trace_idx=i,
                lat=round(float(lats_graphic[i]), 5),
                lon=round(float(lons_180[i]), 5),
                along_track_km=round(float(distances_km[i]), 3),
                surface_elev_m=round(float(surface_elevs[i]), 1)
                    if not np.isnan(surface_elevs[i]) else 0.0,
                interface_detected=bool(detected[i]),
                delta_bins=int(delta_bins_arr[i]) if detected[i] else None,
                twt_us=round(float(twt_us[i]), 4) if detected[i] else None,
                thickness_m=round(float(thickness_m[i]), 1) if detected[i] else None,
                # PHASE 3: epsilon_uncertainty band
                thickness_low_m=round(float(thickness_low_m[i]), 1) if detected[i] and epsilon_uncertainty > 0 else None,
                thickness_high_m=round(float(thickness_high_m[i]), 1) if detected[i] and epsilon_uncertainty > 0 else None,
                snr=round(float(snr_arr[i]), 2) if detected[i] else None,
                confidence=round(float(confidence[i]), 3),
                # PHASE 1: surface ringing rejection
                ringing_rejected=bool(ringing_rejected[i]),
                # PHASE 2: clutter fields
                clutter_available=clutter_available,
                clutter_flagged=bool(clutter_flagged_arr[i]) if clutter_available else False,
                clutter_snr=round(float(clutter_snr_arr[i]), 2) if clutter_available and not np.isnan(clutter_snr_arr[i]) else None,
                # PHASE 1: mode parameter
                mode=mode,
            )
            profile.append(sample)

        # ── Step 9: Overlay segments (downsampled to ≤500) ─────────
        overlay = self._build_overlay(
            lats_graphic, lons_180, detected, thickness_m, n_traces,
        )

        # ── Step 10: Summary stats ──────────────────────────────────
        valid_thick = thickness_m[detected]
        valid_snr = snr_arr[detected]
        valid_conf = confidence[detected]

        # PHASE 1: Calculate ring rejection rate
        ring_reject_rate = float(ringing_rejected.sum()) / max(len(valid_indices), 1) if len(valid_indices) > 0 else 0.0

        # PHASE 2: Calculate clutter statistics
        clutter_flag_rate = 0.0
        clutter_trace_mismatch = 0
        if clutter_available:
            clutter_flag_rate = float(clutter_flagged_arr.sum()) / max(n_valid, 1) if n_valid > 0 else 0.0

        # PHASE 6.1: Calculate truncated trace count
        truncated_traces_count = max(0, len(geom["lat"]) - total_rows) if "lat" in geom else 0

        summary = RegolithSummary(
            product_id=product_id,
            epsilon_r=epsilon_r,
            total_traces=n_traces,
            valid_traces=n_valid,
            detection_rate=round(n_valid / max(n_traces, 1), 4),
            thickness_mean_m=round(float(valid_thick.mean()), 1) if n_valid else None,
            thickness_median_m=round(float(np.median(valid_thick)), 1) if n_valid else None,
            thickness_std_m=round(float(valid_thick.std()), 1) if n_valid else None,
            thickness_min_m=round(float(valid_thick.min()), 1) if n_valid else None,
            thickness_max_m=round(float(valid_thick.max()), 1) if n_valid else None,
            mean_snr=round(float(valid_snr.mean()), 2) if n_valid else None,
            mean_confidence=round(float(valid_conf.mean()), 3) if n_valid else None,
            dem_source=dem_source,
            total_distance_km=round(float(distances_km[-1]), 2) if n_traces else 0.0,
            # PHASE 1: shallow mode fields
            shallow_mode_enabled=(mode == "shallow"),
            ring_reject_rate=round(ring_reject_rate, 4),
            # PHASE 2: clutter fields
            clutter_available=clutter_available,
            clutter_flag_rate=round(clutter_flag_rate, 4),
            clutter_trace_mismatch=clutter_trace_mismatch,
            # PHASE 6.1: truncated traces count
            truncated_traces_count=truncated_traces_count,
            # PHASE 3: epsilon_uncertainty
            epsilon_uncertainty=epsilon_uncertainty,
        )

        params = RegolithParameters(
            epsilon_r=epsilon_r,
            snr_threshold=snr_threshold,
            search_lo=search_lo,
            search_hi=search_hi,
            dem_source=dem_source,
            # PHASE 1: mode parameter
            mode=mode,
            # PHASE 2: clutter parameters
            clutter_mode=clutter_mode,
            clutter_snr_threshold=clutter_snr_threshold,
            clutter_bin_tolerance=clutter_bin_tolerance,
            # PHASE 3: epsilon_uncertainty parameter
            epsilon_uncertainty=epsilon_uncertainty,
        )

        logger.info(
            "RTE: done — %d valid, mean=%.1fm, detection=%.1f%%",
            n_valid,
            summary.thickness_mean_m or 0,
            summary.detection_rate * 100,
        )

        return RegolithResult(
            success=True,
            summary=summary,
            profile=profile,
            overlay_segments=overlay,
            parameters=params,
        )

    # ────────────────────────────────────────────────────────────────
    # Helpers (RTE-specific — not shared)
    # ────────────────────────────────────────────────────────────────

    @staticmethod
    def _coherence_filter(
        detected: np.ndarray,
        delta_bins: np.ndarray,
        snr: np.ndarray,
        coherence: np.ndarray,
        median_kernel: int = 15,
        outlier_threshold: int = 15,
    ) -> np.ndarray:
        """Apply median-based coherence filtering to raw detections.

        Args:
            detected: boolean array of detection flags
            delta_bins: delta bin indices (or 0 if not detected)
            snr: SNR values
            coherence: coherence score array (updated in place)
            median_kernel: kernel size for median filter (PHASE 1: mode-aware)
            outlier_threshold: bin distance threshold for outlier rejection (PHASE 1: mode-aware)
        """
        from scipy.ndimage import median_filter

        det_idx = np.where(detected)[0]
        vals = delta_bins[det_idx].astype(np.float64)

        # Median filter on valid picks only
        filtered = median_filter(vals, size=min(median_kernel, len(vals)))

        # Outlier rejection: > threshold bins from smoothed trend
        outlier_mask = np.abs(vals - filtered) > outlier_threshold
        for j, idx in enumerate(det_idx):
            if outlier_mask[j]:
                detected[idx] = False
                delta_bins[idx] = 0
                snr[idx] = 0

        # Continuity: require ≥5 consecutive valid traces
        run_start = -1
        for i in range(len(detected)):
            if detected[i]:
                if run_start < 0:
                    run_start = i
            else:
                if run_start >= 0:
                    run_len = i - run_start
                    if run_len >= 5:
                        coherence[run_start:i] = 1.0
                    elif run_len >= 2:
                        coherence[run_start:i] = 0.5
                    run_start = -1
        # Handle trailing run
        if run_start >= 0:
            run_len = len(detected) - run_start
            if run_len >= 5:
                coherence[run_start:] = 1.0
            elif run_len >= 2:
                coherence[run_start:] = 0.5

        return coherence

    @staticmethod
    def _build_overlay(
        lats: np.ndarray,
        lons: np.ndarray,
        detected: np.ndarray,
        thickness_m: np.ndarray,
        n_traces: int,
        max_segments: int = 500,
    ) -> List[OverlaySegment]:
        """Build color-coded track overlay, downsampled to max_segments."""
        if n_traces < 2:
            return []

        step = max(1, n_traces // max_segments)
        segments: List[OverlaySegment] = []

        for i in range(0, n_traces - step, step):
            j = min(i + step, n_traces - 1)

            # Average thickness over segment
            seg_det = detected[i:j + 1]
            if seg_det.any():
                seg_thick = float(np.mean(thickness_m[i:j + 1][seg_det]))
            else:
                seg_thick = None

            segments.append(OverlaySegment(
                start_lat=round(float(lats[i]), 5),
                start_lon=round(float(lons[i]), 5),
                end_lat=round(float(lats[j]), 5),
                end_lon=round(float(lons[j]), 5),
                thickness_m=round(seg_thick, 1) if seg_thick is not None else None,
                color=interpolate_colormap(seg_thick, CMAP_THICKNESS),
            ))

        return segments
