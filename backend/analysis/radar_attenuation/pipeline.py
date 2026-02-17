"""
RadarAttenuationMapper — along-track radar attenuation profiling from SHARAD.

Algorithm:
  1. Load SHARAD power array + surface picks (cached)
  2. For each trace, detect near-surface reflector (reuses RTE pattern)
  3. Apply coherence filtering (median + outlier rejection)
  4. Compute two-way attenuation: α = (P_surf_dB - P_sub_dB) / (2 * depth_m)
  5. Classify material transparency per trace
  6. Sample DEM elevation along track
  7. Generate colored overlay segments for map rendering
"""

import logging
import math
import numpy as np
from typing import Dict, List, Optional

from analysis.shared.base import AnalysisModule
from .models import (
    AttenuationSample,
    AttenuationSummary,
    AttenuationResult,
    AttenuationParameters,
    OverlaySegment,
)

from analysis.shared.constants import SPEED_OF_LIGHT, SHARAD_SAMPLE_INTERVAL_US
from analysis.shared.coordinates import lon_to_180, centric_to_graphic, cumulative_distance_m
from analysis.shared.dem_sampling import sample_dem_along_track
from analysis.shared.overlay import interpolate_colormap, CMAP_ATTENUATION

logger = logging.getLogger(__name__)

# ── Transparency classification thresholds (dB/m) ─────────────────
TRANSPARENCY_CLASSES = [
    (0.005, "Pure ice"),
    (0.02,  "Clean ice / low-loss rock"),
    (0.05,  "Dusty ice / porous basalt"),
    (0.08,  "Moderate loss (mixed)"),
    (float("inf"), "Clay-rich / briny"),
]


def _classify_transparency(alpha_dBm: float) -> str:
    """Map attenuation coefficient to material transparency class."""
    for threshold, label in TRANSPARENCY_CLASSES:
        if alpha_dBm < threshold:
            return label
    return TRANSPARENCY_CLASSES[-1][1]


class RadarAttenuationMapper(AnalysisModule):
    """Compute along-track radar attenuation from a single SHARAD track."""

    def __init__(self):
        self._result: Optional[AttenuationResult] = None

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
    ) -> AttenuationResult:
        """Execute the full attenuation mapping pipeline."""
        try:
            self._result = self._run_impl(
                product_id, epsilon_r, snr_threshold,
                search_lo, search_hi, dtm_product_id,
            )
        except Exception as exc:
            logger.exception("Attenuation pipeline failed for %s", product_id)
            self._result = AttenuationResult(success=False, error=str(exc))
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
        d: Dict = {"success": self._result.success, "error": self._result.error}
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
    ) -> AttenuationResult:
        from api.sharad_highres_router import (
            _get_power,
            _get_geometry,
            _pick_surface,
        )

        logger.info(
            "Attenuation: product=%s εr=%.2f snr≥%.1f search=[%d,%d]",
            product_id, epsilon_r, snr_threshold, search_lo, search_hi,
        )

        # ── Step 1: Load SHARAD data ────────────────────────────────
        power, total_rows = _get_power(product_id)
        geom, _ = _get_geometry(product_id)
        surface = _pick_surface(product_id, power)

        lats_centric = geom["lat"][:total_rows]
        lons_360 = geom["lon"][:total_rows]
        lons_180 = lon_to_180(lons_360)
        lats_graphic = centric_to_graphic(lats_centric)

        # ── Step 2: Sample DEM along track ──────────────────────────
        dem_source, surface_elevs = sample_dem_along_track(
            lats_centric, lons_180, dtm_product_id,
        )

        # ── Step 3: Along-track distance ────────────────────────────
        distances_m = cumulative_distance_m(lats_centric, lons_180)
        distances_km = distances_m / 1000.0

        # ── Step 4: Subsurface reflector detection (RTE pattern) ────
        n_traces = total_rows
        n_bins = power.shape[1]

        detected = np.zeros(n_traces, dtype=bool)
        delta_bins_arr = np.zeros(n_traces, dtype=np.int32)
        snr_arr = np.zeros(n_traces, dtype=np.float32)

        valid_surface = surface >= 0
        valid_indices = np.where(valid_surface)[0]

        for i in valid_indices:
            sb = int(surface[i])
            lo = sb + search_lo
            hi = min(sb + search_hi, n_bins)
            if hi <= lo + 5:
                continue

            band = power[i, lo:hi].astype(np.float64)
            noise = float(np.median(band)) + 1e-12
            peak_idx = int(np.argmax(band))
            peak_val = float(band[peak_idx])
            snr = peak_val / noise

            if peak_idx < 5:
                continue

            if snr >= snr_threshold:
                detected[i] = True
                delta_bins_arr[i] = search_lo + peak_idx
                snr_arr[i] = snr

        n_valid = int(detected.sum())
        logger.info("Attenuation: raw detections: %d/%d (%.1f%%)",
                     n_valid, n_traces, 100 * n_valid / max(n_traces, 1))

        # ── Step 5: Coherence filtering ─────────────────────────────
        coherence = np.zeros(n_traces, dtype=np.float32)
        if n_valid >= 10:
            coherence = self._coherence_filter(
                detected, delta_bins_arr, snr_arr, coherence,
            )
        elif n_valid > 0:
            coherence[detected] = 0.5

        # Recount after filtering
        n_valid = int(detected.sum())

        # ── Step 6: Compute attenuation ─────────────────────────────
        velocity = SPEED_OF_LIGHT / math.sqrt(epsilon_r)
        twt_us = delta_bins_arr.astype(np.float64) * SHARAD_SAMPLE_INTERVAL_US
        depth_m = np.where(
            detected,
            (velocity * twt_us * 1e-6) / 2.0,
            0.0,
        )

        # Power in dB (surface and subsurface)
        surface_power_dB = np.full(n_traces, np.nan, dtype=np.float64)
        subsurface_power_dB = np.full(n_traces, np.nan, dtype=np.float64)
        alpha_dBm = np.full(n_traces, np.nan, dtype=np.float64)
        transparency = [None] * n_traces

        for i in valid_indices:
            if not detected[i]:
                continue
            sb = int(surface[i])
            sub_bin = sb + delta_bins_arr[i]
            if sub_bin >= n_bins:
                continue

            p_surf = max(float(power[i, sb]), 1e-12)
            p_sub = max(float(power[i, sub_bin]), 1e-12)
            p_surf_dB = 10.0 * math.log10(p_surf)
            p_sub_dB = 10.0 * math.log10(p_sub)

            surface_power_dB[i] = p_surf_dB
            subsurface_power_dB[i] = p_sub_dB

            d = depth_m[i]
            if d > 5.0:  # Need sufficient depth for reliable α
                a = (p_surf_dB - p_sub_dB) / (2.0 * d)
                if a >= 0:  # Physically valid (surface stronger than subsurface)
                    alpha_dBm[i] = a
                    transparency[i] = _classify_transparency(a)

        # ── Step 7: Build profile ───────────────────────────────────
        profile: List[AttenuationSample] = []
        for i in range(n_traces):
            det = bool(detected[i])
            has_alpha = det and not np.isnan(alpha_dBm[i])
            sample = AttenuationSample(
                trace_idx=i,
                lat=round(float(lats_graphic[i]), 5),
                lon=round(float(lons_180[i]), 5),
                along_track_km=round(float(distances_km[i]), 3),
                surface_elev_m=round(float(surface_elevs[i]), 1)
                    if not np.isnan(surface_elevs[i]) else 0.0,
                interface_detected=det,
                surface_power_dB=round(float(surface_power_dB[i]), 2)
                    if det and not np.isnan(surface_power_dB[i]) else None,
                subsurface_power_dB=round(float(subsurface_power_dB[i]), 2)
                    if det and not np.isnan(subsurface_power_dB[i]) else None,
                depth_m=round(float(depth_m[i]), 1) if det else None,
                alpha_dBm=round(float(alpha_dBm[i]), 4) if has_alpha else None,
                transparency=transparency[i],
                snr=round(float(snr_arr[i]), 2) if det else None,
                confidence=round(float(coherence[i]), 3) if det else None,
            )
            profile.append(sample)

        # ── Step 8: Overlay segments ────────────────────────────────
        alpha_valid = np.where(~np.isnan(alpha_dBm), True, False)
        alpha_values = np.where(alpha_valid, alpha_dBm, 0.0)
        overlay = self._build_overlay(
            lats_graphic, lons_180, alpha_valid, alpha_values, n_traces,
        )

        # ── Step 9: Summary stats ──────────────────────────────────
        valid_alpha = alpha_dBm[~np.isnan(alpha_dBm)]
        n_alpha = len(valid_alpha)

        # Count transparency classes
        trans_counts: Dict[str, int] = {}
        for t in transparency:
            if t is not None:
                trans_counts[t] = trans_counts.get(t, 0) + 1

        dominant_trans = max(trans_counts, key=trans_counts.get) if trans_counts else None

        summary = AttenuationSummary(
            product_id=product_id,
            epsilon_r=epsilon_r,
            total_traces=n_traces,
            valid_traces=n_alpha,
            detection_rate=round(n_alpha / max(n_traces, 1), 4),
            alpha_mean_dBm=round(float(valid_alpha.mean()), 4) if n_alpha else None,
            alpha_median_dBm=round(float(np.median(valid_alpha)), 4) if n_alpha else None,
            alpha_std_dBm=round(float(valid_alpha.std()), 4) if n_alpha else None,
            dominant_transparency=dominant_trans,
            transparency_counts=trans_counts,
            dem_source=dem_source,
            total_distance_km=round(float(distances_km[-1]), 2) if n_traces else 0.0,
        )

        params = AttenuationParameters(
            epsilon_r=epsilon_r,
            snr_threshold=snr_threshold,
            search_lo=search_lo,
            search_hi=search_hi,
            dem_source=dem_source,
        )

        logger.info(
            "Attenuation: done — %d valid α, mean=%.4f dB/m, detection=%.1f%%",
            n_alpha,
            summary.alpha_mean_dBm or 0,
            summary.detection_rate * 100,
        )

        return AttenuationResult(
            success=True,
            summary=summary,
            profile=profile,
            overlay_segments=overlay,
            parameters=params,
        )

    # ────────────────────────────────────────────────────────────────
    # Helpers
    # ────────────────────────────────────────────────────────────────

    @staticmethod
    def _coherence_filter(
        detected: np.ndarray,
        delta_bins: np.ndarray,
        snr: np.ndarray,
        coherence: np.ndarray,
    ) -> np.ndarray:
        """Apply median-based coherence filtering to raw detections."""
        from scipy.ndimage import median_filter

        det_idx = np.where(detected)[0]
        vals = delta_bins[det_idx].astype(np.float64)

        filtered = median_filter(vals, size=min(15, len(vals)))

        outlier_mask = np.abs(vals - filtered) > 15
        for j, idx in enumerate(det_idx):
            if outlier_mask[j]:
                detected[idx] = False
                delta_bins[idx] = 0
                snr[idx] = 0

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
        alpha_values: np.ndarray,
        n_traces: int,
        max_segments: int = 500,
    ) -> List[OverlaySegment]:
        """Build color-coded attenuation overlay, downsampled to max_segments."""
        if n_traces < 2:
            return []

        step = max(1, n_traces // max_segments)
        segments: List[OverlaySegment] = []

        for i in range(0, n_traces - step, step):
            j = min(i + step, n_traces - 1)

            seg_det = detected[i:j + 1]
            if seg_det.any():
                seg_val = float(np.mean(alpha_values[i:j + 1][seg_det]))
            else:
                seg_val = None

            segments.append(OverlaySegment(
                start_lat=round(float(lats[i]), 5),
                start_lon=round(float(lons[i]), 5),
                end_lat=round(float(lats[j]), 5),
                end_lon=round(float(lons[j]), 5),
                alpha_dBm=round(seg_val, 4) if seg_val is not None else None,
                color=interpolate_colormap(seg_val, CMAP_ATTENUATION),
            ))

        return segments
