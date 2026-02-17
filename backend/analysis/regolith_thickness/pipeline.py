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

logger = logging.getLogger(__name__)

# ── Physical constants ──────────────────────────────────────────────
SPEED_OF_LIGHT = 299_792_458.0  # m/s
SHARAD_SAMPLE_INTERVAL_US = 3.0 / 80.0  # 0.0375 µs per range bin
MARS_MEAN_RADIUS_M = (2 * 3_396_190.0 + 3_376_200.0) / 3.0
N_RANGE_BINS = 667

# ── Overlay colormap boundaries ─────────────────────────────────────
# thin → thick  ≡  blue → yellow → red
_CMAP_STOPS = [
    (0,   [70, 130, 230, 220]),    # blue  (thin, 0 m)
    (20,  [70, 200, 220, 220]),    # cyan  (20 m)
    (50,  [240, 220, 80, 220]),    # yellow (50 m)
    (80,  [230, 130, 50, 220]),    # orange (80 m)
    (150, [210, 50, 50, 220]),     # red   (≥150 m)
]


def _thickness_to_color(thickness_m: Optional[float]) -> List[float]:
    """Map thickness to RGBA (0-255) using a blue→red colormap."""
    if thickness_m is None:
        return [120, 120, 120, 120]  # gray for no detection

    t = max(0.0, thickness_m)
    # Find enclosing stops
    for i in range(len(_CMAP_STOPS) - 1):
        lo_t, lo_c = _CMAP_STOPS[i]
        hi_t, hi_c = _CMAP_STOPS[i + 1]
        if t <= hi_t:
            frac = (t - lo_t) / max(hi_t - lo_t, 1e-6)
            return [
                int(lo_c[j] + frac * (hi_c[j] - lo_c[j]))
                for j in range(4)
            ]
    # Clamp to max color
    return list(_CMAP_STOPS[-1][1])


def _haversine_vec(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Cumulative great-circle distance on Mars (metres), vectorized."""
    lat_r = np.radians(lats)
    lon_r = np.radians(lons)
    dlat = np.diff(lat_r)
    dlon = np.diff(lon_r)
    a = (np.sin(dlat / 2) ** 2
         + np.cos(lat_r[:-1]) * np.cos(lat_r[1:]) * np.sin(dlon / 2) ** 2)
    seg = MARS_MEAN_RADIUS_M * 2 * np.arcsin(np.sqrt(np.minimum(a, 1.0)))
    return np.concatenate([[0.0], np.cumsum(seg)])


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
    ) -> RegolithResult:
        """Execute the full RTE pipeline. Returns RegolithResult."""
        try:
            self._result = self._run_impl(
                product_id, epsilon_r, snr_threshold,
                search_lo, search_hi, dtm_product_id,
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
    ) -> RegolithResult:
        # Late imports to avoid circular deps at module load
        from api.sharad_highres_router import (
            _get_power,
            _get_geometry,
            _pick_surface,
            _lon_to_180,
            _centric_to_graphic,
        )

        logger.info(
            "RTE: product=%s εr=%.2f snr≥%.1f search=[%d,%d]",
            product_id, epsilon_r, snr_threshold, search_lo, search_hi,
        )

        # ── Step 1: Load SHARAD data ────────────────────────────────
        power, total_rows = _get_power(product_id)
        geom, _ = _get_geometry(product_id)
        surface = _pick_surface(product_id, power)

        lats_centric = geom["lat"][:total_rows]
        lons_360 = geom["lon"][:total_rows]
        lons_180 = _lon_to_180(lons_360)
        lats_graphic = _centric_to_graphic(lats_centric)

        logger.debug("RTE: loaded %d traces, %d bins", total_rows, power.shape[1])

        # ── Step 2: Sample DEM along track ──────────────────────────
        dem_source, surface_elevs = self._sample_dem(
            lats_centric, lons_180, dtm_product_id,
        )

        # ── Step 3: Along-track distance ────────────────────────────
        distances_m = _haversine_vec(lats_centric, lons_180)
        distances_km = distances_m / 1000.0

        # ── Step 4: Vectorized near-surface reflector detection ─────
        n_traces = total_rows
        n_bins = power.shape[1]

        # Pre-allocate output arrays
        detected = np.zeros(n_traces, dtype=bool)
        delta_bins_arr = np.zeros(n_traces, dtype=np.int32)
        snr_arr = np.zeros(n_traces, dtype=np.float32)

        valid_surface = surface >= 0
        valid_indices = np.where(valid_surface)[0]

        logger.debug("RTE: %d/%d traces have valid surface pick",
                      len(valid_indices), n_traces)

        for i in valid_indices:
            sb = int(surface[i])
            lo = sb + search_lo
            hi = min(sb + search_hi, n_bins)
            if hi <= lo + 5:
                continue  # not enough room to search

            band = power[i, lo:hi].astype(np.float64)
            noise = float(np.median(band)) + 1e-12
            peak_idx = int(np.argmax(band))
            peak_val = float(band[peak_idx])
            snr = peak_val / noise

            # Reject peaks in first 5 bins (surface ringing artifact)
            if peak_idx < 5:
                continue

            if snr >= snr_threshold:
                detected[i] = True
                delta_bins_arr[i] = search_lo + peak_idx
                snr_arr[i] = snr

        n_valid = int(detected.sum())
        logger.info("RTE: raw detections: %d/%d (%.1f%%)",
                     n_valid, n_traces, 100 * n_valid / max(n_traces, 1))

        # ── Step 5: Coherence filtering ─────────────────────────────
        coherence = np.zeros(n_traces, dtype=np.float32)  # 0=none, 0.3=interp, 1=coherent

        if n_valid >= 10:
            coherence = self._coherence_filter(
                detected, delta_bins_arr, snr_arr, coherence,
            )
        elif n_valid > 0:
            # Too few detections for filtering — keep all, low coherence
            coherence[detected] = 0.5

        # ── Step 6: Convert TWT → depth ─────────────────────────────
        velocity = SPEED_OF_LIGHT / math.sqrt(epsilon_r)
        twt_us = delta_bins_arr.astype(np.float64) * SHARAD_SAMPLE_INTERVAL_US
        thickness_m = np.where(
            detected,
            (velocity * twt_us * 1e-6) / 2.0,
            0.0,
        )

        # ── Step 7: Confidence score ────────────────────────────────
        dem_bonus = 1.0 if dem_source == "HiRISE_DTM" else 0.5

        # Median delta_bins for consistency term
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

        # ── Step 8: Build profile ───────────────────────────────────
        # Downsample for output: keep every trace (typically ~2k–10k)
        # Frontend handles rendering; we cap overlay segments separately
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
                snr=round(float(snr_arr[i]), 2) if detected[i] else None,
                confidence=round(float(confidence[i]), 3),
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
        )

        params = RegolithParameters(
            epsilon_r=epsilon_r,
            snr_threshold=snr_threshold,
            search_lo=search_lo,
            search_hi=search_hi,
            dem_source=dem_source,
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

        # Median filter on valid picks only
        filtered = median_filter(vals, size=min(15, len(vals)))

        # Outlier rejection: > 15 bins from smoothed trend
        outlier_mask = np.abs(vals - filtered) > 15
        for j, idx in enumerate(det_idx):
            if outlier_mask[j]:
                detected[idx] = False
                delta_bins[idx] = 0
                snr[idx] = 0

        # Continuity: require ≥5 consecutive valid traces
        # Walk through and label runs
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
                    # Isolated short runs get partial coherence
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
    def _sample_dem(
        lats_centric: np.ndarray,
        lons_180: np.ndarray,
        dtm_product_id: str,
    ) -> tuple:
        """Sample DEM elevation along track. Returns (source_name, elevations)."""
        n = len(lats_centric)
        elevations = np.full(n, np.nan, dtype=np.float64)

        # Try HiRISE DTM if specified
        if dtm_product_id:
            try:
                from api.sharad_report import load_dtm_along_track
                dtm_elevs = load_dtm_along_track(
                    dtm_product_id, lats_centric, lons_180,
                )
                if dtm_elevs is not None:
                    valid = ~np.isnan(dtm_elevs)
                    if valid.sum() > n * 0.3:
                        elevations = dtm_elevs
                        return "HiRISE_DTM", elevations
            except Exception:
                logger.debug("HiRISE DTM fallback to MOLA: %s", dtm_product_id)

        # MOLA fallback (always available)
        try:
            import rasterio
            from rasterio.windows import Window
            from api.sharad_highres_router import _get_dem

            ds = _get_dem()
            inv_transform = ~ds.transform
            cols, rows = inv_transform * (lons_180, lats_centric)
            cols = np.round(cols).astype(int)
            rows = np.round(rows).astype(int)
            cols = np.clip(cols, 0, ds.width - 1)
            rows = np.clip(rows, 0, ds.height - 1)

            row_min, row_max = int(rows.min()), int(rows.max())
            col_min, col_max = int(cols.min()), int(cols.max())
            window = Window(col_min, row_min,
                            col_max - col_min + 1, row_max - row_min + 1)
            block = ds.read(1, window=window).astype(np.float64)
            if ds.nodata is not None:
                block[block == ds.nodata] = np.nan

            local_rows = rows - row_min
            local_cols = cols - col_min
            elevations = block[local_rows, local_cols]
        except Exception:
            logger.warning("MOLA DEM sampling failed; elevations will be NaN")

        return "MOLA", elevations

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
                color=_thickness_to_color(seg_thick),
            ))

        return segments
