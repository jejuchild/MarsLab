from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..shared.constants import SHARAD_SAMPLE_INTERVAL_S, SPEED_OF_LIGHT
from .rdr_loader import TrackData


@dataclass
class DielectricResult:
    lat: np.ndarray
    lon: np.ndarray
    consistency: np.ndarray
    depth_bin: list[str]
    epsilon_r: np.ndarray
    snr: np.ndarray
    product_id: str


def _detect_reflectors(
    power: np.ndarray,
    surface: np.ndarray,
    min_depth_bins: int = 5,
    max_depth_bins: int = 200,
    snr_threshold: float = 3.0,
    continuity_tol_bins: int = 5,
    min_segment_len: int = 10,
) -> list[dict[str, np.ndarray]]:
    n_traces, n_bins = power.shape
    picks = np.full(n_traces, -1, dtype=np.int32)
    snrs = np.zeros(n_traces, dtype=np.float64)

    for t in range(n_traces):
        s = int(surface[t])
        if s < 0:
            continue
        lo = s + min_depth_bins
        hi = min(n_bins, s + max_depth_bins)
        if hi <= lo:
            continue

        band = np.asarray(power[t, lo:hi], dtype=np.float64)
        if band.size == 0:
            continue
        bg = float(np.median(band))
        if not np.isfinite(bg) or bg <= 0:
            continue
        peak = int(np.argmax(band))
        peak_val = float(band[peak])
        local_snr = peak_val / bg
        if np.isfinite(local_snr) and local_snr >= snr_threshold:
            picks[t] = lo + peak
            snrs[t] = local_snr

    segments: list[dict[str, np.ndarray]] = []
    start = -1
    prev_pick = -1

    for t in range(n_traces):
        p = int(picks[t])
        if p >= 0:
            if start < 0:
                start = t
                prev_pick = p
            elif abs(p - prev_pick) <= continuity_tol_bins:
                prev_pick = p
            else:
                if (t - start) >= min_segment_len:
                    seg_slice = slice(start, t)
                    valid = picks[seg_slice] >= 0
                    if np.any(valid):
                        seg_traces = np.arange(start, t, dtype=np.int32)[valid]
                        seg_picks = picks[seg_slice][valid]
                        seg_snrs = snrs[seg_slice][valid]
                        segments.append(
                            {
                                "trace_idx": seg_traces,
                                "pick_bins": seg_picks,
                                "snr": seg_snrs,
                            }
                        )
                start = t
                prev_pick = p
        else:
            if start >= 0 and (t - start) >= min_segment_len:
                seg_slice = slice(start, t)
                valid = picks[seg_slice] >= 0
                if np.any(valid):
                    seg_traces = np.arange(start, t, dtype=np.int32)[valid]
                    seg_picks = picks[seg_slice][valid]
                    seg_snrs = snrs[seg_slice][valid]
                    segments.append(
                        {
                            "trace_idx": seg_traces,
                            "pick_bins": seg_picks,
                            "snr": seg_snrs,
                        }
                    )
            start = -1
            prev_pick = -1

    if start >= 0 and (n_traces - start) >= min_segment_len:
        seg_slice = slice(start, n_traces)
        valid = picks[seg_slice] >= 0
        if np.any(valid):
            seg_traces = np.arange(start, n_traces, dtype=np.int32)[valid]
            seg_picks = picks[seg_slice][valid]
            seg_snrs = snrs[seg_slice][valid]
            segments.append(
                {
                    "trace_idx": seg_traces,
                    "pick_bins": seg_picks,
                    "snr": seg_snrs,
                }
            )

    return segments


def _epsilon_to_consistency(epsilon_r: float) -> float:
    """SWIM2 RD consistency score from real dielectric permittivity.

    Formula (swim.psi.edu/SWIM2Products.php):
        C_rd = -0.5 * epsilon_r + 2.5  (clamped to [-1, +1])

    Calibration:
        epsilon_r = 3.0  (pure ice)  -> C_rd = +1.0
        epsilon_r = 5.0  (mixed)     -> C_rd =  0.0
        epsilon_r = 7.0  (rock)      -> C_rd = -1.0
    """
    c_rd = -0.5 * epsilon_r + 2.5
    return float(max(-1.0, min(1.0, c_rd)))

def compute_dielectric(track: TrackData) -> DielectricResult | None:
    segments = _detect_reflectors(track.power, track.surface_bins)
    if not segments:
        return None

    lat_out: list[float] = []
    lon_out: list[float] = []
    consistency_out: list[float] = []
    depth_bins_out: list[str] = []
    epsilon_out: list[float] = []
    snr_out: list[float] = []

    for seg in segments:
        trace_idx = np.asarray(seg["trace_idx"], dtype=np.int32)
        pick_bins = np.asarray(seg["pick_bins"], dtype=np.int32)
        seg_snr = np.asarray(seg["snr"], dtype=np.float64)
        if trace_idx.size == 0:
            continue

        mid_i = trace_idx.size // 2
        mid_trace = int(trace_idx[mid_i])
        refl_bin_mid = int(pick_bins[mid_i])
        surf_bin_mid = int(track.surface_bins[mid_trace])
        if surf_bin_mid < 0:
            continue

        delta_bins = refl_bin_mid - surf_bin_mid
        if delta_bins <= 0:
            continue

        delta_t_s = delta_bins * SHARAD_SAMPLE_INTERVAL_S
        d_apparent_m = SPEED_OF_LIGHT * delta_t_s / 2.0
        d_ice_m = d_apparent_m / np.sqrt(3.15)
        # SWIM depth zones: 1-5m and >5m.
        # SHARAD vertical resolution in ice ≈ 8.5m, so most detections
        # fall in the >5m zone.  Keep a 1m floor to reject surface clutter.
        if d_ice_m < 1.0:
            continue
        depth_bin = "1-5m" if d_ice_m < 5.0 else "5m-plus"

        surf_bins_seg = track.surface_bins[trace_idx]
        valid = (surf_bins_seg >= 0) & (pick_bins >= 0)
        if not np.any(valid):
            continue
        trace_valid = trace_idx[valid]
        refl_valid = pick_bins[valid]
        surf_valid = surf_bins_seg[valid].astype(np.int32)

        p_refl = np.mean(track.power[trace_valid, refl_valid], dtype=np.float64)
        p_surf = np.mean(track.power[trace_valid, surf_valid], dtype=np.float64)
        if not np.isfinite(p_refl) or not np.isfinite(p_surf) or p_refl <= 0 or p_surf <= 0:
            continue

        r_amp = float(np.sqrt(p_refl / (p_surf + 1e-20)))
        r_amp = min(r_amp, 0.95)
        epsilon_r = float(((1.0 + r_amp) / (1.0 - r_amp)) ** 2)
        if not np.isfinite(epsilon_r):
            continue

        consistency = _epsilon_to_consistency(epsilon_r)
        lat = float(track.lat[mid_trace])
        lon = float(track.lon[mid_trace])
        if not np.isfinite(lat) or not np.isfinite(lon):
            continue

        lat_out.append(lat)
        lon_out.append(lon)
        consistency_out.append(consistency)
        depth_bins_out.append(depth_bin)
        epsilon_out.append(epsilon_r)
        snr_out.append(float(np.mean(seg_snr)))

    if not lat_out:
        return None

    return DielectricResult(
        lat=np.asarray(lat_out, dtype=np.float64),
        lon=np.asarray(lon_out, dtype=np.float64),
        consistency=np.asarray(consistency_out, dtype=np.float32),
        depth_bin=depth_bins_out,
        epsilon_r=np.asarray(epsilon_out, dtype=np.float64),
        snr=np.asarray(snr_out, dtype=np.float64),
        product_id=track.product_id,
    )
