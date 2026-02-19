"""Clutter-aware subsurface reflector picking for discontinuous interfaces.

This module avoids the "single long horizontal horizon" failure mode by:
1) optional adaptive clutter suppression (correlation-gated, per-trace scaling)
2) local-contrast peak scoring (not just argmax)
3) neighborhood support filtering that allows gaps/discontinuous segments
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import find_peaks


def _mad(x: np.ndarray) -> float:
    """Robust scale estimate (median absolute deviation)."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 0.0
    med = float(np.median(x))
    return float(np.median(np.abs(x - med)))


def _to_trace_major_clutter(
    clutter_aligned: Optional[np.ndarray],
    n_traces: int,
    n_bins: int,
) -> Optional[np.ndarray]:
    """Normalize clutter layout to (traces, bins)."""
    if clutter_aligned is None:
        return None
    c = np.asarray(clutter_aligned, dtype=np.float32)
    if c.ndim != 2:
        return None
    if c.shape == (n_traces, n_bins):
        return c
    if c.shape == (n_bins, n_traces):
        return c.T
    return None


def suppress_clutter_adaptive(
    power: np.ndarray,
    surface_bins: np.ndarray,
    clutter_aligned: Optional[np.ndarray],
    search_lo: int,
    search_hi: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Suppress clutter using correlation-gated adaptive subtraction.

    Returns:
      suppressed_power: same shape as power (trace-major)
      scales: per-trace subtraction scale
      corr: per-trace clutter/rdr correlation within search band
    """
    p = np.asarray(power, dtype=np.float32)
    n_traces, n_bins = p.shape
    c_tm = _to_trace_major_clutter(clutter_aligned, n_traces, n_bins)
    if c_tm is None:
        return p.copy(), np.zeros(n_traces, dtype=np.float32), np.zeros(n_traces, dtype=np.float32)

    out = p.copy()
    scales = np.zeros(n_traces, dtype=np.float32)
    corr_arr = np.zeros(n_traces, dtype=np.float32)

    for i in range(n_traces):
        sb = int(surface_bins[i]) if i < len(surface_bins) else -1
        if sb < 0:
            continue
        lo = max(0, sb + search_lo)
        hi = min(n_bins, sb + search_hi)
        if hi <= lo + 8:
            continue

        r = p[i, lo:hi].astype(np.float64)
        c = c_tm[i, lo:hi].astype(np.float64)
        if np.all(c <= 0):
            continue

        # Correlation in log-domain to reduce dynamic range bias.
        lr = np.log10(np.maximum(r, 1e-12))
        lc = np.log10(np.maximum(c, 1e-12))
        if np.std(lr) < 1e-8 or np.std(lc) < 1e-8:
            continue
        corr = float(np.corrcoef(lr, lc)[0, 1])
        if not np.isfinite(corr):
            corr = 0.0
        corr_arr[i] = corr

        # Gate subtraction: low-correlation traces should not be forced.
        if corr < 0.15:
            continue

        c_q = np.percentile(c, 70)
        use = c >= c_q
        if use.sum() < 5:
            use = c > 0
        if use.sum() == 0:
            continue

        ratios = r[use] / np.maximum(c[use], 1e-12)
        alpha = float(np.median(ratios))
        alpha = float(np.clip(alpha, 0.0, 3.0))
        scales[i] = alpha

        resid = r - alpha * c
        # Remove baseline and keep positive residual energy only.
        floor = float(np.percentile(resid, 20))
        resid = np.maximum(resid - floor, 0.0)
        out[i, lo:hi] = resid.astype(np.float32)

    return out, scales, corr_arr


@dataclass
class PickResultArrays:
    detected: np.ndarray
    delta_bins: np.ndarray
    snr: np.ndarray
    support: np.ndarray
    score: np.ndarray
    ringing_rejected: np.ndarray


def pick_discontinuous_reflectors(
    power: np.ndarray,
    surface_bins: np.ndarray,
    search_lo: int,
    search_hi: int,
    min_snr: float,
    clutter_aligned: Optional[np.ndarray] = None,
    ring_guard_bins: int = 3,
    neighbor_window: int = 3,
    neighbor_bin_tol: int = 7,
) -> PickResultArrays:
    """Pick discontinuous subsurface interfaces trace-by-trace.

    Strategy:
      - detect locally prominent peaks in high-passed band
      - apply clutter penalty at candidate bin
      - keep per-trace best candidate
      - enforce local neighborhood support (with high-SNR escape hatch)
    """
    p = np.asarray(power, dtype=np.float32)
    n_traces, n_bins = p.shape
    c_tm = _to_trace_major_clutter(clutter_aligned, n_traces, n_bins)

    detected = np.zeros(n_traces, dtype=bool)
    delta_bins = np.zeros(n_traces, dtype=np.int32)
    snr_arr = np.zeros(n_traces, dtype=np.float32)
    support = np.zeros(n_traces, dtype=np.int16)
    score = np.zeros(n_traces, dtype=np.float32)
    ringing_rejected = np.zeros(n_traces, dtype=bool)

    valid = np.where(surface_bins >= 0)[0]
    for i in valid:
        sb = int(surface_bins[i])
        lo = max(0, sb + search_lo)
        hi = min(n_bins, sb + search_hi)
        if hi <= lo + 8:
            continue

        band = p[i, lo:hi].astype(np.float64)
        if band.size < 8:
            continue

        # High-pass on range axis to favor discrete reflections over broad humps.
        trend = median_filter(band, size=min(11, max(3, (band.size // 8) * 2 + 1)))
        hp = band - trend
        sigma_hp = max(_mad(hp), 1e-12)
        sigma_band = max(_mad(band), 1e-12)
        base = float(np.median(band))

        prom_min = max(0.8 * sigma_hp, 1e-12)
        peaks, props = find_peaks(hp, prominence=prom_min, distance=2)
        if peaks.size == 0:
            pk = int(np.argmax(hp))
            peaks = np.array([pk], dtype=np.int32)
            props = {"prominences": np.array([max(float(hp[pk]), 0.0)], dtype=np.float64)}

        best_local = None
        best_score = -1e9

        for k, pk in enumerate(peaks):
            pk = int(pk)
            if pk < ring_guard_bins:
                ringing_rejected[i] = True
                continue

            amp = float(band[pk] - base)
            snr = amp / sigma_band
            prom = float(props["prominences"][k]) if "prominences" in props else float(max(hp[pk], 0.0))
            sharp = float(hp[pk] - np.median(hp[max(0, pk - 2): min(hp.size, pk + 3)]))

            clutter_pen = 0.0
            if c_tm is not None:
                c_band = c_tm[i, lo:hi].astype(np.float64)
                c_ref = float(np.percentile(c_band, 90)) + 1e-12
                clutter_pen = float(np.clip(c_band[pk] / c_ref, 0.0, 2.0))

            cand_score = (prom / sigma_hp) + 0.35 * snr + 0.15 * sharp - 0.75 * clutter_pen
            if snr >= min_snr and cand_score > best_score:
                best_score = float(cand_score)
                best_local = (pk, snr)

        if best_local is None:
            continue

        pk, snr = best_local
        detected[i] = True
        delta_bins[i] = int(search_lo + pk)
        snr_arr[i] = float(max(snr, 0.0))
        score[i] = float(best_score)

    # Local-support pruning: keep discontinuous segments without forcing long continuity.
    det_idx = np.where(detected)[0]
    if det_idx.size > 0:
        for i in det_idx:
            lo_i = max(0, i - neighbor_window)
            hi_i = min(n_traces, i + neighbor_window + 1)
            neigh = np.where(detected[lo_i:hi_i])[0] + lo_i
            if neigh.size == 0:
                continue
            dd = np.abs(delta_bins[neigh] - delta_bins[i])
            support[i] = int(np.sum(dd <= neighbor_bin_tol) - 1)  # exclude self

        # Keep if has local support OR very strong single-trace evidence.
        strong_snr = min_snr * 1.8
        strong_score = 2.5
        keep = (support >= 1) | (snr_arr >= strong_snr) | (score >= strong_score)
        drop = detected & (~keep)
        detected[drop] = False
        delta_bins[drop] = 0
        snr_arr[drop] = 0.0
        score[drop] = 0.0
        support[drop] = 0

    return PickResultArrays(
        detected=detected,
        delta_bins=delta_bins,
        snr=snr_arr,
        support=support,
        score=score,
        ringing_rejected=ringing_rejected,
    )

