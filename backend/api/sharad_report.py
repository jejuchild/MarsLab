"""
SHARAD Subsurface Interface Analysis Report Generator.

Produces a publication-quality scientific report with:
1. Annotated radargram with detected subsurface interface
2. Depth estimation table for multiple dielectric models
3. Cross-section comparison with HiRISE DTM surface elevation
4. Dielectric constant back-calculation
5. Comprehensive multi-page PDF report

Usage:
    results = await run_report_pipeline(product_id)
"""

import os
import csv
import json
import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Tuple, AsyncGenerator, Any
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO

import numpy as np

logger = logging.getLogger("sharad_report")

# ── Constants ─────────────────────────────────────────
C_VACUUM = 299_792_458.0            # m/s
SHARAD_DT_US = 3.0 / 80.0           # 0.0375 µs per range bin
SHARAD_DT_S = SHARAD_DT_US * 1e-6   # seconds per range bin
MARS_RADIUS_KM = 3389.5
N_RANGE_BINS = 667

DIELECTRIC_MODELS = [
    ("Vacuum",         1.0),
    ("Porous Ice",     2.8),
    ("Pure Water Ice", 3.1),
    ("Basalt (low)",   4.0),
    ("Basalt (mid)",   5.0),
    ("Basalt (high)",  6.0),
]

_BACKEND_DIR = Path(__file__).resolve().parent.parent
REPORT_DIR = _BACKEND_DIR / "sharad_reports"
MAX_DISPLAY_TRACES = 3000   # downsample radargram for figures


# ── Data classes ──────────────────────────────────────
@dataclass
class InterfaceDetection:
    bins: np.ndarray           # interface bin per trace (-1 = none)
    snr: np.ndarray            # SNR per trace
    confidence: str            # high / moderate / low
    detection_rate: float
    coherent_segments: List[Tuple[int, int]]
    mean_depth_bins: float
    surface_smooth: Optional[np.ndarray] = field(default=None)  # smoothed surface used by detection
    segment_statistics: Optional[Dict[str, Any]] = field(default=None)  # continuous vs patchy breakdown


@dataclass
class DepthEstimate:
    label: str
    epsilon_r: float
    depth_m: np.ndarray        # NaN where no detection
    mean_m: float
    min_m: float
    max_m: float


@dataclass
class ReportResult:
    product_id: str
    output_dir: str
    figures: Dict[str, str]    # name → file path
    csv_files: Dict[str, str]
    pdf_path: Optional[str]
    interface: InterfaceDetection
    depths: List[DepthEstimate]
    dtm_product_id: Optional[str]
    summary: str


# ── Helpers ───────────────────────────────────────────
def _haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance on Mars (km)."""
    rlat1, rlat2 = np.radians(lat1), np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(rlat1) * np.cos(rlat2) * np.sin(dlon / 2) ** 2
    return MARS_RADIUS_KM * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def _track_distance_km(lats, lons):
    """Cumulative along-track distance in km."""
    dist = np.zeros(len(lats))
    for i in range(1, len(lats)):
        dist[i] = dist[i - 1] + _haversine_km(lats[i - 1], lons[i - 1], lats[i], lons[i])
    return dist


def _lon180(lon):
    """Convert 0-360 → -180/180."""
    return np.where(lon > 180, lon - 360, lon)


# ── Step 1: Interface Detection (Viterbi DP Ridge Tracking) ──

MAX_DP_TRACES = 40000   # downsample for DP if more traces than this


def _remove_surface_echo_tail(
    power: np.ndarray,
    surface_smooth: np.ndarray,
    suppress_bins: int = 25,
) -> np.ndarray:
    """
    Suppress surface echo ringing in the zone immediately below the
    surface return, where multiples and side-lobe energy dominate.

    Strategy: for each trace, estimate the "deep background" noise level
    from bins well below the surface (depth 40-80), then in the ringing
    zone (surface+1 to surface+suppress_bins) replace any power above
    that background level with the background.  This flattens the
    surface multiple without touching deeper subsurface reflectors.

    Args:
        power: (n_traces, n_bins) raw power array
        surface_smooth: (n_traces,) smoothed surface bin indices
        suppress_bins: how many bins below surface to suppress

    Returns:
        cleaned: (n_traces, n_bins) power with surface ringing suppressed
    """
    n_traces, n_bins = power.shape
    cleaned = power.astype(np.float64).copy()

    for i in range(n_traces):
        s = int(round(surface_smooth[i]))
        if s < 0 or s >= n_bins:
            continue

        # Estimate deep background: median of bins s+40 to s+100
        bg_lo = min(s + 40, n_bins - 1)
        bg_hi = min(s + 100, n_bins)
        if bg_hi <= bg_lo:
            continue
        bg_level = float(np.median(power[i, bg_lo:bg_hi]))

        # Suppress the ringing zone: clamp to background level
        sup_lo = s + 1
        sup_hi = min(s + suppress_bins, n_bins)
        if sup_lo >= sup_hi:
            continue
        cleaned[i, sup_lo:sup_hi] = np.minimum(
            cleaned[i, sup_lo:sup_hi], bg_level
        )

    return cleaned.astype(np.float32)


def _along_track_stack(
    power: np.ndarray,
    half_width: int = 2,
) -> np.ndarray:
    """
    Apply along-track running mean to boost SNR of laterally coherent
    reflectors.  A half_width of 2 gives a 5-trace window (sqrt(5) SNR
    improvement for coherent signals).

    Uses uniform_filter1d for efficiency — O(n) regardless of window size.
    """
    from scipy.ndimage import uniform_filter1d
    width = 2 * half_width + 1
    return uniform_filter1d(
        power.astype(np.float32), size=width, axis=0, mode="nearest"
    )


def _smooth_surface_line(surface: np.ndarray, kernel: int = 21) -> np.ndarray:
    """Smooth and interpolate surface picks for stable search windows."""
    from scipy.ndimage import median_filter

    n = len(surface)
    surf = surface.astype(np.float64).copy()
    surf[surf < 0] = np.nan

    nans = np.isnan(surf)
    if nans.all():
        return np.zeros(n, dtype=np.float64)

    valid_idx = np.where(~nans)[0]
    surf_interp = np.interp(np.arange(n), valid_idx, surf[valid_idx])
    return median_filter(surf_interp, size=kernel).astype(np.float64)


def _viterbi_ridge_track(
    score: np.ndarray,
    valid: np.ndarray,
    smoothness: float,
) -> np.ndarray:
    """
    DP ridge tracker with linear smoothness penalty.
    Uses distance-transform trick for O(n_traces * n_depth) complexity.

    Args:
        score: (n_traces, n_depth) — higher = better candidate
        valid: (n_traces,) bool — traces with valid search windows
        smoothness: penalty per bin of vertical offset between adjacent traces

    Returns:
        path: (n_traces,) int32 — relative depth index (-1 if invalid)
    """
    n_traces, n_depth = score.shape

    # Rolling cost — only keep previous valid trace's cost row
    cost_prev = np.zeros(n_depth, dtype=np.float64)
    parent = np.full((n_traces, n_depth), -1, dtype=np.int32)

    first = -1
    for i in range(n_traces):
        if valid[i]:
            cost_prev = (-score[i]).copy()
            first = i
            break

    if first < 0:
        return np.full(n_traces, -1, dtype=np.int32)

    prev_valid = first

    for i in range(first + 1, n_traces):
        if not valid[i]:
            continue

        gap = i - prev_valid
        lam = smoothness / max(gap, 1)

        # Distance transform: min_{d'} cost_prev[d'] + lam * |d - d'|
        dt = cost_prev.copy()
        dt_parent = np.arange(n_depth, dtype=np.int32)

        # Forward sweep (d' <= d)
        for d in range(1, n_depth):
            c = dt[d - 1] + lam
            if c < dt[d]:
                dt[d] = c
                dt_parent[d] = dt_parent[d - 1]

        # Backward sweep (d' >= d)
        for d in range(n_depth - 2, -1, -1):
            c = dt[d + 1] + lam
            if c < dt[d]:
                dt[d] = c
                dt_parent[d] = dt_parent[d + 1]

        cost_prev = dt + (-score[i])
        parent[i] = dt_parent
        prev_valid = i

    # Backtrack from last valid trace
    last = prev_valid
    path = np.full(n_traces, -1, dtype=np.int32)

    if last < 0 or last == first:
        if first >= 0:
            path[first] = int(np.argmin(-score[first]))
        return path

    path[last] = int(np.argmin(cost_prev))

    for i in range(last - 1, -1, -1):
        if not valid[i]:
            continue
        nxt = i + 1
        while nxt <= last and not valid[nxt]:
            nxt += 1
        if nxt <= last and path[nxt] >= 0:
            path[i] = parent[nxt, path[nxt]]

    return path


def _smooth_picks(iface_bins: np.ndarray, medfilt_k: int = 7, savgol_w: int = 21) -> np.ndarray:
    """Post-process picks with median filter + Savitzky-Golay smoothing."""
    from scipy.signal import medfilt, savgol_filter

    result = iface_bins.copy()
    valid = result >= 0
    if valid.sum() < savgol_w:
        return result

    valid_idx = np.where(valid)[0]
    vals = result[valid_idx].astype(np.float64)

    if len(vals) >= medfilt_k:
        vals = medfilt(vals, kernel_size=medfilt_k)
    if len(vals) >= savgol_w:
        vals = savgol_filter(vals, window_length=savgol_w, polyorder=3)

    result[valid_idx] = np.round(vals).astype(np.int32)
    return result


def _fill_gaps(
    iface_bins: np.ndarray,
    power: np.ndarray,
    surface_smooth: np.ndarray,
    min_depth_bins: int,
    max_depth_bins: int,
    max_gap: int = 100,
    search_radius: int = 8,
    snr_threshold: float = 1.5,
) -> np.ndarray:
    """
    Fill gaps between detected segments by searching for peaks near the
    interpolated expected depth.

    For each gap of ≤ max_gap traces between two detected runs, linearly
    interpolate the expected interface bin from the flanking picks, then
    search within ±search_radius bins for a local peak that exceeds
    snr_threshold.
    """
    n_traces, n_bins = power.shape
    result = iface_bins.copy()
    detected = result >= 0

    # Find runs of detected and undetected traces
    i = 0
    while i < n_traces:
        if detected[i]:
            i += 1
            continue
        # Found start of gap
        gap_start = i
        while i < n_traces and not detected[i]:
            i += 1
        gap_end = i  # first detected trace after gap (or n_traces)

        gap_len = gap_end - gap_start
        if gap_len > max_gap or gap_len < 1:
            continue

        # Need flanking picks
        left_pick = result[gap_start - 1] if gap_start > 0 and result[gap_start - 1] >= 0 else -1
        right_pick = result[gap_end] if gap_end < n_traces and result[gap_end] >= 0 else -1

        if left_pick < 0 and right_pick < 0:
            continue
        if left_pick < 0:
            left_pick = right_pick
        if right_pick < 0:
            right_pick = left_pick

        for j in range(gap_start, gap_end):
            # Interpolate expected bin
            frac = (j - gap_start + 1) / (gap_len + 1)
            expected_bin = int(round(left_pick * (1 - frac) + right_pick * frac))

            surf = int(round(surface_smooth[j]))
            lo = max(expected_bin - search_radius, surf + min_depth_bins)
            hi = min(expected_bin + search_radius + 1, min(surf + max_depth_bins, n_bins))
            if lo >= hi or lo < 0:
                continue

            window = power[j, lo:hi].astype(np.float64)
            # Noise from wider search range
            noise_lo = surf + min_depth_bins
            noise_hi = min(surf + max_depth_bins, n_bins)
            if noise_hi <= noise_lo:
                continue
            noise = float(np.median(power[j, noise_lo:noise_hi]))
            if noise <= 0:
                continue

            peak_rel = int(np.argmax(window))
            peak_bin = lo + peak_rel
            peak_snr = window[peak_rel] / noise

            if peak_snr >= snr_threshold:
                result[j] = peak_bin

    return result


def _debug_detection_plot(
    power: np.ndarray,
    surface_smooth: np.ndarray,
    iface_bins: np.ndarray,
    score_2d: np.ndarray,
    rel_path: np.ndarray,
    min_depth_bins: int,
    debug_dir: str,
):
    """Generate debug plots: radargram overview + score matrix with DP path."""
    plt = _get_plt()
    os.makedirs(debug_dir, exist_ok=True)

    n_traces, n_bins = power.shape
    ds = max(1, n_traces // 3000)

    if ds > 1:
        n = n_traces // ds * ds
        p = power[:n].reshape(-1, ds, n_bins).mean(axis=1)
        surf = surface_smooth[:n:ds]
        iface = iface_bins[:n:ds].copy()
        sc = score_2d[:n].reshape(-1, ds, score_2d.shape[1]).mean(axis=1)
        rp = rel_path[:n:ds].copy()
        x = np.arange(p.shape[0])
    else:
        p, surf = power, surface_smooth
        iface, sc, rp = iface_bins.copy(), score_2d, rel_path.copy()
        x = np.arange(n_traces)

    p_log = np.log10(np.maximum(p, 1e-12))
    vmin, vmax = np.percentile(p_log, [2, 98])

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), dpi=150)

    # Panel 1: Radargram with surface + interface
    ax = axes[0]
    ax.imshow(p_log.T, aspect="auto", cmap="gray", vmin=vmin, vmax=vmax)
    ax.plot(x, surf, color="lime", linewidth=0.8, label="Surface (smoothed)")
    iface_f = iface.astype(np.float64)
    iface_f[iface < 0] = np.nan
    ax.plot(x, iface_f, color="red", linewidth=1.0, label="DP interface")
    ax.plot(x, surf + min_depth_bins, color="cyan", linewidth=0.4,
            linestyle=":", alpha=0.5, label="Search window top")
    ax.set_title("Debug: Interface Detection Overview")
    ax.legend(loc="upper right", fontsize=7)
    ax.set_ylabel("Range bin")

    # Panel 2: Score matrix with DP path
    ax2 = axes[1]
    ax2.imshow(sc.T, aspect="auto", cmap="hot", interpolation="nearest")
    rp_f = rp.astype(np.float64)
    rp_f[rp < 0] = np.nan
    ax2.plot(x, rp_f, color="cyan", linewidth=1.0, label="DP optimal path")
    ax2.set_title("Debug: Score Matrix (relative depth) + DP Path")
    ax2.set_xlabel("Trace index")
    ax2.set_ylabel("Relative depth bin")
    ax2.legend(loc="upper right", fontsize=7)

    fig.tight_layout()
    fig.savefig(os.path.join(debug_dir, "debug_detection.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def detect_interface(
    power: np.ndarray,
    surface: np.ndarray,
    min_depth_bins: int = 25,
    max_depth_bins: int = 300,
    smoothness_weight: float = 0.15,
    snr_threshold: float = 1.5,
    min_segment_len: int = 30,
    stack_half_width: int = 5,
    debug: bool = False,
    debug_dir: str = None,
) -> InterfaceDetection:
    """
    Detect laterally coherent subsurface reflector using Viterbi-style
    dynamic programming ridge tracking with surface echo suppression.

    Pipeline:
      A. Smooth surface line
      B. Remove surface echo exponential tail (suppresses multiples)
      C. Along-track stacking (5-trace running mean, sqrt(5) SNR boost)
      D. Build relative-depth power matrix below surface + min_depth_bins
      E. Per-trace column-normalized scoring (local SNR, not global percentile)
      F. Viterbi DP ridge tracking
      G. Convert to absolute bins, smooth, compute SNR, filter
    """
    n_traces, n_bins = power.shape

    # ── A: Smooth surface line for stable search windows ──
    surface_smooth = _smooth_surface_line(surface)

    # ── B: Remove surface echo tail ──
    power_clean = _remove_surface_echo_tail(power, surface_smooth)

    # ── C: Along-track stacking ──
    power_stacked = _along_track_stack(power_clean, half_width=stack_half_width)

    # ── D: Build relative-depth power matrix ──
    n_depth = max_depth_bins - min_depth_bins
    rel_power = np.zeros((n_traces, n_depth), dtype=np.float32)
    valid_traces = np.ones(n_traces, dtype=bool)

    for i in range(n_traces):
        surf = int(round(surface_smooth[i]))
        lo = surf + min_depth_bins
        hi = min(surf + max_depth_bins, n_bins)
        if lo < 0 or lo >= n_bins or hi <= lo:
            valid_traces[i] = False
            continue
        actual = hi - lo
        rel_power[i, :actual] = power_stacked[i, lo:hi]

    if not valid_traces.any():
        return InterfaceDetection(
            bins=np.full(n_traces, -1, dtype=np.int32),
            snr=np.zeros(n_traces, dtype=np.float32),
            confidence="low", detection_rate=0.0,
            coherent_segments=[], mean_depth_bins=0.0,
            surface_smooth=surface_smooth,
        )

    # ── E: Per-trace column-normalized scoring ──
    # For each trace, score = power / local_median.  This makes the score
    # equal to local SNR rather than absolute power, which equalizes
    # detection sensitivity across traces with different background levels.
    score = np.zeros((n_traces, n_depth), dtype=np.float64)
    for i in range(n_traces):
        if not valid_traces[i]:
            continue
        col = rel_power[i].astype(np.float64)
        col_pos = col[col > 0]
        if len(col_pos) < 5:
            continue
        noise = float(np.median(col_pos))
        if noise > 0:
            score[i] = col / noise
    # Clip extreme outliers and normalize to [0, 1]
    active = score > 0
    if active.any():
        s_max = np.percentile(score[active], 98)
        if s_max > 1.0:
            score = np.clip(score / s_max, 0.0, 1.0)

    # ── F: Viterbi DP ridge tracking ──
    if n_traces > MAX_DP_TRACES:
        # Downsample for DP, then interpolate path back
        ds_dp = (n_traces + MAX_DP_TRACES - 1) // MAX_DP_TRACES
        n_sub = n_traces // ds_dp
        trim = n_sub * ds_dp

        score_sub = score[:trim].reshape(n_sub, ds_dp, n_depth).mean(axis=1)
        valid_sub = valid_traces[:trim].reshape(n_sub, ds_dp).all(axis=1)

        path_sub = _viterbi_ridge_track(score_sub, valid_sub, smoothness_weight)

        sub_centers = np.arange(n_sub) * ds_dp + ds_dp // 2
        valid_path = path_sub >= 0
        if valid_path.any():
            rel_path = np.round(np.interp(
                np.arange(n_traces),
                sub_centers[valid_path],
                path_sub[valid_path].astype(np.float64),
            )).astype(np.int32)
            first_v = int(sub_centers[valid_path][0])
            last_v = int(sub_centers[valid_path][-1])
            rel_path[:max(0, first_v - ds_dp)] = -1
            rel_path[min(n_traces, last_v + ds_dp):] = -1
        else:
            rel_path = np.full(n_traces, -1, dtype=np.int32)
    else:
        rel_path = _viterbi_ridge_track(score, valid_traces, smoothness_weight)

    # ── G: Convert relative depth to absolute bins ──
    iface_bins = np.full(n_traces, -1, dtype=np.int32)
    for i in range(n_traces):
        if valid_traces[i] and 0 <= rel_path[i] < n_depth:
            surf = int(round(surface_smooth[i]))
            iface_bins[i] = surf + min_depth_bins + rel_path[i]

    # ── H: Post-processing — smooth picks ──
    iface_bins = _smooth_picks(iface_bins)

    # ── H2: Fill gaps between segments ──
    iface_bins = _fill_gaps(
        iface_bins, power, surface_smooth,
        min_depth_bins, max_depth_bins,
    )
    iface_bins = _smooth_picks(iface_bins)

    # ── I: Compute SNR along the path (against original power, not cleaned) ──
    iface_snr = np.zeros(n_traces, dtype=np.float32)
    for i in range(n_traces):
        b = iface_bins[i]
        if b < 0 or not valid_traces[i]:
            continue
        surf = int(round(surface_smooth[i]))
        lo = surf + min_depth_bins
        hi = min(surf + max_depth_bins, n_bins)
        if lo >= hi:
            continue
        col = power[i, lo:hi].astype(np.float64)
        noise = np.median(col)
        if noise > 0:
            iface_snr[i] = float(power[i, b]) / noise

    # ── J: Filter weak detections ──
    weak = (iface_snr < snr_threshold) & (iface_bins >= 0)
    iface_bins[weak] = -1
    iface_snr[weak] = 0

    # ── K: Segment detection with continuous + patchy classification ──
    detected = iface_bins >= 0
    all_raw_segments: List[Tuple[int, int]] = []
    i = 0
    while i < n_traces:
        if not detected[i]:
            i += 1
            continue
        start = i
        while i < n_traces and detected[i]:
            i += 1
        all_raw_segments.append((start, i))

    # Classify segments: continuous (≥ min_segment_len) vs patchy (≥ 10)
    PATCHY_MIN_LEN = 10
    PATCHY_SNR_THRESHOLD = 2.0  # stricter SNR for short segments
    continuous_segments: List[Tuple[int, int]] = []
    patchy_segments: List[Tuple[int, int]] = []

    for s, e in all_raw_segments:
        seg_len = e - s
        if seg_len >= min_segment_len:
            continuous_segments.append((s, e))
        elif seg_len >= PATCHY_MIN_LEN:
            # Patchy segments require higher mean SNR to reduce false positives
            seg_snr = iface_snr[s:e]
            mean_seg_snr = float(np.mean(seg_snr[seg_snr > 0])) if np.any(seg_snr > 0) else 0
            if mean_seg_snr >= PATCHY_SNR_THRESHOLD:
                patchy_segments.append((s, e))

    # Combine all accepted segments (continuous + patchy)
    segments = continuous_segments + patchy_segments

    mask = np.zeros(n_traces, dtype=bool)
    for s, e in segments:
        mask[s:e] = True
    iface_bins[~mask] = -1
    iface_snr[~mask] = 0

    # Build segment statistics
    seg_stats: Dict[str, Any] = {
        "continuous_count": len(continuous_segments),
        "patchy_count": len(patchy_segments),
        "continuous_total_traces": sum(e - s for s, e in continuous_segments),
        "patchy_total_traces": sum(e - s for s, e in patchy_segments),
        "total_lateral_extent_traces": sum(e - s for s, e in segments),
        "rejected_short_count": sum(1 for s, e in all_raw_segments if (e - s) < PATCHY_MIN_LEN),
    }

    # ── Metrics ──
    valid = iface_bins >= 0
    rate = valid.sum() / n_traces if n_traces > 0 else 0
    mean_depth = float(np.mean(iface_bins[valid] - surface_smooth[valid])) if valid.any() else 0.0

    if rate > 0.3 and continuous_segments:
        conf = "high"
    elif rate > 0.1:
        conf = "moderate"
    elif patchy_segments and not continuous_segments:
        conf = "low_patchy"
    else:
        conf = "low"

    # ── Debug output ──
    if debug and debug_dir:
        _debug_detection_plot(power, surface_smooth, iface_bins, score,
                              rel_path, min_depth_bins, debug_dir)

    return InterfaceDetection(
        bins=iface_bins, snr=iface_snr,
        confidence=conf, detection_rate=rate,
        coherent_segments=segments, mean_depth_bins=mean_depth,
        surface_smooth=surface_smooth,
        segment_statistics=seg_stats,
    )


# ── Step 2: Depth Estimation ─────────────────────────
def estimate_depths(
    interface: InterfaceDetection,
    surface: np.ndarray,
) -> List[DepthEstimate]:
    """Compute depth for each dielectric model."""
    # Use smoothed surface if available (consistent with DP search window)
    surf = interface.surface_smooth if interface.surface_smooth is not None else surface
    valid = interface.bins >= 0
    dt_bins = np.where(valid, interface.bins - surf, 0).astype(np.float64)
    twtt_s = dt_bins * SHARAD_DT_S

    results = []
    for label, eps in DIELECTRIC_MODELS:
        depth = np.where(valid, (C_VACUUM * twtt_s) / (2.0 * np.sqrt(eps)), np.nan)
        vd = depth[~np.isnan(depth)]
        results.append(DepthEstimate(
            label=label, epsilon_r=eps, depth_m=depth,
            mean_m=float(np.nanmean(vd)) if len(vd) else 0,
            min_m=float(np.nanmin(vd)) if len(vd) else 0,
            max_m=float(np.nanmax(vd)) if len(vd) else 0,
        ))
    return results


# ── Step 3: Find Overlapping DTM ─────────────────────
def find_overlapping_dtm(lats, lons) -> Optional[str]:
    """Find a HiRISE DTM product that overlaps the SHARAD track."""
    index_path = _BACKEND_DIR / "hirise_dtm_data" / "index.geojson"
    if not index_path.exists():
        return None

    try:
        with open(index_path) as f:
            idx = json.load(f)
    except Exception:
        return None

    track_lat_min, track_lat_max = float(np.min(lats)), float(np.max(lats))
    track_lon_min, track_lon_max = float(np.min(lons)), float(np.max(lons))

    for feat in idx.get("features", []):
        geom = feat.get("geometry", {})
        coords = geom.get("coordinates", [[]])
        if geom.get("type") != "Polygon" or not coords[0]:
            continue
        ring = coords[0]
        flons = [c[0] for c in ring]
        flats = [c[1] for c in ring]
        flon_min, flon_max = min(flons), max(flons)
        flat_min, flat_max = min(flats), max(flats)

        # Bbox overlap check
        if (flon_max < track_lon_min or flon_min > track_lon_max or
                flat_max < track_lat_min or flat_min > track_lat_max):
            continue

        pid = feat.get("properties", {}).get("product_id")
        if pid:
            return pid
    return None


def load_dtm_along_track(dtm_product_id: str, lats, lons) -> Optional[np.ndarray]:
    """Extract DTM elevation along SHARAD track coordinates using rasterio."""
    try:
        import rasterio
        from pyproj import Transformer
    except ImportError:
        logger.warning("rasterio/pyproj not available for DTM loading")
        return None

    dtm_dir = _BACKEND_DIR / "hirise_dtm_data"
    # Find the .IMG file
    import glob
    pattern = str(dtm_dir / f"*{dtm_product_id}*")
    matches = [f for f in glob.glob(pattern) if f.upper().endswith(".IMG")]
    if not matches:
        return None

    try:
        with rasterio.open(matches[0]) as ds:
            crs = ds.crs
            # Transform Mars lat/lon to DTM CRS
            transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
            xs, ys = transformer.transform(lons, lats)

            elevations = np.full(len(lats), np.nan)
            for i in range(len(lats)):
                try:
                    row, col = ds.index(xs[i], ys[i])
                    if 0 <= row < ds.height and 0 <= col < ds.width:
                        val = ds.read(1, window=rasterio.windows.Window(col, row, 1, 1))
                        elevations[i] = float(val[0, 0])
                except Exception:
                    continue
            # Filter nodata
            nodata = ds.nodata
            if nodata is not None:
                elevations[elevations == nodata] = np.nan
            return elevations
    except Exception as e:
        logger.warning(f"DTM loading failed: {e}")
        return None


# ── Step 4: Dielectric Back-Calculation ──────────────
def compute_dielectric_profile(
    interface: InterfaceDetection,
    surface: np.ndarray,
    dtm_elevation: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Back-calculate εr from DTM-derived true depth.
    εr = (c * t / (2 * true_depth))²
    """
    valid = (interface.bins >= 0) & ~np.isnan(dtm_elevation)
    if valid.sum() < 10:
        return None

    dt_bins = (interface.bins - surface).astype(np.float64)
    twtt_s = dt_bins * SHARAD_DT_S

    # Use DTM elevation difference as proxy for true depth
    # Assume surface elevation = DTM value, interface at some depth below
    # We need a reference: use the vacuum-depth estimate as starting point
    # and see if DTM relief provides depth constraints
    #
    # For terrace craters / exposed layers: true_depth ≈ local relief
    # For flat terrain: we can't back-calculate, skip
    elev = dtm_elevation[valid]
    elev_smooth = np.convolve(elev, np.ones(50) / 50, mode="same")
    local_relief = np.abs(elev - elev_smooth)

    # Only where relief > 10m (potential exposed subsurface)
    relief_mask = local_relief > 10
    if relief_mask.sum() < 5:
        return None

    eps_profile = np.full(len(interface.bins), np.nan)
    valid_idx = np.where(valid)[0]
    for j, idx in enumerate(valid_idx):
        if not relief_mask[j]:
            continue
        t = twtt_s[idx]
        depth = local_relief[j]
        if depth > 0 and t > 0:
            eps = (C_VACUUM * t / (2.0 * depth)) ** 2
            if 1.0 <= eps <= 20.0:  # physical range
                eps_profile[idx] = eps

    return eps_profile


# ── Figure Generation ─────────────────────────────────
def _get_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def generate_radargram_figure(
    power: np.ndarray,
    surface: np.ndarray,
    interface: InterfaceDetection,
    track_km: np.ndarray,
    output_path: str,
    ds: int = 1,
):
    """Generate annotated radargram (300 DPI)."""
    plt = _get_plt()

    # Downsample
    if ds > 1:
        n_trim = power.shape[0] // ds * ds
        p = power[:n_trim].reshape(-1, ds, power.shape[1]).mean(axis=1)
        s = surface[:n_trim:ds].copy()
        ib = interface.bins[:n_trim:ds].copy()
        isnr = interface.snr[:n_trim:ds].copy()
        tk = track_km[:n_trim:ds].copy()
    else:
        p = power.copy()
        s = surface.copy()
        ib = interface.bins.copy()
        isnr = interface.snr.copy()
        tk = track_km.copy()

    n_traces = p.shape[0]
    # Log scale
    p_log = np.log10(np.maximum(p, 1e-12))
    vmin, vmax = np.percentile(p_log, [2, 98])

    fig, ax = plt.subplots(figsize=(14, 5), dpi=300)
    ax.imshow(
        p_log.T, aspect="auto", cmap="gray",
        vmin=vmin, vmax=vmax,
        extent=[tk[0], tk[-1], N_RANGE_BINS * SHARAD_DT_US, 0],
    )

    # Surface return (green)
    valid_s = s >= 0
    ax.plot(
        tk[valid_s], s[valid_s] * SHARAD_DT_US,
        color="lime", linewidth=0.8, alpha=0.9, label="Surface return",
    )

    # Subsurface interface (red) — NaN gaps for clean line breaks
    valid_i = ib >= 0
    if valid_i.any():
        ib_float = ib.astype(np.float64)
        ib_float[~valid_i] = np.nan
        ax.plot(
            tk, ib_float * SHARAD_DT_US,
            color="red", linewidth=1.2, alpha=0.9, label="Subsurface interface",
        )

        # Peak SNR markers (yellow)
        top_snr = np.argsort(isnr)[::-1][:20]
        top_snr = top_snr[isnr[top_snr] > 0]
        if len(top_snr):
            ax.scatter(
                tk[top_snr], ib[top_snr] * SHARAD_DT_US,
                c="yellow", s=15, zorder=5, edgecolors="k", linewidths=0.3,
                label="Peak reflectors",
            )

    ax.set_xlabel("Along-track distance (km)", fontsize=10)
    ax.set_ylabel("Two-way travel time (µs)", fontsize=10)
    ax.set_title(f"SHARAD Radargram — Subsurface Interface Detection ({interface.confidence} confidence)", fontsize=11)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.8)

    # Scale bar
    bar_len_km = round((tk[-1] - tk[0]) * 0.1, 0)
    if bar_len_km < 1:
        bar_len_km = 1
    bx = tk[0] + (tk[-1] - tk[0]) * 0.02
    by = N_RANGE_BINS * SHARAD_DT_US * 0.92
    ax.plot([bx, bx + bar_len_km], [by, by], color="white", linewidth=2)
    ax.text(bx + bar_len_km / 2, by - 0.3, f"{bar_len_km:.0f} km",
            color="white", ha="center", fontsize=7, fontweight="bold")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def generate_interface_mask(
    interface: InterfaceDetection,
    n_traces: int,
    output_path: str,
):
    """Generate binary interface mask image."""
    from PIL import Image
    mask = np.zeros((N_RANGE_BINS, n_traces), dtype=np.uint8)
    for i in range(n_traces):
        b = interface.bins[i]
        if b >= 0 and b < N_RANGE_BINS:
            mask[b, i] = 255
            # Thicken to ±1 bin
            if b > 0:
                mask[b - 1, i] = 180
            if b < N_RANGE_BINS - 1:
                mask[b + 1, i] = 180
    img = Image.fromarray(mask, mode="L")
    img.save(output_path)


def generate_depth_csv(
    depths: List[DepthEstimate],
    output_path: str,
):
    """Write depth estimation table."""
    with open(output_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Material", "Dielectric_Constant", "Mean_Depth_m",
                     "Min_Depth_m", "Max_Depth_m"])
        for d in depths:
            w.writerow([d.label, f"{d.epsilon_r:.1f}",
                        f"{d.mean_m:.1f}", f"{d.min_m:.1f}", f"{d.max_m:.1f}"])


def generate_cross_section(
    track_km: np.ndarray,
    surface: np.ndarray,
    interface: InterfaceDetection,
    depths: List[DepthEstimate],
    dtm_elevation: np.ndarray,
    output_path: str,
    ds: int = 1,
):
    """Generate surface vs subsurface cross-section plot."""
    plt = _get_plt()

    if ds > 1:
        n = len(track_km) // ds * ds
        tk = track_km[:n:ds]
        dtm = dtm_elevation[:n:ds]
        depth_data = {d.label: d.depth_m[:n:ds] for d in depths}
    else:
        tk = track_km
        dtm = dtm_elevation
        depth_data = {d.label: d.depth_m for d in depths}

    fig, ax = plt.subplots(figsize=(14, 6), dpi=300)

    # Surface elevation (black)
    valid_dtm = ~np.isnan(dtm)
    if valid_dtm.any():
        ax.plot(tk[valid_dtm], dtm[valid_dtm], color="black", linewidth=1.2,
                label="Surface elevation (HiRISE DTM)")

    # Subsurface depths (below surface)
    colors = {"Vacuum": "#AAAAAA", "Porous Ice": "cyan", "Pure Water Ice": "dodgerblue",
              "Basalt (low)": "orange", "Basalt (mid)": "darkorange", "Basalt (high)": "red"}
    styles = {"Vacuum": ":", "Porous Ice": "--", "Pure Water Ice": "-",
              "Basalt (low)": "--", "Basalt (mid)": "-.", "Basalt (high)": ":"}

    # Plot key models only (not all 6 — too cluttered)
    key_models = ["Pure Water Ice", "Porous Ice", "Basalt (low)"]
    for label in key_models:
        if label not in depth_data:
            continue
        d = depth_data[label]
        valid = ~np.isnan(d) & valid_dtm
        if not valid.any():
            continue
        subsurface_elev = dtm[valid] - d[valid]
        ax.plot(tk[valid], subsurface_elev,
                color=colors.get(label, "gray"),
                linestyle=styles.get(label, "--"),
                linewidth=1.0, alpha=0.9, label=f"Interface (εr={dict(DIELECTRIC_MODELS)[label]:.1f}, {label})")

    ax.set_xlabel("Along-track distance (km)", fontsize=10)
    ax.set_ylabel("Elevation (m)", fontsize=10)
    ax.set_title("Surface vs Subsurface Interface Cross-Section", fontsize=11)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def generate_dielectric_figure(
    track_km: np.ndarray,
    eps_profile: np.ndarray,
    output_path: str,
):
    """Plot back-calculated dielectric profile."""
    plt = _get_plt()
    valid = ~np.isnan(eps_profile)
    if valid.sum() < 5:
        return False

    fig, ax = plt.subplots(figsize=(14, 4), dpi=300)
    ax.scatter(track_km[valid], eps_profile[valid], s=3, c="steelblue", alpha=0.6)

    # Reference lines
    ax.axhline(3.1, color="cyan", linestyle="--", linewidth=0.8, label="Pure water ice (3.1)")
    ax.axhline(2.8, color="lightblue", linestyle=":", linewidth=0.8, label="Porous ice (2.8)")
    ax.axhline(4.0, color="orange", linestyle="--", linewidth=0.8, label="Basalt low (4.0)")

    ax.set_xlabel("Along-track distance (km)", fontsize=10)
    ax.set_ylabel("Dielectric constant (εr)", fontsize=10)
    ax.set_title("Back-Calculated Dielectric Constant Profile", fontsize=11)
    ax.set_ylim(0.5, 12)
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return True


def generate_dielectric_csv(
    track_km: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    eps_profile: np.ndarray,
    output_path: str,
):
    """Write dielectric values along track."""
    valid = ~np.isnan(eps_profile)
    with open(output_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Distance_km", "Latitude", "Longitude", "Dielectric_Constant"])
        for i in np.where(valid)[0]:
            w.writerow([f"{track_km[i]:.2f}", f"{lats[i]:.5f}",
                        f"{lons[i]:.5f}", f"{eps_profile[i]:.3f}"])


# ── Step 5: PDF Report ───────────────────────────────
def generate_pdf_report(
    product_id: str,
    interface: InterfaceDetection,
    depths: List[DepthEstimate],
    dtm_product_id: Optional[str],
    eps_profile: Optional[np.ndarray],
    figures: Dict[str, str],
    output_path: str,
):
    """Generate multi-page PDF report using fpdf2."""
    try:
        from fpdf import FPDF
    except ImportError:
        logger.warning("fpdf2 not installed — skipping PDF generation")
        return False

    pdf = FPDF(orientation="L", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)

    # ── Title page ──
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 22)
    pdf.cell(0, 20, "SHARAD Subsurface Interface Report", ln=True, align="C")
    pdf.ln(5)
    pdf.set_font("Helvetica", "", 14)
    pdf.cell(0, 10, f"Product: {product_id}", ln=True, align="C")
    pdf.cell(0, 10, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align="C")
    pdf.ln(10)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Detection Summary", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Confidence: {interface.confidence.upper()}", ln=True)
    pdf.cell(0, 6, f"Detection rate: {interface.detection_rate * 100:.1f}% of traces", ln=True)
    pdf.cell(0, 6, f"Coherent segments: {len(interface.coherent_segments)}", ln=True)
    pdf.cell(0, 6, f"Mean interface depth: {interface.mean_depth_bins:.0f} range bins "
                    f"({interface.mean_depth_bins * SHARAD_DT_US:.2f} us two-way)", ln=True)
    pdf.ln(5)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Depth Estimates", ln=True)
    pdf.set_font("Helvetica", "", 9)
    # Table header
    col_w = [55, 20, 30, 30, 30]
    pdf.cell(col_w[0], 6, "Material", border="B")
    pdf.cell(col_w[1], 6, "er", border="B", align="R")
    pdf.cell(col_w[2], 6, "Mean (m)", border="B", align="R")
    pdf.cell(col_w[3], 6, "Min (m)", border="B", align="R")
    pdf.cell(col_w[4], 6, "Max (m)", border="B", align="R")
    pdf.ln()
    for d in depths:
        pdf.cell(col_w[0], 5, d.label)
        pdf.cell(col_w[1], 5, f"{d.epsilon_r:.1f}", align="R")
        pdf.cell(col_w[2], 5, f"{d.mean_m:.1f}", align="R")
        pdf.cell(col_w[3], 5, f"{d.min_m:.1f}", align="R")
        pdf.cell(col_w[4], 5, f"{d.max_m:.1f}", align="R")
        pdf.ln()

    if dtm_product_id:
        pdf.ln(5)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 6, f"HiRISE DTM used: {dtm_product_id}", ln=True)
    else:
        pdf.ln(5)
        pdf.set_font("Helvetica", "I", 10)
        pdf.cell(0, 6, "Note: No co-located HiRISE DTM available. Cross-section comparison skipped.", ln=True)

    # ── Figure pages ──
    page_w = 277  # A4 landscape usable width
    page_h = 170  # usable height

    for fig_name, fig_path in figures.items():
        if not os.path.exists(fig_path) or not fig_path.endswith(".png"):
            continue
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 12)
        title_map = {
            "radargram_annotated": "Annotated Radargram with Subsurface Interface",
            "interface_mask": "Interface Detection Mask",
            "cross_section": "Surface vs Subsurface Cross-Section Comparison",
            "dielectric_profile": "Back-Calculated Dielectric Constant Profile",
        }
        pdf.cell(0, 10, title_map.get(fig_name, fig_name.replace("_", " ").title()), ln=True, align="C")
        pdf.ln(2)

        # Fit image to page
        try:
            from PIL import Image as PILImage
            with PILImage.open(fig_path) as img:
                iw, ih = img.size
            aspect = iw / ih
            if aspect > page_w / page_h:
                w = page_w
                h = w / aspect
            else:
                h = page_h
                w = h * aspect
            x = (297 - w) / 2  # center on A4 landscape
            pdf.image(fig_path, x=x, y=pdf.get_y(), w=w, h=h)
        except Exception:
            pdf.image(fig_path, x=10, y=pdf.get_y(), w=page_w)

    # ── Interpretation page ──
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Scientific Interpretation", ln=True)
    pdf.ln(3)
    pdf.set_font("Helvetica", "", 10)

    ice_depth = next((d for d in depths if d.label == "Pure Water Ice"), None)
    basalt_depth = next((d for d in depths if d.label == "Basalt (low)"), None)

    lines = []
    if interface.confidence == "high":
        lines.append("A laterally coherent subsurface reflector was detected with HIGH confidence.")
    elif interface.confidence == "moderate":
        lines.append("A subsurface reflector was detected with MODERATE confidence. "
                      "Additional data or alternative viewing geometries may improve characterization.")
    else:
        lines.append("WARNING: Low-confidence reflector detection. Results should be interpreted with caution.")

    lines.append("")
    if ice_depth and ice_depth.mean_m > 0:
        lines.append(f"Assuming pure water ice (er = 3.1), the mean interface depth is {ice_depth.mean_m:.1f} m "
                      f"(range: {ice_depth.min_m:.1f}-{ice_depth.max_m:.1f} m).")

    if basalt_depth and basalt_depth.mean_m > 0:
        lines.append(f"Assuming dry basaltic regolith (er = 4.0), the mean depth is {basalt_depth.mean_m:.1f} m.")

    lines.append("")
    if eps_profile is not None:
        valid_eps = eps_profile[~np.isnan(eps_profile)]
        if len(valid_eps) > 0:
            med_eps = float(np.median(valid_eps))
            lines.append(f"Back-calculated median dielectric constant: er = {med_eps:.2f}")
            if 2.5 <= med_eps <= 3.5:
                lines.append("=> Consistent with water ice composition (er ~ 2.8-3.1).")
                lines.append("  This may indicate a cryosphere boundary or buried ice deposit.")
            elif 3.5 < med_eps <= 6.0:
                lines.append("=> Consistent with dry to moderately porous regolith (er ~ 4-6).")
                lines.append("  Subsurface reflector likely represents a lithological boundary.")
            elif med_eps < 2.5:
                lines.append("=> Anomalously low er suggests highly porous material or measurement artifacts.")
            else:
                lines.append(f"=> er = {med_eps:.2f} is outside typical ranges. Consider measurement uncertainties.")
    else:
        lines.append("Dielectric back-calculation was not possible (insufficient DTM coverage or relief).")

    for line in lines:
        pdf.set_x(10)  # reset to left margin
        pdf.multi_cell(0, 6, line)

    pdf.output(output_path)
    return True


# ── Concurrency control ──────────────────────────────
import re as _re
_PRODUCT_RE = _re.compile(r"^[a-zA-Z0-9_]+$")
_active_locks: Dict[str, asyncio.Lock] = {}
_global_semaphore = asyncio.Semaphore(2)  # max 2 concurrent pipelines


def _run_sync(fn, *args):
    """Run a blocking function in the default executor (thread pool)."""
    loop = asyncio.get_event_loop()
    return loop.run_in_executor(None, fn, *args)


# ── Main Pipeline ─────────────────────────────────────
async def run_report_pipeline(
    product_id: str,
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Run full SHARAD subsurface interface analysis pipeline.
    Yields SSE-compatible progress events.
    """
    # Input validation (defense-in-depth)
    if not _PRODUCT_RE.match(product_id):
        yield {"event": "error", "data": {"message": "Invalid product ID"}}
        return

    pid_key = product_id.upper()

    # Per-product lock to prevent concurrent runs on same product
    if pid_key not in _active_locks:
        _active_locks[pid_key] = asyncio.Lock()
    lock = _active_locks[pid_key]

    if lock.locked():
        yield {"event": "error", "data": {"message": "Report already in progress for this product"}}
        return

    async with _global_semaphore:
        async with lock:
            async for event in _run_report_pipeline_inner(product_id):
                yield event


async def _run_report_pipeline_inner(
    product_id: str,
) -> AsyncGenerator[Dict[str, Any], None]:
    """Inner pipeline logic, called under concurrency guards."""
    out_dir = str(REPORT_DIR / product_id.upper())
    os.makedirs(out_dir, exist_ok=True)

    figures: Dict[str, str] = {}
    csv_files: Dict[str, str] = {}

    # ── Load SHARAD data ──
    yield {"event": "step", "data": {"step": "loading", "progress": 0,
           "message": "Loading SHARAD RDR data..."}}

    try:
        from .sharad_highres_router import _get_power, _pick_surface, _get_geometry, _lon_to_180
    except ImportError:
        from sharad_highres_router import _get_power, _pick_surface, _get_geometry, _lon_to_180

    try:
        power, n_traces = await _run_sync(_get_power, product_id)
        geometry, _ = await _run_sync(_get_geometry, product_id)
        lats = geometry["lat"]
        lons = _lon_to_180(geometry["lon"])
        surface = await _run_sync(_pick_surface, product_id, power)
    except FileNotFoundError:
        yield {"event": "error", "data": {"message": f"SHARAD data not found for {product_id}"}}
        return

    yield {"event": "step", "data": {"step": "loading", "progress": 10,
           "message": f"Loaded {n_traces} traces, {N_RANGE_BINS} range bins"}}

    # Track distance
    track_km = await _run_sync(_track_distance_km, lats, lons)

    # Downsample factor
    ds = max(1, n_traces // MAX_DISPLAY_TRACES)

    # ── Detect interface ──
    yield {"event": "step", "data": {"step": "detection", "progress": 15,
           "message": "Detecting subsurface interface..."}}

    interface = await _run_sync(detect_interface, power, surface)

    n_detected = int((interface.bins >= 0).sum())
    yield {"event": "step", "data": {"step": "detection", "progress": 30,
           "message": f"Detection complete: {n_detected} traces, "
                      f"{len(interface.coherent_segments)} segments, "
                      f"confidence={interface.confidence}"}}

    # ── Generate annotated radargram ──
    yield {"event": "step", "data": {"step": "figures", "progress": 35,
           "message": "Generating annotated radargram..."}}

    fig_path = os.path.join(out_dir, "radargram_annotated.png")
    await _run_sync(generate_radargram_figure, power, surface, interface, track_km, fig_path, ds)
    figures["radargram_annotated"] = fig_path

    # Interface mask
    mask_path = os.path.join(out_dir, "interface_mask.png")
    await _run_sync(generate_interface_mask, interface, n_traces, mask_path)
    figures["interface_mask"] = mask_path

    yield {"event": "figure", "data": {"id": "radargram_annotated", "path": fig_path}}

    # ── Depth estimation ──
    yield {"event": "step", "data": {"step": "depth", "progress": 50,
           "message": "Computing depth estimates for 6 dielectric models..."}}

    depths = estimate_depths(interface, surface)

    csv_path = os.path.join(out_dir, "depth_estimation_table.csv")
    generate_depth_csv(depths, csv_path)
    csv_files["depth_estimation"] = csv_path

    ice_depth = next((d for d in depths if d.label == "Pure Water Ice"), None)
    yield {"event": "step", "data": {"step": "depth", "progress": 60,
           "message": f"Ice-model depth: {ice_depth.mean_m:.1f} m mean" if ice_depth else "Depth computed"}}

    # ── DTM comparison ──
    yield {"event": "step", "data": {"step": "dtm", "progress": 65,
           "message": "Searching for co-located HiRISE DTM..."}}

    dtm_product_id = await _run_sync(find_overlapping_dtm, lats, lons)
    dtm_elevation = None
    eps_profile = None

    if dtm_product_id:
        yield {"event": "step", "data": {"step": "dtm", "progress": 70,
               "message": f"Loading DTM: {dtm_product_id}"}}

        dtm_elevation = await _run_sync(load_dtm_along_track, dtm_product_id, lats, lons)

        if dtm_elevation is not None and not np.all(np.isnan(dtm_elevation)):
            cross_path = os.path.join(out_dir, "cross_section_comparison.png")
            await _run_sync(generate_cross_section, track_km, surface, interface, depths, dtm_elevation, cross_path, ds)
            figures["cross_section"] = cross_path

            yield {"event": "figure", "data": {"id": "cross_section", "path": cross_path}}

            # ── Dielectric back-calculation ──
            yield {"event": "step", "data": {"step": "dielectric", "progress": 80,
                   "message": "Back-calculating dielectric constant..."}}

            eps_profile = compute_dielectric_profile(interface, surface, dtm_elevation)
            if eps_profile is not None:
                eps_fig_path = os.path.join(out_dir, "dielectric_profile.png")
                ok = await _run_sync(generate_dielectric_figure, track_km, eps_profile, eps_fig_path)
                if ok:
                    figures["dielectric_profile"] = eps_fig_path

                eps_csv_path = os.path.join(out_dir, "dielectric_values.csv")
                generate_dielectric_csv(track_km, lats, lons, eps_profile, eps_csv_path)
                csv_files["dielectric_values"] = eps_csv_path
        else:
            yield {"event": "step", "data": {"step": "dtm", "progress": 75,
                   "message": "DTM loaded but no valid elevation along track"}}
    else:
        yield {"event": "step", "data": {"step": "dtm", "progress": 75,
               "message": "No co-located HiRISE DTM found. Skipping cross-section."}}

    # ── PDF Report ──
    yield {"event": "step", "data": {"step": "report", "progress": 90,
           "message": "Generating PDF report..."}}

    pdf_path = os.path.join(out_dir, f"MarsLab_SHARAD_Interface_Report_{product_id.upper()}.pdf")
    pdf_ok = await _run_sync(
        generate_pdf_report, product_id, interface, depths, dtm_product_id,
        eps_profile, figures, pdf_path,
    )

    # Write completion marker
    marker_path = os.path.join(out_dir, ".complete")
    with open(marker_path, "w") as f:
        json.dump({"completed": datetime.now().isoformat()}, f)

    # ── Summary ──
    summary_parts = [
        f"SHARAD Interface Report for {product_id.upper()}",
        f"Confidence: {interface.confidence}",
        f"Detection rate: {interface.detection_rate * 100:.1f}%",
        f"Coherent segments: {len(interface.coherent_segments)}",
    ]
    if ice_depth:
        summary_parts.append(f"Ice-model depth: {ice_depth.mean_m:.1f} m ({ice_depth.min_m:.1f}-{ice_depth.max_m:.1f} m)")
    if dtm_product_id:
        summary_parts.append(f"DTM comparison: {dtm_product_id}")
    if eps_profile is not None:
        valid_eps = eps_profile[~np.isnan(eps_profile)]
        if len(valid_eps):
            summary_parts.append(f"Median er: {np.median(valid_eps):.2f}")

    yield {"event": "step", "data": {"step": "report", "progress": 100,
           "message": "Report generation complete"}}

    yield {"event": "done", "data": {
        "product_id": product_id.upper(),
        "figures": {k: os.path.basename(v) for k, v in figures.items()},
        "csv_files": {k: os.path.basename(v) for k, v in csv_files.items()},
        "pdf_path": os.path.basename(pdf_path) if pdf_ok else None,
        "summary": "\n".join(summary_parts),
        "interface": {
            "confidence": interface.confidence,
            "detection_rate": interface.detection_rate,
            "n_segments": len(interface.coherent_segments),
            "mean_depth_bins": interface.mean_depth_bins,
        },
        "depths": [
            {"label": d.label, "epsilon_r": d.epsilon_r,
             "mean_m": d.mean_m, "min_m": d.min_m, "max_m": d.max_m}
            for d in depths
        ],
    }}
