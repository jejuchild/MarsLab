"""
Semi-automatic hyperbola fitting for SHARAD radargrams.

Fits t(x) = sqrt(t0² + (2x/v)²) to diffraction hyperbolas to
estimate subsurface wave velocity v, then εr = (c/v)².

Includes bootstrap confidence intervals, quality metrics, and
clutter risk heuristics.
"""

from __future__ import annotations

import logging
import math
import os
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import least_squares

from .models import (
    HyperbolaFitRequest,
    HyperbolaFitResult,
    HyperbolaFitQuality,
    OverlayCIPoint,
    OverlayPoint,
)

logger = logging.getLogger(__name__)

SPEED_OF_LIGHT = 299_792_458.0  # m/s
SHARAD_DT_S = 3.75e-8  # 0.0375 µs per bin (26.67 MHz)
SHARAD_DX_M_DEFAULT = 463.0  # ~MRO ground-track spacing per trace


# ── Helpers ──────────────────────────────────────────────

def _load_power_and_surface(product_id: str):
    """Load SHARAD power array and surface picks via the existing router helpers."""
    import sys
    backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)
    from api.sharad_highres_router import _get_power, _pick_surface
    power, total_rows = _get_power(product_id)
    surface = _pick_surface(product_id, power)
    return power, surface, total_rows


def _compute_dx_m(product_id: str) -> float:
    """Compute along-track spacing from geometry lat/lon."""
    try:
        import sys
        backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if backend_dir not in sys.path:
            sys.path.insert(0, backend_dir)
        from api.sharad_highres_router import _get_geometry, _centric_to_graphic
        geom, n_rows = _get_geometry(product_id)
        lats = _centric_to_graphic(geom["lat"])
        lons = geom["lon"].copy()
        # Convert 0-360 to -180/180
        lons[lons > 180] -= 360
        # Haversine for adjacent traces
        R_MARS = 3389_500.0  # metres
        n_sample = min(200, len(lats) - 1)
        step = max(1, len(lats) // n_sample)
        dists = []
        for i in range(0, len(lats) - step, step):
            lat1, lat2 = math.radians(lats[i]), math.radians(lats[i + step])
            dlon = math.radians(lons[i + step] - lons[i])
            a = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
            dists.append(R_MARS * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)) / step)
        dx = float(np.median(dists)) if dists else SHARAD_DX_M_DEFAULT
        logger.info(f"Computed dx_m_per_trace = {dx:.1f} for {product_id}")
        return dx
    except Exception as e:
        logger.warning(f"Cannot compute dx for {product_id}: {e}; using default {SHARAD_DX_M_DEFAULT}")
        return SHARAD_DX_M_DEFAULT


def _extract_ridge_points(
    power: np.ndarray,
    surface: np.ndarray,
    apex_trace: int,
    apex_bin: int,
    roi_traces: int,
    roi_bins: int,
    top_k: int = 3,
) -> List[Tuple[int, int, float]]:
    """Extract bright ridge points inside ROI for fitting.

    Returns list of (trace, bin, energy) tuples.
    """
    n_traces, n_bins = power.shape
    t_lo = max(0, apex_trace - roi_traces)
    t_hi = min(n_traces, apex_trace + roi_traces + 1)
    b_lo = max(0, apex_bin - roi_bins)
    b_hi = min(n_bins, apex_bin + roi_bins + 1)

    # Work in log-power
    roi = power[t_lo:t_hi, b_lo:b_hi].astype(np.float64)
    roi = np.where(roi > 0, np.log10(roi + 1e-30), 0)

    # Percentile clip
    p2, p98 = np.percentile(roi[roi > 0], [2, 98]) if np.any(roi > 0) else (0, 1)
    if p98 <= p2:
        p98 = p2 + 1
    roi = np.clip((roi - p2) / (p98 - p2), 0, 1)

    points = []
    for ti in range(roi.shape[0]):
        abs_trace = t_lo + ti
        col = roi[ti, :]
        # Find top-k local peaks
        if col.max() < 0.15:
            continue
        idxs = np.argsort(col)[-top_k:]
        for idx in idxs:
            if col[idx] < 0.2:
                continue
            abs_bin = b_lo + int(idx)
            points.append((abs_trace, abs_bin, float(col[idx])))

    return points


# ── Core fit ─────────────────────────────────────────────

def _hyperbola_residuals(params, x_m, t_obs_s):
    """Residuals for least-squares: t_model - t_obs."""
    t0, v = params
    t_model = np.sqrt(t0 ** 2 + (2 * x_m / v) ** 2)
    return t_model - t_obs_s


def _fit_hyperbola(
    x_m: np.ndarray,
    t_s: np.ndarray,
    t0_init: float,
    v_init: float = SPEED_OF_LIGHT / math.sqrt(3.15),
) -> Tuple[float, float, float]:
    """Fit v and t0 to hyperbola model. Returns (t0, v, residual_rms)."""
    try:
        result = least_squares(
            _hyperbola_residuals,
            x0=[t0_init, v_init],
            args=(x_m, t_s),
            bounds=([0, 1e6], [1e-4, SPEED_OF_LIGHT]),
            method="trf",
            max_nfev=2000,
        )
        t0_fit, v_fit = result.x
        rms = float(np.sqrt(np.mean(result.fun ** 2)))
        return t0_fit, v_fit, rms
    except Exception as e:
        logger.warning(f"Fit failed: {e}")
        return t0_init, v_init, 999.0


def fit_hyperbola(req: HyperbolaFitRequest) -> HyperbolaFitResult:
    """Run full hyperbola fit pipeline."""

    power, surface, total_rows = _load_power_and_surface(req.product_id)
    n_traces, n_bins = power.shape

    # Compute dx
    dx = req.dx_m_per_trace
    if dx <= 0:
        dx = _compute_dx_m(req.product_id)
    dt = req.dt_s_per_bin if req.dt_s_per_bin > 0 else SHARAD_DT_S

    # Surface bin at apex
    apex_surface = int(surface[req.apex_trace]) if 0 <= req.apex_trace < len(surface) and surface[req.apex_trace] >= 0 else req.apex_bin - 20

    # Extract ridge points
    ridge_pts = _extract_ridge_points(
        power, surface, req.apex_trace, req.apex_bin,
        req.roi_traces, req.roi_bins, top_k=3,
    )

    if len(ridge_pts) < 5:
        return HyperbolaFitResult(
            v_mps=0, epsr=0,
            flags=["INSUFFICIENT_POINTS"],
            quality=HyperbolaFitQuality(support_traces=len(ridge_pts)),
        )

    # Build arrays: x = distance from apex, t = two-way time below surface
    traces = np.array([p[0] for p in ridge_pts])
    bins = np.array([p[1] for p in ridge_pts])

    x_m = (traces - req.apex_trace) * dx
    # Time relative to apex bin (could use surface if available)
    t_s = bins * dt

    # Initial guesses
    t0_init = float(req.apex_bin * dt)
    v_init = SPEED_OF_LIGHT / math.sqrt(3.15)

    # Fit
    t0_fit, v_fit, rms = _fit_hyperbola(x_m, t_s, t0_init, v_init)

    # Convert to εr and depth
    epsr = (SPEED_OF_LIGHT / v_fit) ** 2 if v_fit > 0 else 0
    dt_surface = abs(req.apex_bin - apex_surface) * dt
    depth_m = v_fit * dt_surface / 2 if v_fit > 0 else 0

    # ── Bootstrap CI ──
    epsr_boot = []
    depth_boot = []
    n_boot = req.options.bootstrap_iters
    n_pts = len(x_m)
    rng = np.random.default_rng(42)

    for _ in range(n_boot):
        idx = rng.choice(n_pts, size=n_pts, replace=True)
        _, vb, _ = _fit_hyperbola(x_m[idx], t_s[idx], t0_init, v_fit)
        if vb > 1e6:
            eb = (SPEED_OF_LIGHT / vb) ** 2
            epsr_boot.append(eb)
            depth_boot.append(vb * dt_surface / 2)

    if len(epsr_boot) > 10:
        epsr_ci = [float(np.percentile(epsr_boot, 2.5)), float(np.percentile(epsr_boot, 97.5))]
        depth_ci = [float(np.percentile(depth_boot, 2.5)), float(np.percentile(depth_boot, 97.5))]
    else:
        epsr_ci = [epsr * 0.8, epsr * 1.2]
        depth_ci = [depth_m * 0.8, depth_m * 1.2]

    # ── Quality metrics ──
    # SNR: mean energy along fitted curve vs background
    fitted_bins = []
    for pt in ridge_pts:
        fitted_bins.append(power[pt[0], pt[1]] if 0 <= pt[0] < n_traces and 0 <= pt[1] < n_bins else 0)
    signal = np.mean(fitted_bins) if fitted_bins else 0
    b_lo = max(0, req.apex_bin - req.roi_bins)
    b_hi = min(n_bins, req.apex_bin + req.roi_bins)
    t_lo = max(0, req.apex_trace - req.roi_traces)
    t_hi = min(n_traces, req.apex_trace + req.roi_traces)
    bg = np.median(power[t_lo:t_hi, b_lo:b_hi].astype(np.float64))
    snr = float(signal / bg) if bg > 0 else 0.0

    unique_traces = len(set(p[0] for p in ridge_pts))

    # ── Clutter risk ──
    flags = []
    if epsr >= 2.7 and epsr <= 3.4:
        flags.append("ICE_CONSISTENT")
    elif epsr >= 2.0 and epsr <= 5.0:
        flags.append("REGOLITH_RANGE")

    if dt_surface < 1e-6:
        flags.append("CLUTTER_RISK_HIGH")
    elif dt_surface < 2e-6:
        flags.append("CLUTTER_RISK_MED")
    else:
        flags.append("CLUTTER_RISK_LOW")

    if unique_traces < 10:
        flags.append("LOW_SUPPORT")
    if snr < 2.5:
        flags.append("LOW_SNR")

    # ── Build overlay polyline ──
    overlay = []
    ci_band = []
    t_lo_trace = max(0, req.apex_trace - req.roi_traces)
    t_hi_trace = min(n_traces, req.apex_trace + req.roi_traces)
    for t in range(t_lo_trace, t_hi_trace, max(1, (t_hi_trace - t_lo_trace) // 200)):
        x = (t - req.apex_trace) * dx
        t_model = math.sqrt(t0_fit ** 2 + (2 * x / v_fit) ** 2) if v_fit > 0 else 0
        bin_model = t_model / dt
        overlay.append(OverlayPoint(trace=t, bin=round(bin_model, 2)))

        # CI band from bootstrap
        if len(epsr_boot) > 10:
            v_lo = SPEED_OF_LIGHT / math.sqrt(max(epsr_ci[1], 1.01))
            v_hi = SPEED_OF_LIGHT / math.sqrt(max(epsr_ci[0], 1.01))
            t_lo_m = math.sqrt(t0_fit ** 2 + (2 * x / v_hi) ** 2) if v_hi > 0 else 0
            t_hi_m = math.sqrt(t0_fit ** 2 + (2 * x / v_lo) ** 2) if v_lo > 0 else 0
            ci_band.append(OverlayCIPoint(trace=t, bin_lo=round(t_lo_m / dt, 2), bin_hi=round(t_hi_m / dt, 2)))

    # ── Preset curves ──
    preset_curves = {}
    for preset_epsr in req.options.epsr_presets:
        v_preset = SPEED_OF_LIGHT / math.sqrt(max(preset_epsr, 1.01))
        curve = []
        for t in range(t_lo_trace, t_hi_trace, max(1, (t_hi_trace - t_lo_trace) // 200)):
            x = (t - req.apex_trace) * dx
            t_model = math.sqrt(t0_fit ** 2 + (2 * x / v_preset) ** 2)
            curve.append(OverlayPoint(trace=t, bin=round(t_model / dt, 2)))
        preset_curves[str(preset_epsr)] = curve

    return HyperbolaFitResult(
        v_mps=round(v_fit, 1),
        epsr=round(epsr, 3),
        epsr_ci95=[round(c, 3) for c in epsr_ci],
        dt_surface_s=round(dt_surface, 12),
        depth_m=round(depth_m, 1),
        depth_ci95=[round(c, 1) for c in depth_ci],
        quality=HyperbolaFitQuality(
            snr=round(snr, 2),
            residual=round(rms, 10),
            support_traces=unique_traces,
        ),
        flags=flags,
        overlay_polyline=overlay,
        overlay_ci_band=ci_band if ci_band else None,
        preset_curves=preset_curves,
    )


def auto_detect_apexes(
    product_id: str,
    n_candidates: int = 3,
    min_bin_below_surface: int = 20,
    max_bin_below_surface: int = 300,
) -> List[dict]:
    """Auto-detect top diffraction apex candidates in a SHARAD product.

    Returns list of {trace, bin, energy} dicts sorted by energy descending.
    """
    power, surface, total_rows = _load_power_and_surface(product_id)
    n_traces, n_bins = power.shape

    # Work in log-power
    log_power = np.log10(power.astype(np.float64) + 1e-30)

    # Search zone: min_bin..max_bin below surface per trace
    candidates = []
    step = max(1, n_traces // 500)  # sample every ~500 traces for speed

    for t in range(0, n_traces, step):
        s = int(surface[t])
        if s < 0:
            continue
        b_lo = s + min_bin_below_surface
        b_hi = min(n_bins, s + max_bin_below_surface)
        if b_hi <= b_lo:
            continue
        col = log_power[t, b_lo:b_hi]
        peak_idx = int(np.argmax(col))
        peak_val = col[peak_idx]
        # local SNR
        median_val = float(np.median(col))
        if median_val > 0 and peak_val / median_val > 3.0:
            candidates.append({
                "trace": t,
                "bin": b_lo + peak_idx,
                "energy": float(peak_val),
            })

    # Sort by energy descending, take top n
    candidates.sort(key=lambda c: c["energy"], reverse=True)

    # De-duplicate: remove candidates within 50 traces of a higher-energy one
    filtered = []
    for c in candidates:
        if all(abs(c["trace"] - f["trace"]) > 50 for f in filtered):
            filtered.append(c)
        if len(filtered) >= n_candidates:
            break

    return filtered
