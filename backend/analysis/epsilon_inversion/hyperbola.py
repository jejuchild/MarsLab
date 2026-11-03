"""Diffraction-hyperbola fitting for εr estimation (EXPERIMENTAL).

Model (ground-based NMO form):  t(x) = √(t0² + (2(x−x0)/v)²),  εr = (c/v)²
with t the two-way time *below the surface echo of the same trace* and x the
along-track distance from the RDR geometry (≈18 m/trace SS19, ≈36 m SS4).

Caveat — why this is flagged experimental for SHARAD RDRs:
  * The RDR is already azimuth-focused (Doppler-processed), which collapses
    diffraction hyperbolae; residual curvature is not a clean NMO signal.
  * From ~270 km altitude the free-space moveout 2(√(h²+x²)−h)/c dominates
    the apparent curvature and is independent of εr, so εr is poorly
    constrained.  We fit that orbital model too and flag the result when it
    explains the picks as well as the NMO model.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from analysis.shared.constants import SHARAD_SAMPLE_INTERVAL_US
from analysis.shared.coordinates import cumulative_distance_m, lon_to_180
from analysis.shared.sharad_math import (
    fit_nmo_hyperbola,
    nmo_twt,
    orbital_moveout_twt,
)

logger = logging.getLogger(__name__)

EXPERIMENTAL_NOTE = (
    "Experimental — poorly constrained for focused orbital RDR: the RDR is "
    "Doppler-focused and, from ~270 km altitude, free-space moveout dominates "
    "hyperbola shape independently of εr."
)


@dataclass
class HyperbolaFitResult:
    """Result of hyperbola fitting."""

    epsilon_r_est: Optional[float]  # Estimated εr, None if fit failed
    epsilon_r_std: Optional[float]  # 1σ uncertainty in estimate
    rmse_bins: float  # RMS error of residuals (in bins)
    inlier_ratio: float  # Fraction of picks that are inliers (0–1)
    n_inliers: int  # Number of inlier picks used
    quality_note: str  # Human-readable fit quality
    diagnostics: Dict  # Curve parameters, residuals, etc.
    epsilon_r_ci95: Optional[tuple] = None
    depth_m: Optional[float] = None
    v_ms: Optional[float] = None
    flags: List[str] = field(default_factory=list)
    picks: List[Dict] = field(default_factory=list)      # [{trace, bin}] full-res
    overlay: List[Dict] = field(default_factory=list)    # fitted curve [{trace, bin}]


def _fail(note: str, **kw) -> HyperbolaFitResult:
    return HyperbolaFitResult(
        epsilon_r_est=None, epsilon_r_std=None, rmse_bins=float("nan"),
        inlier_ratio=0.0, n_inliers=0, quality_note=note, diagnostics={},
        flags=["experimental"], **kw,
    )


def pick_hyperbola(power_win: np.ndarray, apex_row: int, apex_bin: int,
                   rel_thresh: float = 0.1, search: int = 4, max_gap: int = 3):
    """Track a diffraction hyperbola outward from the apex.

    The apex is the strongest bin within ±``search`` of (apex_row, apex_bin).
    Moving trace by trace in each direction, the next pick is the strongest
    bin within ±``search`` of the linearly extrapolated position; it is kept
    if ≥ ``rel_thresh`` × apex power.  Tracking stops after ``max_gap``
    consecutive misses.  Continuity keeps flat reflectors / clutter that are
    merely the brightest thing in a trace from being mistaken for the limbs.

    Returns (rows, bins, powers) sorted by row.
    """
    n_rows, n_bins = power_win.shape
    lo = max(0, apex_bin - search)
    hi = min(n_bins, apex_bin + search + 1)
    if not (0 <= apex_row < n_rows) or hi <= lo:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0)
    b_apex = lo + int(np.argmax(power_win[apex_row, lo:hi]))
    p_apex = float(power_win[apex_row, b_apex])
    if p_apex <= 0:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0)
    picks = {apex_row: (b_apex, p_apex)}
    for step in (1, -1):
        prev2, prev = b_apex, b_apex
        gap = 0
        r = apex_row + step
        while 0 <= r < n_rows and gap < max_gap:
            pred = prev + max(0, prev - prev2)
            lo = int(max(0, pred - search))
            hi = int(min(n_bins, pred + search + 1))
            if hi <= lo:
                break
            b = lo + int(np.argmax(power_win[r, lo:hi]))
            pv = float(power_win[r, b])
            if pv >= rel_thresh * p_apex and b >= prev:
                picks[r] = (b, pv)
                prev2, prev = prev, b
                gap = 0
            else:
                gap += 1
            r += step
    rows = np.array(sorted(picks))
    bins = np.array([picks[r][0] for r in rows])
    vals = np.array([picks[r][1] for r in rows])
    return rows, bins, vals


class HyperbolaFitter:
    """Fit a diffraction hyperbola to picks around an apex and estimate εr."""

    def fit(
        self,
        product_id: str,
        trace_idx0: int,
        bin_idx0: int,
        window_traces: int = 150,
        window_bins: int = 120,
    ) -> HyperbolaFitResult:
        """Fit around full-resolution trace ``trace_idx0`` / bin ``bin_idx0``.

        1. Window ±window_traces/2 × ±window_bins/2 around the apex.
        2. Continuity-tracked picks outward from the apex (≥10 % of apex power).
        3. t = (pick − surface_pick[trace]) × 37.5 ns  (time below surface).
        4. x = along-track distance from geometry, relative to the apex trace.
        5. Robust NMO fit (v0 = c/√3, εr bounded to [1, 20]).
        6. Compare with the εr-independent orbital moveout model.
        """
        try:
            from api.sharad_highres_router import _get_power, _get_geometry, _pick_surface

            power, n_traces = _get_power(product_id)
            geom, _ = _get_geometry(product_id)
            surface = _pick_surface(product_id, power)
        except Exception as e:
            return _fail(f"Exception: {e}")

        half_t = max(window_traces // 2, 3)
        half_b = max(window_bins // 2, 3)
        if trace_idx0 < 0 or trace_idx0 >= n_traces:
            return _fail("Apex trace out of range")
        t0i = max(0, trace_idx0 - half_t)
        t1i = min(n_traces, trace_idx0 + half_t + 1)
        b0 = max(0, bin_idx0 - half_b)
        b1 = min(power.shape[1], bin_idx0 + half_b + 1)
        win = np.asarray(power[t0i:t1i, b0:b1], dtype=np.float64)
        if win.size == 0 or not np.isfinite(win).any() or win.max() <= 0:
            return _fail("Window extraction failed")

        rows, peaks, vals = pick_hyperbola(win, trace_idx0 - t0i, bin_idx0 - b0)
        traces = rows + t0i
        bins = peaks + b0
        surf = np.asarray(surface)[traces].astype(np.float64)
        ok = (surf >= 0) & (bins > surf)
        traces, bins, vals, surf = traces[ok], bins[ok], vals[ok], surf[ok]
        if len(traces) < 5:
            return _fail(f"Insufficient picks below the surface ({len(traces)} < 5)")

        lat = np.asarray(geom["lat"][t0i:t1i], dtype=np.float64)
        lon = lon_to_180(np.asarray(geom["lon"][t0i:t1i], dtype=np.float64))
        dist = cumulative_distance_m(lat, lon)
        x = dist[traces - t0i] - dist[trace_idx0 - t0i]
        t = (bins - surf) * SHARAD_SAMPLE_INTERVAL_US * 1e-6
        spacing = float(np.median(np.diff(dist))) if len(dist) > 1 else float("nan")

        fit = fit_nmo_hyperbola(x, t, weights=vals)
        if fit.epsilon_r is None:
            return _fail(fit.note)

        # Orbital (εr-independent) alternative: t = t0 + 2(√(h²+(x−x0)²)−h)/c
        h = float(np.median(np.asarray(geom["alt"][t0i:t1i]))) * 1e3
        m_orb = orbital_moveout_twt(x - fit.x0_m, h)
        orb_t0 = float(np.median(t - m_orb))
        orb_rmse = float(np.sqrt(np.mean((orb_t0 + m_orb - t) ** 2)) / (SHARAD_SAMPLE_INTERVAL_US * 1e-6))

        flags = ["experimental"]
        if fit.note != "ok":
            flags.append("epsilon_at_bound")
        if orb_rmse <= 1.1 * fit.rmse_bins:
            flags.append("orbital_moveout_explains_picks")
        if fit.epsilon_r_ci95 and (fit.epsilon_r_ci95[1] - fit.epsilon_r_ci95[0]) > 0.5 * fit.epsilon_r:
            flags.append("wide_ci")

        # Fitted curve back in radargram coordinates (bin = surface + t/dt)
        tr_curve = np.arange(t0i, t1i)
        surf_curve = np.asarray(surface)[tr_curve].astype(np.float64)
        x_curve = dist - dist[trace_idx0 - t0i]
        t_curve = nmo_twt(x_curve - fit.x0_m, fit.t0_s, fit.v_mps)
        b_curve = surf_curve + t_curve / (SHARAD_SAMPLE_INTERVAL_US * 1e-6)
        overlay = [
            {"trace": int(tt), "bin": round(float(bb), 2)}
            for tt, bb, ss in zip(tr_curve, b_curve, surf_curve)
            if ss >= 0 and b0 <= bb <= b1
        ]

        note = (
            f"εr={fit.epsilon_r:.2f}"
            + (f" ±{fit.epsilon_r_std:.2f}" if fit.epsilon_r_std is not None else "")
            + f", {fit.n_used}/{len(traces)} picks, rmse={fit.rmse_bins:.2f} bins. {EXPERIMENTAL_NOTE}"
        )
        return HyperbolaFitResult(
            epsilon_r_est=fit.epsilon_r,
            epsilon_r_std=fit.epsilon_r_std,
            rmse_bins=fit.rmse_bins,
            inlier_ratio=fit.n_used / len(traces),
            n_inliers=fit.n_used,
            quality_note=note,
            diagnostics={
                "v_ms": round(fit.v_mps, 0),
                "t0_us": round(fit.t0_s * 1e6, 4),
                "x0_m": round(fit.x0_m, 1),
                "n_picks": int(len(traces)),
                "trace_spacing_m": round(spacing, 2),
                "altitude_km": round(h / 1e3, 2),
                "orbital_model_rmse_bins": round(orb_rmse, 3),
                "model": "t(x)=sqrt(t0^2+(2(x-x0)/v)^2), t below surface echo",
            },
            epsilon_r_ci95=fit.epsilon_r_ci95,
            depth_m=fit.depth_m,
            v_ms=fit.v_mps,
            flags=flags,
            picks=[{"trace": int(a), "bin": int(b)} for a, b in zip(traces, bins)],
            overlay=overlay,
        )
