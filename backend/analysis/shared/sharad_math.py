"""Pure SHARAD radargram math (no I/O on caches, no FastAPI).

Everything here is unit-tested in tests/test_sharad_math.py.  The router and
analysis pipelines import these helpers so the physics lives in one place.

Conventions
-----------
* ``bin`` is a range-sample index inside the 667-sample receive window.
* One bin = 37.5 ns of two-way travel time (1 / 26.67 MHz ADC).
* Depths are below the picked surface echo, in metres.
* Attenuation α is reported as one-way loss in dB/m (two-way loss = 2αd),
  matching the historical MarsLab convention and the transparency classes.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np

from analysis.shared.constants import SPEED_OF_LIGHT, SHARAD_SAMPLE_INTERVAL_US

BIN_DT_S = SHARAD_SAMPLE_INTERVAL_US * 1e-6           # 37.5 ns
SHARAD_BANDWIDTH_HZ = 10e6                            # 15–25 MHz chirp
# Free-space range resolution c / (2B) = 15 m
FREE_SPACE_RESOLUTION_M = SPEED_OF_LIGHT / (2.0 * SHARAD_BANDWIDTH_HZ)

ROW_BYTES = 5822
RWOT_OFFSET = 186            # RECEIVE_WINDOW_OPENING_TIME, float32, units of 37.5 ns


# ─────────────────────────────────────────────────────────────────────
# Depth conversion
# ─────────────────────────────────────────────────────────────────────

def depth_from_twt(
    twt_s,
    epsilon_r1: float,
    epsilon_r2: Optional[float] = None,
    boundary_m: float = 0.0,
):
    """Two-way time below the surface echo → depth (m), piecewise 2-layer.

    Layer 1 (εr1) from the surface to ``boundary_m``; layer 2 (εr2) below.
    Uniform εr1 when ``boundary_m <= 0`` or εr2 is None / equal to εr1.
    Works on scalars and numpy arrays.  Negative times (above the surface)
    are clipped to 0.
    """
    t = np.maximum(np.asarray(twt_s, dtype=np.float64), 0.0)
    v1 = SPEED_OF_LIGHT / math.sqrt(epsilon_r1)
    if epsilon_r2 is None or boundary_m <= 0 or epsilon_r2 == epsilon_r1:
        d = v1 * t / 2.0
    else:
        v2 = SPEED_OF_LIGHT / math.sqrt(epsilon_r2)
        t1 = 2.0 * boundary_m / v1
        d = np.where(t <= t1, v1 * t / 2.0, boundary_m + v2 * (t - t1) / 2.0)
    return float(d) if np.ndim(d) == 0 else d


def depth_resolution(
    twt_s: float,
    epsilon_r1: float,
    epsilon_r2: Optional[float] = None,
    boundary_m: float = 0.0,
) -> dict:
    """Depth uncertainty / resolution at a given two-way time.

    * ``bin_uncertainty_m``: ±1 range bin (37.5 ns) of quantisation,
      using the velocity of the layer the point falls in.
    * ``vertical_resolution_m``: c / (2B√εr) with B = 10 MHz (15 m in vacuum).
    """
    d = depth_from_twt(twt_s, epsilon_r1, epsilon_r2, boundary_m)
    eps = epsilon_r1
    if epsilon_r2 is not None and boundary_m > 0 and d > boundary_m:
        eps = epsilon_r2
    v = SPEED_OF_LIGHT / math.sqrt(eps)
    return {
        "bin_uncertainty_m": v * BIN_DT_S / 2.0,
        "vertical_resolution_m": FREE_SPACE_RESOLUTION_M / math.sqrt(eps),
        "epsilon_r_at_depth": eps,
    }


# ─────────────────────────────────────────────────────────────────────
# Surface picking
# ─────────────────────────────────────────────────────────────────────

def pick_surface_leading_edge(
    power: np.ndarray,
    threshold_db: float = 15.0,
    peak_rel_db: float = 15.0,
    peak_window: int = 8,
    min_snr_db: float = 10.0,
    noise_bins: Optional[slice] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Leading-edge ("first return") surface pick.

    For each trace the noise floor is the median power (robust: most bins
    are noise or weak clutter).  The leading edge is the first bin whose power
    exceeds ``max(noise·10^(threshold_db/10), peak·10^(-peak_rel_db/10))`` —
    i.e. clearly above noise AND within ``peak_rel_db`` of the strongest echo,
    so isolated noise spikes and weak early clutter do not trigger it.  The
    pick is refined to the local maximum within ``peak_window`` bins after the
    edge (the echo peak of the first strong return, not the strongest return
    of the whole trace, which is often off-nadir clutter or a subsurface echo).

    Returns ``(bins, snr_db)``; ``bins`` is -1 where the surface echo is
    weaker than ``min_snr_db`` above noise.
    """
    p = np.asarray(power, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] == 0:
        return np.zeros(0, np.int32), np.zeros(0, np.float32)
    n_tr, n_bins = p.shape
    if noise_bins is not None:
        noise = np.median(p[:, noise_bins], axis=1)
    else:
        noise = np.median(p, axis=1)
    noise = np.maximum(noise, 1e-30)
    peak = p.max(axis=1)
    thr = np.maximum(noise * 10 ** (threshold_db / 10.0), peak * 10 ** (-peak_rel_db / 10.0))
    above = p > thr[:, None]
    has = above.any(axis=1)
    edge = np.argmax(above, axis=1)

    # local max within [edge, edge+peak_window)
    offs = np.arange(peak_window)
    cols = np.clip(edge[:, None] + offs[None, :], 0, n_bins - 1)
    local = np.take_along_axis(p, cols, axis=1)
    bins = cols[np.arange(n_tr), np.argmax(local, axis=1)].astype(np.int32)

    snr_db = 10.0 * np.log10(np.maximum(p[np.arange(n_tr), bins], 1e-30) / noise)
    ok = has & (snr_db >= min_snr_db)
    bins = np.where(ok, bins, -1).astype(np.int32)
    snr_db = np.where(ok, snr_db, 0.0).astype(np.float32)
    return bins, snr_db


def reject_pick_outliers(bins: np.ndarray, kernel: int = 31, max_dev: int = 20) -> np.ndarray:
    """Invalidate picks deviating > ``max_dev`` bins from a running median of
    the valid picks (lateral continuity).  Returns a new array."""
    out = np.array(bins, dtype=np.int32, copy=True)
    valid = np.where(out >= 0)[0]
    if len(valid) > kernel:
        from scipy.ndimage import median_filter
        y = out[valid].astype(np.float64)
        med = median_filter(y, size=kernel, mode="nearest")
        out[valid[np.abs(y - med) > max_dev]] = -1
    return out


# ─────────────────────────────────────────────────────────────────────
# Attenuation (power-vs-depth regression)
# ─────────────────────────────────────────────────────────────────────

@dataclass
class AttenuationFit:
    alpha_dBm: Optional[float]        # one-way loss, dB/m  (= -slope/2)
    alpha_se_dBm: Optional[float]     # standard error of α
    slope_dB_per_m: Optional[float]   # d(P_rel)/d(depth)
    intercept_dB: Optional[float]
    r2: Optional[float]
    n: int
    depth_span_m: float


def fit_attenuation(depth_m, rel_power_db, min_points: int = 5, min_span_m: float = 5.0) -> AttenuationFit:
    """Least-squares fit P_rel(dB) = a + s·depth  →  α = −s/2 (one-way dB/m).

    ``rel_power_db`` should be subsurface minus surface echo power for each
    trace so per-trace gain cancels.  Fitting across many traces at different
    depths separates the loss gradient from the (unknown, roughly constant)
    interface reflectivity, unlike the per-trace ratio (P_s − P_ss)/(2d),
    which attributes all of the reflectivity contrast to attenuation.
    """
    d = np.asarray(depth_m, dtype=np.float64)
    y = np.asarray(rel_power_db, dtype=np.float64)
    m = np.isfinite(d) & np.isfinite(y)
    d, y = d[m], y[m]
    n = int(d.size)
    span = float(d.max() - d.min()) if n else 0.0
    if n < min_points or span < min_span_m:
        return AttenuationFit(None, None, None, None, None, n, span)
    A = np.column_stack([np.ones(n), d])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    a, s = float(coef[0]), float(coef[1])
    resid = y - (a + s * d)
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    dof = max(n - 2, 1)
    sxx = float(((d - d.mean()) ** 2).sum())
    se_s = math.sqrt(ss_res / dof / sxx) if sxx > 0 else float("nan")
    return AttenuationFit(
        alpha_dBm=-s / 2.0,
        alpha_se_dBm=se_s / 2.0,
        slope_dB_per_m=s,
        intercept_dB=a,
        r2=r2,
        n=n,
        depth_span_m=span,
    )


def windowed_attenuation(
    depth_m,
    rel_power_db,
    valid,
    half_window: int,
    min_points: int = 10,
    min_span_m: float = 5.0,
) -> dict:
    """Along-track sliding-window power-vs-depth regression (O(n) via prefix sums).

    For each trace i, fits the valid traces in [i−half_window, i+half_window].
    Returns arrays (NaN where unconstrained): ``alpha_dBm`` (one-way, = −slope/2),
    ``alpha_se_dBm``, ``r2``, ``n``, ``depth_span_m``.
    """
    d = np.asarray(depth_m, dtype=np.float64)
    y = np.asarray(rel_power_db, dtype=np.float64)
    v = np.asarray(valid, dtype=bool) & np.isfinite(d) & np.isfinite(y)
    dz = np.where(v, d, 0.0)
    yz = np.where(v, y, 0.0)
    n_tr = d.size

    def cs(a):
        return np.concatenate([[0.0], np.cumsum(a)])

    C1, Cd, Cy = cs(v.astype(np.float64)), cs(dz), cs(yz)
    Cdd, Cdy, Cyy = cs(dz * dz), cs(dz * yz), cs(yz * yz)
    # running min/max depth for the span check
    from scipy.ndimage import maximum_filter1d, minimum_filter1d
    w = 2 * half_window + 1
    dmax = maximum_filter1d(np.where(v, d, -np.inf), size=w, mode="nearest")
    dmin = minimum_filter1d(np.where(v, d, np.inf), size=w, mode="nearest")

    i = np.arange(n_tr)
    a = np.clip(i - half_window, 0, n_tr)
    b = np.clip(i + half_window + 1, 0, n_tr)
    n = C1[b] - C1[a]
    Sd, Sy = Cd[b] - Cd[a], Cy[b] - Cy[a]
    Sdd, Sdy, Syy = Cdd[b] - Cdd[a], Cdy[b] - Cdy[a], Cyy[b] - Cyy[a]
    with np.errstate(invalid="ignore", divide="ignore"):
        sxx = Sdd - Sd * Sd / n
        sxy = Sdy - Sd * Sy / n
        syy = Syy - Sy * Sy / n
        slope = sxy / sxx
        ss_res = np.maximum(syy - slope * sxy, 0.0)
        r2 = np.where(syy > 0, 1.0 - ss_res / syy, 0.0)
        se = np.sqrt(ss_res / np.maximum(n - 2, 1) / sxx)
    span = dmax - dmin
    ok = (n >= min_points) & (span >= min_span_m) & (sxx > 0)
    nan = np.full(n_tr, np.nan)
    return {
        "alpha_dBm": np.where(ok, -slope / 2.0, nan),
        "alpha_se_dBm": np.where(ok, se / 2.0, nan),
        "r2": np.where(ok, r2, nan),
        "n": n.astype(np.int64),
        "depth_span_m": np.where(ok, span, nan),
    }


# ─────────────────────────────────────────────────────────────────────
# Hyperbola (diffraction) fitting
# ─────────────────────────────────────────────────────────────────────

def nmo_twt(x_m, t0_s: float, v_mps: float):
    """Two-way time below the surface of a point diffractor at depth
    d = v·t0/2 observed at horizontal offset x:  t(x) = √(t0² + (2x/v)²)."""
    x = np.asarray(x_m, dtype=np.float64)
    return np.sqrt(t0_s ** 2 + (2.0 * x / v_mps) ** 2)


def orbital_moveout_twt(x_m, altitude_m: float):
    """Extra two-way free-space delay of an off-nadir point at offset x seen
    from altitude h: 2(√(h²+x²) − h)/c.  For an orbital sounder this term
    dominates the shape of any diffraction hyperbola and is independent of εr."""
    x = np.asarray(x_m, dtype=np.float64)
    return 2.0 * (np.sqrt(altitude_m ** 2 + x ** 2) - altitude_m) / SPEED_OF_LIGHT


@dataclass
class HyperbolaFit:
    v_mps: Optional[float]
    t0_s: Optional[float]
    x0_m: Optional[float]
    epsilon_r: Optional[float]
    epsilon_r_std: Optional[float]
    epsilon_r_ci95: Optional[tuple]
    depth_m: Optional[float]
    rmse_bins: float
    n_used: int
    note: str


def fit_nmo_hyperbola(
    x_m,
    t_s,
    weights=None,
    eps_bounds: tuple = (1.0, 20.0),
    fit_x0: bool = True,
) -> HyperbolaFit:
    """Fit t(x) = √(t0² + (2(x−x0)/v)²) to picks; return v, εr = (c/v)², depth.

    ``t_s`` must be two-way time measured from the surface echo of the same
    trace; ``x_m`` along-track distance (m) from geometry.  The initial guess
    is v = c/√3.  Velocity is bounded to the εr range ``eps_bounds``.  Picks
    with residual > 2.5σ are dropped once and the fit repeated.
    """
    from scipy.optimize import least_squares

    x = np.asarray(x_m, dtype=np.float64)
    t = np.asarray(t_s, dtype=np.float64)
    w = np.ones_like(x) if weights is None else np.asarray(weights, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(t) & (t > 0)
    x, t, w = x[ok], t[ok], w[ok]
    if x.size < 5:
        return HyperbolaFit(None, None, None, None, None, None, None, float("nan"), int(x.size),
                            f"insufficient picks ({x.size} < 5)")

    v_lo = SPEED_OF_LIGHT / math.sqrt(eps_bounds[1])
    v_hi = SPEED_OF_LIGHT / math.sqrt(eps_bounds[0])
    sw = np.sqrt(w / w.max())
    # scale: time in µs, x in km, v in km/µs → well-conditioned
    xk, tu = x / 1e3, t * 1e6

    def resid(p, xk, tu, sw):
        v_kmus, t0, x0 = p
        return sw * (np.sqrt(t0 ** 2 + (2.0 * (xk - x0) / v_kmus) ** 2) - tu)

    v0 = SPEED_OF_LIGHT / math.sqrt(3.0) / 1e9  # km/µs
    p0 = [v0, float(np.min(tu)), float(xk[np.argmin(tu)]) if fit_x0 else 0.0]
    lo = [v_lo / 1e9, 0.0, (xk.min() if fit_x0 else -1e-9)]
    hi = [v_hi / 1e9, max(float(tu.max()) * 2, 1e-3), (xk.max() if fit_x0 else 1e-9)]
    p0 = list(np.clip(p0, lo, hi))

    def _solve(xk, tu, sw, p0):
        return least_squares(resid, p0, args=(xk, tu, sw), bounds=(lo, hi), x_scale="jac")

    r = _solve(xk, tu, sw, p0)
    res = r.fun / np.maximum(sw, 1e-12)
    sd = np.std(res)
    keep = np.abs(res) <= max(2.5 * sd, 1e-9)
    if keep.sum() >= 5 and keep.sum() < len(keep):
        xk, tu, sw = xk[keep], tu[keep], sw[keep]
        r = _solve(xk, tu, sw, r.x)
        res = r.fun / np.maximum(sw, 1e-12)

    v_kmus, t0_us, x0_km = (float(v) for v in r.x)
    v = v_kmus * 1e9
    eps = (SPEED_OF_LIGHT / v) ** 2
    rmse_bins = float(np.sqrt(np.mean(res ** 2)) / SHARAD_SAMPLE_INTERVAL_US)
    eps_std = None
    ci = None
    try:
        J = r.jac
        dof = max(len(tu) - 3, 1)
        s2 = float((r.fun ** 2).sum()) / dof
        cov = np.linalg.pinv(J.T @ J) * s2
        v_std = math.sqrt(max(cov[0, 0], 0.0)) * 1e9
        eps_std = abs(-2.0 * SPEED_OF_LIGHT ** 2 / v ** 3) * v_std
        ci = (max(eps_bounds[0], eps - 1.96 * eps_std), min(eps_bounds[1], eps + 1.96 * eps_std))
    except Exception:  # pragma: no cover - degenerate jacobian
        pass

    at_bound = (abs(v - v_lo) / v_lo < 1e-3) or (abs(v - v_hi) / v_hi < 1e-3)
    note = "fit at εr bound — unconstrained" if at_bound else "ok"
    return HyperbolaFit(
        v_mps=v,
        t0_s=t0_us * 1e-6,
        x0_m=x0_km * 1e3,
        epsilon_r=eps,
        epsilon_r_std=eps_std,
        epsilon_r_ci95=ci,
        depth_m=v * t0_us * 1e-6 / 2.0,
        rmse_bins=rmse_bins,
        n_used=int(len(tu)),
        note=note,
    )


# ─────────────────────────────────────────────────────────────────────
# RDR file helpers
# ─────────────────────────────────────────────────────────────────────

class IncompleteProductError(Exception):
    """Raised when a SHARAD .dat is still downloading or truncated."""


def label_file_records(lbl_path: str) -> Optional[int]:
    """FILE_RECORDS from a PDS3 label, or None."""
    if not lbl_path or not os.path.exists(lbl_path):
        return None
    with open(lbl_path, "r", errors="replace") as f:
        for line in f:
            m = re.match(r"\s*FILE_RECORDS\s*=\s*(\d+)", line)
            if m:
                return int(m.group(1))
    return None


def check_dat_complete(dat_path: str, lbl_path: Optional[str], row_bytes: int = ROW_BYTES) -> int:
    """Return the number of complete rows, or raise IncompleteProductError.

    Incomplete = an aria2 control file (``<dat>.aria2``) still exists (aria2
    pre-allocates the full size, so size alone cannot detect it), or the file
    is shorter than FILE_RECORDS × row_bytes from the label.
    """
    if os.path.exists(dat_path + ".aria2"):
        raise IncompleteProductError(
            f"{os.path.basename(dat_path)} is still downloading (aria2 control file present); "
            "re-try once the download has finished."
        )
    size = os.path.getsize(dat_path)
    expected = label_file_records(lbl_path) if lbl_path else None
    if expected is not None and size < expected * row_bytes:
        pct = 100.0 * size / (expected * row_bytes)
        raise IncompleteProductError(
            f"{os.path.basename(dat_path)} is truncated: {size} bytes, label expects "
            f"{expected} rows × {row_bytes} B ({pct:.0f}% present). Re-download the product."
        )
    return expected if expected is not None else size // row_bytes


def source_signature(dat_path: str) -> dict:
    """Identity of the source file used to key derived caches."""
    st = os.stat(dat_path)
    return {"size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}


def read_rwot(dat_path: str, n_rows: int, row_bytes: int = ROW_BYTES) -> np.ndarray:
    """RECEIVE_WINDOW_OPENING_TIME per row (float32, units of 37.5 ns)."""
    mm = np.memmap(dat_path, dtype=np.uint8, mode="r")
    n = min(n_rows, mm.size // row_bytes)
    rows = mm[: n * row_bytes].reshape(n, row_bytes)
    return np.frombuffer(
        np.ascontiguousarray(rows[:, RWOT_OFFSET:RWOT_OFFSET + 4]).tobytes(), dtype="<f4"
    ).astype(np.float64)


def predicted_surface_bins(altitude_m, elevation_m, rwot) -> np.ndarray:
    """Geometric (MOLA) surface position in absolute delay bins, before the
    per-product calibration offset: 2(h − z)/c / 37.5 ns − RWOT."""
    alt = np.asarray(altitude_m, dtype=np.float64)
    z = np.asarray(elevation_m, dtype=np.float64)
    return 2.0 * (alt - z) / SPEED_OF_LIGHT / BIN_DT_S - np.asarray(rwot, dtype=np.float64)


def robust_linear_offset(y, x=None, n_iter: int = 5, k: float = 3.0) -> tuple[np.ndarray, float, int]:
    """Robust (MAD-clipped) linear fit of y against x.

    Returns (coef [slope, intercept], residual MAD, n_used).  Used to calibrate
    the geometric surface prediction against the picked surface: the constant
    absorbs the instrument delay (≈ one PRI) and the slope absorbs the
    spacecraft-altitude vs. DEM datum drift along track.
    """
    y = np.asarray(y, dtype=np.float64)
    x = np.arange(len(y), dtype=np.float64) if x is None else np.asarray(x, dtype=np.float64)
    m = np.isfinite(y) & np.isfinite(x)
    if m.sum() < 2:
        return np.array([0.0, float(np.nanmedian(y)) if m.any() else 0.0]), float("nan"), int(m.sum())
    for _ in range(n_iter):
        coef = np.polyfit(x[m], y[m], 1)
        r = y - np.polyval(coef, x)
        mad = float(np.median(np.abs(r[m] - np.median(r[m])))) or 1e-9
        new = np.isfinite(y) & np.isfinite(x) & (np.abs(r) <= k * 1.4826 * mad)
        if new.sum() < 2 or (new == m).all():
            m = new if new.sum() >= 2 else m
            break
        m = new
    coef = np.polyfit(x[m], y[m], 1)
    r = y[m] - np.polyval(coef, x[m])
    return coef, float(np.median(np.abs(r - np.median(r)))), int(m.sum())
