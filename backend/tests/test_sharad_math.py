"""Physics/maths of the SHARAD viewer (analysis/shared/sharad_math.py)."""

import math

import numpy as np
import pytest

from analysis.shared.constants import SPEED_OF_LIGHT
from analysis.shared.sharad_math import (
    BIN_DT_S,
    depth_from_twt,
    depth_resolution,
    fit_attenuation,
    fit_nmo_hyperbola,
    nmo_twt,
    orbital_moveout_twt,
    pick_surface_leading_edge,
    predicted_surface_bins,
    reject_pick_outliers,
    robust_linear_offset,
)

C = SPEED_OF_LIGHT


# ── depth conversion ─────────────────────────────────────────────────

def test_uniform_depth():
    t = 100 * BIN_DT_S
    assert depth_from_twt(t, 3.1) == pytest.approx(C * t / (2 * math.sqrt(3.1)))
    # 1 bin at εr=1 is 5.62 m
    assert depth_from_twt(BIN_DT_S, 1.0) == pytest.approx(5.6211, abs=1e-3)


def test_two_layer_depth():
    eps1, eps2, zb = 3.0, 8.0, 40.0
    t_b = 2 * zb * math.sqrt(eps1) / C
    assert depth_from_twt(t_b, eps1, eps2, zb) == pytest.approx(zb)
    # 1 µs extra in layer 2
    t = t_b + 1e-6
    assert depth_from_twt(t, eps1, eps2, zb) == pytest.approx(zb + C * 1e-6 / (2 * math.sqrt(eps2)))
    # above the boundary it equals the uniform layer-1 answer
    assert depth_from_twt(t_b / 2, eps1, eps2, zb) == pytest.approx(depth_from_twt(t_b / 2, eps1))
    # equal εr or no boundary → uniform
    assert depth_from_twt(t, eps1, eps1, zb) == pytest.approx(depth_from_twt(t, eps1))
    assert depth_from_twt(t, eps1, eps2, 0.0) == pytest.approx(depth_from_twt(t, eps1))
    # vectorised & negative clipped
    arr = depth_from_twt(np.array([-1e-6, 0.0, t]), eps1, eps2, zb)
    assert arr[0] == 0.0 and arr[1] == 0.0 and arr[2] == pytest.approx(depth_from_twt(t, eps1, eps2, zb))


def test_depth_resolution():
    r = depth_resolution(1e-6, 4.0)
    assert r["bin_uncertainty_m"] == pytest.approx(C * BIN_DT_S / 4)
    assert r["vertical_resolution_m"] == pytest.approx(C / 2e7 / 2)
    # below the boundary the layer-2 velocity is used
    r2 = depth_resolution(10e-6, 3.0, 9.0, 10.0)
    assert r2["epsilon_r_at_depth"] == 9.0
    assert r2["bin_uncertainty_m"] == pytest.approx(C * BIN_DT_S / (2 * 3.0))


# ── surface pick ─────────────────────────────────────────────────────

def _synthetic_traces(n=200, surf=180, seed=1):
    rng = np.random.default_rng(seed)
    p = rng.exponential(1.0, size=(n, 667))
    s = surf + (np.arange(n) // 20)          # gently sloping surface
    for i in range(n):
        p[i, s[i]:s[i] + 3] += [300.0, 600.0, 200.0]   # leading edge + peak
        p[i, s[i] + 60] += 3000.0                      # STRONGER later echo (clutter)
    return p, s


def test_leading_edge_beats_strongest_return():
    p, s = _synthetic_traces()
    bins, snr = pick_surface_leading_edge(p)
    assert (bins >= 0).all()
    # picks the first strong return (peak at s+1), not the brighter s+60 echo
    assert np.abs(bins - (s + 1)).max() <= 1
    assert (snr > 20).all()
    # the old "strongest echo" choice would be 60 bins too deep
    assert np.abs(np.argmax(p, axis=1) - (s + 60)).max() == 0


def test_no_surface_when_only_noise():
    rng = np.random.default_rng(2)
    p = rng.exponential(1.0, size=(50, 667))
    bins, snr = pick_surface_leading_edge(p, min_snr_db=15)
    assert (bins == -1).all() and (snr == 0).all()


def test_outlier_rejection():
    b = np.full(100, 200, np.int32)
    b[50] = 400
    out = reject_pick_outliers(b, kernel=11, max_dev=20)
    assert out[50] == -1 and (out[:50] == 200).all()


# ── attenuation regression ───────────────────────────────────────────

def test_attenuation_regression_recovers_alpha():
    rng = np.random.default_rng(3)
    depth = rng.uniform(10, 200, 400)
    alpha = 0.03                                  # one-way dB/m
    rel = -7.0 - 2 * alpha * depth + rng.normal(0, 0.5, depth.size)
    f = fit_attenuation(depth, rel)
    assert f.alpha_dBm == pytest.approx(alpha, abs=0.002)
    assert f.alpha_se_dBm < 0.002
    assert f.r2 > 0.9
    assert f.intercept_dB == pytest.approx(-7.0, abs=0.3)
    assert f.n == 400


def test_attenuation_regression_needs_depth_span():
    f = fit_attenuation([50, 50.5, 51, 50.2, 50.1], [-10, -11, -9, -10, -10])
    assert f.alpha_dBm is None
    f = fit_attenuation([10, 20], [-1, -2])
    assert f.alpha_dBm is None


# ── hyperbola ────────────────────────────────────────────────────────

@pytest.mark.parametrize("eps", [2.5, 4.0, 7.0])
def test_nmo_hyperbola_recovery(eps):
    v = C / math.sqrt(eps)
    depth = 60.0
    t0 = 2 * depth / v
    x = np.arange(-60, 61) * 18.1                 # SS19 trace spacing
    rng = np.random.default_rng(4)
    t = nmo_twt(x - 30.0, t0, v) + rng.normal(0, 0.05 * BIN_DT_S, x.size)
    f = fit_nmo_hyperbola(x, t)
    assert f.epsilon_r == pytest.approx(eps, rel=0.03)
    assert f.depth_m == pytest.approx(depth, rel=0.03)
    assert f.x0_m == pytest.approx(30.0, abs=10)
    lo, hi = f.epsilon_r_ci95
    assert lo <= eps <= hi or abs(f.epsilon_r - eps) / eps < 0.03
    assert f.note == "ok"


def test_hyperbola_rejects_outliers_and_few_picks():
    v = C / math.sqrt(3.0)
    x = np.arange(-40, 41) * 18.1
    t = nmo_twt(x, 0.5e-6, v)
    t[::10] += 30 * BIN_DT_S                     # gross outliers
    f = fit_nmo_hyperbola(x, t)
    assert f.epsilon_r == pytest.approx(3.0, rel=0.05)
    assert fit_nmo_hyperbola(x[:3], t[:3]).epsilon_r is None


def test_orbital_moveout_is_tiny_and_eps_independent():
    # 1 km offset from 270 km altitude: ~0.33 bins of free-space moveout
    dt = orbital_moveout_twt(1000.0, 270e3)
    assert dt / BIN_DT_S == pytest.approx(0.33, abs=0.02)


# ── geometry / receive window ────────────────────────────────────────

def test_predicted_surface_bins_and_calibration():
    n = 500
    alt = np.linspace(286e3, 266e3, n)
    elev = 1000 * np.sin(np.linspace(0, 6, n))
    # instrument: RWOT tracks range with a 1-PRI offset; surface ~ bin 200
    true_bins = 2 * (alt - elev) / C / BIN_DT_S
    k = 38080.0
    rwot = np.round(true_bins - k - 200)
    pick = true_bins - rwot - k + np.random.default_rng(5).normal(0, 1, n)
    pred = predicted_surface_bins(alt, elev, rwot)
    coef, mad, used = robust_linear_offset(pred - pick)
    assert coef[1] == pytest.approx(k, abs=1)
    assert mad < 2 and used > 400
    aligned = pred - np.polyval(coef, np.arange(n))
    assert np.median(np.abs(aligned - pick)) < 1.5


def test_windowed_attenuation_matches_global_fit_and_localises():
    from analysis.shared.sharad_math import windowed_attenuation
    rng = np.random.default_rng(6)
    n = 2000
    depth = rng.uniform(10, 150, n)
    alpha = np.where(np.arange(n) < 1000, 0.01, 0.05)      # two units along track
    rel = -5 - 2 * alpha * depth + rng.normal(0, 0.3, n)
    valid = rng.uniform(size=n) > 0.2
    w = windowed_attenuation(depth, rel, valid, half_window=150)
    assert np.nanmedian(w["alpha_dBm"][200:800]) == pytest.approx(0.01, abs=0.002)
    assert np.nanmedian(w["alpha_dBm"][1200:1800]) == pytest.approx(0.05, abs=0.002)
    # window at i equals a direct fit of the same points
    i = 500
    sl = slice(i - 150, i + 151)
    f = fit_attenuation(depth[sl][valid[sl]], rel[sl][valid[sl]])
    assert w["alpha_dBm"][i] == pytest.approx(f.alpha_dBm, rel=1e-6)
    assert w["alpha_se_dBm"][i] == pytest.approx(f.alpha_se_dBm, rel=1e-6)
    assert w["r2"][i] == pytest.approx(f.r2, rel=1e-6)
