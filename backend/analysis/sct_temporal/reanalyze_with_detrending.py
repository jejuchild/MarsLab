#!/usr/bin/env python3
"""
Phase 2: Re-analyze existing SCT temporal pair with polynomial detrending.

Loads the existing displacement data and applies polynomial detrending
to assess improvement in stable terrain noise and scarp signal.

Also re-runs phase correlation from scratch if original images are available.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
from scipy import stats

from .phase_correlation import (
    _build_polynomial_matrix,
    _robust_polyfit,
    _min_points_for_order,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parents[3] / "results" / "sct_analysis"
BASELINE_MARS_YR = 8.01


def reanalyze_existing_displacement(
    results_dir: Path = RESULTS_DIR,
    detrend_orders: list[int] | None = None,
) -> dict:
    """
    Apply polynomial detrending to the existing affine-corrected displacement field.

    Loads the saved NPZ data and applies polynomial detrending at multiple orders,
    comparing noise reduction and signal preservation.
    """
    if detrend_orders is None:
        detrend_orders = [0, 1, 2, 3]

    # Load existing data
    npz_path = results_dir / "affine_corrected_displacement.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"No displacement data at {npz_path}")

    data = np.load(npz_path)
    x = data["x"]  # pixel x coordinates (col)
    y = data["y"]  # pixel y coordinates (row)
    dx = data["corr_dx_m"]  # affine-corrected dx in meters
    dy = data["corr_dy_m"]  # affine-corrected dy in meters
    ncc = data["ncc"]
    is_scarp = data["is_scarp"]
    is_stable = data["is_stable"]

    logger.info(f"Loaded {len(x)} chips from {npz_path.name}")

    # Filter by NCC quality
    good = ncc > 0.5
    logger.info(f"Good chips (NCC > 0.5): {good.sum()} / {len(good)}")

    results = {}

    for order in detrend_orders:
        logger.info(f"\n{'='*60}")
        logger.info(f"Detrend order = {order}")
        logger.info(f"{'='*60}")

        result = _apply_detrend_to_flat_arrays(
            x, y, dx, dy, ncc, is_scarp, is_stable, good, order
        )
        results[f"order_{order}"] = result

    # Comparison summary
    comparison = _build_comparison(results, detrend_orders)

    # Save report
    output_path = results_dir / "polynomial_detrending_report.json"
    report = {
        "description": "Polynomial detrending comparison for SCT temporal pair",
        "pair": "PSP_007173_2245 → ESP_077815_2245",
        "baseline_mars_yr": BASELINE_MARS_YR,
        "input_data": str(npz_path),
        "detrend_results": {k: v for k, v in results.items()},
        "comparison": comparison,
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"\nReport saved: {output_path}")

    # Print comparison table
    _print_comparison_table(results, detrend_orders)

    return report


def _apply_detrend_to_flat_arrays(
    x: np.ndarray,
    y: np.ndarray,
    dx_m: np.ndarray,
    dy_m: np.ndarray,
    ncc: np.ndarray,
    is_scarp: np.ndarray,
    is_stable: np.ndarray,
    good: np.ndarray,
    order: int,
) -> dict:
    """Apply polynomial detrending to flat arrays of displacement measurements."""

    # Normalize coordinates
    x_mean, x_std = x.mean(), max(x.std(), 1.0)
    y_mean, y_std = y.mean(), max(y.std(), 1.0)
    x_norm = (x - x_mean) / x_std
    y_norm = (y - y_mean) / y_std

    # Stable terrain for fitting
    stable_good = is_stable & good
    n_stable = int(stable_good.sum())

    if order == 0:
        # Median subtraction (baseline)
        dx_corr = dx_m.copy()
        dy_corr = dy_m.copy()
        if n_stable > 0:
            dx_corr -= np.median(dx_m[stable_good])
            dy_corr -= np.median(dy_m[stable_good])
        model_info = {"type": "median", "n_stable": n_stable}
    else:
        n_coeffs = (order + 1) * (order + 2) // 2
        min_pts = _min_points_for_order(order)

        if n_stable < min_pts:
            logger.warning(
                f"Only {n_stable} stable points, need {min_pts} for order={order}. "
                "Falling back to median."
            )
            return _apply_detrend_to_flat_arrays(
                x, y, dx_m, dy_m, ncc, is_scarp, is_stable, good, order=0
            )

        # Build design matrix for stable terrain
        A_stable = _build_polynomial_matrix(
            x_norm[stable_good], y_norm[stable_good], order
        )

        # Robust fit
        dx_coeffs = _robust_polyfit(A_stable, dx_m[stable_good])
        dy_coeffs = _robust_polyfit(A_stable, dy_m[stable_good])

        # Evaluate model at all good chip positions
        A_all = _build_polynomial_matrix(x_norm[good], y_norm[good], order)
        dx_model = A_all @ dx_coeffs
        dy_model = A_all @ dy_coeffs

        # Subtract model
        dx_corr = dx_m.copy()
        dy_corr = dy_m.copy()
        dx_corr[good] -= dx_model
        dy_corr[good] -= dy_model

        model_info = {
            "type": f"polynomial_order_{order}",
            "n_stable": n_stable,
            "n_coefficients": n_coeffs,
            "dx_coeffs": dx_coeffs.tolist(),
            "dy_coeffs": dy_coeffs.tolist(),
        }

    # Compute statistics
    mag_corr = np.sqrt(dx_corr**2 + dy_corr**2)
    scarp_good = is_scarp & good
    other_good = ~is_scarp & ~is_stable & good

    stable_stats = _compute_terrain_stats(dx_corr, dy_corr, mag_corr, stable_good)
    scarp_stats = _compute_terrain_stats(dx_corr, dy_corr, mag_corr, scarp_good)
    other_stats = _compute_terrain_stats(dx_corr, dy_corr, mag_corr, other_good)

    # Statistical test: scarp vs stable
    if stable_good.sum() > 5 and scarp_good.sum() > 5:
        u_stat, u_p = stats.mannwhitneyu(
            mag_corr[scarp_good], mag_corr[stable_good], alternative="greater"
        )
        ks_stat, ks_p = stats.ks_2samp(mag_corr[scarp_good], mag_corr[stable_good])
    else:
        u_stat, u_p, ks_stat, ks_p = 0, 1, 0, 1

    excess_m = scarp_stats["mean_mag_m"] - stable_stats["mean_mag_m"]
    retreat_rate = excess_m / BASELINE_MARS_YR if BASELINE_MARS_YR > 0 else 0

    noise_2sigma = stable_stats["std_mag_m"] * 2
    min_detectable = noise_2sigma / BASELINE_MARS_YR if BASELINE_MARS_YR > 0 else 0

    # Fraction of scarp chips above noise
    threshold = stable_stats["mean_mag_m"] + noise_2sigma
    scarp_above_noise = int(np.sum(mag_corr[scarp_good] > threshold))
    scarp_total = int(scarp_good.sum())

    return {
        "model": model_info,
        "stable_terrain": stable_stats,
        "scarp_terrain": scarp_stats,
        "other_terrain": other_stats,
        "statistical_tests": {
            "mann_whitney_u": {"statistic": float(u_stat), "p_value": float(u_p)},
            "ks_test": {"statistic": float(ks_stat), "p_value": float(ks_p)},
        },
        "retreat_measurement": {
            "excess_displacement_m": round(float(excess_m), 4),
            "retreat_rate_m_per_mars_yr": round(float(retreat_rate), 4),
            "retreat_rate_cm_per_mars_yr": round(float(retreat_rate * 100), 2),
            "noise_floor_2sigma_m": round(float(noise_2sigma), 4),
            "min_detectable_rate_m_per_mars_yr": round(float(min_detectable), 4),
            "scarp_chips_above_noise": f"{scarp_above_noise}/{scarp_total}",
            "scarp_above_noise_pct": round(100 * scarp_above_noise / max(scarp_total, 1), 1),
            "signal_to_noise": round(float(excess_m / max(noise_2sigma, 1e-6)), 3),
        },
    }


def _compute_terrain_stats(
    dx: np.ndarray, dy: np.ndarray, mag: np.ndarray, mask: np.ndarray
) -> dict:
    """Compute displacement statistics for a terrain class."""
    n = int(mask.sum())
    if n == 0:
        return {
            "n_chips": 0,
            "mean_dx_m": 0.0, "mean_dy_m": 0.0, "mean_mag_m": 0.0,
            "std_dx_m": 0.0, "std_dy_m": 0.0, "std_mag_m": 0.0,
            "median_mag_m": 0.0, "p95_mag_m": 0.0,
        }
    return {
        "n_chips": n,
        "mean_dx_m": round(float(np.mean(dx[mask])), 4),
        "mean_dy_m": round(float(np.mean(dy[mask])), 4),
        "mean_mag_m": round(float(np.mean(mag[mask])), 4),
        "std_dx_m": round(float(np.std(dx[mask])), 4),
        "std_dy_m": round(float(np.std(dy[mask])), 4),
        "std_mag_m": round(float(np.std(mag[mask])), 4),
        "median_mag_m": round(float(np.median(mag[mask])), 4),
        "p95_mag_m": round(float(np.percentile(mag[mask], 95)), 4),
    }


def _build_comparison(results: dict, orders: list[int]) -> dict:
    """Build a comparison across all detrend orders."""
    rows = []
    for order in orders:
        key = f"order_{order}"
        r = results[key]
        rows.append({
            "order": order,
            "stable_mean_mag_m": r["stable_terrain"]["mean_mag_m"],
            "stable_std_mag_m": r["stable_terrain"]["std_mag_m"],
            "scarp_mean_mag_m": r["scarp_terrain"]["mean_mag_m"],
            "excess_m": r["retreat_measurement"]["excess_displacement_m"],
            "retreat_rate_m_my": r["retreat_measurement"]["retreat_rate_m_per_mars_yr"],
            "noise_2sigma_m": r["retreat_measurement"]["noise_floor_2sigma_m"],
            "snr": r["retreat_measurement"]["signal_to_noise"],
            "mann_whitney_p": r["statistical_tests"]["mann_whitney_u"]["p_value"],
            "above_noise_pct": r["retreat_measurement"]["scarp_above_noise_pct"],
        })
    return {"by_order": rows}


def _print_comparison_table(results: dict, orders: list[int]) -> None:
    """Print formatted comparison."""
    print(f"\n{'='*110}")
    print("POLYNOMIAL DETRENDING COMPARISON — PSP_007173_2245 → ESP_077815_2245")
    print(f"{'='*110}")
    print(
        f"{'Order':<8} {'Stable mag':<14} {'Stable std':<14} {'Scarp mag':<14} "
        f"{'Excess':<12} {'Rate(m/My)':<12} {'Noise 2σ':<12} {'SNR':<8} {'p-value':<12}"
    )
    print(f"{'-'*8} {'-'*14} {'-'*14} {'-'*14} {'-'*12} {'-'*12} {'-'*12} {'-'*8} {'-'*12}")

    for order in orders:
        key = f"order_{order}"
        r = results[key]
        st = r["stable_terrain"]
        sc = r["scarp_terrain"]
        rm = r["retreat_measurement"]
        mw = r["statistical_tests"]["mann_whitney_u"]

        label = "median" if order == 0 else f"poly-{order}"
        print(
            f"{label:<8} {st['mean_mag_m']:<14.4f} {st['std_mag_m']:<14.4f} "
            f"{sc['mean_mag_m']:<14.4f} {rm['excess_displacement_m']:<12.4f} "
            f"{rm['retreat_rate_m_per_mars_yr']:<12.4f} {rm['noise_floor_2sigma_m']:<12.4f} "
            f"{rm['signal_to_noise']:<8.3f} {mw['p_value']:<12.2e}"
        )

    print(f"{'='*110}")

    # Highlight best result
    best_snr = -1
    best_order = 0
    for order in orders:
        snr = results[f"order_{order}"]["retreat_measurement"]["signal_to_noise"]
        if snr > best_snr:
            best_snr = snr
            best_order = order

    print(f"\n→ Best SNR: order={best_order} (SNR={best_snr:.3f})")

    baseline = results["order_0"]["stable_terrain"]["mean_mag_m"]
    best = results[f"order_{best_order}"]["stable_terrain"]["mean_mag_m"]
    if baseline > 0:
        print(f"  Stable noise improvement: {baseline:.4f}m → {best:.4f}m ({baseline/max(best,1e-6):.1f}×)")


def main():
    """Run reanalysis from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Re-analyze SCT pair with polynomial detrending")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--orders", type=int, nargs="+", default=[0, 1, 2, 3])
    args = parser.parse_args()

    reanalyze_existing_displacement(
        results_dir=args.results_dir,
        detrend_orders=args.orders,
    )


if __name__ == "__main__":
    main()
