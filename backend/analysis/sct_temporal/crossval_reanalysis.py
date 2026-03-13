#!/usr/bin/env python3
"""Phase 3B — Multi-Pair Cross-Validation with Independent Classification.

Re-analyzes all successfully processed pairs using displacement-INDEPENDENT
terrain classification, eliminating the circular bias of the two-pass approach.

Classification strategies:
  - Pair 0: MarsLandformNet (original, gold standard)
  - Pair 1: MarsLandformNet mapped via shared base image (PSP_007173_2245)
  - Others: Calibrated gradient-percentile (target ~40% scarp fraction)

Key test: if scarp retreat is real, excess displacement should be
PROPORTIONAL to temporal baseline. If it's artifact, excess will be
constant regardless of gap.

Usage:
    python -m backend.analysis.sct_temporal.crossval_reanalysis
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from scipy import ndimage, stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = PROJECT_ROOT / "results" / "sct_analysis" / "phase3"
CHIP_SIZE = 256
STEP_SIZE = 128

# Target scarp fraction — calibrated to match original MLN analysis (48.3%)
TARGET_SCARP_PCT = 40  # top N% of gradient → scarp
TARGET_STABLE_PCT = 25  # bottom N% of gradient → stable


# ── Classification Methods ───────────────────────────────────────────

def _load_mln_classification(
    row_centers: np.ndarray, col_centers: np.ndarray
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Load MarsLandformNet classification for Pair 0."""
    npz_path = PROJECT_ROOT / "results" / "sct_analysis" / "affine_corrected_displacement.npz"
    if not npz_path.exists():
        return None, None

    data = np.load(npz_path)
    x_orig = data["x"]
    y_orig = data["y"]
    scarp_orig = data["is_scarp"].astype(bool)
    stable_orig = data["is_stable"].astype(bool)

    ux = np.unique(x_orig)
    uy = np.unique(y_orig)
    nx, ny = len(ux), len(uy)

    scarp_2d = np.zeros((ny, nx), dtype=bool)
    stable_2d = np.zeros((ny, nx), dtype=bool)
    for idx in range(len(x_orig)):
        ci = np.searchsorted(ux, x_orig[idx])
        ri = np.searchsorted(uy, y_orig[idx])
        if 0 <= ri < ny and 0 <= ci < nx:
            scarp_2d[ri, ci] = scarp_orig[idx]
            stable_2d[ri, ci] = stable_orig[idx]

    nr, nc = len(row_centers), len(col_centers)
    is_scarp = np.zeros((nr, nc), dtype=bool)
    is_stable = np.zeros((nr, nc), dtype=bool)
    matched = 0
    for i, r in enumerate(row_centers):
        for j, c in enumerate(col_centers):
            ri = np.searchsorted(uy, r)
            ci = np.searchsorted(ux, c)
            if (0 <= ri < ny and 0 <= ci < nx
                    and abs(uy[ri] - r) < 1 and abs(ux[ci] - c) < 1):
                is_scarp[i, j] = scarp_2d[ri, ci]
                is_stable[i, j] = stable_2d[ri, ci]
                matched += 1

    if matched < 100:
        return None, None
    return is_stable, is_scarp


def _gradient_percentile_classify(
    img: np.ndarray,
    row_centers: np.ndarray,
    col_centers: np.ndarray,
    chip_size: int = CHIP_SIZE,
    scarp_pct: float = TARGET_SCARP_PCT,
    stable_pct: float = TARGET_STABLE_PCT,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Classify chips by image gradient magnitude using percentile thresholds.

    Independent of displacement — uses only image morphology.
    Scarp: chips whose mean gradient is in top scarp_pct%.
    Stable: chips whose mean gradient is in bottom stable_pct%.
    """
    img_f = img.astype(np.float64)
    if img_f.max() > img_f.min():
        img_f = (img_f - img_f.min()) / (img_f.max() - img_f.min())

    gy = ndimage.sobel(img_f, axis=0)
    gx = ndimage.sobel(img_f, axis=1)
    grad_mag = np.sqrt(gx**2 + gy**2)

    half = chip_size // 2
    H, W = img.shape
    nr, nc = len(row_centers), len(col_centers)
    chip_grad = np.zeros((nr, nc), dtype=np.float64)

    for i, r in enumerate(row_centers):
        for j, c in enumerate(col_centers):
            r0 = max(0, r - half)
            r1 = min(H, r + half)
            c0 = max(0, c - half)
            c1 = min(W, c + half)
            chip_grad[i, j] = grad_mag[r0:r1, c0:c1].mean()

    # Percentile-based thresholds
    flat = chip_grad.ravel()
    scarp_threshold = np.percentile(flat, 100 - scarp_pct)
    stable_threshold = np.percentile(flat, stable_pct)

    is_scarp = chip_grad >= scarp_threshold
    is_stable = chip_grad <= stable_threshold

    return is_stable, is_scarp


# ── Re-analysis ──────────────────────────────────────────────────────

def reanalyze_pair(pair_idx: int, pair_dir: Path) -> Optional[dict]:
    """Re-analyze one pair using independent classification."""
    disp_path = pair_dir / "displacement.npz"
    result_path = pair_dir / "pair_result.json"

    if not disp_path.exists() or not result_path.exists():
        return None

    with open(result_path) as f:
        orig_result = json.load(f)

    if not orig_result.get("success"):
        return None

    data = np.load(disp_path)
    row_centers = data["row_centers"]
    col_centers = data["col_centers"]
    mag_m = data["magnitude_m"]
    valid = data["valid_mask"]

    # Load co-registered image for gradient classification
    # We need the image — it's not saved in displacement.npz
    # Instead, re-open the JP2s and coregister (cached, fast)
    pair_name = orig_result["pair_name"]
    pid_a = orig_result["pid_a"]
    pid_b = orig_result["pid_b"]
    gap = orig_result["gap_mars_yr"]
    score = orig_result["score"]

    # Try to get the coregistered image
    img = _load_coregistered_image(pid_a, pid_b)
    if img is None:
        logger.warning(f"[Pair {pair_idx}] Cannot load coregistered image, skipping")
        return None

    # Choose classification method
    if pair_idx == 0:
        method = "MarsLandformNet"
        is_stable, is_scarp = _load_mln_classification(row_centers, col_centers)
        if is_stable is None:
            logger.warning(f"[Pair {pair_idx}] MLN classification failed, falling back to gradient")
            method = "gradient-percentile"
            is_stable, is_scarp = _gradient_percentile_classify(
                img, row_centers, col_centers
            )
    elif pair_idx == 1:
        # Pair 1 shares PSP_007173_2245 — try MLN mapping
        method = "MarsLandformNet-mapped"
        is_stable, is_scarp = _load_mln_classification(row_centers, col_centers)
        if is_stable is None:
            method = "gradient-percentile"
            is_stable, is_scarp = _gradient_percentile_classify(
                img, row_centers, col_centers
            )
    else:
        method = "gradient-percentile"
        is_stable, is_scarp = _gradient_percentile_classify(
            img, row_centers, col_centers
        )

    # Compute statistics
    stable_valid = valid & is_stable
    scarp_valid = valid & is_scarp
    n_stable = int(stable_valid.sum())
    n_scarp = int(scarp_valid.sum())

    logger.info(
        f"[Pair {pair_idx}] {method}: {n_scarp} scarp, {n_stable} stable "
        f"({n_scarp / max(valid.sum(), 1) * 100:.1f}% scarp)"
    )

    if n_stable < 10 or n_scarp < 10:
        logger.warning(f"[Pair {pair_idx}] Too few chips: stable={n_stable}, scarp={n_scarp}")
        return None

    stable_mags = mag_m[stable_valid]
    scarp_mags = mag_m[scarp_valid]

    excess = float(np.mean(scarp_mags) - np.mean(stable_mags))
    stable_noise = float(np.std(stable_mags))
    snr = excess / (2 * stable_noise) if stable_noise > 0 else 0

    try:
        _, mw_p = stats.mannwhitneyu(scarp_mags, stable_mags, alternative="greater")
        mw_p = float(mw_p)
    except ValueError:
        mw_p = 1.0

    rate = excess / gap if gap > 0 else 0

    return {
        "pair_idx": pair_idx,
        "pair_name": pair_name,
        "pid_a": pid_a,
        "pid_b": pid_b,
        "gap_mars_yr": gap,
        "score": score,
        "classification_method": method,
        "n_stable": n_stable,
        "n_scarp": n_scarp,
        "scarp_fraction_pct": round(n_scarp / max(valid.sum(), 1) * 100, 1),
        "stable_mean_m": round(float(np.mean(stable_mags)), 4),
        "scarp_mean_m": round(float(np.mean(scarp_mags)), 4),
        "excess_m": round(excess, 4),
        "stable_noise_m": round(stable_noise, 4),
        "snr": round(snr, 3),
        "mann_whitney_p": mw_p,
        "raw_rate_m_per_mars_yr": round(rate, 6),
        "calibrated_rate_cm_per_mars_yr": round(rate * 2.38 * 100, 2),
    }


def _load_coregistered_image(pid_a: str, pid_b: str) -> Optional[np.ndarray]:
    """Load and coregister the image pair (uses cache)."""
    from .coregistration import coregister_geotiffs

    cache_dir = PROJECT_ROOT / "Data" / "HiRISE" / "rdr_cache"
    path_a = cache_dir / f"{pid_a}_RED.JP2"
    path_b = cache_dir / f"{pid_b}_RED.JP2"

    if not path_a.exists() or not path_b.exists():
        return None

    try:
        coreg = coregister_geotiffs(path_a, path_b)
        return coreg.img1
    except Exception as e:
        logger.error(f"Coregistration failed: {e}")
        return None


# ── Main ─────────────────────────────────────────────────────────────

def main():
    results = []

    for pair_idx in range(10):
        pair_dir = RESULTS_DIR / f"pair_{pair_idx:02d}"
        result = reanalyze_pair(pair_idx, pair_dir)
        if result is not None:
            results.append(result)

    if not results:
        logger.error("No pairs successfully re-analyzed")
        return

    # ── Proportionality Test ─────────────────────────────────────────
    # Key question: is excess proportional to gap?
    # If real retreat: excess = rate × gap → excess ∝ gap
    # If fixed artifact: excess = const → excess independent of gap

    gaps = np.array([r["gap_mars_yr"] for r in results])
    excesses = np.array([r["excess_m"] for r in results])
    rates = np.array([r["raw_rate_m_per_mars_yr"] for r in results])

    # Linear regression: excess = a × gap + b
    if len(results) >= 3:
        slope, intercept, r_value, p_value, std_err = stats.linregress(gaps, excesses)
        r_sq = r_value ** 2
    else:
        slope = intercept = r_value = p_value = std_err = r_sq = float("nan")

    # Rate consistency: if real, all rates should be similar
    rate_mean = float(np.mean(rates))
    rate_std = float(np.std(rates, ddof=1)) if len(rates) > 1 else 0
    rate_cv = rate_std / rate_mean if rate_mean > 0 else float("inf")

    proportionality = {
        "description": "Test whether excess displacement scales with temporal baseline",
        "n_pairs": len(results),
        "linear_fit": {
            "slope_m_per_mars_yr": round(slope, 4),
            "intercept_m": round(intercept, 4),
            "r_squared": round(r_sq, 4),
            "p_value": f"{p_value:.2e}" if not np.isnan(p_value) else "N/A",
        },
        "rate_consistency": {
            "mean_rate_m_per_mars_yr": round(rate_mean, 4),
            "std_rate_m_per_mars_yr": round(rate_std, 4),
            "coefficient_of_variation": round(rate_cv, 3),
            "interpretation": (
                "CONSISTENT (CV < 0.5)" if rate_cv < 0.5
                else "MODERATE (0.5 < CV < 1.0)" if rate_cv < 1.0
                else "INCONSISTENT (CV > 1.0)"
            ),
        },
        "verdict": (
            "SUPPORTS real retreat" if r_sq > 0.5 and rate_cv < 1.0
            else "INCONCLUSIVE" if r_sq > 0.2 or rate_cv < 1.5
            else "DOES NOT SUPPORT real retreat"
        ),
    }

    # ── Summary table ────────────────────────────────────────────────
    print("\n" + "=" * 120)
    print("CROSS-VALIDATION: Independent Classification Re-Analysis")
    print("=" * 120)
    print(f"{'#':>2} {'Pair':<42} {'Gap':>6} {'Method':<22} {'Scarp%':>7} "
          f"{'Excess':>8} {'Rate':>10} {'SNR':>6} {'p-value':>10}")
    print("-" * 120)

    for r in results:
        print(
            f"{r['pair_idx']:>2} {r['pair_name']:<42} "
            f"{r['gap_mars_yr']:>6.2f} {r['classification_method']:<22} "
            f"{r['scarp_fraction_pct']:>6.1f}% "
            f"{r['excess_m']:>7.3f}m "
            f"{r['raw_rate_m_per_mars_yr']:>9.4f} "
            f"{r['snr']:>6.3f} "
            f"{r['mann_whitney_p']:>10.2e}"
        )

    print("=" * 120)
    print(f"\nPROPORTIONALITY TEST (excess ∝ gap?):")
    print(f"  Linear fit: excess = {slope:.4f} × gap + {intercept:.4f}")
    print(f"  R² = {r_sq:.4f}, p = {p_value:.2e}" if not np.isnan(p_value) else "  R² = N/A")
    print(f"  Rate consistency: mean={rate_mean:.4f} ± {rate_std:.4f} m/My (CV={rate_cv:.3f})")
    print(f"  Verdict: {proportionality['verdict']}")

    cal = 2.38
    print(f"\n  Calibrated mean retreat rate: {rate_mean * cal * 100:.2f} ± {rate_std * cal * 100:.2f} cm/Mars-yr")

    # ── Save ─────────────────────────────────────────────────────────
    report = {
        "description": "Phase 3B — Cross-Validation with Independent Classification",
        "classification_methods": {
            "pair_0": "MarsLandformNet (gold standard)",
            "pair_1": "MarsLandformNet mapped via shared base image",
            "others": f"Gradient-percentile (top {TARGET_SCARP_PCT}% scarp, bottom {TARGET_STABLE_PCT}% stable)",
        },
        "pair_results": results,
        "proportionality_test": proportionality,
        "calibrated_mean_rate_cm_per_mars_yr": round(rate_mean * cal * 100, 2),
        "calibrated_std_cm_per_mars_yr": round(rate_std * cal * 100, 2),
    }

    out_path = RESULTS_DIR / "crossval_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Report saved: {out_path}")


if __name__ == "__main__":
    main()
