#!/usr/bin/env python
"""Phase 3 — Multi-Pair SCT Temporal Change Batch Processing.

Downloads all 10 HiRISE temporal pairs, processes each through
phase correlation with order-3 polynomial detrending, and produces
a meta-analysis of scarp retreat rates across all pairs/sites.

Uses a two-pass approach:
  Pass 1: Raw phase correlation (no detrending) to identify displacement
          outliers and build a data-driven stable/scarp classification.
  Pass 2: Polynomial-detrended correlation using only the low-displacement
          chips as the stable reference.

For Pair 0 (the original analysis pair), the MarsLandformNet classification
is loaded from the Phase 2 results to ensure consistency.

Results saved incrementally to: results/sct_analysis/phase3/

Usage:
    python -m backend.analysis.sct_temporal.batch_phase3

Runtime estimate: 4-8 hours (dominated by ~14 GB of HiRISE downloads).
Each pair's result is cached; safe to re-run after interruption.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage, stats

from .hirise_download import download_hirise_rdr
from .coregistration import coregister_geotiffs
from .phase_correlation import sliding_window_correlation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = PROJECT_ROOT / "results" / "sct_analysis" / "phase3"
PAIRS_FILE = PROJECT_ROOT / "results" / "sct_temporal_pairs_v2.json"

# ── Processing Parameters ────────────────────────────────────────────

CHIP_SIZE = 256          # px — larger chips = better SNR per measurement
STEP_SIZE = 128          # px — 50% overlap
UPSAMPLE_FACTOR = 100    # sub-pixel precision: 0.01 px = 2.5 mm @ HiRISE
SNR_THRESHOLD = 3.0      # minimum phase-correlation SNR
DETREND_ORDER = 3        # polynomial detrending order (cubic)

# Two-pass classification parameters
REFERENCE_PERCENTILE = 40   # bottom N% of displacement → "stable" reference
SCARP_MAD_FACTOR = 2.0      # chips above median + factor*MAD → "candidate scarp"


# ── Data Structures ──────────────────────────────────────────────────

@dataclass
class PairResult:
    """Results from processing one temporal pair."""
    pair_idx: int
    pair_name: str
    pid_a: str
    pid_b: str
    gap_mars_yr: float
    score: float

    # Image dimensions
    img_width: int = 0
    img_height: int = 0
    pixel_scale_m: float = 0.0

    # Chip counts
    n_total_chips: int = 0
    n_valid_chips: int = 0
    n_stable_chips: int = 0
    n_scarp_chips: int = 0

    # Displacement statistics (after poly-3 detrending)
    stable_mean_mag_m: float = 0.0
    stable_std_mag_m: float = 0.0
    stable_median_mag_m: float = 0.0
    scarp_mean_mag_m: float = 0.0
    scarp_std_mag_m: float = 0.0
    scarp_median_mag_m: float = 0.0
    excess_m: float = 0.0

    # Retreat rate
    retreat_rate_m_per_mars_yr: float = 0.0
    retreat_rate_cm_per_mars_yr: float = 0.0
    noise_floor_2sigma_m: float = 0.0
    signal_to_noise: float = 0.0

    # Statistical significance
    mann_whitney_p: float = 1.0
    ks_test_p: float = 1.0

    # Status
    success: bool = False
    error: Optional[str] = None
    processing_time_s: float = 0.0


# ── Classification Helpers ──────────────────────────────────────────

def _load_original_classification(
    batch_row_centers: np.ndarray,
    batch_col_centers: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load MarsLandformNet classification for Pair 0 from original analysis.

    The original analysis stored is_scarp/is_stable as flat boolean arrays
    with pixel coordinates (x, y) on a 45x45 grid (128 to 5760, step=128).
    We map these onto the batch grid (which may be larger or differently positioned).

    Returns (is_stable, is_scarp) as 2D arrays matching the batch grid shape,
    or (None, None) if the data cannot be loaded.
    """
    npz_path = PROJECT_ROOT / "results" / "sct_analysis" / "affine_corrected_displacement.npz"
    if not npz_path.exists():
        logger.warning(f"Original classification not found: {npz_path}")
        return None, None

    try:
        data = np.load(npz_path)
        x_orig = data["x"]      # pixel col coords, flat (2025,)
        y_orig = data["y"]      # pixel row coords, flat (2025,)
        scarp_orig = data["is_scarp"].astype(bool)
        stable_orig = data["is_stable"].astype(bool)
    except Exception as e:
        logger.warning(f"Failed to load original classification: {e}")
        return None, None

    # Determine original grid dimensions
    ux = np.unique(x_orig)
    uy = np.unique(y_orig)
    nx_orig, ny_orig = len(ux), len(uy)
    logger.info(
        f"Original classification grid: {ny_orig}x{nx_orig}, "
        f"x=[{ux.min():.0f},{ux.max():.0f}], y=[{uy.min():.0f},{uy.max():.0f}]"
    )

    # Reshape flat arrays to 2D grid (row=y, col=x)
    scarp_2d = np.zeros((ny_orig, nx_orig), dtype=bool)
    stable_2d = np.zeros((ny_orig, nx_orig), dtype=bool)
    for idx in range(len(x_orig)):
        ci = np.searchsorted(ux, x_orig[idx])
        ri = np.searchsorted(uy, y_orig[idx])
        if 0 <= ri < ny_orig and 0 <= ci < nx_orig:
            scarp_2d[ri, ci] = scarp_orig[idx]
            stable_2d[ri, ci] = stable_orig[idx]

    # Map onto batch grid
    nr = len(batch_row_centers)
    nc = len(batch_col_centers)
    is_scarp_batch = np.zeros((nr, nc), dtype=bool)
    is_stable_batch = np.zeros((nr, nc), dtype=bool)

    matched = 0
    for i, r in enumerate(batch_row_centers):
        for j, c in enumerate(batch_col_centers):
            # Find matching position in original grid
            ri = np.searchsorted(uy, r)
            ci = np.searchsorted(ux, c)
            if (0 <= ri < ny_orig and 0 <= ci < nx_orig
                    and abs(uy[ri] - r) < 1 and abs(ux[ci] - c) < 1):
                is_scarp_batch[i, j] = scarp_2d[ri, ci]
                is_stable_batch[i, j] = stable_2d[ri, ci]
                matched += 1

    logger.info(
        f"Mapped {matched}/{nr*nc} batch chips to original classification "
        f"({is_scarp_batch.sum()} scarp, {is_stable_batch.sum()} stable)"
    )

    if matched < 100:
        logger.warning("Too few chips matched — falling back to two-pass")
        return None, None

    return is_stable_batch, is_scarp_batch


def _build_pixel_stable_mask(
    img_shape: Tuple[int, int],
    row_centers: np.ndarray,
    col_centers: np.ndarray,
    is_stable: np.ndarray,
    chip_size: int,
) -> np.ndarray:
    """
    Expand chip-level stable classification to pixel-level mask.

    For each chip classified as stable, mark its pixel footprint as True.
    Used as the stable_mask input for polynomial detrending.
    """
    H, W = img_shape
    mask = np.zeros((H, W), dtype=bool)
    half = chip_size // 2

    for i, r in enumerate(row_centers):
        for j, c in enumerate(col_centers):
            if is_stable[i, j]:
                r0 = max(0, r - half)
                r1 = min(H, r + half)
                c0 = max(0, c - half)
                c1 = min(W, c + half)
                mask[r0:r1, c0:c1] = True

    return mask


def _two_pass_classify(
    raw_mag: np.ndarray,
    valid: np.ndarray,
    ref_percentile: float = REFERENCE_PERCENTILE,
    mad_factor: float = SCARP_MAD_FACTOR,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Data-driven chip classification from raw (undetrended) displacement field.

    Pass 1 classification logic:
      - Stable reference: valid chips with magnitude < ref_percentile of valid magnitudes
      - Candidate scarp: valid chips with magnitude > median + mad_factor * MAD

    Returns (is_stable, is_scarp) boolean arrays matching raw_mag shape.
    """
    valid_mags = raw_mag[valid]
    if len(valid_mags) < 20:
        return np.zeros_like(valid), np.zeros_like(valid)

    threshold_low = np.percentile(valid_mags, ref_percentile)
    median_mag = np.median(valid_mags)
    mad = np.median(np.abs(valid_mags - median_mag))
    threshold_high = median_mag + mad_factor * mad

    is_stable = valid & (raw_mag <= threshold_low)
    is_scarp = valid & (raw_mag >= threshold_high)

    return is_stable, is_scarp


# ── Per-Pair Processing ──────────────────────────────────────────────

async def process_pair(pair_idx: int, pair_info: Dict[str, object], pair_dir: Path) -> PairResult:
    """
    Full pipeline for one temporal pair using two-pass approach:
      download → coregister → pass1 (raw correlation) → classify →
      pass2 (detrended correlation) → statistics.
    """
    pair_name = pair_info["pair"]
    pid_a = pair_info["pid_a"].replace("_RED", "")
    pid_b = pair_info["pid_b"].replace("_RED", "")
    gap = pair_info["gap"]
    score = pair_info["score"]

    result = PairResult(
        pair_idx=pair_idx,
        pair_name=pair_name,
        pid_a=pid_a,
        pid_b=pid_b,
        gap_mars_yr=gap,
        score=score,
    )

    # ── Cache check ──────────────────────────────────────────────────
    result_file = pair_dir / "pair_result.json"
    if result_file.exists():
        try:
            with open(result_file) as f:
                cached = json.load(f)
            if cached.get("success"):
                for k, v in cached.items():
                    if hasattr(result, k):
                        setattr(result, k, v)
                logger.info(f"[Pair {pair_idx}] Loaded cached result (success)")
                return result
        except (json.JSONDecodeError, KeyError):
            pass  # re-process

    pair_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    try:
        # ── Step 1: Download ─────────────────────────────────────────
        logger.info(f"[Pair {pair_idx}] Downloading {pid_a}...")
        path_a = await download_hirise_rdr(pid_a)
        if path_a is None:
            result.error = f"Download failed: {pid_a}"
            result.processing_time_s = time.time() - t0
            _save_pair_result(result, pair_dir)
            return result

        logger.info(f"[Pair {pair_idx}] Downloading {pid_b}...")
        path_b = await download_hirise_rdr(pid_b)
        if path_b is None:
            result.error = f"Download failed: {pid_b}"
            result.processing_time_s = time.time() - t0
            _save_pair_result(result, pair_dir)
            return result

        # ── Step 2: Coregister ───────────────────────────────────────
        logger.info(f"[Pair {pair_idx}] Co-registering {pid_a} ↔ {pid_b}...")
        coreg = coregister_geotiffs(path_a, path_b)
        result.img_height, result.img_width = coreg.img1.shape
        result.pixel_scale_m = coreg.pixel_scale_m

        logger.info(
            f"[Pair {pair_idx}] Overlap: {result.img_width}×{result.img_height} px, "
            f"{result.pixel_scale_m:.3f} m/px"
        )

        # ── Step 3: Two-pass approach ────────────────────────────────
        # Check if we can use MarsLandformNet classification (Pair 0 only)
        use_mln = False
        is_stable: Optional[np.ndarray] = None
        is_scarp: Optional[np.ndarray] = None

        if pair_idx == 0:
            logger.info(f"[Pair {pair_idx}] Attempting to load MarsLandformNet classification...")
            # First do a quick pass1 to get the grid shape
            disp_pass1 = sliding_window_correlation(
                coreg.img1, coreg.img2,
                chip_size=CHIP_SIZE, step_size=STEP_SIZE,
                upsample_factor=UPSAMPLE_FACTOR, snr_threshold=SNR_THRESHOLD,
                pixel_scale_m=coreg.pixel_scale_m,
                stable_mask=None, detrend_order=0,
            )
            mln_stable, mln_scarp = _load_original_classification(
                disp_pass1.row_centers, disp_pass1.col_centers
            )
            if mln_stable is not None and mln_scarp is not None:
                is_stable = mln_stable
                is_scarp = mln_scarp
                use_mln = True
                logger.info(
                    f"[Pair {pair_idx}] Using MarsLandformNet classification: "
                    f"{is_scarp.sum()} scarp, {is_stable.sum()} stable"
                )

        if not use_mln:
            # ── Pass 1: Raw correlation (no detrending) ──────────────
            logger.info(f"[Pair {pair_idx}] Pass 1: Raw phase correlation (no detrend)...")
            disp_pass1 = sliding_window_correlation(
                coreg.img1, coreg.img2,
                chip_size=CHIP_SIZE, step_size=STEP_SIZE,
                upsample_factor=UPSAMPLE_FACTOR, snr_threshold=SNR_THRESHOLD,
                pixel_scale_m=coreg.pixel_scale_m,
                stable_mask=None, detrend_order=0,
            )

            # Classify chips from raw displacements
            is_stable, is_scarp = _two_pass_classify(
                disp_pass1.magnitude_m, disp_pass1.valid_mask
            )
            logger.info(
                f"[Pair {pair_idx}] Pass 1 classification: "
                f"{is_stable.sum()} stable, {is_scarp.sum()} scarp "
                f"(of {disp_pass1.valid_mask.sum()} valid)"
            )

        # ── Build pixel-level stable mask for detrending ─────────
        stable_mask_px = _build_pixel_stable_mask(
            coreg.img1.shape,
            disp_pass1.row_centers, disp_pass1.col_centers,
            is_stable, CHIP_SIZE,
        )
        stable_frac = stable_mask_px.mean()
        logger.info(f"[Pair {pair_idx}] Pixel-level stable fraction: {stable_frac:.1%}")

        # ── Pass 2: Detrended correlation ────────────────────────
        logger.info(
            f"[Pair {pair_idx}] Pass 2: Phase correlation "
            f"(chip={CHIP_SIZE}, step={STEP_SIZE}, detrend={DETREND_ORDER})..."
        )
        displacement = sliding_window_correlation(
            coreg.img1, coreg.img2,
            chip_size=CHIP_SIZE, step_size=STEP_SIZE,
            upsample_factor=UPSAMPLE_FACTOR, snr_threshold=SNR_THRESHOLD,
            pixel_scale_m=coreg.pixel_scale_m,
            stable_mask=stable_mask_px, detrend_order=DETREND_ORDER,
        )

        # ── Step 4: Refine classification after detrending ───────
        valid = displacement.valid_mask

        if not use_mln:
            # For two-pass: refine classification using detrended magnitudes
            # Keep pass1 scarp candidates but verify they still show excess
            mag_m = displacement.magnitude_m
            valid_stable = valid & is_stable
            if valid_stable.sum() > 0:
                stable_median = np.median(mag_m[valid_stable])
                stable_mad = np.median(np.abs(mag_m[valid_stable] - stable_median))
                # Scarp = pass1 candidates that are still above stable+2*MAD after detrend
                detrend_threshold = stable_median + 2.0 * max(stable_mad, 0.01)
                is_scarp = is_scarp & valid & (mag_m >= detrend_threshold)
                logger.info(
                    f"[Pair {pair_idx}] Refined scarp after detrend: "
                    f"{is_scarp.sum()} chips (threshold={detrend_threshold:.4f}m)"
                )

        result.n_total_chips = int(valid.size)
        result.n_valid_chips = int(valid.sum())

        stable_valid = valid & is_stable
        scarp_valid = valid & is_scarp
        result.n_stable_chips = int(stable_valid.sum())
        result.n_scarp_chips = int(scarp_valid.sum())

        logger.info(
            f"[Pair {pair_idx}] Final chips: {result.n_valid_chips}/{result.n_total_chips} valid, "
            f"{result.n_stable_chips} stable, {result.n_scarp_chips} scarp"
        )

        if result.n_stable_chips < 10:
            result.error = f"Too few stable chips ({result.n_stable_chips})"
            result.processing_time_s = time.time() - t0
            _save_pair_result(result, pair_dir)
            return result

        if result.n_scarp_chips < 10:
            result.error = f"Too few scarp chips ({result.n_scarp_chips})"
            result.processing_time_s = time.time() - t0
            _save_pair_result(result, pair_dir)
            return result

        # ── Step 5: Displacement statistics ──────────────────────
        mag_m = displacement.magnitude_m
        stable_mags = mag_m[stable_valid]
        scarp_mags = mag_m[scarp_valid]

        result.stable_mean_mag_m = float(np.mean(stable_mags))
        result.stable_std_mag_m = float(np.std(stable_mags))
        result.stable_median_mag_m = float(np.median(stable_mags))
        result.scarp_mean_mag_m = float(np.mean(scarp_mags))
        result.scarp_std_mag_m = float(np.std(scarp_mags))
        result.scarp_median_mag_m = float(np.median(scarp_mags))
        result.excess_m = result.scarp_mean_mag_m - result.stable_mean_mag_m

        # Retreat rate
        if gap > 0:
            result.retreat_rate_m_per_mars_yr = result.excess_m / gap
            result.retreat_rate_cm_per_mars_yr = result.retreat_rate_m_per_mars_yr * 100

        result.noise_floor_2sigma_m = 2.0 * result.stable_std_mag_m
        if result.noise_floor_2sigma_m > 0:
            result.signal_to_noise = result.excess_m / result.noise_floor_2sigma_m

        # Statistical tests
        try:
            u_stat, u_p = stats.mannwhitneyu(
                scarp_mags, stable_mags, alternative="greater"
            )
            result.mann_whitney_p = float(u_p)
        except ValueError:
            result.mann_whitney_p = 1.0

        try:
            ks_stat, ks_p = stats.ks_2samp(scarp_mags, stable_mags)
            result.ks_test_p = float(ks_p)  # type: ignore[arg-type]
        except ValueError:
            result.ks_test_p = 1.0

        # ── Step 6: Save displacement data ───────────────────────
        np.savez_compressed(
            pair_dir / "displacement.npz",
            row_centers=displacement.row_centers,
            col_centers=displacement.col_centers,
            row_disp_m=displacement.row_disp_m,
            col_disp_m=displacement.col_disp_m,
            magnitude_m=mag_m,
            snr=displacement.snr,
            valid_mask=valid,
            is_stable=is_stable,
            is_scarp=is_scarp,
        )

        result.success = True
        result.processing_time_s = time.time() - t0

        logger.info(
            f"[Pair {pair_idx}] ✓ stable={result.stable_mean_mag_m:.3f}m, "
            f"scarp={result.scarp_mean_mag_m:.3f}m, excess={result.excess_m:.3f}m, "
            f"SNR={result.signal_to_noise:.2f}, p={result.mann_whitney_p:.2e}, "
            f"time={result.processing_time_s:.0f}s"
            f"{' [MLN]' if use_mln else ' [2-pass]'}"
        )

    except Exception as e:
        result.error = str(e)
        result.processing_time_s = time.time() - t0
        logger.error(f"[Pair {pair_idx}] ✗ FAILED: {e}", exc_info=True)

    _save_pair_result(result, pair_dir)
    return result


def _save_pair_result(result: PairResult, pair_dir: Path) -> None:
    """Persist per-pair result to JSON for caching/resume."""
    pair_dir.mkdir(parents=True, exist_ok=True)
    with open(pair_dir / "pair_result.json", "w") as f:
        json.dump(asdict(result), f, indent=2)


# ── Meta-Analysis ────────────────────────────────────────────────────

def meta_analysis(results: List[PairResult]) -> Dict[str, object]:
    """
    Combine retreat rate measurements across all successful pairs.

    Uses inverse-variance weighting: pairs with lower stable-terrain noise
    contribute more to the combined estimate.
    """
    successful = [r for r in results if r.success and r.excess_m > 0]

    if not successful:
        return {"error": "No successful pairs with positive excess displacement"}

    # Per-pair retreat rates and weights
    rates = np.array([r.retreat_rate_m_per_mars_yr for r in successful])

    # Weight = 1 / (noise_variance / gap^2)  →  pairs with low noise AND long
    # baseline dominate.  This is the standard inverse-variance weight for a
    # rate estimate.
    weights = np.array([
        (r.gap_mars_yr / max(r.stable_std_mag_m, 1e-6)) ** 2
        for r in successful
    ])
    weights /= weights.sum()

    weighted_mean = float(np.average(rates, weights=weights))
    simple_mean = float(np.mean(rates))
    simple_std = float(np.std(rates, ddof=1)) if len(rates) > 1 else 0.0

    # Standard error of the weighted mean
    weighted_se = float(np.sqrt(np.sum((weights * (rates - weighted_mean)) ** 2)))

    # Apply Phase-1 calibration factor (42% recovery → ×2.38)
    calibration_factor = 1.0 / 0.42
    calibrated_rate = weighted_mean * calibration_factor
    calibrated_se = weighted_se * calibration_factor

    return {
        "n_pairs_total": len(results),
        "n_pairs_successful": len(successful),
        "n_pairs_with_positive_excess": len(successful),
        "n_pairs_failed": sum(1 for r in results if not r.success),
        # Raw (uncalibrated) retreat rates
        "raw_weighted_mean_m_per_mars_yr": weighted_mean,
        "raw_weighted_mean_cm_per_mars_yr": weighted_mean * 100,
        "raw_simple_mean_m_per_mars_yr": simple_mean,
        "raw_simple_std_m_per_mars_yr": simple_std,
        "raw_weighted_se_m_per_mars_yr": weighted_se,
        # Calibrated (÷0.42 from synthetic injection test)
        "calibration_factor": calibration_factor,
        "calibrated_weighted_mean_m_per_mars_yr": calibrated_rate,
        "calibrated_weighted_mean_cm_per_mars_yr": calibrated_rate * 100,
        "calibrated_weighted_se_m_per_mars_yr": calibrated_se,
        "calibrated_weighted_se_cm_per_mars_yr": calibrated_se * 100,
        # Noise statistics
        "mean_stable_noise_m": float(np.mean([r.stable_mean_mag_m for r in successful])),
        "noise_reduction_vs_single": (
            simple_std / simple_mean if simple_mean > 0 else float("inf")
        ),
        "theoretical_sqrt_n": 1.0 / np.sqrt(len(successful)),
        # Per-pair breakdown
        "per_pair": [
            {
                "pair": r.pair_name,
                "gap_mars_yr": round(r.gap_mars_yr, 2),
                "score": round(r.score, 3),
                "retreat_rate_m_my": round(r.retreat_rate_m_per_mars_yr, 4),
                "stable_noise_m": round(r.stable_mean_mag_m, 4),
                "snr": round(r.signal_to_noise, 3),
                "p_value": f"{r.mann_whitney_p:.2e}",
                "weight": round(float(w), 4),
            }
            for r, w in zip(successful, weights)
        ],
    }


# ── Summary Printing ─────────────────────────────────────────────────

def print_summary(results: List[PairResult], summary: Dict[str, object]) -> None:
    """Print formatted results table and meta-analysis."""
    W = 130
    print(f"\n{'=' * W}")
    print("PHASE 3 — MULTI-PAIR BATCH RESULTS (two-pass + order-3 polynomial detrending)")
    print(f"{'=' * W}")
    header = (
        f"{'#':>2} {'Pair':<42} {'Gap(My)':>8} {'Score':>6} "
        f"{'Stable(m)':>10} {'Scarp(m)':>10} {'Excess(m)':>10} "
        f"{'Rate(m/My)':>11} {'SNR':>6} {'p-value':>10} {'Time':>6} {'Status':>6}"
    )
    print(header)
    print("-" * W)

    for r in results:
        if r.success:
            print(
                f"{r.pair_idx:>2} {r.pair_name:<42} "
                f"{r.gap_mars_yr:>8.2f} {r.score:>6.3f} "
                f"{r.stable_mean_mag_m:>10.4f} {r.scarp_mean_mag_m:>10.4f} "
                f"{r.excess_m:>10.4f} {r.retreat_rate_m_per_mars_yr:>11.4f} "
                f"{r.signal_to_noise:>6.2f} {r.mann_whitney_p:>10.2e} "
                f"{r.processing_time_s:>5.0f}s {'OK':>6}"
            )
        else:
            print(
                f"{r.pair_idx:>2} {r.pair_name:<42} "
                f"{r.gap_mars_yr:>8.2f} {r.score:>6.3f} "
                f"{'—':>10} {'—':>10} {'—':>10} {'—':>11} "
                f"{'—':>6} {'—':>10} "
                f"{r.processing_time_s:>5.0f}s {'FAIL':>6}"
            )
            if r.error:
                print(f"   └─ {r.error}")

    print(f"{'=' * W}")

    if "error" in summary:
        print(f"\n⚠ {summary['error']}")
        return

    print(f"\n{'─' * 70}")
    print("META-ANALYSIS")
    print(f"{'─' * 70}")
    n_ok = summary["n_pairs_successful"]
    n_tot = summary["n_pairs_total"]
    print(f"  Successful pairs:  {n_ok}/{n_tot}")
    print()
    print(f"  Raw retreat rate (weighted mean):  "
          f"{summary['raw_weighted_mean_cm_per_mars_yr']:.2f} cm/Mars-yr")
    print(f"  Raw retreat rate (simple mean):    "
          f"{summary['raw_simple_mean_m_per_mars_yr'] * 100:.2f} "
          f"± {summary['raw_simple_std_m_per_mars_yr'] * 100:.2f} cm/Mars-yr")
    print()
    print(f"  Calibration factor (Phase 1):      ×{summary['calibration_factor']:.2f}")
    print(f"  Calibrated retreat rate:           "
          f"{summary['calibrated_weighted_mean_cm_per_mars_yr']:.2f} "
          f"± {summary['calibrated_weighted_se_cm_per_mars_yr']:.2f} cm/Mars-yr")
    print()
    print(f"  Mean stable noise:                 {summary['mean_stable_noise_m']:.4f} m")
    print(f"  Observed scatter / mean:           {summary['noise_reduction_vs_single']:.2f}")
    print(f"  Theoretical 1/√N:                  {summary['theoretical_sqrt_n']:.2f}")
    print(f"{'─' * 70}")


# ── Main ─────────────────────────────────────────────────────────────

async def main() -> List[PairResult]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load pairs
    with open(PAIRS_FILE) as f:
        data = json.load(f)

    pairs = data["pairs_ranked_by_illumination"]
    logger.info(f"Loaded {len(pairs)} temporal pairs from {PAIRS_FILE.name}")

    # Identify unique products to download
    unique_pids = set()
    for p in pairs:
        unique_pids.add(p["pid_a"].replace("_RED", ""))
        unique_pids.add(p["pid_b"].replace("_RED", ""))
    logger.info(f"Unique HiRISE products needed: {len(unique_pids)}")

    # Process each pair sequentially (downloads are cached, so safe to re-run)
    results: List[PairResult] = []

    for i, pair_info in enumerate(pairs):
        pair_dir = RESULTS_DIR / f"pair_{i:02d}"
        logger.info(f"\n{'═' * 70}")
        logger.info(f"PAIR {i}/{len(pairs) - 1}: {pair_info['pair']}")
        logger.info(f"  Gap: {pair_info['gap']:.2f} Mars yr | Score: {pair_info['score']:.3f}")
        logger.info(f"{'═' * 70}")

        result = await process_pair(i, pair_info, pair_dir)
        results.append(result)

        # Running tally
        n_ok = sum(1 for r in results if r.success)
        n_fail = sum(1 for r in results if r.error and not r.success)
        logger.info(
            f"Progress: {i + 1}/{len(pairs)} processed "
            f"({n_ok} success, {n_fail} failed)"
        )

    # ── Meta-analysis ────────────────────────────────────────────────
    summary = meta_analysis(results)

    # Save full report
    report = {
        "description": "Phase 3 — Multi-Pair SCT Temporal Change Analysis (two-pass)",
        "parameters": {
            "chip_size": CHIP_SIZE,
            "step_size": STEP_SIZE,
            "upsample_factor": UPSAMPLE_FACTOR,
            "detrend_order": DETREND_ORDER,
            "snr_threshold": SNR_THRESHOLD,
            "reference_percentile": REFERENCE_PERCENTILE,
            "scarp_mad_factor": SCARP_MAD_FACTOR,
        },
        "meta_analysis": summary,
        "pair_results": [asdict(r) for r in results],
    }

    report_file = RESULTS_DIR / "phase3_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Full report saved: {report_file}")

    # Print human-readable summary
    print_summary(results, summary)

    return results


if __name__ == "__main__":
    asyncio.run(main())
