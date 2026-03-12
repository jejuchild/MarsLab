#!/usr/bin/env python3
"""
Phase 1: Synthetic Injection Test for SCT Temporal Change Pipeline.

Validates that the phase correlation pipeline can correctly recover
known sub-pixel displacements. Creates a synthetic image pair where
scarp regions have a known shift and stable regions have zero shift,
then measures recovery accuracy.

This test is independent of co-registration quality — it validates
the measurement methodology itself.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.fft import fft2, ifft2
from scipy import ndimage

from .phase_correlation import (
    cosicorr_phase_correlation,
    sliding_window_correlation,
    DisplacementField,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class InjectionResult:
    """Results of a single injection test."""

    injected_dx_px: float
    injected_dy_px: float
    injected_mag_px: float
    # Scarp region recovery
    scarp_measured_dx_px: float
    scarp_measured_dy_px: float
    scarp_measured_mag_px: float
    scarp_recovery_dx_pct: float
    scarp_recovery_dy_pct: float
    scarp_recovery_mag_pct: float
    scarp_n_valid: int
    scarp_std_dx_px: float
    scarp_std_dy_px: float
    # Stable region noise
    stable_measured_dx_px: float
    stable_measured_dy_px: float
    stable_measured_mag_px: float
    stable_n_valid: int
    stable_std_dx_px: float
    stable_std_dy_px: float
    # Discrimination
    discrimination_snr: float  # (scarp_mag - stable_mag) / stable_std

    @property
    def summary(self) -> dict:
        return {
            "injected_px": {
                "dx": round(self.injected_dx_px, 4),
                "dy": round(self.injected_dy_px, 4),
                "mag": round(self.injected_mag_px, 4),
            },
            "scarp_recovery": {
                "dx_pct": round(self.scarp_recovery_dx_pct, 1),
                "dy_pct": round(self.scarp_recovery_dy_pct, 1),
                "mag_pct": round(self.scarp_recovery_mag_pct, 1),
                "n_valid": self.scarp_n_valid,
                "mean_dx_px": round(self.scarp_measured_dx_px, 4),
                "mean_dy_px": round(self.scarp_measured_dy_px, 4),
                "std_dx_px": round(self.scarp_std_dx_px, 4),
                "std_dy_px": round(self.scarp_std_dy_px, 4),
            },
            "stable_noise": {
                "mean_dx_px": round(self.stable_measured_dx_px, 4),
                "mean_dy_px": round(self.stable_measured_dy_px, 4),
                "mean_mag_px": round(self.stable_measured_mag_px, 4),
                "n_valid": self.stable_n_valid,
                "std_dx_px": round(self.stable_std_dx_px, 4),
                "std_dy_px": round(self.stable_std_dy_px, 4),
            },
            "discrimination_snr": round(self.discrimination_snr, 2),
        }


def generate_mars_terrain(
    size: int = 2048,
    n_scarps: int = 8,
    scarp_width_range: tuple[int, int] = (40, 120),
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic Mars-like terrain image with scarp features.

    Returns (image, scarp_mask) where scarp_mask is True at scarp edges.
    The image has realistic spatial frequency content similar to HiRISE.
    """
    rng = np.random.RandomState(seed)

    # Multi-scale terrain: sum of filtered noise at different scales
    img = np.zeros((size, size), dtype=np.float64)

    # Low frequency: broad terrain undulations
    noise_low = rng.randn(size, size)
    img += ndimage.gaussian_filter(noise_low, sigma=80) * 50

    # Medium frequency: smaller features (boulders, small dunes)
    noise_mid = rng.randn(size, size)
    img += ndimage.gaussian_filter(noise_mid, sigma=15) * 20

    # High frequency: fine texture
    noise_hi = rng.randn(size, size)
    img += ndimage.gaussian_filter(noise_hi, sigma=3) * 8

    # Pixel-level noise (sensor noise)
    img += rng.randn(size, size) * 2

    # Create scarp features: sharp elevation drops
    scarp_mask = np.zeros((size, size), dtype=bool)

    for _ in range(n_scarps):
        # Random scarp position and orientation
        cy = rng.randint(size // 4, 3 * size // 4)
        cx = rng.randint(size // 4, 3 * size // 4)
        angle = rng.uniform(0, np.pi)
        length = rng.randint(size // 6, size // 3)
        width = rng.randint(scarp_width_range[0], scarp_width_range[1])
        drop = rng.uniform(30, 80)  # elevation drop across scarp

        # Create scarp as a sigmoid transition
        yy, xx = np.mgrid[0:size, 0:size]
        # Distance from scarp line (perpendicular)
        dx_line = (xx - cx) * np.sin(angle) - (yy - cy) * np.cos(angle)
        # Distance along scarp line
        dl_line = (xx - cx) * np.cos(angle) + (yy - cy) * np.sin(angle)

        # Sigmoid for elevation drop
        scarp_profile = drop / (1 + np.exp(-dx_line / (width * 0.15)))
        # Limit extent along the scarp line
        along_mask = np.abs(dl_line) < length / 2
        scarp_profile *= along_mask

        img += scarp_profile

        # Mark scarp edge region (high gradient zone)
        edge_zone = (np.abs(dx_line) < width / 2) & along_mask
        scarp_mask |= edge_zone

    # Normalize to 0-1 range (like HiRISE DN)
    img = (img - img.min()) / (img.max() - img.min())

    return img, scarp_mask


def apply_fourier_shift(
    img: np.ndarray,
    dy_px: float,
    dx_px: float,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Apply exact sub-pixel shift using Fourier phase shift.

    If mask is provided, only shift pixels where mask is True,
    with smooth blending at boundaries.
    """
    rows, cols = img.shape

    if mask is None:
        # Shift entire image
        F = fft2(img)
        fy = np.fft.fftfreq(rows)
        fx = np.fft.fftfreq(cols)
        FX, FY = np.meshgrid(fx, fy)
        phase_shift = np.exp(-2j * np.pi * (FY * dy_px + FX * dx_px))
        shifted = np.real(ifft2(F * phase_shift))
        return shifted

    # Shift only masked region with smooth blending
    shifted_full = apply_fourier_shift(img, dy_px, dx_px, mask=None)

    # Smooth the mask boundary to avoid sharp edges
    smooth_mask = ndimage.gaussian_filter(mask.astype(np.float64), sigma=5)
    smooth_mask = np.clip(smooth_mask, 0, 1)

    # Blend: shifted in mask, original outside
    result = img * (1 - smooth_mask) + shifted_full * smooth_mask
    return result


def run_single_injection(
    img_size: int = 2048,
    inject_dx_px: float = 0.5,
    inject_dy_px: float = 0.25,
    chip_size: int = 256,
    step_size: int = 128,
    seed: int = 42,
) -> InjectionResult:
    """
    Run a single synthetic injection test.

    Creates terrain, injects known displacement at scarps,
    runs phase correlation, and measures recovery.
    """
    logger.info(
        f"Injection test: dx={inject_dx_px:.3f}px, dy={inject_dy_px:.3f}px, "
        f"img={img_size}px, chip={chip_size}px"
    )

    # Generate terrain
    img1, scarp_mask = generate_mars_terrain(img_size, seed=seed)
    logger.info(
        f"Generated terrain: {img_size}×{img_size}, "
        f"scarp fraction: {scarp_mask.mean():.1%}"
    )

    # Create displaced image: shift scarp regions only
    img2 = apply_fourier_shift(img1, inject_dy_px, inject_dx_px, mask=scarp_mask)

    # Build stable mask (inverse of scarp, with buffer)
    scarp_dilated = ndimage.binary_dilation(scarp_mask, iterations=chip_size // 4)
    stable_mask = ~scarp_dilated

    logger.info(
        f"Stable fraction: {stable_mask.mean():.1%}, "
        f"Scarp fraction: {scarp_mask.mean():.1%}"
    )

    # Run phase correlation (NO bulk offset removal — we want raw measurements)
    t0 = time.time()
    disp = sliding_window_correlation(
        img1,
        img2,
        chip_size=chip_size,
        step_size=step_size,
        upsample_factor=100,
        snr_threshold=3.0,
        pixel_scale_m=0.25,
        stable_mask=stable_mask,
        remove_bulk_offset=True,  # Remove stable-terrain offset (should be ~0)
    )
    elapsed = time.time() - t0
    logger.info(f"Phase correlation: {elapsed:.1f}s, valid: {disp.valid_count}/{disp.row_disp.size}")

    # Classify chips by terrain type
    half = chip_size // 2
    nr, nc = len(disp.row_centers), len(disp.col_centers)

    scarp_at_centers = np.zeros((nr, nc), dtype=bool)
    stable_at_centers = np.zeros((nr, nc), dtype=bool)

    for i, r in enumerate(disp.row_centers):
        for j, c in enumerate(disp.col_centers):
            r0, r1 = r - half, r + half
            c0, c1 = c - half, c + half
            scarp_frac = scarp_mask[r0:r1, c0:c1].mean()
            stable_frac = stable_mask[r0:r1, c0:c1].mean()
            scarp_at_centers[i, j] = scarp_frac > 0.3
            stable_at_centers[i, j] = stable_frac > 0.8

    valid = disp.valid_mask
    scarp_valid = valid & scarp_at_centers
    stable_valid = valid & stable_at_centers

    # Extract measurements
    scarp_dx = disp.col_disp[scarp_valid]  # col_disp = dx
    scarp_dy = disp.row_disp[scarp_valid]  # row_disp = dy
    stable_dx = disp.col_disp[stable_valid]
    stable_dy = disp.row_disp[stable_valid]

    scarp_mag = np.sqrt(scarp_dx**2 + scarp_dy**2)
    stable_mag = np.sqrt(stable_dx**2 + stable_dy**2)

    inject_mag = np.sqrt(inject_dx_px**2 + inject_dy_px**2)

    # Recovery percentages
    def recovery_pct(measured_mean: float, injected: float) -> float:
        if abs(injected) < 1e-10:
            return 100.0 if abs(measured_mean) < 0.01 else 0.0
        return 100.0 * measured_mean / injected

    scarp_mean_dx = float(np.mean(scarp_dx)) if len(scarp_dx) > 0 else 0.0
    scarp_mean_dy = float(np.mean(scarp_dy)) if len(scarp_dy) > 0 else 0.0
    scarp_mean_mag = float(np.mean(scarp_mag)) if len(scarp_mag) > 0 else 0.0

    stable_mean_dx = float(np.mean(stable_dx)) if len(stable_dx) > 0 else 0.0
    stable_mean_dy = float(np.mean(stable_dy)) if len(stable_dy) > 0 else 0.0
    stable_mean_mag = float(np.mean(stable_mag)) if len(stable_mag) > 0 else 0.0

    # Discrimination SNR
    stable_std = float(np.std(stable_mag)) if len(stable_mag) > 1 else 1.0
    disc_snr = (scarp_mean_mag - stable_mean_mag) / max(stable_std, 1e-6)

    return InjectionResult(
        injected_dx_px=inject_dx_px,
        injected_dy_px=inject_dy_px,
        injected_mag_px=inject_mag,
        scarp_measured_dx_px=scarp_mean_dx,
        scarp_measured_dy_px=scarp_mean_dy,
        scarp_measured_mag_px=scarp_mean_mag,
        scarp_recovery_dx_pct=recovery_pct(scarp_mean_dx, inject_dx_px),
        scarp_recovery_dy_pct=recovery_pct(scarp_mean_dy, inject_dy_px),
        scarp_recovery_mag_pct=recovery_pct(scarp_mean_mag, inject_mag),
        scarp_n_valid=int(scarp_valid.sum()),
        scarp_std_dx_px=float(np.std(scarp_dx)) if len(scarp_dx) > 1 else 0.0,
        scarp_std_dy_px=float(np.std(scarp_dy)) if len(scarp_dy) > 1 else 0.0,
        stable_measured_dx_px=stable_mean_dx,
        stable_measured_dy_px=stable_mean_dy,
        stable_measured_mag_px=stable_mean_mag,
        stable_n_valid=int(stable_valid.sum()),
        stable_std_dx_px=float(np.std(stable_dx)) if len(stable_dx) > 1 else 0.0,
        stable_std_dy_px=float(np.std(stable_dy)) if len(stable_dy) > 1 else 0.0,
        discrimination_snr=disc_snr,
    )


def run_validation_suite(
    output_dir: Optional[Path] = None,
    img_size: int = 2048,
    chip_size: int = 256,
    step_size: int = 128,
) -> dict:
    """
    Run a comprehensive validation suite with multiple injection levels.

    Tests:
    1. Zero displacement (null test)
    2. Sub-pixel displacements at various magnitudes
    3. Different displacement directions
    """
    if output_dir is None:
        output_dir = Path("results/sct_analysis/validation")
    output_dir.mkdir(parents=True, exist_ok=True)

    test_cases = [
        # (name, dx_px, dy_px)
        ("null_test", 0.0, 0.0),
        ("tiny_0.1px", 0.1, 0.0),
        ("small_0.25px", 0.25, 0.0),
        ("medium_0.5px", 0.5, 0.0),
        ("large_1.0px", 1.0, 0.0),
        ("diagonal_0.35px", 0.25, 0.25),
        ("realistic_scarp_retreat", 0.19, 0.0),  # 0.048 m/Mars-yr * 8yr / 0.25m/px ≈ 1.5px total... but per chip it's fractional
        ("y_only_0.5px", 0.0, 0.5),
        ("oblique_0.5px", 0.35, 0.35),
    ]

    results = {}
    all_summaries = []

    for name, dx, dy in test_cases:
        logger.info(f"\n{'='*60}")
        logger.info(f"Test: {name} (dx={dx}, dy={dy})")
        logger.info(f"{'='*60}")

        result = run_single_injection(
            img_size=img_size,
            inject_dx_px=dx,
            inject_dy_px=dy,
            chip_size=chip_size,
            step_size=step_size,
        )

        results[name] = result
        summary = result.summary
        summary["test_name"] = name
        all_summaries.append(summary)

        logger.info(f"  Scarp recovery: dx={result.scarp_recovery_dx_pct:.1f}%, "
                     f"dy={result.scarp_recovery_dy_pct:.1f}%, "
                     f"mag={result.scarp_recovery_mag_pct:.1f}%")
        logger.info(f"  Stable noise: mag={result.stable_measured_mag_px:.4f} px")
        logger.info(f"  Discrimination SNR: {result.discrimination_snr:.2f}")

    # Save results
    report = {
        "config": {
            "img_size": img_size,
            "chip_size": chip_size,
            "step_size": step_size,
            "pixel_scale_m": 0.25,
        },
        "tests": all_summaries,
        "verdict": _compute_verdict(results),
    }

    report_path = output_dir / "synthetic_validation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"\nReport saved: {report_path}")

    # Print summary table
    _print_summary_table(results)

    return report


def _compute_verdict(results: dict[str, InjectionResult]) -> dict:
    """Compute overall validation verdict."""
    verdicts = {}

    # Null test: should show ~0 displacement everywhere
    null = results.get("null_test")
    if null:
        null_ok = null.stable_measured_mag_px < 0.05 and null.scarp_measured_mag_px < 0.05
        verdicts["null_test"] = "PASS" if null_ok else "FAIL"

    # Recovery tests: should recover >70% of injected displacement
    recovery_tests = ["small_0.25px", "medium_0.5px", "large_1.0px"]
    recoveries = []
    for name in recovery_tests:
        r = results.get(name)
        if r:
            recoveries.append(r.scarp_recovery_mag_pct)
    if recoveries:
        avg_recovery = np.mean(recoveries)
        verdicts["mean_recovery_pct"] = round(float(avg_recovery), 1)
        verdicts["recovery_test"] = "PASS" if avg_recovery > 70 else "MARGINAL" if avg_recovery > 50 else "FAIL"

    # Minimum detectable displacement
    if null:
        noise_floor_px = null.stable_std_dx_px * 2  # 2-sigma
        verdicts["noise_floor_2sigma_px"] = round(noise_floor_px, 4)
        verdicts["noise_floor_2sigma_m"] = round(noise_floor_px * 0.25, 4)

    # Overall
    all_pass = all(v == "PASS" for k, v in verdicts.items() if k.endswith("_test"))
    verdicts["overall"] = "PIPELINE VALIDATED" if all_pass else "NEEDS INVESTIGATION"

    return verdicts


def _print_summary_table(results: dict[str, InjectionResult]) -> None:
    """Print a formatted summary table."""
    print(f"\n{'='*90}")
    print("SYNTHETIC INJECTION VALIDATION RESULTS")
    print(f"{'='*90}")
    print(f"{'Test':<25} {'Inject(px)':<12} {'Scarp(px)':<12} {'Recovery%':<12} {'Stable(px)':<12} {'SNR':<8}")
    print(f"{'-'*25} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*8}")

    for name, r in results.items():
        inject_str = f"{r.injected_mag_px:.3f}"
        scarp_str = f"{r.scarp_measured_mag_px:.3f}"
        rec_str = f"{r.scarp_recovery_mag_pct:.1f}%"
        stable_str = f"{r.stable_measured_mag_px:.4f}"
        snr_str = f"{r.discrimination_snr:.2f}"
        print(f"{name:<25} {inject_str:<12} {scarp_str:<12} {rec_str:<12} {stable_str:<12} {snr_str:<8}")

    print(f"{'='*90}")


def main():
    """Run validation suite from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Synthetic Injection Validation for SCT Pipeline")
    parser.add_argument("--output", "-o", type=Path, default=Path("results/sct_analysis/validation"))
    parser.add_argument("--img-size", type=int, default=2048)
    parser.add_argument("--chip-size", type=int, default=256)
    parser.add_argument("--step-size", type=int, default=128)
    args = parser.parse_args()

    run_validation_suite(
        output_dir=args.output,
        img_size=args.img_size,
        chip_size=args.chip_size,
        step_size=args.step_size,
    )


if __name__ == "__main__":
    main()
