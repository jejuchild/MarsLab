#!/usr/bin/env python3
"""Diagnostic: Why does Pair 0 show strong signal but Pair 1 does not?

Both pairs share the same base image (PSP_007173_2245) at the same location.
Pair 0: PSP_007173_2245 → ESP_077815_2245 (excess=0.84m, p=9e-27)
Pair 1: PSP_007173_2245 → ESP_077393_2245 (excess=-0.02m, p=0.68)

Two diagnostics:
1. OVERLAP GEOMETRY: Do the center-crops cover the same ground?
2. ILLUMINATION DIRECTION: Do displacement vectors correlate with sun angle difference?

Usage:
    python -m backend.analysis.sct_temporal.diagnostic_pair01
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = PROJECT_ROOT / "results" / "sct_analysis" / "phase3"
CACHE_DIR = PROJECT_ROOT / "Data" / "HiRISE" / "rdr_cache"
MAX_OVERLAP_PX = 8192


def diagnostic_overlap_geometry():
    """Compare the actual spatial footprints of Pair 0 and Pair 1 center-crops."""
    import rasterio
    from rasterio.windows import from_bounds

    base = CACHE_DIR / "PSP_007173_2245_RED.JP2"
    second_0 = CACHE_DIR / "ESP_077815_2245_RED.JP2"
    second_1 = CACHE_DIR / "ESP_077393_2245_RED.JP2"

    results = {}
    for label, path_b in [("Pair_0", second_0), ("Pair_1", second_1)]:
        with rasterio.open(base) as ds1, rasterio.open(path_b) as ds2:
            b1 = ds1.bounds
            b2 = ds2.bounds
            pixel_scale = abs(ds1.transform.a)

            # Full overlap
            ol_left = max(b1.left, b2.left)
            ol_bottom = max(b1.bottom, b2.bottom)
            ol_right = min(b1.right, b2.right)
            ol_top = min(b1.top, b2.top)

            ol_w_px = int((ol_right - ol_left) / pixel_scale)
            ol_h_px = int((ol_top - ol_bottom) / pixel_scale)

            # Center-crop
            cx = (ol_left + ol_right) / 2
            cy = (ol_bottom + ol_top) / 2
            half_w = min(ol_w_px, MAX_OVERLAP_PX) / 2 * pixel_scale
            half_h = min(ol_h_px, MAX_OVERLAP_PX) / 2 * pixel_scale
            crop_left = cx - half_w
            crop_right = cx + half_w
            crop_bottom = cy - half_h
            crop_top = cy + half_h

            crop_w_px = int((crop_right - crop_left) / pixel_scale)
            crop_h_px = int((crop_top - crop_bottom) / pixel_scale)

            results[label] = {
                "second_image": path_b.name,
                "base_bounds": {"left": b1.left, "bottom": b1.bottom, "right": b1.right, "top": b1.top},
                "second_bounds": {"left": b2.left, "bottom": b2.bottom, "right": b2.right, "top": b2.top},
                "full_overlap": {
                    "left": ol_left, "bottom": ol_bottom, "right": ol_right, "top": ol_top,
                    "width_px": ol_w_px, "height_px": ol_h_px,
                },
                "center_crop": {
                    "left": crop_left, "bottom": crop_bottom, "right": crop_right, "top": crop_top,
                    "width_px": crop_w_px, "height_px": crop_h_px,
                    "center_x": cx, "center_y": cy,
                },
                "pixel_scale_m": pixel_scale,
            }

    # Compare
    c0 = results["Pair_0"]["center_crop"]
    c1 = results["Pair_1"]["center_crop"]

    overlap_of_crops_left = max(c0["left"], c1["left"])
    overlap_of_crops_right = min(c0["right"], c1["right"])
    overlap_of_crops_bottom = max(c0["bottom"], c1["bottom"])
    overlap_of_crops_top = min(c0["top"], c1["top"])

    if overlap_of_crops_left < overlap_of_crops_right and overlap_of_crops_bottom < overlap_of_crops_top:
        overlap_w = overlap_of_crops_right - overlap_of_crops_left
        overlap_h = overlap_of_crops_top - overlap_of_crops_bottom
        crop0_area = (c0["right"] - c0["left"]) * (c0["top"] - c0["bottom"])
        overlap_area = overlap_w * overlap_h
        overlap_pct = overlap_area / crop0_area * 100
    else:
        overlap_w = overlap_h = overlap_pct = 0

    center_dist_m = np.sqrt((c0["center_x"] - c1["center_x"])**2 +
                            (c0["center_y"] - c1["center_y"])**2)

    print("\n" + "=" * 100)
    print("DIAGNOSTIC 1: OVERLAP GEOMETRY")
    print("=" * 100)
    print(f"\nBase image: PSP_007173_2245_RED.JP2")
    print(f"Pixel scale: {results['Pair_0']['pixel_scale_m']:.4f} m/px")

    for label in ["Pair_0", "Pair_1"]:
        r = results[label]
        fo = r["full_overlap"]
        cc = r["center_crop"]
        print(f"\n  {label} ({r['second_image']}):")
        print(f"    Full overlap: {fo['width_px']}×{fo['height_px']} px")
        print(f"    Center-crop:  {cc['width_px']}×{cc['height_px']} px")
        print(f"    Crop center:  ({cc['center_x']:.1f}, {cc['center_y']:.1f}) m")
        print(f"    Crop bounds:  L={cc['left']:.1f} R={cc['right']:.1f} B={cc['bottom']:.1f} T={cc['top']:.1f}")

    print(f"\n  Center offset:   {center_dist_m:.1f} m ({center_dist_m/0.25:.0f} px)")
    print(f"  Crop-crop overlap: {overlap_pct:.1f}%")

    if overlap_pct < 50:
        print(f"  ⚠ CRITICAL: Crops cover DIFFERENT ground! Only {overlap_pct:.1f}% shared.")
        print(f"  This means Pair 0 and Pair 1 are analyzing different terrain patches.")
    elif overlap_pct < 90:
        print(f"  ⚠ WARNING: Crops partially overlap ({overlap_pct:.1f}%). Results not fully comparable.")
    else:
        print(f"  ✓ Crops largely overlap ({overlap_pct:.1f}%). Spatial footprint is similar.")

    return results, overlap_pct, center_dist_m


def diagnostic_illumination_direction():
    """Check if displacement vectors correlate with illumination difference direction.

    For each pair, compute:
    - Illumination difference direction (from sun angle difference)
    - Mean displacement vector direction on scarp vs stable
    - Correlation between them
    """
    with open(PROJECT_ROOT / "results" / "sct_temporal_pairs_v2.json") as f:
        config = json.load(f)

    pairs_config = config["pairs_ranked_by_illumination"]
    metadata = config["metadata"]

    # Successful pairs from crossval
    successful = [0, 1, 4, 7, 8, 9]

    print("\n" + "=" * 100)
    print("DIAGNOSTIC 2: ILLUMINATION DIRECTION ANALYSIS")
    print("=" * 100)

    print(f"\n{'#':>2} {'Pair':<42} {'dInc':>5} {'dAz':>5} "
          f"{'IllumDir':>8} {'DispDir_scarp':>12} {'DispDir_stable':>13} "
          f"{'AngleDiff':>9} {'Excess':>7}")
    print("-" * 120)

    illum_dirs = []
    disp_dirs_scarp = []
    disp_dirs_stable = []
    excesses = []

    for pair_idx in successful:
        pc = pairs_config[pair_idx]
        pid_a = pc["pid_a"]
        pid_b = pc["pid_b"]

        # Illumination geometry
        inc_a = metadata[pid_a]["incidence"]
        inc_b = metadata[pid_b]["incidence"]
        az_a = metadata[pid_a]["sub_solar_az"]
        az_b = metadata[pid_b]["sub_solar_az"]

        # The illumination "shift" direction: how does the sun move between images?
        # Sun position relative to surface in (N, E) components
        # Incidence angle from zenith, azimuth from north
        # Shadow tip displacement direction ≈ illumination difference direction
        sun_x_a = np.sin(np.radians(az_a)) * np.tan(np.radians(inc_a))
        sun_y_a = np.cos(np.radians(az_a)) * np.tan(np.radians(inc_a))
        sun_x_b = np.sin(np.radians(az_b)) * np.tan(np.radians(inc_b))
        sun_y_b = np.cos(np.radians(az_b)) * np.tan(np.radians(inc_b))

        # Difference in shadow-casting direction (in ground meters per meter height)
        d_shadow_x = sun_x_b - sun_x_a  # east component
        d_shadow_y = sun_y_b - sun_y_a  # north component
        illum_dir = np.degrees(np.arctan2(d_shadow_x, d_shadow_y)) % 360  # azimuth from north
        illum_mag = np.sqrt(d_shadow_x**2 + d_shadow_y**2)

        # Load displacement data
        disp_path = RESULTS_DIR / f"pair_{pair_idx:02d}" / "displacement.npz"
        data = np.load(disp_path)
        row_disp = data["row_disp_m"]
        col_disp = data["col_disp_m"]
        valid = data["valid_mask"]

        # Load crossval classification
        # For the crossval, we need to reload classification
        # Use the is_scarp/is_stable from the stored data (from two-pass), but
        # for Pair 0/1 we should use MLN classification
        # For simplicity, use the stored classification from displacement.npz
        is_scarp = data["is_scarp"]
        is_stable = data["is_stable"]

        scarp_valid = valid & is_scarp
        stable_valid = valid & is_stable

        # Mean displacement vectors
        if scarp_valid.sum() > 5:
            scarp_row = np.mean(row_disp[scarp_valid])
            scarp_col = np.mean(col_disp[scarp_valid])
            scarp_dir = np.degrees(np.arctan2(scarp_col, scarp_row)) % 360
        else:
            scarp_row = scarp_col = scarp_dir = float("nan")

        if stable_valid.sum() > 5:
            stable_row = np.mean(row_disp[stable_valid])
            stable_col = np.mean(col_disp[stable_valid])
            stable_dir = np.degrees(np.arctan2(stable_col, stable_row)) % 360
        else:
            stable_row = stable_col = stable_dir = float("nan")

        # Excess displacement
        scarp_mag = np.mean(data["magnitude_m"][scarp_valid]) if scarp_valid.sum() > 0 else 0
        stable_mag = np.mean(data["magnitude_m"][stable_valid]) if stable_valid.sum() > 0 else 0
        excess = scarp_mag - stable_mag

        # Angular difference between illumination direction and scarp displacement
        if not np.isnan(scarp_dir):
            angle_diff = abs(illum_dir - scarp_dir)
            if angle_diff > 180:
                angle_diff = 360 - angle_diff
        else:
            angle_diff = float("nan")

        print(f"{pair_idx:>2} {pc['pair']:<42} "
              f"{inc_b - inc_a:>+5.1f} {az_b - az_a:>+5.1f} "
              f"{illum_dir:>7.1f}° {scarp_dir:>11.1f}° {stable_dir:>12.1f}° "
              f"{angle_diff:>8.1f}° {excess:>+6.3f}m")

        illum_dirs.append(illum_dir)
        disp_dirs_scarp.append(scarp_dir)
        disp_dirs_stable.append(stable_dir)
        excesses.append(excess)

    # Now analyze: do scarp AND stable move in the illumination direction?
    # If both move in the illumination direction → artifact (shadow shift)
    # If only scarp moves → could be real + illumination
    print("\n" + "-" * 100)
    print("ANALYSIS: Illumination artifact signature")
    print("-" * 100)

    # Check if displacement directions are similar between scarp and stable
    # (if both move same way → bulk systematic artifact, not terrain-specific)
    print("\n  Per-pair displacement vector comparison:")
    for i, pair_idx in enumerate(successful):
        disp_path = RESULTS_DIR / f"pair_{pair_idx:02d}" / "displacement.npz"
        data = np.load(disp_path)
        valid = data["valid_mask"]
        is_scarp = data["is_scarp"]
        is_stable = data["is_stable"]

        scarp_valid = valid & is_scarp
        stable_valid = valid & is_stable

        if scarp_valid.sum() > 5 and stable_valid.sum() > 5:
            scarp_row_mean = np.mean(data["row_disp_m"][scarp_valid])
            scarp_col_mean = np.mean(data["col_disp_m"][scarp_valid])
            stable_row_mean = np.mean(data["row_disp_m"][stable_valid])
            stable_col_mean = np.mean(data["col_disp_m"][stable_valid])

            # Excess vector (scarp - stable)
            excess_row = scarp_row_mean - stable_row_mean
            excess_col = scarp_col_mean - stable_col_mean
            excess_mag = np.sqrt(excess_row**2 + excess_col**2)
            excess_dir = np.degrees(np.arctan2(excess_col, excess_row)) % 360

            # Compare excess direction to illumination direction
            angle_diff = abs(illum_dirs[i] - excess_dir)
            if angle_diff > 180:
                angle_diff = 360 - angle_diff

            print(f"    Pair {pair_idx}: scarp_vec=({scarp_row_mean:+.3f}, {scarp_col_mean:+.3f})m "
                  f"stable_vec=({stable_row_mean:+.3f}, {stable_col_mean:+.3f})m "
                  f"EXCESS_vec=({excess_row:+.3f}, {excess_col:+.3f})m |{excess_mag:.3f}m| "
                  f"dir={excess_dir:.1f}° vs illum={illum_dirs[i]:.1f}° (Δ={angle_diff:.1f}°)")
        else:
            print(f"    Pair {pair_idx}: insufficient chips for comparison")

    # Detailed Pair 0 vs Pair 1 comparison
    print("\n" + "-" * 100)
    print("DETAILED PAIR 0 vs PAIR 1 COMPARISON")
    print("-" * 100)

    for pair_idx in [0, 1]:
        pc = pairs_config[pair_idx]
        pid_b = pc["pid_b"]
        meta_b = metadata[pid_b]
        meta_a = metadata[pc["pid_a"]]

        disp_path = RESULTS_DIR / f"pair_{pair_idx:02d}" / "displacement.npz"
        data = np.load(disp_path)
        valid = data["valid_mask"]
        mag = data["magnitude_m"]

        print(f"\n  Pair {pair_idx}: {pc['pair']}")
        print(f"    Second image: {pid_b}")
        print(f"    Incidence: {meta_a['incidence']:.1f}° → {meta_b['incidence']:.1f}° (Δ={meta_b['incidence']-meta_a['incidence']:+.1f}°)")
        print(f"    Azimuth:   {meta_a['sub_solar_az']:.1f}° → {meta_b['sub_solar_az']:.1f}° (Δ={meta_b['sub_solar_az']-meta_a['sub_solar_az']:+.1f}°)")
        print(f"    Emission:  {meta_a['emission']:.2f}° → {meta_b['emission']:.2f}° (Δ={meta_b['emission']-meta_a['emission']:+.2f}°)")
        print(f"    Ls:        {meta_a['Ls']:.1f}° → {meta_b['Ls']:.1f}° (Δ={meta_b['Ls']-meta_a['Ls']:+.1f}°)")
        print(f"    Valid chips: {valid.sum()}")
        print(f"    Displacement stats: mean={mag[valid].mean():.4f}m, "
              f"std={mag[valid].std():.4f}m, "
              f"median={np.median(mag[valid]):.4f}m")

        # Distribution of row/col displacements
        row_d = data["row_disp_m"][valid]
        col_d = data["col_disp_m"][valid]
        print(f"    Row displacement: mean={row_d.mean():+.4f}m, std={row_d.std():.4f}m")
        print(f"    Col displacement: mean={col_d.mean():+.4f}m, std={col_d.std():.4f}m")

    return illum_dirs, disp_dirs_scarp, excesses


def diagnostic_displacement_maps():
    """Compare spatial patterns of displacement between Pair 0 and Pair 1.

    If illumination artifact: displacement should be spatially smooth (whole-image shift)
    If real retreat: displacement should be localized to scarp edges
    """
    print("\n" + "=" * 100)
    print("DIAGNOSTIC 3: SPATIAL PATTERN OF DISPLACEMENT")
    print("=" * 100)

    for pair_idx in [0, 1]:
        disp_path = RESULTS_DIR / f"pair_{pair_idx:02d}" / "displacement.npz"
        data = np.load(disp_path)
        valid = data["valid_mask"]
        mag = data["magnitude_m"]
        row_d = data["row_disp_m"]
        col_d = data["col_disp_m"]

        # Spatial autocorrelation: Moran's I approximation
        # If displacement is spatially smooth → high autocorrelation → systematic artifact
        # If displacement is noisy/localized → low autocorrelation → could be real

        valid_mag = np.where(valid, mag, np.nan)
        valid_row = np.where(valid, row_d, np.nan)
        valid_col = np.where(valid, col_d, np.nan)

        # Simple spatial correlation: correlation with 1-chip-shifted version
        # Row direction
        if valid.shape[0] > 2 and valid.shape[1] > 2:
            # Row-shifted correlation (for magnitude)
            a = valid_mag[:-1, :].ravel()
            b = valid_mag[1:, :].ravel()
            mask = ~(np.isnan(a) | np.isnan(b))
            if mask.sum() > 10:
                lag1_row_corr = np.corrcoef(a[mask], b[mask])[0, 1]
            else:
                lag1_row_corr = float("nan")

            # Col-shifted correlation
            a = valid_mag[:, :-1].ravel()
            b = valid_mag[:, 1:].ravel()
            mask = ~(np.isnan(a) | np.isnan(b))
            if mask.sum() > 10:
                lag1_col_corr = np.corrcoef(a[mask], b[mask])[0, 1]
            else:
                lag1_col_corr = float("nan")
        else:
            lag1_row_corr = lag1_col_corr = float("nan")

        # Spatial gradient of displacement
        mag_valid = mag.copy()
        mag_valid[~valid] = np.nan
        grad_row = np.nanstd(np.diff(mag_valid, axis=0))
        grad_col = np.nanstd(np.diff(mag_valid, axis=1))

        print(f"\n  Pair {pair_idx}:")
        print(f"    Lag-1 spatial autocorrelation (magnitude):")
        print(f"      Row direction: r={lag1_row_corr:.4f}")
        print(f"      Col direction: r={lag1_col_corr:.4f}")
        if not np.isnan(lag1_row_corr):
            avg_corr = (lag1_row_corr + lag1_col_corr) / 2
            if avg_corr > 0.7:
                print(f"      → HIGH spatial autocorrelation ({avg_corr:.3f}): suggests SYSTEMATIC artifact")
            elif avg_corr > 0.3:
                print(f"      → MODERATE spatial autocorrelation ({avg_corr:.3f}): mixed signal")
            else:
                print(f"      → LOW spatial autocorrelation ({avg_corr:.3f}): suggests localized/noisy signal")
        print(f"    Spatial gradient of displacement: row_std={grad_row:.4f}m, col_std={grad_col:.4f}m")


def main():
    """Run all diagnostics."""
    print("\n" + "#" * 100)
    print("# DIAGNOSTIC: Pair 0 vs Pair 1 — Why Opposite Results?")
    print("#" * 100)
    print("\nPair 0: PSP_007173_2245 → ESP_077815_2245  (excess=+0.84m, p=9e-27)")
    print("Pair 1: PSP_007173_2245 → ESP_077393_2245  (excess=-0.02m, p=0.68)")
    print("Same base image, same location, similar temporal gap (~8 My)")

    # Diagnostic 1: Overlap geometry
    geo_results, overlap_pct, center_dist = diagnostic_overlap_geometry()

    # Diagnostic 2: Illumination direction
    illum_dirs, disp_dirs, excesses = diagnostic_illumination_direction()

    # Diagnostic 3: Spatial patterns
    diagnostic_displacement_maps()

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 100)

    print(f"\n  1. SPATIAL OVERLAP of center-crops:")
    print(f"     Center offset: {center_dist:.1f} m ({center_dist/0.25:.0f} px)")
    print(f"     Crop overlap: {overlap_pct:.1f}%")
    if overlap_pct < 50:
        print(f"     → FINDING: Pair 0 and Pair 1 analyze DIFFERENT terrain patches!")
        print(f"       This alone could explain the discrepancy.")
    else:
        print(f"     → Spatial footprint is sufficiently similar.")
        print(f"       The discrepancy is NOT due to different terrain patches.")

    print(f"\n  2. ILLUMINATION DIFFERENCE:")
    with open(PROJECT_ROOT / "results" / "sct_temporal_pairs_v2.json") as f:
        cfg = json.load(f)
    p0 = cfg["pairs_ranked_by_illumination"][0]
    p1 = cfg["pairs_ranked_by_illumination"][1]
    print(f"     Pair 0: dInc={p0['dInc']:.1f}°, dAz={p0['dAz']:.1f}° (dLs={p0['dLs']:.1f}°)")
    print(f"     Pair 1: dInc={p1['dInc']:.1f}°, dAz={p1['dAz']:.1f}° (dLs={p1['dLs']:.1f}°)")
    print(f"     Pair 0 has SMALLER illumination difference → better pair")
    print(f"     Pair 1 has LARGER illumination difference → more contaminated")
    print(f"     BUT: Pair 1 shows NO signal while Pair 0 shows STRONG signal")
    print(f"     → If signal were real, BOTH should show it")
    print(f"     → If signal were illumination artifact, Pair 1 should show MORE, not less")

    # Save diagnostic results
    report = {
        "description": "Diagnostic analysis: Pair 0 vs Pair 1 discrepancy",
        "pair_0": "PSP_007173_2245 → ESP_077815_2245 (excess=+0.84m)",
        "pair_1": "PSP_007173_2245 → ESP_077393_2245 (excess=-0.02m)",
        "overlap_geometry": {
            "center_offset_m": round(center_dist, 1),
            "center_offset_px": int(center_dist / 0.25),
            "crop_overlap_pct": round(overlap_pct, 1),
            "pair_0_crop": geo_results["Pair_0"]["center_crop"],
            "pair_1_crop": geo_results["Pair_1"]["center_crop"],
        },
        "illumination": {
            "pair_0_dInc": p0["dInc"],
            "pair_0_dAz": p0["dAz"],
            "pair_1_dInc": p1["dInc"],
            "pair_1_dAz": p1["dAz"],
        },
    }

    out_path = RESULTS_DIR / "diagnostic_pair01.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Diagnostic report saved: {out_path}")


if __name__ == "__main__":
    main()
