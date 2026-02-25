#!/usr/bin/env python3
"""
Generate CRISM ice_score and hyd_score overlays from browse maps.

Outputs:
- PNG visualizations for map overlay
- NumPy arrays (.npy) for filtering and analysis
- Statistics JSON for efficient filtering queries

Usage:
    python generate_score_maps.py [--force]

    --force: Regenerate all score maps even if they exist (bypass cache)
"""

import os
import sys
import json
import argparse
import numpy as np
from PIL import Image
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

BROWSE_DIR = Path(__file__).parent.parent / "crism_browse"
OUTPUT_DIR = Path(__file__).parent.parent / "crism_browse"
SCORE_DIR = Path(__file__).parent.parent / "crism_score"  # For .npy files
STATS_FILE = Path(__file__).parent.parent / "crism_score" / "score_stats.json"

# Score thresholds for precomputed statistics
THRESHOLDS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5]

# Numerical stability constant
EPS = 1e-6

# ============================================================
# UTILITIES
# ============================================================

def load_png(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load PNG and return R, G, B channels as float32 arrays (0-1)."""
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr[..., 0], arr[..., 1], arr[..., 2]


def normalized_colors(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute normalized color ratios.

    Returns:
        redness:   r / (r + g + b + eps)
        greenness: g / (r + g + b + eps)
        blueness:  b / (r + g + b + eps)
    """
    total = r + g + b + EPS
    redness = r / total
    greenness = g / total
    blueness = b / total
    return redness, greenness, blueness


def compute_hydration_score(r_h: np.ndarray, g_h: np.ndarray, b_h: np.ndarray,
                            r_i: np.ndarray, g_i: np.ndarray, b_i: np.ndarray,
                            r_c: np.ndarray, g_c: np.ndarray, b_c: np.ndarray) -> np.ndarray:
    """
    Compute hydration score from HYD, ICE, IC2 browse maps.

    - Signal: redness from HYD + 0.5 * redness from ICE/IC2
    - Penalty: 0.5 * (greenness + blueness from ICE/IC2)
    """
    redness_h, _, _ = normalized_colors(r_h, g_h, b_h)
    redness_i, greenness_i, blueness_i = normalized_colors(r_i, g_i, b_i)
    redness_c, greenness_c, blueness_c = normalized_colors(r_c, g_c, b_c)

    S_hyd = (
        redness_h
        + 0.5 * (redness_i + redness_c)
        - 0.5 * (greenness_i + blueness_i + greenness_c + blueness_c)
    )
    return np.clip(S_hyd, 0, None)


def compute_ice_score(r_i: np.ndarray, g_i: np.ndarray, b_i: np.ndarray,
                      r_c: np.ndarray, g_c: np.ndarray, b_c: np.ndarray) -> np.ndarray:
    """
    Compute ice score from ICE, IC2 browse maps.

    - Signal: greenness from ICE/IC2
    - Penalty: 0.5 * (redness + blueness from ICE/IC2)
    """
    redness_i, greenness_i, blueness_i = normalized_colors(r_i, g_i, b_i)
    redness_c, greenness_c, blueness_c = normalized_colors(r_c, g_c, b_c)

    S_ice = (
        (greenness_i + greenness_c)
        - 0.5 * (redness_i + blueness_i + redness_c + blueness_c)
    )
    return np.clip(S_ice, 0, None)


def compute_score_stats(score: np.ndarray, valid_mask: np.ndarray) -> dict:
    """
    Compute statistics for a score array.

    Returns dict with:
    - valid_pixels: count of valid pixels
    - max_score: maximum score value
    - mean_score: mean of valid pixels
    - threshold_counts: dict mapping threshold -> count of pixels >= threshold
    """
    valid_scores = score[valid_mask]

    if valid_scores.size == 0:
        return {
            "valid_pixels": 0,
            "max_score": 0.0,
            "mean_score": 0.0,
            "threshold_counts": {str(t): 0 for t in THRESHOLDS}
        }

    stats = {
        "valid_pixels": int(valid_scores.size),
        "max_score": float(valid_scores.max()),
        "mean_score": float(valid_scores.mean()),
        "threshold_counts": {}
    }

    # Precompute counts for each threshold
    for t in THRESHOLDS:
        count = int((valid_scores >= t).sum())
        stats["threshold_counts"][str(t)] = count

    return stats


def score_to_rgba(score: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """
    Convert score array to RGBA image using strict color mapping.

    Color mapping rules:
    - score <= 0:       white (RGB = [255, 255, 255])
    - 0 < score < 1:    grayscale (0 = white, 1 = black)
    - 1 <= score <= 2:  red scale (higher = darker/stronger red)
    - Background (invalid pixels): transparent

    This mapping is consistent for both ice and hydration scores.
    """
    H, W = score.shape
    rgba = np.zeros((H, W, 4), dtype=np.uint8)

    # Clamp score to [0, 2] range for visualization
    clamped = np.clip(score, 0, 2)

    # Create color arrays
    r = np.zeros((H, W), dtype=np.uint8)
    g = np.zeros((H, W), dtype=np.uint8)
    b = np.zeros((H, W), dtype=np.uint8)

    # Region 1: score <= 0 → white
    mask_zero = valid_mask & (score <= 0)
    r[mask_zero] = 255
    g[mask_zero] = 255
    b[mask_zero] = 255

    # Region 2: 0 < score < 1 → grayscale (white to black)
    # score=0 → 255 (white), score=1 → 0 (black)
    mask_gray = valid_mask & (score > 0) & (score < 1)
    gray_value = (255 * (1 - clamped)).astype(np.uint8)
    r[mask_gray] = gray_value[mask_gray]
    g[mask_gray] = gray_value[mask_gray]
    b[mask_gray] = gray_value[mask_gray]

    # Region 3: 1 <= score <= 2 → red scale (black to dark red)
    # score=1 → black (0,0,0), score=2 → dark red (180,0,0)
    mask_red = valid_mask & (score >= 1)
    red_intensity = ((clamped - 1) * 180).astype(np.uint8)  # 0 to 180
    r[mask_red] = red_intensity[mask_red]
    g[mask_red] = 0
    b[mask_red] = 0

    # Assemble RGBA
    rgba[..., 0] = r
    rgba[..., 1] = g
    rgba[..., 2] = b
    # Alpha: fully opaque for valid pixels, transparent for background
    rgba[..., 3] = np.where(valid_mask, 255, 0).astype(np.uint8)

    return rgba


def generate_score_maps(obs_id: str, force: bool = False) -> tuple[bool, bool, dict | None]:
    """
    Generate ice_score and hyd_score outputs for a given observation.

    Outputs:
    - PNG visualizations in BROWSE_DIR
    - NumPy arrays in SCORE_DIR/<obs_id>/

    Returns:
        (ice_generated, hyd_generated, stats) - tuple of booleans and stats dict
    """
    # Output paths
    ice_png = OUTPUT_DIR / f"{obs_id}_score_ice.png"
    hyd_png = OUTPUT_DIR / f"{obs_id}_score_hyd.png"

    score_obs_dir = SCORE_DIR / obs_id
    ice_npy = score_obs_dir / "ice_score.npy"
    hyd_npy = score_obs_dir / "hyd_score.npy"

    # Check cache (all files must exist for cache hit)
    all_exist = ice_png.exists() and hyd_png.exists() and ice_npy.exists() and hyd_npy.exists()

    if not force and all_exist:
        return (False, False, None)  # Cache hit, skip

    # Load browse maps
    try:
        hyd_path = BROWSE_DIR / f"{obs_id}_HYD.png"
        ice_path = BROWSE_DIR / f"{obs_id}_ICE.png"
        ic2_path = BROWSE_DIR / f"{obs_id}_IC2.png"

        r_h, g_h, b_h = load_png(hyd_path)
        r_i, g_i, b_i = load_png(ice_path)
        r_c, g_c, b_c = load_png(ic2_path)
    except FileNotFoundError as e:
        print(f"[SKIP] {obs_id} - missing browse file: {e}")
        return (False, False, None)
    except OSError as e:
        print(f"[SKIP] {obs_id} - corrupted browse file: {e}")
        return (False, False, None)

    # Create valid mask (background = black pixels)
    bg_mask = (r_h + g_h + b_h) == 0
    valid_mask = ~bg_mask

    # Compute scores
    S_ice = compute_ice_score(r_i, g_i, b_i, r_c, g_c, b_c)
    S_hyd = compute_hydration_score(r_h, g_h, b_h, r_i, g_i, b_i, r_c, g_c, b_c)

    # Create score directory for this observation
    score_obs_dir.mkdir(parents=True, exist_ok=True)

    # Save NumPy arrays
    np.save(ice_npy, S_ice.astype(np.float32))
    np.save(hyd_npy, S_hyd.astype(np.float32))

    # Also save the valid mask for filtering
    mask_npy = score_obs_dir / "valid_mask.npy"
    np.save(mask_npy, valid_mask)

    # Save PNGs
    rgba_ice = score_to_rgba(S_ice, valid_mask)
    rgba_hyd = score_to_rgba(S_hyd, valid_mask)
    Image.fromarray(rgba_ice, mode="RGBA").save(ice_png)
    Image.fromarray(rgba_hyd, mode="RGBA").save(hyd_png)

    # Compute statistics
    ice_stats = compute_score_stats(S_ice, valid_mask)
    hyd_stats = compute_score_stats(S_hyd, valid_mask)

    stats = {
        "obs_id": obs_id,
        "ice": ice_stats,
        "hyd": hyd_stats
    }

    return (True, True, stats)


def main():
    parser = argparse.ArgumentParser(description="Generate CRISM score maps and statistics")
    parser.add_argument("--force", action="store_true", help="Regenerate all (bypass cache)")
    parser.add_argument("--obs", type=str, help="Process specific observation ID only")
    args = parser.parse_args()

    if not BROWSE_DIR.exists():
        print(f"Error: Browse directory not found: {BROWSE_DIR}")
        sys.exit(1)

    # Ensure output directories exist
    SCORE_DIR.mkdir(parents=True, exist_ok=True)

    # Find all observations with HYD browse maps
    if args.obs:
        obs_ids = [args.obs]
    else:
        obs_ids = sorted({
            f.split("_")[0]
            for f in os.listdir(BROWSE_DIR)
            if f.endswith("_HYD.png")
        })

    print(f"[INFO] Processing {len(obs_ids)} observations")
    print(f"[INFO] PNG output: {OUTPUT_DIR}")
    print(f"[INFO] NPY output: {SCORE_DIR}")
    print(f"[INFO] Force regenerate: {args.force}")

    # Load existing stats if available
    all_stats = {}
    if STATS_FILE.exists() and not args.force:
        try:
            with open(STATS_FILE, "r") as f:
                all_stats = json.load(f)
            print(f"[INFO] Loaded existing stats for {len(all_stats)} observations")
        except (json.JSONDecodeError, OSError) as e:
            print(f"[WARN] Failed to load existing stats: {e}")

    gen_count = 0
    skip_count = 0

    for obs_id in obs_ids:
        ice_gen, hyd_gen, stats = generate_score_maps(obs_id, force=args.force)

        if ice_gen or hyd_gen:
            print(f"[GEN] {obs_id}")
            gen_count += 1
            if stats:
                all_stats[obs_id] = stats
        else:
            skip_count += 1

    # Save consolidated stats file
    with open(STATS_FILE, "w") as f:
        json.dump(all_stats, f, indent=2)

    print(f"\n[DONE] Generated: {gen_count} | Skipped (cached): {skip_count}")
    print(f"[INFO] Stats saved to: {STATS_FILE}")


if __name__ == "__main__":
    main()
