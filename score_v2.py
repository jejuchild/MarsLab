#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CRISM browse hydration / ice score statistics generator (v2)

Revision from score.py:
- REMOVED: dominance metric (max(r-g, r-b) style)
- REPLACED: normalized redness = r / (r + g + b + eps)
- CHANGED: penalty coefficient from 0.7 to 0.5
- KEPT: file I/O, masks, statistics structure unchanged

All color-based terms now use normalized color ratios instead of dominance.
"""

import os
import json
import numpy as np
from PIL import Image

# ============================================================
# CONFIG
# ============================================================

BROWSE_DIR = "./arcadia_browse/browse"
STATS_DIR  = "./arcadia_browse/stats"

N_OBS = 10000

# Score thresholds for product-level statistics
THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 1.2, 1.5, 1.8]

# Numerical stability constant
EPS = 1e-6

os.makedirs(STATS_DIR, exist_ok=True)

# ============================================================
# UTILITIES
# ============================================================

def load_png(path):
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr[..., 0], arr[..., 1], arr[..., 2]

def normalized_colors(r, g, b):
    """
    Compute normalized color ratios.

    Returns:
        redness:   r / (r + g + b + eps)
        greenness: g / (r + g + b + eps)
        blueness:  b / (r + g + b + eps)
    """
    total = r + g + b + EPS
    redness   = r / total
    greenness = g / total
    blueness  = b / total
    return redness, greenness, blueness

def compute_score_stats(S, valid_mask):
    valid_scores = S[valid_mask]

    stats = {
        "valid_pixel_count": int(valid_scores.size),
        "score_stats": {},
        "max_score": float(valid_scores.max()) if valid_scores.size > 0 else 0.0,
        "mean_score": float(valid_scores.mean()) if valid_scores.size > 0 else 0.0,
    }

    for t in THRESHOLDS:
        cnt = int((valid_scores >= t).sum())
        stats["score_stats"][str(t)] = {
            "count": cnt,
            "area_fraction": cnt / valid_scores.size if valid_scores.size > 0 else 0.0
        }

    return stats

# ============================================================
# MAIN
# ============================================================

def main():

    obs_ids = sorted({f.split("_")[0] for f in os.listdir(BROWSE_DIR)})
    obs_ids = obs_ids[:N_OBS]

    print(f"[INFO] Processing {len(obs_ids)} observations")

    for obs in obs_ids:

        try:
            r_h, g_h, b_h = load_png(f"{BROWSE_DIR}/{obs}_HYD.png")
            r_i, g_i, b_i = load_png(f"{BROWSE_DIR}/{obs}_ICE.png")
            r_c, g_c, b_c = load_png(f"{BROWSE_DIR}/{obs}_IC2.png")
        except FileNotFoundError:
            print(f"[SKIP] {obs} missing files")
            continue

        # ----------------------------------------------------
        # background / valid mask
        # ----------------------------------------------------
        bg_mask = (r_h + g_h + b_h) == 0
        valid_mask = ~bg_mask

        # ----------------------------------------------------
        # normalized color ratios (replaces dominance)
        # ----------------------------------------------------
        redness_h, greenness_h, blueness_h = normalized_colors(r_h, g_h, b_h)
        redness_i, greenness_i, blueness_i = normalized_colors(r_i, g_i, b_i)
        redness_c, greenness_c, blueness_c = normalized_colors(r_c, g_c, b_c)

        # ----------------------------------------------------
        # Hydration score
        # - Signal: redness from HYD + 0.5 * redness from ICE/IC2
        # - Penalty: 0.5 * (greenness + blueness from ICE/IC2)
        # ----------------------------------------------------
        S_hyd = (
            redness_h
            + 0.5 * (redness_i + redness_c)
            - 0.5 * (greenness_i + blueness_i + greenness_c + blueness_c)
        )
        S_hyd = np.clip(S_hyd, 0, None)

        # ----------------------------------------------------
        # Ice score
        # - Signal: greenness from ICE/IC2
        # - Penalty: 0.5 * (redness + blueness from ICE/IC2)
        # ----------------------------------------------------
        S_ice = (
            (greenness_i + greenness_c)
            - 0.5 * (redness_i + blueness_i + redness_c + blueness_c)
        )
        S_ice = np.clip(S_ice, 0, None)

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------
        hyd_stats = compute_score_stats(S_hyd, valid_mask)
        ice_stats = compute_score_stats(S_ice, valid_mask)

        out = {
            "obs_id": obs,
            "hydration": hyd_stats,
            "ice": ice_stats
        }

        out_path = f"{STATS_DIR}/{obs}_score_stats.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)

        print(f"[SAVE] {out_path}")

    print("[DONE] Score statistics generated")

if __name__ == "__main__":
    main()
