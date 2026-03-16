#!/usr/bin/env python3
"""Run all three RDR pairs: D (primary), E (validation), F (null test)."""

import sys
import logging
import json
from pathlib import Path
from dataclasses import asdict

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.analysis.sct_temporal.ortho_pipeline_v2 import (
    TEMPORAL_PAIRS, RESULTS_DIR, run_pair_analysis, IMAGE_CATALOG,
    cross_validate_pairs,
)

if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(RESULTS_DIR / "all_rdr_pairs.log"),
        ],
    )

    results = {}
    disp_data_all = {}

    for pair_cfg in TEMPORAL_PAIRS:
        name = pair_cfg["name"]
        ref_info = IMAGE_CATALOG.get(pair_cfg["ref"])
        tgt_info = IMAGE_CATALOG.get(pair_cfg["tgt"])

        if ref_info is None or tgt_info is None:
            print(f"SKIP {name}: unknown image key")
            continue
        if not ref_info.path.exists() or not tgt_info.path.exists():
            print(f"SKIP {name}: missing file")
            continue

        print(f"\n{'=' * 70}")
        print(f"  {name}: {pair_cfg['desc']}")
        print(f"{'=' * 70}")

        try:
            result, disp_data = run_pair_analysis(pair_cfg)
            results[name] = result
            disp_data_all[name] = disp_data

            pair_dir = RESULTS_DIR / name
            pair_dir.mkdir(exist_ok=True)
            np.savez_compressed(
                pair_dir / "displacement.npz",
                magnitude_m=disp_data["magnitude_m"],
                row_disp_m=disp_data["row_disp_m"],
                col_disp_m=disp_data["col_disp_m"],
                snr=disp_data["snr"],
                valid_mask=disp_data["valid_mask"],
            )
        except Exception as e:
            print(f"FAILED {name}: {e}")
            import traceback
            traceback.print_exc()

    # Cross-validation
    if "D_rdr_primary" in disp_data_all and "E_rdr_validation" in disp_data_all:
        da = disp_data_all["D_rdr_primary"]
        db = disp_data_all["E_rdr_validation"]
        min_r = min(da["magnitude_m"].shape[0], db["magnitude_m"].shape[0])
        min_c = min(da["magnitude_m"].shape[1], db["magnitude_m"].shape[1])
        crossval = cross_validate_pairs(
            da["magnitude_m"][:min_r, :min_c],
            db["magnitude_m"][:min_r, :min_c],
            da["valid_mask"][:min_r, :min_c],
            db["valid_mask"][:min_r, :min_c],
        )
        print(f"\nCross-validation D vs E: R²={crossval.r_squared:.4f}, RMSE={crossval.rmse_m:.4f}m, n={crossval.n_common}")
    else:
        crossval = None

    # Summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    for name, r in results.items():
        print(f"  {name:25s}  noise={r.noise_floor_m:.4f}m  excess={r.scarp_excess_m:.4f}m  "
              f"rate={r.retreat_rate_m_yr:.4f}m/yr  upper={r.retreat_rate_upper_bound_m_yr:.4f}m/yr  "
              f"inliers={r.coreg['n_inliers']}")

    null_r = results.get("F_rdr_null")
    if null_r:
        passed = abs(null_r.scarp_excess_m) < 2 * null_r.noise_floor_m if not np.isnan(null_r.scarp_excess_m) else False
        print(f"\n  Null test: {'PASS ✓' if passed else 'FAIL ✗'} (excess={null_r.scarp_excess_m:.4f}m vs 2σ={2*null_r.noise_floor_m:.4f}m)")

    if crossval:
        print(f"  Cross-validation: R²={crossval.r_squared:.4f}")

    primary = results.get("D_rdr_primary")
    if primary:
        if crossval and crossval.r_squared > 0.3 and primary.scarp_excess_m > 2 * primary.noise_floor_m:
            verdict = "POSITIVE_DETECTION"
        elif primary.scarp_excess_m > primary.noise_floor_m:
            verdict = "MARGINAL_DETECTION"
        else:
            verdict = "NO_DETECTION — UPPER_BOUND"
        print(f"\n  VERDICT: {verdict}")
        print(f"  Retreat rate upper bound: {primary.retreat_rate_upper_bound_m_yr:.4f} m/yr = {primary.retreat_rate_upper_bound_m_yr*100:.2f} cm/yr")
