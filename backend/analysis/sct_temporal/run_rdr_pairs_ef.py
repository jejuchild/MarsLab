#!/usr/bin/env python3
"""Run Pair E (validation) and Pair F (null test) — RDR↔RDR."""

import sys
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.analysis.sct_temporal.ortho_pipeline_v2 import (
    TEMPORAL_PAIRS, RESULTS_DIR, run_pair_analysis, IMAGE_CATALOG,
)
import numpy as np

def run_named_pair(pair_name):
    pair_cfg = next(p for p in TEMPORAL_PAIRS if p["name"] == pair_name)
    
    # Check files exist
    for key in [pair_cfg["ref"], pair_cfg["tgt"]]:
        info = IMAGE_CATALOG[key]
        if not info.path.exists():
            print(f"SKIPPING {pair_name}: {key} not found at {info.path}")
            return None, None
    
    print(f"\n{'=' * 60}")
    print(f"Running {pair_name}: {pair_cfg['desc']}")
    print(f"  Ref: {pair_cfg['ref']} → Tgt: {pair_cfg['tgt']}")
    print(f"  Baseline: {pair_cfg['baseline_yr']:.2f} yr")
    print(f"{'=' * 60}")
    
    result, disp_data = run_pair_analysis(pair_cfg)
    
    print(f"\n--- {pair_name} Results ---")
    print(f"  Co-reg: RMSE={result.coreg['affine_rmse_px']:.3f}px, inliers={result.coreg['n_inliers']}, ECC={result.coreg['ecc_correction_px']:.3f}px")
    print(f"  Noise floor: {result.noise_floor_m:.4f} m")
    print(f"  Scarp excess: {result.scarp_excess_m:.4f} m")
    print(f"  Retreat rate: {result.retreat_rate_m_yr:.4f} m/yr")
    print(f"  Upper bound: {result.retreat_rate_upper_bound_m_yr:.4f} m/yr")
    
    pair_dir = RESULTS_DIR / pair_name
    pair_dir.mkdir(exist_ok=True)
    np.savez_compressed(
        pair_dir / "displacement.npz",
        magnitude_m=disp_data["magnitude_m"],
        row_disp_m=disp_data["row_disp_m"],
        col_disp_m=disp_data["col_disp_m"],
        snr=disp_data["snr"],
        valid_mask=disp_data["valid_mask"],
    )
    return result, disp_data


if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(RESULTS_DIR / "pairs_ef_run.log"),
        ],
    )

    # Pair E: validation
    result_e, disp_e = run_named_pair("E_rdr_validation")
    
    # Pair F: null test
    result_f, disp_f = run_named_pair("F_rdr_null")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if result_e:
        print(f"Pair E (validation): noise={result_e.noise_floor_m:.4f}m, excess={result_e.scarp_excess_m:.4f}m, rate={result_e.retreat_rate_m_yr:.4f}m/yr")
    if result_f:
        print(f"Pair F (null test):  noise={result_f.noise_floor_m:.4f}m, excess={result_f.scarp_excess_m:.4f}m")
        passed = abs(result_f.scarp_excess_m) < 2 * result_f.noise_floor_m
        print(f"  Null test: {'PASS' if passed else 'FAIL'}")
