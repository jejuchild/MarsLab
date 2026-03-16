#!/usr/bin/env python3
"""Run Pair D: RDR_000856 (2006) → RDR_064072 (2020) — primary RDR pair."""

import sys
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.analysis.sct_temporal.ortho_pipeline_v2 import (
    TEMPORAL_PAIRS, RESULTS_DIR, run_pair_analysis,
)
import numpy as np

if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(RESULTS_DIR / "pair_d_run.log"),
        ],
    )

    pair_d = next(p for p in TEMPORAL_PAIRS if p["name"] == "D_rdr_primary")

    print("=" * 60)
    print(f"Running Pair D: {pair_d['desc']}")
    print(f"  Ref: {pair_d['ref']} → Tgt: {pair_d['tgt']}")
    print(f"  Baseline: {pair_d['baseline_yr']:.2f} yr")
    print("=" * 60)

    result, disp_data = run_pair_analysis(pair_d)

    print("\n" + "=" * 60)
    print("PAIR D RESULTS (RDR↔RDR)")
    print("=" * 60)
    print(f"  Co-registration:")
    print(f"    Affine RMSE: {result.coreg['affine_rmse_px']:.3f} px ({result.coreg['affine_rmse_px']*0.25:.4f} m)")
    print(f"    Inliers: {result.coreg['n_inliers']}")
    print(f"    ECC applied: {result.coreg['ecc_applied']}")
    print(f"    ECC correction: {result.coreg['ecc_correction_px']:.3f} px")
    print(f"    Final RMSE: {result.coreg['final_rmse_px']:.3f} px")
    print(f"  Parallax:")
    print(f"    Correction applied: {result.parallax['correction_applied']}")
    print(f"    Emission angle: {result.parallax['emission_angle_deg']:.2f}°")
    print(f"    Max displacement: {result.parallax['max_displacement_m']:.2f} m")
    print(f"  Displacement:")
    print(f"    Noise floor: {result.noise_floor_m:.4f} m")
    print(f"    Scarp excess: {result.scarp_excess_m:.4f} m")
    print(f"    Retreat rate: {result.retreat_rate_m_yr:.4f} m/yr")
    print(f"    Upper bound: {result.retreat_rate_upper_bound_m_yr:.4f} m/yr")

    pair_dir = RESULTS_DIR / "pair_D"
    pair_dir.mkdir(exist_ok=True)
    np.savez_compressed(
        pair_dir / "displacement.npz",
        magnitude_m=disp_data["magnitude_m"],
        row_disp_m=disp_data["row_disp_m"],
        col_disp_m=disp_data["col_disp_m"],
        snr=disp_data["snr"],
        valid_mask=disp_data["valid_mask"],
    )
    print(f"\nDisplacement saved to {pair_dir}/displacement.npz")
