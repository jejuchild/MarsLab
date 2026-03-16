#!/usr/bin/env python3
"""Run only Pair B (ORTHO_002439 → ESP_064072) with fixed co-registration."""

import sys
import logging
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.analysis.sct_temporal.ortho_pipeline_v2 import (
    TEMPORAL_PAIRS, RESULTS_DIR, run_pair_analysis,
)

if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(RESULTS_DIR / "pair_b_rerun.log"),
        ],
    )

    # Find Pair B config
    pair_b = next(p for p in TEMPORAL_PAIRS if p["name"] == "B_validation")

    print("=" * 60)
    print("Re-running Pair B with fixed co-registration")
    print(f"  Ref: {pair_b['ref']} → Tgt: {pair_b['tgt']}")
    print(f"  Baseline: {pair_b['baseline_yr']:.2f} yr")
    print("=" * 60)

    result, disp_data = run_pair_analysis(pair_b)

    print("\n" + "=" * 60)
    print("PAIR B RESULTS")
    print("=" * 60)
    print(f"  Co-registration:")
    print(f"    Affine RMSE: {result.coreg['affine_rmse_px']:.3f} px")
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

    import numpy as np
    pair_dir = RESULTS_DIR / "pair_B"
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
