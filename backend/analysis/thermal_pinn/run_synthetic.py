#!/usr/bin/env python3
"""
Phase 1: Synthetic data validation for thermal PINN.

Generates a known k(z) profile, solves the heat equation numerically,
then trains the PINN to recover k(z) from surface temperature observations.

Usage:
    python -m backend.analysis.thermal_pinn.run_synthetic
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    from .synthetic import (
        SyntheticConfig,
        generate_synthetic_dataset,
        two_layer_k_profile,
        three_layer_k_profile,
    )
    from .pinn_model import PINNConfig, ThermalPINN, train_pinn, evaluate_pinn

    # ── Configuration ────────────────────────────────────────
    synth_config = SyntheticConfig(
        z_max=3.0,
        n_z=200,
        n_sols=20,
        dt_per_sol=144,
        T_mean=210.0,
        T_amp_diurnal=40.0,
        T_amp_seasonal=20.0,
    )

    pinn_config = PINNConfig(
        n_hidden_T=4,
        n_neurons_T=64,
        n_hidden_k=3,
        n_neurons_k=32,
        k_min=0.005,
        k_max=4.0,
        T_min=140.0,
        T_max=300.0,
        w_physics=1.0,
        w_data=10.0,
        w_k_smooth=0.01,
        lr=1e-3,
        n_epochs=5000,
        batch_colloc=2048,
        scheduler_step=1500,
        scheduler_gamma=0.5,
    )

    # ── Generate synthetic data ──────────────────────────────
    logger.info("=" * 60)
    logger.info("Phase 1: Synthetic PINN Validation")
    logger.info("=" * 60)

    logger.info("Generating two-layer synthetic data...")
    data = generate_synthetic_dataset(
        k_profile_fn=two_layer_k_profile,
        config=synth_config,
        n_surface_obs=500,
    )

    logger.info("Surface T range: %.1f - %.1f K",
                data["T_obs"].min(), data["T_obs"].max())
    logger.info("Ground truth k range: %.4f - %.4f W/m/K",
                data["k_true"].min(), data["k_true"].max())

    # ── Train PINN ───────────────────────────────────────────
    logger.info("Training PINN...")
    model = ThermalPINN(pinn_config)
    logger.info("Device: %s", model.device)
    logger.info("Parameters: %d",
                sum(p.numel() for p in model.parameters()))

    history = train_pinn(model, data, pinn_config)

    # ── Evaluate ─────────────────────────────────────────────
    logger.info("Evaluating...")
    results = evaluate_pinn(model, data)

    # ── Save results ─────────────────────────────────────────
    out_dir = Path(__file__).resolve().parents[2] / "data" / "thermal_pinn"
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    torch.save(model.state_dict(), out_dir / "pinn_synthetic_weights.pt")

    np.savez_compressed(
        out_dir / "pinn_synthetic_results.npz",
        z=results["z"],
        k_pred=results["k_pred"],
        k_true=results["k_true"],
        TI_pred=results["TI_pred"],
        TI_true=results["TI_true"],
        T_pred_surface=results["T_pred_surface"],
        T_true_surface=results["T_true_surface"],
        loss_total=np.array(history["loss_total"]),
        loss_physics=np.array(history["loss_physics"]),
        loss_data=np.array(history["loss_data"]),
    )

    logger.info("Results saved to %s", out_dir)
    logger.info("=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("  k(z) RMSE:     %.4f W/m/K", results["k_rmse"])
    logger.info("  Surface T RMSE: %.2f K", results["T_rmse"])
    logger.info("  TI predicted:   %.0f - %.0f tiu",
                results["TI_pred"].min(), results["TI_pred"].max())
    logger.info("  TI true:        %.0f - %.0f tiu",
                results["TI_true"].min(), results["TI_true"].max())
    logger.info("=" * 60)

    return results


if __name__ == "__main__":
    main()
