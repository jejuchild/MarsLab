#!/usr/bin/env python3
"""
Phase 1: Synthetic data validation for thermal PINN.
V2: Dual-coordinate time encoding with 1 Mars year simulation.

Generates a known k(z) profile, solves the heat equation numerically,
then trains the PINN to recover k(z) from surface temperature observations.

Usage:
    python -m backend.analysis.thermal_pinn.run_synthetic
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
)
logger = logging.getLogger(__name__)


def visualize_results(results: dict, data: dict, history: dict,
                      out_dir: Path) -> None:
    """Generate diagnostic plots and save as PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping visualization")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("PINN Synthetic Validation — k(z) Recovery", fontsize=14)

    # ── Top-left: k(z) comparison ────────────────────────
    ax = axes[0, 0]
    z = results["z"]
    ax.plot(z, results["k_true"], "b-", linewidth=2, label="k true")
    ax.plot(z, results["k_pred"], "r--", linewidth=2, label="k predicted")
    ax.set_xlabel("Depth [m]")
    ax.set_ylabel("k [W/m/K]")
    ax.set_title(f"Thermal Conductivity (RMSE={results['k_rmse']:.4f})")
    ax.legend()
    ax.set_xlim(0, 2.0)
    ax.grid(True, alpha=0.3)

    # Secondary y-axis for TI
    ax2 = ax.twinx()
    ax2.plot(z, results["TI_true"], "b:", alpha=0.4, linewidth=1)
    ax2.plot(z, results["TI_pred"], "r:", alpha=0.4, linewidth=1)
    ax2.set_ylabel("TI [tiu]", alpha=0.5)

    # ── Top-right: Surface T fit (3 sols detail) ─────────
    ax = axes[0, 1]
    t_sols = data["t_obs"] / 88_775.0  # convert to sols
    T_true = results["T_true_surface"]
    T_pred = results["T_pred_surface"]

    # Show first 3 sols
    mask_3sol = t_sols < 3.0
    if mask_3sol.sum() > 10:
        ax.plot(t_sols[mask_3sol], T_true[mask_3sol], "b.", markersize=2,
                label="True", alpha=0.6)
        ax.plot(t_sols[mask_3sol], T_pred[mask_3sol], "r.", markersize=2,
                label="Predicted", alpha=0.6)
        ax.set_xlabel("Time [sols]")
        ax.set_ylabel("T [K]")
        ax.set_title(f"Surface T — Diurnal (RMSE={results['T_rmse']:.2f} K)")
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "Insufficient data for 3-sol view",
                transform=ax.transAxes, ha="center")

    # ── Bottom-left: Surface T full year (seasonal) ──────
    ax = axes[1, 0]
    ax.plot(t_sols, T_true, "b.", markersize=1, alpha=0.3, label="True")
    ax.plot(t_sols, T_pred, "r.", markersize=1, alpha=0.3, label="Predicted")
    ax.set_xlabel("Time [sols]")
    ax.set_ylabel("T [K]")
    ax.set_title("Surface T — Full Year (seasonal)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Bottom-right: Loss curves ────────────────────────
    ax = axes[1, 1]
    epochs = np.arange(1, len(history["loss_total"]) + 1)
    ax.semilogy(epochs, history["loss_data"], "b-", alpha=0.7, label="Data loss")
    ax.semilogy(epochs, history["loss_physics"], "r-", alpha=0.7,
                label="Physics loss")
    ax.semilogy(epochs, history["loss_total"], "k-", alpha=0.3,
                label="Total loss")

    # Mark phase boundaries
    n_ep = len(epochs)
    p1 = int(0.25 * n_ep)
    p2 = int(0.60 * n_ep)
    ax.axvline(p1, color="gray", linestyle=":", alpha=0.5, label="P1→P2")
    ax.axvline(p2, color="gray", linestyle="--", alpha=0.5, label="P2→P3")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = out_dir / "pinn_synthetic_v2.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved visualization: %s", plot_path)


def main():
    from .synthetic import (
        SyntheticConfig,
        generate_synthetic_dataset,
        two_layer_k_profile,
    )
    from .pinn_model import PINNConfig, ThermalPINN, train_pinn, evaluate_pinn

    # ── Configuration ────────────────────────────────────────
    synth_config = SyntheticConfig(
        z_max=3.0,
        n_z=200,
        n_sols=668,             # 1 Mars year — full seasonal coverage
        dt_per_sol=144,
        spinup_sols=668,        # 1 year spinup for equilibration
        T_mean=210.0,
        T_amp_diurnal=40.0,
        T_amp_seasonal=20.0,
    )

    pinn_config = PINNConfig(
        # Architecture
        n_hidden_T=4,
        n_neurons_T=64,
        n_hidden_k=3,
        n_neurons_k=32,
        n_harmonics_diurnal=4,
        n_harmonics_seasonal=3,
        n_depth_modes=4,
        # Bounds
        k_min=0.005,
        k_max=4.0,
        T_min=140.0,
        T_max=300.0,
        # Loss weights
        w_physics=1.0,
        w_data=10.0,
        w_k_smooth=1e-4,       # very low — allow sharp transitions
        # Training
        lr=1e-3,
        n_epochs=6000,
        batch_colloc=4096,
        scheduler_step=2000,
        scheduler_gamma=0.5,
    )

    # ── Generate synthetic data ──────────────────────────────
    logger.info("=" * 60)
    logger.info("Phase 1: Synthetic PINN Validation (V2 — Dual Coords)")
    logger.info("=" * 60)

    t0 = time.time()
    logger.info("Generating two-layer synthetic data (668 sols + 668 spinup)...")
    data = generate_synthetic_dataset(
        k_profile_fn=two_layer_k_profile,
        config=synth_config,
        n_surface_obs=3000,
        n_colloc=8000,
    )
    t_gen = time.time() - t0

    logger.info("Data generation: %.1f seconds", t_gen)
    logger.info("Surface T range: %.1f - %.1f K",
                data["T_obs"].min(), data["T_obs"].max())
    logger.info("Ground truth k range: %.4f - %.4f W/m/K",
                data["k_true"].min(), data["k_true"].max())
    logger.info("Diurnal phase range: %.3f - %.3f",
                data["t_obs_diurnal"].min(), data["t_obs_diurnal"].max())
    logger.info("Seasonal phase range: %.3f - %.3f",
                data["t_obs_seasonal"].min(), data["t_obs_seasonal"].max())

    # ── Train PINN ───────────────────────────────────────────
    logger.info("\nTraining PINN (V2 — dual time coordinates)...")
    model = ThermalPINN(pinn_config)
    logger.info("Device: %s", model.device)
    logger.info("Parameters: %d",
                sum(p.numel() for p in model.parameters()))
    logger.info("T_net input dim: %d",
                1 + 2 * pinn_config.n_harmonics_diurnal
                + 2 * pinn_config.n_harmonics_seasonal
                + 2 * pinn_config.n_depth_modes)

    t0 = time.time()
    history = train_pinn(model, data, pinn_config)
    t_train = time.time() - t0
    logger.info("Training: %.1f seconds (%.1f ms/epoch)",
                t_train, 1000 * t_train / pinn_config.n_epochs)

    # ── Evaluate ─────────────────────────────────────────────
    logger.info("\nEvaluating...")
    results = evaluate_pinn(model, data)

    # ── Save results ─────────────────────────────────────────
    out_dir = Path(__file__).resolve().parents[2] / "data" / "thermal_pinn"
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    torch.save(model.state_dict(), out_dir / "pinn_synthetic_weights_v2.pt")

    np.savez_compressed(
        out_dir / "pinn_synthetic_results_v2.npz",
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

    # ── Visualization ────────────────────────────────────────
    visualize_results(results, data, history, out_dir)

    # ── Summary ──────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("  k(z) RMSE:      %.4f W/m/K", results["k_rmse"])
    logger.info("  Surface T RMSE: %.2f K", results["T_rmse"])
    logger.info("  TI predicted:   %.0f - %.0f tiu",
                results["TI_pred"].min(), results["TI_pred"].max())
    logger.info("  TI true:        %.0f - %.0f tiu",
                results["TI_true"].min(), results["TI_true"].max())
    logger.info("  Layer boundary: %.2f m (detected)",
                results["boundary_depth"])
    logger.info("  Time: %.0fs gen + %.0fs train = %.0fs total",
                t_gen, t_train, t_gen + t_train)
    logger.info("  Saved to: %s", out_dir)
    logger.info("=" * 60)

    return results


if __name__ == "__main__":
    main()
