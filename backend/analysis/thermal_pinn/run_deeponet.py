#!/usr/bin/env python3
"""
DeepONet surrogate: train and run inversion, compare with FDM results.

Pipeline:
    1. Generate training data (FDM forward runs with random k_params)
    2. Train DeepONet surrogate model
    3. Run inversion at target location using DeepONet
    4. Compare with existing FDM inversion results

Usage:
    python -m backend.analysis.thermal_pinn.run_deeponet
    python -m backend.analysis.thermal_pinn.run_deeponet --n-samples 5000 --epochs 200
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from backend.analysis.thermal_pinn.deeponet_surrogate import (
    DeepONetConfig,
    DeepONet,
    generate_training_data,
    train_deeponet,
    run_inversion_deeponet,
)
from backend.analysis.thermal_pinn.run_real import (
    extract_observations,
    load_swim_ct,
    k_to_ct_score,
)
from backend.analysis.thermal_pinn.pinn_model import RHO, C_P

logger = logging.getLogger(__name__)

DATA_DIR = Path("backend/data/thermal_pinn")
HIGHLAT_DIR = Path("backend/data/themis_irbtr_highlat")


def plot_comparison(don_result, fdm_results_path, observations, swim_ct,
                    target_lat, target_lon, save_path):
    """6-panel comparison plot: DeepONet vs FDM inversion."""
    # Load FDM results
    fdm = np.load(fdm_results_path, allow_pickle=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    tag = f"{target_lat:.0f}°N, {target_lon:.0f}°E"
    fig.suptitle(f"DeepONet vs FDM Inversion — {tag}", fontsize=14, fontweight="bold")

    # Panel 1: k(z) profiles comparison
    ax = axes[0, 0]
    z = fdm["z"]
    k_fdm = fdm["k_pred"]
    # Reconstruct DeepONet k(z)
    from backend.analysis.thermal_pinn.deeponet_surrogate import k_z_from_params
    k_don = k_z_from_params(don_result.k_upper, don_result.k_lower,
                            don_result.boundary, don_result.width, z)
    ax.semilogy(z, k_fdm, "b-", lw=2, label="FDM inversion")
    ax.semilogy(z, k_don, "r--", lw=2, label="DeepONet inversion")
    ax.set_xlabel("Depth (m)")
    ax.set_ylabel("k (W/m/K)")
    ax.set_title("Conductivity Profile Comparison")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: Parameter comparison bar chart
    ax = axes[0, 1]
    params = ["k_upper", "k_lower", "boundary"]
    fdm_vals = [float(fdm["k_upper"]), float(fdm["k_lower"]), float(fdm["boundary"])]
    don_vals = [don_result.k_upper, don_result.k_lower, don_result.boundary]
    x = np.arange(len(params))
    w = 0.35
    bars1 = ax.bar(x - w/2, fdm_vals, w, label="FDM", color="#42a5f5")
    bars2 = ax.bar(x + w/2, don_vals, w, label="DeepONet", color="#ef5350")
    ax.set_xticks(x)
    ax.set_xticklabels(params)
    ax.set_ylabel("Value")
    ax.set_title("Parameter Comparison")
    ax.legend()
    ax.set_yscale("log")
    # Add value labels
    for bar, val in zip(bars1, fdm_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                f"{val:.4f}", ha="center", fontsize=7)
    for bar, val in zip(bars2, don_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                f"{val:.4f}", ha="center", fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 3: TI comparison
    ax = axes[0, 2]
    ti_fdm = [float(fdm["TI_upper"]), float(fdm["TI_lower"])]
    ti_don = [don_result.TI_upper, don_result.TI_lower]
    labels = ["TI_upper", "TI_lower"]
    x = np.arange(len(labels))
    ax.bar(x - w/2, ti_fdm, w, label="FDM", color="#42a5f5")
    ax.bar(x + w/2, ti_don, w, label="DeepONet", color="#ef5350")
    for xi, v1, v2 in zip(x, ti_fdm, ti_don):
        ax.text(xi - w/2, v1 + 20, f"{v1:.0f}", ha="center", fontsize=9)
        ax.text(xi + w/2, v2 + 20, f"{v2:.0f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Thermal Inertia (tiu)")
    ax.set_title("Thermal Inertia Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 4: DeepONet loss convergence
    ax = axes[1, 0]
    ax.semilogy(don_result.loss_history, "r-", lw=1)
    ax.set_xlabel("Step")
    ax.set_ylabel("MSE Loss (K²)")
    ax.set_title(f"DeepONet Inversion Loss\n(final={don_result.loss_final:.1f} K², "
                 f"{don_result.elapsed:.1f}s)")
    ax.grid(True, alpha=0.3)

    # Panel 5: Observations
    ax = axes[1, 1]
    night_obs = [o for o in observations if o["is_night"]]
    day_obs = [o for o in observations if not o["is_night"]]
    if night_obs:
        ax.scatter([o["solar_lon"] for o in night_obs],
                   [o["bt_kelvin"] for o in night_obs],
                   c="blue", s=30, label=f"Night ({len(night_obs)})")
    if day_obs:
        ax.scatter([o["solar_lon"] for o in day_obs],
                   [o["bt_kelvin"] for o in day_obs],
                   c="red", s=30, label=f"Day ({len(day_obs)})")
    ax.set_xlabel("Ls (°)")
    ax.set_ylabel("T_surface (K)")
    ax.set_title("THEMIS Observations")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 6: CT comparison
    ax = axes[1, 2]
    ct_don = k_to_ct_score(don_result.k_upper, don_result.k_lower, don_result.boundary)
    ct_fdm = k_to_ct_score(float(fdm["k_upper"]), float(fdm["k_lower"]),
                           float(fdm["boundary"]))
    bars_labels = ["SWIM CT"]
    bars_vals = [swim_ct if swim_ct is not None else 0]
    bars_colors = ["#66bb6a"]
    bars_labels.extend(["FDM CT", "DeepONet CT"])
    bars_vals.extend([ct_fdm, ct_don])
    bars_colors.extend(["#42a5f5", "#ef5350"])

    ax.barh(bars_labels, bars_vals, color=bars_colors, height=0.5)
    ax.set_xlim(-1, 1)
    ax.axvline(0, color="k", lw=0.5)
    ax.axvline(0.3, color="g", ls=":", alpha=0.5)
    ax.set_xlabel("Consistency Score")
    ax.set_title("CT Thermal Consistency")
    for i, v in enumerate(bars_vals):
        ax.text(v + 0.05 if v >= 0 else v - 0.15, i, f"{v:.3f}",
                va="center", fontsize=11)
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved comparison plot → %s", save_path)


def main():
    parser = argparse.ArgumentParser(description="DeepONet surrogate training & inversion")
    parser.add_argument("--target-lat", type=float, default=57.0)
    parser.add_argument("--target-lon", type=float, default=175.0)
    parser.add_argument("--n-samples", type=int, default=25000)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--inv-steps", type=int, default=500)
    parser.add_argument("--inv-lr", type=float, default=0.03)
    parser.add_argument("--data-dir", type=str, default=str(HIGHLAT_DIR))
    parser.add_argument("--skip-datagen", action="store_true",
                        help="Skip data generation, load from cache")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training, load saved model")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    tag = f"{int(args.target_lat)}N_{int(args.target_lon)}E"
    dataset_path = DATA_DIR / "deeponet_training_data.npz"
    model_path = DATA_DIR / "deeponet_model.pt"

    don_cfg = DeepONetConfig(
        n_samples=args.n_samples,
        epochs=args.epochs,
    )

    # ── Step 1: Generate training data ──
    logger.info("=" * 60)
    logger.info("STEP 1: Training Data Generation")
    logger.info("=" * 60)

    if args.skip_datagen and dataset_path.exists():
        logger.info("Loading cached training data from %s", dataset_path)
        cached = np.load(dataset_path, allow_pickle=True)
        data = {k: cached[k] for k in cached.files}
    else:
        t0 = time.time()
        data = generate_training_data(don_cfg)
        elapsed = time.time() - t0
        logger.info("Data generation: %.1fs", elapsed)

        # Cache to disk
        np.savez_compressed(
            dataset_path,
            branch_data=data["branch_data"],
            query_T=data["query_T"],
            trunk_enc=data["trunk_enc"],
            query_Ls=data["query_Ls"],
            query_lt=data["query_lt"],
        )
        logger.info("Saved training data → %s (%.1f MB)",
                    dataset_path, dataset_path.stat().st_size / 1e6)

    logger.info("  Samples: %d", data["branch_data"].shape[0])
    logger.info("  Query points per sample: %d", data["query_T"].shape[1])
    logger.info("  T range: %.1f – %.1f K",
                data["query_T"].min(), data["query_T"].max())

    # ── Step 2: Train DeepONet ──
    logger.info("=" * 60)
    logger.info("STEP 2: DeepONet Training")
    logger.info("=" * 60)

    if args.skip_training and model_path.exists():
        logger.info("Loading saved model from %s", model_path)
        model = DeepONet(
            branch_dim=don_cfg.branch_dim, trunk_dim=don_cfg.trunk_dim,
            hidden_dim=don_cfg.hidden_dim, latent_dim=don_cfg.latent_dim,
            n_layers=don_cfg.n_layers,
        )
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
        train_hist = {"train_loss": []}
    else:
        model, train_hist = train_deeponet(data, don_cfg)
        torch.save(model.state_dict(), model_path)
        logger.info("Saved model → %s", model_path)

    # ── Step 3: Extract observations ──
    logger.info("=" * 60)
    logger.info("STEP 3: Extract Observations")
    logger.info("=" * 60)

    data_dir = Path(args.data_dir)
    observations = extract_observations(
        args.target_lat, args.target_lon, data_dir=data_dir,
    )
    logger.info("Extracted %d observations", len(observations))

    # ── Step 4: DeepONet Inversion ──
    logger.info("=" * 60)
    logger.info("STEP 4: DeepONet Inversion")
    logger.info("=" * 60)

    don_result = run_inversion_deeponet(
        observations, model, args.target_lat,
        cfg=don_cfg, n_steps=args.inv_steps, lr=args.inv_lr,
    )

    # CT score
    ct_don = k_to_ct_score(don_result.k_upper, don_result.k_lower, don_result.boundary)
    swim_ct = load_swim_ct(args.target_lat, args.target_lon)

    # ── Step 5: Compare with FDM ──
    logger.info("=" * 60)
    logger.info("STEP 5: Comparison with FDM Inversion")
    logger.info("=" * 60)

    fdm_path = DATA_DIR / f"real_ebbc_results_{tag}.npz"
    if fdm_path.exists():
        fdm = np.load(fdm_path, allow_pickle=True)

        logger.info("")
        logger.info("%-25s %-18s %-18s %-10s", "", "FDM", "DeepONet", "Diff %")
        logger.info("-" * 75)
        for name, fdm_v, don_v in [
            ("k_upper (W/m/K)", float(fdm["k_upper"]), don_result.k_upper),
            ("k_lower (W/m/K)", float(fdm["k_lower"]), don_result.k_lower),
            ("boundary (m)", float(fdm["boundary"]), don_result.boundary),
            ("TI_upper (tiu)", float(fdm["TI_upper"]), don_result.TI_upper),
            ("TI_lower (tiu)", float(fdm["TI_lower"]), don_result.TI_lower),
        ]:
            diff_pct = abs(fdm_v - don_v) / max(abs(fdm_v), 1e-10) * 100
            logger.info("%-25s %-18.5f %-18.5f %.1f%%", name, fdm_v, don_v, diff_pct)

        ct_fdm = k_to_ct_score(float(fdm["k_upper"]), float(fdm["k_lower"]),
                               float(fdm["boundary"]))
        logger.info("%-25s %-18.3f %-18.3f", "Our CT score", ct_fdm, ct_don)
        if swim_ct is not None:
            logger.info("%-25s %-18.3f", "SWIM CT score", swim_ct)

        fdm_time = float(fdm["elapsed"])
        logger.info("")
        logger.info("Time: FDM=%.0fs, DeepONet=%.2fs → %.0fx speedup",
                    fdm_time, don_result.elapsed, fdm_time / don_result.elapsed)

        # Plot comparison
        plot_path = DATA_DIR / f"deeponet_vs_fdm_{tag}.png"
        plot_comparison(don_result, fdm_path, observations, swim_ct,
                        args.target_lat, args.target_lon, plot_path)
    else:
        logger.warning("No FDM results at %s — skipping comparison", fdm_path)
        logger.info("DeepONet result: k_upper=%.5f, k_lower=%.4f, boundary=%.3f",
                    don_result.k_upper, don_result.k_lower, don_result.boundary)
        logger.info("Our CT=%.3f, SWIM CT=%s", ct_don,
                    f"{swim_ct:.3f}" if swim_ct else "N/A")

    # ── Save DeepONet results ──
    results_path = DATA_DIR / f"deeponet_results_{tag}.npz"
    np.savez(
        results_path,
        k_upper=don_result.k_upper, k_lower=don_result.k_lower,
        boundary=don_result.boundary, width=don_result.width,
        TI_upper=don_result.TI_upper, TI_lower=don_result.TI_lower,
        loss_history=np.array(don_result.loss_history),
        elapsed=don_result.elapsed,
        our_ct=ct_don, swim_ct=swim_ct if swim_ct else np.nan,
    )
    logger.info("Saved DeepONet results → %s", results_path)
    logger.info("=" * 60)
    logger.info("DONE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
