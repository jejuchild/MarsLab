#!/usr/bin/env python3
"""
V5 Differentiable FDM — Synthetic Validation Runner.

Generates high-resolution ground truth with Crank-Nicolson FDM, then runs the
differentiable FDM inversion to recover k(z) from subsurface observations.

Usage:
    python -m backend.analysis.thermal_pinn.run_synthetic          # default 300 steps
    python -m backend.analysis.thermal_pinn.run_synthetic --steps 500
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from backend.analysis.thermal_pinn.synthetic import (
    SyntheticConfig,
    two_layer_k_profile,
    solve_heat_equation,
)
from backend.analysis.thermal_pinn.pinn_model import (
    InversionConfig,
    ParametricConductivity,
    run_inversion,
    evaluate_inversion,
    MARS_SOL,
)

logger = logging.getLogger(__name__)
OUT_DIR = Path("backend/data/thermal_pinn")

# ── Ground truth parameters ─────────────────────────────────────────
K_UPPER = 0.02      # W/m/K — dust
K_LOWER = 2.0       # W/m/K — ice
BOUNDARY = 0.5      # m
TRANS_WIDTH = 0.05   # m


def generate_truth(cfg_inv: InversionConfig) -> dict:
    """Generate high-res truth, then extract observations on the inversion grid."""

    # High-res FDM truth (nz=200, 144 steps/sol) — much finer than inversion grid
    cfg_syn = SyntheticConfig(
        z_max=cfg_inv.z_max,
        n_z=200,
        n_sols=cfg_inv.n_sols,
        dt_per_sol=144,
        spinup_sols=cfg_inv.spinup_sols,
        T_mean=cfg_inv.T_mean,
        T_amp_diurnal=cfg_inv.T_amp_diurnal,
        T_amp_seasonal=cfg_inv.T_amp_seasonal,
    )

    z_hires = np.linspace(0, cfg_inv.z_max, cfg_syn.n_z)
    k_true_hires = two_layer_k_profile(
        z_hires, K_UPPER, K_LOWER, BOUNDARY, TRANS_WIDTH
    )

    logger.info("Generating high-res truth: nz=%d, %d+%d sols ...",
                cfg_syn.n_z, cfg_syn.n_sols, cfg_syn.spinup_sols)
    T_full, z_syn, t_syn = solve_heat_equation(k_true_hires, cfg_syn)
    logger.info("Truth T range: %.1f – %.1f K, shape=%s",
                T_full.min(), T_full.max(), T_full.shape)

    # Inversion grid (coarse)
    z_inv = np.linspace(0, cfg_inv.z_max, cfg_inv.nz)
    k_true_inv = two_layer_k_profile(z_inv, K_UPPER, K_LOWER, BOUNDARY, TRANS_WIDTH)

    # Map inversion depth indices to hires indices for interpolation
    depth_hires_idx = [np.argmin(np.abs(z_hires - zi)) for zi in z_inv]

    # Inversion time grid
    dt_inv = MARS_SOL / cfg_inv.dt_per_sol
    dt_syn = MARS_SOL / cfg_syn.dt_per_sol
    n_out_inv = cfg_inv.n_sols * cfg_inv.dt_per_sol

    # Select observation depths: dense near surface + sparse deep
    # k_upper is only constrained by shallow nodes (skin depth ~1.7cm for dust)
    # so we oversample the top ~0.3m, then sparse deeper
    shallow_nodes = list(range(1, min(5, cfg_inv.nz)))  # z ≈ 0.08, 0.15, 0.23, 0.31m
    deep_nodes = list(range(5, cfg_inv.nz, cfg_inv.obs_subsample))  # every 6th after that
    obs_depth_indices = sorted(set(shallow_nodes + deep_nodes))
    logger.info("Observation depths: %d nodes at z = %s m",
                len(obs_depth_indices),
                [f"{z_inv[i]:.2f}" for i in obs_depth_indices])

    # Select observation times: subsample output timesteps
    # Take every 4th timestep to reduce compute while maintaining coverage
    time_subsample = 4
    obs_time_indices = list(range(0, n_out_inv, time_subsample))
    logger.info("Observation times: %d of %d output steps",
                len(obs_time_indices), n_out_inv)

    # Extract T_obs by interpolating from hires truth
    T_obs_list = []
    for t_idx_inv in obs_time_indices:
        # Map inversion output time to hires output time index
        t_seconds = t_idx_inv * dt_inv
        t_idx_hires = int(round(t_seconds / dt_syn))
        t_idx_hires = min(t_idx_hires, T_full.shape[0] - 1)

        row = []
        for d_idx_inv in obs_depth_indices:
            d_idx_hires = depth_hires_idx[d_idx_inv]
            row.append(T_full[t_idx_hires, d_idx_hires])
        T_obs_list.append(row)

    T_obs_np = np.array(T_obs_list, dtype=np.float64)  # (n_obs_times, n_depths)
    T_obs = torch.tensor(T_obs_np, dtype=torch.float64)

    logger.info("T_obs shape: %s, range: %.1f – %.1f K",
                T_obs.shape, T_obs_np.min(), T_obs_np.max())

    return {
        "T_obs": T_obs,
        "obs_time_indices": obs_time_indices,
        "obs_depth_indices": obs_depth_indices,
        "z_inv": z_inv,
        "k_true_inv": k_true_inv,
        "z_hires": z_hires,
        "k_true_hires": k_true_hires,
        "T_full": T_full,
        "t_syn": t_syn,
    }


def plot_results(truth: dict, results: dict, hist: dict, cfg: InversionConfig,
                 elapsed: float, save_path: Path):
    """4-panel figure: k(z), loss, parameter trajectories, T residual."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"V5 Differentiable FDM — Synthetic Validation\n"
        f"nz={cfg.nz}, {cfg.spinup_sols}+{cfg.n_sols} sols, "
        f"{cfg.n_steps} steps, {elapsed:.0f}s",
        fontsize=13, fontweight="bold",
    )

    # ── Panel 1: k(z) true vs recovered ──
    ax = axes[0, 0]
    z = results["z"]
    ax.semilogy(z, results["k_true"], "k-", lw=2, label="True k(z)")
    ax.semilogy(z, results["k_pred"], "r--", lw=2, label="Recovered k(z)")
    ax.axhline(K_UPPER, color="blue", ls=":", alpha=0.5, label=f"k_upper={K_UPPER}")
    ax.axhline(K_LOWER, color="green", ls=":", alpha=0.5, label=f"k_lower={K_LOWER}")
    ax.axvline(BOUNDARY, color="gray", ls=":", alpha=0.5, label=f"boundary={BOUNDARY}m")
    ax.set_xlabel("Depth (m)")
    ax.set_ylabel("k (W/m/K)")
    ax.set_title(f"Conductivity Profile — RMSE={results['k_rmse']:.4f}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 2: Loss convergence ──
    ax = axes[0, 1]
    ax.semilogy(hist["loss"], "b-", lw=1)
    ax.set_xlabel("Optimization Step")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Loss Convergence")
    ax.grid(True, alpha=0.3)

    # ── Panel 3: Parameter trajectories ──
    ax = axes[1, 0]
    steps = np.arange(len(hist["k_upper"]))

    ax.plot(steps, hist["k_upper"], "b-", label=f"k_upper (true={K_UPPER})")
    ax.axhline(K_UPPER, color="b", ls=":", alpha=0.5)

    ax2 = ax.twinx()
    ax2.plot(steps, hist["k_lower"], "r-", label=f"k_lower (true={K_LOWER})")
    ax2.axhline(K_LOWER, color="r", ls=":", alpha=0.5)
    ax2.set_ylabel("k_lower (W/m/K)", color="r")

    ax.set_xlabel("Step")
    ax.set_ylabel("k_upper (W/m/K)", color="b")
    ax.set_title("Parameter Convergence")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="center right")
    ax.grid(True, alpha=0.3)

    # ── Panel 4: Boundary & width trajectories ──
    ax = axes[1, 1]
    ax.plot(steps, hist["boundary"], "g-", label=f"boundary (true={BOUNDARY}m)")
    ax.axhline(BOUNDARY, color="g", ls=":", alpha=0.5)

    ax2 = ax.twinx()
    ax2.plot(steps, hist["width"], "m-", label=f"width (true={TRANS_WIDTH}m)")
    ax2.axhline(TRANS_WIDTH, color="m", ls=":", alpha=0.5)
    ax2.set_ylabel("width (m)", color="m")

    ax.set_xlabel("Step")
    ax.set_ylabel("boundary depth (m)", color="g")
    ax.set_title("Boundary & Width Convergence")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="center right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved plot → %s", save_path)


def main():
    parser = argparse.ArgumentParser(description="V5 Differentiable FDM synthetic validation")
    parser.add_argument("--steps", type=int, default=300, help="Optimization steps")
    parser.add_argument("--nz", type=int, default=40, help="Depth grid nodes")
    parser.add_argument("--n-sols", type=int, default=50, help="Output sols")
    parser.add_argument("--spinup-sols", type=int, default=50, help="Spinup sols")
    parser.add_argument("--lr", type=float, default=0.03, help="Learning rate")
    parser.add_argument("--z-max", type=float, default=3.0, help="Max depth (m)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg = InversionConfig(
        nz=args.nz,
        z_max=args.z_max,
        n_sols=args.n_sols,
        spinup_sols=args.spinup_sols,
        n_steps=args.steps,
        lr=args.lr,
    )

    logger.info("=" * 60)
    logger.info("V5 DIFFERENTIABLE FDM — SYNTHETIC VALIDATION")
    logger.info("=" * 60)
    logger.info("Config: nz=%d z_max=%.1f n_sols=%d spinup=%d steps=%d lr=%.4f",
                cfg.nz, cfg.z_max, cfg.n_sols, cfg.spinup_sols, cfg.n_steps, cfg.lr)
    logger.info("Ground truth: k_upper=%.3f k_lower=%.1f boundary=%.2fm width=%.3fm",
                K_UPPER, K_LOWER, BOUNDARY, TRANS_WIDTH)

    # Step 1: Generate truth
    truth = generate_truth(cfg)

    # Step 2: Run inversion
    logger.info("-" * 60)
    logger.info("Starting inversion ...")
    t0 = time.time()

    k_model, hist = run_inversion(
        T_obs=truth["T_obs"],
        obs_time_indices=truth["obs_time_indices"],
        obs_depth_indices=truth["obs_depth_indices"],
        z_np=truth["z_inv"],
        cfg=cfg,
    )

    elapsed = time.time() - t0
    logger.info("Inversion complete in %.1fs", elapsed)

    # Step 3: Evaluate
    results = evaluate_inversion(k_model, truth["z_inv"], truth["k_true_inv"], cfg.z_max)

    # Step 4: Summary
    kp = results["k_params"]
    logger.info("")
    logger.info("=" * 60)
    logger.info("FINAL RESULTS")
    logger.info("=" * 60)
    logger.info("  k_upper:   %.5f  (true=%.3f, err=%.1f%%)",
                kp["k_upper"], K_UPPER, abs(kp["k_upper"] - K_UPPER) / K_UPPER * 100)
    logger.info("  k_lower:   %.4f   (true=%.1f,  err=%.1f%%)",
                kp["k_lower"], K_LOWER, abs(kp["k_lower"] - K_LOWER) / K_LOWER * 100)
    logger.info("  boundary:  %.4fm  (true=%.2fm, err=%.1fmm)",
                kp["boundary"], BOUNDARY, abs(kp["boundary"] - BOUNDARY) * 1000)
    logger.info("  width:     %.4fm  (true=%.3fm, err=%.1fmm)",
                kp["width"], TRANS_WIDTH, abs(kp["width"] - TRANS_WIDTH) * 1000)
    logger.info("  k(z) RMSE: %.4f", results["k_rmse"])
    logger.info("  Time:      %.1fs", elapsed)
    logger.info("=" * 60)

    # Step 5: Save
    weights_path = OUT_DIR / "dfm_v5_weights.pt"
    torch.save(k_model.state_dict(), weights_path)
    logger.info("Saved weights → %s", weights_path)

    results_path = OUT_DIR / "dfm_v5_results.npz"
    np.savez(
        results_path,
        z=results["z"],
        k_pred=results["k_pred"],
        k_true=results["k_true"],
        TI_pred=results["TI_pred"],
        TI_true=results["TI_true"],
        k_rmse=results["k_rmse"],
        boundary_depth=results["boundary_depth"],
        k_upper=kp["k_upper"],
        k_lower=kp["k_lower"],
        boundary=kp["boundary"],
        width=kp["width"],
        loss_history=np.array(hist["loss"]),
        k_upper_history=np.array(hist["k_upper"]),
        k_lower_history=np.array(hist["k_lower"]),
        boundary_history=np.array(hist["boundary"]),
        width_history=np.array(hist["width"]),
        elapsed=elapsed,
    )
    logger.info("Saved results → %s", results_path)

    # Step 6: Plot
    plot_path = OUT_DIR / "dfm_v5_validation.png"
    plot_results(truth, results, hist, cfg, elapsed, plot_path)

    logger.info("Done. All outputs in %s/", OUT_DIR)


if __name__ == "__main__":
    main()
