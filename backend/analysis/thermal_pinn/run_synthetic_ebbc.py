#!/usr/bin/env python3
"""
Energy-balance BC — Synthetic Validation.

Validates that the differentiable FDM with energy-balance surface BC can
recover k(z) from SURFACE-ONLY temperature observations (no subsurface data).

This is the critical test before applying to real THEMIS data:
- Generate truth using forward EBBC model with known k(z)
- Extract T_surface at THEMIS-like observation times (~3AM, ~3PM)
- Run inversion → verify k(z) recovery

Usage:
    python -m backend.analysis.thermal_pinn.run_synthetic_ebbc
    python -m backend.analysis.thermal_pinn.run_synthetic_ebbc --steps 300
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

from backend.analysis.thermal_pinn.synthetic import two_layer_k_profile
from backend.analysis.thermal_pinn.inversion_real import (
    EBBCConfig,
    forward_ebbc_numpy,
    map_observations_to_steps,
    run_inversion_ebbc,
)
from backend.analysis.thermal_pinn.pinn_model import evaluate_inversion

logger = logging.getLogger(__name__)
OUT_DIR = Path("backend/data/thermal_pinn")

# Ground truth
K_UPPER = 0.02       # W/m/K (dust)
K_LOWER = 2.0        # W/m/K (ice)
BOUNDARY = 0.5       # m
TRANS_WIDTH = 0.05   # m


def generate_synthetic_observations(cfg: EBBCConfig, n_night=30, n_day=20):
    """
    Generate truth with forward EBBC model, then extract surface-only observations
    mimicking THEMIS orbital passes.

    Mars Odyssey observes at ~3:00-4:00 AM (nighttime) and ~14:30-16:00 (daytime).
    """
    z = np.linspace(0, cfg.z_max, cfg.nz)
    k_true = two_layer_k_profile(z, K_UPPER, K_LOWER, BOUNDARY, TRANS_WIDTH)

    logger.info("Generating truth: forward EBBC (nz=%d, %d+%d sols, lat=%.1f°N) ...",
                cfg.nz, cfg.spinup_sols, cfg.sim_sols, cfg.latitude)
    T_surf, T_full, z_out, Ls_out, lt_out = forward_ebbc_numpy(k_true, cfg)

    logger.info("Truth T_surface range: %.1f – %.1f K", T_surf.min(), T_surf.max())

    # Check for valid temperatures
    if T_surf.max() < 100 or T_surf.min() > 400:
        raise RuntimeError(f"Invalid T_surface range: {T_surf.min():.1f}-{T_surf.max():.1f} K")

    # Extract THEMIS-like observations
    observations = []
    n_sim_steps = len(T_surf)

    # Find nighttime steps (~3:00-4:00 AM → local_time 3.0-4.0)
    night_mask = (lt_out >= 3.0) & (lt_out <= 4.0)
    night_indices = np.where(night_mask)[0]
    if len(night_indices) > n_night:
        # Subsample uniformly across available nights
        sel = np.linspace(0, len(night_indices) - 1, n_night, dtype=int)
        night_indices = night_indices[sel]

    for idx in night_indices:
        observations.append({
            "local_time": float(lt_out[idx]),
            "solar_lon": float(Ls_out[idx]),
            "bt_kelvin": float(T_surf[idx]),
        })

    # Find daytime steps (~14:30-16:00 → local_time 14.5-16.0)
    day_mask = (lt_out >= 14.5) & (lt_out <= 16.0)
    day_indices = np.where(day_mask)[0]
    if len(day_indices) > n_day:
        sel = np.linspace(0, len(day_indices) - 1, n_day, dtype=int)
        day_indices = day_indices[sel]

    for idx in day_indices:
        observations.append({
            "local_time": float(lt_out[idx]),
            "solar_lon": float(Ls_out[idx]),
            "bt_kelvin": float(T_surf[idx]),
        })

    logger.info("Generated %d observations: %d nighttime + %d daytime",
                len(observations), len(night_indices), len(day_indices))
    logger.info("  Night BT range: %.1f – %.1f K",
                min(o["bt_kelvin"] for o in observations if o["local_time"] < 6),
                max(o["bt_kelvin"] for o in observations if o["local_time"] < 6))
    if any(o["local_time"] > 12 for o in observations):
        logger.info("  Day   BT range: %.1f – %.1f K",
                    min(o["bt_kelvin"] for o in observations if o["local_time"] > 12),
                    max(o["bt_kelvin"] for o in observations if o["local_time"] > 12))

    return observations, z, k_true, T_surf, Ls_out, lt_out


def plot_results(z, k_true, k_pred, hist, T_surf_truth, Ls_truth, lt_truth,
                 observations, cfg, elapsed, save_path):
    """5-panel figure for EBBC validation."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"V5 Energy-Balance BC — Synthetic Validation\n"
        f"nz={cfg.nz}, lat={cfg.latitude}°N, {cfg.spinup_sols}+{cfg.sim_sols} sols, "
        f"{cfg.n_steps} steps, {elapsed:.0f}s",
        fontsize=13, fontweight="bold",
    )

    # Panel 1: k(z) recovery
    ax = axes[0, 0]
    ax.semilogy(z, k_true, "k-", lw=2, label="True k(z)")
    ax.semilogy(z, k_pred, "r--", lw=2, label="Recovered k(z)")
    ax.axhline(K_UPPER, color="blue", ls=":", alpha=0.5, label=f"k_upper={K_UPPER}")
    ax.axhline(K_LOWER, color="green", ls=":", alpha=0.5, label=f"k_lower={K_LOWER}")
    ax.axvline(BOUNDARY, color="gray", ls=":", alpha=0.5, label=f"boundary={BOUNDARY}m")
    rmse = np.sqrt(np.mean((k_pred - k_true)**2))
    ax.set_title(f"Conductivity Profile — RMSE={rmse:.4f}")
    ax.set_xlabel("Depth (m)")
    ax.set_ylabel("k (W/m/K)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel 2: Loss convergence
    ax = axes[0, 1]
    ax.semilogy(hist["loss"], "b-", lw=1)
    ax.set_xlabel("Step")
    ax.set_ylabel("MSE Loss (K²)")
    ax.set_title("Loss Convergence")
    ax.grid(True, alpha=0.3)

    # Panel 3: Parameter convergence
    ax = axes[0, 2]
    steps = np.arange(len(hist["k_upper"]))
    ax.plot(steps, hist["k_upper"], "b-", label=f"k_upper (true={K_UPPER})")
    ax.axhline(K_UPPER, color="b", ls=":", alpha=0.5)
    ax2 = ax.twinx()
    ax2.plot(steps, hist["k_lower"], "r-", label=f"k_lower (true={K_LOWER})")
    ax2.axhline(K_LOWER, color="r", ls=":", alpha=0.5)
    ax2.set_ylabel("k_lower", color="r")
    ax.set_xlabel("Step")
    ax.set_ylabel("k_upper", color="b")
    ax.set_title("k Parameter Convergence")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel 4: Boundary/width convergence
    ax = axes[1, 0]
    ax.plot(steps, hist["boundary"], "g-", label=f"boundary (true={BOUNDARY}m)")
    ax.axhline(BOUNDARY, color="g", ls=":", alpha=0.5)
    ax2 = ax.twinx()
    ax2.plot(steps, hist["width"], "m-", label=f"width (true={TRANS_WIDTH}m)")
    ax2.axhline(TRANS_WIDTH, color="m", ls=":", alpha=0.5)
    ax2.set_ylabel("width (m)", color="m")
    ax.set_xlabel("Step")
    ax.set_ylabel("boundary (m)", color="g")
    ax.set_title("Boundary & Width Convergence")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel 5: T_surface diurnal curve (truth vs observations)
    ax = axes[1, 1]
    # Plot a few sols of T_surface truth
    dt_per_sol = cfg.dt_per_sol
    # Show first 5 sols of output
    n_show = min(5 * dt_per_sol, len(T_surf_truth))
    lt_show = lt_truth[:n_show]
    T_show = T_surf_truth[:n_show]
    ax.scatter(lt_show, T_show, c="gray", s=1, alpha=0.3, label="Truth T_surface")

    # Overlay observations
    obs_lt = [o["local_time"] for o in observations]
    obs_bt = [o["bt_kelvin"] for o in observations]
    night_mask = [lt < 6 for lt in obs_lt]
    day_mask = [lt > 12 for lt in obs_lt]
    ax.scatter([lt for lt, m in zip(obs_lt, night_mask) if m],
               [bt for bt, m in zip(obs_bt, night_mask) if m],
               c="blue", s=20, label=f"Night obs ({sum(night_mask)})", zorder=5)
    ax.scatter([lt for lt, m in zip(obs_lt, day_mask) if m],
               [bt for bt, m in zip(obs_bt, day_mask) if m],
               c="red", s=20, label=f"Day obs ({sum(day_mask)})", zorder=5)
    ax.set_xlabel("Local Solar Time (hr)")
    ax.set_ylabel("T_surface (K)")
    ax.set_title("Diurnal Temperature Curve")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel 6: T_surface seasonal variation
    ax = axes[1, 2]
    ax.scatter(Ls_truth[:len(T_surf_truth)], T_surf_truth, c="gray", s=1, alpha=0.1)
    # Overlay observations colored by local time
    for o in observations:
        c = "blue" if o["local_time"] < 6 else "red"
        ax.scatter(o["solar_lon"], o["bt_kelvin"], c=c, s=15, zorder=5)
    ax.set_xlabel("Ls (°)")
    ax.set_ylabel("T_surface (K)")
    ax.set_title("Seasonal Temperature Variation")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved plot → %s", save_path)


def main():
    parser = argparse.ArgumentParser(description="EBBC synthetic validation")
    parser.add_argument("--steps", type=int, default=200, help="Optimization steps")
    parser.add_argument("--nz", type=int, default=40, help="Depth nodes")
    parser.add_argument("--sim-sols", type=int, default=200, help="Simulation sols")
    parser.add_argument("--spinup-sols", type=int, default=200, help="Spinup sols")
    parser.add_argument("--lat", type=float, default=45.0, help="Latitude (°N)")
    parser.add_argument("--lr", type=float, default=0.02, help="Learning rate")
    parser.add_argument("--n-night", type=int, default=30, help="Number of nighttime obs")
    parser.add_argument("--n-day", type=int, default=20, help="Number of daytime obs")
    parser.add_argument("--sched-step", type=int, default=None, help="LR scheduler step")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg = EBBCConfig(
        nz=args.nz,
        sim_sols=args.sim_sols,
        spinup_sols=args.spinup_sols,
        latitude=args.lat,
        n_steps=args.steps,
        lr=args.lr,
        scheduler_step=args.sched_step if args.sched_step else max(args.steps // 3, 80),
    )

    logger.info("=" * 60)
    logger.info("ENERGY-BALANCE BC — SYNTHETIC VALIDATION")
    logger.info("=" * 60)
    logger.info("Config: nz=%d lat=%.1f sim=%d spinup=%d steps=%d lr=%.4f",
                cfg.nz, cfg.latitude, cfg.sim_sols, cfg.spinup_sols,
                cfg.n_steps, cfg.lr)

    # Step 1: Generate truth + synthetic observations
    observations, z, k_true, T_surf_truth, Ls_truth, lt_truth = \
        generate_synthetic_observations(cfg, n_night=args.n_night, n_day=args.n_day)

    # Step 2: Map observations to simulation steps
    obs_mapped = map_observations_to_steps(observations, cfg)
    if not obs_mapped:
        logger.error("No observations mapped! Check Ls range.")
        return

    # Step 3: Run inversion
    logger.info("-" * 60)
    logger.info("Starting EBBC inversion (%d mapped observations) ...", len(obs_mapped))
    t0 = time.time()

    k_model, hist = run_inversion_ebbc(obs_mapped, z, cfg)

    elapsed = time.time() - t0
    logger.info("Inversion complete in %.1fs", elapsed)

    # Step 4: Evaluate
    results = evaluate_inversion(k_model, z, k_true, cfg.z_max)
    kp = results["k_params"]

    logger.info("")
    logger.info("=" * 60)
    logger.info("FINAL RESULTS (EBBC)")
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
    weights_path = OUT_DIR / "ebbc_v5_weights.pt"
    torch.save(k_model.state_dict(), weights_path)
    logger.info("Saved weights → %s", weights_path)

    results_path = OUT_DIR / "ebbc_v5_results.npz"
    np.savez(
        results_path,
        z=z, k_pred=results["k_pred"], k_true=k_true,
        k_rmse=results["k_rmse"],
        k_upper=kp["k_upper"], k_lower=kp["k_lower"],
        boundary=kp["boundary"], width=kp["width"],
        loss_history=np.array(hist["loss"]),
        k_upper_history=np.array(hist["k_upper"]),
        k_lower_history=np.array(hist["k_lower"]),
        boundary_history=np.array(hist["boundary"]),
        width_history=np.array(hist["width"]),
        elapsed=elapsed,
    )
    logger.info("Saved results → %s", results_path)

    # Step 6: Plot
    with torch.no_grad():
        k_pred = k_model(torch.tensor(z, dtype=torch.float64)).numpy()

    plot_path = OUT_DIR / "ebbc_v5_validation.png"
    plot_results(z, k_true, k_pred, hist, T_surf_truth, Ls_truth, lt_truth,
                 observations, cfg, elapsed, plot_path)

    logger.info("Done.")


if __name__ == "__main__":
    main()
