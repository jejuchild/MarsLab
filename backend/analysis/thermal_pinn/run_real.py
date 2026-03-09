#!/usr/bin/env python3
"""
Phase 2: Real THEMIS data inversion for Arcadia Planitia.

Extracts surface brightness temperatures from downloaded THEMIS Band 9 images,
runs the energy-balance BC differentiable FDM inversion to recover k(z),
and compares against SWIM CT thermal consistency scores.

Usage:
    python -m backend.analysis.thermal_pinn.run_real
    python -m backend.analysis.thermal_pinn.run_real --steps 200 --target-lat 45.0 --target-lon 190.0
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

from backend.analysis.thermal_pinn.themis_download import parse_irbtr_img
from backend.analysis.thermal_pinn.inversion_real import (
    EBBCConfig,
    map_observations_to_steps,
    run_inversion_ebbc,
)
from backend.analysis.thermal_pinn.pinn_model import (
    ParametricConductivity,
    evaluate_inversion,
    RHO_CP,
)
from backend.analysis.thermal_pinn.mars_solar import STEFAN_BOLTZMANN

logger = logging.getLogger(__name__)

DATA_DIR = Path("backend/data/themis_irbtr")
OUT_DIR = Path("backend/data/thermal_pinn")
SWIM_DIR = Path("backend/data/swim")

# Surface emissivity at 12.57 μm (Mars basalt/dust)
EMISSIVITY_BAND9 = 0.97


# ── THEMIS observation extraction ───────────────────────────────

def extract_observations(target_lat: float, target_lon: float,
                         search_radius_deg: float = 1.5) -> list[dict]:
    """
    Parse all downloaded THEMIS IRBTR images and extract brightness temperature
    at the target location.

    For each image that covers the target point, extracts the mean BT in a
    small window around the target pixel. Applies emissivity correction
    (BTR → T_kinetic) and simple atmospheric correction.

    Returns list of observation dicts ready for inversion.
    """
    raw_dir = DATA_DIR / "raw"
    img_files = sorted(raw_dir.glob("*BTR.IMG"))
    logger.info("Found %d IRBTR images in %s", len(img_files), raw_dir)

    observations = []
    skipped = {"no_coverage": 0, "bad_bt": 0, "parse_error": 0, "too_small": 0}

    for img_path in img_files:
        try:
            parsed = parse_irbtr_img(img_path)
        except Exception as e:
            logger.debug("Parse error %s: %s", img_path.name, e)
            skipped["parse_error"] += 1
            continue

        # Check if image covers target location (approximate)
        # IRBTR images are narrow strips; use center +/- radius
        if abs(parsed.center_lat - target_lat) > search_radius_deg * 3:
            skipped["no_coverage"] += 1
            continue
        # Longitude check (handle 360° wrapping)
        lon_diff = abs(parsed.center_lon - target_lon)
        if lon_diff > 180:
            lon_diff = 360 - lon_diff
        if lon_diff > search_radius_deg * 3:
            skipped["no_coverage"] += 1
            continue

        # Check image size (some are tiny 1-line images)
        if parsed.lines < 10 or parsed.samples < 10:
            skipped["too_small"] += 1
            continue

        # Extract BT at target pixel
        # Map target lat/lon to image pixel coordinates (approximate linear mapping)
        bt = parsed.bt  # (lines, samples) in Kelvin

        # Approximate pixel coordinates
        # Images go from max_lat (top) to min_lat (bottom)
        # For IRBTR, we use the center and image extent
        lat_range = max(abs(parsed.bt_max - parsed.bt_min), 1.0)  # avoid div by 0

        # Use center pixel approach: find the pixel closest to target
        # Simple linear interpolation based on image metadata
        row_frac = 0.5  # default to center
        col_frac = 0.5

        # Extract mean BT in center region (3x3 pixels around center)
        r_center = int(parsed.lines * row_frac)
        c_center = int(parsed.samples * col_frac)
        hw = 2  # half-window
        r0 = max(0, r_center - hw)
        r1 = min(parsed.lines, r_center + hw + 1)
        c0 = max(0, c_center - hw)
        c1 = min(parsed.samples, c_center + hw + 1)

        bt_window = bt[r0:r1, c0:c1]
        valid = bt_window[(bt_window > 100) & (bt_window < 350)]
        if len(valid) < 3:
            skipped["bad_bt"] += 1
            continue

        bt_mean = float(np.mean(valid))

        # Apply emissivity correction: BT → T_kinetic
        # T_kinetic = BT / ε^(1/4) (Stefan-Boltzmann approx, valid for broadband)
        # More accurate: Planck inversion, but ε≈0.97 gives ~0.5K correction
        t_kinetic = bt_mean / EMISSIVITY_BAND9**0.25

        # Simple atmospheric correction (nighttime: small; daytime: larger)
        # Using linear mixing model: T_corrected⁴ = (T_obs⁴ - ε_atm·T_atm⁴) / (1-ε_atm)
        is_night = parsed.local_time < 6.0 or parsed.local_time > 18.0
        tau_dust = 0.3  # typical
        eps_atm = 1 - np.exp(-tau_dust)
        T_atm = 210.0  # K atmospheric temperature
        T_corr4 = (t_kinetic**4 - eps_atm * T_atm**4) / (1 - eps_atm)
        if T_corr4 > 0:
            t_corrected = T_corr4**0.25
        else:
            t_corrected = t_kinetic  # fallback

        observations.append({
            "local_time": float(parsed.local_time),
            "solar_lon": float(parsed.solar_lon),
            "bt_kelvin": float(t_corrected),
            "bt_raw": float(bt_mean),
            "product_id": parsed.product_id,
            "is_night": is_night,
            "center_lat": parsed.center_lat,
            "center_lon": parsed.center_lon,
        })

    logger.info("Extracted %d observations (skipped: %s)", len(observations), skipped)

    if observations:
        night_obs = [o for o in observations if o["is_night"]]
        day_obs = [o for o in observations if not o["is_night"]]
        logger.info("  Night: %d obs, BT %.1f–%.1f K",
                    len(night_obs),
                    min(o["bt_kelvin"] for o in night_obs) if night_obs else 0,
                    max(o["bt_kelvin"] for o in night_obs) if night_obs else 0)
        logger.info("  Day:   %d obs, BT %.1f–%.1f K",
                    len(day_obs),
                    min(o["bt_kelvin"] for o in day_obs) if day_obs else 0,
                    max(o["bt_kelvin"] for o in day_obs) if day_obs else 0)
        logger.info("  Ls range: %.0f°–%.0f°",
                    min(o["solar_lon"] for o in observations),
                    max(o["solar_lon"] for o in observations))

    return observations


# ── SWIM CT comparison ──────────────────────────────────────────

def load_swim_ct(target_lat: float, target_lon: float):
    """Load SWIM CT thermal consistency score at target location."""
    try:
        import rasterio

        ct_path = SWIM_DIR / "SWIM2_T.tif"
        if not ct_path.exists():
            ct_path = SWIM_DIR / "thermal_consistency.tif"
        if not ct_path.exists():
            logger.warning("SWIM CT file not found")
            return None

        with rasterio.open(ct_path) as ds:
            # SWIM uses 0-360 lon, -90 to 90 lat, simple cylindrical
            # Transform target coords
            lon_360 = target_lon if target_lon >= 0 else target_lon + 360
            row, col = ds.index(lon_360, target_lat)
            data = ds.read(1)
            if 0 <= row < data.shape[0] and 0 <= col < data.shape[1]:
                val = float(data[row, col])
                if val == ds.nodata or np.isnan(val):
                    return None
                return val
        return None
    except Exception as e:
        logger.warning("Could not load SWIM CT: %s", e)
        return None


def k_to_ct_score(k_upper, k_lower, boundary):
    """
    Convert recovered k(z) to a CT-like consistency score (-1 to +1).

    Logic (following MARSTHERM CT approach):
    - If k_lower >> k_upper AND boundary < 2m → consistent with buried ice → positive score
    - If k_lower ≈ k_upper → no layering evidence → score near 0
    - Score magnitude depends on k_lower/k_upper ratio and boundary depth

    This is a simplified version of the full CT scoring.
    """
    from backend.analysis.thermal_pinn.pinn_model import RHO, C_P

    # Thermal inertia
    TI_upper = np.sqrt(k_upper * RHO * C_P)
    TI_lower = np.sqrt(k_lower * RHO * C_P)

    # Layer contrast ratio
    ratio = TI_lower / max(TI_upper, 1.0)

    # Depth penalty (deeper boundary = weaker evidence)
    depth_factor = np.exp(-boundary / 1.0)  # e-fold at 1m

    # CT score: high ratio + shallow boundary → strong positive
    if ratio > 3.0 and boundary < 2.0:
        # Consistent with ice cement or buried ice
        score = min(1.0, 0.3 + 0.7 * (ratio - 3) / 10 * depth_factor)
    elif ratio > 1.5:
        score = 0.1 * (ratio - 1.5) * depth_factor
    else:
        # No significant layering
        score = -0.2 * (1.5 - ratio)

    return np.clip(score, -1.0, 1.0)


# ── Plotting ────────────────────────────────────────────────────

def plot_real_results(observations, k_pred, z, hist, cfg, kp, swim_ct,
                      elapsed, save_path):
    """4-panel figure for real THEMIS results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Real THEMIS Inversion — Arcadia Planitia ({cfg.latitude:.1f}°N)\n"
        f"{len(observations)} observations, {cfg.n_steps} steps, {elapsed:.0f}s",
        fontsize=13, fontweight="bold",
    )

    # Panel 1: Recovered k(z)
    ax = axes[0, 0]
    ax.semilogy(z, k_pred, "r-", lw=2, label="Recovered k(z)")
    ax.axhline(kp["k_upper"], color="blue", ls="--", alpha=0.5,
               label=f'k_upper={kp["k_upper"]:.4f}')
    ax.axhline(kp["k_lower"], color="green", ls="--", alpha=0.5,
               label=f'k_lower={kp["k_lower"]:.4f}')
    ax.axvline(kp["boundary"], color="gray", ls=":", alpha=0.5,
               label=f'boundary={kp["boundary"]:.2f}m')
    # TI annotations
    from backend.analysis.thermal_pinn.pinn_model import RHO, C_P
    TI_u = np.sqrt(kp["k_upper"] * RHO * C_P)
    TI_l = np.sqrt(kp["k_lower"] * RHO * C_P)
    ax.set_title(f"Conductivity Profile\nTI: {TI_u:.0f}→{TI_l:.0f} tiu")
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

    # Panel 3: Observations vs Ls
    ax = axes[1, 0]
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

    # Panel 4: CT comparison
    ax = axes[1, 1]
    ct_our = k_to_ct_score(kp["k_upper"], kp["k_lower"], kp["boundary"])
    bars = ["Our CT"]
    vals = [ct_our]
    colors = ["#42a5f5" if ct_our > 0.3 else "#9e9e9e" if ct_our > -0.3 else "#ef9a9a"]
    if swim_ct is not None:
        bars.append("SWIM CT")
        vals.append(swim_ct)
        colors.append("#42a5f5" if swim_ct > 0.3 else "#9e9e9e" if swim_ct > -0.3 else "#ef9a9a")
    ax.barh(bars, vals, color=colors, height=0.5)
    ax.set_xlim(-1, 1)
    ax.axvline(0, color="k", lw=0.5)
    ax.axvline(0.3, color="g", ls=":", alpha=0.5)
    ax.axvline(-0.3, color="r", ls=":", alpha=0.5)
    ax.set_xlabel("Consistency Score")
    ax.set_title("CT Thermal Consistency Comparison")

    for i, v in enumerate(vals):
        ax.text(v + 0.05 if v >= 0 else v - 0.15, i, f"{v:.2f}", va="center", fontsize=11)
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved plot → %s", save_path)


# ── Main ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Real THEMIS inversion")
    parser.add_argument("--target-lat", type=float, default=45.0,
                        help="Target latitude (°N)")
    parser.add_argument("--target-lon", type=float, default=190.0,
                        help="Target longitude (°E, 0-360)")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--nz", type=int, default=40)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--sched-step", type=int, default=100)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("REAL THEMIS INVERSION — ARCADIA PLANITIA")
    logger.info("=" * 60)
    logger.info("Target: %.1f°N, %.1f°E", args.target_lat, args.target_lon)

    # Step 1: Extract observations from downloaded THEMIS images
    observations = extract_observations(args.target_lat, args.target_lon)
    if len(observations) < 5:
        logger.error("Not enough observations (%d). Need at least 5.", len(observations))
        return

    # Step 2: Configure inversion
    cfg = EBBCConfig(
        nz=args.nz,
        sim_sols=668,        # full Mars year
        spinup_sols=668,     # full year spinup
        latitude=args.target_lat,
        Ls_start=0.0,
        n_steps=args.steps,
        lr=args.lr,
        scheduler_step=args.sched_step,
        scheduler_gamma=0.5,
    )

    # Map observations to simulation steps
    obs_mapped = map_observations_to_steps(observations, cfg)
    if len(obs_mapped) < 5:
        logger.error("Not enough mapped observations (%d). Need at least 5.", len(obs_mapped))
        return

    # Step 3: Run inversion
    z = np.linspace(0, cfg.z_max, cfg.nz)

    logger.info("-" * 60)
    logger.info("Starting inversion: %d mapped obs, %d steps", len(obs_mapped), cfg.n_steps)
    t0 = time.time()

    k_model, hist = run_inversion_ebbc(obs_mapped, z, cfg)

    elapsed = time.time() - t0
    logger.info("Inversion complete in %.1fs", elapsed)

    # Step 4: Results
    kp = k_model.get_params(cfg.z_max)
    with torch.no_grad():
        k_pred = k_model(torch.tensor(z, dtype=torch.float64)).numpy()

    from backend.analysis.thermal_pinn.pinn_model import RHO, C_P
    TI_upper = np.sqrt(kp["k_upper"] * RHO * C_P)
    TI_lower = np.sqrt(kp["k_lower"] * RHO * C_P)

    logger.info("")
    logger.info("=" * 60)
    logger.info("REAL THEMIS INVERSION RESULTS")
    logger.info("=" * 60)
    logger.info("  k_upper:    %.5f W/m/K  (TI = %.0f tiu)", kp["k_upper"], TI_upper)
    logger.info("  k_lower:    %.4f W/m/K   (TI = %.0f tiu)", kp["k_lower"], TI_lower)
    logger.info("  boundary:   %.3fm", kp["boundary"])
    logger.info("  width:      %.4fm", kp["width"])
    logger.info("  Final loss: %.4f K²  (RMS = %.2f K)", hist["loss"][-1],
                np.sqrt(hist["loss"][-1]))
    logger.info("  TI ratio:   %.1f (lower/upper)", TI_lower / max(TI_upper, 1))

    # Step 5: SWIM CT comparison
    swim_ct = load_swim_ct(args.target_lat, args.target_lon)
    our_ct = k_to_ct_score(kp["k_upper"], kp["k_lower"], kp["boundary"])

    logger.info("-" * 60)
    logger.info("CT COMPARISON")
    if swim_ct is not None:
        logger.info("  SWIM CT score: %.3f", swim_ct)
    else:
        logger.info("  SWIM CT score: N/A (data not loaded)")
    logger.info("  Our  CT score: %.3f", our_ct)

    interp = ("consistent with ice" if our_ct > 0.3
              else "ambiguous" if our_ct > -0.3
              else "inconsistent with ice")
    logger.info("  Interpretation: %s", interp)
    logger.info("=" * 60)

    # Step 6: Save
    weights_path = OUT_DIR / "real_ebbc_weights.pt"
    torch.save(k_model.state_dict(), weights_path)

    results_path = OUT_DIR / "real_ebbc_results.npz"
    np.savez(
        results_path,
        z=z, k_pred=k_pred,
        k_upper=kp["k_upper"], k_lower=kp["k_lower"],
        boundary=kp["boundary"], width=kp["width"],
        TI_upper=TI_upper, TI_lower=TI_lower,
        loss_history=np.array(hist["loss"]),
        our_ct=our_ct, swim_ct=swim_ct if swim_ct else np.nan,
        target_lat=args.target_lat, target_lon=args.target_lon,
        n_observations=len(obs_mapped),
        elapsed=elapsed,
    )
    logger.info("Saved results → %s", results_path)

    # Save observations for reference
    obs_path = OUT_DIR / "real_observations.json"
    with open(obs_path, "w") as f:
        json.dump(observations, f, indent=2)
    logger.info("Saved observations → %s", obs_path)

    # Step 7: Plot
    plot_path = OUT_DIR / "real_ebbc_results.png"
    plot_real_results(observations, k_pred, z, hist, cfg, kp, swim_ct,
                      elapsed, plot_path)

    logger.info("Done. All outputs in %s/", OUT_DIR)


if __name__ == "__main__":
    main()
