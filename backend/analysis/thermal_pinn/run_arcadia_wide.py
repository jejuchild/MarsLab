#!/usr/bin/env python3
"""
Arcadia-wide DeepONet inversion: CT thermal consistency map.

Pipeline:
    1. Query ODE for ALL THEMIS IRBTR products covering Arcadia Planitia
    2. Download images (bulk, shared across grid points)
    3. For each grid point: extract observations from nearby images
    4. Run DeepONet inversion (if ≥10 observations available)
    5. Generate CT comparison map vs SWIM

Usage:
    python -m backend.analysis.thermal_pinn.run_arcadia_wide
    python -m backend.analysis.thermal_pinn.run_arcadia_wide --grid-step 5 --max-products 200
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from backend.analysis.thermal_pinn.themis_download import (
    query_ode_products,
    download_img,
    IRBTRProduct,
)
from backend.analysis.thermal_pinn.run_real import (
    extract_observations,
    load_swim_ct,
    k_to_ct_score,
)
from backend.analysis.thermal_pinn.deeponet_surrogate import (
    DeepONet,
    DeepONetConfig,
    run_inversion_deeponet,
)

logger = logging.getLogger(__name__)

DATA_DIR = Path("backend/data/thermal_pinn")
ARCADIA_DIR = Path("backend/data/themis_irbtr_arcadia")

# Arcadia Planitia bounds (generous for THEMIS coverage)
ARCADIA_BBOX = {
    "minlat": 38.0,
    "maxlat": 62.0,
    "westernlon": 150.0,   # 0-360 East-positive
    "easternlon": 220.0,
}

MIN_OBS_FOR_INVERSION = 10  # minimum observations to attempt inversion


def download_arcadia_themis(max_products: int = 3000,
                            skip_download: bool = False) -> Path:
    """
    Query ODE and bulk-download THEMIS IRBTR images for Arcadia.
    Returns path to the raw image directory.
    """
    raw_dir = ARCADIA_DIR / "raw"
    catalog_path = ARCADIA_DIR / "product_catalog.json"

    if skip_download and catalog_path.exists():
        logger.info("Skipping download, using cached catalog at %s", catalog_path)
        with open(catalog_path) as f:
            catalog = json.load(f)
        logger.info("Catalog has %d products", len(catalog))
        return raw_dir

    # Query ODE
    products = query_ode_products(
        minlat=ARCADIA_BBOX["minlat"],
        maxlat=ARCADIA_BBOX["maxlat"],
        westernlon=ARCADIA_BBOX["westernlon"],
        easternlon=ARCADIA_BBOX["easternlon"],
        max_products=max_products,
    )
    logger.info("ODE query returned %d products", len(products))

    if not products:
        raise RuntimeError("No THEMIS IRBTR products found for Arcadia region!")

    # Save catalog
    ARCADIA_DIR.mkdir(parents=True, exist_ok=True)
    with open(catalog_path, "w") as f:
        json.dump([asdict(p) for p in products], f, indent=2)
    logger.info("Saved catalog: %s", catalog_path)

    # Download all images
    raw_dir.mkdir(parents=True, exist_ok=True)
    n_downloaded = 0
    n_skipped = 0
    n_failed = 0

    for i, prod in enumerate(products):
        try:
            out_path = raw_dir / f"{prod.product_id}.IMG"
            if out_path.exists():
                n_skipped += 1
            else:
                download_img(prod, out_dir=raw_dir)
                n_downloaded += 1

            if (i + 1) % 100 == 0:
                logger.info("  Download progress: %d/%d (new=%d, cached=%d, failed=%d)",
                            i + 1, len(products), n_downloaded, n_skipped, n_failed)
        except Exception as e:
            logger.warning("Failed to download %s: %s", prod.product_id, e)
            n_failed += 1

    logger.info("Download complete: %d new, %d cached, %d failed",
                n_downloaded, n_skipped, n_failed)
    return raw_dir


def run_arcadia_grid(
    raw_dir: Path,
    model: DeepONet,
    cfg: DeepONetConfig,
    lat_range: tuple[float, float] = (40.0, 60.0),
    lon_range: tuple[float, float] = (155.0, 215.0),
    grid_step: float = 3.0,
    inv_steps: int = 500,
    inv_lr: float = 0.03,
) -> dict:
    """
    Run DeepONet inversion at each grid point over Arcadia.

    Returns dict with numpy arrays:
        lats, lons: (N,) grid coordinates
        k_upper, k_lower, boundary: (N,) inverted parameters
        ct_ours, ct_swim: (N,) CT scores
        n_obs: (N,) number of observations used
        loss: (N,) final inversion loss
        status: (N,) 0=success, 1=insufficient data, 2=failed
    """
    lats = np.arange(lat_range[0], lat_range[1] + 0.1, grid_step)
    lons = np.arange(lon_range[0], lon_range[1] + 0.1, grid_step)
    grid_lats, grid_lons = np.meshgrid(lats, lons, indexing="ij")
    flat_lats = grid_lats.ravel()
    flat_lons = grid_lons.ravel()
    n_points = len(flat_lats)

    logger.info("Grid: lat %.0f–%.0f, lon %.0f–%.0f, step=%.0f° → %d points (%d×%d)",
                lat_range[0], lat_range[1], lon_range[0], lon_range[1],
                grid_step, n_points, len(lats), len(lons))

    # Result arrays
    results = {
        "lats": flat_lats,
        "lons": flat_lons,
        "grid_lats": lats,
        "grid_lons": lons,
        "k_upper": np.full(n_points, np.nan),
        "k_lower": np.full(n_points, np.nan),
        "boundary": np.full(n_points, np.nan),
        "TI_upper": np.full(n_points, np.nan),
        "TI_lower": np.full(n_points, np.nan),
        "ct_ours": np.full(n_points, np.nan),
        "ct_swim": np.full(n_points, np.nan),
        "n_obs": np.zeros(n_points, dtype=int),
        "loss": np.full(n_points, np.nan),
        "elapsed": np.full(n_points, np.nan),
        "status": np.full(n_points, 1, dtype=int),  # 1=insufficient by default
    }

    # Load SWIM CT for all points first
    logger.info("Loading SWIM CT values...")
    for i in range(n_points):
        ct = load_swim_ct(flat_lats[i], flat_lons[i])
        if ct is not None:
            results["ct_swim"][i] = ct

    n_swim_valid = np.isfinite(results["ct_swim"]).sum()
    logger.info("SWIM CT loaded: %d/%d points have valid values", n_swim_valid, n_points)

    # Process each grid point
    n_success = 0
    n_insufficient = 0
    n_failed = 0
    t0_total = time.time()

    for i in range(n_points):
        lat = flat_lats[i]
        lon = flat_lons[i]
        tag = f"[{i+1}/{n_points}] ({lat:.0f}°N, {lon:.0f}°E)"

        # Extract observations
        try:
            observations = extract_observations(lat, lon, data_dir=ARCADIA_DIR)
        except Exception as e:
            logger.warning("%s extract failed: %s", tag, e)
            results["status"][i] = 2
            n_failed += 1
            continue

        results["n_obs"][i] = len(observations)

        if len(observations) < MIN_OBS_FOR_INVERSION:
            logger.info("%s skipped: only %d obs (need %d)",
                        tag, len(observations), MIN_OBS_FOR_INVERSION)
            n_insufficient += 1
            continue

        # Run DeepONet inversion
        try:
            t0 = time.time()
            inv_result = run_inversion_deeponet(
                observations, model, lat,
                cfg=cfg, n_steps=inv_steps, lr=inv_lr,
            )
            elapsed = time.time() - t0

            results["k_upper"][i] = inv_result.k_upper
            results["k_lower"][i] = inv_result.k_lower
            results["boundary"][i] = inv_result.boundary
            results["TI_upper"][i] = inv_result.TI_upper
            results["TI_lower"][i] = inv_result.TI_lower
            results["ct_ours"][i] = k_to_ct_score(
                inv_result.k_upper, inv_result.k_lower, inv_result.boundary)
            results["loss"][i] = inv_result.loss_final
            results["elapsed"][i] = elapsed
            results["status"][i] = 0
            n_success += 1

            logger.info("%s ✓ k_u=%.4f k_l=%.3f bd=%.2fm CT=%.3f (%d obs, %.1fs)",
                        tag, inv_result.k_upper, inv_result.k_lower,
                        inv_result.boundary,
                        results["ct_ours"][i], len(observations), elapsed)

        except Exception as e:
            logger.warning("%s inversion failed: %s", tag, e)
            results["status"][i] = 2
            n_failed += 1

    total_elapsed = time.time() - t0_total
    logger.info("=" * 60)
    logger.info("Arcadia grid complete: %d success, %d insufficient, %d failed (%.1fs total)",
                n_success, n_insufficient, n_failed, total_elapsed)

    return results


def plot_ct_comparison(results: dict, save_path: Path):
    """Generate side-by-side CT map: our inversion vs SWIM."""
    lats = results["grid_lats"]
    lons = results["grid_lons"]
    n_lat = len(lats)
    n_lon = len(lons)

    # Reshape to grid
    ct_ours = results["ct_ours"].reshape(n_lat, n_lon)
    ct_swim = results["ct_swim"].reshape(n_lat, n_lon)
    n_obs_grid = results["n_obs"].reshape(n_lat, n_lon)
    status = results["status"].reshape(n_lat, n_lon)

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle("Arcadia Planitia — Thermal Consistency (CT) Map\n"
                 "DeepONet Surrogate Inversion vs SWIM",
                 fontsize=14, fontweight="bold")

    # Shared colorbar range
    vmin, vmax = -0.5, 1.0

    # Panel 1: Our CT
    ax = axes[0, 0]
    im = ax.pcolormesh(lons, lats, ct_ours, cmap="RdYlBu_r",
                       vmin=vmin, vmax=vmax, shading="nearest")
    ax.set_title("Our CT (DeepONet Inversion)")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    plt.colorbar(im, ax=ax, label="CT Score")

    # Panel 2: SWIM CT
    ax = axes[0, 1]
    im = ax.pcolormesh(lons, lats, ct_swim, cmap="RdYlBu_r",
                       vmin=vmin, vmax=vmax, shading="nearest")
    ax.set_title("SWIM CT (Reference)")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    plt.colorbar(im, ax=ax, label="CT Score")

    # Panel 3: Difference (ours - SWIM)
    ax = axes[0, 2]
    diff = ct_ours - ct_swim
    im = ax.pcolormesh(lons, lats, diff, cmap="RdBu_r",
                       vmin=-1.0, vmax=1.0, shading="nearest")
    ax.set_title("Difference (Ours − SWIM)")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    plt.colorbar(im, ax=ax, label="ΔCT")

    # Panel 4: k_upper map
    ax = axes[1, 0]
    ku = results["k_upper"].reshape(n_lat, n_lon)
    im = ax.pcolormesh(lons, lats, np.log10(np.clip(ku, 1e-4, 10)),
                       cmap="viridis", shading="nearest")
    ax.set_title("log₁₀(k_upper) [W/m/K]")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    plt.colorbar(im, ax=ax, label="log₁₀(k)")

    # Panel 5: boundary depth map
    ax = axes[1, 1]
    bd = results["boundary"].reshape(n_lat, n_lon)
    im = ax.pcolormesh(lons, lats, bd, cmap="YlOrRd",
                       vmin=0, vmax=2.5, shading="nearest")
    ax.set_title("Boundary Depth [m]")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    plt.colorbar(im, ax=ax, label="Depth (m)")

    # Panel 6: Observation count / data coverage
    ax = axes[1, 2]
    im = ax.pcolormesh(lons, lats, n_obs_grid, cmap="YlGn",
                       vmin=0, shading="nearest")
    # Mark failed/insufficient points
    fail_mask = status > 0
    if fail_mask.any():
        fail_lats, fail_lons = np.where(fail_mask)
        ax.scatter(lons[fail_lons], lats[fail_lats], marker="x",
                   color="red", s=30, label="No data")
        ax.legend(fontsize=8)
    ax.set_title("Observation Count per Grid Point")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    plt.colorbar(im, ax=ax, label="# Obs")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved CT comparison map → %s", save_path)


def print_summary(results: dict):
    """Print summary statistics."""
    success = results["status"] == 0
    n_success = success.sum()
    n_total = len(results["status"])

    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info("Grid points: %d total, %d successful inversions", n_total, n_success)
    logger.info("")

    if n_success > 0:
        ct_ours = results["ct_ours"][success]
        ct_swim = results["ct_swim"][success]
        valid_both = np.isfinite(ct_ours) & np.isfinite(ct_swim)

        logger.info("Our CT:  min=%.3f, max=%.3f, mean=%.3f",
                    ct_ours.min(), ct_ours.max(), ct_ours.mean())
        if valid_both.sum() > 0:
            logger.info("SWIM CT: min=%.3f, max=%.3f, mean=%.3f",
                        ct_swim[valid_both].min(), ct_swim[valid_both].max(),
                        ct_swim[valid_both].mean())

            # Correlation
            corr = np.corrcoef(ct_ours[valid_both], ct_swim[valid_both])[0, 1]
            rmse = np.sqrt(((ct_ours[valid_both] - ct_swim[valid_both])**2).mean())
            logger.info("Correlation (ours vs SWIM): r=%.3f", corr)
            logger.info("RMSE (ours vs SWIM): %.3f", rmse)

            # Ice detection agreement
            ours_ice = ct_ours[valid_both] > 0.3
            swim_ice = ct_swim[valid_both] > 0.3
            agreement = (ours_ice == swim_ice).mean() * 100
            logger.info("Ice detection agreement (CT>0.3): %.1f%%", agreement)

        logger.info("")
        logger.info("Parameter ranges (successful points):")
        logger.info("  k_upper: %.5f – %.5f W/m/K", results["k_upper"][success].min(),
                    results["k_upper"][success].max())
        logger.info("  k_lower: %.4f – %.4f W/m/K", results["k_lower"][success].min(),
                    results["k_lower"][success].max())
        logger.info("  boundary: %.2f – %.2f m", results["boundary"][success].min(),
                    results["boundary"][success].max())
        logger.info("  Mean inversion time: %.1fs per point",
                    results["elapsed"][success].mean())
        logger.info("  Total inversion time: %.1fs",
                    results["elapsed"][success].sum())


def main():
    parser = argparse.ArgumentParser(description="Arcadia-wide DeepONet CT mapping")
    parser.add_argument("--grid-step", type=float, default=3.0,
                        help="Grid spacing in degrees (default: 3)")
    parser.add_argument("--lat-min", type=float, default=40.0)
    parser.add_argument("--lat-max", type=float, default=60.0)
    parser.add_argument("--lon-min", type=float, default=155.0)
    parser.add_argument("--lon-max", type=float, default=215.0)
    parser.add_argument("--max-products", type=int, default=3000,
                        help="Max THEMIS products to download")
    parser.add_argument("--inv-steps", type=int, default=500)
    parser.add_argument("--inv-lr", type=float, default=0.03)
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip THEMIS download, use cached images")
    parser.add_argument("--model-path", type=str,
                        default=str(DATA_DIR / "deeponet_model.pt"))
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ARCADIA_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Download THEMIS data ──
    logger.info("=" * 60)
    logger.info("STEP 1: THEMIS IRBTR Download for Arcadia Planitia")
    logger.info("=" * 60)
    raw_dir = download_arcadia_themis(
        max_products=args.max_products,
        skip_download=args.skip_download,
    )

    # ── Step 2: Load DeepONet model ──
    logger.info("=" * 60)
    logger.info("STEP 2: Load DeepONet Model")
    logger.info("=" * 60)

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"DeepONet model not found at {model_path}. "
            "Run run_deeponet.py first to train the model."
        )

    cfg = DeepONetConfig()
    model = DeepONet(
        branch_dim=cfg.branch_dim, trunk_dim=cfg.trunk_dim,
        hidden_dim=cfg.hidden_dim, latent_dim=cfg.latent_dim,
        n_layers=cfg.n_layers,
    )
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    logger.info("Loaded DeepONet model from %s (%d params)",
                model_path, sum(p.numel() for p in model.parameters()))

    # ── Step 3: Grid inversion ──
    logger.info("=" * 60)
    logger.info("STEP 3: Arcadia Grid Inversion")
    logger.info("=" * 60)

    results = run_arcadia_grid(
        raw_dir, model, cfg,
        lat_range=(args.lat_min, args.lat_max),
        lon_range=(args.lon_min, args.lon_max),
        grid_step=args.grid_step,
        inv_steps=args.inv_steps,
        inv_lr=args.inv_lr,
    )

    # ── Step 4: Save results ──
    results_path = DATA_DIR / "arcadia_wide_results.npz"
    np.savez_compressed(results_path, **results)
    logger.info("Saved results → %s", results_path)

    # ── Step 5: Generate maps ──
    logger.info("=" * 60)
    logger.info("STEP 5: Generate CT Comparison Map")
    logger.info("=" * 60)

    plot_path = DATA_DIR / "arcadia_ct_comparison.png"
    plot_ct_comparison(results, plot_path)

    # ── Summary ──
    print_summary(results)

    logger.info("=" * 60)
    logger.info("DONE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
