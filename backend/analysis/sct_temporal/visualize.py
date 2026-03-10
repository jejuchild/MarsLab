"""Visualization for SCT temporal change detection results."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def plot_displacement_field(
    displacement,
    output_path: Path,
    title: str = "Displacement Field",
    vmax_m: Optional[float] = None,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    mag = displacement.magnitude_m.copy()
    mag[~displacement.valid_mask] = np.nan
    if vmax_m is None:
        vmax_m = float(np.nanpercentile(mag, 98)) if np.any(np.isfinite(mag)) else 1.0

    im0 = axes[0].imshow(mag, cmap="hot", vmin=0, vmax=vmax_m)
    axes[0].set_title("Displacement Magnitude (m)")
    plt.colorbar(im0, ax=axes[0], shrink=0.8)

    row_m = displacement.row_disp_m.copy()
    row_m[~displacement.valid_mask] = np.nan
    vlim = max(abs(np.nanmin(row_m)), abs(np.nanmax(row_m)), 0.01)
    im1 = axes[1].imshow(row_m, cmap="RdBu_r", vmin=-vlim, vmax=vlim)
    axes[1].set_title("N-S Displacement (m)")
    plt.colorbar(im1, ax=axes[1], shrink=0.8)

    col_m = displacement.col_disp_m.copy()
    col_m[~displacement.valid_mask] = np.nan
    vlim = max(abs(np.nanmin(col_m)), abs(np.nanmax(col_m)), 0.01)
    im2 = axes[2].imshow(col_m, cmap="RdBu_r", vmin=-vlim, vmax=vlim)
    axes[2].set_title("E-W Displacement (m)")
    plt.colorbar(im2, ax=axes[2], shrink=0.8)

    fig.suptitle(title, fontsize=14)
    for ax in axes:
        ax.set_xlabel(f"Grid index (step={displacement.col_centers[1]-displacement.col_centers[0] if len(displacement.col_centers)>1 else 0} px)")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved displacement field: {output_path}")


def plot_retreat_analysis(
    analysis,
    output_path: Path,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_segments = len(analysis.segments)
    if n_segments == 0:
        logger.warning("No segments to plot")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    retreat_rates = [
        s.mean_retreat_m / analysis.time_gap_mars_years for s in analysis.segments
    ]
    axes[0].barh(range(n_segments), retreat_rates, color="steelblue")
    axes[0].set_xlabel("Retreat Rate (m/Mars yr)")
    axes[0].set_ylabel("Scarp Segment")
    axes[0].set_title("Scarp Retreat Rates")
    axes[0].axvline(analysis.mean_retreat_rate_m_per_yr, color="red", ls="--", label="Mean")
    axes[0].legend()

    orientations = [s.orientation_deg for s in analysis.segments]
    lengths = [s.length_m for s in analysis.segments]
    axes[1].scatter(orientations, retreat_rates, s=[l/10 for l in lengths], alpha=0.7)
    axes[1].set_xlabel("Scarp Orientation (°)")
    axes[1].set_ylabel("Retreat Rate (m/Mars yr)")
    axes[1].set_title("Retreat vs Orientation")

    for s in analysis.segments:
        rect = plt.Rectangle(
            (s.col_start, s.row_start),
            s.col_end - s.col_start,
            s.row_end - s.row_start,
            fill=False,
            edgecolor="red",
            linewidth=1.5,
        )
        axes[2].add_patch(rect)

    mag = analysis.displacement_field.magnitude_m.copy()
    mag[~analysis.displacement_field.valid_mask] = np.nan
    step = analysis.displacement_field.row_centers[1] - analysis.displacement_field.row_centers[0] if len(analysis.displacement_field.row_centers) > 1 else 1
    axes[2].imshow(mag, cmap="hot", vmin=0, vmax=float(np.nanpercentile(mag, 98)) if np.any(np.isfinite(mag)) else 1)
    axes[2].set_title("Scarp Segments on Displacement Map")

    fig.suptitle(
        f"SCT Retreat Analysis — {analysis.time_gap_mars_years:.1f} Mars yr gap\n"
        f"Mean: {analysis.mean_retreat_rate_m_per_yr:.2f} m/yr | "
        f"Segments: {n_segments} | "
        f"Total scarp: {analysis.total_scarp_length_m:.0f} m",
        fontsize=12,
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved retreat analysis: {output_path}")


def plot_temporal_comparison(
    img1: np.ndarray,
    img2: np.ndarray,
    displacement,
    output_path: Path,
    title: str = "Temporal Comparison",
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(img1, cmap="gray")
    axes[0].set_title("T1 (Reference)")

    axes[1].imshow(img2, cmap="gray")
    axes[1].set_title("T2 (Secondary)")

    mag = displacement.magnitude_m.copy()
    mag[~displacement.valid_mask] = np.nan
    vmax = float(np.nanpercentile(mag, 98)) if np.any(np.isfinite(mag)) else 1.0
    im = axes[2].imshow(mag, cmap="hot", vmin=0, vmax=vmax)
    axes[2].set_title("Displacement Magnitude (m)")
    plt.colorbar(im, ax=axes[2], shrink=0.8)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved temporal comparison: {output_path}")
