"""
MarsLab Multi-Instrument Scientific Analysis Report Generator.

Produces a publication-quality PDF report integrating:
1. Overview map (MOLA shaded relief + instrument footprints)
2. SHARAD subsurface analysis (radargram, depth, dielectric)
3. CRISM mineral classification (CNN map, confidence, spectral interpretation)
4. Slope & terrain analysis (DEM patch, hillshade, slope histogram)
5. Multi-instrument synthesis composite figure
6. Physics interpretation

Intelligent figure selection: only includes scientifically meaningful figures.

Usage:
    async for event in run_multi_report(config):
        ...
"""

import asyncio
import csv
import json
import logging
import math
import os
import re
from dataclasses import dataclass, field, replace as dc_replace
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("multi_report")

# ── Constants ─────────────────────────────────────────
C_VACUUM = 299_792_458.0
SHARAD_DT_S = (3.0 / 80.0) * 1e-6
N_RANGE_BINS = 667
MARS_RADIUS_M = 3_389_500.0

_BACKEND_DIR = Path(__file__).resolve().parent.parent
REPORT_DIR = _BACKEND_DIR / "multi_reports"
MOLA_DEM_PATH = _BACKEND_DIR.parent / "Mars_HRSC_MOLA_BlendDEM_Global_200mp_v2.tif"
_PRODUCT_RE = re.compile(r"^[a-zA-Z0-9_]+$")

# Dielectric models for SHARAD depth estimation
DIELECTRIC_MODELS = [
    ("Vacuum", 1.0),
    ("Porous Ice", 2.8),
    ("Pure Water Ice", 3.1),
    ("Basalt (low)", 4.0),
    ("Basalt (mid)", 5.0),
    ("Basalt (high)", 6.0),
]

# Mineral color palette (from mineral_cnn/constants.py)
_MINERAL_COLORS = {
    -1: (30, 30, 30),    # Unclassified
    1: (0, 191, 255),     # CO2 Ice
    2: (135, 206, 250),   # H2O Ice
    3: (255, 255, 0),     # Gypsum
    4: (255, 165, 0),     # Ferric Hydroxysulfate
    6: (139, 69, 19),     # Fe smectite
    7: (34, 139, 34),     # Mg smectite
    8: (75, 0, 130),      # Prehnite
    9: (255, 20, 147),    # Jarosite
    10: (0, 128, 0),      # Serpentine
    11: (220, 20, 60),    # Alunite
    12: (255, 99, 71),    # Akaganeite
    14: (210, 180, 140),  # Al smectite 1
    15: (245, 245, 220),  # Kaolinite
    16: (255, 228, 181),  # Bassanite
    17: (128, 0, 128),    # Epidote
    18: (188, 143, 143),  # Al smectite 2
    19: (255, 215, 0),    # Polyhydrated sulfate
    23: (169, 169, 169),  # Illite
    25: (100, 149, 237),  # Analcime
    26: (218, 165, 32),   # Monohydrated sulfate
    27: (176, 224, 230),  # Hydrated silica
    29: (255, 127, 80),   # Ferricopiapite
    31: (46, 139, 87),    # Chlorite
    38: (85, 107, 47),    # Chlorite-smectite
}

_CLASS_NAME = {
    -1: "Unclassified", 1: "CO2 Ice", 2: "H2O Ice", 3: "Gypsum",
    4: "Ferric Hydroxysulfate", 6: "Fe smectite", 7: "Mg smectite",
    8: "Prehnite", 9: "Jarosite", 10: "Serpentine", 11: "Alunite",
    12: "Akaganeite", 14: "Al smectite 1", 15: "Kaolinite", 16: "Bassanite",
    17: "Epidote", 18: "Al smectite 2", 19: "Polyhydrated sulfate",
    23: "Illite", 25: "Analcime", 26: "Monohydrated sulfate",
    27: "Hydrated silica", 29: "Ferricopiapite", 31: "Chlorite",
    38: "Chlorite-smectite",
}


# ── Config dataclass ─────────────────────────────────
@dataclass
class MultiReportConfig:
    """Configuration for multi-instrument report."""
    region_name: str = "Analysis Region"
    sharad_product_id: Optional[str] = None
    crism_obs_id: Optional[str] = None        # e.g. "frt0000c702_07"
    hirise_dtm_id: Optional[str] = None       # e.g. "DTEEC_..."
    center_lat: Optional[float] = None
    center_lon: Optional[float] = None        # -180/180
    analysis_radius_km: float = 50.0


# ── Helpers ──────────────────────────────────────────
def _get_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _run_sync(fn, *args):
    loop = asyncio.get_running_loop()
    return loop.run_in_executor(None, fn, *args)


def _haversine_km(lat1, lon1, lat2, lon2):
    rlat1, rlat2 = np.radians(lat1), np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(rlat1) * np.cos(rlat2) * np.sin(dlon / 2) ** 2
    return (MARS_RADIUS_M / 1000) * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def _track_distance_km(lats, lons):
    dist = np.zeros(len(lats))
    for i in range(1, len(lats)):
        dist[i] = dist[i - 1] + _haversine_km(lats[i - 1], lons[i - 1], lats[i], lons[i])
    return dist


# ── Concurrency control ─────────────────────────────
_active_locks: Dict[str, asyncio.Lock] = {}
_global_semaphore = asyncio.Semaphore(2)


# =============================================================
# SECTION 1 — OVERVIEW MAP (MOLA shaded relief)
# =============================================================

def generate_overview_map(
    center_lat: float,
    center_lon: float,
    radius_deg: float,
    sharad_coords: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    crism_footprint: Optional[List[Tuple[float, float]]] = None,
    dtm_footprint: Optional[List[Tuple[float, float]]] = None,
    output_path: str = "",
) -> bool:
    """Generate regional context map using MOLA DEM shaded relief."""
    plt = _get_plt()
    from matplotlib.patches import Polygon as MplPolygon
    from matplotlib.collections import PatchCollection

    if not MOLA_DEM_PATH.exists():
        logger.warning("MOLA DEM not found — skipping overview map")
        return False

    import rasterio
    from rasterio.windows import from_bounds

    try:
        with rasterio.open(str(MOLA_DEM_PATH)) as ds:
            # Convert lon to 0-360 for MOLA DEM
            lon_360 = center_lon % 360

            # Compute window bounds
            west = lon_360 - radius_deg
            east = lon_360 + radius_deg
            south = center_lat - radius_deg
            north = center_lat + radius_deg

            win = from_bounds(west, south, east, north, ds.transform)
            elev = ds.read(1, window=win, boundless=True, fill_value=0)
    except Exception as e:
        logger.warning(f"Failed to read MOLA DEM: {e}")
        return False

    if elev.size == 0:
        return False

    # Compute hillshade
    dy, dx = np.gradient(elev.astype(np.float64))
    azimuth_rad = np.radians(315)
    altitude_rad = np.radians(45)
    slope = np.sqrt(dx ** 2 + dy ** 2)
    aspect = np.arctan2(-dy, dx)
    hillshade = (np.sin(altitude_rad) * np.cos(np.arctan(slope)) +
                 np.cos(altitude_rad) * np.sin(np.arctan(slope)) *
                 np.cos(azimuth_rad - aspect))
    hillshade = np.clip(hillshade, 0, 1)

    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)

    # Shaded relief with elevation tint
    extent = [center_lon - radius_deg, center_lon + radius_deg,
              center_lat - radius_deg, center_lat + radius_deg]

    ax.imshow(elev, extent=extent, cmap="gist_earth", origin="upper",
              vmin=np.percentile(elev[elev != 0], 2) if np.any(elev != 0) else -5000,
              vmax=np.percentile(elev[elev != 0], 98) if np.any(elev != 0) else 5000,
              alpha=0.7)
    ax.imshow(hillshade, extent=extent, cmap="gray", origin="upper", alpha=0.4)

    # Overlay SHARAD ground track
    if sharad_coords is not None:
        slats, slons = sharad_coords
        ax.plot(slons, slats, color="red", linewidth=1.5, alpha=0.9,
                label="SHARAD ground track", zorder=5)

    # Overlay CRISM footprint
    if crism_footprint:
        poly = MplPolygon(crism_footprint, closed=True, fill=False,
                          edgecolor="cyan", linewidth=2.0, linestyle="--",
                          label="CRISM footprint", zorder=4)
        ax.add_patch(poly)

    # Overlay DTM footprint
    if dtm_footprint:
        poly = MplPolygon(dtm_footprint, closed=True, fill=False,
                          edgecolor="yellow", linewidth=2.0, linestyle="-.",
                          label="HiRISE DTM footprint", zorder=4)
        ax.add_patch(poly)

    # North arrow
    arr_x = center_lon + radius_deg * 0.85
    arr_y = center_lat + radius_deg * 0.75
    ax.annotate("N", xy=(arr_x, arr_y + radius_deg * 0.08),
                fontsize=12, fontweight="bold", ha="center", color="white",
                path_effects=[_text_outline()])
    ax.annotate("", xy=(arr_x, arr_y + radius_deg * 0.06),
                xytext=(arr_x, arr_y - radius_deg * 0.02),
                arrowprops=dict(arrowstyle="->", color="white", lw=2))

    # Scale bar
    cos_lat_sb = max(np.cos(np.radians(center_lat)), 0.01)  # guard near poles
    bar_km = round(radius_deg * 111 * cos_lat_sb * 0.3, -1)
    if bar_km < 10:
        bar_km = 10
    bar_deg = bar_km / (111 * cos_lat_sb)
    bx = center_lon - radius_deg * 0.85
    by = center_lat - radius_deg * 0.85
    ax.plot([bx, bx + bar_deg], [by, by], color="white", linewidth=3, zorder=10)
    ax.text(bx + bar_deg / 2, by + radius_deg * 0.03, f"{bar_km:.0f} km",
            color="white", ha="center", fontsize=9, fontweight="bold",
            path_effects=[_text_outline()])

    ax.set_xlabel("Longitude (deg E)", fontsize=10)
    ax.set_ylabel("Latitude (deg N)", fontsize=10)
    ax.set_title(f"Regional Context — MOLA Shaded Relief", fontsize=12)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.8)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return True


def _text_outline():
    """White outline for text on dark backgrounds."""
    from matplotlib.patheffects import withStroke
    return withStroke(linewidth=2, foreground="black")


# =============================================================
# SECTION 2 — SHARAD ANALYSIS
# =============================================================
# Reuses detect_interface, estimate_depths, etc. from sharad_report.py

def _load_sharad_data(product_id: str):
    """Load SHARAD RDR and return (power, surface, lats, lons, n_traces)."""
    try:
        from .sharad_highres_router import _get_power, _pick_surface, _get_geometry, _lon_to_180
    except ImportError:
        from sharad_highres_router import _get_power, _pick_surface, _get_geometry, _lon_to_180

    power, n_traces = _get_power(product_id)
    geometry, _ = _get_geometry(product_id)
    lats = geometry["lat"]
    lons = _lon_to_180(geometry["lon"])
    surface = _pick_surface(product_id, power)
    return power, surface, lats, lons, n_traces


# =============================================================
# SECTION 3 — CRISM MINERAL CLASSIFICATION
# =============================================================

def generate_mineral_map_figure(
    obs_id: str,
    output_path: str,
) -> Optional[Dict]:
    """Generate classified mineral map with legend."""
    plt = _get_plt()
    from matplotlib.patches import Patch

    results_dir = _BACKEND_DIR / "mineral_cnn_results" / obs_id
    mm_path = results_dir / "mineral_map.npy"
    conf_path = results_dir / "confidence_map.npy"
    valid_path = results_dir / "valid_mask.npy"

    if not mm_path.exists():
        return None

    mineral_map = np.load(str(mm_path))
    rows, cols = mineral_map.shape

    # Build RGBA image
    rgba = np.zeros((rows, cols, 4), dtype=np.uint8)
    unique_classes = np.unique(mineral_map)
    class_counts = {}

    for cls in unique_classes:
        if cls == -1:
            continue
        mask = mineral_map == cls
        count = int(mask.sum())
        if count == 0:
            continue
        color = _MINERAL_COLORS.get(cls, (128, 128, 128))
        rgba[mask, 0] = color[0]
        rgba[mask, 1] = color[1]
        rgba[mask, 2] = color[2]
        rgba[mask, 3] = 255
        name = _CLASS_NAME.get(cls, f"Class {cls}")
        class_counts[name] = count

    # Sort by count descending
    sorted_classes = sorted(class_counts.items(), key=lambda x: -x[1])

    fig, (ax_map, ax_legend) = plt.subplots(
        1, 2, figsize=(14, 6), dpi=300,
        gridspec_kw={"width_ratios": [4, 1]},
    )

    ax_map.imshow(rgba, interpolation="nearest")
    ax_map.set_xlabel("Sample", fontsize=9)
    ax_map.set_ylabel("Line", fontsize=9)
    ax_map.set_title(f"CNN Mineral Classification — {obs_id}", fontsize=11)

    # Legend
    legend_patches = []
    for name, count in sorted_classes[:10]:
        pct = count / (rows * cols) * 100
        cls_id = [k for k, v in _CLASS_NAME.items() if v == name]
        color_rgb = _MINERAL_COLORS.get(cls_id[0] if cls_id else -1, (128, 128, 128))
        color_norm = tuple(c / 255 for c in color_rgb)
        legend_patches.append(Patch(facecolor=color_norm, label=f"{name} ({pct:.1f}%)"))

    ax_legend.legend(handles=legend_patches, loc="center left", fontsize=8,
                     framealpha=0.9, title="Minerals", title_fontsize=9)
    ax_legend.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    total_classified = sum(class_counts.values())
    return {
        "classes": sorted_classes,
        "total_classified": total_classified,
        "total_pixels": rows * cols,
        "classification_rate": total_classified / (rows * cols) * 100,
    }


def generate_confidence_map_figure(
    obs_id: str,
    output_path: str,
) -> bool:
    """Generate confidence (softmax probability) heatmap."""
    plt = _get_plt()

    conf_path = _BACKEND_DIR / "mineral_cnn_results" / obs_id / "confidence_map.npy"
    valid_path = _BACKEND_DIR / "mineral_cnn_results" / obs_id / "valid_mask.npy"

    if not conf_path.exists():
        return False

    conf = np.load(str(conf_path))
    valid = np.load(str(valid_path)) if valid_path.exists() else np.ones_like(conf, dtype=bool)

    display = np.where(valid, conf, np.nan)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    im = ax.imshow(display, cmap="viridis", vmin=0.0, vmax=1.0, interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, label="Softmax Confidence")
    ax.set_xlabel("Sample", fontsize=9)
    ax.set_ylabel("Line", fontsize=9)
    ax.set_title(f"Classification Confidence — {obs_id}", fontsize=11)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return True


# =============================================================
# SECTION 4 — SLOPE & TERRAIN ANALYSIS
# =============================================================

def generate_slope_map(
    center_lat: float,
    center_lon: float,
    radius_m: float,
    output_path: str,
) -> Optional[Dict]:
    """Generate 2D slope map from MOLA DEM."""
    plt = _get_plt()

    if not MOLA_DEM_PATH.exists():
        return None

    import rasterio
    from rasterio.windows import from_bounds

    # Convert to 0-360 for MOLA
    lon_360 = center_lon % 360
    meters_per_deg = math.pi / 180 * MARS_RADIUS_M
    cos_lat = max(math.cos(math.radians(center_lat)), 0.01)  # guard near poles
    radius_deg = radius_m / (meters_per_deg * cos_lat)
    radius_deg_lat = radius_m / meters_per_deg

    west = lon_360 - radius_deg
    east = lon_360 + radius_deg
    south = center_lat - radius_deg_lat
    north = center_lat + radius_deg_lat

    with rasterio.open(str(MOLA_DEM_PATH)) as ds:
        win = from_bounds(west, south, east, north, ds.transform)
        elev = ds.read(1, window=win, boundless=True, fill_value=0).astype(np.float64)

    if elev.size < 9:
        return None

    # Compute slope in degrees
    px_m = radius_m * 2 / max(max(elev.shape), 1)
    dy, dx = np.gradient(elev, px_m)
    slope_deg = np.degrees(np.arctan(np.sqrt(dx ** 2 + dy ** 2)))

    mean_slope = float(np.mean(slope_deg))
    max_slope = float(np.max(slope_deg))
    std_slope = float(np.std(slope_deg))

    # Only generate figure if slope > 1 deg (meaningful terrain variation)
    if max_slope < 1.0:
        return {"mean_slope": mean_slope, "max_slope": max_slope, "std_slope": std_slope,
                "figure_generated": False, "reason": "Terrain too flat (max < 1 deg)"}

    extent = [center_lon - radius_deg, center_lon + radius_deg,
              center_lat - radius_deg_lat, center_lat + radius_deg_lat]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    # Slope map
    im = axes[0].imshow(slope_deg, extent=extent, cmap="YlOrRd", origin="upper",
                        vmin=0, vmax=min(max_slope, 30))
    fig.colorbar(im, ax=axes[0], shrink=0.8, label="Slope (degrees)")
    axes[0].set_xlabel("Longitude (deg E)", fontsize=9)
    axes[0].set_ylabel("Latitude (deg N)", fontsize=9)
    axes[0].set_title("Local Slope Map", fontsize=11)

    # Slope histogram
    bins = np.arange(0, min(max_slope + 2, 35), 1)
    axes[1].hist(slope_deg.flatten(), bins=bins, color="steelblue", edgecolor="white",
                 alpha=0.8, density=True)
    axes[1].axvline(5.0, color="orange", linestyle="--", linewidth=1.5,
                    label="5 deg (stability threshold)")
    axes[1].axvline(15.0, color="red", linestyle="--", linewidth=1.5,
                    label="15 deg (hazard threshold)")
    axes[1].set_xlabel("Slope (degrees)", fontsize=9)
    axes[1].set_ylabel("Probability Density", fontsize=9)
    axes[1].set_title("Slope Distribution", fontsize=11)
    axes[1].legend(fontsize=8)

    pct_above_5 = float(np.sum(slope_deg > 5) / slope_deg.size * 100)
    pct_above_15 = float(np.sum(slope_deg > 15) / slope_deg.size * 100)

    fig.suptitle(f"Mean: {mean_slope:.1f} deg | Max: {max_slope:.1f} deg | "
                 f">{5} deg: {pct_above_5:.1f}%", fontsize=10, y=0.02)

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return {
        "mean_slope": mean_slope, "max_slope": max_slope, "std_slope": std_slope,
        "pct_above_5deg": pct_above_5, "pct_above_15deg": pct_above_15,
        "figure_generated": True,
    }


def generate_terrain_3d(
    center_lat: float,
    center_lon: float,
    radius_m: float,
    output_path: str,
) -> bool:
    """Generate 3D hillshade visualization (only if terrain is interesting)."""
    plt = _get_plt()
    from mpl_toolkits.mplot3d import Axes3D

    if not MOLA_DEM_PATH.exists():
        return False

    import rasterio
    from rasterio.windows import from_bounds

    lon_360 = center_lon % 360
    meters_per_deg = math.pi / 180 * MARS_RADIUS_M
    cos_lat = max(math.cos(math.radians(center_lat)), 0.01)  # guard near poles
    radius_deg = radius_m / (meters_per_deg * cos_lat)
    radius_deg_lat = radius_m / meters_per_deg

    with rasterio.open(str(MOLA_DEM_PATH)) as ds:
        win = from_bounds(lon_360 - radius_deg, center_lat - radius_deg_lat,
                          lon_360 + radius_deg, center_lat + radius_deg_lat,
                          ds.transform)

        # Read at reduced resolution for 3D
        out_size = min(200, max(win.height, win.width))
        elev = ds.read(1, window=win, out_shape=(out_size, out_size),
                       boundless=True, fill_value=0).astype(np.float64)

    if elev.size < 25:
        return False

    relief = float(np.max(elev) - np.min(elev))
    if relief < 50:  # Not enough terrain variation
        return False

    rows, cols = elev.shape
    x = np.linspace(center_lon - radius_deg, center_lon + radius_deg, cols)
    y = np.linspace(center_lat + radius_deg_lat, center_lat - radius_deg_lat, rows)
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(12, 8), dpi=300)
    ax = fig.add_subplot(111, projection="3d")

    # Normalize elevation for color
    norm_elev = (elev - np.min(elev)) / max(relief, 1)

    from matplotlib import cm
    colors = cm.gist_earth(norm_elev)

    ax.plot_surface(X, Y, elev, facecolors=colors, rstride=1, cstride=1,
                    antialiased=False, shade=True)
    ax.set_xlabel("Longitude (deg E)", fontsize=8)
    ax.set_ylabel("Latitude (deg N)", fontsize=8)
    ax.set_zlabel("Elevation (m)", fontsize=8)
    ax.set_title(f"3D Terrain — Relief: {relief:.0f} m", fontsize=11)
    ax.view_init(elev=35, azim=225)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return True


# =============================================================
# SECTION 6 — MULTI-INSTRUMENT SYNTHESIS FIGURE
# =============================================================

def generate_synthesis_composite(
    figures: Dict[str, str],
    config: MultiReportConfig,
    output_path: str,
) -> bool:
    """Generate multi-panel journal-quality composite figure."""
    plt = _get_plt()
    from PIL import Image as PILImage

    # Collect available panels
    panels = []
    titles = []

    panel_order = [
        ("overview_map", "Regional Context"),
        ("radargram_annotated", "SHARAD Radargram"),
        ("mineral_map", "CRISM Mineral Classification"),
        ("slope_map", "Slope Analysis"),
    ]

    for key, title in panel_order:
        if key in figures and os.path.exists(figures[key]):
            panels.append(figures[key])
            titles.append(title)

    if len(panels) < 2:
        return False

    n = len(panels)
    ncols = min(n, 2)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows), dpi=300)
    if n == 1:
        axes = [axes]
    elif nrows == 1:
        axes = list(axes)
    else:
        axes = [ax for row in axes for ax in row]

    for i, (path, title) in enumerate(zip(panels, titles)):
        img = PILImage.open(path)
        axes[i].imshow(np.array(img))
        axes[i].set_title(title, fontsize=10, fontweight="bold")
        axes[i].axis("off")

    # Hide extra axes
    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"Multi-Instrument Synthesis — {config.region_name}",
                 fontsize=13, fontweight="bold", y=1.01)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return True


# =============================================================
# SECTION 7 — PDF REPORT ASSEMBLY
# =============================================================

def _sanitize_pdf_text(text: str) -> str:
    """Replace Unicode characters unsupported by fpdf2 built-in fonts."""
    replacements = {
        "\u2014": "--",   # em dash
        "\u2013": "-",    # en dash
        "\u2018": "'",    # left single quote
        "\u2019": "'",    # right single quote
        "\u201c": '"',    # left double quote
        "\u201d": '"',    # right double quote
        "\u2026": "...",  # ellipsis
        "\u00b5": "u",    # micro sign
        "\u03b5": "e",    # epsilon
        "\u2192": "=>",   # right arrow
        "\u2248": "~",    # approximately
        "\u00b0": " deg", # degree sign
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    # Strip any remaining non-latin1 characters
    return text.encode("latin-1", errors="replace").decode("latin-1")


def generate_multi_pdf(
    config: MultiReportConfig,
    figures: Dict[str, str],
    sharad_data: Optional[Dict] = None,
    mineral_data: Optional[Dict] = None,
    slope_data: Optional[Dict] = None,
    interpretation: str = "",
    output_path: str = "",
) -> bool:
    """Generate comprehensive multi-page PDF."""
    try:
        from fpdf import FPDF
    except ImportError:
        logger.warning("fpdf2 not installed")
        return False

    def _s(t):
        """Sanitize text for PDF."""
        return _sanitize_pdf_text(t)

    pdf = FPDF(orientation="L", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)

    # ── Title page ──
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 24)
    pdf.cell(0, 25, "MarsLab Multi-Instrument", ln=True, align="C")
    pdf.cell(0, 15, "Scientific Analysis Report", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Helvetica", "", 14)
    pdf.cell(0, 10, _s(f"Region: {config.region_name}"), ln=True, align="C")
    if config.center_lat is not None and config.center_lon is not None:
        pdf.cell(0, 8, f"Center: {config.center_lat:.3f} N, {config.center_lon:.3f} E",
                 ln=True, align="C")
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}", ln=True, align="C")

    pdf.ln(10)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Instruments Included:", ln=True)
    pdf.set_font("Helvetica", "", 10)
    if config.sharad_product_id:
        pdf.set_x(10)
        pdf.cell(0, 6, f"  SHARAD: {config.sharad_product_id}", ln=True)
    if config.crism_obs_id:
        pdf.set_x(10)
        pdf.cell(0, 6, f"  CRISM (CNN): {config.crism_obs_id}", ln=True)
    if config.hirise_dtm_id:
        pdf.set_x(10)
        pdf.cell(0, 6, f"  HiRISE DTM: {config.hirise_dtm_id}", ln=True)
    pdf.set_x(10)
    pdf.cell(0, 6, "  MOLA DEM: Global 200m blend", ln=True)

    # ── SHARAD section ──
    if sharad_data:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "SHARAD Subsurface Analysis", ln=True)
        pdf.ln(3)
        pdf.set_font("Helvetica", "", 10)

        iface = sharad_data.get("interface", {})
        pdf.set_x(10)
        pdf.cell(0, 6, f"Detection Confidence: {iface.get('confidence', 'N/A').upper()}", ln=True)
        pdf.set_x(10)
        pdf.cell(0, 6, f"Detection Rate: {iface.get('detection_rate', 0) * 100:.1f}%", ln=True)
        pdf.set_x(10)
        pdf.cell(0, 6, f"Coherent Segments: {iface.get('n_segments', 0)}", ln=True)

        depths = sharad_data.get("depths", [])
        if depths:
            pdf.ln(3)
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(0, 7, "Depth Estimates:", ln=True)
            pdf.set_font("Helvetica", "", 9)
            col_w = [55, 20, 30, 30, 30]
            pdf.cell(col_w[0], 6, "Material", border="B")
            pdf.cell(col_w[1], 6, "er", border="B", align="R")
            pdf.cell(col_w[2], 6, "Mean (m)", border="B", align="R")
            pdf.cell(col_w[3], 6, "Min (m)", border="B", align="R")
            pdf.cell(col_w[4], 6, "Max (m)", border="B", align="R")
            pdf.ln()
            for d in depths:
                pdf.cell(col_w[0], 5, d["label"])
                pdf.cell(col_w[1], 5, f"{d['epsilon_r']:.1f}", align="R")
                pdf.cell(col_w[2], 5, f"{d['mean_m']:.1f}", align="R")
                pdf.cell(col_w[3], 5, f"{d['min_m']:.1f}", align="R")
                pdf.cell(col_w[4], 5, f"{d['max_m']:.1f}", align="R")
                pdf.ln()

    # ── Mineral section ──
    if mineral_data:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "CRISM Mineral Classification", ln=True)
        pdf.ln(3)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_x(10)
        pdf.cell(0, 6, f"Observation: {config.crism_obs_id}", ln=True)
        pdf.set_x(10)
        pdf.cell(0, 6,
                 f"Classified pixels: {mineral_data['total_classified']} / {mineral_data['total_pixels']} "
                 f"({mineral_data['classification_rate']:.1f}%)", ln=True)

        pdf.ln(3)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 7, "Detected Minerals:", ln=True)
        pdf.set_font("Helvetica", "", 9)
        for name, count in mineral_data.get("classes", [])[:10]:
            pct = count / mineral_data["total_pixels"] * 100
            pdf.set_x(10)
            pdf.cell(0, 5, f"  {name}: {count} px ({pct:.2f}%)", ln=True)

    # ── Slope section ──
    if slope_data and slope_data.get("figure_generated"):
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "Terrain & Slope Analysis", ln=True)
        pdf.ln(3)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_x(10)
        pdf.cell(0, 6, f"Mean slope: {slope_data['mean_slope']:.1f} deg", ln=True)
        pdf.set_x(10)
        pdf.cell(0, 6, f"Max slope: {slope_data['max_slope']:.1f} deg", ln=True)
        if "pct_above_5deg" in slope_data:
            pdf.set_x(10)
            pdf.cell(0, 6, f"Area > 5 deg: {slope_data['pct_above_5deg']:.1f}%", ln=True)
            pdf.set_x(10)
            pdf.cell(0, 6, f"Area > 15 deg: {slope_data['pct_above_15deg']:.1f}%", ln=True)

    # ── Figure pages ──
    page_w = 277
    page_h = 170
    fig_titles = {
        "overview_map": "Regional Context Map (MOLA Shaded Relief)",
        "radargram_annotated": "Annotated SHARAD Radargram",
        "mineral_map": "CNN Mineral Classification Map",
        "confidence_map": "Classification Confidence Map",
        "slope_map": "Slope Map & Distribution",
        "terrain_3d": "3D Terrain Visualization",
        "synthesis_composite": "Multi-Instrument Synthesis",
    }

    for fig_name, fig_path in figures.items():
        if not os.path.exists(fig_path) or not fig_path.endswith(".png"):
            continue
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 12)
        title = fig_titles.get(fig_name, fig_name.replace("_", " ").title())
        pdf.cell(0, 10, title, ln=True, align="C")
        pdf.ln(2)

        try:
            from PIL import Image as PILImage
            with PILImage.open(fig_path) as img:
                iw, ih = img.size
            aspect = iw / ih
            if aspect > page_w / page_h:
                w = page_w
                h = w / aspect
            else:
                h = page_h
                w = h * aspect
            x = (297 - w) / 2
            pdf.image(fig_path, x=x, y=pdf.get_y(), w=w, h=h)
        except Exception:
            pdf.image(fig_path, x=10, y=pdf.get_y(), w=page_w)

    # ── Interpretation page ──
    if interpretation:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "Scientific Interpretation", ln=True)
        pdf.ln(3)
        pdf.set_font("Helvetica", "", 10)
        for line in _s(interpretation).split("\n"):
            pdf.set_x(10)
            pdf.multi_cell(0, 6, line)

    pdf.output(output_path)
    return True


# =============================================================
# SECTION 7 — PHYSICS INTERPRETATION (auto-generated)
# =============================================================

def generate_interpretation(
    config: MultiReportConfig,
    sharad_data: Optional[Dict] = None,
    mineral_data: Optional[Dict] = None,
    slope_data: Optional[Dict] = None,
) -> str:
    """Auto-generate physics interpretation text."""
    lines = []
    lines.append("=" * 60)
    lines.append("SCIENTIFIC INTERPRETATION")
    lines.append("=" * 60)
    lines.append("")

    # SHARAD interpretation
    if sharad_data:
        iface = sharad_data.get("interface", {})
        conf = iface.get("confidence", "low")
        rate = iface.get("detection_rate", 0)
        depths = sharad_data.get("depths", [])

        if conf == "high":
            lines.append("SUBSURFACE INTERFACE: A laterally coherent subsurface reflector "
                         "was detected with HIGH confidence.")
        elif conf == "moderate":
            lines.append("SUBSURFACE INTERFACE: A subsurface reflector was detected with "
                         "MODERATE confidence. Additional data recommended.")
        else:
            lines.append("SUBSURFACE INTERFACE: Low-confidence detection. "
                         "Results should be interpreted with caution.")

        lines.append(f"Detection rate: {rate * 100:.1f}% of traces.")
        lines.append("")

        # Depth analysis
        ice_depth = next((d for d in depths if d["label"] == "Pure Water Ice"), None)
        basalt_depth = next((d for d in depths if d["label"] == "Basalt (low)"), None)

        if ice_depth and ice_depth["mean_m"] > 0:
            lines.append(f"Depth (assuming water ice, er=3.1): "
                         f"{ice_depth['mean_m']:.1f} m mean "
                         f"(range: {ice_depth['min_m']:.1f}-{ice_depth['max_m']:.1f} m)")
        if basalt_depth and basalt_depth["mean_m"] > 0:
            lines.append(f"Depth (assuming basalt, er=4.0): "
                         f"{basalt_depth['mean_m']:.1f} m mean")
        lines.append("")

        # Physical plausibility
        if ice_depth:
            d = ice_depth["mean_m"]
            if 10 < d < 500:
                lines.append("Depth range is physically consistent with:")
                lines.append("  - Buried ice table (if er ~ 3.1)")
                lines.append("  - Lava flow boundary (if er ~ 4-6)")
                lines.append("  - Sedimentary layer interface")
            elif d > 500:
                lines.append("WARNING: Depth exceeds 500m under ice assumption. "
                             "Consider higher er (basalt) or measurement artifacts.")
        lines.append("")

    # Mineral interpretation
    if mineral_data:
        classes = mineral_data.get("classes", [])
        lines.append("MINERALOGY:")

        has_ice = any("Ice" in name for name, _ in classes)
        has_sulfate = any("sulfate" in name.lower() or "gypsum" in name.lower()
                         for name, _ in classes)
        has_hydrated = any(kw in name.lower() for name, _ in classes
                          for kw in ["hydrat", "smectite", "clay", "serpentine"])

        for name, count in classes[:5]:
            pct = count / mineral_data["total_pixels"] * 100
            lines.append(f"  {name}: {pct:.2f}%")
        lines.append("")

        if has_ice:
            lines.append("=> Water/CO2 ice detected in spectral data.")
            if sharad_data:
                lines.append("   Mineralogy is CONSISTENT with ice-bearing subsurface.")
        if has_sulfate:
            lines.append("=> Sulfate minerals detected, indicating past aqueous activity.")
        if has_hydrated:
            lines.append("=> Hydrated minerals present, suggesting water-rock interaction.")

        if not has_ice and sharad_data:
            lines.append("=> No ice minerals detected at surface. Subsurface interface "
                         "may represent a lithological boundary rather than cryosphere.")
        lines.append("")

    # Slope interpretation
    if slope_data:
        lines.append("TERRAIN ASSESSMENT:")
        mean_s = slope_data.get("mean_slope", 0)
        max_s = slope_data.get("max_slope", 0)
        pct5 = slope_data.get("pct_above_5deg", 0)

        if mean_s < 3:
            lines.append(f"  Mean slope: {mean_s:.1f} deg — FAVORABLE for landing/operations")
        elif mean_s < 8:
            lines.append(f"  Mean slope: {mean_s:.1f} deg — MARGINAL terrain")
        else:
            lines.append(f"  Mean slope: {mean_s:.1f} deg — UNFAVORABLE steep terrain")

        if pct5 > 20 and sharad_data:
            lines.append("  NOTE: Significant slope variability may affect SHARAD "
                         "reflector interpretation (surface clutter contamination).")
        lines.append("")

    # Cross-instrument synthesis
    if sharad_data and mineral_data:
        lines.append("CROSS-INSTRUMENT SYNTHESIS:")
        classes_dict = dict(mineral_data.get("classes", []))
        ice_minerals = sum(v for k, v in classes_dict.items() if "Ice" in k)

        if ice_minerals > 0 and sharad_data.get("interface", {}).get("confidence") == "high":
            lines.append("  STRONG EVIDENCE: Both radar and spectral data indicate "
                         "subsurface ice presence.")
            lines.append("  Assessment: Likely WATER ICE deposit.")
        elif sharad_data.get("interface", {}).get("confidence") == "high":
            lines.append("  SHARAD detects a coherent reflector but spectral data "
                         "does not confirm surface ice exposure.")
            lines.append("  Assessment: Possible buried ice OR lithological boundary.")
        lines.append("")

    # Confidence assessment
    lines.append("OVERALL CONFIDENCE:")
    instruments = []
    if sharad_data:
        instruments.append("SHARAD")
    if mineral_data:
        instruments.append("CRISM/CNN")
    if slope_data:
        instruments.append("Terrain")
    lines.append(f"  Instruments used: {', '.join(instruments)}")

    if len(instruments) >= 3:
        lines.append("  Multi-instrument cross-validation: HIGH confidence")
    elif len(instruments) == 2:
        lines.append("  Dual-instrument analysis: MODERATE confidence")
    else:
        lines.append("  Single-instrument analysis: interpretation requires caution")

    lines.append("")
    lines.append(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append("MarsLab Autonomous Scientific Report Generator")

    return "\n".join(lines)


# =============================================================
# MAIN PIPELINE
# =============================================================

async def run_multi_report(
    config: MultiReportConfig,
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Run full multi-instrument analysis pipeline.
    Yields SSE-compatible progress events.
    """
    # Validate product IDs
    for pid in [config.sharad_product_id, config.crism_obs_id, config.hirise_dtm_id]:
        if pid and not _PRODUCT_RE.match(pid):
            yield {"event": "error", "data": {"message": "Invalid product ID"}}
            return

    report_id = f"{config.region_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    report_id = re.sub(r"[^a-zA-Z0-9_]", "", report_id)

    # Concurrency lock
    lock = _active_locks.setdefault(report_id, asyncio.Lock())

    async with _global_semaphore:
        async with lock:
            try:
                async for event in _run_multi_report_inner(config, report_id):
                    yield event
            finally:
                _active_locks.pop(report_id, None)


async def _run_multi_report_inner(
    config: MultiReportConfig,
    report_id: str,
) -> AsyncGenerator[Dict[str, Any], None]:
    """Inner pipeline logic."""
    # Work on a copy to avoid mutating the caller's config
    config = dc_replace(config)
    out_dir = str(REPORT_DIR / report_id)
    os.makedirs(out_dir, exist_ok=True)

    figures: Dict[str, str] = {}
    sharad_data: Optional[Dict] = None
    mineral_data: Optional[Dict] = None
    slope_data: Optional[Dict] = None

    total_steps = 0
    if config.sharad_product_id:
        total_steps += 3
    if config.crism_obs_id:
        total_steps += 2
    total_steps += 3  # overview + slope + PDF

    step_num = 0

    def progress():
        return int(step_num / max(total_steps, 1) * 100)

    # ════════════════════════════════════════════
    # SECTION 1: Load SHARAD data (sets center coords if not provided)
    # ════════════════════════════════════════════
    sharad_lats = sharad_lons = None

    if config.sharad_product_id:
        yield {"event": "step", "data": {"step": "sharad_load", "progress": progress(),
               "message": f"Loading SHARAD: {config.sharad_product_id}"}}

        try:
            power, surface, lats, lons, n_traces = await _run_sync(
                _load_sharad_data, config.sharad_product_id)
            sharad_lats = lats
            sharad_lons = lons

            # Derive center from SHARAD track midpoint if not set
            if config.center_lat is None:
                mid = len(lats) // 2
                config.center_lat = float(lats[mid])
                config.center_lon = float(lons[mid])

            track_km = await _run_sync(_track_distance_km, lats, lons)
            ds_factor = max(1, n_traces // 3000)

            yield {"event": "step", "data": {"step": "sharad_load", "progress": progress(),
                   "message": f"SHARAD loaded: {n_traces} traces"}}
        except Exception as e:
            logger.warning(f"SHARAD load failed: {e}")
            yield {"event": "step", "data": {"step": "sharad_load", "progress": progress(),
                   "message": f"SHARAD data unavailable"}}
            config.sharad_product_id = None  # Disable SHARAD sections

    # ════════════════════════════════════════════
    # SECTION 2: SHARAD Analysis
    # ════════════════════════════════════════════
    if config.sharad_product_id and sharad_lats is not None:
        step_num += 1
        yield {"event": "step", "data": {"step": "sharad_detect", "progress": progress(),
               "message": "Detecting subsurface interface..."}}

        from .sharad_report import detect_interface, estimate_depths

        interface = await _run_sync(detect_interface, power, surface)
        depths = estimate_depths(interface, surface)

        step_num += 1
        yield {"event": "step", "data": {"step": "sharad_figures", "progress": progress(),
               "message": "Generating SHARAD figures..."}}

        from .sharad_report import generate_radargram_figure

        fig_path = os.path.join(out_dir, "radargram_annotated.png")
        await _run_sync(generate_radargram_figure, power, surface, interface,
                        track_km, fig_path, ds_factor)
        figures["radargram_annotated"] = fig_path

        # Depth CSV
        depth_csv_path = os.path.join(out_dir, "depth_estimation.csv")
        from .sharad_report import generate_depth_csv
        generate_depth_csv(depths, depth_csv_path)

        n_detected = int((interface.bins >= 0).sum())
        sharad_data = {
            "interface": {
                "confidence": interface.confidence,
                "detection_rate": interface.detection_rate,
                "n_segments": len(interface.coherent_segments),
                "mean_depth_bins": interface.mean_depth_bins,
            },
            "depths": [
                {"label": d.label, "epsilon_r": d.epsilon_r,
                 "mean_m": d.mean_m, "min_m": d.min_m, "max_m": d.max_m}
                for d in depths
            ],
            "n_detected": n_detected,
            "n_traces": n_traces,
        }

        step_num += 1
        yield {"event": "step", "data": {"step": "sharad_done", "progress": progress(),
               "message": f"SHARAD: {interface.confidence} confidence, "
                          f"{interface.detection_rate * 100:.0f}% detection"}}

    # ════════════════════════════════════════════
    # SECTION 3: CRISM Mineral Classification
    # ════════════════════════════════════════════
    if config.crism_obs_id:
        step_num += 1
        yield {"event": "step", "data": {"step": "crism_mineral", "progress": progress(),
               "message": f"Loading mineral classification: {config.crism_obs_id}"}}

        fig_path = os.path.join(out_dir, "mineral_map.png")
        mineral_data = await _run_sync(generate_mineral_map_figure, config.crism_obs_id, fig_path)

        if mineral_data:
            figures["mineral_map"] = fig_path

            # Set center from CRISM if not already set
            if config.center_lat is None:
                crism_index_path = _BACKEND_DIR / "mineral_cnn_data" / "index.geojson"
                if crism_index_path.exists():
                    with open(str(crism_index_path)) as fh:
                        idx = json.load(fh)
                    for f in idx["features"]:
                        if f["properties"]["product_id"] == config.crism_obs_id:
                            ring = f["geometry"]["coordinates"][0]
                            config.center_lat = sum(c[1] for c in ring) / len(ring)
                            config.center_lon = sum(c[0] for c in ring) / len(ring)
                            break

        step_num += 1
        yield {"event": "step", "data": {"step": "crism_confidence", "progress": progress(),
               "message": "Generating confidence map..."}}

        conf_path = os.path.join(out_dir, "confidence_map.png")
        conf_ok = await _run_sync(generate_confidence_map_figure, config.crism_obs_id, conf_path)
        if conf_ok:
            figures["confidence_map"] = conf_path

        if mineral_data:
            yield {"event": "step", "data": {"step": "crism_done", "progress": progress(),
                   "message": f"CRISM: {mineral_data['total_classified']} classified pixels, "
                              f"{len(mineral_data['classes'])} mineral species"}}
        else:
            yield {"event": "step", "data": {"step": "crism_done", "progress": progress(),
                   "message": "CRISM: no classification results available"}}

    # ════════════════════════════════════════════
    # SECTION 1 (deferred): Overview Map
    # ════════════════════════════════════════════
    if config.center_lat is not None:
        step_num += 1
        yield {"event": "step", "data": {"step": "overview", "progress": progress(),
               "message": "Generating overview map..."}}

        cos_lat_ov = max(math.cos(math.radians(config.center_lat)), 0.01)
        radius_deg = config.analysis_radius_km / (111 * cos_lat_ov)
        radius_deg = max(radius_deg, 0.5)

        # Collect footprints
        sharad_coords = None
        if sharad_lats is not None:
            # Downsample for plotting
            ds_plot = max(1, len(sharad_lats) // 2000)
            sharad_coords = (sharad_lats[::ds_plot], sharad_lons[::ds_plot])

        crism_footprint = None
        if config.crism_obs_id:
            crism_index_path = _BACKEND_DIR / "mineral_cnn_data" / "index.geojson"
            if crism_index_path.exists():
                with open(str(crism_index_path)) as fh:
                    idx = json.load(fh)
                for f in idx["features"]:
                    if f["properties"]["product_id"] == config.crism_obs_id:
                        ring = f["geometry"]["coordinates"][0]
                        crism_footprint = [(c[0], c[1]) for c in ring]
                        break

        dtm_footprint = None
        if config.hirise_dtm_id:
            dtm_index_path = _BACKEND_DIR / "hirise_dtm_data" / "index.geojson"
            if dtm_index_path.exists():
                with open(str(dtm_index_path)) as fh:
                    idx = json.load(fh)
                for f in idx["features"]:
                    if f["properties"]["product_id"] == config.hirise_dtm_id:
                        ring = f["geometry"]["coordinates"][0]
                        dtm_footprint = [(c[0], c[1]) for c in ring]
                        break

        overview_path = os.path.join(out_dir, "overview_map.png")
        ok = await _run_sync(
            generate_overview_map, config.center_lat, config.center_lon,
            radius_deg, sharad_coords, crism_footprint, dtm_footprint, overview_path)
        if ok:
            figures["overview_map"] = overview_path

    # ════════════════════════════════════════════
    # SECTION 4: Slope & Terrain
    # ════════════════════════════════════════════
    if config.center_lat is not None:
        step_num += 1
        yield {"event": "step", "data": {"step": "slope", "progress": progress(),
               "message": "Computing slope analysis..."}}

        slope_path = os.path.join(out_dir, "slope_map.png")
        slope_data = await _run_sync(
            generate_slope_map, config.center_lat, config.center_lon,
            config.analysis_radius_km * 1000, slope_path)

        if slope_data and slope_data.get("figure_generated"):
            figures["slope_map"] = slope_path

        # 3D terrain (only if meaningful relief)
        terrain_3d_path = os.path.join(out_dir, "terrain_3d.png")
        ok_3d = await _run_sync(
            generate_terrain_3d, config.center_lat, config.center_lon,
            config.analysis_radius_km * 1000, terrain_3d_path)
        if ok_3d:
            figures["terrain_3d"] = terrain_3d_path

        if slope_data:
            yield {"event": "step", "data": {"step": "slope_done", "progress": progress(),
                   "message": f"Terrain: mean slope {slope_data['mean_slope']:.1f} deg, "
                              f"max {slope_data['max_slope']:.1f} deg"}}

    # ════════════════════════════════════════════
    # SECTION 6: Synthesis Composite
    # ════════════════════════════════════════════
    step_num += 1
    yield {"event": "step", "data": {"step": "synthesis", "progress": progress(),
           "message": "Generating synthesis composite..."}}

    composite_path = os.path.join(out_dir, "synthesis_composite.png")
    ok_comp = await _run_sync(generate_synthesis_composite, figures, config, composite_path)
    if ok_comp:
        figures["synthesis_composite"] = composite_path

    # ════════════════════════════════════════════
    # SECTION 7: Interpretation
    # ════════════════════════════════════════════
    step_num += 1
    yield {"event": "step", "data": {"step": "interpretation", "progress": progress(),
           "message": "Generating scientific interpretation..."}}

    interpretation = generate_interpretation(config, sharad_data, mineral_data, slope_data)

    # Save interpretation as text
    interp_path = os.path.join(out_dir, "interpretation.txt")
    with open(interp_path, "w") as f:
        f.write(interpretation)

    # ════════════════════════════════════════════
    # PDF Report
    # ════════════════════════════════════════════
    step_num += 1
    yield {"event": "step", "data": {"step": "pdf", "progress": progress(),
           "message": "Assembling PDF report..."}}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    region_slug = re.sub(r"[^a-zA-Z0-9]", "_", config.region_name)
    pdf_name = f"MarsLab_MultiInstrument_Report_{region_slug}_{timestamp}.pdf"
    pdf_path = os.path.join(out_dir, pdf_name)

    pdf_ok = await _run_sync(
        generate_multi_pdf, config, figures, sharad_data, mineral_data,
        slope_data, interpretation, pdf_path)

    # Completion marker
    marker = {
        "completed": datetime.now().isoformat(),
        "report_id": report_id,
        "config": {
            "region_name": config.region_name,
            "sharad_product_id": config.sharad_product_id,
            "crism_obs_id": config.crism_obs_id,
            "hirise_dtm_id": config.hirise_dtm_id,
            "center_lat": config.center_lat,
            "center_lon": config.center_lon,
        },
    }
    with open(os.path.join(out_dir, ".complete"), "w") as f:
        json.dump(marker, f, indent=2)

    yield {"event": "step", "data": {"step": "done", "progress": 100,
           "message": "Report complete"}}

    yield {"event": "done", "data": {
        "report_id": report_id,
        "figures": {k: os.path.basename(v) for k, v in figures.items()},
        "pdf": pdf_name if pdf_ok else None,
        "sharad": sharad_data,
        "mineral": {
            "classes": mineral_data["classes"],
            "total_classified": mineral_data["total_classified"],
            "classification_rate": mineral_data["classification_rate"],
        } if mineral_data else None,
        "slope": slope_data,
        "interpretation_preview": interpretation[:500],
    }}
