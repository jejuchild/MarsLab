#!/usr/bin/env python3
"""
Comprehensive Arcadia Planitia Scientific Report Generator.

Runs ALL MarsLab analysis algorithms on Arcadia Planitia and generates
a full report with figures:
  1. Landform detection (craters, LDAs, channels, ridges)
  2. SHARAD subsurface interface detection + radargram
  3. Regolith thickness estimation (RTE)
  4. Radar attenuation + material classification
  5. CRISM mineral classification (CNN)
  6. Stratigraphic column
  7. Terrain slope analysis + 3D visualization
  8. Mars climate model
  9. Overview map
  10. Synthesis composite + markdown report

Usage:
    cd backend && python scripts/generate_arcadia_report.py
"""

import json
import math
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

# ── Paths ─────────────────────────────────────────────────
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

REPORT_DIR = BACKEND_DIR.parent / "reports" / "arcadia_planitia_full"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ── Arcadia Planitia Parameters ───────────────────────────
CENTER_LAT = 47.5
CENTER_LON = -172.0   # -180/180 convention
SCAN_RADIUS_KM = 200  # for landform detection (large area)
DETAIL_RADIUS_KM = 50 # for terrain slope / 3D / overview
DETAIL_RADIUS_M = 50000

# SHARAD tracks to analyze (validated from previous work)
SHARAD_TRACKS = [
    "R_3933702_001_SS19_700_A",
    "R_3898101_001_SS19_700_A",
    "R_4043102_001_SS19_700_A",
    "R_3940302_001_SS19_700_A",
    "R_4018103_001_SS19_700_A",
]

# Primary SHARAD track for detailed figures
PRIMARY_SHARAD = "R_3933702_001_SS19_700_A"

# CRISM observations with CNN results
CRISM_OBS_IDS = [
    "frt0000a255_07",  # best classified (67.8%)
    "frt00009e0b_07",
    "frt00017af8_07",
    "frt00016511_07",
]
PRIMARY_CRISM = "frt0000a255_07"

# Test crater for stratigraphy
CRATER_LAT = 47.2
CRATER_LON = -166.3
CRATER_DIAMETER_KM = 20.0


# ── Utility ───────────────────────────────────────────────
def section(title: str):
    bar = "=" * 70
    print(f"\n{bar}")
    print(f"  {title}")
    print(f"{bar}\n", flush=True)


def subsection(title: str):
    print(f"\n  --- {title} ---\n", flush=True)


def save_json(data, filename):
    path = REPORT_DIR / filename
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Saved: {path.name}")
    return path


# ── Collect results for final report ─────────────────────
report_sections = {}
figures = {}
analysis_results = {}


# ══════════════════════════════════════════════════════════
# 1. LANDFORM DETECTION
# ══════════════════════════════════════════════════════════
def run_landform_detection():
    section("1. LANDFORM DETECTION (MOLA DEM)")
    try:
        from analysis.mola_detect.common import build_shared_context
        from analysis.mola_detect.crater_detect import detect_craters_and_volcanics
        from analysis.mola_detect.lda_detect import detect_ldas
        from analysis.mola_detect.ridge_channel_detect import detect_channels, detect_ridges

        print(f"  Center: {CENTER_LAT}°N, {CENTER_LON}°E")
        print(f"  Radius: {SCAN_RADIUS_KM} km")

        subsection("Building shared DEM context")
        t0 = time.time()
        ctx = build_shared_context(CENTER_LAT, CENTER_LON, SCAN_RADIUS_KM)
        print(f"  DEM extracted: {ctx.elev.shape[1]}x{ctx.elev.shape[0]} pixels in {time.time()-t0:.1f}s")

        # --- Craters & Volcanics ---
        subsection("Detecting craters & volcanics")
        t0 = time.time()
        crater_features = detect_craters_and_volcanics(
            CENTER_LAT, CENTER_LON, SCAN_RADIUS_KM,
            min_diameter_km=5.0, min_depth_m=100.0, ctx=ctx,
        )
        print(f"  Found {len(crater_features)} crater/volcanic features in {time.time()-t0:.1f}s")
        for f in crater_features[:10]:
            print(f"    {f.feature_type}: {f.diameter_km:.1f} km, depth {f.depth_m:.0f} m, "
                  f"circ {f.circularity:.3f}, conf {f.confidence:.3f} @ ({f.lat:.2f}, {f.lon:.2f})")

        # --- Channels ---
        subsection("Detecting channels/valleys")
        t0 = time.time()
        channel_features = detect_channels(
            CENTER_LAT, CENTER_LON, SCAN_RADIUS_KM,
            min_length_km=5.0, ctx=ctx,
        )
        print(f"  Found {len(channel_features)} channels in {time.time()-t0:.1f}s")
        for f in channel_features[:5]:
            print(f"    {f.morphology}: {f.length_km:.1f} km, depth {f.depth_m:.0f} m, "
                  f"sinuosity {f.sinuosity:.2f}, conf {f.confidence:.3f}")

        # --- Ridges ---
        subsection("Detecting wrinkle ridges")
        t0 = time.time()
        ridge_features = detect_ridges(
            CENTER_LAT, CENTER_LON, SCAN_RADIUS_KM,
            min_length_km=5.0, ctx=ctx,
        )
        print(f"  Found {len(ridge_features)} ridges in {time.time()-t0:.1f}s")
        for f in ridge_features[:5]:
            print(f"    {f.morphology}: {f.length_km:.1f} km, height {f.depth_m:.0f} m, "
                  f"conf {f.confidence:.3f}")

        # --- LDAs ---
        subsection("Detecting Lobate Debris Aprons")
        t0 = time.time()
        lda_features = detect_ldas(
            CENTER_LAT, CENTER_LON, SCAN_RADIUS_KM,
            min_area_km2=10.0, latitude_filter=True, ctx=ctx,
        )
        print(f"  Found {len(lda_features)} LDAs in {time.time()-t0:.1f}s")
        for f in lda_features[:5]:
            print(f"    {f.morphology}: area {f.area_km2:.1f} km², "
                  f"conf {f.confidence:.3f} @ ({f.lat:.2f}, {f.lon:.2f})")

        all_features = crater_features + channel_features + ridge_features + lda_features

        # --- Generate landform map figure ---
        subsection("Generating landform detection figure")
        _generate_landform_figure(ctx, all_features)

        analysis_results["landform"] = {
            "craters": len([f for f in all_features if "crater" in f.feature_type]),
            "volcanics": len([f for f in all_features if f.feature_type == "volcanic"]),
            "graben": len([f for f in all_features if f.feature_type == "graben"]),
            "channels": len(channel_features),
            "ridges": len(ridge_features),
            "ldas": len(lda_features),
            "total": len(all_features),
            "features": [
                {
                    "type": f.feature_type, "lat": f.lat, "lon": f.lon,
                    "diameter_km": f.diameter_km, "depth_m": f.depth_m,
                    "confidence": f.confidence, "morphology": f.morphology,
                    "description": f.description,
                }
                for f in all_features
            ],
        }
        save_json(analysis_results["landform"], "landform_results.json")
        return True

    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        return False


def _generate_landform_figure(ctx, features):
    """Generate a map of detected landforms overlaid on DEM."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(14, 10), dpi=200)

    # DEM hillshade
    elev = ctx.elev_smooth
    meta = ctx.meta
    vmin, vmax = np.nanpercentile(elev, [2, 98])

    # Compute extent in lat/lon
    nrows, ncols = elev.shape
    lat_top = meta["transform_f"] - meta["row0"] * meta["px_deg_ns"]
    lat_bot = lat_top - nrows * meta["px_deg_ns"]
    lon_left = meta["transform_c"] + meta["col0"] * meta["px_deg_ew"]
    lon_right = lon_left + ncols * meta["px_deg_ew"]
    extent = [lon_left, lon_right, lat_bot, lat_top]

    # Hillshade
    dz_dx = np.gradient(elev, axis=1)
    dz_dy = np.gradient(elev, axis=0)
    azimuth_rad = np.radians(315)
    altitude_rad = np.radians(45)
    hillshade = (
        np.cos(altitude_rad) * np.cos(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))
        + np.sin(altitude_rad) * np.sin(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))
        * np.cos(azimuth_rad - np.arctan2(-dz_dy, dz_dx))
    )
    hillshade = np.clip(hillshade, 0, 1)

    ax.imshow(elev, extent=extent, cmap="gist_earth", vmin=vmin, vmax=vmax,
              origin="upper", aspect="auto", alpha=0.7)
    ax.imshow(hillshade, extent=extent, cmap="gray", origin="upper",
              aspect="auto", alpha=0.35)

    # Plot features
    colors = {
        "crater": "#FF4444", "terraced_crater": "#FF0000", "complex": "#CC0000",
        "volcanic": "#FF8800", "graben": "#8844FF",
        "channel": "#4488FF", "wrinkle_ridge": "#44FF44", "lda": "#FFAA00",
    }
    markers = {
        "crater": "o", "terraced_crater": "s", "volcanic": "^",
        "graben": "D", "channel": "v", "wrinkle_ridge": "+", "lda": "p",
    }

    for f in features:
        color = colors.get(f.feature_type, "#FFFFFF")
        marker = markers.get(f.feature_type, "o")
        size = max(20, min(200, f.diameter_km * 3 if f.diameter_km > 0 else f.area_km2 * 0.5))

        # Plot point
        ax.scatter(f.lon, f.lat, c=color, marker=marker, s=size,
                   edgecolors="white", linewidths=0.5, zorder=5, alpha=0.85)

        # Plot paths for channels/ridges
        if f.path and len(f.path) >= 2:
            lats = [p[0] for p in f.path]
            lons = [p[1] for p in f.path]
            ax.plot(lons, lats, color=color, linewidth=1.5, alpha=0.7, zorder=4)

        # Plot boundaries for LDAs
        if f.boundary and len(f.boundary) >= 3:
            lats = [p[0] for p in f.boundary]
            lons = [p[1] for p in f.boundary]
            ax.fill(lons, lats, color=color, alpha=0.2, zorder=3)
            ax.plot(lons, lats, color=color, linewidth=1.0, alpha=0.7, zorder=4)

    # Legend
    legend_elements = []
    type_counts = {}
    for f in features:
        type_counts[f.feature_type] = type_counts.get(f.feature_type, 0) + 1
    for ftype, count in sorted(type_counts.items()):
        color = colors.get(ftype, "#FFFFFF")
        marker = markers.get(ftype, "o")
        label = f"{ftype.replace('_', ' ').title()} ({count})"
        legend_elements.append(Line2D([0], [0], marker=marker, color="w",
                               markerfacecolor=color, markersize=8, label=label))
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8,
              fancybox=True, framealpha=0.9)

    ax.set_xlabel("Longitude (°E)", fontsize=10)
    ax.set_ylabel("Latitude (°N)", fontsize=10)
    ax.set_title(f"Arcadia Planitia — Landform Detection\n"
                 f"({len(features)} features, {SCAN_RADIUS_KM} km radius from "
                 f"{CENTER_LAT}°N, {CENTER_LON}°E)", fontsize=12)
    ax.grid(True, alpha=0.3, linestyle="--")

    out = REPORT_DIR / "fig_landform_map.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    figures["landform_map"] = str(out)
    print(f"  Saved: {out.name}")


# ══════════════════════════════════════════════════════════
# 2. SHARAD SUBSURFACE INTERFACE DETECTION
# ══════════════════════════════════════════════════════════
def run_sharad_analysis():
    section("2. SHARAD SUBSURFACE INTERFACE DETECTION")
    try:
        from api.sharad_report import (
            detect_interface, estimate_depths, generate_radargram_figure,
        )
        from api.sharad_highres_router import _get_power, _get_geometry, _pick_surface, _lon_to_180

        print(f"  Primary track: {PRIMARY_SHARAD}")

        subsection("Loading SHARAD data")
        power, n_traces = _get_power(PRIMARY_SHARAD)
        geometry, _ = _get_geometry(PRIMARY_SHARAD)
        lats = geometry["lat"]
        lons = _lon_to_180(geometry["lon"])
        surface = _pick_surface(PRIMARY_SHARAD, power)
        print(f"  Power array: {power.shape} ({power.shape[0]} traces, {power.shape[1]} bins)")
        print(f"  Lat range: {np.nanmin(lats):.2f} to {np.nanmax(lats):.2f}")

        # Along-track distance
        R_MARS = 3389500.0
        dlat = np.diff(lats)
        dlon = np.diff(lons)
        seg = R_MARS * np.sqrt(np.radians(dlat)**2 + (np.radians(dlon) * np.cos(np.radians(lats[:-1])))**2)
        track_km = np.concatenate([[0], np.cumsum(seg)]) / 1000.0

        subsection("Detecting subsurface interface (Viterbi DP)")
        t0 = time.time()
        interface = detect_interface(power, surface)
        dt = time.time() - t0
        print(f"  Detection complete in {dt:.1f}s")
        print(f"  Detection rate: {interface.detection_rate:.1%}")
        print(f"  Confidence: {interface.confidence}")
        print(f"  Mean depth (bins): {interface.mean_depth_bins:.1f}")
        if hasattr(interface, "segment_statistics"):
            stats = interface.segment_statistics
            print(f"  Continuous segments: {stats.get('n_continuous', 0)}")
            print(f"  Patchy segments: {stats.get('n_patchy', 0)}")

        subsection("Estimating depths (6 dielectric models)")
        depths = estimate_depths(interface, surface)
        for d in depths:
            print(f"  {d.label} (ε={d.epsilon_r:.2f}): "
                  f"mean={d.mean_m:.1f} m, range={d.min_m:.1f}-{d.max_m:.1f} m")

        subsection("Generating radargram figure")
        out_path = str(REPORT_DIR / "fig_radargram.png")
        generate_radargram_figure(power, surface, interface, track_km, out_path)
        figures["radargram"] = out_path
        print(f"  Saved: fig_radargram.png")

        # Also generate a depth profile figure
        _generate_depth_profile_figure(track_km, lats, depths, interface)

        analysis_results["sharad"] = {
            "product_id": PRIMARY_SHARAD,
            "n_traces": int(power.shape[0]),
            "n_bins": int(power.shape[1]),
            "detection_rate": float(interface.detection_rate),
            "confidence": interface.confidence,
            "mean_depth_bins": float(interface.mean_depth_bins),
            "depth_estimates": [
                {"label": d.label, "epsilon_r": d.epsilon_r,
                 "mean_m": d.mean_m, "min_m": d.min_m, "max_m": d.max_m}
                for d in depths
            ],
        }
        save_json(analysis_results["sharad"], "sharad_results.json")
        return True

    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        return False


def _generate_depth_profile_figure(track_km, lats, depths, interface):
    """Generate depth profile along track for each dielectric model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), dpi=200,
                                    gridspec_kw={"height_ratios": [2, 1]})

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0", "#795548"]
    for i, d in enumerate(depths[:6]):
        valid = ~np.isnan(d.depth_m)
        if np.sum(valid) == 0:
            continue
        c = colors[i % len(colors)]
        ax1.scatter(track_km[valid], d.depth_m[valid], s=0.3, alpha=0.3, c=c, rasterized=True)
        # Running median for trend
        from scipy.ndimage import median_filter
        depth_filled = np.where(valid, d.depth_m, np.nan)
        # Simple binned average
        n_bins = 200
        bin_edges = np.linspace(track_km[0], track_km[-1], n_bins + 1)
        bin_means = []
        bin_centers = []
        for j in range(n_bins):
            mask = valid & (track_km >= bin_edges[j]) & (track_km < bin_edges[j+1])
            if np.sum(mask) > 5:
                bin_means.append(np.nanmedian(d.depth_m[mask]))
                bin_centers.append((bin_edges[j] + bin_edges[j+1]) / 2)
        if bin_centers:
            ax1.plot(bin_centers, bin_means, color=c, linewidth=1.5,
                     label=f"{d.label} (ε={d.epsilon_r:.2f}, μ={d.mean_m:.0f} m)")

    ax1.set_ylabel("Depth (m)", fontsize=10)
    ax1.set_title(f"SHARAD Subsurface Depth Profile — {PRIMARY_SHARAD}", fontsize=11)
    ax1.legend(fontsize=7, loc="upper right", ncol=2)
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3)

    # SNR plot
    ax2.fill_between(track_km, interface.snr, alpha=0.4, color="steelblue")
    ax2.axhline(y=3.5, color="red", linestyle="--", linewidth=0.8, label="SNR threshold (3.5)")
    ax2.set_xlabel("Along-track distance (km)", fontsize=10)
    ax2.set_ylabel("SNR", fontsize=10)
    ax2.set_ylim(0, min(50, np.nanpercentile(interface.snr[interface.snr > 0], 99) * 1.2) if np.any(interface.snr > 0) else 10)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    out = REPORT_DIR / "fig_depth_profile.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    figures["depth_profile"] = str(out)
    print(f"  Saved: fig_depth_profile.png")


# ══════════════════════════════════════════════════════════
# 3. REGOLITH THICKNESS ESTIMATION
# ══════════════════════════════════════════════════════════
def run_regolith_thickness():
    section("3. REGOLITH THICKNESS ESTIMATION (RTE)")
    try:
        from analysis.regolith_thickness.pipeline import RegolithThicknessEstimator

        all_summaries = []
        for i, track_id in enumerate(SHARAD_TRACKS):
            subsection(f"Track {i+1}/{len(SHARAD_TRACKS)}: {track_id}")
            t0 = time.time()
            estimator = RegolithThicknessEstimator()
            result = estimator.run(
                product_id=track_id,
                epsilon_r=3.0,
                snr_threshold=3.5,
                search_lo=10,
                search_hi=150,
            )
            dt = time.time() - t0

            if result.success:
                s = result.summary
                print(f"    Mean thickness: {s.thickness_mean_m:.1f} m")
                print(f"    Median thickness: {s.thickness_median_m:.1f} m")
                print(f"    Detection rate: {s.detection_rate:.1%}")
                print(f"    Time: {dt:.1f}s")
                all_summaries.append({
                    "product_id": track_id,
                    "mean_m": s.thickness_mean_m,
                    "median_m": s.thickness_median_m,
                    "std_m": s.thickness_std_m or 0.0,
                    "min_m": s.thickness_min_m or 0.0,
                    "max_m": s.thickness_max_m or 0.0,
                    "detection_rate": s.detection_rate,
                    "n_detections": s.valid_traces,
                    "n_traces": s.total_traces,
                    "mean_snr": s.mean_snr or 0.0,
                })
            else:
                print(f"    FAILED: {result.error}")

        if all_summaries:
            _generate_rte_figure(all_summaries)

        analysis_results["rte"] = {
            "epsilon_r": 3.0,
            "n_tracks": len(all_summaries),
            "tracks": all_summaries,
            "aggregate_mean_m": np.mean([s["mean_m"] for s in all_summaries]) if all_summaries else 0,
            "aggregate_median_m": np.mean([s["median_m"] for s in all_summaries]) if all_summaries else 0,
            "aggregate_detection_rate": np.mean([s["detection_rate"] for s in all_summaries]) if all_summaries else 0,
        }
        save_json(analysis_results["rte"], "rte_results.json")
        return True

    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        return False


def _generate_rte_figure(summaries):
    """Generate regolith thickness comparison across tracks."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=200)

    labels = [s["product_id"].split("_")[1] for s in summaries]
    means = [s["mean_m"] for s in summaries]
    medians = [s["median_m"] for s in summaries]
    stds = [s["std_m"] for s in summaries]
    det_rates = [s["detection_rate"] * 100 for s in summaries]

    x = np.arange(len(labels))
    width = 0.35

    ax1.bar(x - width/2, means, width, label="Mean", color="#2196F3", alpha=0.8,
            yerr=stds, capsize=3)
    ax1.bar(x + width/2, medians, width, label="Median", color="#4CAF50", alpha=0.8)
    ax1.set_xlabel("SHARAD Track (orbit)", fontsize=10)
    ax1.set_ylabel("Regolith Thickness (m)", fontsize=10)
    ax1.set_title("Regolith Thickness by Track (ε=3.0)", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    ax2.barh(x, det_rates, color="#FF9800", alpha=0.8)
    ax2.set_xlabel("Detection Rate (%)", fontsize=10)
    ax2.set_yticks(x)
    ax2.set_yticklabels(labels, fontsize=8)
    ax2.set_title("Subsurface Detection Rate", fontsize=11)
    ax2.axvline(x=50, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
    ax2.grid(True, alpha=0.3, axis="x")

    fig.suptitle("Arcadia Planitia — Regolith Thickness Estimation (5 SHARAD tracks)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    out = REPORT_DIR / "fig_regolith_thickness.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    figures["regolith_thickness"] = str(out)
    print(f"  Saved: fig_regolith_thickness.png")


# ══════════════════════════════════════════════════════════
# 4. RADAR ATTENUATION
# ══════════════════════════════════════════════════════════
def run_radar_attenuation():
    section("4. RADAR ATTENUATION & MATERIAL CLASSIFICATION")
    try:
        from analysis.radar_attenuation.pipeline import RadarAttenuationMapper

        all_summaries = []
        for i, track_id in enumerate(SHARAD_TRACKS[:3]):  # top 3 tracks
            subsection(f"Track {i+1}/3: {track_id}")
            t0 = time.time()
            mapper = RadarAttenuationMapper()
            result = mapper.run(
                product_id=track_id,
                epsilon_r=3.0,
                snr_threshold=3.5,
            )
            dt = time.time() - t0

            if result.success:
                s = result.summary
                print(f"    Mean α: {s.alpha_mean_dBm:.4f} dB/m")
                print(f"    Dominant material: {s.dominant_transparency}")
                print(f"    Detection rate: {s.detection_rate:.1%}")
                print(f"    Time: {dt:.1f}s")

                # Material distribution
                profile = mapper.generate_profile()
                mat_counts = {}
                for p in profile:
                    mat = p.get("transparency") or "unclassified"
                    mat_counts[mat] = mat_counts.get(mat, 0) + 1

                all_summaries.append({
                    "product_id": track_id,
                    "alpha_mean": s.alpha_mean_dBm,
                    "alpha_std": s.alpha_std_dBm,
                    "dominant_material": s.dominant_transparency,
                    "detection_rate": s.detection_rate,
                    "material_distribution": mat_counts,
                })
            else:
                print(f"    FAILED: {result.error}")

        if all_summaries:
            _generate_attenuation_figure(all_summaries)

        analysis_results["attenuation"] = {
            "n_tracks": len(all_summaries),
            "tracks": all_summaries,
        }
        save_json(analysis_results["attenuation"], "attenuation_results.json")
        return True

    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        return False


def _generate_attenuation_figure(summaries):
    """Generate radar attenuation material classification figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=200)

    # Attenuation values
    labels = [s["product_id"].split("_")[1] for s in summaries]
    alphas = [s["alpha_mean"] for s in summaries]
    alpha_stds = [s["alpha_std"] for s in summaries]

    ax1.barh(range(len(labels)), alphas, xerr=alpha_stds, color="#9C27B0",
             alpha=0.8, capsize=3)
    ax1.set_yticks(range(len(labels)))
    ax1.set_yticklabels(labels, fontsize=9)
    ax1.set_xlabel("Attenuation α (dB/m)", fontsize=10)
    ax1.set_title("Radar Attenuation by Track", fontsize=11)

    # Reference lines
    ax1.axvline(x=0.005, color="blue", linestyle="--", linewidth=0.8, alpha=0.6, label="Clean ice (<0.005)")
    ax1.axvline(x=0.02, color="green", linestyle="--", linewidth=0.8, alpha=0.6, label="Dusty ice (<0.02)")
    ax1.axvline(x=0.05, color="orange", linestyle="--", linewidth=0.8, alpha=0.6, label="Clay-rich (<0.05)")
    ax1.legend(fontsize=7, loc="lower right")
    ax1.grid(True, alpha=0.3, axis="x")

    # Material distribution (stacked bar)
    all_materials = set()
    for s in summaries:
        all_materials.update(k for k in s["material_distribution"].keys() if k is not None)
    all_materials = sorted(all_materials)

    mat_colors = {
        "clean_ice": "#42A5F5", "dusty_ice": "#66BB6A",
        "clay_rich": "#FFA726", "basalt": "#EF5350",
        "unknown": "#BDBDBD",
    }
    bottom = np.zeros(len(summaries))
    for mat in all_materials:
        values = []
        for s in summaries:
            total = sum(s["material_distribution"].values())
            count = s["material_distribution"].get(mat, 0)
            values.append(count / total * 100 if total > 0 else 0)
        color = mat_colors.get(mat, "#BDBDBD")
        ax2.bar(range(len(labels)), values, bottom=bottom, color=color,
                alpha=0.8, label=mat.replace("_", " ").title())
        bottom += values

    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("Material Fraction (%)", fontsize=10)
    ax2.set_title("Material Classification Distribution", fontsize=11)
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Arcadia Planitia — Radar Attenuation & Material Classification",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    out = REPORT_DIR / "fig_attenuation.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    figures["attenuation"] = str(out)
    print(f"  Saved: fig_attenuation.png")


# ══════════════════════════════════════════════════════════
# 5. CRISM MINERAL CLASSIFICATION
# ══════════════════════════════════════════════════════════
def run_crism_analysis():
    section("5. CRISM MINERAL CLASSIFICATION (CNN)")
    try:
        from api.multi_report import generate_mineral_map_figure, generate_confidence_map_figure

        mineral_results = {}
        for obs_id in CRISM_OBS_IDS:
            subsection(f"Observation: {obs_id}")

            # Try to generate mineral map figure
            out_mineral = str(REPORT_DIR / f"fig_mineral_{obs_id}.png")
            result = generate_mineral_map_figure(obs_id, out_mineral)

            if result:
                print(f"    Classes: {result['classes']}")
                print(f"    Classified pixels: {result['total_classified']}")
                print(f"    Classification rate: {result['classification_rate']:.1%}")
                mineral_results[obs_id] = result
                if obs_id == PRIMARY_CRISM:
                    figures["mineral_map"] = out_mineral
            else:
                # Try with stripped obs_id
                stripped = obs_id.replace("_07", "")
                result = generate_mineral_map_figure(stripped, out_mineral)
                if result:
                    mineral_results[obs_id] = result
                    if obs_id == PRIMARY_CRISM:
                        figures["mineral_map"] = out_mineral
                    print(f"    Classes: {result['classes']}")
                else:
                    print(f"    No CNN results available")

        # Confidence map for primary
        out_conf = str(REPORT_DIR / "fig_confidence_map.png")
        if generate_confidence_map_figure(PRIMARY_CRISM, out_conf):
            figures["confidence_map"] = out_conf
            print(f"  Saved: fig_confidence_map.png")
        else:
            stripped = PRIMARY_CRISM.replace("_07", "")
            if generate_confidence_map_figure(stripped, out_conf):
                figures["confidence_map"] = out_conf

        # Aggregate mineral statistics
        if mineral_results:
            _generate_mineral_summary_figure(mineral_results)

        analysis_results["crism"] = {
            "n_observations": len(mineral_results),
            "observations": {k: v for k, v in mineral_results.items()},
        }
        save_json(analysis_results["crism"], "crism_results.json")
        return True

    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        return False


def _generate_mineral_summary_figure(mineral_results):
    """Generate aggregate mineral distribution across all CRISM observations."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Aggregate mineral counts
    mineral_totals = {}
    for obs_id, result in mineral_results.items():
        for cls_info in result.get("classes", []):
            if isinstance(cls_info, dict):
                name = cls_info.get("name", "unknown")
                count = cls_info.get("count", 0)
            elif isinstance(cls_info, (list, tuple)):
                name, count = cls_info[0], cls_info[1]
            else:
                continue
            mineral_totals[name] = mineral_totals.get(name, 0) + count

    if not mineral_totals:
        return

    # Sort by count
    sorted_minerals = sorted(mineral_totals.items(), key=lambda x: x[1], reverse=True)
    names = [m[0] for m in sorted_minerals[:15]]
    counts = [m[1] for m in sorted_minerals[:15]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=200)

    # Bar chart
    colors_list = plt.cm.Set3(np.linspace(0, 1, len(names)))
    ax1.barh(range(len(names)), counts, color=colors_list, alpha=0.85)
    ax1.set_yticks(range(len(names)))
    ax1.set_yticklabels(names, fontsize=8)
    ax1.set_xlabel("Classified Pixels", fontsize=10)
    ax1.set_title("Mineral Distribution (All Observations)", fontsize=11)
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3, axis="x")

    # Pie chart
    if len(names) > 5:
        top_names = names[:5]
        top_counts = counts[:5]
        other = sum(counts[5:])
        if other > 0:
            top_names.append("Other")
            top_counts.append(other)
    else:
        top_names, top_counts = names, counts

    wedges, texts, autotexts = ax2.pie(
        top_counts, labels=top_names, autopct="%1.1f%%",
        colors=plt.cm.Set2(np.linspace(0, 1, len(top_names))),
        textprops={"fontsize": 8},
    )
    ax2.set_title("Top Mineral Classes", fontsize=11)

    fig.suptitle(f"Arcadia Planitia — CRISM CNN Mineral Classification "
                 f"({len(mineral_results)} observations)", fontsize=12, y=1.02)
    fig.tight_layout()
    out = REPORT_DIR / "fig_mineral_summary.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    figures["mineral_summary"] = str(out)
    print(f"  Saved: fig_mineral_summary.png")


# ══════════════════════════════════════════════════════════
# 6. STRATIGRAPHIC COLUMN
# ══════════════════════════════════════════════════════════
def run_stratigraphy():
    section("6. STRATIGRAPHIC COLUMN")
    try:
        from analysis.strat_column.pipeline import StratigraphicColumnBuilder

        subsection(f"Building column at ({CRATER_LAT}°N, {CRATER_LON}°E), D={CRATER_DIAMETER_KM} km")
        builder = StratigraphicColumnBuilder()
        result = builder.run(
            crater_lat=CRATER_LAT,
            crater_lon=CRATER_LON,
            diameter_km=CRATER_DIAMETER_KM,
            buffer_km=30.0,
            include_crism=True,
            include_sharad=True,
        )

        if result.success:
            print(f"  Layers: {result.summary.n_layers}")
            print(f"  Total depth: {result.summary.total_depth_m:.1f} m")
            print(f"  Instruments: {result.summary.instruments_used}")
            for layer in result.layers:
                eps_str = f"ε={layer.epsilon_r:.2f}" if layer.epsilon_r is not None else "ε=N/A"
                print(f"    {layer.depth_top_m:.1f}-{layer.depth_bottom_m:.1f} m: "
                      f"{layer.mineral_name or 'unknown'} ({eps_str})")

            _generate_strat_column_figure(result)
            analysis_results["stratigraphy"] = {
                "crater_lat": CRATER_LAT,
                "crater_lon": CRATER_LON,
                "n_layers": result.summary.n_layers,
                "total_depth_m": result.summary.total_depth_m,
                "instruments": result.summary.instruments_used,
                "layers": [
                    {
                        "top_m": l.depth_top_m, "bottom_m": l.depth_bottom_m,
                        "thickness_m": l.thickness_m, "mineral": l.mineral_name,
                        "epsilon_r": l.epsilon_r if l.epsilon_r is not None else 0.0,
                    }
                    for l in result.layers
                ],
            }
        else:
            print(f"  FAILED: {result.error}")
            analysis_results["stratigraphy"] = {"error": result.error}

        save_json(analysis_results.get("stratigraphy", {}), "stratigraphy_results.json")
        return result.success if hasattr(result, 'success') else False

    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        return False


def _generate_strat_column_figure(result):
    """Generate a visual stratigraphic column."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(6, 10), dpi=200)

    layer_colors = {
        "al_phyllosilicate": "#C8A2C8",
        "fe_mg_phyllosilicate": "#8B4513",
        "sulfate": "#FFD700",
        "olivine": "#556B2F",
        "pyroxene": "#2F4F4F",
        "ice": "#87CEEB",
        "basalt": "#696969",
    }

    max_depth = max(l.depth_bottom_m for l in result.layers) if result.layers else 100
    for layer in result.layers:
        mineral = (layer.mineral_name or "unknown").lower().replace(" ", "_")
        color = layer_colors.get(mineral, "#CCCCCC")
        rect = Rectangle(
            (0.1, layer.depth_top_m), 0.8, layer.thickness_m,
            facecolor=color, edgecolor="black", linewidth=0.5, alpha=0.8,
        )
        ax.add_patch(rect)
        # Label
        mid = (layer.depth_top_m + layer.depth_bottom_m) / 2
        eps_label = f"ε={layer.epsilon_r:.2f}" if layer.epsilon_r is not None else ""
        ax.text(0.5, mid, f"{layer.mineral_name or '?'}\n{eps_label}",
                ha="center", va="center", fontsize=7, fontweight="bold")
        # Depth labels
        ax.text(0.95, layer.depth_top_m, f"{layer.depth_top_m:.1f} m",
                ha="left", va="center", fontsize=7, color="gray")

    ax.set_xlim(0, 1.3)
    ax.set_ylim(max_depth * 1.05, -max_depth * 0.05)
    ax.set_ylabel("Depth (m)", fontsize=11)
    ax.set_title(f"Stratigraphic Column\n({CRATER_LAT}°N, {CRATER_LON}°E, D={CRATER_DIAMETER_KM} km)",
                 fontsize=11)
    ax.set_xticks([])
    ax.grid(True, alpha=0.2, axis="y")

    out = REPORT_DIR / "fig_strat_column.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    figures["strat_column"] = str(out)
    print(f"  Saved: fig_strat_column.png")


# ══════════════════════════════════════════════════════════
# 7. TERRAIN SLOPE ANALYSIS
# ══════════════════════════════════════════════════════════
def run_terrain_analysis():
    section("7. TERRAIN SLOPE ANALYSIS")
    try:
        from analysis.mola_detect.common import extract_dem_window, compute_slope_map

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        subsection("Extracting DEM and computing slope")
        elev, meta = extract_dem_window(CENTER_LAT, CENTER_LON, DETAIL_RADIUS_KM)
        px_m_ns = meta["px_m_ns"]
        px_m_ew = meta["px_m_ew"]

        # Fill NaN
        nan_mask = np.isnan(elev)
        fill_val = float(np.nanmean(elev)) if not np.all(nan_mask) else 0.0
        elev_filled = np.where(nan_mask, fill_val, elev)

        slope_deg = compute_slope_map(elev_filled, px_m_ns, px_m_ew)

        mean_slope = float(np.nanmean(slope_deg))
        max_slope = float(np.nanmax(slope_deg))
        std_slope = float(np.nanstd(slope_deg))
        pct_5 = float(np.sum(slope_deg > 5.0) / slope_deg.size * 100)
        pct_15 = float(np.sum(slope_deg > 15.0) / slope_deg.size * 100)

        print(f"  DEM size: {elev.shape}")
        print(f"  Mean slope: {mean_slope:.2f}°")
        print(f"  Max slope: {max_slope:.2f}°")
        print(f"  >5° fraction: {pct_5:.1f}%")
        print(f"  >15° fraction: {pct_15:.1f}%")

        # Compute extent
        nrows, ncols = elev.shape
        lat_top = meta["transform_f"] - meta["row0"] * meta["px_deg_ns"]
        lat_bot = lat_top - nrows * meta["px_deg_ns"]
        lon_left = meta["transform_c"] + meta["col0"] * meta["px_deg_ew"]
        lon_right = lon_left + ncols * meta["px_deg_ew"]
        extent = [lon_left, lon_right, lat_bot, lat_top]

        # Generate slope figure
        subsection("Generating slope map figure")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=200)

        im = ax1.imshow(slope_deg, extent=extent, cmap="YlOrRd", origin="upper",
                        aspect="auto", vmin=0, vmax=min(max_slope, 30))
        plt.colorbar(im, ax=ax1, shrink=0.8, label="Slope (°)")
        ax1.set_xlabel("Longitude (°E)")
        ax1.set_ylabel("Latitude (°N)")
        ax1.set_title(f"Slope Map — {DETAIL_RADIUS_KM} km radius")
        ax1.grid(True, alpha=0.3, linestyle="--")

        ax2.hist(slope_deg.flatten(), bins=np.arange(0, min(max_slope + 1, 30), 0.5),
                 color="steelblue", edgecolor="white", alpha=0.8, density=True)
        ax2.axvline(x=5, color="orange", linestyle="--", linewidth=1.5, label="5° threshold")
        ax2.axvline(x=15, color="red", linestyle="--", linewidth=1.5, label="15° threshold")
        ax2.set_xlabel("Slope (°)")
        ax2.set_ylabel("Density")
        ax2.set_title(f"Slope Distribution (mean={mean_slope:.2f}°, max={max_slope:.1f}°)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        fig.suptitle(f"Arcadia Planitia — Terrain Slope Analysis\n"
                     f"({CENTER_LAT}°N, {CENTER_LON}°E, {DETAIL_RADIUS_KM} km radius)",
                     fontsize=12, y=1.02)
        fig.tight_layout()
        out_slope = REPORT_DIR / "fig_slope_map.png"
        fig.savefig(out_slope, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        figures["slope_map"] = str(out_slope)
        print(f"  Saved: fig_slope_map.png")

        # Generate 3D terrain
        subsection("Generating 3D terrain visualization")
        relief = float(np.nanmax(elev_filled) - np.nanmin(elev_filled))
        print(f"  Relief: {relief:.0f} m")
        if relief > 20:
            from scipy.ndimage import zoom
            # Downsample for 3D rendering
            target = 200
            factor = min(1.0, target / max(elev_filled.shape))
            if factor < 1.0:
                elev_small = zoom(elev_filled, factor, order=1)
            else:
                elev_small = elev_filled

            fig = plt.figure(figsize=(12, 8), dpi=200)
            ax = fig.add_subplot(111, projection="3d")
            nr, nc = elev_small.shape
            X = np.linspace(extent[0], extent[1], nc)
            Y = np.linspace(extent[3], extent[2], nr)
            X, Y = np.meshgrid(X, Y)

            vmin, vmax = np.nanpercentile(elev_small, [2, 98])
            norm_elev = (elev_small - vmin) / max(vmax - vmin, 1)
            norm_elev = np.clip(norm_elev, 0, 1)
            from matplotlib import cm
            colors = cm.gist_earth(norm_elev)

            ax.plot_surface(X, Y, elev_small, facecolors=colors,
                           rstride=1, cstride=1, antialiased=False, shade=True)
            ax.view_init(elev=35, azim=225)
            ax.set_xlabel("Longitude (°E)")
            ax.set_ylabel("Latitude (°N)")
            ax.set_zlabel("Elevation (m)")
            ax.set_title(f"3D Terrain — Arcadia Planitia\nRelief: {relief:.0f} m")

            out_3d = REPORT_DIR / "fig_terrain_3d.png"
            fig.savefig(out_3d, dpi=200, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            figures["terrain_3d"] = str(out_3d)
            print(f"  Saved: fig_terrain_3d.png")
        else:
            print("  Relief too low for 3D visualization")

        analysis_results["terrain"] = {
            "mean_slope": mean_slope, "max_slope": max_slope, "std_slope": std_slope,
            "pct_above_5deg": pct_5, "pct_above_15deg": pct_15,
            "relief_m": relief,
        }
        save_json(analysis_results["terrain"], "terrain_results.json")
        return True

    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        return False


# ══════════════════════════════════════════════════════════
# 8. MARS CLIMATE MODEL
# ══════════════════════════════════════════════════════════
def run_climate_analysis():
    section("8. MARS CLIMATE MODEL")
    try:
        from api.mars_climate import analyze_climate

        subsection(f"Climate analysis at ({CENTER_LAT}°N, {CENTER_LON}°E)")
        result = analyze_climate(CENTER_LAT, CENTER_LON)

        print(f"  Temperature: mean={result.temperature['mean_k']:.1f} K, "
              f"range={result.temperature['min_k']:.1f}-{result.temperature['max_k']:.1f} K")
        print(f"  Pressure: {result.pressure_pa:.0f} Pa")
        print(f"  Dust: τ_mean={result.dust['tau_mean']:.2f}, storm risk={result.dust['storm_risk']}")
        print(f"  Wind: mean={result.wind['mean_ms']:.1f} m/s, gust={result.wind['gust_ms']:.1f} m/s")
        print(f"  Frost: prob={result.frost['frost_probability']:.1%}, seasonal={result.frost['seasonal_frost']}")
        print(f"  Climate score: {result.climate_score}/10")

        _generate_climate_figure(result)

        analysis_results["climate"] = {
            "temperature": result.temperature,
            "pressure_pa": result.pressure_pa,
            "dust": result.dust,
            "wind": result.wind,
            "frost": result.frost,
            "climate_score": result.climate_score,
            "seasonal_profile": result.seasonal_profile if hasattr(result, "seasonal_profile") else [],
        }
        save_json(analysis_results["climate"], "climate_results.json")
        return True

    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        return False


def _generate_climate_figure(result):
    """Generate climate seasonal profile figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seasonal = result.seasonal_profile if hasattr(result, "seasonal_profile") else []
    if not seasonal:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=200)

    ls_vals = [s.get("ls_deg", i * 30) for i, s in enumerate(seasonal)]

    # Temperature
    ax = axes[0, 0]
    mean_temps = [s.get("temperature", {}).get("mean_k", 200) for s in seasonal]
    max_temps = [s.get("temperature", {}).get("max_k", 220) for s in seasonal]
    min_temps = [s.get("temperature", {}).get("min_k", 180) for s in seasonal]
    ax.fill_between(ls_vals, min_temps, max_temps, alpha=0.3, color="#F44336")
    ax.plot(ls_vals, mean_temps, "o-", color="#F44336", linewidth=2, markersize=4)
    ax.set_ylabel("Temperature (K)", fontsize=10)
    ax.set_title("Surface Temperature", fontsize=11)
    ax.axhline(y=273.15, color="blue", linestyle=":", alpha=0.5, label="273 K (H₂O melting)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Dust opacity
    ax = axes[0, 1]
    tau_mean = [s.get("dust", {}).get("tau_mean", 0.3) for s in seasonal]
    tau_peak = [s.get("dust", {}).get("tau_peak", 0.5) for s in seasonal]
    ax.fill_between(ls_vals, tau_mean, tau_peak, alpha=0.3, color="#FF9800")
    ax.plot(ls_vals, tau_mean, "o-", color="#FF9800", linewidth=2, markersize=4, label="Mean τ")
    ax.plot(ls_vals, tau_peak, "s--", color="#E65100", linewidth=1, markersize=3, label="Peak τ")
    ax.set_ylabel("Dust Optical Depth", fontsize=10)
    ax.set_title("Atmospheric Dust", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Wind
    ax = axes[1, 0]
    wind_mean = [s.get("wind", {}).get("mean_ms", 5) for s in seasonal]
    wind_gust = [s.get("wind", {}).get("gust_ms", 15) for s in seasonal]
    ax.fill_between(ls_vals, wind_mean, wind_gust, alpha=0.3, color="#4CAF50")
    ax.plot(ls_vals, wind_mean, "o-", color="#4CAF50", linewidth=2, markersize=4, label="Mean")
    ax.plot(ls_vals, wind_gust, "s--", color="#1B5E20", linewidth=1, markersize=3, label="Gust")
    ax.set_xlabel("Solar Longitude Ls (°)", fontsize=10)
    ax.set_ylabel("Wind Speed (m/s)", fontsize=10)
    ax.set_title("Surface Winds", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Frost probability
    ax = axes[1, 1]
    frost_prob = [s.get("frost", {}).get("frost_probability", 0) for s in seasonal]
    ax.bar(ls_vals, [p * 100 for p in frost_prob], width=25, color="#2196F3", alpha=0.8)
    ax.set_xlabel("Solar Longitude Ls (°)", fontsize=10)
    ax.set_ylabel("CO₂ Frost Probability (%)", fontsize=10)
    ax.set_title("Seasonal Frost", fontsize=11)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Arcadia Planitia Climate Profile — {CENTER_LAT}°N, {CENTER_LON}°E\n"
                 f"Climate Score: {result.climate_score}/10",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    out = REPORT_DIR / "fig_climate.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    figures["climate"] = str(out)
    print(f"  Saved: fig_climate.png")


# ══════════════════════════════════════════════════════════
# 9. OVERVIEW MAP
# ══════════════════════════════════════════════════════════
def run_overview_map():
    section("9. OVERVIEW MAP (MOLA DEM)")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from analysis.mola_detect.common import extract_dem_window

        # Load SHARAD track coordinates for overlay
        sharad_lats, sharad_lons = None, None
        try:
            from api.sharad_highres_router import _get_geometry, _lon_to_180
            geometry, _ = _get_geometry(PRIMARY_SHARAD)
            sharad_lats = geometry["lat"]
            sharad_lons = _lon_to_180(geometry["lon"])
        except Exception:
            pass

        subsection("Extracting MOLA DEM for overview")
        elev, meta = extract_dem_window(CENTER_LAT, CENTER_LON, DETAIL_RADIUS_KM)

        nrows, ncols = elev.shape
        lat_top = meta["transform_f"] - meta["row0"] * meta["px_deg_ns"]
        lat_bot = lat_top - nrows * meta["px_deg_ns"]
        lon_left = meta["transform_c"] + meta["col0"] * meta["px_deg_ew"]
        lon_right = lon_left + ncols * meta["px_deg_ew"]
        extent = [lon_left, lon_right, lat_bot, lat_top]

        # NaN fill
        nan_mask = np.isnan(elev)
        fill_val = float(np.nanmean(elev)) if not np.all(nan_mask) else 0.0
        elev_filled = np.where(nan_mask, fill_val, elev)
        vmin, vmax = np.nanpercentile(elev_filled, [2, 98])

        # Hillshade
        dz_dx = np.gradient(elev_filled, axis=1)
        dz_dy = np.gradient(elev_filled, axis=0)
        azimuth_rad = np.radians(315)
        altitude_rad = np.radians(45)
        slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
        aspect_rad = np.arctan2(-dz_dy, dz_dx)
        hillshade = (
            np.cos(altitude_rad) * np.cos(slope_rad)
            + np.sin(altitude_rad) * np.sin(slope_rad) * np.cos(azimuth_rad - aspect_rad)
        )
        hillshade = np.clip(hillshade, 0, 1)

        fig, ax = plt.subplots(figsize=(12, 10), dpi=200)
        ax.imshow(elev_filled, extent=extent, cmap="gist_earth", vmin=vmin, vmax=vmax,
                  origin="upper", aspect="auto", alpha=0.7)
        ax.imshow(hillshade, extent=extent, cmap="gray", origin="upper",
                  aspect="auto", alpha=0.35)

        # Overlay SHARAD track
        if sharad_lats is not None:
            # Filter to visible extent
            mask = ((sharad_lats >= extent[2]) & (sharad_lats <= extent[3]) &
                    (sharad_lons >= extent[0]) & (sharad_lons <= extent[1]))
            if np.any(mask):
                ax.plot(sharad_lons[mask], sharad_lats[mask], color="red",
                        linewidth=2, alpha=0.8, label=f"SHARAD {PRIMARY_SHARAD[:15]}...")

        # Center marker
        ax.scatter(CENTER_LON, CENTER_LAT, marker="+", s=200, color="yellow",
                   linewidths=2, zorder=10, label="Region center")

        ax.set_xlabel("Longitude (°E)", fontsize=10)
        ax.set_ylabel("Latitude (°N)", fontsize=10)
        ax.set_title(f"Arcadia Planitia — MOLA DEM Overview\n"
                     f"({CENTER_LAT}°N, {CENTER_LON}°E, {DETAIL_RADIUS_KM} km radius)",
                     fontsize=12)
        ax.legend(loc="upper left", fontsize=8, framealpha=0.8)
        ax.grid(True, alpha=0.3, linestyle="--")

        out = REPORT_DIR / "fig_overview_map.png"
        fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        figures["overview_map"] = str(out)
        print(f"  Saved: fig_overview_map.png")
        return True

    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        return False


# ══════════════════════════════════════════════════════════
# 10. SYNTHESIS COMPOSITE + MARKDOWN REPORT
# ══════════════════════════════════════════════════════════
def generate_synthesis():
    section("10. SYNTHESIS COMPOSITE FIGURE")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image

        # Select best figures for composite
        panel_keys = ["overview_map", "radargram", "landform_map", "mineral_summary",
                      "regolith_thickness", "climate"]
        available = [(k, figures[k]) for k in panel_keys if k in figures]

        if len(available) < 2:
            print("  Not enough figures for composite")
            return False

        n = len(available)
        ncols = min(3, n)
        nrows = math.ceil(n / ncols)

        fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows), dpi=200)
        if nrows == 1 and ncols == 1:
            axes = np.array([[axes]])
        elif nrows == 1:
            axes = axes[np.newaxis, :]
        elif ncols == 1:
            axes = axes[:, np.newaxis]

        for idx, (key, path) in enumerate(available):
            r, c = divmod(idx, ncols)
            ax = axes[r, c]
            try:
                img = Image.open(path)
                ax.imshow(np.array(img))
            except Exception:
                ax.text(0.5, 0.5, f"[{key}]", ha="center", va="center", fontsize=12)
            ax.axis("off")
            label = key.replace("_", " ").title()
            ax.set_title(f"({chr(97+idx)}) {label}", fontsize=10, fontweight="bold")

        # Hide empty axes
        for idx in range(len(available), nrows * ncols):
            r, c = divmod(idx, ncols)
            axes[r, c].axis("off")

        fig.suptitle("Arcadia Planitia — Comprehensive Multi-Instrument Analysis",
                     fontsize=14, fontweight="bold", y=1.01)
        fig.tight_layout()
        out = REPORT_DIR / "fig_synthesis_composite.png"
        fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        figures["synthesis"] = str(out)
        print(f"  Saved: fig_synthesis_composite.png")
        return True

    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        return False


def write_markdown_report():
    section("WRITING FINAL MARKDOWN REPORT")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

    md = []
    md.append("# Arcadia Planitia — Comprehensive Multi-Instrument Scientific Report")
    md.append(f"\n**Generated:** {timestamp}")
    md.append(f"**Center:** {CENTER_LAT}°N, {CENTER_LON}°E")
    md.append(f"**Scan radius:** {SCAN_RADIUS_KM} km (landform), {DETAIL_RADIUS_KM} km (detail)")
    md.append(f"**Instruments:** MOLA DEM, SHARAD, CRISM, Mars Climate Model")
    md.append("")

    # Synthesis figure
    if "synthesis" in figures:
        md.append("![Synthesis Composite](fig_synthesis_composite.png)")
        md.append("")

    # ── Section 1: Landform Detection ──
    md.append("## 1. Landform Detection (MOLA DEM)")
    md.append("")
    if "landform" in analysis_results:
        r = analysis_results["landform"]
        md.append(f"**Total features detected:** {r['total']}")
        md.append("")
        md.append("| Feature Type | Count |")
        md.append("|---|---|")
        for key in ["craters", "volcanics", "graben", "channels", "ridges", "ldas"]:
            md.append(f"| {key.replace('_', ' ').title()} | {r.get(key, 0)} |")
        md.append("")
        if "landform_map" in figures:
            md.append("![Landform Detection Map](fig_landform_map.png)")
            md.append("")
        # Top features table
        feats = r.get("features", [])
        if feats:
            md.append("### Notable Features")
            md.append("")
            md.append("| Type | Lat | Lon | Size | Depth/Height | Confidence | Description |")
            md.append("|---|---|---|---|---|---|---|")
            for f in feats[:20]:
                size = f"{f['diameter_km']:.1f} km" if f['diameter_km'] > 0 else "-"
                md.append(f"| {f['type']} | {f['lat']:.2f} | {f['lon']:.2f} | "
                         f"{size} | {f['depth_m']:.0f} m | {f['confidence']:.3f} | "
                         f"{f['description'][:60]} |")
            md.append("")

    # ── Section 2: SHARAD Subsurface ──
    md.append("## 2. SHARAD Subsurface Interface Detection")
    md.append("")
    if "sharad" in analysis_results:
        r = analysis_results["sharad"]
        md.append(f"**Track:** {r['product_id']}")
        md.append(f"**Traces:** {r['n_traces']:,}")
        md.append(f"**Detection rate:** {r['detection_rate']:.1%}")
        md.append(f"**Confidence:** {r['confidence']}")
        md.append("")
        if "radargram" in figures:
            md.append("![Annotated Radargram](fig_radargram.png)")
            md.append("")
        md.append("### Depth Estimates by Dielectric Model")
        md.append("")
        md.append("| Material | ε_r | Mean Depth (m) | Min (m) | Max (m) |")
        md.append("|---|---|---|---|---|")
        for d in r.get("depth_estimates", []):
            md.append(f"| {d['label']} | {d['epsilon_r']:.2f} | {d['mean_m']:.1f} | "
                     f"{d['min_m']:.1f} | {d['max_m']:.1f} |")
        md.append("")
        if "depth_profile" in figures:
            md.append("![Depth Profile](fig_depth_profile.png)")
            md.append("")

    # ── Section 3: Regolith Thickness ──
    md.append("## 3. Regolith Thickness Estimation (RTE)")
    md.append("")
    if "rte" in analysis_results:
        r = analysis_results["rte"]
        md.append(f"**Tracks analyzed:** {r['n_tracks']}")
        md.append(f"**Assumed ε_r:** {r['epsilon_r']}")
        md.append(f"**Aggregate mean thickness:** {r['aggregate_mean_m']:.1f} m")
        md.append(f"**Aggregate median thickness:** {r['aggregate_median_m']:.1f} m")
        md.append(f"**Aggregate detection rate:** {r['aggregate_detection_rate']:.1%}")
        md.append("")
        md.append("| Track | Mean (m) | Median (m) | Std (m) | Det. Rate | SNR |")
        md.append("|---|---|---|---|---|---|")
        for t in r.get("tracks", []):
            md.append(f"| {t['product_id'][:20]} | {t['mean_m']:.1f} | {t['median_m']:.1f} | "
                     f"{t['std_m']:.1f} | {t['detection_rate']:.1%} | {t['mean_snr']:.1f} |")
        md.append("")
        if "regolith_thickness" in figures:
            md.append("![Regolith Thickness](fig_regolith_thickness.png)")
            md.append("")

    # ── Section 4: Radar Attenuation ──
    md.append("## 4. Radar Attenuation & Material Classification")
    md.append("")
    if "attenuation" in analysis_results:
        r = analysis_results["attenuation"]
        md.append(f"**Tracks analyzed:** {r['n_tracks']}")
        md.append("")
        md.append("| Track | α (dB/m) | Dominant Material | Det. Rate |")
        md.append("|---|---|---|---|")
        for t in r.get("tracks", []):
            md.append(f"| {t['product_id'][:20]} | {t['alpha_mean']:.4f} | "
                     f"{t['dominant_material']} | {t['detection_rate']:.1%} |")
        md.append("")
        if "attenuation" in figures:
            md.append("![Radar Attenuation](fig_attenuation.png)")
            md.append("")

    # ── Section 5: CRISM Minerals ──
    md.append("## 5. CRISM Mineral Classification (CNN)")
    md.append("")
    if "crism" in analysis_results:
        r = analysis_results["crism"]
        md.append(f"**Observations with CNN results:** {r['n_observations']}")
        md.append("")
        if "mineral_map" in figures:
            md.append(f"![Mineral Map — {PRIMARY_CRISM}](fig_mineral_{PRIMARY_CRISM}.png)")
            md.append("")
        if "mineral_summary" in figures:
            md.append("![Mineral Distribution Summary](fig_mineral_summary.png)")
            md.append("")
        if "confidence_map" in figures:
            md.append("![CNN Confidence Map](fig_confidence_map.png)")
            md.append("")

    # ── Section 6: Stratigraphy ──
    md.append("## 6. Integrated Stratigraphic Column")
    md.append("")
    if "stratigraphy" in analysis_results:
        r = analysis_results["stratigraphy"]
        if "error" not in r:
            md.append(f"**Location:** {CRATER_LAT}°N, {CRATER_LON}°E (D={CRATER_DIAMETER_KM} km)")
            md.append(f"**Layers:** {r.get('n_layers', 0)}")
            md.append(f"**Total depth:** {r.get('total_depth_m', 0):.1f} m")
            md.append(f"**Instruments:** {', '.join(r.get('instruments', []))}")
            md.append("")
            md.append("| Depth Top (m) | Depth Bottom (m) | Thickness (m) | Mineral | ε_r |")
            md.append("|---|---|---|---|---|")
            for l in r.get("layers", []):
                eps_val = l.get('epsilon_r')
                eps_str = f"{eps_val:.2f}" if eps_val else "-"
                md.append(f"| {l['top_m']:.1f} | {l['bottom_m']:.1f} | {l['thickness_m']:.1f} | "
                         f"{l.get('mineral', '-')} | {eps_str} |")
            md.append("")
            if "strat_column" in figures:
                md.append("![Stratigraphic Column](fig_strat_column.png)")
                md.append("")
        else:
            md.append(f"*Analysis failed: {r['error']}*")
            md.append("")

    # ── Section 7: Terrain ──
    md.append("## 7. Terrain Slope Analysis")
    md.append("")
    if "terrain" in analysis_results:
        r = analysis_results["terrain"]
        md.append(f"**Mean slope:** {r.get('mean_slope', 0):.2f}°")
        md.append(f"**Max slope:** {r.get('max_slope', 0):.2f}°")
        md.append(f"**Std slope:** {r.get('std_slope', 0):.2f}°")
        md.append(f"**>5° fraction:** {r.get('pct_above_5deg', 0):.1f}%")
        md.append(f"**>15° fraction:** {r.get('pct_above_15deg', 0):.1f}%")
        md.append("")
    if "slope_map" in figures:
        md.append("![Slope Map](fig_slope_map.png)")
        md.append("")
    if "terrain_3d" in figures:
        md.append("![3D Terrain](fig_terrain_3d.png)")
        md.append("")

    # ── Section 8: Climate ──
    md.append("## 8. Mars Climate Model")
    md.append("")
    if "climate" in analysis_results:
        r = analysis_results["climate"]
        md.append(f"**Climate score:** {r.get('climate_score', '-')}/10")
        md.append(f"**Mean temperature:** {r['temperature']['mean_k']:.1f} K "
                 f"({r['temperature']['mean_k'] - 273.15:.1f} °C)")
        md.append(f"**Temperature range:** {r['temperature']['min_k']:.1f} — "
                 f"{r['temperature']['max_k']:.1f} K")
        md.append(f"**Surface pressure:** {r['pressure_pa']:.0f} Pa")
        md.append(f"**Dust opacity:** τ_mean = {r['dust']['tau_mean']:.2f}, "
                 f"storm risk = {r['dust']['storm_risk']}")
        md.append(f"**Wind:** mean = {r['wind']['mean_ms']:.1f} m/s, "
                 f"gust = {r['wind']['gust_ms']:.1f} m/s")
        md.append(f"**CO₂ frost:** probability = {r['frost']['frost_probability']:.1%}, "
                 f"seasonal = {r['frost']['seasonal_frost']}")
        md.append("")
    if "climate" in figures:
        md.append("![Climate Profile](fig_climate.png)")
        md.append("")

    # ── Conclusions ──
    md.append("## 9. Synthesis & Conclusions")
    md.append("")
    md.append("### Key Findings")
    md.append("")

    findings = []
    if "landform" in analysis_results:
        r = analysis_results["landform"]
        findings.append(f"- **Landforms:** {r['total']} features detected across {SCAN_RADIUS_KM} km "
                       f"radius, including {r['craters']} craters, {r['channels']} channels, "
                       f"{r['ridges']} ridges, and {r['ldas']} LDAs.")
    if "sharad" in analysis_results:
        r = analysis_results["sharad"]
        if r.get("depth_estimates"):
            # Pick the pure water ice model (epsilon ~3.1) or third entry
            ice_model = next((d for d in r["depth_estimates"] if d["epsilon_r"] > 2.5 and d["epsilon_r"] < 3.5), r["depth_estimates"][0])
            findings.append(f"- **Subsurface:** SHARAD interface detected with {r['detection_rate']:.1%} "
                           f"rate ({r['confidence']} confidence). {ice_model['label']} model (ε={ice_model['epsilon_r']:.2f}) gives "
                           f"mean depth ~{ice_model['mean_m']:.0f} m (range {ice_model['min_m']:.0f}-{ice_model['max_m']:.0f} m).")
    if "rte" in analysis_results:
        r = analysis_results["rte"]
        findings.append(f"- **Regolith:** Mean thickness {r['aggregate_mean_m']:.1f} m across "
                       f"{r['n_tracks']} tracks (ε=3.0), detection rate {r['aggregate_detection_rate']:.1%}.")
    if "attenuation" in analysis_results:
        r = analysis_results["attenuation"]
        if r.get("tracks"):
            dominant = r["tracks"][0].get("dominant_material", "unknown")
            findings.append(f"- **Material:** Dominant subsurface material classified as "
                          f"**{dominant.replace('_', ' ')}** from radar attenuation analysis.")
    if "climate" in analysis_results:
        r = analysis_results["climate"]
        findings.append(f"- **Climate:** Score {r['climate_score']}/10. Mean temperature "
                       f"{r['temperature']['mean_k']:.0f} K, CO₂ frost probability "
                       f"{r['frost']['frost_probability']:.0%}.")

    md.extend(findings)
    md.append("")

    md.append("### Figures Index")
    md.append("")
    md.append("| # | Figure | Description |")
    md.append("|---|---|---|")
    for i, (key, path) in enumerate(figures.items(), 1):
        fname = Path(path).name
        md.append(f"| {i} | `{fname}` | {key.replace('_', ' ').title()} |")
    md.append("")

    md.append("---")
    md.append(f"*Report generated by MarsLab v2.0 — {timestamp}*")

    # Write
    report_path = REPORT_DIR / "arcadia_planitia_comprehensive_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(md))
    print(f"  Report written: {report_path}")
    return report_path


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════
def main():
    t_start = time.time()
    section("ARCADIA PLANITIA — COMPREHENSIVE ANALYSIS")
    print(f"  Output directory: {REPORT_DIR}")
    print(f"  Center: {CENTER_LAT}°N, {CENTER_LON}°E")
    print(f"  Scan radius: {SCAN_RADIUS_KM} km")
    print(f"  SHARAD tracks: {len(SHARAD_TRACKS)}")
    print(f"  CRISM observations: {len(CRISM_OBS_IDS)}")

    results = {}

    # Run all analyses
    results["landform"] = run_landform_detection()
    results["sharad"] = run_sharad_analysis()
    results["rte"] = run_regolith_thickness()
    results["attenuation"] = run_radar_attenuation()
    results["crism"] = run_crism_analysis()
    results["stratigraphy"] = run_stratigraphy()
    results["terrain"] = run_terrain_analysis()
    results["climate"] = run_climate_analysis()
    results["overview"] = run_overview_map()

    # Synthesis
    generate_synthesis()
    report_path = write_markdown_report()

    # Summary
    elapsed = time.time() - t_start
    section("COMPLETE")
    print(f"  Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Figures generated: {len(figures)}")
    print(f"  Report: {report_path}")
    print()
    print("  Analysis Status:")
    for name, success in results.items():
        status = "OK" if success else "FAILED"
        print(f"    {name:20s} [{status}]")
    print()
    print(f"  All outputs in: {REPORT_DIR}")


if __name__ == "__main__":
    main()
