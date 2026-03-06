#!/usr/bin/env python3
"""
Generate publication-quality figures for the Mars Landing Site Report.
Uses actual grid search data from arcadia_refinement_results.json.

Generates:
  1. fig1_composite_heatmap.png   — Composite score heatmap over Arcadia grid
  2. fig2_swim_ice_heatmap.png    — SWIM ice consistency heatmap
  3. fig3_elevation_contour.png   — MOLA elevation contour with optimal site
  4. fig4_spacex_comparison.png   — Our site vs SpaceX 7 downselected sites
  5. fig5_swim_depth_breakdown.png — SWIM by depth layer at optimal site vs neighbors
  6. fig6_regional_overview.png   — Full 55-region analysis overview

Usage:
  cd backend && python -m analysis.integration.generate_figures
"""

import json
import os
import sys

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D

_DIR = os.path.dirname(os.path.abspath(__file__))
_FIG_DIR = os.path.join(_DIR, "figures")
os.makedirs(_FIG_DIR, exist_ok=True)

# Load grid data
with open(os.path.join(_DIR, "arcadia_refinement_results.json")) as f:
    data = json.load(f)

all_pts = data["all_viable"]
top20 = data["top_20"]
optimal = data["optimal_site"]

# Load regional results
with open(os.path.join(_DIR, "landing_site_results.json")) as f:
    regional = json.load(f)


# ═══════════════════════════════════════════════════════════════
# Color schemes
# ═══════════════════════════════════════════════════════════════

MARS_CMAP = plt.cm.get_cmap('YlOrRd')  # warm tones for Mars
ICE_CMAP = plt.cm.get_cmap('YlGnBu')   # cool tones for ice
ELEV_CMAP = plt.cm.get_cmap('terrain')  # topographic

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'savefig.facecolor': '#1a1a2e',
    'axes.facecolor': '#16213e',
    'figure.facecolor': '#1a1a2e',
    'text.color': '#e0e0e0',
    'axes.labelcolor': '#e0e0e0',
    'xtick.color': '#b0b0b0',
    'ytick.color': '#b0b0b0',
})


def _grid_arrays():
    """Convert flat point list to 2D arrays for pcolormesh."""
    lats = sorted(set(p["lat"] for p in all_pts))
    lons = sorted(set(p["lon"] for p in all_pts))
    
    lat_idx = {v: i for i, v in enumerate(lats)}
    lon_idx = {v: i for i, v in enumerate(lons)}
    
    composite = np.full((len(lats), len(lons)), np.nan)
    swim = np.full((len(lats), len(lons)), np.nan)
    elev = np.full((len(lats), len(lons)), np.nan)
    slope = np.full((len(lats), len(lons)), np.nan)
    
    for p in all_pts:
        i, j = lat_idx[p["lat"]], lon_idx[p["lon"]]
        composite[i, j] = p["composite"]
        swim[i, j] = p["swim_avg"]
        elev[i, j] = p["elevation_m"]
        slope[i, j] = p["slope_deg"]
    
    return np.array(lats), np.array(lons), composite, swim, elev, slope


def fig1_composite_heatmap():
    """Composite score heatmap over Arcadia Planitia."""
    lats, lons, composite, _, _, _ = _grid_arrays()
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Pcolormesh
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    im = ax.pcolormesh(lon_grid, lat_grid, composite, 
                       cmap='magma', vmin=0.55, vmax=0.82,
                       shading='nearest', edgecolors='none')
    
    # Optimal site marker
    ax.plot(optimal["lon"], optimal["lat"], '*', color='#00ff88', 
            markersize=22, markeredgecolor='white', markeredgewidth=1.5,
            zorder=10, label=f'Optimal: {optimal["lat"]}°N, {optimal["lon"]}°E')
    
    # Top 5 markers
    for i, p in enumerate(top20[:5]):
        if i == 0:
            continue
        ax.plot(p["lon"], p["lat"], 'D', color='#ff6b6b', 
                markersize=10, markeredgecolor='white', markeredgewidth=1,
                zorder=9)
        ax.annotate(f'#{p["rank"]}', (p["lon"], p["lat"]), 
                   xytext=(5, 5), textcoords='offset points',
                   color='white', fontsize=9, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, pad=0.02, shrink=0.85)
    cbar.set_label('Composite Score', color='#e0e0e0', fontsize=12)
    cbar.ax.yaxis.set_tick_params(color='#b0b0b0')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#b0b0b0')
    
    ax.set_xlabel('Longitude (°E)', fontsize=12)
    ax.set_ylabel('Latitude (°N)', fontsize=12)
    ax.set_title('Arcadia Planitia — Composite Landing Site Score\n'
                 '(493 points, 0.5° resolution)', fontsize=14, color='#ffffff')
    
    ax.legend(loc='lower right', fontsize=10, framealpha=0.8,
             facecolor='#2a2a4e', edgecolor='#555')
    
    # Grid
    ax.grid(True, alpha=0.15, color='#666')
    ax.set_aspect('equal')
    
    path = os.path.join(_FIG_DIR, "fig1_composite_heatmap.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {path}")


def fig2_swim_ice_heatmap():
    """SWIM ice consistency heatmap."""
    lats, lons, _, swim, _, _ = _grid_arrays()
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    im = ax.pcolormesh(lon_grid, lat_grid, swim,
                       cmap='YlGnBu', vmin=0.3, vmax=0.85,
                       shading='nearest')
    
    # Optimal site
    ax.plot(optimal["lon"], optimal["lat"], '*', color='#ff4444',
            markersize=22, markeredgecolor='white', markeredgewidth=1.5,
            zorder=10, label=f'Optimal (SWIM={optimal["swim_avg"]:.3f})')
    
    # Annotate the ice hotspot
    ax.annotate(f'SWIM = {optimal["swim_avg"]:.3f}\n1-5m: {optimal["swim_1_5m"]:.3f}',
               (optimal["lon"], optimal["lat"]),
               xytext=(20, -30), textcoords='offset points',
               color='#ff6666', fontsize=11, fontweight='bold',
               arrowprops=dict(arrowstyle='->', color='#ff6666', lw=1.5))
    
    cbar = plt.colorbar(im, ax=ax, pad=0.02, shrink=0.85)
    cbar.set_label('SWIM Ice Consistency (avg 3 depths)', color='#e0e0e0', fontsize=12)
    cbar.ax.yaxis.set_tick_params(color='#b0b0b0')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#b0b0b0')
    
    ax.set_xlabel('Longitude (°E)', fontsize=12)
    ax.set_ylabel('Latitude (°N)', fontsize=12)
    ax.set_title('SWIM Subsurface Ice Consistency — Arcadia Planitia\n'
                 '(Morgan et al. 2021/2025 integrated geophysics)', fontsize=14, color='#ffffff')
    ax.legend(loc='lower right', fontsize=10, framealpha=0.8,
             facecolor='#2a2a4e', edgecolor='#555')
    ax.grid(True, alpha=0.15, color='#666')
    ax.set_aspect('equal')
    
    path = os.path.join(_FIG_DIR, "fig2_swim_ice_heatmap.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {path}")


def fig3_elevation_contour():
    """MOLA elevation contour map."""
    lats, lons, _, _, elev, slope = _grid_arrays()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    # Left: Elevation
    im1 = ax1.pcolormesh(lon_grid, lat_grid, elev,
                         cmap='terrain', vmin=-4500, vmax=-3500,
                         shading='nearest')
    contours = ax1.contour(lon_grid, lat_grid, elev, 
                           levels=[-4200, -4100, -4000, -3900, -3800],
                           colors='white', linewidths=0.5, alpha=0.6)
    ax1.clabel(contours, inline=True, fontsize=8, fmt='%dm', colors='white')
    
    ax1.plot(optimal["lon"], optimal["lat"], '*', color='#ff4444',
             markersize=20, markeredgecolor='white', markeredgewidth=1.5, zorder=10)
    
    cbar1 = plt.colorbar(im1, ax=ax1, pad=0.02, shrink=0.85)
    cbar1.set_label('Elevation (m MOLA)', color='#e0e0e0')
    cbar1.ax.yaxis.set_tick_params(color='#b0b0b0')
    plt.setp(plt.getp(cbar1.ax.axes, 'yticklabels'), color='#b0b0b0')
    
    ax1.set_title('MOLA Elevation', color='#ffffff')
    ax1.set_xlabel('Longitude (°E)')
    ax1.set_ylabel('Latitude (°N)')
    ax1.grid(True, alpha=0.15, color='#666')
    ax1.set_aspect('equal')
    
    # Right: Slope
    im2 = ax2.pcolormesh(lon_grid, lat_grid, slope,
                         cmap='inferno', vmin=0, vmax=3.0,
                         shading='nearest')
    
    ax2.plot(optimal["lon"], optimal["lat"], '*', color='#00ff88',
             markersize=20, markeredgecolor='white', markeredgewidth=1.5, zorder=10)
    
    cbar2 = plt.colorbar(im2, ax=ax2, pad=0.02, shrink=0.85)
    cbar2.set_label('Mean Slope (°)', color='#e0e0e0')
    cbar2.ax.yaxis.set_tick_params(color='#b0b0b0')
    plt.setp(plt.getp(cbar2.ax.axes, 'yticklabels'), color='#b0b0b0')
    
    ax2.set_title('Terrain Slope (2km radius)', color='#ffffff')
    ax2.set_xlabel('Longitude (°E)')
    ax2.set_ylabel('Latitude (°N)')
    ax2.grid(True, alpha=0.15, color='#666')
    ax2.set_aspect('equal')
    
    fig.suptitle('Arcadia Planitia — Terrain Analysis (MOLA-derived)',
                fontsize=15, color='#ffffff', y=1.02)
    
    path = os.path.join(_FIG_DIR, "fig3_elevation_contour.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {path}")


def fig4_spacex_comparison():
    """Our site vs SpaceX 7 downselected sites on regional map."""
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # SpaceX 7 sites (Golombek et al. 2021, LPSC #2420)
    spacex_primary = [
        ("PM-1", 35.5, 163.6, "Phlegra Montes"),
        ("AP-1", 39.8, 192.1, "Arcadia Planitia"),
        ("AP-9", 39.1, 196.7, "Arcadia Planitia"),
        ("EM-16", 38.6, 190.2, "Erebus Montes"),
    ]
    spacex_secondary = [
        ("AP-8", 39.1, 189.8, "Arcadia Planitia"),
        ("EM-15", 39.8, 195.6, "Erebus Montes"),
        ("PM-7", 35.5, 163.6, "Phlegra Montes"),
    ]
    
    # Background: simple terrain proxy using our grid data (if available)
    # Just show the regional context with labeled regions
    
    # Draw SpaceX latitude constraint line
    ax.axhline(y=40, color='#ff6b6b', linestyle='--', linewidth=2, alpha=0.7,
               label='SpaceX lat constraint (<40°N)')
    ax.fill_between([155, 210], 40, 50, color='#ff6b6b', alpha=0.05)
    ax.text(200, 40.5, 'SpaceX: "latitude must be <40°"',
            color='#ff8888', fontsize=9, style='italic')
    
    # Our optimal site (big star)
    ax.plot(176.0, 42.0, '*', color='#00ff88', markersize=30,
            markeredgecolor='white', markeredgewidth=2, zorder=20,
            label=f'MarsLab Optimal (42.0°N, 176.0°E)')
    
    # Our grid search area (rectangle)
    from matplotlib.patches import Rectangle
    grid_rect = Rectangle((170, 38), 14, 8, linewidth=2,
                          edgecolor='#00ff88', facecolor='#00ff88',
                          alpha=0.08, linestyle='-', zorder=5)
    ax.add_patch(grid_rect)
    ax.text(170.5, 46.3, 'MarsLab 0.5° Grid\n(493 points)',
            color='#00ff88', fontsize=9, fontweight='bold')
    
    # SpaceX primary sites (red diamonds)
    for name, lat, lon, region in spacex_primary:
        ax.plot(lon, lat, 'D', color='#ff4444', markersize=14,
                markeredgecolor='white', markeredgewidth=1.5, zorder=15)
        ax.annotate(f'{name}', (lon, lat),
                   xytext=(8, 8), textcoords='offset points',
                   color='#ff6666', fontsize=10, fontweight='bold')
    
    # SpaceX secondary sites (orange circles)
    for name, lat, lon, region in spacex_secondary:
        ax.plot(lon, lat, 'o', color='#ffa500', markersize=12,
                markeredgecolor='white', markeredgewidth=1.5, zorder=15)
        ax.annotate(f'{name}', (lon, lat),
                   xytext=(8, -12), textcoords='offset points',
                   color='#ffbb44', fontsize=10, fontweight='bold')
    
    # Region labels
    ax.text(163, 33.5, 'PHLEGRA\nMONTES', color='#888', fontsize=12,
            ha='center', style='italic', alpha=0.7)
    ax.text(190, 36.5, 'EREBUS MONTES', color='#888', fontsize=12,
            ha='center', style='italic', alpha=0.7)
    ax.text(176, 43.5, 'WESTERN\nARCADIA PLANITIA', color='#aaffaa', fontsize=11,
            ha='center', style='italic', alpha=0.8, fontweight='bold')
    ax.text(195, 42, 'ARCADIA / AMAZONIS\nBOUNDARY', color='#888', fontsize=10,
            ha='center', style='italic', alpha=0.7)
    
    # Legend
    legend_elements = [
        Line2D([0], [0], marker='*', color='w', markerfacecolor='#00ff88',
               markersize=18, label=f'MarsLab Optimal (42.0°N, 176.0°E)', linestyle=''),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='#ff4444',
               markersize=10, label='SpaceX Primary (4 sites)', linestyle=''),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#ffa500',
               markersize=10, label='SpaceX Secondary (3 sites)', linestyle=''),
        Line2D([0], [0], color='#ff6b6b', linestyle='--', linewidth=2,
               label='SpaceX lat constraint (<40°N)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10,
             framealpha=0.8, facecolor='#2a2a4e', edgecolor='#555')
    
    ax.set_xlim(155, 210)
    ax.set_ylim(32, 48)
    ax.set_xlabel('Longitude (°E)', fontsize=12)
    ax.set_ylabel('Latitude (°N)', fontsize=12)
    ax.set_title('MarsLab vs SpaceX/Golombek (2021) Landing Site Candidates\n'
                 'Arcadia Planitia / Erebus Montes / Phlegra Montes Region',
                fontsize=14, color='#ffffff')
    ax.grid(True, alpha=0.15, color='#666')
    ax.set_aspect(1.2)  # Approximate Mercator correction at 40°N
    
    path = os.path.join(_FIG_DIR, "fig4_spacex_comparison.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {path}")


def fig5_swim_depth_breakdown():
    """SWIM ice consistency by depth at optimal site vs neighbors."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Left: Bar chart — optimal vs top 5 neighbors
    sites = []
    for p in top20[:6]:
        sites.append({
            "label": f'{p["lat"]}°N\n{p["lon"]}°E',
            "0-1m": p["swim_0_1m"] or 0,
            "1-5m": p["swim_1_5m"] or 0,
            "5m+": p["swim_5m_plus"] or 0,
            "rank": p["rank"],
        })
    
    x = np.arange(len(sites))
    width = 0.25
    
    colors = ['#4ecdc4', '#45b7d1', '#96ceb4']
    
    bars1 = ax1.bar(x - width, [s["0-1m"] for s in sites], width,
                    label='0-1m (shallow)', color=colors[0], edgecolor='white', linewidth=0.5)
    bars2 = ax1.bar(x, [s["1-5m"] for s in sites], width,
                    label='1-5m (medium)', color=colors[1], edgecolor='white', linewidth=0.5)
    bars3 = ax1.bar(x + width, [s["5m+"] for s in sites], width,
                    label='5m+ (deep)', color=colors[2], edgecolor='white', linewidth=0.5)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels([s["label"] for s in sites], fontsize=9)
    ax1.set_ylabel('SWIM Consistency', fontsize=12)
    ax1.set_title('SWIM Ice by Depth — Top 6 Sites', color='#ffffff')
    ax1.legend(fontsize=10, framealpha=0.8, facecolor='#2a2a4e', edgecolor='#555')
    ax1.set_ylim(0, 1.1)
    ax1.grid(True, axis='y', alpha=0.2, color='#666')
    
    # Highlight #1
    ax1.patches[0].set_edgecolor('#00ff88')
    ax1.patches[0].set_linewidth(2)
    ax1.annotate('★ #1', (0, 0.52), fontsize=11, color='#00ff88',
                fontweight='bold', ha='center')
    
    # Right: Radar/spider chart for optimal site
    categories = ['SWIM\n0-1m', 'SWIM\n1-5m', 'SWIM\n5m+', 'Landing\nScore', 
                  'Terrain\nQuality', 'Climate\nResil.']
    
    # Optimal site values (normalized 0-1)
    opt_vals = [
        optimal["swim_0_1m"] or 0,
        optimal["swim_1_5m"] or 0,
        optimal["swim_5m_plus"] or 0,
        optimal["landing_score"] / 100.0,
        1.0 - (optimal["slope_deg"] / 5.0),  # lower slope = better
        optimal["climate_resilience"],
    ]
    
    # Average of top 20 for comparison
    avg_vals = [
        np.mean([p["swim_0_1m"] or 0 for p in top20]),
        np.mean([p["swim_1_5m"] or 0 for p in top20]),
        np.mean([p["swim_5m_plus"] or 0 for p in top20]),
        np.mean([p["landing_score"] for p in top20]) / 100.0,
        np.mean([1.0 - (p["slope_deg"] / 5.0) for p in top20]),
        np.mean([p["climate_resilience"] for p in top20]),
    ]
    
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    opt_vals += opt_vals[:1]
    avg_vals += avg_vals[:1]
    angles += angles[:1]
    
    ax2 = fig.add_subplot(122, polar=True)
    ax2.set_facecolor('#16213e')
    
    ax2.plot(angles, opt_vals, 'o-', linewidth=2, color='#00ff88', label='Optimal (#1)')
    ax2.fill(angles, opt_vals, alpha=0.15, color='#00ff88')
    ax2.plot(angles, avg_vals, 's--', linewidth=1.5, color='#ff6b6b', alpha=0.7, label='Top 20 avg')
    ax2.fill(angles, avg_vals, alpha=0.08, color='#ff6b6b')
    
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(categories, fontsize=9, color='#e0e0e0')
    ax2.set_ylim(0, 1.05)
    ax2.set_title('Multi-Criteria Profile\n42.0°N, 176.0°E', 
                  color='#ffffff', fontsize=13, pad=20)
    ax2.legend(loc='lower right', fontsize=9, framealpha=0.8,
              facecolor='#2a2a4e', edgecolor='#555')
    ax2.grid(True, alpha=0.2, color='#666')
    
    path = os.path.join(_FIG_DIR, "fig5_swim_depth_breakdown.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {path}")


def fig6_regional_overview():
    """Full 55-region analysis overview — the 5 viable candidates."""
    results = regional["results"]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Left: Horizontal bar chart of final scores
    names = [r["name"] for r in results]
    scores = [r["final_score"] for r in results]
    swim_scores = [r["swim_avg"] for r in results]
    
    colors = ['#00ff88' if i == 0 else '#45b7d1' if i < 3 else '#ff6b6b' 
              for i in range(len(results))]
    
    y_pos = np.arange(len(names))
    bars = ax1.barh(y_pos, scores, color=colors, edgecolor='white', linewidth=0.5)
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names, fontsize=11)
    ax1.set_xlabel('Final Composite Score', fontsize=12)
    ax1.set_title('Phase 1: Regional Ranking\n(55 regions → 5 viable)', color='#ffffff')
    ax1.set_xlim(0, 85)
    ax1.grid(True, axis='x', alpha=0.2, color='#666')
    ax1.invert_yaxis()
    
    # Add score labels
    for i, (bar, score) in enumerate(zip(bars, scores)):
        ax1.text(score + 1, bar.get_y() + bar.get_height()/2,
                f'{score:.1f}', va='center', fontsize=11, color='white', fontweight='bold')
    
    # Right: Multi-metric comparison
    metrics = ['Landing\nScorer', 'SWIM\nIce', 'ISRU\nAccess', 'Climate\nResil.']
    
    x = np.arange(len(metrics))
    width = 0.15
    
    site_colors = ['#00ff88', '#45b7d1', '#96ceb4', '#ffa07a', '#ff6b6b']
    
    for i, r in enumerate(results):
        vals = [
            r["seasonal_avg"] / 100.0,
            r["swim_avg"],
            r["accessibility_score"],
            r["climate_resilience"],
        ]
        offset = (i - 2) * width
        ax2.bar(x + offset, vals, width, label=r["name"],
                color=site_colors[i], edgecolor='white', linewidth=0.5)
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics, fontsize=10)
    ax2.set_ylabel('Normalized Score (0–1)', fontsize=12)
    ax2.set_title('Multi-Metric Comparison\n(5 Candidate Regions)', color='#ffffff')
    ax2.set_ylim(0, 1.1)
    ax2.legend(fontsize=9, framealpha=0.8, facecolor='#2a2a4e', edgecolor='#555',
              loc='upper right')
    ax2.grid(True, axis='y', alpha=0.2, color='#666')
    
    fig.suptitle('Mars Landing Site Analysis — Phase 1 Overview',
                fontsize=15, color='#ffffff', y=1.02)
    
    path = os.path.join(_FIG_DIR, "fig6_regional_overview.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {path}")


def main():
    print("Generating figures for Landing Site Report...")
    print()
    fig1_composite_heatmap()
    fig2_swim_ice_heatmap()
    fig3_elevation_contour()
    fig4_spacex_comparison()
    fig5_swim_depth_breakdown()
    fig6_regional_overview()
    print()
    print(f"All figures saved to {_FIG_DIR}/")


if __name__ == "__main__":
    main()
