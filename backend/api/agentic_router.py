"""
Agentic AI Router — SSE streaming endpoint + report download.

POST /api/agent/run  — Start agent in background and stream SSE events
GET  /api/agent/resume/{session_id} — Reconnect to a running/completed session
GET  /api/agent/sessions — List all past sessions (summary)
GET  /api/agent/session/{session_id} — Get session state (polling fallback)
GET  /api/agent/report/{session_id} — Download report (md or pdf)
GET  /api/agent/status — Check Ollama/Llama availability
"""

import base64
import io
import json
import logging
import os
from datetime import datetime, timezone
from typing import Dict, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel

from .agent_orchestrator import (
    cancel_session,
    get_session,
    list_sessions,
    start_agent_background,
    stream_session_events,
    _check_ollama,
    AgentSession,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/agent", tags=["agentic-ai"])


# =============================================================================
# Request Models
# =============================================================================

class AgentRunRequest(BaseModel):
    """Request to start an agent session."""
    objective: str
    auto_download: bool = True


# =============================================================================
# Report Builder
# =============================================================================

def _fmt_lat(v: Optional[float]) -> str:
    if v is None:
        return "N/A"
    return f"{abs(v):.4f} {'N' if v >= 0 else 'S'}"

def _fmt_lon(v: Optional[float]) -> str:
    if v is None:
        return "N/A"
    return f"{abs(v):.4f} {'E' if v >= 0 else 'W'}"


def _generate_figures(session: AgentSession) -> Dict[str, str]:
    """Generate base64-encoded PNG figures for the report.

    Returns dict of figure_name -> base64 string.
    Gracefully handles missing dependencies or data.
    """
    figures: Dict[str, str] = {}
    synthesis = session.synthesis or {}

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.info("matplotlib not available, skipping figure generation")
        return figures

    # ── SHARAD radargram ──
    sub_result = session.all_results.get("subsurface")
    if sub_result and sub_result.success:
        tracks = sub_result.data.get("tracks", [])
        # Pick best track: prefer one with subsurface detection
        best_track = None
        for t in tracks:
            if t.get("subsurface_detected") and t.get("analyzed"):
                best_track = t
                break
        if not best_track:
            for t in tracks:
                if t.get("analyzed"):
                    best_track = t
                    break
        if best_track:
            try:
                from .sharad_highres_router import _get_power, _pick_surface
                pid = best_track["product_id"]
                power, n_traces = _get_power(pid)
                surface = _pick_surface(pid, power)

                fig, ax = plt.subplots(figsize=(12, 4))
                power_db = 10 * np.log10(power.T.astype(np.float64) + 1e-12)
                vmin, vmax = np.percentile(power_db, 5), np.percentile(power_db, 95)
                ax.imshow(power_db, aspect="auto", cmap="gray", vmin=vmin, vmax=vmax)

                valid = surface >= 0
                ax.plot(np.where(valid)[0], surface[valid], "r-", linewidth=0.5, alpha=0.8, label="Surface return")

                if valid.any():
                    mean_surf = int(np.mean(surface[valid]))
                    ax.axhline(y=mean_surf + 20, color="cyan", linewidth=0.3, alpha=0.5, linestyle="--")
                    ax.axhline(
                        y=min(mean_surf + 200, power.shape[1] - 1),
                        color="cyan", linewidth=0.3, alpha=0.5, linestyle="--",
                        label="Subsurface search zone",
                    )

                ax.set_xlabel("Trace Number")
                ax.set_ylabel("Range Bin (two-way travel time)")
                ax.set_title(f"SHARAD Radargram — {pid}")
                ax.legend(loc="upper right", fontsize=8)

                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor="white")
                plt.close(fig)
                buf.seek(0)
                figures["sharad_radargram"] = base64.b64encode(buf.read()).decode()
                figures["sharad_radargram_pid"] = pid
            except Exception as e:
                logger.warning(f"Failed to generate SHARAD figure: {e}")

    # ── CRISM ice score map ──
    mineral_result = session.all_results.get("mineral")
    if mineral_result and mineral_result.success:
        top_ice = mineral_result.data.get("top_ice_candidates", [])
        if top_ice:
            best_obs = top_ice[0].get("obs_id")
            if best_obs:
                try:
                    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    score_dir = os.path.join(base_dir, "crism_score", best_obs)
                    ice_npy = os.path.join(score_dir, "ice_score.npy")

                    if os.path.exists(ice_npy):
                        ice_arr = np.load(ice_npy)
                        mask_path = os.path.join(score_dir, "valid_mask.npy")
                        if os.path.exists(mask_path):
                            mask = np.load(mask_path)
                            ice_arr = np.where(mask, ice_arr, np.nan)

                        fig, ax = plt.subplots(figsize=(8, 6))
                        im = ax.imshow(ice_arr, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
                        plt.colorbar(im, ax=ax, label="Ice Score (0–1)")
                        ax.set_title(f"CRISM Ice Score Map — {best_obs}")
                        ax.set_xlabel("Sample")
                        ax.set_ylabel("Line")

                        buf = io.BytesIO()
                        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor="white")
                        plt.close(fig)
                        buf.seek(0)
                        figures["crism_ice_map"] = base64.b64encode(buf.read()).decode()
                        figures["crism_ice_map_obs_id"] = best_obs
                except Exception as e:
                    logger.warning(f"Failed to generate CRISM figure: {e}")

    # ── Slope grid heatmap ──
    slope_result = session.all_results.get("slope")
    if slope_result and slope_result.success:
        grid_points = slope_result.data.get("grid_points", [])
        if len(grid_points) >= 4:
            try:
                lats = sorted(set(p["lat"] for p in grid_points))
                lons = sorted(set(p["lon"] for p in grid_points))

                if len(lats) > 1 and len(lons) > 1:
                    grid = np.full((len(lats), len(lons)), np.nan)
                    for p in grid_points:
                        i = lats.index(p["lat"])
                        j = lons.index(p["lon"])
                        grid[i, j] = p["mean_slope"]

                    fig, ax = plt.subplots(figsize=(8, 6))
                    im = ax.imshow(
                        grid, extent=[min(lons), max(lons), min(lats), max(lats)],
                        origin="lower", cmap="RdYlGn_r", vmin=0, vmax=15,
                        aspect="auto", interpolation="nearest",
                    )
                    plt.colorbar(im, ax=ax, label="Mean Slope (deg)")
                    # 5-degree contour
                    try:
                        ax.contour(lons, lats, grid, levels=[5], colors="white", linewidths=1.5, linestyles="--")
                    except Exception:
                        pass
                    ax.set_xlabel("Longitude (deg E)")
                    ax.set_ylabel("Latitude (deg N)")
                    ax.set_title("Slope Grid — Engineering Feasibility (white dashed = 5 deg threshold)")

                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor="white")
                    plt.close(fig)
                    buf.seek(0)
                    figures["slope_grid"] = base64.b64encode(buf.read()).decode()
            except Exception as e:
                logger.warning(f"Failed to generate slope figure: {e}")

    return figures


# =============================================================================
# Evidence Figures for live SSE stream (annotated, richer than report figures)
# =============================================================================

def generate_evidence_figures(session: AgentSession) -> Dict:
    """Generate annotated evidence figures for the live SSE stream.

    Returns {"figures": [{"id", "title", "caption", "instrument", "base64"}, ...]}.
    Each figure is independently wrapped in try/except so partial results are fine.
    """
    figures: list[dict] = []

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.info("matplotlib not available, skipping evidence figures")
        return {"figures": []}

    def _fig_to_b64(fig) -> str:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                    facecolor="#0d1520", edgecolor="none")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()

    # ── Figure A: Annotated SHARAD Radargram ──
    sub_result = session.all_results.get("subsurface")
    if sub_result and sub_result.success:
        tracks = sub_result.data.get("tracks", [])
        # Pick best track: prefer one with subsurface_picks
        best_track = None
        for t in tracks:
            if t.get("subsurface_picks") and t.get("analyzed"):
                best_track = t
                break
        if not best_track:
            for t in tracks:
                if t.get("subsurface_detected") and t.get("analyzed"):
                    best_track = t
                    break
        if not best_track:
            for t in tracks:
                if t.get("analyzed"):
                    best_track = t
                    break

        if best_track:
            try:
                from .sharad_highres_router import _get_power, _pick_surface
                pid = best_track["product_id"]
                power, n_traces = _get_power(pid)
                surface = _pick_surface(pid, power)

                fig, ax = plt.subplots(figsize=(14, 5))
                fig.patch.set_facecolor("#0d1520")
                ax.set_facecolor("#0d1520")

                power_db = 10 * np.log10(power.T.astype(np.float64) + 1e-12)
                vmin_p, vmax_p = np.percentile(power_db, 5), np.percentile(power_db, 95)
                ax.imshow(power_db, aspect="auto", cmap="gray", vmin=vmin_p, vmax=vmax_p)

                # Surface return (red)
                valid = surface >= 0
                ax.plot(np.where(valid)[0], surface[valid], "r-", linewidth=0.6,
                        alpha=0.8, label="Surface return")

                picks = best_track.get("subsurface_picks", [])
                depth_info = best_track.get("estimated_depth_m", {})
                caption_parts = [f"Product: {pid}"]

                if picks:
                    # Plot subsurface reflector picks (yellow dots)
                    tr_idxs = [pk["trace_idx"] for pk in picks]
                    bin_idxs = [pk["bin_idx"] for pk in picks]
                    ax.scatter(tr_idxs, bin_idxs, c="yellow", s=3, alpha=0.5,
                               zorder=5, label="Subsurface reflector")

                    # Annotate strongest detection
                    strongest = max(picks, key=lambda p: p["snr"])
                    ax.annotate(
                        f"Reflector\n{strongest['depth_m']:.0f} m depth\n"
                        f"SNR = {strongest['snr']:.1f}",
                        xy=(strongest["trace_idx"], strongest["bin_idx"]),
                        xytext=(strongest["trace_idx"] + n_traces * 0.08,
                                max(strongest["bin_idx"] - 60, 10)),
                        fontsize=8, color="yellow", fontweight="bold",
                        arrowprops=dict(arrowstyle="->", color="yellow", lw=1.5),
                        bbox=dict(facecolor="black", alpha=0.8, edgecolor="yellow",
                                  lw=0.8, boxstyle="round,pad=0.3"),
                    )
                    caption_parts.append(
                        f"{len(picks)} subsurface reflectors detected (SNR >= 2.5)"
                    )

                    # Depth range info box
                    if depth_info:
                        info_text = (
                            f"εr = {depth_info.get('epsilon_r', 3.15)}\n"
                            f"Depth: {depth_info.get('min', 0):.0f}"
                            f"–{depth_info.get('max', 0):.0f} m\n"
                            f"Median: {depth_info.get('median', 0):.0f} m"
                        )
                        ax.text(
                            0.02, 0.96, info_text, transform=ax.transAxes,
                            fontsize=8, color="cyan", va="top", fontfamily="monospace",
                            bbox=dict(facecolor="black", alpha=0.8, edgecolor="cyan",
                                      lw=0.8, boxstyle="round,pad=0.4"),
                        )
                        caption_parts.append(
                            f"Estimated depth {depth_info.get('min', 0):.0f}"
                            f"–{depth_info.get('max', 0):.0f} m "
                            f"(εr = {depth_info.get('epsilon_r', 3.15)}, water-ice)"
                        )
                else:
                    # Fallback: show search zone lines only
                    if valid.any():
                        mean_surf = int(np.mean(surface[valid]))
                        ax.axhline(y=mean_surf + 20, color="cyan", lw=0.4,
                                   alpha=0.5, linestyle="--")
                        ax.axhline(y=min(mean_surf + 200, power.shape[1] - 1),
                                   color="cyan", lw=0.4, alpha=0.5, linestyle="--",
                                   label="Search zone")
                    caption_parts.append("No clear subsurface reflector detected")

                ax.set_xlabel("Trace Number", color="#92a4c9", fontsize=9)
                ax.set_ylabel("Range Bin", color="#92a4c9", fontsize=9)
                ax.set_title(f"SHARAD Radargram — {pid}", color="white",
                             fontsize=11, fontweight="bold")
                ax.tick_params(colors="#6b7c9c", labelsize=8)
                ax.legend(loc="upper right", fontsize=7, facecolor="black",
                          edgecolor="#232f48", labelcolor="white")

                figures.append({
                    "id": "sharad_radargram",
                    "title": f"SHARAD Radargram — {pid}",
                    "caption": ". ".join(caption_parts) + ".",
                    "instrument": "SHARAD",
                    "base64": _fig_to_b64(fig),
                })
            except Exception as e:
                logger.warning(f"Evidence SHARAD figure failed: {e}")

    # ── Figure B: CRISM Ice Score Map ──
    mineral_result = session.all_results.get("mineral")
    if mineral_result and mineral_result.success:
        top_ice = mineral_result.data.get("top_ice_candidates", [])
        if top_ice:
            best_obs = top_ice[0].get("obs_id")
            if best_obs:
                try:
                    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    score_dir = os.path.join(base_dir, "crism_score", best_obs)
                    ice_npy = os.path.join(score_dir, "ice_score.npy")

                    if os.path.exists(ice_npy):
                        ice_arr = np.load(ice_npy)
                        mask_path = os.path.join(score_dir, "valid_mask.npy")
                        if os.path.exists(mask_path):
                            mask = np.load(mask_path)
                            ice_arr = np.where(mask, ice_arr, np.nan)

                        fig, ax = plt.subplots(figsize=(8, 6))
                        fig.patch.set_facecolor("#0d1520")
                        ax.set_facecolor("#0d1520")

                        im = ax.imshow(ice_arr, cmap="YlOrRd", vmin=0, vmax=1,
                                       aspect="auto")
                        cb = plt.colorbar(im, ax=ax, label="Ice Score (0–1)")
                        cb.ax.yaxis.label.set_color("white")
                        cb.ax.tick_params(colors="#92a4c9")

                        # Contour at detection threshold
                        try:
                            ice_clean = np.nan_to_num(ice_arr, nan=0.0)
                            ax.contour(ice_clean, levels=[0.3], colors="white",
                                       linewidths=0.8, linestyles="--")
                        except Exception:
                            pass

                        ax.set_title(f"CRISM Ice Score — {best_obs}", color="white",
                                     fontsize=11, fontweight="bold")
                        ax.set_xlabel("Sample", color="#92a4c9", fontsize=9)
                        ax.set_ylabel("Line", color="#92a4c9", fontsize=9)
                        ax.tick_params(colors="#6b7c9c", labelsize=8)

                        ice_pct = top_ice[0].get("ice_percent", 0)
                        figures.append({
                            "id": "crism_ice_map",
                            "title": f"CRISM Ice Score — {best_obs}",
                            "caption": (
                                f"Ice score heatmap for {best_obs}. "
                                f"White dashed contour marks 0.3 detection threshold. "
                                f"{ice_pct:.1f}% of pixels above ice threshold."
                            ),
                            "instrument": "CRISM",
                            "base64": _fig_to_b64(fig),
                        })
                except Exception as e:
                    logger.warning(f"Evidence CRISM figure failed: {e}")

    # ── Figure C: CNN Mineral Classification Map ──
    cnn_result = session.all_results.get("mineral_cnn")
    if cnn_result and cnn_result.success:
        observations = cnn_result.data.get("observations", [])
        if observations:
            best_cnn_obs = observations[0].get("obs_id")
            if best_cnn_obs:
                try:
                    from .mineral_cnn.constants import CLASS_NAME, RESULTS_DIR
                    from .mineral_cnn.router import _MINERAL_COLORS
                    result_dir = os.path.join(RESULTS_DIR, best_cnn_obs)
                    mmap_path = os.path.join(result_dir, "mineral_map.npy")
                    vmask_path = os.path.join(result_dir, "valid_mask.npy")

                    if os.path.exists(mmap_path):
                        mineral_map = np.load(mmap_path)
                        valid_mask = (np.load(vmask_path)
                                      if os.path.exists(vmask_path)
                                      else mineral_map >= 0)

                        rows, cols = mineral_map.shape
                        rgba = np.zeros((rows, cols, 4), dtype=np.uint8)
                        rgba[:, :, :3] = 30  # dark bg for unclassified

                        legend_items = []
                        for mid, color in _MINERAL_COLORS.items():
                            m = mineral_map == mid
                            if m.any():
                                rgba[m, 0] = color[0]
                                rgba[m, 1] = color[1]
                                rgba[m, 2] = color[2]
                                rgba[m, 3] = 255
                                legend_items.append((
                                    int(m.sum()),
                                    mid,
                                    CLASS_NAME.get(mid, f"Class {mid}"),
                                    color,
                                ))
                        # Invalid pixels stay dark
                        rgba[~valid_mask, 3] = 80

                        # Sort legend by pixel count descending
                        legend_items.sort(key=lambda x: -x[0])

                        fig, ax = plt.subplots(figsize=(8, 6))
                        fig.patch.set_facecolor("#0d1520")
                        ax.set_facecolor("#0d1520")
                        ax.imshow(rgba, aspect="auto")

                        # Draw legend for top 5 minerals
                        for i, (cnt, mid, name, color) in enumerate(legend_items[:5]):
                            norm_color = (color[0] / 255, color[1] / 255, color[2] / 255)
                            ax.plot([], [], "s", color=norm_color, markersize=8,
                                    label=f"{name} ({cnt:,}px)")
                        ax.legend(loc="upper right", fontsize=7, facecolor="black",
                                  edgecolor="#232f48", labelcolor="white")

                        ax.set_title(f"CNN Mineral Map — {best_cnn_obs}", color="white",
                                     fontsize=11, fontweight="bold")
                        ax.set_xlabel("Sample", color="#92a4c9", fontsize=9)
                        ax.set_ylabel("Line", color="#92a4c9", fontsize=9)
                        ax.tick_params(colors="#6b7c9c", labelsize=8)

                        top_names = [x[2] for x in legend_items[:3]]
                        figures.append({
                            "id": "cnn_mineral_map",
                            "title": f"CNN Mineral Map — {best_cnn_obs}",
                            "caption": (
                                f"CNN classification (95% confidence) for {best_cnn_obs}. "
                                f"Top minerals: {', '.join(top_names)}. "
                                f"{len(legend_items)} mineral classes detected."
                            ),
                            "instrument": "CRISM_CNN",
                            "base64": _fig_to_b64(fig),
                        })
                except Exception as e:
                    logger.warning(f"Evidence CNN mineral figure failed: {e}")

    # ── Figure D: DTM Elevation Patch ──
    # Show elevation near the best subsurface detection if DTM available
    if sub_result and sub_result.success:
        tracks = sub_result.data.get("tracks", [])
        best_sub = None
        for t in tracks:
            if t.get("subsurface_detected") and t.get("lat") is not None:
                best_sub = t
                break
        if best_sub:
            # Find DTM products from search results (stored as search_HIRISE_DTM etc.)
            dtm_products = []
            for key, res in session.all_results.items():
                if not key.startswith("search_"):
                    continue
                if res and res.success:
                    for p in res.data.get("products", []):
                        if p.get("instrument") == "HIRISE_DTM":
                            dtm_products.append(p)

            if dtm_products:
                sub_lat = best_sub.get("lat", 0)
                sub_lon = best_sub.get("lon", 0)
                # Find nearest DTM
                nearest_dtm = min(dtm_products, key=lambda d: (
                    (d.get("lat", 999) - sub_lat) ** 2 +
                    (d.get("lon", 999) - sub_lon) ** 2
                ))
                dist = ((nearest_dtm.get("lat", 999) - sub_lat) ** 2 +
                        (nearest_dtm.get("lon", 999) - sub_lon) ** 2) ** 0.5
                if dist < 1.0:  # within ~1 degree
                    try:
                        from .terrain_router import compute_hirise_dtm_patch
                        dtm_id = nearest_dtm["product_id"]
                        patch = compute_hirise_dtm_patch(
                            dtm_id, sub_lat, sub_lon,
                            radius_m=2000, grid_size=128,
                        )
                        elev = np.array(patch.get("elevation") or patch.get("grid"))
                        if elev is not None and elev.size > 0:
                            fig, ax = plt.subplots(figsize=(7, 6))
                            fig.patch.set_facecolor("#0d1520")
                            ax.set_facecolor("#0d1520")

                            im = ax.imshow(elev, cmap="gist_earth", aspect="equal")
                            cb = plt.colorbar(im, ax=ax, label="Elevation (m)")
                            cb.ax.yaxis.label.set_color("white")
                            cb.ax.tick_params(colors="#92a4c9")

                            # Mark subsurface detection location (center)
                            cy, cx = elev.shape[0] // 2, elev.shape[1] // 2
                            ax.plot(cx, cy, "*", color="yellow", markersize=14,
                                    markeredgecolor="black", markeredgewidth=0.5,
                                    label="Subsurface detection")
                            ax.legend(loc="upper right", fontsize=7, facecolor="black",
                                      edgecolor="#232f48", labelcolor="white")

                            ax.set_title(
                                f"DTM Elevation — {dtm_id}", color="white",
                                fontsize=11, fontweight="bold",
                            )
                            ax.tick_params(colors="#6b7c9c", labelsize=8)

                            relief = float(np.nanmax(elev) - np.nanmin(elev))
                            figures.append({
                                "id": "dtm_elevation",
                                "title": f"DTM Elevation — {dtm_id}",
                                "caption": (
                                    f"HiRISE DTM elevation near subsurface detection "
                                    f"({sub_lat:.3f}N, {sub_lon:.3f}E). "
                                    f"Relief: {relief:.0f} m over 4 km patch. "
                                    f"Yellow star marks radar detection."
                                ),
                                "instrument": "HIRISE_DTM",
                                "base64": _fig_to_b64(fig),
                            })
                    except Exception as e:
                        logger.warning(f"Evidence DTM figure failed: {e}")

    # ── Figure E: Slope Grid ──
    slope_result = session.all_results.get("slope")
    if slope_result and slope_result.success:
        grid_points = slope_result.data.get("grid_points", [])
        if len(grid_points) >= 4:
            try:
                lats = sorted(set(p["lat"] for p in grid_points))
                lons = sorted(set(p["lon"] for p in grid_points))

                if len(lats) > 1 and len(lons) > 1:
                    grid = np.full((len(lats), len(lons)), np.nan)
                    for p in grid_points:
                        i = lats.index(p["lat"])
                        j = lons.index(p["lon"])
                        grid[i, j] = p["mean_slope"]

                    fig, ax = plt.subplots(figsize=(8, 6))
                    fig.patch.set_facecolor("#0d1520")
                    ax.set_facecolor("#0d1520")

                    im = ax.imshow(
                        grid, extent=[min(lons), max(lons), min(lats), max(lats)],
                        origin="lower", cmap="RdYlGn_r", vmin=0, vmax=15,
                        aspect="auto", interpolation="nearest",
                    )
                    cb = plt.colorbar(im, ax=ax, label="Mean Slope (deg)")
                    cb.ax.yaxis.label.set_color("white")
                    cb.ax.tick_params(colors="#92a4c9")

                    try:
                        ax.contour(lons, lats, grid, levels=[5], colors="white",
                                   linewidths=1.5, linestyles="--")
                    except Exception:
                        pass

                    # Mark best point with star
                    best_pt = slope_result.data.get("best_point")
                    if best_pt:
                        ax.plot(best_pt["lon"], best_pt["lat"], "*", color="cyan",
                                markersize=14, markeredgecolor="black",
                                markeredgewidth=0.5, label="Best site")
                        ax.legend(loc="upper right", fontsize=7, facecolor="black",
                                  edgecolor="#232f48", labelcolor="white")

                    ax.set_xlabel("Longitude (°E)", color="#92a4c9", fontsize=9)
                    ax.set_ylabel("Latitude (°N)", color="#92a4c9", fontsize=9)
                    ax.set_title("Slope Grid — Engineering Feasibility",
                                 color="white", fontsize=11, fontweight="bold")
                    ax.tick_params(colors="#6b7c9c", labelsize=8)

                    fav = slope_result.data.get("favorable_zones", 0)
                    total_z = len(grid_points)
                    figures.append({
                        "id": "slope_grid",
                        "title": "Slope Grid — Engineering Feasibility",
                        "caption": (
                            f"Slope analysis over {total_z}-point grid. "
                            f"White dashed line = 5° safety threshold. "
                            f"{fav}/{total_z} zones FAVORABLE. "
                            f"Cyan star marks optimal landing site."
                        ),
                        "instrument": "MOLA_DTM",
                        "base64": _fig_to_b64(fig),
                    })
            except Exception as e:
                logger.warning(f"Evidence slope figure failed: {e}")

    # ── Figure F: Multi-Instrument Comparison Composite ──
    # Only create if we have >= 2 instrument-specific figures
    instrument_figs = [f for f in figures if f["id"] != "slope_grid"]
    if len(instrument_figs) >= 2:
        try:
            n = min(len(instrument_figs), 4)
            fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
            fig.patch.set_facecolor("#0d1520")
            if n == 1:
                axes = [axes]

            for i, ifig in enumerate(instrument_figs[:n]):
                ax = axes[i]
                ax.set_facecolor("#0d1520")
                # Decode the base64 image to display as thumbnail
                img_data = base64.b64decode(ifig["base64"])
                img_arr = plt.imread(io.BytesIO(img_data), format="png")
                ax.imshow(img_arr)
                ax.set_title(ifig["instrument"], color="white", fontsize=10,
                             fontweight="bold")
                ax.tick_params(left=False, bottom=False, labelleft=False,
                               labelbottom=False)
                for spine in ax.spines.values():
                    spine.set_color("#232f48")

            fig.suptitle("Multi-Instrument Evidence Comparison", color="white",
                         fontsize=13, fontweight="bold", y=1.02)
            fig.tight_layout()

            figures.append({
                "id": "comparison_composite",
                "title": "Multi-Instrument Evidence Comparison",
                "caption": (
                    f"Side-by-side comparison of {n} instrument datasets. "
                    f"Instruments: {', '.join(f['instrument'] for f in instrument_figs[:n])}."
                ),
                "instrument": "COMPOSITE",
                "base64": _fig_to_b64(fig),
            })
        except Exception as e:
            logger.warning(f"Evidence composite figure failed: {e}")

    logger.info(f"Generated {len(figures)} evidence figures for session {session.session_id}")
    return {"figures": figures}


def _fig_markdown(b64: str, caption: str, explanation: str) -> list[str]:
    """Return markdown lines embedding a base64 PNG with caption."""
    return [
        f"![{caption}](data:image/png;base64,{b64})",
        "",
        f"*{caption}.* {explanation}",
        "",
    ]


def _build_report_markdown(session: AgentSession) -> str:
    """Build a decision-oriented Markdown mission assessment report.

    Order: Objective → SHARAD → CRISM → Cross-Instrument → Engineering →
    Landing Site Decision → Assessment Score → Data Confidence → Log.
    """
    synthesis = session.synthesis or {}
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines: list[str] = []

    # Generate figures
    figures = _generate_figures(session)

    # ── Title ──
    lines.append("# MarsLab — Mission Assessment Report")
    lines.append("")
    lines.append(f"**Generated:** {now}  ")
    lines.append(f"**Session:** `{session.session_id}`  ")
    lines.append(f"**Region:** {session.region_name or 'Unknown'}  ")
    lines.append(f"**Instruments:** {', '.join(synthesis.get('instruments_searched', []))}  ")
    lines.append("")

    # ── 1. Mission Objective ──
    lines.append("## 1. Mission Objective")
    lines.append("")
    lines.append(session.objective)
    lines.append("")

    # ── 2. Subsurface Potential (SHARAD) ──
    sub = synthesis.get("subsurface_coverage", {})
    if sub:
        lines.append("## 2. Subsurface Potential (SHARAD Radar Analysis)")
        lines.append("")
        n_analyzed = sub.get("analyzed_count", 0)
        n_detect = sub.get("subsurface_detections", 0)
        depth = sub.get("depth_summary", {})

        lines.append("| Parameter | Value |")
        lines.append("|-----------|-------|")
        if n_analyzed:
            lines.append(f"| Radargrams Analyzed | {n_analyzed} |")
        lines.append(f"| Subsurface Reflector Detections | **{n_detect}** |")
        if depth:
            lines.append(f"| Estimated Depth Range | {depth.get('min_depth_m', 'N/A')} – {depth.get('max_depth_m', 'N/A')} m |")
            lines.append(f"| Median Depth | {depth.get('median_depth_m', 'N/A')} m |")
            lines.append(f"| Dielectric Constant (εr) | {depth.get('epsilon_r_assumed', 3.15)} (water-ice) |")
        lines.append("")

        if n_detect > 0 and depth:
            try:
                med_d = float(depth.get('median_depth_m', 50))
            except (TypeError, ValueError):
                med_d = 50.0
            if med_d <= 10:
                depth_qual = (
                    "This is **shallow ice** — potentially accessible for in-situ "
                    "resource extraction without deep drilling."
                )
            elif med_d <= 30:
                depth_qual = (
                    "This depth is moderate — drilling infrastructure would be "
                    "required for extraction."
                )
            elif med_d <= 80:
                depth_qual = (
                    "This depth is significant and would require substantial "
                    "drilling capability."
                )
            else:
                depth_qual = (
                    "This depth is **too deep for practical ice extraction** "
                    "and reduces the site's utility for ISRU."
                )
            lines.append(
                f"SHARAD analysis detected **{n_detect} subsurface reflector(s)**, "
                f"providing direct evidence of a dielectric interface. "
                f"Assuming εr = {depth.get('epsilon_r_assumed', 3.15)} (water-ice), "
                f"the ice-table depth is estimated at "
                f"**{depth['min_depth_m']}–{depth['max_depth_m']} m** "
                f"(median {depth['median_depth_m']} m). {depth_qual}"
            )
        elif n_analyzed > 0:
            lines.append(
                f"{n_analyzed} SHARAD radargrams were analyzed but no subsurface "
                f"reflectors exceeded the detection threshold (SNR ≥ 2.5). "
                f"This does not rule out ice — deposits may be below SHARAD "
                f"vertical resolution (~15 m) or masked by surface clutter."
            )
        else:
            lines.append(
                f"No high-resolution radargram data was available for quantitative analysis. "
                f"Coverage level: {sub.get('coverage', 'NONE')}."
            )
        lines.append("")

        # SHARAD figure
        if "sharad_radargram" in figures:
            lines.extend(_fig_markdown(
                figures["sharad_radargram"],
                f"Figure 1: SHARAD Radargram ({figures.get('sharad_radargram_pid', 'N/A')})",
                "Red line marks the auto-picked surface return. Cyan dashed lines delimit "
                "the subsurface search zone (20–200 range bins below surface). "
                "Any subsurface reflector within this zone is evaluated for SNR ≥ 2.5.",
            ))

    # ── 3. Surface / Near-Surface Composition (CRISM) ──
    mineral = synthesis.get("mineral_signatures", {})
    if mineral:
        lines.append("## 3. Surface / Near-Surface Composition (CRISM)")
        lines.append("")

        high_ice = mineral.get("high_ice_count", 0)
        high_hyd = mineral.get("high_hyd_count", 0)
        crism_n = mineral.get("crism_count", 0)

        lines.append("| Parameter | Value |")
        lines.append("|-----------|-------|")
        lines.append(f"| CRISM Products Analyzed | {crism_n} |")
        lines.append(f"| Significant Ice Signatures | **{high_ice}** |")
        lines.append(f"| Significant Hydration | {high_hyd} |")
        lines.append("")

        if high_ice > 0:
            lines.append(
                f"**{high_ice}** CRISM products show ice spectral signatures "
                f"(>5% of pixels above 0.3 threshold). These represent *indirect* "
                f"evidence — spectral proxies for surface or near-surface ice/hydration, "
                f"not a direct detection of buried ice."
            )
        else:
            lines.append(
                "No significant CRISM ice signatures exceeded the detection threshold. "
                "This may indicate ice is absent at the surface, or masked by dust/regolith."
            )
        lines.append("")

        # Top ice candidates table
        top_ice = mineral.get("top_ice_candidates", [])
        if top_ice:
            lines.append("### Ranked Ice Candidates")
            lines.append("")
            lines.append("| Obs ID | Lat | Lon | Ice Score | Ice % |")
            lines.append("|--------|-----|-----|-----------|-------|")
            for c in top_ice[:10]:
                lat = c.get("lat")
                lon = c.get("lon")
                lat_s = f"{lat:.3f}" if lat is not None else "N/A"
                lon_s = f"{lon:.3f}" if lon is not None else "N/A"
                lines.append(
                    f"| {c.get('obs_id', '?')} | {lat_s} | {lon_s} | "
                    f"{c.get('ice_mean_score', 0):.3f} | {c.get('ice_percent', 0):.1f}% |"
                )
            lines.append("")

        hotspot = mineral.get("ice_hotspot", {})
        if hotspot and hotspot.get("center_lat") is not None:
            lines.append(
                f"**Ice Hotspot Centroid:** ({hotspot['center_lat']:.4f}, "
                f"{hotspot['center_lon']:.4f}) — {hotspot.get('n_products', 0)} "
                f"high-ice products, maximum {hotspot.get('max_ice_pct', 0)}% ice."
            )
            lines.append("")

        # CRISM figure
        if "crism_ice_map" in figures:
            lines.extend(_fig_markdown(
                figures["crism_ice_map"],
                f"Figure 2: CRISM Ice Score Map ({figures.get('crism_ice_map_obs_id', 'N/A')})",
                "Per-pixel ice probability score (0–1) from CRISM MTRDR browse product. "
                "Warmer colors indicate higher ice spectral response. "
                "Pixels above 0.3 are counted as significant ice detections.",
            ))

    # ── 4. Cross-Instrument Consistency Analysis ──
    cross = synthesis.get("cross_instrument", {})
    if cross and cross.get("notes"):
        lines.append("## 4. Cross-Instrument Consistency Analysis")
        lines.append("")
        for note in cross["notes"]:
            lines.append(note)
            lines.append("")

        dist = cross.get("sharad_crism_min_distance_km")
        if dist is not None:
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            lines.append(f"| SHARAD–CRISM Minimum Separation | **{dist:.0f} km** |")
            lines.append(f"| Coincident Detections (within 100 km) | {cross.get('coincident_detections', 0)} |")
            lines.append(f"| Evidence Consistency | {cross.get('evidence_consistency', 'N/A')} |")
            lines.append("")

        n_direct = len(cross.get("direct_ice_evidence", []))
        n_indirect = len(cross.get("indirect_ice_evidence", []))
        if n_direct or n_indirect:
            lines.append(
                f"**Evidence inventory:** {n_direct} direct (SHARAD subsurface reflector) "
                f"and {n_indirect} indirect (CRISM spectral proxy) ice indicators."
            )
            lines.append("")

    # ── 5. Engineering Feasibility (Terrain — Final Filter) ──
    eng = synthesis.get("engineering_feasibility", {})
    if eng:
        lines.append("## 5. Engineering Feasibility (Terrain — Final Filter)")
        lines.append("")
        lines.append(
            "*Terrain slope is applied as a final feasibility constraint, "
            "not as the primary site-selection driver.*"
        )
        lines.append("")
        lines.append("| Parameter | Value |")
        lines.append("|-----------|-------|")
        lines.append(f"| Safety Rating | **{eng.get('safety', 'UNKNOWN')}** |")
        lines.append(f"| Mean Slope | {eng.get('mean_slope', 'N/A')} deg |")
        lines.append(f"| Max Slope | {eng.get('max_slope', 'N/A')} deg |")
        grid_size = eng.get("grid_size")
        if grid_size:
            fav = eng.get("favorable_zones", 0)
            lines.append(f"| Grid Points Sampled | {grid_size} |")
            lines.append(f"| Favorable Zones (< 5 deg) | {fav} / {grid_size} |")
        lines.append("")

        # Slope figure
        if "slope_grid" in figures:
            lines.extend(_fig_markdown(
                figures["slope_grid"],
                "Figure 3: Slope Grid Analysis",
                "Mean slope (degrees) at each grid sample point. Green zones are below "
                "5 degrees (favorable for landing). Red zones exceed the engineering threshold. "
                "White dashed contour marks the 5-degree boundary.",
            ))

    # ── 6. Climate Constraints (MCD Model) ──
    clim = synthesis.get("climate", {})
    if clim:
        lines.append("## 6. Climate Constraints (Parametric MCD Model)")
        lines.append("")
        annual = clim.get("annual_stats", {})
        if annual:
            lines.append("| Parameter | Value |")
            lines.append("|-----------|-------|")
            lines.append(f"| Mean Annual Temperature | **{annual.get('temp_mean_k', 'N/A')} K** |")
            lines.append(f"| Temperature Range | {annual.get('temp_min_k', 'N/A')} – {annual.get('temp_max_k', 'N/A')} K |")
            lines.append(f"| Surface Pressure | {annual.get('pressure_pa', 'N/A')} Pa |")
            lines.append(f"| Elevation | {annual.get('elevation_m', 'N/A')} m |")
            lines.append(f"| Dust Opacity (mean / peak) | {annual.get('dust_tau_mean', 'N/A')} / {annual.get('dust_tau_peak', 'N/A')} |")
            lines.append(f"| Wind (mean / max gust) | {annual.get('wind_mean_ms', 'N/A')} / {annual.get('wind_gust_max_ms', 'N/A')} m/s |")
            lines.append(f"| CO2 Frost Probability | {annual.get('frost_max_probability', 0):.0%} |")
            lines.append(f"| Climate Score | **{clim.get('climate_score', 0)} / 10** |")
            lines.append("")

        summary = clim.get("climate_summary", "")
        if summary:
            lines.append(summary)
            lines.append("")

    # ── 7. Thermal Inertia (TES) ──
    ti = synthesis.get("thermal_inertia", {})
    if ti:
        lines.append("## 7. Thermal Inertia (TES)")
        lines.append("")
        if ti.get("ti_median") is not None:
            lines.append("| Parameter | Value |")
            lines.append("|-----------|-------|")
            lines.append(f"| Median TI | **{ti.get('ti_median', 'N/A')}** J/(m²·K·s^0.5) |")
            lines.append(f"| Mean TI | {ti.get('ti_mean', 'N/A')} |")
            lines.append(f"| Classification | {ti.get('classification', 'N/A')} |")
            lines.append(f"| TI Score | **{ti.get('ti_score', 0)} / 10** |")
            lines.append("")

            dist = ti.get("distribution_pct", {})
            if dist:
                lines.append("**Surface Material Distribution:**")
                lines.append("")
                lines.append(f"- Dusty (TI < 150): {dist.get('dusty_lt150', 0)}%")
                lines.append(f"- Mixed (150-300): {dist.get('mixed_150_300', 0)}%")
                lines.append(f"- Consolidated (300-600): {dist.get('consolidated_300_600', 0)}%")
                lines.append(f"- Bedrock (> 600): {dist.get('bedrock_gt600', 0)}%")
                lines.append("")

        explanation = ti.get("ti_explanation", "")
        if explanation:
            lines.append(explanation)
            lines.append("")

    # ── 8. Landing Site Decision ──
    rec_data = synthesis.get("recommended_site", {})
    primary = rec_data.get("primary_site") or rec_data.get("best_site") if rec_data else None
    secondary = rec_data.get("secondary_site") if rec_data else None
    science_targets = rec_data.get("science_targets", []) if rec_data else []
    trade_offs = rec_data.get("trade_offs", []) if rec_data else []

    lines.append("## 8. Landing Site Decision")
    lines.append("")

    if primary and primary.get("lat") is not None:
        lines.append("### Primary Landing Site")
        lines.append("")
        lines.append("| Parameter | Value |")
        lines.append("|-----------|-------|")
        lines.append(f"| **Latitude** | **{_fmt_lat(primary['lat'])}** |")
        lines.append(f"| **Longitude** | **{_fmt_lon(primary['lon'])}** |")
        lines.append(f"| Composite Score | {primary.get('score', 'N/A')} / 100 |")
        lines.append(f"| Mean Slope | {primary.get('mean_slope', 'N/A')} deg |")
        lines.append(f"| Elevation | {primary.get('elevation_m', 'N/A')} m |")
        lines.append("")
        reasons = primary.get("reasons", [])
        if reasons:
            lines.append("**Why this site:**")
            for r in reasons:
                lines.append(f"- {r}")
            lines.append("")
    else:
        lines.append("*No landing site could be determined from available data.*")
        lines.append("")

    if secondary and secondary.get("lat") is not None:
        lines.append("### Secondary / Backup Site")
        lines.append("")
        lines.append("| Parameter | Value |")
        lines.append("|-----------|-------|")
        lines.append(f"| Latitude | {_fmt_lat(secondary['lat'])} |")
        lines.append(f"| Longitude | {_fmt_lon(secondary['lon'])} |")
        lines.append(f"| Composite Score | {secondary.get('score', 'N/A')} / 100 |")
        lines.append(f"| Mean Slope | {secondary.get('mean_slope', 'N/A')} deg |")
        lines.append("")

    if science_targets:
        lines.append("### Science Targets (Non-Landing)")
        lines.append("")
        lines.append("These locations have high science value but may require extended traverse:")
        lines.append("")
        for st in science_targets:
            lat_s = f"{st['lat']:.3f}" if st.get("lat") is not None else "N/A"
            lon_s = f"{st['lon']:.3f}" if st.get("lon") is not None else "N/A"
            lines.append(f"- ({lat_s}, {lon_s}): {st.get('reason', 'High science value')}")
        lines.append("")

    if trade_offs:
        lines.append("### Trade-offs")
        lines.append("")
        for t in trade_offs:
            lines.append(f"- {t}")
        lines.append("")

    # ── 7. Assessment Score ──
    score_range = synthesis.get("score_range", {})
    score_lo = score_range.get("low", synthesis.get("overall_score", 0))
    score_hi = score_range.get("high", synthesis.get("overall_score", 0))
    rec_label = str(synthesis.get("recommendation", "N/A")).replace("_", " ")
    strengths = synthesis.get("strengths", [])
    uncertainties_list = synthesis.get("uncertainties", [])

    lines.append("## 9. Assessment Score")
    lines.append("")
    lines.append(f"**Score: {score_lo}–{score_hi} / 100** (point estimate: {synthesis.get('overall_score', 'N/A')})")
    lines.append(f"**Recommendation:** {rec_label}")
    lines.append("")

    if strengths:
        lines.append("**Primary Strengths:**")
        lines.append("")
        for s in strengths:
            lines.append(f"- {s}")
        lines.append("")

    if uncertainties_list:
        lines.append("**Primary Uncertainties:**")
        lines.append("")
        for u in uncertainties_list:
            lines.append(f"- {u}")
        lines.append("")

    # ── Detailed Analysis (narrative) ──
    if session.narrative:
        lines.append("## Detailed Analysis")
        lines.append("")
        lines.append(session.narrative)
        lines.append("")

    # ── Appendix A: Data Confidence ──
    lines.append("## Appendix A: Data Confidence")
    lines.append("")
    lines.append("*Coverage statistics are a confidence indicator, not a primary result.*")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total Products Found | {synthesis.get('total_products_found', 0)} |")
    lines.append(f"| Available Locally | {synthesis.get('total_available_locally', 0)} |")
    lines.append(f"| Downloaded This Session | {synthesis.get('total_downloaded', 0)} |")
    sub_tracks = synthesis.get("subsurface_coverage", {}).get("total_tracks", 0)
    lines.append(f"| SHARAD Total Tracks | {sub_tracks} |")
    crism_n = synthesis.get("mineral_signatures", {}).get("crism_count", 0)
    lines.append(f"| CRISM Products | {crism_n} |")
    lines.append("")

    # ── Appendix B: Execution Log ──
    lines.append("## Appendix B: Execution Log")
    lines.append("")
    lines.append("| # | Step | Instrument | Status | Result |")
    lines.append("|---|------|------------|--------|--------|")
    for i, step in enumerate(session.steps, 1):
        inst = step.instrument or "—"
        status = step.status.value.upper()
        summary = step.result.summary if step.result else (step.error or "—")
        lines.append(f"| {i} | {step.description} | {inst} | {status} | {summary} |")
    lines.append("")

    # ── Footer ──
    lines.append("---")
    lines.append("*Report generated by MarsLab Agentic AI*")

    return "\n".join(lines)


# =============================================================================
# Endpoints
# =============================================================================

@router.get("/status")
async def agent_status():
    """Check if Ollama/Llama is available for agentic AI."""
    available = await _check_ollama()
    return {
        "ollama_available": available,
        "model": "llama3.3",
        "fallback": "rule-based planning + template narrative",
        "message": "Llama available via Ollama" if available else "Ollama not running — will use rule-based fallback",
    }


@router.get("/sessions")
async def agent_sessions():
    """List all past agent sessions (summary only)."""
    sessions = list_sessions()
    summaries = []
    for s in sessions:
        overall_score = None
        if s.synthesis and isinstance(s.synthesis, dict):
            overall_score = s.synthesis.get("overall_score")
        # Detect stale sessions: non-terminal but background task is gone
        # (e.g. server restarted mid-execution)
        effective_status = s.status
        if not s.is_terminal and (s._task is None or s._task.done()):
            effective_status = "done" if s.synthesis else "error"
        summaries.append({
            "session_id": s.session_id,
            "objective": s.objective,
            "status": effective_status,
            "region_name": s.region_name,
            "created_at": s.created_at,
            "overall_score": overall_score,
        })
    return summaries


def _sse_response(session: AgentSession, from_index: int = 0) -> StreamingResponse:
    """Create an SSE StreamingResponse that replays + follows a session's events."""

    async def event_generator():
        try:
            async for event in stream_session_events(session, from_index=from_index):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            logger.error(f"SSE stream error for {session.session_id}: {e}")
            yield f"data: {json.dumps({'event': 'error', 'data': {'error': str(e)}})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/run")
async def agent_run(request: AgentRunRequest):
    """
    Start an agent session in the background and stream SSE events.

    The agent keeps running even if the client disconnects.
    Use GET /resume/{session_id} to reconnect later.
    """
    if not request.objective.strip():
        raise HTTPException(status_code=400, detail="Objective cannot be empty")

    session = start_agent_background(request.objective, request.auto_download)
    return _sse_response(session, from_index=0)


@router.get("/resume/{session_id}")
async def agent_resume(session_id: str, from_index: int = Query(0, ge=0)):
    """
    Reconnect to a running or completed agent session.

    Replays all buffered events from *from_index*, then streams
    live events until the session finishes. This allows the frontend
    to resume watching an in-progress session after closing the panel.
    """
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    return _sse_response(session, from_index=from_index)


@router.post("/stop/{session_id}")
async def agent_stop(session_id: str):
    """Cancel a running agent session."""
    cancelled = await cancel_session(session_id)
    if not cancelled:
        raise HTTPException(status_code=400, detail="Session not found or already finished")
    return {"status": "cancelled", "session_id": session_id}


@router.get("/session/{session_id}")
async def agent_session(session_id: str):
    """Get current state of an agent session (polling fallback)."""
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    data = session.to_dict()
    # Fix stale sessions: non-terminal but task is gone (server restarted)
    if not session.is_terminal and (session._task is None or session._task.done()):
        data["status"] = "done" if session.synthesis else "error"
    return data


@router.get("/report/{session_id}")
async def agent_report(
    session_id: str,
    format: str = Query("md", pattern="^(md|pdf)$"),
):
    """
    Download a report for a completed agent session.

    Supports:
    - format=md  — Markdown report (always available)
    - format=pdf — PDF report (requires weasyprint; falls back to md)
    """
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    if session.status != "done":
        raise HTTPException(status_code=400, detail="Session has not completed yet")

    md_content = _build_report_markdown(session)
    region_slug = (session.region_name or "report").replace(" ", "_").lower()[:30]
    base_filename = f"marslab_{region_slug}_{session_id}"

    if format == "pdf":
        try:
            import markdown as md_lib
            from weasyprint import HTML

            html_body = md_lib.markdown(md_content, extensions=["tables"])
            styled_html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
  body {{ font-family: 'Helvetica Neue', Arial, sans-serif; margin: 40px; color: #1a1a2e; line-height: 1.6; font-size: 11pt; }}
  h1 {{ color: #16213e; border-bottom: 2px solid #0f3460; padding-bottom: 8px; font-size: 20pt; }}
  h2 {{ color: #0f3460; margin-top: 24px; font-size: 14pt; }}
  table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
  th, td {{ border: 1px solid #c8d6e5; padding: 6px 10px; text-align: left; font-size: 10pt; }}
  th {{ background: #0f3460; color: white; }}
  tr:nth-child(even) {{ background: #f0f4f8; }}
  strong {{ color: #0f3460; }}
  hr {{ border: none; border-top: 1px solid #c8d6e5; margin: 24px 0; }}
  em {{ color: #6b7c9c; }}
  img {{ max-width: 100%; height: auto; margin: 12px 0; border: 1px solid #c8d6e5; }}
  .figure {{ text-align: center; margin: 16px 0; }}
</style>
</head><body>{html_body}</body></html>"""

            pdf_bytes = HTML(string=styled_html).write_pdf()
            return Response(
                content=pdf_bytes,
                media_type="application/pdf",
                headers={"Content-Disposition": f'attachment; filename="{base_filename}.pdf"'},
            )
        except ImportError:
            logger.warning("weasyprint or markdown not installed, falling back to Markdown")
            # Fall through to markdown
        except Exception as e:
            logger.error(f"PDF generation failed: {e}")
            # Fall through to markdown

    # Markdown download
    return Response(
        content=md_content.encode("utf-8"),
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{base_filename}.md"'},
    )
