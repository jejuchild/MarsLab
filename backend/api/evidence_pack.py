"""
EvidencePack — Structured evidence container for B-level agentic reports.

Replaces the flat synthesis dict with a typed, hierarchical JSON structure.
All report generation reads from the EvidencePack; no reaching back into raw results.
"""

from __future__ import annotations

import base64
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .agent_orchestrator import AgentSession

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# EvidencePack Assembly
# ═══════════════════════════════════════════════════════════════════════

def assemble_evidence_pack(session: "AgentSession") -> Dict[str, Any]:
    """
    Reorganize session.synthesis into a structured EvidencePack dict.

    Adds Gaussian sigma and sensitivity table for εr when physics inversion
    data is available. The pack is JSON-serializable and backward-compatible.
    """
    synthesis = session.synthesis or {}
    now = datetime.now(timezone.utc).isoformat()

    # ── Region ──
    bbox_dict = {}
    if session.bbox:
        bbox_dict = {
            "min_lat": session.bbox.min_lat,
            "max_lat": session.bbox.max_lat,
            "min_lon": session.bbox.min_lon,
            "max_lon": session.bbox.max_lon,
        }
    center_lat = (bbox_dict.get("min_lat", 0) + bbox_dict.get("max_lat", 0)) / 2 if bbox_dict else 0
    center_lon = (bbox_dict.get("min_lon", 0) + bbox_dict.get("max_lon", 0)) / 2 if bbox_dict else 0

    region = {
        "name": session.region_name or "Unknown",
        "bbox": bbox_dict,
        "center_lat": round(center_lat, 4),
        "center_lon": round(center_lon, 4),
        "science_context": synthesis.get("region_science_context", ""),
    }

    # ── SHARAD Evidence ──
    sub = synthesis.get("subsurface_coverage", {})
    sharad = {
        "coverage": sub.get("coverage", "NONE"),
        "total_tracks": sub.get("total_tracks", 0),
        "analyzed_count": sub.get("analyzed_count", 0),
        "subsurface_detections": sub.get("subsurface_detections", 0),
        "reflector_summary": sub.get("reflector_summary"),
        "depth_summary": sub.get("depth_summary"),  # backward-compat alias
        "snr_distribution": sub.get("snr_distribution"),
        "clutter_rejection": sub.get("clutter_rejection"),
        "clutter_validation": sub.get("clutter_validation"),
        "reflector_density_per_km": sub.get("reflector_density_per_km"),
        "epsilon_r_source": sub.get("epsilon_r_source", "not_estimated"),
    }

    # ── Dielectric Evidence ──
    diel = synthesis.get("dielectric_analysis", {})
    terrace_diel = synthesis.get("terrace_dielectric", {})
    physics_inv = synthesis.get("sharad_physics_inversion", {})
    hyperbola = synthesis.get("hyperbola_epsilon", {})
    cross_val = synthesis.get("epsilon_cross_validation", {})
    method_hierarchy = synthesis.get("dielectric_method_hierarchy", [])

    # Determine best method
    best_method = "none"
    for entry in method_hierarchy:
        if entry.get("status") == "success":
            best_method = entry["method"]
            break

    # Build Gaussian + sensitivity from physics inversion data
    gaussian = None
    sensitivity_table = None
    # Phase 2+3: prefer quality-weighted terrace εr, then hyperbola, then raw median
    terrace_weighted_eps = (terrace_diel.get("weighted_aggregate") or {}).get("weighted_median_epsilon_r")
    # Phase 3: use cross-validation consensus if available
    consensus_eps = cross_val.get("consensus_epsilon_r")
    eps_val = (
        physics_inv.get("best_epsilon_r")
        or consensus_eps
        or terrace_weighted_eps
        or terrace_diel.get("median_epsilon_r")
        or hyperbola.get("median_epsilon_r")
        or diel.get("median_epsilon_r")
    )
    eps_ci = physics_inv.get("best_epsilon_r_ci")

    # Pull Gaussian data from physics inversion pipeline result
    eps_sigma = physics_inv.get("best_epsilon_r_sigma")
    eps_1sigma = physics_inv.get("best_epsilon_r_1sigma")
    sensitivity_raw = physics_inv.get("sensitivity")

    if eps_sigma is not None and eps_val is not None:
        gaussian = {
            "mean": eps_val,
            "sigma": eps_sigma,
            "1sigma_lo": eps_1sigma[0] if eps_1sigma else eps_val - eps_sigma,
            "1sigma_hi": eps_1sigma[1] if eps_1sigma else eps_val + eps_sigma,
            "2sigma_lo": eps_val - 2 * eps_sigma,
            "2sigma_hi": eps_val + 2 * eps_sigma,
        }

    if sensitivity_raw and eps_val is not None:
        sensitivity_table = [
            {"perturbation": "Nominal", "epsilon_r": round(eps_val, 3), "delta": None},
            {"perturbation": "Depth +10%", "epsilon_r": round(sensitivity_raw["depth_plus10"], 3),
             "delta": round(sensitivity_raw["depth_plus10"] - eps_val, 3)},
            {"perturbation": "Depth -10%", "epsilon_r": round(sensitivity_raw["depth_minus10"], 3),
             "delta": round(sensitivity_raw["depth_minus10"] - eps_val, 3)},
            {"perturbation": "TWTT +1 bin", "epsilon_r": round(sensitivity_raw["twt_plus1bin"], 3),
             "delta": round(sensitivity_raw["twt_plus1bin"] - eps_val, 3)},
            {"perturbation": "TWTT -1 bin", "epsilon_r": round(sensitivity_raw["twt_minus1bin"], 3),
             "delta": round(sensitivity_raw["twt_minus1bin"] - eps_val, 3)},
        ]
        if gaussian:
            sensitivity_table.append({
                "perturbation": "Gaussian 1σ bounds",
                "epsilon_r": f"{gaussian['1sigma_lo']:.3f} – {gaussian['1sigma_hi']:.3f}",
                "delta": None,
            })

    dielectric = {
        "method_hierarchy": method_hierarchy,
        "best_method": best_method,
        "epsilon_r": eps_val,
        "epsilon_r_ci": eps_ci,
        "epsilon_r_gaussian": gaussian,
        "sensitivity_table": sensitivity_table,
        "reflector_confidence": physics_inv.get("reflector_confidence", 0),
        "interpretation": physics_inv.get("best_interpretation") or diel.get("interpretation", ""),
        "is_fallback": synthesis.get("is_fallback", True),
        "physics_attempted": synthesis.get("physics_inversion_attempted", False),
        "assumptions": [a if isinstance(a, dict) else a.dict() if hasattr(a, "dict") else str(a)
                        for a in physics_inv.get("assumptions", [])],
        "derivation_log": [d if isinstance(d, dict) else d.dict() if hasattr(d, "dict") else str(d)
                           for d in physics_inv.get("derivation_log", [])],
        "terrace_estimates": terrace_diel.get("estimates", []),
        "terrace_median_epsilon_r": terrace_diel.get("median_epsilon_r"),
        "terrace_estimates_count": terrace_diel.get("estimates_count", 0),
        # Phase 2: quality-weighted terrace aggregate
        "terrace_weighted_aggregate": terrace_diel.get("weighted_aggregate"),
        "terrace_weighted_epsilon_r": (terrace_diel.get("weighted_aggregate") or {}).get("weighted_median_epsilon_r"),
        "terrace_confidence_interval_68": (terrace_diel.get("weighted_aggregate") or {}).get("confidence_interval_68"),
        "terrace_confidence_interval_95": (terrace_diel.get("weighted_aggregate") or {}).get("confidence_interval_95"),
        "terrace_n_good": (terrace_diel.get("weighted_aggregate") or {}).get("n_good", 0),
        "terrace_n_marginal": (terrace_diel.get("weighted_aggregate") or {}).get("n_marginal", 0),
        "physics_inversions_completed": physics_inv.get("inversions_completed", 0),
        "physics_methodology": physics_inv.get("methodology", ""),
        # Phase 3: Hyperbola εr
        "hyperbola_estimates_count": hyperbola.get("estimates_count", 0),
        "hyperbola_median_epsilon_r": hyperbola.get("median_epsilon_r"),
        "hyperbola_epsilon_r_ci95": hyperbola.get("epsilon_r_ci95"),
        "hyperbola_n_good": hyperbola.get("n_good", 0),
        "hyperbola_ice_consistent": hyperbola.get("ice_consistent_count", 0),
        "hyperbola_fits": hyperbola.get("fits", []),
        # Phase 3: Cross-validation
        "cross_validation": cross_val,
    }

    # ── CRISM Evidence ──
    mineral = synthesis.get("mineral_signatures", {})
    crism_spectral = synthesis.get("crism_spectral_analysis", {})
    crism = {
        "crism_count": mineral.get("crism_count", 0),
        "high_ice_count": mineral.get("high_ice_count", 0),
        "high_hyd_count": mineral.get("high_hyd_count", 0),
        "top_ice_candidates": mineral.get("top_ice_candidates", []),
        "ice_hotspot": mineral.get("ice_hotspot"),
        # Phase 0.3: CRISM physics analysis
        "spectral_analysis": {
            "observations_analyzed": crism_spectral.get("observations_analyzed", 0),
            "mean_physics_score": crism_spectral.get("mean_physics_score", 0),
            "max_physics_score": crism_spectral.get("max_physics_score", 0),
            "interpretation_distribution": crism_spectral.get("interpretation_distribution", {}),
            "top_physics_candidates": crism_spectral.get("top_physics_candidates", []),
            "mean_band_params": crism_spectral.get("mean_band_params", {}),
            "water_ice_overall_fraction": crism_spectral.get("water_ice_overall_fraction", 0),
            "methodology": crism_spectral.get("methodology", ""),
        } if crism_spectral else None,
    }

    # ── CNN Evidence ──
    cnn_raw = synthesis.get("cnn_mineral_classification", {})
    cnn = {
        "observations_classified": cnn_raw.get("observations_classified", 0),
        "top_minerals": cnn_raw.get("top_minerals", []),
        "ice_detected": cnn_raw.get("ice_detected", False),
        "ice_types": cnn_raw.get("ice_types", []),
        "h2o_total_pixels": cnn_raw.get("h2o_total_pixels", 0),
        "co2_total_pixels": cnn_raw.get("co2_total_pixels", 0),
        "h2o_rich_observations": cnn_raw.get("h2o_rich_observations", 0),
        "h2o_hotspot": cnn_raw.get("h2o_hotspot"),
    }

    # ── Cross-Instrument ──
    cross_raw = synthesis.get("cross_instrument", {})
    targeted = synthesis.get("targeted_ice_subsurface", {})
    cross_instrument = {
        "evidence_consistency": cross_raw.get("evidence_consistency", "insufficient_data"),
        "coincident_detections": cross_raw.get("coincident_detections", 0),
        "sharad_crism_min_distance_km": cross_raw.get("sharad_crism_min_distance_km"),
        "direct_ice_evidence": cross_raw.get("direct_ice_evidence", []),
        "indirect_ice_evidence": cross_raw.get("indirect_ice_evidence", []),
        "targeted_ice_subsurface": targeted,
        "stratigraphic_interpretation": cross_raw.get("stratigraphic_interpretation"),
        "notes": cross_raw.get("notes", []),
        "sharad_crism_geometric_intersections": cross_raw.get("sharad_crism_geometric_intersections", 0),
        "sharad_hirise_geometric_intersections": cross_raw.get("sharad_hirise_geometric_intersections", 0),
        "sharad_dtm_geometric_intersections": cross_raw.get("sharad_dtm_geometric_intersections", 0),
    }

    # ── Engineering ──
    eng_raw = synthesis.get("engineering_feasibility", {})
    engineering = {
        "safety": eng_raw.get("safety", "UNKNOWN"),
        "mean_slope": eng_raw.get("mean_slope", 0),
        "max_slope": eng_raw.get("max_slope", 0),
        "favorable_zones": eng_raw.get("favorable_zones", 0),
        "grid_size": eng_raw.get("grid_size", 0),
        "elevation_m": eng_raw.get("elevation_m", 0),
        "best_site": eng_raw.get("best_site"),
    }

    # ── ISRU Accessibility (Phase 1) ──
    isru_raw = synthesis.get("isru_assessment", {})
    isru = {
        "depth_m": isru_raw.get("depth_m"),
        "depth_uncertainty_m": isru_raw.get("depth_uncertainty_m"),
        "depth_source": isru_raw.get("depth_source", "not_available"),
        "accessibility_category": isru_raw.get("accessibility_category", "depth_unknown"),
        "accessibility_score": isru_raw.get("accessibility_score", 0.0),
        "isru_tier": isru_raw.get("isru_tier", "unknown"),
        "slope_penalty_factor": isru_raw.get("slope_penalty_factor", 1.0),
        "slope_stability": isru_raw.get("slope_stability", "unknown"),
        "epsilon_r": isru_raw.get("epsilon_r"),
        "epsilon_r_ci": isru_raw.get("epsilon_r_ci"),
        "ice_purity_estimate": isru_raw.get("ice_purity_estimate", "unknown"),
        "notes": isru_raw.get("notes", []),
    }

    # ── Climate ──
    clim_raw = synthesis.get("climate", {})
    climate = {
        "climate_score": clim_raw.get("climate_score", 0),
        "climate_summary": clim_raw.get("climate_summary", ""),
        "annual_stats": clim_raw.get("annual_stats", {}),
    }

    # ── Thermal Inertia ──
    ti_raw = synthesis.get("thermal_inertia", {})
    thermal_inertia = {
        "ti_score": ti_raw.get("ti_score", 0),
        "ti_median": ti_raw.get("ti_median"),
        "ti_mean": ti_raw.get("ti_mean"),
        "classification": ti_raw.get("classification", ""),
        "distribution_pct": ti_raw.get("distribution_pct", {}),
        "ti_explanation": ti_raw.get("ti_explanation", ""),
    }

    # ── Scoring ──
    scoring_model = synthesis.get("scoring_model", {})
    scoring = {
        "overall_score": synthesis.get("overall_score", 0),
        "score_range": synthesis.get("score_range", {}),
        "recommendation": synthesis.get("recommendation", ""),
        "recommendation_v2": synthesis.get("recommendation_v2"),
        "evidence_strength": synthesis.get("evidence_strength"),
        "scoring_model": scoring_model,
        "strengths": synthesis.get("strengths", []),
        "uncertainties": synthesis.get("uncertainties", []),
    }

    # ── Landing Site ──
    rec = synthesis.get("recommended_site", {})
    landing_site = {
        "primary_site": rec.get("primary_site") or rec.get("best_site"),
        "secondary_site": rec.get("secondary_site"),
        "science_targets": rec.get("science_targets", []),
        "trade_offs": rec.get("trade_offs", []),
    }

    # ── Assemble ──
    evidence_pack: Dict[str, Any] = {
        "version": "2.0",
        "session_id": session.session_id,
        "objective": session.objective,
        "generated_at": now,
        "region": region,
        "sharad": sharad,
        "dielectric": dielectric,
        "crism": crism,
        "cnn": cnn,
        "cross_instrument": cross_instrument,
        "engineering": engineering,
        "isru": isru,
        "climate": climate,
        "thermal_inertia": thermal_inertia,
        "scoring": scoring,
        "landing_site": landing_site,
        "instruments_searched": synthesis.get("instruments_searched", []),
        "total_products_found": synthesis.get("total_products_found", 0),
        "total_available_locally": synthesis.get("total_available_locally", 0),
        "total_downloaded": synthesis.get("total_downloaded", 0),
        "physics_pipeline_warnings": synthesis.get("physics_pipeline_warnings", []),
        "narrative": session.narrative or "",
    }

    return evidence_pack


# ═══════════════════════════════════════════════════════════════════════
# Artifact Saving
# ═══════════════════════════════════════════════════════════════════════

def save_session_artifacts(session: "AgentSession") -> str:
    """
    Save all session artifacts to disk.

    Returns the output directory path.
    Directory: backend/agent_reports/{session_id}/
    """
    reports_base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent_reports")
    session_dir = os.path.join(reports_base, session.session_id)
    figures_dir = os.path.join(session_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    # 1. EvidencePack JSON
    if session.evidence_pack:
        ep_path = os.path.join(session_dir, "evidence_pack.json")
        try:
            with open(ep_path, "w") as f:
                json.dump(session.evidence_pack, f, indent=2, default=str)
            logger.info(f"Saved evidence_pack.json to {ep_path}")
        except Exception as e:
            logger.warning(f"Failed to save evidence_pack.json: {e}")

    # 2. Report Markdown
    if session.report_draft:
        md_path = os.path.join(session_dir, "report.md")
        try:
            with open(md_path, "w") as f:
                f.write(session.report_draft)
            logger.info(f"Saved report.md to {md_path}")
        except Exception as e:
            logger.warning(f"Failed to save report.md: {e}")

    # 3. Figures (decode base64 → PNG)
    for fig in (session.figures or []):
        fig_b64 = fig.get("base64")
        fig_id = fig.get("id")
        if fig_b64 and fig_id:
            fig_path = os.path.join(figures_dir, f"{fig_id}.png")
            try:
                with open(fig_path, "wb") as f:
                    f.write(base64.b64decode(fig_b64))
            except Exception as e:
                logger.warning(f"Failed to save figure {fig_id}: {e}")

    # 4. Sensitivity table figure
    if session.evidence_pack:
        sens_table = session.evidence_pack.get("dielectric", {}).get("sensitivity_table")
        if sens_table:
            try:
                _render_sensitivity_figure(sens_table, os.path.join(figures_dir, "sensitivity_table.png"))
            except Exception as e:
                logger.warning(f"Failed to render sensitivity figure: {e}")

    # 5. Critique log
    if session.report_critique:
        critique_path = os.path.join(session_dir, "critique_log.json")
        try:
            with open(critique_path, "w") as f:
                json.dump(session.report_critique, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to save critique_log.json: {e}")

    # 6. PDF (best-effort)
    if session.report_draft:
        try:
            _render_pdf(session.report_draft, os.path.join(session_dir, "report.pdf"))
        except Exception:
            pass  # PDF is optional

    session.artifacts_dir = session_dir
    logger.info(f"All artifacts saved to {session_dir}")
    return session_dir


def _render_sensitivity_figure(sensitivity_table: List[Dict], output_path: str):
    """Render a sensitivity analysis table as a matplotlib figure."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    # Filter rows with numeric epsilon_r values
    rows = []
    for row in sensitivity_table:
        eps = row.get("epsilon_r")
        delta = row.get("delta")
        if isinstance(eps, (int, float)):
            rows.append((row["perturbation"], eps, delta))

    if not rows:
        return

    fig, ax = plt.subplots(figsize=(8, 3))
    fig.patch.set_facecolor("#0d1520")
    ax.set_facecolor("#0d1520")

    labels = [r[0] for r in rows]
    eps_vals = [r[1] for r in rows]
    nominal = rows[0][1]

    colors = []
    for r in rows:
        if r[2] is None:
            colors.append("#14b8a6")  # teal for nominal
        elif abs(r[2]) < 0.2:
            colors.append("#3b82f6")  # blue for small delta
        else:
            colors.append("#f59e0b")  # amber for large delta

    bars = ax.barh(labels, eps_vals, color=colors, height=0.5, edgecolor="#232f48")
    ax.axvline(x=nominal, color="#14b8a6", linestyle="--", linewidth=1, alpha=0.7)

    for bar, eps in zip(bars, eps_vals):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{eps:.3f}", va="center", ha="left", color="white", fontsize=9)

    ax.set_xlabel("εr", color="#92a4c9", fontsize=10)
    ax.set_title("Dielectric Constant Sensitivity Analysis", color="white", fontsize=11, fontweight="bold")
    ax.tick_params(colors="#6b7c9c", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#232f48")
    ax.spines["left"].set_color("#232f48")
    ax.invert_yaxis()

    fig.savefig(output_path, format="png", dpi=120, bbox_inches="tight",
                facecolor="#0d1520", edgecolor="none")
    plt.close(fig)
    logger.info(f"Saved sensitivity figure to {output_path}")


def _render_pdf(report_md: str, output_path: str):
    """Render markdown report as PDF. Requires weasyprint + markdown."""
    import markdown as md_lib
    from weasyprint import HTML

    html_body = md_lib.markdown(report_md, extensions=["tables"])
    styled_html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
  body {{ font-family: 'Helvetica Neue', Arial, sans-serif; margin: 40px; color: #1a1a2e; line-height: 1.6; font-size: 11pt; }}
  h1 {{ color: #16213e; border-bottom: 2px solid #0f3460; padding-bottom: 8px; font-size: 20pt; }}
  h2 {{ color: #0f3460; margin-top: 24px; font-size: 14pt; }}
  h3 {{ color: #16213e; margin-top: 16px; font-size: 12pt; }}
  table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
  th, td {{ border: 1px solid #c8d6e5; padding: 6px 10px; text-align: left; font-size: 10pt; }}
  th {{ background: #0f3460; color: white; }}
  tr:nth-child(even) {{ background: #f0f4f8; }}
  strong {{ color: #0f3460; }}
  hr {{ border: none; border-top: 1px solid #c8d6e5; margin: 24px 0; }}
  em {{ color: #6b7c9c; }}
  img {{ max-width: 100%; height: auto; margin: 12px 0; border: 1px solid #c8d6e5; }}
</style>
</head><body>{html_body}</body></html>"""

    pdf_bytes = HTML(string=styled_html).write_pdf()
    with open(output_path, "wb") as f:
        f.write(pdf_bytes)
