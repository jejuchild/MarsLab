"""
Report Self-Critique — Automated quality review for agentic AI reports.

Implements a review loop that checks report completeness, εr consistency,
figure coverage, and section presence. Uses Ollama/Llama first, falls back
to rule-based checklist when LLM is unavailable.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .agent_orchestrator import AgentSession

logger = logging.getLogger(__name__)

# Required section headers in the report
REQUIRED_SECTIONS = [
    ("## 1. Mission Objective", "1"),
    ("## 2. Subsurface Potential", "2"),
    ("## 2b. Dielectric Constant", "2b"),
    ("## 3. Surface", "3"),
    ("## 4. Cross-Instrument", "4"),
    ("## 5. Engineering", "5"),
    ("## 6. Climate", "6"),
    ("## 7. Assessment Score", "7"),
]


# ═══════════════════════════════════════════════════════════════════════
# Main Critique Loop
# ═══════════════════════════════════════════════════════════════════════

async def self_critique_loop(
    report_md: str,
    evidence_pack: Dict[str, Any],
    session: "AgentSession",
    max_iterations: int = 2,
    timeout_s: float = 60.0,
) -> Dict[str, Any]:
    """
    Run self-critique on the report against the evidence pack.

    Returns {final_report, iterations, critique_log}.
    """
    critique_log: List[Dict[str, Any]] = []
    current_report = report_md

    for iteration in range(max_iterations):
        logger.info(f"Self-critique iteration {iteration + 1}/{max_iterations}")

        try:
            issues = await asyncio.wait_for(
                _run_critique(current_report, evidence_pack),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError:
            logger.warning(f"Critique timed out after {timeout_s}s on iteration {iteration + 1}")
            critique_log.append({
                "iteration": iteration + 1,
                "issues_found": 0,
                "timed_out": True,
            })
            break
        except Exception as e:
            logger.warning(f"Critique failed on iteration {iteration + 1}: {e}")
            critique_log.append({
                "iteration": iteration + 1,
                "issues_found": 0,
                "error": str(e),
            })
            break

        critique_log.append({
            "iteration": iteration + 1,
            "issues_found": len(issues),
            "issues": issues,
        })

        if not issues:
            logger.info(f"No issues found on iteration {iteration + 1} — report passes")
            break

        # Patch the report
        current_report = _patch_report(current_report, issues, evidence_pack)
        logger.info(f"Patched {len(issues)} issues on iteration {iteration + 1}")

    return {
        "final_report": current_report,
        "iterations": len(critique_log),
        "critique_log": critique_log,
    }


async def _run_critique(
    report_md: str,
    evidence_pack: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Run critique using LLM first, then fall back to rule-based.
    """
    # Try Ollama/Llama first
    try:
        from .agent_orchestrator import _check_ollama
        ollama_ok = await _check_ollama()
        if ollama_ok:
            result = await _critique_with_ollama(report_md, evidence_pack)
            if result is not None:
                return result
    except Exception as e:
        logger.debug(f"Ollama critique failed: {e}")

    # Fallback: rule-based critique (always available)
    return _rule_based_critique(report_md, evidence_pack)


# ═══════════════════════════════════════════════════════════════════════
# Rule-Based Critique (Always Available)
# ═══════════════════════════════════════════════════════════════════════

def _rule_based_critique(
    report_md: str,
    evidence_pack: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Check report against structured checklist. No LLM needed.
    """
    issues: List[Dict[str, Any]] = []

    # 1. Check all required sections are present
    for header, section_id in REQUIRED_SECTIONS:
        if header not in report_md:
            issues.append({
                "section": section_id,
                "type": "missing_section",
                "fix": f"Add {header} section",
            })

    # 2. Check εr confusion: assumed described as measured
    dielectric = evidence_pack.get("dielectric", {})
    if dielectric.get("is_fallback", True):
        # εr was assumed — report should NOT use "measured" language
        # Look for misleading phrases
        lower_report = report_md.lower()
        misleading = [
            "measured εr",
            "measured dielectric",
            "independently measured",
            "physics-based" if dielectric.get("best_method") == "none" else None,
        ]
        for phrase in misleading:
            if phrase and phrase in lower_report:
                # Check it's not in a negation context
                context_check = f"not {phrase}" in lower_report or f"no {phrase}" in lower_report
                if not context_check:
                    issues.append({
                        "section": "2b",
                        "type": "epsilon_confusion",
                        "fix": f"Report uses '{phrase}' but εr was assumed (fallback). "
                               f"Replace with 'assumed εr = 3.15 (water-ice)'.",
                    })
                    break  # One issue is enough

    # 3. Check figure count
    figure_refs = len(re.findall(r"Figure \d+", report_md))
    inline_imgs = report_md.count("![")
    total_figs = max(figure_refs, inline_imgs)
    if total_figs < 3:
        issues.append({
            "section": "figures",
            "type": "insufficient_figures",
            "fix": f"Only {total_figs} figure references found, minimum 3 required "
                   f"(SHARAD radargram, CRISM ice map, slope grid).",
        })

    # 4. Check sensitivity table when εr was measured
    if not dielectric.get("is_fallback", True) and dielectric.get("epsilon_r") is not None:
        if "sensitivity" not in report_md.lower() and "Sensitivity" not in report_md:
            issues.append({
                "section": "2b",
                "type": "missing_sensitivity_table",
                "fix": "Add εr sensitivity analysis table (depth ±10%, TWTT ±1 bin).",
            })

    # 5. Check empty sections have explanation
    section_pattern = re.compile(r"^## \d+[a-z]?\. .+$", re.MULTILINE)
    section_positions = [(m.start(), m.group()) for m in section_pattern.finditer(report_md)]
    for i, (pos, header) in enumerate(section_positions):
        next_pos = section_positions[i + 1][0] if i + 1 < len(section_positions) else len(report_md)
        content = report_md[pos + len(header):next_pos].strip()
        # Check if section is essentially empty (just whitespace or very short)
        if len(content) < 20:
            section_id = re.match(r"## (\d+[a-z]?)\.", header)
            sid = section_id.group(1) if section_id else "?"
            issues.append({
                "section": sid,
                "type": "empty_section",
                "fix": f"Section '{header}' appears empty. Add data or explain why unavailable.",
            })

    # 6. Check score justification
    if "## 7. Assessment Score" in report_md or "## 9. Assessment Score" in report_md:
        score_section_start = report_md.find("## 7. Assessment Score")
        if score_section_start == -1:
            score_section_start = report_md.find("## 9. Assessment Score")
        if score_section_start >= 0:
            # Check that there's a score value and some justification
            score_content = report_md[score_section_start:score_section_start + 500]
            if "Score:" not in score_content and "score:" not in score_content.lower():
                issues.append({
                    "section": "7",
                    "type": "missing_score",
                    "fix": "Assessment score section missing the actual score value.",
                })

    return issues


# ═══════════════════════════════════════════════════════════════════════
# LLM-Based Critique (Ollama/Llama)
# ═══════════════════════════════════════════════════════════════════════

async def _critique_with_ollama(
    report_md: str,
    evidence_pack: Dict[str, Any],
) -> Optional[List[Dict[str, Any]]]:
    """
    Use Ollama/Llama to critique the report. Returns issues list or None on failure.
    """
    try:
        import httpx
    except ImportError:
        return None

    # Build a summary of the evidence pack for the prompt (not the full thing — too large)
    ep_summary = {
        "region": evidence_pack.get("region", {}).get("name"),
        "sharad_detections": evidence_pack.get("sharad", {}).get("subsurface_detections", 0),
        "epsilon_r": evidence_pack.get("dielectric", {}).get("epsilon_r"),
        "epsilon_r_method": evidence_pack.get("dielectric", {}).get("best_method"),
        "is_fallback": evidence_pack.get("dielectric", {}).get("is_fallback"),
        "has_sensitivity": evidence_pack.get("dielectric", {}).get("sensitivity_table") is not None,
        "crism_ice_count": evidence_pack.get("crism", {}).get("high_ice_count", 0),
        "cnn_ice": evidence_pack.get("cnn", {}).get("ice_detected", False),
        "overall_score": evidence_pack.get("scoring", {}).get("overall_score", 0),
    }

    prompt = f"""You are a reviewer for a Mars mission assessment report.
Given the report and evidence summary, check for these issues:

1. MISSING SECTIONS: Required sections: Mission Objective (1), Subsurface (2), Dielectric (2b), Surface (3), Cross-Instrument (4), Engineering (5), Climate (6), Assessment Score (7). List any missing.
2. EPSILON CONFUSION: If is_fallback=true, the report must NOT describe εr as "measured" or "physics-based". Check.
3. FIGURE COUNT: Report must reference at least 3 figures. Count "Figure N" references.
4. SENSITIVITY TABLE: If has_sensitivity=true and is_fallback=false, a sensitivity table must be present.
5. EMPTY SECTIONS: Any section with no meaningful content needs explanation.

Evidence summary: {json.dumps(ep_summary)}

Report (first 3000 chars):
{report_md[:3000]}

Respond with ONLY a JSON array of issues. Each issue: {{"section": "2b", "type": "issue_type", "fix": "description"}}
If no issues, respond with: []"""

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3.3",
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 500},
                },
            )
            if resp.status_code != 200:
                return None

            result_text = resp.json().get("response", "")

            # Parse JSON from response
            # Try to find JSON array in the response
            json_match = re.search(r"\[.*\]", result_text, re.DOTALL)
            if json_match:
                issues = json.loads(json_match.group())
                if isinstance(issues, list):
                    return issues
            return []

    except Exception as e:
        logger.debug(f"Ollama critique request failed: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════
# Report Patching
# ═══════════════════════════════════════════════════════════════════════

def _patch_report(
    report_md: str,
    issues: List[Dict[str, Any]],
    evidence_pack: Dict[str, Any],
) -> str:
    """
    Apply patches to the report based on critique issues.

    Each issue type maps to a specific patch function.
    """
    patched = report_md

    for issue in issues:
        issue_type = issue.get("type", "")

        if issue_type == "missing_section":
            patched = _patch_missing_section(patched, issue, evidence_pack)
        elif issue_type == "epsilon_confusion":
            patched = _patch_epsilon_confusion(patched, evidence_pack)
        elif issue_type == "missing_sensitivity_table":
            patched = _patch_sensitivity_table(patched, evidence_pack)
        elif issue_type == "empty_section":
            patched = _patch_empty_section(patched, issue)
        # insufficient_figures and missing_score are informational — hard to auto-patch

    return patched


def _patch_missing_section(
    report_md: str,
    issue: Dict[str, Any],
    evidence_pack: Dict[str, Any],
) -> str:
    """Insert a missing section with default content."""
    section_id = issue.get("section", "")

    section_templates = {
        "1": "\n## 1. Mission Objective\n\n{objective}\n",
        "2": "\n## 2. Subsurface Potential (SHARAD Radar Analysis)\n\n*No SHARAD subsurface data available for this region.*\n",
        "2b": "\n## 2b. Dielectric Constant Estimation (εr)\n\n*No dielectric inversion was performed. All depth estimates use assumed εr = 3.15 (water-ice).*\n",
        "3": "\n## 3. Surface / Near-Surface Composition (CRISM)\n\n*No CRISM mineral data available for this region.*\n",
        "4": "\n## 4. Cross-Instrument Consistency Analysis\n\n*Insufficient data for cross-instrument analysis.*\n",
        "5": "\n## 5. Engineering Feasibility (Terrain)\n\n*Slope data not available for this region.*\n",
        "6": "\n## 6. Climate + Thermal Inertia\n\n*Climate model data not computed for this region.*\n",
        "7": "\n## 7. Assessment Score + Landing Site Decision\n\n**Score: {score} / 100**\n**Recommendation:** {rec}\n",
    }

    template = section_templates.get(section_id, "")
    if not template:
        return report_md

    # Fill template
    content = template.format(
        objective=evidence_pack.get("objective", "N/A"),
        score=evidence_pack.get("scoring", {}).get("overall_score", 0),
        rec=evidence_pack.get("scoring", {}).get("recommendation", "N/A"),
    )

    # Find insertion point — before the next higher-numbered section or at end
    section_order = ["1", "2", "2b", "3", "3b", "4", "5", "6", "7"]
    try:
        idx = section_order.index(section_id)
    except ValueError:
        idx = len(section_order)

    # Find the next existing section after this one
    insert_pos = len(report_md)
    for next_id in section_order[idx + 1:]:
        for header, sid in REQUIRED_SECTIONS:
            if sid == next_id and header in report_md:
                insert_pos = report_md.index(header)
                break
        if insert_pos < len(report_md):
            break

    # Also check appendices
    for marker in ["## Appendix", "---\n*Report generated"]:
        pos = report_md.find(marker)
        if 0 < pos < insert_pos:
            insert_pos = pos
            break

    return report_md[:insert_pos] + content + "\n" + report_md[insert_pos:]


def _patch_epsilon_confusion(report_md: str, evidence_pack: Dict[str, Any]) -> str:
    """Replace misleading 'measured' language when εr is actually assumed."""
    patched = report_md
    replacements = [
        ("measured εr", "assumed εr"),
        ("measured dielectric", "assumed dielectric"),
        ("independently measured", "assumed (water-ice fallback)"),
    ]
    for old, new in replacements:
        # Case-insensitive replace
        pattern = re.compile(re.escape(old), re.IGNORECASE)
        patched = pattern.sub(new, patched)
    return patched


def _patch_sensitivity_table(report_md: str, evidence_pack: Dict[str, Any]) -> str:
    """Insert sensitivity table after the εr section."""
    sens_table = evidence_pack.get("dielectric", {}).get("sensitivity_table")
    if not sens_table:
        return report_md

    # Build markdown table
    lines = [
        "",
        "#### Uncertainty & Sensitivity Analysis",
        "",
        "| Perturbation | εr | Δεr |",
        "|---|---|---|",
    ]
    for row in sens_table:
        eps = row.get("epsilon_r", "—")
        delta = row.get("delta")
        delta_str = f"{delta:+.3f}" if isinstance(delta, (int, float)) else "—"
        eps_str = f"{eps:.3f}" if isinstance(eps, (int, float)) else str(eps)
        lines.append(f"| {row['perturbation']} | {eps_str} | {delta_str} |")
    lines.append("")

    gaussian = evidence_pack.get("dielectric", {}).get("epsilon_r_gaussian")
    if gaussian:
        lines.append(f"*Gaussian 1σ uncertainty: εr = {gaussian['mean']:.3f} ± {gaussian['sigma']:.3f}*")
        lines.append("")

    table_content = "\n".join(lines)

    # Insert after "## 2b. Dielectric" section, before next section
    marker = "## 2b. Dielectric Constant"
    pos = report_md.find(marker)
    if pos < 0:
        return report_md

    # Find next ## section
    next_section = report_md.find("\n## ", pos + len(marker))
    if next_section < 0:
        next_section = len(report_md)

    return report_md[:next_section] + table_content + "\n" + report_md[next_section:]


def _patch_empty_section(report_md: str, issue: Dict[str, Any]) -> str:
    """Add placeholder text to empty sections."""
    section_id = issue.get("section", "")
    placeholder = f"\n*Data not available for this analysis. This section is intentionally empty.*\n"

    # Find the section header
    pattern = re.compile(rf"^(## {re.escape(section_id)}\. .+)$", re.MULTILINE)
    match = pattern.search(report_md)
    if not match:
        return report_md

    header_end = match.end()
    # Find next section
    next_section = report_md.find("\n## ", header_end)
    if next_section < 0:
        next_section = len(report_md)

    content = report_md[header_end:next_section].strip()
    if len(content) < 20:
        return report_md[:header_end] + "\n" + placeholder + "\n" + report_md[next_section:]

    return report_md
