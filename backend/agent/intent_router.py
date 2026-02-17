"""
Intent router: maps natural-language goals to workflow templates.

Supports Korean and English. Falls back to generic LLM planner
when no template matches.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple


def detect_language(text: str) -> str:
    """Detect if text contains Korean (Hangul) characters."""
    if re.search(r"[\uAC00-\uD7AF\u1100-\u11FF\u3130-\u318F]", text):
        return "ko"
    return "en"


def route_intent(goal: str) -> Tuple[str, Optional[str]]:
    """Route user goal to best agent template.

    Returns:
        (agent_type, region_hint)
        agent_type: "lda_sharad" | "terrace_epsilon" | "ice_search" | "generic"
        region_hint: optional extracted region name
    """
    goal_lower = goal.lower()

    # --- Terraced crater / dielectric ---
    terrace_keywords = [
        "terraced crater", "terrace", "epsilon", "유전율",
        "dielectric", "ε_r", "εr", "epsilon_r",
        "crater depth", "테라스", "유전",
    ]
    if any(kw in goal_lower for kw in terrace_keywords):
        region = _extract_region(goal)
        return "terrace_epsilon", region

    # --- LDA / subsurface interface ---
    lda_keywords = [
        "lda", "lobate debris", "debris apron",
        "subsurface interface", "서브서피스",
        "interface detection", "얼음 아래",
    ]
    if any(kw in goal_lower for kw in lda_keywords):
        region = _extract_region(goal)
        return "lda_sharad", region

    # --- Ice search ---
    ice_keywords = [
        "ice", "얼음", "crism", "mineral", "hydration",
        "h2o", "water", "물", "광물",
        "ice probability", "ice evidence",
    ]
    if any(kw in goal_lower for kw in ice_keywords):
        region = _extract_region(goal)
        return "ice_search", region

    # --- Generic (no template match) ---
    region = _extract_region(goal)
    return "generic", region


# Known Mars region names for extraction
_KNOWN_REGIONS = [
    "arcadia planitia", "utopia planitia", "amazonis planitia",
    "deuteronilus mensae", "protonilus mensae", "nilosyrtis mensae",
    "hellas planitia", "isidis planitia", "elysium planitia",
    "acidalia planitia", "chryse planitia", "syrtis major",
    "jezero", "gale crater", "valles marineris",
    "olympus mons", "tharsis", "arabia terra",
    "noachis terra", "terra cimmeria", "terra sirenum",
    "아르카디아", "유토피아", "엘리시움", "제저로",
]


def _extract_region(goal: str) -> Optional[str]:
    """Try to extract a Mars region name from the goal text."""
    goal_lower = goal.lower()
    for region in _KNOWN_REGIONS:
        if region in goal_lower:
            return region.title()
    return None
