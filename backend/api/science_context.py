"""
Science context loader for MarsLab agentic AI system.

Loads mars_science_context.json and knowledge files from /knowledge/,
providing formatted text blocks for injection into Llama prompts
(plan generation, narrative synthesis, report generation).
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Load and cache the science context at import time
_DATA_FILE = Path(__file__).parent.parent / "data" / "mars_science_context.json"
_CONTEXT: Dict = {}

try:
    with open(_DATA_FILE) as f:
        _CONTEXT = json.load(f)
    logger.info(f"Loaded science context: {len(_CONTEXT.get('regions', {}))} regions")
except Exception as e:
    logger.warning(f"Failed to load science context: {e}")

# Load knowledge files from /knowledge/ directory
_KNOWLEDGE_DIR = Path(__file__).parent.parent.parent / "knowledge"
_KNOWLEDGE: Dict[str, str] = {}  # filename_stem -> content

if _KNOWLEDGE_DIR.is_dir():
    for md_file in sorted(_KNOWLEDGE_DIR.glob("*.md")):
        try:
            _KNOWLEDGE[md_file.stem] = md_file.read_text(encoding="utf-8")
            logger.info(f"Loaded knowledge file: {md_file.name}")
        except Exception as e:
            logger.warning(f"Failed to load knowledge file {md_file.name}: {e}")
    if _KNOWLEDGE:
        logger.info(f"Loaded {len(_KNOWLEDGE)} knowledge file(s) from {_KNOWLEDGE_DIR}")
else:
    logger.info(f"No knowledge directory at {_KNOWLEDGE_DIR}")


def get_region_context(region_id: str) -> Optional[str]:
    """
    Get formatted science context for a region.

    Args:
        region_id: Region key (e.g., 'jezero_crater', 'arcadia_planitia')

    Returns:
        Formatted text block for prompt injection, or None if not found.
    """
    regions = _CONTEXT.get("regions", {})
    region = regions.get(region_id)
    if not region:
        return None

    lines = [f"Region: {region['display_name']}"]

    if region.get("science_context"):
        lines.append(f"Description: {region['science_context']}")

    if region.get("ice_confidence"):
        lines.append(f"Ice confidence: {region['ice_confidence']}")

    if region.get("relevant_minerals"):
        lines.append(f"Known minerals: {', '.join(region['relevant_minerals'])}")

    if region.get("key_findings"):
        lines.append("Key findings:")
        for finding in region["key_findings"]:
            lines.append(f"  - {finding}")

    if region.get("landing_site_suitability"):
        lines.append(f"Landing suitability: {region['landing_site_suitability']}")

    if region.get("references"):
        lines.append(f"References: {', '.join(region['references'])}")

    return "\n".join(lines)


def get_region_context_by_name(name: str) -> Optional[str]:
    """
    Fuzzy-match a region by display name and return its context.

    Args:
        name: Region name (e.g., 'Jezero Crater', 'Arcadia Planitia')

    Returns:
        Formatted text block, or None if no match.
    """
    name_lower = name.lower().strip()
    regions = _CONTEXT.get("regions", {})

    # Try exact region_id match first
    if name_lower.replace(" ", "_") in regions:
        return get_region_context(name_lower.replace(" ", "_"))

    # Try matching display_name
    for rid, region in regions.items():
        if region.get("display_name", "").lower() == name_lower:
            return get_region_context(rid)

    # Partial match
    for rid, region in regions.items():
        display = region.get("display_name", "").lower()
        if name_lower in display or display in name_lower:
            return get_region_context(rid)

    return None


def get_instrument_context(instruments: List[str]) -> str:
    """
    Get formatted instrument interpretation context.

    Args:
        instruments: List of instrument names (e.g., ['CRISM', 'SHARAD'])

    Returns:
        Formatted text block with instrument details.
    """
    inst_data = _CONTEXT.get("instruments", {})
    lines = []

    for inst_name in instruments:
        inst = inst_data.get(inst_name)
        if not inst:
            continue

        lines.append(f"\n{inst_name} ({inst.get('full_name', '')}):")

        if inst.get("interpretation_notes"):
            lines.append(f"  {inst['interpretation_notes']}")

        if inst_name == "CRISM":
            if inst.get("ice_relevant_signatures"):
                lines.append(f"  Ice signatures: {inst['ice_relevant_signatures']}")
        elif inst_name == "SHARAD":
            if inst.get("ice_detection_method"):
                lines.append(f"  Ice detection: {inst['ice_detection_method']}")
            if inst.get("clutter_notes"):
                lines.append(f"  Clutter warning: {inst['clutter_notes']}")
        elif inst_name == "HIRISE":
            if inst.get("ice_relevant_features"):
                lines.append(f"  Ice features: {inst['ice_relevant_features']}")
        elif inst_name == "HIRISE_DTM":
            if inst.get("slope_analysis_notes"):
                lines.append(f"  Slope analysis: {inst['slope_analysis_notes']}")

    return "\n".join(lines) if lines else ""


def get_general_context(topics: Optional[List[str]] = None) -> str:
    """
    Get formatted general Mars knowledge.

    Args:
        topics: List of topic keys (e.g., ['shallow_ice', 'landing_constraints']).
                If None, returns all topics.

    Returns:
        Formatted text block.
    """
    gen = _CONTEXT.get("general_knowledge", {})
    if topics is None:
        topics = list(gen.keys())

    lines = []
    for topic in topics:
        data = gen.get(topic)
        if not data:
            continue

        header = topic.replace("_", " ").title()
        lines.append(f"\n{header}:")

        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, list):
                    lines.append(f"  {key}: {', '.join(str(v) for v in value)}")
                elif value:
                    lines.append(f"  {key}: {value}")
        elif isinstance(data, str):
            lines.append(f"  {data}")

    return "\n".join(lines) if lines else ""


def get_knowledge(name: str) -> Optional[str]:
    """
    Get a specific knowledge file by stem name.

    Args:
        name: Knowledge file stem (e.g., 'dielectric_constant_terraced_crater')

    Returns:
        Full markdown content, or None if not found.
    """
    return _KNOWLEDGE.get(name)


def get_knowledge_by_tags(tags: List[str]) -> List[str]:
    """
    Find knowledge files whose Tags section contains any of the given tags.

    Args:
        tags: List of tag strings to match (case-insensitive)

    Returns:
        List of matching knowledge file contents.
    """
    tags_lower = {t.lower().strip() for t in tags}
    results = []
    for _name, content in _KNOWLEDGE.items():
        # Extract tags from ## Tags section
        file_tags = set()
        in_tags = False
        for line in content.splitlines():
            if line.strip().lower().startswith("## tags"):
                in_tags = True
                continue
            if in_tags:
                if line.startswith("##"):
                    break
                for tag in line.split(","):
                    tag = tag.strip().lower().strip("-").strip()
                    if tag:
                        file_tags.add(tag)
        if file_tags & tags_lower:
            results.append(content)
    return results


def list_knowledge() -> List[str]:
    """Return list of available knowledge file names."""
    return list(_KNOWLEDGE.keys())


def get_context_for_agent(region_name: Optional[str] = None,
                          instruments: Optional[List[str]] = None) -> str:
    """
    Build a combined context block for agent prompts.

    Includes region context, instrument context, general knowledge,
    and relevant knowledge files matched by instrument tags.

    Args:
        region_name: Region name or ID to look up
        instruments: List of instruments being used

    Returns:
        Combined formatted text block ready for prompt injection.
    """
    parts = []

    # Region context
    if region_name:
        ctx = get_region_context_by_name(region_name)
        if ctx:
            parts.append("=== Region Science Context ===")
            parts.append(ctx)

    # Instrument context
    if instruments:
        inst_ctx = get_instrument_context(instruments)
        if inst_ctx:
            parts.append("\n=== Instrument Context ===")
            parts.append(inst_ctx)

    # General ice/landing context (always useful for agent)
    gen_ctx = get_general_context(["shallow_ice", "landing_constraints"])
    if gen_ctx:
        parts.append("\n=== General Mars Knowledge ===")
        parts.append(gen_ctx)

    # Relevant knowledge files — match by instrument tags + common analysis tags
    knowledge_tags = ["subsurface", "ice-detection", "methodology", "dielectric"]
    if instruments:
        knowledge_tags.extend(t.lower() for t in instruments)
    matched = get_knowledge_by_tags(knowledge_tags)
    for content in matched:
        parts.append("\n=== Reference Methodology ===")
        parts.append(content)

    return "\n".join(parts) if parts else ""
