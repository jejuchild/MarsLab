#!/usr/bin/env python3
"""
MarsLab Daily AI Discussion Generator
======================================

Generates daily AI-driven team discussions that produce:
  1. MarsLab platform improvement suggestions
  2. Scientific research insights for Arcadia Planitia rover traverse planning

Uses Groq LLaMA API for generation with structured prompt templates.

Usage:
  python daily_discussion.py                    # Generate today's discussion
  python daily_discussion.py --date 2026-02-25  # Generate for a specific date
  python daily_discussion.py --dry-run          # Print prompt without calling API
  python daily_discussion.py --list-topics      # Show available topic focus areas

Cron setup (daily at 7am):
  0 7 * * * cd /disk1/cspark/MarsLab && python backend/scripts/daily_discussion.py

Output: backend/daily_discussions/YYYY-MM-DD.md
"""

import argparse
import datetime
import hashlib
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BACKEND_DIR = Path(__file__).parent.parent
PROJECT_ROOT = BACKEND_DIR.parent
OUTPUT_DIR = BACKEND_DIR / "daily_discussions"

# Load .env
from dotenv import load_dotenv
load_dotenv(BACKEND_DIR / ".env")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("daily_discussion")


# ---------------------------------------------------------------------------
# Topic Rotation
# ---------------------------------------------------------------------------

TOPIC_AREAS = [
    {
        "focus": "SHARAD Subsurface Analysis",
        "description": "Radar sounding interpretation, dielectric constant estimation, ice detection methodology",
        "science_keywords": ["SHARAD", "dielectric constant", "subsurface ice", "radargram", "reflectors", "permittivity"],
        "marslab_features": ["SharadHiresInspector", "Subsurface3DViewer", "RegolithPanel", "depth conversion"],
    },
    {
        "focus": "CRISM Mineral Mapping",
        "description": "Spectral analysis, CNN mineral classification, band ratio interpretation, dust contamination",
        "science_keywords": ["CRISM", "mineral classification", "spectral analysis", "band depth", "phyllosilicates", "hydration"],
        "marslab_features": ["Inspector", "BandRatioCalculator", "SpectralComparison", "CNN Mineral Classification"],
    },
    {
        "focus": "Traverse Hazard Assessment",
        "description": "Slope analysis, terrain roughness, thermal constraints, ice proximity hazards",
        "science_keywords": ["slope", "traverse planning", "rover safety", "terrain roughness", "thermal inertia"],
        "marslab_features": ["SlopeAnalysis3DTab", "HiRiseDTM3DViewer", "MeasurementTools", "DTMHoverReadout"],
    },
    {
        "focus": "Ice-Science Integration",
        "description": "Combining orbital data types to constrain ice distribution for traverse targets",
        "science_keywords": ["excess ice", "lobate debris aprons", "ice stability", "SWIM", "hydrogen"],
        "marslab_features": ["StratigraphyPanel", "AgenticPanel", "MapView overlays", "ComparisonMode"],
    },
    {
        "focus": "Data Pipeline Quality",
        "description": "Atmospheric correction, dust detection, calibration, confidence scoring methodology",
        "science_keywords": ["JCAT", "atmospheric correction", "dust opacity", "calibration", "confidence threshold"],
        "marslab_features": ["CNN confidence scores", "dust detection", "spectral pipeline", "data validation"],
    },
    {
        "focus": "Multi-Instrument Correlation",
        "description": "Cross-referencing CTX, HiRISE, CRISM, SHARAD for comprehensive site characterization",
        "science_keywords": ["multi-instrument", "CTX context", "HiRISE detail", "data fusion", "spatial correlation"],
        "marslab_features": ["LayerPanel", "TimelineNavigator", "FieldNoteModal", "ReportPanel"],
    },
    {
        "focus": "Arcadia Planitia Regional Science",
        "description": "Regional geological context, landing site constraints, science return optimization",
        "science_keywords": ["Arcadia Planitia", "mid-latitude ice", "periglacial features", "polygonal terrain", "landing site"],
        "marslab_features": ["MapView", "RegionDashboard", "DataDownloadPage", "terrain overlays"],
    },
]

CHARACTERS = [
    {
        "name": "Dr. Elena Vasquez",
        "role": "Mars Geologist / Principal Investigator",
        "personality": "Passionate about mineralogy, always connects observations to geological processes. Pushes for science return.",
    },
    {
        "name": "James Park",
        "role": "Project Lead / Systems Engineer",
        "personality": "Practical, schedule-conscious, bridges science and engineering. Asks 'how long will this take?'",
    },
    {
        "name": "Dr. Anika Rao",
        "role": "Rover Software Engineer",
        "personality": "Safety-first mindset. Knows the rover's physical limits. Thinks about autonomy constraints.",
    },
    {
        "name": "Marcus Chen",
        "role": "Ground Operations Lead",
        "personality": "MarsLab power user. Operates the tools daily. First to notice UI issues or suggest workflow improvements.",
    },
    {
        "name": "Dr. Fatima Al-Rashid",
        "role": "Spectral Data Scientist",
        "personality": "Quantitative, careful with uncertainty. CNN expert. Pushes for statistical rigor in interpretations.",
    },
    {
        "name": "Dr. Yuri Petrov",
        "role": "Planetary Geophysicist / Radar Science Lead",
        "personality": "Deep SHARAD expertise. Cites papers naturally. Thinks in permittivity and loss tangent. Collaborative, never pedantic.",
    },
]


# ---------------------------------------------------------------------------
# Context Gathering
# ---------------------------------------------------------------------------

def get_recent_git_changes(days: int = 7) -> str:
    """Get a summary of recent git changes to MarsLab."""
    try:
        since = (datetime.date.today() - datetime.timedelta(days=days)).isoformat()
        result = subprocess.run(
            ["git", "log", f"--since={since}", "--oneline", "--no-merges", "-20"],
            capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return "(no recent changes detected)"


def get_previous_topics(n: int = 5) -> list[str]:
    """Get topics from the N most recent discussions to avoid repetition."""
    topics = []
    if OUTPUT_DIR.exists():
        files = sorted(OUTPUT_DIR.glob("*.md"), reverse=True)[:n]
        for f in files:
            # Extract topic from first heading
            for line in f.read_text(encoding="utf-8").splitlines()[:5]:
                if line.startswith("## Focus:"):
                    topics.append(line.replace("## Focus:", "").strip())
                    break
    return topics


def select_topic(date: datetime.date) -> dict:
    """Deterministically select today's topic, avoiding recent repeats."""
    previous = get_previous_topics()

    # Use date hash to pick topic, skipping recently used ones
    seed = int(hashlib.sha256(date.isoformat().encode()).hexdigest()[:8], 16)
    candidates = [t for t in TOPIC_AREAS if t["focus"] not in previous]
    if not candidates:
        candidates = TOPIC_AREAS  # All used recently — reset

    return candidates[seed % len(candidates)]


# ---------------------------------------------------------------------------
# Prompt Construction
# ---------------------------------------------------------------------------

def build_prompt(date: datetime.date, topic: dict, git_context: str) -> str:
    """Build the LLM prompt for generating the discussion."""

    characters_desc = "\n".join(
        f"- **{c['name']}** ({c['role']}): {c['personality']}"
        for c in CHARACTERS
    )

    return f"""You are a creative technical writer generating a simulated team meeting discussion for the MarsLab project — a Mars scientific data visualization platform used for rover traverse planning in Arcadia Planitia.

## Date: {date.isoformat()}

## Today's Focus: {topic['focus']}
{topic['description']}

## Science Keywords: {', '.join(topic['science_keywords'])}
## MarsLab Features to Reference: {', '.join(topic['marslab_features'])}

## Characters
{characters_desc}

## Recent MarsLab Development Context
{git_context}

## Research Context
The team is planning a rover traverse in Arcadia Planitia (lat ~40-50°N, lon ~180-200°E). Key background:
- Arcadia Planitia has widespread subsurface ice (Bramson et al. 2015, GRL)
- SHARAD radar reveals ice deposits extending >100m deep
- Exposed ice scarps found in mid-latitudes (Dundas et al. 2018, Science)
- Dielectric constants of ~3.0-3.15 indicate nearly pure water ice (Petersen et al. 2018)
- The team uses MarsLab to visualize CRISM, HiRISE, CTX, SHARAD, and DTM data together

## Instructions
Write a discussion of 1500-2500 words with 3 sections:

### Section 1: Today's Analysis (~600 words)
The team works through a specific analysis task related to today's focus using MarsLab tools. They click buttons, load data, interpret results — natural workflow. Include specific data values, coordinates, and measurements.

### Section 2: Scientific Interpretation (~500 words)
The team debates what the data means. Dr. Petrov contributes radar/geophysics expertise. Reference at least one real paper. Characters disagree respectfully on interpretations.

### Section 3: Actionable Outcomes (~400 words)
End with TWO concrete lists:

**MarsLab Improvements** (2-3 specific, implementable features):
Each should reference specific components/endpoints and describe the user benefit.

**Research Insights** (2-3 findings relevant to the Arcadia traverse):
Each should be a specific observation or hypothesis that impacts traverse planning.

## Style
- Natural conversation — interruptions, follow-ups, humor
- Technical but not textbook — explain naturally when concepts arise
- Characters use MarsLab features by name (these are real UI components)
- Specific numbers: coordinates, band depths, dielectric constants, slopes
- When citing papers, mention them naturally: "Bramson's 2015 paper showed..." not "[Bramson et al., 2015]"

Write the discussion now. Start with a markdown heading: ## Focus: {topic['focus']}"""


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_discussion(prompt: str) -> Optional[str]:
    """Call Groq API to generate the discussion."""
    if not GROQ_API_KEY:
        logger.error("GROQ_API_KEY not set in environment. Cannot generate discussion.")
        return None

    try:
        resp = requests.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.85,
                "max_tokens": 4096,
                "top_p": 0.95,
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except requests.RequestException as e:
        logger.error("Groq API request failed: %s", e)
        return None
    except (KeyError, IndexError) as e:
        logger.error("Unexpected API response format: %s", e)
        return None


def save_discussion(date: datetime.date, topic: dict, content: str) -> Path:
    """Save generated discussion to markdown file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filepath = OUTPUT_DIR / f"{date.isoformat()}.md"

    header = f"""# MarsLab Daily Discussion — {date.strftime('%B %d, %Y')}

**Generated**: {datetime.datetime.now().isoformat()}
**Topic**: {topic['focus']}
**Science Keywords**: {', '.join(topic['science_keywords'])}
**MarsLab Features**: {', '.join(topic['marslab_features'])}

---

"""
    filepath.write_text(header + content, encoding="utf-8")
    return filepath


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="MarsLab Daily AI Discussion Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--date", type=str, default=None,
        help="Generate for specific date (YYYY-MM-DD). Default: today.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print prompt without calling API.",
    )
    parser.add_argument(
        "--list-topics", action="store_true",
        help="List available topic focus areas.",
    )
    parser.add_argument(
        "--topic", type=str, default=None,
        help="Override topic selection (use exact focus name from --list-topics).",
    )
    args = parser.parse_args()

    if args.list_topics:
        print("Available topic focus areas:\n")
        for i, t in enumerate(TOPIC_AREAS, 1):
            print(f"  {i}. {t['focus']}")
            print(f"     {t['description']}")
            print(f"     Keywords: {', '.join(t['science_keywords'])}")
            print(f"     Features: {', '.join(t['marslab_features'])}")
            print()
        return

    date = datetime.date.fromisoformat(args.date) if args.date else datetime.date.today()

    # Select topic
    if args.topic:
        matching = [t for t in TOPIC_AREAS if t["focus"].lower() == args.topic.lower()]
        if not matching:
            logger.error("Topic '%s' not found. Use --list-topics to see options.", args.topic)
            sys.exit(1)
        topic = matching[0]
    else:
        topic = select_topic(date)

    logger.info("Date: %s | Topic: %s", date.isoformat(), topic["focus"])

    # Gather context
    git_context = get_recent_git_changes()

    # Build prompt
    prompt = build_prompt(date, topic, git_context)

    if args.dry_run:
        print("=" * 72)
        print("DRY RUN — Prompt that would be sent to Groq API:")
        print("=" * 72)
        print(prompt)
        print("=" * 72)
        print(f"\nPrompt length: {len(prompt)} chars")
        print(f"Topic: {topic['focus']}")
        return

    # Generate
    logger.info("Generating discussion via Groq API (%s)...", GROQ_MODEL)
    content = generate_discussion(prompt)

    if content is None:
        logger.error("Generation failed. No output produced.")
        sys.exit(1)

    # Save
    filepath = save_discussion(date, topic, content)
    logger.info("Discussion saved to %s", filepath)
    logger.info("Word count: ~%d", len(content.split()))

    # Print summary
    print(f"\n✓ Daily discussion generated: {filepath}")
    print(f"  Topic: {topic['focus']}")
    print(f"  Words: ~{len(content.split())}")


if __name__ == "__main__":
    main()
