#!/usr/bin/env python3
"""
MarsLab Mars Research Crawler
=============================

Generates a daily Mars research digest using Groq.

Usage:
  python backend/scripts/mars_research_crawler.py
  python backend/scripts/mars_research_crawler.py --date 2026-02-25
  python backend/scripts/mars_research_crawler.py --dry-run
  python backend/scripts/mars_research_crawler.py --list-topics
  python backend/scripts/mars_research_crawler.py --topic "Ice Detection"
"""

import argparse
import datetime
import hashlib
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Optional

import requests
from dotenv import load_dotenv


BACKEND_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BACKEND_DIR / "mars_research"

load_dotenv(BACKEND_DIR / ".env")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mars_research_crawler")


RESEARCH_TOPICS = [
    {
        "focus": "Mineral Classification",
        "description": "CRISM spectral analysis, machine learning for mineral detection, phyllosilicate/sulfate/oxide identification",
        "keywords": ["CRISM", "spectral analysis", "mineral mapping", "machine learning", "phyllosilicates"],
    },
    {
        "focus": "Ice Detection",
        "description": "SHARAD radar subsurface ice, mid-latitude glaciers, ice table depth estimation, SWIM mapping",
        "keywords": ["SHARAD", "subsurface ice", "radar sounding", "ice table", "SWIM"],
    },
    {
        "focus": "Arcadia Planitia",
        "description": "Landing site analysis, subsurface ice mapping, terrain characterization, ISRU potential",
        "keywords": ["Arcadia Planitia", "landing site", "ISRU", "terrain", "ice deposits"],
    },
    {
        "focus": "Human Mars Exploration",
        "description": "Habitat design, life support, radiation shielding, ISRU water extraction, mission architecture",
        "keywords": ["human Mars", "habitat", "ISRU", "radiation", "life support"],
    },
    {
        "focus": "Mars Sample Return",
        "description": "Sample collection, MAV design, orbital rendezvous, Earth return, contamination prevention",
        "keywords": ["Mars Sample Return", "MSR", "Perseverance", "sample caching"],
    },
    {
        "focus": "Mars Rover Technology",
        "description": "Autonomous navigation, instrument development, power systems, drilling technology",
        "keywords": ["rover", "autonomy", "drilling", "instruments"],
    },
]


def get_previous_topics(n: int = 5) -> list[str]:
    topics: list[str] = []
    if not OUTPUT_DIR.exists():
        return topics

    files = [
        fpath
        for fpath in sorted(OUTPUT_DIR.glob("*.json"), reverse=True)
        if re.match(r"^\d{4}-\d{2}-\d{2}\.json$", fpath.name)
    ][:n]

    for fpath in files:
        try:
            payload = json.loads(fpath.read_text(encoding="utf-8"))
            focus = payload.get("topic", {}).get("focus", "")
            if isinstance(focus, str) and focus:
                topics.append(focus)
        except Exception:
            continue
    return topics


def select_topic(date: datetime.date) -> dict[str, Any]:
    previous = get_previous_topics()
    seed = int(hashlib.sha256(date.isoformat().encode()).hexdigest()[:8], 16)
    candidates = [t for t in RESEARCH_TOPICS if t["focus"] not in previous]
    if not candidates:
        candidates = RESEARCH_TOPICS
    return candidates[seed % len(candidates)]


def build_prompt(date: datetime.date, topic: dict[str, Any]) -> str:
    return f"""You are a Mars research librarian preparing a daily research digest for MarsLab.

Date: {date.isoformat()}
Topic Focus: {topic['focus']}
Topic Description: {topic['description']}
Topic Keywords: {', '.join(topic['keywords'])}

Task:
1) Provide recent and significant papers/studies relevant to this topic.
2) Prioritize credible scientific sources (journals, conference proceedings, mission teams, agency reports).
3) Keep each paper entry concise, technically precise, and useful for engineering/science planning.
4) Include both high-level trends and practical relevance to MarsLab workflows.

Return only valid JSON with this exact shape:
{{
  "papers": [
    {{
      "title": "string",
      "authors": "string",
      "year": "int or string",
      "journal": "string",
      "key_findings": "string",
      "methodology": "string",
      "relevance": "string",
      "category": "string",
      "url": "https://doi.org/... or https://arxiv.org/... or null"
    }}
  ],
  "trend_analysis": "string",
  "connections_to_marslab": "string"
}}

Rules:
- Output JSON only. Do not include markdown, comments, or code fences.
- Provide 5-10 paper entries.
- Include DOI or arXiv URLs where known. Use null if URL is uncertain.
- If exact citation details are uncertain, state uncertainty clearly inside the relevant fields.
"""


def _extract_json_object(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        return stripped[start : end + 1]
    return stripped


def _normalize_paper(paper: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": str(paper.get("title", "")).strip(),
        "authors": str(paper.get("authors", "")).strip(),
        "year": paper.get("year", ""),
        "journal": str(paper.get("journal", "")).strip(),
        "key_findings": str(paper.get("key_findings", "")).strip(),
        "methodology": str(paper.get("methodology", "")).strip(),
        "relevance": str(paper.get("relevance", "")).strip(),
        "category": str(paper.get("category", "")).strip(),
        "url": paper.get("url") if paper.get("url") else None,
    }


def generate_research(prompt: str) -> Optional[dict[str, Any]]:
    if not GROQ_API_KEY:
        logger.error("GROQ_API_KEY not set in backend/.env or environment.")
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
                "temperature": 0.5,
                "max_tokens": 8192,
            },
            timeout=180,
        )
        resp.raise_for_status()
        data = resp.json()
        raw_content = data["choices"][0]["message"]["content"]
    except requests.RequestException as exc:
        logger.error("Groq API request failed: %s", exc)
        return None
    except (KeyError, IndexError) as exc:
        logger.error("Unexpected Groq response format: %s", exc)
        return None

    try:
        parsed = json.loads(_extract_json_object(raw_content))
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse Groq JSON output: %s", exc)
        return None

    papers_raw = parsed.get("papers", [])
    papers: list[dict[str, Any]] = []
    if isinstance(papers_raw, list):
        for item in papers_raw:
            if isinstance(item, dict):
                papers.append(_normalize_paper(item))

    return {
        "papers": papers,
        "trend_analysis": str(parsed.get("trend_analysis", "")).strip(),
        "connections_to_marslab": str(parsed.get("connections_to_marslab", "")).strip(),
    }


def build_summary_markdown(date: datetime.date, topic: dict[str, Any], data: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Mars Research Digest")
    lines.append("")
    lines.append(f"**Date**: {date.isoformat()}")
    lines.append(f"**Topic**: {topic['focus']}")
    lines.append("**Keywords**: " + ", ".join(topic["keywords"]))
    lines.append("")
    lines.append("## Trend Analysis")
    lines.append(data.get("trend_analysis", ""))
    lines.append("")
    lines.append("## Significant Papers")

    papers = data.get("papers", [])
    if not papers:
        lines.append("- No papers returned.")
    else:
        for idx, paper in enumerate(papers, 1):
            lines.append("")
            lines.append(f"### {idx}. {paper.get('title', '')}")
            lines.append(f"- Authors: {paper.get('authors', '')}")
            lines.append(f"- Year: {paper.get('year', '')}")
            lines.append(f"- Journal: {paper.get('journal', '')}")
            lines.append(f"- Category: {paper.get('category', '')}")
            lines.append(f"- Key Findings: {paper.get('key_findings', '')}")
            lines.append(f"- Methodology: {paper.get('methodology', '')}")
            lines.append(f"- Relevance: {paper.get('relevance', '')}")

    lines.append("")
    lines.append("## Connections to MarsLab")
    lines.append(data.get("connections_to_marslab", ""))
    lines.append("")

    return "\n".join(lines)


def save_outputs(date: datetime.date, topic: dict[str, Any], data: dict[str, Any]) -> tuple[Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    json_path = OUTPUT_DIR / f"{date.isoformat()}.json"
    summary_path = OUTPUT_DIR / f"{date.isoformat()}_summary.md"

    payload = {
        "date": date.isoformat(),
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "topic": {
            "focus": topic["focus"],
            "keywords": topic["keywords"],
        },
        "papers": data.get("papers", []),
        "trend_analysis": data.get("trend_analysis", ""),
        "connections_to_marslab": data.get("connections_to_marslab", ""),
    }

    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_path.write_text(build_summary_markdown(date, topic, payload), encoding="utf-8")
    return json_path, summary_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MarsLab Mars research crawler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Generate for specific date (YYYY-MM-DD). Default: today.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print prompt without calling API.",
    )
    parser.add_argument(
        "--list-topics",
        action="store_true",
        help="List available research topics.",
    )
    parser.add_argument(
        "--topic",
        type=str,
        default=None,
        help="Override topic selection (use exact focus name from --list-topics).",
    )
    args = parser.parse_args()

    if args.list_topics:
        print("Available research topics:\n")
        for idx, topic_item in enumerate(RESEARCH_TOPICS, 1):
            print(f"  {idx}. {topic_item['focus']}")
            print(f"     {topic_item['description']}")
            print("     Keywords: " + ", ".join(topic_item["keywords"]))
            print()
        return

    try:
        target_date = datetime.date.fromisoformat(args.date) if args.date else datetime.date.today()
    except ValueError:
        logger.error("Invalid --date value. Use YYYY-MM-DD.")
        sys.exit(1)

    topic_value: str = ""
    if isinstance(args.topic, str):
        topic_value = args.topic

    if topic_value:
        matches = []
        for t in RESEARCH_TOPICS:
            focus_value = t.get("focus")
            if isinstance(focus_value, str) and focus_value.lower() == topic_value.lower():
                matches.append(t)
        if not matches:
            logger.error("Topic '%s' not found. Use --list-topics to inspect options.", topic_value)
            sys.exit(1)
        topic = matches[0]
    else:
        topic = select_topic(target_date)

    logger.info("Date: %s | Topic: %s", target_date.isoformat(), topic["focus"])
    prompt = build_prompt(target_date, topic)
    logger.info("Prompt length: %d chars (~%d tokens est.)", len(prompt), len(prompt) // 4)

    if args.dry_run:
        print("=" * 72)
        print("DRY RUN - Prompt that would be sent to Groq API:")
        print("=" * 72)
        print(prompt)
        print("=" * 72)
        return

    data = generate_research(prompt)
    if data is None:
        logger.error("Generation failed. No output produced.")
        sys.exit(1)

    json_path, summary_path = save_outputs(target_date, topic, data)
    logger.info("Saved JSON to %s", json_path)
    logger.info("Saved summary to %s", summary_path)

    print("\n+ Mars research digest generated")
    print(f"  JSON: {json_path}")
    print(f"  Summary: {summary_path}")
    print(f"  Papers: {len(data.get('papers', []))}")


if __name__ == "__main__":
    main()
