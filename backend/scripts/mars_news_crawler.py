#!/usr/bin/env python3
import argparse
import datetime
import hashlib
import json
import logging
import os
import random
import re
import sys
from pathlib import Path
from typing import Any

import requests

BACKEND_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BACKEND_DIR / "mars_news"

from dotenv import load_dotenv
_ = load_dotenv(BACKEND_DIR / ".env")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"

NEWS_CATEGORIES = [
    "missions",
    "discoveries",
    "technology",
    "human_exploration",
    "sample_return",
    "international",
    "commercial",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mars_news_crawler")


def _date_seed(date: datetime.date) -> int:
    return int(hashlib.sha256(date.isoformat().encode()).hexdigest()[:8], 16)


def build_prompt(date: datetime.date) -> str:
    seed = _date_seed(date)
    rng = random.Random(seed)
    category_order = NEWS_CATEGORIES[:]
    rng.shuffle(category_order)

    return f"""You are producing a production-grade Mars exploration news digest for MarsLab.

Target digest date: {date.isoformat()}
Deterministic seed hint: {seed}

Collect and synthesize RECENT, HIGH-CONFIDENCE Mars exploration updates.
Focus on events and status changes from official agencies, mission teams, or credible space reporting.
If an item cannot be verified from reliable public reporting, exclude it.

You must cover these topic requirements:
1) Recent Mars mission updates (Perseverance, Curiosity, MAVEN, Mars Odyssey, etc.)
2) Mars Sample Return status
3) SpaceX Starship Mars plans
4) ESA/JAXA/CNSA Mars missions and plans
5) Ice/water discoveries and subsurface findings
6) Human Mars exploration planning
7) Mars surface/orbital technology developments

Primary category order for this run: {', '.join(category_order)}
Allowed categories: {', '.join(NEWS_CATEGORIES)}

Return ONLY valid JSON (no markdown fences, no prose before/after) with exactly this schema:
{{
  "date": "YYYY-MM-DD",
  "generated_at": "ISO-8601 timestamp",
  "categories": {{
    "missions": "brief category trend",
    "discoveries": "brief category trend",
    "technology": "brief category trend",
    "human_exploration": "brief category trend",
    "sample_return": "brief category trend",
    "international": "brief category trend",
    "commercial": "brief category trend"
  }},
  "items": [
    {{
      "title": "string",
      "date": "YYYY-MM-DD or approximate",
      "source": "publisher or agency",
      "summary": "2-4 sentences",
      "category": "one of allowed categories",
      "significance": "why this matters for Mars exploration",
      "url": "https://... or null"
    }}
  ],
  "trend_summary": "multi-sentence synthesis across all categories"
}}

Constraints:
- Include 10-20 items total.
- Ensure category values are exactly one of the allowed categories.
- Use null for unknown URLs.
- Prefer factual updates over speculation.
"""


def _extract_json_object(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].startswith("```"):
            stripped = "\n".join(lines[1:-1]).strip()
            if stripped.startswith("json"):
                stripped = stripped[4:].strip()

    match = re.search(r"\{[\s\S]*\}", stripped)
    return match.group(0) if match else stripped


def generate_news_payload(prompt: str) -> dict[str, Any] | None:
    if not GROQ_API_KEY:
        logger.error("GROQ_API_KEY not set in backend/.env or environment. Cannot generate Mars news.")
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
        content = data["choices"][0]["message"]["content"]

        parsed = json.loads(_extract_json_object(content))
        return parsed
    except requests.RequestException as e:
        logger.error("Groq API request failed: %s", e)
        return None
    except (KeyError, IndexError) as e:
        logger.error("Unexpected API response format: %s", e)
        return None
    except json.JSONDecodeError as e:
        logger.error("Failed to parse JSON from Groq response: %s", e)
        return None


def normalize_payload(date: datetime.date, payload: dict[str, Any]) -> dict[str, Any]:
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    items_obj = payload.get("items", [])
    items = items_obj if isinstance(items_obj, list) else []

    normalized_items = []
    for item in items:
        if not isinstance(item, dict):
            continue
        category = str(item.get("category", "")).strip()
        if category not in NEWS_CATEGORIES:
            category = "missions"

        normalized_items.append({
            "title": str(item.get("title", "")).strip(),
            "date": str(item.get("date", "")).strip(),
            "source": str(item.get("source", "")).strip(),
            "summary": str(item.get("summary", "")).strip(),
            "category": category,
            "significance": str(item.get("significance", "")).strip(),
            "url": item.get("url") if item.get("url") else None,
        })

    incoming_categories = payload.get("categories", {})
    category_trends = {}
    for name in NEWS_CATEGORIES:
        if isinstance(incoming_categories, dict):
            category_trends[name] = str(incoming_categories.get(name, "")).strip()
        else:
            category_trends[name] = ""

    return {
        "date": date.isoformat(),
        "generated_at": payload.get("generated_at", now_iso),
        "categories": category_trends,
        "items": normalized_items,
        "trend_summary": str(payload.get("trend_summary", "")).strip(),
    }


def build_markdown_digest(payload: dict[str, Any]) -> str:
    date = str(payload.get("date", ""))
    generated_at = str(payload.get("generated_at", ""))
    categories_obj = payload.get("categories", {})
    items_obj = payload.get("items", [])
    trend_summary = str(payload.get("trend_summary", ""))
    categories = categories_obj if isinstance(categories_obj, dict) else {}
    items = items_obj if isinstance(items_obj, list) else []

    lines = [
        f"# Mars News Digest - {date}",
        "",
        f"Generated: {generated_at}",
        "",
        "## Trend Analysis",
        trend_summary,
        "",
        "## Category Highlights",
    ]

    for category in NEWS_CATEGORIES:
        lines.append(f"- **{category}**: {categories.get(category, '')}")

    lines.append("")
    lines.append("## News Items")
    lines.append("")

    for idx, item in enumerate(items, 1):
        lines.append(f"### {idx}. {item.get('title', '')}")
        lines.append(f"- Date: {item.get('date', '')}")
        lines.append(f"- Source: {item.get('source', '')}")
        lines.append(f"- Category: {item.get('category', '')}")
        lines.append(f"- Significance: {item.get('significance', '')}")
        lines.append(f"- URL: {item.get('url') if item.get('url') else 'N/A'}")
        lines.append(f"- Summary: {item.get('summary', '')}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def save_outputs(date: datetime.date, payload: dict[str, Any]) -> tuple[Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / f"{date.isoformat()}.json"
    md_path = OUTPUT_DIR / f"{date.isoformat()}_summary.md"

    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(build_markdown_digest(payload), encoding="utf-8")
    return json_path, md_path


def main():
    parser = argparse.ArgumentParser(
        description="MarsLab Mars News Crawler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
        "--list-categories", action="store_true",
        help="List available Mars news categories.",
    )
    args = parser.parse_args()

    if args.list_categories:
        print("Available Mars news categories:\n")
        for i, category in enumerate(NEWS_CATEGORIES, 1):
            print(f"  {i}. {category}")
        return

    date = datetime.date.fromisoformat(args.date) if args.date else datetime.date.today()
    logger.info("Date: %s", date.isoformat())

    prompt = build_prompt(date)
    logger.info("Prompt length: %d chars (~%d tokens est.)", len(prompt), len(prompt) // 4)

    if args.dry_run:
        print("=" * 72)
        print("DRY RUN - Prompt that would be sent to Groq API:")
        print("=" * 72)
        print(prompt)
        print("=" * 72)
        print(f"\nPrompt length: {len(prompt)} chars (~{len(prompt) // 4} tokens est.)")
        return

    logger.info("Generating Mars news digest via Groq API (%s)...", GROQ_MODEL)
    payload = generate_news_payload(prompt)

    if payload is None:
        logger.error("Generation failed. No output produced.")
        sys.exit(1)

    normalized = normalize_payload(date, payload)
    json_path, md_path = save_outputs(date, normalized)

    logger.info("Mars news JSON saved to %s", json_path)
    logger.info("Mars news markdown summary saved to %s", md_path)

    print(f"\n✓ Mars news digest generated: {json_path}")
    print(f"  Summary: {md_path}")
    normalized_items = normalized.get("items", [])
    item_count = len(normalized_items) if isinstance(normalized_items, list) else 0
    print(f"  Items: {item_count}")


if __name__ == "__main__":
    main()
