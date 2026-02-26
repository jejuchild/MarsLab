#!/usr/bin/env python3
"""
MarsLab Mars News Crawler
=========================

Fetches REAL Mars news from NASA and space news RSS feeds,
then uses Groq LLM for categorization and trend analysis only.

Sources:
  - NASA Mars Exploration Program (mars.nasa.gov) — JSON API
  - NASA News Releases (nasa.gov) — RSS
  - SpaceNews (spacenews.com) — RSS

Usage:
  python backend/scripts/mars_news_crawler.py
  python backend/scripts/mars_news_crawler.py --date 2026-02-26
  python backend/scripts/mars_news_crawler.py --dry-run
  python backend/scripts/mars_news_crawler.py --list-categories
"""

import argparse
import datetime
import json
import logging
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


BACKEND_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BACKEND_DIR / "mars_news"

load_dotenv(BACKEND_DIR / ".env")

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

MARS_KEYWORDS = [
    "mars", "martian", "perseverance", "curiosity", "ingenuity",
    "jezero", "gale crater", "rover", "red planet", "msl",
    "maven", "odyssey", "mro", "reconnaissance", "sample return",
    "starship mars", "mars helicopter", "zhurong", "tianwen",
    "exomars", "rosalind franklin",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mars_news_crawler")

HTTP_HEADERS = {"User-Agent": "MarsLab/1.0 (Mars exploration research platform)"}
HTTP_TIMEOUT = 20

# Generic catch-all pages — if a redirect lands here, keep original URL
GENERIC_CATCH_ALL = {
    "https://science.nasa.gov/mars/stories",
    "https://science.nasa.gov/mars/stories/",
    "https://science.nasa.gov/mars",
    "https://science.nasa.gov/mars/",
}

# ======================================================================
# Feed fetchers — return real news items with real URLs
# ======================================================================

def _resolve_redirect(url: str) -> str:
    """Follow redirects and return the final URL. If it's a generic catch-all, keep the original."""
    try:
        resp = requests.head(url, headers=HTTP_HEADERS, timeout=10, allow_redirects=True)
        final = resp.url.rstrip("/")
        if final.rstrip("/") in {u.rstrip("/") for u in GENERIC_CATCH_ALL}:
            return url  # keep original — it at least identifies the specific article
        return resp.url
    except Exception:
        return url


def fetch_nasa_mars_api() -> list[dict[str, str]]:
    """Fetch from NASA Mars Exploration Program JSON API, resolve redirects to get real URLs."""
    url = "https://mars.nasa.gov/rss/api/?feed=news&category=all&feedtype=json"
    items = []
    try:
        resp = requests.get(url, headers=HTTP_HEADERS, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        for entry in data.get("newsitems", []):
            title = entry.get("TITLE", "").strip()
            link = entry.get("LINK", "").strip()
            desc = entry.get("DESCRIPTION", "").strip()
            posted = entry.get("POSTED", "").strip()
            if title and link:
                # Resolve NASA redirect to get the real article URL
                resolved = _resolve_redirect(link)
                items.append({
                    "title": title,
                    "url": resolved,
                    "summary": desc,
                    "date": posted,
                    "source": "NASA Mars Exploration Program",
                })
        logger.info("NASA Mars API: %d items", len(items))
    except Exception as exc:
        logger.warning("NASA Mars API failed: %s", exc)
    return items


def _parse_rss_items(xml_content: bytes, source_name: str, filter_mars: bool = True) -> list[dict[str, str]]:
    """Parse RSS XML and extract items, optionally filtering for Mars keywords."""
    items = []
    try:
        root = ET.fromstring(xml_content)
        for item_el in root.findall(".//item"):
            title = (item_el.findtext("title") or "").strip()
            link = (item_el.findtext("link") or "").strip()
            desc = (item_el.findtext("description") or "").strip()
            pubdate = (item_el.findtext("pubDate") or "").strip()

            if not title or not link:
                continue

            if filter_mars:
                combined = (title + " " + desc).lower()
                if not any(kw in combined for kw in MARS_KEYWORDS):
                    continue

            # Clean HTML from description
            desc_clean = re.sub(r"<[^>]+>", "", desc).strip()
            if len(desc_clean) > 400:
                desc_clean = desc_clean[:397] + "..."

            items.append({
                "title": title,
                "url": link,
                "summary": desc_clean,
                "date": pubdate,
                "source": source_name,
            })
    except ET.ParseError as exc:
        logger.warning("RSS parse error for %s: %s", source_name, exc)
    return items


def fetch_nasa_general_rss() -> list[dict[str, str]]:
    """Fetch NASA news releases RSS, filter for Mars content."""
    url = "https://www.nasa.gov/news-release/feed/"
    try:
        resp = requests.get(url, headers=HTTP_HEADERS, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        items = _parse_rss_items(resp.content, "NASA", filter_mars=True)
        logger.info("NASA general RSS: %d Mars items", len(items))
        return items
    except Exception as exc:
        logger.warning("NASA general RSS failed: %s", exc)
        return []


def fetch_spacenews_rss() -> list[dict[str, str]]:
    """Fetch SpaceNews RSS, filter for Mars content."""
    url = "https://spacenews.com/feed/"
    try:
        resp = requests.get(url, headers=HTTP_HEADERS, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        items = _parse_rss_items(resp.content, "SpaceNews", filter_mars=True)
        logger.info("SpaceNews RSS: %d Mars items", len(items))
        return items
    except Exception as exc:
        logger.warning("SpaceNews RSS failed: %s", exc)
        return []


def fetch_esa_mars_rss() -> list[dict[str, str]]:
    """Fetch ESA Mars Express RSS, filter for Mars content."""
    url = "https://www.esa.int/rssfeed/Our_Activities/Space_Science/Mars_Express"
    try:
        resp = requests.get(url, headers=HTTP_HEADERS, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        items = _parse_rss_items(resp.content, "ESA", filter_mars=False)  # already Mars-specific
        logger.info("ESA Mars Express RSS: %d items", len(items))
        return items
    except Exception as exc:
        logger.warning("ESA Mars RSS failed: %s", exc)
        return []


def fetch_arxiv_mars() -> list[dict[str, str]]:
    """Fetch recent Mars papers from arXiv astro-ph via Atom feed."""
    # arXiv API: search for Mars-related papers in astro-ph.EP (Earth and Planetary Astrophysics)
    url = (
        "https://export.arxiv.org/api/query"
        "?search_query=all:Mars+AND+(cat:astro-ph.EP+OR+cat:physics.geo-ph)"
        "&sortBy=submittedDate&sortOrder=descending&max_results=15"
    )
    items = []
    try:
        resp = requests.get(url, headers=HTTP_HEADERS, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        # arXiv uses Atom namespace
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(resp.content)
        for entry in root.findall("atom:entry", ns):
            title = (entry.findtext("atom:title", "", ns) or "").strip().replace("\n", " ")
            summary = (entry.findtext("atom:summary", "", ns) or "").strip().replace("\n", " ")
            published = (entry.findtext("atom:published", "", ns) or "").strip()
            # Get the abstract page link (not PDF)
            link = ""
            for link_el in entry.findall("atom:link", ns):
                if link_el.get("type") == "text/html":
                    link = link_el.get("href", "")
                    break
            if not link:
                link = (entry.findtext("atom:id", "", ns) or "").strip()
            if not title or not link:
                continue
            # Check it's actually Mars-related (not just mentions Mars in passing)
            combined = (title + " " + summary).lower()
            mars_score = sum(1 for kw in ["mars", "martian", "jezero", "gale crater",
                                          "perseverance", "curiosity", "arcadia"]
                            if kw in combined)
            if mars_score < 1:
                continue
            if len(summary) > 300:
                summary = summary[:297] + "..."
            items.append({
                "title": title,
                "url": link,
                "summary": summary,
                "date": published[:10] if published else "",
                "source": "arXiv",
            })
        logger.info("arXiv Mars: %d items", len(items))
    except Exception as exc:
        logger.warning("arXiv Mars failed: %s", exc)
    return items

def fetch_all_news() -> list[dict[str, str]]:
    """Fetch from all sources, deduplicate by title similarity."""
    all_items: list[dict[str, str]] = []
    all_items.extend(fetch_nasa_mars_api())
    all_items.extend(fetch_nasa_general_rss())
    all_items.extend(fetch_spacenews_rss())
    all_items.extend(fetch_esa_mars_rss())
    all_items.extend(fetch_arxiv_mars())

    # Deduplicate by normalized title
    seen_titles: set[str] = set()
    unique: list[dict[str, str]] = []
    for item in all_items:
        normalized = re.sub(r"[^a-z0-9]", "", item["title"].lower())
        if normalized not in seen_titles:
            seen_titles.add(normalized)
            unique.append(item)

    logger.info("Total unique news items: %d", len(unique))
    return unique

# ======================================================================
# LLM — categorization and trend analysis ONLY (not content generation)
# ======================================================================

def _extract_json_object(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].startswith("```"):
            stripped = "\n".join(lines[1:-1]).strip()
            if stripped.startswith("json"):
                stripped = stripped[4:].strip()
    match = re.search(r"\{[\s\S]*\}", stripped)
    result = match.group(0) if match else stripped
    # Fix trailing commas before ] or } (common LLM JSON error)
    result = re.sub(r",\s*]", "]", result)
    result = re.sub(r",\s*}", "}", result)
    return result

def categorize_with_llm(items: list[dict[str, str]]) -> dict[str, Any] | None:
    """Send real news items to LLM for categorization and trend analysis."""
    if not GROQ_API_KEY:
        logger.warning("No GROQ_API_KEY — skipping LLM categorization")
        return None
    if not items:
        return None

    # Build a compact summary of items for the LLM
    items_text = ""
    for i, item in enumerate(items, 1):
        items_text += f"{i}. [{item['source']}] {item['title']}\n   {item['summary'][:200]}\n\n"

    prompt = f"""You are categorizing real Mars news items for MarsLab.

Here are {len(items)} real news items from NASA and space news sources:

{items_text}

For each item (by number), assign exactly ONE category from: {', '.join(NEWS_CATEGORIES)}

Also provide:
- A brief trend for each category
- An overall trend_summary (3-5 sentences)

Return ONLY valid JSON:
{{
  "assignments": [
    {{"item": 1, "category": "missions", "significance": "brief reason"}}
  ],
  "categories": {{
    "missions": "brief trend",
    "discoveries": "brief trend",
    "technology": "brief trend",
    "human_exploration": "brief trend",
    "sample_return": "brief trend",
    "international": "brief trend",
    "commercial": "brief trend"
  }},
  "trend_summary": "3-5 sentence overall analysis"
}}"""

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
                "temperature": 0.3,
                "max_tokens": 4096,
            },
            timeout=120,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        return json.loads(_extract_json_object(content))
    except Exception as exc:
        logger.warning("LLM categorization failed: %s", exc)
        return None


# ======================================================================
# Assembly — merge real items with LLM categories
# ======================================================================

def build_payload(date: datetime.date, items: list[dict[str, str]], llm_result: dict[str, Any] | None) -> dict[str, Any]:
    """Merge real news items with LLM categorization."""
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # Build assignment lookup
    assignments: dict[int, dict[str, str]] = {}
    if llm_result and "assignments" in llm_result:
        for a in llm_result["assignments"]:
            idx = a.get("item", 0)
            if isinstance(idx, int) and 1 <= idx <= len(items):
                assignments[idx] = a

    normalized_items = []
    for i, item in enumerate(items, 1):
        assignment = assignments.get(i, {})
        category = str(assignment.get("category", "missions")).strip()
        if category not in NEWS_CATEGORIES:
            category = "missions"
        significance = str(assignment.get("significance", "")).strip()

        normalized_items.append({
            "title": item["title"],
            "date": item.get("date", ""),
            "source": item["source"],
            "summary": item["summary"],
            "category": category,
            "significance": significance,
            "url": item["url"],
        })

    # Category trends from LLM
    category_trends = {}
    llm_cats = (llm_result or {}).get("categories", {})
    for name in NEWS_CATEGORIES:
        if isinstance(llm_cats, dict):
            category_trends[name] = str(llm_cats.get(name, "")).strip()
        else:
            category_trends[name] = ""

    return {
        "date": date.isoformat(),
        "generated_at": now_iso,
        "categories": category_trends,
        "items": normalized_items,
        "trend_summary": str((llm_result or {}).get("trend_summary", "")).strip(),
    }


def build_markdown_digest(payload: dict[str, Any]) -> str:
    date = str(payload.get("date", ""))
    generated_at = str(payload.get("generated_at", ""))
    categories = payload.get("categories", {})
    items = payload.get("items", [])
    trend_summary = str(payload.get("trend_summary", ""))
    categories = categories if isinstance(categories, dict) else {}
    items = items if isinstance(items, list) else []

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
        url = item.get("url", "")
        title = item.get("title", "")
        if url:
            lines.append(f"### {idx}. [{title}]({url})")
        else:
            lines.append(f"### {idx}. {title}")
        lines.append(f"- Date: {item.get('date', '')}")
        lines.append(f"- Source: {item.get('source', '')}")
        lines.append(f"- Category: {item.get('category', '')}")
        lines.append(f"- Significance: {item.get('significance', '')}")
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
        description="MarsLab Mars News Crawler — fetches real news from NASA & space news feeds",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--date", type=str, default=None, help="Date label (YYYY-MM-DD). Default: today.")
    parser.add_argument("--dry-run", action="store_true", help="Fetch feeds but skip LLM analysis.")
    parser.add_argument("--list-categories", action="store_true", help="List available categories.")
    args = parser.parse_args()

    if args.list_categories:
        print("Available Mars news categories:\n")
        for i, category in enumerate(NEWS_CATEGORIES, 1):
            print(f"  {i}. {category}")
        return

    date = datetime.date.fromisoformat(args.date) if args.date else datetime.date.today()
    logger.info("Date: %s", date.isoformat())

    # Step 1: Fetch real news from RSS feeds
    logger.info("Fetching Mars news from RSS feeds...")
    items = fetch_all_news()

    if not items:
        logger.error("No news items fetched from any source.")
        sys.exit(1)

    logger.info("Fetched %d unique news items with real URLs", len(items))

    if args.dry_run:
        print("=" * 72)
        print(f"DRY RUN — {len(items)} items fetched:")
        print("=" * 72)
        for i, item in enumerate(items, 1):
            print(f"  {i}. [{item['source']}] {item['title']}")
            print(f"     URL: {item['url']}")
        return

    # Step 2: Use LLM for categorization + trend analysis only
    logger.info("Categorizing with Groq LLM (%s)...", GROQ_MODEL)
    llm_result = categorize_with_llm(items)

    # Step 3: Build and save
    payload = build_payload(date, items, llm_result)
    json_path, md_path = save_outputs(date, payload)

    logger.info("Mars news JSON saved to %s", json_path)
    logger.info("Mars news markdown saved to %s", md_path)

    news_items = payload.get("items", [])
    item_count = len(news_items) if isinstance(news_items, list) else 0
    urls_count = sum(1 for it in news_items if it.get("url") and not it["url"].startswith("https://www.google.com"))
    print(f"\n✓ Mars news digest generated: {json_path}")
    print(f"  Summary: {md_path}")
    print(f"  Items: {item_count} ({urls_count} with direct URLs)")


if __name__ == "__main__":
    main()
