#!/usr/bin/env python3
"""
MarsLab Mars Research Crawler
=============================

Fetches REAL Mars research papers from Crossref API with real DOI URLs,
then uses Groq LLM for trend analysis only.

Sources:
  - Crossref API (api.crossref.org) — real papers with DOI URLs
  - LLM (Groq) — trend analysis and relevance scoring ONLY

Usage:
  python backend/scripts/mars_research_crawler.py
  python backend/scripts/mars_research_crawler.py --date 2026-02-26
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
import time
from pathlib import Path
from typing import Any

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

CROSSREF_BASE = "https://api.crossref.org/works"
CROSSREF_HEADERS = {
    "User-Agent": "MarsLab/1.0 (mailto:marslab@research.dev)",
}
CROSSREF_TIMEOUT = 30

# Planetary-science focused topics (NOT engineering)
RESEARCH_TOPICS = [
    {
        "focus": "Mineral Classification",
        "description": "CRISM spectral analysis, phyllosilicate/sulfate/oxide identification, orbital mineralogy",
        "keywords": ["CRISM", "spectral analysis", "mineral mapping", "phyllosilicates", "sulfates"],
        "queries": [
            "Mars CRISM phyllosilicate mineral spectroscopy",
            "Mars sulfate detection orbital spectrometer",
            "Mars olivine pyroxene CRISM mapping",
        ],
    },
    {
        "focus": "Ice Detection & Geomorphology",
        "description": "SHARAD radar subsurface ice, mid-latitude glaciers, lobate debris aprons, ice table depth",
        "keywords": ["SHARAD", "subsurface ice", "radar sounding", "lobate debris aprons", "glacial"],
        "queries": [
            "Mars subsurface ice SHARAD radar sounding",
            "Mars lobate debris aprons ice glacial",
            "Mars ice table depth mid-latitude",
        ],
    },
    {
        "focus": "Arcadia Planitia",
        "description": "Landing site analysis, subsurface ice mapping, terrain characterization, ISRU potential",
        "keywords": ["Arcadia Planitia", "landing site", "ISRU", "subsurface ice", "terrain"],
        "queries": [
            "Mars Arcadia Planitia subsurface ice",
            "Mars Arcadia Planitia landing site characterization",
            "Mars Arcadia ice deposit radar",
        ],
    },
    {
        "focus": "Mars Geochemistry",
        "description": "In-situ elemental composition, Perseverance PIXL, sample analysis, rock geochemistry",
        "keywords": ["geochemistry", "PIXL", "elemental composition", "Perseverance", "Jezero"],
        "queries": [
            "Mars Perseverance PIXL elemental composition Jezero",
            "Mars in-situ geochemistry rover analysis",
            "Mars rock sample mineralogy Jezero crater",
        ],
    },
    {
        "focus": "Mars Climate & Atmosphere",
        "description": "Dust storms, atmospheric dynamics, methane detection, trace gases, climate evolution",
        "keywords": ["atmosphere", "dust storm", "methane", "climate", "trace gas"],
        "queries": [
            "Mars atmosphere dust storm climate dynamics",
            "Mars methane trace gas detection orbital",
            "Mars climate evolution atmospheric loss",
        ],
    },
    {
        "focus": "Polar Science",
        "description": "Polar layered deposits, south polar ice cap, CO2 ice, polar stratigraphy",
        "keywords": ["polar layered deposits", "south polar", "CO2 ice", "stratigraphy", "polar cap"],
        "queries": [
            "Mars polar layered deposits stratigraphy",
            "Mars south polar ice cap radar MARSIS",
            "Mars polar CO2 ice seasonal dynamics",
        ],
    },
    {
        "focus": "Water & Habitability",
        "description": "Ancient water evidence, recurring slope lineae, brines, habitability assessment",
        "keywords": ["water", "habitability", "recurring slope lineae", "brine", "ancient lake"],
        "queries": [
            "Mars recurring slope lineae water evidence",
            "Mars ancient lake habitability biosignature",
            "Mars brine liquid water subsurface",
        ],
    },
    {
        "focus": "Surface Processes",
        "description": "Aeolian processes, impact cratering, mass wasting, surface morphology",
        "keywords": ["aeolian", "dunes", "impact crater", "mass wasting", "HiRISE"],
        "queries": [
            "Mars aeolian dune migration HiRISE",
            "Mars impact crater morphology CTX",
            "Mars gully formation slope processes",
        ],
    },
]


# ======================================================================
# Cross-day deduplication
# ======================================================================

def _load_seen_papers(n_days: int = 30) -> set[str]:
    """Load DOIs and normalized titles from previous N days to avoid cross-day duplicates."""
    seen: set[str] = set()
    if not OUTPUT_DIR.is_dir():
        return seen
    for fpath in sorted(OUTPUT_DIR.glob("*.json"), reverse=True):
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", fpath.stem):
            continue
        if len(seen) > 0 and n_days <= 0:
            break
        n_days -= 1
        try:
            data = json.loads(fpath.read_text(encoding="utf-8"))
            for paper in data.get("papers", []):
                url = str(paper.get("url", ""))
                if url:
                    seen.add(url)
                norm = re.sub(r"[^a-z0-9]", "", str(paper.get("title", "")).lower())
                if norm:
                    seen.add(norm)
        except Exception:
            continue
    logger.info("Loaded %d seen paper keys from previous files", len(seen))
    return seen


# ======================================================================
# Recency scoring — newer papers get higher priority
# ======================================================================

CURRENT_YEAR = datetime.date.today().year


def _recency_score(year_str: str) -> float:
    """Score 0.0-1.0 based on publication year. Current year = 1.0, 20+ years old = 0.0."""
    try:
        year = int(year_str)
    except (ValueError, TypeError):
        return 0.0
    age = max(0, CURRENT_YEAR - year)
    # Linear decay over 20 years, clamped to [0, 1]
    return max(0.0, min(1.0, 1.0 - age / 20.0))


# ======================================================================
# Crossref API — fetch REAL papers with real DOI URLs
# ======================================================================

def fetch_crossref_papers(query: str, rows: int = 10) -> list[dict[str, Any]]:
    """Fetch papers from Crossref API for a given query, preferring recent publications."""
    params = {
        "query": query,
        "rows": rows,
        "filter": "type:journal-article",
        "select": "DOI,title,author,published-print,published-online,container-title,abstract,subject",
        "sort": "published",
        "order": "desc",
    }
    papers = []
    try:
        resp = requests.get(
            CROSSREF_BASE,
            params=params,
            headers=CROSSREF_HEADERS,
            timeout=CROSSREF_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        items = data.get("message", {}).get("items", [])
        for item in items:
            title_list = item.get("title", [])
            title = title_list[0] if title_list else ""
            if not title:
                continue

            doi = item.get("DOI", "")
            url = f"https://doi.org/{doi}" if doi else ""

            # Authors
            author_list = item.get("author", [])
            authors_parts = []
            for a in author_list:
                family = a.get("family", "")
                given = a.get("given", "")
                if family:
                    if given:
                        authors_parts.append(f"{given} {family}")
                    else:
                        authors_parts.append(family)
            authors = ", ".join(authors_parts[:6])
            if len(author_list) > 6:
                authors += " et al."

            # Year
            pub_print = item.get("published-print", {})
            pub_online = item.get("published-online", {})
            date_parts = pub_print.get("date-parts", [[None]]) or pub_online.get("date-parts", [[None]])
            year = ""
            if date_parts and date_parts[0] and date_parts[0][0]:
                year = str(date_parts[0][0])

            # Journal
            container = item.get("container-title", [])
            journal = container[0] if container else ""

            # Abstract (clean HTML)
            abstract_raw = item.get("abstract", "")
            abstract = re.sub(r"<[^>]+>", "", abstract_raw).strip()
            if len(abstract) > 500:
                abstract = abstract[:497] + "..."

            # Subjects
            subjects = item.get("subject", [])

            papers.append({
                "title": title,
                "authors": authors,
                "year": year,
                "journal": journal,
                "doi": doi,
                "url": url,
                "abstract": abstract,
                "subjects": subjects,
            })
        logger.info("Crossref query '%s': %d papers", query[:60], len(papers))
    except requests.exceptions.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 429:
            logger.warning("Crossref rate limited for query '%s', waiting 5s...", query[:40])
            time.sleep(5)
        else:
            logger.warning("Crossref HTTP error for '%s': %s", query[:40], exc)
    except Exception as exc:
        logger.warning("Crossref failed for '%s': %s", query[:40], exc)
    return papers


def fetch_papers_for_topic(topic: dict[str, Any], max_per_query: int = 8) -> list[dict[str, Any]]:
    """Fetch papers for all queries in a topic, deduplicate by DOI (within-day + cross-day)."""
    all_papers: list[dict[str, Any]] = []
    seen_keys = _load_seen_papers()

    queries = topic.get("queries", [])
    for query in queries:
        papers = fetch_crossref_papers(query, rows=max_per_query)
        for paper in papers:
            doi = paper.get("doi", "")
            url = paper.get("url", "")
            norm_title = re.sub(r"[^a-z0-9]", "", paper.get("title", "").lower())

            # Check both DOI/URL and normalized title for cross-day dedup
            if url and url in seen_keys:
                continue
            if norm_title and norm_title in seen_keys:
                continue
            if doi and doi in seen_keys:
                continue

            # Mark as seen
            if url:
                seen_keys.add(url)
            if norm_title:
                seen_keys.add(norm_title)
            if doi:
                seen_keys.add(doi)
            all_papers.append(paper)
        # Be polite to Crossref
        time.sleep(1)

    # Sort by recency — newer papers first
    all_papers.sort(key=lambda p: _recency_score(p.get("year", "")), reverse=True)

    logger.info("Topic '%s': %d unique papers from %d queries (after cross-day dedup)", topic["focus"], len(all_papers), len(queries))
    return all_papers

# ======================================================================
# LLM — trend analysis and relevance scoring ONLY
# ======================================================================

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


def analyze_with_llm(topic: dict[str, Any], papers: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Use LLM to analyze papers and provide trend analysis + relevance scoring."""
    if not GROQ_API_KEY:
        logger.warning("No GROQ_API_KEY — skipping LLM analysis")
        return None
    if not papers:
        return None

    # Build compact paper list for LLM
    papers_text = ""
    for i, p in enumerate(papers, 1):
        abstract_snippet = p.get("abstract", "")[:200]
        papers_text += (
            f"{i}. \"{p['title']}\"\n"
            f"   Authors: {p['authors']}\n"
            f"   Year: {p['year']} | Journal: {p['journal']}\n"
            f"   Abstract: {abstract_snippet}\n\n"
        )

    prompt = f"""You are analyzing real Mars research papers for MarsLab, a Mars exploration research platform.

Topic: {topic['focus']}
Description: {topic['description']}

Here are {len(papers)} real papers fetched from Crossref:

{papers_text}

For each paper (by number), provide:
1. key_findings — 1-2 sentence summary of the paper's main contribution
2. methodology — brief description of methods used
3. relevance — how this relates to MarsLab's Mars exploration goals
4. category — one of: {topic['focus']}, or a relevant subcategory

Also provide:
- trend_analysis: 3-5 sentences analyzing research trends across these papers
- connections_to_marslab: How these findings connect to MarsLab's mission (mineral classification, ice detection, Arcadia Planitia site selection, human exploration planning)

Return ONLY valid JSON:
{{
  "paper_analysis": [
    {{"item": 1, "key_findings": "...", "methodology": "...", "relevance": "...", "category": "..."}}
  ],
  "trend_analysis": "3-5 sentence analysis",
  "connections_to_marslab": "..."
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
                "max_tokens": 8192,
            },
            timeout=180,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        return json.loads(_extract_json_object(content))
    except Exception as exc:
        logger.warning("LLM analysis failed: %s", exc)
        return None


# ======================================================================
# Assembly — merge real papers with LLM analysis
# ======================================================================

def build_payload(
    date: datetime.date,
    topic: dict[str, Any],
    papers: list[dict[str, Any]],
    llm_result: dict[str, Any] | None,
) -> dict[str, Any]:
    """Merge real Crossref papers with LLM analysis."""
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # Build LLM analysis lookup
    analysis_lookup: dict[int, dict[str, str]] = {}
    if llm_result and "paper_analysis" in llm_result:
        for a in llm_result["paper_analysis"]:
            idx = a.get("item", 0)
            if isinstance(idx, int) and 1 <= idx <= len(papers):
                analysis_lookup[idx] = a

    normalized_papers = []
    for i, paper in enumerate(papers, 1):
        analysis = analysis_lookup.get(i, {})
        score = round(_recency_score(paper["year"]), 3)
        normalized_papers.append({
            "title": paper["title"],
            "authors": paper["authors"],
            "year": paper["year"],
            "journal": paper["journal"],
            "key_findings": str(analysis.get("key_findings", paper.get("abstract", "")[:200])).strip(),
            "methodology": str(analysis.get("methodology", "")).strip(),
            "relevance": str(analysis.get("relevance", "")).strip(),
            "category": str(analysis.get("category", topic["focus"])).strip(),
            "url": paper["url"],
            "recency_score": score,
        })

    # Sort by recency score descending (newest first)
    normalized_papers.sort(key=lambda p: p.get("recency_score", 0), reverse=True)
    return {
        "date": date.isoformat(),
        "generated_at": now_iso,
        "topic": {
            "focus": topic["focus"],
            "keywords": topic["keywords"],
        },
        "papers": normalized_papers,
        "trend_analysis": str((llm_result or {}).get("trend_analysis", "")).strip(),
        "connections_to_marslab": str((llm_result or {}).get("connections_to_marslab", "")).strip(),
    }


def build_summary_markdown(payload: dict[str, Any]) -> str:
    date = str(payload.get("date", ""))
    topic_obj = payload.get("topic", {})
    focus = topic_obj.get("focus", "") if isinstance(topic_obj, dict) else ""
    keywords = topic_obj.get("keywords", []) if isinstance(topic_obj, dict) else []
    papers = payload.get("papers", [])
    trend_analysis = str(payload.get("trend_analysis", ""))
    connections = str(payload.get("connections_to_marslab", ""))

    lines = [
        f"# Mars Research Digest — {date}",
        "",
        f"**Topic**: {focus}",
        f"**Keywords**: {', '.join(keywords)}",
        "",
        "## Trend Analysis",
        trend_analysis,
        "",
        "## Significant Papers",
    ]

    if not papers:
        lines.append("- No papers found.")
    else:
        for idx, paper in enumerate(papers, 1):
            url = paper.get("url", "")
            title = paper.get("title", "")
            lines.append("")
            if url:
                lines.append(f"### {idx}. [{title}]({url})")
            else:
                lines.append(f"### {idx}. {title}")
            lines.append(f"- **Authors**: {paper.get('authors', '')}")
            lines.append(f"- **Year**: {paper.get('year', '')}")
            lines.append(f"- **Journal**: {paper.get('journal', '')}")
            lines.append(f"- **Category**: {paper.get('category', '')}")
            lines.append(f"- **Key Findings**: {paper.get('key_findings', '')}")
            lines.append(f"- **Methodology**: {paper.get('methodology', '')}")
            lines.append(f"- **Relevance**: {paper.get('relevance', '')}")

    lines.append("")
    lines.append("## Connections to MarsLab")
    lines.append(connections)
    lines.append("")

    return "\n".join(lines)


def save_outputs(date: datetime.date, payload: dict[str, Any]) -> tuple[Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / f"{date.isoformat()}.json"
    md_path = OUTPUT_DIR / f"{date.isoformat()}_summary.md"

    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(build_summary_markdown(payload), encoding="utf-8")
    return json_path, md_path


# ======================================================================
# Topic selection
# ======================================================================

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
    """Select today's topic, avoiding recently covered ones."""
    previous = get_previous_topics()
    seed = int(hashlib.sha256(date.isoformat().encode()).hexdigest()[:8], 16)
    candidates = [t for t in RESEARCH_TOPICS if t["focus"] not in previous]
    if not candidates:
        candidates = RESEARCH_TOPICS
    return candidates[seed % len(candidates)]


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MarsLab Mars Research Crawler — fetches real papers from Crossref with DOI URLs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--date", type=str, default=None, help="Date label (YYYY-MM-DD). Default: today.")
    parser.add_argument("--dry-run", action="store_true", help="Fetch papers but skip LLM analysis.")
    parser.add_argument("--list-topics", action="store_true", help="List available research topics.")
    parser.add_argument("--topic", type=str, default=None, help="Override topic selection (use exact focus name).")
    parser.add_argument("--all-topics", action="store_true", help="Crawl ALL topics (for initial seeding).")
    args = parser.parse_args()

    if args.list_topics:
        print("Available research topics:\n")
        for idx, t in enumerate(RESEARCH_TOPICS, 1):
            print(f"  {idx}. {t['focus']}")
            print(f"     {t['description']}")
            print(f"     Queries: {', '.join(t['queries'][:2])}...")
            print()
        return

    date = datetime.date.fromisoformat(args.date) if args.date else datetime.date.today()
    logger.info("Date: %s", date.isoformat())

    # Determine which topics to crawl
    if args.all_topics:
        topics_to_crawl = RESEARCH_TOPICS
    elif args.topic:
        topic_value = args.topic.strip()
        matches = [t for t in RESEARCH_TOPICS if str(t["focus"]).lower() == topic_value.lower()]
        if not matches:
            logger.error("Topic '%s' not found. Use --list-topics.", topic_value)
            sys.exit(1)
        topics_to_crawl = [matches[0]]
    else:
        topics_to_crawl = [select_topic(date)]

    for topic in topics_to_crawl:
        logger.info("=" * 60)
        logger.info("Topic: %s", topic["focus"])
        logger.info("=" * 60)

        # Step 1: Fetch real papers from Crossref
        logger.info("Fetching papers from Crossref API...")
        papers = fetch_papers_for_topic(topic, max_per_query=8)

        if not papers:
            logger.warning("No papers found for topic '%s', skipping.", topic["focus"])
            continue

        logger.info("Fetched %d unique papers with real DOI URLs", len(papers))

        if args.dry_run:
            print(f"\nDRY RUN — Topic: {topic['focus']} — {len(papers)} papers:")
            print("=" * 72)
            for i, p in enumerate(papers, 1):
                print(f"  {i}. {p['title']}")
                print(f"     Authors: {p['authors']}")
                print(f"     Year: {p['year']} | Journal: {p['journal']}")
                print(f"     URL: {p['url']}")
            print()
            continue

        # Step 2: LLM analysis (trend + relevance scoring only)
        logger.info("Analyzing with Groq LLM (%s)...", GROQ_MODEL)
        llm_result = analyze_with_llm(topic, papers)

        # Step 3: Build and save
        payload = build_payload(date, topic, papers, llm_result)
        json_path, md_path = save_outputs(date, payload)

        paper_list = payload.get("papers", [])
        paper_count = len(paper_list) if isinstance(paper_list, list) else 0
        url_count = sum(1 for p in paper_list if p.get("url"))

        logger.info("Saved JSON to %s", json_path)
        logger.info("Saved markdown to %s", md_path)
        print(f"\n✓ Research digest generated: {json_path}")
        print(f"  Summary: {md_path}")
        print(f"  Papers: {paper_count} ({url_count} with DOI URLs)")


if __name__ == "__main__":
    main()
