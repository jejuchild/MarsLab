import json
import logging
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/mars-research", tags=["Mars Research"])

BASE_DIR = Path(__file__).parent.parent
RESEARCH_DIR = BASE_DIR / "mars_research"
DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _list_research_files() -> list[Path]:
    if not RESEARCH_DIR.is_dir():
        return []
    files = []
    for fpath in sorted(RESEARCH_DIR.glob("*.json"), reverse=True):
        if DATE_PATTERN.match(fpath.stem):
            files.append(fpath)
    return files


def _load_research_file(fpath: Path) -> dict[str, object]:
    try:
        return json.loads(fpath.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to read research file %s: %s", fpath, exc)
        return {}


def _topic_focus(data: dict[str, object]) -> str:
    topic = data.get("topic")
    if isinstance(topic, dict):
        focus = topic.get("focus")
        if isinstance(focus, str):
            return focus
    return ""


@router.get("")
def list_mars_research():
    entries = []
    for fpath in _list_research_files():
        data = _load_research_file(fpath)
        topic_obj = data.get("topic", {})
        topic_result = {}
        if isinstance(topic_obj, dict):
            topic_result = {
                "focus": str(topic_obj.get("focus", "")).strip(),
                "keywords": topic_obj.get("keywords", []) if isinstance(topic_obj.get("keywords"), list) else [],
            }
        papers = data.get("papers", [])
        entries.append(
            {
                "date": fpath.stem,
                "topic": topic_result,
                "paper_count": len(papers) if isinstance(papers, list) else 0,
            }
        )
    return JSONResponse(content={"research": entries})


@router.get("/latest")
def get_latest_mars_research():
    files = _list_research_files()
    if not files:
        raise HTTPException(status_code=404, detail="No Mars research data found")

    data = _load_research_file(files[0])
    if not data:
        raise HTTPException(status_code=404, detail="Latest Mars research entry is unreadable")
    summary_path = RESEARCH_DIR / f"{files[0].stem}_summary.md"
    summary_md = summary_path.read_text(encoding="utf-8") if summary_path.is_file() else ""
    data["summary_md"] = summary_md
    return JSONResponse(content=data)


@router.get("/search")
def search_mars_research(q: str):
    if not q or len(q.strip()) < 2:
        raise HTTPException(status_code=400, detail="Query must be at least 2 characters")

    query = q.strip().lower()
    results = []

    for fpath in _list_research_files():
        data = _load_research_file(fpath)
        if not data:
            continue

        papers = data.get("papers", [])
        if not isinstance(papers, list):
            continue

        total_matches = 0
        matched_papers = []

        for paper in papers:
            if not isinstance(paper, dict):
                continue
            searchable = " ".join(
                [
                    str(paper.get("title", "")),
                    str(paper.get("authors", "")),
                    str(paper.get("year", "")),
                    str(paper.get("journal", "")),
                    str(paper.get("key_findings", "")),
                    str(paper.get("methodology", "")),
                    str(paper.get("relevance", "")),
                    str(paper.get("category", "")),
                ]
            )
            match_count = searchable.lower().count(query)
            if match_count > 0:
                total_matches += match_count
                matched_papers.append(
                    {
                        "title": paper.get("title", ""),
                        "authors": paper.get("authors", ""),
                        "journal": paper.get("journal", ""),
                        "year": paper.get("year", ""),
                    }
                )

        if total_matches > 0:
            results.append(
                {
                    "date": data.get("date", fpath.stem),
                    "topic": _topic_focus(data),
                    "match_count": total_matches,
                    "papers": matched_papers,
                }
            )

    return JSONResponse(content={"query": q, "results": results, "total": len(results)})


@router.get("/topics")
def list_mars_research_topics():
    topic_counts: dict[str, int] = {}
    for fpath in _list_research_files():
        data = _load_research_file(fpath)
        focus = _topic_focus(data)
        if not isinstance(focus, str) or not focus:
            continue
        topic_counts[focus] = topic_counts.get(focus, 0) + 1

    topics = [
        {"topic": topic, "count": count}
        for topic, count in sorted(topic_counts.items(), key=lambda item: item[0].lower())
    ]
    return JSONResponse(content={"topics": topics, "total": len(topics)})


@router.get("/{date}")
def get_mars_research_by_date(date: str):
    if not DATE_PATTERN.match(date):
        raise HTTPException(status_code=400, detail="Invalid date format (use YYYY-MM-DD)")

    fpath = RESEARCH_DIR / f"{date}.json"
    if not fpath.is_file():
        raise HTTPException(status_code=404, detail=f"No Mars research found for {date}")

    data = _load_research_file(fpath)
    if not data:
        raise HTTPException(status_code=404, detail=f"Mars research file unreadable for {date}")
    summary_path = RESEARCH_DIR / f"{date}_summary.md"
    summary_md = summary_path.read_text(encoding="utf-8") if summary_path.is_file() else ""
    data["summary_md"] = summary_md
    return JSONResponse(content=data)
