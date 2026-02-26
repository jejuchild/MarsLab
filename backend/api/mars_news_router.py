import json
import logging
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/mars-news", tags=["Mars News"])

BASE_DIR = Path(__file__).parent.parent
NEWS_DIR = BASE_DIR / "mars_news"


def _valid_date(date: str) -> bool:
    return bool(re.match(r"^\d{4}-\d{2}-\d{2}$", date))


def _load_news_json(date: str) -> dict:
    fpath = NEWS_DIR / f"{date}.json"
    if not fpath.is_file():
        raise HTTPException(status_code=404, detail=f"No Mars news found for {date}")
    try:
        return json.loads(fpath.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.exception("Invalid JSON for Mars news date %s", date)
        raise HTTPException(status_code=500, detail=f"Mars news data is invalid for {date}")


def _load_summary_md(date: str) -> str:
    fpath = NEWS_DIR / f"{date}_summary.md"
    if not fpath.is_file():
        return ""
    return fpath.read_text(encoding="utf-8")


@router.get("")
def list_mars_news():
    if not NEWS_DIR.is_dir():
        return JSONResponse(content={"news": []})

    news_entries = []
    for fpath in sorted(NEWS_DIR.glob("*.json"), reverse=True):
        date = fpath.stem
        if not _valid_date(date):
            continue
        try:
            payload = json.loads(fpath.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("Skipping invalid Mars news JSON file: %s", fpath)
            continue

        categories = payload.get("categories", {})
        trend_summary = str(payload.get("trend_summary", "")).strip()
        news_entries.append({
            "date": date,
            "item_count": len(payload.get("items", [])) if isinstance(payload.get("items", []), list) else 0,
            "categories": categories if isinstance(categories, dict) else {},
            "trend_summary_preview": trend_summary[:240],
        })

    return JSONResponse(content={"news": news_entries})


@router.get("/latest")
def get_latest_mars_news():
    if not NEWS_DIR.is_dir():
        raise HTTPException(status_code=404, detail="No Mars news directory found")

    candidates = [fpath.stem for fpath in sorted(NEWS_DIR.glob("*.json"), reverse=True) if _valid_date(fpath.stem)]
    if not candidates:
        raise HTTPException(status_code=404, detail="No Mars news entries found")

    latest_date = candidates[0]
    payload = _load_news_json(latest_date)
    summary_md = _load_summary_md(latest_date)
    return JSONResponse(content={"date": latest_date, "data": payload, "summary_markdown": summary_md})


@router.get("/search")
def search_mars_news(q: str):
    if not q or len(q) < 2:
        raise HTTPException(status_code=400, detail="Query must be at least 2 characters")

    results = []
    if NEWS_DIR.is_dir():
        for fpath in sorted(NEWS_DIR.glob("*.json"), reverse=True):
            date = fpath.stem
            if not _valid_date(date):
                continue

            try:
                payload = json.loads(fpath.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue

            trend_summary = str(payload.get("trend_summary", ""))
            categories = payload.get("categories", {})
            category_text = ""
            if isinstance(categories, dict):
                category_text = " ".join([str(v) for v in categories.values()])

            matched_items = []
            for item in payload.get("items", []):
                if not isinstance(item, dict):
                    continue
                combined = " ".join([
                    str(item.get("title", "")),
                    str(item.get("summary", "")),
                    str(item.get("source", "")),
                    str(item.get("significance", "")),
                    str(item.get("category", "")),
                ])
                if q.lower() in combined.lower():
                    matched_items.append(item)

            combined_top = f"{trend_summary} {category_text}"
            if matched_items or q.lower() in combined_top.lower():
                excerpt = trend_summary.replace("\n", " ").strip()[:240]
                results.append({
                    "date": date,
                    "match_count": len(matched_items),
                    "trend_match": q.lower() in combined_top.lower(),
                    "excerpt": excerpt,
                    "items": matched_items,
                })

    return JSONResponse(content={"query": q, "results": results, "total": len(results)})


@router.get("/{date}")
def get_mars_news(date: str):
    if not _valid_date(date):
        raise HTTPException(status_code=400, detail="Invalid date format (use YYYY-MM-DD)")

    payload = _load_news_json(date)
    summary_md = _load_summary_md(date)
    return JSONResponse(content={
        "date": date,
        "data": payload,
        "summary_markdown": summary_md,
    })
