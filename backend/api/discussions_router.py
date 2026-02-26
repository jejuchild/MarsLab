"""
Daily AI Discussion API — serves generated discussion markdown files.

Endpoints:
  GET /api/discussions          — list all discussions (newest first)
  GET /api/discussions/{date}   — get discussion content by date (YYYY-MM-DD)
"""

import logging
import os
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/discussions", tags=["Daily Discussions"])

BASE_DIR = Path(__file__).parent.parent
DISCUSSIONS_DIR = BASE_DIR / "daily_discussions"


def _parse_metadata(content: str) -> dict:
    """Extract YAML-like front matter from discussion markdown."""
    meta: dict = {}
    for line in content.splitlines()[:10]:
        if line.startswith("**Generated**:"):
            meta["generated"] = line.split(":", 1)[1].strip().rstrip("*")
        elif line.startswith("**Topic**:"):
            meta["topic"] = line.split(":", 1)[1].strip().rstrip("*")
        elif line.startswith("**Science Keywords**:"):
            meta["science_keywords"] = line.split(":", 1)[1].strip().rstrip("*")
        elif line.startswith("**MarsLab Features**:"):
            meta["marslab_features"] = line.split(":", 1)[1].strip().rstrip("*")
    return meta


@router.get("")
def list_discussions():
    """List all available discussions, newest first."""
    if not DISCUSSIONS_DIR.is_dir():
        return JSONResponse(content={"discussions": []})

    discussions = []
    for fpath in sorted(DISCUSSIONS_DIR.glob("*.md"), reverse=True):
        date = fpath.stem  # e.g. "2026-02-25"
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", date):
            continue  # skip non-date files

        content = fpath.read_text(encoding="utf-8")
        meta = _parse_metadata(content)

        # Word count (skip front matter)
        body_start = content.find("## Focus:")
        body = content[body_start:] if body_start > 0 else content
        word_count = len(body.split())

        discussions.append({
            "date": date,
            "topic": meta.get("topic", ""),
            "science_keywords": meta.get("science_keywords", ""),
            "marslab_features": meta.get("marslab_features", ""),
            "word_count": word_count,
            "size_bytes": fpath.stat().st_size,
        })

    return JSONResponse(content={"discussions": discussions})


@router.get("/{date}")
def get_discussion(date: str):
    """Get discussion content for a specific date."""
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", date):
        raise HTTPException(status_code=400, detail="Invalid date format (use YYYY-MM-DD)")

    fpath = DISCUSSIONS_DIR / f"{date}.md"
    if not fpath.is_file():
        raise HTTPException(status_code=404, detail=f"No discussion found for {date}")

    content = fpath.read_text(encoding="utf-8")
    meta = _parse_metadata(content)

    return JSONResponse(content={
        "date": date,
        "topic": meta.get("topic", ""),
        "science_keywords": meta.get("science_keywords", ""),
        "marslab_features": meta.get("marslab_features", ""),
        "content": content,
    })
