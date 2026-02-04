"""Feature suggestions board – lightweight JSON-file backed API."""

import json
import os
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/feature_suggestions", tags=["Feature Suggestions"])

# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUGGESTIONS_FILE = os.path.join(BASE_DIR, "feature_suggestions.json")


def _load() -> list[dict]:
    if not os.path.exists(SUGGESTIONS_FILE):
        return []
    with open(SUGGESTIONS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _save(data: list[dict]) -> None:
    with open(SUGGESTIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
VALID_STATUSES = {"unread", "in_progress", "resolved"}


class SuggestionCreate(BaseModel):
    title: str
    description: str


class StatusUpdate(BaseModel):
    status: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@router.post("")
def create_suggestion(body: SuggestionCreate):
    title = body.title.strip()
    description = body.description.strip()
    if not title:
        raise HTTPException(status_code=400, detail="Title is required")

    suggestion = {
        "id": uuid.uuid4().hex[:8],
        "title": title,
        "description": description,
        "status": "unread",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    data = _load()
    data.insert(0, suggestion)
    _save(data)
    return suggestion


@router.get("")
def list_suggestions():
    data = _load()
    # Already stored newest-first, but ensure sort
    data.sort(key=lambda s: s.get("created_at", ""), reverse=True)
    return data


@router.patch("/{suggestion_id}/status")
def update_status(suggestion_id: str, body: StatusUpdate):
    status = body.status.strip()
    if status not in VALID_STATUSES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status. Must be one of: {', '.join(sorted(VALID_STATUSES))}",
        )

    data = _load()
    for item in data:
        if item["id"] == suggestion_id:
            item["status"] = status
            _save(data)
            return item

    raise HTTPException(status_code=404, detail="Suggestion not found")
