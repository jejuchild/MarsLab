"""Field Notes – lightweight JSON-file backed API for research annotations."""

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.validation import validate_product_id

router = APIRouter(prefix="/api/fieldnotes", tags=["Field Notes"])

# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTES_FILE = os.path.join(BASE_DIR, "data", "field_notes.json")


def _load() -> list[dict]:
    if not os.path.exists(NOTES_FILE):
        return []
    with open(NOTES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _save(data: list[dict]) -> None:
    os.makedirs(os.path.dirname(NOTES_FILE), exist_ok=True)
    with open(NOTES_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class NoteCreate(BaseModel):
    product_id: str
    instrument: str
    lat: float = 0.0
    lon: float = 0.0
    tags: list[str] = []
    memo: str = ""


class NoteUpdate(BaseModel):
    tags: Optional[list[str]] = None
    memo: Optional[str] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@router.get("")
def list_notes():
    """List all field notes, newest first."""
    data = _load()
    data.sort(key=lambda n: n.get("updated_at", n.get("created_at", "")), reverse=True)
    return data


@router.post("")
def create_note(body: NoteCreate):
    """Create a new field note."""
    product_id = body.product_id.strip()
    if not product_id:
        raise HTTPException(status_code=400, detail="product_id is required")

    now = datetime.now(timezone.utc).isoformat()
    note = {
        "id": uuid.uuid4().hex[:8],
        "product_id": product_id,
        "instrument": body.instrument.strip().upper(),
        "lat": body.lat,
        "lon": body.lon,
        "tags": [t.strip() for t in body.tags if t.strip()],
        "memo": body.memo.strip(),
        "created_at": now,
        "updated_at": now,
    }

    data = _load()
    data.insert(0, note)
    _save(data)
    return note


@router.put("/{note_id}")
def update_note(note_id: str, body: NoteUpdate):
    """Update tags and/or memo of an existing note."""
    data = _load()
    for item in data:
        if item["id"] == note_id:
            if body.tags is not None:
                item["tags"] = [t.strip() for t in body.tags if t.strip()]
            if body.memo is not None:
                item["memo"] = body.memo.strip()
            item["updated_at"] = datetime.now(timezone.utc).isoformat()
            _save(data)
            return item

    raise HTTPException(status_code=404, detail="Note not found")


@router.delete("/{note_id}")
def delete_note(note_id: str):
    """Delete a field note."""
    data = _load()
    original_len = len(data)
    data = [item for item in data if item["id"] != note_id]

    if len(data) == original_len:
        raise HTTPException(status_code=404, detail="Note not found")

    _save(data)
    return {"deleted": True}


@router.get("/tags")
def list_tags():
    """Return all unique tags across all notes, sorted alphabetically."""
    data = _load()
    tag_set: set[str] = set()
    for note in data:
        for tag in note.get("tags", []):
            if tag.strip():
                tag_set.add(tag.strip())
    return sorted(tag_set)


@router.get("/product/{product_id}")
def get_notes_for_product(product_id: str):
    """Get all notes for a specific product."""
    try:
        product_id = validate_product_id(product_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid product ID format")

    data = _load()
    return [n for n in data if n["product_id"] == product_id]
