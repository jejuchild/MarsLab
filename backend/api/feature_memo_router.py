"""Feature memo (developer notepad): POST text, saved under private_memos/."""
import os
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.paths import BASE_DIR

router = APIRouter()

MEMO_DIR = os.path.join(BASE_DIR, "private_memos")
os.makedirs(MEMO_DIR, exist_ok=True)


class FeatureMemo(BaseModel):
    text: str


@router.post("/api/feature_memo")
def save_feature_memo(memo: FeatureMemo):
    text = memo.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty memo")
    now = datetime.now()
    filename = now.strftime("feature_%Y%m%d_%H%M%S.txt")
    content = f"[{now.strftime('%Y-%m-%d %H:%M')}]\n{text}\n"
    with open(os.path.join(MEMO_DIR, filename), "w", encoding="utf-8") as f:
        f.write(content)
    return {"ok": True}
