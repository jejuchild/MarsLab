# api/mastcam_router.py
"""
Mastcam-Z 360° Panorama API
Serves panorama metadata, thumbnails, previews, and equirectangular images
crawled from FU Berlin Jezero Crater virtual tour.
"""

import os
import logging
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse

logger = logging.getLogger("marslab.mastcam")
router = APIRouter(prefix="/api/mastcam", tags=["Mastcam-Z"])

# Data directories
MASTCAM_DIR = Path(os.environ.get(
    "MASTCAM_DATA_DIR",
    "/disk1/cspark/mastcam/downloads"
))
PREVIEW_DIR = MASTCAM_DIR / "previews"
FULL_DIR = MASTCAM_DIR / "full"

# Build scene index once at import time
_scene_cache: list[dict] | None = None


def _build_scene_index() -> list[dict]:
    """Scan preview directory to build scene list."""
    scenes = []
    if not PREVIEW_DIR.exists():
        logger.warning(f"[MASTCAM] Preview dir not found: {PREVIEW_DIR}")
        return scenes

    seen = set()
    for f in sorted(PREVIEW_DIR.iterdir()):
        if f.name.endswith("_preview.jpg"):
            name = f.name.replace("_preview.jpg", "")
            if name in seen:
                continue
            seen.add(name)

            thumb = PREVIEW_DIR / f"{name}_thumb.jpg"
            equirect = FULL_DIR / f"{name}_equirectangular.jpg"

            scenes.append({
                "id": name,
                "title": name.replace("_", " "),
                "has_thumb": thumb.exists(),
                "has_preview": f.exists(),
                "has_equirectangular": equirect.exists(),
                "equirect_size_mb": round(equirect.stat().st_size / 1024 / 1024, 1) if equirect.exists() else None,
            })

    logger.info(f"[MASTCAM] Indexed {len(scenes)} panoramas")
    return scenes


def _get_scenes() -> list[dict]:
    global _scene_cache
    if _scene_cache is None:
        _scene_cache = _build_scene_index()
    return _scene_cache


@router.get("/scenes")
def list_scenes():
    """List all available Mastcam-Z panorama scenes."""
    return JSONResponse(content=_get_scenes())


@router.get("/scenes/refresh")
def refresh_scenes():
    """Force re-scan of panorama directories."""
    global _scene_cache
    _scene_cache = None
    return JSONResponse(content={"status": "ok", "count": len(_get_scenes())})


def _safe_resolve(base: Path, filename: str) -> Path:
    """Resolve path and ensure it stays within base directory."""
    resolved = (base / filename).resolve()
    if not str(resolved).startswith(str(base.resolve())):
        raise HTTPException(400, "Invalid scene ID")
    return resolved


@router.get("/thumb/{scene_id}")
def get_thumb(scene_id: str):
    """Serve thumbnail image for a scene."""
    path = _safe_resolve(PREVIEW_DIR, f"{scene_id}_thumb.jpg")
    if not path.exists():
        raise HTTPException(404, f"Thumbnail not found: {scene_id}")
    return FileResponse(path, media_type="image/jpeg")


@router.get("/preview/{scene_id}")
def get_preview(scene_id: str):
    """Serve preview image for a scene."""
    path = _safe_resolve(PREVIEW_DIR, f"{scene_id}_preview.jpg")
    if not path.exists():
        raise HTTPException(404, f"Preview not found: {scene_id}")
    return FileResponse(path, media_type="image/jpeg")


@router.get("/panorama/{scene_id}")
def get_panorama(scene_id: str):
    """Serve equirectangular panorama image for a scene."""
    path = _safe_resolve(FULL_DIR, f"{scene_id}_equirectangular.jpg")
    if not path.exists():
        raise HTTPException(404, f"Panorama not found: {scene_id}")
    return FileResponse(path, media_type="image/jpeg")
