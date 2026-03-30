# api/mastcam_router.py
"""
Mastcam-Z 360° Panorama API
Serves panorama metadata, thumbnails, previews, and equirectangular images
crawled from FU Berlin Jezero Crater virtual tour.
"""

import os
import json
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
WEBVIEW_DIR = FULL_DIR / "webview"
METADATA_PATH = MASTCAM_DIR / "panorama_metadata.json"

_scene_cache: list[dict] | None = None
_coord_cache: dict[str, dict] | None = None


def _load_coordinates() -> dict[str, dict]:
    """Load WFS coordinate metadata, keyed by fuzzy name match."""
    global _coord_cache
    if _coord_cache is not None:
        return _coord_cache

    _coord_cache = {}
    if METADATA_PATH.exists():
        with open(METADATA_PATH) as f:
            data = json.load(f)
        for entry in data:
            # Key by lowercased name for fuzzy matching
            key = entry["name"].lower().strip()
            _coord_cache[key] = {
                "lon": entry["lon"],
                "lat": entry["lat"],
                "pano_id": entry.get("pano_id"),
            }
    logger.info(f"[MASTCAM] Loaded {len(_coord_cache)} coordinate entries")
    return _coord_cache


def _match_coordinates(scene_title: str) -> dict | None:
    """Try to match a scene title to WFS coordinate data."""
    coords = _load_coordinates()
    title_lower = scene_title.lower().replace("_", " ").replace("-", " ")

    # Try progressively shorter substrings of the title
    for key, val in coords.items():
        key_words = set(key.split())
        # Check if most words from the coordinate name appear in the title
        if len(key_words) >= 2:
            matches = sum(1 for w in key_words if w in title_lower)
            if matches >= len(key_words) * 0.6:
                return val

    return None


def _build_scene_index() -> list[dict]:
    """Scan preview directory to build scene list with coordinates."""
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
            webview = WEBVIEW_DIR / f"{name}_web.jpg"

            coord = _match_coordinates(name)

            scenes.append({
                "id": name,
                "title": name.replace("_", " "),
                "has_thumb": thumb.exists(),
                "has_preview": f.exists(),
                "has_equirectangular": equirect.exists(),
                "has_webview": webview.exists(),
                "equirect_size_mb": round(equirect.stat().st_size / 1024 / 1024, 1) if equirect.exists() else None,
                "lon": coord["lon"] if coord else None,
                "lat": coord["lat"] if coord else None,
                "pano_id": coord["pano_id"] if coord else None,
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
    path = _safe_resolve(PREVIEW_DIR, f"{scene_id}_thumb.jpg")
    if not path.exists():
        raise HTTPException(404, f"Thumbnail not found: {scene_id}")
    return FileResponse(path, media_type="image/jpeg")


@router.get("/preview/{scene_id}")
def get_preview(scene_id: str):
    path = _safe_resolve(PREVIEW_DIR, f"{scene_id}_preview.jpg")
    if not path.exists():
        raise HTTPException(404, f"Preview not found: {scene_id}")
    return FileResponse(path, media_type="image/jpeg")


@router.get("/panorama/{scene_id}")
def get_panorama(scene_id: str):
    """Serve full-res equirectangular panorama (100MB+)."""
    path = _safe_resolve(FULL_DIR, f"{scene_id}_equirectangular.jpg")
    if not path.exists():
        raise HTTPException(404, f"Panorama not found: {scene_id}")
    return FileResponse(path, media_type="image/jpeg")


@router.get("/webview/{scene_id}")
def get_webview(scene_id: str):
    """Serve optimized 4096px equirectangular for 3D viewer (~1-3MB)."""
    path = _safe_resolve(WEBVIEW_DIR, f"{scene_id}_web.jpg")
    if not path.exists():
        # Fallback to full-res if webview not generated yet
        path = _safe_resolve(FULL_DIR, f"{scene_id}_equirectangular.jpg")
        if not path.exists():
            raise HTTPException(404, f"Panorama not found: {scene_id}")
    return FileResponse(path, media_type="image/jpeg")
