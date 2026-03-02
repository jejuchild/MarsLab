from __future__ import annotations

import io
import logging
from pathlib import Path
import numpy as np
import requests
from PIL import Image

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = ROOT / "backend" / "data" / "hirise_landforms"
BROWSE_CACHE_DIR = DATA_ROOT / "cache" / "browse"
LEGACY_BROWSE_DIR = ROOT / "Data" / "HiRISE" / "midlat_browse"
MOLA_FEATURE_DIM = 23
_mola_warning_logged = False


def _candidate_local_paths(product_id: str) -> list[Path]:
    lowered = product_id.lower()
    names = [
        product_id,
        product_id.upper(),
        product_id.lower(),
        f"{product_id}_browse",
        f"{product_id}_RED",
    ]
    extensions = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
    out: list[Path] = []
    for base in (BROWSE_CACHE_DIR, LEGACY_BROWSE_DIR):
        for name in names:
            for ext in extensions:
                out.append(base / f"{name}{ext}")
        if base.exists():
            for path in base.glob(f"*{lowered}*"):
                out.append(path)
            for path in base.glob(f"*{product_id.upper()}*"):
                out.append(path)
    return out


def _candidate_urls(product_id: str) -> list[str]:
    return [
        f"https://www.uahirise.org/jpeg/{product_id}",
        f"https://www.uahirise.org/images/{product_id}.jpg",
    ]


def fetch_hirise_browse(product_id: str) -> Image.Image:
    BROWSE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    for path in _candidate_local_paths(product_id):
        if path.is_file():
            with Image.open(path) as img:
                return img.convert("RGB")

    errors: list[str] = []
    for url in _candidate_urls(product_id):
        try:
            response = requests.get(url, timeout=20)
            if response.status_code != 200:
                errors.append(f"{url}: HTTP {response.status_code}")
                continue
            with Image.open(io.BytesIO(response.content)) as img:
                rgb = img.convert("RGB")
                cache_path = BROWSE_CACHE_DIR / f"{product_id}.jpg"
                rgb.save(cache_path, format="JPEG", quality=95)
                return rgb
        except Exception as exc:
            errors.append(f"{url}: {exc}")

    error_text = "; ".join(errors) if errors else "no candidate URLs available"
    raise FileNotFoundError(f"Could not fetch HiRISE browse image for '{product_id}'. {error_text}")


def tile_image(image: Image.Image, tile_size: int = 224) -> list[tuple[int, int, Image.Image]]:
    if tile_size <= 0:
        raise ValueError("tile_size must be > 0")

    rgb = image.convert("RGB")
    width, height = rgb.size
    tiles: list[tuple[int, int, Image.Image]] = []

    for y0 in range(0, height, tile_size):
        for x0 in range(0, width, tile_size):
            crop = rgb.crop((x0, y0, min(x0 + tile_size, width), min(y0 + tile_size, height)))
            if crop.size != (tile_size, tile_size):
                padded = Image.new("RGB", (tile_size, tile_size))
                padded.paste(crop, (0, 0))
                crop = padded
            tiles.append((x0 // tile_size, y0 // tile_size, crop))

    if not tiles:
        resized = rgb.resize((tile_size, tile_size), Image.Resampling.BICUBIC)
        tiles.append((0, 0, resized))

    return tiles


def extract_mola_features(lat: float, lon: float) -> np.ndarray:
    global _mola_warning_logged
    if not _mola_warning_logged:
        logger.warning("MOLA DEM not available for lat=%s lon=%s; returning zeros", lat, lon)
        _mola_warning_logged = True
    return np.zeros((MOLA_FEATURE_DIM,), dtype=np.float32)
