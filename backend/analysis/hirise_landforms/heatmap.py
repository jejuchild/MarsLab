from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from PIL import Image, ImageDraw

from .models import TileResult

ROOT = Path(__file__).resolve().parents[3]
HEATMAP_DIR = ROOT / "backend" / "data" / "hirise_landforms" / "cache" / "heatmaps"


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def generate_heatmap(image: Image.Image, tiles: list[TileResult], tile_size: int = 224) -> Image.Image:
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for tile in tiles:
        weight = _clamp01(tile.attention_weight)
        x0 = int(tile.x * tile_size)
        y0 = int(tile.y * tile_size)
        x1 = min(base.size[0], x0 + tile_size)
        y1 = min(base.size[1], y0 + tile_size)
        if x0 >= base.size[0] or y0 >= base.size[1] or x1 <= x0 or y1 <= y0:
            continue

        alpha = int(180 * weight)
        red = int(255 * weight)
        blue = int(255 * (1.0 - weight))
        draw.rectangle((x0, y0, x1, y1), fill=(red, 64, blue, alpha))

    return Image.alpha_composite(base, overlay).convert("RGB")


def save_heatmap(heatmap: Image.Image, product_id: str) -> str:
    HEATMAP_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"{product_id}_{ts}.png"
    path = HEATMAP_DIR / filename
    heatmap.save(path, format="PNG")
    return f"/cache/hirise_landforms/heatmaps/{filename}"
