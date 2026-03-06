from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw

from .models import TilePrediction

ROOT = Path(__file__).resolve().parents[3]
HEATMAP_DIR = ROOT / "backend" / "data" / "hirise_landforms" / "cache" / "heatmaps"

CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "LDA": (59, 130, 246),    # blue
    "LVF": (16, 185, 129),    # green
    "CCF": (245, 158, 11),    # amber
    "OTHER": (100, 116, 139), # slate
    "SCT": (168, 85, 247),    # purple — scalloped terrain
    "Uncertain": (55, 55, 75),
}


def generate_class_map(
    image: Image.Image,
    tile_predictions: list[TilePrediction],
    tile_size: int = 224,
) -> Image.Image:
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for tile in tile_predictions:
        x0 = int(tile.x * tile_size)
        y0 = int(tile.y * tile_size)
        x1 = min(base.size[0], x0 + tile_size)
        y1 = min(base.size[1], y0 + tile_size)
        if x0 >= base.size[0] or y0 >= base.size[1] or x1 <= x0 or y1 <= y0:
            continue

        color = CLASS_COLORS.get(tile.predicted_class, CLASS_COLORS["OTHER"])
        alpha = int(40 + 140 * tile.confidence)
        draw.rectangle((x0, y0, x1, y1), fill=(*color, alpha))

    return Image.alpha_composite(base, overlay).convert("RGB")


def save_heatmap(heatmap: Image.Image, product_id: str) -> str:
    HEATMAP_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"{product_id}_{ts}.png"
    path = HEATMAP_DIR / filename
    heatmap.save(path, format="PNG")
    return f"/cache/hirise_landforms/heatmaps/{filename}"
