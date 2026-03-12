from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw
import numpy as np

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


# Accessibility score → color (red=inaccessible → yellow → green=excellent)
def _score_to_rgb(score: float) -> tuple[int, int, int]:
    """Map accessibility score [0,1] to RGB color.

    0.0 = dark red (inaccessible)
    0.3 = red-orange (challenging)
    0.5 = yellow (moderate/good)
    0.7 = lime (good/excellent)
    1.0 = bright green (excellent)
    """
    s = max(0.0, min(1.0, score))
    if s < 0.5:
        # Red → Yellow
        t = s / 0.5
        r = int(200 + 55 * t)  # 200→255
        g = int(30 + 195 * t)  # 30→225
        b = int(30 - 10 * t)   # 30→20
    else:
        # Yellow → Green
        t = (s - 0.5) / 0.5
        r = int(255 - 215 * t)  # 255→40
        g = int(225 + 20 * t)   # 225→245
        b = int(20 + 40 * t)    # 20→60
    return (r, g, b)


def generate_accessibility_map(
    image: Image.Image,
    tile_predictions: list[TilePrediction],
    tile_size: int = 224,
) -> Image.Image:
    """Generate accessibility heatmap overlay on browse image.

    Color scale: dark red (0) → yellow (0.5) → green (1.0)
    Alpha scales with confidence level.
    """
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

        color = _score_to_rgb(tile.accessibility_score)
        # Higher confidence → more opaque
        conf_level = {"high": 1.0, "medium": 0.8, "low": 0.6, "insufficient": 0.4}
        conf_mult = conf_level.get(tile.accessibility_confidence, 0.5)
        alpha = int((60 + 120 * conf_mult))
        draw.rectangle((x0, y0, x1, y1), fill=(*color, alpha))

    return Image.alpha_composite(base, overlay).convert("RGB")


def generate_mola_accessibility_map(
    image: Image.Image,
    scores: np.ndarray,
    alpha: int = 160,
) -> Image.Image:
    """Overlay MOLA-pixel accessibility scores on the HiRISE browse image.

    Parameters
    ----------
    image : PIL.Image
        HiRISE browse image.
    scores : (H, W) float32
        MOLA-pixel accessibility scores [0, 1].
    alpha : int
        Overlay opacity (0-255).
    """
    base = image.convert("RGBA")
    bw, bh = base.size
    gh, gw = scores.shape

    # Build RGBA overlay at MOLA resolution
    rgba = np.zeros((gh, gw, 4), dtype=np.uint8)
    for r in range(gh):
        for c in range(gw):
            s = float(scores[r, c])
            if np.isnan(s):
                continue
            rgb = _score_to_rgb(s)
            rgba[r, c] = (*rgb, alpha)

    # Upscale to browse image size (nearest for crisp MOLA pixels)
    overlay_small = Image.fromarray(rgba, "RGBA")
    overlay = overlay_small.resize((bw, bh), Image.NEAREST)

    return Image.alpha_composite(base, overlay).convert("RGB")
