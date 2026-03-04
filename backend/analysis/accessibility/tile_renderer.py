"""Tile renderer for the accessibility heatmap layer.

Colourmap: dark-red (0) → orange → yellow → light-green → dark-green (1).
Transparent where no data.
"""

import io
from typing import Optional

import numpy as np

# Accessibility colour stops (value, RGBA uint8)
_ACCESSIBILITY_STOPS = [
    (0.00, np.array([183,  28,  28, 200], dtype=np.uint8)),  # dark red
    (0.25, np.array([239, 108,   0, 200], dtype=np.uint8)),  # orange
    (0.50, np.array([255, 235,  59, 200], dtype=np.uint8)),  # yellow
    (0.75, np.array([139, 195,  74, 200], dtype=np.uint8)),  # light green
    (1.00, np.array([ 27,  94,  32, 200], dtype=np.uint8)),  # dark green
]


def colorize_accessibility(grid: np.ndarray) -> np.ndarray:
    """Convert a 2D float32 grid (0-1) to RGBA (H, W, 4) uint8."""
    h, w = grid.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    valid_mask = np.isfinite(grid)
    clamped = np.clip(grid, 0.0, 1.0)

    for i in range(len(_ACCESSIBILITY_STOPS) - 1):
        v0, c0 = _ACCESSIBILITY_STOPS[i]
        v1, c1 = _ACCESSIBILITY_STOPS[i + 1]

        if i == 0:
            band = valid_mask & (clamped <= v1)
        elif i == len(_ACCESSIBILITY_STOPS) - 2:
            band = valid_mask & (clamped > v0)
        else:
            band = valid_mask & (clamped > v0) & (clamped <= v1)

        if not np.any(band):
            continue

        dv = v1 - v0
        t = np.clip((clamped - v0) / dv, 0.0, 1.0) if dv > 0 else np.zeros_like(clamped)

        for ch in range(4):
            rgba[band, ch] = (
                c0[ch] * (1 - t[band]) + c1[ch] * t[band]
            ).astype(np.uint8)

    return rgba


def render_accessibility_tile(
    score_grid: np.ndarray,
    tile_size: int = 256,
) -> Optional[bytes]:
    """Render an accessibility score grid to a PNG tile."""
    if score_grid is None or score_grid.size == 0:
        return None

    from PIL import Image

    rgba = colorize_accessibility(score_grid)
    img = Image.fromarray(rgba, "RGBA")
    if img.size != (tile_size, tile_size):
        img = img.resize((tile_size, tile_size), Image.Resampling.NEAREST)

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()
