"""Fusion tile renderer — accessibility heatmap with landform overlay.

Renders accessibility scores as the base colormap (red→green), then
overlays landform coverage regions with a distinctive border/hatch
to indicate where geomorphological evidence enhances the score.
"""

from __future__ import annotations

import io
from typing import Optional

import numpy as np

from analysis.accessibility.tile_renderer import colorize_accessibility


def render_fusion_tile(
    score_grid: np.ndarray,
    landform_grid: Optional[np.ndarray],
    tile_size: int = 256,
) -> Optional[bytes]:
    """Render a fusion tile as PNG.

    - Base colour: accessibility score (red→green gradient)
    - Landform overlay: where landform_grid > 0, add a semi-transparent
      tint indicating landform evidence strength
    """
    if score_grid is None or score_grid.size == 0:
        return None

    from PIL import Image, ImageDraw

    # Base layer: accessibility colourmap
    rgba = colorize_accessibility(score_grid)

    # Overlay landform coverage if available
    if landform_grid is not None:
        valid_lf = np.isfinite(landform_grid) & (landform_grid > 0)
        if np.any(valid_lf):
            _apply_landform_overlay(rgba, landform_grid, valid_lf)

    img = Image.fromarray(rgba, "RGBA")
    if img.size != (tile_size, tile_size):
        img = img.resize((tile_size, tile_size), Image.Resampling.NEAREST)

    # Draw landform coverage borders
    if landform_grid is not None:
        valid_lf = np.isfinite(landform_grid) & (landform_grid > 0)
        if np.any(valid_lf):
            _draw_landform_borders(img, valid_lf, tile_size)

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _apply_landform_overlay(
    rgba: np.ndarray,
    landform_grid: np.ndarray,
    valid_mask: np.ndarray,
) -> None:
    """Apply a semi-transparent tint to landform regions.

    Colour by landform strength:
    - bonus >= 0.8 (LDA/LVF): blue-cyan tint
    - bonus >= 0.5 (CCF): amber tint
    - bonus > 0 (weak): light tint
    """
    # Blue-cyan tint for strong glacial features (LDA, LVF)
    strong_mask = valid_mask & (landform_grid >= 0.8)
    if np.any(strong_mask):
        # Blend existing colour with blue-cyan
        alpha = 0.25
        rgba[strong_mask, 0] = (rgba[strong_mask, 0] * (1 - alpha) + 60 * alpha).astype(np.uint8)
        rgba[strong_mask, 1] = (rgba[strong_mask, 1] * (1 - alpha) + 180 * alpha).astype(np.uint8)
        rgba[strong_mask, 2] = (rgba[strong_mask, 2] * (1 - alpha) + 220 * alpha).astype(np.uint8)

    # Amber tint for moderate features (CCF)
    moderate_mask = valid_mask & (landform_grid >= 0.5) & (landform_grid < 0.8)
    if np.any(moderate_mask):
        alpha = 0.2
        rgba[moderate_mask, 0] = (rgba[moderate_mask, 0] * (1 - alpha) + 255 * alpha).astype(np.uint8)
        rgba[moderate_mask, 1] = (rgba[moderate_mask, 1] * (1 - alpha) + 193 * alpha).astype(np.uint8)
        rgba[moderate_mask, 2] = (rgba[moderate_mask, 2] * (1 - alpha) + 7 * alpha).astype(np.uint8)


def _draw_landform_borders(
    img,
    valid_mask: np.ndarray,
    tile_size: int,
) -> None:
    """Draw thin border around landform coverage regions.

    Uses edge detection on the valid_mask to find boundary pixels,
    then draws them in a distinctive colour.
    """
    from PIL import ImageDraw

    # Scale mask to tile size
    from PIL import Image as PILImage
    mask_img = PILImage.fromarray((valid_mask * 255).astype(np.uint8), "L")
    if mask_img.size != (tile_size, tile_size):
        mask_img = mask_img.resize((tile_size, tile_size), PILImage.Resampling.NEAREST)

    mask_arr = np.array(mask_img) > 128

    # Simple edge detection: pixel is border if it's in the mask
    # but has a neighbor that isn't
    padded = np.pad(mask_arr, 1, mode="constant", constant_values=False)
    borders = mask_arr & ~(
        padded[:-2, 1:-1] &
        padded[2:, 1:-1] &
        padded[1:-1, :-2] &
        padded[1:-1, 2:]
    )

    if not np.any(borders):
        return

    draw = ImageDraw.Draw(img)
    border_color = (100, 200, 255, 200)  # Light blue, semi-transparent

    # Draw border pixels
    ys, xs = np.where(borders)
    for x, y in zip(xs, ys):
        draw.point((x, y), fill=border_color)
