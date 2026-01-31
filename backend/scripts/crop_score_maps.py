#!/usr/bin/env python3
"""
Crop score map images to remove padding.
The original images have ~20px padding on all sides which causes misalignment
when displayed as map overlays.
"""

import os
import sys
from pathlib import Path
from PIL import Image
import numpy as np

def find_content_bounds(img):
    """Find the bounding box of non-transparent content."""
    arr = np.array(img)
    if arr.shape[2] < 4:  # No alpha channel
        return 0, 0, img.width, img.height

    alpha = arr[:, :, 3]
    non_trans = np.where(alpha > 0)

    if len(non_trans[0]) == 0:
        return 0, 0, img.width, img.height

    top = int(non_trans[0].min())
    bottom = int(non_trans[0].max())
    left = int(non_trans[1].min())
    right = int(non_trans[1].max())

    return left, top, right + 1, bottom + 1

def crop_score_map(input_path, output_path=None):
    """Crop a score map image to remove transparent padding."""
    if output_path is None:
        output_path = input_path

    img = Image.open(input_path)
    original_size = img.size

    # Find content bounds
    left, top, right, bottom = find_content_bounds(img)
    content_size = (right - left, bottom - top)

    # Skip if no significant padding
    padding = (left, top, original_size[0] - right, original_size[1] - bottom)
    if max(padding) < 5:
        return False

    # Crop
    cropped = img.crop((left, top, right, bottom))

    # Save cropped image
    cropped.save(output_path, format="PNG")

    print(f"  Cropped {os.path.basename(input_path)}: {original_size} -> {content_size}")
    return True

def main():
    score_maps_dir = Path(__file__).parent.parent / "crism_score_maps"

    if not score_maps_dir.exists():
        print(f"Error: Score maps directory not found: {score_maps_dir}")
        sys.exit(1)

    print(f"Processing score maps in: {score_maps_dir}")

    png_files = list(score_maps_dir.glob("*.png"))
    print(f"Found {len(png_files)} PNG files")

    cropped_count = 0
    for png_file in sorted(png_files):
        if png_file.name.endswith(".bak.png"):
            continue
        if crop_score_map(str(png_file)):
            cropped_count += 1

    print(f"\nDone! Cropped {cropped_count} images.")

if __name__ == "__main__":
    main()
