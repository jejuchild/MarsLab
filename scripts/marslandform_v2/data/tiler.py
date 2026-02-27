"""
HiRISE browse image tiling for MarsLandformNet V2.

Takes full HiRISE browse JPEG images and extracts 224×224 tiles.
Each tile gets a position-based lat/lon computed from the image center.
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.marslandform_v2.config import (
    BROWSE_DIR, METADATA_JSON, V2_OUTPUT, DINOv2Config, get_config,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TILE_SIZE = 224  # DINOv2 input size
MIN_TILE_CONTENT = 0.3  # minimum non-black fraction to keep a tile


def extract_tiles(
    image_path: Path,
    tile_size: int = TILE_SIZE,
    min_content: float = MIN_TILE_CONTENT,
) -> List[Dict]:
    """
    Extract non-overlapping tiles from a browse image.
    Returns list of tile metadata dicts (without saving to disk yet).
    """
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        logger.warning(f"Cannot open {image_path}: {e}")
        return []

    w, h = img.size
    arr = np.array(img)
    tiles = []

    for row in range(0, h - tile_size + 1, tile_size):
        for col in range(0, w - tile_size + 1, tile_size):
            tile_arr = arr[row : row + tile_size, col : col + tile_size]

            # Skip mostly-black tiles (edges of browse images)
            content_frac = np.mean(tile_arr > 10)
            if content_frac < min_content:
                continue

            tiles.append({
                "row": row // tile_size,
                "col": col // tile_size,
                "pixel_row": row,
                "pixel_col": col,
                "tile_array": tile_arr,
                "content_fraction": float(content_frac),
            })

    return tiles


def compute_tile_coords(
    center_lat: float,
    center_lon: float,
    img_width: int,
    img_height: int,
    tile_row: int,
    tile_col: int,
    tile_size: int = TILE_SIZE,
    pixel_scale_m: float = 25.0,  # HiRISE browse is ~25 m/px
) -> Tuple[float, float]:
    """
    Compute approximate lat/lon for a tile center.
    Uses simple equirectangular projection from image center.
    """
    mars_circumference_m = 2 * np.pi * 3389500  # Mars equatorial radius
    deg_per_meter_lat = 360.0 / mars_circumference_m
    deg_per_meter_lon = deg_per_meter_lat / max(np.cos(np.radians(center_lat)), 0.01)

    # Pixel offset from image center
    img_center_row = img_height / 2
    img_center_col = img_width / 2
    tile_center_row = tile_row * tile_size + tile_size / 2
    tile_center_col = tile_col * tile_size + tile_size / 2

    dy_px = img_center_row - tile_center_row  # positive = north
    dx_px = tile_center_col - img_center_col  # positive = east

    tile_lat = center_lat + dy_px * pixel_scale_m * deg_per_meter_lat
    tile_lon = center_lon + dx_px * pixel_scale_m * deg_per_meter_lon

    return float(tile_lat), float(tile_lon)


def tile_images(
    image_ids: List[str],
    metadata: Dict[str, dict],
    browse_dir: Path,
    output_dir: Path,
    tile_size: int = TILE_SIZE,
) -> Tuple[Path, int]:
    """
    Tile a list of HiRISE browse images, saving tiles and metadata.

    Returns: (metadata_csv_path, total_tile_count)
    """
    tiles_dir = output_dir / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = output_dir / "tile_metadata.csv"
    total_tiles = 0

    with open(metadata_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "image_id", "tile_idx", "tile_row", "tile_col",
            "lat", "lon", "class_label", "tile_path", "content_fraction",
        ])
        writer.writeheader()

        for i, img_id in enumerate(image_ids):
            if img_id not in metadata:
                continue

            meta = metadata[img_id]
            # HiRISE browse files are named: {image_id}_RED.abrowse.jpg
            browse_path = browse_dir / f"{img_id}_RED.abrowse.jpg"
            if not browse_path.exists():
                # Try alternate naming conventions
                for pattern in [f"{img_id}.jpg", f"{img_id}.jpeg", f"{img_id}_RED.browse.jpg"]:
                    alt = browse_dir / pattern
                    if alt.exists():
                        browse_path = alt
                        break
                else:
                    # Try glob as last resort
                    matches = list(browse_dir.glob(f"{img_id}*"))
                    if matches:
                        browse_path = matches[0]
                    else:
                        continue

            tiles = extract_tiles(browse_path, tile_size)
            if not tiles:
                continue

            # Get image dimensions for coordinate computation
            img = Image.open(browse_path)
            img_w, img_h = img.size
            center_lat = meta.get("lat")
            center_lon = meta.get("lon")

            # Create per-image tile directory
            img_tile_dir = tiles_dir / img_id
            img_tile_dir.mkdir(exist_ok=True)

            for tile_idx, tile in enumerate(tiles):
                # Save tile image
                tile_filename = f"tile_{tile['row']:03d}_{tile['col']:03d}.jpg"
                tile_path = img_tile_dir / tile_filename
                Image.fromarray(tile["tile_array"]).save(tile_path, quality=95)

                # Compute coordinates
                if center_lat is not None and center_lon is not None:
                    t_lat, t_lon = compute_tile_coords(
                        center_lat, center_lon, img_w, img_h,
                        tile["row"], tile["col"], tile_size,
                    )
                else:
                    t_lat, t_lon = None, None

                writer.writerow({
                    "image_id": img_id,
                    "tile_idx": tile_idx,
                    "tile_row": tile["row"],
                    "tile_col": tile["col"],
                    "lat": t_lat,
                    "lon": t_lon,
                    "class_label": meta.get("final_class", meta.get("class", "UNLABELED")),
                    "tile_path": str(tile_path.relative_to(output_dir)),
                    "content_fraction": f"{tile['content_fraction']:.3f}",
                })
                total_tiles += 1

            if (i + 1) % 50 == 0:
                logger.info(f"  Tiled {i + 1}/{len(image_ids)} images ({total_tiles} tiles)")

    logger.info(f"Tiling complete: {total_tiles} tiles from {len(image_ids)} images")
    return metadata_path, total_tiles


def main():
    parser = argparse.ArgumentParser(description="Tile HiRISE browse images")
    parser.add_argument("--labels", type=str,
                       default=str(V2_OUTPUT / "unified_labels.json"),
                       help="Path to unified_labels.json (or midlat_metadata.json)")
    parser.add_argument("--browse-dir", type=str, default=str(BROWSE_DIR))
    parser.add_argument("--output-dir", type=str, default=str(V2_OUTPUT))
    parser.add_argument("--tile-size", type=int, default=TILE_SIZE)
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of images to tile (for testing)")
    args = parser.parse_args()

    # Load labels
    labels_path = Path(args.labels)
    with open(labels_path) as f:
        labels_data = json.load(f)

    # Support both unified_labels.json format and midlat_metadata.json format
    if isinstance(labels_data, list):
        metadata = {m["image_id"]: m for m in labels_data}
    else:
        metadata = labels_data

    image_ids = list(metadata.keys())
    if args.limit:
        image_ids = image_ids[: args.limit]

    logger.info(f"Tiling {len(image_ids)} images from {labels_path.name}")

    _, total = tile_images(
        image_ids=image_ids,
        metadata=metadata,
        browse_dir=Path(args.browse_dir),
        output_dir=Path(args.output_dir),
        tile_size=args.tile_size,
    )

    logger.info(f"Done. {total} tiles saved to {args.output_dir}/tiles/")


if __name__ == "__main__":
    main()
