#!/usr/bin/env python3
"""
Extract DINOv2 ViT-L/14 patch tokens for all training tiles.

For each 224×224 tile, extracts 16×16=256 patch tokens of 1024-dim each.
Uses the frozen pretrained backbone (no LoRA).

Output: patch_tokens_v8/ directory with one .npy file per product:
  {product_id}.npy → dict {row_col: np.array(256, 1024, dtype=float16)}

Total expected: ~67,904 tiles × 256 × 1024 × 2 bytes = ~33GB

Usage:
  python extract_patch_tokens_v8.py [--batch-size 4] [--resume]
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from transformers import Dinov2Model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path("/disk1/cspark/MarsLab")
BROWSE_DIR = ROOT / "Data/HiRISE/midlat_browse"
TILE_LABELS = ROOT / "Data/HiRISE/v5_retrain/tile_labels_v5.json"
OUTPUT_DIR = ROOT / "Data/HiRISE/v8_segmentation/patch_tokens_v8"

MODEL_NAME = "facebook/dinov2-large"  # ViT-L/14, 1024-dim, 256 patches for 224×224
EMBED_DIM = 1024
PATCHES_PER_SIDE = 16
TILE_SIZE = 224

# ImageNet normalization (standard for DINOv2)
_transform = transforms.Compose([
    transforms.Resize(TILE_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(TILE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


def find_browse_image(product_id: str) -> Path | None:
    """Find the browse image for a product ID."""
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG"):
        for name in (product_id, product_id.upper(), product_id.lower()):
            path = BROWSE_DIR / f"{name}{ext}"
            if path.exists():
                return path
    # Glob fallback
    matches = list(BROWSE_DIR.glob(f"*{product_id}*"))
    return matches[0] if matches else None


def tile_image(img: Image.Image, tile_size: int = 224, min_content: float = 0.3):
    """
    Tile a browse image into tile_size × tile_size patches.
    Returns: list of (grid_x, grid_y, PIL.Image)
    """
    img_array = np.array(img)
    if img_array.ndim == 2:
        img_array = np.stack([img_array] * 3, axis=-1)

    h, w = img_array.shape[:2]
    tiles = []
    for y0 in range(0, h - tile_size + 1, tile_size):
        for x0 in range(0, w - tile_size + 1, tile_size):
            tile = img_array[y0:y0 + tile_size, x0:x0 + tile_size]
            # Check content (skip mostly-black tiles)
            if float(np.mean(tile > 10)) < min_content:
                continue
            tiles.append((x0 // tile_size, y0 // tile_size, Image.fromarray(tile)))
    return tiles


def extract_for_product(
    model: Dinov2Model,
    product_id: str,
    tile_keys: list[str],
    device: torch.device,
    batch_size: int = 4,
) -> dict[str, np.ndarray]:
    """
    Extract patch tokens for all tiles in a product.

    Returns: {row_col: np.array(256, 1024, dtype=float16)}
    """
    browse_path = find_browse_image(product_id)
    if browse_path is None:
        logger.warning(f"  Browse image not found for {product_id}, skipping")
        return {}

    try:
        img = Image.open(browse_path).convert("RGB")
    except Exception as e:
        logger.warning(f"  Failed to open {browse_path}: {e}")
        return {}

    # Tile the image
    tiles = tile_image(img, TILE_SIZE)
    # Build lookup: (gx, gy) → tile_image
    tile_lookup = {f"{gy}_{gx}": tile_img for gx, gy, tile_img in tiles}

    # Filter to tiles that exist in our label set
    result = {}
    matching_keys = [k for k in tile_keys if k in tile_lookup]

    if not matching_keys:
        return {}

    # Process in batches
    for batch_start in range(0, len(matching_keys), batch_size):
        batch_keys = matching_keys[batch_start:batch_start + batch_size]
        batch_tensors = []
        for key in batch_keys:
            tensor = _transform(tile_lookup[key])
            batch_tensors.append(tensor)

        batch = torch.stack(batch_tensors).to(device)

        with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=False):
            outputs = model(pixel_values=batch)
            # last_hidden_state: (B, 1 + num_patches, embed_dim)
            # First token is CLS, rest are patch tokens
            patch_tokens = outputs.last_hidden_state[:, 1:]  # (B, 256, 1024)

        # Store as float16 to save space
        patch_np = patch_tokens.cpu().numpy().astype(np.float16)
        for i, key in enumerate(batch_keys):
            result[key] = patch_np[i]  # (256, 1024)

    return result


def main():
    parser = argparse.ArgumentParser(description="Extract DINOv2 ViT-L patch tokens")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for inference")
    parser.add_argument("--resume", action="store_true", help="Skip already-processed products")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load tile labels → group by product
    logger.info("Loading tile labels...")
    with open(TILE_LABELS) as f:
        tiles = json.load(f)

    product_tiles: dict[str, list[str]] = defaultdict(list)
    for t in tiles:
        pid = t["image_id"]
        key = f"{t['tile_row']}_{t['tile_col']}"
        product_tiles[pid].append(key)

    products = sorted(product_tiles.keys())
    logger.info(f"  {len(tiles)} tiles across {len(products)} products")

    # Check resume
    if args.resume:
        done = {p.stem for p in OUTPUT_DIR.glob("*.npy")}
        products = [p for p in products if p not in done]
        logger.info(f"  Resuming: {len(done)} already done, {len(products)} remaining")

    # Load model
    logger.info(f"Loading DINOv2 model: {MODEL_NAME}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Dinov2Model.from_pretrained(MODEL_NAME)
    model.eval()
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Model loaded: {total_params / 1e6:.0f}M params on {device}")

    # Extract
    logger.info("=" * 60)
    logger.info(f"Extracting patch tokens (batch_size={args.batch_size})...")
    t0 = time.time()
    total_tiles_done = 0
    total_tiles_skipped = 0

    for i, pid in enumerate(tqdm(products, desc="Products")):
        tile_keys = product_tiles[pid]
        result = extract_for_product(model, pid, tile_keys, device, args.batch_size)

        if result:
            out_path = OUTPUT_DIR / f"{pid}.npy"
            np.save(out_path, result, allow_pickle=True)
            total_tiles_done += len(result)
        else:
            total_tiles_skipped += len(tile_keys)

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(products) - i - 1) / rate
            logger.info(
                f"  {i + 1}/{len(products)} products, "
                f"{total_tiles_done} tiles extracted, "
                f"{total_tiles_skipped} skipped, "
                f"ETA: {eta / 3600:.1f}h"
            )

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info(f"DONE in {elapsed / 3600:.1f}h")
    logger.info(f"  Tiles extracted: {total_tiles_done}")
    logger.info(f"  Tiles skipped: {total_tiles_skipped}")
    logger.info(f"  Output dir: {OUTPUT_DIR}")

    # Check total size
    total_bytes = sum(f.stat().st_size for f in OUTPUT_DIR.glob("*.npy"))
    logger.info(f"  Total size: {total_bytes / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
