#!/usr/bin/env python3
"""Fast SSL re-embedding: only labeled images, subsample tiles, CPU-optimized."""
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

ROOT = Path("/disk1/cspark/MarsLab")
sys.path.insert(0, str(ROOT))

from scripts.marslandform_v2.config import V2_OUTPUT, get_config
from scripts.marslandform_v2.models.dinov2_lora import DinoV2LoRA


MAX_TILES_PER_IMAGE = None  # Use ALL tiles for fair comparison with frozen


def main():
    cfg = get_config().dinov2
    device = torch.device("cpu")
    lora_path = ROOT / "Data/HiRISE/v2_output/ssl_lora_weights/best_model.pt"

    # Load labels
    labels = json.loads((V2_OUTPUT / "labels_simple.json").read_text())
    print(f"Labeled images: {len(labels)}")

    # Load model with LoRA
    print("Loading DINOv2 + LoRA...")
    model = DinoV2LoRA(cfg, use_lora=True)
    ckpt = torch.load(lora_path, map_location="cpu", weights_only=False)
    print(f"Checkpoint epoch: {ckpt['epoch']}, loss: {ckpt['loss']:.4f}")

    # Load student backbone weights
    missing, unexpected = model.load_state_dict(ckpt["student_backbone"], strict=False)
    lora_loaded = sum(1 for k in ckpt["student_backbone"] if "lora" in k)
    print(f"LoRA weights loaded: {lora_loaded} keys, missing: {len(missing)}, unexpected: {len(unexpected)}")

    model.eval()
    model.to(device)

    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    tiles_dir = V2_OUTPUT / "tiles"
    per_image_embeddings = {}
    total_tiles = 0
    random.seed(42)

    for img_id in tqdm(labels.keys(), desc="Embedding labeled images"):
        img_dir = tiles_dir / img_id
        if not img_dir.exists():
            continue

        tile_paths = sorted(img_dir.glob("*.jpg"))
        if not tile_paths:
            tile_paths = sorted(img_dir.glob("*.jpeg"))
        if not tile_paths:
            continue

        # Subsample tiles if limit set
        if MAX_TILES_PER_IMAGE is not None and len(tile_paths) > MAX_TILES_PER_IMAGE:
            tile_paths = random.sample(tile_paths, MAX_TILES_PER_IMAGE)

        # Process tiles in small batches
        all_embeddings = []
        batch_size = 16
        for i in range(0, len(tile_paths), batch_size):
            batch_paths = tile_paths[i:i + batch_size]
            batch_tensors = []
            for tp in batch_paths:
                try:
                    with Image.open(tp) as img:
                        img = img.convert("RGB")
                        batch_tensors.append(transform(img))
                except Exception:
                    continue

            if not batch_tensors:
                continue

            batch = torch.stack(batch_tensors).to(device)
            with torch.no_grad():
                emb = model(batch)  # [B, 768]
            all_embeddings.append(emb.numpy())

        if all_embeddings:
            per_image_embeddings[img_id] = np.concatenate(all_embeddings, axis=0).astype(np.float32)
            total_tiles += per_image_embeddings[img_id].shape[0]

    # Save
    out_dir = V2_OUTPUT / "embeddings_ssl"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "embeddings_by_image.npy"
    np.save(out_path, per_image_embeddings)

    print(f"\n{'='*50}")
    print(f"SSL embeddings saved: {out_path}")
    print(f"  Images: {len(per_image_embeddings)}")
    print(f"  Total tiles: {total_tiles:,}")
    first_key = next(iter(per_image_embeddings))
    print(f"  Example: {first_key} → {per_image_embeddings[first_key].shape}")
    print(f"  Embedding dim: {per_image_embeddings[first_key].shape[1]}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
