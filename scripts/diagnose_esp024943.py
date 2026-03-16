#!/usr/bin/env python3
"""
Diagnose why ESP_024943_2345 classifies as all-OTHER.
Extract embeddings and MOLA features, then test different alpha values.
"""

import sys
import numpy as np
import torch
from pathlib import Path
from collections import Counter
from PIL import Image

sys.path.insert(0, "/disk1/cspark/hirise-api")
sys.path.insert(0, "/disk1/cspark/MarsLab/backend")

from scripts.marslandform_v2.models.film_classifier import FiLMClassifier
from analysis.hirise_landforms.preprocessing import (
    fetch_hirise_browse,
    tile_image,
    extract_mola_features,
)

CHECKPOINT = Path("/disk1/cspark/MarsLab/Data/HiRISE/v5_retrain/film_classifier_v5c.pt")
CLASS_NAMES = ["LDA", "LVF", "CCF", "OTHER"]


def load_model(device):
    model = FiLMClassifier(visual_dim=768, mola_dim=25, num_classes=4,
                           film_hidden=64, head_hidden=128, dropout=0.4)
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(sd)
    model.eval()
    return model.to(device)


def get_embeddings_for_product(product_id, lat, lon):
    """Extract DINOv2 embeddings and MOLA features for a product."""
    from scripts.marslandform_v2.config import DINOv2Config
    from scripts.marslandform_v2.models.dinov2_lora import DinoV2LoRA

    device = torch.device("cpu")

    # Get browse image
    browse_dir = Path("/disk1/cspark/MarsLab/Data/HiRISE/midlat_browse")
    browse_files = list(browse_dir.glob(f"{product_id}*"))
    if browse_files:
        img = Image.open(browse_files[0])
        print(f"Browse image: {browse_files[0]} ({img.size})")
    else:
        print(f"Fetching browse image for {product_id}...")
        img = fetch_hirise_browse(product_id)
        print(f"Fetched: {img.size}")

    # Tile
    tiles = tile_image(img, tile_size=224)
    print(f"Tiles: {len(tiles)}")

    # MOLA features - single lat/lon for the whole product
    mola_feat = extract_mola_features(lat, lon)
    print(f"MOLA features shape: {mola_feat.shape}")
    print(f"MOLA features: {mola_feat}")
    # Replicate for all tiles
    mola_array = np.tile(mola_feat, (len(tiles), 1))

    # Extract embeddings with vanilla DINOv2
    cfg = DINOv2Config()
    backbone = DinoV2LoRA(cfg, use_lora=False)
    backbone.eval()
    backbone = backbone.to(device)

    # Process tiles
    tile_images = []
    for row, col, tile_img in tiles:
        arr = np.array(tile_img.convert("RGB")).astype(np.float32) / 255.0
        tile_images.append(arr)

    tile_array = np.stack(tile_images)  # (N, 224, 224, 3)
    tile_tensor = torch.from_numpy(tile_array).permute(0, 3, 1, 2).float()

    # DINOv2 normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    tile_tensor = (tile_tensor - mean) / std

    # Extract embeddings
    embeddings = []
    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(tile_tensor), batch_size):
            batch = tile_tensor[i:i + batch_size].to(device)
            emb = backbone(batch)
            embeddings.append(emb.cpu().numpy())

    embeddings = np.concatenate(embeddings)
    print(f"Embeddings shape: {embeddings.shape}")

    return embeddings, mola_array


def test_alphas(model, embeddings, mola, device):
    """Test different alpha values and biases."""
    alphas = [0.0, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0]

    PROD_BIAS = np.array([-1.65, -1.85, -1.90, 0.0], dtype=np.float32)
    OLD_BIAS = np.array([-0.25, 0.15, -0.15, 0.0], dtype=np.float32)

    emb_t = torch.from_numpy(embeddings).float().to(device)
    mola_t = torch.from_numpy(mola).float().to(device)

    original_forward = model.film.forward

    print(f"\n{'Alpha':>6} | {'Bias':>8} | {'LDA':>5} | {'LVF':>5} | {'CCF':>5} | {'OTH':>5} | Top-5 non-OTHER probs | Mean raw logits")
    print("-" * 130)

    for alpha in alphas:
        for bias_name, bias in [("prod", PROD_BIAS), ("old", OLD_BIAS), ("none", np.zeros(4, dtype=np.float32))]:
            def dampened_forward(visual_features, mola_features, a=alpha):
                h = model.film.mola_encoder(mola_features)
                gamma = model.film.gamma_proj(h)
                beta = model.film.beta_proj(h)
                gamma_d = 1.0 + a * (gamma - 1.0)
                beta_d = a * beta
                return gamma_d * visual_features + beta_d

            model.film.forward = dampened_forward
            with torch.no_grad():
                logits = model(emb_t, mola_t)

            bias_t = torch.from_numpy(bias).float().to(device)
            logits_biased = logits + bias_t
            probs = torch.softmax(logits_biased, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            dist = Counter(preds)

            non_other_conf = probs[:, :3].max(axis=1)
            top5 = sorted(non_other_conf, reverse=True)[:5]
            top5_str = ", ".join(f"{c:.4f}" for c in top5)

            mean_logits = logits.mean(0).cpu().numpy()
            ml_str = "[" + ", ".join(f"{v:.2f}" for v in mean_logits) + "]"

            print(f"{alpha:>6.1f} | {bias_name:>8} | {dist.get(0, 0):>5} | {dist.get(1, 0):>5} | {dist.get(2, 0):>5} | {dist.get(3, 0):>5} | {top5_str} | {ml_str}")

    model.film.forward = original_forward


def main():
    device = torch.device("cpu")
    model = load_model(device)

    print("=== ESP_024943_2345 (lat=54.3374, lon=212.0251) ===")
    embeddings, mola = get_embeddings_for_product("ESP_024943_2345", 54.3374, 212.0251)

    print(f"\nMOLA feature values (same for all tiles):")
    for i in range(25):
        print(f"  feat[{i:2d}]: {mola[0, i]:10.4f}")

    test_alphas(model, embeddings, mola, device)


if __name__ == "__main__":
    main()
