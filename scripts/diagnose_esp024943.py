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

sys.path.insert(0, "/disk1/cspark/hirise-api")
from scripts.marslandform_v2.models.film_classifier import FiLMClassifier

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


def get_embeddings_for_product(product_id):
    """Run the actual pipeline preprocessing to get embeddings and MOLA features."""
    sys.path.insert(0, "/disk1/cspark/MarsLab/backend")
    from analysis.hirise_landforms.preprocessing import (
        fetch_hirise_browse,
        tile_image,
        extract_mola_features,
    )
    from scripts.marslandform_v2.config import DINOv2Config
    from scripts.marslandform_v2.models.dinov2_lora import DinoV2LoRA

    device = torch.device("cpu")

    # Get browse image
    browse_dir = Path("/disk1/cspark/MarsLab/Data/HiRISE/midlat_browse")
    browse_files = list(browse_dir.glob(f"{product_id}*"))
    if not browse_files:
        print(f"No browse image found for {product_id}, trying download...")
        browse_path = download_browse_image(product_id, browse_dir)
    else:
        browse_path = browse_files[0]
    
    print(f"Browse image: {browse_path}")
    
    # Tile
    tiles, coords, meta = tile_browse_image(str(browse_path), tile_size=224)
    print(f"Tiles: {len(tiles)} tiles, shape: {tiles[0].shape if tiles else 'N/A'}")
    
    # Extract MOLA features
    lat = meta.get("center_lat", 54.3374)
    lon = meta.get("center_lon", 212.0251)
    mola_features = extract_mola_features(
        lat, lon, coords, meta,
        dem_path="/disk1/cspark/hirise-api/Mars_HRSC_MOLA_BlendDEM_Global_200mp_v2.tif"
    )
    print(f"MOLA features: {mola_features.shape}")
    
    # Extract embeddings with vanilla DINOv2
    cfg = DINOv2Config()
    backbone = DinoV2LoRA(cfg, use_lora=False)
    backbone.eval()
    backbone = backbone.to(device)
    
    # Stack tiles
    tile_array = np.stack(tiles)  # (N, H, W, 3)
    # Normalize to [0, 1] if needed
    if tile_array.max() > 1:
        tile_array = tile_array.astype(np.float32) / 255.0
    
    # Transpose to (N, 3, H, W)
    tile_tensor = torch.from_numpy(tile_array).permute(0, 3, 1, 2).float()
    
    # DINOv2 normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    tile_tensor = (tile_tensor - mean) / std
    
    # Extract embeddings in batches
    embeddings = []
    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(tile_tensor), batch_size):
            batch = tile_tensor[i:i+batch_size].to(device)
            emb = backbone(batch)
            embeddings.append(emb.cpu().numpy())
    
    embeddings = np.concatenate(embeddings)
    print(f"Embeddings: {embeddings.shape}")
    
    return embeddings, mola_features


def test_alphas(model, embeddings, mola, device):
    """Test different alpha values and biases."""
    alphas = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0]
    
    # Current production bias
    PROD_BIAS = np.array([-1.65, -1.85, -1.90, 0.0], dtype=np.float32)
    # Old bias
    OLD_BIAS = np.array([-0.25, 0.15, -0.15, 0.0], dtype=np.float32)
    
    emb_t = torch.from_numpy(embeddings).float().to(device)
    mola_t = torch.from_numpy(mola).float().to(device)
    
    print(f"\n{'Alpha':>6} | {'Bias':>20} | {'LDA':>5} | {'LVF':>5} | {'CCF':>5} | {'OTH':>5} | Top non-OTHER conf")
    print("-" * 90)
    
    for alpha in alphas:
        for bias_name, bias in [("prod", PROD_BIAS), ("old", OLD_BIAS), ("none", np.zeros(4, dtype=np.float32))]:
            original_forward = model.film.forward
            
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
            model.film.forward = original_forward
            
            bias_t = torch.from_numpy(bias).float().to(device)
            logits_biased = logits + bias_t
            probs = torch.softmax(logits_biased, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            dist = Counter(preds)
            
            # Find top non-OTHER confidence
            non_other_conf = probs[:, :3].max(axis=1)
            top5 = sorted(non_other_conf, reverse=True)[:5]
            top5_str = ", ".join(f"{c:.4f}" for c in top5)
            
            print(f"{alpha:>6.1f} | {bias_name:>20} | {dist.get(0,0):>5} | {dist.get(1,0):>5} | {dist.get(2,0):>5} | {dist.get(3,0):>5} | {top5_str}")
    
    # Detailed analysis at alpha=0.0 (no MOLA)
    print("\n=== Alpha=0.0 (visual-only) detailed ===")
    model.film.forward = lambda vis, mola: vis  # Identity
    with torch.no_grad():
        logits = model(emb_t, mola_t)
    model.film.forward = model.film.__class__.forward.__get__(model.film)  # Restore
    
    probs = torch.softmax(logits, dim=1).cpu().numpy()
    preds = np.argmax(probs, axis=1)
    
    print(f"Predictions: {Counter(preds)}")
    print(f"Mean logits: {logits.mean(0).cpu().numpy()}")
    print(f"Mean probs:  {probs.mean(0)}")
    
    # Show histogram of max non-OTHER prob
    non_other_max = probs[:, :3].max(axis=1)
    for thresh in [0.1, 0.2, 0.3, 0.4, 0.5]:
        count = (non_other_max > thresh).sum()
        print(f"  Tiles with max non-OTHER prob > {thresh}: {count}/{len(probs)}")


def main():
    device = torch.device("cpu")
    model = load_model(device)
    
    print("=== Extracting features for ESP_024943_2345 ===")
    embeddings, mola = get_embeddings_for_product("ESP_024943_2345")
    
    # Print MOLA feature stats
    print(f"\nMOLA feature stats for ESP_024943_2345:")
    for i in range(25):
        print(f"  feat[{i:2d}]: mean={mola[:,i].mean():8.4f}  std={mola[:,i].std():8.4f}  min={mola[:,i].min():8.4f}  max={mola[:,i].max():8.4f}")
    
    test_alphas(model, embeddings, mola, device)


if __name__ == "__main__":
    main()
