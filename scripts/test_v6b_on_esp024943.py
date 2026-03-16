#!/usr/bin/env python3
"""Test V6b model on ESP_024943_2345 to check OOD MOLA behavior."""
import sys
import json
import numpy as np
import torch

sys.path.insert(0, "/disk1/cspark/hirise-api")
from scripts.marslandform_v2.models.late_fusion_classifier import LateFusionClassifier

DATA_DIR = "/disk1/cspark/MarsLab/Data/HiRISE/v5_retrain"
CLASS_NAMES = ["LDA", "LVF", "CCF", "OTHER"]

# Load V6b model
ckpt = torch.load(f"{DATA_DIR}/late_fusion_v6b.pt", map_location="cpu", weights_only=False)
cfg = ckpt["cfg"]
model = LateFusionClassifier(
    visual_dim=cfg["visual_dim"],
    mola_dim=cfg["mola_dim"],
    visual_hidden=cfg["visual_hidden"],
    mola_hidden=cfg["mola_hidden"],
    num_classes=cfg["num_classes"],
    dropout=cfg["dropout"],
)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

print(f"Model version: {ckpt.get('version')}")
print(f"Val F1: {ckpt.get('val_f1'):.4f}")
print(f"Weights: vis={model.vis_weight.item():.4f}, mola={model.mola_weight.item():.4f}")
print(f"MOLA mean buffer: {model.mola_mean[:5].numpy()} ...")
print(f"MOLA std buffer:  {model.mola_std[:5].numpy()} ...")

# Load ESP_024943_2345 data
import os
results_dir = "/disk1/cspark/hirise-api/results/ESP_024943_2345"
embeddings_path = os.path.join(results_dir, "embeddings.npy")
mola_path = os.path.join(results_dir, "mola_features.npy")

if os.path.exists(embeddings_path) and os.path.exists(mola_path):
    embeddings = np.load(embeddings_path)
    mola_dict = np.load(mola_path, allow_pickle=True).item()
    
    # Reconstruct MOLA array matching embeddings
    coords_path = os.path.join(results_dir, "tile_coords.json")
    if os.path.exists(coords_path):
        with open(coords_path) as f:
            coords = json.load(f)
        mola_list = []
        for c in coords:
            key = f"{c['row']}_{c['col']}"
            if key in mola_dict:
                feat = mola_dict[key]
                if len(feat) < 25:
                    feat = np.pad(feat, (0, 25 - len(feat)))
                mola_list.append(feat[:25])
            else:
                mola_list.append(np.zeros(25, dtype=np.float32))
        mola_arr = np.array(mola_list, dtype=np.float32)
    else:
        # Try flattening mola_dict
        keys = sorted(mola_dict.keys(), key=lambda x: (int(x.split("_")[0]), int(x.split("_")[1])))
        mola_list = []
        for k in keys:
            feat = mola_dict[k]
            if len(feat) < 25:
                feat = np.pad(feat, (0, 25 - len(feat)))
            mola_list.append(feat[:25])
        mola_arr = np.array(mola_list, dtype=np.float32)
    
    print(f"\nESP_024943_2345: {len(embeddings)} tiles, {mola_arr.shape} MOLA features")
    print(f"MOLA sample: {mola_arr[0][:5]} ... (feat 21-24: {mola_arr[0][21:]})")
    
    # Check MOLA after normalization
    mola_t = torch.from_numpy(mola_arr).float()
    mola_normed = model._normalize_mola(mola_t)
    print(f"\nMOLA after z-score normalization:")
    print(f"  Range: [{mola_normed.min().item():.2f}, {mola_normed.max().item():.2f}]")
    print(f"  Mean: {mola_normed.mean(0)[:5].numpy()}")
    
    # Get predictions
    emb_t = torch.from_numpy(embeddings).float()
    with torch.no_grad():
        logits = model(emb_t, mola_t)
        vis_logits = model.get_visual_logits(emb_t)
        mola_logits = model.get_mola_logits(mola_t)
        
    probs = torch.softmax(logits, dim=1).numpy()
    vis_probs = torch.softmax(vis_logits, dim=1).numpy()
    mola_probs_arr = torch.softmax(mola_logits, dim=1).numpy()
    preds = np.argmax(probs, axis=1)
    vis_preds = np.argmax(vis_probs, axis=1)
    mola_preds = np.argmax(mola_probs_arr, axis=1)
    
    print(f"\n=== Prediction Distribution ===")
    print(f"{'Class':>8} | {'Combined':>10} | {'Visual-only':>12} | {'MOLA-only':>10}")
    print("-" * 55)
    for c, name in enumerate(CLASS_NAMES):
        print(f"{name:>8} | {(preds == c).sum():>10} | {(vis_preds == c).sum():>12} | {(mola_preds == c).sum():>10}")
    
    print(f"\n=== Logit Statistics ===")
    print(f"Combined logits: mean={logits.numpy().mean(0)}, std={logits.numpy().std(0)}")
    print(f"Visual logits:   mean={vis_logits.numpy().mean(0)}, std={vis_logits.numpy().std(0)}")
    print(f"MOLA logits:     mean={mola_logits.numpy().mean(0)}, std={mola_logits.numpy().std(0)}")
    
    # Show a few example tiles
    print(f"\n=== Sample Tiles ===")
    for i in range(min(5, len(embeddings))):
        print(f"Tile {i}: combined={CLASS_NAMES[preds[i]]} (prob={probs[i].max():.3f}), "
              f"visual={CLASS_NAMES[vis_preds[i]]} (prob={vis_probs[i].max():.3f}), "
              f"mola={CLASS_NAMES[mola_preds[i]]} (prob={mola_probs_arr[i].max():.3f})")
else:
    print(f"\nCannot find cached data for ESP_024943_2345 at {results_dir}")
    print("Will test with synthetic MOLA features representing this product's characteristics")
    
    # Use known MOLA stats for ESP_024943_2345
    # slope=0.1055, elevation=-4251, latitude=54.34
    esp_mola = np.zeros((1, 25), dtype=np.float32)
    esp_mola[0, 0] = 0.1055  # slope
    esp_mola[0, 21] = -4251.0  # elevation
    esp_mola[0, 22] = 54.34  # latitude
    
    mola_t = torch.from_numpy(esp_mola).float()
    mola_normed = model._normalize_mola(mola_t)
    print(f"\nESP_024943 MOLA after z-score: {mola_normed.numpy().flatten()}")
    
    with torch.no_grad():
        mola_logits = model.get_mola_logits(mola_t)
    print(f"MOLA logits: {mola_logits.numpy().flatten()}")
    print(f"MOLA class: {CLASS_NAMES[mola_logits.argmax().item()]}")
