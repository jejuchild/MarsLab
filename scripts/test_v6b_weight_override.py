#!/usr/bin/env python3
"""
Test V6b with weight override: train with learnable weights, inference at vis=0.7/mola=0.3.
Uses the first V6b model (unfrozen weights, vis=0.35/mola=0.65 learned).
We reload and override weights to see effect on val F1 and ESP_024943.
"""
import sys
import json
import numpy as np
import torch

sys.path.insert(0, "/disk1/cspark/hirise-api")
from scripts.marslandform_v2.models.late_fusion_classifier import LateFusionClassifier

DATA_DIR = "/disk1/cspark/MarsLab/Data/HiRISE/v5_retrain"
CLASS_NAMES = ["LDA", "LVF", "CCF", "OTHER"]
NUM_CLASSES = 4

def compute_metrics(preds, labels):
    acc = (preds == labels).mean()
    f1s = []
    for c in range(NUM_CLASSES):
        tp = ((preds == c) & (labels == c)).sum()
        fp = ((preds == c) & (labels != c)).sum()
        fn = ((preds != c) & (labels == c)).sum()
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        f1s.append(f1)
    return acc, np.mean(f1s), f1s

# Load training data for validation
embeddings = np.load(f"{DATA_DIR}/embeddings_v5.npy")
mola_dict = np.load(f"{DATA_DIR}/mola_features_v5.npy", allow_pickle=True).item()
with open(f"{DATA_DIR}/tile_labels_v5.json") as f:
    labels_list = json.load(f)

emb_list, mola_list, label_list, pids = [], [], [], []
for i, tile_info in enumerate(labels_list):
    pid = tile_info["image_id"]
    key = f"{tile_info['tile_row']}_{tile_info['tile_col']}"
    if pid in mola_dict and key in mola_dict[pid]:
        emb_list.append(i)
        mola_list.append(mola_dict[pid][key])
        lbl = tile_info["label"]
        label_list.append(CLASS_NAMES.index(lbl) if lbl in CLASS_NAMES else 3)
        pids.append(pid)

embeddings = embeddings[emb_list].astype(np.float32)
mola = np.array(mola_list, dtype=np.float32)
labels = np.array(label_list, dtype=np.int64)

# Split same way as training
rng = np.random.RandomState(42)
unique_pids = sorted(set(pids))
rng.shuffle(unique_pids)
n_val = max(1, int(len(unique_pids) * 0.15))
val_pids = set(unique_pids[:n_val])
val_idx = np.array([i for i, p in enumerate(pids) if p in val_pids])

# We need to retrain with unfrozen weights. Let's check if there's a backup.
# Actually, let's just retrain quickly. Or better: test different weight combinations
# using the frozen-weight model (which already exists) by overriding _log_vis_weight.

# Load the FROZEN weight model (vis=0.7 forced)
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

print(f"Loaded V6b: vis={model.vis_weight.item():.3f}, mola={model.mola_weight.item():.3f}")
print(f"Val F1: {ckpt['val_f1']:.4f}")

# Test val performance
emb_t = torch.from_numpy(embeddings[val_idx]).float()
mola_t = torch.from_numpy(mola[val_idx]).float()
labels_val = labels[val_idx]

# Test with different weight ratios
print(f"\n=== Weight sweep on validation set ===")
print(f"{'vis':>5} | {'mola':>5} | {'Acc':>6} | {'F1':>6} | Per-class F1")
print("-" * 80)

for vis_w in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    mola_w = 1.0 - vis_w
    with torch.no_grad():
        vis_logits = model.get_visual_logits(emb_t)
        mola_logits = model.get_mola_logits(mola_t)
        combined = vis_w * vis_logits + mola_w * mola_logits
        preds = combined.argmax(1).numpy()
    
    acc, f1, f1s = compute_metrics(preds, labels_val)
    f1_str = " ".join(f"{CLASS_NAMES[c]}={f1s[c]:.3f}" for c in range(NUM_CLASSES))
    print(f"{vis_w:>5.1f} | {mola_w:>5.1f} | {acc:>6.3f} | {f1:>6.3f} | {f1_str}")

# Test on ESP_024943 MOLA features
print(f"\n=== ESP_024943_2345 MOLA logits ===")
esp_mola = np.zeros((1, 25), dtype=np.float32)
esp_mola[0, 0] = 0.1055
esp_mola[0, 21] = -4251.0
esp_mola[0, 22] = 54.34
esp_mola_t = torch.from_numpy(esp_mola).float()

with torch.no_grad():
    mola_logits = model.get_mola_logits(esp_mola_t)
print(f"MOLA logits: {mola_logits.numpy().flatten()}")
print(f"MOLA pred: {CLASS_NAMES[mola_logits.argmax().item()]}")
print(f"MOLA logit max-min spread: {mola_logits.max().item() - mola_logits.min().item():.2f}")
