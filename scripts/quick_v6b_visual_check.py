#!/usr/bin/env python3
"""Quick check: V6b visual-only classification quality on training data."""
import sys
import json
import numpy as np
import torch

sys.path.insert(0, "/disk1/cspark/hirise-api")
from scripts.marslandform_v2.models.late_fusion_classifier import LateFusionClassifier

DATA_DIR = "/disk1/cspark/MarsLab/Data/HiRISE/v5_retrain"
CLASS_NAMES = ["LDA", "LVF", "CCF", "OTHER"]

# Load V6b
ckpt = torch.load(f"{DATA_DIR}/late_fusion_v6b.pt", map_location="cpu", weights_only=False)
cfg = ckpt["cfg"]
model = LateFusionClassifier(
    visual_dim=cfg["visual_dim"], mola_dim=cfg["mola_dim"],
    visual_hidden=cfg["visual_hidden"], mola_hidden=cfg["mola_hidden"],
    num_classes=cfg["num_classes"], dropout=cfg["dropout"],
)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Load data
embeddings = np.load(f"{DATA_DIR}/embeddings_v5.npy")
mola_dict = np.load(f"{DATA_DIR}/mola_features_v5.npy", allow_pickle=True).item()
with open(f"{DATA_DIR}/tile_labels_v5.json") as f:
    labels_list = json.load(f)

emb_list, mola_list, label_list = [], [], []
for i, tile_info in enumerate(labels_list):
    pid = tile_info["image_id"]
    key = f"{tile_info['tile_row']}_{tile_info['tile_col']}"
    if pid in mola_dict and key in mola_dict[pid]:
        emb_list.append(i)
        mola_list.append(mola_dict[pid][key])
        lbl = tile_info["label"]
        label_list.append(CLASS_NAMES.index(lbl) if lbl in CLASS_NAMES else 3)

embeddings = embeddings[emb_list].astype(np.float32)
mola = np.array(mola_list, dtype=np.float32)
labels = np.array(label_list, dtype=np.int64)

emb_t = torch.from_numpy(embeddings).float()
mola_t = torch.from_numpy(mola).float()

with torch.no_grad():
    vis_logits = model.get_visual_logits(emb_t)
    mola_logits = model.get_mola_logits(mola_t)
    combined = model(emb_t, mola_t)

vis_preds = vis_logits.argmax(1).numpy()
mola_preds = mola_logits.argmax(1).numpy()
comb_preds = combined.argmax(1).numpy()

# Per-class metrics
print("=== Visual-only per-class accuracy ===")
for c in range(4):
    mask = labels == c
    if mask.sum() > 0:
        acc = (vis_preds[mask] == c).mean()
        print(f"  {CLASS_NAMES[c]}: {acc:.3f} ({(vis_preds[mask] == c).sum()}/{mask.sum()})")

print("\n=== Visual logit distribution ===")
for c in range(4):
    col = vis_logits.numpy()[:, c]
    print(f"  {CLASS_NAMES[c]}: mean={col.mean():.3f}, std={col.std():.3f}")

print("\n=== MOLA logit distribution ===")
for c in range(4):
    col = mola_logits.numpy()[:, c]
    print(f"  {CLASS_NAMES[c]}: mean={col.mean():.3f}, std={col.std():.3f}, max={col.max():.3f}")

# Check: what happens when visual says LDA but MOLA says OTHER?
vis_lda = vis_preds == 0
mola_other = mola_preds == 3
conflict = vis_lda & mola_other
print(f"\nVisual=LDA but MOLA=OTHER: {conflict.sum()} tiles")
print(f"  Of those, actual labels: {dict(zip(*np.unique(labels[conflict], return_counts=True)))}")
print(f"  Combined predicts: {dict(zip(*np.unique(comb_preds[conflict], return_counts=True)))}")
