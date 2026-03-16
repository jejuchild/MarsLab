#!/usr/bin/env python3
"""Compare V5c (FiLM) vs V6b (Late Fusion Normalized) on the same val set."""
import sys, json, numpy as np, torch

sys.path.insert(0, "/disk1/cspark/hirise-api")
sys.path.insert(0, "/disk1/cspark/MarsLab")

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

# Load data
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

# Same split as training
rng = np.random.RandomState(42)
unique_pids = sorted(set(pids))
rng.shuffle(unique_pids)
n_val = max(1, int(len(unique_pids) * 0.15))
val_pids = set(unique_pids[:n_val])
val_idx = np.array([i for i, p in enumerate(pids) if p in val_pids])

emb_val = embeddings[val_idx]
mola_val = mola[val_idx]
labels_val = labels[val_idx]

print(f"Val set: {len(val_idx)} tiles")
print(f"Val label distribution: { {CLASS_NAMES[c]: int((labels_val==c).sum()) for c in range(4)} }")

# ===== V5c FiLM =====
print("\n" + "="*70)
print("V5c FiLM Classifier")
print("="*70)
from scripts.marslandform_v2.models.film_classifier import FiLMClassifier

v5_ckpt = torch.load(f"{DATA_DIR}/film_classifier_v5c.pt", map_location="cpu", weights_only=False)
v5_cfg = v5_ckpt.get("cfg", {})
v5_model = FiLMClassifier(
    visual_dim=int(v5_cfg.get("visual_dim", 768)),
    mola_dim=int(v5_cfg.get("mola_dim", 25)),
    hidden_dim=int(v5_cfg.get("hidden_dim", 256)),
    num_classes=int(v5_cfg.get("num_classes", 4)),
)
v5_model.load_state_dict(v5_ckpt["model_state_dict"], strict=False)
v5_model.eval()

emb_t = torch.from_numpy(emb_val).float()
mola_t = torch.from_numpy(mola_val).float()

with torch.no_grad():
    v5_logits = v5_model(emb_t, mola_t)
    # Visual-only: pass zeros for MOLA
    v5_vis_logits = v5_model(emb_t, torch.zeros_like(mola_t))

v5_preds = v5_logits.argmax(1).numpy()
v5_vis_preds = v5_vis_logits.argmax(1).numpy()

v5_acc, v5_f1, v5_f1s = compute_metrics(v5_preds, labels_val)
v5v_acc, v5v_f1, v5v_f1s = compute_metrics(v5_vis_preds, labels_val)

print(f"  Val Acc:       {v5_acc:.4f}")
print(f"  Val Macro F1:  {v5_f1:.4f}")
for c in range(4):
    print(f"    {CLASS_NAMES[c]}: F1={v5_f1s[c]:.4f}")
print(f"  Visual-only F1: {v5v_f1:.4f}")
for c in range(4):
    print(f"    {CLASS_NAMES[c]}: F1={v5v_f1s[c]:.4f}")

# Logit stats
v5_logits_np = v5_logits.numpy()
print(f"\n  Logit ranges:")
for c in range(4):
    col = v5_logits_np[:, c]
    print(f"    {CLASS_NAMES[c]}: [{col.min():.1f}, {col.max():.1f}], mean={col.mean():.2f}")

# ESP_024943 MOLA test
esp_mola = np.zeros((1, 25), dtype=np.float32)
esp_mola[0, 0] = 0.1055; esp_mola[0, 21] = -4251.0; esp_mola[0, 22] = 54.34
with torch.no_grad():
    esp_logits_v5 = v5_model(torch.randn(1, 768), torch.from_numpy(esp_mola).float())
print(f"\n  ESP_024943 (random vis + real MOLA):")
print(f"    Logits: {esp_logits_v5.numpy().flatten()}")
print(f"    Pred: {CLASS_NAMES[esp_logits_v5.argmax().item()]}")

# ===== V6b Late Fusion =====
print("\n" + "="*70)
print("V6b Late Fusion (MOLA Normalized)")
print("="*70)
from scripts.marslandform_v2.models.late_fusion_classifier import LateFusionClassifier

v6_ckpt = torch.load(f"{DATA_DIR}/late_fusion_v6b.pt", map_location="cpu", weights_only=False)
v6_cfg = v6_ckpt["cfg"]
v6_model = LateFusionClassifier(
    visual_dim=v6_cfg["visual_dim"], mola_dim=v6_cfg["mola_dim"],
    visual_hidden=v6_cfg["visual_hidden"], mola_hidden=v6_cfg["mola_hidden"],
    num_classes=v6_cfg["num_classes"], dropout=v6_cfg["dropout"],
)
v6_model.load_state_dict(v6_ckpt["model_state_dict"])
v6_model.eval()

with torch.no_grad():
    v6_logits = v6_model(emb_t, mola_t)
    v6_vis_logits = v6_model.get_visual_logits(emb_t)
    v6_mola_logits = v6_model.get_mola_logits(mola_t)

v6_preds = v6_logits.argmax(1).numpy()
v6_vis_preds = v6_vis_logits.argmax(1).numpy()
v6_mola_preds = v6_mola_logits.argmax(1).numpy()

v6_acc, v6_f1, v6_f1s = compute_metrics(v6_preds, labels_val)
v6v_acc, v6v_f1, v6v_f1s = compute_metrics(v6_vis_preds, labels_val)
v6m_acc, v6m_f1, v6m_f1s = compute_metrics(v6_mola_preds, labels_val)

print(f"  Weights: vis={v6_model.vis_weight.item():.3f}, mola={v6_model.mola_weight.item():.3f}")
print(f"  Val Acc:       {v6_acc:.4f}")
print(f"  Val Macro F1:  {v6_f1:.4f}")
for c in range(4):
    print(f"    {CLASS_NAMES[c]}: F1={v6_f1s[c]:.4f}")
print(f"  Visual-only F1: {v6v_f1:.4f}")
for c in range(4):
    print(f"    {CLASS_NAMES[c]}: F1={v6v_f1s[c]:.4f}")
print(f"  MOLA-only F1:  {v6m_f1:.4f}")

# Logit stats
v6_logits_np = v6_logits.numpy()
print(f"\n  Combined logit ranges:")
for c in range(4):
    col = v6_logits_np[:, c]
    print(f"    {CLASS_NAMES[c]}: [{col.min():.1f}, {col.max():.1f}], mean={col.mean():.2f}")

v6m_np = v6_mola_logits.numpy()
print(f"\n  MOLA logit ranges:")
for c in range(4):
    col = v6m_np[:, c]
    print(f"    {CLASS_NAMES[c]}: [{col.min():.1f}, {col.max():.1f}], mean={col.mean():.2f}")

# ESP_024943 MOLA test
with torch.no_grad():
    esp_mola_logits = v6_model.get_mola_logits(torch.from_numpy(esp_mola).float())
print(f"\n  ESP_024943 MOLA logits: {esp_mola_logits.numpy().flatten()}")
print(f"    MOLA pred: {CLASS_NAMES[esp_mola_logits.argmax().item()]}")

# ===== Summary =====
print("\n" + "="*70)
print("SUMMARY COMPARISON")
print("="*70)
print(f"{'':>20} | {'V5c FiLM':>12} | {'V6b LateFusion':>14}")
print("-"*55)
print(f"{'Val Macro F1':>20} | {v5_f1:>12.4f} | {v6_f1:>14.4f}")
print(f"{'Val Accuracy':>20} | {v5_acc:>12.4f} | {v6_acc:>14.4f}")
print(f"{'Visual-only F1':>20} | {v5v_f1:>12.4f} | {v6v_f1:>14.4f}")
print(f"{'MOLA-only F1':>20} | {'N/A':>12} | {v6m_f1:>14.4f}")
for c in range(4):
    print(f"{'F1 ' + CLASS_NAMES[c]:>20} | {v5_f1s[c]:>12.4f} | {v6_f1s[c]:>14.4f}")
print(f"{'ESP024943 MOLA pred':>20} | {'OTHER':>12} | {CLASS_NAMES[esp_mola_logits.argmax().item()]:>14}")
