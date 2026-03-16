#!/usr/bin/env python3
"""
Analyze false positive rate: In HiRISE images that contain Levy polygons,
are the OTHER tiles (tiles NOT in Levy polygons) correctly classified as OTHER?
Or does the model over-classify them as LDA/LVF/CCF?

This checks BOTH V5c and V6b models.
"""
import sys, json, numpy as np, torch
from collections import defaultdict

sys.path.insert(0, "/disk1/cspark/hirise-api")
sys.path.insert(0, "/disk1/cspark/MarsLab")

DATA_DIR = "/disk1/cspark/MarsLab/Data/HiRISE/v5_retrain"
CLASS_NAMES = ["LDA", "LVF", "CCF", "OTHER"]

# Load data
embeddings = np.load(f"{DATA_DIR}/embeddings_v5.npy")
mola_dict = np.load(f"{DATA_DIR}/mola_features_v5.npy", allow_pickle=True).item()
with open(f"{DATA_DIR}/tile_labels_v5.json") as f:
    labels_list = json.load(f)

emb_list, mola_list, label_list, pids_list = [], [], [], []
for i, tile_info in enumerate(labels_list):
    pid = tile_info["image_id"]
    key = f"{tile_info['tile_row']}_{tile_info['tile_col']}"
    if pid in mola_dict and key in mola_dict[pid]:
        emb_list.append(i)
        mola_list.append(mola_dict[pid][key])
        lbl = tile_info["label"]
        label_list.append(CLASS_NAMES.index(lbl) if lbl in CLASS_NAMES else 3)
        pids_list.append(pid)

embeddings = embeddings[emb_list].astype(np.float32)
mola = np.array(mola_list, dtype=np.float32)
labels = np.array(label_list, dtype=np.int64)
pids = np.array(pids_list)

# Find products that have BOTH glacial (LDA/LVF/CCF) AND OTHER tiles
# These are the Levy-polygon-containing images
product_classes = defaultdict(set)
product_indices = defaultdict(list)
for i, (pid, lbl) in enumerate(zip(pids, labels)):
    product_classes[pid].add(int(lbl))
    product_indices[pid].append(i)

# Products with at least one glacial class + OTHER
mixed_products = [pid for pid, classes in product_classes.items()
                  if classes & {0, 1, 2} and 3 in classes]
glacial_only = [pid for pid, classes in product_classes.items()
                if classes & {0, 1, 2} and 3 not in classes]
other_only = [pid for pid, classes in product_classes.items()
              if classes == {3}]

print(f"Total products: {len(product_classes)}")
print(f"  Mixed (glacial + OTHER): {len(mixed_products)}")
print(f"  Glacial only: {len(glacial_only)}")
print(f"  OTHER only: {len(other_only)}")

# Get indices for mixed products
mixed_idx = []
for pid in mixed_products:
    mixed_idx.extend(product_indices[pid])
mixed_idx = np.array(mixed_idx)

mixed_labels = labels[mixed_idx]
mixed_glacial_mask = mixed_labels < 3  # LDA/LVF/CCF
mixed_other_mask = mixed_labels == 3

print(f"\nIn mixed products:")
print(f"  Total tiles: {len(mixed_idx)}")
print(f"  Glacial tiles (LDA/LVF/CCF): {mixed_glacial_mask.sum()}")
print(f"  OTHER tiles: {mixed_other_mask.sum()}")
for c in range(4):
    print(f"    {CLASS_NAMES[c]}: {(mixed_labels == c).sum()}")

# ===== Load both models =====
emb_t = torch.from_numpy(embeddings[mixed_idx]).float()
mola_t = torch.from_numpy(mola[mixed_idx]).float()

# V5c FiLM
from scripts.marslandform_v2.models.film_classifier import FiLMClassifier
v5_ckpt = torch.load(f"{DATA_DIR}/film_classifier_v5c.pt", map_location="cpu", weights_only=False)
v5_cfg = v5_ckpt.get("cfg", {})
v5_model = FiLMClassifier(
    visual_dim=int(v5_cfg.get("visual_dim", 768)),
    mola_dim=int(v5_cfg.get("mola_dim", 25)),
    num_classes=int(v5_cfg.get("num_classes", 4)),
)
v5_model.load_state_dict(v5_ckpt["model_state_dict"], strict=False)
v5_model.eval()

# V6b Late Fusion
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

# Get predictions
with torch.no_grad():
    v5_logits = v5_model(emb_t, mola_t)
    v5_preds = v5_logits.argmax(1).numpy()
    
    v6_logits = v6_model(emb_t, mola_t)
    v6_preds = v6_logits.argmax(1).numpy()
    v6_vis_preds = v6_model.get_visual_logits(emb_t).argmax(1).numpy()
    v6_mola_preds = v6_model.get_mola_logits(mola_t).argmax(1).numpy()

def analyze_model(name, preds, labels, glacial_mask, other_mask):
    print(f"\n{'='*70}")
    print(f"{name}")
    print(f"{'='*70}")
    
    # 1. Glacial tiles: recall (correctly classified as their class)
    print(f"\n[Glacial tiles recall] — Levy가 마킹한 타일을 제대로 분류하나?")
    for c in range(3):
        mask_c = labels == c
        if mask_c.sum() > 0:
            correct = (preds[mask_c] == c).sum()
            total = mask_c.sum()
            print(f"  {CLASS_NAMES[c]}: {correct}/{total} = {correct/total:.1%}")
    
    glacial_correct = sum((preds[labels == c] == c).sum() for c in range(3))
    glacial_total = glacial_mask.sum()
    print(f"  전체 glacial recall: {glacial_correct}/{glacial_total} = {glacial_correct/glacial_total:.1%}")
    
    # 2. OTHER tiles: are they correctly classified as OTHER? (specificity)
    print(f"\n[OTHER 타일 정확도] — Levy polygon 밖 타일이 OTHER로 분류되나?")
    other_correct = (preds[other_mask] == 3).sum()
    other_total = other_mask.sum()
    other_as_glacial = (preds[other_mask] < 3).sum()
    print(f"  OTHER → OTHER (정확): {other_correct}/{other_total} = {other_correct/other_total:.1%}")
    print(f"  OTHER → LDA/LVF/CCF (오분류): {other_as_glacial}/{other_total} = {other_as_glacial/other_total:.1%}")
    
    # Break down false positives
    for c in range(3):
        fp = (preds[other_mask] == c).sum()
        if fp > 0:
            print(f"    OTHER → {CLASS_NAMES[c]}: {fp} tiles")
    
    # 3. Per-product analysis
    print(f"\n[Per-product false positive rate] — 이미지별 OTHER 오분류율")
    fp_rates = []
    print(f"  {'Product':>25} | {'OTHER tiles':>11} | {'→ OTHER':>8} | {'→ Glacial':>10} | {'FP rate':>8}")
    print(f"  {'-'*75}")
    
    product_fps = []
    for pid in mixed_products:
        idx_local = [j for j, i in enumerate(mixed_idx) if pids[i] == pid]
        if not idx_local:
            continue
        local_labels = mixed_labels[idx_local]
        local_preds = preds[idx_local]
        local_other = local_labels == 3
        if local_other.sum() == 0:
            continue
        
        n_other = local_other.sum()
        n_correct = (local_preds[local_other] == 3).sum()
        n_fp = (local_preds[local_other] < 3).sum()
        fp_rate = n_fp / n_other
        fp_rates.append(fp_rate)
        product_fps.append((pid, int(n_other), int(n_correct), int(n_fp), fp_rate))
    
    # Sort by FP rate descending
    product_fps.sort(key=lambda x: -x[4])
    for pid, n_other, n_correct, n_fp, fp_rate in product_fps[:15]:
        print(f"  {pid:>25} | {n_other:>11} | {n_correct:>8} | {n_fp:>10} | {fp_rate:>7.1%}")
    if len(product_fps) > 15:
        print(f"  ... ({len(product_fps) - 15} more products)")
    
    avg_fp = np.mean(fp_rates) if fp_rates else 0
    print(f"\n  평균 FP rate: {avg_fp:.1%}")
    
    return {
        "glacial_recall": glacial_correct / glacial_total,
        "other_accuracy": other_correct / other_total,
        "false_positive_rate": other_as_glacial / other_total,
        "avg_per_product_fp": avg_fp,
    }

v5_stats = analyze_model("V5c FiLM", v5_preds, mixed_labels, mixed_glacial_mask, mixed_other_mask)
v6_stats = analyze_model("V6b Late Fusion (Normalized)", v6_preds, mixed_labels, mixed_glacial_mask, mixed_other_mask)

# Also show V6b visual-only and MOLA-only
analyze_model("V6b Visual-only", v6_vis_preds, mixed_labels, mixed_glacial_mask, mixed_other_mask)
analyze_model("V6b MOLA-only", v6_mola_preds, mixed_labels, mixed_glacial_mask, mixed_other_mask)

# Final comparison
print(f"\n{'='*70}")
print(f"SUMMARY")
print(f"{'='*70}")
print(f"{'':>30} | {'V5c FiLM':>10} | {'V6b LF':>10}")
print(f"{'-'*55}")
print(f"{'Glacial recall':>30} | {v5_stats['glacial_recall']:>9.1%} | {v6_stats['glacial_recall']:>9.1%}")
print(f"{'OTHER accuracy':>30} | {v5_stats['other_accuracy']:>9.1%} | {v6_stats['other_accuracy']:>9.1%}")
print(f"{'False positive rate':>30} | {v5_stats['false_positive_rate']:>9.1%} | {v6_stats['false_positive_rate']:>9.1%}")
print(f"{'Avg per-product FP rate':>30} | {v5_stats['avg_per_product_fp']:>9.1%} | {v6_stats['avg_per_product_fp']:>9.1%}")
