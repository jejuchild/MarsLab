#!/usr/bin/env python3
"""Optimize per-class confidence thresholds for v5c FiLM classifier.

Current pipeline: export_geojson.py uses --confidence-threshold 0.5 (global).
Problem: LDA tiles often have softmax ~0.6-0.7, getting filtered or confused with LVF.

This script:
  1. Sweeps global + per-class thresholds on val/test
  2. Finds optimal values maximizing landform F1
  3. Shows impact on key presentation images
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, classification_report

ROOT = Path("/disk1/cspark/hirise-api")
MARSLAB = Path("/disk1/cspark/MarsLab")
V5_DIR = MARSLAB / "Data/HiRISE/v5_retrain"
V4_DIR = MARSLAB / "Data/HiRISE/v4_colab_data_expanded"

if not V5_DIR.exists():
    V5_DIR = ROOT / "data/HiRISE/v5_retrain"
if not V4_DIR.exists():
    V4_DIR = ROOT / "data/HiRISE/v4_colab_data_expanded"

sys.path.insert(0, str(ROOT))
CLASS_NAMES = ["LDA", "LVF", "CCF", "OTHER"]

SHOWCASE_IMAGES = [
    "ESP_026058_2230",  # User's preferred (LDA+LVF balanced)
    "ESP_074782_1435",  # Best margin (LDA+LVF)
    "ESP_016391_2195",  # Rank 2 margin
    "ESP_050255_2195",  # CCF+LDA
    "ESP_080217_2230",  # CCF+LDA
]


def load_model(model_path):
    from scripts.marslandform_v2.models.film_classifier import FiLMClassifier
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("cfg", {})
    model = FiLMClassifier(
        visual_dim=cfg.get("visual_dim", 768), mola_dim=cfg.get("mola_dim", 25),
        num_classes=cfg.get("num_classes", 4), film_hidden=cfg.get("film_hidden", 64),
        head_hidden=cfg.get("head_hidden", 128), dropout=cfg.get("dropout", 0.4),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def main():
    print("=" * 70)
    print("V5c Per-Class Threshold Optimization")
    print("=" * 70)

    # Load data
    with open(V4_DIR / "tile_index.json") as f:
        tile_index = json.load(f)
    embeddings = np.load(str(V5_DIR / "embeddings_v5.npy"))
    mola = np.load(str(V5_DIR / "mola_features_v5.npy"), allow_pickle=True).item()
    with open(V5_DIR / "tile_labels_v5.json") as f:
        labels_list = json.load(f)
    with open(V5_DIR / "tile_splits_v5.json") as f:
        splits = json.load(f)
    model = load_model(V5_DIR / "film_classifier_v5c.pt")
    print("Data loaded.")

    # Build arrays
    labels_raw, label_meta = {}, {}
    for entry in labels_list:
        key = f"{entry['image_id']}_{entry['tile_row']}_{entry['tile_col']}"
        labels_raw[key] = entry.get("label", "OTHER")
        label_meta[key] = entry

    c2i = {c: i for i, c in enumerate(CLASS_NAMES)}
    tile_keys = list(tile_index.keys())
    n = len(tile_keys)
    mola_arr = np.zeros((n, 25), dtype=np.float32)
    label_arr = np.full(n, -1, dtype=np.int64)
    tile_image_ids = []

    for i, key in enumerate(tile_keys):
        parts = key.rsplit("_", 2)
        img_id, rc = parts[0], f"{parts[1]}_{parts[2]}"
        tile_image_ids.append(img_id)
        if img_id in mola and rc in mola[img_id]:
            mola_arr[i] = mola[img_id][rc]
        if key in labels_raw and labels_raw[key] in c2i:
            label_arr[i] = c2i[labels_raw[key]]

    # Split indices
    val_set = set(splits["val"])
    test_set = set(splits["test"])

    labeled_mask = label_arr >= 0
    labeled_indices = np.where(labeled_mask)[0]

    val_indices = np.array([i for i in labeled_indices if i in val_set])
    test_indices = np.array([i for i in labeled_indices if i in test_set])
    print(f"Val: {len(val_indices)} tiles, Test: {len(test_indices)} tiles")

    # Inference on all labeled
    print("Running inference...")
    emb_t = torch.tensor(embeddings[labeled_indices], dtype=torch.float32)
    mola_t = torch.tensor(mola_arr[labeled_indices], dtype=torch.float32)
    all_probs = []
    with torch.no_grad():
        for s in range(0, len(labeled_indices), 512):
            e = min(s + 512, len(labeled_indices))
            logits = model(emb_t[s:e], mola_t[s:e])
            all_probs.append(F.softmax(logits, dim=1).numpy())
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = label_arr[labeled_indices]

    # Map labeled_indices position to val/test
    labeled_to_pos = {gi: j for j, gi in enumerate(labeled_indices)}
    val_pos = np.array([labeled_to_pos[i] for i in val_indices if i in labeled_to_pos])
    test_pos = np.array([labeled_to_pos[i] for i in test_indices if i in labeled_to_pos])

    val_probs = all_probs[val_pos]
    val_labels = all_labels[val_pos]
    test_probs = all_probs[test_pos]
    test_labels = all_labels[test_pos]

    # ═══════════════════════════════════════════════════════════════════
    # 1. Baseline: argmax (no threshold)
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("1. BASELINE — Argmax (no threshold)")
    print("=" * 70)
    baseline_preds = test_probs.argmax(axis=1)
    print(classification_report(test_labels, baseline_preds, target_names=CLASS_NAMES, digits=4))
    baseline_f1 = f1_score(test_labels, baseline_preds, average="macro")
    lf_mask = test_labels != c2i["OTHER"]
    baseline_lf_f1 = f1_score(test_labels[lf_mask], baseline_preds[lf_mask], average="macro")
    print(f"  Macro F1: {baseline_f1:.4f} | Landform F1: {baseline_lf_f1:.4f}")

    # ═══════════════════════════════════════════════════════════════════
    # 2. Global threshold sweep
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("2. GLOBAL THRESHOLD SWEEP (val set)")
    print("=" * 70)
    print(f"{'Threshold':>10} {'Macro F1':>9} {'LF F1':>8} {'LDA F1':>8} {'LVF F1':>8} {'CCF F1':>8} {'Filtered':>9}")
    print("-" * 70)

    best_global_t = 0.0
    best_global_f1 = 0.0

    for t_int in range(20, 80, 2):
        t = t_int / 100.0
        preds = apply_global_threshold(val_probs, t, c2i["OTHER"])
        mf1 = f1_score(val_labels, preds, average="macro")
        lf_m = val_labels != c2i["OTHER"]
        lf_f1 = f1_score(val_labels[lf_m], preds[lf_m], average="macro") if lf_m.sum() > 0 else 0

        per_cls = {}
        for ci, cn in enumerate(CLASS_NAMES):
            cm = val_labels == ci
            if cm.sum() > 0:
                per_cls[cn] = f1_score(cm, preds == ci)
            else:
                per_cls[cn] = 0.0

        filtered = (preds == c2i["OTHER"]).sum() - (val_labels == c2i["OTHER"]).sum()
        print(f"{t:>10.2f} {mf1:>9.4f} {lf_f1:>8.4f} "
              f"{per_cls['LDA']:>8.4f} {per_cls['LVF']:>8.4f} {per_cls['CCF']:>8.4f} "
              f"{filtered:>+9d}")

        if lf_f1 > best_global_f1:
            best_global_f1 = lf_f1
            best_global_t = t

    print(f"\n  Best global threshold: {best_global_t:.2f} (LF F1 = {best_global_f1:.4f})")

    # Apply to test
    test_preds_global = apply_global_threshold(test_probs, best_global_t, c2i["OTHER"])
    test_f1_global = f1_score(test_labels, test_preds_global, average="macro")
    test_lf_f1_global = f1_score(test_labels[lf_mask], test_preds_global[lf_mask], average="macro")
    print(f"  Test: Macro F1 = {test_f1_global:.4f} | LF F1 = {test_lf_f1_global:.4f}")

    # ═══════════════════════════════════════════════════════════════════
    # 3. Per-class threshold optimization (val set)
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("3. PER-CLASS THRESHOLD OPTIMIZATION (val set)")
    print("=" * 70)

    per_class_thresholds = np.full(len(CLASS_NAMES), 0.5)
    for ci, cn in enumerate(CLASS_NAMES):
        binary_true = (val_labels == ci).astype(int)
        if binary_true.sum() == 0:
            continue

        best_f1_cls = -1
        best_t_cls = 0.5

        for t_int in range(15, 85):
            t = t_int / 100.0
            binary_pred = (val_probs[:, ci] >= t).astype(int)
            tp = ((binary_pred == 1) & (binary_true == 1)).sum()
            fp = ((binary_pred == 1) & (binary_true == 0)).sum()
            fn = ((binary_pred == 0) & (binary_true == 1)).sum()
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

            if f1 > best_f1_cls:
                best_f1_cls = f1
                best_t_cls = t

        per_class_thresholds[ci] = best_t_cls
        print(f"  {cn}: optimal threshold = {best_t_cls:.2f} (val binary F1 = {best_f1_cls:.4f})")

    # Apply per-class thresholds to test
    test_preds_perclass = apply_perclass_threshold(test_probs, per_class_thresholds)
    test_f1_perclass = f1_score(test_labels, test_preds_perclass, average="macro")
    test_lf_f1_perclass = f1_score(test_labels[lf_mask], test_preds_perclass[lf_mask], average="macro")
    print(f"\n  Test with per-class thresholds:")
    print(f"  Macro F1 = {test_f1_perclass:.4f} | LF F1 = {test_lf_f1_perclass:.4f}")
    print(classification_report(test_labels, test_preds_perclass, target_names=CLASS_NAMES, digits=4))

    # ═══════════════════════════════════════════════════════════════════
    # 4. Summary
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Method':<30} {'Macro F1':>9} {'LF F1':>8} {'Δ LF F1':>8}")
    print("-" * 60)
    print(f"{'Baseline (argmax)':<30} {baseline_f1:>9.4f} {baseline_lf_f1:>8.4f} {'—':>8}")
    print(f"{'Global t={best_global_t:.2f}':<30} {test_f1_global:>9.4f} {test_lf_f1_global:>8.4f} {test_lf_f1_global-baseline_lf_f1:>+8.4f}")
    thstr = "/".join(f"{t:.2f}" for t in per_class_thresholds)
    print(f"{'Per-class [' + thstr + ']':<30} {test_f1_perclass:>9.4f} {test_lf_f1_perclass:>8.4f} {test_lf_f1_perclass-baseline_lf_f1:>+8.4f}")

    # ═══════════════════════════════════════════════════════════════════
    # 5. Impact on showcase images
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("SHOWCASE IMAGE IMPACT")
    print("=" * 70)

    # Build image-level probs
    image_tiles = defaultdict(list)
    for j, gi in enumerate(labeled_indices):
        img_id = tile_image_ids[gi]
        label = int(all_labels[j])
        probs_vec = all_probs[j]
        image_tiles[img_id].append({"label": label, "probs": probs_vec})

    for img_id in SHOWCASE_IMAGES:
        if img_id not in image_tiles:
            print(f"\n  {img_id}: NOT IN DATASET")
            continue

        tiles = image_tiles[img_id]
        labels_img = np.array([t["label"] for t in tiles])
        probs_img = np.array([t["probs"] for t in tiles])

        print(f"\n  {img_id} ({len(tiles)} tiles)")
        print(f"    GT distribution: {dict(Counter(CLASS_NAMES[l] for l in labels_img))}")

        # Argmax
        preds_argmax = probs_img.argmax(axis=1)
        print(f"    Argmax preds:    {dict(Counter(CLASS_NAMES[p] for p in preds_argmax))}")

        # Global threshold
        preds_global = apply_global_threshold(probs_img, best_global_t, c2i["OTHER"])
        print(f"    Global t={best_global_t:.2f}:   {dict(Counter(CLASS_NAMES[p] for p in preds_global))}")

        # Per-class threshold
        preds_pc = apply_perclass_threshold(probs_img, per_class_thresholds)
        print(f"    Per-class:       {dict(Counter(CLASS_NAMES[p] for p in preds_pc))}")

        # Current pipeline default (t=0.50)
        preds_050 = apply_global_threshold(probs_img, 0.50, c2i["OTHER"])
        print(f"    Pipeline t=0.50: {dict(Counter(CLASS_NAMES[p] for p in preds_050))}")

        # Show LDA tile confidence details
        lda_gt_mask = labels_img == c2i["LDA"]
        if lda_gt_mask.sum() > 0:
            lda_probs = probs_img[lda_gt_mask]
            print(f"    LDA tiles ({lda_gt_mask.sum()}):")
            print(f"      P(LDA): {lda_probs[:, c2i['LDA']].mean():.3f} ± {lda_probs[:, c2i['LDA']].std():.3f} "
                  f"[{lda_probs[:, c2i['LDA']].min():.3f} ~ {lda_probs[:, c2i['LDA']].max():.3f}]")
            print(f"      P(LVF): {lda_probs[:, c2i['LVF']].mean():.3f} ± {lda_probs[:, c2i['LVF']].std():.3f}")

    # Save results
    output_dir = ROOT / "results" / "levy_match_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "baseline_macro_f1": float(baseline_f1),
        "baseline_landform_f1": float(baseline_lf_f1),
        "best_global_threshold": float(best_global_t),
        "global_macro_f1": float(test_f1_global),
        "global_landform_f1": float(test_lf_f1_global),
        "per_class_thresholds": {cn: float(per_class_thresholds[ci]) for ci, cn in enumerate(CLASS_NAMES)},
        "perclass_macro_f1": float(test_f1_perclass),
        "perclass_landform_f1": float(test_lf_f1_perclass),
    }
    with open(output_dir / "threshold_optimization.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {output_dir / 'threshold_optimization.json'}")


def apply_global_threshold(probs, threshold, other_idx):
    """If max prob < threshold, classify as OTHER."""
    preds = probs.argmax(axis=1)
    max_conf = probs.max(axis=1)
    preds[max_conf < threshold] = other_idx
    return preds


def apply_perclass_threshold(probs, thresholds):
    """Per-class: scale by threshold, pick highest passing class."""
    preds = []
    for i in range(len(probs)):
        p = probs[i]
        scaled = p / thresholds
        passing = p >= thresholds
        if passing.any():
            masked = np.where(passing, scaled, -np.inf)
            preds.append(int(np.argmax(masked)))
        else:
            preds.append(int(np.argmax(p)))
    return np.array(preds)


if __name__ == "__main__":
    main()
