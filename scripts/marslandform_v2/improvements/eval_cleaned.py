#!/usr/bin/env python3
"""Evaluate the cleaned-labels retrained model (saved at best epoch) + threshold opt."""
import json
import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from copy import deepcopy
from scripts.marslandform_v2.config import CLASS_ORDER, get_config
from scripts.marslandform_v2.models.mil_classifier import (
    AttentionMILClassifier,
    MILDataset,
    load_embeddings,
    load_labels,
    load_mola_features,
    mil_collate_fn,
    set_seed,
    compute_metrics,
)
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

DATA_ROOT = ROOT / "Data/HiRISE/v2_output"


def eval_model_on_split(model, loader, device, num_classes=5):
    """Evaluate model and return predictions + true labels."""
    model.eval()
    all_preds = []
    all_true = []
    all_probs = []
    all_ids = []
    
    with torch.no_grad():
        for batch in loader:
            tiles = batch["tile_embeddings"].to(device)
            mask = batch["tile_mask"].to(device)
            mola = batch["mola_features"].to(device)
            labels = batch["labels"]
            
            logits, _ = model(tiles, mask, mola)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(dim=1).cpu().tolist()
            
            all_preds.extend(preds)
            all_true.extend(labels.tolist())
            all_probs.append(probs)
            all_ids.extend(batch["image_ids"])
    
    all_probs = np.concatenate(all_probs, axis=0)
    return all_preds, all_true, all_probs, all_ids


def threshold_optimize(val_probs, val_true, test_probs, test_true, num_classes=5):
    """Find optimal per-class thresholds on val set, apply to test set."""
    # Grid search thresholds on validation set
    best_thresholds = [0.5] * num_classes
    best_f1 = 0.0
    
    # Try different thresholds per class
    threshold_candidates = np.arange(0.10, 0.60, 0.03)
    
    for c in range(num_classes):
        best_c_f1 = 0.0
        best_c_thr = 0.5
        
        for thr in threshold_candidates:
            # Adjust class c threshold
            test_thresholds = best_thresholds.copy()
            test_thresholds[c] = thr
            
            # Predict with these thresholds
            adjusted_scores = val_probs.copy()
            for j in range(num_classes):
                adjusted_scores[:, j] = adjusted_scores[:, j] / test_thresholds[j]
            
            preds = adjusted_scores.argmax(axis=1).tolist()
            
            # Compute landform F1 (exclude background)
            lf_f1s = []
            for cls_idx in range(4):  # LDA, LVF, CCF, GLF
                tp = sum(1 for p, t in zip(preds, val_true) if p == cls_idx and t == cls_idx)
                fp = sum(1 for p, t in zip(preds, val_true) if p == cls_idx and t != cls_idx)
                fn = sum(1 for p, t in zip(preds, val_true) if p != cls_idx and t == cls_idx)
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
                lf_f1s.append(f1)
            
            lf_f1 = np.mean(lf_f1s)
            if lf_f1 > best_c_f1:
                best_c_f1 = lf_f1
                best_c_thr = thr
        
        best_thresholds[c] = best_c_thr
    
    # Apply best thresholds to test set
    adjusted_test = test_probs.copy()
    for j in range(num_classes):
        adjusted_test[:, j] = adjusted_test[:, j] / best_thresholds[j]
    
    test_preds = adjusted_test.argmax(axis=1).tolist()
    test_metrics = compute_metrics(test_true, test_preds, num_classes=num_classes)
    
    return best_thresholds, test_preds, test_metrics


def main():
    print("=" * 60)
    print("EVALUATE CLEANED V4 MODEL + THRESHOLD OPT")
    print("=" * 60)
    
    set_seed(42)
    cfg = get_config()
    mil_cfg = cfg.mil
    device = torch.device("cpu")
    
    model_path = DATA_ROOT / "models/cleaned_v4/best_mil_model.pt"
    if not model_path.exists():
        print(f"ERROR: No model found at {model_path}")
        return
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    saved_cfg = checkpoint.get("mil_config", {})
    model_cfg = deepcopy(mil_cfg)
    for k, v in saved_cfg.items():
        if hasattr(model_cfg, k):
            setattr(model_cfg, k, v)
    
    model = AttentionMILClassifier(model_cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded model from epoch {checkpoint.get('best_epoch', '?')}")
    
    # Load data
    emb_dict = load_embeddings(DATA_ROOT / "embeddings_mil")
    mola_dict = load_mola_features(DATA_ROOT / "mola_features_by_image.npy")
    labels_dict = load_labels(DATA_ROOT / "label_audit/labels_cleaned.json")
    
    # Use canonical split
    split_path = DATA_ROOT / "models/multihead_improved/data_split.json"
    split = json.loads(split_path.read_text())
    
    # Build datasets with cleaned labels
    valid_ids = set(emb_dict.keys()) & set(mola_dict.keys()) & set(labels_dict.keys())
    train_ids = [i for i in split["train_ids"] if i in valid_ids]
    val_ids = [i for i in split["val_ids"] if i in valid_ids]
    test_ids = [i for i in split["test_ids"] if i in valid_ids]
    
    print(f"Split: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")
    
    val_ds = MILDataset(val_ids, emb_dict, mola_dict, labels_dict, min_tiles_per_image=1, max_tiles_per_image=128)
    test_ds = MILDataset(test_ids, emb_dict, mola_dict, labels_dict, min_tiles_per_image=1, max_tiles_per_image=128)
    
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=mil_collate_fn, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=mil_collate_fn, num_workers=0)
    
    # Evaluate on test set (raw)
    print("\n--- Raw Predictions ---")
    test_preds, test_true, test_probs, test_ids_out = eval_model_on_split(model, test_loader, device)
    raw_metrics = compute_metrics(test_true, test_preds, num_classes=5)
    
    print(f"  Macro F1: {raw_metrics['macro_f1_all']:.4f}")
    print(f"  Landform F1: {raw_metrics['landform_macro_f1']:.4f}")
    for i, cls in enumerate(CLASS_ORDER):
        print(f"    {cls}: P={raw_metrics['precision'][i]:.3f} R={raw_metrics['recall'][i]:.3f} F1={raw_metrics['f1'][i]:.3f}")
    
    # Evaluate on val for threshold opt
    val_preds, val_true, val_probs, _ = eval_model_on_split(model, val_loader, device)
    
    # Threshold optimization
    print("\n--- Threshold Optimization ---")
    thresholds, opt_preds, opt_metrics = threshold_optimize(val_probs, val_true, test_probs, test_true)
    
    print(f"  Thresholds: {dict(zip(CLASS_ORDER, [f'{t:.2f}' for t in thresholds]))}")
    print(f"  Macro F1: {opt_metrics['macro_f1_all']:.4f}")
    print(f"  Landform F1: {opt_metrics['landform_macro_f1']:.4f}")
    for i, cls in enumerate(CLASS_ORDER):
        print(f"    {cls}: P={opt_metrics['precision'][i]:.3f} R={opt_metrics['recall'][i]:.3f} F1={opt_metrics['f1'][i]:.3f}")
    
    # Save results
    out_dir = DATA_ROOT / "models/cleaned_v4"
    results = {
        "raw": {
            "macro_f1_all": raw_metrics["macro_f1_all"],
            "landform_macro_f1": raw_metrics["landform_macro_f1"],
            "precision": raw_metrics["precision"],
            "recall": raw_metrics["recall"],
            "f1": raw_metrics["f1"],
        },
        "threshold_optimized": {
            "thresholds": dict(zip(CLASS_ORDER, thresholds)),
            "macro_f1_all": opt_metrics["macro_f1_all"],
            "landform_macro_f1": opt_metrics["landform_macro_f1"],
            "precision": opt_metrics["precision"],
            "recall": opt_metrics["recall"],
            "f1": opt_metrics["f1"],
        },
    }
    (out_dir / "test_metrics.json").write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_dir / 'test_metrics.json'}")
    
    # Summary comparison
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"  V3 Baseline (original labels):     Landform F1 = 0.571")
    print(f"  V3 + Threshold Opt (original):     Landform F1 = 0.618")
    print(f"  Cleaned V4 (raw):                  Landform F1 = {raw_metrics['landform_macro_f1']:.3f}")
    print(f"  Cleaned V4 + Threshold Opt:        Landform F1 = {opt_metrics['landform_macro_f1']:.3f}")
    print(f"{'='*60}")
    
    return results


if __name__ == "__main__":
    main()
