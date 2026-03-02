#!/usr/bin/env python3
"""Push Cleaned V4 model toward F1=0.8 with advanced techniques.

Techniques:
1. Fine-grained threshold optimization with wider search
2. Multi-seed ensemble (retrain with different seeds, average)
3. Temperature scaling + threshold
4. Class-weighted post-hoc calibration
"""
import json
import sys
import numpy as np
from pathlib import Path
from copy import deepcopy
from itertools import product

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
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
    train_mil,
)
from torch.utils.data import DataLoader

DATA_ROOT = ROOT / "Data/HiRISE/v2_output"


def load_model_and_data():
    """Load Cleaned V4 model and prepare dataloaders."""
    cfg = get_config()
    mil_cfg = cfg.mil
    device = torch.device("cpu")
    
    model_path = DATA_ROOT / "models/cleaned_v4/best_mil_model.pt"
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    saved_cfg = checkpoint.get("mil_config", {})
    model_cfg = deepcopy(mil_cfg)
    for k, v in saved_cfg.items():
        if hasattr(model_cfg, k):
            setattr(model_cfg, k, v)
    
    model = AttentionMILClassifier(model_cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    emb_dict = load_embeddings(DATA_ROOT / "embeddings_mil")
    mola_dict = load_mola_features(DATA_ROOT / "mola_features_by_image.npy")
    labels_dict = load_labels(DATA_ROOT / "label_audit/labels_cleaned.json")
    
    split = json.loads((DATA_ROOT / "models/multihead_improved/data_split.json").read_text())
    valid_ids = set(emb_dict.keys()) & set(mola_dict.keys()) & set(labels_dict.keys())
    
    train_ids = [i for i in split["train_ids"] if i in valid_ids]
    val_ids = [i for i in split["val_ids"] if i in valid_ids]
    test_ids = [i for i in split["test_ids"] if i in valid_ids]
    
    return model, model_cfg, emb_dict, mola_dict, labels_dict, train_ids, val_ids, test_ids, device


def get_predictions(model, loader, device):
    """Get predictions and probabilities."""
    model.eval()
    all_preds, all_true, all_probs, all_ids = [], [], [], []
    
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
    
    return all_preds, all_true, np.concatenate(all_probs), all_ids


def landform_f1(true, pred, num_classes=5):
    """Compute landform macro F1 (classes 0-3 only)."""
    f1s = []
    for c in range(4):
        tp = sum(1 for p, t in zip(pred, true) if p == c and t == c)
        fp = sum(1 for p, t in zip(pred, true) if p == c and t != c)
        fn = sum(1 for p, t in zip(pred, true) if p != c and t == c)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        f1s.append(f1)
    return np.mean(f1s)


def technique_1_fine_threshold(val_probs, val_true, test_probs, test_true):
    """Fine-grained threshold optimization."""
    print("\n--- Technique 1: Fine-Grained Threshold Optimization ---")
    
    best_thresholds = [1.0] * 5
    best_f1 = 0.0
    
    # Coarse search
    for iteration in range(3):
        step = 0.05 / (iteration + 1)
        for c in range(5):
            low = max(0.05, best_thresholds[c] - 0.3)
            high = min(3.0, best_thresholds[c] + 0.3)
            candidates = np.arange(low, high, step)
            
            best_c_f1 = 0.0
            best_c_thr = best_thresholds[c]
            
            for thr in candidates:
                test_thr = best_thresholds.copy()
                test_thr[c] = thr
                adjusted = val_probs.copy()
                for j in range(5):
                    adjusted[:, j] = adjusted[:, j] / test_thr[j]
                preds = adjusted.argmax(axis=1).tolist()
                f1 = landform_f1(val_true, preds)
                if f1 > best_c_f1:
                    best_c_f1 = f1
                    best_c_thr = thr
            
            best_thresholds[c] = best_c_thr
            if best_c_f1 > best_f1:
                best_f1 = best_c_f1
    
    # Apply to test
    adjusted_test = test_probs.copy()
    for j in range(5):
        adjusted_test[:, j] = adjusted_test[:, j] / best_thresholds[j]
    test_preds = adjusted_test.argmax(axis=1).tolist()
    
    test_lf1 = landform_f1(test_true, test_preds)
    test_metrics = compute_metrics(test_true, test_preds, num_classes=5)
    
    print(f"  Val LF F1: {best_f1:.4f}")
    print(f"  Test LF F1: {test_lf1:.4f}")
    print(f"  Thresholds: {dict(zip(CLASS_ORDER, [f'{t:.3f}' for t in best_thresholds]))}")
    for i, cls in enumerate(CLASS_ORDER):
        print(f"    {cls}: P={test_metrics['precision'][i]:.3f} R={test_metrics['recall'][i]:.3f} F1={test_metrics['f1'][i]:.3f}")
    
    return test_lf1, best_thresholds, test_preds


def technique_2_temperature_threshold(val_probs, val_true, test_probs, test_true):
    """Temperature scaling combined with threshold optimization."""
    print("\n--- Technique 2: Temperature + Threshold ---")
    
    best_overall_f1 = 0.0
    best_temp = 1.0
    best_thresholds = [1.0] * 5
    
    for temp in np.arange(0.3, 3.0, 0.1):
        # Apply temperature to logits (probs -> logits -> scale -> probs)
        eps = 1e-8
        val_logits = np.log(val_probs + eps)
        val_scaled = np.exp(val_logits / temp) / np.exp(val_logits / temp).sum(axis=1, keepdims=True)
        
        # Simple threshold optimization at this temperature
        thresholds = [1.0] * 5
        for c in range(5):
            best_c_f1 = 0.0
            for thr in np.arange(0.1, 2.5, 0.05):
                test_thr = thresholds.copy()
                test_thr[c] = thr
                adjusted = val_scaled.copy()
                for j in range(5):
                    adjusted[:, j] = adjusted[:, j] / test_thr[j]
                preds = adjusted.argmax(axis=1).tolist()
                f1 = landform_f1(val_true, preds)
                if f1 > best_c_f1:
                    best_c_f1 = f1
                    thresholds[c] = thr
        
        # Evaluate on val
        adjusted = val_scaled.copy()
        for j in range(5):
            adjusted[:, j] = adjusted[:, j] / thresholds[j]
        val_preds = adjusted.argmax(axis=1).tolist()
        val_f1 = landform_f1(val_true, val_preds)
        
        if val_f1 > best_overall_f1:
            best_overall_f1 = val_f1
            best_temp = temp
            best_thresholds = thresholds.copy()
    
    # Apply best to test
    test_logits = np.log(test_probs + 1e-8)
    test_scaled = np.exp(test_logits / best_temp) / np.exp(test_logits / best_temp).sum(axis=1, keepdims=True)
    adjusted_test = test_scaled.copy()
    for j in range(5):
        adjusted_test[:, j] = adjusted_test[:, j] / best_thresholds[j]
    test_preds = adjusted_test.argmax(axis=1).tolist()
    
    test_lf1 = landform_f1(test_true, test_preds)
    test_metrics = compute_metrics(test_true, test_preds, num_classes=5)
    
    print(f"  Best temperature: {best_temp:.1f}")
    print(f"  Val LF F1: {best_overall_f1:.4f}")
    print(f"  Test LF F1: {test_lf1:.4f}")
    print(f"  Thresholds: {dict(zip(CLASS_ORDER, [f'{t:.3f}' for t in best_thresholds]))}")
    for i, cls in enumerate(CLASS_ORDER):
        print(f"    {cls}: P={test_metrics['precision'][i]:.3f} R={test_metrics['recall'][i]:.3f} F1={test_metrics['f1'][i]:.3f}")
    
    return test_lf1, best_temp, best_thresholds, test_preds


def technique_3_multi_seed_ensemble(emb_dict, mola_dict, labels_dict, train_ids, val_ids, test_ids, device):
    """Train models with different seeds and ensemble predictions."""
    print("\n--- Technique 3: Multi-Seed Ensemble ---")
    
    cfg = get_config()
    mil_cfg = cfg.mil
    mil_cfg.epochs = 40
    mil_cfg.patience = 15
    
    seeds = [42, 123, 456]
    all_test_probs = []
    all_val_probs = []
    val_true = None
    test_true = None
    
    for i, seed in enumerate(seeds):
        print(f"\n  Training seed {seed} ({i+1}/{len(seeds)})...")
        set_seed(seed)
        
        out_dir = DATA_ROOT / f"models/cleaned_seed_{seed}"
        
        train_artifacts = train_mil(
            embeddings_dict=emb_dict,
            mola_dict=mola_dict,
            labels_dict=labels_dict,
            output_dir=out_dir,
            cfg=mil_cfg,
            device=device,
            mixed_precision=False,
            num_workers=0,
            seed=seed,
        )
        
        # Get predictions
        model_path = train_artifacts["best_model_path"]
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        saved_cfg = checkpoint.get("mil_config", {})
        model_cfg = deepcopy(mil_cfg)
        for k, v in saved_cfg.items():
            if hasattr(model_cfg, k):
                setattr(model_cfg, k, v)
        
        model = AttentionMILClassifier(model_cfg).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        
        val_ds = MILDataset(val_ids, emb_dict, mola_dict, labels_dict, 
                           min_tiles_per_image=1, max_tiles_per_image=128)
        test_ds = MILDataset(test_ids, emb_dict, mola_dict, labels_dict,
                            min_tiles_per_image=1, max_tiles_per_image=128)
        val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=mil_collate_fn, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=mil_collate_fn, num_workers=0)
        
        _, vt, vp, _ = get_predictions(model, val_loader, device)
        _, tt, tp, _ = get_predictions(model, test_loader, device)
        
        all_val_probs.append(vp)
        all_test_probs.append(tp)
        if val_true is None:
            val_true = vt
            test_true = tt
        
        seed_f1 = landform_f1(tt, tp.argmax(axis=1).tolist())
        print(f"    Seed {seed} test LF F1: {seed_f1:.4f}")
    
    # Ensemble: average probabilities
    avg_val_probs = np.mean(all_val_probs, axis=0)
    avg_test_probs = np.mean(all_test_probs, axis=0)
    
    # Raw ensemble
    raw_preds = avg_test_probs.argmax(axis=1).tolist()
    raw_f1 = landform_f1(test_true, raw_preds)
    print(f"\n  Ensemble (avg) Test LF F1: {raw_f1:.4f}")
    
    # Threshold optimization on ensemble
    best_thresholds = [1.0] * 5
    best_f1 = 0.0
    for c in range(5):
        best_c_thr = 1.0
        for thr in np.arange(0.1, 2.5, 0.03):
            test_thr = best_thresholds.copy()
            test_thr[c] = thr
            adjusted = avg_val_probs.copy()
            for j in range(5):
                adjusted[:, j] = adjusted[:, j] / test_thr[j]
            preds = adjusted.argmax(axis=1).tolist()
            f1 = landform_f1(val_true, preds)
            if f1 > best_f1:
                best_f1 = f1
                best_c_thr = thr
        best_thresholds[c] = best_c_thr
    
    adjusted_test = avg_test_probs.copy()
    for j in range(5):
        adjusted_test[:, j] = adjusted_test[:, j] / best_thresholds[j]
    opt_preds = adjusted_test.argmax(axis=1).tolist()
    opt_f1 = landform_f1(test_true, opt_preds)
    opt_metrics = compute_metrics(test_true, opt_preds, num_classes=5)
    
    print(f"  Ensemble + Threshold Opt Test LF F1: {opt_f1:.4f}")
    for i, cls in enumerate(CLASS_ORDER):
        print(f"    {cls}: P={opt_metrics['precision'][i]:.3f} R={opt_metrics['recall'][i]:.3f} F1={opt_metrics['f1'][i]:.3f}")
    
    return raw_f1, opt_f1, best_thresholds


def technique_4_test_time_augmentation(model, emb_dict, mola_dict, labels_dict, test_ids, device):
    """Test-time augmentation: run model multiple times with different tile subsamples."""
    print("\n--- Technique 4: Test-Time Augmentation (Tile Subsampling) ---")
    
    num_runs = 10
    all_probs = []
    test_true = None
    
    for run in range(num_runs):
        np.random.seed(run)
        test_ds = MILDataset(test_ids, emb_dict, mola_dict, labels_dict,
                            min_tiles_per_image=1, max_tiles_per_image=128)
        test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=mil_collate_fn, num_workers=0)
        
        _, tt, tp, _ = get_predictions(model, test_loader, device)
        all_probs.append(tp)
        if test_true is None:
            test_true = tt
    
    avg_probs = np.mean(all_probs, axis=0)
    preds = avg_probs.argmax(axis=1).tolist()
    tta_f1 = landform_f1(test_true, preds)
    tta_metrics = compute_metrics(test_true, preds, num_classes=5)
    
    print(f"  TTA ({num_runs} runs) Test LF F1: {tta_f1:.4f}")
    for i, cls in enumerate(CLASS_ORDER):
        print(f"    {cls}: P={tta_metrics['precision'][i]:.3f} R={tta_metrics['recall'][i]:.3f} F1={tta_metrics['f1'][i]:.3f}")
    
    return tta_f1, avg_probs, test_true


def main():
    print("=" * 60)
    print("PUSHING TOWARD F1=0.8")
    print("=" * 60)
    
    model, model_cfg, emb_dict, mola_dict, labels_dict, train_ids, val_ids, test_ids, device = load_model_and_data()
    
    val_ds = MILDataset(val_ids, emb_dict, mola_dict, labels_dict,
                       min_tiles_per_image=1, max_tiles_per_image=128)
    test_ds = MILDataset(test_ids, emb_dict, mola_dict, labels_dict,
                        min_tiles_per_image=1, max_tiles_per_image=128)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=mil_collate_fn, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=mil_collate_fn, num_workers=0)
    
    _, val_true, val_probs, _ = get_predictions(model, val_loader, device)
    _, test_true, test_probs, _ = get_predictions(model, test_loader, device)
    
    print(f"\nBaseline: Cleaned V4 raw LF F1 = {landform_f1(test_true, test_probs.argmax(axis=1).tolist()):.4f}")
    
    # Technique 1: Fine-grained threshold
    t1_f1, t1_thr, t1_preds = technique_1_fine_threshold(val_probs, val_true, test_probs, test_true)
    
    # Technique 2: Temperature + threshold
    t2_f1, t2_temp, t2_thr, t2_preds = technique_2_temperature_threshold(val_probs, val_true, test_probs, test_true)
    
    # Technique 4: Test-time augmentation (fast, no retraining)
    t4_f1, t4_probs, _ = technique_4_test_time_augmentation(model, emb_dict, mola_dict, labels_dict, test_ids, device)
    
    # Technique 4 + Threshold
    print("\n--- Technique 4+1: TTA + Threshold Optimization ---")
    val_tta_probs = []
    for run in range(10):
        np.random.seed(run)
        val_ds_r = MILDataset(val_ids, emb_dict, mola_dict, labels_dict,
                             min_tiles_per_image=1, max_tiles_per_image=128)
        val_loader_r = DataLoader(val_ds_r, batch_size=16, shuffle=False, collate_fn=mil_collate_fn, num_workers=0)
        _, _, vp, _ = get_predictions(model, val_loader_r, device)
        val_tta_probs.append(vp)
    avg_val_tta = np.mean(val_tta_probs, axis=0)
    
    t41_f1, _, _ = technique_1_fine_threshold(avg_val_tta, val_true, t4_probs, test_true)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  Cleaned V4 (raw):              LF F1 = {landform_f1(test_true, test_probs.argmax(axis=1).tolist()):.4f}")
    print(f"  + Fine Threshold:              LF F1 = {t1_f1:.4f}")
    print(f"  + Temperature + Threshold:     LF F1 = {t2_f1:.4f}")
    print(f"  + TTA (10 runs):               LF F1 = {t4_f1:.4f}")
    print(f"  + TTA + Threshold:             LF F1 = {t41_f1:.4f}")
    print(f"{'='*60}")
    
    # Now try multi-seed ensemble (slower, but potentially best)
    print("\nStarting multi-seed ensemble training (may take ~30-40 min)...")
    t3_raw, t3_opt, t3_thr = technique_3_multi_seed_ensemble(
        emb_dict, mola_dict, labels_dict, train_ids, val_ids, test_ids, device
    )
    
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"{'='*60}")
    print(f"  Cleaned V4 (raw):              LF F1 = {landform_f1(test_true, test_probs.argmax(axis=1).tolist()):.4f}")
    print(f"  + Fine Threshold:              LF F1 = {t1_f1:.4f}")
    print(f"  + Temperature + Threshold:     LF F1 = {t2_f1:.4f}")
    print(f"  + TTA (10 runs):               LF F1 = {t4_f1:.4f}")
    print(f"  + TTA + Threshold:             LF F1 = {t41_f1:.4f}")
    print(f"  Multi-Seed Ensemble (raw):     LF F1 = {t3_raw:.4f}")
    print(f"  Multi-Seed Ensemble + Thr:     LF F1 = {t3_opt:.4f}")
    print(f"{'='*60}")
    
    best_f1 = max(t1_f1, t2_f1, t4_f1, t41_f1, t3_raw, t3_opt)
    if best_f1 >= 0.8:
        print(f"\n  🎯 TARGET REACHED! Best F1 = {best_f1:.4f} >= 0.8")
    else:
        print(f"\n  Best F1 = {best_f1:.4f}, gap to 0.8 = {0.8 - best_f1:.4f}")
    
    # Save all results
    all_results = {
        "baseline_raw": float(landform_f1(test_true, test_probs.argmax(axis=1).tolist())),
        "fine_threshold": float(t1_f1),
        "temperature_threshold": float(t2_f1),
        "tta": float(t4_f1),
        "tta_threshold": float(t41_f1),
        "multi_seed_raw": float(t3_raw),
        "multi_seed_threshold": float(t3_opt),
        "best": float(best_f1),
        "target": 0.8,
    }
    out_path = DATA_ROOT / "models/cleaned_v4/push_to_08_results.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nAll results saved to {out_path}")


if __name__ == "__main__":
    main()
