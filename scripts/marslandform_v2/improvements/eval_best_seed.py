#!/usr/bin/env python3
"""Evaluate seed-123 model (the champion) + threshold opt + ensemble of seed42+seed123."""
import json
import sys
import numpy as np
from pathlib import Path
from copy import deepcopy

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
)
from torch.utils.data import DataLoader

DATA_ROOT = ROOT / "Data/HiRISE/v2_output"


def load_model(model_path, device, mil_cfg):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    saved_cfg = checkpoint.get("mil_config", {})
    model_cfg = deepcopy(mil_cfg)
    for k, v in saved_cfg.items():
        if hasattr(model_cfg, k):
            setattr(model_cfg, k, v)
    model = AttentionMILClassifier(model_cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint.get("best_epoch", "?")


def get_predictions(model, loader, device):
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


def landform_f1(true, pred):
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


def threshold_optimize(val_probs, val_true, test_probs, test_true):
    best_thresholds = [1.0] * 5
    for iteration in range(3):
        step = 0.03 / (iteration + 1)
        for c in range(5):
            low = max(0.05, best_thresholds[c] - 0.4)
            high = min(3.0, best_thresholds[c] + 0.4)
            best_c_f1 = 0.0
            for thr in np.arange(low, high, step):
                test_thr = best_thresholds.copy()
                test_thr[c] = thr
                adjusted = val_probs.copy()
                for j in range(5):
                    adjusted[:, j] = adjusted[:, j] / test_thr[j]
                preds = adjusted.argmax(axis=1).tolist()
                f1 = landform_f1(val_true, preds)
                if f1 > best_c_f1:
                    best_c_f1 = f1
                    best_thresholds[c] = thr
    
    adjusted_test = test_probs.copy()
    for j in range(5):
        adjusted_test[:, j] = adjusted_test[:, j] / best_thresholds[j]
    test_preds = adjusted_test.argmax(axis=1).tolist()
    return best_thresholds, test_preds


def main():
    print("=" * 60)
    print("EVALUATE BEST MODELS")
    print("=" * 60)

    set_seed(42)
    cfg = get_config()
    mil_cfg = cfg.mil
    device = torch.device("cpu")

    emb_dict = load_embeddings(DATA_ROOT / "embeddings_mil")
    mola_dict = load_mola_features(DATA_ROOT / "mola_features_by_image.npy")
    labels_dict = load_labels(DATA_ROOT / "label_audit/labels_cleaned.json")

    split = json.loads((DATA_ROOT / "models/multihead_improved/data_split.json").read_text())
    valid_ids = set(emb_dict.keys()) & set(mola_dict.keys()) & set(labels_dict.keys())
    train_ids = [i for i in split["train_ids"] if i in valid_ids]
    val_ids = [i for i in split["val_ids"] if i in valid_ids]
    test_ids = [i for i in split["test_ids"] if i in valid_ids]

    val_ds = MILDataset(val_ids, emb_dict, mola_dict, labels_dict, 1, 128)
    test_ds = MILDataset(test_ids, emb_dict, mola_dict, labels_dict, 1, 128)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=mil_collate_fn, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=mil_collate_fn, num_workers=0)

    results = {}

    # Evaluate each model
    model_paths = {
        "V3 (original labels, seed 42)": DATA_ROOT / "models/multihead_improved/best_mil_model.pt",
        "Cleaned V4 (seed 42)": DATA_ROOT / "models/cleaned_v4/best_mil_model.pt",
        "Cleaned V4 (seed 123)": DATA_ROOT / "models/cleaned_seed_123/best_mil_model.pt",
        "Cleaned V4 (seed 42, retrain)": DATA_ROOT / "models/cleaned_seed_42/best_mil_model.pt",
    }

    all_probs = {}
    test_true = None

    for name, path in model_paths.items():
        if not path.exists():
            print(f"\n{name}: SKIP (not found)")
            continue

        print(f"\n{'='*60}")
        print(f"  {name}")
        model, epoch = load_model(path, device, mil_cfg)
        print(f"  Best epoch: {epoch}")

        _, vt, vp, _ = get_predictions(model, val_loader, device)
        _, tt, tp, tids = get_predictions(model, test_loader, device)
        if test_true is None:
            test_true = tt

        all_probs[name] = tp

        # Raw
        raw_preds = tp.argmax(axis=1).tolist()
        raw_f1 = landform_f1(tt, raw_preds)
        raw_metrics = compute_metrics(tt, raw_preds, num_classes=5)
        
        print(f"  Raw Landform F1: {raw_f1:.4f}")
        print(f"  Raw Macro F1: {raw_metrics['macro_f1_all']:.4f}")
        for i, cls in enumerate(CLASS_ORDER):
            print(f"    {cls}: P={raw_metrics['precision'][i]:.3f} R={raw_metrics['recall'][i]:.3f} F1={raw_metrics['f1'][i]:.3f}")

        # Threshold opt
        thresholds, opt_preds = threshold_optimize(vp, vt, tp, tt)
        opt_f1 = landform_f1(tt, opt_preds)
        opt_metrics = compute_metrics(tt, opt_preds, num_classes=5)

        print(f"  + Threshold Opt Landform F1: {opt_f1:.4f}")
        print(f"  + Threshold Opt Macro F1: {opt_metrics['macro_f1_all']:.4f}")
        print(f"  Thresholds: {dict(zip(CLASS_ORDER, [f'{t:.3f}' for t in thresholds]))}")
        for i, cls in enumerate(CLASS_ORDER):
            print(f"    {cls}: P={opt_metrics['precision'][i]:.3f} R={opt_metrics['recall'][i]:.3f} F1={opt_metrics['f1'][i]:.3f}")

        results[name] = {
            "raw_landform_f1": raw_f1,
            "raw_macro_f1": raw_metrics["macro_f1_all"],
            "opt_landform_f1": opt_f1,
            "opt_macro_f1": opt_metrics["macro_f1_all"],
            "raw_per_class": {CLASS_ORDER[i]: raw_metrics["f1"][i] for i in range(5)},
            "opt_per_class": {CLASS_ORDER[i]: opt_metrics["f1"][i] for i in range(5)},
        }

    # Ensemble of seed42 + seed123
    if "Cleaned V4 (seed 42, retrain)" in all_probs and "Cleaned V4 (seed 123)" in all_probs:
        print(f"\n{'='*60}")
        print(f"  Ensemble: seed42 + seed123")
        ens_probs = (all_probs["Cleaned V4 (seed 42, retrain)"] + all_probs["Cleaned V4 (seed 123)"]) / 2
        ens_preds = ens_probs.argmax(axis=1).tolist()
        ens_f1 = landform_f1(test_true, ens_preds)
        ens_metrics = compute_metrics(test_true, ens_preds, num_classes=5)
        
        print(f"  Ensemble Raw Landform F1: {ens_f1:.4f}")
        print(f"  Ensemble Raw Macro F1: {ens_metrics['macro_f1_all']:.4f}")
        for i, cls in enumerate(CLASS_ORDER):
            print(f"    {cls}: P={ens_metrics['precision'][i]:.3f} R={ens_metrics['recall'][i]:.3f} F1={ens_metrics['f1'][i]:.3f}")

        results["Ensemble (seed42+seed123)"] = {
            "raw_landform_f1": ens_f1,
            "raw_macro_f1": ens_metrics["macro_f1_all"],
        }

    # Ensemble all 3 (V4 + seed42 + seed123)
    valid_keys = [k for k in ["Cleaned V4 (seed 42)", "Cleaned V4 (seed 42, retrain)", "Cleaned V4 (seed 123)"] if k in all_probs]
    if len(valid_keys) >= 2:
        print(f"\n{'='*60}")
        print(f"  Ensemble: all cleaned models ({len(valid_keys)} models)")
        all_ens = np.mean([all_probs[k] for k in valid_keys], axis=0)
        all_ens_preds = all_ens.argmax(axis=1).tolist()
        all_ens_f1 = landform_f1(test_true, all_ens_preds)
        all_ens_metrics = compute_metrics(test_true, all_ens_preds, num_classes=5)

        print(f"  All-Ensemble Raw Landform F1: {all_ens_f1:.4f}")
        print(f"  All-Ensemble Raw Macro F1: {all_ens_metrics['macro_f1_all']:.4f}")
        for i, cls in enumerate(CLASS_ORDER):
            print(f"    {cls}: P={all_ens_metrics['precision'][i]:.3f} R={all_ens_metrics['recall'][i]:.3f} F1={all_ens_metrics['f1'][i]:.3f}")

        results["Ensemble (all cleaned)"] = {
            "raw_landform_f1": all_ens_f1,
            "raw_macro_f1": all_ens_metrics["macro_f1_all"],
        }

    # Final leaderboard
    print(f"\n{'='*60}")
    print(f"LEADERBOARD (Landform F1)")
    print(f"{'='*60}")
    sorted_results = sorted(results.items(), key=lambda x: max(x[1].get("raw_landform_f1", 0), x[1].get("opt_landform_f1", 0)), reverse=True)
    for name, r in sorted_results:
        best = max(r.get("raw_landform_f1", 0), r.get("opt_landform_f1", 0))
        marker = " 🎯" if best >= 0.8 else ""
        print(f"  {best:.4f} — {name}{marker}")

    # Save
    out_path = DATA_ROOT / "models/cleaned_v4/leaderboard.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
