#!/usr/bin/env python3
"""Per-class threshold optimization + temperature scaling for MIL classifier."""
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.marslandform_v2.config import CLASS_ORDER, get_config
from scripts.marslandform_v2.models.mil_classifier import (
    AttentionMILClassifier,
    MILDataset,
    compute_metrics,
    load_embeddings,
    load_labels,
    load_mola_features,
    mil_collate_fn,
    set_seed,
)

DATA_ROOT = ROOT / "Data/HiRISE/v2_output"


@torch.no_grad()
def collect_predictions(
    model: AttentionMILClassifier,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[List[int], np.ndarray, np.ndarray]:
    """Collect ground truth, logits, and probabilities."""
    model.eval()
    y_true = []
    all_logits = []
    all_probs = []
    
    for batch in loader:
        tiles = batch["tile_embeddings"].to(device)
        mask = batch["tile_mask"].to(device)
        mola = batch["mola_features"].to(device)
        labels = batch["labels"]
        
        logits, _ = model(tiles, mask, mola)
        probs = torch.softmax(logits, dim=1)
        
        y_true.extend(labels.tolist())
        all_logits.append(logits.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
    
    return y_true, np.concatenate(all_logits), np.concatenate(all_probs)


def optimize_thresholds(
    y_true: List[int], probs: np.ndarray, num_classes: int = 5
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Find optimal per-class probability thresholds that maximize F1."""
    best_thresholds = np.full(num_classes, 0.5)
    threshold_details = {}
    
    for cls in range(num_classes):
        binary_true = np.array([1 if y == cls else 0 for y in y_true])
        if binary_true.sum() == 0:
            threshold_details[CLASS_ORDER[cls]] = {"best_threshold": 0.5, "best_f1": 0.0, "support": 0}
            continue
        
        best_f1 = -1.0
        best_t = 0.5
        
        for t_int in range(10, 91):
            t = t_int / 100.0
            binary_pred = (probs[:, cls] >= t).astype(int)
            
            tp = np.sum((binary_pred == 1) & (binary_true == 1))
            fp = np.sum((binary_pred == 1) & (binary_true == 0))
            fn = np.sum((binary_pred == 0) & (binary_true == 1))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        
        best_thresholds[cls] = best_t
        threshold_details[CLASS_ORDER[cls]] = {
            "best_threshold": best_t,
            "best_f1": best_f1,
            "support": int(binary_true.sum()),
        }
    
    return best_thresholds, threshold_details


def apply_thresholds(probs: np.ndarray, thresholds: np.ndarray) -> List[int]:
    """Apply per-class thresholds with conflict resolution."""
    preds = []
    for i in range(len(probs)):
        p = probs[i]
        # Scale probabilities by threshold: p_i / t_i (higher = more confident relative to threshold)
        scaled = p / thresholds
        
        # Check if any class passes its threshold
        passing = p >= thresholds
        if passing.any():
            # Among passing classes, pick highest scaled score
            masked_scaled = np.where(passing, scaled, -np.inf)
            preds.append(int(np.argmax(masked_scaled)))
        else:
            # No class passes: fall back to argmax
            preds.append(int(np.argmax(p)))
    
    return preds


def fit_temperature(logits: np.ndarray, y_true: List[int], max_iter: int = 200) -> float:
    """Learn temperature T that minimizes NLL on validation logits."""
    logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_t = torch.tensor(y_true, dtype=torch.long)
    
    # Initialize temperature
    temperature = nn.Parameter(torch.ones(1) * 1.5)
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=max_iter)
    
    def eval_loss():
        optimizer.zero_grad()
        scaled = logits_t / temperature
        loss = nn.functional.cross_entropy(scaled, labels_t)
        loss.backward()
        return loss
    
    optimizer.step(eval_loss)
    
    return float(temperature.item())


def run_threshold_opt():
    print("=" * 60)
    print("THRESHOLD OPTIMIZATION + TEMPERATURE SCALING")
    print("=" * 60)
    
    set_seed(42)
    cfg = get_config()
    mil_cfg = cfg.mil
    device = torch.device("cpu")
    
    # Load data
    emb_dict = load_embeddings(DATA_ROOT / "embeddings_mil")
    mola_raw = np.load(DATA_ROOT / "mola_features_by_image.npy", allow_pickle=True).item()
    mola_dict = {str(k): np.asarray(v, dtype=np.float32) for k, v in mola_raw.items()}
    labels_raw = json.loads((DATA_ROOT / "labels_simple.json").read_text())
    labels_dict = {k: CLASS_ORDER.index(v) if isinstance(v, str) else v for k, v in labels_raw.items()}
    
    # Load canonical split
    split = json.loads((DATA_ROOT / "models/multihead_improved/data_split.json").read_text())
    val_ids = split["val_ids"]
    test_ids = split["test_ids"]
    print(f"Val: {len(val_ids)} images, Test: {len(test_ids)} images")
    
    # Load V3 model
    model_path = DATA_ROOT / "models/multihead_improved/best_mil_model.pt"
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    saved_cfg = checkpoint.get("mil_config", {})
    model_cfg = deepcopy(mil_cfg)
    for k, v in saved_cfg.items():
        if hasattr(model_cfg, k):
            setattr(model_cfg, k, v)
    
    model = AttentionMILClassifier(model_cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Create dataloaders
    def make_loader(ids):
        ds = MILDataset(ids, emb_dict, mola_dict, labels_dict,
                       min_tiles_per_image=model_cfg.min_tiles_per_image,
                       max_tiles_per_image=model_cfg.max_tiles_per_image)
        return DataLoader(ds, batch_size=16, shuffle=False, collate_fn=mil_collate_fn, num_workers=0)
    
    val_loader = make_loader(val_ids)
    test_loader = make_loader(test_ids)
    
    # Collect predictions
    print("\nCollecting validation predictions...")
    val_true, val_logits, val_probs = collect_predictions(model, val_loader, device)
    print("Collecting test predictions...")
    test_true, test_logits, test_probs = collect_predictions(model, test_loader, device)
    
    results = {}
    
    # --- Baseline: argmax ---
    print("\n--- Baseline: Argmax ---")
    baseline_preds = np.argmax(test_probs, axis=1).tolist()
    baseline_metrics = compute_metrics(test_true, baseline_preds, num_classes=5)
    print(f"  Macro F1: {baseline_metrics['macro_f1_all']:.4f}")
    print(f"  Landform F1: {baseline_metrics['landform_macro_f1']:.4f}")
    results["baseline_argmax"] = baseline_metrics
    
    # --- Strategy 1: Optimized per-class thresholds ---
    print("\n--- Strategy 1: Per-Class Threshold Optimization ---")
    thresholds, threshold_details = optimize_thresholds(val_true, val_probs)
    print(f"  Optimized thresholds: {dict(zip(CLASS_ORDER, [f'{t:.2f}' for t in thresholds]))}")
    for cls, detail in threshold_details.items():
        print(f"    {cls}: t={detail['best_threshold']:.2f}, val_f1={detail['best_f1']:.3f}, support={detail['support']}")
    
    thresh_preds = apply_thresholds(test_probs, thresholds)
    thresh_metrics = compute_metrics(test_true, thresh_preds, num_classes=5)
    print(f"  Test Macro F1: {thresh_metrics['macro_f1_all']:.4f}")
    print(f"  Test Landform F1: {thresh_metrics['landform_macro_f1']:.4f}")
    results["threshold_optimized"] = thresh_metrics
    results["threshold_optimized"]["thresholds"] = thresholds.tolist()
    
    # --- Strategy 2: Temperature scaling ---
    print("\n--- Strategy 2: Temperature Scaling ---")
    T = fit_temperature(val_logits, val_true)
    print(f"  Learned temperature T = {T:.4f}")
    
    temp_probs = torch.softmax(torch.tensor(test_logits) / T, dim=1).numpy()
    temp_preds = np.argmax(temp_probs, axis=1).tolist()
    temp_metrics = compute_metrics(test_true, temp_preds, num_classes=5)
    print(f"  Test Macro F1: {temp_metrics['macro_f1_all']:.4f}")
    print(f"  Test Landform F1: {temp_metrics['landform_macro_f1']:.4f}")
    results["temperature_scaled"] = temp_metrics
    results["temperature_scaled"]["temperature"] = T
    
    # --- Strategy 3: Temperature + Thresholds ---
    print("\n--- Strategy 3: Temperature + Thresholds ---")
    # Re-optimize thresholds on temperature-scaled val probs
    val_temp_probs = torch.softmax(torch.tensor(val_logits) / T, dim=1).numpy()
    thresholds_t, _ = optimize_thresholds(val_true, val_temp_probs)
    print(f"  Thresholds (temp-scaled): {dict(zip(CLASS_ORDER, [f'{t:.2f}' for t in thresholds_t]))}")
    
    combo_preds = apply_thresholds(temp_probs, thresholds_t)
    combo_metrics = compute_metrics(test_true, combo_preds, num_classes=5)
    print(f"  Test Macro F1: {combo_metrics['macro_f1_all']:.4f}")
    print(f"  Test Landform F1: {combo_metrics['landform_macro_f1']:.4f}")
    results["temperature_plus_threshold"] = combo_metrics
    results["temperature_plus_threshold"]["temperature"] = T
    results["temperature_plus_threshold"]["thresholds"] = thresholds_t.tolist()
    
    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    best_key = max(results.keys(), key=lambda k: results[k]["landform_macro_f1"])
    for key in results:
        lf = results[key]["landform_macro_f1"]
        mf = results[key]["macro_f1_all"]
        marker = " <<<" if key == best_key else ""
        print(f"  {key}: Landform F1={lf:.4f}, Macro F1={mf:.4f}{marker}")
    
    print(f"\n*** Best: {best_key} (Landform F1 = {results[best_key]['landform_macro_f1']:.4f}) ***")
    
    # Save
    out_dir = DATA_ROOT / "models/threshold_opt"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "test_metrics.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {out_dir / 'test_metrics.json'}")
    
    return results


if __name__ == "__main__":
    run_threshold_opt()
