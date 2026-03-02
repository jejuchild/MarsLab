#!/usr/bin/env python3
"""Ensemble classifier: combine multiple trained MIL models for better predictions."""
import json
import sys
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
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

# Model configs: name, model_dir, embeddings_dir
MODELS = [
    {
        "name": "V3_MultiHead",
        "model_dir": DATA_ROOT / "models/multihead_improved",
        "embeddings_dir": DATA_ROOT / "embeddings_mil",
        "val_landform_f1": 0.571,
    },
    {
        "name": "Frozen_30",
        "model_dir": DATA_ROOT / "models/frozen_30_test",
        "embeddings_dir": DATA_ROOT / "embeddings_frozen_30",
        "val_landform_f1": 0.543,
    },
    {
        "name": "SSL_30",
        "model_dir": DATA_ROOT / "models/ssl_multihead",
        "embeddings_dir": DATA_ROOT / "embeddings_ssl",
        "val_landform_f1": 0.461,
    },
]


@torch.no_grad()
def get_model_predictions(
    model_path: Path,
    embeddings_dict: Dict[str, np.ndarray],
    mola_dict: Dict[str, np.ndarray],
    test_ids: List[str],
    cfg: Any,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """Load model and get probability predictions for test_ids."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Build model with config from checkpoint
    saved_cfg = checkpoint.get("mil_config", {})
    model_cfg = deepcopy(cfg)
    for k, v in saved_cfg.items():
        if hasattr(model_cfg, k):
            setattr(model_cfg, k, v)
    
    model = AttentionMILClassifier(model_cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    ds = MILDataset(
        test_ids, embeddings_dict, mola_dict,
        {img: 0 for img in test_ids},  # dummy labels
        min_tiles_per_image=model_cfg.min_tiles_per_image,
        max_tiles_per_image=model_cfg.max_tiles_per_image,
    )
    loader = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=mil_collate_fn, num_workers=0)
    
    probs_dict: Dict[str, np.ndarray] = {}
    for batch in loader:
        tiles = batch["tile_embeddings"].to(device)
        mask = batch["tile_mask"].to(device)
        mola = batch["mola_features"].to(device)
        image_ids = batch["image_ids"]
        
        logits, _ = model(tiles, mask, mola)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        
        for i, img_id in enumerate(image_ids):
            probs_dict[img_id] = probs[i]
    
    return probs_dict


def ensemble_average(all_probs: List[Dict[str, np.ndarray]], test_ids: List[str]) -> np.ndarray:
    """Simple average of probability vectors."""
    n_models = len(all_probs)
    results = []
    for img_id in test_ids:
        avg = np.zeros(5)
        count = 0
        for mp in all_probs:
            if img_id in mp:
                avg += mp[img_id]
                count += 1
        if count > 0:
            avg /= count
        results.append(avg)
    return np.array(results)


def ensemble_weighted(
    all_probs: List[Dict[str, np.ndarray]], 
    weights: List[float],
    test_ids: List[str]
) -> np.ndarray:
    """Weighted average of probability vectors."""
    results = []
    total_w = sum(weights)
    for img_id in test_ids:
        avg = np.zeros(5)
        w_sum = 0.0
        for mp, w in zip(all_probs, weights):
            if img_id in mp:
                avg += mp[img_id] * w
                w_sum += w
        if w_sum > 0:
            avg /= w_sum
        results.append(avg)
    return np.array(results)


def ensemble_majority_vote(all_probs: List[Dict[str, np.ndarray]], test_ids: List[str]) -> np.ndarray:
    """Majority vote across models."""
    results = []
    for img_id in test_ids:
        votes = []
        for mp in all_probs:
            if img_id in mp:
                votes.append(int(np.argmax(mp[img_id])))
        if not votes:
            results.append(np.array([0.2, 0.2, 0.2, 0.2, 0.2]))
            continue
        vote_counts = Counter(votes)
        winner = vote_counts.most_common(1)[0][0]
        # Create one-hot probability for the winner
        prob = np.zeros(5)
        prob[winner] = 1.0
        results.append(prob)
    return np.array(results)


def run_ensemble():
    print("=" * 60)
    print("ENSEMBLE CLASSIFIER")
    print("=" * 60)
    
    set_seed(42)
    cfg = get_config()
    mil_cfg = cfg.mil
    device = torch.device("cpu")
    
    # Load canonical test split from V3
    split_path = DATA_ROOT / "models/multihead_improved/data_split.json"
    split = json.loads(split_path.read_text())
    test_ids = split["test_ids"]
    print(f"Test set: {len(test_ids)} images")
    
    # Load labels
    labels_dict = {k: CLASS_ORDER.index(v) if isinstance(v, str) else v 
                   for k, v in json.loads((DATA_ROOT / "labels_simple.json").read_text()).items()}
    
    # Load MOLA (shared across all models)
    mola_dict_raw = np.load(DATA_ROOT / "mola_features_by_image.npy", allow_pickle=True).item()
    mola_dict = {str(k): np.asarray(v, dtype=np.float32) for k, v in mola_dict_raw.items()}
    
    # Get predictions from each model
    all_probs: List[Dict[str, np.ndarray]] = []
    model_names = []
    model_weights = []
    
    for model_info in MODELS:
        name = model_info["name"]
        model_dir = model_info["model_dir"]
        emb_dir = model_info["embeddings_dir"]
        
        model_path = model_dir / "best_mil_model.pt"
        if not model_path.exists():
            print(f"  Skipping {name}: no checkpoint")
            continue
        
        print(f"\nLoading {name}...")
        emb_dict = load_embeddings(emb_dir)
        
        # Check which test_ids have embeddings
        available_ids = [tid for tid in test_ids if tid in emb_dict and tid in mola_dict]
        print(f"  {len(available_ids)}/{len(test_ids)} test images have embeddings")
        
        probs = get_model_predictions(model_path, emb_dict, mola_dict, available_ids, mil_cfg, device)
        all_probs.append(probs)
        model_names.append(name)
        model_weights.append(model_info["val_landform_f1"])
        print(f"  Got predictions for {len(probs)} images")
    
    if not all_probs:
        print("No models loaded!")
        return
    
    # Get ground truth for test set
    y_true = [labels_dict[tid] for tid in test_ids]
    
    # Output dir
    out_dir = DATA_ROOT / "models/ensemble"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Strategy 1: Simple average
    print("\n--- Strategy 1: Simple Average ---")
    avg_probs = ensemble_average(all_probs, test_ids)
    y_pred_avg = np.argmax(avg_probs, axis=1).tolist()
    metrics_avg = compute_metrics(y_true, y_pred_avg, num_classes=5)
    print(f"  Macro F1: {metrics_avg['macro_f1_all']:.4f}")
    print(f"  Landform F1: {metrics_avg['landform_macro_f1']:.4f}")
    for i, cls in enumerate(CLASS_ORDER):
        print(f"    {cls}: P={metrics_avg['precision'][i]:.3f} R={metrics_avg['recall'][i]:.3f} F1={metrics_avg['f1'][i]:.3f}")
    
    # Strategy 2: Weighted average
    print(f"\n--- Strategy 2: Weighted Average (weights={[f'{w:.3f}' for w in model_weights]}) ---")
    wavg_probs = ensemble_weighted(all_probs, model_weights, test_ids)
    y_pred_wavg = np.argmax(wavg_probs, axis=1).tolist()
    metrics_wavg = compute_metrics(y_true, y_pred_wavg, num_classes=5)
    print(f"  Macro F1: {metrics_wavg['macro_f1_all']:.4f}")
    print(f"  Landform F1: {metrics_wavg['landform_macro_f1']:.4f}")
    for i, cls in enumerate(CLASS_ORDER):
        print(f"    {cls}: P={metrics_wavg['precision'][i]:.3f} R={metrics_wavg['recall'][i]:.3f} F1={metrics_wavg['f1'][i]:.3f}")
    
    # Strategy 3: Majority vote
    print("\n--- Strategy 3: Majority Vote ---")
    vote_probs = ensemble_majority_vote(all_probs, test_ids)
    y_pred_vote = np.argmax(vote_probs, axis=1).tolist()
    metrics_vote = compute_metrics(y_true, y_pred_vote, num_classes=5)
    print(f"  Macro F1: {metrics_vote['macro_f1_all']:.4f}")
    print(f"  Landform F1: {metrics_vote['landform_macro_f1']:.4f}")
    for i, cls in enumerate(CLASS_ORDER):
        print(f"    {cls}: P={metrics_vote['precision'][i]:.3f} R={metrics_vote['recall'][i]:.3f} F1={metrics_vote['f1'][i]:.3f}")
    
    # Pick best strategy
    strategies = {
        "simple_average": metrics_avg,
        "weighted_average": metrics_wavg,
        "majority_vote": metrics_vote,
    }
    best_strategy = max(strategies.items(), key=lambda x: x[1]["landform_macro_f1"])
    print(f"\n*** Best strategy: {best_strategy[0]} (Landform F1 = {best_strategy[1]['landform_macro_f1']:.4f}) ***")
    
    # Save results
    result = {
        "models_used": model_names,
        "model_weights": model_weights,
        "strategies": {k: v for k, v in strategies.items()},
        "best_strategy": best_strategy[0],
        "best_landform_f1": best_strategy[1]["landform_macro_f1"],
        "best_macro_f1": best_strategy[1]["macro_f1_all"],
    }
    (out_dir / "test_metrics.json").write_text(json.dumps(result, indent=2))
    print(f"\nSaved to {out_dir / 'test_metrics.json'}")
    
    return result


if __name__ == "__main__":
    run_ensemble()
