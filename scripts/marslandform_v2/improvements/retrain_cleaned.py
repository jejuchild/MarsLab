#!/usr/bin/env python3
"""Retrain MIL classifier with cleaned labels from VLM audit.

Uses same V3 architecture but with corrected labels.
Also applies threshold optimization to the retrained model.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.marslandform_v2.config import CLASS_ORDER, get_config
from scripts.marslandform_v2.models.mil_classifier import (
    evaluate_best_model,
    load_embeddings,
    load_labels,
    load_mola_features,
    set_seed,
    train_mil,
)

DATA_ROOT = ROOT / "Data/HiRISE/v2_output"


def run_retrain():
    print("=" * 60)
    print("RETRAIN WITH CLEANED LABELS")
    print("=" * 60)
    
    set_seed(42)
    cfg = get_config()
    mil_cfg = cfg.mil
    mil_cfg.epochs = 60
    mil_cfg.patience = 20
    
    import torch
    device = torch.device("cpu")
    
    # Load embeddings and MOLA (same as before)
    emb_dict = load_embeddings(DATA_ROOT / "embeddings_mil")
    mola_dict = load_mola_features(DATA_ROOT / "mola_features_by_image.npy")
    
    # Use cleaned labels
    cleaned_path = DATA_ROOT / "label_audit/labels_cleaned.json"
    if not cleaned_path.exists():
        print("ERROR: No cleaned labels found. Run label_audit.py first.")
        return
    
    labels_dict = load_labels(cleaned_path)
    print(f"Loaded {len(labels_dict)} cleaned labels")
    
    # Count changes
    original = json.loads((DATA_ROOT / "labels_simple.json").read_text())
    orig_labels = {k: CLASS_ORDER.index(v) if isinstance(v, str) else v for k, v in original.items()}
    changes = sum(1 for k in labels_dict if k in orig_labels and labels_dict[k] != orig_labels[k])
    print(f"Labels changed: {changes}/{len(labels_dict)}")
    
    # Train
    out_dir = DATA_ROOT / "models/cleaned_v4"
    print(f"\nTraining to {out_dir}...")
    
    train_artifacts = train_mil(
        embeddings_dict=emb_dict,
        mola_dict=mola_dict,
        labels_dict=labels_dict,
        output_dir=out_dir,
        cfg=mil_cfg,
        device=device,
        mixed_precision=False,
        num_workers=0,
        seed=42,
    )
    
    # Evaluate
    predictions = evaluate_best_model(
        model_path=train_artifacts["best_model_path"],
        test_loader=train_artifacts["test_loader"],
        device=device,
        cfg=mil_cfg,
        output_dir=out_dir,
        mixed_precision=False,
    )
    
    # Load and print test metrics
    metrics = json.loads((out_dir / "test_metrics.json").read_text())
    print(f"\n{'='*60}")
    print(f"CLEANED V4 RESULTS")
    print(f"  Macro F1: {metrics['macro_f1_all']:.4f}")
    print(f"  Landform F1: {metrics['landform_macro_f1']:.4f}")
    for i, cls in enumerate(CLASS_ORDER):
        print(f"    {cls}: P={metrics['precision'][i]:.3f} R={metrics['recall'][i]:.3f} F1={metrics['f1'][i]:.3f}")
    print(f"{'='*60}")
    
    return metrics


if __name__ == "__main__":
    run_retrain()
