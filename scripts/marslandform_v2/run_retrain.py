#!/usr/bin/env python3
"""Retrain MIL with cleaned labels and focal loss."""
import sys
from pathlib import Path

import torch

ROOT = Path("/disk1/cspark/MarsLab")
sys.path.insert(0, str(ROOT))

from scripts.marslandform_v2.config import CLASS_ORDER, get_config
from scripts.marslandform_v2.models.mil_classifier import (
    FocalLoss,
    evaluate_best_model,
    load_embeddings,
    load_labels,
    load_mola_features,
    train_mil,
)


cfg = get_config()
mil_cfg = cfg.mil
mil_cfg.epochs = 100
mil_cfg.patience = 20

device = torch.device(cfg.device)
output_dir = ROOT / "Data" / "HiRISE" / "v2_output" / "models" / "cleaned_focal"

embeddings = load_embeddings(ROOT / "Data" / "HiRISE" / "v2_output" / "embeddings_mil")
mola = load_mola_features(ROOT / "Data" / "HiRISE" / "v2_output" / "mola_features.npy")
labels = load_labels(ROOT / "Data" / "HiRISE" / "v2_output" / "labels_simple.json")

print(f"Embeddings: {len(embeddings)} images")
print(f"MOLA: {len(mola)} images")
print(f"Labels: {len(labels)} images")
print(f"Device: {device}")

artifacts = train_mil(
    embeddings_dict=embeddings,
    mola_dict=mola,
    labels_dict=labels,
    output_dir=output_dir,
    cfg=mil_cfg,
    device=device,
    mixed_precision=cfg.mixed_precision,
    num_workers=cfg.num_workers,
    seed=cfg.seed,
)

predictions = evaluate_best_model(
    model_path=artifacts["best_model_path"],
    test_loader=artifacts["test_loader"],
    device=device,
    cfg=mil_cfg,
    output_dir=output_dir,
    mixed_precision=cfg.mixed_precision,
)
print(f"Training complete. Results in {output_dir}")
