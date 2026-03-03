#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

ROOT = Path(os.getenv("MARSLAB_ROOT", "/disk1/cspark/MarsLab"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.marslandform_v2.config import V3_CLASSES, get_config
from scripts.marslandform_v2.models.tile_classifier import TileLabelDataset, TileLandformClassifier


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate V3 tile classifier F1 on val/test.")
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "Data/HiRISE/v3_output/models/best_tile_classifier.pt",
    )
    p.add_argument(
        "--tile-labels",
        type=Path,
        default=ROOT / "Data/HiRISE/v3_output/tile_labels_v3.json",
    )
    p.add_argument(
        "--tile-splits",
        type=Path,
        default=ROOT / "Data/HiRISE/v3_output/tile_splits_v3.json",
    )
    p.add_argument(
        "--embeddings-dir",
        type=Path,
        default=ROOT / "Data/HiRISE/v2_output/embeddings_ssl",
    )
    p.add_argument("--split", choices=["val", "test", "both"], default="both")
    return p.parse_args()


def _load_dataset(split_name: str, cfg, tile_labels, splits, embeddings_by_image):
    return TileLabelDataset(
        tile_labels=tile_labels,
        tile_indices=splits[split_name],
        embeddings_dir=None,
        config=cfg,
        is_train=False,
        embeddings_by_image=embeddings_by_image,
    )


def _eval_dataset(model: TileLandformClassifier, ds: TileLabelDataset) -> dict[str, object]:
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for i in range(len(ds)):
            sample = ds[i]
            logits = model(sample["embedding"].unsqueeze(0), sample["mola"].unsqueeze(0))
            y_pred.append(int(torch.argmax(logits, dim=1).item()))
            y_true.append(int(sample["label"].item()))

    overall_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division="warn"))
    landform_idx = [i for i, label in enumerate(y_true) if label < 3]
    if landform_idx:
        yt = [y_true[i] for i in landform_idx]
        yp = [y_pred[i] for i in landform_idx]
        landform_f1 = float(f1_score(yt, yp, average="macro", zero_division="warn"))
    else:
        landform_f1 = 0.0

    per_class: dict[str, float] = {}
    for idx, cls_name in enumerate(V3_CLASSES):
        yt = [1 if y == idx else 0 for y in y_true]
        yp = [1 if y == idx else 0 for y in y_pred]
        per_class[cls_name] = float(f1_score(yt, yp, zero_division="warn"))

    return {
        "samples": len(ds),
        "overall_macro_f1": overall_f1,
        "landform_macro_f1": landform_f1,
        "per_class_f1": per_class,
    }


def main() -> None:
    args = parse_args()

    with args.tile_labels.open() as f:
        tile_labels = json.load(f)
    with args.tile_splits.open() as f:
        splits = json.load(f)

    obj = np.load(args.embeddings_dir / "embeddings_by_image.npy", allow_pickle=True)
    embeddings_by_image = obj.item()

    cfg = get_config().tile_classifier
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = TileLandformClassifier(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    requested = [args.split] if args.split != "both" else ["val", "test"]
    results = {}
    for split_name in requested:
        ds = _load_dataset(split_name, cfg, tile_labels, splits, embeddings_by_image)
        results[split_name] = _eval_dataset(model, ds)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
