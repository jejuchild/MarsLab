#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/disk1/cspark/MarsLab")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.marslandform_v2.config import V3_EVAL_DIR, get_config
from scripts.marslandform_v2.models.tile_classifier import TileClassifierTrainer, TileLabelDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MarsLandformNet V3 tile classifier.")
    parser.add_argument(
        "--tile-labels",
        type=Path,
        default=ROOT / "Data/HiRISE/v3_output/tile_labels_v3.json",
        help="Path to tile_labels_v3.json",
    )
    parser.add_argument(
        "--tile-splits",
        type=Path,
        default=ROOT / "Data/HiRISE/v3_output/tile_splits_v3.json",
        help="Path to tile_splits_v3.json",
    )
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=ROOT / "Data/HiRISE/v2_output/embeddings_ssl",
        help="Directory with per-image embedding .npy files",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Max epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Tile batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--patience", type=int, default=15, help="Early-stopping patience")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Training device",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print dataset stats without training",
    )
    return parser.parse_args()


def _resolve_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _require_exists(path: Path, kind: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {kind}: {path}")


def _load_embeddings(embeddings_dir: Path) -> tuple[Path | None, dict[str, np.ndarray] | None]:
    by_image_path = embeddings_dir / "embeddings_by_image.npy"
    if by_image_path.exists():
        obj = np.load(by_image_path, allow_pickle=True)
        data = obj.item() if hasattr(obj, "item") else None
        if isinstance(data, dict):
            return None, data
    return embeddings_dir, None


def main() -> None:
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger("v3_train")

    _require_exists(args.tile_labels, "tile labels")
    _require_exists(args.tile_splits, "tile splits")
    _require_exists(args.embeddings_dir, "embeddings directory")

    with args.tile_labels.open() as f:
        tile_labels = json.load(f)
    with args.tile_splits.open() as f:
        splits = json.load(f)

    train_indices = splits.get("train", [])
    val_indices = splits.get("val", [])

    if not train_indices or not val_indices:
        raise ValueError("tile_splits must include non-empty train and val index arrays")

    cfg = get_config().tile_classifier
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.lr = args.lr
    cfg.patience = args.patience

    device = _resolve_device(args.device)
    logger.info("Device: %s", device)
    if device == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    per_image_dir, embeddings_by_image = _load_embeddings(args.embeddings_dir)

    train_dataset = TileLabelDataset(
        tile_labels=tile_labels,
        tile_indices=train_indices,
        embeddings_dir=per_image_dir,
        config=cfg,
        is_train=True,
        embeddings_by_image=embeddings_by_image,
    )
    val_dataset = TileLabelDataset(
        tile_labels=tile_labels,
        tile_indices=val_indices,
        embeddings_dir=per_image_dir,
        config=cfg,
        is_train=False,
        embeddings_by_image=embeddings_by_image,
    )

    logger.info("Train samples: %d", len(train_dataset))
    logger.info("Val samples: %d", len(val_dataset))

    if args.dry_run:
        logger.info("Dry-run complete. Inputs and dataset construction are valid.")
        return

    trainer = TileClassifierTrainer(
        config=cfg,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
    )
    result = trainer.train()

    V3_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = V3_EVAL_DIR / "v3_training_summary.json"
    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "best_f1": result["best_f1"],
        "best_epoch": result["best_epoch"],
        "device": device,
        "tile_labels": str(args.tile_labels),
        "tile_splits": str(args.tile_splits),
        "embeddings_dir": str(args.embeddings_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "patience": args.patience,
    }
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Saved training summary: %s", summary_path)


if __name__ == "__main__":
    main()
