#!/usr/bin/env python3
"""
V5c Retraining — Class-balanced FiLM classifier with Focal Loss + EMA.

Improvements over V5b:
  1. Focal Loss (gamma=2.0) — down-weight easy samples, focus on hard ones
  2. WeightedRandomSampler — oversample minority classes (LVF, CCF, LDA)
  3. EMA (Exponential Moving Average) — smoother, more robust model
  4. Cosine warmup scheduler — avoid early divergence
  5. Gradient clipping — stabilize training

Uses pre-computed V5 assets:
  - embeddings_v5.npy (DINOv2+LoRA, 67904×768)
  - mola_features_v5.npy (corrected PDS extent, 8128 images)
  - tile_labels_v5.json (PDS extent-based polygon overlap)
  - tile_splits_v5.json (spatial splits)

Usage:
  nohup python3 retrain_v5c_balanced.py > /tmp/retrain_v5c.log 2>&1 &
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

ROOT = Path("/disk1/cspark/MarsLab")
V4_DIR = ROOT / "Data" / "HiRISE" / "v4_colab_data_expanded"
V5_DIR = ROOT / "Data" / "HiRISE" / "v5_retrain"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("/tmp/retrain_v5c.log")],
)
logger = logging.getLogger(__name__)

CLASS_NAMES = ["LDA", "LVF", "CCF", "OTHER"]


# ============================================================================
# Focal Loss
# ============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Down-weights easy (well-classified) samples, focuses learning on hard ones.
    With gamma=0, reduces to standard CrossEntropyLoss.
    """

    def __init__(
        self,
        alpha: torch.Tensor | None = None,
        gamma: float = 2.0,
        label_smoothing: float = 0.05,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.register_buffer("alpha", alpha)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(1)

        # Label smoothing
        if self.label_smoothing > 0:
            with torch.no_grad():
                smooth_targets = torch.full_like(logits, self.label_smoothing / (num_classes - 1))
                smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        else:
            smooth_targets = F.one_hot(targets, num_classes).float()

        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)

        # Focal modulation: (1 - p_t)^gamma
        focal_weight = (1.0 - probs) ** self.gamma

        # Per-class alpha weighting
        if self.alpha is not None:
            alpha_weight = self.alpha.unsqueeze(0).expand_as(logits)
            focal_weight = focal_weight * alpha_weight

        loss = -focal_weight * smooth_targets * log_probs
        loss = loss.sum(dim=1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ============================================================================
# EMA (Exponential Moving Average)
# ============================================================================

class EMA:
    """Maintains EMA of model parameters for smoother evaluation."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {name: param.clone().detach() for name, param in model.named_parameters()}

    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def apply(self, model: nn.Module) -> dict:
        """Apply EMA params and return backup of original params."""
        backup = {}
        for name, param in model.named_parameters():
            backup[name] = param.data.clone()
            param.data.copy_(self.shadow[name])
        return backup

    def restore(self, model: nn.Module, backup: dict) -> None:
        """Restore original params from backup."""
        for name, param in model.named_parameters():
            param.data.copy_(backup[name])


# ============================================================================
# Training
# ============================================================================

def train_v5c(
    embeddings: np.ndarray,
    mola_features: dict,
    tile_index: dict,
    labels_path: Path,
    splits_path: Path,
    output_dir: Path,
    epochs: int = 150,
    patience: int = 25,
    batch_size: int = 256,
    lr: float = 3e-4,
    focal_gamma: float = 2.0,
    ema_decay: float = 0.999,
):
    """Train FiLM classifier with focal loss + oversampling + EMA."""
    logger.info("V5c Training: Focal Loss + Oversampling + EMA")

    # Load labels & splits
    with open(labels_path) as f:
        labels_list = json.load(f)
    with open(splits_path) as f:
        splits = json.load(f)

    # Build label lookup
    labels_raw = {}
    for entry in labels_list:
        key = f"{entry['image_id']}_{entry['tile_row']}_{entry['tile_col']}"
        labels_raw[key] = entry.get("label", "OTHER")
    logger.info(f"  Labels loaded: {len(labels_raw)} entries")

    class_to_idx = {c: i for i, c in enumerate(CLASS_NAMES)}
    tile_keys = list(tile_index.keys())
    n = len(tile_keys)

    # Build aligned arrays
    mola_arr = np.zeros((n, 25), dtype=np.float32)
    label_arr = np.full(n, -1, dtype=np.int64)
    label_matched = 0

    for i, key in enumerate(tile_keys):
        parts = key.rsplit("_", 2)
        image_id = parts[0]
        row_col = f"{parts[1]}_{parts[2]}"

        if image_id in mola_features and row_col in mola_features[image_id]:
            mola_arr[i] = mola_features[image_id][row_col]

        if key in labels_raw:
            cls = labels_raw[key]
            label_arr[i] = class_to_idx.get(cls, class_to_idx["OTHER"])
            label_matched += 1

    logger.info(f"  Labels matched: {label_matched}/{n}")

    # Split indices
    train_idx = [int(i) for i in splits.get("train", []) if int(i) < n]
    val_idx = [int(i) for i in splits.get("val", []) if int(i) < n]
    test_idx = [int(i) for i in splits.get("test", []) if int(i) < n]

    # Filter unlabeled
    train_idx = [i for i in train_idx if label_arr[i] >= 0]
    val_idx = [i for i in val_idx if label_arr[i] >= 0]
    test_idx = [i for i in test_idx if label_arr[i] >= 0]

    logger.info(f"  Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # Class distribution
    train_labels = label_arr[train_idx]
    class_counts = np.bincount(train_labels, minlength=len(CLASS_NAMES))
    for i, name in enumerate(CLASS_NAMES):
        logger.info(f"    {name}: {class_counts[i]} ({100 * class_counts[i] / len(train_idx):.1f}%)")

    # ── WeightedRandomSampler for oversampling minority classes ──
    sample_weights = np.zeros(len(train_idx), dtype=np.float64)
    weight_per_class = 1.0 / (class_counts.astype(np.float64) + 1e-6)
    for i, idx in enumerate(train_idx):
        sample_weights[i] = weight_per_class[label_arr[idx]]

    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights),
        num_samples=len(train_idx),
        replacement=True,
    )

    # ── DataLoaders ──
    device = torch.device("cpu")

    def make_tensors(indices):
        emb_t = torch.tensor(embeddings[indices], dtype=torch.float32)
        mola_t = torch.tensor(mola_arr[indices], dtype=torch.float32)
        lab_t = torch.tensor(label_arr[indices], dtype=torch.long)
        return emb_t, mola_t, lab_t

    train_emb, train_mola, train_lab = make_tensors(train_idx)
    train_ds = TensorDataset(train_emb, train_mola, train_lab)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0)

    val_emb, val_mola, val_lab = make_tensors(val_idx)
    val_ds = TensorDataset(val_emb, val_mola, val_lab)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    test_emb, test_mola, test_lab = make_tensors(test_idx)
    test_ds = TensorDataset(test_emb, test_mola, test_lab)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # ── Model ──
    from scripts.marslandform_v2.models.film_classifier import FiLMClassifier

    model = FiLMClassifier(
        visual_dim=768,
        mola_dim=25,
        num_classes=len(CLASS_NAMES),
        film_hidden=64,
        head_hidden=128,
        dropout=0.4,
    )
    model.to(device)

    # ── Focal Loss with per-class alpha ──
    # Alpha = inverse sqrt frequency (less aggressive than inverse frequency)
    alpha = 1.0 / np.sqrt(class_counts.astype(np.float32) + 1e-6)
    alpha /= alpha.sum()
    alpha_t = torch.tensor(alpha, dtype=torch.float32).to(device)
    logger.info(f"  Focal alpha: {dict(zip(CLASS_NAMES, alpha.round(4)))}")
    logger.info(f"  Focal gamma: {focal_gamma}")

    criterion = FocalLoss(alpha=alpha_t, gamma=focal_gamma, label_smoothing=0.05)

    # ── Optimizer + Scheduler ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)

    # Cosine with linear warmup
    warmup_epochs = 10
    total_steps = epochs * len(train_loader)
    warmup_steps = warmup_epochs * len(train_loader)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── EMA ──
    ema = EMA(model, decay=ema_decay)

    # ── Training Loop ──
    best_f1 = 0.0
    best_epoch = 0
    no_improve = 0
    best_state = None
    global_step = 0
    t0 = time.time()

    from sklearn.metrics import f1_score, accuracy_score

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0

        for emb_b, mola_b, lab_b in train_loader:
            emb_b, mola_b, lab_b = emb_b.to(device), mola_b.to(device), lab_b.to(device)

            optimizer.zero_grad()
            logits = model(emb_b, mola_b)
            loss = criterion(logits, lab_b)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()
            ema.update(model)

            train_loss += loss.item()
            n_batches += 1
            global_step += 1

        train_loss /= max(n_batches, 1)

        # ── Validate with EMA model ──
        backup = ema.apply(model)
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for emb_b, mola_b, lab_b in val_loader:
                logits = model(emb_b, mola_b)
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.numpy())
                all_labels.extend(lab_b.numpy())

        val_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        val_acc = accuracy_score(all_labels, all_preds)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        ema.restore(model, backup)

        if epoch % 5 == 0 or epoch == 1 or no_improve == 0:
            elapsed = time.time() - t0
            logger.info(
                f"  Epoch {epoch:3d}: loss={train_loss:.4f} "
                f"val_f1={val_f1:.4f} val_acc={val_acc:.4f} "
                f"lr={optimizer.param_groups[0]['lr']:.2e} "
                f"[{elapsed / 60:.0f}m] "
                f"{'*best*' if no_improve == 0 else ''}"
            )

        if no_improve >= patience:
            logger.info(f"  Early stopping at epoch {epoch} (best={best_epoch})")
            break

    # ── Test with best EMA model ──
    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for emb_b, mola_b, lab_b in test_loader:
            logits = model(emb_b, mola_b)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.numpy())
            all_labels.extend(lab_b.numpy())

    from sklearn.metrics import classification_report
    report = classification_report(all_labels, all_preds, target_names=CLASS_NAMES, digits=4)
    test_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    test_acc = accuracy_score(all_labels, all_preds)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"TRAINING COMPLETE — best epoch {best_epoch}")
    logger.info(f"  Val  F1={best_f1:.4f}")
    logger.info(f"  Test F1={test_f1:.4f}, Acc={test_acc:.4f}")
    logger.info(f"\n{report}")

    # ── Save ──
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "film_classifier_v5c.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": best_epoch,
            "val_f1": best_f1,
            "test_f1": test_f1,
            "test_acc": test_acc,
            "cfg": {
                "visual_dim": 768,
                "mola_dim": 25,
                "num_classes": len(CLASS_NAMES),
                "film_hidden": 64,
                "head_hidden": 128,
                "dropout": 0.4,
                "class_names": CLASS_NAMES,
                "pixel_scale_fix": "pds_extent",
                "focal_gamma": focal_gamma,
                "ema_decay": ema_decay,
                "balancing": "weighted_random_sampler + focal_loss",
            },
        },
        save_path,
    )
    logger.info(f"  Model saved: {save_path}")

    with open(output_dir / "test_report_v5c.txt", "w") as f:
        f.write(f"V5c — Focal Loss + Oversampling + EMA\n")
        f.write(f"Best epoch: {best_epoch}\n")
        f.write(f"Val F1: {best_f1:.4f}\n")
        f.write(f"Test F1: {test_f1:.4f}\n")
        f.write(f"Test Acc: {test_acc:.4f}\n\n")
        f.write(report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("V5c Retraining — Class-balanced FiLM classifier")
    logger.info("=" * 60)

    # Load pre-computed assets
    with open(V4_DIR / "tile_index.json") as f:
        tile_index = json.load(f)
    logger.info(f"Tile index: {len(tile_index)} tiles")

    embeddings = np.load(str(V5_DIR / "embeddings_v5.npy"))
    logger.info(f"Embeddings: {embeddings.shape}")

    mola = np.load(str(V5_DIR / "mola_features_v5.npy"), allow_pickle=True).item()
    logger.info(f"MOLA features: {len(mola)} images")

    train_v5c(
        embeddings=embeddings,
        mola_features=mola,
        tile_index=tile_index,
        labels_path=V5_DIR / "tile_labels_v5.json",
        splits_path=V5_DIR / "tile_splits_v5.json",
        output_dir=V5_DIR,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        lr=args.lr,
        focal_gamma=args.focal_gamma,
    )

    logger.info("\n" + "=" * 60)
    logger.info("ALL DONE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
