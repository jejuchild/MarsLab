"""
V3 Tile-Level Classifier.

Pure per-tile classification: DINOv2 embedding (768) + MOLA features (25) → 4 classes.
No MIL aggregation, no attention — each tile is independently classified.

Classes: LDA (0), LVF (1), CCF (2), OTHER (3)
"""
from __future__ import annotations

import json
import logging
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from ..config import TileClassifierConfig, V3_CLASSES, V3_OUTPUT, V3_MODELS_DIR

logger = logging.getLogger(__name__)


# ── Model ────────────────────────────────────────────────────────────────────

class TileLandformClassifier(nn.Module):
    """Simple tile-level classifier: concat(embedding, mola) → MLP → logits."""

    def __init__(self, config: TileClassifierConfig):
        super().__init__()
        self.config = config
        input_dim = config.embed_dim + config.mola_dim  # 768 + 25 = 793

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_classes),
        )

    def forward(
        self,
        embeddings: torch.Tensor,  # (B, 768)
        mola: torch.Tensor,        # (B, 25)
    ) -> torch.Tensor:
        """Returns logits (B, num_classes)."""
        x = torch.cat([embeddings, mola], dim=1)  # (B, 793)
        return self.classifier(x)


# ── Focal Loss ───────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """Focal loss with label smoothing and per-class weights."""

    def __init__(
        self,
        gamma: float = 1.5,
        label_smoothing: float = 0.1,
        weight: torch.Tensor | None = None,
    ):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.class_weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, C) raw logits
        targets: (B,) integer class labels
        """
        num_classes = logits.shape[1]
        ce = F.cross_entropy(
            logits, targets,
            weight=self.class_weight,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        probs = F.softmax(logits, dim=1)
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1.0 - target_probs) ** self.gamma
        return (focal_weight * ce).mean()


# ── Dataset ──────────────────────────────────────────────────────────────────

class TileLabelDataset(Dataset[dict[str, Any]]):
    """Dataset for tile-level classification from precomputed embeddings."""

    def __init__(
        self,
        tile_labels: list[dict[str, Any]],
        tile_indices: list[int],
        embeddings_dir: Path | None,
        config: TileClassifierConfig,
        is_train: bool = True,
        embeddings_by_image: dict[str, np.ndarray] | None = None,
    ):
        self.config = config
        self.is_train = is_train
        self.embeddings_dir = embeddings_dir
        self.embeddings_by_image = embeddings_by_image

        # Filter to usable tiles (confident, mixed, or OTHER)
        self.samples: list[dict[str, Any]] = []
        other_samples: list[dict[str, Any]] = []

        for idx in tile_indices:
            t = tile_labels[idx]
            label = t.get("label")
            label_type = t.get("label_type", "")

            if label == "UNLABELED" or label is None:
                continue

            # Load embedding path
            img_id = t["image_id"]
            emb_path = None
            if self.embeddings_by_image is None:
                if self.embeddings_dir is None:
                    continue
                emb_candidate = self.embeddings_dir / f"{img_id}.npy"
                if not emb_candidate.exists():
                    continue
                emb_path = emb_candidate
            elif img_id not in self.embeddings_by_image:
                continue

            sample = {
                "image_id": img_id,
                "tile_idx": t["tile_idx"],
                "emb_path": str(emb_path) if emb_path is not None else None,
                "label": label,
                "label_type": label_type,
                "coverage": t.get("coverage", {}),
            }

            if label == "OTHER":
                other_samples.append(sample)
            else:
                self.samples.append(sample)

        # Subsample OTHER to prevent class imbalance
        if is_train and other_samples:
            import random
            n_keep = max(1, int(len(other_samples) * config.other_subsample_ratio))
            random.shuffle(other_samples)
            other_samples = other_samples[:n_keep]

        self.samples.extend(other_samples)

        # Class to index mapping
        self.class_to_idx = {cls: i for i, cls in enumerate(V3_CLASSES)}

        logger.info(
            "TileLabelDataset: %d samples (%s)",
            len(self.samples),
            dict(Counter(s["label"] for s in self.samples)),
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]

        if self.embeddings_by_image is not None:
            all_embeddings = self.embeddings_by_image[sample["image_id"]]
        else:
            emb_path = sample["emb_path"]
            if emb_path is None:
                all_embeddings = np.zeros((1, self.config.embed_dim), dtype=np.float32)
            else:
                all_embeddings = np.load(emb_path)

        tile_idx = sample["tile_idx"]
        if tile_idx < len(all_embeddings):
            embedding = all_embeddings[tile_idx]
        else:
            embedding = np.zeros((self.config.embed_dim,), dtype=np.float32)

        # MOLA features placeholder (filled by collate or precomputed)
        mola = np.zeros((self.config.mola_dim,), dtype=np.float32)

        # Label
        label_idx = self.class_to_idx.get(sample["label"], self.class_to_idx["OTHER"])

        result = {
            "embedding": torch.from_numpy(embedding).float(),
            "mola": torch.from_numpy(mola).float(),
            "label": torch.tensor(label_idx, dtype=torch.long),
            "label_type": sample["label_type"],
        }

        # For mixed tiles: soft target from coverage fractions
        if sample["label_type"] == "mixed":
            coverage = sample.get("coverage", {})
            soft_target = torch.zeros(self.config.num_classes, dtype=torch.float32)
            for cls, frac in coverage.items():
                if cls in self.class_to_idx:
                    soft_target[self.class_to_idx[cls]] = frac
            # Fill remaining with OTHER
            remaining = 1.0 - soft_target.sum()
            if remaining > 0:
                soft_target[self.class_to_idx["OTHER"]] = remaining
            result["soft_target"] = soft_target

        return result

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights."""
        counts = Counter(s["label"] for s in self.samples)
        total = len(self.samples)
        weights = torch.zeros(len(V3_CLASSES))
        for cls, idx in self.class_to_idx.items():
            cnt = counts.get(cls, 1)
            weights[idx] = total / (len(V3_CLASSES) * cnt)
        return weights

    def get_sample_weights(self) -> torch.Tensor:
        """Per-sample weights for WeightedRandomSampler."""
        class_weights = self.get_class_weights()
        return torch.tensor([
            float(class_weights[self.class_to_idx.get(s["label"], 3)])
            for s in self.samples
        ])


# ── Trainer ──────────────────────────────────────────────────────────────────

class TileClassifierTrainer:
    """Training loop for V3 tile classifier."""

    def __init__(
        self,
        config: TileClassifierConfig,
        train_dataset: TileLabelDataset,
        val_dataset: TileLabelDataset,
        device: torch.device | str = "cpu",
    ):
        self.config = config
        self.device = torch.device(device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        # Model
        self.model = TileLandformClassifier(config).to(self.device)
        logger.info("Model params: %d", sum(p.numel() for p in self.model.parameters()))

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        # Scheduler: warmup + cosine
        total_steps = config.epochs * (len(train_dataset) // config.batch_size + 1)
        warmup_steps = int(total_steps * 0.1)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.lr,
            total_steps=total_steps,
            pct_start=warmup_steps / total_steps,
            anneal_strategy="cos",
        )

        # Loss
        class_weights = train_dataset.get_class_weights().to(self.device)
        self.criterion = FocalLoss(
            gamma=config.focal_gamma,
            label_smoothing=config.label_smoothing,
            weight=class_weights,
        )

        # Dataloaders
        train_sampler = WeightedRandomSampler(
            weights=train_dataset.get_sample_weights().tolist(),
            num_samples=len(train_dataset),
            replacement=True,
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            sampler=train_sampler,
            num_workers=2,
            pin_memory=True,
            collate_fn=self._collate,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size * 2,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            collate_fn=self._collate,
        )

        # Tracking
        self.best_f1 = 0.0
        self.best_epoch = -1
        self.patience_counter = 0

    @staticmethod
    def _collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor | torch.BoolTensor]:
        """Custom collate to handle variable-presence soft_target."""
        result = {
            "embedding": torch.stack([b["embedding"] for b in batch]),
            "mola": torch.stack([b["mola"] for b in batch]),
            "label": torch.stack([b["label"] for b in batch]),
        }
        # Soft targets only for mixed tiles
        if any("soft_target" in b for b in batch):
            soft = []
            has_soft = []
            for b in batch:
                if "soft_target" in b:
                    soft.append(b["soft_target"])
                    has_soft.append(True)
                else:
                    soft.append(torch.zeros_like(batch[0].get("soft_target", torch.zeros(4))))
                    has_soft.append(False)
            result["soft_target"] = torch.stack(soft)
            result["has_soft"] = torch.tensor(has_soft, dtype=torch.bool)
        return result

    def train_epoch(self) -> float:
        """Run one training epoch. Returns average loss."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in self.train_loader:
            emb = batch["embedding"].to(self.device)
            mola = batch["mola"].to(self.device)
            labels = batch["label"].to(self.device)

            logits = self.model(emb, mola)

            # Hard label loss (focal)
            loss = self.criterion(logits, labels)

            # Soft label loss for mixed tiles (KL-divergence)
            if "soft_target" in batch and "has_soft" in batch:
                soft_targets = batch["soft_target"].to(self.device)
                has_soft = batch["has_soft"].to(self.device)
                if has_soft.any():
                    soft_logits = logits[has_soft]
                    soft_t = soft_targets[has_soft]
                    log_probs = F.log_softmax(soft_logits, dim=1)
                    kl_loss = F.kl_div(log_probs, soft_t, reduction="batchmean")
                    loss = loss + self.config.mixed_loss_weight * kl_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def evaluate(self) -> dict[str, Any]:
        """Evaluate on validation set. Returns metrics dict."""
        self.model.eval()
        all_preds: list[int] = []
        all_labels: list[int] = []
        total_loss = 0.0
        n_batches = 0

        for batch in self.val_loader:
            emb = batch["embedding"].to(self.device)
            mola = batch["mola"].to(self.device)
            labels = batch["label"].to(self.device)

            logits = self.model(emb, mola)
            loss = self.criterion(logits, labels)
            total_loss += loss.item()
            n_batches += 1

            preds = logits.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())

        # Compute per-class metrics
        from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

        # Landform macro-F1 (exclude OTHER = index 3)
        landform_mask = [i for i, l in enumerate(all_labels) if l < 3]
        if landform_mask:
            lf_labels = [all_labels[i] for i in landform_mask]
            lf_preds = [all_preds[i] for i in landform_mask]
            landform_f1 = f1_score(lf_labels, lf_preds, average="macro", zero_division="warn")
        else:
            landform_f1 = 0.0

        overall_f1 = f1_score(all_labels, all_preds, average="macro", zero_division="warn")
        accuracy = accuracy_score(all_labels, all_preds)

        per_class: dict[str, dict[str, float]] = {}
        for cls_idx, cls_name in enumerate(V3_CLASSES):
            cls_mask_true = [1 if l == cls_idx else 0 for l in all_labels]
            cls_mask_pred = [1 if p == cls_idx else 0 for p in all_preds]
            per_class[cls_name] = {
                "precision": precision_score(cls_mask_true, cls_mask_pred, zero_division="warn"),
                "recall": recall_score(cls_mask_true, cls_mask_pred, zero_division="warn"),
                "f1": f1_score(cls_mask_true, cls_mask_pred, zero_division="warn"),
                "support": sum(cls_mask_true),
            }

        return {
            "val_loss": total_loss / max(n_batches, 1),
            "accuracy": accuracy,
            "overall_macro_f1": overall_f1,
            "landform_macro_f1": landform_f1,
            "per_class": per_class,
        }

    def train(self) -> dict[str, Any]:
        """Full training loop with early stopping."""
        logger.info("Starting V3 tile classifier training...")
        logger.info("  Train: %d tiles, Val: %d tiles", len(self.train_dataset), len(self.val_dataset))
        logger.info("  Config: %s", asdict(self.config))

        history: list[dict[str, Any]] = []

        for epoch in range(1, self.config.epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch()
            metrics = self.evaluate()
            elapsed = time.time() - t0

            metrics["epoch"] = epoch
            metrics["train_loss"] = train_loss
            metrics["elapsed_s"] = round(elapsed, 1)
            history.append(metrics)

            lf1 = metrics["landform_macro_f1"]
            logger.info(
                "Epoch %3d/%d  train_loss=%.4f  val_loss=%.4f  lf_F1=%.4f  acc=%.4f  (%.1fs)",
                epoch, self.config.epochs, train_loss, metrics["val_loss"],
                lf1, metrics["accuracy"], elapsed,
            )

            # Early stopping on landform macro-F1
            if lf1 > self.best_f1:
                self.best_f1 = lf1
                self.best_epoch = epoch
                self.patience_counter = 0
                self._save_checkpoint(epoch, metrics)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.patience:
                    logger.info("Early stopping at epoch %d (patience=%d)", epoch, self.config.patience)
                    break

        logger.info("Training complete. Best landform F1=%.4f at epoch %d", self.best_f1, self.best_epoch)

        # Save training history
        history_path = V3_MODELS_DIR / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2, default=str)

        return {"best_f1": self.best_f1, "best_epoch": self.best_epoch, "history": history}

    def _save_checkpoint(self, epoch: int, metrics: dict[str, Any]) -> None:
        """Save best model checkpoint."""
        V3_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        path = V3_MODELS_DIR / "best_tile_classifier.pt"
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": asdict(self.config),
            "epoch": epoch,
            "best_landform_macro_f1": metrics["landform_macro_f1"],
            "metrics": metrics,
            "classes": V3_CLASSES,
        }, path)
        logger.info("  Saved checkpoint: %s (F1=%.4f)", path, metrics["landform_macro_f1"])
