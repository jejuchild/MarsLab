from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

try:
    from ..config import CLASS_ORDER, MILConfig, get_config
except ImportError:
    import sys

    ROOT = Path(__file__).resolve().parents[3]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scripts.marslandform_v2.config import CLASS_ORDER, MILConfig, get_config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _label_to_int(label: Any) -> int:
    if isinstance(label, int):
        return label
    if isinstance(label, np.integer):
        return int(label)
    if isinstance(label, str):
        if label in CLASS_ORDER:
            return CLASS_ORDER.index(label)
        if label.isdigit():
            return int(label)
    raise ValueError(f"Unsupported label format: {label}")


def _coerce_dict(payload: Any, expected_ndim: int | None = None) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    if isinstance(payload, dict):
        items = payload.items()
    else:
        raise ValueError("Expected dictionary-like payload")
    for key, value in items:
        arr = np.asarray(value, dtype=np.float32)
        if expected_ndim is not None and arr.ndim != expected_ndim:
            raise ValueError(f"Array for {key} has ndim={arr.ndim}, expected {expected_ndim}")
        out[str(key)] = arr
    return out


def load_embeddings(embeddings_dir: Path) -> Dict[str, np.ndarray]:
    if not embeddings_dir.exists():
        raise FileNotFoundError(f"Embeddings dir not found: {embeddings_dir}")

    embeddings: Dict[str, np.ndarray] = {}
    files = sorted([p for p in embeddings_dir.iterdir() if p.suffix.lower() in {".npy", ".npz", ".pt", ".pth"}])
    if not files:
        raise ValueError(f"No embedding files found in {embeddings_dir}")

    for file_path in files:
        suffix = file_path.suffix.lower()
        if suffix == ".npy":
            data = np.load(file_path, allow_pickle=True)
            if isinstance(data, np.ndarray) and data.dtype == object and data.shape == ():
                payload = data.item()
                embeddings.update(_coerce_dict(payload, expected_ndim=2))
            elif isinstance(data, np.ndarray) and data.ndim == 2:
                embeddings[file_path.stem] = data.astype(np.float32)
            else:
                raise ValueError(f"Unsupported .npy format in {file_path}")
        elif suffix == ".npz":
            data = np.load(file_path, allow_pickle=True)
            for key in data.files:
                arr = np.asarray(data[key], dtype=np.float32)
                if arr.ndim != 2:
                    raise ValueError(f"Embedding for {key} in {file_path} must be 2D")
                embeddings[str(key)] = arr
        else:
            data = torch.load(file_path, map_location="cpu")
            if isinstance(data, dict):
                for key, value in data.items():
                    arr = np.asarray(value, dtype=np.float32)
                    if arr.ndim != 2:
                        raise ValueError(f"Embedding for {key} in {file_path} must be 2D")
                    embeddings[str(key)] = arr
            else:
                raise ValueError(f"Unsupported torch payload in {file_path}")

    return embeddings


def load_mola_features(mola_path: Path) -> Dict[str, np.ndarray]:
    if not mola_path.exists():
        raise FileNotFoundError(f"MOLA path not found: {mola_path}")

    suffix = mola_path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(mola_path.read_text())
        return _coerce_dict(payload, expected_ndim=1)
    if suffix == ".npz":
        data = np.load(mola_path, allow_pickle=True)
        return {str(key): np.asarray(data[key], dtype=np.float32) for key in data.files}
    if suffix == ".npy":
        data = np.load(mola_path, allow_pickle=True)
        if isinstance(data, np.ndarray) and data.dtype == object and data.shape == ():
            payload = data.item()
            return _coerce_dict(payload, expected_ndim=1)
        raise ValueError("Expected dict-like object in mola .npy file")
    raise ValueError(f"Unsupported MOLA format: {mola_path}")


def load_labels(labels_path: Path) -> Dict[str, int]:
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels path not found: {labels_path}")
    payload = json.loads(labels_path.read_text())
    if isinstance(payload, dict) and "labels" in payload and isinstance(payload["labels"], dict):
        payload = payload["labels"]
    if not isinstance(payload, dict):
        raise ValueError("labels_path must contain a dict-like mapping image_id -> label")
    return {str(k): _label_to_int(v) for k, v in payload.items()}


def intersect_image_ids(
    embeddings_dict: Dict[str, np.ndarray],
    mola_dict: Dict[str, np.ndarray],
    labels_dict: Dict[str, int],
    mola_dim: int,
) -> List[str]:
    common = sorted(set(embeddings_dict) & set(mola_dict) & set(labels_dict))
    valid: List[str] = []
    for image_id in common:
        emb = np.asarray(embeddings_dict[image_id])
        mola = np.asarray(mola_dict[image_id])
        if emb.ndim != 2:
            continue
        if mola.ndim != 1 or mola.shape[0] != mola_dim:
            continue
        if emb.shape[0] <= 0:
            continue
        valid.append(image_id)
    if not valid:
        raise ValueError("No overlapping valid image ids across embeddings, MOLA, labels")
    return valid


class MILDataset(Dataset[Tuple[torch.Tensor, torch.Tensor, int, str]]):
    def __init__(
        self,
        image_ids: Sequence[str],
        embeddings_dict: Dict[str, np.ndarray],
        mola_dict: Dict[str, np.ndarray],
        labels_dict: Dict[str, int],
        min_tiles_per_image: int,
        max_tiles_per_image: int,
    ) -> None:
        self.image_ids = list(image_ids)
        self.embeddings_dict = embeddings_dict
        self.mola_dict = mola_dict
        self.labels_dict = labels_dict
        self.min_tiles_per_image = min_tiles_per_image
        self.max_tiles_per_image = max_tiles_per_image

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, str]:
        image_id = self.image_ids[idx]
        tiles = np.asarray(self.embeddings_dict[image_id], dtype=np.float32)
        mola = np.asarray(self.mola_dict[image_id], dtype=np.float32)
        label = int(self.labels_dict[image_id])

        n_tiles = tiles.shape[0]
        if n_tiles > self.max_tiles_per_image:
            keep = np.random.choice(n_tiles, size=self.max_tiles_per_image, replace=False)
            tiles = tiles[keep]
            n_tiles = tiles.shape[0]

        if n_tiles < self.min_tiles_per_image:
            needed = self.min_tiles_per_image - n_tiles
            pad = np.zeros((needed, tiles.shape[1]), dtype=np.float32)
            tiles = np.concatenate([tiles, pad], axis=0)

        return torch.from_numpy(tiles), torch.from_numpy(mola), label, image_id


def mil_collate_fn(batch: Sequence[Tuple[torch.Tensor, torch.Tensor, int, str]]) -> Dict[str, Any]:
    tile_seqs, mola_feats, labels, image_ids = zip(*batch)
    batch_size = len(tile_seqs)
    max_tiles = max(x.size(0) for x in tile_seqs)
    embed_dim = tile_seqs[0].size(1)

    padded = torch.zeros(batch_size, max_tiles, embed_dim, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_tiles, dtype=torch.bool)
    for i, tiles in enumerate(tile_seqs):
        n = tiles.size(0)
        padded[i, :n] = tiles
        mask[i, :n] = True

    return {
        "tile_embeddings": padded,
        "tile_mask": mask,
        "mola_features": torch.stack(mola_feats).float(),
        "labels": torch.tensor(labels, dtype=torch.long),
        "image_ids": list(image_ids),
    }


class AttentionMILClassifier(nn.Module):
    def __init__(self, cfg: MILConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.tile_transform = nn.Sequential(
            nn.Linear(cfg.embed_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )

        self.attention_v = nn.Linear(cfg.hidden_dim, cfg.attention_dim)
        self.attention_u = nn.Linear(cfg.hidden_dim, cfg.attention_dim)
        self.attention_w = nn.Linear(cfg.attention_dim, 1)

        self.classifier = nn.Sequential(
            nn.Linear(cfg.hidden_dim + cfg.mola_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.num_classes),
        )

    def forward(
        self,
        tile_embeddings: torch.Tensor,
        tile_mask: torch.Tensor,
        mola_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.tile_transform(tile_embeddings)

        att_v = torch.tanh(self.attention_v(features))
        att_u = torch.sigmoid(self.attention_u(features))
        att_logits = self.attention_w(att_v * att_u).squeeze(-1)
        att_logits = att_logits.masked_fill(~tile_mask, torch.finfo(att_logits.dtype).min)
        att_weights = torch.softmax(att_logits, dim=1)

        bag_feature = torch.sum(att_weights.unsqueeze(-1) * features, dim=1)
        fused = torch.cat([bag_feature, mola_features], dim=1)
        logits = self.classifier(fused)
        return logits, att_weights


def build_scheduler(optimizer: torch.optim.Optimizer, total_steps: int, warmup_ratio: float = 0.1) -> LambdaLR:
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def compute_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    num_classes: int,
    background_idx: int = 4,
) -> Dict[str, Any]:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for truth, pred in zip(y_true, y_pred):
        if 0 <= truth < num_classes and 0 <= pred < num_classes:
            cm[truth, pred] += 1

    precision = np.zeros(num_classes, dtype=np.float32)
    recall = np.zeros(num_classes, dtype=np.float32)
    f1 = np.zeros(num_classes, dtype=np.float32)
    support = cm.sum(axis=1)

    for idx in range(num_classes):
        tp = float(cm[idx, idx])
        fp = float(cm[:, idx].sum() - cm[idx, idx])
        fn = float(cm[idx, :].sum() - cm[idx, idx])
        precision[idx] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall[idx] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        denom = precision[idx] + recall[idx]
        f1[idx] = (2.0 * precision[idx] * recall[idx] / denom) if denom > 0 else 0.0

    macro_f1_all = float(np.mean(f1))
    landform_indices = [i for i in range(num_classes) if i != background_idx]
    landform_macro_f1 = float(np.mean(f1[landform_indices])) if landform_indices else 0.0

    return {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "support": support.tolist(),
        "confusion_matrix": cm.tolist(),
        "macro_f1_all": macro_f1_all,
        "landform_macro_f1": landform_macro_f1,
    }


@torch.no_grad()
def run_epoch_eval(
    model: AttentionMILClassifier,
    loader: DataLoader[Any],
    device: torch.device,
    criterion: nn.Module,
    use_amp: bool,
) -> Dict[str, Any]:
    model.eval()
    losses: List[float] = []
    y_true: List[int] = []
    y_pred: List[int] = []

    for batch in loader:
        tiles = batch["tile_embeddings"].to(device)
        mask = batch["tile_mask"].to(device)
        mola = batch["mola_features"].to(device)
        labels = batch["labels"].to(device)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits, _ = model(tiles, mask, mola)
            loss = criterion(logits, labels)

        losses.append(float(loss.detach().cpu().item()))
        preds = torch.argmax(logits, dim=1)
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())

    metrics = compute_metrics(y_true, y_pred, num_classes=model.cfg.num_classes)
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    return metrics


def print_epoch_metrics(epoch: int, metrics: Dict[str, Any]) -> None:
    print(
        f"[Val][Epoch {epoch}] loss={metrics['loss']:.4f} "
        f"macro_f1={metrics['macro_f1_all']:.4f} "
        f"landform_macro_f1={metrics['landform_macro_f1']:.4f}"
    )
    print("Class     Precision  Recall  F1   Support")
    for i, cls_name in enumerate(CLASS_ORDER):
        print(
            f"{cls_name:<10}{metrics['precision'][i]:>8.3f}"
            f"{metrics['recall'][i]:>8.3f}{metrics['f1'][i]:>7.3f}{int(metrics['support'][i]):>9d}"
        )
    print("Confusion Matrix:")
    cm = np.asarray(metrics["confusion_matrix"], dtype=int)
    for row in cm:
        print(" ".join(f"{v:4d}" for v in row))


def split_image_ids(image_ids: Sequence[str], labels_dict: Dict[str, int], cfg: MILConfig, seed: int) -> Tuple[List[str], List[str], List[str]]:
    labels = [labels_dict[i] for i in image_ids]
    train_ids, temp_ids, train_labels, temp_labels = train_test_split(
        list(image_ids),
        labels,
        test_size=(1.0 - cfg.train_ratio),
        random_state=seed,
        stratify=labels,
    )
    val_fraction_of_temp = cfg.val_ratio / (cfg.val_ratio + cfg.test_ratio)
    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=(1.0 - val_fraction_of_temp),
        random_state=seed,
        stratify=temp_labels,
    )
    return train_ids, val_ids, test_ids


def make_loaders(
    train_ids: Sequence[str],
    val_ids: Sequence[str],
    test_ids: Sequence[str],
    embeddings_dict: Dict[str, np.ndarray],
    mola_dict: Dict[str, np.ndarray],
    labels_dict: Dict[str, int],
    cfg: MILConfig,
    num_workers: int,
) -> Tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]]:
    train_ds = MILDataset(
        train_ids,
        embeddings_dict,
        mola_dict,
        labels_dict,
        min_tiles_per_image=cfg.min_tiles_per_image,
        max_tiles_per_image=cfg.max_tiles_per_image,
    )
    val_ds = MILDataset(
        val_ids,
        embeddings_dict,
        mola_dict,
        labels_dict,
        min_tiles_per_image=cfg.min_tiles_per_image,
        max_tiles_per_image=cfg.max_tiles_per_image,
    )
    test_ds = MILDataset(
        test_ids,
        embeddings_dict,
        mola_dict,
        labels_dict,
        min_tiles_per_image=cfg.min_tiles_per_image,
        max_tiles_per_image=cfg.max_tiles_per_image,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=mil_collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=mil_collate_fn,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=mil_collate_fn,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader


def make_class_weights(labels: Iterable[int], num_classes: int, device: torch.device) -> torch.Tensor:
    labels_list = list(labels)
    counts = np.bincount(labels_list, minlength=num_classes).astype(np.float32)
    weights = np.zeros_like(counts, dtype=np.float32)
    nonzero = counts > 0
    weights[nonzero] = 1.0 / counts[nonzero]
    if np.sum(weights) > 0:
        weights = weights * (num_classes / np.sum(weights))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def save_confusion_matrix_png(confusion: Sequence[Sequence[int]], class_names: Sequence[str], out_path: Path) -> None:
    cm = np.asarray(confusion)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(class_names)), yticks=np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")
    thresh = cm.max() / 2.0 if cm.size > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(int(cm[i, j]), "d"), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


class FocalLoss(nn.Module):
    """Focal Loss for class-imbalanced classification (Lin et al. 2017)."""

    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.0, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.register_buffer("weight", weight)

    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        return focal_loss


def train_mil(
    embeddings_dict: Dict[str, np.ndarray],
    mola_dict: Dict[str, np.ndarray],
    labels_dict: Dict[str, int],
    output_dir: Path,
    cfg: MILConfig,
    device: torch.device,
    mixed_precision: bool,
    num_workers: int,
    seed: int,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    image_ids = intersect_image_ids(embeddings_dict, mola_dict, labels_dict, mola_dim=cfg.mola_dim)
    train_ids, val_ids, test_ids = split_image_ids(image_ids, labels_dict, cfg, seed)

    train_loader, val_loader, test_loader = make_loaders(
        train_ids,
        val_ids,
        test_ids,
        embeddings_dict,
        mola_dict,
        labels_dict,
        cfg,
        num_workers,
    )

    model = AttentionMILClassifier(cfg).to(device)
    class_weights = make_class_weights((labels_dict[i] for i in train_ids), cfg.num_classes, device)
    criterion = FocalLoss(weight=class_weights, gamma=2.0, label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = max(1, len(train_loader) * cfg.epochs)
    scheduler = build_scheduler(optimizer, total_steps=total_steps, warmup_ratio=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=mixed_precision)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_macro_f1": [],
        "val_landform_macro_f1": [],
    }

    best_metric = -1.0
    best_epoch = -1
    patience_counter = 0
    best_model_path = output_dir / "best_mil_model.pt"

    global_step = 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_losses: List[float] = []

        for batch in train_loader:
            tiles = batch["tile_embeddings"].to(device)
            mask = batch["tile_mask"].to(device)
            mola = batch["mola_features"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=mixed_precision):
                logits, _ = model(tiles, mask, mola)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1
            train_losses.append(float(loss.detach().cpu().item()))

        val_metrics = run_epoch_eval(model, val_loader, device, criterion, mixed_precision)
        print_epoch_metrics(epoch, val_metrics)

        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_macro_f1"].append(val_metrics["macro_f1_all"])
        history["val_landform_macro_f1"].append(val_metrics["landform_macro_f1"])

        metric_for_early_stop = val_metrics["landform_macro_f1"]
        if metric_for_early_stop > best_metric:
            best_metric = metric_for_early_stop
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "mil_config": asdict(cfg),
                    "class_names": CLASS_ORDER,
                    "best_epoch": best_epoch,
                    "best_landform_macro_f1": best_metric,
                },
                best_model_path,
            )
        else:
            patience_counter += 1

        if patience_counter >= cfg.patience:
            print(f"Early stopping at epoch {epoch} (best_epoch={best_epoch})")
            break

    curves_path = output_dir / "training_curves.json"
    curves_path.write_text(json.dumps(history, indent=2))

    split_path = output_dir / "data_split.json"
    split_path.write_text(json.dumps({"train_ids": train_ids, "val_ids": val_ids, "test_ids": test_ids}, indent=2))

    return {
        "best_model_path": best_model_path,
        "history_path": curves_path,
        "split_path": split_path,
        "test_loader": test_loader,
        "test_ids": test_ids,
    }


@torch.no_grad()
def evaluate_best_model(
    model_path: Path,
    test_loader: DataLoader[Any],
    device: torch.device,
    cfg: MILConfig,
    output_dir: Path,
    mixed_precision: bool,
) -> List[Dict[str, Any]]:
    checkpoint = torch.load(model_path, map_location=device)
    model = AttentionMILClassifier(cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    y_true: List[int] = []
    y_pred: List[int] = []
    predictions: List[Dict[str, Any]] = []

    for batch in test_loader:
        tiles = batch["tile_embeddings"].to(device)
        mask = batch["tile_mask"].to(device)
        mola = batch["mola_features"].to(device)
        labels = batch["labels"].to(device)
        image_ids = batch["image_ids"]

        with torch.cuda.amp.autocast(enabled=mixed_precision):
            logits, att_weights = model(tiles, mask, mola)

        probs = torch.softmax(logits, dim=1)
        conf, preds = torch.max(probs, dim=1)

        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

        att_cpu = att_weights.detach().cpu()
        mask_cpu = mask.detach().cpu()
        probs_cpu = probs.detach().cpu()
        conf_cpu = conf.detach().cpu()

        for i, image_id in enumerate(image_ids):
            valid_len = int(mask_cpu[i].sum().item())
            predictions.append(
                {
                    "image_id": image_id,
                    "true_label": int(labels[i].item()),
                    "pred_label": int(preds[i].item()),
                    "pred_label_name": CLASS_ORDER[int(preds[i].item())],
                    "confidence": float(conf_cpu[i].item()),
                    "probabilities": probs_cpu[i].tolist(),
                    "attention_weights": att_cpu[i, :valid_len].tolist(),
                }
            )

    metrics = compute_metrics(y_true, y_pred, num_classes=cfg.num_classes)
    print("[Test] Per-class metrics")
    print("Class     Precision  Recall  F1   Support")
    for i, cls_name in enumerate(CLASS_ORDER):
        print(
            f"{cls_name:<10}{metrics['precision'][i]:>8.3f}"
            f"{metrics['recall'][i]:>8.3f}{metrics['f1'][i]:>7.3f}{int(metrics['support'][i]):>9d}"
        )
    print(f"[Test] Macro-F1 (all classes): {metrics['macro_f1_all']:.4f}")
    print(f"[Test] Landform Macro-F1 (exclude BACKGROUND): {metrics['landform_macro_f1']:.4f}")

    cm_png = output_dir / "test_confusion_matrix.png"
    save_confusion_matrix_png(metrics["confusion_matrix"], CLASS_ORDER, cm_png)

    report_path = output_dir / "test_metrics.json"
    report_path.write_text(json.dumps(metrics, indent=2))
    preds_path = output_dir / "test_predictions_with_attention.json"
    preds_path.write_text(json.dumps(predictions, indent=2))

    return predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate Attention-based MIL classifier")
    parser.add_argument("--embeddings_dir", type=Path, required=True)
    parser.add_argument("--mola_path", type=Path, required=True)
    parser.add_argument("--labels_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipe_cfg = get_config()
    mil_cfg = pipe_cfg.mil

    if args.epochs is not None:
        mil_cfg.epochs = args.epochs
    if args.patience is not None:
        mil_cfg.patience = args.patience
    if args.batch_size is not None:
        mil_cfg.batch_size = args.batch_size
    if args.lr is not None:
        mil_cfg.lr = args.lr
    if args.weight_decay is not None:
        mil_cfg.weight_decay = args.weight_decay

    seed = args.seed if args.seed is not None else pipe_cfg.seed
    set_seed(seed)

    embeddings_dict = load_embeddings(args.embeddings_dir)
    mola_dict = load_mola_features(args.mola_path)
    labels_dict = load_labels(args.labels_path)

    device = torch.device(pipe_cfg.device)
    train_artifacts = train_mil(
        embeddings_dict=embeddings_dict,
        mola_dict=mola_dict,
        labels_dict=labels_dict,
        output_dir=args.output_dir,
        cfg=mil_cfg,
        device=device,
        mixed_precision=pipe_cfg.mixed_precision,
        num_workers=pipe_cfg.num_workers,
        seed=seed,
    )

    evaluate_best_model(
        model_path=train_artifacts["best_model_path"],
        test_loader=train_artifacts["test_loader"],
        device=device,
        cfg=mil_cfg,
        output_dir=args.output_dir,
        mixed_precision=pipe_cfg.mixed_precision,
    )


if __name__ == "__main__":
    main()
