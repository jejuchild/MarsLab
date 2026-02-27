#!/usr/bin/env python3

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


CLASS_ORDER = ["LDA", "CCF", "LVF", "GLF", "BACKGROUND"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_ORDER)}


def parse_args():
    parser = argparse.ArgumentParser(description="Train pseudo-label classifier for Martian landforms")
    _ = parser.add_argument("--cluster-dir", default="Data/HiRISE/pipeline_output/clusters")
    _ = parser.add_argument("--embeddings", default="Data/HiRISE/pipeline_output/embeddings.npy")
    _ = parser.add_argument("--tile-metadata", default="Data/HiRISE/pipeline_output/tile_metadata.csv")
    _ = parser.add_argument("--mola-features", default="Data/HiRISE/pipeline_output/mola_features.npy")
    _ = parser.add_argument("--use-mola", action="store_true", help="Use MOLA features as additional inputs")
    _ = parser.add_argument("--output-dir", default="Data/HiRISE/pipeline_output/classifier")
    _ = parser.add_argument("--enrichment-threshold", type=float, default=2.0)
    _ = parser.add_argument("--epochs", type=int, default=30)
    _ = parser.add_argument("--lr", type=float, default=1e-3)
    _ = parser.add_argument("--batch-size", type=int, default=256)
    _ = parser.add_argument("--val-split", type=float, default=0.15)
    _ = parser.add_argument("--finetune-backbone", action="store_true", default=False)
    return parser.parse_args()


def count_csv_rows(csv_path):
    count = 0
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for _ in reader:
            count += 1
    return count


def load_cluster_label_map(cluster_summary_path, threshold):
    with open(cluster_summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    clusters = summary.get("clusters", [])
    cluster_to_label = {}
    for cluster in clusters:
        cluster_id = int(cluster["id"])
        dominant_class = str(cluster.get("dominant_class", "BACKGROUND"))
        enrichment_score = float(cluster.get("enrichment_score", 0.0))

        if enrichment_score >= threshold and dominant_class in CLASS_TO_IDX and dominant_class != "BACKGROUND":
            cluster_to_label[cluster_id] = CLASS_TO_IDX[dominant_class]
        else:
            cluster_to_label[cluster_id] = CLASS_TO_IDX["BACKGROUND"]

    return cluster_to_label


def load_assignments_and_labels(assignments_path, cluster_to_label):
    labels = []
    source_paths = []
    with open(assignments_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cluster_id = int(row["cluster_id"])
            label = cluster_to_label.get(cluster_id, CLASS_TO_IDX["BACKGROUND"])
            labels.append(label)
            source_paths.append(row.get("source_path", ""))
    return np.array(labels, dtype=np.int64), source_paths


def load_feature_matrix(embeddings_path, mola_path=None, use_mola=False):
    embeddings = np.load(embeddings_path).astype(np.float32)
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings array, got shape={embeddings.shape}")
    if embeddings.shape[1] != 384:
        raise ValueError(f"Expected embeddings dim=384, got dim={embeddings.shape[1]}")

    if use_mola:
        if mola_path is None:
            raise ValueError("--use-mola enabled but no --mola-features path provided")
        mola = np.load(mola_path).astype(np.float32)
        if mola.ndim != 2:
            raise ValueError(f"Expected 2D MOLA array, got shape={mola.shape}")
        if mola.shape[0] != embeddings.shape[0]:
            raise ValueError("MOLA rows do not match embedding rows")
        return np.concatenate([embeddings, mola], axis=1), mola

    return embeddings, None


def compute_class_weights(y_train):
    counts = np.bincount(y_train, minlength=len(CLASS_ORDER)).astype(np.float32)
    total = float(np.sum(counts))
    weights = np.zeros(len(CLASS_ORDER), dtype=np.float32)
    for i in range(len(CLASS_ORDER)):
        if counts[i] > 0:
            weights[i] = total / (len(CLASS_ORDER) * counts[i])
        else:
            weights[i] = 0.0
    return weights


class FeatureDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, features, labels):
        self.features = torch.from_numpy(features.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.int64))

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class ImageFeatureDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, source_paths, labels, mola_features=None, image_size=224):
        self.source_paths = source_paths
        self.labels = labels.astype(np.int64)
        self.mola_features = mola_features
        self.image_size = image_size
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.source_paths)

    def __getitem__(self, idx):
        img_path = self.source_paths[idx]
        if not img_path:
            raise ValueError(f"Missing source_path for sample index {idx}")

        with Image.open(img_path) as img:
            img = img.convert("RGB")
            img = img.resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
            arr = np.asarray(img, dtype=np.float32) / 255.0

        arr = (arr - self.mean) / self.std
        arr = np.transpose(arr, (2, 0, 1))

        if self.mola_features is not None:
            extra = self.mola_features[idx].astype(np.float32)
        else:
            extra = np.zeros((0,), dtype=np.float32)

        return (
            torch.from_numpy(arr),
            torch.from_numpy(extra),
            torch.tensor(self.labels[idx], dtype=torch.int64),
        )


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class DinoClassifier(nn.Module):
    def __init__(self, backbone, extra_dim=0, num_classes=5):
        super().__init__()
        self.backbone = backbone
        self.extra_dim = extra_dim
        self.head = MLPClassifier(384 + extra_dim, num_classes=num_classes)

    def forward(self, images, extra_features=None):
        feats = self.backbone(images)
        if isinstance(feats, (tuple, list)):
            feats = feats[0]
        if extra_features is not None and extra_features.numel() > 0:
            feats = torch.cat([feats, extra_features], dim=1)
        return self.head(feats)


def run_epoch_feature(model, loader, criterion, optimizer, device, train_mode=True):
    if train_mode:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    all_preds = []
    all_targets = []

    pbar = tqdm(loader, desc="train" if train_mode else "val", leave=False)
    for features, targets in pbar:
        features = features.to(device)
        targets = targets.to(device)

        if train_mode:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train_mode):
            logits = model(features)
            loss = criterion(logits, targets)
            if train_mode:
                loss.backward()
                optimizer.step()

        preds = torch.argmax(logits, dim=1)
        running_loss += loss.item() * targets.size(0)
        all_preds.append(preds.detach().cpu().numpy())
        all_targets.append(targets.detach().cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    avg_loss = running_loss / len(loader.dataset)
    acc = float((y_pred == y_true).mean())
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_ORDER))),
        zero_division="warn",
    )
    precision = np.asarray(precision, dtype=np.float32)
    recall = np.asarray(recall, dtype=np.float32)
    f1 = np.asarray(f1, dtype=np.float32)

    return {
        "loss": float(avg_loss),
        "acc": acc,
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def run_epoch_dino(model, loader, criterion, optimizer, device, train_mode=True):
    if train_mode:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    all_preds = []
    all_targets = []

    pbar = tqdm(loader, desc="train" if train_mode else "val", leave=False)
    for images, extra, targets in pbar:
        images = images.to(device)
        extra = extra.to(device)
        targets = targets.to(device)

        if train_mode:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train_mode):
            logits = model(images, extra)
            loss = criterion(logits, targets)
            if train_mode:
                loss.backward()
                optimizer.step()

        preds = torch.argmax(logits, dim=1)
        running_loss += loss.item() * targets.size(0)
        all_preds.append(preds.detach().cpu().numpy())
        all_targets.append(targets.detach().cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    avg_loss = running_loss / len(loader.dataset)
    acc = float((y_pred == y_true).mean())
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_ORDER))),
        zero_division="warn",
    )
    precision = np.asarray(precision, dtype=np.float32)
    recall = np.asarray(recall, dtype=np.float32)
    f1 = np.asarray(f1, dtype=np.float32)

    return {
        "loss": float(avg_loss),
        "acc": acc,
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def save_confusion_matrix(y_true, y_pred, output_path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_ORDER))))
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASS_ORDER)))
    ax.set_yticks(range(len(CLASS_ORDER)))
    ax.set_xticklabels(CLASS_ORDER, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_ORDER)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Validation Confusion Matrix")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    start_time = time.time()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cluster_dir = Path(args.cluster_dir)
    cluster_summary_path = cluster_dir / "cluster_summary.json"
    cluster_assignments_path = cluster_dir / "cluster_assignments.csv"

    print("Loading cluster enrichment and assignments...")
    cluster_to_label = load_cluster_label_map(cluster_summary_path, args.enrichment_threshold)
    pseudo_labels, source_paths = load_assignments_and_labels(cluster_assignments_path, cluster_to_label)

    print("Loading embeddings...")
    features_or_embeddings, mola = load_feature_matrix(
        args.embeddings,
        mola_path=args.mola_features,
        use_mola=args.use_mola,
    )

    n_metadata = count_csv_rows(args.tile_metadata)
    n_labels = pseudo_labels.shape[0]
    n_emb = features_or_embeddings.shape[0]
    if not (n_metadata == n_labels == n_emb):
        raise ValueError(
            f"Row count mismatch: tile_metadata={n_metadata}, cluster_assignments={n_labels}, embeddings={n_emb}"
        )

    indices = np.arange(n_labels)
    print("Creating train/val split...")
    try:
        train_idx, val_idx = train_test_split(
            indices,
            test_size=args.val_split,
            random_state=42,
            stratify=pseudo_labels,
        )
    except ValueError as exc:
        print(f"Warning: stratified split failed ({exc}); falling back to random split")
        train_idx, val_idx = train_test_split(
            indices,
            test_size=args.val_split,
            random_state=42,
            stratify=None,
        )

    y_train = pseudo_labels[train_idx]
    y_val = pseudo_labels[val_idx]

    weights = compute_class_weights(y_train)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Training class distribution:")
    for i, class_name in enumerate(CLASS_ORDER):
        print(f"  {class_name}: {int(np.sum(y_train == i))}")

    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))

    if args.finetune_backbone:
        print("Loading DINOv2 ViT-S/14 backbone...")
        backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        if not isinstance(backbone, nn.Module):
            raise TypeError("Loaded DINOv2 backbone is not a torch.nn.Module")
        extra_dim = 23 if args.use_mola else 0
        model = DinoClassifier(backbone, extra_dim=extra_dim, num_classes=len(CLASS_ORDER)).to(device)

        train_source_paths = [source_paths[i] for i in train_idx]
        val_source_paths = [source_paths[i] for i in val_idx]
        if args.use_mola:
            if mola is None:
                raise ValueError("--use-mola enabled but MOLA features were not loaded")
            train_mola = mola[train_idx]
            val_mola = mola[val_idx]
        else:
            train_mola = None
            val_mola = None

        train_ds = ImageFeatureDataset(train_source_paths, y_train, mola_features=train_mola)
        val_ds = ImageFeatureDataset(val_source_paths, y_val, mola_features=val_mola)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

        optimizer = torch.optim.Adam(
            [
                {"params": model.backbone.parameters(), "lr": 1e-5},
                {"params": model.head.parameters(), "lr": args.lr},
            ]
        )
        epoch_runner = run_epoch_dino
    else:
        x_train = features_or_embeddings[train_idx]
        x_val = features_or_embeddings[val_idx]

        train_ds = FeatureDataset(x_train, y_train)
        val_ds = FeatureDataset(x_val, y_val)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

        input_dim = x_train.shape[1]
        model = MLPClassifier(input_dim=input_dim, num_classes=len(CLASS_ORDER)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        epoch_runner = run_epoch_feature

    best_val_loss = float("inf")
    best_epoch = -1
    patience = 5
    wait = 0
    history = []

    checkpoint_path = output_dir / "best_model.pt"

    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_stats = epoch_runner(model, train_loader, criterion, optimizer, device, train_mode=True)
        val_stats = epoch_runner(model, val_loader, criterion, optimizer, device, train_mode=False)
        val_f1 = np.asarray(val_stats["f1"], dtype=np.float32)

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "val_loss": val_stats["loss"],
            "val_acc": val_stats["acc"],
            "val_f1_per_class": {
                class_name: float(val_f1[i]) for i, class_name in enumerate(CLASS_ORDER)
            },
        }
        history.append(epoch_record)

        print(
            f"  train_loss={train_stats['loss']:.4f} val_loss={val_stats['loss']:.4f} val_acc={val_stats['acc']:.4f}"
        )

        if val_stats["loss"] < best_val_loss:
            best_val_loss = val_stats["loss"]
            best_epoch = epoch
            wait = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_val_loss": best_val_loss,
                    "class_names": CLASS_ORDER,
                    "class_order": CLASS_ORDER,
                    "input_dim": 384,
                    "use_mola": args.use_mola,
                    "mola_dim": 23 if args.use_mola else 0,
                    "finetune_backbone": args.finetune_backbone,
                    "args": vars(args),
                },
                checkpoint_path,
            )
            print(f"  Saved new best model to {checkpoint_path}")
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping triggered at epoch {epoch} (patience={patience})")
                break

    print("Loading best model for final evaluation...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    final_stats = epoch_runner(model, val_loader, criterion, optimizer, device, train_mode=False)
    y_true = final_stats["y_true"]
    y_pred = final_stats["y_pred"]

    cm_path = output_dir / "confusion_matrix.png"
    save_confusion_matrix(y_true, y_pred, cm_path)
    print(f"Saved confusion matrix: {cm_path}")

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_ORDER))),
        target_names=CLASS_ORDER,
        zero_division="warn",
        output_dict=False,
    )
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_ORDER))),
        target_names=CLASS_ORDER,
        zero_division="warn",
        output_dict=True,
    )

    print("Per-class precision/recall/F1 on validation set:")
    print(report)
    final_precision = np.asarray(final_stats["precision"], dtype=np.float32)
    final_recall = np.asarray(final_stats["recall"], dtype=np.float32)
    final_f1 = np.asarray(final_stats["f1"], dtype=np.float32)

    metrics = {
        "class_order": CLASS_ORDER,
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss),
        "history": history,
        "final_val": {
            "loss": float(final_stats["loss"]),
            "acc": float(final_stats["acc"]),
            "precision_per_class": {
                CLASS_ORDER[i]: float(final_precision[i]) for i in range(len(CLASS_ORDER))
            },
            "recall_per_class": {
                CLASS_ORDER[i]: float(final_recall[i]) for i in range(len(CLASS_ORDER))
            },
            "f1_per_class": {
                CLASS_ORDER[i]: float(final_f1[i]) for i in range(len(CLASS_ORDER))
            },
            "classification_report": report_dict,
        },
    }

    metrics_path = output_dir / "train_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved training metrics: {metrics_path}")

    elapsed = time.time() - start_time
    print(f"Done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
