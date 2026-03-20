#!/usr/bin/env python3
"""
Train V8.1 Segmentation Head on pre-extracted DINOv2 ViT-L patch tokens.

V8.1 improvements over V8.0:
  - Spatial conv layers (3×3 + dilated 3×3 with residual) for local context
  - MOLA elevation feature fusion (25-dim per tile, broadcast to 16×16 grid)
  - Focal Loss (gamma=2.0) for class imbalance
  - Lower LR (1e-3) with warmup + cosine annealing
  - AdamW with weight decay + gradient clipping

Usage:
  nohup python train_v8_segmentation.py > /tmp/train_v8.log 2>&1 &
"""
from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, "/disk1/cspark/hirise-api")
from scripts.marslandform_v2.models.seg_head import PatchSegmentationHead

ROOT = Path("/disk1/cspark/MarsLab")
DATA_DIR = ROOT / "Data/HiRISE/v8_segmentation"
PATCH_TOKENS_DIR = DATA_DIR / "patch_tokens_v8"
PATCH_LABELS_PATH = DATA_DIR / "patch_labels_v8.npy"
MOLA_PATH = ROOT / "Data/HiRISE/v5_retrain/mola_features_v5.npy"
OUT_DIR = DATA_DIR

CLASS_NAMES = ["LDA", "LVF", "CCF", "OTHER"]
NUM_CLASSES = 4
UNLABELED = 255
EMBED_DIM = 1024
MOLA_DIM = 25
PATCHES_PER_SIDE = 16


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None, ignore_index: int = -100):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.weight, ignore_index=self.ignore_index, reduction="none")
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma * ce)
        return focal.mean()


def split_by_product(all_pids: list[str], val_ratio: float = 0.15, seed: int = 42):
    rng = np.random.RandomState(seed)
    unique_pids = sorted(set(all_pids))
    rng.shuffle(unique_pids)
    n_val = max(1, int(len(unique_pids) * val_ratio))
    val_pids = set(unique_pids[:n_val])
    train_pids = [p for p in unique_pids if p not in val_pids]
    return train_pids, list(val_pids)


def compute_metrics(all_preds: np.ndarray, all_labels: np.ndarray):
    mask = all_labels != UNLABELED
    preds = all_preds[mask]
    labels = all_labels[mask]

    if len(labels) == 0:
        return 0.0, 0.0, [0.0] * NUM_CLASSES

    acc = float((preds == labels).mean())
    f1s = []
    for c in range(NUM_CLASSES):
        tp = ((preds == c) & (labels == c)).sum()
        fp = ((preds == c) & (labels != c)).sum()
        fn = ((preds != c) & (labels == c)).sum()
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        f1s.append(f1)

    return acc, float(np.mean(f1s)), f1s


def preload_split(
    pids: list[str],
    patch_labels: dict[str, dict[str, np.ndarray]],
    tokens_dir: Path,
    mola_dict: dict[str, dict[str, np.ndarray]],
    split_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_tokens = []
    all_labels = []
    all_mola = []
    skipped = 0

    for i, pid in enumerate(pids):
        token_file = tokens_dir / f"{pid}.npy"
        if not token_file.exists():
            skipped += 1
            continue
        if pid not in patch_labels:
            skipped += 1
            continue

        token_dict = np.load(token_file, allow_pickle=True).item()
        label_dict = patch_labels[pid]
        mola_features_for_pid = mola_dict.get(pid, {})

        for tile_key in label_dict:
            if tile_key not in token_dict:
                continue
            mola_vec = mola_features_for_pid.get(tile_key)
            if mola_vec is None:
                skipped += 1
                continue
            all_tokens.append(token_dict[tile_key])
            all_labels.append(label_dict[tile_key])
            all_mola.append(mola_vec)

        if (i + 1) % 1000 == 0:
            print(f"    {split_name}: loaded {i + 1}/{len(pids)} products, {len(all_tokens)} tiles")

    tokens = np.stack(all_tokens, axis=0)
    labels = np.stack(all_labels, axis=0)
    mola = np.stack(all_mola, axis=0)
    print(f"    {split_name}: {len(tokens)} tiles from {len(pids)} products (skipped {skipped})")
    return tokens, labels, mola


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading patch labels...")
    patch_labels = np.load(PATCH_LABELS_PATH, allow_pickle=True).item()
    all_pids = sorted(patch_labels.keys())
    print(f"  Products with labels: {len(all_pids)}")

    print("Loading MOLA features...")
    mola_dict = np.load(MOLA_PATH, allow_pickle=True).item()
    print(f"  Products with MOLA: {len(mola_dict)}")

    class_counts = Counter()
    total_patches = 0
    for pid in all_pids:
        for tile_key, lbl in patch_labels[pid].items():
            unique, counts = np.unique(lbl, return_counts=True)
            for u, c in zip(unique, counts):
                if u != UNLABELED:
                    class_counts[int(u)] += int(c)
                total_patches += int(c)
    print(f"  Total patches: {total_patches:,}")
    print(f"  Trainable patches: {sum(class_counts.values()):,}")
    for c in range(NUM_CLASSES):
        print(f"    {CLASS_NAMES[c]}: {class_counts.get(c, 0):,}")

    train_pids, val_pids = split_by_product(all_pids)
    print(f"  Train products: {len(train_pids)}, Val products: {len(val_pids)}")

    print("\nPreloading tokens + labels + MOLA into RAM...")
    t_load = time.time()
    train_tokens, train_labels, train_mola = preload_split(
        train_pids, patch_labels, PATCH_TOKENS_DIR, mola_dict, "train",
    )
    val_tokens, val_labels, val_mola = preload_split(
        val_pids, patch_labels, PATCH_TOKENS_DIR, mola_dict, "val",
    )
    load_time = time.time() - t_load
    print(f"  Loaded in {load_time:.1f}s")
    print(f"  Train: {train_tokens.shape} tokens ({train_tokens.nbytes / 1e9:.1f}GB), "
          f"{train_labels.shape} labels, {train_mola.shape} MOLA")
    print(f"  Val: {val_tokens.shape} tokens ({val_tokens.nbytes / 1e9:.1f}GB), "
          f"{val_labels.shape} labels, {val_mola.shape} MOLA")

    # Z-score normalize MOLA using training set stats
    mola_mean = train_mola.mean(axis=0).astype(np.float32)
    mola_std = train_mola.std(axis=0).astype(np.float32)
    mola_std = np.maximum(mola_std, 1e-6)
    train_mola_normed = ((train_mola - mola_mean) / mola_std).astype(np.float32)
    val_mola_normed = ((val_mola - mola_mean) / mola_std).astype(np.float32)
    print(f"  MOLA z-score normalized (mean range: [{mola_mean.min():.1f}, {mola_mean.max():.1f}], "
          f"std range: [{mola_std.min():.4f}, {mola_std.max():.1f}])")

    train_ds = TensorDataset(
        torch.from_numpy(train_tokens),
        torch.from_numpy(train_labels.astype(np.int64)),
        torch.from_numpy(train_mola_normed),
    )
    del train_tokens, train_labels, train_mola, train_mola_normed
    val_ds = TensorDataset(
        torch.from_numpy(val_tokens),
        torch.from_numpy(val_labels.astype(np.int64)),
        torch.from_numpy(val_mola_normed),
    )
    del val_tokens, val_labels, val_mola, val_mola_normed, patch_labels, mola_dict

    print(f"  Train tiles: {len(train_ds)}, Val tiles: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=0)

    model = PatchSegmentationHead(
        embed_dim=EMBED_DIM,
        num_classes=NUM_CLASSES,
        patches_per_side=PATCHES_PER_SIDE,
        hidden_dim=64,
        mola_dim=MOLA_DIM,
        mola_hidden=16,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model params: {trainable:,} trainable / {total_params:,} total")

    class_weight_vals = torch.zeros(NUM_CLASSES)
    for c in range(NUM_CLASSES):
        cnt = class_counts.get(c, 1)
        class_weight_vals[c] = 1.0 / (cnt + 1e-6)
    class_weight_vals = class_weight_vals / class_weight_vals.sum() * NUM_CLASSES
    print(f"  Class weights: {class_weight_vals.tolist()}")

    criterion = FocalLoss(
        gamma=2.0,
        weight=class_weight_vals.to(device),
        ignore_index=UNLABELED,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    warmup_epochs = 5
    n_epochs = 50
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs,
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs - warmup_epochs, eta_min=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )

    best_val_f1 = 0
    patience = 15
    no_improve = 0

    print(f"\n{'Ep':>3} | {'TrLoss':>7} | {'TrAcc':>6} | {'TrF1':>6} | "
          f"{'VlAcc':>6} | {'VlF1':>6} | {'LR':>8} | Per-class Val F1")
    print("-" * 110)

    for epoch in range(1, n_epochs + 1):
        t_ep = time.time()

        model.train()
        total_loss = 0
        n_samples = 0
        all_preds_tr, all_labels_tr = [], []

        for tokens, labels, mola in train_loader:
            tokens = tokens.float().to(device)
            labels = labels.to(device)
            mola = mola.to(device)

            logits = model(tokens, mola)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * tokens.shape[0]
            n_samples += tokens.shape[0]
            all_preds_tr.append(logits.argmax(1).cpu().numpy())
            all_labels_tr.append(labels.cpu().numpy())

        scheduler.step()
        avg_loss = total_loss / max(n_samples, 1)
        preds_tr = np.concatenate(all_preds_tr)
        labels_tr = np.concatenate(all_labels_tr)
        tr_acc, tr_f1, _ = compute_metrics(preds_tr, labels_tr)

        model.eval()
        all_preds_val, all_labels_val = [], []
        with torch.no_grad():
            for tokens, labels, mola in val_loader:
                tokens = tokens.float().to(device)
                mola = mola.to(device)
                logits = model(tokens, mola)
                all_preds_val.append(logits.argmax(1).cpu().numpy())
                all_labels_val.append(labels.numpy())

        preds_val = np.concatenate(all_preds_val)
        labels_val = np.concatenate(all_labels_val)
        val_acc, val_f1, val_f1s = compute_metrics(preds_val, labels_val)

        ep_time = time.time() - t_ep
        lr = scheduler.get_last_lr()[0]
        f1_str = " ".join(f"{CLASS_NAMES[c]}={val_f1s[c]:.3f}" for c in range(NUM_CLASSES))
        print(f"{epoch:>3} | {avg_loss:>7.4f} | {tr_acc:>6.3f} | {tr_f1:>6.3f} | "
              f"{val_acc:>6.3f} | {val_f1:>6.3f} | {lr:>8.6f} | {f1_str}  ({ep_time:.0f}s)")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            no_improve = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_f1": val_f1,
                "val_acc": val_acc,
                "train_f1": tr_f1,
                "class_weights": class_weight_vals.tolist(),
                "mola_mean": mola_mean.tolist(),
                "mola_std": mola_std.tolist(),
                "cfg": {
                    "embed_dim": EMBED_DIM,
                    "num_classes": NUM_CLASSES,
                    "patches_per_side": PATCHES_PER_SIDE,
                    "hidden_dim": 64,
                    "mola_dim": MOLA_DIM,
                    "mola_hidden": 16,
                    "class_names": CLASS_NAMES,
                    "architecture": "v8.1-patch-segmentation",
                    "backbone": "facebook/dinov2-large",
                },
                "version": "v8.1-segmentation",
            }, OUT_DIR / "seg_head_v8.pt")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    print(f"\nBest val F1: {best_val_f1:.4f}")
    print(f"Saved to: {OUT_DIR / 'seg_head_v8.pt'}")

    best_ckpt = torch.load(OUT_DIR / "seg_head_v8.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    print(f"\n=== Final Evaluation (best epoch={best_ckpt['epoch']}) ===")

    all_preds, all_labels = [], []
    with torch.no_grad():
        for tokens, labels, mola in val_loader:
            tokens = tokens.float().to(device)
            mola = mola.to(device)
            logits = model(tokens, mola)
            all_preds.append(logits.argmax(1).cpu().numpy())
            all_labels.append(labels.numpy())

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    mask = labels != UNLABELED

    print(f"\nVal set confusion (patches):")
    print(f"{'':>8}", end="")
    for c in range(NUM_CLASSES):
        print(f"  pred_{CLASS_NAMES[c]:>5}", end="")
    print()
    for true_c in range(NUM_CLASSES):
        true_mask = (labels == true_c) & mask
        print(f"  {CLASS_NAMES[true_c]:>5}", end="")
        for pred_c in range(NUM_CLASSES):
            cnt = ((preds == pred_c) & true_mask).sum()
            print(f"  {cnt:>10,}", end="")
        print()

    other_mask = (labels == 3) & mask
    other_as_glacial = ((preds != 3) & other_mask).sum()
    other_total = other_mask.sum()
    print(f"\n  FP rate (OTHER→glacial): {other_as_glacial}/{other_total} = "
          f"{other_as_glacial / max(other_total, 1) * 100:.2f}%")

    print(f"\n  {'Class':>8} | {'Precision':>9} | {'Recall':>9} | {'F1':>9} | {'Support':>9}")
    print(f"  {'-' * 55}")
    for c in range(NUM_CLASSES):
        tp = ((preds == c) & (labels == c) & mask).sum()
        fp = ((preds == c) & (labels != c) & mask).sum()
        fn = ((preds != c) & (labels == c) & mask).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        support = ((labels == c) & mask).sum()
        print(f"  {CLASS_NAMES[c]:>8} | {prec:>9.4f} | {rec:>9.4f} | {f1:>9.4f} | {support:>9,}")


if __name__ == "__main__":
    t0 = time.time()
    train()
    print(f"\nTotal time: {time.time() - t0:.1f}s")
