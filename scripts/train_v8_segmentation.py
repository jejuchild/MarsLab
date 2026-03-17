#!/usr/bin/env python3
"""
Train V8 Segmentation Head on pre-extracted DINOv2 ViT-L patch tokens.

Architecture: frozen ViT-L backbone → BN + Conv1×1 → per-patch class prediction.
Data: pre-extracted patch tokens (256×1024 per tile) + patch labels (16×16 per tile).
Split: product-level, seed=42, 15% val (same as V6b).

Trainable params: ~6K. Training is fast even on CPU.

Strategy: Preload ALL tokens+labels into RAM as flat arrays, then train with
TensorDataset. Eliminates I/O bottleneck entirely.

Usage:
  python train_v8_segmentation.py
"""
from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, "/disk1/cspark/hirise-api")
from scripts.marslandform_v2.models.seg_head import PatchSegmentationHead

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path("/disk1/cspark/MarsLab")
DATA_DIR = ROOT / "Data/HiRISE/v8_segmentation"
PATCH_TOKENS_DIR = DATA_DIR / "patch_tokens_v8"
PATCH_LABELS_PATH = DATA_DIR / "patch_labels_v8.npy"
OUT_DIR = DATA_DIR

CLASS_NAMES = ["LDA", "LVF", "CCF", "OTHER"]
NUM_CLASSES = 4
UNLABELED = 255
EMBED_DIM = 1024
PATCHES_PER_SIDE = 16


def split_by_product(all_pids: list[str], val_ratio: float = 0.15, seed: int = 42):
    """Split products into train/val (same strategy as V6b)."""
    rng = np.random.RandomState(seed)
    unique_pids = sorted(set(all_pids))
    rng.shuffle(unique_pids)
    n_val = max(1, int(len(unique_pids) * val_ratio))
    val_pids = set(unique_pids[:n_val])
    train_pids = [p for p in unique_pids if p not in val_pids]
    return train_pids, list(val_pids)


def compute_metrics(all_preds: np.ndarray, all_labels: np.ndarray):
    """Compute accuracy and per-class F1 (ignoring UNLABELED)."""
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
    patch_labels: dict,
    tokens_dir: Path,
    split_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Preload all tokens + labels for a split into flat arrays.

    Returns:
        tokens: (N, 256, 1024) float16
        labels: (N, 16, 16) uint8
    """
    all_tokens = []
    all_labels = []
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

        for tile_key in label_dict:
            if tile_key not in token_dict:
                continue
            all_tokens.append(token_dict[tile_key])   # (256, 1024) float16
            all_labels.append(label_dict[tile_key])    # (16, 16) uint8

        if (i + 1) % 1000 == 0:
            print(f"    {split_name}: loaded {i + 1}/{len(pids)} products, {len(all_tokens)} tiles")

    tokens = np.stack(all_tokens, axis=0)  # (N, 256, 1024)
    labels = np.stack(all_labels, axis=0)  # (N, 16, 16)
    print(f"    {split_name}: {len(tokens)} tiles from {len(pids) - skipped} products (skipped {skipped})")
    return tokens, labels


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load patch labels
    print("Loading patch labels...")
    patch_labels = np.load(PATCH_LABELS_PATH, allow_pickle=True).item()
    all_pids = sorted(patch_labels.keys())
    print(f"  Products with labels: {len(all_pids)}")

    # Count class distribution
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

    # Split
    train_pids, val_pids = split_by_product(all_pids)
    print(f"  Train products: {len(train_pids)}, Val products: {len(val_pids)}")

    # Preload all data into RAM
    print("\nPreloading tokens + labels into RAM...")
    t_load = time.time()
    train_tokens, train_labels = preload_split(train_pids, patch_labels, PATCH_TOKENS_DIR, "train")
    val_tokens, val_labels = preload_split(val_pids, patch_labels, PATCH_TOKENS_DIR, "val")
    load_time = time.time() - t_load
    print(f"  Loaded in {load_time:.1f}s")
    print(f"  Train: {train_tokens.shape} tokens ({train_tokens.nbytes / 1e9:.1f}GB), {train_labels.shape} labels")
    print(f"  Val: {val_tokens.shape} tokens ({val_tokens.nbytes / 1e9:.1f}GB), {val_labels.shape} labels")

    # Keep as float16 torch tensors to save RAM (convert to float32 per-batch)
    train_ds = TensorDataset(
        torch.from_numpy(train_tokens),  # (N, 256, 1024) float16
        torch.from_numpy(train_labels.astype(np.int64)),
    )
    del train_tokens, train_labels
    val_ds = TensorDataset(
        torch.from_numpy(val_tokens),  # (N, 256, 1024) float16
        torch.from_numpy(val_labels.astype(np.int64)),
    )
    del val_tokens, val_labels, patch_labels

    print(f"  Train tiles: {len(train_ds)}, Val tiles: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=0)

    # Model
    model = PatchSegmentationHead(
        embed_dim=EMBED_DIM,
        num_classes=NUM_CLASSES,
        patches_per_side=PATCHES_PER_SIDE,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model params: {trainable:,} trainable / {total_params:,} total")

    # Loss: CE with class weights + ignore unlabeled
    class_weight_vals = torch.zeros(NUM_CLASSES)
    for c in range(NUM_CLASSES):
        cnt = class_counts.get(c, 1)
        class_weight_vals[c] = 1.0 / (cnt + 1e-6)
    class_weight_vals = class_weight_vals / class_weight_vals.sum() * NUM_CLASSES
    print(f"  Class weights: {class_weight_vals.tolist()}")

    criterion = nn.CrossEntropyLoss(
        weight=class_weight_vals.to(device),
        ignore_index=UNLABELED,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-5)

    # Training loop
    best_val_f1 = 0
    patience = 10
    no_improve = 0
    n_epochs = 30

    print(f"\n{'Ep':>3} | {'TrLoss':>7} | {'TrAcc':>6} | {'TrF1':>6} | {'VlAcc':>6} | {'VlF1':>6} | {'LR':>8} | Per-class Val F1")
    print("-" * 110)

    for epoch in range(1, n_epochs + 1):
        t_ep = time.time()

        # Train
        model.train()
        total_loss = 0
        n_samples = 0
        all_preds_tr, all_labels_tr = [], []

        for tokens, labels in train_loader:
            tokens, labels = tokens.float().to(device), labels.to(device)
            logits = model(tokens)  # (B, 4, 16, 16)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
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

        # Validate
        model.eval()
        all_preds_val, all_labels_val = [], []
        with torch.no_grad():
            for tokens, labels in val_loader:
                tokens = tokens.float().to(device)
                logits = model(tokens)
                all_preds_val.append(logits.argmax(1).cpu().numpy())
                all_labels_val.append(labels.numpy())

        preds_val = np.concatenate(all_preds_val)
        labels_val = np.concatenate(all_labels_val)
        val_acc, val_f1, val_f1s = compute_metrics(preds_val, labels_val)

        ep_time = time.time() - t_ep
        lr = scheduler.get_last_lr()[0]
        f1_str = " ".join(f"{CLASS_NAMES[c]}={val_f1s[c]:.3f}" for c in range(NUM_CLASSES))
        print(f"{epoch:>3} | {avg_loss:>7.4f} | {tr_acc:>6.3f} | {tr_f1:>6.3f} | {val_acc:>6.3f} | {val_f1:>6.3f} | {lr:>8.6f} | {f1_str}  ({ep_time:.0f}s)")

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
                "cfg": {
                    "embed_dim": EMBED_DIM,
                    "num_classes": NUM_CLASSES,
                    "patches_per_side": PATCHES_PER_SIDE,
                    "class_names": CLASS_NAMES,
                    "architecture": "v8-patch-segmentation",
                    "backbone": "facebook/dinov2-large",
                },
                "version": "v8-segmentation",
            }, OUT_DIR / "seg_head_v8.pt")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    print(f"\nBest val F1: {best_val_f1:.4f}")
    print(f"Saved to: {OUT_DIR / 'seg_head_v8.pt'}")

    # Final evaluation
    best_ckpt = torch.load(OUT_DIR / "seg_head_v8.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    print(f"\n=== Final Evaluation (best epoch={best_ckpt['epoch']}) ===")

    # Per-class detailed metrics on val set
    all_preds, all_labels = [], []
    with torch.no_grad():
        for tokens, labels in val_loader:
            tokens = tokens.float().to(device)
            logits = model(tokens)
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

    # FP rate: OTHER predicted as glacial
    other_mask = (labels == 3) & mask
    other_as_glacial = ((preds != 3) & other_mask).sum()
    other_total = other_mask.sum()
    print(f"\n  FP rate (OTHER→glacial): {other_as_glacial}/{other_total} = {other_as_glacial/max(other_total,1)*100:.2f}%")

    # Per-class precision/recall
    print(f"\n  {'Class':>8} | {'Precision':>9} | {'Recall':>9} | {'F1':>9} | {'Support':>9}")
    print(f"  {'-'*55}")
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
