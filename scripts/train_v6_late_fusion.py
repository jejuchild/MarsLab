#!/usr/bin/env python3
"""
Train V6b Late Fusion classifier with explicit MOLA z-score normalization.

Uses pre-extracted DINOv2 embeddings (768-dim) + MOLA features (25-dim).
Architecture: independent visual_head + mola_head → weighted logit fusion.

Key change from V6: MOLA features are z-score normalized using training
distribution statistics (stored as registered buffers in the model).
This replaces BatchNorm1d for more robust OOD handling.

Data: /disk1/cspark/MarsLab/Data/HiRISE/v5_retrain/
  embeddings_v5.npy   (67904, 768)
  mola_features_v5.npy  {product_id: {row_col: 25-dim}}
  tile_labels_v5.json   [{image_id, tile_row, tile_col, label, ...}, ...]
"""

import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from collections import Counter
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

sys.path.insert(0, "/disk1/cspark/hirise-api")
from scripts.marslandform_v2.models.late_fusion_classifier import LateFusionClassifier

DATA_DIR = Path("/disk1/cspark/MarsLab/Data/HiRISE/v5_retrain")
OUT_DIR = DATA_DIR
CLASS_NAMES = ["LDA", "LVF", "CCF", "OTHER"]
NUM_CLASSES = 4


def load_data():
    """Load and align training data. Returns (embeddings, mola, labels, product_ids)."""
    embeddings = np.load(DATA_DIR / "embeddings_v5.npy")
    mola_dict = np.load(DATA_DIR / "mola_features_v5.npy", allow_pickle=True).item()
    with open(DATA_DIR / "tile_labels_v5.json") as f:
        labels_list = json.load(f)

    emb_list, mola_list, label_list, pids = [], [], [], []
    for i, tile_info in enumerate(labels_list):
        pid = tile_info["image_id"]
        key = f"{tile_info['tile_row']}_{tile_info['tile_col']}"
        if pid in mola_dict and key in mola_dict[pid]:
            emb_list.append(i)
            mola_list.append(mola_dict[pid][key])
            lbl = tile_info["label"]
            label_list.append(CLASS_NAMES.index(lbl) if lbl in CLASS_NAMES else 3)
            pids.append(pid)

    return (
        embeddings[emb_list].astype(np.float32),
        np.array(mola_list, dtype=np.float32),
        np.array(label_list, dtype=np.int64),
        pids,
    )


def split_by_product(pids, labels, val_ratio=0.15, seed=42):
    """Split by product_id to prevent data leakage."""
    rng = np.random.RandomState(seed)
    unique_pids = sorted(set(pids))
    rng.shuffle(unique_pids)

    n_val = max(1, int(len(unique_pids) * val_ratio))
    val_pids = set(unique_pids[:n_val])

    train_idx = [i for i, p in enumerate(pids) if p not in val_pids]
    val_idx = [i for i, p in enumerate(pids) if p in val_pids]

    return np.array(train_idx), np.array(val_idx)


def compute_metrics(preds, labels):
    """Compute accuracy and per-class F1."""
    acc = (preds == labels).mean()
    f1s = []
    for c in range(NUM_CLASSES):
        tp = ((preds == c) & (labels == c)).sum()
        fp = ((preds == c) & (labels != c)).sum()
        fn = ((preds != c) & (labels == c)).sum()
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        f1s.append(f1)
    return acc, np.mean(f1s), f1s


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    embeddings, mola, labels, pids = load_data()
    print(f"Total tiles: {len(embeddings)}")
    print(f"Label distribution: {dict(Counter(labels))}")

    # Split
    train_idx, val_idx = split_by_product(pids, labels)
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")

    # Compute MOLA normalization stats from training set only
    mola_train = mola[train_idx]
    mola_mean = mola_train.mean(axis=0).astype(np.float32)
    mola_std = mola_train.std(axis=0).astype(np.float32)
    print(f"\nMOLA normalization stats (from training set):")
    print(f"  Mean range: [{mola_mean.min():.2f}, {mola_mean.max():.2f}]")
    print(f"  Std range:  [{mola_std.min():.6f}, {mola_std.max():.2f}]")
    n_const = (mola_std < 1e-6).sum()
    if n_const > 0:
        print(f"  WARNING: {n_const} constant features (std < 1e-6) — will be clamped to 1e-6")

    X_emb_tr = torch.from_numpy(embeddings[train_idx]).float()
    X_mola_tr = torch.from_numpy(mola[train_idx]).float()
    y_tr = torch.from_numpy(labels[train_idx]).long()

    X_emb_val = torch.from_numpy(embeddings[val_idx]).float()
    X_mola_val = torch.from_numpy(mola[val_idx]).float()
    y_val = torch.from_numpy(labels[val_idx]).long()

    # Weighted random sampler for class balance
    class_counts = np.bincount(labels[train_idx], minlength=NUM_CLASSES)
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = class_weights[labels[train_idx]]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_ds = TensorDataset(X_emb_tr, X_mola_tr, y_tr)
    val_ds = TensorDataset(X_emb_val, X_mola_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=512, sampler=sampler, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=2048, shuffle=False)

    # Model
    model = LateFusionClassifier(
        visual_dim=768,
        mola_dim=25,
        visual_hidden=128,
        mola_hidden=64,
        num_classes=NUM_CLASSES,
        dropout=0.5,
        init_vis_weight=0.7,
    ).to(device)

    # Let fusion weights learn freely — heads optimize better this way.
    # At inference, we'll override weights to vis=0.7/mola=0.3 if needed.
    print(f"Initial fusion weights: vis={model.vis_weight.item():.3f}, mola={model.mola_weight.item():.3f}")

    # Set MOLA normalization stats (baked into model as registered buffers)
    model.set_mola_stats(
        mean=torch.from_numpy(mola_mean),
        std=torch.from_numpy(mola_std),
    )
    print(f"\nMOLA stats set in model (as registered buffers)")

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {trainable:,} trainable / {total_params:,} total")

    # Loss: focal loss for hard examples + MOLA logit regularization
    class FocalLoss(nn.Module):
        def __init__(self, gamma=2.0):
            super().__init__()
            self.gamma = gamma

        def forward(self, logits, targets):
            ce = nn.functional.cross_entropy(logits, targets, reduction="none")
            pt = torch.exp(-ce)
            return ((1 - pt) ** self.gamma * ce).mean()

    criterion = FocalLoss(gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60, eta_min=1e-5)

    # Training loop
    best_val_f1 = 0
    patience = 15
    no_improve = 0
    n_epochs = 60
    mola_logit_reg_weight = 0.01  # Soft penalty to keep MOLA logits bounded

    print(f"\n{'Epoch':>5} | {'TrLoss':>7} | {'TrAcc':>6} | {'TrF1':>6} | {'VlAcc':>6} | {'VlF1':>6} | {'w_vis':>5} | {'w_mola':>5} | {'LR':>8} | Per-class Val F1")
    print("-" * 130)

    for epoch in range(1, n_epochs + 1):
        # Train
        model.train()
        total_loss = 0
        all_preds_tr, all_labels_tr = [], []

        for emb_b, mola_b, y_b in train_loader:
            emb_b, mola_b, y_b = emb_b.to(device), mola_b.to(device), y_b.to(device)
            logits = model(emb_b, mola_b)
            loss = criterion(logits, y_b)

            # Soft regularization: penalize large MOLA logits to prevent extreme OOD outputs
            mola_normed = model._normalize_mola(mola_b)
            mola_logits = model.mola_head(mola_normed)
            mola_logit_penalty = (mola_logits ** 2).mean()
            loss = loss + mola_logit_reg_weight * mola_logit_penalty

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * len(y_b)
            all_preds_tr.append(logits.argmax(1).cpu().numpy())
            all_labels_tr.append(y_b.cpu().numpy())

        scheduler.step()
        avg_loss = total_loss / len(train_idx)
        preds_tr = np.concatenate(all_preds_tr)
        labels_tr_np = np.concatenate(all_labels_tr)
        tr_acc, tr_f1, _ = compute_metrics(preds_tr, labels_tr_np)

        # Validate
        model.eval()
        all_preds_val, all_labels_val = [], []
        with torch.no_grad():
            for emb_b, mola_b, y_b in val_loader:
                emb_b, mola_b = emb_b.to(device), mola_b.to(device)
                logits = model(emb_b, mola_b)
                all_preds_val.append(logits.argmax(1).cpu().numpy())
                all_labels_val.append(y_b.numpy())

        preds_val = np.concatenate(all_preds_val)
        labels_val_np = np.concatenate(all_labels_val)
        val_acc, val_f1, val_f1s = compute_metrics(preds_val, labels_val_np)

        w_vis = model.vis_weight.item()
        w_mola = model.mola_weight.item()
        lr = scheduler.get_last_lr()[0]

        f1_str = " ".join(f"{CLASS_NAMES[c]}={val_f1s[c]:.3f}" for c in range(NUM_CLASSES))
        print(f"{epoch:>5} | {avg_loss:>7.4f} | {tr_acc:>6.3f} | {tr_f1:>6.3f} | {val_acc:>6.3f} | {val_f1:>6.3f} | {w_vis:>5.3f} | {w_mola:>5.3f} | {lr:>8.6f} | {f1_str}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            no_improve = 0
            # Save best — includes mola_mean and mola_std as part of state_dict
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_f1": val_f1,
                "val_acc": val_acc,
                "train_f1": tr_f1,
                "vis_weight": w_vis,
                "mola_weight": w_mola,
                "mola_mean": mola_mean.tolist(),
                "mola_std": mola_std.tolist(),
                "cfg": {
                    "visual_dim": 768,
                    "mola_dim": 25,
                    "visual_hidden": 128,
                    "mola_hidden": 64,
                    "num_classes": NUM_CLASSES,
                    "dropout": 0.5,
                    "init_vis_weight": 0.7,
                    "class_names": CLASS_NAMES,
                    "architecture": "late_fusion_v6b",
                },
                "version": "v6b-late-fusion-normalized",
            }, OUT_DIR / "late_fusion_v6b.pt")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    print(f"\nBest val F1: {best_val_f1:.4f}")
    print(f"Saved to: {OUT_DIR / 'late_fusion_v6b.pt'}")

    # Final evaluation with best model
    best_ckpt = torch.load(OUT_DIR / "late_fusion_v6b.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    print(f"\n=== Final Evaluation (best epoch={best_ckpt['epoch']}) ===")
    print(f"Learned weights: vis={best_ckpt['vis_weight']:.4f}, mola={best_ckpt['mola_weight']:.4f}")

    # Full dataset evaluation
    with torch.no_grad():
        all_emb = torch.from_numpy(embeddings).float().to(device)
        all_mola_t = torch.from_numpy(mola).float().to(device)
        all_logits = model(all_emb, all_mola_t)
        all_preds = all_logits.argmax(1).cpu().numpy()

    acc, f1, f1s = compute_metrics(all_preds, labels)
    print(f"Full dataset: acc={acc:.4f}, macro_f1={f1:.4f}")
    for c in range(NUM_CLASSES):
        print(f"  {CLASS_NAMES[c]}: F1={f1s[c]:.4f}")

    # Visual-only evaluation
    with torch.no_grad():
        vis_logits = model.get_visual_logits(all_emb)
        vis_preds = vis_logits.argmax(1).cpu().numpy()
    acc_v, f1_v, f1s_v = compute_metrics(vis_preds, labels)
    print(f"\nVisual-only: acc={acc_v:.4f}, macro_f1={f1_v:.4f}")

    # MOLA-only evaluation
    with torch.no_grad():
        mola_logits = model.get_mola_logits(all_mola_t)
        mola_preds = mola_logits.argmax(1).cpu().numpy()
    acc_m, f1_m, f1s_m = compute_metrics(mola_preds, labels)
    print(f"MOLA-only:   acc={acc_m:.4f}, macro_f1={f1_m:.4f}")

    # Check MOLA logit range on training data
    with torch.no_grad():
        mola_logits_np = mola_logits.cpu().numpy()
    print(f"\nMOLA logit range on full dataset:")
    for c in range(NUM_CLASSES):
        col = mola_logits_np[:, c]
        print(f"  {CLASS_NAMES[c]}: mean={col.mean():.2f}, std={col.std():.2f}, "
              f"min={col.min():.2f}, max={col.max():.2f}")


if __name__ == "__main__":
    t0 = time.time()
    train()
    print(f"\nTotal training time: {time.time() - t0:.1f}s")
