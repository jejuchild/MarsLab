#!/usr/bin/env python3
"""Multi-label MIL classifier: support multiple landforms per image.

Key changes from single-label:
1. Binary cross-entropy per class instead of softmax cross-entropy
2. Sigmoid outputs instead of softmax
3. Per-class F1 thresholds instead of argmax
4. Label format: image_id -> list of classes (e.g., ["LDA", "LVF"])
"""
import json
import math
import random
import sys
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.marslandform_v2.config import CLASS_ORDER, get_config
from scripts.marslandform_v2.models.mil_classifier import (
    AttentionMILClassifier,
    MultiHeadGatedAttention,
    MOLACrossModalFusion,
    build_scheduler,
    load_embeddings,
    load_labels,
    load_mola_features,
    mil_collate_fn,
    set_seed,
)

DATA_ROOT = ROOT / "Data/HiRISE/v2_output"
LANDFORM_CLASSES = CLASS_ORDER[:4]  # ["LDA", "LVF", "CCF", "GLF"]


class MultiLabelMILDataset(Dataset):
    """Dataset that supports multi-label targets as binary vectors."""
    
    def __init__(
        self,
        image_ids: List[str],
        embeddings_dict: Dict[str, np.ndarray],
        mola_dict: Dict[str, np.ndarray],
        multi_labels: Dict[str, List[int]],  # image_id -> list of class indices
        max_tiles: int = 128,
    ):
        self.image_ids = list(image_ids)
        self.emb = embeddings_dict
        self.mola = mola_dict
        self.multi_labels = multi_labels
        self.max_tiles = max_tiles
        self.num_classes = 5  # LDA, LVF, CCF, GLF, BACKGROUND
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        tiles = np.asarray(self.emb[img_id], dtype=np.float32)
        mola = np.asarray(self.mola[img_id], dtype=np.float32)
        
        # Subsample tiles
        if tiles.shape[0] > self.max_tiles:
            keep = np.random.choice(tiles.shape[0], self.max_tiles, replace=False)
            tiles = tiles[keep]
        
        # Multi-hot label vector
        label_vec = np.zeros(self.num_classes, dtype=np.float32)
        for cls_idx in self.multi_labels.get(img_id, []):
            if 0 <= cls_idx < self.num_classes:
                label_vec[cls_idx] = 1.0
        
        return torch.from_numpy(tiles), torch.from_numpy(mola), torch.from_numpy(label_vec), img_id


def multi_label_collate(batch):
    """Collate with padding for variable-length tile sequences."""
    tile_seqs, mola_feats, label_vecs, image_ids = zip(*batch)
    batch_size = len(tile_seqs)
    max_tiles = max(x.size(0) for x in tile_seqs)
    embed_dim = tile_seqs[0].size(1)
    
    padded = torch.zeros(batch_size, max_tiles, embed_dim)
    mask = torch.zeros(batch_size, max_tiles, dtype=torch.bool)
    for i, tiles in enumerate(tile_seqs):
        n = tiles.size(0)
        padded[i, :n] = tiles
        mask[i, :n] = True
    
    return {
        "tile_embeddings": padded,
        "tile_mask": mask,
        "mola_features": torch.stack(mola_feats),
        "labels": torch.stack(label_vecs),  # (B, num_classes) multi-hot
        "image_ids": list(image_ids),
    }


class MultiLabelMILClassifier(nn.Module):
    """MIL classifier with sigmoid outputs for multi-label classification."""
    
    def __init__(self, embed_dim=768, hidden_dim=256, attention_dim=128,
                 num_heads=4, mola_dim=23, num_classes=5, dropout=0.3):
        super().__init__()
        
        self.tile_transform = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        self.attention = MultiHeadGatedAttention(hidden_dim, attention_dim, num_heads)
        self.mola_fusion = MOLACrossModalFusion(mola_dim, hidden_dim, dropout)
        
        # Separate head per class for multi-label
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            )
            for _ in range(num_classes)
        ])
        
        self.num_classes = num_classes
    
    def forward(self, tile_embeddings, tile_mask, mola_features):
        features = self.tile_transform(tile_embeddings)
        bag_feature, att_weights = self.attention(features, tile_mask)
        fused = self.mola_fusion(bag_feature, mola_features)
        
        # Per-class logits
        logits = torch.cat([clf(fused) for clf in self.classifiers], dim=1)  # (B, num_classes)
        return logits, att_weights


class AsymmetricLoss(nn.Module):
    """Asymmetric Loss for multi-label classification (Ridnik et al. 2021).
    Better than BCE for imbalanced multi-label problems."""
    
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
    
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        
        # Asymmetric clipping for negative samples
        probs_neg = probs.clamp(max=1 - self.clip) if self.clip > 0 else probs
        
        # Basic BCE
        loss_pos = targets * torch.log(probs + self.eps)
        loss_neg = (1 - targets) * torch.log(1 - probs_neg + self.eps)
        
        # Asymmetric focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt_pos = probs
            pt_neg = 1 - probs_neg
            loss_pos *= (1 - pt_pos) ** self.gamma_pos
            loss_neg *= pt_neg ** self.gamma_neg
        
        loss = -(loss_pos + loss_neg)
        return loss.mean()


def convert_to_multi_labels(
    labels_path: Path,
    multi_candidates_path: Optional[Path] = None,
) -> Dict[str, List[int]]:
    """Convert single-label to multi-label format, using multi-label candidates if available."""
    labels_raw = json.loads(labels_path.read_text())
    
    multi_labels = {}
    for img_id, label in labels_raw.items():
        if isinstance(label, str):
            idx = CLASS_ORDER.index(label) if label in CLASS_ORDER else 4
        else:
            idx = int(label)
        multi_labels[img_id] = [idx]
    
    # Add multi-label candidates from audit
    if multi_candidates_path and multi_candidates_path.exists():
        candidates = json.loads(multi_candidates_path.read_text())
        multi_count = 0
        for c in candidates:
            img_id = c["image_id"]
            if img_id in multi_labels:
                # Add secondary labels where model gives high probability
                high_classes = c.get("high_prob_classes", [])
                for cls_name, prob in high_classes:
                    cls_idx = CLASS_ORDER.index(cls_name)
                    if cls_idx not in multi_labels[img_id] and prob > 0.30:
                        multi_labels[img_id].append(cls_idx)
                        multi_count += 1
        
        print(f"  Added {multi_count} secondary labels from multi-label candidates")
    
    return multi_labels


def compute_multi_label_metrics(
    y_true: np.ndarray,  # (N, C) binary
    y_pred: np.ndarray,  # (N, C) binary
    class_names: List[str],
) -> Dict[str, Any]:
    """Compute per-class and overall metrics for multi-label classification."""
    num_classes = y_true.shape[1]
    results = {"per_class": {}, "overall": {}}
    
    f1_scores = []
    for c in range(num_classes):
        tp = np.sum((y_pred[:, c] == 1) & (y_true[:, c] == 1))
        fp = np.sum((y_pred[:, c] == 1) & (y_true[:, c] == 0))
        fn = np.sum((y_pred[:, c] == 0) & (y_true[:, c] == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        results["per_class"][class_names[c]] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(y_true[:, c].sum()),
        }
        f1_scores.append(f1)
    
    # Landform F1 (exclude BACKGROUND)
    landform_f1s = f1_scores[:4]
    results["overall"]["landform_macro_f1"] = float(np.mean(landform_f1s))
    results["overall"]["macro_f1_all"] = float(np.mean(f1_scores))
    
    # Also compute single-label metrics (primary class = argmax of true labels, or highest-prob)
    return results


def train_multi_label(
    train_ids: List[str],
    val_ids: List[str],
    emb_dict: Dict[str, np.ndarray],
    mola_dict: Dict[str, np.ndarray],
    multi_labels: Dict[str, List[int]],
    output_dir: Path,
    epochs: int = 50,
    patience: int = 15,
    lr: float = 1e-3,
):
    """Train multi-label MIL classifier."""
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    
    train_ds = MultiLabelMILDataset(train_ids, emb_dict, mola_dict, multi_labels)
    val_ds = MultiLabelMILDataset(val_ids, emb_dict, mola_dict, multi_labels)
    
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=multi_label_collate, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=multi_label_collate, num_workers=0)
    
    model = MultiLabelMILClassifier().to(device)
    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    total_steps = len(train_loader) * epochs
    scheduler = build_scheduler(optimizer, total_steps, warmup_ratio=0.1)
    
    best_f1 = -1.0
    best_epoch = -1
    patience_counter = 0
    best_state = None
    history = {"train_loss": [], "val_loss": [], "val_landform_f1": []}
    
    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_losses = []
        for batch in train_loader:
            tiles = batch["tile_embeddings"].to(device)
            mask = batch["tile_mask"].to(device)
            mola = batch["mola_features"].to(device)
            labels = batch["labels"].to(device)
            
            optimizer.zero_grad()
            logits, _ = model(tiles, mask, mola)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_losses.append(loss.item())
        
        # Validate
        model.eval()
        val_losses = []
        all_true = []
        all_probs = []
        
        with torch.no_grad():
            for batch in val_loader:
                tiles = batch["tile_embeddings"].to(device)
                mask = batch["tile_mask"].to(device)
                mola = batch["mola_features"].to(device)
                labels = batch["labels"].to(device)
                
                logits, _ = model(tiles, mask, mola)
                loss = criterion(logits, labels)
                val_losses.append(loss.item())
                
                probs = torch.sigmoid(logits).cpu().numpy()
                all_true.append(labels.cpu().numpy())
                all_probs.append(probs)
        
        all_true = np.concatenate(all_true)
        all_probs = np.concatenate(all_probs)
        
        # Use 0.5 threshold for validation
        all_pred = (all_probs > 0.5).astype(int)
        metrics = compute_multi_label_metrics(all_true, all_pred, CLASS_ORDER)
        val_f1 = metrics["overall"]["landform_macro_f1"]
        
        history["train_loss"].append(float(np.mean(train_losses)))
        history["val_loss"].append(float(np.mean(val_losses)))
        history["val_landform_f1"].append(val_f1)
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"  [Epoch {epoch}] train_loss={np.mean(train_losses):.4f} val_loss={np.mean(val_losses):.4f} val_LF_F1={val_f1:.4f}")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            best_state = deepcopy(model.state_dict())
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch} (best={best_epoch})")
            break
    
    # Save best model
    if best_state:
        model.load_state_dict(best_state)
    torch.save({"model_state_dict": best_state or model.state_dict(), "best_epoch": best_epoch}, 
               output_dir / "best_multi_label_model.pt")
    
    (output_dir / "training_curves.json").write_text(json.dumps(history, indent=2))
    print(f"  Best val LF F1: {best_f1:.4f} at epoch {best_epoch}")
    
    return model, best_f1


@torch.no_grad()
def evaluate_multi_label(
    model: MultiLabelMILClassifier,
    test_ids: List[str],
    emb_dict: Dict[str, np.ndarray],
    mola_dict: Dict[str, np.ndarray],
    multi_labels: Dict[str, List[int]],
    output_dir: Path,
):
    """Evaluate multi-label model and also compute single-label metrics for comparison."""
    from scripts.marslandform_v2.models.mil_classifier import compute_metrics
    
    device = torch.device("cpu")
    model.eval()
    
    test_ds = MultiLabelMILDataset(test_ids, emb_dict, mola_dict, multi_labels)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=multi_label_collate, num_workers=0)
    
    all_true = []
    all_probs = []
    all_image_ids = []
    
    for batch in test_loader:
        tiles = batch["tile_embeddings"].to(device)
        mask = batch["tile_mask"].to(device)
        mola = batch["mola_features"].to(device)
        labels = batch["labels"]
        
        logits, _ = model(tiles, mask, mola)
        probs = torch.sigmoid(logits).cpu().numpy()
        
        all_true.append(labels.numpy())
        all_probs.append(probs)
        all_image_ids.extend(batch["image_ids"])
    
    all_true = np.concatenate(all_true)
    all_probs = np.concatenate(all_probs)
    
    # Multi-label metrics with 0.5 threshold
    all_pred = (all_probs > 0.5).astype(int)
    ml_metrics = compute_multi_label_metrics(all_true, all_pred, CLASS_ORDER)
    
    # Single-label comparison: argmax of probs
    single_pred = np.argmax(all_probs, axis=1).tolist()
    single_true = [np.argmax(t) if t.sum() > 0 else 4 for t in all_true]  # primary class
    sl_metrics = compute_metrics(single_true, single_pred, num_classes=5)
    
    results = {
        "multi_label": ml_metrics,
        "single_label_comparison": {
            "landform_macro_f1": sl_metrics["landform_macro_f1"],
            "macro_f1_all": sl_metrics["macro_f1_all"],
            "per_class_f1": {CLASS_ORDER[i]: sl_metrics["f1"][i] for i in range(5)},
        },
    }
    
    (output_dir / "test_metrics.json").write_text(json.dumps(results, indent=2))
    
    print(f"\n  Multi-label Landform F1: {ml_metrics['overall']['landform_macro_f1']:.4f}")
    print(f"  Single-label Landform F1: {sl_metrics['landform_macro_f1']:.4f}")
    for cls in CLASS_ORDER:
        ml_f1 = ml_metrics["per_class"][cls]["f1"]
        print(f"    {cls}: ML F1={ml_f1:.3f}")
    
    return results


def run_multi_label():
    print("=" * 60)
    print("MULTI-LABEL MIL CLASSIFIER")
    print("=" * 60)
    
    set_seed(42)
    cfg = get_config()
    device = torch.device("cpu")
    
    # Load data
    print("\nLoading data...")
    emb_dict = load_embeddings(DATA_ROOT / "embeddings_mil")
    mola_raw = np.load(DATA_ROOT / "mola_features_by_image.npy", allow_pickle=True).item()
    mola_dict = {str(k): np.asarray(v, dtype=np.float32) for k, v in mola_raw.items()}
    
    # Convert to multi-labels
    labels_path = DATA_ROOT / "labels_simple.json"
    multi_candidates_path = DATA_ROOT / "label_audit/multi_label_candidates.json"
    
    # Use cleaned labels if available
    cleaned_path = DATA_ROOT / "label_audit/labels_cleaned.json"
    if cleaned_path.exists():
        print(f"  Using cleaned labels from {cleaned_path}")
        labels_path = cleaned_path
    
    print("  Converting to multi-label format...")
    multi_labels = convert_to_multi_labels(labels_path, multi_candidates_path)
    
    # Filter to valid images
    valid_ids = sorted(set(emb_dict.keys()) & set(mola_dict.keys()) & set(multi_labels.keys()))
    multi_labels = {k: multi_labels[k] for k in valid_ids}
    
    multi_count = sum(1 for v in multi_labels.values() if len(v) > 1)
    print(f"  Valid images: {len(valid_ids)}")
    print(f"  Multi-label images: {multi_count}")
    
    # Use canonical split
    split_path = DATA_ROOT / "models/multihead_improved/data_split.json"
    split = json.loads(split_path.read_text())
    train_ids = [i for i in split["train_ids"] if i in multi_labels]
    val_ids = [i for i in split["val_ids"] if i in multi_labels]
    test_ids = [i for i in split["test_ids"] if i in multi_labels]
    
    print(f"  Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")
    
    # Train
    out_dir = DATA_ROOT / "models/multi_label"
    print(f"\nTraining multi-label MIL...")
    model, best_f1 = train_multi_label(
        train_ids, val_ids, emb_dict, mola_dict, multi_labels, out_dir,
        epochs=50, patience=15
    )
    
    # Evaluate
    print(f"\nEvaluating on test set...")
    results = evaluate_multi_label(model, test_ids, emb_dict, mola_dict, multi_labels, out_dir)
    
    return results


if __name__ == "__main__":
    run_multi_label()
