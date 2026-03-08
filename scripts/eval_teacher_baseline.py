#!/usr/bin/env python3
"""Evaluate V4b teacher model on V5 validation set — per-class P/R/F1 for LDA/LVF/CCF."""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ── Config ──
DATA_DIR = Path("/disk1/cspark/hirise-api/data/HiRISE/v5_merged_data")
CKPT_PATH = DATA_DIR / "marslandform_v4b_deploy.pt"
TILES_DIR = DATA_DIR / "tiles"
LABELS_PATH = DATA_DIR / "tile_labels_v5.json"
SPLITS_PATH = DATA_DIR / "tile_splits_v5.json"
MOLA_PATH = DATA_DIR / "mola_features_by_tile.npy"

CLASS_NAMES_4 = ["LDA", "LVF", "CCF", "OTHER"]
OLD_LANDFORM_INDICES = [0, 1, 2]  # LDA, LVF, CCF
CLASS_TO_IDX = {"LDA": 0, "LVF": 1, "CCF": 2, "OTHER": 3, "SCT": 4}
BATCH_SIZE = 8
NUM_WORKERS = 2


# ── Model Architecture (from Colab notebook) ──
class FiLMLayer(nn.Module):
    def __init__(self, mola_dim: int, visual_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.mola_encoder = nn.Sequential(
            nn.BatchNorm1d(mola_dim),
            nn.Linear(mola_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
        )
        self.gamma_proj = nn.Linear(hidden_dim, visual_dim)
        self.beta_proj = nn.Linear(hidden_dim, visual_dim)
        nn.init.ones_(self.gamma_proj.bias)
        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)

    def forward(self, visual_features, mola_features):
        h = self.mola_encoder(mola_features)
        gamma = self.gamma_proj(h)
        beta = self.beta_proj(h)
        return gamma * visual_features + beta


class FiLMClassifier(nn.Module):
    def __init__(self, visual_dim=768, mola_dim=25, film_hidden=64,
                 head_hidden=128, num_classes=4, dropout=0.4):
        super().__init__()
        self.film = FiLMLayer(mola_dim, visual_dim, film_hidden)
        self.classifier = nn.Sequential(
            nn.Linear(visual_dim, head_hidden),
            nn.BatchNorm1d(head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, num_classes),
        )

    def forward(self, embeddings, mola):
        modulated = self.film(embeddings, mola)
        return self.classifier(modulated)


# ── Dataset ──
class TileDataset(Dataset):
    def __init__(self, indices, labels, mola_dict, tiles_dir, transform):
        self.indices = indices
        self.labels = labels
        self.mola_dict = mola_dict  # {image_id: {row_col: np.array(25,)}}
        self.tiles_dir = tiles_dir
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        entry = self.labels[idx]
        image_id = entry["image_id"]
        row = entry.get("tile_row", 0)
        col = entry.get("tile_col", 0)

        # Find tile file
        tile_path = self.tiles_dir / image_id / f"tile_{row:03d}_{col:03d}.jpg"
        if not tile_path.exists():
            # Fallback: search directory
            img_dir = self.tiles_dir / image_id
            if img_dir.exists():
                files = sorted(img_dir.iterdir())
                tile_idx = entry.get("tile_idx", 0)
                if tile_idx < len(files):
                    tile_path = files[tile_idx]

        img = Image.open(tile_path).convert("RGB")
        pixel_values = self.transform(img)

        # MOLA features: nested dict {image_id: {"row_col": array(25,)}}
        mola_key = f"{row}_{col}"
        img_mola = self.mola_dict.get(image_id, {})
        mola_feat = img_mola.get(mola_key, np.zeros(25, dtype=np.float32))
        mola_feat = torch.tensor(np.array(mola_feat), dtype=torch.float32)

        label = CLASS_TO_IDX.get(entry["label"], 3)
        return pixel_values, mola_feat, label


def main():
    print("=" * 72)
    print("V4b Teacher Model Evaluation on V5 Validation Set")
    print("=" * 72)

    device = torch.device("cpu")

    # ── Load data ──
    print("\n[1/4] Loading data...")
    labels = json.loads(LABELS_PATH.read_text())
    splits = json.loads(SPLITS_PATH.read_text())
    mola_features = np.load(MOLA_PATH, allow_pickle=True).item()

    val_indices = splits["val"]
    print(f"  Total val indices: {len(val_indices)}")

    # Filter to old landform classes only (LDA, LVF, CCF) for speed
    # Also include OTHER to check false positives from other classes
    old_val_indices = []
    for idx in val_indices:
        lbl = CLASS_TO_IDX.get(labels[idx]["label"], -1)
        if lbl in OLD_LANDFORM_INDICES:
            old_val_indices.append(idx)

    from collections import Counter
    dist = Counter(labels[i]["label"] for i in old_val_indices)
    print(f"  Old-class val tiles: {len(old_val_indices)}")
    print(f"  Distribution: {dict(dist)}")

    # DINOv2 preprocessing
    transform = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = TileDataset(old_val_indices, labels, mola_features, TILES_DIR, transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=False)

    # ── Load model ──
    print("\n[2/4] Loading V4b teacher model...")
    from transformers import Dinov2Model
    from peft import LoraConfig, get_peft_model

    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    state_dict = ckpt["model_state_dict"]

    # Split state dict
    backbone_state = {}
    head_state = {}
    for k, v in state_dict.items():
        if k.startswith("backbone."):
            backbone_state[k[len("backbone."):]] = v
        elif k.startswith("head."):
            head_state[k[len("head."):]] = v
        elif k.startswith("film.") or k.startswith("classifier."):
            head_state[k] = v

    # Build backbone
    backbone = Dinov2Model.from_pretrained(cfg["model_name"])
    lora_config = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg.get("lora_dropout", 0.1),
        target_modules=cfg.get("lora_targets", ["query", "key", "value"]),
        bias="none",
    )
    backbone = get_peft_model(backbone, lora_config)
    result = backbone.load_state_dict(backbone_state, strict=False)
    print(f"  Backbone: {len(backbone_state)} tensors loaded "
          f"(missing={len(result.missing_keys)}, unexpected={len(result.unexpected_keys)})")
    backbone.eval()
    backbone.to(device)

    # Build FiLM classifier
    classifier = FiLMClassifier(
        visual_dim=int(cfg.get("hidden_dim", 768)),
        mola_dim=int(cfg.get("mola_dim", 25)),
        film_hidden=64,
        head_hidden=int(cfg.get("head_hidden", 128)),
        num_classes=int(cfg.get("num_classes", 4)),
        dropout=float(cfg.get("dropout", 0.4)),
    )
    result = classifier.load_state_dict(head_state, strict=True)
    print(f"  FiLM classifier loaded (num_classes={cfg.get('num_classes')})")
    classifier.eval()
    classifier.to(device)

    # ── Run inference ──
    print(f"\n[3/4] Running inference on {len(old_val_indices)} tiles (CPU)...")
    y_true, y_pred, y_prob = [], [], []
    total = len(loader)

    with torch.no_grad():
        for batch_idx, (pixel_values, mola_feat, targets) in enumerate(loader):
            pixel_values = pixel_values.to(device)
            mola_feat = mola_feat.to(device)

            # Forward: backbone → CLS token → FiLM classifier
            outputs = backbone(pixel_values=pixel_values)
            cls_token = outputs.last_hidden_state[:, 0]
            logits = classifier(cls_token, mola_feat)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            y_true.extend(targets.numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            y_prob.extend(probs.cpu().numpy().tolist())

            if (batch_idx + 1) % 50 == 0 or batch_idx == total - 1:
                print(f"    Batch {batch_idx + 1}/{total} done")

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # ── Compute metrics ──
    print(f"\n[4/4] Computing metrics...")

    # Teacher baseline: macro-F1 over old landform classes
    teacher_baseline = f1_score(
        y_true, y_pred,
        labels=OLD_LANDFORM_INDICES,
        average="macro",
        zero_division=0,
    )

    # Per-class precision/recall/F1
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred,
        labels=OLD_LANDFORM_INDICES,
        zero_division=0,
    )

    # Full classification report (including OTHER predictions)
    all_labels = list(range(4))
    report = classification_report(
        y_true, y_pred,
        labels=all_labels,
        target_names=CLASS_NAMES_4,
        digits=4,
        zero_division=0,
    )

    # ── Print results ──
    print("\n" + "=" * 72)
    print("RESULTS: V4b Teacher on V5 Val Set (Old Landform Classes)")
    print("=" * 72)

    print(f"\nTeacher baseline old-class macro-F1 (LDA/LVF/CCF): {teacher_baseline:.4f}")
    print(f"\nPer-class breakdown:")
    print(f"{'Class':>8} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print("-" * 52)
    for i, cls_name in enumerate(["LDA", "LVF", "CCF"]):
        print(f"{cls_name:>8} {precision[i]:>10.4f} {recall[i]:>10.4f} {f1[i]:>10.4f} {int(support[i]):>10}")

    print(f"\n{'macro':>8} {np.mean(precision):>10.4f} {np.mean(recall):>10.4f} {teacher_baseline:>10.4f} {int(np.sum(support)):>10}")

    print(f"\n\nFull classification report (true labels are LDA/LVF/CCF only,")
    print(f"but model can predict OTHER too):\n")
    print(report)

    # Also compute 4-class macro F1 for reference
    # For this we need to run on ALL val data including OTHER
    print(f"\nNote: This evaluation only includes tiles with true labels LDA/LVF/CCF.")
    print(f"      V4b overall 4-class macro-F1 requires evaluating all val tiles including OTHER.")
    print(f"      Checkpoint metadata: val_f1={ckpt.get('val_f1', '?'):.4f}, epoch={ckpt.get('epoch', '?')}")


if __name__ == "__main__":
    main()
