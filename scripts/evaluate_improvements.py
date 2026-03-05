#!/usr/bin/env python3
"""
Evaluate F1 improvements from TTA, Overlap voting, and CRF smoothing.

Runs on V4 expanded test split with the V4b FiLM classifier.
Pre-computes embeddings once, then evaluates classifier variants.

Usage:
    python scripts/evaluate_improvements.py [--max-tiles 2000]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import f1_score, classification_report
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR = ROOT / "Data" / "HiRISE" / "v4_colab_data_expanded"
CLASSES = ["LDA", "LVF", "CCF", "OTHER"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

TILE_TRANSFORM = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

# TTA augmentations (applied to PIL images BEFORE transform)
TTA_AUGMENTS = [
    ("original", lambda img: img),
    ("hflip", lambda img: img.transpose(Image.FLIP_LEFT_RIGHT)),
    ("vflip", lambda img: img.transpose(Image.FLIP_TOP_BOTTOM)),
    ("rot180", lambda img: img.rotate(180)),
]


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_test_data(max_tiles: int | None = None):
    """Load test split tile images, labels, and MOLA features."""
    with open(DATA_DIR / "tile_labels_v4.json") as f:
        labels = json.load(f)
    with open(DATA_DIR / "tile_splits_v4.json") as f:
        splits = json.load(f)
    with open(DATA_DIR / "tile_index.json") as f:
        tile_index = json.load(f)
    
    mola_dict = np.load(DATA_DIR / "mola_features_by_tile.npy", allow_pickle=True).item()
    
    test_indices = splits["test"]
    if max_tiles:
        # Stratified sampling
        by_class: dict[str, list[int]] = defaultdict(list)
        for idx in test_indices:
            by_class[labels[idx]["label"]].append(idx)
        per_class = max_tiles // len(CLASSES)
        sampled = []
        for cls in CLASSES:
            sampled.extend(by_class[cls][:per_class])
        test_indices = sampled
    
    tiles_data = []
    for idx in test_indices:
        t = labels[idx]
        key = f"{t['image_id']}_{t['tile_row']}_{t['tile_col']}"
        rel_path = tile_index.get(key)
        if not rel_path:
            continue
        full_path = DATA_DIR / rel_path
        if not full_path.exists():
            continue
        
        # MOLA features
        mola_img = mola_dict.get(t["image_id"], {})
        mola_key = f"{t['tile_row']}_{t['tile_col']}"
        mola_feat = mola_img.get(mola_key)
        if mola_feat is None or not isinstance(mola_feat, np.ndarray):
            mola_feat = np.zeros(25, dtype=np.float32)
        elif mola_feat.shape[0] < 25:
            padded = np.zeros(25, dtype=np.float32)
            padded[:mola_feat.shape[0]] = mola_feat
            mola_feat = padded
        
        tiles_data.append({
            "path": str(full_path),
            "label": t["label"],
            "label_idx": CLASS_TO_IDX[t["label"]],
            "mola": mola_feat.astype(np.float32),
            "image_id": t["image_id"],
            "tile_row": t["tile_row"],
            "tile_col": t["tile_col"],
        })
    
    return tiles_data


# ── Model Loading ─────────────────────────────────────────────────────────────

def load_models(device: torch.device):
    """Load DINOv2 backbone + V4b FiLM classifier."""
    from scripts.marslandform_v2.models.dinov2_lora import DinoV2LoRA
    from scripts.marslandform_v2.models.film_classifier import FiLMClassifier
    from scripts.marslandform_v2.config import DINOv2Config
    
    # Backbone
    print("Loading DINOv2 backbone...")
    cfg = DINOv2Config()
    backbone = DinoV2LoRA(cfg, use_lora=True)
    
    # Load V4b fine-tuned backbone weights (includes LoRA + last block)
    film_path = ROOT / "Data" / "HiRISE" / "v3_output" / "models" / "marslandform_v4b_deploy.pt"
    v4b_ckpt = torch.load(film_path, map_location="cpu", weights_only=False)
    backbone_state = {k: v for k, v in v4b_ckpt["model_state_dict"].items()
                      if k.startswith("backbone.")}
    if backbone_state:
        result = backbone.load_state_dict(backbone_state, strict=False)
        matched = len(backbone_state) - len(result.unexpected_keys)
        print(f"  V4b backbone: {matched}/{len(backbone_state)} tensors matched")
    else:
        # Fallback to SSL weights
        ssl_path = ROOT / "Data" / "HiRISE" / "v2_output" / "ssl_lora_weights" / "best_model.pt"
        if ssl_path.exists():
            ckpt = torch.load(ssl_path, map_location="cpu", weights_only=False)
            if "student_backbone" in ckpt:
                backbone.load_state_dict(ckpt["student_backbone"], strict=False)
                print("  SSL LoRA weights loaded (fallback)")
    
    backbone.eval().to(device)
    
    # Classifier (V4b FiLM) — reuse already-loaded checkpoint
    print("Loading V4b FiLM classifier...")
    cfg_data = v4b_ckpt.get("cfg", {})
    
    head_state = {k: v for k, v in v4b_ckpt["model_state_dict"].items()
                  if k.startswith("film.") or k.startswith("classifier.")}
    
    classifier = FiLMClassifier(
        visual_dim=int(cfg_data.get("hidden_dim", 768)),
        mola_dim=int(cfg_data.get("mola_dim", 25)),
        film_hidden=64,
        head_hidden=int(cfg_data.get("head_hidden", 128)),
        num_classes=int(cfg_data.get("num_classes", 4)),
        dropout=float(cfg_data.get("dropout", 0.4)),
    )
    classifier.load_state_dict(head_state, strict=False)
    classifier.eval().to(device)
    
    print(f"  FiLM classifier: {sum(p.numel() for p in classifier.parameters())} params")
    print(f"  Val F1 from training: {v4b_ckpt.get('val_f1', '?'):.4f}")
    
    return backbone, classifier


# ── Embedding Extraction ──────────────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(backbone, images_pil: list[Image.Image], device: torch.device,
                       batch_size: int = 32) -> np.ndarray:
    """Run backbone on PIL images, return (N, 768) embeddings."""
    tensors = [TILE_TRANSFORM(img.convert("RGB")) for img in images_pil]
    all_emb = []
    for i in range(0, len(tensors), batch_size):
        batch = torch.stack(tensors[i:i+batch_size]).to(device)
        emb = backbone(batch)
        all_emb.append(emb.cpu().numpy())
    return np.concatenate(all_emb, axis=0)


@torch.no_grad()
def run_classifier(classifier, embeddings: np.ndarray, mola: np.ndarray,
                   device: torch.device) -> np.ndarray:
    """Run classifier, return (N, 4) logits."""
    emb_t = torch.from_numpy(embeddings).float().to(device)
    mola_t = torch.from_numpy(mola).float().to(device)
    logits = classifier(emb_t, mola_t)
    return logits.cpu().numpy()


# ── Evaluation Methods ────────────────────────────────────────────────────────

def evaluate_baseline(backbone, classifier, tiles_data, device):
    """Baseline: single-pass inference."""
    print("\n" + "="*70)
    print("BASELINE — Single pass, no augmentation")
    print("="*70)
    
    images = [Image.open(t["path"]) for t in tiles_data]
    mola = np.stack([t["mola"] for t in tiles_data])
    labels = np.array([t["label_idx"] for t in tiles_data])
    
    t0 = time.time()
    embeddings = extract_embeddings(backbone, images, device)
    logits = run_classifier(classifier, embeddings, mola, device)
    elapsed = time.time() - t0
    
    preds = np.argmax(logits, axis=1)
    probs = _softmax(logits)
    
    f1 = f1_score(labels, preds, average="macro")
    print(f"\nF1 (macro): {f1:.4f}  ({elapsed:.1f}s)")
    print(classification_report(labels, preds, target_names=CLASSES, digits=4))
    
    return embeddings, logits, probs, preds, labels, mola


def evaluate_tta(backbone, classifier, tiles_data, device,
                 baseline_logits: np.ndarray, labels: np.ndarray, mola: np.ndarray):
    """TTA: 4 augmentations, average logits."""
    print("\n" + "="*70)
    print("TTA — 4 augmentations (original + hflip + vflip + rot180)")
    print("="*70)
    
    images_pil = [Image.open(t["path"]) for t in tiles_data]
    
    # baseline_logits already has the original pass
    all_logits = [baseline_logits]
    
    t0 = time.time()
    for aug_name, aug_fn in TTA_AUGMENTS[1:]:  # Skip original (already computed)
        print(f"  Augmentation: {aug_name}...")
        aug_images = [aug_fn(img) for img in images_pil]
        aug_emb = extract_embeddings(backbone, aug_images, device)
        aug_logits = run_classifier(classifier, aug_emb, mola, device)
        all_logits.append(aug_logits)
    
    # Average logits
    avg_logits = np.mean(all_logits, axis=0)
    preds = np.argmax(avg_logits, axis=1)
    elapsed = time.time() - t0
    
    f1 = f1_score(labels, preds, average="macro")
    print(f"\nF1 (macro): {f1:.4f}  (+{elapsed:.1f}s for augmentations)")
    print(classification_report(labels, preds, target_names=CLASSES, digits=4))
    
    return avg_logits, preds


def evaluate_crf(tiles_data, probs: np.ndarray, labels: np.ndarray,
                 smoothing_weight: float = 1.5, n_iterations: int = 10):
    """CRF: Potts-model spatial smoothing on per-image tile grids."""
    print("\n" + "="*70)
    print(f"CRF — Potts smoothing (weight={smoothing_weight}, iters={n_iterations})")
    print("="*70)
    
    # Compatibility matrix: which classes are allowed to be neighbors
    # LDA↔LVF: very compatible (glacial flow transitions)
    # CCF↔OTHER: compatible (CCF can be scattered)
    # LDA↔CCF: somewhat compatible
    # LVF↔CCF: somewhat compatible
    compat = np.array([
        #  LDA   LVF   CCF   OTHER
        [1.0,  0.8,  0.3,  0.2],   # LDA
        [0.8,  1.0,  0.4,  0.2],   # LVF
        [0.3,  0.4,  1.0,  0.5],   # CCF
        [0.2,  0.2,  0.5,  1.0],   # OTHER
    ], dtype=np.float32)
    
    # Group tiles by image
    by_image: dict[str, list[int]] = defaultdict(list)
    for i, t in enumerate(tiles_data):
        by_image[t["image_id"]].append(i)
    
    smoothed_probs = probs.copy()
    n_images_smoothed = 0
    
    for img_id, tile_indices in by_image.items():
        if len(tile_indices) < 2:
            continue  # Can't smooth a single tile
        
        # Build grid
        rows = [tiles_data[i]["tile_row"] for i in tile_indices]
        cols = [tiles_data[i]["tile_col"] for i in tile_indices]
        
        # Create adjacency (4-connected grid)
        pos_to_idx = {}
        for local_i, global_i in enumerate(tile_indices):
            pos_to_idx[(rows[local_i], cols[local_i])] = local_i
        
        local_probs = probs[tile_indices].copy()
        n_tiles = len(tile_indices)
        
        # Mean-field CRF iteration
        for iteration in range(n_iterations):
            new_probs = np.log(local_probs + 1e-10)  # Unary potential (log-prob)
            
            # Pairwise potential: for each tile, aggregate neighbor messages
            for local_i, global_i in enumerate(tile_indices):
                r, c = rows[local_i], cols[local_i]
                neighbor_msg = np.zeros(len(CLASSES), dtype=np.float32)
                n_neighbors = 0
                
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) in pos_to_idx:
                        nidx = pos_to_idx[(nr, nc)]
                        # Message = compatibility * neighbor's current belief
                        neighbor_msg += compat @ local_probs[nidx]
                        n_neighbors += 1
                
                if n_neighbors > 0:
                    new_probs[local_i] += smoothing_weight * neighbor_msg / n_neighbors
            
            # Normalize
            local_probs = _softmax(new_probs)
        
        smoothed_probs[tile_indices] = local_probs
        n_images_smoothed += 1
    
    preds = np.argmax(smoothed_probs, axis=1)
    f1 = f1_score(labels, preds, average="macro")
    print(f"\nImages smoothed: {n_images_smoothed}")
    print(f"F1 (macro): {f1:.4f}")
    print(classification_report(labels, preds, target_names=CLASSES, digits=4))
    
    return smoothed_probs, preds


def evaluate_tta_plus_crf(tta_logits, tiles_data, labels,
                          smoothing_weight=1.5, n_iterations=10):
    """TTA + CRF combined."""
    print("\n" + "="*70)
    print("TTA + CRF — Combined")
    print("="*70)
    
    tta_probs = _softmax(tta_logits)
    smoothed_probs, preds = evaluate_crf(
        tiles_data, tta_probs, labels,
        smoothing_weight=smoothing_weight,
        n_iterations=n_iterations,
    )
    return smoothed_probs, preds


# ── Helpers ───────────────────────────────────────────────────────────────────

def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    exp_x = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tiles", type=int, default=2000,
                        help="Max test tiles (stratified sample). 0 = all.")
    parser.add_argument("--crf-weight", type=float, default=1.5)
    parser.add_argument("--crf-iters", type=int, default=10)
    parser.add_argument("--skip-tta", action="store_true",
                        help="Skip TTA evaluation (confirmed useless for DINOv2).")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load data
    max_t = args.max_tiles if args.max_tiles > 0 else None
    print(f"\nLoading test data (max_tiles={max_t})...")
    tiles_data = load_test_data(max_tiles=max_t)
    label_counts = Counter(t["label"] for t in tiles_data)
    print(f"Loaded {len(tiles_data)} tiles: {dict(label_counts)}")
    
    # Group by image for CRF stats
    by_img = defaultdict(int)
    for t in tiles_data:
        by_img[t["image_id"]] += 1
    print(f"Across {len(by_img)} images (avg {np.mean(list(by_img.values())):.1f} tiles/image)")
    
    # Load models
    backbone, classifier = load_models(device)
    
    # ── 1. Baseline ──
    embeddings, logits, probs, preds, labels, mola = evaluate_baseline(
        backbone, classifier, tiles_data, device
    )
    baseline_f1 = f1_score(labels, preds, average="macro")
    
    # ── 2. TTA (skip if --skip-tta) ──
    if args.skip_tta:
        tta_logits, tta_preds = logits, preds
        tta_f1 = baseline_f1
        print("\n[TTA skipped via --skip-tta]")
    else:
        tta_logits, tta_preds = evaluate_tta(
            backbone, classifier, tiles_data, device, logits, labels, mola
        )
        tta_f1 = f1_score(labels, tta_preds, average="macro")
    
    # ── 3. CRF on baseline ──
    crf_probs, crf_preds = evaluate_crf(
        tiles_data, probs, labels,
        smoothing_weight=args.crf_weight, n_iterations=args.crf_iters,
    )
    crf_f1 = f1_score(labels, crf_preds, average="macro")
    
    # ── 4. TTA + CRF (skip if --skip-tta) ──
    if args.skip_tta:
        combined_f1 = crf_f1
        print("\n[TTA+CRF skipped via --skip-tta]")
    else:
        tta_probs = _softmax(tta_logits)
        _, combined_preds = evaluate_crf(
            tiles_data, tta_probs, labels,
            smoothing_weight=args.crf_weight, n_iterations=args.crf_iters,
        )
        combined_f1 = f1_score(labels, combined_preds, average="macro")
    
    # ── Summary ──
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Method':<25} {'F1':>8} {'Δ vs baseline':>15} {'Retrain?':>10}")
    print("-"*60)
    print(f"{'Baseline (V4b FiLM)':<25} {baseline_f1:>8.4f} {'—':>15} {'No':>10}")
    print(f"{'+ TTA (4x augment)':<25} {tta_f1:>8.4f} {tta_f1-baseline_f1:>+15.4f} {'No':>10}")
    print(f"{'+ CRF (spatial smooth)':<25} {crf_f1:>8.4f} {crf_f1-baseline_f1:>+15.4f} {'No':>10}")
    print(f"{'+ TTA + CRF':<25} {combined_f1:>8.4f} {combined_f1-baseline_f1:>+15.4f} {'No':>10}")
    print("="*70)


if __name__ == "__main__":
    main()
