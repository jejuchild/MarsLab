#!/usr/bin/env python3
"""Expand dataset: tile + embed browse images, pseudo-label with VLM+model ensemble.

Phase 2 of the accuracy improvement pipeline:
1. Select a subset of browse images (2000-3000) for embedding
2. Tile them (same as training pipeline)
3. Embed with frozen DINOv2 
4. Pseudo-label with best model + GroqVLM verification
5. Merge with cleaned labels from Phase 1
"""
import json
import os
import sys
import time
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_ROOT = ROOT / "Data/HiRISE/v2_output"
BROWSE_DIR = ROOT / "Data/HiRISE/midlat_browse"

# Target: embed N new images (CPU ~5 min each with tiling + DINOv2)
MAX_NEW_IMAGES = 2000  # ~7 days on CPU for full embedding... let's be realistic
BATCH_SIZE_EMBED = 50  # Process in batches of 50


def get_unlabeled_browse_ids(max_count: int = MAX_NEW_IMAGES) -> List[str]:
    """Get unlabeled browse image IDs."""
    labels = json.loads((DATA_ROOT / "labels_simple.json").read_text())
    labeled_ids = set(labels.keys())
    
    browse_files = sorted(BROWSE_DIR.glob("*.jpg"))
    all_browse_ids = []
    for f in browse_files:
        img_id = f.stem.replace("_RED.abrowse", "")
        if img_id not in labeled_ids:
            all_browse_ids.append(img_id)
    
    # Randomly sample
    random.seed(42)
    if len(all_browse_ids) > max_count:
        all_browse_ids = random.sample(all_browse_ids, max_count)
    
    return sorted(all_browse_ids)


def tile_browse_image(
    image_id: str,
    tile_size: int = 224,
    stride: int = 112,
    max_tiles: int = 128,
) -> Optional[np.ndarray]:
    """Tile a browse image into patches for DINOv2 embedding."""
    browse_path = BROWSE_DIR / f"{image_id}_RED.abrowse.jpg"
    if not browse_path.exists():
        return None
    
    try:
        img = Image.open(browse_path).convert("RGB")
        w, h = img.size
        
        if w < tile_size or h < tile_size:
            return None
        
        tiles = []
        for y in range(0, h - tile_size + 1, stride):
            for x in range(0, w - tile_size + 1, stride):
                tile = img.crop((x, y, x + tile_size, y + tile_size))
                tiles.append(np.array(tile))
        
        if not tiles:
            return None
        
        # Subsample if too many tiles
        if len(tiles) > max_tiles:
            indices = np.random.choice(len(tiles), max_tiles, replace=False)
            tiles = [tiles[i] for i in sorted(indices)]
        
        return np.stack(tiles)  # (N, 224, 224, 3)
    except Exception as e:
        print(f"  Error tiling {image_id}: {e}")
        return None


def embed_tiles_batch(tiles_batch: np.ndarray) -> np.ndarray:
    """Embed a batch of tiles using frozen DINOv2 ViT-B/14."""
    import torch
    
    # Load DINOv2
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", verbose=False)
    model.eval()
    
    # Normalize tiles
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    all_embeddings = []
    
    with torch.no_grad():
        for i in range(0, len(tiles_batch), 32):
            batch = tiles_batch[i:i+32].astype(np.float32) / 255.0
            batch = (batch - mean) / std
            batch = torch.tensor(batch).permute(0, 3, 1, 2)  # NCHW
            
            features = model(batch)
            all_embeddings.append(features.cpu().numpy())
    
    return np.concatenate(all_embeddings, axis=0)


def pseudo_label_with_model(
    image_id: str,
    embedding: np.ndarray,
    mola_features: np.ndarray,
    model_path: Path,
    cfg: Any,
) -> Dict[str, Any]:
    """Get pseudo-label from best MIL model."""
    import torch
    from copy import deepcopy
    from scripts.marslandform_v2.config import CLASS_ORDER
    from scripts.marslandform_v2.models.mil_classifier import AttentionMILClassifier
    
    device = torch.device("cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    saved_cfg = checkpoint.get("mil_config", {})
    model_cfg = deepcopy(cfg)
    for k, v in saved_cfg.items():
        if hasattr(model_cfg, k):
            setattr(model_cfg, k, v)
    
    model = AttentionMILClassifier(model_cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Prepare input
    tiles_t = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)  # (1, N, 768)
    mask = torch.ones(1, embedding.shape[0], dtype=torch.bool)
    mola_t = torch.tensor(mola_features, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        logits, att = model(tiles_t, mask, mola_t)
        probs = torch.softmax(logits, dim=1).squeeze().numpy()
    
    pred_cls = int(np.argmax(probs))
    confidence = float(probs[pred_cls])
    
    return {
        "image_id": image_id,
        "pred_label": CLASS_ORDER[pred_cls],
        "pred_idx": pred_cls,
        "confidence": confidence,
        "probabilities": probs.tolist(),
    }


def run_expand_dataset():
    """Main expansion pipeline. Can be run in stages."""
    print("=" * 60)
    print("DATASET EXPANSION PIPELINE")
    print("=" * 60)
    
    # Check what's already done
    expand_dir = DATA_ROOT / "expansion"
    expand_dir.mkdir(parents=True, exist_ok=True)
    
    progress_path = expand_dir / "progress.json"
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
    else:
        progress = {"embedded_ids": [], "pseudo_labels": {}, "stage": "init"}
    
    # Step 1: Select images to embed
    print("\nStep 1: Selecting unlabeled browse images...")
    unlabeled_ids = get_unlabeled_browse_ids(MAX_NEW_IMAGES)
    already_embedded = set(progress["embedded_ids"])
    remaining = [i for i in unlabeled_ids if i not in already_embedded]
    print(f"  Total unlabeled: {len(unlabeled_ids)}")
    print(f"  Already embedded: {len(already_embedded)}")
    print(f"  Remaining: {len(remaining)}")
    
    if not remaining:
        print("  All selected images already embedded!")
    else:
        # Step 2: Tile and embed in batches
        print(f"\nStep 2: Tiling + embedding {min(len(remaining), BATCH_SIZE_EMBED)} images...")
        print(f"  (Processing {BATCH_SIZE_EMBED} at a time, rerun to continue)")
        
        batch_to_process = remaining[:BATCH_SIZE_EMBED]
        new_embeddings = {}
        
        for idx, img_id in enumerate(batch_to_process):
            tiles = tile_browse_image(img_id)
            if tiles is None:
                print(f"  [{idx+1}/{len(batch_to_process)}] {img_id}: SKIP (too small or error)")
                continue
            
            print(f"  [{idx+1}/{len(batch_to_process)}] {img_id}: {tiles.shape[0]} tiles", end="")
            
            try:
                embeddings = embed_tiles_batch(tiles)
                new_embeddings[img_id] = embeddings  # (N, 768)
                progress["embedded_ids"].append(img_id)
                print(f" -> {embeddings.shape}")
            except Exception as e:
                print(f" -> ERROR: {e}")
        
        # Save embeddings
        if new_embeddings:
            emb_path = expand_dir / "new_embeddings.npy"
            if emb_path.exists():
                existing = np.load(emb_path, allow_pickle=True).item()
                existing.update(new_embeddings)
                np.save(emb_path, existing)
            else:
                np.save(emb_path, new_embeddings)
            print(f"\n  Saved {len(new_embeddings)} new embeddings to {emb_path}")
        
        progress_path.write_text(json.dumps(progress, indent=2))
    
    # Step 3: Pseudo-label embedded images
    emb_path = expand_dir / "new_embeddings.npy"
    if emb_path.exists():
        print(f"\nStep 3: Pseudo-labeling with best model...")
        new_emb = np.load(emb_path, allow_pickle=True).item()
        
        model_path = DATA_ROOT / "models/multihead_improved/best_mil_model.pt"
        cfg = __import__("scripts.marslandform_v2.config", fromlist=["get_config"]).get_config().mil
        
        # Load MOLA (use mean values for images without MOLA)
        mola_raw = np.load(DATA_ROOT / "mola_features_by_image.npy", allow_pickle=True).item()
        mean_mola = np.mean([np.asarray(v, dtype=np.float32) for v in mola_raw.values()], axis=0)
        
        pseudo_labels = {}
        for img_id in new_emb:
            if img_id in progress["pseudo_labels"]:
                continue
            
            mola = mola_raw.get(img_id, mean_mola)
            mola = np.asarray(mola, dtype=np.float32)
            
            result = pseudo_label_with_model(img_id, new_emb[img_id], mola, model_path, cfg)
            pseudo_labels[img_id] = result
        
        progress["pseudo_labels"].update({k: v for k, v in pseudo_labels.items()})
        progress_path.write_text(json.dumps(progress, indent=2, default=str))
        
        # Summary
        all_pl = progress["pseudo_labels"]
        print(f"  Total pseudo-labeled: {len(all_pl)}")
        if all_pl:
            conf_values = [v["confidence"] for v in all_pl.values()]
            print(f"  Confidence: mean={np.mean(conf_values):.3f}, median={np.median(conf_values):.3f}")
            
            high_conf = {k: v for k, v in all_pl.items() if v["confidence"] > 0.7}
            print(f"  High confidence (>0.7): {len(high_conf)}")
            
            label_dist = Counter(v["pred_label"] for v in all_pl.values())
            print(f"  Label distribution: {dict(label_dist)}")
            
            # Save high-confidence pseudo-labels as expanded label set
            expanded_labels_path = expand_dir / "pseudo_labels_highconf.json"
            expanded = {k: v["pred_label"] for k, v in high_conf.items()}
            expanded_labels_path.write_text(json.dumps(expanded, indent=2))
            print(f"\n  Saved {len(expanded)} high-confidence labels to {expanded_labels_path}")
    else:
        print("\n  No embeddings yet — run again after embedding completes")
    
    print(f"\n{'='*60}")
    print(f"To continue embedding more images, run this script again.")
    print(f"Progress saved to {progress_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_expand_dataset()
