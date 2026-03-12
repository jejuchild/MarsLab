#!/usr/bin/env python3
"""
V5 Retraining — Fix pixel_scale_m by using PDS extent for tile coordinates.

Phases:
  1. Fetch PDS extents for all training images (parallel HTTP)
  2. Recompute tile lat/lon from PDS extent + image dimensions
  3. Re-extract MOLA features at corrected coordinates
  4. Pre-compute DINOv2+LoRA embeddings (CPU, one-time)
  5. Train FiLM classifier head with new MOLA features

Usage:
  nohup python3 retrain_v5_fixed_mola.py > /tmp/retrain_v5.log 2>&1 &
  # Resume from a phase:
  python3 retrain_v5_fixed_mola.py --start-phase 3
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
import requests
from PIL import Image

# ── Paths ──
ROOT = Path("/disk1/cspark/MarsLab")
DATA_DIR = ROOT / "Data" / "HiRISE"
BROWSE_DIR = DATA_DIR / "midlat_browse"
V4_DIR = DATA_DIR / "v4_colab_data_expanded"
OUTPUT_DIR = DATA_DIR / "v5_retrain"
MOLA_DEM_PATH = ROOT / "Mars_HRSC_MOLA_BlendDEM_Global_200mp_v2.tif"

# Add MarsLab to path
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/tmp/retrain_v5.log"),
    ],
)
logger = logging.getLogger(__name__)

TILE_SIZE = 224
MARS_RADIUS_M = 3389500.0
DEM_RESOLUTION_M = 200.0


# ============================================================================
# Phase 1: Fetch PDS extents
# ============================================================================

def fetch_pds_extent(product_id: str, max_retries: int = 3) -> Optional[dict]:
    """Fetch lat/lon extent from Arizona PDS LBL for one image (with retries)."""
    base_id = re.sub(r"_(RED|COLOR|BG|IR)$", "", product_id)
    parts = base_id.split("_")
    if len(parts) < 3:
        return None
    try:
        orbit_num = int(parts[1])
    except ValueError:
        return None

    prefix = parts[0]
    orbit_base = (orbit_num // 100) * 100
    orbit_dir = f"ORB_{orbit_base:06d}_{orbit_base + 99:06d}"
    lbl_url = (
        f"https://hirise-pds.lpl.arizona.edu/PDS/RDR/{prefix}/"
        f"{orbit_dir}/{base_id}/{base_id}_RED.LBL"
    )

    for attempt in range(max_retries):
        try:
            resp = requests.get(lbl_url, timeout=30)
            if resp.status_code == 200:
                return _parse_extent(resp.text, base_id)
            if resp.status_code == 404:
                return None  # No point retrying 404
            # Server error (5xx) — retry
        except (requests.Timeout, requests.ConnectionError):
            pass  # Retry on network errors
        except Exception:
            return None  # Unexpected error — don't retry
        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s
    return None

def _parse_extent(lbl_text: str, product_id: str) -> Optional[dict]:
    """Parse lat/lon bounds from PDS LBL text."""
    vals = {}
    mapping = {
        "MINIMUM_LATITUDE": "lat_min",
        "MAXIMUM_LATITUDE": "lat_max",
        "WESTERNMOST_LONGITUDE": "lon_min",
        "EASTERNMOST_LONGITUDE": "lon_max",
    }
    for line in lbl_text.splitlines():
        stripped = line.strip()
        for pds_key, name in mapping.items():
            if pds_key in stripped:
                m = re.search(r"=\s*([-\d.]+)", stripped)
                if m:
                    vals[name] = float(m.group(1))
    if len(vals) == 4:
        vals["product_id"] = product_id
        return vals
    return None


def phase1_fetch_extents(image_ids: list[str], cache_path: Path, workers: int = 4) -> dict:
    """Fetch PDS extents for all images. Returns {image_id: extent_dict}."""
    if cache_path.exists():
        logger.info(f"Phase 1: Loading cached extents from {cache_path}")
        with open(cache_path) as f:
            cached = json.load(f)
        logger.info(f"  Cached: {len(cached)} extents")
        missing = [img for img in image_ids if img not in cached]
        if not missing:
            return cached
        logger.info(f"  Missing: {len(missing)} — fetching...")
    else:
        cached = {}
        missing = image_ids

    results = dict(cached)
    failed = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_pds_extent, img): img for img in missing}
        for i, future in enumerate(as_completed(futures), 1):
            img = futures[future]
            try:
                extent = future.result()
                if extent:
                    results[img] = extent
                else:
                    failed += 1
            except Exception:
                failed += 1

            if i % 200 == 0 or i == len(missing):
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else 0
                eta = (len(missing) - i) / rate if rate > 0 else 0
                logger.info(
                    f"  PDS fetch: {i}/{len(missing)} "
                    f"(ok={len(results)-len(cached)}, fail={failed}) "
                    f"[{rate:.0f}/s, ETA {eta/60:.0f}m]"
                )

            # Incremental save every 500 fetches
            if i % 500 == 0:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, "w") as f:
                    json.dump(results, f, indent=2)
                logger.info(f"  Checkpoint saved: {len(results)} extents")

    # Final save
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Phase 1 done: {len(results)} extents saved ({failed} failed)")
    return results


# ============================================================================
# Phase 2: Recompute tile coordinates from PDS extent
# ============================================================================

def compute_tile_coord_from_extent(
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
    img_w: int, img_h: int,
    tile_row: int, tile_col: int,
) -> tuple[float, float]:
    """Compute tile center lat/lon from actual PDS image extent."""
    tile_center_row = tile_row * TILE_SIZE + TILE_SIZE / 2
    tile_center_col = tile_col * TILE_SIZE + TILE_SIZE / 2

    # Linear interpolation: pixel -> geographic
    # Row 0 = top = lat_max (north), last row = lat_min (south)
    lat = lat_max - (tile_center_row / img_h) * (lat_max - lat_min)
    # Col 0 = left = lon_min (west), last col = lon_max (east)
    lon = lon_min + (tile_center_col / img_w) * (lon_max - lon_min)
    return float(lat), float(lon)


# ============================================================================
# Phase 3: Re-extract MOLA features at corrected coordinates
# ============================================================================

def phase3_extract_mola(
    tile_index: dict,
    extents: dict,
    mola_path: str,
    cache_path: Path,
) -> dict:
    """Re-extract MOLA features for all tiles using corrected coordinates."""
    if cache_path.exists():
        logger.info(f"Phase 3: Loading cached MOLA features from {cache_path}")
        data = np.load(str(cache_path), allow_pickle=True).item()
        logger.info(f"  Cached: {len(data)} images")
        return data

    import rasterio

    # Group tiles by image
    tiles_by_image: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for tile_key in tile_index:
        parts = tile_key.rsplit("_", 2)
        image_id = parts[0]
        row, col = int(parts[1]), int(parts[2])
        tiles_by_image[image_id].append((row, col))

    logger.info(f"Phase 3: Extracting MOLA for {len(tiles_by_image)} images, {len(tile_index)} tiles")

    # Load DEM once
    ds = rasterio.open(mola_path)

    # Load MOLA extraction functions
    from scripts.marslandform_v2.data.mola import (
        extract_features_at_scale,
        extract_window,
    )

    mola_features: dict[str, dict[str, np.ndarray]] = {}
    skipped_images = 0
    t0 = time.time()

    for i, (image_id, tile_coords) in enumerate(tiles_by_image.items()):
        extent = extents.get(image_id)
        if extent is None:
            skipped_images += 1
            # Fallback: use old coordinates (center-based)
            mola_features[image_id] = {}
            for row, col in tile_coords:
                mola_features[image_id][f"{row}_{col}"] = np.zeros(25, dtype=np.float32)
            continue

        lat_min = extent["lat_min"]
        lat_max = extent["lat_max"]
        lon_min = extent["lon_min"]
        lon_max = extent["lon_max"]

        # Get image dimensions
        browse_path = _find_browse(image_id)
        if browse_path is None:
            skipped_images += 1
            mola_features[image_id] = {}
            for row, col in tile_coords:
                mola_features[image_id][f"{row}_{col}"] = np.zeros(25, dtype=np.float32)
            continue

        img = Image.open(browse_path)
        img_w, img_h = img.size

        image_mola: dict[str, np.ndarray] = {}
        center_lat = (lat_min + lat_max) / 2
        center_lon = (lon_min + lon_max) / 2

        # Collect all tile features for relative computation
        all_feats = []
        tile_keys = []

        for row, col in tile_coords:
            t_lat, t_lon = compute_tile_coord_from_extent(
                lat_min, lat_max, lon_min, lon_max, img_w, img_h, row, col,
            )

            # Extract 23 base features (7 × 3 scales + 2 global)
            features = []
            for scale_km in [1.0, 5.0, 20.0]:
                sf = extract_features_at_scale(ds, t_lat, t_lon, scale_km)
                features.extend([
                    sf["slope_mean"], sf["slope_std"], sf["curvature_mean"],
                    sf["TPI"], sf["TRI"], sf["roughness"], sf["lobateness"],
                ])
            # Global: elevation + abs_latitude
            w1km = extract_window(ds, t_lat, t_lon, 1.0)
            elev = float(np.nanmean(w1km)) if w1km.size > 0 else 0.0
            features.append(elev)
            features.append(abs(t_lat))

            feat_arr = np.array(features, dtype=np.float32)  # (23,)
            all_feats.append(feat_arr)
            tile_keys.append(f"{row}_{col}")

        # Compute relative features (25-dim = 23 base + 2 relative)
        if all_feats:
            all_feats_arr = np.stack(all_feats)  # (N, 23)
            mean_elev = float(np.mean(all_feats_arr[:, 21]))
            mean_slope = float(np.mean(all_feats_arr[:, 0]))
            for j, key in enumerate(tile_keys):
                full_feat = np.zeros(25, dtype=np.float32)
                full_feat[:23] = all_feats_arr[j]
                full_feat[23] = all_feats_arr[j, 21] - mean_elev   # relative elevation
                full_feat[24] = all_feats_arr[j, 0] - mean_slope   # relative slope
                image_mola[key] = full_feat

        mola_features[image_id] = image_mola

        if (i + 1) % 100 == 0 or i + 1 == len(tiles_by_image):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(tiles_by_image) - i - 1) / rate if rate > 0 else 0
            logger.info(
                f"  MOLA extract: {i+1}/{len(tiles_by_image)} images "
                f"[{rate:.1f}/s, ETA {eta/60:.0f}m, skipped={skipped_images}]"
            )

    ds.close()

    # Save
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(cache_path), mola_features)
    logger.info(f"Phase 3 done: {len(mola_features)} images, saved to {cache_path}")
    return mola_features


def _find_browse(image_id: str) -> Optional[Path]:
    """Find browse image file."""
    for pattern in [
        f"{image_id}_RED.abrowse.jpg",
        f"{image_id}.jpg",
        f"{image_id}_RED.browse.jpg",
    ]:
        p = BROWSE_DIR / pattern
        if p.exists():
            return p
    matches = list(BROWSE_DIR.glob(f"{image_id}*"))
    return matches[0] if matches else None


# ============================================================================
# Phase 4: Pre-compute DINOv2+LoRA embeddings
# ============================================================================

def phase4_embeddings(
    tile_index: dict,
    ssl_weights_path: Path,
    cache_path: Path,
    batch_size: int = 32,
) -> np.ndarray:
    """Pre-compute DINOv2+LoRA CLS embeddings for all tiles."""
    if cache_path.exists():
        logger.info(f"Phase 4: Loading cached embeddings from {cache_path}")
        emb = np.load(str(cache_path))
        logger.info(f"  Shape: {emb.shape}")
        return emb

    import torch
    from torchvision import transforms

    logger.info("Phase 4: Computing DINOv2+LoRA embeddings on CPU")

    # Load backbone
    sys.path.insert(0, str(ROOT / "scripts"))
    from scripts.marslandform_v2.models.dinov2_lora import DinoV2LoRA
    from scripts.marslandform_v2.config import get_config

    cfg = get_config()
    device = torch.device("cpu")

    backbone = DinoV2LoRA(cfg.dinov2)
    if ssl_weights_path.exists():
        sd = torch.load(str(ssl_weights_path), map_location="cpu", weights_only=False)
        if "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        # Filter to LoRA keys only
        lora_keys = {k: v for k, v in sd.items() if "lora" in k.lower()}
        if lora_keys:
            backbone.load_state_dict(lora_keys, strict=False)
            logger.info(f"  Loaded {len(lora_keys)} LoRA keys from SSL weights")
        else:
            backbone.load_state_dict(sd, strict=False)
            logger.info(f"  Loaded {len(sd)} keys from SSL weights")

    backbone.eval()
    backbone.to(device)

    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Process tiles in order
    tile_keys = list(tile_index.keys())
    n = len(tile_keys)
    embeddings = np.zeros((n, 768), dtype=np.float32)

    t0 = time.time()
    with torch.no_grad():
        for batch_start in range(0, n, batch_size):
            batch_end = min(batch_start + batch_size, n)
            batch_imgs = []

            for idx in range(batch_start, batch_end):
                tile_path = V4_DIR / tile_index[tile_keys[idx]]
                try:
                    img = Image.open(tile_path).convert("RGB")
                    batch_imgs.append(transform(img))
                except Exception:
                    batch_imgs.append(torch.zeros(3, 224, 224))

            batch_tensor = torch.stack(batch_imgs).to(device)
            out = backbone(batch_tensor)  # (B, 768)
            embeddings[batch_start:batch_end] = out.numpy()

            if (batch_start // batch_size + 1) % 50 == 0:
                elapsed = time.time() - t0
                done = batch_end
                rate = done / elapsed
                eta = (n - done) / rate if rate > 0 else 0
                logger.info(
                    f"  Embeddings: {done}/{n} tiles "
                    f"[{rate:.0f} tiles/s, ETA {eta/60:.0f}m]"
                )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(cache_path), embeddings)
    elapsed = time.time() - t0
    logger.info(f"Phase 4 done: {embeddings.shape} embeddings in {elapsed/60:.1f}m")
    return embeddings


# ============================================================================
# Phase 5: Train FiLM classifier
# ============================================================================

def phase5_train(
    embeddings: np.ndarray,
    mola_features: dict,
    tile_index: dict,
    labels_path: Path,
    splits_path: Path,
    output_dir: Path,
    epochs: int = 100,
    patience: int = 15,
    batch_size: int = 256,
    lr: float = 1e-4,
):
    """Train FiLM classifier with pre-computed embeddings + new MOLA features."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    logger.info("Phase 5: Training FiLM classifier")

    # Load labels & splits
    with open(labels_path) as f:
        labels_raw = json.load(f)
    with open(splits_path) as f:
        splits = json.load(f)

    CLASS_NAMES = ["LDA", "LVF", "CCF", "OTHER"]
    class_to_idx = {c: i for i, c in enumerate(CLASS_NAMES)}
    tile_keys = list(tile_index.keys())

    # Build aligned arrays
    n = len(tile_keys)
    mola_arr = np.zeros((n, 25), dtype=np.float32)
    label_arr = np.full(n, -1, dtype=np.int64)

    for i, key in enumerate(tile_keys):
        # Parse image_id and row_col
        parts = key.rsplit("_", 2)
        image_id = parts[0]
        row_col = f"{parts[1]}_{parts[2]}"

        # MOLA features
        if image_id in mola_features and row_col in mola_features[image_id]:
            mola_arr[i] = mola_features[image_id][row_col]

        # Labels
        if key in labels_raw:
            label_info = labels_raw[key]
            if isinstance(label_info, dict):
                cls = label_info.get("class", label_info.get("label", "OTHER"))
            else:
                cls = str(label_info)
            label_arr[i] = class_to_idx.get(cls, class_to_idx["OTHER"])

    # Split indices
    train_idx = [i for i, k in enumerate(tile_keys) if k in set(splits.get("train", []))]
    val_idx = [i for i, k in enumerate(tile_keys) if k in set(splits.get("val", []))]
    test_idx = [i for i, k in enumerate(tile_keys) if k in set(splits.get("test", []))]

    logger.info(f"  Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # Filter out unlabeled
    train_idx = [i for i in train_idx if label_arr[i] >= 0]
    val_idx = [i for i in val_idx if label_arr[i] >= 0]
    test_idx = [i for i in test_idx if label_arr[i] >= 0]

    logger.info(f"  After filter: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # Build FiLM classifier
    from scripts.marslandform_v2.models.film_classifier import FiLMClassifier

    model = FiLMClassifier(
        visual_dim=768,
        mola_dim=25,
        num_classes=len(CLASS_NAMES),
        hidden_dim=256,
        dropout=0.3,
    )
    device = torch.device("cpu")
    model.to(device)

    # DataLoaders
    def make_loader(indices, shuffle=False):
        emb_t = torch.tensor(embeddings[indices], dtype=torch.float32)
        mola_t = torch.tensor(mola_arr[indices], dtype=torch.float32)
        lab_t = torch.tensor(label_arr[indices], dtype=torch.long)
        ds = TensorDataset(emb_t, mola_t, lab_t)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    train_loader = make_loader(train_idx, shuffle=True)
    val_loader = make_loader(val_idx)
    test_loader = make_loader(test_idx)

    # Class weights for focal loss
    train_labels = label_arr[train_idx]
    class_counts = np.bincount(train_labels, minlength=len(CLASS_NAMES)).astype(np.float32)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights /= class_weights.sum()
    weights_t = torch.tensor(class_weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights_t, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    best_f1 = 0.0
    best_epoch = 0
    no_improve = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for emb_b, mola_b, lab_b in train_loader:
            emb_b, mola_b, lab_b = emb_b.to(device), mola_b.to(device), lab_b.to(device)
            optimizer.zero_grad()
            logits = model(emb_b, mola_b)
            loss = criterion(logits, lab_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * emb_b.size(0)
        train_loss /= len(train_idx)
        scheduler.step()

        # Validate
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for emb_b, mola_b, lab_b in val_loader:
                logits = model(emb_b, mola_b)
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.numpy())
                all_labels.extend(lab_b.numpy())

        # Macro F1
        from sklearn.metrics import f1_score, accuracy_score
        val_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        val_acc = accuracy_score(all_labels, all_preds)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 5 == 0 or epoch == 1 or no_improve == 0:
            logger.info(
                f"  Epoch {epoch:3d}: loss={train_loss:.4f} "
                f"val_f1={val_f1:.4f} val_acc={val_acc:.4f} "
                f"{'*best*' if no_improve == 0 else ''}"
            )

        if no_improve >= patience:
            logger.info(f"  Early stopping at epoch {epoch} (best={best_epoch})")
            break

    # Load best and evaluate on test
    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for emb_b, mola_b, lab_b in test_loader:
            logits = model(emb_b, mola_b)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.numpy())
            all_labels.extend(lab_b.numpy())

    from sklearn.metrics import classification_report
    report = classification_report(all_labels, all_preds, target_names=CLASS_NAMES, digits=4)
    test_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    test_acc = accuracy_score(all_labels, all_preds)

    logger.info(f"\n{'='*60}")
    logger.info(f"TRAINING COMPLETE — best epoch {best_epoch}")
    logger.info(f"  Val  F1={best_f1:.4f}")
    logger.info(f"  Test F1={test_f1:.4f}, Acc={test_acc:.4f}")
    logger.info(f"\n{report}")

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    import torch as _torch
    _torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": best_epoch,
            "val_f1": best_f1,
            "test_f1": test_f1,
            "test_acc": test_acc,
            "cfg": {
                "visual_dim": 768,
                "mola_dim": 25,
                "num_classes": len(CLASS_NAMES),
                "hidden_dim": 256,
                "dropout": 0.3,
                "class_names": CLASS_NAMES,
                "pixel_scale_fix": "pds_extent",
            },
        },
        output_dir / "film_classifier_v5.pt",
    )
    logger.info(f"  Model saved: {output_dir / 'film_classifier_v5.pt'}")

    # Save report
    with open(output_dir / "test_report.txt", "w") as f:
        f.write(f"Best epoch: {best_epoch}\n")
        f.write(f"Val F1: {best_f1:.4f}\n")
        f.write(f"Test F1: {test_f1:.4f}\n")
        f.write(f"Test Acc: {test_acc:.4f}\n\n")
        f.write(report)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-phase", type=int, default=1, help="Start from phase N")
    parser.add_argument("--workers", type=int, default=4, help="HTTP fetch parallelism (default: 4)")
    parser.add_argument("--batch-size", type=int, default=32, help="DINOv2 batch size")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load tile index
    with open(V4_DIR / "tile_index.json") as f:
        tile_index = json.load(f)
    logger.info(f"Tile index: {len(tile_index)} tiles")

    # Unique images
    image_ids = sorted(set(k.rsplit("_", 2)[0] for k in tile_index))
    logger.info(f"Unique images: {len(image_ids)}")

    # ── Phase 1: PDS extents ──
    if args.start_phase <= 1:
        logger.info(f"\n{'='*60}\nPHASE 1: Fetch PDS extents\n{'='*60}")
        extents = phase1_fetch_extents(
            image_ids,
            OUTPUT_DIR / "pds_extents.json",
            workers=args.workers,
        )
    else:
        with open(OUTPUT_DIR / "pds_extents.json") as f:
            extents = json.load(f)
        logger.info(f"Phase 1 skipped: loaded {len(extents)} cached extents")

    # ── Phase 2 is merged into Phase 3 ──

    # ── Phase 3: MOLA re-extraction ──
    if args.start_phase <= 3:
        logger.info(f"\n{'='*60}\nPHASE 3: Re-extract MOLA features\n{'='*60}")
        mola = phase3_extract_mola(
            tile_index, extents, str(MOLA_DEM_PATH),
            OUTPUT_DIR / "mola_features_v5.npy",
        )
    else:
        mola = np.load(str(OUTPUT_DIR / "mola_features_v5.npy"), allow_pickle=True).item()
        logger.info(f"Phase 3 skipped: loaded {len(mola)} cached MOLA features")

    # ── Phase 4: DINOv2 embeddings ──
    if args.start_phase <= 4:
        logger.info(f"\n{'='*60}\nPHASE 4: DINOv2+LoRA embeddings\n{'='*60}")
        embeddings = phase4_embeddings(
            tile_index,
            V4_DIR / "ssl_lora_weights.pt",
            OUTPUT_DIR / "embeddings_v5.npy",
            batch_size=args.batch_size,
        )
    else:
        embeddings = np.load(str(OUTPUT_DIR / "embeddings_v5.npy"))
        logger.info(f"Phase 4 skipped: loaded {embeddings.shape} cached embeddings")

    # ── Phase 5: Train FiLM ──
    logger.info(f"\n{'='*60}\nPHASE 5: Train FiLM classifier\n{'='*60}")
    phase5_train(
        embeddings=embeddings,
        mola_features=mola,
        tile_index=tile_index,
        labels_path=V4_DIR / "tile_labels_v4.json",
        splits_path=V4_DIR / "tile_splits_v4.json",
        output_dir=OUTPUT_DIR,
    )

    logger.info(f"\n{'='*60}\nALL DONE — outputs in {OUTPUT_DIR}\n{'='*60}")


if __name__ == "__main__":
    main()
