#!/usr/bin/env python3
"""
Evaluate V8 Segmentation:
  1. Patch-level metrics (already done in training, reproduced here)
  2. Tile-level comparison with V6b (aggregate patch predictions → tile label)
  3. Polygon contour extraction & Levy IoU evaluation

Usage:
  python -u evaluate_v8_segmentation.py
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/disk1/cspark/hirise-api")
from scripts.marslandform_v2.models.seg_head import PatchSegmentationHead

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path("/disk1/cspark/MarsLab")
DATA_DIR = ROOT / "Data/HiRISE/v8_segmentation"
PATCH_TOKENS_DIR = DATA_DIR / "patch_tokens_v8"
PATCH_LABELS_PATH = DATA_DIR / "patch_labels_v8.npy"
MODEL_PATH = DATA_DIR / "seg_head_v8.pt"
TILE_LABELS_PATH = ROOT / "Data/HiRISE/v5_retrain/tile_labels_v5.json"
LEVY_SHP = Path("/disk1/cspark/hirise-api/data/external_datasets/levy_2014_glacial/extracted/AreaIceUnits.shp")
OUT_DIR = DATA_DIR

CLASS_NAMES = ["LDA", "LVF", "CCF", "OTHER"]
NUM_CLASSES = 4
UNLABELED = 255
EMBED_DIM = 1024
PATCHES_PER_SIDE = 16
TILE_SIZE_PX = 224
BROWSE_SCALE_M_PER_PX = 25.0
PATCH_SIZE_M = (TILE_SIZE_PX / PATCHES_PER_SIDE) * BROWSE_SCALE_M_PER_PX  # 350m


def split_by_product(all_pids: list[str], val_ratio: float = 0.15, seed: int = 42):
    rng = np.random.RandomState(seed)
    unique_pids = sorted(set(all_pids))
    rng.shuffle(unique_pids)
    n_val = max(1, int(len(unique_pids) * val_ratio))
    val_pids = set(unique_pids[:n_val])
    train_pids = [p for p in unique_pids if p not in val_pids]
    return train_pids, list(val_pids)


def predict_all_val(model, val_pids, patch_labels, tokens_dir, device):
    """Run V8 model on all val tiles, return predictions + labels per product."""
    model.eval()
    results = {}  # pid → {tile_key: {"pred": (16,16), "label": (16,16)}}

    for i, pid in enumerate(val_pids):
        token_file = tokens_dir / f"{pid}.npy"
        if not token_file.exists() or pid not in patch_labels:
            continue

        token_dict = np.load(token_file, allow_pickle=True).item()
        label_dict = patch_labels[pid]
        pid_results = {}

        # Batch all tiles for this product
        keys = [k for k in label_dict if k in token_dict]
        if not keys:
            continue

        tokens_list = [token_dict[k] for k in keys]
        tokens_batch = torch.from_numpy(np.stack(tokens_list)).float().to(device)

        with torch.no_grad():
            logits = model(tokens_batch)  # (N, 4, 16, 16)
            preds = logits.argmax(1).cpu().numpy()  # (N, 16, 16)

        for j, k in enumerate(keys):
            pid_results[k] = {
                "pred": preds[j],       # (16, 16)
                "label": label_dict[k],  # (16, 16)
            }

        results[pid] = pid_results

        if (i + 1) % 200 == 0:
            print(f"  Predicted {i + 1}/{len(val_pids)} products")

    return results


def patch_to_tile_label(patch_preds: np.ndarray) -> int:
    """Aggregate 16x16 patch predictions to a single tile label via majority vote (excluding OTHER)."""
    glacial_mask = patch_preds != 3  # not OTHER
    if glacial_mask.sum() == 0:
        return 3  # all OTHER
    glacial_preds = patch_preds[glacial_mask]
    counts = np.bincount(glacial_preds, minlength=NUM_CLASSES)[:3]  # LDA, LVF, CCF
    if counts.sum() < patch_preds.size * 0.1:  # less than 10% glacial
        return 3
    return int(counts.argmax())


def evaluate_patch_level(results: dict):
    """Compute patch-level metrics."""
    all_preds = []
    all_labels = []
    for pid, tiles in results.items():
        for tk, data in tiles.items():
            all_preds.append(data["pred"].flatten())
            all_labels.append(data["label"].flatten())

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    mask = labels != UNLABELED

    p = preds[mask]
    l = labels[mask]

    acc = float((p == l).mean())
    print(f"\n{'='*60}")
    print(f"PATCH-LEVEL METRICS (val set)")
    print(f"{'='*60}")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Total patches: {len(p):,}")

    print(f"\n  {'Class':>8} | {'Precision':>9} | {'Recall':>9} | {'F1':>9} | {'Support':>9}")
    print(f"  {'-'*55}")
    f1s = []
    for c in range(NUM_CLASSES):
        tp = ((p == c) & (l == c)).sum()
        fp = ((p == c) & (l != c)).sum()
        fn = ((p != c) & (l == c)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        f1s.append(f1)
        support = ((l == c)).sum()
        print(f"  {CLASS_NAMES[c]:>8} | {prec:>9.4f} | {rec:>9.4f} | {f1:>9.4f} | {support:>9,}")

    macro_f1 = float(np.mean(f1s))
    print(f"\n  Macro F1: {macro_f1:.4f}")

    # FP rate
    other_mask = l == 3
    other_as_glacial = ((p != 3) & other_mask).sum()
    other_total = other_mask.sum()
    print(f"  FP rate (OTHER→glacial): {other_as_glacial}/{other_total} = {other_as_glacial/max(other_total,1)*100:.2f}%")

    return {"accuracy": acc, "macro_f1": macro_f1, "per_class_f1": f1s}


def evaluate_tile_level(results: dict, tile_labels_lookup: dict):
    """Aggregate V8 patch predictions to tile-level and compare with ground truth tile labels."""
    # tile_labels_lookup: {(pid, row_col) → tile_type} from tile_labels_v5.json
    v8_tile_preds = []
    v8_tile_trues = []

    for pid, tiles in results.items():
        for tk, data in tiles.items():
            pred_tile = patch_to_tile_label(data["pred"])
            # Ground truth: majority of non-UNLABELED patch labels
            label_flat = data["label"].flatten()
            labeled_mask = label_flat != UNLABELED
            if labeled_mask.sum() == 0:
                continue
            true_vals = label_flat[labeled_mask]
            counts = np.bincount(true_vals, minlength=NUM_CLASSES)
            true_tile = int(counts.argmax())

            v8_tile_preds.append(pred_tile)
            v8_tile_trues.append(true_tile)

    preds = np.array(v8_tile_preds)
    trues = np.array(v8_tile_trues)
    acc = float((preds == trues).mean())

    print(f"\n{'='*60}")
    print(f"TILE-LEVEL METRICS (V8 aggregated, val set)")
    print(f"{'='*60}")
    print(f"  Tiles: {len(preds):,}")
    print(f"  Accuracy: {acc:.4f}")

    print(f"\n  {'Class':>8} | {'Precision':>9} | {'Recall':>9} | {'F1':>9} | {'Support':>9}")
    print(f"  {'-'*55}")
    f1s = []
    for c in range(NUM_CLASSES):
        tp = ((preds == c) & (trues == c)).sum()
        fp = ((preds == c) & (trues != c)).sum()
        fn = ((preds != c) & (trues == c)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        f1s.append(f1)
        support = (trues == c).sum()
        print(f"  {CLASS_NAMES[c]:>8} | {prec:>9.4f} | {rec:>9.4f} | {f1:>9.4f} | {support:>9,}")

    macro_f1 = float(np.mean(f1s))
    print(f"\n  Macro F1: {macro_f1:.4f}")
    print(f"\n  V6b reference: val F1=0.697, val Acc=0.815 (tile-level)")

    return {"accuracy": acc, "macro_f1": macro_f1, "per_class_f1": f1s}


def extract_polygons_and_levy_iou(results: dict, tile_meta: dict):
    """
    Extract polygon contours from V8 predictions and compare with Levy polygons.

    For each val product:
    1. Assemble full prediction map from tiles
    2. Extract connected components per glacial class
    3. Convert to geographic polygons (lon/lat)
    4. Compare with Levy reference polygons via IoU
    """
    try:
        import geopandas as gpd
        from shapely.geometry import box, MultiPolygon, Polygon
        from shapely.ops import unary_union
        from shapely.prepared import prep
    except ImportError:
        print("\n  WARNING: geopandas/shapely not available. Skipping polygon evaluation.")
        return None

    print(f"\n{'='*60}")
    print(f"POLYGON CONTOUR EXTRACTION & LEVY IoU")
    print(f"{'='*60}")

    # Load Levy polygons
    print("  Loading Levy shapefile...")
    levy = gpd.read_file(LEVY_SHP)
    levy_by_class = {}
    for cls_name in ["ccf", "lda", "lvf"]:
        polys = levy[levy["code"] == cls_name].geometry.values
        if len(polys) > 0:
            levy_by_class[cls_name] = unary_union(polys)
    levy_all = unary_union(levy.geometry.values)
    levy_all_prep = prep(levy_all)

    # tile_meta: {pid: {tile_key: {"lat": float, "lon": float}}}
    # Build geographic prediction map per product
    print("  Extracting prediction polygons from val products...")

    # For each val product, build prediction polygons
    v8_polygons_by_class = {0: [], 1: [], 2: []}  # LDA, LVF, CCF
    products_with_predictions = 0

    for pid, tiles in results.items():
        if pid not in tile_meta:
            continue

        for tk, data in tiles.items():
            if tk not in tile_meta[pid]:
                continue

            meta = tile_meta[pid][tk]
            lat, lon = meta["lat"], meta["lon"]
            pred = data["pred"]  # (16, 16)

            # For each patch, create a geographic box
            for pr in range(PATCHES_PER_SIDE):
                for pc in range(PATCHES_PER_SIDE):
                    cls = pred[pr, pc]
                    if cls >= 3:  # OTHER or UNLABELED
                        continue

                    # Geographic extent of this patch
                    # Patch (pr, pc): row=pr (top-down), col=pc (left-right)
                    # lat decreases downward, lon increases rightward
                    patch_lat_top = lat + (PATCHES_PER_SIDE / 2 - pr) * PATCH_SIZE_M / 59274.0  # m→deg lat
                    patch_lat_bot = lat + (PATCHES_PER_SIDE / 2 - pr - 1) * PATCH_SIZE_M / 59274.0
                    cos_lat = max(np.cos(np.radians(lat)), 0.01)
                    m_per_deg_lon = 59274.0 * cos_lat
                    patch_lon_left = lon + (pc - PATCHES_PER_SIDE / 2) * PATCH_SIZE_M / m_per_deg_lon
                    patch_lon_right = lon + (pc + 1 - PATCHES_PER_SIDE / 2) * PATCH_SIZE_M / m_per_deg_lon

                    patch_box = box(patch_lon_left, patch_lat_bot, patch_lon_right, patch_lat_top)
                    v8_polygons_by_class[cls].append(patch_box)

        products_with_predictions += 1

    print(f"  Products processed: {products_with_predictions}")
    for c in range(3):
        print(f"    {CLASS_NAMES[c]}: {len(v8_polygons_by_class[c]):,} patch polygons")

    # Merge adjacent patch polygons into contiguous regions
    print("  Merging patch polygons...")
    v8_merged = {}
    for c in range(3):
        if v8_polygons_by_class[c]:
            merged = unary_union(v8_polygons_by_class[c])
            v8_merged[c] = merged
            if hasattr(merged, 'geoms'):
                n_polys = len(list(merged.geoms))
            else:
                n_polys = 1
            print(f"    {CLASS_NAMES[c]}: {n_polys} merged polygons, area={merged.area:.4f} deg²")
        else:
            v8_merged[c] = MultiPolygon()
            print(f"    {CLASS_NAMES[c]}: 0 polygons")

    # Union all V8 glacial predictions
    all_v8_glacial_parts = [v8_merged[c] for c in range(3) if not v8_merged[c].is_empty]
    if not all_v8_glacial_parts:
        print("  No glacial predictions found.")
        return None
    v8_all_glacial = unary_union(all_v8_glacial_parts)

    # Compute IoU: V8 predicted glacial vs Levy all glacial
    intersection = v8_all_glacial.intersection(levy_all)
    union = v8_all_glacial.union(levy_all)
    iou_all = intersection.area / union.area if union.area > 0 else 0
    print(f"\n  Overall Glacial IoU (V8 vs Levy): {iou_all:.4f}")
    print(f"    V8 area: {v8_all_glacial.area:.4f} deg²")
    print(f"    Levy area: {levy_all.area:.4f} deg²")
    print(f"    Intersection: {intersection.area:.4f} deg²")

    # Per-class IoU
    cls_name_map = {0: "lda", 1: "lvf", 2: "ccf"}
    print(f"\n  Per-class IoU:")
    for c in range(3):
        cls_key = cls_name_map[c]
        if cls_key in levy_by_class and not v8_merged[c].is_empty:
            inter = v8_merged[c].intersection(levy_by_class[cls_key])
            uni = v8_merged[c].union(levy_by_class[cls_key])
            cls_iou = inter.area / uni.area if uni.area > 0 else 0
            print(f"    {CLASS_NAMES[c]}: IoU={cls_iou:.4f}")
        else:
            print(f"    {CLASS_NAMES[c]}: N/A")

    # Detection rate: what fraction of Levy polygons overlap with V8 predictions?
    print(f"\n  Levy polygon detection rate:")
    for cls_key_upper, cls_key in [("LDA", "lda"), ("LVF", "lvf"), ("CCF", "ccf")]:
        cls_polys = levy[levy["code"] == cls_key]
        if len(cls_polys) == 0:
            continue
        detected = 0
        for _, row in cls_polys.iterrows():
            poly = row.geometry
            if v8_all_glacial.intersects(poly):
                overlap = v8_all_glacial.intersection(poly).area / poly.area
                if overlap > 0.1:  # at least 10% overlap
                    detected += 1
        print(f"    {cls_key_upper}: {detected}/{len(cls_polys)} ({detected/len(cls_polys)*100:.1f}%)")

    return {
        "overall_iou": iou_all,
        "v8_area": v8_all_glacial.area,
        "levy_area": levy_all.area,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print("Loading V8 model...")
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model = PatchSegmentationHead(
        embed_dim=EMBED_DIM,
        num_classes=NUM_CLASSES,
        patches_per_side=PATCHES_PER_SIDE,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  Loaded epoch {ckpt['epoch']}, val_f1={ckpt['val_f1']:.4f}")

    # Load patch labels
    print("Loading patch labels...")
    patch_labels = np.load(PATCH_LABELS_PATH, allow_pickle=True).item()
    all_pids = sorted(patch_labels.keys())

    # Split (same as training)
    _, val_pids = split_by_product(all_pids)
    print(f"  Val products: {len(val_pids)}")

    # Load tile metadata (for geographic coordinates)
    print("Loading tile metadata...")
    with open(TILE_LABELS_PATH) as f:
        tile_list = json.load(f)

    tile_meta = defaultdict(dict)  # pid → {tile_key → {lat, lon}}
    for t in tile_list:
        pid = t["image_id"]
        tk = f"{t['tile_row']}_{t['tile_col']}"
        tile_meta[pid][tk] = {"lat": t["lat"], "lon": t["lon"]}

    # Run predictions on val set
    print("\nPredicting on val set...")
    t0 = time.time()
    results = predict_all_val(model, val_pids, patch_labels, PATCH_TOKENS_DIR, device)
    print(f"  Done in {time.time() - t0:.1f}s, {sum(len(v) for v in results.values())} tiles")

    # 1. Patch-level metrics
    patch_metrics = evaluate_patch_level(results)

    # 2. Tile-level metrics (aggregated, comparable to V6b)
    tile_metrics = evaluate_tile_level(results, tile_meta)

    # 3. Polygon contour extraction & Levy IoU
    poly_metrics = extract_polygons_and_levy_iou(results, tile_meta)

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  V8 Patch-level:  macro F1 = {patch_metrics['macro_f1']:.4f}")
    print(f"  V8 Tile-level:   macro F1 = {tile_metrics['macro_f1']:.4f}, acc = {tile_metrics['accuracy']:.4f}")
    print(f"  V6b Tile-level:  macro F1 = 0.6970, acc = 0.8154 (reference)")
    if poly_metrics:
        print(f"  Levy IoU:        {poly_metrics['overall_iou']:.4f}")

    # Save results
    summary = {
        "patch_level": patch_metrics,
        "tile_level": tile_metrics,
        "v6b_reference": {"macro_f1": 0.697, "accuracy": 0.815},
        "polygon_iou": poly_metrics,
    }
    out_path = OUT_DIR / "v8_evaluation_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
