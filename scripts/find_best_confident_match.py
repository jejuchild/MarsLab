#!/usr/bin/env python3
"""Find HiRISE image where v5c predictions are BOTH accurate AND high-confidence.

Problem: previous analysis only checked argmax accuracy.
In practice, the pipeline applies confidence thresholds — if softmax prob is low,
tiles get filtered out. LDA often has low confidence → disappears.

This script finds images where:
  1. Multiple landform classes present
  2. High accuracy
  3. HIGH CONFIDENCE on landform tiles (so they survive thresholding)
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("/disk1/cspark/hirise-api")
MARSLAB = Path("/disk1/cspark/MarsLab")
V5_DIR = MARSLAB / "Data/HiRISE/v5_retrain"
V4_DIR = MARSLAB / "Data/HiRISE/v4_colab_data_expanded"
BROWSE_DIR = MARSLAB / "Data/HiRISE/midlat_browse"

if not V5_DIR.exists():
    V5_DIR = ROOT / "data/HiRISE/v5_retrain"
if not V4_DIR.exists():
    V4_DIR = ROOT / "data/HiRISE/v4_colab_data_expanded"

sys.path.insert(0, str(ROOT))

CLASS_NAMES = ["LDA", "LVF", "CCF", "OTHER"]


def load_model(model_path: Path):
    from scripts.marslandform_v2.models.film_classifier import FiLMClassifier
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("cfg", {})
    model = FiLMClassifier(
        visual_dim=cfg.get("visual_dim", 768),
        mola_dim=cfg.get("mola_dim", 25),
        num_classes=cfg.get("num_classes", 4),
        film_hidden=cfg.get("film_hidden", 64),
        head_hidden=cfg.get("head_hidden", 128),
        dropout=cfg.get("dropout", 0.4),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def main():
    print("=" * 70)
    print("Finding best HIGH-CONFIDENCE multi-class Levy match (v5c)")
    print("=" * 70)

    # Load
    with open(V4_DIR / "tile_index.json") as f:
        tile_index = json.load(f)
    embeddings = np.load(str(V5_DIR / "embeddings_v5.npy"))
    mola = np.load(str(V5_DIR / "mola_features_v5.npy"), allow_pickle=True).item()
    with open(V5_DIR / "tile_labels_v5.json") as f:
        labels_list = json.load(f)
    model = load_model(V5_DIR / "film_classifier_v5c.pt")

    # Build arrays
    labels_raw = {}
    label_meta = {}
    for entry in labels_list:
        key = f"{entry['image_id']}_{entry['tile_row']}_{entry['tile_col']}"
        labels_raw[key] = entry.get("label", "OTHER")
        label_meta[key] = entry

    class_to_idx = {c: i for i, c in enumerate(CLASS_NAMES)}
    tile_keys = list(tile_index.keys())
    n = len(tile_keys)

    mola_arr = np.zeros((n, 25), dtype=np.float32)
    label_arr = np.full(n, -1, dtype=np.int64)
    tile_image_ids = []
    tile_rows_list = []
    tile_cols_list = []

    for i, key in enumerate(tile_keys):
        parts = key.rsplit("_", 2)
        image_id = parts[0]
        row_col = f"{parts[1]}_{parts[2]}"
        tile_image_ids.append(image_id)
        tile_rows_list.append(int(parts[1]))
        tile_cols_list.append(int(parts[2]))
        if image_id in mola and row_col in mola[image_id]:
            mola_arr[i] = mola[image_id][row_col]
        if key in labels_raw and labels_raw[key] in class_to_idx:
            label_arr[i] = class_to_idx[labels_raw[key]]

    labeled_indices = np.where(label_arr >= 0)[0]

    # Inference with SOFTMAX PROBABILITIES
    print("Running inference with softmax probabilities...")
    emb_t = torch.tensor(embeddings[labeled_indices], dtype=torch.float32)
    mola_t = torch.tensor(mola_arr[labeled_indices], dtype=torch.float32)

    all_preds = []
    all_probs = []
    all_max_conf = []

    with torch.no_grad():
        for start in range(0, len(labeled_indices), 512):
            end = min(start + 512, len(labeled_indices))
            logits = model(emb_t[start:end], mola_t[start:end])
            probs = F.softmax(logits, dim=1).numpy()
            preds = logits.argmax(dim=1).numpy()
            max_conf = probs.max(axis=1)

            all_preds.extend(preds)
            all_probs.extend(probs)
            all_max_conf.extend(max_conf)

    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_max_conf = np.array(all_max_conf)
    all_labels = label_arr[labeled_indices]

    # Show overall confidence stats
    print(f"\nOverall confidence stats:")
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        mask = all_labels == cls_idx
        if mask.sum() == 0:
            continue
        cls_conf = all_max_conf[mask & (all_preds == all_labels)]
        cls_conf_wrong = all_max_conf[mask & (all_preds != all_labels)]
        print(f"  {cls_name}: correct conf={np.mean(cls_conf):.3f}±{np.std(cls_conf):.3f} "
              f"(n={len(cls_conf)}) | wrong conf={np.mean(cls_conf_wrong):.3f}±{np.std(cls_conf_wrong):.3f} "
              f"(n={len(cls_conf_wrong)})")

    # Group by image
    image_results = defaultdict(list)
    for j, global_idx in enumerate(labeled_indices):
        img_id = tile_image_ids[global_idx]
        key = tile_keys[global_idx]
        meta = label_meta.get(key, {})
        image_results[img_id].append({
            "pred": int(all_preds[j]),
            "label": int(all_labels[j]),
            "confidence": float(all_max_conf[j]),
            "probs": all_probs[j].tolist(),
            "row": tile_rows_list[global_idx],
            "col": tile_cols_list[global_idx],
            "label_type": meta.get("label_type", "unknown"),
        })

    # Rank: multi-class, high confidence, high accuracy
    print("\nRanking images by confidence + accuracy...")

    CONF_THRESHOLD = 0.5  # Typical pipeline threshold

    rankings = []
    for img_id, tiles in image_results.items():
        browse_path = BROWSE_DIR / f"{img_id}_RED.abrowse.jpg"
        if not browse_path.exists():
            continue

        landform_tiles = [t for t in tiles if t["label"] != class_to_idx["OTHER"]]
        if len(landform_tiles) < 3:
            continue

        gt_classes = set(CLASS_NAMES[t["label"]] for t in landform_tiles)

        # Per-class analysis
        per_class = {}
        all_classes_confident = True
        for cls_name in gt_classes:
            cls_idx = class_to_idx[cls_name]
            cls_tiles = [t for t in tiles if t["label"] == cls_idx]
            cls_correct = [t for t in cls_tiles if t["pred"] == t["label"]]
            cls_confident = [t for t in cls_correct if t["confidence"] >= CONF_THRESHOLD]

            n_total = len(cls_tiles)
            n_correct = len(cls_correct)
            n_confident = len(cls_confident)
            mean_conf = np.mean([t["confidence"] for t in cls_correct]) if cls_correct else 0

            per_class[cls_name] = {
                "total": n_total,
                "correct": n_correct,
                "confident": n_confident,  # correct AND above threshold
                "mean_conf": mean_conf,
                "accuracy": n_correct / n_total if n_total > 0 else 0,
                "confident_ratio": n_confident / n_total if n_total > 0 else 0,
            }

            if n_confident < 1:
                all_classes_confident = False

        # Require at least 1 confident-correct tile per class
        if not all_classes_confident:
            continue

        # Compute overall metrics
        total = len(tiles)
        lf_correct = sum(1 for t in landform_tiles if t["pred"] == t["label"])
        lf_acc = lf_correct / len(landform_tiles)

        lf_confident_correct = sum(1 for t in landform_tiles
                                    if t["pred"] == t["label"] and t["confidence"] >= CONF_THRESHOLD)
        lf_mean_conf = np.mean([t["confidence"] for t in landform_tiles
                                 if t["pred"] == t["label"]])

        # Min class confidence (worst class must still be good)
        min_class_conf = min(pc["mean_conf"] for pc in per_class.values())
        min_class_confident_ratio = min(pc["confident_ratio"] for pc in per_class.values())

        label_dist = Counter(CLASS_NAMES[t["label"]] for t in tiles)
        pred_dist = Counter(CLASS_NAMES[t["pred"]] for t in tiles)

        rankings.append({
            "image_id": img_id,
            "n_classes": len(gt_classes),
            "gt_classes": sorted(gt_classes),
            "total_tiles": total,
            "n_landform": len(landform_tiles),
            "landform_accuracy": lf_acc,
            "lf_confident_correct": lf_confident_correct,
            "lf_mean_conf": lf_mean_conf,
            "min_class_conf": min_class_conf,
            "min_class_confident_ratio": min_class_confident_ratio,
            "per_class": per_class,
            "label_dist": dict(label_dist),
            "pred_dist": dict(pred_dist),
            "tiles": tiles,
        })

    # Sort by:
    # 1. More classes better
    # 2. Higher min-class confidence (weakest class must be strong)
    # 3. Higher accuracy
    # 4. More landform tiles
    rankings.sort(key=lambda r: (
        r["n_classes"],
        r["min_class_confident_ratio"],
        r["min_class_conf"],
        r["landform_accuracy"],
        r["n_landform"],
    ), reverse=True)

    print(f"\nQualified images (multi-class, all classes with confident tiles): {len(rankings)}")

    # Print top results
    print(f"\n{'='*140}")
    print(f"{'Rk':>3} {'Image ID':<24} {'#C':>3} {'Classes':<12} {'LF Acc':>7} "
          f"{'MinConf':>7} {'MinRatio':>8} {'Per-class detail':<55}")
    print(f"{'='*140}")
    for i, r in enumerate(rankings[:30]):
        cls_str = "+".join(r["gt_classes"])
        detail = "  ".join(
            f"{k}:{v['correct']}/{v['total']}@{v['mean_conf']:.2f}({v['confident']}/{v['total']}>{CONF_THRESHOLD})"
            for k, v in sorted(r["per_class"].items())
        )
        print(
            f"{i+1:>3} {r['image_id']:<24} {r['n_classes']:>3} {cls_str:<12} "
            f"{r['landform_accuracy']:>6.0%} "
            f"{r['min_class_conf']:>7.3f} "
            f"{r['min_class_confident_ratio']:>7.0%} "
            f"{detail}"
        )

    if not rankings:
        print("No qualified images found!")
        return

    best = rankings[0]
    print(f"\n{'='*70}")
    print(f"BEST HIGH-CONFIDENCE MULTI-CLASS: {best['image_id']}")
    print(f"  Classes: {best['gt_classes']}")
    print(f"  Landform accuracy: {best['landform_accuracy']:.1%}")
    print(f"  Min class confidence: {best['min_class_conf']:.3f}")
    for cls, info in sorted(best["per_class"].items()):
        print(f"    {cls}: {info['correct']}/{info['total']} correct, "
              f"mean_conf={info['mean_conf']:.3f}, "
              f"{info['confident']}/{info['total']} above {CONF_THRESHOLD}")
    print(f"{'='*70}")

    # Generate visualization
    output_dir = ROOT / "results" / "levy_match_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    save_data = [{k: v for k, v in r.items() if k != "tiles"} for r in rankings[:50]]
    with open(output_dir / "confident_multiclass_rankings.json", "w") as f:
        json.dump(save_data, f, indent=2, default=str)

    generate_confident_presentation(rankings[:5], output_dir, CONF_THRESHOLD)


def generate_confident_presentation(top_images, output_dir, conf_threshold):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import to_rgba
    from PIL import Image

    CLASS_COLORS = {"LDA": "#e74c3c", "LVF": "#3498db", "CCF": "#2ecc71", "OTHER": "#7f8c8d"}

    for rank, img_data in enumerate(top_images):
        img_id = img_data["image_id"]
        tiles = img_data["tiles"]
        browse_path = BROWSE_DIR / f"{img_id}_RED.abrowse.jpg"
        if not browse_path.exists():
            continue

        browse_img = Image.open(browse_path).convert("RGB")
        img_w, img_h = browse_img.size

        rows = [t["row"] for t in tiles]
        cols = [t["col"] for t in tiles]
        max_row, max_col = max(rows), max(cols)
        tile_h = img_h / (max_row + 1) if max_row > 0 else img_h
        tile_w = img_w / (max_col + 1) if max_col > 0 else img_w

        fig, axes = plt.subplots(1, 3, figsize=(27, 9))

        # Panel 1: Original
        axes[0].imshow(browse_img, cmap="gray")
        axes[0].set_title(f"HiRISE Browse\n{img_id}", fontsize=14, fontweight="bold")
        axes[0].axis("off")

        # Panel 2: Ground Truth
        axes[1].imshow(browse_img, cmap="gray", alpha=0.35)
        for t in tiles:
            label_name = CLASS_NAMES[t["label"]]
            if label_name == "OTHER":
                continue
            color = to_rgba(CLASS_COLORS[label_name], alpha=0.6)
            rect = plt.Rectangle(
                (t["col"] * tile_w, t["row"] * tile_h), tile_w, tile_h,
                facecolor=color, edgecolor="white", linewidth=0.3)
            axes[1].add_patch(rect)
        gt_str = "+".join(img_data["gt_classes"])
        axes[1].set_title(f"Ground Truth (Levy 2014)\n{gt_str}", fontsize=14, fontweight="bold")
        axes[1].axis("off")

        # Panel 3: Predictions with confidence shading
        axes[2].imshow(browse_img, cmap="gray", alpha=0.35)
        for t in tiles:
            pred_name = CLASS_NAMES[t["pred"]]
            if pred_name == "OTHER":
                continue
            conf = t["confidence"]
            # Alpha proportional to confidence (0.3 at 0.3, 0.8 at 1.0)
            alpha = 0.3 + 0.5 * min(1.0, max(0.0, (conf - 0.3) / 0.7))
            color = to_rgba(CLASS_COLORS[pred_name], alpha=alpha)

            is_correct = t["pred"] == t["label"]
            below_thresh = conf < conf_threshold
            if not is_correct:
                ec, lw = "yellow", 2.0
            elif below_thresh:
                ec, lw = "orange", 1.5
            else:
                ec, lw = "white", 0.3

            rect = plt.Rectangle(
                (t["col"] * tile_w, t["row"] * tile_h), tile_w, tile_h,
                facecolor=color, edgecolor=ec, linewidth=lw)
            axes[2].add_patch(rect)

            # Show confidence number for landform tiles
            if pred_name != "OTHER":
                cx = t["col"] * tile_w + tile_w / 2
                cy = t["row"] * tile_h + tile_h / 2
                fontsize = max(4, min(7, int(tile_w / 8)))
                axes[2].text(cx, cy, f"{conf:.0%}", ha="center", va="center",
                            fontsize=fontsize, color="white", fontweight="bold",
                            path_effects=[
                                __import__('matplotlib.patheffects', fromlist=['withStroke']).withStroke(linewidth=1.5, foreground="black")
                            ])

        pca = img_data["per_class"]
        pca_str = " | ".join(f"{k}: {v['correct']}/{v['total']} (conf={v['mean_conf']:.2f})"
                             for k, v in sorted(pca.items()))
        axes[2].set_title(f"V5c Predictions (α ∝ confidence)\n{pca_str}", fontsize=13, fontweight="bold")
        axes[2].axis("off")

        # Legend
        legend_patches = [
            mpatches.Patch(color=CLASS_COLORS["LDA"], label="LDA"),
            mpatches.Patch(color=CLASS_COLORS["LVF"], label="LVF"),
            mpatches.Patch(color=CLASS_COLORS["CCF"], label="CCF"),
            mpatches.Patch(facecolor="white", edgecolor="yellow", linewidth=2, label="Wrong pred"),
            mpatches.Patch(facecolor="white", edgecolor="orange", linewidth=1.5,
                          label=f"Below threshold ({conf_threshold})"),
        ]
        fig.legend(handles=legend_patches, loc="lower center", ncol=5, fontsize=11,
                   frameon=True, bbox_to_anchor=(0.5, 0.01))

        lf_acc = img_data["landform_accuracy"]
        min_conf = img_data["min_class_conf"]
        fig.suptitle(
            f"Levy vs V5c — Rank #{rank+1} | LF Acc: {lf_acc:.0%} | Min class conf: {min_conf:.2f}",
            fontsize=16, fontweight="bold", y=0.98)

        plt.tight_layout(rect=[0, 0.06, 1, 0.94])
        out_path = output_dir / f"confident_rank{rank+1}_{img_id}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
