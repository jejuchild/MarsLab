#!/usr/bin/env python3
"""Find the HiRISE image with MULTIPLE landform classes where v5c model best matches Levy polygons.

More impressive for presentation: images with LDA+LVF, LDA+CCF, etc. all correctly classified.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

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


def load_model(model_path: Path) -> torch.nn.Module:
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
    print("=" * 60)
    print("Finding best MULTI-CLASS Levy-matching image (v5c)")
    print("=" * 60)

    # Load assets
    with open(V4_DIR / "tile_index.json") as f:
        tile_index = json.load(f)
    embeddings = np.load(str(V5_DIR / "embeddings_v5.npy"))
    mola = np.load(str(V5_DIR / "mola_features_v5.npy"), allow_pickle=True).item()
    with open(V5_DIR / "tile_labels_v5.json") as f:
        labels_list = json.load(f)
    model = load_model(V5_DIR / "film_classifier_v5c.pt")
    print("Assets loaded.")

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
        if key in labels_raw:
            cls = labels_raw[key]
            if cls in class_to_idx:
                label_arr[i] = class_to_idx[cls]

    labeled_indices = np.where(label_arr >= 0)[0]

    # Inference
    print("Running inference...")
    emb_t = torch.tensor(embeddings[labeled_indices], dtype=torch.float32)
    mola_t = torch.tensor(mola_arr[labeled_indices], dtype=torch.float32)
    all_preds = []
    with torch.no_grad():
        for start in range(0, len(labeled_indices), 512):
            end = min(start + 512, len(labeled_indices))
            logits = model(emb_t[start:end], mola_t[start:end])
            all_preds.extend(logits.argmax(dim=1).numpy())
    all_preds = np.array(all_preds)
    all_labels = label_arr[labeled_indices]

    # Group by image
    image_results = defaultdict(list)
    for j, global_idx in enumerate(labeled_indices):
        img_id = tile_image_ids[global_idx]
        key = tile_keys[global_idx]
        meta = label_meta.get(key, {})
        image_results[img_id].append({
            "pred": int(all_preds[j]),
            "label": int(all_labels[j]),
            "row": tile_rows_list[global_idx],
            "col": tile_cols_list[global_idx],
            "coverage": meta.get("coverage", {}),
            "label_type": meta.get("label_type", "unknown"),
        })

    # Find multi-class images
    print("\nSearching for multi-class images...")

    rankings = []
    for img_id, tiles in image_results.items():
        browse_path = BROWSE_DIR / f"{img_id}_RED.abrowse.jpg"
        if not browse_path.exists():
            continue

        landform_tiles = [t for t in tiles if t["label"] != class_to_idx["OTHER"]]
        if len(landform_tiles) < 3:
            continue

        # Count distinct landform classes in ground truth
        gt_classes = set(CLASS_NAMES[t["label"]] for t in landform_tiles)
        n_classes = len(gt_classes)

        # We want MULTI-CLASS (at least 2 different landform types)
        if n_classes < 2:
            continue

        total = len(tiles)
        correct = sum(1 for t in tiles if t["pred"] == t["label"])
        accuracy = correct / total

        lf_correct = sum(1 for t in landform_tiles if t["pred"] == t["label"])
        lf_acc = lf_correct / len(landform_tiles)

        confident_lf = [t for t in tiles if t["label_type"] == "confident" and t["label"] != class_to_idx["OTHER"]]
        n_conf = len(confident_lf)
        conf_correct = sum(1 for t in confident_lf if t["pred"] == t["label"])
        conf_acc = conf_correct / n_conf if n_conf > 0 else 0

        label_dist = Counter(CLASS_NAMES[t["label"]] for t in tiles)
        pred_dist = Counter(CLASS_NAMES[t["pred"]] for t in tiles)

        # Per-class accuracy breakdown
        per_class_acc = {}
        for cls_name in gt_classes:
            cls_idx = class_to_idx[cls_name]
            cls_tiles = [t for t in tiles if t["label"] == cls_idx]
            cls_correct = sum(1 for t in cls_tiles if t["pred"] == t["label"])
            per_class_acc[cls_name] = f"{cls_correct}/{len(cls_tiles)}"

        rankings.append({
            "image_id": img_id,
            "n_classes": n_classes,
            "gt_classes": sorted(gt_classes),
            "total_tiles": total,
            "accuracy": accuracy,
            "n_landform": len(landform_tiles),
            "landform_accuracy": lf_acc,
            "n_confident": n_conf,
            "confident_accuracy": conf_acc,
            "label_dist": dict(label_dist),
            "pred_dist": dict(pred_dist),
            "per_class_acc": per_class_acc,
            "tiles": tiles,
        })

    # Sort: (1) more classes better, (2) higher landform accuracy, (3) more landform tiles
    rankings.sort(key=lambda r: (r["n_classes"], r["landform_accuracy"], r["n_landform"]), reverse=True)

    print(f"Multi-class images found: {len(rankings)}")

    # Print top results
    print(f"\n{'='*120}")
    print(f"{'Rank':>4} {'Image ID':<24} {'#Cls':>4} {'Classes':<12} {'LF Acc':>7} {'LF':>6} {'Total':>5} {'Per-class accuracy':<40}")
    print(f"{'='*120}")
    for i, r in enumerate(rankings[:30]):
        cls_str = "+".join(r["gt_classes"])
        pca_str = "  ".join(f"{k}:{v}" for k, v in sorted(r["per_class_acc"].items()))
        print(
            f"{i+1:>4} {r['image_id']:<24} "
            f"{r['n_classes']:>4} {cls_str:<12} "
            f"{r['landform_accuracy']:>6.1%} "
            f"{r['n_landform']:>6} {r['total_tiles']:>5} "
            f"{pca_str:<40}"
        )

    if not rankings:
        print("No multi-class images found!")
        return

    best = rankings[0]
    print(f"\n{'='*60}")
    print(f"BEST MULTI-CLASS MATCH: {best['image_id']}")
    print(f"  Classes: {best['gt_classes']}")
    print(f"  Landform accuracy: {best['landform_accuracy']:.1%}")
    print(f"  Per-class: {best['per_class_acc']}")
    print(f"  Label dist: {best['label_dist']}")
    print(f"  Pred dist: {best['pred_dist']}")
    print(f"{'='*60}")

    # Generate visualization for top 5
    output_dir = ROOT / "results" / "levy_match_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save rankings
    save_rankings = [{k: v for k, v in r.items() if k != "tiles"} for r in rankings[:50]]
    with open(output_dir / "multiclass_rankings.json", "w") as f:
        json.dump(save_rankings, f, indent=2)

    generate_multiclass_presentation(rankings[:5], output_dir)


def generate_multiclass_presentation(top_images: list[dict], output_dir: Path):
    """Generate presentation-ready visualization for multi-class images."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from PIL import Image

    CLASS_COLORS = {
        "LDA": "#e74c3c",
        "LVF": "#3498db",
        "CCF": "#2ecc71",
        "OTHER": "#7f8c8d",
    }
    CLASS_COLORS_RGBA = {
        "LDA": (0.91, 0.30, 0.24, 0.6),
        "LVF": (0.20, 0.60, 0.86, 0.6),
        "CCF": (0.18, 0.80, 0.44, 0.6),
        "OTHER": (0.50, 0.55, 0.55, 0.2),
    }

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
        max_row = max(rows) if rows else 0
        max_col = max(cols) if cols else 0

        tile_h = img_h / (max_row + 1) if max_row > 0 else img_h
        tile_w = img_w / (max_col + 1) if max_col > 0 else img_w

        fig, axes = plt.subplots(1, 3, figsize=(27, 9))

        # 1. Original
        axes[0].imshow(browse_img, cmap="gray")
        axes[0].set_title(f"HiRISE Browse Image\n{img_id}", fontsize=14, fontweight="bold")
        axes[0].axis("off")

        # 2. Ground Truth
        axes[1].imshow(browse_img, cmap="gray", alpha=0.35)
        for t in tiles:
            label_name = CLASS_NAMES[t["label"]]
            rgba = CLASS_COLORS_RGBA[label_name]
            y0 = t["row"] * tile_h
            x0 = t["col"] * tile_w
            rect = plt.Rectangle((x0, y0), tile_w, tile_h,
                                 facecolor=rgba[:3], alpha=rgba[3],
                                 edgecolor="white", linewidth=0.3)
            axes[1].add_patch(rect)

        gt_classes = "+".join(img_data["gt_classes"])
        axes[1].set_title(f"Ground Truth (Levy 2014)\nClasses: {gt_classes}", fontsize=14, fontweight="bold")
        axes[1].axis("off")

        # 3. Predictions with correctness
        axes[2].imshow(browse_img, cmap="gray", alpha=0.35)
        n_correct = 0
        n_wrong = 0
        for t in tiles:
            pred_name = CLASS_NAMES[t["pred"]]
            label_name = CLASS_NAMES[t["label"]]
            rgba = CLASS_COLORS_RGBA[pred_name]
            y0 = t["row"] * tile_h
            x0 = t["col"] * tile_w

            is_correct = t["pred"] == t["label"]
            if is_correct:
                n_correct += 1
                ec = "white"
                lw = 0.3
            else:
                n_wrong += 1
                ec = "yellow"
                lw = 2.0

            rect = plt.Rectangle((x0, y0), tile_w, tile_h,
                                 facecolor=rgba[:3], alpha=rgba[3],
                                 edgecolor=ec, linewidth=lw)
            axes[2].add_patch(rect)

        lf_acc = img_data["landform_accuracy"]
        pca = img_data["per_class_acc"]
        pca_str = " | ".join(f"{k}: {v}" for k, v in sorted(pca.items()))
        axes[2].set_title(
            f"V5c Predictions (LF Acc: {lf_acc:.0%})\n{pca_str}",
            fontsize=14, fontweight="bold"
        )
        axes[2].axis("off")

        # Legend
        legend_patches = [
            mpatches.Patch(color=CLASS_COLORS["LDA"], label="LDA"),
            mpatches.Patch(color=CLASS_COLORS["LVF"], label="LVF"),
            mpatches.Patch(color=CLASS_COLORS["CCF"], label="CCF"),
            mpatches.Patch(facecolor="white", edgecolor="yellow", linewidth=2, label="Mismatch"),
        ]
        fig.legend(handles=legend_patches, loc="lower center", ncol=4, fontsize=12,
                   frameon=True, fancybox=True, shadow=True, bbox_to_anchor=(0.5, 0.01))

        fig.suptitle(
            f"Levy 2014 vs V5c Model — Multi-class Rank #{rank+1} | "
            f"Accuracy: {n_correct}/{n_correct+n_wrong} ({n_correct/(n_correct+n_wrong):.0%})",
            fontsize=16, fontweight="bold", y=0.98
        )

        plt.tight_layout(rect=[0, 0.05, 1, 0.94])
        out_path = output_dir / f"multiclass_rank{rank+1}_{img_id}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  Saved: {out_path}")

    print(f"\nAll visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
