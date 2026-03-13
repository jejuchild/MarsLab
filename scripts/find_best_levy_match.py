#!/usr/bin/env python3
"""Find the HiRISE image where v5c model predictions best match Levy polygon labels.

Outputs:
  1. Ranked list of images by per-image accuracy (landform tiles only)
  2. Presentation-ready visualization of the best image
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

# Fallback paths (hirise-api copy)
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
    print("Finding best Levy-matching HiRISE image (v5c model)")
    print("=" * 60)

    # 1. Load assets
    print("\n[1] Loading assets...")
    with open(V4_DIR / "tile_index.json") as f:
        tile_index = json.load(f)
    print(f"  Tile index: {len(tile_index)} tiles")

    embeddings = np.load(str(V5_DIR / "embeddings_v5.npy"))
    print(f"  Embeddings: {embeddings.shape}")

    mola = np.load(str(V5_DIR / "mola_features_v5.npy"), allow_pickle=True).item()
    print(f"  MOLA features: {len(mola)} images")

    with open(V5_DIR / "tile_labels_v5.json") as f:
        labels_list = json.load(f)
    print(f"  Labels: {len(labels_list)} entries")

    model = load_model(V5_DIR / "film_classifier_v5c.pt")
    print("  Model loaded: film_classifier_v5c.pt")

    # 2. Build aligned arrays (same logic as retrain_v5c_balanced.py)
    print("\n[2] Building aligned arrays...")
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

    # Map tile keys to image_id
    tile_image_ids = []
    tile_rows = []
    tile_cols = []

    for i, key in enumerate(tile_keys):
        parts = key.rsplit("_", 2)
        image_id = parts[0]
        row_col = f"{parts[1]}_{parts[2]}"
        tile_image_ids.append(image_id)
        tile_rows.append(int(parts[1]))
        tile_cols.append(int(parts[2]))

        if image_id in mola and row_col in mola[image_id]:
            mola_arr[i] = mola[image_id][row_col]

        if key in labels_raw:
            cls = labels_raw[key]
            if cls in class_to_idx:
                label_arr[i] = class_to_idx[cls]

    # 3. Get ALL labeled (non-UNLABELED) tile indices
    labeled_mask = label_arr >= 0
    labeled_indices = np.where(labeled_mask)[0]
    print(f"  Labeled tiles: {len(labeled_indices)} / {n}")

    # 4. Run inference
    print("\n[3] Running inference on all labeled tiles...")
    emb_t = torch.tensor(embeddings[labeled_indices], dtype=torch.float32)
    mola_t = torch.tensor(mola_arr[labeled_indices], dtype=torch.float32)

    batch_size = 512
    all_preds = []
    with torch.no_grad():
        for start in range(0, len(labeled_indices), batch_size):
            end = min(start + batch_size, len(labeled_indices))
            logits = model(emb_t[start:end], mola_t[start:end])
            preds = logits.argmax(dim=1).numpy()
            all_preds.extend(preds)

    all_preds = np.array(all_preds)
    all_labels = label_arr[labeled_indices]
    print(f"  Total predictions: {len(all_preds)}")
    print(f"  Overall accuracy: {(all_preds == all_labels).mean():.4f}")

    # 5. Group by image and compute per-image stats
    print("\n[4] Computing per-image statistics...")

    # image_id -> list of (pred, label, row, col, coverage_info)
    image_results = defaultdict(list)
    for j, global_idx in enumerate(labeled_indices):
        img_id = tile_image_ids[global_idx]
        pred = int(all_preds[j])
        label = int(all_labels[j])
        row = tile_rows[global_idx]
        col = tile_cols[global_idx]
        key = tile_keys[global_idx]
        meta = label_meta.get(key, {})
        image_results[img_id].append({
            "pred": pred,
            "label": label,
            "row": row,
            "col": col,
            "coverage": meta.get("coverage", {}),
            "label_type": meta.get("label_type", "unknown"),
        })

    # 6. Rank images — focus on those with Levy landform tiles
    print("\n[5] Ranking images by Levy match quality...")

    rankings = []
    for img_id, tiles in image_results.items():
        total = len(tiles)
        correct = sum(1 for t in tiles if t["pred"] == t["label"])
        accuracy = correct / total if total > 0 else 0

        # Count landform tiles (non-OTHER)
        landform_tiles = [t for t in tiles if t["label"] != class_to_idx["OTHER"]]
        n_landform = len(landform_tiles)
        if n_landform == 0:
            continue  # Skip images with no landform labels

        landform_correct = sum(1 for t in landform_tiles if t["pred"] == t["label"])
        landform_acc = landform_correct / n_landform if n_landform > 0 else 0

        # Count per-class
        label_dist = Counter(CLASS_NAMES[t["label"]] for t in tiles)
        pred_dist = Counter(CLASS_NAMES[t["pred"]] for t in tiles)

        # Count confident landform tiles
        confident_landform = [t for t in tiles if t["label_type"] == "confident" and t["label"] != class_to_idx["OTHER"]]
        n_confident = len(confident_landform)
        confident_correct = sum(1 for t in confident_landform if t["pred"] == t["label"])
        confident_acc = confident_correct / n_confident if n_confident > 0 else 0

        # Check browse image exists
        browse_path = BROWSE_DIR / f"{img_id}_RED.abrowse.jpg"
        has_browse = browse_path.exists()

        rankings.append({
            "image_id": img_id,
            "total_tiles": total,
            "accuracy": accuracy,
            "n_landform": n_landform,
            "landform_accuracy": landform_acc,
            "landform_correct": landform_correct,
            "n_confident": n_confident,
            "confident_accuracy": confident_acc,
            "confident_correct": confident_correct,
            "label_dist": dict(label_dist),
            "pred_dist": dict(pred_dist),
            "has_browse": has_browse,
            "tiles": tiles,
        })

    # Sort by: (1) has browse image, (2) confident landform accuracy, (3) n_confident tiles, (4) landform accuracy
    # Require at least 3 confident landform tiles for a meaningful comparison
    qualified = [r for r in rankings if r["n_confident"] >= 3 and r["has_browse"]]
    qualified.sort(key=lambda r: (r["confident_accuracy"], r["n_confident"], r["landform_accuracy"]), reverse=True)

    print(f"\n  Total images with landform tiles: {len(rankings)}")
    print(f"  Qualified (≥3 confident landform tiles + browse): {len(qualified)}")

    # Print top 20
    print(f"\n{'='*100}")
    print(f"{'Rank':>4} {'Image ID':<24} {'Conf.Acc':>8} {'Conf':>5} {'LF Acc':>7} {'LF':>4} {'Total':>5} {'Labels':<30} {'Preds':<30}")
    print(f"{'='*100}")
    for i, r in enumerate(qualified[:30]):
        label_str = " ".join(f"{k}:{v}" for k, v in sorted(r["label_dist"].items()) if k != "OTHER")
        pred_str = " ".join(f"{k}:{v}" for k, v in sorted(r["pred_dist"].items()) if k != "OTHER")
        print(
            f"{i+1:>4} {r['image_id']:<24} "
            f"{r['confident_accuracy']:>7.1%} {r['confident_correct']:>2}/{r['n_confident']:<2} "
            f"{r['landform_accuracy']:>6.1%} {r['landform_correct']:>2}/{r['n_landform']:<2} "
            f"{r['total_tiles']:>5} "
            f"{label_str:<30} {pred_str:<30}"
        )

    if not qualified:
        print("No qualified images found!")
        return

    # 7. Save results
    best = qualified[0]
    print(f"\n{'='*60}")
    print(f"BEST MATCH: {best['image_id']}")
    print(f"  Confident landform accuracy: {best['confident_accuracy']:.1%} ({best['confident_correct']}/{best['n_confident']})")
    print(f"  All landform accuracy: {best['landform_accuracy']:.1%} ({best['landform_correct']}/{best['n_landform']})")
    print(f"  Total tiles: {best['total_tiles']}")
    print(f"  Label distribution: {best['label_dist']}")
    print(f"  Pred distribution: {best['pred_dist']}")
    print(f"{'='*60}")

    # Save top results JSON
    output_dir = ROOT / "results" / "levy_match_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    top_results = []
    for r in qualified[:50]:
        top_results.append({k: v for k, v in r.items() if k != "tiles"})

    with open(output_dir / "levy_match_rankings.json", "w") as f:
        json.dump(top_results, f, indent=2)
    print(f"\nRankings saved: {output_dir / 'levy_match_rankings.json'}")

    # 8. Generate visualization for top candidates
    print("\n[6] Generating presentation visualization...")
    generate_presentation(qualified[:5], output_dir)


def generate_presentation(top_images: list[dict], output_dir: Path):
    """Generate presentation-ready visualization for the best matching image."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from PIL import Image

    CLASS_COLORS = {
        "LDA": "#e74c3c",   # red
        "LVF": "#3498db",   # blue
        "CCF": "#2ecc71",   # green
        "OTHER": "#95a5a6", # gray
    }

    for rank, img_data in enumerate(top_images):
        img_id = img_data["image_id"]
        tiles = img_data["tiles"]

        browse_path = BROWSE_DIR / f"{img_id}_RED.abrowse.jpg"
        if not browse_path.exists():
            continue

        browse_img = Image.open(browse_path).convert("RGB")
        img_w, img_h = browse_img.size

        # Build prediction and label grids
        rows = [t["row"] for t in tiles]
        cols = [t["col"] for t in tiles]
        if not rows:
            continue

        max_row, max_col = max(rows), max(cols)
        min_row, min_col = min(rows), min(cols)
        grid_rows = max_row - min_row + 1
        grid_cols = max_col - min_col + 1

        # Create color overlays
        tile_h = img_h / (max_row + 1) if max_row > 0 else img_h
        tile_w = img_w / (max_col + 1) if max_col > 0 else img_w

        fig, axes = plt.subplots(1, 3, figsize=(24, 8))

        # Left: Original browse image
        axes[0].imshow(browse_img, cmap="gray")
        axes[0].set_title(f"HiRISE Browse Image\n{img_id}", fontsize=13, fontweight="bold")
        axes[0].axis("off")

        # Middle: Ground truth (Levy polygon labels)
        axes[1].imshow(browse_img, cmap="gray", alpha=0.4)
        for t in tiles:
            if t["label"] == CLASS_NAMES.index("OTHER"):
                continue
            label_name = CLASS_NAMES[t["label"]]
            color = CLASS_COLORS[label_name]
            y0 = t["row"] * tile_h
            x0 = t["col"] * tile_w
            rect = plt.Rectangle((x0, y0), tile_w, tile_h,
                                 facecolor=color, alpha=0.55, edgecolor="white", linewidth=0.5)
            axes[1].add_patch(rect)
        axes[1].set_title("Ground Truth (Levy 2014 Polygons)", fontsize=13, fontweight="bold")
        axes[1].axis("off")

        # Right: Model predictions
        axes[2].imshow(browse_img, cmap="gray", alpha=0.4)
        for t in tiles:
            if t["pred"] == CLASS_NAMES.index("OTHER"):
                continue
            pred_name = CLASS_NAMES[t["pred"]]
            color = CLASS_COLORS[pred_name]
            y0 = t["row"] * tile_h
            x0 = t["col"] * tile_w

            # Mark correctness
            is_correct = t["pred"] == t["label"]
            ec = "white" if is_correct else "black"
            lw = 0.5 if is_correct else 1.5

            rect = plt.Rectangle((x0, y0), tile_w, tile_h,
                                 facecolor=color, alpha=0.55, edgecolor=ec, linewidth=lw)
            axes[2].add_patch(rect)

        # Stats annotation
        conf_acc = img_data["confident_accuracy"]
        lf_acc = img_data["landform_accuracy"]
        axes[2].set_title(
            f"V5c Model Predictions\n"
            f"Landform Acc: {lf_acc:.0%} | Confident Acc: {conf_acc:.0%}",
            fontsize=13, fontweight="bold"
        )
        axes[2].axis("off")

        # Legend
        legend_patches = [
            mpatches.Patch(color=CLASS_COLORS["LDA"], label="LDA (Lobate Debris Apron)"),
            mpatches.Patch(color=CLASS_COLORS["LVF"], label="LVF (Lineated Valley Fill)"),
            mpatches.Patch(color=CLASS_COLORS["CCF"], label="CCF (Concentric Crater Fill)"),
            mpatches.Patch(color=CLASS_COLORS["OTHER"], label="OTHER (Background)"),
        ]
        fig.legend(handles=legend_patches, loc="lower center", ncol=4, fontsize=11,
                   frameon=True, fancybox=True, shadow=True)

        fig.suptitle(
            f"Levy 2014 Polygon vs V5c Model — Rank #{rank+1}",
            fontsize=16, fontweight="bold", y=0.98
        )

        plt.tight_layout(rect=[0, 0.06, 1, 0.95])
        out_path = output_dir / f"levy_match_rank{rank+1}_{img_id}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
