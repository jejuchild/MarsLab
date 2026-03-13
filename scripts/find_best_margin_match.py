#!/usr/bin/env python3
"""Find image where LDA predictions have large MARGIN over LVF.

Problem: Even when argmax says LDA, the LDA prob might be 0.40 and LVF 0.35.
In the real pipeline, this gets classified as LVF or filtered out.

Solution: Find images where LDA tiles have:
  - High P(LDA) AND
  - Large gap P(LDA) - P(LVF) (margin)
  - Same for other class pairs
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


def load_model(model_path):
    from scripts.marslandform_v2.models.film_classifier import FiLMClassifier
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("cfg", {})
    model = FiLMClassifier(
        visual_dim=cfg.get("visual_dim", 768), mola_dim=cfg.get("mola_dim", 25),
        num_classes=cfg.get("num_classes", 4), film_hidden=cfg.get("film_hidden", 64),
        head_hidden=cfg.get("head_hidden", 128), dropout=cfg.get("dropout", 0.4),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def main():
    print("=" * 70)
    print("Finding best HIGH-MARGIN multi-class match (v5c)")
    print("Focus: LDA tiles must clearly beat LVF probability")
    print("=" * 70)

    with open(V4_DIR / "tile_index.json") as f:
        tile_index = json.load(f)
    embeddings = np.load(str(V5_DIR / "embeddings_v5.npy"))
    mola = np.load(str(V5_DIR / "mola_features_v5.npy"), allow_pickle=True).item()
    with open(V5_DIR / "tile_labels_v5.json") as f:
        labels_list = json.load(f)
    model = load_model(V5_DIR / "film_classifier_v5c.pt")

    labels_raw, label_meta = {}, {}
    for entry in labels_list:
        key = f"{entry['image_id']}_{entry['tile_row']}_{entry['tile_col']}"
        labels_raw[key] = entry.get("label", "OTHER")
        label_meta[key] = entry

    c2i = {c: i for i, c in enumerate(CLASS_NAMES)}
    tile_keys = list(tile_index.keys())
    n = len(tile_keys)
    mola_arr = np.zeros((n, 25), dtype=np.float32)
    label_arr = np.full(n, -1, dtype=np.int64)
    tile_image_ids, tile_rows_list, tile_cols_list = [], [], []

    for i, key in enumerate(tile_keys):
        parts = key.rsplit("_", 2)
        img_id, rc = parts[0], f"{parts[1]}_{parts[2]}"
        tile_image_ids.append(img_id)
        tile_rows_list.append(int(parts[1]))
        tile_cols_list.append(int(parts[2]))
        if img_id in mola and rc in mola[img_id]:
            mola_arr[i] = mola[img_id][rc]
        if key in labels_raw and labels_raw[key] in c2i:
            label_arr[i] = c2i[labels_raw[key]]

    labeled_indices = np.where(label_arr >= 0)[0]

    # Inference — get full probability vectors
    print("Running inference...")
    emb_t = torch.tensor(embeddings[labeled_indices], dtype=torch.float32)
    mola_t = torch.tensor(mola_arr[labeled_indices], dtype=torch.float32)
    all_probs = []
    with torch.no_grad():
        for s in range(0, len(labeled_indices), 512):
            e = min(s + 512, len(labeled_indices))
            logits = model(emb_t[s:e], mola_t[s:e])
            all_probs.append(F.softmax(logits, dim=1).numpy())
    all_probs = np.concatenate(all_probs, axis=0)
    all_preds = all_probs.argmax(axis=1)
    all_labels = label_arr[labeled_indices]

    # Show global LDA confusion stats
    lda_idx, lvf_idx = c2i["LDA"], c2i["LVF"]
    lda_mask = all_labels == lda_idx
    lda_correct = lda_mask & (all_preds == lda_idx)
    print(f"\n=== LDA Global Stats ===")
    print(f"  LDA tiles: {lda_mask.sum()}")
    print(f"  Correctly predicted: {lda_correct.sum()} ({lda_correct.sum()/lda_mask.sum():.1%})")
    if lda_correct.sum() > 0:
        lda_probs_correct = all_probs[lda_correct]
        print(f"  When correct:")
        print(f"    P(LDA):  {lda_probs_correct[:, lda_idx].mean():.3f} ± {lda_probs_correct[:, lda_idx].std():.3f}")
        print(f"    P(LVF):  {lda_probs_correct[:, lvf_idx].mean():.3f} ± {lda_probs_correct[:, lvf_idx].std():.3f}")
        print(f"    Margin (LDA-LVF): {(lda_probs_correct[:, lda_idx] - lda_probs_correct[:, lvf_idx]).mean():.3f}")
    lda_wrong_as_lvf = lda_mask & (all_preds == lvf_idx)
    if lda_wrong_as_lvf.sum() > 0:
        lda_probs_wrong = all_probs[lda_wrong_as_lvf]
        print(f"  When misclassified as LVF ({lda_wrong_as_lvf.sum()}):")
        print(f"    P(LDA):  {lda_probs_wrong[:, lda_idx].mean():.3f}")
        print(f"    P(LVF):  {lda_probs_wrong[:, lvf_idx].mean():.3f}")
        print(f"    Margin:  {(lda_probs_wrong[:, lda_idx] - lda_probs_wrong[:, lvf_idx]).mean():.3f}")

    # Group by image
    image_results = defaultdict(list)
    for j, gi in enumerate(labeled_indices):
        img_id = tile_image_ids[gi]
        key = tile_keys[gi]
        meta = label_meta.get(key, {})
        image_results[img_id].append({
            "pred": int(all_preds[j]),
            "label": int(all_labels[j]),
            "probs": all_probs[j].tolist(),
            "row": tile_rows_list[gi],
            "col": tile_cols_list[gi],
            "label_type": meta.get("label_type", "unknown"),
        })

    # Rank by margin
    print("\nRanking by probability margin...")

    rankings = []
    for img_id, tiles in image_results.items():
        browse_path = BROWSE_DIR / f"{img_id}_RED.abrowse.jpg"
        if not browse_path.exists():
            continue

        landform_tiles = [t for t in tiles if t["label"] != c2i["OTHER"]]
        if len(landform_tiles) < 3:
            continue

        gt_classes = set(CLASS_NAMES[t["label"]] for t in landform_tiles)
        if len(gt_classes) < 2:
            continue

        # Per-class margin analysis
        per_class = {}
        min_margin = float("inf")
        all_ok = True

        for cls_name in gt_classes:
            cls_idx = c2i[cls_name]
            cls_tiles = [t for t in tiles if t["label"] == cls_idx]
            cls_correct = [t for t in cls_tiles if t["pred"] == t["label"]]

            if len(cls_correct) == 0:
                all_ok = False
                break

            # For each correct tile: margin = P(correct_class) - max(P(other_classes))
            margins = []
            p_correct_list = []
            p_runner_up_list = []
            runner_up_names = []

            for t in cls_correct:
                p = np.array(t["probs"])
                p_correct = p[cls_idx]
                p_others = p.copy()
                p_others[cls_idx] = -1
                runner_up_idx = p_others.argmax()
                p_runner = p[runner_up_idx]
                margin = p_correct - p_runner

                margins.append(margin)
                p_correct_list.append(p_correct)
                p_runner_up_list.append(p_runner)
                runner_up_names.append(CLASS_NAMES[runner_up_idx])

            mean_margin = np.mean(margins)
            min_tile_margin = np.min(margins)
            if mean_margin < min_margin:
                min_margin = mean_margin

            # Most common runner-up
            runner_counter = Counter(runner_up_names)
            top_runner = runner_counter.most_common(1)[0][0]

            per_class[cls_name] = {
                "total": len(cls_tiles),
                "correct": len(cls_correct),
                "accuracy": len(cls_correct) / len(cls_tiles),
                "mean_p_correct": float(np.mean(p_correct_list)),
                "mean_p_runner": float(np.mean(p_runner_up_list)),
                "mean_margin": float(mean_margin),
                "min_margin": float(min_tile_margin),
                "top_runner_up": top_runner,
            }

        if not all_ok:
            continue

        lf_correct = sum(1 for t in landform_tiles if t["pred"] == t["label"])
        lf_acc = lf_correct / len(landform_tiles)

        rankings.append({
            "image_id": img_id,
            "n_classes": len(gt_classes),
            "gt_classes": sorted(gt_classes),
            "total_tiles": len(tiles),
            "n_landform": len(landform_tiles),
            "landform_accuracy": lf_acc,
            "min_margin": min_margin,
            "per_class": per_class,
            "tiles": tiles,
        })

    # Sort: high margin, then accuracy, then more tiles
    rankings.sort(key=lambda r: (r["min_margin"], r["landform_accuracy"], r["n_landform"]), reverse=True)

    print(f"\nQualified: {len(rankings)}")
    print(f"\n{'='*160}")
    print(f"{'Rk':>3} {'Image ID':<24} {'Cls':<12} {'LF Acc':>6} {'MinMar':>7} "
          f"{'Per-class: P(cls) vs P(runner) margin runner':<80}")
    print(f"{'='*160}")
    for i, r in enumerate(rankings[:40]):
        cls_str = "+".join(r["gt_classes"])
        detail = "  |  ".join(
            f"{k}: P={v['mean_p_correct']:.2f} vs {v['top_runner_up']}={v['mean_p_runner']:.2f} "
            f"margin={v['mean_margin']:.2f} (min={v['min_margin']:.2f})"
            for k, v in sorted(r["per_class"].items())
        )
        print(f"{i+1:>3} {r['image_id']:<24} {cls_str:<12} {r['landform_accuracy']:>5.0%} "
              f"{r['min_margin']:>7.3f} {detail}")

    if not rankings:
        print("No qualified images!")
        return

    # Show the best
    best = rankings[0]
    print(f"\n{'='*70}")
    print(f"BEST HIGH-MARGIN: {best['image_id']}")
    print(f"  Classes: {best['gt_classes']}")
    print(f"  LF accuracy: {best['landform_accuracy']:.0%}")
    print(f"  Min margin: {best['min_margin']:.3f}")
    for cls, info in sorted(best["per_class"].items()):
        print(f"    {cls}: P({cls})={info['mean_p_correct']:.3f} vs "
              f"P({info['top_runner_up']})={info['mean_p_runner']:.3f} "
              f"margin={info['mean_margin']:.3f} (min={info['min_margin']:.3f}) "
              f"[{info['correct']}/{info['total']}]")
    print(f"{'='*70}")

    output_dir = ROOT / "results" / "levy_match_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    save = [{k: v for k, v in r.items() if k != "tiles"} for r in rankings[:50]]
    with open(output_dir / "margin_rankings.json", "w") as f:
        json.dump(save, f, indent=2, default=str)
    print(f"Saved: {output_dir / 'margin_rankings.json'}")

    # Generate viz for top 5
    generate_margin_viz(rankings[:5], output_dir)


def generate_margin_viz(top_images, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.patheffects as pe
    from matplotlib.colors import to_rgba
    from PIL import Image

    CC = {"LDA": "#e74c3c", "LVF": "#3498db", "CCF": "#2ecc71", "OTHER": "#7f8c8d"}

    for rank, img_data in enumerate(top_images):
        img_id = img_data["image_id"]
        tiles = img_data["tiles"]
        bp = BROWSE_DIR / f"{img_id}_RED.abrowse.jpg"
        if not bp.exists():
            continue

        browse = Image.open(bp).convert("RGB")
        iw, ih = browse.size
        rows = [t["row"] for t in tiles]
        cols = [t["col"] for t in tiles]
        mr, mc = max(rows), max(cols)
        th, tw = ih / (mr + 1), iw / (mc + 1)

        fig, axes = plt.subplots(1, 3, figsize=(27, 9))

        # Panel 1: Original
        axes[0].imshow(browse, cmap="gray")
        axes[0].set_title(f"HiRISE Browse\n{img_id}", fontsize=14, fontweight="bold")
        axes[0].axis("off")

        # Panel 2: Ground Truth
        axes[1].imshow(browse, cmap="gray", alpha=0.35)
        for t in tiles:
            ln = CLASS_NAMES[t["label"]]
            if ln == "OTHER":
                continue
            rect = plt.Rectangle((t["col"]*tw, t["row"]*th), tw, th,
                                 facecolor=to_rgba(CC[ln], 0.6), edgecolor="white", linewidth=0.3)
            axes[1].add_patch(rect)
        axes[1].set_title(f"Ground Truth (Levy 2014)\n{'+'.join(img_data['gt_classes'])}",
                         fontsize=14, fontweight="bold")
        axes[1].axis("off")

        # Panel 3: Predictions with probability bars
        axes[2].imshow(browse, cmap="gray", alpha=0.35)
        for t in tiles:
            pn = CLASS_NAMES[t["pred"]]
            if pn == "OTHER":
                continue
            p = t["probs"]
            conf = max(p)
            alpha = 0.3 + 0.5 * min(1.0, max(0, (conf - 0.3) / 0.7))
            correct = t["pred"] == t["label"]
            ec = "white" if correct else "yellow"
            lw = 0.3 if correct else 2.0

            rect = plt.Rectangle((t["col"]*tw, t["row"]*th), tw, th,
                                 facecolor=to_rgba(CC[pn], alpha), edgecolor=ec, linewidth=lw)
            axes[2].add_patch(rect)

            # Show P(pred) / P(runner)
            pred_idx = t["pred"]
            p_arr = np.array(p)
            p_others = p_arr.copy(); p_others[pred_idx] = -1
            runner_idx = p_others.argmax()
            margin = p_arr[pred_idx] - p_arr[runner_idx]

            cx = t["col"] * tw + tw / 2
            cy = t["row"] * th + th / 2
            fs = max(4, min(6, int(tw / 10)))
            txt = f"{p_arr[pred_idx]:.0%}"
            axes[2].text(cx, cy, txt, ha="center", va="center", fontsize=fs,
                        color="white", fontweight="bold",
                        path_effects=[pe.withStroke(linewidth=1.5, foreground="black")])

        pca = img_data["per_class"]
        detail = " | ".join(f"{k}: P={v['mean_p_correct']:.0%} (margin {v['mean_margin']:.0%})"
                            for k, v in sorted(pca.items()))
        axes[2].set_title(f"V5c Predictions\n{detail}", fontsize=13, fontweight="bold")
        axes[2].axis("off")

        legend_patches = [
            mpatches.Patch(color=CC["LDA"], label="LDA"),
            mpatches.Patch(color=CC["LVF"], label="LVF"),
            mpatches.Patch(color=CC["CCF"], label="CCF"),
            mpatches.Patch(facecolor="white", edgecolor="yellow", linewidth=2, label="Mismatch"),
        ]
        fig.legend(handles=legend_patches, loc="lower center", ncol=4, fontsize=11,
                   frameon=True, bbox_to_anchor=(0.5, 0.01))

        fig.suptitle(
            f"Levy vs V5c — Margin Rank #{rank+1} | "
            f"LF Acc: {img_data['landform_accuracy']:.0%} | Min margin: {img_data['min_margin']:.2f}",
            fontsize=16, fontweight="bold", y=0.98)

        plt.tight_layout(rect=[0, 0.06, 1, 0.94])
        out = output_dir / f"margin_rank{rank+1}_{img_id}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  Saved: {out}")


if __name__ == "__main__":
    main()
