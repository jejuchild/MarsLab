#!/usr/bin/env python3
"""Analyze landform classification accuracy under resolution degradation.

Experiment: keep tile size 224×224 constant, but simulate lower-resolution
input by downscaling the image to (224/s, 224/s) then upscaling back to 224.
This mimics what would happen if the same model were applied to lower-resolution
imagery (e.g., CTX ~6m/px vs HiRISE ~25cm/px browse ~25m/px).

Scale factors tested:
  1× = original (~25 m/px browse)
  2× = ~50 m/px equivalent
  4× = ~100 m/px equivalent
  8× = ~200 m/px equivalent (MOLA-scale)
  16× = ~400 m/px equivalent

Pipeline: tile_image → degrade → DINOv2 → CLS embedding → FiLM classifier → prediction
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F_torch
from PIL import Image
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from torchvision import transforms

ROOT = Path("/disk1/cspark/hirise-api")
MARSLAB = Path("/disk1/cspark/MarsLab")
V5_DIR = MARSLAB / "Data/HiRISE/v5_retrain"
V4_DIR = MARSLAB / "Data/HiRISE/v4_colab_data_expanded"
TILES_DIR = V4_DIR / "tiles"
TILES_DIR = V4_DIR / "tiles"
OUTPUT_DIR = MARSLAB / "Data/HiRISE/v5_retrain/resolution_analysis"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(MARSLAB))

from scripts.marslandform_v2.models.dinov2_lora import DinoV2LoRA
from scripts.marslandform_v2.config import get_config

CLASS_NAMES = ["LDA", "LVF", "CCF", "OTHER"]
SCALE_FACTORS = [1, 2, 4, 8, 16]
MAX_PER_CLASS = 300  # stratified subsample cap
BATCH_SIZE = 8
IMAGE_SIZE = 224


# ── Data loading ─────────────────────────────────────────────────────────────

def load_experiment_data() -> dict:
    """Load tile index, labels, splits, MOLA features, and FiLM model."""
    with open(V4_DIR / "tile_index.json") as f:
        tile_index = json.load(f)

    with open(V5_DIR / "tile_labels_v5.json") as f:
        labels_list = json.load(f)

    with open(V5_DIR / "tile_splits_v5.json") as f:
        splits = json.load(f)

    mola = np.load(str(V5_DIR / "mola_features_v5.npy"), allow_pickle=True).item()

    # Load FiLM classifier
    from scripts.marslandform_v2.models.film_classifier import FiLMClassifier

    ckpt = torch.load(V5_DIR / "film_classifier_v5c.pt", map_location="cpu", weights_only=False)
    cfg = ckpt.get("cfg", {})
    film_model = FiLMClassifier(
        visual_dim=cfg.get("visual_dim", 768),
        mola_dim=cfg.get("mola_dim", 25),
        num_classes=cfg.get("num_classes", 4),
        film_hidden=cfg.get("film_hidden", 64),
        head_hidden=cfg.get("head_hidden", 128),
        dropout=cfg.get("dropout", 0.4),
    )
    film_model.load_state_dict(ckpt["model_state_dict"])
    film_model.eval()

    return {
        "tile_index": tile_index,
        "labels_list": labels_list,
        "splits": splits,
        "mola": mola,
        "film_model": film_model,
    }


def build_eval_subset(data: dict) -> dict:
    """Build a class-stratified subset of val+test tiles."""
    tile_index = data["tile_index"]
    splits = data["splits"]
    mola = data["mola"]

    c2i = {c: i for i, c in enumerate(CLASS_NAMES)}

    # Build label map
    label_map: dict[str, int] = {}
    for entry in data["labels_list"]:
        key = f"{entry['image_id']}_{entry['tile_row']}_{entry['tile_col']}"
        lbl = entry.get("label", "OTHER")
        if lbl in c2i:
            label_map[key] = c2i[lbl]

    # Get val+test indices
    eval_set = set(splits["val"]) | set(splits["test"])
    tile_keys = list(tile_index.keys())

    # Collect labeled eval tiles with valid image paths and MOLA
    candidates_by_class: dict[int, list] = {i: [] for i in range(len(CLASS_NAMES))}

    for idx in eval_set:
        if idx >= len(tile_keys):
            continue
        key = tile_keys[idx]
        if key not in label_map:
            continue
        label = label_map[key]

        # Check tile image exists
        rel_path = tile_index[key]
        img_path = TILES_DIR / "/".join(rel_path.split("/")[1:])  # strip "tiles/" prefix
        if not img_path.exists():
            img_path = V4_DIR / rel_path
        if not img_path.exists():
            continue

        # Check MOLA features
        parts = key.rsplit("_", 2)
        img_id = parts[0]
        rc = f"{parts[1]}_{parts[2]}"
        if img_id not in mola or rc not in mola[img_id]:
            continue

        candidates_by_class[label].append({
            "key": key,
            "idx": idx,
            "label": label,
            "img_path": str(img_path),
            "mola_key": (img_id, rc),
        })

    # Stratified subsample
    rng = np.random.default_rng(42)
    selected = []
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        pool = candidates_by_class[cls_idx]
        n = min(len(pool), MAX_PER_CLASS)
        chosen = rng.choice(len(pool), size=n, replace=False)
        for c in chosen:
            selected.append(pool[c])
        print(f"  {cls_name}: {n}/{len(pool)} tiles selected")

    print(f"  Total: {len(selected)} tiles")
    return {"tiles": selected, "mola_data": mola}


# ── Resolution degradation ──────────────────────────────────────────────────

def make_transform(scale_factor: int) -> transforms.Compose:
    """Build a transform that degrades resolution by scale_factor.

    Process: original 224 → downscale to 224/s → upscale back to 224 → normalize.
    scale_factor=1 means no degradation (original pipeline).
    """
    steps = []
    if scale_factor > 1:
        small_size = IMAGE_SIZE // scale_factor
        # Downscale (simulates lower resolution capture)
        steps.append(transforms.Resize(
            (small_size, small_size),
            interpolation=transforms.InterpolationMode.BILINEAR,
        ))
        # Upscale back to DINOv2 input size (simulates resampling)
        steps.append(transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=transforms.InterpolationMode.BILINEAR,
        ))
    steps.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return transforms.Compose(steps)


# ── Embedding + classification ───────────────────────────────────────────────

@torch.no_grad()
def embed_and_classify(
    dino_model: DinoV2LoRA,
    film_model: torch.nn.Module,
    tiles: list[dict],
    mola_data: dict,
    scale_factor: int,
) -> dict:
    """Embed tiles at given scale factor and classify with FiLM model."""
    transform = make_transform(scale_factor)
    n = len(tiles)
    all_probs = []
    all_labels = []

    for start in range(0, n, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n)
        batch_imgs = []
        batch_mola = []
        batch_labels = []

        for tile in tiles[start:end]:
            # Load and transform image
            img = Image.open(tile["img_path"]).convert("RGB")
            tensor_img = transform(img)
            batch_imgs.append(tensor_img)

            # MOLA features
            img_id, rc = tile["mola_key"]
            mola_feat = mola_data[img_id][rc]
            batch_mola.append(mola_feat)
            batch_labels.append(tile["label"])

        # Stack batch
        img_batch = torch.stack(batch_imgs)  # (B, 3, 224, 224)
        mola_batch = torch.tensor(np.array(batch_mola), dtype=torch.float32)

        # DINOv2 forward → CLS token
        cls_emb = dino_model(img_batch)  # DinoV2LoRA returns (B, 768) CLS directly

        # FiLM classifier
        logits = film_model(cls_emb, mola_batch)
        probs = F_torch.softmax(logits, dim=1).numpy()

        all_probs.append(probs)
        all_labels.extend(batch_labels)

    all_probs_arr = np.concatenate(all_probs, axis=0)
    all_labels_arr = np.array(all_labels)
    preds = all_probs_arr.argmax(axis=1)

    # Metrics
    macro_f1 = f1_score(all_labels_arr, preds, average="macro")
    per_class_f1 = f1_score(all_labels_arr, preds, average=None, labels=list(range(len(CLASS_NAMES))))
    accuracy = float(np.mean(preds == all_labels_arr))
    cm = confusion_matrix(all_labels_arr, preds, labels=list(range(len(CLASS_NAMES))))

    # Landform-only F1 (exclude OTHER)
    lf_mask = all_labels_arr != CLASS_NAMES.index("OTHER")
    lf_f1 = f1_score(all_labels_arr[lf_mask], preds[lf_mask], average="macro") if lf_mask.sum() > 0 else 0.0

    return {
        "scale_factor": scale_factor,
        "effective_resolution_m": 25.0 * scale_factor,
        "macro_f1": float(macro_f1),
        "landform_f1": float(lf_f1),
        "accuracy": float(accuracy),
        "per_class_f1": {cn: float(f) for cn, f in zip(CLASS_NAMES, per_class_f1)},
        "confusion_matrix": cm.tolist(),
        "n_tiles": len(all_labels_arr),
        "class_distribution": {cn: int((all_labels_arr == i).sum()) for i, cn in enumerate(CLASS_NAMES)},
    }


# ── Visualization ────────────────────────────────────────────────────────────

def plot_results(results: list[dict], output_dir: Path) -> None:
    """Generate publication-quality degradation analysis plots."""
    scales = [r["scale_factor"] for r in results]
    resolutions = [r["effective_resolution_m"] for r in results]
    x_labels = [f"{s}×\n({r:.0f} m/px)" for s, r in zip(scales, resolutions)]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle("Landform Classification Accuracy vs. Resolution Degradation",
                 fontsize=14, fontweight="bold", y=1.02)

    colors = {"LDA": "#e74c3c", "LVF": "#3498db", "CCF": "#2ecc71", "OTHER": "#95a5a6"}

    # ── Panel 1: Per-class F1 ──
    ax = axes[0]
    for cls_name in CLASS_NAMES:
        f1s = [r["per_class_f1"][cls_name] for r in results]
        ax.plot(range(len(scales)), f1s, "o-", label=cls_name,
                color=colors[cls_name], linewidth=2, markersize=7)
    ax.set_xticks(range(len(scales)))
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_ylabel("F1 Score", fontsize=11)
    ax.set_xlabel("Downscale Factor (effective resolution)", fontsize=10)
    ax.set_title("Per-Class F1", fontsize=12)
    ax.legend(fontsize=9, loc="lower left")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)

    # ── Panel 2: Macro F1 + Landform F1 ──
    ax = axes[1]
    macro_f1s = [r["macro_f1"] for r in results]
    lf_f1s = [r["landform_f1"] for r in results]
    accs = [r["accuracy"] for r in results]

    ax.plot(range(len(scales)), macro_f1s, "s-", label="Macro F1",
            color="#2c3e50", linewidth=2.5, markersize=8)
    ax.plot(range(len(scales)), lf_f1s, "D-", label="Landform F1 (LDA+LVF+CCF)",
            color="#e67e22", linewidth=2.5, markersize=8)
    ax.plot(range(len(scales)), accs, "^-", label="Accuracy",
            color="#8e44ad", linewidth=2, markersize=7, alpha=0.7)

    # Annotate values
    for i, (mf, lf) in enumerate(zip(macro_f1s, lf_f1s)):
        ax.annotate(f"{mf:.3f}", (i, mf), textcoords="offset points",
                    xytext=(0, 10), fontsize=8, ha="center", color="#2c3e50")
        ax.annotate(f"{lf:.3f}", (i, lf), textcoords="offset points",
                    xytext=(0, -15), fontsize=8, ha="center", color="#e67e22")

    ax.set_xticks(range(len(scales)))
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_xlabel("Downscale Factor (effective resolution)", fontsize=10)
    ax.set_title("Overall Metrics", fontsize=12)
    ax.legend(fontsize=9, loc="lower left")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # ── Panel 3: F1 drop relative to 1× ──
    ax = axes[2]
    baseline = results[0]
    for cls_name in CLASS_NAMES:
        base_f1 = baseline["per_class_f1"][cls_name]
        if base_f1 > 0:
            drops = [(r["per_class_f1"][cls_name] - base_f1) / base_f1 * 100 for r in results]
        else:
            drops = [0.0] * len(results)
        ax.plot(range(len(scales)), drops, "o-", label=cls_name,
                color=colors[cls_name], linewidth=2, markersize=7)

    macro_base = baseline["macro_f1"]
    macro_drops = [(r["macro_f1"] - macro_base) / macro_base * 100 for r in results]
    ax.plot(range(len(scales)), macro_drops, "s--", label="Macro F1",
            color="#2c3e50", linewidth=2.5, markersize=8)

    ax.set_xticks(range(len(scales)))
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_ylabel("F1 Change (%)", fontsize=11)
    ax.set_xlabel("Downscale Factor (effective resolution)", fontsize=10)
    ax.set_title("Relative F1 Drop from Baseline (1×)", fontsize=12)
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.5, linewidth=0.8)
    ax.axhline(y=-10, color="red", linestyle="--", alpha=0.3, linewidth=0.8)
    ax.axhline(y=-25, color="red", linestyle="--", alpha=0.3, linewidth=0.8)

    plt.tight_layout()
    plt.savefig(output_dir / "resolution_degradation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {output_dir / 'resolution_degradation.png'}")

    # ── Confusion matrices ──
    fig, axes_cm = plt.subplots(1, len(results), figsize=(5 * len(results), 4.5))
    if len(results) == 1:
        axes_cm = [axes_cm]
    for i, r in enumerate(results):
        ax = axes_cm[i]
        cm = np.array(r["confusion_matrix"])
        # Normalize by row (true labels)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

        im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues")
        ax.set_xticks(range(len(CLASS_NAMES)))
        ax.set_xticklabels(CLASS_NAMES, fontsize=9)
        ax.set_yticks(range(len(CLASS_NAMES)))
        ax.set_yticklabels(CLASS_NAMES, fontsize=9)
        ax.set_title(f"{r['scale_factor']}× ({r['effective_resolution_m']:.0f} m/px)\n"
                     f"Macro F1={r['macro_f1']:.3f}", fontsize=10)

        # Annotate cells
        for row in range(len(CLASS_NAMES)):
            for col in range(len(CLASS_NAMES)):
                val = cm_norm[row, col]
                count = cm[row, col]
                color = "white" if val > 0.5 else "black"
                ax.text(col, row, f"{val:.2f}\n({count})",
                        ha="center", va="center", fontsize=8, color=color)

        if i == 0:
            ax.set_ylabel("True Label", fontsize=10)
        ax.set_xlabel("Predicted", fontsize=10)

    plt.suptitle("Confusion Matrices at Each Resolution", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrices_by_resolution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {output_dir / 'confusion_matrices_by_resolution.png'}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("Resolution Degradation Analysis — Landform Classification")
    print("=" * 70)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\n[1/4] Loading data and models...")
    data = load_experiment_data()

    # Build eval subset
    print("\n[2/4] Building stratified eval subset...")
    subset = build_eval_subset(data)
    tiles = subset["tiles"]
    mola_data = subset["mola_data"]

    # Load DINOv2 backbone.
    # NOTE: embeddings_v5.npy was extracted with DinoV2LoRA(use_lora=True) but the
    # SSL LoRA weights were NOT correctly loaded (checkpoint nesting bug). Since
    # LoRA B-matrix inits to zero, the LoRA contribution was zero → effectively
    # vanilla DINOv2-base output. The FiLM classifier was trained on these vanilla
    # embeddings, so we use vanilla DINOv2-base here to match.
    print("\n[3/4] Loading DINOv2-base backbone...")
    t0 = time.time()
    cfg = get_config()
    dino_model = DinoV2LoRA(cfg.dinov2, use_lora=False)
    dino_model.eval()
    print(f"  DINOv2-base loaded in {time.time() - t0:.1f}s")

    # Run experiment
    print(f"\n[4/4] Running degradation experiment ({len(tiles)} tiles × {len(SCALE_FACTORS)} scales)...")
    results = []
    total_start = time.time()

    for sf in SCALE_FACTORS:
        print(f"\n  Scale {sf}× (effective ~{25 * sf} m/px)...")
        t0 = time.time()
        result = embed_and_classify(dino_model, data["film_model"], tiles, mola_data, sf)
        elapsed = time.time() - t0
        results.append(result)

        print(f"    Macro F1: {result['macro_f1']:.4f} | "
              f"Landform F1: {result['landform_f1']:.4f} | "
              f"Accuracy: {result['accuracy']:.4f} | "
              f"Time: {elapsed:.1f}s")
        for cn in CLASS_NAMES:
            print(f"      {cn}: F1={result['per_class_f1'][cn]:.4f}")

    total_elapsed = time.time() - total_start
    print(f"\n  Total experiment time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    header = f"{'Scale':>6} {'Res(m/px)':>10} {'Macro F1':>9} {'LF F1':>8} {'Accuracy':>9}"
    for cn in CLASS_NAMES:
        header += f" {cn:>8}"
    print(header)
    print("-" * len(header))
    for r in results:
        line = f"{r['scale_factor']:>5}× {r['effective_resolution_m']:>9.0f} {r['macro_f1']:>9.4f} {r['landform_f1']:>8.4f} {r['accuracy']:>9.4f}"
        for cn in CLASS_NAMES:
            line += f" {r['per_class_f1'][cn]:>8.4f}"
        print(line)

    # Relative drop
    print("\n  Relative F1 drop from 1× baseline:")
    base = results[0]
    for r in results[1:]:
        drop_macro = (r["macro_f1"] - base["macro_f1"]) / base["macro_f1"] * 100
        drop_lf = (r["landform_f1"] - base["landform_f1"]) / base["landform_f1"] * 100 if base["landform_f1"] > 0 else 0
        print(f"    {r['scale_factor']:>2}×: Macro F1 {drop_macro:+.1f}% | Landform F1 {drop_lf:+.1f}%")

    # Save results
    json_results = {
        "experiment": "resolution_degradation",
        "model": "v5c_film_classifier + dinov2-base",
        "tile_size": IMAGE_SIZE,
        "max_per_class": MAX_PER_CLASS,
        "scale_factors": SCALE_FACTORS,
        "results": results,
        "total_time_s": total_elapsed,
    }
    with open(OUTPUT_DIR / "resolution_degradation_results.json", "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\n  Results JSON saved: {OUTPUT_DIR / 'resolution_degradation_results.json'}")

    # Plots
    print("\n  Generating plots...")
    plot_results(results, OUTPUT_DIR)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
