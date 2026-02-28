#!/usr/bin/env python3
"""Generate all visualizations from trained MIL model with actual tile images."""
import sys
import json
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from collections import Counter
from PIL import Image

ROOT = Path("/disk1/cspark/MarsLab")
sys.path.insert(0, str(ROOT))

CLASS_ORDER = ["LDA", "LVF", "CCF", "GLF", "BACKGROUND"]
CLASS_COLORS = {
    "LDA": "#e74c3c",
    "LVF": "#3498db",
    "CCF": "#2ecc71",
    "GLF": "#f39c12",
    "BACKGROUND": "#9b59b6",
}

model_dir = ROOT / "Data" / "HiRISE" / "v2_output" / "models" / "cleaned_focal"
eval_dir = ROOT / "Data" / "HiRISE" / "v2_output" / "eval"
tiles_dir = ROOT / "Data" / "HiRISE" / "v2_output" / "tiles"
eval_dir.mkdir(parents=True, exist_ok=True)

# Load results
metrics = json.loads((model_dir / "test_metrics.json").read_text())
predictions = json.loads((model_dir / "test_predictions_with_attention.json").read_text())
curves = json.loads((model_dir / "training_curves.json").read_text())


def get_tile_files(image_id: str) -> list:
    """Get sorted tile files for an image."""
    tile_dir = tiles_dir / image_id
    if not tile_dir.exists():
        return []
    files = sorted(glob.glob(str(tile_dir / "*.jpg")))
    if not files:
        files = sorted(glob.glob(str(tile_dir / "*.png")))
    return files


# ─── 1. Training Curves ────────────────────────────────────────────────────────
print("1. Training curves...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
epochs = range(1, len(curves["train_loss"]) + 1)
ax1.plot(epochs, curves["train_loss"], "b-", label="Train Loss", linewidth=2)
ax1.plot(epochs, curves["val_loss"], "r-", label="Val Loss", linewidth=2)
ax1.set_xlabel("Epoch", fontsize=12)
ax1.set_ylabel("Loss", fontsize=12)
ax1.legend(fontsize=11)
ax1.set_title("Training / Validation Loss", fontsize=13)
ax1.grid(True, alpha=0.3)

ax2.plot(epochs, curves["val_macro_f1"], "g-", label="Macro F1 (all)", linewidth=2)
ax2.plot(epochs, curves["val_landform_macro_f1"], "m-", label="Landform F1", linewidth=2)
best_epoch = int(np.argmax(curves["val_landform_macro_f1"])) + 1
best_f1 = max(curves["val_landform_macro_f1"])
ax2.axvline(best_epoch, color="gray", linestyle="--", alpha=0.5)
ax2.annotate(f"Best: {best_f1:.3f}\n(epoch {best_epoch})",
             xy=(best_epoch, best_f1), fontsize=10,
             xytext=(best_epoch + 2, best_f1 - 0.05),
             arrowprops=dict(arrowstyle="->", color="gray"))
ax2.set_xlabel("Epoch", fontsize=12)
ax2.set_ylabel("F1 Score", fontsize=12)
ax2.legend(fontsize=11)
ax2.set_title("Validation F1 Scores", fontsize=13)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 0.8)

fig.tight_layout()
fig.savefig(eval_dir / "training_curves.png", dpi=200)
plt.close(fig)
print("  Saved training_curves.png")


# ─── 2. Per-class F1 Bar Chart ────────────────────────────────────────────────
print("2. Per-class F1 bar chart...")
f1_scores = metrics["f1"]
fig, ax = plt.subplots(figsize=(10, 6))
colors = [CLASS_COLORS[c] for c in CLASS_ORDER]
bars = ax.bar(CLASS_ORDER, f1_scores, color=colors, edgecolor="white", linewidth=1.5)
for bar, val in zip(bars, f1_scores):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
            f"{val:.3f}", ha="center", fontsize=12, fontweight="bold")
ax.set_ylabel("F1 Score", fontsize=13)
ax.set_title(f"Per-class F1 Score — Macro F1: {metrics['macro_f1_all']:.3f} | Landform F1: {metrics['landform_macro_f1']:.3f}",
             fontsize=13)
ax.set_ylim(0, 1.0)
ax.grid(axis="y", alpha=0.3)
# Add support counts
for i, cls in enumerate(CLASS_ORDER):
    support = metrics["support"][i]
    ax.text(bars[i].get_x() + bars[i].get_width() / 2, 0.02,
            f"n={int(support)}", ha="center", fontsize=9, color="white", fontweight="bold")
fig.tight_layout()
fig.savefig(eval_dir / "per_class_f1.png", dpi=200)
plt.close(fig)
print("  Saved per_class_f1.png")


# ─── 3. Confusion Matrix (enhanced) ──────────────────────────────────────────
print("3. Confusion matrix...")
cm = np.array(metrics["confusion_matrix"])
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
ax.set_xticks(range(len(CLASS_ORDER)))
ax.set_yticks(range(len(CLASS_ORDER)))
ax.set_xticklabels(CLASS_ORDER, fontsize=11)
ax.set_yticklabels(CLASS_ORDER, fontsize=11)
ax.set_xlabel("Predicted", fontsize=12)
ax.set_ylabel("True", fontsize=12)
ax.set_title("Confusion Matrix — MIL + FocalLoss + Levy2014 Labels", fontsize=12)

thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, str(int(cm[i, j])), ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black", fontsize=13)
fig.colorbar(im, ax=ax, shrink=0.8)
fig.tight_layout()
fig.savefig(eval_dir / "confusion_matrix.png", dpi=200)
plt.close(fig)
print("  Saved confusion_matrix.png")


# ─── 4. Confidence Distribution ──────────────────────────────────────────────
print("4. Confidence distribution...")
fig, axes = plt.subplots(1, 5, figsize=(22, 4), sharey=True)
for i, cls in enumerate(CLASS_ORDER):
    cls_preds = [p for p in predictions if p["true_label"] == i]
    correct = [p["confidence"] for p in cls_preds if p["pred_label"] == i]
    wrong = [p["confidence"] for p in cls_preds if p["pred_label"] != i]
    axes[i].hist(correct, bins=10, alpha=0.7, label="Correct", color="green", range=(0, 1))
    axes[i].hist(wrong, bins=10, alpha=0.7, label="Wrong", color="red", range=(0, 1))
    axes[i].set_title(f"{cls} (n={len(cls_preds)})", fontsize=11)
    axes[i].set_xlabel("Confidence")
    if i == 0:
        axes[i].set_ylabel("Count")
        axes[i].legend(fontsize=9)
fig.suptitle("Prediction Confidence Distribution by True Class", fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(eval_dir / "confidence_distribution.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("  Saved confidence_distribution.png")


# ─── 5. Per-class Example Tiles with Attention ──────────────────────────────
print("5. Per-class example tiles with attention...")
for cls_idx, cls_name in enumerate(CLASS_ORDER):
    cls_preds = [p for p in predictions if p["true_label"] == cls_idx]
    cls_preds.sort(key=lambda x: x["confidence"], reverse=True)

    # Top 4 correct, top 4 wrong
    correct = [p for p in cls_preds if p["pred_label"] == cls_idx][:4]
    wrong = [p for p in cls_preds if p["pred_label"] != cls_idx][:4]
    shown = correct + wrong

    if not shown:
        print(f"  Skipping {cls_name} (no predictions)")
        continue

    n_cols = min(4, len(shown))
    n_rows = max(1, (len(shown) + n_cols - 1) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4.5 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    for idx, pred in enumerate(shown):
        r, c = idx // n_cols, idx % n_cols
        ax = axes[r, c]
        img_id = pred["image_id"]

        tile_files = get_tile_files(img_id)
        if tile_files:
            att_weights = pred.get("attention_weights", [])
            if att_weights and len(att_weights) <= len(tile_files):
                top_idx = int(np.argmax(att_weights[:len(tile_files)]))
                tile_img = Image.open(tile_files[top_idx])
                att_val = att_weights[top_idx]
            else:
                tile_img = Image.open(tile_files[len(tile_files) // 2])
                att_val = None
            ax.imshow(tile_img, cmap="gray")
            if att_val is not None:
                ax.text(0.02, 0.98, f"att={att_val:.4f}", transform=ax.transAxes,
                        fontsize=7, va="top", color="yellow",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7))

        is_correct = pred["pred_label"] == pred["true_label"]
        color = "green" if is_correct else "red"
        pred_name = CLASS_ORDER[pred["pred_label"]]
        title = f"{img_id[:24]}\nPred: {pred_name} ({pred['confidence']:.2f})"
        if not is_correct:
            title += f"\nTrue: {cls_name}"
        ax.set_title(title, fontsize=8, color=color, fontweight="bold")
        ax.axis("off")

    for idx in range(len(shown), n_rows * n_cols):
        r, c = idx // n_cols, idx % n_cols
        axes[r, c].axis("off")

    fig.suptitle(
        f"Class: {cls_name} — Highest-attention tiles (green=correct, red=wrong)",
        fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(eval_dir / f"examples_{cls_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved examples_{cls_name}.png")


# ─── 6. Attention Heatmap — Top Predicted Images per Class ───────────────────
print("6. Attention heatmaps...")
for cls_idx, cls_name in enumerate(CLASS_ORDER):
    # Get correctly classified images for this class
    cls_correct = [p for p in predictions
                   if p["true_label"] == cls_idx and p["pred_label"] == cls_idx]
    cls_correct.sort(key=lambda x: x["confidence"], reverse=True)

    if not cls_correct:
        continue

    # Take top 2 most confident correct predictions
    for rank, pred in enumerate(cls_correct[:2]):
        img_id = pred["image_id"]
        att_weights = pred.get("attention_weights", [])
        tile_files = get_tile_files(img_id)

        if not tile_files or not att_weights:
            continue

        n_tiles = min(len(att_weights), len(tile_files))
        weights = np.array(att_weights[:n_tiles])
        # Normalize weights for visualization
        w_min, w_max = weights.min(), weights.max()
        if w_max > w_min:
            w_norm = (weights - w_min) / (w_max - w_min)
        else:
            w_norm = np.ones_like(weights)

        # Sort by attention weight (highest first)
        sorted_idx = np.argsort(weights)[::-1]

        # Show top 8 and bottom 4 tiles
        n_show = min(12, n_tiles)
        show_idx = list(sorted_idx[:8]) + list(sorted_idx[-min(4, n_tiles):])
        show_idx = show_idx[:n_show]

        n_cols_h = min(4, n_show)
        n_rows_h = max(1, (n_show + n_cols_h - 1) // n_cols_h)
        fig, axes = plt.subplots(n_rows_h, n_cols_h, figsize=(4 * n_cols_h, 4.5 * n_rows_h))
        if n_rows_h == 1 and n_cols_h == 1:
            axes = np.array([[axes]])
        elif n_rows_h == 1:
            axes = axes[np.newaxis, :]
        elif n_cols_h == 1:
            axes = axes[:, np.newaxis]

        for pos, tile_idx in enumerate(show_idx):
            r, c = pos // n_cols_h, pos % n_cols_h
            ax = axes[r, c]
            if tile_idx < len(tile_files):
                tile_img = Image.open(tile_files[tile_idx])
                ax.imshow(tile_img, cmap="gray")
            w = weights[tile_idx]
            border_color = plt.cm.hot(w_norm[tile_idx])
            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(4)
            ax.set_title(f"Tile {tile_idx} | att={w:.5f}", fontsize=8,
                         color="red" if w_norm[tile_idx] > 0.7 else "black")
            ax.set_xticks([])
            ax.set_yticks([])

        for pos in range(len(show_idx), n_rows_h * n_cols_h):
            r, c = pos // n_cols_h, pos % n_cols_h
            axes[r, c].axis("off")

        fig.suptitle(
            f"Attention Heatmap: {img_id} (True: {cls_name}, Conf: {pred['confidence']:.3f})\n"
            f"Top rows = high attention, bottom = low attention",
            fontsize=11, fontweight="bold")
        fig.tight_layout()
        fig.savefig(eval_dir / f"attention_{cls_name}_{rank}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved attention_{cls_name}_{rank}.png")


# ─── 7. t-SNE of Embeddings ─────────────────────────────────────────────────
print("7. t-SNE of test predictions...")
try:
    from sklearn.manifold import TSNE

    # Extract probabilities as feature vectors for t-SNE
    probs = np.array([p["probabilities"] for p in predictions])
    true_labels = [p["true_label"] for p in predictions]
    pred_labels = [p["pred_label"] for p in predictions]

    if len(probs) > 10:
        perplexity = min(30, len(probs) - 1)
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
        coords = tsne.fit_transform(probs)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # By TRUE label
        for cls_idx, cls_name in enumerate(CLASS_ORDER):
            mask = [i for i, l in enumerate(true_labels) if l == cls_idx]
            if mask:
                ax1.scatter(coords[mask, 0], coords[mask, 1],
                            c=CLASS_COLORS[cls_name], label=cls_name,
                            s=40, alpha=0.7, edgecolors="white", linewidths=0.5)
        ax1.set_title("t-SNE by True Label", fontsize=13)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.2)

        # By PREDICTED label (correct vs wrong)
        correct_mask = [i for i in range(len(predictions)) if predictions[i]["pred_label"] == predictions[i]["true_label"]]
        wrong_mask = [i for i in range(len(predictions)) if predictions[i]["pred_label"] != predictions[i]["true_label"]]
        if correct_mask:
            ax2.scatter(coords[correct_mask, 0], coords[correct_mask, 1],
                        c="green", label="Correct", s=40, alpha=0.6, edgecolors="white", linewidths=0.5)
        if wrong_mask:
            ax2.scatter(coords[wrong_mask, 0], coords[wrong_mask, 1],
                        c="red", label="Wrong", s=40, alpha=0.6, marker="x", linewidths=1.5)
        ax2.set_title("t-SNE by Prediction Correctness", fontsize=13)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.2)

        fig.suptitle("t-SNE of MIL Class Probabilities (Test Set)", fontsize=14)
        fig.tight_layout()
        fig.savefig(eval_dir / "tsne_predictions.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print("  Saved tsne_predictions.png")
except Exception as e:
    print(f"  t-SNE failed: {e}")


# ─── 8. Label Distribution Comparison ────────────────────────────────────────
print("8. Label distribution comparison...")
# Load split data
split_data = json.loads((model_dir / "data_split.json").read_text())
labels = json.loads((ROOT / "Data/HiRISE/v2_output/labels_simple.json").read_text())

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (split_name, ids) in zip(axes, [("train", split_data["train_ids"]),
                                          ("val", split_data["val_ids"]),
                                          ("test", split_data["test_ids"])]):
    class_counts = Counter(labels.get(i, "UNK") for i in ids)
    counts = [class_counts.get(c, 0) for c in CLASS_ORDER]
    colors = [CLASS_COLORS[c] for c in CLASS_ORDER]
    bars = ax.bar(CLASS_ORDER, counts, color=colors, edgecolor="white")
    for bar, val in zip(bars, counts):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    str(val), ha="center", fontsize=10)
    ax.set_title(f"{split_name.title()} (n={len(ids)})", fontsize=12)
    ax.set_ylabel("Count" if split_name == "train" else "")
fig.suptitle("Label Distribution per Split (Levy 2014 + Hepburn + Pearson Labels)", fontsize=13)
fig.tight_layout()
fig.savefig(eval_dir / "label_distribution.png", dpi=200)
plt.close(fig)
print("  Saved label_distribution.png")


# ─── 9. Summary Card ─────────────────────────────────────────────────────────
print("9. Summary card...")
fig = plt.figure(figsize=(12, 8))
gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.3)

# Top-left: metrics table
ax = fig.add_subplot(gs[0, 0])
ax.axis("off")
table_data = []
for i, cls in enumerate(CLASS_ORDER):
    table_data.append([cls, f"{metrics['precision'][i]:.3f}",
                        f"{metrics['recall'][i]:.3f}", f"{metrics['f1'][i]:.3f}",
                        str(int(metrics['support'][i]))])
table = ax.table(cellText=table_data,
                 colLabels=["Class", "Precision", "Recall", "F1", "Support"],
                 loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.0, 1.4)
for i in range(len(CLASS_ORDER)):
    table[i + 1, 0].set_facecolor(CLASS_COLORS[CLASS_ORDER[i]])
    table[i + 1, 0].set_text_props(color="white", fontweight="bold")
ax.set_title("Test Metrics", fontsize=12, fontweight="bold", pad=20)

# Top-right: F1 bars
ax2 = fig.add_subplot(gs[0, 1])
colors = [CLASS_COLORS[c] for c in CLASS_ORDER]
ax2.barh(CLASS_ORDER[::-1], f1_scores[::-1], color=colors[::-1])
ax2.set_xlim(0, 1)
ax2.set_title("F1 Score by Class", fontsize=12, fontweight="bold")
for i, v in enumerate(f1_scores[::-1]):
    ax2.text(v + 0.02, i, f"{v:.3f}", va="center", fontsize=10)

# Bottom: text summary
ax3 = fig.add_subplot(gs[1, :])
ax3.axis("off")
total_test = sum(metrics["support"])
total_correct = sum(cm[i][i] for i in range(len(CLASS_ORDER)))
accuracy = total_correct / total_test if total_test > 0 else 0

summary_text = (
    f"MarsLandformNet V2 — Classification Results\n"
    f"{'=' * 50}\n"
    f"Model: AttentionMIL + FocalLoss (γ=2.0, ε=0.1)\n"
    f"Features: DINOv2-B/14 (frozen) + MOLA topography (23-dim)\n"
    f"Labels: Levy 2014 polygons + Hepburn SGLF + Pearson brain terrain\n"
    f"{'─' * 50}\n"
    f"Test Accuracy: {accuracy:.1%} ({int(total_correct)}/{int(total_test)})\n"
    f"Macro F1 (all):      {metrics['macro_f1_all']:.4f}\n"
    f"Landform Macro F1:   {metrics['landform_macro_f1']:.4f}\n"
    f"Best Epoch: {best_epoch} / {len(curves['train_loss'])}\n"
    f"{'─' * 50}\n"
    f"Key improvement: Levy 2014 polygon labels matched 372/639 images\n"
    f"LDA F1: 0.28 → {metrics['f1'][0]:.3f}  (polygon labels + focal loss)"
)
ax3.text(0.05, 0.95, summary_text, transform=ax3.transAxes,
         fontsize=10, va="top", fontfamily="monospace",
         bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", edgecolor="gray"))

fig.suptitle("Mars HiRISE Mid-Latitude Landform Classification", fontsize=14, fontweight="bold", y=0.98)
fig.savefig(eval_dir / "summary_card.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("  Saved summary_card.png")

print(f"\n{'=' * 60}")
print(f"All visualizations saved to: {eval_dir}")
print(f"Total files: {len(list(eval_dir.glob('*.png')))}")
print(f"{'=' * 60}")
