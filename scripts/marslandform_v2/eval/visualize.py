from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.marslandform_v2.config import CLASS_ORDER, EVAL_DIR


def plot_confusion_matrix(
    cm: Sequence[Sequence[float]],
    class_names: Sequence[str],
    output_path: Path,
    normalize: bool = True,
) -> None:
    cm_arr = np.asarray(cm, dtype=float)
    if normalize:
        row_sums = cm_arr.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cm_display = cm_arr / row_sums
    else:
        cm_display = cm_arr

    sns.set_theme(style="white", context="paper", font_scale=1.2)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm_display,
        cmap="Blues",
        annot=True,
        fmt=".2f" if normalize else ".0f",
        square=True,
        cbar=True,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.tick_params(axis="x", rotation=45, labelsize=12)
    ax.tick_params(axis="y", rotation=0, labelsize=12)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _parse_tile_coords(tile_path: Path) -> tuple[int, int] | None:
    stem = tile_path.stem
    patterns = [
        r"(?:^|_)r(?P<row>\d+)[_\-]c(?P<col>\d+)(?:_|$)",
        r"(?:^|_)row(?P<row>\d+)[_\-]col(?P<col>\d+)(?:_|$)",
        r"(?:^|_)y(?P<row>\d+)[_\-]x(?P<col>\d+)(?:_|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, stem)
        if match is not None:
            return int(match.group("row")), int(match.group("col"))
    return None


def _to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return np.stack([image, image, image], axis=-1)
    if image.ndim == 3 and image.shape[2] == 1:
        return np.repeat(image, repeats=3, axis=2)
    if image.ndim == 3 and image.shape[2] >= 3:
        return image[:, :, :3]
    raise ValueError("Unsupported image shape for tile visualization")


def plot_attention_weights(
    tile_paths: Sequence[str | Path],
    attention_weights: Sequence[float],
    image_id: str,
    output_path: Path,
) -> None:
    if len(tile_paths) == 0:
        raise ValueError("tile_paths is empty")
    if len(tile_paths) != len(attention_weights):
        raise ValueError("tile_paths and attention_weights must have the same length")

    tiles = [Path(path) for path in tile_paths]
    coords = [_parse_tile_coords(path) for path in tiles]
    if any(coord is None for coord in coords):
        grid_side = int(np.ceil(np.sqrt(len(tiles))))
        coords = [(idx // grid_side, idx % grid_side) for idx in range(len(tiles))]

    valid_coords = [(int(coord[0]), int(coord[1])) for coord in coords]
    rows = [coord[0] for coord in valid_coords]
    cols = [coord[1] for coord in valid_coords]
    min_row, max_row = min(rows), max(rows)
    min_col, max_col = min(cols), max(cols)

    first_tile = _to_rgb(plt.imread(tiles[0]))
    tile_h, tile_w = first_tile.shape[:2]
    grid_h = (max_row - min_row + 1) * tile_h
    grid_w = (max_col - min_col + 1) * tile_w

    canvas = np.zeros((grid_h, grid_w, 3), dtype=float)
    attention_grid = np.zeros((max_row - min_row + 1, max_col - min_col + 1), dtype=float)
    coverage = np.zeros_like(attention_grid, dtype=bool)

    weights = np.asarray(attention_weights, dtype=float)
    if np.sum(weights) > 0:
        weights = weights / np.sum(weights)

    for idx, tile_path in enumerate(tiles):
        row, col = valid_coords[idx]
        rr = row - min_row
        cc = col - min_col
        image = _to_rgb(plt.imread(tile_path)).astype(float)
        if image.max() > 1.0:
            image = image / 255.0
        y0, y1 = rr * tile_h, (rr + 1) * tile_h
        x0, x1 = cc * tile_w, (cc + 1) * tile_w
        canvas[y0:y1, x0:x1, :] = image
        attention_grid[rr, cc] = weights[idx]
        coverage[rr, cc] = True

    attention_grid[~coverage] = np.nan
    heatmap = np.kron(attention_grid, np.ones((tile_h, tile_w)))

    sns.set_theme(style="white", context="paper", font_scale=1.2)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(canvas)
    im = ax.imshow(heatmap, cmap="inferno", alpha=0.45)
    ax.set_title(f"Attention Heatmap - {image_id}", fontsize=13)
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Attention Weight", fontsize=12)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_per_class_f1(metrics_dict: dict[str, Any], output_path: Path) -> None:
    per_class = metrics_dict.get("per_class_f1")
    if isinstance(per_class, dict):
        class_names = list(per_class.keys())
        f1_values = [float(per_class[name]) for name in class_names]
    elif isinstance(per_class, list):
        class_names = list(CLASS_ORDER[: len(per_class)])
        f1_values = [float(value) for value in per_class]
    elif isinstance(metrics_dict.get("f1"), list):
        class_names = list(CLASS_ORDER[: len(metrics_dict["f1"])])
        f1_values = [float(value) for value in metrics_dict["f1"]]
    else:
        raise ValueError("No per-class F1 information found in metrics_dict")

    chart_df = (
        np.array(list(zip(class_names, f1_values)), dtype=object)
        if class_names
        else np.empty((0, 2), dtype=object)
    )

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    fig, ax = plt.subplots(figsize=(8, 5))
    if chart_df.size > 0:
        sns.barplot(x=chart_df[:, 1].astype(float), y=chart_df[:, 0], orient="h", palette="viridis", ax=ax)
    ax.set_xlabel("F1 Score", fontsize=12)
    ax.set_ylabel("Class", fontsize=12)
    ax.set_xlim(0.0, 1.0)
    ax.tick_params(axis="both", labelsize=12)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(
    train_losses: Sequence[float],
    val_losses: Sequence[float],
    val_f1s: Sequence[float],
    output_path: Path,
) -> None:
    epochs = np.arange(1, len(train_losses) + 1)
    if len(val_losses) != len(train_losses) or len(val_f1s) != len(train_losses):
        raise ValueError("train_losses, val_losses, and val_f1s must have the same length")

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()

    line1 = ax1.plot(epochs, train_losses, color="#1f77b4", linewidth=2.0, label="Train Loss")
    line2 = ax1.plot(epochs, val_losses, color="#ff7f0e", linewidth=2.0, label="Val Loss")
    line3 = ax2.plot(epochs, val_f1s, color="#2ca02c", linewidth=2.0, label="Val F1")

    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax2.set_ylabel("F1", fontsize=12)
    ax1.tick_params(axis="both", labelsize=12)
    ax2.tick_params(axis="y", labelsize=12)
    ax2.set_ylim(0.0, 1.0)

    lines = line1 + line2 + line3
    labels = [str(line.get_label()) for line in lines]
    ax1.legend(lines, labels, loc="best", fontsize=11)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_tsne_embeddings(
    embeddings: np.ndarray,
    labels: Sequence[int],
    class_names: Sequence[str],
    output_path: Path,
) -> None:
    emb = np.asarray(embeddings, dtype=float)
    if emb.ndim != 2:
        raise ValueError("embeddings must be 2D with shape (n_samples, n_features)")
    labels_arr = np.asarray(labels, dtype=int)
    if emb.shape[0] != labels_arr.shape[0]:
        raise ValueError("embeddings and labels must have the same number of samples")

    tsne = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=min(30, max(5, emb.shape[0] // 5)))
    proj = tsne.fit_transform(emb)

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    fig, ax = plt.subplots(figsize=(8, 7))
    palette = sns.color_palette("tab10", n_colors=max(2, len(class_names)))

    for class_idx, class_name in enumerate(class_names):
        mask = labels_arr == class_idx
        if not np.any(mask):
            continue
        ax.scatter(
            proj[mask, 0],
            proj[mask, 1],
            s=25,
            alpha=0.8,
            label=class_name,
            color=palette[class_idx % len(palette)],
            edgecolors="none",
        )

    ax.set_xlabel("t-SNE 1", fontsize=12)
    ax.set_ylabel("t-SNE 2", fontsize=12)
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(title="Class", fontsize=11, title_fontsize=12, frameon=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create MarsLandformNet V2 evaluation figures")
    subparsers = parser.add_subparsers(dest="command", required=True)

    cm_parser = subparsers.add_parser("confusion_matrix")
    cm_parser.add_argument("--cm_json", type=Path, required=True)
    cm_parser.add_argument("--output", type=Path, default=EVAL_DIR / "confusion_matrix_paper.png")
    cm_parser.add_argument("--class_names", type=str, default=",".join(CLASS_ORDER))
    cm_parser.add_argument("--normalize", action="store_true")

    att_parser = subparsers.add_parser("attention")
    att_parser.add_argument("--tile_paths_json", type=Path, required=True)
    att_parser.add_argument("--attention_json", type=Path, required=True)
    att_parser.add_argument("--image_id", type=str, required=True)
    att_parser.add_argument("--output", type=Path, default=EVAL_DIR / "attention_heatmap.png")

    f1_parser = subparsers.add_parser("per_class_f1")
    f1_parser.add_argument("--metrics_json", type=Path, required=True)
    f1_parser.add_argument("--output", type=Path, default=EVAL_DIR / "per_class_f1.png")

    curve_parser = subparsers.add_parser("training_curves")
    curve_parser.add_argument("--curves_json", type=Path, required=True)
    curve_parser.add_argument("--output", type=Path, default=EVAL_DIR / "training_curves.png")

    tsne_parser = subparsers.add_parser("tsne")
    tsne_parser.add_argument("--embeddings_npy", type=Path, required=True)
    tsne_parser.add_argument("--labels_json", type=Path, required=True)
    tsne_parser.add_argument("--class_names", type=str, default=",".join(CLASS_ORDER))
    tsne_parser.add_argument("--output", type=Path, default=EVAL_DIR / "tsne_embeddings.png")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "confusion_matrix":
        cm = _load_json(args.cm_json)
        class_names = [name.strip() for name in args.class_names.split(",") if name.strip()]
        plot_confusion_matrix(cm=cm, class_names=class_names, output_path=args.output, normalize=args.normalize)
        print(f"Saved confusion matrix to {args.output}")
        return

    if args.command == "attention":
        tile_paths = _load_json(args.tile_paths_json)
        attention = _load_json(args.attention_json)
        plot_attention_weights(tile_paths=tile_paths, attention_weights=attention, image_id=args.image_id, output_path=args.output)
        print(f"Saved attention heatmap to {args.output}")
        return

    if args.command == "per_class_f1":
        metrics = _load_json(args.metrics_json)
        plot_per_class_f1(metrics_dict=metrics, output_path=args.output)
        print(f"Saved per-class F1 chart to {args.output}")
        return

    if args.command == "training_curves":
        curves = _load_json(args.curves_json)
        train_losses = curves["train_loss"]
        val_losses = curves["val_loss"]
        val_f1s = curves.get("val_macro_f1", curves.get("val_landform_macro_f1"))
        if val_f1s is None:
            raise KeyError("curves_json must contain val_macro_f1 or val_landform_macro_f1")
        plot_training_curves(train_losses=train_losses, val_losses=val_losses, val_f1s=val_f1s, output_path=args.output)
        print(f"Saved training curves to {args.output}")
        return

    if args.command == "tsne":
        embeddings = np.load(args.embeddings_npy)
        labels = _load_json(args.labels_json)
        class_names = [name.strip() for name in args.class_names.split(",") if name.strip()]
        plot_tsne_embeddings(
            embeddings=embeddings,
            labels=labels,
            class_names=class_names,
            output_path=args.output,
        )
        print(f"Saved t-SNE plot to {args.output}")
        return

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
