#!/usr/bin/env python3
"""
Post-SSL Pipeline: Re-embed tiles with LoRA weights → Retrain MIL → Predict → Export

Usage (after downloading LoRA weights from Colab):
    python scripts/marslandform_v2/run_after_ssl.py --lora-path /path/to/marslandform_lora_weights

This script:
  1. Re-extracts DINOv2 embeddings using LoRA-finetuned backbone
  2. Aggregates per-tile embeddings into per-image embeddings  
  3. Retrains the MultiHead MIL classifier on SSL-enhanced embeddings
  4. Runs predictions (fast + agent modes)
  5. Generates comparison visualizations (frozen vs SSL)
  6. Exports GeoJSON
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/disk1/cspark/MarsLab")
sys.path.insert(0, str(ROOT))

from scripts.marslandform_v2.config import V2_OUTPUT, METADATA_JSON, get_config


def stage_reembed(lora_path: Path) -> Path:
    """Re-extract embeddings with LoRA backbone, then aggregate per-image."""
    import torch
    from scripts.marslandform_v2.models.embedder import load_model, extract_embeddings

    tiles_dir = V2_OUTPUT / "tiles"
    ssl_emb_dir = V2_OUTPUT / "embeddings_ssl"
    ssl_emb_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"STAGE 1: Re-embed tiles with LoRA weights")
    print(f"  LoRA path: {lora_path}")
    print(f"  Tiles dir: {tiles_dir}")
    print(f"  Output: {ssl_emb_dir}")
    print(f"{'='*60}\n")

    # Load LoRA model
    model = load_model(lora_path)
    device = next(model.parameters()).device
    print(f"Model loaded on {device}")

    # Extract flat embeddings
    batch_size = 64 if torch.cuda.is_available() else 16
    npy_path, csv_path = extract_embeddings(
        model=model,
        image_dir=tiles_dir,
        output_dir=ssl_emb_dir,
        batch_size=batch_size,
    )
    print(f"Flat embeddings saved: {npy_path}")

    # Aggregate per-image (tiles → image bags)
    print("\nAggregating tile embeddings per image...")
    flat_embeddings = np.load(npy_path)
    
    import csv
    tile_paths = []
    with open(csv_path) as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            tile_paths.append(row[1])

    # Group by parent directory (image_id)
    image_embeddings = defaultdict(list)
    for i, tile_path in enumerate(tile_paths):
        p = Path(tile_path)
        # tile_path looks like .../tiles/IMAGE_ID/tile_000.jpg
        image_id = p.parent.name
        image_embeddings[image_id].append(flat_embeddings[i])

    # Stack into per-image arrays
    per_image = {}
    for image_id, tile_embs in image_embeddings.items():
        per_image[image_id] = np.stack(tile_embs, axis=0).astype(np.float32)

    out_path = ssl_emb_dir / "embeddings_by_image.npy"
    np.save(out_path, per_image)
    print(f"Per-image embeddings saved: {out_path}")
    print(f"  Images: {len(per_image)}")
    first_key = next(iter(per_image))
    print(f"  Example: {first_key} → {per_image[first_key].shape}")

    return ssl_emb_dir


def stage_retrain_mil(ssl_emb_dir: Path) -> Path:
    """Retrain MIL classifier with SSL-enhanced embeddings."""
    import subprocess

    output_dir = V2_OUTPUT / "models" / "ssl_multihead"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"STAGE 2: Retrain MIL classifier")
    print(f"  Embeddings: {ssl_emb_dir / 'embeddings_by_image.npy'}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}\n")

    cmd = [
        sys.executable, "-m", "scripts.marslandform_v2.models.mil_classifier",
        "--embeddings_dir", str(ssl_emb_dir),
        "--mola_path", str(V2_OUTPUT / "mola_features_by_image.npy"),
        "--labels_path", str(V2_OUTPUT / "labels_simple.json"),
        "--output_dir", str(output_dir),
        "--epochs", "100",
        "--patience", "20",
        "--seed", "42",
    ]

    start = time.time()
    result = subprocess.run(cmd, cwd=str(ROOT))
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"✗ MIL training FAILED (exit code {result.returncode}) after {elapsed:.1f}s")
        return output_dir

    print(f"✓ MIL training completed in {elapsed:.1f}s")

    # Print test results
    metrics_path = output_dir / "test_metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        print(f"\n  Macro F1: {metrics['macro_f1_all']:.4f}")
        print(f"  Landform F1: {metrics['landform_macro_f1']:.4f}")
        class_order = ["LDA", "LVF", "CCF", "GLF", "BACKGROUND"]
        for i, cls in enumerate(class_order):
            print(f"    {cls}: P={metrics['precision'][i]:.3f} R={metrics['recall'][i]:.3f} F1={metrics['f1'][i]:.3f}")

    return output_dir


def stage_predict_and_export(model_dir: Path):
    """Run predictions with the SSL model and export GeoJSON."""
    import argparse as ap
    from scripts.marslandform_v2.run_pipeline import stage_predict, stage_export

    print(f"\n{'='*60}")
    print(f"STAGE 3: Predict + Export")
    print(f"{'='*60}\n")

    args = ap.Namespace(
        mode="fast", limit=None, batch_size=32, model_path=None,
        epochs=100, patience=15, stop_on_failure=False,
    )
    stage_predict(args)
    stage_export(args)


def stage_comparison(ssl_model_dir: Path):
    """Generate 4-way comparison: baseline vs cleaned_focal vs multihead vs ssl_multihead."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eval_dir = V2_OUTPUT / "eval"
    class_order = ["LDA", "LVF", "CCF", "GLF", "BACKGROUND"]

    print(f"\n{'='*60}")
    print(f"STAGE 4: Generate comparison report")
    print(f"{'='*60}\n")

    variants = {}
    variant_dirs = {
        "Baseline\n(frozen, CE)": V2_OUTPUT / "models" / "frozen_baseline",
        "V2\n(Focal+Levy)": V2_OUTPUT / "models" / "cleaned_focal",
        "V3\n(MultiHead)": V2_OUTPUT / "models" / "multihead_improved",
        "V4\n(SSL+MultiHead)": ssl_model_dir,
    }

    for name, d in variant_dirs.items():
        mp = d / "test_metrics.json"
        if mp.exists():
            variants[name] = json.loads(mp.read_text())
            print(f"  {name.replace(chr(10), ' ')}: landform_f1={variants[name]['landform_macro_f1']:.4f}")

    if len(variants) < 2:
        print("Not enough variants for comparison")
        return

    colors = ["#95a5a6", "#2980b9", "#e74c3c", "#27ae60"]
    variant_names = list(variants.keys())

    # F1 comparison
    fig, ax = plt.subplots(figsize=(16, 7))
    x = np.arange(len(class_order))
    n = len(variant_names)
    width = 0.8 / n
    for i, (name, metrics) in enumerate(variants.items()):
        offset = (i - n / 2 + 0.5) * width
        bars = ax.bar(x + offset, metrics["f1"], width, label=name.replace("\n", " "),
                      color=colors[i % len(colors)], edgecolor="white")
        for bar, val in zip(bars, metrics["f1"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", fontsize=7, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(class_order, fontsize=12)
    ax.set_ylabel("F1 Score", fontsize=13)
    ax.set_title("Per-Class F1: Full Ablation (Baseline → SSL)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(eval_dir / "ablation_full_f1.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Macro F1 progression
    fig, ax = plt.subplots(figsize=(12, 6))
    lf1 = [variants[n]["landform_macro_f1"] for n in variant_names]
    bars = ax.bar(range(len(variant_names)), lf1,
                  color=colors[:len(variant_names)], edgecolor="white")
    for bar, val in zip(bars, lf1):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.4f}", ha="center", fontsize=13, fontweight="bold")
    ax.set_xticks(range(len(variant_names)))
    ax.set_xticklabels([n.replace("\n", " ") for n in variant_names], fontsize=10)
    ax.set_ylabel("Landform F1", fontsize=13)
    ax.set_title("Landform F1 Progression: Baseline → SSL LoRA", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 0.9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(eval_dir / "landform_f1_progression.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Save JSON
    ablation = {
        name.replace("\n", " "): {
            "macro_f1": m["macro_f1_all"],
            "landform_f1": m["landform_macro_f1"],
            "per_class_f1": {class_order[i]: m["f1"][i] for i in range(len(class_order))},
        }
        for name, m in variants.items()
    }
    (eval_dir / "ablation_full.json").write_text(json.dumps(ablation, indent=2))

    print(f"\n  Saved: ablation_full_f1.png, landform_f1_progression.png, ablation_full.json")


def main():
    parser = argparse.ArgumentParser(description="Post-SSL pipeline: re-embed → retrain → predict → export")
    parser.add_argument("--lora-path", type=Path, required=True,
                        help="Path to LoRA weights directory (from Colab) or .pt checkpoint")
    parser.add_argument("--skip-embed", action="store_true",
                        help="Skip re-embedding (use existing SSL embeddings)")
    args = parser.parse_args()

    if not args.lora_path.exists():
        print(f"ERROR: LoRA path not found: {args.lora_path}")
        sys.exit(1)

    print("=" * 60)
    print("MarsLandformNet — Post-SSL Pipeline")
    print("=" * 60)

    # Stage 1: Re-embed
    ssl_emb_dir = V2_OUTPUT / "embeddings_ssl"
    if not args.skip_embed:
        ssl_emb_dir = stage_reembed(args.lora_path)
    else:
        print(f"Skipping re-embedding, using: {ssl_emb_dir}")

    # Stage 2: Retrain MIL
    model_dir = stage_retrain_mil(ssl_emb_dir)

    # Stage 3: Predict + Export
    stage_predict_and_export(model_dir)

    # Stage 4: Comparison
    stage_comparison(model_dir)

    print("\n" + "=" * 60)
    print("POST-SSL PIPELINE COMPLETE")
    print("=" * 60)
    metrics_path = model_dir / "test_metrics.json"
    if metrics_path.exists():
        m = json.loads(metrics_path.read_text())
        print(f"  Final Landform F1: {m['landform_macro_f1']:.4f}")
        print(f"  Final Macro F1:    {m['macro_f1_all']:.4f}")
    print(f"\n  Visualizations: {V2_OUTPUT / 'eval'}")
    print(f"  GeoJSON:        {V2_OUTPUT / 'geojson'}")
    print(f"  Predictions:    {V2_OUTPUT / 'predictions'}")


if __name__ == "__main__":
    main()
