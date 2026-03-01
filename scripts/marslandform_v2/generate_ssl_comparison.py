#!/usr/bin/env python3
"""Generate SSL vs Frozen comparison charts from MIL training logs.

Parses /tmp/mil_frozen30.log and /tmp/mil_ssl30.log to produce:
1. Convergence plot (loss + F1 over epochs)
2. Final bar chart comparison
3. Per-class F1 comparison
"""
import re
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path("/disk1/cspark/MarsLab")
EVAL_DIR = ROOT / "Data/HiRISE/v2_output/eval"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

CLASS_ORDER = ["LDA", "LVF", "CCF", "GLF", "BACKGROUND"]


def parse_training_log(log_path: str) -> dict:
    """Parse MIL training log into structured epoch data.
    
    Expected log format per epoch:
    [Val][Epoch N] loss=X.XXXX macro_f1=X.XXXX landform_macro_f1=X.XXXX
    Class     Precision  Recall  F1   Support
    LDA          X.XXX   X.XXX  X.XXX       NN
    ...
    """
    log_path = Path(log_path)
    if not log_path.exists():
        print(f"  Log not found: {log_path}")
        return {"epochs": [], "loss": [], "macro_f1": [], "landform_f1": [], "per_class_f1": {}}
    
    text = log_path.read_text()
    
    # Parse epoch summary lines
    epoch_pattern = re.compile(
        r'\[Val\]\[Epoch\s+(\d+)\]\s+loss=([\d.]+)\s+macro_f1=([\d.]+)\s+landform_macro_f1=([\d.]+)'
    )
    
    # Parse per-class lines
    class_pattern = re.compile(
        r'^(LDA|LVF|CCF|GLF|BACKGROUND)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)',
        re.MULTILINE
    )
    
    epochs = []
    losses = []
    macro_f1s = []
    landform_f1s = []
    per_class_f1 = {cls: [] for cls in CLASS_ORDER}
    
    # Split by epoch blocks
    lines = text.split('\n')
    current_epoch_classes = {}
    current_epoch_num = None
    
    for line in lines:
        epoch_match = epoch_pattern.search(line)
        if epoch_match:
            # Save previous epoch's class data if exists
            if current_epoch_num is not None and current_epoch_classes:
                for cls in CLASS_ORDER:
                    per_class_f1[cls].append(current_epoch_classes.get(cls, 0.0))
            
            current_epoch_num = int(epoch_match.group(1))
            epochs.append(current_epoch_num)
            losses.append(float(epoch_match.group(2)))
            macro_f1s.append(float(epoch_match.group(3)))
            landform_f1s.append(float(epoch_match.group(4)))
            current_epoch_classes = {}
            continue
        
        class_match = class_pattern.search(line)
        if class_match and current_epoch_num is not None:
            cls_name = class_match.group(1)
            f1_val = float(class_match.group(4))
            current_epoch_classes[cls_name] = f1_val
    
    # Don't forget last epoch
    if current_epoch_num is not None and current_epoch_classes:
        for cls in CLASS_ORDER:
            per_class_f1[cls].append(current_epoch_classes.get(cls, 0.0))
    
    print(f"  Parsed {len(epochs)} epochs from {log_path.name}")
    if epochs:
        print(f"    Last epoch {epochs[-1]}: loss={losses[-1]:.4f}, macro_f1={macro_f1s[-1]:.4f}, landform_f1={landform_f1s[-1]:.4f}")
    
    return {
        "epochs": epochs,
        "loss": losses,
        "macro_f1": macro_f1s,
        "landform_f1": landform_f1s,
        "per_class_f1": per_class_f1,
    }


def plot_convergence(frozen: dict, ssl: dict, save_path: Path):
    """Plot side-by-side convergence: loss and F1."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # 1. Loss convergence
    ax = axes[0]
    if frozen["epochs"]:
        ax.plot(frozen["epochs"], frozen["loss"], label="Frozen DINOv2", color="#3498db", linewidth=2)
    if ssl["epochs"]:
        ax.plot(ssl["epochs"], ssl["loss"], label="SSL LoRA (14ep)", color="#e74c3c", linewidth=2)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Validation Loss", fontsize=12)
    ax.set_title("Loss Convergence", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    # 2. Landform F1 convergence
    ax = axes[1]
    if frozen["epochs"]:
        ax.plot(frozen["epochs"], frozen["landform_f1"], label="Frozen DINOv2", color="#3498db", linewidth=2)
        best_idx = np.argmax(frozen["landform_f1"])
        ax.scatter([frozen["epochs"][best_idx]], [frozen["landform_f1"][best_idx]], 
                   color="#3498db", s=100, zorder=5, marker="*")
        ax.annotate(f'{frozen["landform_f1"][best_idx]:.3f}', 
                   (frozen["epochs"][best_idx], frozen["landform_f1"][best_idx]),
                   textcoords="offset points", xytext=(10, 5), fontsize=9, color="#3498db", fontweight="bold")
    if ssl["epochs"]:
        ax.plot(ssl["epochs"], ssl["landform_f1"], label="SSL LoRA (14ep)", color="#e74c3c", linewidth=2)
        best_idx = np.argmax(ssl["landform_f1"])
        ax.scatter([ssl["epochs"][best_idx]], [ssl["landform_f1"][best_idx]], 
                   color="#e74c3c", s=100, zorder=5, marker="*")
        ax.annotate(f'{ssl["landform_f1"][best_idx]:.3f}', 
                   (ssl["epochs"][best_idx], ssl["landform_f1"][best_idx]),
                   textcoords="offset points", xytext=(10, -10), fontsize=9, color="#e74c3c", fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Landform Macro F1", fontsize=12)
    ax.set_title("Landform F1 Convergence", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 0.7)
    
    # 3. Macro F1 convergence  
    ax = axes[2]
    if frozen["epochs"]:
        ax.plot(frozen["epochs"], frozen["macro_f1"], label="Frozen DINOv2", color="#3498db", linewidth=2)
    if ssl["epochs"]:
        ax.plot(ssl["epochs"], ssl["macro_f1"], label="SSL LoRA (14ep)", color="#e74c3c", linewidth=2)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Macro F1 (all classes)", fontsize=12)
    ax.set_title("Overall Macro F1 Convergence", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 0.6)
    
    fig.suptitle("SSL LoRA (14 epochs) vs Frozen DINOv2 — MIL Training Convergence\n(30 tiles/image, same architecture)", 
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path.name}")


def plot_final_comparison(frozen: dict, ssl: dict, save_path: Path):
    """Bar chart: final epoch F1 per class + overall."""
    if not frozen["epochs"] or not ssl["epochs"]:
        print("  Skipping final comparison — insufficient data")
        return
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Get last epoch per-class F1
    classes = CLASS_ORDER[:4]  # LDA, LVF, CCF, GLF only
    frozen_f1 = [frozen["per_class_f1"][c][-1] if frozen["per_class_f1"][c] else 0 for c in classes]
    ssl_f1 = [ssl["per_class_f1"][c][-1] if ssl["per_class_f1"][c] else 0 for c in classes]
    
    # Add overall landform F1
    classes_ext = classes + ["LANDFORM\n(macro)"]
    frozen_f1.append(frozen["landform_f1"][-1])
    ssl_f1.append(ssl["landform_f1"][-1])
    
    x = np.arange(len(classes_ext))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, frozen_f1, width, label="Frozen DINOv2", color="#3498db", edgecolor="white")
    bars2 = ax.bar(x + width/2, ssl_f1, width, label="SSL LoRA (14ep)", color="#e74c3c", edgecolor="white")
    
    for bar, val in zip(bars1, frozen_f1):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{val:.3f}",
                ha="center", fontsize=10, fontweight="bold", color="#2c3e50")
    for bar, val in zip(bars2, ssl_f1):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{val:.3f}",
                ha="center", fontsize=10, fontweight="bold", color="#c0392b")
    
    ax.set_xticks(x)
    ax.set_xticklabels(classes_ext, fontsize=12)
    ax.set_ylabel("F1 Score", fontsize=13)
    ax.set_title("SSL LoRA (14ep) vs Frozen DINOv2 — Per-Class F1\n(30 tiles/image, same MIL architecture)", 
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_ylim(0, 0.75)
    ax.grid(axis="y", alpha=0.3)
    
    # Add verdict box
    frozen_lf = frozen["landform_f1"][-1]
    ssl_lf = ssl["landform_f1"][-1]
    delta = ssl_lf - frozen_lf
    verdict = "SSL WINS ✓" if delta > 0 else "Frozen WINS ✓ (SSL needs more training)"
    box_color = "#27ae60" if delta > 0 else "#e67e22"
    ax.text(0.98, 0.95, f"Δ Landform F1: {delta:+.3f}\n{verdict}", 
            transform=ax.transAxes, fontsize=11, fontweight="bold",
            va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.5", facecolor=box_color, alpha=0.2))
    
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path.name}")


def plot_full_model_comparison(frozen30: dict, ssl30: dict, save_path: Path):
    """5-model comparison: V1 baseline, V2, V3, Frozen-30, SSL-30."""
    
    # Load V1, V2, V3 from saved metrics
    model_dirs = {
        "V1: Baseline": ROOT / "Data/HiRISE/v2_output/models/frozen_baseline",
        "V2: Focal+Levy": ROOT / "Data/HiRISE/v2_output/models/cleaned_focal",
        "V3: MultiHead\n(128 tiles)": ROOT / "Data/HiRISE/v2_output/models/multihead_improved",
    }
    
    models = {}
    for name, d in model_dirs.items():
        mp = d / "test_metrics.json"
        if mp.exists():
            m = json.loads(mp.read_text())
            models[name] = {"landform_f1": m["landform_macro_f1"], "macro_f1": m["macro_f1_all"],
                           "per_class_f1": {CLASS_ORDER[i]: m["f1"][i] for i in range(len(CLASS_ORDER))}}
    
    # Add frozen-30 and SSL-30 from logs
    if frozen30["epochs"]:
        best_idx = max(range(len(frozen30["landform_f1"])), key=lambda i: frozen30["landform_f1"][i])
        models["Frozen-30\n(30 tiles)"] = {
            "landform_f1": frozen30["landform_f1"][best_idx],
            "macro_f1": frozen30["macro_f1"][best_idx],
            "per_class_f1": {c: frozen30["per_class_f1"][c][best_idx] if best_idx < len(frozen30["per_class_f1"][c]) else 0 
                           for c in CLASS_ORDER}
        }
    
    if ssl30["epochs"]:
        best_idx = max(range(len(ssl30["landform_f1"])), key=lambda i: ssl30["landform_f1"][i])
        models["SSL-30\n(14ep LoRA)"] = {
            "landform_f1": ssl30["landform_f1"][best_idx],
            "macro_f1": ssl30["macro_f1"][best_idx],
            "per_class_f1": {c: ssl30["per_class_f1"][c][best_idx] if best_idx < len(ssl30["per_class_f1"][c]) else 0 
                           for c in CLASS_ORDER}
        }
    
    if len(models) < 2:
        print("  Not enough models for full comparison")
        return
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    colors = ["#95a5a6", "#2980b9", "#e74c3c", "#27ae60", "#f39c12"]
    
    # Left: Landform F1 + Macro F1 per model
    names = list(models.keys())
    lf_f1s = [models[n]["landform_f1"] for n in names]
    macro_f1s = [models[n]["macro_f1"] for n in names]
    
    x = np.arange(len(names))
    width = 0.35
    bars1 = ax1.bar(x - width/2, lf_f1s, width, label="Landform F1", color="#e74c3c", edgecolor="white")
    bars2 = ax1.bar(x + width/2, macro_f1s, width, label="Macro F1", color="#3498db", edgecolor="white")
    
    for bar, val in zip(bars1, lf_f1s):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{val:.3f}",
                ha="center", fontsize=9, fontweight="bold")
    for bar, val in zip(bars2, macro_f1s):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{val:.3f}",
                ha="center", fontsize=9, fontweight="bold")
    
    ax1.set_xticks(x)
    ax1.set_xticklabels([n.replace('\n', ' ') for n in names], fontsize=8, rotation=15)
    ax1.set_ylabel("F1 Score", fontsize=12)
    ax1.set_title("All Models: Landform & Macro F1", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.set_ylim(0, 0.75)
    ax1.grid(axis="y", alpha=0.3)
    
    # Right: Per-class F1 for landform classes only (best 3 models)
    # Sort by landform F1, take top 3
    top_models = sorted(models.items(), key=lambda x: x[1]["landform_f1"], reverse=True)[:3]
    classes = CLASS_ORDER[:4]  # LDA, LVF, CCF, GLF
    
    x2 = np.arange(len(classes))
    n = len(top_models)
    width2 = 0.8 / n
    top_colors = ["#e74c3c", "#3498db", "#27ae60"]
    
    for i, (name, data) in enumerate(top_models):
        offset = (i - n/2 + 0.5) * width2
        f1s = [data["per_class_f1"].get(c, 0) for c in classes]
        clean_name = name.replace('\n', ' ')
        bars = ax2.bar(x2 + offset, f1s, width2, label=f"{clean_name} (LF={data['landform_f1']:.3f})", 
                      color=top_colors[i], edgecolor="white", alpha=0.85)
        for bar, val in zip(bars, f1s):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{val:.2f}",
                    ha="center", fontsize=7, fontweight="bold")
    
    ax2.set_xticks(x2)
    ax2.set_xticklabels(classes, fontsize=12)
    ax2.set_ylabel("F1 Score", fontsize=12)
    ax2.set_title("Top 3 Models: Per-Class F1", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.set_ylim(0, 0.8)
    ax2.grid(axis="y", alpha=0.3)
    
    fig.suptitle("MarsLandformNet — Full Model Comparison", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path.name}")


def main():
    print("=" * 60)
    print("Generating SSL vs Frozen comparison charts")
    print("=" * 60)
    
    # Parse logs
    print("\nParsing training logs...")
    frozen = parse_training_log("/tmp/mil_frozen30.log")
    ssl = parse_training_log("/tmp/mil_ssl30.log")
    
    # 1. Convergence plot
    print("\n1. Convergence chart...")
    plot_convergence(frozen, ssl, EVAL_DIR / "ssl_vs_frozen_convergence.png")
    
    # 2. Final per-class comparison
    print("\n2. Final comparison chart...")
    plot_final_comparison(frozen, ssl, EVAL_DIR / "ssl_vs_frozen_final.png")
    
    # 3. Full 5-model comparison
    print("\n3. Full model comparison...")
    plot_full_model_comparison(frozen, ssl, EVAL_DIR / "model_comparison_full.png")
    
    # Save parsed data as JSON for reference
    summary = {
        "frozen_30": {
            "total_epochs": len(frozen["epochs"]),
            "best_landform_f1": max(frozen["landform_f1"]) if frozen["landform_f1"] else 0,
            "best_epoch": frozen["epochs"][np.argmax(frozen["landform_f1"])] if frozen["landform_f1"] else 0,
            "final_loss": frozen["loss"][-1] if frozen["loss"] else 0,
        },
        "ssl_30": {
            "total_epochs": len(ssl["epochs"]),
            "best_landform_f1": max(ssl["landform_f1"]) if ssl["landform_f1"] else 0,
            "best_epoch": ssl["epochs"][np.argmax(ssl["landform_f1"])] if ssl["landform_f1"] else 0,
            "final_loss": ssl["loss"][-1] if ssl["loss"] else 0,
        }
    }
    (EVAL_DIR / "ssl_vs_frozen_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved summary JSON")
    
    print(f"\n{'=' * 60}")
    print(f"All charts saved to: {EVAL_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
