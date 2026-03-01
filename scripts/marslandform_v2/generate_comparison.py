#!/usr/bin/env python3
"""Generate 3-way ablation comparison: Baseline vs Cleaned+Focal vs MultiHead+Improved."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

ROOT = Path("/disk1/cspark/MarsLab")
CLASS_ORDER = ["LDA", "LVF", "CCF", "GLF", "BACKGROUND"]
eval_dir = ROOT / "Data/HiRISE/v2_output/eval"
eval_dir.mkdir(parents=True, exist_ok=True)

# Load all three model results
variants = {}
variant_dirs = {
    "Baseline\n(title regex, CE)": ROOT / "Data/HiRISE/v2_output/models/frozen_baseline",
    "V2: Focal+Levy2014\n(single-head)": ROOT / "Data/HiRISE/v2_output/models/cleaned_focal",
    "V3: MultiHead+MOLA\n(4-head, stratified)": ROOT / "Data/HiRISE/v2_output/models/multihead_improved",
}

for name, d in variant_dirs.items():
    metrics_path = d / "test_metrics.json"
    if metrics_path.exists():
        variants[name] = json.loads(metrics_path.read_text())
        print(f"Loaded {name}: macro_f1={variants[name]['macro_f1_all']:.4f}, landform_f1={variants[name]['landform_macro_f1']:.4f}")

if len(variants) < 2:
    print("Need at least 2 variants for comparison")
    exit(1)

variant_names = list(variants.keys())
colors = ["#95a5a6", "#2980b9", "#e74c3c"]

# ─── 1. 3-Way F1 Comparison ───────────────────────────────────────────────
print("\n1. 3-way F1 comparison chart...")
fig, ax = plt.subplots(figsize=(14, 7))
x = np.arange(len(CLASS_ORDER))
n = len(variant_names)
width = 0.8 / n

for i, (name, metrics) in enumerate(variants.items()):
    offset = (i - n/2 + 0.5) * width
    bars = ax.bar(x + offset, metrics["f1"], width, label=name, color=colors[i], edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, metrics["f1"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{val:.2f}",
                ha="center", fontsize=8, fontweight="bold", color=colors[i])

ax.set_xticks(x)
ax.set_xticklabels(CLASS_ORDER, fontsize=12)
ax.set_ylabel("F1 Score", fontsize=13)
ax.set_title("Per-Class F1 Score: 3-Way Ablation Comparison", fontsize=14, fontweight="bold")
ax.legend(fontsize=9, loc="upper right")
ax.set_ylim(0, 1.0)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(eval_dir / "ablation_3way_f1.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("  Saved ablation_3way_f1.png")

# ─── 2. Macro F1 Progression ─────────────────────────────────────────────
print("2. Macro F1 progression...")
fig, ax = plt.subplots(figsize=(10, 6))
macro_f1s = [variants[n]["macro_f1_all"] for n in variant_names]
landform_f1s = [variants[n]["landform_macro_f1"] for n in variant_names]

x = np.arange(len(variant_names))
width = 0.35
bars1 = ax.bar(x - width/2, macro_f1s, width, label="Macro F1 (all)", color="#3498db", edgecolor="white")
bars2 = ax.bar(x + width/2, landform_f1s, width, label="Landform F1", color="#e74c3c", edgecolor="white")

for bar, val in zip(bars1, macro_f1s):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{val:.3f}",
            ha="center", fontsize=11, fontweight="bold")
for bar, val in zip(bars2, landform_f1s):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{val:.3f}",
            ha="center", fontsize=11, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels([n.replace('\n', ' ') for n in variant_names], fontsize=9)
ax.set_ylabel("F1 Score", fontsize=13)
ax.set_title("Model Evolution: Macro F1 Progression", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.set_ylim(0, 0.8)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(eval_dir / "ablation_macro_f1_progression.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("  Saved ablation_macro_f1_progression.png")

# ─── 3. Delta from baseline ──────────────────────────────────────────────
print("3. Delta from baseline...")
baseline_key = variant_names[0]
best_key = variant_names[-1]
fig, ax = plt.subplots(figsize=(10, 6))
deltas = [variants[best_key]["f1"][i] - variants[baseline_key]["f1"][i] for i in range(len(CLASS_ORDER))]
bar_colors = ["#27ae60" if d >= 0 else "#e74c3c" for d in deltas]
bars = ax.bar(CLASS_ORDER, deltas, color=bar_colors, edgecolor="white", linewidth=1.5)
for bar, val in zip(bars, deltas):
    y_pos = bar.get_height() + 0.01 if val >= 0 else bar.get_height() - 0.04
    ax.text(bar.get_x() + bar.get_width()/2, y_pos, f"{val:+.3f}",
            ha="center", fontsize=13, fontweight="bold")
ax.axhline(0, color="black", linewidth=0.5)
ax.set_ylabel("ΔF1 Score", fontsize=13)
macro_delta = variants[best_key]["macro_f1_all"] - variants[baseline_key]["macro_f1_all"]
lf_delta = variants[best_key]["landform_macro_f1"] - variants[baseline_key]["landform_macro_f1"]
ax.set_title(f"F1 Improvement: V3 MultiHead vs Baseline\n"
             f"ΔMacro F1: {macro_delta:+.3f} | ΔLandform F1: {lf_delta:+.3f}",
             fontsize=13, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(eval_dir / "ablation_delta_v3.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("  Saved ablation_delta_v3.png")

# ─── 4. Training curves for new model ────────────────────────────────────
print("4. Training curves...")
curves_path = ROOT / "Data/HiRISE/v2_output/models/multihead_improved/training_curves.json"
if curves_path.exists():
    curves = json.loads(curves_path.read_text())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(curves["train_loss"]) + 1)
    ax1.plot(epochs, curves["train_loss"], label="Train Loss", color="#e74c3c", linewidth=1.5)
    ax1.plot(epochs, curves["val_loss"], label="Val Loss", color="#3498db", linewidth=1.5)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curves (MultiHead V3)", fontsize=13)
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(epochs, curves["val_macro_f1"], label="Val Macro F1", color="#2ecc71", linewidth=1.5)
    ax2.plot(epochs, curves["val_landform_macro_f1"], label="Val Landform F1", color="#e74c3c", linewidth=1.5)
    best_epoch = np.argmax(curves["val_landform_macro_f1"]) + 1
    best_f1 = max(curves["val_landform_macro_f1"])
    ax2.axvline(best_epoch, color="gray", linestyle="--", alpha=0.5, label=f"Best epoch ({best_epoch})")
    ax2.scatter([best_epoch], [best_f1], color="red", s=100, zorder=5, marker="*")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("F1 Score")
    ax2.set_title(f"F1 Curves — Best Landform F1: {best_f1:.4f} @ Epoch {best_epoch}", fontsize=13)
    ax2.legend(); ax2.grid(alpha=0.3)

    fig.suptitle("MultiHead V3: Training Progress", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(eval_dir / "training_curves_v3.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved training_curves_v3.png")

# ─── 5. Save ablation JSON ───────────────────────────────────────────────
print("5. Saving ablation JSON...")
ablation_data = {}
for name, metrics in variants.items():
    clean_name = name.replace('\n', ' ')
    ablation_data[clean_name] = {
        "macro_f1_all": metrics["macro_f1_all"],
        "landform_macro_f1": metrics["landform_macro_f1"],
        "per_class_f1": {CLASS_ORDER[i]: metrics["f1"][i] for i in range(len(CLASS_ORDER))},
        "per_class_precision": {CLASS_ORDER[i]: metrics["precision"][i] for i in range(len(CLASS_ORDER))},
        "per_class_recall": {CLASS_ORDER[i]: metrics["recall"][i] for i in range(len(CLASS_ORDER))},
    }
(eval_dir / "ablation_3way.json").write_text(json.dumps(ablation_data, indent=2))
print("  Saved ablation_3way.json")

# ─── 6. Summary HTML Report ──────────────────────────────────────────────
print("6. Generating HTML report...")

def make_metrics_table(metrics, name):
    rows = ""
    for i, cls in enumerate(CLASS_ORDER):
        rows += f"<tr><td>{cls}</td><td>{metrics['precision'][i]:.3f}</td><td>{metrics['recall'][i]:.3f}</td><td><strong>{metrics['f1'][i]:.3f}</strong></td><td>{int(metrics['support'][i])}</td></tr>"
    return f"""
    <h3>{name}</h3>
    <p>Macro F1: <strong>{metrics['macro_f1_all']:.4f}</strong> | Landform F1: <strong>{metrics['landform_macro_f1']:.4f}</strong></p>
    <table><thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr></thead>
    <tbody>{rows}</tbody></table>
    """

tables_html = ""
for name, metrics in variants.items():
    tables_html += make_metrics_table(metrics, name.replace('\n', ' '))

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>MarsLandformNet V3 — Ablation Report</title>
<style>
body {{ font-family: -apple-system, Helvetica, Arial, sans-serif; max-width: 1100px; margin: 2rem auto; padding: 0 1rem; line-height: 1.5; }}
table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
th, td {{ border: 1px solid #ddd; padding: 0.5rem; text-align: left; }}
th {{ background: #f4f7fa; }}
img {{ max-width: 100%; margin: 1rem 0; border: 1px solid #eee; border-radius: 4px; }}
h1 {{ color: #2c3e50; }} h2 {{ color: #34495e; border-bottom: 2px solid #eee; padding-bottom: 0.3rem; }}
.highlight {{ background: #e8f8f5; padding: 1rem; border-radius: 8px; margin: 1rem 0; }}
</style>
</head>
<body>
<h1>MarsLandformNet V3 — Classification Report</h1>
<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>

<div class="highlight">
<h2>Key Results</h2>
<ul>
<li><strong>Best Model (V3 MultiHead):</strong> Landform F1 = <strong>{variants[variant_names[-1]]['landform_macro_f1']:.4f}</strong></li>
<li><strong>Improvement over Baseline:</strong> +{variants[variant_names[-1]]['landform_macro_f1'] - variants[variant_names[0]]['landform_macro_f1']:.4f} landform F1 ({((variants[variant_names[-1]]['landform_macro_f1'] / max(variants[variant_names[0]]['landform_macro_f1'], 0.001)) - 1) * 100:.1f}% relative)</li>
<li><strong>Architecture:</strong> 4-head gated attention + cross-modal MOLA fusion + stratified batch sampling</li>
<li><strong>Loss:</strong> FocalLoss (γ=1.5, label_smoothing=0.15) with inverse-frequency class weights</li>
</ul>
</div>

<h2>1. Ablation Comparison</h2>
<img src="ablation_3way_f1.png" alt="3-way F1 comparison">
<img src="ablation_macro_f1_progression.png" alt="Macro F1 progression">
<img src="ablation_delta_v3.png" alt="Delta from baseline">

<h2>2. Training Progress (V3)</h2>
<img src="training_curves_v3.png" alt="Training curves">

<h2>3. Per-Model Metrics</h2>
{tables_html}

<h2>4. Visualizations</h2>
<h3>Confusion Matrix</h3>
<img src="confusion_matrix.png" alt="Confusion matrix">
<h3>Confidence Distribution</h3>
<img src="confidence_distribution.png" alt="Confidence distribution">
<h3>Per-Class F1</h3>
<img src="per_class_f1.png" alt="Per-class F1">

<h2>5. Example Tiles with Attention</h2>
<img src="examples_LDA.png" alt="LDA examples">
<img src="examples_LVF.png" alt="LVF examples">
<img src="examples_CCF.png" alt="CCF examples">
<img src="examples_GLF.png" alt="GLF examples">

<h2>6. Attention Heatmaps</h2>
<img src="attention_LDA_0.png" alt="LDA attention 0">
<img src="attention_LVF_0.png" alt="LVF attention 0">
<img src="attention_CCF_0.png" alt="CCF attention 0">
<img src="attention_GLF_0.png" alt="GLF attention 0">

<hr>
<p><em>MarsLandformNet V2/V3 Pipeline — Multi-head MIL + DINOv2 + MOLA + ReACT Agent</em></p>
</body></html>"""

(eval_dir / "classification_report_v3.html").write_text(html)
print("  Saved classification_report_v3.html")

# ─── Done ─────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"All comparison artifacts saved to: {eval_dir}")
print(f"{'='*60}")
