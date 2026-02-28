#!/usr/bin/env python3
"""Generate ablation comparison and final HTML report."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

ROOT = Path("/disk1/cspark/MarsLab")
CLASS_ORDER = ["LDA", "LVF", "CCF", "GLF", "BACKGROUND"]
CLASS_COLORS = {"LDA": "#e74c3c", "LVF": "#3498db", "CCF": "#2ecc71", "GLF": "#f39c12", "BACKGROUND": "#9b59b6"}

eval_dir = ROOT / "Data/HiRISE/v2_output/eval"
eval_dir.mkdir(parents=True, exist_ok=True)

# Load both model results
baseline_dir = ROOT / "Data/HiRISE/v2_output/models/frozen_baseline"
focal_dir = ROOT / "Data/HiRISE/v2_output/models/cleaned_focal"

baseline_metrics = json.loads((baseline_dir / "test_metrics.json").read_text())
focal_metrics = json.loads((focal_dir / "test_metrics.json").read_text())

# ─── 1. Ablation Bar Chart ──────────────────────────────────────────────────
print("1. Ablation comparison chart...")
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# F1 per class comparison
x = np.arange(len(CLASS_ORDER))
width = 0.35
bars1 = axes[0].bar(x - width/2, baseline_metrics["f1"], width, label="Baseline (title regex)",
                     color="#95a5a6", edgecolor="white")
bars2 = axes[0].bar(x + width/2, focal_metrics["f1"], width, label="FocalLoss + Levy2014",
                     color="#2980b9", edgecolor="white")
axes[0].set_xticks(x)
axes[0].set_xticklabels(CLASS_ORDER)
axes[0].set_ylabel("F1 Score", fontsize=12)
axes[0].set_title("Per-Class F1 Comparison", fontsize=13)
axes[0].legend(fontsize=10)
axes[0].set_ylim(0, 1.0)
axes[0].grid(axis="y", alpha=0.3)
for bar, val in zip(bars1, baseline_metrics["f1"]):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{val:.2f}",
                 ha="center", fontsize=8, color="#95a5a6")
for bar, val in zip(bars2, focal_metrics["f1"]):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{val:.2f}",
                 ha="center", fontsize=8, color="#2980b9", fontweight="bold")

# Precision comparison
bars1 = axes[1].bar(x - width/2, baseline_metrics["precision"], width, label="Baseline",
                     color="#95a5a6", edgecolor="white")
bars2 = axes[1].bar(x + width/2, focal_metrics["precision"], width, label="FocalLoss + Levy2014",
                     color="#27ae60", edgecolor="white")
axes[1].set_xticks(x)
axes[1].set_xticklabels(CLASS_ORDER)
axes[1].set_ylabel("Precision", fontsize=12)
axes[1].set_title("Per-Class Precision Comparison", fontsize=13)
axes[1].legend(fontsize=10)
axes[1].set_ylim(0, 1.0)
axes[1].grid(axis="y", alpha=0.3)

# Recall comparison
bars1 = axes[2].bar(x - width/2, baseline_metrics["recall"], width, label="Baseline",
                     color="#95a5a6", edgecolor="white")
bars2 = axes[2].bar(x + width/2, focal_metrics["recall"], width, label="FocalLoss + Levy2014",
                     color="#e67e22", edgecolor="white")
axes[2].set_xticks(x)
axes[2].set_xticklabels(CLASS_ORDER)
axes[2].set_ylabel("Recall", fontsize=12)
axes[2].set_title("Per-Class Recall Comparison", fontsize=13)
axes[2].legend(fontsize=10)
axes[2].set_ylim(0, 1.0)
axes[2].grid(axis="y", alpha=0.3)

fig.suptitle("Ablation Study: Baseline vs FocalLoss + Levy 2014 Polygon Labels", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(eval_dir / "ablation_comparison.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("  Saved ablation_comparison.png")


# ─── 2. Delta Chart ─────────────────────────────────────────────────────────
print("2. Delta chart...")
fig, ax = plt.subplots(figsize=(10, 6))
deltas = [focal_metrics["f1"][i] - baseline_metrics["f1"][i] for i in range(len(CLASS_ORDER))]
colors = ["#27ae60" if d >= 0 else "#e74c3c" for d in deltas]
bars = ax.bar(CLASS_ORDER, deltas, color=colors, edgecolor="white", linewidth=1.5)
for bar, val in zip(bars, deltas):
    y_pos = bar.get_height() + 0.01 if val >= 0 else bar.get_height() - 0.03
    ax.text(bar.get_x() + bar.get_width()/2, y_pos, f"{val:+.3f}",
            ha="center", fontsize=12, fontweight="bold")
ax.axhline(0, color="black", linewidth=0.5)
ax.set_ylabel("ΔF1 Score", fontsize=13)
macro_delta = focal_metrics["macro_f1_all"] - baseline_metrics["macro_f1_all"]
lf_delta = focal_metrics["landform_macro_f1"] - baseline_metrics["landform_macro_f1"]
ax.set_title(f"F1 Change: FocalLoss+Levy2014 vs Baseline\n"
             f"ΔMacro F1: {macro_delta:+.3f} | ΔLandform F1: {lf_delta:+.3f}",
             fontsize=13)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(eval_dir / "ablation_delta.png", dpi=200)
plt.close(fig)
print("  Saved ablation_delta.png")


# ─── 3. HTML Report ─────────────────────────────────────────────────────────
print("3. Generating HTML report...")

# Load label stats
levy_report = {}
levy_path = ROOT / "Data/HiRISE/v2_output/levy2014_integration_report.json"
if levy_path.exists():
    levy_report = json.loads(levy_path.read_text())

label_stats = {}
stats_path = ROOT / "Data/HiRISE/v2_output/label_stats.json"
if stats_path.exists():
    label_stats = json.loads(stats_path.read_text())

# Collect all eval images
eval_images = sorted(eval_dir.glob("*.png"))

def img_to_base64(path):
    import base64
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>MarsLandformNet V2 — Classification Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #0d1117; color: #c9d1d9; }}
h1 {{ color: #e74c3c; border-bottom: 2px solid #e74c3c; padding-bottom: 10px; }}
h2 {{ color: #58a6ff; margin-top: 40px; }}
h3 {{ color: #8b949e; }}
table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
th, td {{ border: 1px solid #30363d; padding: 8px 12px; text-align: center; }}
th {{ background: #161b22; color: #58a6ff; }}
tr:nth-child(even) {{ background: #161b22; }}
.metric-card {{ display: inline-block; background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px 25px; margin: 5px; text-align: center; }}
.metric-card .value {{ font-size: 2em; font-weight: bold; color: #58a6ff; }}
.metric-card .label {{ font-size: 0.9em; color: #8b949e; }}
.improved {{ color: #3fb950; font-weight: bold; }}
.degraded {{ color: #f85149; }}
img {{ max-width: 100%; border-radius: 8px; margin: 10px 0; border: 1px solid #30363d; }}
.section {{ background: #161b22; padding: 20px; border-radius: 12px; margin: 20px 0; border: 1px solid #30363d; }}
code {{ background: #1f2937; padding: 2px 6px; border-radius: 4px; font-size: 0.9em; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 15px; }}
</style>
</head>
<body>

<h1>🔴 MarsLandformNet V2 — Classification Report</h1>
<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Mars HiRISE Mid-Latitude Landform Classification</p>

<div class="section">
<h2>📊 Key Metrics</h2>
<div>
<div class="metric-card"><div class="value">{focal_metrics['macro_f1_all']:.3f}</div><div class="label">Macro F1 (all)</div></div>
<div class="metric-card"><div class="value">{focal_metrics['landform_macro_f1']:.3f}</div><div class="label">Landform F1</div></div>
<div class="metric-card"><div class="value">{sum(focal_metrics['confusion_matrix'][i][i] for i in range(5))}/{int(sum(focal_metrics['support']))}</div><div class="label">Correct/Total</div></div>
<div class="metric-card"><div class="value">639</div><div class="label">Total Images</div></div>
<div class="metric-card"><div class="value">{levy_report.get('total_matches', '?')}</div><div class="label">Levy Polygon Matches</div></div>
</div>
</div>

<div class="section">
<h2>🔬 Ablation Study: Baseline vs Improved</h2>
<table>
<tr><th>Metric</th><th>Baseline<br>(title regex labels)</th><th>Improved<br>(FocalLoss + Levy2014)</th><th>Δ Change</th></tr>
<tr>
<td><b>Macro F1</b></td>
<td>{baseline_metrics['macro_f1_all']:.4f}</td>
<td>{focal_metrics['macro_f1_all']:.4f}</td>
<td class="{'improved' if focal_metrics['macro_f1_all'] > baseline_metrics['macro_f1_all'] else 'degraded'}">{focal_metrics['macro_f1_all'] - baseline_metrics['macro_f1_all']:+.4f}</td>
</tr>
<tr>
<td><b>Landform F1</b></td>
<td>{baseline_metrics['landform_macro_f1']:.4f}</td>
<td>{focal_metrics['landform_macro_f1']:.4f}</td>
<td class="{'improved' if focal_metrics['landform_macro_f1'] > baseline_metrics['landform_macro_f1'] else 'degraded'}">{focal_metrics['landform_macro_f1'] - baseline_metrics['landform_macro_f1']:+.4f}</td>
</tr>
"""

for i, cls in enumerate(CLASS_ORDER):
    bf1 = baseline_metrics["f1"][i]
    ff1 = focal_metrics["f1"][i]
    delta = ff1 - bf1
    css = "improved" if delta > 0 else "degraded" if delta < 0 else ""
    html += f"""<tr>
<td><b>{cls} F1</b></td>
<td>{bf1:.3f} (n={int(baseline_metrics['support'][i])})</td>
<td>{ff1:.3f} (n={int(focal_metrics['support'][i])})</td>
<td class="{css}">{delta:+.3f}</td>
</tr>
"""

html += """</table>
<img src="ablation_comparison.png" alt="Ablation Comparison">
<img src="ablation_delta.png" alt="Ablation Delta">
</div>

<div class="section">
<h2>📈 Training Progress</h2>
<img src="training_curves.png" alt="Training Curves">
</div>

<div class="section">
<h2>🎯 Per-Class Performance</h2>
<img src="per_class_f1.png" alt="Per-class F1">
<img src="confusion_matrix.png" alt="Confusion Matrix">
<img src="confidence_distribution.png" alt="Confidence Distribution">
<img src="label_distribution.png" alt="Label Distribution">
</div>

<div class="section">
<h2>🗺️ t-SNE Embedding Visualization</h2>
<img src="tsne_predictions.png" alt="t-SNE">
</div>

<div class="section">
<h2>🔍 Per-Class Example Tiles (Highest Attention)</h2>
<p>Green = correct prediction, Red = misclassified. Tiles shown are those with highest attention weights.</p>
"""

for cls in CLASS_ORDER:
    cls_desc = {"LDA": "Lobate Debris Apron", "LVF": "Lineated Valley Fill", "CCF": "Concentric Crater Fill", "GLF": "Glacier-Like Form", "BACKGROUND": "Background/Uncertain"}[cls]
    html += f'<h3>{cls} \u2014 {cls_desc}</h3>\n'
    html += f'<img src="examples_{cls}.png" alt="Examples {cls}">\n'

html += """</div>

<div class="section">
<h2>🔥 Attention Heatmaps</h2>
<p>High-attention tiles (red border) are regions the classifier focuses on. Low-attention tiles show ignored areas.</p>
"""

for cls in CLASS_ORDER:
    for rank in range(2):
        fname = f"attention_{cls}_{rank}.png"
        if (eval_dir / fname).exists():
            html += f'<img src="{fname}" alt="Attention {cls} #{rank}">\n'

html += f"""</div>

<div class="section">
<h2>📋 Pipeline Configuration</h2>
<table>
<tr><th>Component</th><th>Setting</th></tr>
<tr><td>Backbone</td><td>DINOv2 ViT-B/14 (frozen, 768-dim)</td></tr>
<tr><td>Classifier</td><td>Gated Attention MIL (256 hidden, 128 attention dim)</td></tr>
<tr><td>Loss</td><td>FocalLoss (γ=2.0) + Label Smoothing (ε=0.1)</td></tr>
<tr><td>MOLA Features</td><td>23-dim topographic features (3 scales × 7 features + 2 global)</td></tr>
<tr><td>Label Sources</td><td>Levy 2014 polygons > Hepburn SGLF > Pearson brain terrain > title regex</td></tr>
<tr><td>Classes</td><td>LDA, LVF, CCF, GLF, BACKGROUND (5-class)</td></tr>
<tr><td>Train/Val/Test</td><td>70% / 15% / 15% (spatial-aware split, 20km groups)</td></tr>
<tr><td>Optimizer</td><td>AdamW (lr=1e-4, weight_decay=1e-4)</td></tr>
<tr><td>Scheduler</td><td>Cosine annealing with 10% warmup</td></tr>
<tr><td>Early Stopping</td><td>Patience=15 on landform_macro_f1</td></tr>
</table>
</div>

<div class="section">
<h2>📚 Data Sources</h2>
<table>
<tr><th>Source</th><th>Type</th><th>Count</th><th>Classes</th><th>Priority</th></tr>
<tr><td>Levy et al. 2014</td><td>Expert polygons</td><td>6,385 polygons → {levy_report.get('total_matches', '?')} HiRISE matches</td><td>CCF, LDA, LVF</td><td>🥇 Highest</td></tr>
<tr><td>Hepburn et al. 2020</td><td>SGLF inventory</td><td>320 locations</td><td>GLF</td><td>🥈 High</td></tr>
<tr><td>Pearson et al. 2024</td><td>Brain terrain</td><td>90 in catalog</td><td>CCF/LDA indicator</td><td>🥉 Medium</td></tr>
<tr><td>Title regex</td><td>Metadata parse</td><td>385 images</td><td>All</td><td>⬇️ Low (weak)</td></tr>
</table>
</div>

<div class="section">
<h2>🚀 Next Steps</h2>
<ul>
<li><b>SSL LoRA Fine-tuning</b>: Upload <code>mars_tiles.tar.gz</code> (1.9GB) to Google Drive, open <code>colab_ssl_training.ipynb</code> in Colab, run all cells. Takes ~2-3 hours on T4 GPU.</li>
<li><b>Re-embed tiles</b>: After SSL, re-extract DINOv2 embeddings using LoRA weights, retrain MIL.</li>
<li><b>Download more data</b>: Brough 2019 GLF polygons (1,243 features) — contact s.brough@liverpool.ac.uk</li>
<li><b>5-fold cross-validation</b>: Current results are single split — add spatial-aware k-fold for robust estimates.</li>
</ul>
</div>

<footer style="text-align: center; color: #484f58; margin-top: 40px; padding: 20px; border-top: 1px solid #30363d;">
MarsLandformNet V2 | {datetime.now().strftime('%Y-%m-%d')} | DINOv2 + Attention MIL + MOLA
</footer>
</body>
</html>"""

report_path = eval_dir / "classification_report.html"
report_path.write_text(html)
print(f"  Saved {report_path}")

# ─── 4. Save ablation summary JSON ──────────────────────────────────────────
print("4. Saving ablation summary...")
ablation = {
    "experiments": {
        "frozen_baseline": {
            "description": "Frozen DINOv2 + MIL + CrossEntropyLoss + title_regex labels",
            "macro_f1": baseline_metrics["macro_f1_all"],
            "landform_f1": baseline_metrics["landform_macro_f1"],
            "per_class_f1": dict(zip(CLASS_ORDER, baseline_metrics["f1"])),
            "per_class_support": dict(zip(CLASS_ORDER, [int(s) for s in baseline_metrics["support"]])),
        },
        "cleaned_focal": {
            "description": "Frozen DINOv2 + MIL + FocalLoss(γ=2.0,ε=0.1) + Levy2014+Hepburn+Pearson labels",
            "macro_f1": focal_metrics["macro_f1_all"],
            "landform_f1": focal_metrics["landform_macro_f1"],
            "per_class_f1": dict(zip(CLASS_ORDER, focal_metrics["f1"])),
            "per_class_support": dict(zip(CLASS_ORDER, [int(s) for s in focal_metrics["support"]])),
        },
    },
    "improvements": {
        "label_changes": "Levy 2014 polygon matching added 372 spatial labels",
        "loss_function": "CrossEntropyLoss → FocalLoss(γ=2.0) + label_smoothing(ε=0.1)",
        "label_priority": "Levy2014 polygons > SGLF > brain_terrain > title_regex",
    },
    "delta": {
        "macro_f1": focal_metrics["macro_f1_all"] - baseline_metrics["macro_f1_all"],
        "landform_f1": focal_metrics["landform_macro_f1"] - baseline_metrics["landform_macro_f1"],
        "per_class_f1_delta": {cls: focal_metrics["f1"][i] - baseline_metrics["f1"][i]
                                for i, cls in enumerate(CLASS_ORDER)},
    },
    "pending": {
        "ssl_lora": "Colab notebook ready, tile archive being created",
        "expected_improvement": "SSL LoRA typically adds +5-15% F1 on domain-shifted visual tasks",
    },
}
ablation_path = eval_dir / "ablation_results.json"
ablation_path.write_text(json.dumps(ablation, indent=2))
print(f"  Saved {ablation_path}")

print(f"\n{'=' * 60}")
print("All outputs generated!")
print(f"HTML report: {report_path}")
print(f"{'=' * 60}")
