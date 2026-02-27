from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.marslandform_v2.config import CLASS_NAMES, EVAL_DIR


VARIANT_ORDER = ["frozen_baseline", "ssl_lora", "frozen_mola", "ssl_mola"]


def _load_metrics(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_class_f1(metrics: dict[str, Any], class_name: str, class_idx: int) -> float:
    per_class_f1 = metrics.get("per_class_f1")
    if isinstance(per_class_f1, dict):
        value = per_class_f1.get(class_name)
        return float(value) if value is not None else 0.0
    if isinstance(per_class_f1, list) and class_idx < len(per_class_f1):
        return float(per_class_f1[class_idx])
    legacy_f1 = metrics.get("f1")
    if isinstance(legacy_f1, list) and class_idx < len(legacy_f1):
        return float(legacy_f1[class_idx])
    return 0.0


def run_ablation(results_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variant in VARIANT_ORDER:
        metrics_path = results_dir / variant / "test_metrics.json"
        if not metrics_path.exists():
            continue
        metrics = _load_metrics(metrics_path)
        row: dict[str, Any] = {
            "variant": variant,
            "accuracy": float(metrics.get("accuracy", 0.0)),
            "macro_f1": float(metrics.get("macro_f1", metrics.get("macro_f1_all", 0.0))),
        }
        for class_idx, class_name in enumerate(CLASS_NAMES):
            row[f"f1_{class_name}"] = _extract_class_f1(metrics, class_name, class_idx)
        rows.append(row)

    if not rows:
        raise FileNotFoundError(f"No ablation metrics found in {results_dir}")

    df = pd.DataFrame(rows)
    df["variant"] = pd.Categorical(df["variant"], categories=VARIANT_ORDER, ordered=True)
    df = df.sort_values("variant").reset_index(drop=True)

    output_csv = results_dir / "ablation_summary.csv"
    output_tex = results_dir / "ablation_summary.tex"
    df.to_csv(output_csv, index=False)

    latex_df = df.copy()
    numeric_cols = [col for col in latex_df.columns if col != "variant"]
    latex_df[numeric_cols] = latex_df[numeric_cols].astype(float).round(4)
    output_tex.write_text(
        latex_df.to_latex(index=False, float_format=lambda x: f"{x:.4f}"),
        encoding="utf-8",
    )

    return df


def plot_ablation(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    class_cols = [f"f1_{name}" for name in CLASS_NAMES]
    long_df = df[["variant", *class_cols]].melt(
        id_vars="variant",
        value_vars=class_cols,
        var_name="class_name",
        value_name="f1",
    )
    long_df["class_name"] = long_df["class_name"].str.replace("f1_", "", regex=False)

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=long_df, x="class_name", y="f1", hue="variant", ax=ax)
    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_ylim(0.0, 1.0)
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(title="Variant", fontsize=11, title_fontsize=12, frameon=True)
    fig.tight_layout()
    fig.savefig(output_dir / "ablation_f1_by_class.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MarsLandformNet V2 ablation comparisons")
    parser.add_argument("--results_dir", type=Path, default=EVAL_DIR, help="Root directory containing variant folders")
    parser.add_argument("--output_dir", type=Path, default=None, help="Plot output directory (default: results_dir)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = run_ablation(args.results_dir)
    out_dir = args.output_dir or args.results_dir
    plot_ablation(df, out_dir)
    print(f"Saved ablation CSV/TEX in {args.results_dir}")
    print(f"Saved ablation figure in {out_dir}")


if __name__ == "__main__":
    main()
