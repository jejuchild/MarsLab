from __future__ import annotations

import argparse
import html
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.marslandform_v2.config import CLASS_ORDER, EVAL_DIR


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _extract_per_class_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    precision = metrics.get("precision", [])
    recall = metrics.get("recall", [])
    f1 = metrics.get("f1", [])
    support = metrics.get("support", [])

    n_rows = max(len(precision), len(recall), len(f1), len(support), len(CLASS_ORDER))
    rows: list[dict[str, Any]] = []
    for idx in range(n_rows):
        class_name = CLASS_ORDER[idx] if idx < len(CLASS_ORDER) else f"CLASS_{idx}"
        rows.append(
            {
                "class": class_name,
                "precision": _safe_float(precision[idx]) if idx < len(precision) else 0.0,
                "recall": _safe_float(recall[idx]) if idx < len(recall) else 0.0,
                "f1": _safe_float(f1[idx]) if idx < len(f1) else 0.0,
                "support": int(_safe_float(support[idx], default=0.0)) if idx < len(support) else 0,
            }
        )
    return rows


def _find_confusion_matrix_image(metrics_path: Path) -> Path | None:
    candidates = [
        metrics_path.parent / "test_confusion_matrix.png",
        metrics_path.parent / "confusion_matrix.png",
        metrics_path.parent / "confusion.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    raw = _read_json(path)
    return raw if isinstance(raw, dict) else {}


def _extract_dataset_stats(metrics: dict[str, Any], metrics_path: Path) -> dict[str, Any]:
    stats = metrics.get("dataset_stats")
    if isinstance(stats, dict):
        return stats

    split_json = _load_optional_json(metrics_path.parent / "data_split.json")
    labels_json = _load_optional_json(metrics_path.parent / "label_stats.json")

    train_ids = split_json.get("train_ids", []) if isinstance(split_json.get("train_ids"), list) else []
    val_ids = split_json.get("val_ids", []) if isinstance(split_json.get("val_ids"), list) else []
    test_ids = split_json.get("test_ids", []) if isinstance(split_json.get("test_ids"), list) else []

    class_distribution = labels_json.get("class_distribution")
    if not isinstance(class_distribution, dict):
        class_distribution = metrics.get("class_distribution", {})
    if not isinstance(class_distribution, dict):
        class_distribution = {}

    return {
        "train_size": len(train_ids),
        "val_size": len(val_ids),
        "test_size": len(test_ids),
        "class_distribution": class_distribution,
    }


def _extract_model_config(metrics: dict[str, Any], metrics_path: Path) -> dict[str, Any]:
    model_cfg = metrics.get("model_config")
    if isinstance(model_cfg, dict):
        return model_cfg

    checkpoint_json = _load_optional_json(metrics_path.parent / "best_model_config.json")
    if checkpoint_json:
        return checkpoint_json

    return {
        "model": "Attention MIL",
        "version": "MarsLandformNet V2",
    }


def _extract_ablations(metrics: dict[str, Any], metrics_path: Path) -> dict[str, dict[str, Any]]:
    ablations = metrics.get("ablation_results")
    if isinstance(ablations, dict):
        return {str(k): v for k, v in ablations.items() if isinstance(v, dict)}

    ablation_path = metrics_path.parent / "ablation_results.json"
    if not ablation_path.exists():
        return {}

    raw = _read_json(ablation_path)
    if isinstance(raw, dict):
        return {str(k): v for k, v in raw.items() if isinstance(v, dict)}
    return {}


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([head, sep, *body])


def _html_table(headers: list[str], rows: list[list[str]]) -> str:
    th = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    tr = "".join(
        "<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table>"


def _render_markdown_report(
    metrics: dict[str, Any],
    metrics_path: Path,
    report_path: Path,
) -> str:
    overall_accuracy = _safe_float(metrics.get("accuracy", metrics.get("overall_accuracy", 0.0)))
    macro_f1 = _safe_float(metrics.get("macro_f1_all", metrics.get("macro_f1", 0.0)))
    per_class_rows = _extract_per_class_rows(metrics)
    dataset_stats = _extract_dataset_stats(metrics, metrics_path)
    model_config = _extract_model_config(metrics, metrics_path)
    ablations = _extract_ablations(metrics, metrics_path)
    confusion_img = _find_confusion_matrix_image(metrics_path)

    per_class_table = _markdown_table(
        ["Class", "Precision", "Recall", "F1", "Support"],
        [
            [
                row["class"],
                f"{row['precision']:.4f}",
                f"{row['recall']:.4f}",
                f"{row['f1']:.4f}",
                str(row["support"]),
            ]
            for row in per_class_rows
        ],
    )

    class_dist = dataset_stats.get("class_distribution", {})
    class_dist_lines = "\n".join(f"- {k}: {v}" for k, v in class_dist.items()) if class_dist else "- Not available"
    model_cfg_lines = "\n".join(f"- {k}: {v}" for k, v in model_config.items())

    if ablations:
        ablation_table = _markdown_table(
            ["Variant", "Accuracy", "Macro-F1", "Landform Macro-F1"],
            [
                [
                    variant,
                    f"{_safe_float(values.get('accuracy', values.get('overall_accuracy', 0.0))):.4f}",
                    f"{_safe_float(values.get('macro_f1_all', values.get('macro_f1', 0.0))):.4f}",
                    f"{_safe_float(values.get('landform_macro_f1', 0.0)):.4f}",
                ]
                for variant, values in ablations.items()
            ],
        )
    else:
        ablation_table = "No ablation results available."

    if confusion_img is not None:
        relative_img = confusion_img.relative_to(report_path.parent)
        confusion_section = f"![Confusion Matrix]({relative_img.as_posix()})"
    else:
        confusion_section = "Confusion matrix image not found."

    return "\n\n".join(
        [
            "# MarsLandformNet V2 Evaluation Report",
            f"Generated: {datetime.now(UTC).isoformat()}",
            "## 1) Executive Summary",
            f"- Overall accuracy: **{overall_accuracy:.4f}**",
            f"- Macro F1: **{macro_f1:.4f}**",
            "## 2) Per-Class Performance",
            per_class_table,
            "## 3) Confusion Matrix",
            confusion_section,
            "## 4) Dataset Statistics",
            f"- Train size: {dataset_stats.get('train_size', 0)}\n- Val size: {dataset_stats.get('val_size', 0)}\n- Test size: {dataset_stats.get('test_size', 0)}\n- Class distribution:\n{class_dist_lines}",
            "## 5) Model Configuration Summary",
            model_cfg_lines if model_cfg_lines else "- Not available",
            "## 6) Ablation Results",
            ablation_table,
        ]
    ) + "\n"


def _render_html_report(metrics: dict[str, Any], metrics_path: Path, report_path: Path) -> str:
    overall_accuracy = _safe_float(metrics.get("accuracy", metrics.get("overall_accuracy", 0.0)))
    macro_f1 = _safe_float(metrics.get("macro_f1_all", metrics.get("macro_f1", 0.0)))
    per_class_rows = _extract_per_class_rows(metrics)
    dataset_stats = _extract_dataset_stats(metrics, metrics_path)
    model_config = _extract_model_config(metrics, metrics_path)
    ablations = _extract_ablations(metrics, metrics_path)
    confusion_img = _find_confusion_matrix_image(metrics_path)

    per_class_table = _html_table(
        ["Class", "Precision", "Recall", "F1", "Support"],
        [
            [
                row["class"],
                f"{row['precision']:.4f}",
                f"{row['recall']:.4f}",
                f"{row['f1']:.4f}",
                str(row["support"]),
            ]
            for row in per_class_rows
        ],
    )

    class_dist = dataset_stats.get("class_distribution", {})
    class_dist_list = "".join(f"<li>{html.escape(str(k))}: {html.escape(str(v))}</li>" for k, v in class_dist.items())
    model_cfg_list = "".join(f"<li>{html.escape(str(k))}: {html.escape(str(v))}</li>" for k, v in model_config.items())

    if ablations:
        ablation_table = _html_table(
            ["Variant", "Accuracy", "Macro-F1", "Landform Macro-F1"],
            [
                [
                    variant,
                    f"{_safe_float(values.get('accuracy', values.get('overall_accuracy', 0.0))):.4f}",
                    f"{_safe_float(values.get('macro_f1_all', values.get('macro_f1', 0.0))):.4f}",
                    f"{_safe_float(values.get('landform_macro_f1', 0.0)):.4f}",
                ]
                for variant, values in ablations.items()
            ],
        )
    else:
        ablation_table = "<p>No ablation results available.</p>"

    if confusion_img is not None:
        relative_img = confusion_img.relative_to(report_path.parent).as_posix()
        confusion_block = f'<img src="{html.escape(relative_img)}" alt="Confusion Matrix" style="max-width: 900px; width: 100%;">'
    else:
        confusion_block = "<p>Confusion matrix image not found.</p>"

    return f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
  <title>MarsLandformNet V2 Evaluation Report</title>
  <style>
    body {{ font-family: Helvetica, Arial, sans-serif; margin: 2rem auto; max-width: 1000px; line-height: 1.45; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
    th, td {{ border: 1px solid #ddd; padding: 0.55rem; text-align: left; }}
    th {{ background: #f4f7fa; }}
    h1, h2 {{ margin-top: 1.2rem; }}
  </style>
</head>
<body>
  <h1>MarsLandformNet V2 Evaluation Report</h1>
  <p><strong>Generated:</strong> {html.escape(datetime.now(UTC).isoformat())}</p>

  <h2>1) Executive Summary</h2>
  <ul>
    <li>Overall accuracy: <strong>{overall_accuracy:.4f}</strong></li>
    <li>Macro F1: <strong>{macro_f1:.4f}</strong></li>
  </ul>

  <h2>2) Per-Class Performance</h2>
  {per_class_table}

  <h2>3) Confusion Matrix</h2>
  {confusion_block}

  <h2>4) Dataset Statistics</h2>
  <ul>
    <li>Train size: {int(dataset_stats.get('train_size', 0))}</li>
    <li>Val size: {int(dataset_stats.get('val_size', 0))}</li>
    <li>Test size: {int(dataset_stats.get('test_size', 0))}</li>
  </ul>
  <p><strong>Class distribution</strong></p>
  <ul>{class_dist_list or '<li>Not available</li>'}</ul>

  <h2>5) Model Configuration Summary</h2>
  <ul>{model_cfg_list or '<li>Not available</li>'}</ul>

  <h2>6) Ablation Results</h2>
  {ablation_table}
</body>
</html>
"""


def generate_report(metrics_path: Path, output_dir: Path, format: str = "markdown") -> Path:
    metrics_raw = _read_json(metrics_path)
    if not isinstance(metrics_raw, dict):
        raise ValueError(f"Metrics JSON at {metrics_path} must contain an object")

    output_dir.mkdir(parents=True, exist_ok=True)

    fmt = format.lower()
    if fmt not in {"markdown", "html"}:
        raise ValueError("format must be either 'markdown' or 'html'")

    if fmt == "markdown":
        report_path = output_dir / "evaluation_report.md"
        report_text = _render_markdown_report(metrics_raw, metrics_path, report_path)
    else:
        report_path = output_dir / "evaluation_report.html"
        report_text = _render_html_report(metrics_raw, metrics_path, report_path)

    report_path.write_text(report_text, encoding="utf-8")
    return report_path


def generate_comparison_table(variant_metrics: dict[str, Any], output_path: Path) -> Path:
    if not isinstance(variant_metrics, dict):
        raise ValueError("variant_metrics must be a dictionary mapping variant -> metric dict")

    rows: list[str] = []
    for variant, metrics in variant_metrics.items():
        metric_dict = metrics if isinstance(metrics, dict) else {}
        accuracy = _safe_float(metric_dict.get("accuracy", metric_dict.get("overall_accuracy", 0.0)))
        macro_f1 = _safe_float(metric_dict.get("macro_f1_all", metric_dict.get("macro_f1", 0.0)))
        landform_macro_f1 = _safe_float(metric_dict.get("landform_macro_f1", 0.0))
        rows.append(
            f"{variant} & {accuracy:.4f} & {macro_f1:.4f} & {landform_macro_f1:.4f} \\\\"
        )

    table = "\n".join(
        [
            "\\begin{table}[t]",
            "\\centering",
            "\\caption{Ablation comparison for MarsLandformNet V2 variants}",
            "\\label{tab:ablation_comparison}",
            "\\begin{tabular}{lccc}",
            "\\hline",
            "Variant & Accuracy & Macro-F1 & Landform Macro-F1 \\\\",
            "\\hline",
            *rows,
            "\\hline",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(table, encoding="utf-8")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate MarsLandformNet V2 evaluation reports")
    subparsers = parser.add_subparsers(dest="command", required=True)

    report_parser = subparsers.add_parser("report", help="Generate markdown/html evaluation report")
    report_parser.add_argument("--metrics-path", type=Path, default=EVAL_DIR / "test_metrics.json")
    report_parser.add_argument("--output-dir", type=Path, default=EVAL_DIR)
    report_parser.add_argument("--format", choices=["markdown", "html"], default="markdown")

    compare_parser = subparsers.add_parser("compare", help="Generate LaTeX ablation comparison table")
    compare_parser.add_argument("--variant-metrics-path", type=Path, required=True)
    compare_parser.add_argument("--output-path", type=Path, default=EVAL_DIR / "ablation_comparison.tex")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "report":
        out_path = generate_report(
            metrics_path=args.metrics_path,
            output_dir=args.output_dir,
            format=args.format,
        )
    else:
        variant_metrics = _read_json(args.variant_metrics_path)
        if not isinstance(variant_metrics, dict):
            raise ValueError("variant_metrics JSON must be an object mapping variant names to metrics")
        out_path = generate_comparison_table(variant_metrics=variant_metrics, output_path=args.output_path)
    print(f"Report artifact written to: {out_path}")


if __name__ == "__main__":
    main()
