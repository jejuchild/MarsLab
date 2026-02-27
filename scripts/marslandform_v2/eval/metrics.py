from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.marslandform_v2.config import CLASS_NAMES, CLASS_ORDER, EVAL_DIR


def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int], class_names: Sequence[str]) -> dict[str, Any]:
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    labels = list(range(len(class_names)))

    accuracy = float(accuracy_score(y_true_arr, y_pred_arr))
    macro_f1 = float(f1_score(y_true_arr, y_pred_arr, labels=labels, average="macro"))
    weighted_f1 = float(f1_score(y_true_arr, y_pred_arr, labels=labels, average="weighted"))
    precision_raw, recall_raw, f1_raw, support_raw = precision_recall_fscore_support(
        y_true_arr,
        y_pred_arr,
        labels=labels,
    )
    precision = np.atleast_1d(np.asarray(precision_raw, dtype=float))
    recall = np.atleast_1d(np.asarray(recall_raw, dtype=float))
    f1 = np.atleast_1d(np.asarray(f1_raw, dtype=float))
    support = np.atleast_1d(np.asarray(0 if support_raw is None else support_raw, dtype=int))
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=labels)

    report = classification_report(
        y_true_arr,
        y_pred_arr,
        labels=labels,
        target_names=list(class_names),
        output_dict=True,
    )

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class_precision": {name: float(precision[idx]) for idx, name in enumerate(class_names)},
        "per_class_recall": {name: float(recall[idx]) for idx, name in enumerate(class_names)},
        "per_class_f1": {name: float(f1[idx]) for idx, name in enumerate(class_names)},
        "per_class_support": {name: int(support[idx]) for idx, name in enumerate(class_names)},
        "confusion_matrix": cm.astype(int).tolist(),
        "classification_report": report,
    }


def _expected_calibration_error(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    n_bins: int = 10,
) -> tuple[float, dict[str, list[float] | list[int]]]:
    confidences = y_pred_probs.max(axis=1)
    predictions = y_pred_probs.argmax(axis=1)
    correctness = (predictions == y_true).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(confidences, bin_edges[1:-1], right=False)

    bin_accuracy: list[float] = []
    bin_confidence: list[float] = []
    bin_counts: list[int] = []
    ece = 0.0

    total = max(1, len(confidences))
    for idx in range(n_bins):
        mask = bin_indices == idx
        count = int(mask.sum())
        if count == 0:
            bin_accuracy.append(0.0)
            bin_confidence.append(0.0)
            bin_counts.append(0)
            continue

        acc_bin = float(correctness[mask].mean())
        conf_bin = float(confidences[mask].mean())
        ece += abs(acc_bin - conf_bin) * (count / total)

        bin_accuracy.append(acc_bin)
        bin_confidence.append(conf_bin)
        bin_counts.append(count)

    reliability = {
        "bin_edges": bin_edges.tolist(),
        "bin_accuracy": bin_accuracy,
        "bin_confidence": bin_confidence,
        "bin_counts": bin_counts,
    }
    return float(ece), reliability


def compute_confidence_metrics(y_true: Sequence[int], y_pred_probs: Sequence[Sequence[float]]) -> dict[str, Any]:
    y_true_arr = np.asarray(y_true, dtype=int)
    probs_arr = np.asarray(y_pred_probs, dtype=float)
    if probs_arr.ndim != 2:
        raise ValueError("y_pred_probs must be a 2D array-like of shape (n_samples, n_classes)")
    if probs_arr.shape[0] != y_true_arr.shape[0]:
        raise ValueError("y_true and y_pred_probs must have same number of samples")

    row_sums = probs_arr.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    probs_arr = probs_arr / row_sums

    ece, reliability = _expected_calibration_error(y_true_arr, probs_arr)

    num_classes = probs_arr.shape[1]
    auroc_per_class: dict[str, float | None] = {}
    for class_idx in range(num_classes):
        class_name = CLASS_ORDER[class_idx] if class_idx < len(CLASS_ORDER) else f"class_{class_idx}"
        class_targets = (y_true_arr == class_idx).astype(int)
        if np.unique(class_targets).size < 2:
            auroc_per_class[class_name] = None
            continue
        auroc_per_class[class_name] = float(roc_auc_score(class_targets, probs_arr[:, class_idx]))

    return {
        "ece": ece,
        "auroc_per_class": auroc_per_class,
        "reliability_diagram": reliability,
    }


def save_metrics(metrics: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute MarsLandformNet V2 evaluation metrics")
    parser.add_argument("--y_true", type=Path, required=True, help="Path to JSON list of true class indices")
    parser.add_argument("--y_pred", type=Path, required=True, help="Path to JSON list of predicted class indices")
    parser.add_argument(
        "--y_pred_probs",
        type=Path,
        default=None,
        help="Optional path to JSON list of prediction probabilities (n_samples x n_classes)",
    )
    parser.add_argument("--output", type=Path, default=EVAL_DIR / "test_metrics.json", help="Output JSON path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    y_true = _load_json(args.y_true)
    y_pred = _load_json(args.y_pred)

    metrics = compute_metrics(y_true=y_true, y_pred=y_pred, class_names=CLASS_ORDER)
    metrics["landform_class_names"] = list(CLASS_NAMES)
    metrics["class_order"] = list(CLASS_ORDER)

    if args.y_pred_probs is not None:
        probs = _load_json(args.y_pred_probs)
        metrics.update(compute_confidence_metrics(y_true=y_true, y_pred_probs=probs))

    save_metrics(metrics, args.output)
    print(f"Saved metrics to {args.output}")


if __name__ == "__main__":
    main()
