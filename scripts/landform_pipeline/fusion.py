#!/usr/bin/env python3
"""Fuse CNN and DEM logits with class-wise calibrated Bayesian weights."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


CLASS_ORDER = ["LDA", "CCF", "LVF", "GLF", "BACKGROUND"]
LAND_CLASSES = set(CLASS_ORDER[:-1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bayesian fusion of CNN logits and MOLA DEM logits."
    )
    _ = parser.add_argument(
        "--embeddings",
        default="Data/HiRISE/pipeline_output/embeddings.npy",
        help="Path to tile embeddings (.npy).",
    )
    _ = parser.add_argument(
        "--mola-features",
        default="Data/HiRISE/pipeline_output/mola_features.npy",
        help="Path to MOLA feature matrix (.npy).",
    )
    _ = parser.add_argument(
        "--tile-metadata",
        default="Data/HiRISE/pipeline_output/tile_metadata.csv",
        help="Path to tile metadata CSV.",
    )
    _ = parser.add_argument(
        "--classifier-model",
        default="Data/HiRISE/pipeline_output/classifier/best_model.pt",
        help="Path to trained classifier checkpoint (.pt).",
    )
    _ = parser.add_argument(
        "--cluster-dir",
        default="Data/HiRISE/pipeline_output/clusters",
        help="Directory containing cluster_summary.json and cluster_assignments.csv.",
    )
    _ = parser.add_argument(
        "--output-dir",
        default="Data/HiRISE/pipeline_output/fusion",
        help="Directory for fusion outputs.",
    )
    _ = parser.add_argument(
        "--enrichment-threshold",
        type=float,
        default=2.0,
        help="Minimum enrichment score to trust cluster dominant class.",
    )
    _ = parser.add_argument(
        "--calibration-split",
        type=float,
        default=0.2,
        help="Fraction used for held-out calibration.",
    )
    _ = parser.add_argument(
        "--temperature",
        type=float,
        default=1.5,
        help="Temperature used to smooth logits before calibration.",
    )
    return parser.parse_args()


def _state_tensor_to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy().astype(np.float64)


def _find_linear_keys(state_dict: dict[str, object]) -> list[tuple[str, str]]:
    """Find (weight_key, bias_key) pairs for all linear layers, sorted by key name."""
    weight_keys = [
        k for k, v in state_dict.items()
        if k.endswith(".weight") and hasattr(v, "ndim") and v.ndim == 2  # type: ignore[union-attr]
    ]
    weight_keys.sort()
    pairs = []
    for wk in weight_keys:
        bk = wk[:-7] + ".bias"  # replace '.weight' with '.bias'
        pairs.append((wk, bk))
    return pairs


def load_classifier_head(classifier_model_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the 2-layer MLP head from a classifier checkpoint.

    Supports any key prefix (net.X, layers.X, head.net.X) by dynamically finding
    the first and last Linear weight tensors in the state_dict.
    """
    checkpoint = torch.load(classifier_model_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint does not contain a valid model_state_dict.")

    linear_pairs = _find_linear_keys(state_dict)
    if len(linear_pairs) < 2:
        raise ValueError(
            f"Expected at least 2 linear layers in classifier, found {len(linear_pairs)}: "
            f"{[p[0] for p in linear_pairs]}"
        )

    # First linear layer (input → hidden) and last linear layer (hidden → output)
    w0_key, b0_key = linear_pairs[0]
    w1_key, b1_key = linear_pairs[-1]

    w0 = _state_tensor_to_numpy(state_dict[w0_key])
    b0 = _state_tensor_to_numpy(state_dict[b0_key]) if b0_key in state_dict else np.zeros(state_dict[w0_key].shape[0], dtype=np.float64)
    w1 = _state_tensor_to_numpy(state_dict[w1_key])
    b1 = _state_tensor_to_numpy(state_dict[b1_key]) if b1_key in state_dict else np.zeros(state_dict[w1_key].shape[0], dtype=np.float64)

    if w1.shape[0] != len(CLASS_ORDER):
        raise ValueError(
            f"Classifier head output dim is {w1.shape[0]}, expected {len(CLASS_ORDER)}."
        )

    return w0, b0, w1, b1


def forward_head_logits(
    embeddings: np.ndarray,
    w0: np.ndarray,
    b0: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
) -> np.ndarray:
    hidden = embeddings @ w0.T + b0[None, :]
    hidden = np.maximum(hidden, 0.0)
    logits = hidden @ w1.T + b1[None, :]
    return logits.astype(np.float64)


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_vals = np.exp(shifted)
    denom = np.sum(exp_vals, axis=1, keepdims=True)
    return exp_vals / np.clip(denom, 1e-12, None)


def build_pseudo_labels(
    cluster_dir: Path,
    threshold: float,
    tile_count: int,
    metadata: pd.DataFrame,
) -> np.ndarray:
    summary_path = cluster_dir / "cluster_summary.json"
    assignments_path = cluster_dir / "cluster_assignments.csv"

    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    assignments = pd.read_csv(assignments_path)
    if "cluster_id" not in assignments.columns:
        raise ValueError("cluster_assignments.csv must contain a 'cluster_id' column.")

    if len(assignments) != tile_count:
        join_cols = ["image_id", "tile_row", "tile_col"]
        if not all(c in assignments.columns for c in join_cols):
            raise ValueError(
                "cluster_assignments.csv length mismatch and merge keys are unavailable."
            )
        aligned = metadata[join_cols].merge(
            assignments[join_cols + ["cluster_id"]],
            on=join_cols,
            how="left",
        )
        cluster_ids = aligned["cluster_id"].to_numpy()
    else:
        cluster_ids = assignments["cluster_id"].to_numpy()

    cluster_to_label: dict[int, str] = {}
    for row in summary.get("clusters", []):
        cluster_id = int(row.get("id", -1))
        dominant = str(row.get("dominant_class", "BACKGROUND")).upper()
        score = float(row.get("enrichment_score", 0.0))
        if dominant in LAND_CLASSES and score >= threshold:
            cluster_to_label[cluster_id] = dominant
        else:
            cluster_to_label[cluster_id] = "BACKGROUND"

    labels = np.full(tile_count, CLASS_ORDER.index("BACKGROUND"), dtype=np.int64)
    for i in range(tile_count):
        cid = cluster_ids[i]
        if pd.isna(cid):
            continue
        label_name = cluster_to_label.get(int(cid), "BACKGROUND")
        labels[i] = CLASS_ORDER.index(label_name)

    return labels


def train_dem_classifier(
    mola_features: np.ndarray,
    train_labels: np.ndarray,
    train_indices: np.ndarray,
) -> tuple[LogisticRegression, StandardScaler]:
    scaler = StandardScaler()
    mola_scaled = scaler.fit_transform(mola_features)
    model = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
    )
    model.fit(mola_scaled[train_indices], train_labels[train_indices])
    return model, scaler


def dem_logits_from_model(model: LogisticRegression, features: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    scaled = scaler.transform(features)
    probs = model.predict_proba(scaled)
    logits = np.full((features.shape[0], len(CLASS_ORDER)), -20.0, dtype=np.float64)
    for i, cls_idx in enumerate(model.classes_):
        logits[:, int(cls_idx)] = np.log(np.clip(probs[:, i], 1e-9, 1.0))
    return logits


def split_indices(labels: np.ndarray, calibration_split: float) -> tuple[np.ndarray, np.ndarray]:
    n_samples = labels.shape[0]
    if n_samples < 10:
        raise ValueError("Need at least 10 samples for train/calibration split.")

    calibration_split = float(np.clip(calibration_split, 0.05, 0.5))
    unique, counts = np.unique(labels, return_counts=True)
    can_stratify = unique.size > 1 and np.all(counts >= 2)
    stratify_labels = labels if can_stratify else None

    all_idx = np.arange(n_samples)
    train_idx, cal_idx = train_test_split(
        all_idx,
        test_size=calibration_split,
        random_state=42,
        stratify=stratify_labels,
    )
    return np.asarray(train_idx), np.asarray(cal_idx)


def optimize_fusion_weights(
    cnn_logits: np.ndarray,
    dem_logits: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_classes = len(CLASS_ORDER)

    def objective(params: np.ndarray) -> float:
        w = params[:n_classes]
        b = params[n_classes:]
        fused = cnn_logits + dem_logits * w[None, :] + b[None, :]
        probs = stable_softmax(fused)
        nll = -np.mean(np.log(np.clip(probs[np.arange(labels.shape[0]), labels], 1e-12, 1.0)))
        return float(nll)

    init = np.concatenate([np.ones(n_classes), np.zeros(n_classes)])
    bounds = [(-5.0, 5.0)] * (2 * n_classes)

    result = minimize(
        objective,
        x0=init,
        method="L-BFGS-B",
        bounds=bounds,
    )
    if not result.success:
        print(f"[warn] Fusion optimization did not fully converge: {result.message}")

    learned = result.x
    return learned[:n_classes], learned[n_classes:]


def evaluate_predictions(
    labels: np.ndarray,
    probs: np.ndarray,
) -> tuple[float, dict[str, float]]:
    preds = np.argmax(probs, axis=1)
    acc = float(accuracy_score(labels, preds))
    f1_map: dict[str, float] = {}
    for i, class_name in enumerate(CLASS_ORDER):
        tp = float(np.sum((labels == i) & (preds == i)))
        fp = float(np.sum((labels != i) & (preds == i)))
        fn = float(np.sum((labels == i) & (preds != i)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall == 0:
            f1_map[class_name] = 0.0
        else:
            f1_map[class_name] = 2.0 * precision * recall / (precision + recall)
    return acc, f1_map


def print_metrics(
    labels: np.ndarray,
    cnn_probs: np.ndarray,
    dem_probs: np.ndarray,
    fused_probs: np.ndarray,
) -> None:
    cnn_acc, cnn_f1 = evaluate_predictions(labels, cnn_probs)
    dem_acc, dem_f1 = evaluate_predictions(labels, dem_probs)
    fused_acc, fused_f1 = evaluate_predictions(labels, fused_probs)

    print("\n=== Fusion Quality Metrics ===")
    print(f"Accuracy (CNN-only):   {cnn_acc:.4f}")
    print(f"Accuracy (DEM-only):   {dem_acc:.4f}")
    print(f"Accuracy (Fused):      {fused_acc:.4f}")
    print(f"Improvement vs CNN:    {fused_acc - cnn_acc:+.4f}")
    print(f"Improvement vs DEM:    {fused_acc - dem_acc:+.4f}")

    print("\nPer-class F1:")
    for cls in CLASS_ORDER:
        print(
            f"  {cls:<11} CNN={cnn_f1[cls]:.4f}  DEM={dem_f1[cls]:.4f}  FUSED={fused_f1[cls]:.4f}"
        )


def build_output_table(metadata: pd.DataFrame, fused_probs: np.ndarray) -> pd.DataFrame:
    pred_idx = np.argmax(fused_probs, axis=1)
    conf = np.max(fused_probs, axis=1)
    pred_class = [CLASS_ORDER[i] for i in pred_idx]

    output = pd.DataFrame(
        {
            "image_id": metadata.get("image_id", pd.Series([""] * len(metadata))),
            "tile_row": metadata.get("tile_row", pd.Series([-1] * len(metadata))),
            "tile_col": metadata.get("tile_col", pd.Series([-1] * len(metadata))),
            "lat": pd.to_numeric(metadata.get("lat", pd.Series([np.nan] * len(metadata))), errors="coerce"),
            "lon": pd.to_numeric(metadata.get("lon", pd.Series([np.nan] * len(metadata))), errors="coerce"),
            "pred_class": pred_class,
            "confidence": conf,
            "prob_LDA": fused_probs[:, 0],
            "prob_CCF": fused_probs[:, 1],
            "prob_LVF": fused_probs[:, 2],
            "prob_GLF": fused_probs[:, 3],
            "prob_BACKGROUND": fused_probs[:, 4],
        }
    )
    return output


def main() -> None:
    start_time = time.time()
    args = parse_args()

    embeddings_path = Path(args.embeddings)
    mola_features_path = Path(args.mola_features)
    metadata_path = Path(args.tile_metadata)
    classifier_model_path = Path(args.classifier_model)
    cluster_dir = Path(args.cluster_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("[info] Loading inputs...")
    embeddings = np.load(embeddings_path)
    mola_features = np.load(mola_features_path)
    metadata = pd.read_csv(metadata_path)

    if embeddings.ndim != 2 or mola_features.ndim != 2:
        raise ValueError("Embeddings and MOLA features must be 2D arrays.")
    if embeddings.shape[0] != mola_features.shape[0] or embeddings.shape[0] != len(metadata):
        raise ValueError(
            "Row mismatch across embeddings, mola_features, and tile_metadata."
        )
    if mola_features.shape[1] != 23:
        print(f"[warn] Expected 23 MOLA features, found {mola_features.shape[1]}.")

    print("[info] Building pseudo-labels from cluster enrichment...")
    labels = build_pseudo_labels(
        cluster_dir=cluster_dir,
        threshold=args.enrichment_threshold,
        tile_count=embeddings.shape[0],
        metadata=metadata,
    )

    train_idx, cal_idx = split_indices(labels, args.calibration_split)
    print(f"[info] Train samples: {len(train_idx)} | Calibration samples: {len(cal_idx)}")

    print("[info] Stage 1/2: training DEM logistic regression...")
    dem_model, mola_scaler = train_dem_classifier(mola_features, labels, train_idx)
    dem_logits = dem_logits_from_model(dem_model, mola_features, mola_scaler)

    print("[info] Computing CNN logits from classifier head...")
    w0, b0, w1, b1 = load_classifier_head(classifier_model_path)
    cnn_logits = forward_head_logits(embeddings, w0, b0, w1, b1)

    temperature = float(max(args.temperature, 1e-6))
    cnn_logits_scaled = cnn_logits / temperature
    dem_logits_scaled = dem_logits / temperature

    print("[info] Stage 2/2: calibrating fusion weights on held-out split...")
    fusion_w, fusion_b = optimize_fusion_weights(
        cnn_logits=cnn_logits_scaled[cal_idx],
        dem_logits=dem_logits_scaled[cal_idx],
        labels=labels[cal_idx],
    )

    fused_logits = cnn_logits_scaled + fusion_w[None, :] * dem_logits_scaled + fusion_b[None, :]
    cnn_probs = stable_softmax(cnn_logits_scaled)
    dem_probs = stable_softmax(dem_logits_scaled)
    fused_probs = stable_softmax(fused_logits)

    print_metrics(labels, cnn_probs, dem_probs, fused_probs)

    predictions = build_output_table(metadata, fused_probs)
    fused_csv_path = output_dir / "fused_predictions.csv"
    predictions.to_csv(fused_csv_path, index=False)

    fusion_json = {
        "class_names": CLASS_ORDER,
        "class_order": CLASS_ORDER,
        "temperature": temperature,
        "weights": {cls: float(fusion_w[i]) for i, cls in enumerate(CLASS_ORDER)},
        "bias": {cls: float(fusion_b[i]) for i, cls in enumerate(CLASS_ORDER)},
    }
    fusion_weights_path = output_dir / "fusion_weights.json"
    with fusion_weights_path.open("w", encoding="utf-8") as f:
        json.dump(fusion_json, f, indent=2)

    fusion_model = {
        "class_names": CLASS_ORDER,
        "class_order": CLASS_ORDER,
        "temperature": temperature,
        "dem_classifier": dem_model,
        "mola_scaler": mola_scaler,
        "fusion_weights": fusion_w.astype(np.float64),
        "fusion_biases": fusion_b.astype(np.float64),
    }
    fusion_model_path = output_dir / "fusion_model.pt"
    torch.save(fusion_model, fusion_model_path)

    elapsed = time.time() - start_time
    print("\n[done] Saved fused predictions to:", fused_csv_path)
    print("[done] Saved fusion weights JSON to:", fusion_weights_path)
    print("[done] Saved fusion model to:", fusion_model_path)
    print(f"[done] Total time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
