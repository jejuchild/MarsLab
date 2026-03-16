#!/usr/bin/env python3
"""
Test FiLM weight scaling (dampening) at various alpha values.

Evaluates on training data (embeddings_v5 + mola_features_v5).
Goal: find alpha that retains training accuracy while fixing over-conditioning.
"""

import sys
import json
import numpy as np
import torch
from pathlib import Path
from collections import Counter

sys.path.insert(0, "/disk1/cspark/hirise-api")
from scripts.marslandform_v2.models.film_classifier import FiLMClassifier

DATA_DIR = Path("/disk1/cspark/MarsLab/Data/HiRISE/v5_retrain")
CHECKPOINT = DATA_DIR / "film_classifier_v5c.pt"
CLASS_NAMES = ["LDA", "LVF", "CCF", "OTHER"]
LOGIT_BIAS = np.array([-0.25, 0.15, -0.15, 0.0], dtype=np.float32)


def load_training_data():
    """Load aligned embeddings, MOLA features, and labels."""
    embeddings = np.load(DATA_DIR / "embeddings_v5.npy")  # (67904, 768)
    mola_dict = np.load(DATA_DIR / "mola_features_v5.npy", allow_pickle=True).item()
    with open(DATA_DIR / "tile_labels_v5.json") as f:
        labels_list = json.load(f)  # list of dicts

    # Labels are aligned with embeddings by index
    # MOLA is nested: {product_id: {row_col: 25-dim}}
    mola_arr = []
    label_arr = []
    valid_idx = []

    for i, tile_info in enumerate(labels_list):
        pid = tile_info["image_id"]
        row = tile_info["tile_row"]
        col = tile_info["tile_col"]
        tile_key = f"{row}_{col}"

        if pid in mola_dict and tile_key in mola_dict[pid]:
            mola_feat = mola_dict[pid][tile_key]
            mola_arr.append(mola_feat)
            lbl_str = tile_info["label"]
            label_arr.append(CLASS_NAMES.index(lbl_str) if lbl_str in CLASS_NAMES else 3)
            valid_idx.append(i)

    emb_valid = embeddings[valid_idx]
    return emb_valid, np.array(mola_arr, dtype=np.float32), np.array(label_arr)


def load_model(device):
    model = FiLMClassifier(
        visual_dim=768, mola_dim=25, num_classes=4,
        film_hidden=64, head_hidden=128, dropout=0.4
    )
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    model.to(device)
    return model


def evaluate_with_alpha(model, embeddings, mola, alpha, device, logit_bias=None, batch_size=4096):
    """Run inference with a given FiLM dampening alpha."""
    all_preds = []
    all_probs = []

    original_forward = model.film.forward

    def dampened_forward(visual_features, mola_features):
        h = model.film.mola_encoder(mola_features)
        gamma = model.film.gamma_proj(h)
        beta = model.film.beta_proj(h)
        gamma_d = 1.0 + alpha * (gamma - 1.0)
        beta_d = alpha * beta
        return gamma_d * visual_features + beta_d

    if alpha < 1.0:
        model.film.forward = dampened_forward

    n = len(embeddings)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        emb_t = torch.from_numpy(embeddings[start:end]).float().to(device)
        mola_t = torch.from_numpy(mola[start:end]).float().to(device)

        with torch.no_grad():
            logits = model(emb_t, mola_t)

        if logit_bias is not None:
            bias_t = torch.from_numpy(logit_bias).float().to(device)
            logits = logits + bias_t

        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
        all_preds.append(preds)
        all_probs.append(probs)

    if alpha < 1.0:
        model.film.forward = original_forward

    return np.concatenate(all_preds), np.concatenate(all_probs)


def compute_metrics(preds, labels):
    acc = (preds == labels).mean()
    results = {}
    for c in range(4):
        tp = ((preds == c) & (labels == c)).sum()
        fp = ((preds == c) & (labels != c)).sum()
        fn = ((preds != c) & (labels == c)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        results[CLASS_NAMES[c]] = {"f1": f1, "precision": prec, "recall": rec, "support": int((labels == c).sum())}
    macro_f1 = np.mean([r["f1"] for r in results.values()])
    return acc, macro_f1, results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\n=== Loading training data ===")
    embeddings, mola, labels = load_training_data()
    print(f"Loaded: {len(embeddings)} tiles")
    label_counts = Counter(labels)
    print(f"Label distribution: {dict(sorted(label_counts.items()))}")
    print(f"  LDA={label_counts[0]}, LVF={label_counts[1]}, CCF={label_counts[2]}, OTHER={label_counts[3]}")

    print("\n=== Loading model ===")
    model = load_model(device)
    print("Model loaded successfully")

    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    print("\n=== Training Data Evaluation (with logit bias) ===")
    print(f"{'Alpha':>6} | {'Acc':>7} | {'MacF1':>7} | {'LDA-F1':>7} | {'LVF-F1':>7} | {'CCF-F1':>7} | {'OTH-F1':>7} | Pred Distribution")
    print("-" * 100)

    for alpha in alphas:
        preds, probs = evaluate_with_alpha(model, embeddings, mola, alpha, device, LOGIT_BIAS)
        acc, macro_f1, cr = compute_metrics(preds, labels)
        dist = Counter(preds)
        print(f"{alpha:>6.1f} | {acc:>7.4f} | {macro_f1:>7.4f} | "
              f"{cr['LDA']['f1']:>7.4f} | {cr['LVF']['f1']:>7.4f} | "
              f"{cr['CCF']['f1']:>7.4f} | {cr['OTHER']['f1']:>7.4f} | "
              f"LDA={dist.get(0,0):>5} LVF={dist.get(1,0):>5} CCF={dist.get(2,0):>5} OTH={dist.get(3,0):>5}")

    print("\n=== No logit bias (raw model) ===")
    print(f"{'Alpha':>6} | {'Acc':>7} | {'MacF1':>7} | {'LDA-F1':>7} | {'LVF-F1':>7} | {'CCF-F1':>7} | {'OTH-F1':>7}")
    print("-" * 75)

    for alpha in [0.0, 0.3, 0.5, 1.0]:
        preds, probs = evaluate_with_alpha(model, embeddings, mola, alpha, device, None)
        acc, macro_f1, cr = compute_metrics(preds, labels)
        print(f"{alpha:>6.1f} | {acc:>7.4f} | {macro_f1:>7.4f} | "
              f"{cr['LDA']['f1']:>7.4f} | {cr['LVF']['f1']:>7.4f} | "
              f"{cr['CCF']['f1']:>7.4f} | {cr['OTHER']['f1']:>7.4f}")


if __name__ == "__main__":
    main()
