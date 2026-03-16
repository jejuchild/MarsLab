#!/usr/bin/env python3
"""
Optimize logit bias for a given FiLM dampening alpha.

For each alpha, find the logit bias that maximizes macro F1 on training data.
Then test ESP_024943_2345 with that alpha + bias combination.
"""

import sys
import json
import numpy as np
import torch
from pathlib import Path
from collections import Counter, defaultdict
from itertools import product as iter_product

sys.path.insert(0, "/disk1/cspark/hirise-api")
from scripts.marslandform_v2.models.film_classifier import FiLMClassifier

DATA_DIR = Path("/disk1/cspark/MarsLab/Data/HiRISE/v5_retrain")
CHECKPOINT = DATA_DIR / "film_classifier_v5c.pt"
CLASS_NAMES = ["LDA", "LVF", "CCF", "OTHER"]


def load_training_data():
    embeddings = np.load(DATA_DIR / "embeddings_v5.npy")
    mola_dict = np.load(DATA_DIR / "mola_features_v5.npy", allow_pickle=True).item()
    with open(DATA_DIR / "tile_labels_v5.json") as f:
        labels_list = json.load(f)

    mola_arr, label_arr, valid_idx = [], [], []
    for i, tile_info in enumerate(labels_list):
        pid = tile_info["image_id"]
        key = f"{tile_info['tile_row']}_{tile_info['tile_col']}"
        if pid in mola_dict and key in mola_dict[pid]:
            mola_arr.append(mola_dict[pid][key])
            lbl = tile_info["label"]
            label_arr.append(CLASS_NAMES.index(lbl) if lbl in CLASS_NAMES else 3)
            valid_idx.append(i)

    return embeddings[valid_idx], np.array(mola_arr, dtype=np.float32), np.array(label_arr)


def load_model(device):
    model = FiLMClassifier(visual_dim=768, mola_dim=25, num_classes=4,
                           film_hidden=64, head_hidden=128, dropout=0.4)
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(sd)
    model.eval()
    return model.to(device)


def get_raw_logits(model, embeddings, mola, alpha, device, batch_size=4096):
    """Get raw logits (no bias) with given alpha."""
    all_logits = []
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

    for start in range(0, len(embeddings), batch_size):
        end = min(start + batch_size, len(embeddings))
        emb_t = torch.from_numpy(embeddings[start:end]).float().to(device)
        mola_t = torch.from_numpy(mola[start:end]).float().to(device)
        with torch.no_grad():
            logits = model(emb_t, mola_t)
        all_logits.append(logits.cpu().numpy())

    if alpha < 1.0:
        model.film.forward = original_forward

    return np.concatenate(all_logits)


def compute_macro_f1(preds, labels):
    f1s = []
    for c in range(4):
        tp = ((preds == c) & (labels == c)).sum()
        fp = ((preds == c) & (labels != c)).sum()
        fn = ((preds != c) & (labels == c)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        f1s.append(f1)
    return np.mean(f1s)


def optimize_bias(logits, labels):
    """Grid search for optimal per-class bias to maximize macro F1."""
    best_f1, best_bias = 0, np.zeros(4)

    # Coarse grid
    bias_range = np.arange(-1.0, 1.1, 0.1)
    # Only search LDA, LVF, CCF biases (OTHER fixed at 0)
    for b0 in bias_range:
        for b1 in bias_range:
            for b2 in bias_range:
                bias = np.array([b0, b1, b2, 0.0])
                preds = np.argmax(logits + bias, axis=1)
                f1 = compute_macro_f1(preds, labels)
                if f1 > best_f1:
                    best_f1 = f1
                    best_bias = bias.copy()

    # Fine grid around best
    fine_range = np.arange(-0.15, 0.16, 0.05)
    for d0 in fine_range:
        for d1 in fine_range:
            for d2 in fine_range:
                bias = best_bias + np.array([d0, d1, d2, 0.0])
                preds = np.argmax(logits + bias, axis=1)
                f1 = compute_macro_f1(preds, labels)
                if f1 > best_f1:
                    best_f1 = f1
                    best_bias = bias.copy()

    return best_bias, best_f1


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    embeddings, mola, labels = load_training_data()
    print(f"Training tiles: {len(embeddings)}")

    model = load_model(device)

    test_alphas = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    print(f"\n{'Alpha':>6} | {'Opt Bias':>30} | {'MacF1':>7} | {'Acc':>7} | Pred Distribution")
    print("-" * 110)

    for alpha in test_alphas:
        print(f"\n--- Optimizing for alpha={alpha} ---")
        logits = get_raw_logits(model, embeddings, mola, alpha, device)

        bias, f1 = optimize_bias(logits, labels)
        preds = np.argmax(logits + bias, axis=1)
        acc = (preds == labels).mean()
        dist = Counter(preds)

        bias_str = f"[{bias[0]:+.2f}, {bias[1]:+.2f}, {bias[2]:+.2f}, {bias[3]:+.2f}]"
        print(f"{alpha:>6.1f} | {bias_str:>30} | {f1:>7.4f} | {acc:>7.4f} | "
              f"LDA={dist.get(0,0):>5} LVF={dist.get(1,0):>5} CCF={dist.get(2,0):>5} OTH={dist.get(3,0):>5}")

        # Per-class F1
        for c in range(4):
            tp = ((preds == c) & (labels == c)).sum()
            fp = ((preds == c) & (labels != c)).sum()
            fn = ((preds != c) & (labels == c)).sum()
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1c = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            print(f"  {CLASS_NAMES[c]}: P={prec:.4f} R={rec:.4f} F1={f1c:.4f} (support={int((labels==c).sum())})")


if __name__ == "__main__":
    main()
