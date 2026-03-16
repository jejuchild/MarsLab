#!/usr/bin/env python3
"""
Optimize dual-path ensemble: HiRISE (visual) = main, MOLA = sub.

For each visual weight, find the optimal logit bias that maximizes macro F1.
Then test on ESP_024943_2345.
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
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model.to(device)


def get_logits(model, emb_t, mola_t, alpha):
    """Get logits with given FiLM alpha."""
    original = model.film.forward
    def damp(vis, mola, a=alpha):
        h = model.film.mola_encoder(mola)
        g = model.film.gamma_proj(h)
        b = model.film.beta_proj(h)
        return (1.0 + a * (g - 1.0)) * vis + a * b
    model.film.forward = damp
    with torch.no_grad():
        logits = model(emb_t, mola_t)
    model.film.forward = original
    return logits.cpu().numpy()


def compute_macro_f1(preds, labels):
    f1s = []
    for c in range(4):
        tp = ((preds == c) & (labels == c)).sum()
        fp = ((preds == c) & (labels != c)).sum()
        fn = ((preds != c) & (labels == c)).sum()
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        f1s.append(f1)
    return np.mean(f1s), f1s


def optimize_bias(logits, labels):
    best_f1, best_bias = 0, np.zeros(4)
    # Coarse
    for b0 in np.arange(-2.0, 2.1, 0.2):
        for b1 in np.arange(-2.0, 2.1, 0.2):
            for b2 in np.arange(-2.0, 2.1, 0.2):
                bias = np.array([b0, b1, b2, 0.0])
                preds = np.argmax(logits + bias, axis=1)
                f1, _ = compute_macro_f1(preds, labels)
                if f1 > best_f1:
                    best_f1 = f1
                    best_bias = bias.copy()
    # Fine
    for d0 in np.arange(-0.25, 0.26, 0.05):
        for d1 in np.arange(-0.25, 0.26, 0.05):
            for d2 in np.arange(-0.25, 0.26, 0.05):
                bias = best_bias + np.array([d0, d1, d2, 0.0])
                preds = np.argmax(logits + bias, axis=1)
                f1, _ = compute_macro_f1(preds, labels)
                if f1 > best_f1:
                    best_f1 = f1
                    best_bias = bias.copy()
    return best_bias, best_f1


def main():
    device = torch.device("cpu")
    model = load_model(device)

    embeddings, mola, labels = load_training_data()
    print(f"Training: {len(embeddings)} tiles")
    emb_t = torch.from_numpy(embeddings).float().to(device)
    mola_t = torch.from_numpy(mola).float().to(device)

    # Pre-compute logits for alpha=0 and alpha=1
    print("Computing visual-only logits (alpha=0)...")
    logits_vis = get_logits(model, emb_t, mola_t, alpha=0.0)
    print("Computing full-FiLM logits (alpha=1)...")
    logits_film = get_logits(model, emb_t, mola_t, alpha=1.0)

    # Test: HiRISE main (0.6-0.9), MOLA sub (0.1-0.4)
    configs = [
        (0.9, 0.1),
        (0.8, 0.2),
        (0.75, 0.25),
        (0.7, 0.3),
        (0.6, 0.4),
        (0.5, 0.5),
        (0.0, 1.0),  # pure FiLM baseline
        (1.0, 0.0),  # pure visual baseline
    ]

    print(f"\n{'w_vis':>6} | {'w_mola':>6} | {'Opt Bias':>30} | {'MacF1':>7} | {'Acc':>7} | LDA-F1 | LVF-F1 | CCF-F1 | OTH-F1 | Pred Dist")
    print("-" * 140)

    results = {}
    for w_vis, w_mola in configs:
        ensemble = w_vis * logits_vis + w_mola * logits_film
        bias, f1 = optimize_bias(ensemble, labels)
        preds = np.argmax(ensemble + bias, axis=1)
        acc = (preds == labels).mean()
        _, f1_per = compute_macro_f1(preds, labels)
        dist = Counter(preds)
        bias_str = f"[{bias[0]:+.2f}, {bias[1]:+.2f}, {bias[2]:+.2f}, {bias[3]:+.2f}]"
        f1_str = " | ".join(f"{f:.4f}" for f in f1_per)
        dist_str = f"L={dist.get(0,0):>5} V={dist.get(1,0):>5} C={dist.get(2,0):>5} O={dist.get(3,0):>5}"
        print(f"{w_vis:>6.2f} | {w_mola:>6.2f} | {bias_str:>30} | {f1:>7.4f} | {acc:>7.4f} | {f1_str} | {dist_str}")
        results[(w_vis, w_mola)] = (bias, f1)

    # Print recommendation
    best = max(results.items(), key=lambda x: x[1][1])
    print(f"\nBest config: w_vis={best[0][0]}, w_mola={best[0][1]}, bias={best[1][0].tolist()}, F1={best[1][1]:.4f}")


if __name__ == "__main__":
    main()
