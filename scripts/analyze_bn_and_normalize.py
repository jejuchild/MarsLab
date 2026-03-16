#!/usr/bin/env python3
"""
Analyze BatchNorm running stats vs actual training MOLA distribution,
and test the effect of explicit z-score normalization on OOD MOLA inputs.
"""
import sys
import json
import numpy as np
import torch

sys.path.insert(0, "/disk1/cspark/hirise-api")
from scripts.marslandform_v2.models.late_fusion_classifier import LateFusionClassifier

DATA_DIR = "/disk1/cspark/MarsLab/Data/HiRISE/v5_retrain"
CLASS_NAMES = ["LDA", "LVF", "CCF", "OTHER"]

# Load V6 model
ckpt = torch.load(f"{DATA_DIR}/late_fusion_v6.pt", map_location="cpu", weights_only=False)
model = LateFusionClassifier(**{k: v for k, v in ckpt["cfg"].items() if k != "class_names" and k != "architecture"})
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Extract BatchNorm running stats
bn = model.mola_head[0]  # First layer is BatchNorm1d
print("=== BatchNorm1d Running Stats ===")
print(f"running_mean shape: {bn.running_mean.shape}")
print(f"running_var shape:  {bn.running_var.shape}")
print()

bn_mean = bn.running_mean.numpy()
bn_var = bn.running_var.numpy()
bn_std = np.sqrt(bn_var + 1e-5)

# Load actual training MOLA data
mola_dict = np.load(f"{DATA_DIR}/mola_features_v5.npy", allow_pickle=True).item()
with open(f"{DATA_DIR}/tile_labels_v5.json") as f:
    labels_list = json.load(f)

mola_list = []
for tile_info in labels_list:
    pid = tile_info["image_id"]
    key = f"{tile_info['tile_row']}_{tile_info['tile_col']}"
    if pid in mola_dict and key in mola_dict[pid]:
        mola_list.append(mola_dict[pid][key])

mola_arr = np.array(mola_list, dtype=np.float32)
actual_mean = mola_arr.mean(axis=0)
actual_std = mola_arr.std(axis=0)

print(f"{'Feat':>5} | {'BN_mean':>12} | {'Actual_mean':>12} | {'BN_std':>12} | {'Actual_std':>12} | {'Match?':>8}")
print("-" * 80)
for i in range(25):
    match = "✓" if abs(bn_mean[i] - actual_mean[i]) / (abs(actual_mean[i]) + 1e-8) < 0.1 else "✗"
    print(f"{i:>5} | {bn_mean[i]:>12.4f} | {actual_mean[i]:>12.4f} | {bn_std[i]:>12.4f} | {actual_std[i]:>12.4f} | {match:>8}")

# Now test: what does the MOLA head produce for ESP_024943_2345's features?
print("\n\n=== ESP_024943_2345 MOLA Feature Analysis ===")

# Typical ESP_024943 MOLA features (from previous analysis)
# It has 23 features (missing rel_elev and rel_slope), padded to 25
esp_mola = np.array([
    0.1055, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,   # slope histogram
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # more histogram bins
    -4251.0,  # elevation
    54.34,    # latitude
    0.0, 0.0  # rel_elev, rel_slope (padded zeros)
], dtype=np.float32)

# What BN does: (x - running_mean) / sqrt(running_var + eps) * weight + bias
bn_weight = bn.weight.detach().numpy()
bn_bias = bn.bias.detach().numpy()

normalized_by_bn = (esp_mola - bn_mean) / bn_std * bn_weight + bn_bias
print(f"\nESP_024943 raw features (first 5): {esp_mola[:5]}")
print(f"After BN normalization (first 5):  {normalized_by_bn[:5]}")

# Now compare: what if we do explicit z-score normalization BEFORE model?
z_scored = (esp_mola - actual_mean) / (actual_std + 1e-8)
print(f"\nAfter z-score normalization (first 5): {z_scored[:5]}")
print(f"Then BN would see (first 5):           {(z_scored - bn_mean) / bn_std}")

# The key question: are BN stats close to actual distribution?
# If yes, BN already normalizes correctly. If not, explicit z-score helps.
print("\n\n=== KEY INSIGHT ===")
rel_diff_mean = np.abs(bn_mean - actual_mean) / (np.abs(actual_mean) + 1e-8)
rel_diff_std = np.abs(bn_std - actual_std) / (np.abs(actual_std) + 1e-8)
print(f"Mean relative difference (BN vs actual mean): {rel_diff_mean.mean():.4f}")
print(f"Mean relative difference (BN vs actual std):  {rel_diff_std.mean():.4f}")
print(f"Max relative difference (mean): feat[{rel_diff_mean.argmax()}] = {rel_diff_mean.max():.4f}")
print(f"Max relative difference (std):  feat[{rel_diff_std.argmax()}] = {rel_diff_std.max():.4f}")

# Test MOLA logits with and without pre-normalization
mola_t = torch.from_numpy(esp_mola).float().unsqueeze(0)
with torch.no_grad():
    mola_logits_raw = model.get_mola_logits(mola_t)
print(f"\nMOLA logits (raw input):     {mola_logits_raw.numpy().flatten()}")
print(f"  → class: {CLASS_NAMES[mola_logits_raw.argmax().item()]}")

# With z-score pre-normalization
z_t = torch.from_numpy(z_scored).float().unsqueeze(0)
with torch.no_grad():
    mola_logits_zscore = model.get_mola_logits(z_t)
print(f"MOLA logits (z-scored input): {mola_logits_zscore.numpy().flatten()}")
print(f"  → class: {CLASS_NAMES[mola_logits_zscore.argmax().item()]}")

# Save training stats for use in normalization
print("\n\n=== Saving Training MOLA Stats ===")
stats = {
    "mean": actual_mean.tolist(),
    "std": actual_std.tolist(),
    "n_samples": len(mola_list),
    "n_features": 25,
}
np.savez(f"{DATA_DIR}/mola_training_stats.npz", mean=actual_mean, std=actual_std)
print(f"Saved to {DATA_DIR}/mola_training_stats.npz")
print(f"Mean: {actual_mean}")
print(f"Std:  {actual_std}")
