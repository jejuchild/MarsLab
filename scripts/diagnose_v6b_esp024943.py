#!/usr/bin/env python3
"""
Diagnose V6b predictions on ESP_024943_2345 using the actual pipeline flow.
"""
import sys
import numpy as np
import torch

sys.path.insert(0, "/disk1/cspark/hirise-api")
sys.path.insert(0, "/disk1/cspark/MarsLab")

from scripts.marslandform_v2.models.late_fusion_classifier import LateFusionClassifier
from backend.analysis.hirise_landforms.preprocessing import (
    extract_mola_features, extract_mola_features_batch, fetch_hirise_browse, tile_image
)

DATA_DIR = "/disk1/cspark/MarsLab/Data/HiRISE/v5_retrain"
CLASS_NAMES = ["LDA", "LVF", "CCF", "OTHER"]

# Load V6b model
ckpt = torch.load(f"{DATA_DIR}/late_fusion_v6b.pt", map_location="cpu", weights_only=False)
cfg = ckpt["cfg"]
model = LateFusionClassifier(
    visual_dim=cfg["visual_dim"],
    mola_dim=cfg["mola_dim"],
    visual_hidden=cfg["visual_hidden"],
    mola_hidden=cfg["mola_hidden"],
    num_classes=cfg["num_classes"],
    dropout=cfg["dropout"],
)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

print(f"V6b: vis={model.vis_weight.item():.4f}, mola={model.mola_weight.item():.4f}")
print(f"mola_mean[:5] = {model.mola_mean[:5].numpy()}")
print(f"mola_std[:5] = {model.mola_std[:5].numpy()}")

# Check: do mola_mean/std have reasonable values (not zeros)?
print(f"\nmola_mean all zeros? {(model.mola_mean == 0).all().item()}")
print(f"mola_std all ones?  {(model.mola_std == 1).all().item()}")

# Simulate with known MOLA values
print("\n=== Synthetic test ===")
# Typical LDA MOLA features (mid-latitude, some slope, moderate elevation)
lda_mola = np.zeros(25, dtype=np.float32)
lda_mola[0] = 5.0    # typical slope for LDA
lda_mola[4] = 10.0   # slope histogram bin
lda_mola[5] = 35.0   # slope histogram bin
lda_mola[21] = -2300  # typical LDA elevation
lda_mola[22] = 42.0   # typical LDA latitude

# ESP_024943 MOLA features (very low slope, high latitude)
esp_mola = np.zeros(25, dtype=np.float32)
esp_mola[0] = 0.1055
esp_mola[21] = -4251.0
esp_mola[22] = 54.34

for name, feat in [("Typical LDA", lda_mola), ("ESP_024943", esp_mola)]:
    feat_t = torch.from_numpy(feat).float().unsqueeze(0)
    normed = model._normalize_mola(feat_t)
    with torch.no_grad():
        mola_logits = model.get_mola_logits(feat_t)
        vis_logits = model.get_visual_logits(torch.randn(1, 768))  # random visual for comparison
    print(f"\n{name}:")
    print(f"  Raw: {feat[[0,21,22]]}")
    print(f"  Normed: {normed.numpy().flatten()[[0,21,22]]}")
    print(f"  MOLA logits: {mola_logits.numpy().flatten()} → {CLASS_NAMES[mola_logits.argmax().item()]}")

# Now load actual DINOv2 backbone and process ESP_024943
print("\n\n=== Loading DINOv2 backbone for actual visual features ===")
from torchvision import transforms
from PIL import Image
import transformers

backbone = transformers.AutoModel.from_pretrained("facebook/dinov2-base")
backbone.eval()

tile_transform = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

# Fetch and tile the ESP_024943 image
print("Fetching ESP_024943_2345 browse image...")
img = fetch_hirise_browse("ESP_024943_2345")
if img is not None:
    print(f"Image size: {img.size}")
    tile_results = tile_image(img, tile_size=224)
    tiles = [t[2] for t in tile_results]  # (x, y, PIL.Image)
    coords = [(t[1], t[0]) for t in tile_results]  # (row=y, col=x)
    print(f"Tiles: {len(tiles)}")
    
    # Extract embeddings
    embeddings = []
    for i, tile in enumerate(tiles):
        tile_t = tile_transform(tile).unsqueeze(0)
        with torch.no_grad():
            out = backbone(tile_t)
            emb = out.last_hidden_state[:, 0, :]  # CLS token
        embeddings.append(emb.numpy())
    
    embeddings = np.concatenate(embeddings, axis=0)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Extract MOLA features
    mola_feats = extract_mola_features_batch("ESP_024943_2345", coords)
    print(f"MOLA features: {len(mola_feats)} tiles")
    
    # Build MOLA array (pad to 25 if needed)
    mola_arr = []
    for r, c in coords:
        key = f"{r}_{c}"
        if key in mola_feats:
            feat = np.array(mola_feats[key], dtype=np.float32)
            if len(feat) < 25:
                feat = np.pad(feat, (0, 25 - len(feat)))
            mola_arr.append(feat[:25])
        else:
            mola_arr.append(np.zeros(25, dtype=np.float32))
    mola_arr = np.array(mola_arr, dtype=np.float32)
    print(f"MOLA array shape: {mola_arr.shape}")
    print(f"MOLA sample [0]: slope={mola_arr[0,0]:.4f}, elev={mola_arr[0,21]:.1f}, lat={mola_arr[0,22]:.2f}")
    
    # Run model
    emb_t = torch.from_numpy(embeddings).float()
    mola_t = torch.from_numpy(mola_arr).float()
    
    with torch.no_grad():
        logits = model(emb_t, mola_t)
        vis_logits = model.get_visual_logits(emb_t)
        mola_logits = model.get_mola_logits(mola_t)
    
    probs = torch.softmax(logits, dim=1).numpy()
    vis_probs = torch.softmax(vis_logits, dim=1).numpy()
    mola_probs = torch.softmax(mola_logits, dim=1).numpy()
    
    preds = np.argmax(probs, axis=1)
    vis_preds = np.argmax(vis_probs, axis=1)
    mola_preds = np.argmax(mola_probs, axis=1)
    
    print(f"\n=== Results ===")
    print(f"{'':>15} | {'LDA':>5} | {'LVF':>5} | {'CCF':>5} | {'OTHER':>5}")
    print(f"{'Combined':>15} | {(preds==0).sum():>5} | {(preds==1).sum():>5} | {(preds==2).sum():>5} | {(preds==3).sum():>5}")
    print(f"{'Visual-only':>15} | {(vis_preds==0).sum():>5} | {(vis_preds==1).sum():>5} | {(vis_preds==2).sum():>5} | {(vis_preds==3).sum():>5}")
    print(f"{'MOLA-only':>15} | {(mola_preds==0).sum():>5} | {(mola_preds==1).sum():>5} | {(mola_preds==2).sum():>5} | {(mola_preds==3).sum():>5}")
    
    print(f"\nVisual logit stats:")
    for c in range(4):
        col = vis_logits.numpy()[:, c]
        print(f"  {CLASS_NAMES[c]}: mean={col.mean():.3f}, std={col.std():.3f}, min={col.min():.3f}, max={col.max():.3f}")
    
    print(f"\nMOLA logit stats:")
    for c in range(4):
        col = mola_logits.numpy()[:, c]
        print(f"  {CLASS_NAMES[c]}: mean={col.mean():.3f}, std={col.std():.3f}, min={col.min():.3f}, max={col.max():.3f}")
    
    print(f"\nCombined logit stats:")
    for c in range(4):
        col = logits.numpy()[:, c]
        print(f"  {CLASS_NAMES[c]}: mean={col.mean():.3f}, std={col.std():.3f}, min={col.min():.3f}, max={col.max():.3f}")
    
    # Show tiles where visual says non-OTHER
    non_other_vis = np.where(vis_preds != 3)[0]
    print(f"\n{len(non_other_vis)} tiles where visual predicts non-OTHER:")
    for idx in non_other_vis[:10]:
        r, c = coords[idx]
        print(f"  Tile ({r},{c}): vis={CLASS_NAMES[vis_preds[idx]]} (conf={vis_probs[idx].max():.3f}), "
              f"combined={CLASS_NAMES[preds[idx]]} (conf={probs[idx].max():.3f})")
else:
    print("Failed to fetch image")
