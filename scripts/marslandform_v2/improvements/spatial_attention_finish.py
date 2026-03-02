#!/usr/bin/env python3
"""Finish generating spatial attention overlays (skip already-generated ones)."""
import csv
import json
import sys
import numpy as np
from pathlib import Path
from copy import deepcopy
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from scipy.ndimage import gaussian_filter

from scripts.marslandform_v2.config import CLASS_ORDER, get_config
from scripts.marslandform_v2.models.mil_classifier import (
    AttentionMILClassifier, MILDataset, load_embeddings,
    load_labels, load_mola_features, mil_collate_fn, set_seed,
)
from torch.utils.data import DataLoader

DATA_ROOT = ROOT / "Data/HiRISE/v2_output"
BROWSE_DIR = ROOT / "Data/HiRISE/midlat_browse"
EVAL_DIR = DATA_ROOT / "eval"
TILE_SIZE = 224

CLASS_COLORS = {
    "LDA": "#e74c3c", "LVF": "#3498db", "CCF": "#2ecc71",
    "GLF": "#f39c12", "BACKGROUND": "#95a5a6",
}

def load_tile_metadata():
    meta = defaultdict(list)
    csv_path = DATA_ROOT / "tile_metadata.csv"
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            meta[row["image_id"]].append({
                "tile_idx": int(row["tile_idx"]),
                "tile_row": int(row["tile_row"]),
                "tile_col": int(row["tile_col"]),
                "lat": float(row["lat"]) if row["lat"] else None,
                "lon": float(row["lon"]) if row["lon"] else None,
                "tile_path": row["tile_path"],
                "content_fraction": float(row["content_fraction"]),
            })
    return dict(meta)


def load_best_model():
    cfg = get_config()
    mil_cfg = cfg.mil
    device = torch.device("cpu")
    model_path = DATA_ROOT / "models/cleaned_seed_123/best_mil_model.pt"
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    saved_cfg = checkpoint.get("mil_config", {})
    model_cfg = deepcopy(mil_cfg)
    for k, v in saved_cfg.items():
        if hasattr(model_cfg, k):
            setattr(model_cfg, k, v)
    model = AttentionMILClassifier(model_cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, device


def get_predictions_with_attention(model, device):
    emb_dict = load_embeddings(DATA_ROOT / "embeddings_mil")
    mola_dict = load_mola_features(DATA_ROOT / "mola_features_by_image.npy")
    labels_dict = load_labels(DATA_ROOT / "label_audit/labels_cleaned.json")
    split = json.loads((DATA_ROOT / "models/multihead_improved/data_split.json").read_text())
    valid_ids = set(emb_dict.keys()) & set(mola_dict.keys()) & set(labels_dict.keys())
    test_ids = [i for i in split["test_ids"] if i in valid_ids]
    test_ds = MILDataset(test_ids, emb_dict, mola_dict, labels_dict, 1, 128)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=mil_collate_fn, num_workers=0)
    results = []
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            tiles = batch["tile_embeddings"].to(device)
            mask = batch["tile_mask"].to(device)
            mola = batch["mola_features"].to(device)
            labels = batch["labels"]
            logits, attention = model(tiles, mask, mola)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            pred_idx = logits.argmax(dim=1).item()
            img_id = batch["image_ids"][0]
            n_real = mask[0].sum().item()
            att = attention[0, :int(n_real)].cpu().numpy()
            results.append({
                "image_id": img_id,
                "true": CLASS_ORDER[labels[0].item()],
                "pred": CLASS_ORDER[pred_idx],
                "confidence": float(probs[0][pred_idx]),
                "correct": CLASS_ORDER[labels[0].item()] == CLASS_ORDER[pred_idx],
                "attention": att,
                "n_tiles": int(n_real),
            })
            if (idx + 1) % 20 == 0:
                print(f"  Inference: {idx+1}/{len(test_loader)}")
    return results


def generate_attention_overlay(image_id, attention_weights, tile_meta_list, pred_cls, true_cls, confidence, correct, out_path):
    browse_path = BROWSE_DIR / f"{image_id}_RED.abrowse.jpg"
    if not browse_path.exists():
        return False
    browse_img = Image.open(browse_path).convert("RGB")
    img_w, img_h = browse_img.size
    browse_arr = np.array(browse_img)
    if not tile_meta_list:
        return False
    max_row = max(t["tile_row"] for t in tile_meta_list)
    max_col = max(t["tile_col"] for t in tile_meta_list)
    att_grid = np.full((max_row + 1, max_col + 1), np.nan)
    sorted_meta = sorted(tile_meta_list, key=lambda t: t["tile_idx"])
    n_att = len(attention_weights)
    for i, tile_info in enumerate(sorted_meta):
        if i < n_att:
            att_grid[tile_info["tile_row"], tile_info["tile_col"]] = attention_weights[i]
    valid_mask = ~np.isnan(att_grid)
    if not valid_mask.any():
        return False
    att_min, att_max = np.nanmin(att_grid), np.nanmax(att_grid)
    if att_max > att_min:
        att_norm = (att_grid - att_min) / (att_max - att_min)
    else:
        att_norm = np.where(valid_mask, 0.5, np.nan)

    grid_h, grid_w = att_grid.shape
    heatmap_full = np.zeros((img_h, img_w), dtype=np.float64)
    weight_map = np.zeros((img_h, img_w), dtype=np.float64)
    for r in range(grid_h):
        for c in range(grid_w):
            if np.isnan(att_norm[r, c]):
                continue
            y0 = r * TILE_SIZE; x0 = c * TILE_SIZE
            y1 = min(y0 + TILE_SIZE, img_h); x1 = min(x0 + TILE_SIZE, img_w)
            heatmap_full[y0:y1, x0:x1] = att_norm[r, c]
            weight_map[y0:y1, x0:x1] = 1.0
    heatmap_smooth = gaussian_filter(heatmap_full, sigma=TILE_SIZE * 0.4)
    weight_smooth = gaussian_filter(weight_map, sigma=TILE_SIZE * 0.4)
    weight_smooth[weight_smooth == 0] = 1.0
    heatmap_smooth = heatmap_smooth / weight_smooth
    hm_min, hm_max = heatmap_smooth.min(), heatmap_smooth.max()
    if hm_max > hm_min:
        heatmap_smooth = (heatmap_smooth - hm_min) / (hm_max - hm_min)

    flat_att = []
    for i, tile_info in enumerate(sorted_meta):
        if i < n_att:
            flat_att.append((attention_weights[i], tile_info))
    flat_att.sort(key=lambda x: -x[0])
    top5 = flat_att[:5]

    fig = plt.figure(figsize=(20, 8))
    ax1 = fig.add_axes([0.01, 0.08, 0.30, 0.82])
    ax1.imshow(browse_arr); ax1.set_title("Original Browse Image", fontsize=11, fontweight="bold"); ax1.axis("off")
    ax2 = fig.add_axes([0.33, 0.08, 0.30, 0.82])
    ax2.imshow(browse_arr)
    cmap = plt.cm.jet.copy(); cmap.set_bad(alpha=0)
    heatmap_masked = np.ma.masked_where(weight_map == 0, heatmap_smooth)
    im = ax2.imshow(heatmap_masked, cmap=cmap, alpha=0.55, vmin=0, vmax=1)
    for rank, (att_w, tile_info) in enumerate(top5):
        y0 = tile_info["tile_row"] * TILE_SIZE; x0 = tile_info["tile_col"] * TILE_SIZE
        rect = Rectangle((x0, y0), TILE_SIZE, TILE_SIZE, linewidth=2, edgecolor="white", facecolor="none")
        ax2.add_patch(rect)
        ax2.text(x0+5, y0+20, f"#{rank+1}", color="white", fontsize=8, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7))
    ax2.set_title("Attention Heatmap Overlay\n(hot = high attention)", fontsize=11, fontweight="bold"); ax2.axis("off")
    cbar_ax = fig.add_axes([0.635, 0.08, 0.008, 0.82])
    fig.colorbar(im, cax=cbar_ax, label="Attention")
    ax3_base = fig.add_axes([0.68, 0.08, 0.30, 0.82]); ax3_base.axis("off")
    ax3_base.set_title("Top-5 Attention Tiles", fontsize=11, fontweight="bold")
    for rank, (att_w, tile_info) in enumerate(top5):
        tile_path = DATA_ROOT / tile_info["tile_path"]
        if tile_path.exists():
            tile_img = Image.open(tile_path).convert("RGB")
            ax_tile = fig.add_axes([0.69 + (rank % 3) * 0.10, 0.52 - (rank // 3) * 0.45, 0.09, 0.35])
            ax_tile.imshow(np.array(tile_img))
            att_pct = att_w / sum(a for a, _ in flat_att) * 100
            ax_tile.set_title(f"#{rank+1}: {att_pct:.1f}%", fontsize=8, fontweight="bold"); ax_tile.axis("off")
            for spine in ax_tile.spines.values():
                spine.set_visible(True); spine.set_edgecolor("red" if rank == 0 else "orange"); spine.set_linewidth(2)
    status = "\u2713" if correct else "\u2717"; status_color = "#2ecc71" if correct else "#e74c3c"
    fig.suptitle(f"{image_id}  |  True: {true_cls}  |  Pred: {pred_cls} ({confidence:.0%})  |  {status}",
                 fontsize=14, fontweight="bold", color=status_color, y=0.98)
    plt.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="white"); plt.close()
    return True


def generate_class_gallery(results, tile_meta, out_dir):
    print("Generating per-class localization galleries...")
    for cls in CLASS_ORDER[:4]:
        cls_results = [r for r in results if r["true"] == cls and r["correct"]]
        cls_results.sort(key=lambda x: -x["confidence"])
        n_show = min(6, len(cls_results))
        if n_show == 0:
            continue
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f"Landform Localization: {cls}", fontsize=14, fontweight="bold", y=0.98)
        for i in range(6):
            ax = axes[i // 3][i % 3]
            if i >= n_show:
                ax.axis("off"); continue
            r = cls_results[i]
            img_id = r["image_id"]
            meta_list = tile_meta.get(img_id, [])
            browse_path = BROWSE_DIR / f"{img_id}_RED.abrowse.jpg"
            if not browse_path.exists() or not meta_list:
                ax.axis("off"); continue
            browse_img = np.array(Image.open(browse_path).convert("RGB"))
            img_h, img_w = browse_img.shape[:2]
            max_row = max(t["tile_row"] for t in meta_list)
            max_col = max(t["tile_col"] for t in meta_list)
            att_grid = np.full((max_row + 1, max_col + 1), np.nan)
            sorted_meta = sorted(meta_list, key=lambda t: t["tile_idx"])
            for j, ti in enumerate(sorted_meta):
                if j < len(r["attention"]):
                    att_grid[ti["tile_row"], ti["tile_col"]] = r["attention"][j]
            valid = ~np.isnan(att_grid)
            if not valid.any():
                ax.axis("off"); continue
            att_min, att_max = np.nanmin(att_grid), np.nanmax(att_grid)
            att_norm = (att_grid - att_min) / (att_max - att_min) if att_max > att_min else np.where(valid, 0.5, np.nan)
            heatmap = np.zeros((img_h, img_w)); weight = np.zeros((img_h, img_w))
            for rr in range(att_grid.shape[0]):
                for cc in range(att_grid.shape[1]):
                    if np.isnan(att_norm[rr, cc]): continue
                    y0, x0 = rr*TILE_SIZE, cc*TILE_SIZE
                    y1, x1 = min(y0+TILE_SIZE, img_h), min(x0+TILE_SIZE, img_w)
                    heatmap[y0:y1, x0:x1] = att_norm[rr, cc]; weight[y0:y1, x0:x1] = 1.0
            heatmap = gaussian_filter(heatmap, sigma=TILE_SIZE*0.3)
            ws = gaussian_filter(weight, sigma=TILE_SIZE*0.3); ws[ws==0]=1.0; heatmap = heatmap/ws
            hm_min, hm_max = heatmap.min(), heatmap.max()
            if hm_max > hm_min: heatmap = (heatmap - hm_min)/(hm_max - hm_min)
            heatmap_masked = np.ma.masked_where(weight == 0, heatmap)
            ax.imshow(browse_img); ax.imshow(heatmap_masked, cmap="jet", alpha=0.5, vmin=0, vmax=1)
            ax.set_title(f"{img_id[:20]}\nConf: {r['confidence']:.0%}", fontsize=9); ax.axis("off")
            for spine in ax.spines.values():
                spine.set_visible(True); spine.set_edgecolor(CLASS_COLORS[cls]); spine.set_linewidth(3)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        out_path = out_dir / f"localization_{cls}.png"
        plt.savefig(out_path, dpi=120, bbox_inches="tight"); plt.close()
        print(f"  Saved: {out_path}")


def main():
    print("=" * 60)
    print("SPATIAL ATTENTION OVERLAY — COMPLETION RUN")
    print("=" * 60)
    set_seed(42)
    loc_dir = EVAL_DIR / "localization"
    loc_dir.mkdir(parents=True, exist_ok=True)

    # Check what already exists
    existing = set(p.name for p in loc_dir.glob("heatmap_*.png"))
    print(f"Already generated: {len(existing)} overlays")

    print("Loading model and tile metadata...")
    model, device = load_best_model()
    tile_meta = load_tile_metadata()
    print(f"  Tile metadata: {len(tile_meta)} images")

    print("Running inference on test set...")
    results = get_predictions_with_attention(model, device)
    print(f"  Test set: {len(results)} images")

    # Generate WRONG overlays (skipping existing)
    print("\nGenerating overlays for misclassifications...")
    wrong = [r for r in results if not r["correct"]]
    wrong.sort(key=lambda x: -x["confidence"])
    generated = 0
    for rank, r in enumerate(wrong[:5]):
        img_id = r["image_id"]
        fname = f"heatmap_WRONG_{rank}_{r['true']}_as_{r['pred']}_{img_id}.png"
        out_path = loc_dir / fname
        if fname in existing:
            print(f"  [SKIP] {fname}")
            continue
        meta_list = tile_meta.get(img_id, [])
        success = generate_attention_overlay(
            img_id, r["attention"], meta_list,
            r["pred"], r["true"], r["confidence"], r["correct"], out_path
        )
        if success:
            generated += 1
            print(f"  [WRONG] {img_id}: {r['true']}->{r['pred']} conf={r['confidence']:.2f}")
    print(f"  Generated {generated} misclassification overlays")

    # Generate per-class galleries
    generate_class_gallery(results, tile_meta, EVAL_DIR)

    print(f"\n{'='*60}")
    print(f"Done! Check:")
    print(f"  {loc_dir}/heatmap_WRONG_*.png")
    print(f"  {EVAL_DIR}/localization_*.png")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
