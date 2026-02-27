#!/usr/bin/env python3

import argparse
import json
import math
import pickle
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn as nn
from PIL import Image
from scipy import ndimage
from tqdm import tqdm


CLASS_ORDER = ["LDA", "CCF", "LVF", "GLF", "BACKGROUND"]
CLASS_COLORS = {
    "LDA": [215, 25, 28],
    "CCF": [44, 123, 182],
    "LVF": [253, 174, 97],
    "GLF": [26, 150, 65],
    "BACKGROUND": [200, 200, 200],
}

MARS_RADIUS_M = 3389500.0
DEFAULT_MPP = 6.0


def parse_args():
    parser = argparse.ArgumentParser(description="HiRISE landform inference pipeline")
    parser.add_argument(
        "--image-dirs",
        default="Data/HiRISE/midlat_browse,arcadia_hirise/jpeg",
        help="Comma-separated image directories and/or image paths",
    )
    parser.add_argument("--metadata-json", default="Data/HiRISE/midlat_metadata.json")
    parser.add_argument(
        "--classifier-model",
        default="Data/HiRISE/pipeline_output/classifier/best_model.pt",
    )
    parser.add_argument(
        "--fusion-model",
        default="Data/HiRISE/pipeline_output/fusion/fusion_model.pt",
    )
    parser.add_argument(
        "--dem-path",
        default="Mars_HRSC_MOLA_BlendDEM_Global_200mp_v2.tif",
    )
    parser.add_argument(
        "--output-dir",
        default="Data/HiRISE/pipeline_output/predictions",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patch-size", type=int, default=224)
    parser.add_argument("--black-threshold", type=float, default=0.3)
    parser.add_argument("--save-maps", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def safe_torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def parse_image_inputs(image_dirs_arg):
    inputs = [x.strip() for x in image_dirs_arg.split(",") if x.strip()]
    images = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            for pattern in ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png", "*.PNG"):
                images.extend(sorted(p.glob(pattern)))
        elif p.is_file():
            images.append(p)
    dedup = sorted({x.resolve() for x in images})
    return [Path(x) for x in dedup]


def load_metadata(metadata_json):
    meta_path = Path(metadata_json)
    if not meta_path.exists():
        return {}

    with open(meta_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    lookup = {}

    def register(key, val):
        if key is None:
            return
        key_s = str(key).strip()
        if not key_s:
            return
        lookup[key_s] = val
        lookup[key_s.lower()] = val

    if isinstance(raw, list):
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            image_id = (
                entry.get("image_id")
                or entry.get("product_id")
                or entry.get("obsid")
                or entry.get("id")
                or entry.get("name")
            )
            register(image_id, entry)
    elif isinstance(raw, dict):
        if "images" in raw and isinstance(raw["images"], list):
            for entry in raw["images"]:
                if not isinstance(entry, dict):
                    continue
                image_id = (
                    entry.get("image_id")
                    or entry.get("product_id")
                    or entry.get("obsid")
                    or entry.get("id")
                    or entry.get("name")
                )
                register(image_id, entry)
        else:
            for k, v in raw.items():
                register(k, v)

    return lookup


def get_meta_record(metadata_lookup, image_id):
    if image_id in metadata_lookup:
        return metadata_lookup[image_id]
    if image_id.lower() in metadata_lookup:
        return metadata_lookup[image_id.lower()]
    if image_id.upper() in metadata_lookup:
        return metadata_lookup[image_id.upper()]
    return {}


def get_first_number(dct, keys, default=None):
    for key in keys:
        if key in dct and dct[key] is not None:
            try:
                return float(dct[key])
            except (TypeError, ValueError):
                continue
    return default


def tile_center_lat_lon(meta, tile_row, tile_col, patch_size, image_w, image_h):
    center_lat = get_first_number(
        meta,
        [
            "center_latitude",
            "center_lat",
            "lat",
            "latitude",
            "CenterLatitude",
            "centerLatitude",
        ],
        default=0.0,
    )
    center_lon = get_first_number(
        meta,
        [
            "center_longitude",
            "center_lon",
            "lon",
            "longitude",
            "CenterLongitude",
            "centerLongitude",
        ],
        default=0.0,
    )
    mpp = get_first_number(
        meta,
        [
            "m_per_pixel",
            "meters_per_pixel",
            "map_scale",
            "map_scale_m_per_px",
            "pixel_scale",
            "resolution",
        ],
        default=DEFAULT_MPP,
    )

    center_x = image_w / 2.0
    center_y = image_h / 2.0
    tile_center_x = (tile_col + 0.5) * patch_size
    tile_center_y = (tile_row + 0.5) * patch_size

    dx_px = tile_center_x - center_x
    dy_px = tile_center_y - center_y

    dx_m = dx_px * mpp
    dy_m = dy_px * mpp

    lat_rad = math.radians(center_lat)
    cos_lat = max(1e-6, abs(math.cos(lat_rad)))

    dlat_deg = -(dy_m / MARS_RADIUS_M) * (180.0 / math.pi)
    dlon_deg = (dx_m / (MARS_RADIUS_M * cos_lat)) * (180.0 / math.pi)

    lat = center_lat + dlat_deg
    lon = center_lon + dlon_deg
    while lon > 180.0:
        lon -= 360.0
    while lon < -180.0:
        lon += 360.0
    return lat, lon


def is_black_tile(patch_rgb, black_threshold):
    gray = np.mean(patch_rgb, axis=2)
    black_frac = np.mean(gray < 15.0)
    return black_frac >= black_threshold


def preprocess_patch(patch_rgb, patch_size):
    img = Image.fromarray(patch_rgb.astype(np.uint8), mode="RGB")
    if img.size != (patch_size, patch_size):
        img = img.resize((patch_size, patch_size), resample=Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    arr = (arr - mean) / std
    return torch.from_numpy(arr)


def tile_image(image_arr, patch_size, black_threshold):
    h, w = image_arr.shape[:2]
    n_rows = h // patch_size
    n_cols = w // patch_size

    tiles = []
    positions = []
    skipped = []

    for row in range(n_rows):
        y0 = row * patch_size
        y1 = y0 + patch_size
        for col in range(n_cols):
            x0 = col * patch_size
            x1 = x0 + patch_size
            patch = image_arr[y0:y1, x0:x1]
            if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                skipped.append((row, col))
                continue
            if is_black_tile(patch, black_threshold):
                skipped.append((row, col))
                continue
            tiles.append(patch)
            positions.append((row, col))

    return tiles, positions, skipped, n_rows, n_cols


def load_dino(device):
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    eval_fn = getattr(model, "eval", None)
    to_fn = getattr(model, "to", None)
    if not callable(eval_fn) or not callable(to_fn):
        raise RuntimeError("Unexpected DINOv2 model object returned by torch.hub.load")
    eval_fn()
    to_fn(device)
    return model


def infer_dino_embeddings(model, patches, batch_size, patch_size, device):
    if not patches:
        return np.zeros((0, 384), dtype=np.float32)

    outs = []
    with torch.no_grad():
        for i in range(0, len(patches), batch_size):
            batch_patches = patches[i : i + batch_size]
            batch = torch.stack([preprocess_patch(p, patch_size) for p in batch_patches], dim=0)
            batch = batch.to(device)
            emb = model(batch)
            if isinstance(emb, (tuple, list)):
                emb = emb[0]
            outs.append(emb.detach().cpu().numpy())

    return np.concatenate(outs, axis=0).astype(np.float32)


def get_dem_pixel_size_m(dem_ds, lat_deg):
    xres = abs(dem_ds.transform.a)
    yres = abs(dem_ds.transform.e)
    if dem_ds.crs is not None and dem_ds.crs.is_geographic:
        m_per_deg_lat = 2.0 * math.pi * MARS_RADIUS_M / 360.0
        m_per_deg_lon = m_per_deg_lat * max(1e-6, abs(math.cos(math.radians(lat_deg))))
        return math.sqrt(max(1e-6, xres * m_per_deg_lon) * max(1e-6, yres * m_per_deg_lat))
    return max(1e-6, math.sqrt(xres * yres))


def resolve_lon_for_dem(dem_ds, lon_deg):
    candidates = [lon_deg, lon_deg % 360.0, ((lon_deg + 180.0) % 360.0) - 180.0]
    for cand in candidates:
        if dem_ds.bounds.left <= cand <= dem_ds.bounds.right:
            return cand
    return candidates[0]


def read_dem_window(dem_ds, lat_deg, lon_deg, radius_km):
    lon_use = resolve_lon_for_dem(dem_ds, lon_deg)
    row, col = dem_ds.index(lon_use, lat_deg)

    px_m = get_dem_pixel_size_m(dem_ds, lat_deg)
    rad_px = max(1, int(round((radius_km * 1000.0) / px_m)))

    row0 = max(0, row - rad_px)
    row1 = min(dem_ds.height, row + rad_px + 1)
    col0 = max(0, col - rad_px)
    col1 = min(dem_ds.width, col + rad_px + 1)

    if row1 <= row0 or col1 <= col0:
        return None, px_m

    window = ((row0, row1), (col0, col1))
    data = dem_ds.read(1, window=window)

    if dem_ds.nodata is not None:
        data = np.where(data == dem_ds.nodata, np.nan, data)
    data = data.astype(np.float32)
    return data, px_m


def slope_degrees(elev_window, pixel_size_m):
    if elev_window is None or elev_window.size == 0:
        return None
    arr = elev_window.copy()
    valid = np.isfinite(arr)
    if not np.any(valid):
        return None
    fill_val = float(np.nanmean(arr[valid]))
    arr[~valid] = fill_val
    gx = ndimage.sobel(arr, axis=1, mode="nearest") / (8.0 * pixel_size_m)
    gy = ndimage.sobel(arr, axis=0, mode="nearest") / (8.0 * pixel_size_m)
    slope = np.degrees(np.arctan(np.sqrt(gx * gx + gy * gy)))
    slope[~valid] = np.nan
    return slope


def nan_stats(arr):
    if arr is None or arr.size == 0:
        return [np.nan, np.nan, np.nan, np.nan]
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return [np.nan, np.nan, np.nan, np.nan]
    return [
        float(np.mean(valid)),
        float(np.std(valid)),
        float(np.min(valid)),
        float(np.max(valid)),
    ]


def extract_mola_features_23(dem_ds, lat_deg, lon_deg):
    radii = [1.0, 2.0, 5.0, 10.0]
    features = []

    win_1km, px_m_1km = read_dem_window(dem_ds, lat_deg, lon_deg, radii[0])
    center_elev = np.nan
    mean_1km = np.nan
    if win_1km is not None and win_1km.size > 0:
        c_row = win_1km.shape[0] // 2
        c_col = win_1km.shape[1] // 2
        center_elev = float(win_1km[c_row, c_col])
        mean_1km = float(np.nanmean(win_1km)) if np.any(np.isfinite(win_1km)) else np.nan

    relief_1km = center_elev - mean_1km if np.isfinite(center_elev) and np.isfinite(mean_1km) else np.nan
    features.extend([center_elev, abs(float(lat_deg)), relief_1km])

    for radius_km in radii:
        elev_win, px_m = read_dem_window(dem_ds, lat_deg, lon_deg, radius_km)
        elev_mean, elev_std, elev_min, elev_max = nan_stats(elev_win)
        slope_win = slope_degrees(elev_win, px_m)
        slope_mean = nan_stats(slope_win)[0]
        features.extend([elev_mean, elev_std, elev_min, elev_max, slope_mean])

    feats = np.asarray(features, dtype=np.float32)
    if feats.shape[0] != 23:
        out = np.zeros((23,), dtype=np.float32)
        n = min(23, feats.shape[0])
        out[:n] = feats[:n]
        feats = out
    feats[~np.isfinite(feats)] = 0.0
    return feats


class LinearHead(nn.Module):
    def __init__(self, state_dict):
        super().__init__()
        linear_keys = []
        for key, tensor in state_dict.items():
            if key.endswith(".weight") and hasattr(tensor, "ndim") and tensor.ndim == 2:
                linear_keys.append(key)
        linear_keys.sort()
        if not linear_keys:
            raise RuntimeError("No linear layers found in classifier model_state_dict")

        self.layer_order = linear_keys
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()

        for w_key in self.layer_order:
            b_key = w_key[:-7] + ".bias"
            w = state_dict[w_key].detach().float()
            b = state_dict.get(b_key, None)
            if b is None:
                b = torch.zeros(w.shape[0], dtype=torch.float32)
            else:
                b = b.detach().float()
            self.weights.append(nn.Parameter(w, requires_grad=False))
            self.biases.append(nn.Parameter(b, requires_grad=False))

    def forward(self, x):
        h = x
        n_layers = len(self.weights)
        for i in range(n_layers):
            w = self.weights[i]
            b = self.biases[i]
            h = torch.matmul(h, w.t()) + b
            if i < n_layers - 1:
                h = torch.relu(h)
        return h


def align_logits(logits, src_classes, dst_classes):
    if src_classes is None:
        return logits
    src_idx = {name: i for i, name in enumerate(src_classes)}
    out = np.zeros((logits.shape[0], len(dst_classes)), dtype=np.float32)
    for j, cls in enumerate(dst_classes):
        if cls in src_idx:
            out[:, j] = logits[:, src_idx[cls]]
    return out


def softmax_np(logits):
    z = logits - np.max(logits, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


def dem_logits_from_classifier(dem_classifier, x_scaled):
    if hasattr(dem_classifier, "decision_function"):
        dec = dem_classifier.decision_function(x_scaled)
        dec = np.asarray(dec, dtype=np.float32)
        if dec.ndim == 1:
            dec = np.stack([-dec, dec], axis=1)
        return dec
    if hasattr(dem_classifier, "predict_log_proba"):
        return np.asarray(dem_classifier.predict_log_proba(x_scaled), dtype=np.float32)
    if hasattr(dem_classifier, "predict_proba"):
        p = np.asarray(dem_classifier.predict_proba(x_scaled), dtype=np.float32)
        return np.log(np.clip(p, 1e-12, 1.0))
    raise RuntimeError("Unsupported DEM classifier object")


def save_label_map_png(output_path, pred_grid, patch_size):
    n_rows, n_cols = pred_grid.shape
    h = n_rows * patch_size
    w = n_cols * patch_size
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    color_lut = np.array([CLASS_COLORS[c] for c in CLASS_ORDER], dtype=np.uint8)

    for r in range(n_rows):
        y0 = r * patch_size
        y1 = y0 + patch_size
        for c in range(n_cols):
            x0 = c * patch_size
            x1 = x0 + patch_size
            canvas[y0:y1, x0:x1, :] = color_lut[pred_grid[r, c]]

    Image.fromarray(canvas, mode="RGB").save(output_path)


def classify_cnn(head, embeddings, mola_features, use_mola, device):
    if embeddings.shape[0] == 0:
        return np.zeros((0, len(CLASS_ORDER)), dtype=np.float32)
    x = embeddings
    if use_mola and mola_features is not None and mola_features.shape[0] == embeddings.shape[0]:
        x = np.concatenate([embeddings, mola_features], axis=1)
    with torch.no_grad():
        x_t = torch.from_numpy(x.astype(np.float32)).to(device)
        logits = head(x_t).detach().cpu().numpy().astype(np.float32)
    return logits


def summarize_image(image_id, probs, pred_idx):
    n = probs.shape[0]
    if n == 0:
        return {
            "image_id": image_id,
            "dominant_class": "BACKGROUND",
            "n_tiles": 0,
            "pct_LDA": 0.0,
            "pct_CCF": 0.0,
            "pct_LVF": 0.0,
            "pct_GLF": 0.0,
            "pct_BACKGROUND": 0.0,
            "mean_confidence": 0.0,
        }

    counts = Counter(pred_idx.tolist())
    dominant = int(np.argmax(np.bincount(pred_idx, minlength=len(CLASS_ORDER))))
    mean_conf = float(np.mean(np.max(probs, axis=1)))

    return {
        "image_id": image_id,
        "dominant_class": CLASS_ORDER[dominant],
        "n_tiles": int(n),
        "pct_LDA": 100.0 * counts.get(0, 0) / n,
        "pct_CCF": 100.0 * counts.get(1, 0) / n,
        "pct_LVF": 100.0 * counts.get(2, 0) / n,
        "pct_GLF": 100.0 * counts.get(3, 0) / n,
        "pct_BACKGROUND": 100.0 * counts.get(4, 0) / n,
        "mean_confidence": mean_conf,
    }


def ensure_2d_features(features, expected_dim):
    arr = np.asarray(features, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[1] != expected_dim:
        out = np.zeros((arr.shape[0], expected_dim), dtype=np.float32)
        n = min(expected_dim, arr.shape[1])
        out[:, :n] = arr[:, :n]
        arr = out
    arr[~np.isfinite(arr)] = 0.0
    return arr


def main():
    args = parse_args()
    t0 = time.time()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = parse_image_inputs(args.image_dirs)
    if args.limit is not None:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        print("No input images found.")
        return

    metadata_lookup = load_metadata(args.metadata_json)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading DINOv2 model...")
    dino = load_dino(device)

    print("Loading classifier model...")
    clf_ckpt = safe_torch_load(args.classifier_model, map_location="cpu")
    clf_state = clf_ckpt.get("model_state_dict", clf_ckpt)
    clf_classes = clf_ckpt.get("class_names", None)
    use_mola_in_cnn = bool(clf_ckpt.get("use_mola", False))
    cnn_head = LinearHead(clf_state).to(device)
    cnn_head.eval()

    print("Loading fusion model...")
    fusion_obj = safe_torch_load(args.fusion_model, map_location="cpu")
    if not isinstance(fusion_obj, dict):
        if isinstance(fusion_obj, (bytes, bytearray)):
            fusion_obj = pickle.loads(fusion_obj)
        else:
            raise RuntimeError("Fusion model file must contain a dict")

    dem_classifier = fusion_obj["dem_classifier"]
    fusion_weights = np.asarray(fusion_obj["fusion_weights"], dtype=np.float32)
    fusion_biases = np.asarray(fusion_obj["fusion_biases"], dtype=np.float32)
    mola_scaler = fusion_obj["mola_scaler"]
    fusion_classes = fusion_obj.get("class_names", CLASS_ORDER)

    with rasterio.open(args.dem_path) as dem_ds:
        expected_mola_dim = int(getattr(mola_scaler, "mean_", np.zeros((23,))).shape[0])
        summary_rows = []

        for image_path in tqdm(image_paths, desc="Predicting images"):
            image_id = image_path.stem
            meta = get_meta_record(metadata_lookup, image_id)

            with Image.open(image_path) as im:
                im = im.convert("RGB")
                image_arr = np.asarray(im)

            tiles, positions, skipped, n_rows, n_cols = tile_image(
                image_arr,
                patch_size=args.patch_size,
                black_threshold=args.black_threshold,
            )

            pred_grid = np.full((n_rows, n_cols), CLASS_ORDER.index("BACKGROUND"), dtype=np.int32)

            if not tiles:
                empty_df = pd.DataFrame(
                    columns=[
                        "image_id",
                        "tile_row",
                        "tile_col",
                        "pred_class",
                        "confidence",
                        "prob_LDA",
                        "prob_CCF",
                        "prob_LVF",
                        "prob_GLF",
                        "prob_BACKGROUND",
                    ]
                )
                empty_df.to_csv(output_dir / f"{image_id}_tile_predictions.csv", index=False)
                if args.save_maps:
                    save_label_map_png(output_dir / f"{image_id}_label_map.png", pred_grid, args.patch_size)
                summary_rows.append(
                    {
                        "image_id": image_id,
                        "dominant_class": "BACKGROUND",
                        "n_tiles": 0,
                        "pct_LDA": 0.0,
                        "pct_CCF": 0.0,
                        "pct_LVF": 0.0,
                        "pct_GLF": 0.0,
                        "pct_BACKGROUND": 0.0,
                        "mean_confidence": 0.0,
                    }
                )
                continue

            dino_embeddings = infer_dino_embeddings(
                dino,
                tiles,
                batch_size=args.batch_size,
                patch_size=args.patch_size,
                device=device,
            )

            mola_feats = []
            for row, col in positions:
                lat, lon = tile_center_lat_lon(
                    meta,
                    tile_row=row,
                    tile_col=col,
                    patch_size=args.patch_size,
                    image_w=image_arr.shape[1],
                    image_h=image_arr.shape[0],
                )
                mola_feats.append(extract_mola_features_23(dem_ds, lat, lon))
            mola_feats = ensure_2d_features(mola_feats, expected_mola_dim)

            cnn_logits_raw = classify_cnn(
                head=cnn_head,
                embeddings=dino_embeddings,
                mola_features=mola_feats,
                use_mola=use_mola_in_cnn,
                device=device,
            )
            cnn_logits = align_logits(cnn_logits_raw, clf_classes, CLASS_ORDER)

            scaled_mola = mola_scaler.transform(mola_feats)
            dem_logits_raw = dem_logits_from_classifier(dem_classifier, scaled_mola)

            dem_src_classes = getattr(dem_classifier, "classes_", None)
            dem_src_names = None
            if dem_src_classes is not None:
                dem_src_names = [str(x) for x in dem_src_classes]
                if set(dem_src_names) != set(CLASS_ORDER):
                    dem_src_names = fusion_classes
            else:
                dem_src_names = fusion_classes

            dem_logits = align_logits(dem_logits_raw, dem_src_names, CLASS_ORDER)

            fw = align_logits(fusion_weights[None, :], fusion_classes, CLASS_ORDER)[0]
            fb = align_logits(fusion_biases[None, :], fusion_classes, CLASS_ORDER)[0]

            fused_logits = cnn_logits + dem_logits * fw[None, :] + fb[None, :]
            probs = softmax_np(fused_logits)
            pred_idx = np.argmax(probs, axis=1)
            confidence = np.max(probs, axis=1)

            rows = []
            for i, (row, col) in enumerate(positions):
                cls_i = int(pred_idx[i])
                pred_grid[row, col] = cls_i
                rows.append(
                    {
                        "image_id": image_id,
                        "tile_row": int(row),
                        "tile_col": int(col),
                        "pred_class": CLASS_ORDER[cls_i],
                        "confidence": float(confidence[i]),
                        "prob_LDA": float(probs[i, 0]),
                        "prob_CCF": float(probs[i, 1]),
                        "prob_LVF": float(probs[i, 2]),
                        "prob_GLF": float(probs[i, 3]),
                        "prob_BACKGROUND": float(probs[i, 4]),
                    }
                )

            df = pd.DataFrame(rows)
            df.to_csv(output_dir / f"{image_id}_tile_predictions.csv", index=False)

            if args.save_maps:
                save_label_map_png(output_dir / f"{image_id}_label_map.png", pred_grid, args.patch_size)

            summary_rows.append(summarize_image(image_id, probs, pred_idx))

        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(output_dir / "aggregated_predictions.csv", index=False)

    dt = time.time() - t0
    print(f"Done. Processed {len(image_paths)} images in {dt:.1f}s")
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
