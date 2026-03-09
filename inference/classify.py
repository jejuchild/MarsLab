#!/usr/bin/env python3
"""
MarsLandformNet — Standalone Inference Script

HiRISE 이미지를 입력받아 224×224 타일로 분할한 뒤,
각 타일을 DINOv2-LoRA + FiLM 모델로 분류합니다.

Classes:
  0: LDA  (Lobate Debris Apron)
  1: LVF  (Lineated Valley Fill)
  2: CCF  (Concentric Crater Fill)
  3: OTHER

Usage:
  # Basic (MOLA 없이 — visual features only)
  python classify.py image.jpg -c model.pt

  # With MOLA DEM (full accuracy)
  python classify.py image.jpg -c model.pt --dem MOLA_DEM.tif --lat 43.68 --lon 13.66

  # GPU + JSON output
  python classify.py image.jpg -c model.pt --dem MOLA_DEM.tif --lat 43.68 --lon 13.66 -d cuda -o result.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from transformers import Dinov2Model
from peft import LoraConfig, get_peft_model, TaskType


# =============================================================================
# Constants
# =============================================================================

CLASSES = ["LDA", "LVF", "CCF", "OTHER"]
NUM_CLASSES = 4
TILE_SIZE = 224
MIN_CONTENT_FRACTION = 0.3

CLASS_THRESHOLDS = {
    "LDA":   0.65,
    "LVF":   0.50,
    "CCF":   0.45,
    "OTHER": 0.55,
}

TILE_TRANSFORM = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

DEM_RESOLUTION_M = 200.0  # MOLA 200m/pixel


# =============================================================================
# Model Definitions
# =============================================================================

class DinoV2LoRA(nn.Module):
    """DINOv2-Base (ViT-B/14) backbone with LoRA adapters."""

    def __init__(
        self,
        model_name: str = "facebook/dinov2-base",
        embed_dim: int = 768,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        lora_targets: list[str] | None = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.backbone = Dinov2Model.from_pretrained(model_name)
        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=lora_targets or ["query", "key", "value"],
            task_type=TaskType.FEATURE_EXTRACTION,
            bias="none",
        )
        self.backbone = get_peft_model(self.backbone, lora_cfg)
        for param in self.backbone.parameters():
            param.requires_grad = False
        for name, param in self.backbone.named_parameters():
            if "lora_" in name:
                param.requires_grad = True

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(pixel_values=pixel_values)
        return outputs.last_hidden_state[:, 0]


class FiLMLayer(nn.Module):
    def __init__(self, mola_dim: int, visual_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.mola_encoder = nn.Sequential(
            nn.BatchNorm1d(mola_dim),
            nn.Linear(mola_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.gamma_proj = nn.Linear(hidden_dim, visual_dim)
        self.beta_proj = nn.Linear(hidden_dim, visual_dim)
        nn.init.ones_(self.gamma_proj.bias)
        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)

    def forward(self, visual: torch.Tensor, mola: torch.Tensor) -> torch.Tensor:
        h = self.mola_encoder(mola)
        return self.gamma_proj(h) * visual + self.beta_proj(h)


class FiLMClassifier(nn.Module):
    def __init__(
        self,
        visual_dim: int = 768,
        mola_dim: int = 25,
        film_hidden: int = 64,
        head_hidden: int = 128,
        num_classes: int = 4,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.film = FiLMLayer(mola_dim, visual_dim, film_hidden)
        self.classifier = nn.Sequential(
            nn.Linear(visual_dim, head_hidden),
            nn.BatchNorm1d(head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, num_classes),
        )

    def forward(self, embeddings: torch.Tensor, mola: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.film(embeddings, mola))


# =============================================================================
# MOLA Feature Extraction (23-dim base + 2 relative = 25-dim)
# =============================================================================

def _latlon_to_pixel(ds, lat: float, lon: float) -> tuple[int, int]:
    row, col = ds.index(lon, lat)
    row = max(0, min(row, ds.height - 1))
    col = max(0, min(col, ds.width - 1))
    return int(row), int(col)


def _extract_window(ds, lat: float, lon: float, radius_km: float) -> np.ndarray:
    radius_px = max(1, int(radius_km * 1000 / DEM_RESOLUTION_M))
    row, col = _latlon_to_pixel(ds, lat, lon)
    r0 = max(0, row - radius_px)
    r1 = min(ds.height, row + radius_px + 1)
    c0 = max(0, col - radius_px)
    c1 = min(ds.width, col + radius_px + 1)
    window = ds.read(1, window=((r0, r1), (c0, c1)))
    nodata = ds.nodata
    if nodata is not None:
        window = window.astype(np.float64)
        window[window == nodata] = np.nan
    return window


def _compute_slope(elev: np.ndarray) -> np.ndarray:
    if elev.shape[0] < 3 or elev.shape[1] < 3:
        return np.zeros_like(elev)
    dy, dx = np.gradient(elev, DEM_RESOLUTION_M)
    return np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))


def _compute_curvature(elev: np.ndarray) -> np.ndarray:
    if elev.shape[0] < 3 or elev.shape[1] < 3:
        return np.zeros_like(elev)
    dy, dx = np.gradient(elev, DEM_RESOLUTION_M)
    dyy, _ = np.gradient(dy, DEM_RESOLUTION_M)
    _, dxx = np.gradient(dx, DEM_RESOLUTION_M)
    return dyy + dxx


def _compute_tpi(elev: np.ndarray, radius_px: int = 5) -> np.ndarray:
    from scipy.ndimage import uniform_filter
    return elev - uniform_filter(elev, size=2 * radius_px + 1, mode="nearest")


def _compute_tri(elev: np.ndarray) -> np.ndarray:
    if elev.shape[0] < 3 or elev.shape[1] < 3:
        return np.zeros_like(elev)
    padded = np.pad(elev, 1, mode="edge")
    h, w = elev.shape
    total = np.zeros_like(elev, dtype=np.float64)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            total += np.abs(padded[1 + dr : h + 1 + dr, 1 + dc : w + 1 + dc] - elev)
    return total / 9.0


def _compute_roughness(elev: np.ndarray) -> np.ndarray:
    from scipy.ndimage import maximum_filter, minimum_filter
    return maximum_filter(elev, size=3) - minimum_filter(elev, size=3)


def _compute_lobateness(slope: np.ndarray) -> float:
    mean_s = float(np.nanmean(slope))
    return float(np.nanmax(slope) / mean_s) if mean_s > 0.1 else 0.0


def _extract_7_features(ds, lat: float, lon: float, radius_km: float) -> list[float]:
    """7 geomorphometric features at one spatial scale."""
    window = _extract_window(ds, lat, lon, radius_km)
    if window.size == 0 or np.all(np.isnan(window)):
        return [0.0] * 7
    valid = window[~np.isnan(window)]
    med = float(np.median(valid)) if len(valid) > 0 else 0.0
    filled = np.where(np.isnan(window), med, window)
    slope = _compute_slope(filled)
    curv = _compute_curvature(filled)
    tpi = _compute_tpi(filled, radius_px=max(1, filled.shape[0] // 4))
    tri = _compute_tri(filled)
    rough = _compute_roughness(filled)
    return [
        float(np.nanmean(slope)),
        float(np.nanstd(slope)),
        float(np.nanmean(curv)),
        float(np.nanmean(tpi)),
        float(np.nanmean(tri)),
        float(np.nanmean(rough)),
        _compute_lobateness(slope),
    ]


def extract_mola_features(ds, lat: float, lon: float) -> np.ndarray:
    """
    Extract 23-dim MOLA geomorphometric features at (lat, lon).
      7 features × 3 scales (1, 5, 20 km) + elevation_mean + abs_latitude = 23
    """
    feats: list[float] = []
    for scale in [1.0, 5.0, 20.0]:
        try:
            feats.extend(_extract_7_features(ds, lat, lon, scale))
        except Exception:
            feats.extend([0.0] * 7)
    # Global: elevation_mean (1km window), abs_latitude
    try:
        w = _extract_window(ds, lat, lon, 1.0)
        feats.append(float(np.nanmean(w)) if w.size > 0 else 0.0)
    except Exception:
        feats.append(0.0)
    feats.append(abs(lat))
    return np.array(feats, dtype=np.float32)


def extract_mola_features_batch(
    ds,
    coords: list[tuple[float, float]],
) -> np.ndarray:
    """
    Extract 25-dim MOLA features for multiple (lat, lon) tiles.
      23 base features + 2 relative (elevation_rel, slope_rel) = 25
    Single DEM read covering all tiles.
    """
    n = len(coords)
    if n == 0:
        return np.zeros((0, 25), dtype=np.float32)

    scales_km = [1.0, 5.0, 20.0]
    max_radius_px = int(max(scales_km) * 1000 / DEM_RESOLUTION_M) + 5

    pixel_coords = [_latlon_to_pixel(ds, lat, lon) for lat, lon in coords]
    rows = [r for r, _ in pixel_coords]
    cols = [c for _, c in pixel_coords]

    origin_row = max(0, min(rows) - max_radius_px)
    origin_col = max(0, min(cols) - max_radius_px)
    end_row = min(ds.height, max(rows) + max_radius_px + 1)
    end_col = min(ds.width, max(cols) + max_radius_px + 1)

    try:
        big = ds.read(1, window=((origin_row, end_row), (origin_col, end_col)))
        nodata = ds.nodata
        if nodata is not None:
            big = big.astype(np.float64)
            big[big == nodata] = np.nan
    except Exception:
        return np.zeros((n, 25), dtype=np.float32)

    bh, bw = big.shape
    base_arr = np.zeros((n, 23), dtype=np.float32)

    for i, (prow, pcol) in enumerate(pixel_coords):
        lr = prow - origin_row
        lc = pcol - origin_col
        feats: list[float] = []

        for scale in scales_km:
            rpx = max(1, int(scale * 1000 / DEM_RESOLUTION_M))
            r0, r1 = max(0, lr - rpx), min(bh, lr + rpx + 1)
            c0, c1 = max(0, lc - rpx), min(bw, lc + rpx + 1)
            sub = big[r0:r1, c0:c1]
            if sub.size == 0 or np.all(np.isnan(sub)):
                feats.extend([0.0] * 7)
                continue
            valid = sub[~np.isnan(sub)]
            med = float(np.median(valid)) if len(valid) > 0 else 0.0
            filled = np.where(np.isnan(sub), med, sub)
            slope = _compute_slope(filled)
            curv = _compute_curvature(filled)
            tpi = _compute_tpi(filled, max(1, filled.shape[0] // 4))
            tri = _compute_tri(filled)
            rough = _compute_roughness(filled)
            feats.extend([
                float(np.nanmean(slope)), float(np.nanstd(slope)),
                float(np.nanmean(curv)), float(np.nanmean(tpi)),
                float(np.nanmean(tri)), float(np.nanmean(rough)),
                _compute_lobateness(slope),
            ])

        # elevation_mean (1km), abs_latitude
        r1km = max(1, int(1000 / DEM_RESOLUTION_M))
        r0, r1 = max(0, lr - r1km), min(bh, lr + r1km + 1)
        c0, c1 = max(0, lc - r1km), min(bw, lc + r1km + 1)
        sub1 = big[r0:r1, c0:c1]
        feats.append(float(np.nanmean(sub1)) if sub1.size > 0 else 0.0)
        feats.append(abs(coords[i][0]))
        base_arr[i] = feats

    # +2 relative features: per-tile deviation from image mean
    mean_elev = float(np.mean(base_arr[:, 21]))   # elevation_mean index
    mean_slope = float(np.mean(base_arr[:, 0]))    # slope_mean index
    rel_elev = (base_arr[:, 21] - mean_elev).reshape(-1, 1)
    rel_slope = (base_arr[:, 0] - mean_slope).reshape(-1, 1)
    return np.concatenate([base_arr, rel_elev, rel_slope], axis=1).astype(np.float32)


# =============================================================================
# Result / Tiling
# =============================================================================

@dataclass
class TileResult:
    grid_x: int
    grid_y: int
    predicted_class: str
    confidence: float
    probabilities: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "grid_x": self.grid_x,
            "grid_y": self.grid_y,
            "predicted_class": self.predicted_class,
            "confidence": round(self.confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
        }


def tile_image(
    image: Image.Image,
    tile_size: int = TILE_SIZE,
    min_content: float = MIN_CONTENT_FRACTION,
) -> list[tuple[int, int, Image.Image]]:
    rgb = image.convert("RGB")
    w, h = rgb.size
    tiles: list[tuple[int, int, Image.Image]] = []
    arr = np.array(rgb)
    for y0 in range(0, h - tile_size + 1, tile_size):
        for x0 in range(0, w - tile_size + 1, tile_size):
            t = arr[y0 : y0 + tile_size, x0 : x0 + tile_size]
            if float(np.mean(t > 10)) < min_content:
                continue
            tiles.append((x0 // tile_size, y0 // tile_size, Image.fromarray(t)))
    if not tiles:
        tiles.append((0, 0, rgb.resize((tile_size, tile_size), Image.Resampling.BICUBIC)))
    return tiles


# =============================================================================
# Model Loading
# =============================================================================

def load_model(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[DinoV2LoRA, FiLMClassifier]:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("cfg", ckpt.get("config", {}))
    sd = ckpt.get("model_state_dict", {})

    backbone = DinoV2LoRA(
        model_name=cfg.get("model_name", "facebook/dinov2-base"),
        embed_dim=cfg.get("hidden_dim", 768),
        lora_r=cfg.get("lora_r", 16),
        lora_alpha=cfg.get("lora_alpha", 32),
        lora_dropout=cfg.get("lora_dropout", 0.1),
        lora_targets=cfg.get("lora_targets", ["query", "key", "value"]),
    )
    bb_sd = {k: v for k, v in sd.items() if k.startswith("backbone.")}
    if bb_sd:
        backbone.load_state_dict(bb_sd, strict=False)
    backbone.eval().to(device)

    classifier = FiLMClassifier(
        visual_dim=cfg.get("hidden_dim", 768),
        mola_dim=cfg.get("mola_dim", 25),
        film_hidden=64,
        head_hidden=cfg.get("head_hidden", 128),
        num_classes=cfg.get("num_classes", NUM_CLASSES),
        dropout=cfg.get("dropout", 0.4),
    )
    head_sd = {k: v for k, v in sd.items() if k.startswith("film.") or k.startswith("classifier.")}
    if head_sd:
        classifier.load_state_dict(head_sd, strict=False)
    classifier.eval().to(device)

    epoch = ckpt.get("epoch", "?")
    val_f1 = ckpt.get("val_f1", ckpt.get("test_f1", "?"))
    classes = cfg.get("class_names", CLASSES[:cfg.get("num_classes", NUM_CLASSES)])
    print(f"[Model] {Path(checkpoint_path).name}  epoch={epoch}  val_f1={val_f1}  classes={classes}  device={device}")

    return backbone, classifier


# =============================================================================
# Inference
# =============================================================================

@torch.no_grad()
def classify_image(
    image_path: str | Path,
    backbone: DinoV2LoRA,
    classifier: FiLMClassifier,
    device: torch.device,
    batch_size: int = 32,
    mola_ds=None,
    lat: float | None = None,
    lon: float | None = None,
) -> list[TileResult]:
    """
    Full pipeline:
      1. Tile image into 224×224 patches
      2. Extract DINOv2 embeddings (batched)
      3. Extract MOLA features (if DEM + lat/lon provided)
      4. Run FiLM classifier
      5. Apply per-class confidence thresholds
    """
    image = Image.open(image_path).convert("RGB")
    tiles = tile_image(image)
    n_tiles = len(tiles)
    if n_tiles == 0:
        return []

    img_w, img_h = image.size
    print(f"[Tile]  {img_w}×{img_h} → {n_tiles} tiles ({TILE_SIZE}×{TILE_SIZE})")

    # ── DINOv2 embeddings ──
    tensors = [TILE_TRANSFORM(t.convert("RGB")) for _, _, t in tiles]
    all_emb: list[np.ndarray] = []
    it = range(0, n_tiles, batch_size)
    if tqdm:
        it = tqdm(it, desc="Extracting embeddings", unit="batch")
    for i in it:
        batch = torch.stack(tensors[i : i + batch_size]).to(device)
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            emb = backbone(batch)
        all_emb.append(emb.cpu().numpy())
    embeddings = np.concatenate(all_emb, axis=0)

    # ── MOLA features ──
    mola_dim = classifier.film.mola_encoder[1].in_features
    if mola_ds is not None and lat is not None and lon is not None:
        print(f"[MOLA] Extracting terrain features from DEM at ({lat:.2f}°N, {lon:.2f}°E)...")
        tile_coords = _compute_tile_latlon(tiles, img_w, img_h, lat, lon)
        mola_features = extract_mola_features_batch(mola_ds, tile_coords)
        print(f"[MOLA] {mola_features.shape[1]}-dim features for {n_tiles} tiles ✓")
    else:
        if mola_ds is None and (lat is not None or lon is not None):
            print("[Warn] --lat/--lon given but --dem not provided. MOLA features zeroed.")
        mola_features = np.zeros((n_tiles, mola_dim), dtype=np.float32)

    # ── FiLM classifier ──
    emb_t = torch.from_numpy(embeddings).float().to(device)
    mola_t = torch.from_numpy(mola_features).float().to(device)
    with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
        logits = classifier(emb_t, mola_t)
    probs = torch.softmax(logits, dim=1).cpu().numpy()
    preds = np.argmax(probs, axis=1)

    num_classes = probs.shape[1]
    class_names = CLASSES[:num_classes]

    # ── Build results ──
    results: list[TileResult] = []
    for idx, (gx, gy, _) in enumerate(tiles):
        pred_idx = int(preds[idx])
        raw_class = class_names[pred_idx]
        conf = float(probs[idx, pred_idx])
        threshold = CLASS_THRESHOLDS.get(raw_class, 0.55)
        predicted = raw_class if conf >= threshold else "Uncertain"
        results.append(TileResult(
            grid_x=gx, grid_y=gy,
            predicted_class=predicted, confidence=conf,
            probabilities={class_names[c]: float(probs[idx, c]) for c in range(num_classes)},
        ))
    return results


def _compute_tile_latlon(
    tiles: list[tuple[int, int, Image.Image]],
    img_w: int, img_h: int,
    center_lat: float, center_lon: float,
) -> list[tuple[float, float]]:
    """Approximate per-tile lat/lon from grid position + image center."""
    # HiRISE browse images: ~5-6 km/pixel at ~25 cm/pixel full res, 10x downsampled
    # Rough estimate: image spans ~(img_w * 5) meters
    mars_r = 3389500.0  # meters
    deg_per_m = 180.0 / (math.pi * mars_r)
    # Browse pixel ≈ 50m (very rough — varies by image)
    m_per_px = 50.0

    cx_px = img_w / 2.0
    cy_px = img_h / 2.0

    coords = []
    for gx, gy, _ in tiles:
        tx = (gx + 0.5) * TILE_SIZE
        ty = (gy + 0.5) * TILE_SIZE
        dx_m = (tx - cx_px) * m_per_px
        dy_m = -(ty - cy_px) * m_per_px  # y-axis flipped
        dlat = dy_m * deg_per_m
        dlon = dx_m * deg_per_m / max(math.cos(math.radians(center_lat)), 0.01)
        coords.append((center_lat + dlat, center_lon + dlon))
    return coords


# =============================================================================
# Summary
# =============================================================================

def summarize(results: list[TileResult]) -> dict:
    n = len(results)
    if n == 0:
        return {"total_tiles": 0, "classes": {}}
    counts: dict[str, list[float]] = {}
    for r in results:
        counts.setdefault(r.predicted_class, []).append(r.confidence)
    summary = {}
    for cls in sorted(counts):
        c = counts[cls]
        summary[cls] = {
            "count": len(c),
            "percentage": round(100 * len(c) / n, 1),
            "mean_confidence": round(float(np.mean(c)), 4),
        }
    real = {k: v for k, v in summary.items() if k != "Uncertain"}
    dominant = max(real, key=lambda k: real[k]["count"]) if real else "Uncertain"
    return {
        "total_tiles": n,
        "dominant_class": dominant,
        "dominant_confidence": summary.get(dominant, {}).get("mean_confidence", 0),
        "classes": summary,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="MarsLandformNet — HiRISE landform classification (LDA / LVF / CCF)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visual-only (no MOLA)
  python classify.py image.jpg -c marslandform_v4b_deploy.pt

  # Full accuracy with MOLA DEM
  python classify.py image.jpg -c model.pt --dem Mars_HRSC_MOLA_BlendDEM_Global_200mp_v2.tif --lat 43.68 --lon 13.66

  # GPU + JSON output
  python classify.py image.jpg -c model.pt --dem MOLA.tif --lat 38.2 --lon -26.1 -d cuda -o result.json
        """,
    )
    parser.add_argument("image", help="Path to HiRISE browse image (JPG/PNG)")
    parser.add_argument("-c", "--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("-o", "--output", default=None, help="Output JSON path (default: stdout)")
    parser.add_argument("-d", "--device", default=None, help="cuda / cpu (default: auto)")
    parser.add_argument("-b", "--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--dem", default=None, help="Path to MOLA DEM GeoTIFF")
    parser.add_argument("--lat", type=float, default=None, help="Image center latitude (°N)")
    parser.add_argument("--lon", type=float, default=None, help="Image center longitude (°E)")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for path, label in [(args.image, "Image"), (args.checkpoint, "Checkpoint")]:
        if not Path(path).exists():
            print(f"[Error] {label} not found: {path}", file=sys.stderr)
            sys.exit(1)

    # ── Open MOLA DEM ──
    mola_ds = None
    if args.dem:
        if not Path(args.dem).exists():
            print(f"[Error] DEM not found: {args.dem}", file=sys.stderr)
            sys.exit(1)
        import rasterio
        mola_ds = rasterio.open(args.dem)
        print(f"[MOLA] Loaded DEM: {mola_ds.width}×{mola_ds.height} pixels")

    if args.dem and (args.lat is None or args.lon is None):
        print("[Warn] --dem given but --lat/--lon missing. MOLA features zeroed.", file=sys.stderr)

    # ── Run ──
    t0 = time.time()
    backbone, classifier = load_model(args.checkpoint, device)
    results = classify_image(
        args.image, backbone, classifier, device, args.batch_size,
        mola_ds=mola_ds, lat=args.lat, lon=args.lon,
    )
    summary = summarize(results)
    elapsed = time.time() - t0

    output = {
        "image": str(Path(args.image).name),
        "lat": args.lat,
        "lon": args.lon,
        "mola_dem": str(Path(args.dem).name) if args.dem else None,
        "summary": summary,
        "elapsed_seconds": round(elapsed, 2),
        "tiles": [r.to_dict() for r in results],
    }

    text = json.dumps(output, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"\n[Done] {summary['total_tiles']} tiles → {summary['dominant_class']} "
              f"({summary['dominant_confidence']:.1%}) in {elapsed:.1f}s")
        print(f"       Saved to {args.output}")
    else:
        print(text)

    if mola_ds:
        mola_ds.close()


if __name__ == "__main__":
    main()
