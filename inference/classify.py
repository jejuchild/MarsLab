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
  python classify.py image.jpg --checkpoint model.pt
  python classify.py image.jpg --checkpoint model.pt --output results.json
  python classify.py image.jpg --checkpoint model.pt --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# Optional: tqdm for progress bar (graceful fallback)
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# ── Hugging Face / PEFT imports ──────────────────────────────────────────────
from transformers import Dinov2Model
from peft import LoraConfig, get_peft_model, TaskType


# =============================================================================
# Constants
# =============================================================================

CLASSES = ["LDA", "LVF", "CCF", "OTHER"]
NUM_CLASSES = 4
TILE_SIZE = 224
MIN_CONTENT_FRACTION = 0.3  # skip tiles with >70% black pixels

# Per-class confidence thresholds (below → "Uncertain")
CLASS_THRESHOLDS = {
    "LDA":   0.65,
    "LVF":   0.50,
    "CCF":   0.45,
    "OTHER": 0.55,
}

# ImageNet normalization (DINOv2 pretrained)
TILE_TRANSFORM = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


# =============================================================================
# Model Definitions  (self-contained — no external module imports)
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

        # Freeze everything, unfreeze LoRA adapters
        for param in self.backbone.parameters():
            param.requires_grad = False
        for name, param in self.backbone.named_parameters():
            if "lora_" in name:
                param.requires_grad = True

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Extract CLS token embedding. Returns (B, 768)."""
        outputs = self.backbone(pixel_values=pixel_values)
        return outputs.last_hidden_state[:, 0]


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation: MOLA terrain features condition visual features."""

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
        gamma = self.gamma_proj(h)
        beta = self.beta_proj(h)
        return gamma * visual + beta


class FiLMClassifier(nn.Module):
    """V4b FiLM classifier head: DINOv2 CLS (768) + MOLA (25) → logits."""

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
        modulated = self.film(embeddings, mola)
        return self.classifier(modulated)


# =============================================================================
# Result
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


# =============================================================================
# Tiling
# =============================================================================

def tile_image(
    image: Image.Image,
    tile_size: int = TILE_SIZE,
    min_content: float = MIN_CONTENT_FRACTION,
) -> list[tuple[int, int, Image.Image]]:
    """
    Extract non-overlapping 224×224 tiles from a browse image.
    Filters out tiles with >70% black pixels (edges of browse images).
    Returns list of (grid_x, grid_y, PIL.Image) tuples.
    """
    rgb = image.convert("RGB")
    width, height = rgb.size
    tiles: list[tuple[int, int, Image.Image]] = []
    arr = np.array(rgb)

    for y0 in range(0, height - tile_size + 1, tile_size):
        for x0 in range(0, width - tile_size + 1, tile_size):
            tile_arr = arr[y0 : y0 + tile_size, x0 : x0 + tile_size]
            content_frac = float(np.mean(tile_arr > 10))
            if content_frac < min_content:
                continue
            crop = Image.fromarray(tile_arr)
            tiles.append((x0 // tile_size, y0 // tile_size, crop))

    if not tiles:
        resized = rgb.resize((tile_size, tile_size), Image.Resampling.BICUBIC)
        tiles.append((0, 0, resized))

    return tiles


# =============================================================================
# Model Loading
# =============================================================================

def load_model(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[DinoV2LoRA, FiLMClassifier]:
    """Load backbone + classifier from a single checkpoint file."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("cfg", ckpt.get("config", {}))
    state_dict = ckpt.get("model_state_dict", {})

    # ── Backbone ──
    backbone = DinoV2LoRA(
        model_name=cfg.get("model_name", "facebook/dinov2-base"),
        embed_dim=cfg.get("hidden_dim", 768),
        lora_r=cfg.get("lora_r", 16),
        lora_alpha=cfg.get("lora_alpha", 32),
        lora_dropout=cfg.get("lora_dropout", 0.1),
        lora_targets=cfg.get("lora_targets", ["query", "key", "value"]),
    )
    backbone_state = {k: v for k, v in state_dict.items() if k.startswith("backbone.")}
    if backbone_state:
        backbone.load_state_dict(backbone_state, strict=False)
    backbone.eval().to(device)

    # ── FiLM Classifier Head ──
    classifier = FiLMClassifier(
        visual_dim=cfg.get("hidden_dim", 768),
        mola_dim=cfg.get("mola_dim", 25),
        film_hidden=64,
        head_hidden=cfg.get("head_hidden", 128),
        num_classes=cfg.get("num_classes", NUM_CLASSES),
        dropout=cfg.get("dropout", 0.4),
    )
    head_state = {
        k: v for k, v in state_dict.items()
        if k.startswith("film.") or k.startswith("classifier.")
    }
    if head_state:
        classifier.load_state_dict(head_state, strict=False)
    classifier.eval().to(device)

    epoch = ckpt.get("epoch", "?")
    val_f1 = ckpt.get("val_f1", ckpt.get("test_f1", "?"))
    n_cls = cfg.get("num_classes", NUM_CLASSES)
    classes = cfg.get("class_names", CLASSES[:n_cls])
    print(f"[Model] Loaded from {Path(checkpoint_path).name}")
    print(f"        epoch={epoch}, val_f1={val_f1}, classes={classes}")
    print(f"        device={device}")

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
) -> list[TileResult]:
    """
    Full inference pipeline:
      1. Load image
      2. Tile into 224×224 patches
      3. Extract DINOv2 embeddings (batched)
      4. Run FiLM classifier (MOLA features zeroed — standalone mode)
      5. Apply per-class confidence thresholds
      6. Return per-tile predictions
    """
    image = Image.open(image_path).convert("RGB")
    tiles = tile_image(image)
    n_tiles = len(tiles)

    if n_tiles == 0:
        print("[Warn] No valid tiles extracted.")
        return []

    print(f"[Tile]  {image.size[0]}×{image.size[1]} → {n_tiles} tiles ({TILE_SIZE}×{TILE_SIZE})")

    # ── Step 1: Prepare tile tensors ──
    tile_tensors = []
    for _, _, tile_img in tiles:
        t = TILE_TRANSFORM(tile_img.convert("RGB"))
        tile_tensors.append(t)

    # ── Step 2: Extract DINOv2 embeddings ──
    all_embeddings: list[np.ndarray] = []
    it = range(0, n_tiles, batch_size)
    if tqdm:
        it = tqdm(it, desc="Extracting embeddings", unit="batch")

    for i in it:
        batch = torch.stack(tile_tensors[i : i + batch_size]).to(device)
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            emb = backbone(batch)
        all_embeddings.append(emb.cpu().numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)  # (N, 768)

    # ── Step 3: Run FiLM classifier ──
    #   MOLA features are zeroed in standalone mode (no DEM available).
    #   FiLM is identity-initialized, so gamma≈1, beta≈0 → minimal impact.
    mola_dim = classifier.film.mola_encoder[1].in_features
    mola_zeros = np.zeros((n_tiles, mola_dim), dtype=np.float32)

    emb_t = torch.from_numpy(embeddings).float().to(device)
    mola_t = torch.from_numpy(mola_zeros).float().to(device)

    with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
        logits = classifier(emb_t, mola_t)

    probs = torch.softmax(logits, dim=1).cpu().numpy()  # (N, num_classes)
    preds = np.argmax(probs, axis=1)

    num_classes = probs.shape[1]
    class_names = CLASSES[:num_classes]

    # ── Step 4: Build per-tile results with confidence thresholds ──
    results: list[TileResult] = []
    for idx, (gx, gy, _) in enumerate(tiles):
        pred_idx = int(preds[idx])
        raw_class = class_names[pred_idx]
        conf = float(probs[idx, pred_idx])

        threshold = CLASS_THRESHOLDS.get(raw_class, 0.55)
        predicted = raw_class if conf >= threshold else "Uncertain"

        prob_dict = {class_names[c]: float(probs[idx, c]) for c in range(num_classes)}

        results.append(TileResult(
            grid_x=gx,
            grid_y=gy,
            predicted_class=predicted,
            confidence=conf,
            probabilities=prob_dict,
        ))

    return results


# =============================================================================
# Summary
# =============================================================================

def summarize(results: list[TileResult]) -> dict:
    """Compute per-class summary statistics."""
    n = len(results)
    if n == 0:
        return {"total_tiles": 0, "classes": {}}

    counts: dict[str, list[float]] = {}
    for r in results:
        counts.setdefault(r.predicted_class, []).append(r.confidence)

    summary = {}
    for cls_name in sorted(counts.keys()):
        confs = counts[cls_name]
        summary[cls_name] = {
            "count": len(confs),
            "percentage": round(100 * len(confs) / n, 1),
            "mean_confidence": round(float(np.mean(confs)), 4),
        }

    # Dominant class (excluding "Uncertain")
    real_classes = {k: v for k, v in summary.items() if k != "Uncertain"}
    if real_classes:
        dominant = max(real_classes, key=lambda k: real_classes[k]["count"])
    else:
        dominant = "Uncertain"

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
  python classify.py ESP_016142_2240.jpg --checkpoint marslandform_v4b_deploy.pt
  python classify.py image.png --checkpoint model.pt --device cuda --output result.json
  python classify.py image.jpg --checkpoint model.pt --batch-size 16
        """,
    )
    parser.add_argument("image", type=str, help="Path to HiRISE browse image (JPG/PNG)")
    parser.add_argument("--checkpoint", "-c", required=True, help="Path to .pt checkpoint file")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output JSON path (default: stdout)")
    parser.add_argument("--device", "-d", type=str, default=None, help="Device: cuda / cpu (default: auto)")
    parser.add_argument("--batch-size", "-b", type=int, default=32, help="Batch size for embedding extraction")
    args = parser.parse_args()

    # ── Device ──
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Validate inputs ──
    image_path = Path(args.image)
    ckpt_path = Path(args.checkpoint)
    if not image_path.exists():
        print(f"[Error] Image not found: {image_path}", file=sys.stderr)
        sys.exit(1)
    if not ckpt_path.exists():
        print(f"[Error] Checkpoint not found: {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    # ── Run ──
    t0 = time.time()

    backbone, classifier = load_model(ckpt_path, device)
    tile_results = classify_image(image_path, backbone, classifier, device, args.batch_size)
    summary = summarize(tile_results)

    elapsed = time.time() - t0

    output = {
        "image": str(image_path),
        "summary": summary,
        "elapsed_seconds": round(elapsed, 2),
        "tiles": [r.to_dict() for r in tile_results],
    }

    json_str = json.dumps(output, indent=2, ensure_ascii=False)

    if args.output:
        Path(args.output).write_text(json_str, encoding="utf-8")
        print(f"\n[Done] {summary['total_tiles']} tiles → {summary['dominant_class']} "
              f"({summary['dominant_confidence']:.1%}) in {elapsed:.1f}s")
        print(f"       Saved to {args.output}")
    else:
        print(json_str)


if __name__ == "__main__":
    main()
