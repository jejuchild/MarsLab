"""
DINOv2 Feature Map 기반 Mars 지형 클러스터링 테스트
===================================================
Fine-tuned DINOv2 LoRA의 CLS 토큰을 타일별로 추출하여
K-Means 클러스터링 + UMAP 시각화로 지형 경계 검출 가능성을 평가.

출력:
  results/dinov2_cluster_test/
    ├── {product_id}_cluster_k{k}.png       (클러스터 맵)
    ├── {product_id}_umap.png               (UMAP scatter)
    ├── {product_id}_comparison.png         (클러스터 vs 기존 분류)
    └── summary_report.txt
"""

import sys
import os
import time
import json
import numpy as np
from pathlib import Path
from PIL import Image
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

import torch
from torchvision import transforms

# -- Paths --
PROJECT_ROOT = Path("/disk1/cspark/hirise-api")
MARSLAB_ROOT = Path("/disk1/cspark/MarsLab")
DATA_DIR = MARSLAB_ROOT / "Data" / "HiRISE"
BROWSE_DIR = DATA_DIR / "midlat_browse"
CHECKPOINT = DATA_DIR / "v3_output" / "models" / "marslandform_v4b_deploy.pt"
OUTPUT_DIR = PROJECT_ROOT / "results" / "dinov2_cluster_test"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -- Inject paths --
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(MARSLAB_ROOT))
sys.path.insert(0, str(MARSLAB_ROOT / "scripts" / "marslandform_v2"))
os.chdir(str(PROJECT_ROOT))

# Ensure PROJECT_ROOT is first in sys.path for 'training' package
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
else:
    sys.path.remove(str(PROJECT_ROOT))
    sys.path.insert(0, str(PROJECT_ROOT))

# -- Test images (same as SAM test) --
TEST_IMAGES = [
    "ESP_011286_2200_RED.abrowse.jpg",
    "ESP_011358_1255_RED.abrowse.jpg",
    "ESP_011365_1365_RED.abrowse.jpg",
    "ESP_083500_1400_RED.abrowse.jpg",
    "ESP_083513_2060_RED.abrowse.jpg",
]

TILE_SIZE = 224
MIN_CONTENT = 0.3
K_VALUES = [4, 6, 8]

# Class colors (consistent with existing pipeline)
CLASS_COLORS = {
    "LDA": "#e74c3c",
    "LVF": "#3498db",
    "CCF": "#2ecc71",
    "OTHER": "#95a5a6",
    "SCT": "#f39c12",
}


def extract_tiles(image_path):
    """224x224 타일 추출 (기존 파이프라인과 동일)."""
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img)
    w, h = img.size
    tiles = []

    for row in range(0, arr.shape[0] - TILE_SIZE + 1, TILE_SIZE):
        for col in range(0, arr.shape[1] - TILE_SIZE + 1, TILE_SIZE):
            tile_arr = arr[row:row + TILE_SIZE, col:col + TILE_SIZE]
            content_frac = np.mean(tile_arr > 10)
            if content_frac < MIN_CONTENT:
                continue
            tiles.append({
                "row": row // TILE_SIZE,
                "col": col // TILE_SIZE,
                "pixel_row": row,
                "pixel_col": col,
                "tile_array": tile_arr,
            })
    return tiles, img


def load_dinov2_backbone():
    """V4b 체크포인트에서 DINOv2 backbone 로드."""
    from training.models.dinov2_lora import DinoV2LoRA
    from scripts.marslandform_v2.config import DINOv2Config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # DINOv2 LoRA 초기화 (pretrained base)
    config = DINOv2Config()
    model = DinoV2LoRA(config=config, use_lora=True)

    # V4b 체크포인트에서 backbone weights 로드
    checkpoint = torch.load(str(CHECKPOINT), map_location="cpu", weights_only=False)

    # 체크포인트 키 확인
    if "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    elif "student_backbone" in checkpoint:
        state = checkpoint["student_backbone"]
    else:
        state = checkpoint

    # backbone 관련 키만 추출
    backbone_keys = {k: v for k, v in state.items() if k.startswith("backbone.")}
    if backbone_keys:
        # backbone. prefix 제거
        backbone_state = {k.replace("backbone.", "", 1): v for k, v in backbone_keys.items()}
        model.backbone.load_state_dict(backbone_state, strict=False)
        print(f"  Loaded {len(backbone_keys)} backbone keys from checkpoint")
    else:
        # 직접 로드 시도
        loaded = model.load_state_dict(state, strict=False)
        print(f"  Direct load: missing={len(loaded.missing_keys)}, unexpected={len(loaded.unexpected_keys)}")

    model.eval()
    model.to(device)
    return model, device


@torch.no_grad()
def extract_features(model, tiles, device, batch_size=32):
    """타일들에서 CLS 토큰 추출."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    all_features = []
    for i in range(0, len(tiles), batch_size):
        batch_tiles = tiles[i:i + batch_size]
        batch_tensors = torch.stack([
            transform(Image.fromarray(t["tile_array"])) for t in batch_tiles
        ]).to(device)

        # CLS token
        cls = model(batch_tensors)  # (batch, 768)
        all_features.append(cls.cpu().numpy())

    return np.concatenate(all_features, axis=0)  # (n_tiles, 768)


def cluster_features(features, k):
    """K-Means 클러스터링."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features_scaled)

    return labels, kmeans, features_scaled


def compute_umap(features_scaled):
    """UMAP 2D 임베딩."""
    import umap
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    embedding = reducer.fit_transform(features_scaled)
    return embedding


def visualize_cluster_map(image, tiles, labels, k, save_path):
    """타일 위치에 클러스터 색상을 오버레이."""
    img_arr = np.array(image)
    overlay = img_arr.copy().astype(float)

    cmap = plt.cm.get_cmap("tab10", k)
    colors = [cmap(i)[:3] for i in range(k)]

    for tile, label in zip(tiles, labels):
        r, c = tile["pixel_row"], tile["pixel_col"]
        color = np.array(colors[label]) * 255
        overlay[r:r+TILE_SIZE, c:c+TILE_SIZE] = (
            overlay[r:r+TILE_SIZE, c:c+TILE_SIZE] * 0.4 + color * 0.6
        )

    fig, ax = plt.subplots(1, 1, figsize=(12, 16))
    ax.imshow(overlay.astype(np.uint8))
    ax.set_title(f"DINOv2 K-Means Clustering (k={k})", fontsize=14)

    patches = [mpatches.Patch(color=colors[i], label=f"Cluster {i}") for i in range(k)]
    ax.legend(handles=patches, loc='upper right', fontsize=9)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_umap(embedding, labels_dict, save_path):
    """UMAP scatter — 여러 k 값 동시 시각화."""
    n_plots = len(labels_dict)
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 6))
    if n_plots == 1:
        axes = [axes]

    for ax, (k, labels) in zip(axes, labels_dict.items()):
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1],
                           c=labels, cmap='tab10', s=8, alpha=0.7)
        ax.set_title(f"UMAP (k={k})", fontsize=12)
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        plt.colorbar(scatter, ax=ax, label="Cluster")

    plt.suptitle("DINOv2 Feature Space — UMAP Projection", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_comparison(image, tiles, cluster_labels, k, save_path):
    """클러스터 맵 + 원본 + 타일 그리드 비교."""
    img_arr = np.array(image)

    fig, axes = plt.subplots(1, 3, figsize=(24, 14))

    # 1) 원본
    axes[0].imshow(img_arr)
    axes[0].set_title("Original", fontsize=12)
    axes[0].axis('off')

    # 2) 타일 그리드 (기존 방식)
    axes[1].imshow(img_arr)
    for tile in tiles:
        r, c = tile["pixel_row"], tile["pixel_col"]
        rect = plt.Rectangle((c, r), TILE_SIZE, TILE_SIZE,
                            linewidth=0.3, edgecolor='cyan', facecolor='none')
        axes[1].add_patch(rect)
    axes[1].set_title(f"Fixed 224×224 Grid ({len(tiles)} tiles)", fontsize=12)
    axes[1].axis('off')

    # 3) 클러스터 맵
    overlay = img_arr.copy().astype(float)
    cmap = plt.cm.get_cmap("tab10", k)
    colors = [cmap(i)[:3] for i in range(k)]
    for tile, label in zip(tiles, cluster_labels):
        r, c = tile["pixel_row"], tile["pixel_col"]
        color = np.array(colors[label]) * 255
        overlay[r:r+TILE_SIZE, c:c+TILE_SIZE] = (
            overlay[r:r+TILE_SIZE, c:c+TILE_SIZE] * 0.4 + color * 0.6
        )
    axes[2].imshow(overlay.astype(np.uint8))
    patches = [mpatches.Patch(color=colors[i], label=f"Cluster {i}") for i in range(k)]
    axes[2].legend(handles=patches, loc='upper right', fontsize=8)
    axes[2].set_title(f"DINOv2 Clusters (k={k})", fontsize=12)
    axes[2].axis('off')

    plt.suptitle("Original → Tile Grid → DINOv2 Clustering", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def run_test():
    print("=" * 60)
    print("DINOv2 Feature Clustering — Mars HiRISE Test")
    print("=" * 60)

    # 1. 모델 로드
    print("\n[1] Loading DINOv2 LoRA backbone...")
    model, device = load_dinov2_backbone()

    # 이미지 필터링
    available = []
    for name in TEST_IMAGES:
        path = BROWSE_DIR / name
        if path.exists():
            available.append((name, path))
        else:
            print(f"  SKIP: {name}")

    print(f"\n[2] Testing {len(available)} images with k={K_VALUES}\n")
    report_lines = ["DINOv2 Clustering Test Report", "=" * 40, ""]

    for idx, (name, path) in enumerate(available):
        product_id = name.replace("_RED.abrowse.jpg", "")
        print(f"\n--- [{idx+1}/{len(available)}] {product_id} ---")

        # 타일 추출
        tiles, img = extract_tiles(path)
        print(f"  Extracted {len(tiles)} tiles from {img.size}")

        if len(tiles) < 10:
            print(f"  Too few tiles, skipping")
            continue

        # 피처 추출
        t0 = time.time()
        features = extract_features(model, tiles, device)
        feat_time = time.time() - t0
        print(f"  Features: {features.shape} in {feat_time:.1f}s")

        # 클러스터링 (여러 k)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        labels_dict = {}
        for k in K_VALUES:
            labels, kmeans, _ = cluster_features(features, k)
            labels_dict[k] = labels

            # 클러스터 분포
            dist = Counter(labels)
            dist_str = ", ".join(f"C{c}:{n}" for c, n in sorted(dist.items()))
            print(f"  k={k}: {dist_str}")

            # 클러스터 맵 시각화
            save = OUTPUT_DIR / f"{product_id}_cluster_k{k}.png"
            visualize_cluster_map(img, tiles, labels, k, save)

        # UMAP
        print(f"  Computing UMAP...")
        embedding = compute_umap(features_scaled)
        visualize_umap(embedding, labels_dict, OUTPUT_DIR / f"{product_id}_umap.png")

        # 비교 시각화 (k=6 기본)
        best_k = 6
        visualize_comparison(img, tiles, labels_dict[best_k], best_k,
                           OUTPUT_DIR / f"{product_id}_comparison.png")

        # 리포트
        report_lines.append(f"Image: {product_id}")
        report_lines.append(f"  Size: {img.size}, Tiles: {len(tiles)}")
        report_lines.append(f"  Feature extraction: {feat_time:.1f}s")
        for k in K_VALUES:
            dist = Counter(labels_dict[k])
            report_lines.append(f"  k={k}: {dict(sorted(dist.items()))}")
        report_lines.append("")

    # 저장
    report_path = OUTPUT_DIR / "summary_report.txt"
    report_path.write_text("\n".join(report_lines))
    print(f"\nReport: {report_path}")
    print(f"Results: {OUTPUT_DIR}")
    print("Done!")


if __name__ == "__main__":
    run_test()
