"""
SAM (Segment Anything Model) - Mars HiRISE 이미지 성능 테스트
=============================================================
HiRISE browse 이미지에 SAM ViT-B를 zero-shot으로 적용하여
Mars 지형 세그멘테이션 품질을 평가한다.

출력:
  results/sam_test/
    ├── {product_id}_original.png
    ├── {product_id}_sam_overlay.png
    ├── {product_id}_sam_segments_top30.png
    ├── {product_id}_size_distribution.png
    ├── {product_id}_grid_comparison.png
    └── summary_report.txt
"""

import sys
import os
import time
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch

# -- Config --
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path("/disk1/cspark/MarsLab/Data")
BROWSE_DIR = DATA_ROOT / "HiRISE" / "midlat_browse"
SAM_CHECKPOINT = Path("/disk1/cspark/hirise-api/sam_checkpoints") / "sam_vit_b_01ec64.pth"
OUTPUT_DIR = PROJECT_ROOT / "results" / "sam_test"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 테스트할 이미지 (다양한 위도 = 다양한 지형)
TEST_IMAGES = [
    "ESP_011286_2200_RED.abrowse.jpg",  # 고위도 (~40°N) - LDA/LVF 가능
    "ESP_011358_1255_RED.abrowse.jpg",  # 저위도 (~-54°S) - CCF 가능
    "ESP_011365_1365_RED.abrowse.jpg",  # 중위도 (~-43°S) - 혼합
    "ESP_083500_1400_RED.abrowse.jpg",  # 중위도 (~-40°S)
    "ESP_083513_2060_RED.abrowse.jpg",  # 고위도 (~26°N)
]

# SAM 파라미터
MAX_IMAGE_SIZE = 1024  # 리사이즈 (SAM 입력)
MIN_MASK_AREA_RATIO = 0.001  # 전체 이미지 대비 최소 마스크 면적
TILE_SIZE = 224  # 기존 파이프라인 비교용


def load_sam_model():
    """SAM ViT-B 모델 로드."""
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

    if not SAM_CHECKPOINT.exists():
        print(f"ERROR: SAM checkpoint not found at {SAM_CHECKPOINT}")
        print("Download: wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    sam = sam_model_registry["vit_b"](checkpoint=str(SAM_CHECKPOINT))
    sam.to(device)

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,          # 그리드 밀도
        pred_iou_thresh=0.86,        # IoU 임계값
        stability_score_thresh=0.92, # 안정성 임계값
        min_mask_region_area=500,    # 최소 마스크 픽셀 수
    )
    return mask_generator


def load_and_resize(image_path, max_size=MAX_IMAGE_SIZE):
    """이미지 로드 및 리사이즈."""
    img = Image.open(image_path).convert("RGB")
    original_size = img.size

    # 긴 변 기준 리사이즈
    w, h = img.size
    scale = max_size / max(w, h)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)

    return img, original_size, scale


def generate_masks(mask_generator, image):
    """SAM으로 마스크 생성."""
    img_np = np.array(image)
    t0 = time.time()
    masks = mask_generator.generate(img_np)
    elapsed = time.time() - t0

    # 면적 기준 정렬 (큰 것부터)
    masks = sorted(masks, key=lambda x: x['area'], reverse=True)

    # 최소 면적 필터링
    total_area = img_np.shape[0] * img_np.shape[1]
    masks = [m for m in masks if m['area'] / total_area >= MIN_MASK_AREA_RATIO]

    return masks, elapsed


def visualize_overlay(image, masks, save_path):
    """전체 세그먼트 오버레이 시각화."""
    img_np = np.array(image)
    overlay = img_np.copy().astype(float)

    np.random.seed(42)
    colors = np.random.random((len(masks), 3))

    for i, mask in enumerate(masks):
        seg = mask['segmentation']
        color_mask = np.zeros_like(img_np, dtype=float)
        color_mask[seg] = colors[i] * 255
        overlay[seg] = overlay[seg] * 0.5 + color_mask[seg] * 0.5

    fig, ax = plt.subplots(1, 1, figsize=(12, 16))
    ax.imshow(overlay.astype(np.uint8))
    ax.set_title(f"SAM Segments: {len(masks)} masks", fontsize=14)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_top_segments(image, masks, save_path, top_n=30):
    """상위 N개 세그먼트를 개별적으로 시각화."""
    top_masks = masks[:top_n]
    n_cols = 6
    n_rows = (len(top_masks) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 3 * n_rows))
    if n_rows == 1:
        axes = [axes]
    axes_flat = [ax for row in axes for ax in row]

    img_np = np.array(image)

    for i, (mask, ax) in enumerate(zip(top_masks, axes_flat)):
        seg = mask['segmentation']
        # 마스크 영역만 보여주기
        masked = img_np.copy()
        masked[~seg] = 0

        # 바운딩 박스로 크롭
        ys, xs = np.where(seg)
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        pad = 10
        y0, x0 = max(0, y0 - pad), max(0, x0 - pad)
        y1 = min(img_np.shape[0], y1 + pad)
        x1 = min(img_np.shape[1], x1 + pad)

        ax.imshow(masked[y0:y1, x0:x1])
        ax.set_title(f"#{i+1} area={mask['area']}\niou={mask['predicted_iou']:.2f} stab={mask['stability_score']:.2f}",
                     fontsize=7)
        ax.axis('off')

    for ax in axes_flat[len(top_masks):]:
        ax.axis('off')

    plt.suptitle(f"Top {top_n} Segments by Area", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_size_distribution(masks, save_path, total_area):
    """세그먼트 크기 분포."""
    areas = [m['area'] for m in masks]
    area_pcts = [a / total_area * 100 for a in areas]
    ious = [m['predicted_iou'] for m in masks]
    stabs = [m['stability_score'] for m in masks]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 크기 분포 (히스토그램)
    axes[0].hist(area_pcts, bins=50, color='steelblue', edgecolor='white')
    axes[0].set_xlabel("Segment Area (% of image)")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Size Distribution (n={len(masks)})")
    axes[0].axvline(x=(TILE_SIZE**2) / total_area * 100, color='red',
                     linestyle='--', label=f'{TILE_SIZE}×{TILE_SIZE} tile')
    axes[0].legend()

    # IoU 분포
    axes[1].hist(ious, bins=30, color='coral', edgecolor='white')
    axes[1].set_xlabel("Predicted IoU")
    axes[1].set_title("IoU Score Distribution")

    # Stability 분포
    axes[2].hist(stabs, bins=30, color='seagreen', edgecolor='white')
    axes[2].set_xlabel("Stability Score")
    axes[2].set_title("Stability Score Distribution")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_grid_comparison(image, masks, save_path):
    """기존 224×224 타일 그리드 vs SAM 세그먼트 비교."""
    img_np = np.array(image)
    h, w = img_np.shape[:2]

    fig, axes = plt.subplots(1, 2, figsize=(20, 14))

    # 좌: 기존 타일 그리드
    axes[0].imshow(img_np)
    # 그리드 오버레이 (리사이즈된 이미지 기준 타일 크기 조정)
    scaled_tile = int(TILE_SIZE * (min(w, h) / 2048))  # 대략적인 스케일
    if scaled_tile < 20:
        scaled_tile = 50
    for x in range(0, w, scaled_tile):
        axes[0].axvline(x=x, color='cyan', linewidth=0.3, alpha=0.6)
    for y in range(0, h, scaled_tile):
        axes[0].axhline(y=y, color='cyan', linewidth=0.3, alpha=0.6)
    n_tiles = (w // scaled_tile) * (h // scaled_tile)
    axes[0].set_title(f"Fixed Grid ({scaled_tile}×{scaled_tile}px tiles, ~{n_tiles} tiles)", fontsize=12)
    axes[0].axis('off')

    # 우: SAM 세그먼트 경계
    axes[1].imshow(img_np)
    np.random.seed(42)
    for mask in masks:
        seg = mask['segmentation'].astype(np.uint8)
        # 경계 추출
        from scipy import ndimage
        boundary = seg - ndimage.binary_erosion(seg, iterations=1).astype(np.uint8)
        ys, xs = np.where(boundary)
        color = np.random.random(3)
        axes[1].scatter(xs, ys, c=[color], s=0.1, alpha=0.8)

    axes[1].set_title(f"SAM Segments ({len(masks)} segments)", fontsize=12)
    axes[1].axis('off')

    plt.suptitle("Fixed Grid vs SAM Segmentation", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def run_test():
    """메인 테스트 루프."""
    print("=" * 60)
    print("SAM Mars HiRISE Performance Test")
    print("=" * 60)

    # SAM 로드
    print("\n[1/2] Loading SAM ViT-B...")
    mask_generator = load_sam_model()

    # 테스트 이미지 필터링 (존재하는 것만)
    available = []
    for name in TEST_IMAGES:
        path = BROWSE_DIR / name
        if path.exists():
            available.append((name, path))
        else:
            print(f"  SKIP: {name} (not found)")

    if not available:
        print("ERROR: No test images found!")
        sys.exit(1)

    print(f"\n[2/2] Testing {len(available)} images...\n")

    report_lines = ["SAM Mars HiRISE Test Report", "=" * 40, ""]

    for idx, (name, path) in enumerate(available):
        product_id = name.replace("_RED.abrowse.jpg", "")
        print(f"\n--- [{idx+1}/{len(available)}] {product_id} ---")

        # 로드 & 리사이즈
        img, orig_size, scale = load_and_resize(path)
        print(f"  Original: {orig_size}, Resized: {img.size}, Scale: {scale:.3f}")

        # SAM 실행
        print(f"  Running SAM...")
        masks, elapsed = generate_masks(mask_generator, img)
        print(f"  Generated {len(masks)} masks in {elapsed:.1f}s")

        # 통계
        total_area = img.size[0] * img.size[1]
        areas = [m['area'] for m in masks]
        coverage = sum(areas) / total_area  # 겹침 때문에 1 초과 가능
        avg_iou = np.mean([m['predicted_iou'] for m in masks]) if masks else 0
        avg_stab = np.mean([m['stability_score'] for m in masks]) if masks else 0

        stats = (
            f"  Masks: {len(masks)}, "
            f"Coverage: {coverage:.1%}, "
            f"Avg IoU: {avg_iou:.3f}, "
            f"Avg Stability: {avg_stab:.3f}, "
            f"Time: {elapsed:.1f}s"
        )
        print(stats)

        # 시각화
        prefix = OUTPUT_DIR / product_id

        # 원본 저장
        fig, ax = plt.subplots(1, 1, figsize=(10, 14))
        ax.imshow(np.array(img))
        ax.set_title(f"{product_id} (original)", fontsize=12)
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(f"{prefix}_original.png", dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  Saving visualizations...")
        visualize_overlay(img, masks, f"{prefix}_sam_overlay.png")
        visualize_top_segments(img, masks, f"{prefix}_sam_segments_top30.png")
        visualize_size_distribution(masks, f"{prefix}_size_distribution.png", total_area)
        visualize_grid_comparison(img, masks, f"{prefix}_grid_comparison.png")

        # 리포트
        report_lines.append(f"Image: {product_id}")
        report_lines.append(f"  Original size: {orig_size}")
        report_lines.append(f"  Resized: {img.size}")
        report_lines.append(f"  Num masks: {len(masks)}")
        report_lines.append(f"  Coverage: {coverage:.1%}")
        report_lines.append(f"  Avg predicted IoU: {avg_iou:.3f}")
        report_lines.append(f"  Avg stability: {avg_stab:.3f}")
        report_lines.append(f"  Inference time: {elapsed:.1f}s")
        if masks:
            report_lines.append(f"  Largest segment: {max(areas)/total_area:.1%} of image")
            report_lines.append(f"  Smallest segment: {min(areas)/total_area:.2%} of image")
            report_lines.append(f"  Median segment: {np.median(areas)/total_area:.2%} of image")
        report_lines.append("")

    # Summary
    report_lines.append("=" * 40)
    report_lines.append("CONCLUSION")
    report_lines.append("Check the generated images in results/sam_test/")
    report_lines.append("Compare *_grid_comparison.png to see SAM vs fixed tiles")

    report_path = OUTPUT_DIR / "summary_report.txt"
    report_path.write_text("\n".join(report_lines))
    print(f"\nReport saved: {report_path}")
    print(f"Results dir: {OUTPUT_DIR}")
    print("\nDone!")


if __name__ == "__main__":
    run_test()
