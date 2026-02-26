#!/usr/bin/env python3
# pyright: reportMissingTypeStubs=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportAny=false, reportMissingTypeArgument=false, reportUnusedCallResult=false, reportUnknownMemberType=false, reportAttributeAccessIssue=false, reportArgumentType=false, reportUnknownArgumentType=false

import argparse
import csv
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import cast

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from tqdm import tqdm


MARS_RADIUS_KM = 3389.5
PIXEL_SIZE_M = 3.3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tile HiRISE browse images and extract DINOv2 embeddings "
            "without storing intermediate tiles."
        )
    )
    parser.add_argument(
        "--image-dirs",
        type=str,
        default="Data/HiRISE/midlat_browse,arcadia_hirise/jpeg",
        help="Comma-separated directories containing browse JPEG images.",
    )
    parser.add_argument(
        "--metadata-json",
        type=Path,
        default=Path("Data/HiRISE/midlat_metadata.json"),
        help="Path to metadata JSON list.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Data/HiRISE/pipeline_output"),
        help="Output directory for embeddings.npy and tile_metadata.csv.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for DINOv2 inference.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=224,
        help="Tile size in pixels.",
    )
    parser.add_argument(
        "--black-threshold",
        type=float,
        default=0.3,
        help="Maximum black pixel fraction allowed before skipping a tile.",
    )
    parser.add_argument(
        "--labeled-only",
        action="store_true",
        help="Process only images with class != UNLABELED.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of images to process (for testing).",
    )
    return parser.parse_args()


def load_metadata(metadata_json: Path, labeled_only: bool) -> dict[str, dict]:
    with metadata_json.open("r", encoding="utf-8") as f:
        records = json.load(f)

    metadata_by_id: dict[str, dict] = {}
    for rec in records:
        image_id = rec.get("image_id")
        if not image_id:
            continue
        class_label = rec.get("class", "UNLABELED")
        if labeled_only and class_label == "UNLABELED":
            continue
        metadata_by_id[image_id] = {
            "class_label": class_label,
            "lat": rec.get("lat"),
            "lon": rec.get("lon"),
        }
    return metadata_by_id


def meters_per_degree_lat() -> float:
    return 2.0 * math.pi * MARS_RADIUS_KM * 1000.0 / 360.0


def meters_per_degree_lon(center_lat_deg: float) -> float:
    cos_lat = math.cos(math.radians(center_lat_deg))
    cos_lat = max(abs(cos_lat), 1e-8)
    return meters_per_degree_lat() * cos_lat


def tile_center_lat_lon(
    center_lat: float,
    center_lon: float,
    tile_row: int,
    tile_col: int,
    patch_size: int,
    img_height: int,
    img_width: int,
) -> tuple[float, float]:
    tile_row_center = tile_row + patch_size / 2.0
    tile_col_center = tile_col + patch_size / 2.0

    row_offset_px = tile_row_center - (img_height / 2.0)
    col_offset_px = tile_col_center - (img_width / 2.0)

    m_per_deg_lat = meters_per_degree_lat()
    m_per_deg_lon = meters_per_degree_lon(center_lat)

    tile_lat = center_lat - (row_offset_px * PIXEL_SIZE_M / m_per_deg_lat)
    tile_lon = center_lon + (col_offset_px * PIXEL_SIZE_M / m_per_deg_lon)
    return tile_lat, tile_lon


def flush_batch(
    model: torch.nn.Module,
    batch_tensors: list[torch.Tensor],
    batch_meta: list[dict],
    all_embeddings: list[np.ndarray],
    all_metadata: list[dict],
) -> None:
    if not batch_tensors:
        return
    batch = torch.stack(batch_tensors, dim=0)
    with torch.no_grad():
        outputs = model(batch)
    all_embeddings.append(outputs.cpu().numpy().astype(np.float32, copy=False))
    all_metadata.extend(batch_meta)
    batch_tensors.clear()
    batch_meta.clear()


def main() -> None:
    args = parse_args()
    start_time = time.time()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata_by_id = load_metadata(args.metadata_json, args.labeled_only)

    # Parse comma-separated image directories
    image_dirs = [Path(d.strip()) for d in args.image_dirs.split(",") if d.strip()]
    image_files: list[Path] = []
    for d in image_dirs:
        if d.is_dir():
            image_files.extend(
                p for p in d.glob("*")
                if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"}
            )
    image_files.sort()

    # Apply --limit if set
    if args.limit is not None and args.limit > 0:
        image_files = image_files[:args.limit]

    print(f"Found {len(image_files)} images across {len(image_dirs)} directories")

    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    model = cast(
        torch.nn.Module,
        torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False),
    )
    model.eval()

    all_embeddings: list[np.ndarray] = []
    all_metadata: list[dict] = []
    batch_tensors: list[torch.Tensor] = []
    batch_meta: list[dict] = []
    class_counter: Counter = Counter()

    processed_images = 0
    skipped_corrupt = 0
    skipped_missing_metadata = 0

    progress = tqdm(image_files, desc="Images", unit="img")
    for image_path in progress:
        # Strip _RED.abrowse suffix: ESP_011357_2285_RED.abrowse.jpg -> ESP_011357_2285
        stem = image_path.stem
        for suffix in ("_RED.abrowse", "_RED", ".abrowse"):
            if stem.endswith(suffix):
                stem = stem[:-len(suffix)]
                break
        image_id = stem
        meta = metadata_by_id.get(image_id)
        if meta is None:
            skipped_missing_metadata += 1
            continue

        center_lat = meta.get("lat")
        center_lon = meta.get("lon")
        if center_lat is None or center_lon is None:
            skipped_missing_metadata += 1
            continue

        try:
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                img_width, img_height = img.size

                for tile_row in range(0, img_height, args.patch_size):
                    for tile_col in range(0, img_width, args.patch_size):
                        if tile_row + args.patch_size > img_height or tile_col + args.patch_size > img_width:
                            continue

                        patch = img.crop(
                            (
                                tile_col,
                                tile_row,
                                tile_col + args.patch_size,
                                tile_row + args.patch_size,
                            )
                        )

                        patch_arr = np.asarray(patch)
                        black_fraction = (patch_arr < 5).mean()
                        if black_fraction > args.black_threshold:
                            continue

                        patch_tensor = cast(torch.Tensor, transform(patch))
                        tile_lat, tile_lon = tile_center_lat_lon(
                            center_lat=float(center_lat),
                            center_lon=float(center_lon),
                            tile_row=tile_row,
                            tile_col=tile_col,
                            patch_size=args.patch_size,
                            img_height=img_height,
                            img_width=img_width,
                        )

                        batch_tensors.append(patch_tensor)
                        batch_meta.append(
                            {
                                "image_id": image_id,
                                "tile_row": tile_row,
                                "tile_col": tile_col,
                                "lat": tile_lat,
                                "lon": tile_lon,
                                "class_label": meta["class_label"],
                                "source_path": str(image_path),
                            }
                        )

                        if len(batch_tensors) >= args.batch_size:
                            flush_batch(
                                model=model,
                                batch_tensors=batch_tensors,
                                batch_meta=batch_meta,
                                all_embeddings=all_embeddings,
                                all_metadata=all_metadata,
                            )

        except (UnidentifiedImageError, OSError):
            skipped_corrupt += 1
            continue

        processed_images += 1
        if processed_images % 100 == 0:
            tqdm.write(
                (
                    f"Processed {processed_images} images | "
                    f"tiles kept: {len(all_metadata) + len(batch_meta)}"
                )
            )

    flush_batch(
        model=model,
        batch_tensors=batch_tensors,
        batch_meta=batch_meta,
        all_embeddings=all_embeddings,
        all_metadata=all_metadata,
    )

    if all_embeddings:
        embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32, copy=False)
    else:
        embeddings = np.empty((0, 384), dtype=np.float32)

    for rec in all_metadata:
        class_counter[rec["class_label"]] += 1

    embeddings_path = args.output_dir / "embeddings.npy"
    metadata_csv_path = args.output_dir / "tile_metadata.csv"

    np.save(embeddings_path, embeddings)

    with metadata_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_id",
                "tile_row",
                "tile_col",
                "lat",
                "lon",
                "class_label",
                "source_path",
            ],
        )
        writer.writeheader()
        writer.writerows(all_metadata)

    elapsed = time.time() - start_time
    print("Done.")
    print(f"Processed images: {processed_images}")
    print(f"Skipped images (missing metadata/coords): {skipped_missing_metadata}")
    print(f"Skipped corrupt images: {skipped_corrupt}")
    print(f"Total tiles: {embeddings.shape[0]}")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Embeddings saved to: {embeddings_path}")
    print(f"Tile metadata saved to: {metadata_csv_path}")
    print(f"Elapsed seconds: {elapsed:.2f}")
    print("Tiles per class:")
    for class_label, count in sorted(class_counter.items()):
        print(f"  {class_label}: {count}")


if __name__ == "__main__":
    main()
