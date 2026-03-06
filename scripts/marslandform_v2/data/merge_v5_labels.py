#!/usr/bin/env python3
"""
Merge V4b (LDA/LVF/CCF/OTHER) and SCT tile labels into unified V5 dataset.

V5 classes: LDA(0), LVF(1), CCF(2), OTHER(3), SCT(4)

Steps:
1. Load existing tile_labels_v3.json (V4b labels: LDA/LVF/CCF/OTHER)
2. Load sct_tile_labels.json (SCT labels from Wang et al. 2026)
3. Filter out UNLABELED tiles from both
4. Handle overlapping image_ids (same image may appear in both sets)
5. Create spatial splits (train/val/test with 20km exclusion radius)
6. Sample exemplar buffer from old classes
7. Output: tile_labels_v5.json + tile_splits_v5.json
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(os.getenv("MARSLAB_ROOT", "/disk1/cspark/MarsLab"))
V3_LABELS = ROOT / "Data" / "HiRISE" / "v3_output" / "tile_labels_v3.json"
SCT_LABELS = ROOT / "Data" / "HiRISE" / "v3_output" / "sct_tile_labels.json"
V3_SPLITS = ROOT / "Data" / "HiRISE" / "v3_output" / "tile_splits_v3.json"
OUTPUT_DIR = ROOT / "Data" / "HiRISE" / "v3_output"

MARS_RADIUS_KM = 3389.5
V5_CLASSES = ["LDA", "LVF", "CCF", "OTHER", "SCT"]

# Subsampling caps per class (train set)
SUBSAMPLE_CAPS = {
    "LDA": 20000,
    "LVF": None,     # keep all (minority)
    "CCF": None,     # keep all (minority)
    "OTHER": 20000,
    "SCT": None,     # keep all (new class, want maximum data)
}

EXEMPLAR_PER_CLASS = 2000  # for replay buffer
SPATIAL_SPLIT_RADIUS_KM = 20.0
SEED = 42


def haversine_km(lat1, lon1, lat2, lon2):
    """Haversine distance on Mars in km."""
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return 2 * MARS_RADIUS_KM * math.asin(min(1.0, math.sqrt(a)))


def spatial_split(image_ids_with_coords: list[tuple[str, float, float]],
                  radius_km: float = SPATIAL_SPLIT_RADIUS_KM,
                  ratios: tuple[float, ...] = (0.70, 0.15, 0.15),
                  seed: int = SEED) -> dict[str, set[str]]:
    """
    Spatial split of images into train/val/test ensuring
    no val/test image is within radius_km of any train image.
    """
    rng = random.Random(seed)

    # Deduplicate
    seen = set()
    unique = []
    for img_id, lat, lon in image_ids_with_coords:
        if img_id not in seen:
            seen.add(img_id)
            unique.append((img_id, lat, lon))

    rng.shuffle(unique)
    n = len(unique)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])

    splits = {"train": set(), "val": set(), "test": set()}
    assigned = {}  # img_id → split

    for img_id, lat, lon in unique:
        if len(splits["train"]) < n_train:
            splits["train"].add(img_id)
            assigned[img_id] = "train"
        elif len(splits["val"]) < n_val:
            # Check distance to any train image
            too_close = False
            for tid in list(splits["train"])[:200]:  # sample for speed
                tlat, tlon = next((la, lo) for i, la, lo in unique if i == tid)
                if haversine_km(lat, lon, tlat, tlon) < radius_km:
                    too_close = True
                    break
            if too_close:
                splits["train"].add(img_id)
                assigned[img_id] = "train"
            else:
                splits["val"].add(img_id)
                assigned[img_id] = "val"
        else:
            splits["test"].add(img_id)
            assigned[img_id] = "test"

    return splits, assigned


def main():
    parser = argparse.ArgumentParser(description="Merge V4b + SCT labels into V5")
    parser.add_argument("--v3-labels", type=Path, default=V3_LABELS)
    parser.add_argument("--sct-labels", type=Path, default=SCT_LABELS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)

    # 1. Load existing V4b labels
    logger.info(f"Loading V4b labels from {args.v3_labels}")
    with open(args.v3_labels) as f:
        v3_labels = json.load(f)
    logger.info(f"  V4b tiles: {len(v3_labels)}")

    # Filter UNLABELED
    v3_labels = [t for t in v3_labels if t["label"] != "UNLABELED"]
    v3_counts = Counter(t["label"] for t in v3_labels)
    logger.info(f"  After filter: {len(v3_labels)} — {dict(v3_counts)}")

    # 2. Load SCT labels
    logger.info(f"Loading SCT labels from {args.sct_labels}")
    with open(args.sct_labels) as f:
        sct_labels = json.load(f)
    logger.info(f"  SCT tiles: {len(sct_labels)}")

    sct_labels = [t for t in sct_labels if t["label"] != "UNLABELED"]
    sct_counts = Counter(t["label"] for t in sct_labels)
    logger.info(f"  After filter: {len(sct_labels)} — {dict(sct_counts)}")

    # 3. Handle overlaps — if a tile appears in both V4b and SCT:
    # - V4b gave it a landform label (LDA/LVF/CCF) → keep V4b label (Levy is expert-annotated)
    # - V4b gave it OTHER + SCT gave it SCT → use SCT label
    # - Both give OTHER → keep as OTHER
    v3_by_key = {}
    for t in v3_labels:
        key = (t["image_id"], t["tile_row"], t["tile_col"])
        v3_by_key[key] = t

    merged = list(v3_labels)  # start with all V4b tiles
    added_sct = 0
    upgraded_sct = 0
    skipped_overlap = 0

    for t in sct_labels:
        key = (t["image_id"], t["tile_row"], t["tile_col"])

        if key in v3_by_key:
            existing = v3_by_key[key]
            if existing["label"] in ("LDA", "LVF", "CCF"):
                # Keep expert label — Levy annotation wins
                skipped_overlap += 1
            elif existing["label"] == "OTHER" and t["label"] == "SCT":
                # Upgrade to SCT
                existing["label"] = "SCT"
                existing["label_type"] = t["label_type"]
                existing["coverage"]["SCT"] = t["coverage"].get("SCT", 0)
                existing["max_coverage"] = max(existing["max_coverage"], t["max_coverage"])
                upgraded_sct += 1
            else:
                skipped_overlap += 1
        else:
            # New tile not in V4b
            merged.append(t)
            added_sct += 1

    logger.info(f"\nMerge results:")
    logger.info(f"  Added new SCT tiles: {added_sct}")
    logger.info(f"  Upgraded OTHER→SCT: {upgraded_sct}")
    logger.info(f"  Skipped (Levy wins): {skipped_overlap}")
    logger.info(f"  Total merged: {len(merged)}")

    final_counts = Counter(t["label"] for t in merged)
    logger.info(f"  Class distribution: {dict(final_counts)}")

    # 4. Collect image coords for spatial split
    image_coords = {}
    for t in merged:
        iid = t["image_id"]
        if iid not in image_coords and t.get("lat") and t.get("lon"):
            image_coords[iid] = (t["lat"], t["lon"])

    images_with_coords = [(iid, lat, lon) for iid, (lat, lon) in image_coords.items()]
    logger.info(f"\n  Unique images: {len(images_with_coords)}")

    # 5. Spatial split
    logger.info("Computing spatial split...")
    splits_sets, img_to_split = spatial_split(images_with_coords, seed=args.seed)
    for name, ids in splits_sets.items():
        logger.info(f"  {name}: {len(ids)} images")

    # 6. Build split indices
    split_indices = {"train": [], "val": [], "test": []}
    for i, t in enumerate(merged):
        split_name = img_to_split.get(t["image_id"], "train")
        split_indices[split_name].append(i)

    for name, indices in split_indices.items():
        counts = Counter(merged[i]["label"] for i in indices)
        logger.info(f"  {name}: {len(indices)} tiles — {dict(counts)}")

    # 7. Subsample majority classes in train
    logger.info("\nSubsampling...")
    train_by_class = defaultdict(list)
    for idx in split_indices["train"]:
        label = merged[idx]["label"]
        train_by_class[label].append(idx)

    subsampled_train = []
    for label, indices in train_by_class.items():
        cap = SUBSAMPLE_CAPS.get(label)
        if cap is not None and len(indices) > cap:
            chosen = rng.choice(indices, size=cap, replace=False).tolist()
            logger.info(f"  {label}: {len(indices)} → {len(chosen)} (capped at {cap})")
        else:
            chosen = indices
            logger.info(f"  {label}: {len(indices)} (kept all)")
        subsampled_train.extend(chosen)

    split_indices["train"] = sorted(subsampled_train)

    # 8. Sample exemplar buffer from old classes
    logger.info("\nSampling exemplar buffer...")
    exemplar_buffer = {}
    for label in ["LDA", "LVF", "CCF", "OTHER"]:
        class_train = [i for i in split_indices["train"] if merged[i]["label"] == label]
        n = min(EXEMPLAR_PER_CLASS, len(class_train))
        chosen = rng.choice(class_train, size=n, replace=False).tolist() if class_train else []
        exemplar_buffer[label] = chosen
        logger.info(f"  {label}: {len(chosen)} exemplars")

    # 9. Save
    out_labels = args.output_dir / "tile_labels_v5.json"
    out_splits = args.output_dir / "tile_splits_v5.json"
    out_exemplars = args.output_dir / "exemplar_buffer_v5.json"

    with open(out_labels, "w") as f:
        json.dump(merged, f, indent=1)
    logger.info(f"\nSaved {len(merged)} labels to {out_labels}")

    with open(out_splits, "w") as f:
        json.dump(split_indices, f, indent=1)
    logger.info(f"Saved splits to {out_splits}")

    with open(out_exemplars, "w") as f:
        json.dump(exemplar_buffer, f, indent=2)
    logger.info(f"Saved exemplar buffer to {out_exemplars}")

    # Final summary
    logger.info("\n=== V5 Dataset Summary ===")
    for split_name in ["train", "val", "test"]:
        counts = Counter(merged[i]["label"] for i in split_indices[split_name])
        logger.info(f"  {split_name}: {len(split_indices[split_name])} tiles — {dict(counts)}")
    total_exemplars = sum(len(v) for v in exemplar_buffer.values())
    logger.info(f"  exemplar buffer: {total_exemplars} tiles")


if __name__ == "__main__":
    main()
