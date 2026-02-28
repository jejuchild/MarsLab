"""
Data collection & label unification for MarsLandformNet V2.

Combines label sources:
1. Existing midlat_metadata.json (385 image-level labels from title regex)
2. Hepburn 2020 SGLF inventory (320 GLF locations with lat/lon)
3. Pearson 2024 brain terrain (199 overlapping HiRISE images)
4. Hepburn 2020 crater shapefile (6,520 crater polygons for spatial context)

Outputs:
- unified_labels.json: Per-image labels with source tracking
- dataset_splits.json: Train/val/test splits (stratified by class)
- label_stats.json: Summary statistics
"""

import json
import argparse
import logging
import random
from pathlib import Path
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Mars coordinate utils
MARS_RADIUS_KM = 3389.5
LabelInfo = dict[str, Any]
SpatialGroup = dict[str, Any]

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance on Mars in km."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return MARS_RADIUS_KM * c


def load_existing_labels(metadata_path: Path) -> dict[str, LabelInfo]:
    """Load existing image-level labels from midlat_metadata.json."""
    with open(metadata_path) as f:
        meta = json.load(f)

    labels = {}
    for m in meta:
        img_id = m["image_id"]
        labels[img_id] = {
            "image_id": img_id,
            "title": m.get("title", ""),
            "lat": m.get("lat"),
            "lon": m.get("lon"),
            "existing_class": m.get("class", "UNLABELED"),
            "label_sources": [],
            "final_class": None,
        }
    return labels


def load_hepburn_sglf(excel_path: Path) -> pd.DataFrame:
    """Load Hepburn 2020 SGLF inventory from Excel."""
    df_raw = pd.read_excel(excel_path, header=None)

    # Find header row containing 'Longitude'
    header_idx = None
    for i in range(min(len(df_raw), 15)):
        row_str = " ".join(str(v) for v in df_raw.iloc[i].tolist())
        if "Longitude" in row_str:
            header_idx = i
            break

    if header_idx is None:
        logger.warning(f"Could not find header in {excel_path}")
        return pd.DataFrame()

    # Read data rows (skip header + subheader)
    col_names = [
        "SGLF_ID", "Longitude", "Latitude", "Length_km", "Width_km",
        "Area_km2", "Slope_MOLA", "Slope_HRSC", "Elevation_m",
        "Relief_m", "Orientation_deg",
    ]
    df = df_raw.iloc[header_idx + 2 :].copy()
    df.columns = col_names[: df.shape[1]]
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df = df.dropna(subset=["Longitude", "Latitude"])

    logger.info(f"Loaded {len(df)} SGLFs from Hepburn 2020")
    logger.info(f"  Lat range: {df['Latitude'].min():.1f} to {df['Latitude'].max():.1f}")
    logger.info(f"  Lon range: {df['Longitude'].min():.1f} to {df['Longitude'].max():.1f}")
    return df


def load_pearson_brain_terrain(csv_path: Path) -> pd.DataFrame:
    """Load Pearson 2024 brain terrain assessments."""
    df = pd.read_csv(csv_path)
    # Normalize comment to yes/no/maybe
    df["bt_label"] = df["comment"].str.strip().str.lower()
    df["is_brain_terrain"] = df["bt_label"].apply(
        lambda x: "yes" if isinstance(x, str) and x.startswith("yes") else
                  "maybe" if isinstance(x, str) and x.startswith("maybe") else
                  "partial" if isinstance(x, str) and "partial" in x else "no"
    )
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

    logger.info(f"Loaded {len(df)} Pearson brain terrain assessments")
    logger.info(f"  yes: {(df['is_brain_terrain'] == 'yes').sum()}")
    logger.info(f"  maybe: {(df['is_brain_terrain'] == 'maybe').sum()}")
    logger.info(f"  partial: {(df['is_brain_terrain'] == 'partial').sum()}")
    logger.info(f"  no: {(df['is_brain_terrain'] == 'no').sum()}")
    return df


def match_sglf_to_hirise(
    labels: dict[str, LabelInfo],
    sglf_df: pd.DataFrame,
    radius_km: float = 10.0,
) -> int:
    """
    Match SGLF locations to HiRISE images by proximity.
    HiRISE browse images are ~6×12 km, so 10km radius catches most overlaps.
    """
    matched = 0
    hirise_with_coords = [
        (img_id, info) for img_id, info in labels.items()
        if info["lat"] is not None and info["lon"] is not None
    ]

    if not hirise_with_coords:
        logger.warning("No HiRISE images with coordinates found")
        return 0

    hirise_lats = np.array([info["lat"] for _, info in hirise_with_coords])
    hirise_lons = np.array([info["lon"] for _, info in hirise_with_coords])
    hirise_ids = [img_id for img_id, _ in hirise_with_coords]

    for sglf in sglf_df.to_dict(orient="records"):
        lat_raw = sglf.get("Latitude")
        lon_raw = sglf.get("Longitude")
        if lat_raw is None or lon_raw is None:
            continue
        slat = float(lat_raw)
        slon = float(lon_raw)

        # Vectorized distance computation
        dists = np.array([
            haversine_km(slat, slon, float(hlat), float(hlon))
            for hlat, hlon in zip(hirise_lats, hirise_lons)
        ])
        nearby_idx = np.where(dists < radius_km)[0]

        for idx in nearby_idx:
            img_id = hirise_ids[idx]
            sglf_id_raw = sglf.get("SGLF_ID", 0)
            sglf_id = int(sglf_id_raw) if pd.notna(sglf_id_raw) else 0
            labels[img_id]["label_sources"].append({
                "source": "hepburn_sglf",
                "class": "GLF",
                "sglf_id": sglf_id,
                "distance_km": float(dists[idx]),
                "sglf_lat": slat,
                "sglf_lon": slon,
            })
            matched += 1

    unique_images = len(set(
        img_id for img_id, info in labels.items()
        if any(s["source"] == "hepburn_sglf" for s in info["label_sources"])
    ))
    logger.info(f"SGLF→HiRISE matches: {matched} total, {unique_images} unique images (radius={radius_km}km)")
    return unique_images


def match_pearson_to_hirise(
    labels: dict[str, LabelInfo],
    pearson_df: pd.DataFrame,
) -> int:
    """Match Pearson brain terrain assessments to our HiRISE catalog by image ID."""
    matched = 0
    for row in pearson_df.to_dict(orient="records"):
        img_id = str(row.get("name", ""))
        if img_id in labels:
            bt_label = row.get("is_brain_terrain")
            if bt_label in ("yes", "partial", "maybe"):
                # Brain terrain is associated with CCF, LDA, LVF surfaces
                labels[img_id]["label_sources"].append({
                    "source": "pearson_brain_terrain",
                    "brain_terrain": bt_label,
                    "confidence": row.get("percent", 0),
                    "original_comment": row.get("comment", ""),
                })
                matched += 1

    logger.info(f"Pearson brain terrain matches: {matched} images in our catalog")
    return matched


def _reclassify_brain_terrain(title: str, bt_source: dict[str, object]) -> Tuple[str, str]:
    del bt_source
    title_lower = (title or "").lower()

    crater_keywords = ["crater", "concentric", "crater fill", "ccf", "impact"]
    apron_keywords = ["debris apron", "lobate", "lda", "apron", "scarp", "mesa"]
    valley_keywords = ["valley fill", "lineated", "lvf", "valley"]

    for kw in crater_keywords:
        if kw in title_lower:
            return "CCF", kw
    for kw in apron_keywords:
        if kw in title_lower:
            return "LDA", kw
    for kw in valley_keywords:
        if kw in title_lower:
            return "LVF", kw

    return "CCF", "default_ccf"


def resolve_labels(
    labels: dict[str, LabelInfo],
    sglf_max_distance_km: float = 5.0,
    title_regex_mode: str = "weak",
    reclassify_periglacial: bool = True,
) -> dict[str, LabelInfo]:
    class_counts = Counter()

    for img_id, info in labels.items():
        sources = info["label_sources"]
        existing = info["existing_class"]
        sglf_sources = [s for s in sources if s["source"] == "hepburn_sglf"]
        bt_sources = [s for s in sources if s["source"] == "pearson_brain_terrain"]

        min_dist = None
        if sglf_sources:
            min_dist = min(float(s["distance_km"]) for s in sglf_sources)
            info["sglf_distance_km"] = min_dist
            if min_dist < 2.0:
                info["sglf_distance_bucket"] = "0-2km"
            elif min_dist < 5.0:
                info["sglf_distance_bucket"] = "2-5km"
            elif min_dist < 10.0:
                info["sglf_distance_bucket"] = "5-10km"
            else:
                info["sglf_distance_bucket"] = ">=10km"

        if min_dist is not None and min_dist < sglf_max_distance_km:
            info["final_class"] = "GLF"
            info["label_confidence"] = "expert" if min_dist < 2.0 else "catalog"
            info["label_method"] = "sglf_spatial_close"

        elif bt_sources and any(s.get("brain_terrain") == "yes" for s in bt_sources):
            if reclassify_periglacial:
                cls, reason = _reclassify_brain_terrain(info.get("title", ""), bt_sources[0])
                info["final_class"] = cls
                info["label_confidence"] = "expert"
                info["label_method"] = "brain_terrain_reclassified"
                info["brain_terrain_reclassified"] = True
                info["brain_terrain_reclass_reason"] = reason
                logger.info(f"  Brain terrain reclassified: {img_id} -> {cls} (from title: {reason})")
            else:
                info["final_class"] = "PERIGLACIAL"
                info["label_confidence"] = "uncertain"
                info["label_method"] = "brain_terrain_indicator"

        elif title_regex_mode != "exclude" and existing and existing != "UNLABELED":
            info["final_class"] = existing
            info["label_confidence"] = "weak"
            info["label_method"] = "title_regex"

        elif min_dist is not None and sglf_max_distance_km <= min_dist < 10.0:
            info["final_class"] = "GLF"
            info["label_confidence"] = "low"
            info["label_method"] = "sglf_spatial_far"

        elif bt_sources and any(s.get("brain_terrain") in ("maybe", "partial") for s in bt_sources):
            info["final_class"] = "BACKGROUND"
            info["label_confidence"] = "uncertain"
            info["label_method"] = "brain_terrain_uncertain"

        else:
            info["final_class"] = "UNLABELED"
            info["label_confidence"] = None
            info["label_method"] = None

        class_counts[info["final_class"]] += 1

    logger.info(f"Label resolution complete:")
    for cls, count in sorted(class_counts.items()):
        logger.info(f"  {cls}: {count}")

    return labels


def _build_spatial_groups(
    labels: dict[str, LabelInfo],
    radius_km: float = 20.0,
) -> list[SpatialGroup]:
    candidates = [
        (img_id, info)
        for img_id, info in labels.items()
        if info.get("final_class") not in (None, "UNLABELED")
        and info.get("lat") is not None
        and info.get("lon") is not None
    ]
    candidates = sorted(candidates, key=lambda x: (float(x[1]["lat"]), float(x[1]["lon"])))

    groups: list[SpatialGroup] = []
    for img_id, info in candidates:
        lat = float(info["lat"])
        lon = float(info["lon"])
        assigned = False
        for group in groups:
            dist = haversine_km(lat, lon, group["centroid_lat"], group["centroid_lon"])
            if dist <= radius_km:
                group["image_ids"].append(img_id)
                group["sum_lat"] += lat
                group["sum_lon"] += lon
                n = len(group["image_ids"])
                group["centroid_lat"] = group["sum_lat"] / n
                group["centroid_lon"] = group["sum_lon"] / n
                group["class_counts"][info["final_class"]] += 1
                assigned = True
                break
        if not assigned:
            groups.append(
                {
                    "image_ids": [img_id],
                    "sum_lat": lat,
                    "sum_lon": lon,
                    "centroid_lat": lat,
                    "centroid_lon": lon,
                    "class_counts": Counter([info["final_class"]]),
                }
            )

    return groups


def create_spatial_splits(
    labels: dict[str, LabelInfo],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    radius_km: float = 20.0,
) -> dict[str, list[str]]:
    del test_ratio
    rng = random.Random(seed)

    groups = _build_spatial_groups(labels, radius_km=radius_km)
    if not groups:
        logger.warning("No geolocated labeled images for spatial split; falling back to non-spatial split")
        return create_splits(labels, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=0.15, seed=seed)

    grouped_ids = {img_id for g in groups for img_id in g["image_ids"]}
    for img_id, info in labels.items():
        if info.get("final_class") in (None, "UNLABELED"):
            continue
        if img_id in grouped_ids:
            continue
        groups.append(
            {
                "image_ids": [img_id],
                "sum_lat": 0.0,
                "sum_lon": 0.0,
                "centroid_lat": 0.0,
                "centroid_lon": 0.0,
                "class_counts": Counter([info["final_class"]]),
            }
        )

    total = sum(len(g["image_ids"]) for g in groups)
    target_sizes = {
        "train": int(round(total * train_ratio)),
        "val": int(round(total * val_ratio)),
    }
    target_sizes["test"] = max(0, total - target_sizes["train"] - target_sizes["val"])

    class_totals = Counter()
    for g in groups:
        class_totals.update(g["class_counts"])

    class_targets = {
        "train": {cls: class_totals[cls] * train_ratio for cls in class_totals},
        "val": {cls: class_totals[cls] * val_ratio for cls in class_totals},
        "test": {cls: class_totals[cls] * (1.0 - train_ratio - val_ratio) for cls in class_totals},
    }

    splits = {"train": [], "val": [], "test": []}
    split_sizes = Counter()
    split_class_counts = {"train": Counter(), "val": Counter(), "test": Counter()}

    rng.shuffle(groups)
    groups = sorted(groups, key=lambda g: len(g["image_ids"]), reverse=True)

    split_order = ["train", "val", "test"]
    for g in groups:
        g_size = len(g["image_ids"])
        best_split = None
        best_score = -1e9

        for split in split_order:
            size_need = target_sizes[split] - split_sizes[split]
            size_penalty = 0.0
            if size_need <= 0:
                size_penalty = -0.25 * g_size

            class_need = 0.0
            for cls, cnt in g["class_counts"].items():
                deficit = class_targets[split].get(cls, 0.0) - split_class_counts[split].get(cls, 0)
                class_need += min(cnt, max(deficit, 0.0))

            score = class_need + 0.10 * size_need + size_penalty
            if score > best_score:
                best_score = score
                best_split = split

        assert best_split is not None
        splits[best_split].extend(g["image_ids"])
        split_sizes[best_split] += g_size
        split_class_counts[best_split].update(g["class_counts"])

    for split_name, ids in splits.items():
        counts = Counter(labels[i]["final_class"] for i in ids)
        logger.info(f"  {split_name}: {len(ids)} images — {dict(counts)}")

    return splits


def create_splits(
    labels: dict[str, LabelInfo],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, list[str]]:
    """
    Create stratified train/val/test splits.
    Only includes images with non-UNLABELED labels.
    """
    rng = np.random.RandomState(seed)

    # Group by class
    class_images = defaultdict(list)
    for img_id, info in labels.items():
        cls = info["final_class"]
        if cls not in ("UNLABELED",):
            class_images[cls].append(img_id)

    splits = {"train": [], "val": [], "test": []}

    for cls, img_ids in class_images.items():
        rng.shuffle(img_ids)
        n = len(img_ids)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))
        n_test = n - n_train - n_val
        if n_test < 1:
            n_test = 1
            n_train = n - n_val - n_test

        splits["train"].extend(img_ids[:n_train])
        splits["val"].extend(img_ids[n_train : n_train + n_val])
        splits["test"].extend(img_ids[n_train + n_val :])

    # Shuffle each split
    for k in splits:
        rng.shuffle(splits[k])

    for split_name, ids in splits.items():
        counts = Counter(labels[i]["final_class"] for i in ids)
        logger.info(f"  {split_name}: {len(ids)} images — {dict(counts)}")

    return splits


def main():
    parser = argparse.ArgumentParser(description="Collect and unify Mars landform labels")
    parser.add_argument("--metadata", type=str,
                       default="/disk1/cspark/MarsLab/Data/HiRISE/midlat_metadata.json")
    parser.add_argument("--hepburn-excel", type=str,
                       default="/disk1/cspark/MarsLab/Data/external_datasets/hepburn_glf/Data_File_S1.xlsx")
    parser.add_argument("--pearson-csv", type=str,
                       default="/disk1/cspark/MarsLab/Data/external_datasets/pearson_brain_terrain/Brain Coral Assessment - Revised_Table.csv")
    parser.add_argument("--output-dir", type=str,
                       default="/disk1/cspark/MarsLab/Data/HiRISE/v2_output")
    parser.add_argument("--sglf-radius-km", type=float, default=None,
                       help="Deprecated alias for --sglf-max-distance-km")
    parser.add_argument("--sglf-max-distance-km", type=float, default=5.0,
                       help="Close SGLF distance threshold for GLF labeling")
    parser.add_argument("--title-regex-mode", type=str, default="weak",
                       choices=["primary", "weak", "exclude"],
                       help="How to treat metadata title-regex labels")
    parser.add_argument(
        "--reclassify-periglacial",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reclassify expert-confirmed brain terrain into CCF/LDA/LVF",
    )
    parser.add_argument(
        "--spatial-split",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep nearby images in the same split",
    )
    parser.add_argument("--spatial-split-radius-km", type=float, default=20.0,
                       help="Distance threshold for spatial cluster grouping")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sglf_max_distance_km = args.sglf_max_distance_km
    if args.sglf_radius_km is not None:
        sglf_max_distance_km = args.sglf_radius_km

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load existing labels
    logger.info("=" * 60)
    logger.info("Step 1: Loading existing metadata...")
    labels = load_existing_labels(Path(args.metadata))
    existing_labeled = sum(1 for v in labels.values() if v["existing_class"] != "UNLABELED")
    logger.info(f"  Total images: {len(labels)}")
    logger.info(f"  Already labeled: {existing_labeled}")
    existing_counts = Counter(v["existing_class"] for v in labels.values() if v["existing_class"] != "UNLABELED")
    for cls, cnt in sorted(existing_counts.items()):
        logger.info(f"    {cls}: {cnt}")

    # Step 2: Load Hepburn SGLF data
    logger.info("=" * 60)
    logger.info("Step 2: Loading Hepburn 2020 SGLF inventory...")
    hepburn_path = Path(args.hepburn_excel)
    if hepburn_path.exists():
        sglf_df = load_hepburn_sglf(hepburn_path)
        if not sglf_df.empty:
            match_sglf_to_hirise(labels, sglf_df, radius_km=max(10.0, sglf_max_distance_km))
    else:
        logger.warning(f"Hepburn data not found: {hepburn_path}")

    # Step 3: Load Pearson brain terrain
    logger.info("=" * 60)
    logger.info("Step 3: Loading Pearson 2024 brain terrain data...")
    pearson_path = Path(args.pearson_csv)
    if pearson_path.exists():
        pearson_df = load_pearson_brain_terrain(pearson_path)
        match_pearson_to_hirise(labels, pearson_df)
    else:
        logger.warning(f"Pearson data not found: {pearson_path}")

    # Step 4: Resolve final labels
    logger.info("=" * 60)
    logger.info("Step 4: Resolving final labels...")
    labels = resolve_labels(
        labels,
        sglf_max_distance_km=sglf_max_distance_km,
        title_regex_mode=args.title_regex_mode,
        reclassify_periglacial=args.reclassify_periglacial,
    )

    # Step 5: Create splits
    logger.info("=" * 60)
    logger.info("Step 5: Creating train/val/test splits...")
    if args.spatial_split:
        splits = create_spatial_splits(
            labels,
            seed=args.seed,
            radius_km=args.spatial_split_radius_km,
        )
    else:
        splits = create_splits(labels, seed=args.seed)

    # Step 6: Save outputs
    logger.info("=" * 60)
    logger.info("Step 6: Saving outputs...")

    # Save unified labels (only labeled images, to keep file manageable)
    labeled_only = {
        img_id: info for img_id, info in labels.items()
        if info["final_class"] not in ("UNLABELED",)
    }
    with open(output_dir / "unified_labels.json", "w") as f:
        json.dump(labeled_only, f, indent=2, default=str)
    logger.info(f"  Saved unified_labels.json ({len(labeled_only)} images)")

    # Save splits
    with open(output_dir / "dataset_splits.json", "w") as f:
        json.dump(splits, f, indent=2)
    logger.info(f"  Saved dataset_splits.json")

    # Save stats
    stats = {
        "total_images": len(labels),
        "labeled_images": len(labeled_only),
        "class_distribution": dict(Counter(
            info["final_class"] for info in labels.values()
            if info["final_class"] != "UNLABELED"
        )),
        "label_method_distribution": dict(Counter(
            info.get("label_method", "none") for info in labeled_only.values()
        )),
        "label_confidence_distribution": {
            level: sum(1 for info in labeled_only.values() if info.get("label_confidence") == level)
            for level in ["expert", "catalog", "weak", "low", "uncertain"]
        },
        "reclassified_brain_terrain": {
            "count": sum(1 for info in labeled_only.values() if info.get("brain_terrain_reclassified")),
            "by_class": dict(Counter(
                info["final_class"] for info in labeled_only.values() if info.get("brain_terrain_reclassified")
            )),
        },
        "sglf_distance_distribution": {
            "0-2km": sum(1 for info in labeled_only.values() if 0.0 <= float(info.get("sglf_distance_km", 1e9)) < 2.0),
            "2-5km": sum(1 for info in labeled_only.values() if 2.0 <= float(info.get("sglf_distance_km", 1e9)) < 5.0),
            "5-10km": sum(1 for info in labeled_only.values() if 5.0 <= float(info.get("sglf_distance_km", 1e9)) < 10.0),
        },
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "split_class_distribution": {
            split: dict(Counter(labels[i]["final_class"] for i in ids))
            for split, ids in splits.items()
        },
    }
    with open(output_dir / "label_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"  Saved label_stats.json")

    # Summary
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info(f"  Labeled images: {stats['labeled_images']} / {stats['total_images']}")
    logger.info(f"  Class distribution: {stats['class_distribution']}")
    logger.info(f"  Split sizes: {stats['split_sizes']}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
