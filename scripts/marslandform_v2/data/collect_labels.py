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
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Mars coordinate utils
MARS_RADIUS_KM = 3389.5

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance on Mars in km."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return MARS_RADIUS_KM * c


def load_existing_labels(metadata_path: Path) -> Dict[str, dict]:
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
    labels: Dict[str, dict],
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

    for _, sglf in sglf_df.iterrows():
        slat = float(sglf["Latitude"])
        slon = float(sglf["Longitude"])

        # Vectorized distance computation
        dists = np.array([
            haversine_km(slat, slon, float(hlat), float(hlon))
            for hlat, hlon in zip(hirise_lats, hirise_lons)
        ])
        nearby_idx = np.where(dists < radius_km)[0]

        for idx in nearby_idx:
            img_id = hirise_ids[idx]
            labels[img_id]["label_sources"].append({
                "source": "hepburn_sglf",
                "class": "GLF",
                "sglf_id": int(sglf.get("SGLF_ID", 0)),
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
    labels: Dict[str, dict],
    pearson_df: pd.DataFrame,
) -> int:
    """Match Pearson brain terrain assessments to our HiRISE catalog by image ID."""
    matched = 0
    for _, row in pearson_df.iterrows():
        img_id = row["name"]
        if img_id in labels:
            bt_label = row["is_brain_terrain"]
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


def resolve_labels(labels: Dict[str, dict]) -> Dict[str, dict]:
    """
    Resolve final class for each image using priority:
    1. Existing regex-based label (if not UNLABELED)
    2. SGLF proximity → GLF
    3. Brain terrain → periglacial indicator (doesn't override class)
    4. Default: UNLABELED
    """
    class_counts = Counter()

    for img_id, info in labels.items():
        sources = info["label_sources"]
        existing = info["existing_class"]

        # Priority 1: Existing regex label
        if existing and existing != "UNLABELED":
            info["final_class"] = existing
            info["label_confidence"] = "high"
            info["label_method"] = "title_regex"

        # Priority 2: SGLF spatial match → GLF
        elif any(s["source"] == "hepburn_sglf" for s in sources):
            sglf_sources = [s for s in sources if s["source"] == "hepburn_sglf"]
            min_dist = min(s["distance_km"] for s in sglf_sources)
            info["final_class"] = "GLF"
            info["label_confidence"] = "medium" if min_dist < 5.0 else "low"
            info["label_method"] = "sglf_spatial"
            info["sglf_distance_km"] = min_dist

        # Priority 3: Brain terrain (marks as periglacial, but don't assign specific class)
        elif any(s["source"] == "pearson_brain_terrain" for s in sources):
            # Brain terrain can be on CCF, LDA, or LVF — we can't determine which
            # Mark as PERIGLACIAL for now (useful for filtering but not classification)
            info["final_class"] = "PERIGLACIAL"
            info["label_confidence"] = "low"
            info["label_method"] = "brain_terrain_indicator"

        else:
            info["final_class"] = "UNLABELED"
            info["label_confidence"] = None
            info["label_method"] = None

        class_counts[info["final_class"]] += 1

    logger.info(f"Label resolution complete:")
    for cls, count in sorted(class_counts.items()):
        logger.info(f"  {cls}: {count}")

    return labels


def create_splits(
    labels: Dict[str, dict],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """
    Create stratified train/val/test splits.
    Only includes images with non-UNLABELED labels.
    """
    rng = np.random.RandomState(seed)

    # Group by class
    class_images = defaultdict(list)
    for img_id, info in labels.items():
        cls = info["final_class"]
        if cls not in ("UNLABELED", "PERIGLACIAL"):
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
    parser.add_argument("--sglf-radius-km", type=float, default=10.0,
                       help="Matching radius for SGLF→HiRISE spatial join")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

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
            match_sglf_to_hirise(labels, sglf_df, radius_km=args.sglf_radius_km)
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
    labels = resolve_labels(labels)

    # Step 5: Create splits
    logger.info("=" * 60)
    logger.info("Step 5: Creating train/val/test splits...")
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
