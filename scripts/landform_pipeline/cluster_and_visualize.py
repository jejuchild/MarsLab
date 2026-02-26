#!/usr/bin/env python3
"""Cluster combined DINO+MOLA tile features and generate visual outputs."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


CLASS_ORDER = ["LDA", "CCF", "LVF", "GLF", "UNLABELED"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run K-Means clustering on DINO+MOLA features and export visualizations."
    )
    parser.add_argument(
        "--embeddings",
        default="Data/HiRISE/pipeline_output/embeddings.npy",
        help="Path to DINO embeddings .npy file.",
    )
    parser.add_argument(
        "--mola-features",
        default="Data/HiRISE/pipeline_output/mola_features.npy",
        help="Path to MOLA features .npy file.",
    )
    parser.add_argument(
        "--tile-metadata",
        default="Data/HiRISE/pipeline_output/tile_metadata.csv",
        help="Path to tile metadata CSV.",
    )
    parser.add_argument(
        "--image-dirs",
        default="Data/HiRISE/midlat_browse,arcadia_hirise/jpeg",
        help="Comma-separated directories containing source browse JPGs.",
    )
    parser.add_argument(
        "--output-dir",
        default="Data/HiRISE/pipeline_output/clusters",
        help="Output directory for clustering artifacts.",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=40,
        help="Number of clusters for MiniBatchKMeans.",
    )
    parser.add_argument(
        "--mola-weight",
        type=float,
        default=1.0,
        help="Weight multiplier for normalized MOLA features.",
    )
    parser.add_argument(
        "--samples-per-cluster",
        type=int,
        default=25,
        help="Number of sampled tiles saved per cluster (capped at 50).",
    )
    parser.add_argument(
        "--skip-umap",
        action="store_true",
        help="Skip UMAP/t-SNE overview scatter generation.",
    )
    parser.add_argument(
        "--no-pca",
        action="store_true",
        help="Disable PCA reduction of DINO embeddings before clustering.",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=64,
        help="Number of PCA components for DINO embeddings.",
    )
    return parser.parse_args()


def sanitize_class_labels(values: Any) -> pd.Series:
    series = pd.Series(values)
    labels = series.fillna("UNLABELED").astype(str).str.upper()
    labels = labels.where(labels.isin(CLASS_ORDER), "UNLABELED")
    return labels


def safe_series_mean(series: pd.Series) -> float:
    values = np.asarray(pd.to_numeric(series, errors="coerce"), dtype=float)
    if values.size == 0:
        return float("nan")
    return float(np.nanmean(values))


def find_image(image_id: str, image_dirs: Sequence[Path], source_path: Optional[str]) -> Optional[Path]:
    if source_path:
        candidate = Path(source_path)
        if candidate.exists():
            return candidate
    for image_dir in image_dirs:
        candidate = image_dir / f"{image_id}_RED.abrowse.jpg"
        if candidate.exists():
            return candidate
    return None


def extract_tile(image_path: Path, tile_row: int, tile_col: int, patch_size: int = 224) -> Optional[Image.Image]:
    """Extract tile from source image. tile_row/tile_col are pixel offsets, not grid indices."""
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        x = tile_col  # already pixel offset from tile_and_extract.py
        y = tile_row  # already pixel offset
        if x + patch_size > img.width or y + patch_size > img.height:
            return None
        return img.crop((x, y, x + patch_size, y + patch_size))


def maybe_reduce_dino(dino_norm: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    if args.no_pca:
        print("[info] PCA disabled; using full normalized DINO embeddings.")
        return dino_norm

    n_samples, n_features = dino_norm.shape
    n_components = min(args.pca_components, n_samples, n_features)
    if n_components < 2:
        print("[warn] PCA components too small for reduction; using full DINO embeddings.")
        return dino_norm

    print(f"[info] Running PCA on DINO embeddings: {n_features} -> {n_components}")
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=42)
    return pca.fit_transform(dino_norm)


def build_overview_scatter(
    combined_features: np.ndarray,
    cluster_labels: np.ndarray,
    class_labels: np.ndarray,
    output_dir: Path,
    skip_umap: bool,
    max_points: int = 10_000,
) -> None:
    if skip_umap:
        print("[info] --skip-umap set; skipping overview scatter generation.")
        return

    n_tiles = combined_features.shape[0]
    rng = np.random.default_rng(42)
    if n_tiles > max_points:
        sample_idx = rng.choice(n_tiles, size=max_points, replace=False)
    else:
        sample_idx = np.arange(n_tiles)

    sampled_features = combined_features[sample_idx]
    sampled_clusters = cluster_labels[sample_idx]
    sampled_classes = class_labels[sample_idx]

    reducer_name = "UMAP"
    try:
        umap_module = importlib.import_module("umap")
        reducer = umap_module.UMAP(n_components=2, random_state=42)
    except ImportError:
        from sklearn.manifold import TSNE

        reducer_name = "t-SNE"
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)

    print(f"[info] Running {reducer_name} on {sampled_features.shape[0]} sampled tiles...")
    coords_2d = reducer.fit_transform(sampled_features)

    plt.figure(figsize=(12, 10))
    unique_clusters = np.unique(sampled_clusters)
    cmap = plt.cm.get_cmap("tab20", max(20, len(unique_clusters)))
    for i, cluster_id in enumerate(unique_clusters):
        mask = sampled_clusters == cluster_id
        plt.scatter(
            coords_2d[mask, 0],
            coords_2d[mask, 1],
            s=8,
            alpha=0.7,
            c=[cmap(i)],
            label=f"C{cluster_id}",
        )
    plt.title(f"{reducer_name} Overview Colored by Cluster")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    if len(unique_clusters) <= 40:
        plt.legend(markerscale=2, fontsize=7, ncol=4, frameon=False)
    plt.tight_layout()
    plt.savefig(output_dir / "overview_scatter.png", dpi=220)
    plt.close()

    class_colors = {
        "LDA": "#d7191c",
        "CCF": "#2c7bb6",
        "LVF": "#fdae61",
        "GLF": "#1a9641",
        "UNLABELED": "#8c8c8c",
    }
    plt.figure(figsize=(12, 10))
    for cls in CLASS_ORDER:
        mask = sampled_classes == cls
        if mask.any():
            plt.scatter(
                coords_2d[mask, 0],
                coords_2d[mask, 1],
                s=8,
                alpha=0.7,
                c=class_colors.get(cls, "#8c8c8c"),
                label=cls,
            )
    plt.title(f"{reducer_name} Overview Colored by Class")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(markerscale=2, fontsize=9, frameon=False)
    plt.tight_layout()
    plt.savefig(output_dir / "overview_scatter_by_class.png", dpi=220)
    plt.close()


def save_cluster_grid(
    cluster_id: int,
    cluster_indices: np.ndarray,
    metadata: pd.DataFrame,
    image_dirs: Sequence[Path],
    output_root: Path,
    samples_per_cluster: int,
    rng: np.random.Generator,
) -> Tuple[Optional[Path], int]:
    cluster_dir = output_root / f"cluster_{cluster_id:02d}"
    cluster_dir.mkdir(parents=True, exist_ok=True)

    if len(cluster_indices) == 0:
        return None, 0

    n_samples = min(samples_per_cluster, len(cluster_indices))
    sampled = rng.choice(cluster_indices, size=n_samples, replace=False)

    tiles: List[Image.Image] = []
    for out_i, tile_idx in enumerate(sampled):
        row = metadata.iloc[int(tile_idx)]
        image_id = str(row.get("image_id", "")).strip()
        source_path = row.get("source_path")
        image_path = find_image(image_id, image_dirs, source_path if isinstance(source_path, str) else None)
        if image_path is None:
            continue

        tile_row = int(row.get("tile_row", -1))
        tile_col = int(row.get("tile_col", -1))
        if tile_row < 0 or tile_col < 0:
            continue

        tile = extract_tile(image_path, tile_row=tile_row, tile_col=tile_col, patch_size=224)
        if tile is None:
            continue

        tile.save(cluster_dir / f"tile_{out_i:03d}.jpg", quality=95)
        tiles.append(tile)

    if not tiles:
        return None, 0

    n_tiles = len(tiles)
    n_cols = int(math.ceil(math.sqrt(n_tiles)))
    n_rows = int(math.ceil(n_tiles / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.2, n_rows * 2.2))

    axes_arr = np.atleast_1d(axes).reshape(-1)
    for i, ax in enumerate(axes_arr):
        if i < n_tiles:
            ax.imshow(tiles[i])
        ax.axis("off")

    fig.suptitle(f"Cluster {cluster_id:02d} (n={len(cluster_indices)})", fontsize=12)
    fig.tight_layout()
    grid_path = cluster_dir / "grid.png"
    fig.savefig(grid_path, dpi=180)
    plt.close(fig)
    return grid_path, n_tiles


def main() -> None:
    start_time = time.time()
    args = parse_args()

    samples_per_cluster = max(1, min(args.samples_per_cluster, 50))
    if args.samples_per_cluster > 50:
        print("[warn] --samples-per-cluster capped to 50 to limit disk usage.")

    embeddings_path = Path(args.embeddings)
    mola_features_path = Path(args.mola_features)
    metadata_path = Path(args.tile_metadata)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_dirs = [Path(p.strip()) for p in args.image_dirs.split(",") if p.strip()]

    print("[info] Loading inputs...")
    dino_emb = np.load(embeddings_path)
    mola_feat = np.load(mola_features_path)
    metadata = pd.read_csv(metadata_path)

    if dino_emb.ndim != 2 or mola_feat.ndim != 2:
        raise ValueError("Embeddings and MOLA features must be 2D arrays.")

    n_rows = len(metadata)
    if not (dino_emb.shape[0] == mola_feat.shape[0] == n_rows):
        raise ValueError(
            "Row mismatch: embeddings (%d), mola_features (%d), metadata (%d)."
            % (dino_emb.shape[0], mola_feat.shape[0], n_rows)
        )

    metadata = metadata.copy()
    class_series = metadata["class_label"] if "class_label" in metadata.columns else ["UNLABELED"] * n_rows
    metadata["class_label"] = sanitize_class_labels(class_series)

    print("[info] Standardizing DINO and MOLA features...")
    dino_norm = StandardScaler().fit_transform(dino_emb)
    mola_norm = StandardScaler().fit_transform(mola_feat)

    dino_for_cluster = maybe_reduce_dino(dino_norm, args)
    combined = np.hstack([dino_for_cluster, mola_norm * float(args.mola_weight)])

    print(
        f"[info] Clustering with MiniBatchKMeans: n_clusters={args.n_clusters}, feature_dim={combined.shape[1]}"
    )
    kmeans = MiniBatchKMeans(
        n_clusters=args.n_clusters,
        batch_size=4096,
        random_state=42,
    )
    cluster_labels = kmeans.fit_predict(combined)
    metadata["cluster_id"] = cluster_labels

    assignments_path = output_dir / "cluster_assignments.csv"
    metadata.to_csv(assignments_path, index=False)
    print(f"[info] Saved cluster assignments: {assignments_path}")

    global_fractions: Dict[str, float] = {}
    class_counts = metadata["class_label"].value_counts()
    total_tiles = float(len(metadata))
    for cls in CLASS_ORDER:
        raw_count = class_counts.get(cls, 0)
        if raw_count is None:
            cls_count = 0.0
        else:
            try:
                cls_count = float(raw_count)
            except (TypeError, ValueError):
                cls_count = 0.0
        global_fractions[cls] = float(cls_count) / total_tiles if total_tiles else 0.0

    rng = np.random.default_rng(42)
    cluster_summaries: List[Dict[str, Any]] = []
    all_indices = np.arange(len(metadata))

    for cluster_id in range(args.n_clusters):
        mask = cluster_labels == cluster_id
        cluster_indices = all_indices[mask]
        cluster_df = metadata.loc[mask]
        n_cluster = int(np.count_nonzero(mask))

        class_fracs: Dict[str, float] = {}
        if n_cluster > 0:
            cluster_classes = cluster_df["class_label"].value_counts()
            for cls in CLASS_ORDER:
                class_fracs[cls] = float(cluster_classes.get(cls, 0)) / float(n_cluster)
            dominant_class = max(class_fracs.items(), key=lambda item: item[1])[0]
            global_frac = global_fractions.get(dominant_class, 0.0)
            enrichment_score = (
                class_fracs[dominant_class] / global_frac if global_frac > 0 else 0.0
            )
        else:
            class_fracs = {cls: 0.0 for cls in CLASS_ORDER}
            dominant_class = "UNLABELED"
            enrichment_score = 0.0

        if "lat" in cluster_df.columns:
            mean_lat = safe_series_mean(cluster_df["lat"])
        else:
            mean_lat = float("nan")

        if "lon" in cluster_df.columns:
            mean_lon = safe_series_mean(cluster_df["lon"])
        else:
            mean_lon = float("nan")

        if "elevation" in cluster_df.columns:
            mean_elevation = safe_series_mean(cluster_df["elevation"])
        else:
            mean_elevation = float(mola_feat[mask, 0].mean()) if n_cluster > 0 else float("nan")

        if "slope" in cluster_df.columns:
            mean_slope = safe_series_mean(cluster_df["slope"])
        else:
            mean_slope = float(mola_feat[mask, 1].mean()) if n_cluster > 0 and mola_feat.shape[1] > 1 else float("nan")

        grid_path, n_tiles_saved = save_cluster_grid(
            cluster_id=cluster_id,
            cluster_indices=cluster_indices,
            metadata=metadata,
            image_dirs=image_dirs,
            output_root=output_dir,
            samples_per_cluster=samples_per_cluster,
            rng=rng,
        )
        if grid_path is None:
            rel_grid_path = None
        else:
            rel_grid_path = str(grid_path.relative_to(output_dir.parent))

        cluster_summaries.append(
            {
                "id": cluster_id,
                "n_tiles": n_cluster,
                "class_enrichment": class_fracs,
                "dominant_class": dominant_class,
                "enrichment_score": float(enrichment_score),
                "mean_lat": mean_lat,
                "mean_lon": mean_lon,
                "mean_elevation": mean_elevation,
                "mean_slope": mean_slope,
                "grid_path": rel_grid_path,
                "n_tiles_saved": n_tiles_saved,
            }
        )

    summary = {
        "n_clusters": int(args.n_clusters),
        "n_tiles": int(len(metadata)),
        "mola_weight": float(args.mola_weight),
        "use_pca": not args.no_pca,
        "pca_components": int(min(args.pca_components, dino_norm.shape[0], dino_norm.shape[1]))
        if not args.no_pca
        else None,
        "clusters": cluster_summaries,
    }

    summary_path = output_dir / "cluster_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[info] Saved cluster summary: {summary_path}")

    build_overview_scatter(
        combined_features=combined,
        cluster_labels=cluster_labels,
        class_labels=metadata["class_label"].to_numpy(),
        output_dir=output_dir,
        skip_umap=args.skip_umap,
    )

    print("\n=== Cluster Enrichment Report ===")
    for target_class in ["LDA", "CCF", "LVF", "GLF"]:
        enriched_rows: List[Tuple[int, float, int]] = []
        global_frac = global_fractions.get(target_class, 0.0)
        for cs in cluster_summaries:
            class_enrichment = cs.get("class_enrichment", {})
            frac = float(class_enrichment.get(target_class, 0.0))
            score = frac / global_frac if global_frac > 0 else 0.0
            enriched_rows.append((int(cs["id"]), score, int(cs["n_tiles"])))
        enriched_rows.sort(key=lambda x: x[1], reverse=True)
        top5 = enriched_rows[:5]
        top5_str = ", ".join(
            [f"C{cid:02d} score={score:.2f} n={n}" for cid, score, n in top5]
        )
        print(f"Top 5 {target_class} enriched clusters: {top5_str}")

    print("\nCluster sizes:")
    cluster_sizes = sorted(
        [(int(cs["id"]), int(cs["n_tiles"])) for cs in cluster_summaries],
        key=lambda x: x[1],
        reverse=True,
    )
    for cid, size in cluster_sizes:
        print(f"  Cluster {cid:02d}: {size}")

    elapsed = time.time() - start_time
    print(f"\n[done] Processing time: {elapsed:.2f}s")
    print(f"[done] Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
