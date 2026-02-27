#!/usr/bin/env python3

import argparse
import json
import math
import time
from collections import defaultdict
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, QhullError
from sklearn.cluster import DBSCAN


CLASS_ORDER = ["LDA", "CCF", "LVF", "GLF"]
MARS_RADIUS_KM = 3389.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert tile-level landform predictions into class-wise GeoJSON polygons."
    )
    parser.add_argument(
        "--predictions-dir",
        default="Data/HiRISE/pipeline_output/predictions",
        help="Directory containing per-image prediction CSVs, or a single prediction CSV path.",
    )
    parser.add_argument(
        "--tile-metadata",
        default="Data/HiRISE/pipeline_output/tile_metadata.csv",
        help="Optional tile metadata CSV used to fill lat/lon when missing.",
    )
    parser.add_argument(
        "--output-dir",
        default="Data/HiRISE/pipeline_output/geojson",
        help="Output directory for per-class and combined GeoJSON files.",
    )
    parser.add_argument(
        "--min-tiles",
        type=int,
        default=3,
        help="Minimum number of clustered tiles required to export a polygon.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum tile confidence for class filtering.",
    )
    parser.add_argument(
        "--method",
        choices=["convex_hull", "bbox"],
        default="convex_hull",
        help="Polygon generation method from clustered tile centers.",
    )
    return parser.parse_args()


def resolve_prediction_files(predictions_dir: Path) -> list[Path]:
    if predictions_dir.is_file():
        return [predictions_dir]

    csv_files = sorted(Path(p) for p in glob(str(predictions_dir / "*.csv")))
    if csv_files:
        return csv_files

    fused_path = predictions_dir.parent / "fusion" / "fused_predictions.csv"
    if fused_path.exists():
        return [fused_path]

    return []


def normalize_prediction_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "pred_class" not in df.columns:
        raise ValueError("Prediction CSV missing required column: pred_class")
    if "confidence" not in df.columns:
        raise ValueError("Prediction CSV missing required column: confidence")
    for col in ["image_id", "tile_row", "tile_col"]:
        if col not in df.columns:
            raise ValueError(f"Prediction CSV missing required column: {col}")

    out = df.copy()
    out["pred_class"] = out["pred_class"].astype(str).str.upper()
    out["confidence"] = pd.to_numeric(out["confidence"], errors="coerce")
    out["tile_row"] = pd.to_numeric(out["tile_row"], errors="coerce")
    out["tile_col"] = pd.to_numeric(out["tile_col"], errors="coerce")
    if "lat" in out.columns:
        out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
    if "lon" in out.columns:
        out["lon"] = pd.to_numeric(out["lon"], errors="coerce")
    return out


def load_predictions(predictions_dir: Path) -> pd.DataFrame:
    files = resolve_prediction_files(predictions_dir)
    if not files:
        raise FileNotFoundError(
            f"No prediction CSV files found in {predictions_dir} and no fused_predictions.csv fallback found."
        )

    frames = []
    for path in files:
        frame = pd.read_csv(path)
        frame = normalize_prediction_columns(frame)
        frame["_source_csv"] = path.name
        frames.append(frame)

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.dropna(subset=["tile_row", "tile_col", "confidence"])
    merged["tile_row"] = merged["tile_row"].astype(int)
    merged["tile_col"] = merged["tile_col"].astype(int)
    return merged


def merge_tile_metadata(pred_df: pd.DataFrame, tile_metadata_path: Path) -> pd.DataFrame:
    need_coords = "lat" not in pred_df.columns or "lon" not in pred_df.columns
    if not need_coords:
        lat_ok = bool(np.all(np.asarray(pred_df["lat"].notna(), dtype=bool)))
        lon_ok = bool(np.all(np.asarray(pred_df["lon"].notna(), dtype=bool)))
        if lat_ok and lon_ok:
            return pred_df
    if not tile_metadata_path.exists():
        return pred_df

    metadata = pd.read_csv(tile_metadata_path)
    required_cols = ["image_id", "tile_row", "tile_col", "lat", "lon"]
    missing = set(required_cols).difference(metadata.columns)
    if missing:
        raise ValueError(f"Tile metadata missing required columns: {sorted(missing)}")

    metadata = metadata[required_cols].copy()
    metadata["tile_row"] = pd.to_numeric(metadata["tile_row"], errors="coerce")
    metadata["tile_col"] = pd.to_numeric(metadata["tile_col"], errors="coerce")
    metadata["lat"] = pd.to_numeric(metadata["lat"], errors="coerce")
    metadata["lon"] = pd.to_numeric(metadata["lon"], errors="coerce")
    valid_mask = (
        np.isfinite(np.asarray(metadata["tile_row"], dtype=float))
        & np.isfinite(np.asarray(metadata["tile_col"], dtype=float))
        & np.isfinite(np.asarray(metadata["lat"], dtype=float))
        & np.isfinite(np.asarray(metadata["lon"], dtype=float))
    )
    metadata = pd.DataFrame(metadata[valid_mask]).copy()
    metadata["tile_row"] = metadata["tile_row"].astype(int)
    metadata["tile_col"] = metadata["tile_col"].astype(int)

    joined = pred_df.merge(
        metadata,
        on=["image_id", "tile_row", "tile_col"],
        how="left",
        suffixes=("", "_meta"),
    )

    if "lat" not in joined.columns:
        joined["lat"] = joined["lat_meta"]
    else:
        joined["lat"] = joined["lat"].where(joined["lat"].notna(), joined["lat_meta"])

    if "lon" not in joined.columns:
        joined["lon"] = joined["lon_meta"]
    else:
        joined["lon"] = joined["lon"].where(joined["lon"].notna(), joined["lon_meta"])

    drop_cols = [col for col in ["lat_meta", "lon_meta"] if col in joined.columns]
    if drop_cols:
        joined = joined.drop(columns=drop_cols)
    return joined


def estimate_dbscan_eps_deg(df: pd.DataFrame) -> float:
    values = []
    if "lat" in df.columns:
        lat_values = np.asarray(df["lat"], dtype=float)
        lat_values = lat_values[np.isfinite(lat_values)]
        unique_lat = np.unique(lat_values)
        if unique_lat.size > 1:
            lat_steps = np.abs(np.diff(unique_lat))
            lat_steps = lat_steps[lat_steps > 0]
            if lat_steps.size > 0:
                values.append(float(np.median(lat_steps)))
    if "lon" in df.columns:
        lon_values = np.asarray(df["lon"], dtype=float)
        lon_values = lon_values[np.isfinite(lon_values)]
        unique_lon = np.unique(lon_values)
        if unique_lon.size > 1:
            lon_steps = np.abs(np.diff(unique_lon))
            lon_steps = lon_steps[lon_steps > 0]
            if lon_steps.size > 0:
                values.append(float(np.median(lon_steps)))

    if not values:
        return 0.05
    est_spacing = float(np.median(np.asarray(values, dtype=float)))
    eps = max(0.05, est_spacing * 4.0)
    return float(min(max(eps, 0.01), 0.2))


def polygon_from_points(points_lon_lat: np.ndarray, method: str) -> list[list[float]]:
    lons = points_lon_lat[:, 0]
    lats = points_lon_lat[:, 1]

    if method == "bbox":
        min_lon = float(np.min(lons))
        max_lon = float(np.max(lons))
        min_lat = float(np.min(lats))
        max_lat = float(np.max(lats))
        ring = [
            [min_lon, min_lat],
            [max_lon, min_lat],
            [max_lon, max_lat],
            [min_lon, max_lat],
            [min_lon, min_lat],
        ]
        return ring

    if points_lon_lat.shape[0] < 3:
        return polygon_from_points(points_lon_lat, "bbox")

    try:
        hull = ConvexHull(points_lon_lat)
        hull_points = points_lon_lat[hull.vertices]
        ring = [[float(p[0]), float(p[1])] for p in hull_points]
    except QhullError:
        return polygon_from_points(points_lon_lat, "bbox")

    if len(ring) < 3:
        return polygon_from_points(points_lon_lat, "bbox")
    ring.append(ring[0])
    return ring


def polygon_area_km2(ring_lon_lat: list[list[float]]) -> float:
    if len(ring_lon_lat) < 4:
        return 0.0

    coords = np.asarray(ring_lon_lat, dtype=float)
    core = coords[:-1]
    centroid_lon = float(np.mean(core[:, 0]))
    centroid_lat = float(np.mean(core[:, 1]))

    m_per_deg_lat = (2.0 * math.pi * MARS_RADIUS_KM * 1000.0) / 360.0
    m_per_deg_lon = m_per_deg_lat * max(abs(math.cos(math.radians(centroid_lat))), 1e-8)

    x = (coords[:, 0] - centroid_lon) * m_per_deg_lon
    y = (coords[:, 1] - centroid_lat) * m_per_deg_lat
    area_m2 = 0.5 * abs(np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:]))
    return float(area_m2 / 1_000_000.0)


def build_features_for_class(
    data: pd.DataFrame,
    class_name: str,
    confidence_threshold: float,
    min_tiles: int,
    method: str,
    eps_deg: float,
) -> list[dict[str, object]]:
    class_df = data[
        (data["pred_class"] == class_name)
        & (data["confidence"] >= confidence_threshold)
        & (data["lat"].notna())
        & (data["lon"].notna())
    ].copy()

    if class_df.empty:
        return []

    coords_lat_lon = np.asarray(class_df[["lat", "lon"]], dtype=float)
    labels = DBSCAN(eps=eps_deg, min_samples=1).fit_predict(coords_lat_lon)
    class_df["cluster_id"] = labels

    features = []
    for _, cluster_df in class_df.groupby("cluster_id"):
        n_tiles = int(cluster_df.shape[0])
        if n_tiles < min_tiles:
            continue

        points_lon_lat = np.asarray(cluster_df[["lon", "lat"]], dtype=float)
        ring = polygon_from_points(points_lon_lat, method=method)
        area_km2 = polygon_area_km2(ring)
        mean_conf = float(np.mean(np.asarray(cluster_df["confidence"], dtype=float)))

        if mean_conf < confidence_threshold:
            continue

        centroid_lat = float(np.mean(np.asarray(cluster_df["lat"], dtype=float)))
        centroid_lon = float(np.mean(np.asarray(cluster_df["lon"], dtype=float)))
        image_ids = sorted(np.unique(np.asarray(cluster_df["image_id"], dtype=str)).tolist())

        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [ring],
            },
            "properties": {
                "class": class_name,
                "n_tiles": n_tiles,
                "mean_confidence": round(mean_conf, 6),
                "area_km2": round(area_km2, 6),
                "centroid_lat": round(centroid_lat, 6),
                "centroid_lon": round(centroid_lon, 6),
                "image_ids": image_ids,
            },
        }
        features.append(feature)

    return features


def write_geojson(path: Path, features: list[dict[str, object]]) -> None:
    content = {
        "type": "FeatureCollection",
        "features": features,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(content, f, indent=2)


def main() -> None:
    args = parse_args()
    start = time.time()

    predictions_dir = Path(args.predictions_dir)
    tile_metadata_path = Path(args.tile_metadata)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = load_predictions(predictions_dir)
    predictions = merge_tile_metadata(predictions, tile_metadata_path)
    if "lat" not in predictions.columns or "lon" not in predictions.columns:
        raise ValueError(
            "Predictions do not contain lat/lon and could not be enriched from tile metadata."
        )

    eps_deg = estimate_dbscan_eps_deg(predictions)
    print(f"[info] Loaded {len(predictions)} predictions")
    print(f"[info] DBSCAN eps={eps_deg:.5f} degrees (Mars areocentric lon/lat)")
    print(f"[info] Polygon method={args.method}")

    features_by_class = defaultdict(list)
    combined_features = []

    for class_name in CLASS_ORDER:
        class_features = build_features_for_class(
            data=predictions,
            class_name=class_name,
            confidence_threshold=float(args.confidence_threshold),
            min_tiles=int(args.min_tiles),
            method=args.method,
            eps_deg=eps_deg,
        )
        features_by_class[class_name] = class_features
        combined_features.extend(class_features)

        class_out = output_dir / f"landforms_{class_name}.geojson"
        write_geojson(class_out, class_features)

    write_geojson(output_dir / "landforms_all.geojson", combined_features)

    print("\n=== GeoJSON Export Summary ===")
    total_area = 0.0
    for class_name in CLASS_ORDER:
        class_features = features_by_class[class_name]
        class_area = float(np.sum([float(feature["properties"]["area_km2"]) for feature in class_features]))
        total_area += class_area
        print(f"{class_name}: {len(class_features)} polygons, area={class_area:.3f} km^2")
    print(f"TOTAL: {len(combined_features)} polygons, area={total_area:.3f} km^2")
    print(f"[done] Outputs written to: {output_dir}")
    print(f"[done] Elapsed: {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()
