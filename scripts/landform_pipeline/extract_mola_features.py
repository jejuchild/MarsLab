#!/usr/bin/env python3
# pyright: basic
"""Extract MOLA DEM geomorphometric features for HiRISE tiles."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import rowcol
from rasterio.windows import Window
from tqdm import tqdm


MARS_MEAN_RADIUS_M = 3_389_500.0
EPS_COS_LAT = 1e-4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute MOLA geomorphometric features for each tile."
    )
    _ = parser.add_argument(
        "--tile-metadata",
        default="Data/HiRISE/pipeline_output/tile_metadata.csv",
        help="Path to tile_metadata.csv",
    )
    _ = parser.add_argument(
        "--dem-path",
        default="Mars_HRSC_MOLA_BlendDEM_Global_200mp_v2.tif",
        help="Path to global MOLA DEM GeoTIFF",
    )
    _ = parser.add_argument(
        "--output-dir",
        default="Data/HiRISE/pipeline_output",
        help="Directory to save outputs",
    )
    _ = parser.add_argument(
        "--window-radii",
        default="1,5,20",
        help="Comma-separated radii in km (e.g., '1,5,20')",
    )
    return parser.parse_args()


def parse_window_radii_km(raw: str) -> list[float]:
    radii = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not radii:
        raise ValueError("--window-radii must contain at least one radius.")
    if any(r <= 0 for r in radii):
        raise ValueError("All radii in --window-radii must be positive.")
    return sorted(radii)


def feature_names_for_radii(radii_km: list[float]) -> list[str]:
    names: list[str] = []
    per_window = [
        "slope_mean",
        "slope_std",
        "curvature_mean",
        "TPI",
        "TRI",
        "roughness",
        "lobateness",
    ]
    for radius in radii_km:
        r_label = str(int(radius)) if float(radius).is_integer() else str(radius).replace(".", "p")
        for fname in per_window:
            names.append(f"{fname}_r{r_label}km")
    smallest = radii_km[0]
    smallest_label = (
        str(int(smallest)) if float(smallest).is_integer() else str(smallest).replace(".", "p")
    )
    names.append(f"elevation_mean_r{smallest_label}km")
    names.append("abs_latitude")
    return names


def read_tile_metadata(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, list[int]]]:
    image_ids: list[str] = []
    lats: list[float] = []
    lons: list[float] = []
    grouped_indices: dict[str, list[int]] = defaultdict(list)

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"image_id", "lat", "lon"}
        if not required.issubset(set(reader.fieldnames or [])):
            missing = sorted(required - set(reader.fieldnames or []))
            raise ValueError(f"tile_metadata.csv missing required columns: {missing}")

        for idx, row in enumerate(reader):
            image_id = row["image_id"]
            lat = float(row["lat"])
            lon = float(row["lon"])
            image_ids.append(image_id)
            lats.append(lat)
            lons.append(lon)
            grouped_indices[image_id].append(idx)

    if not lats:
        raise ValueError("tile_metadata.csv is empty.")

    return (
        np.asarray(image_ids, dtype=object),
        np.asarray(lats, dtype=np.float64),
        np.asarray(lons, dtype=np.float64),
        grouped_indices,
    )


def meters_per_degree() -> float:
    return (math.pi / 180.0) * MARS_MEAN_RADIUS_M


def compute_window_features(
    elev: np.ndarray,
    center_r: int,
    center_c: int,
    px_m_ns: float,
    px_m_ew: float,
) -> tuple[float, float, float, float, float, float, float, float]:
    if elev.size == 0:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    center_elev = float(elev[center_r, center_c])

    if not np.isfinite(px_m_ew) or px_m_ew <= 0.0:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    dz_dy, dz_dx = np.gradient(elev, px_m_ns, px_m_ew)
    slope = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))

    d2z_dy2 = np.gradient(dz_dy, px_m_ns, axis=0)
    d2z_dx2 = np.gradient(dz_dx, px_m_ew, axis=1)
    curvature = -(d2z_dx2 + d2z_dy2)

    slope_mean = float(np.nanmean(slope))
    slope_std = float(np.nanstd(slope))
    curvature_mean = float(np.nanmean(curvature))

    elev_mean = float(np.nanmean(elev))
    tpi = center_elev - elev_mean
    tri = float(np.sqrt(np.nanmean((elev - center_elev) ** 2)))
    roughness = float(np.nanmax(elev) - np.nanmin(elev))

    aspect = np.arctan2(dz_dy, dz_dx)
    sin_a = np.sin(aspect)
    cos_a = np.cos(aspect)
    resultant_length = float(
        np.sqrt(np.nanmean(sin_a) ** 2 + np.nanmean(cos_a) ** 2)
    )
    lobateness = 1.0 - resultant_length

    return slope_mean, slope_std, curvature_mean, tpi, tri, roughness, lobateness, elev_mean


def main() -> None:
    args = parse_args()
    t0 = time.time()

    tile_metadata_path = Path(args.tile_metadata)
    dem_path = Path(args.dem_path)
    output_dir = Path(args.output_dir)
    radii_km = parse_window_radii_km(args.window_radii)
    radii_m = np.asarray(radii_km, dtype=np.float64) * 1000.0
    max_radius_m = float(np.max(radii_m))

    output_dir.mkdir(parents=True, exist_ok=True)

    _, lats, lons, grouped_indices = read_tile_metadata(tile_metadata_path)
    n_tiles = lats.shape[0]

    feature_names = feature_names_for_radii(radii_km)
    n_features = len(feature_names)
    features = np.zeros((n_tiles, n_features), dtype=np.float32)

    deg_to_m = meters_per_degree()

    with rasterio.open(dem_path) as ds:
        px_deg_ns = abs(ds.transform.e)
        px_deg_ew = abs(ds.transform.a)
        px_m_ns = px_deg_ns * deg_to_m

        for _image_id, idx_list in tqdm(
            grouped_indices.items(),
            total=len(grouped_indices),
            desc="Processing image groups",
        ):
            idx_arr = np.asarray(idx_list, dtype=np.int64)
            lat_group = lats[idx_arr]
            lon_group = lons[idx_arr]

            rows, cols = rowcol(ds.transform, lon_group, lat_group)
            rows = np.asarray(rows, dtype=np.int64)
            cols = np.asarray(cols, dtype=np.int64)

            cos_lat = np.cos(np.radians(lat_group))
            cos_lat = np.where(np.abs(cos_lat) < EPS_COS_LAT, np.sign(cos_lat) * EPS_COS_LAT, cos_lat)
            cos_lat = np.where(cos_lat == 0.0, EPS_COS_LAT, cos_lat)
            px_m_ew_group = px_deg_ew * deg_to_m * np.abs(cos_lat)

            half_rows_max = int(max(1, math.ceil(max_radius_m / px_m_ns)))
            half_cols_max = np.maximum(
                1,
                np.ceil(max_radius_m / np.maximum(px_m_ew_group, 1e-6)).astype(np.int64),
            )

            row_min = int(np.min(rows - half_rows_max))
            row_max = int(np.max(rows + half_rows_max))
            col_min = int(np.min(cols - half_cols_max))
            col_max = int(np.max(cols + half_cols_max))

            win_height = int(row_max - row_min + 1)
            win_width = int(col_max - col_min + 1)

            window = Window(col_min, row_min, win_width, win_height)
            dem_patch_masked = ds.read(1, window=window, boundless=True, masked=True)
            dem_patch = dem_patch_masked.astype(np.float32).filled(np.nan)

            local_rows = rows - row_min
            local_cols = cols - col_min

            for local_i, tile_idx in enumerate(idx_arr):
                lat = float(lat_group[local_i])
                px_m_ew = float(px_m_ew_group[local_i])
                c_row = int(local_rows[local_i])
                c_col = int(local_cols[local_i])

                tile_features: list[float] = []
                smallest_elev = 0.0

                for radius_m in radii_m:
                    half_rows = int(max(1, math.ceil(radius_m / px_m_ns)))
                    half_cols = int(max(1, math.ceil(radius_m / max(px_m_ew, 1e-6))))

                    r0 = c_row - half_rows
                    r1 = c_row + half_rows + 1
                    c0 = c_col - half_cols
                    c1 = c_col + half_cols + 1

                    elev = dem_patch[r0:r1, c0:c1]

                    if elev.shape[0] < 3 or elev.shape[1] < 3:
                        fvals = (0.0,) * 8
                    else:
                        center_r = c_row - r0
                        center_c = c_col - c0
                        fvals = compute_window_features(
                            elev=elev,
                            center_r=center_r,
                            center_c=center_c,
                            px_m_ns=px_m_ns,
                            px_m_ew=px_m_ew,
                        )

                    tile_features.extend(fvals[:7])
                    if radius_m == radii_m[0]:
                        smallest_elev = float(fvals[7])

                tile_features.append(smallest_elev)
                tile_features.append(abs(lat))

                tile_feature_arr = np.asarray(tile_features, dtype=np.float32)
                tile_feature_arr[~np.isfinite(tile_feature_arr)] = 0.0
                features[tile_idx] = tile_feature_arr

    feature_path = output_dir / "mola_features.npy"
    names_path = output_dir / "mola_feature_names.json"

    np.save(feature_path, features.astype(np.float32))
    with names_path.open("w") as f:
        json.dump(feature_names, f, indent=2)

    elapsed = time.time() - t0

    print(f"Saved features: {feature_path}")
    print(f"Saved feature names: {names_path}")
    print(f"Feature matrix shape: {features.shape}")
    print(f"Processing time: {elapsed:.2f} s")
    print("Feature statistics (min/mean/max/std):")
    for i, name in enumerate(feature_names):
        col = features[:, i]
        print(
            f"  {name}: {np.min(col):.4f} / {np.mean(col):.4f} / {np.max(col):.4f} / {np.std(col):.4f}"
        )


if __name__ == "__main__":
    main()
