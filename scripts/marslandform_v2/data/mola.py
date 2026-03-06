"""
MOLA DEM feature extraction at IMAGE-level for MarsLandformNet V2.

Extracts 23 geomorphometric features from Mars HRSC/MOLA DEM:
- 7 features (slope_mean, slope_std, curvature, TPI, TRI, roughness, lobateness) × 3 scales (1, 5, 20 km)
- 2 global features (elevation_mean, abs_latitude)

CRITICAL: Features are extracted at IMAGE-level, not tile-level.
HiRISE tiles (~56m) are smaller than 1 MOLA DEM pixel (200m),
so tile-level MOLA features are meaningless.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.marslandform_v2.config import MOLA_DEM, METADATA_JSON, V2_OUTPUT, MOLAConfig, get_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MARS_RADIUS_M = 3389500.0
DEM_RESOLUTION_M = 200.0  # 200m/pixel MOLA DEM


def load_dem(dem_path: str) -> Tuple:
    """Load MOLA DEM GeoTIFF. Returns (array, transform, crs)."""
    try:
        import rasterio
    except ImportError:
        raise ImportError("rasterio required: pip install rasterio")

    ds = rasterio.open(dem_path)
    return ds


def latlon_to_pixel(ds, lat: float, lon: float) -> Tuple[int, int]:
    """Convert Mars lat/lon to pixel coordinates in DEM."""
    # MOLA DEM uses simple cylindrical: lon in [-180, 180], lat in [-90, 90]
    row, col = ds.index(lon, lat)
    row = max(0, min(row, ds.height - 1))
    col = max(0, min(col, ds.width - 1))
    return int(row), int(col)


def extract_window(ds, lat: float, lon: float, radius_km: float) -> np.ndarray:
    """Extract a square window from DEM centered on (lat, lon) with given radius."""
    radius_px = max(1, int(radius_km * 1000 / DEM_RESOLUTION_M))
    row, col = latlon_to_pixel(ds, lat, lon)

    row_min = max(0, row - radius_px)
    row_max = min(ds.height, row + radius_px + 1)
    col_min = max(0, col - radius_px)
    col_max = min(ds.width, col + radius_px + 1)

    window = ds.read(1, window=((row_min, row_max), (col_min, col_max)))
    # Replace nodata with NaN
    nodata = ds.nodata
    if nodata is not None:
        window = window.astype(np.float64)
        window[window == nodata] = np.nan
    return window


def compute_slope(elevation: np.ndarray, cell_size_m: float = DEM_RESOLUTION_M) -> np.ndarray:
    """Compute slope in degrees using central differences."""
    if elevation.shape[0] < 3 or elevation.shape[1] < 3:
        return np.zeros_like(elevation)
    dy, dx = np.gradient(elevation, cell_size_m)
    slope_rad = np.arctan(np.sqrt(dx ** 2 + dy ** 2))
    return np.degrees(slope_rad)


def compute_curvature(elevation: np.ndarray, cell_size_m: float = DEM_RESOLUTION_M) -> np.ndarray:
    """Compute profile curvature (second derivative of slope)."""
    if elevation.shape[0] < 3 or elevation.shape[1] < 3:
        return np.zeros_like(elevation)
    dy, dx = np.gradient(elevation, cell_size_m)
    dyy, _ = np.gradient(dy, cell_size_m)
    _, dxx = np.gradient(dx, cell_size_m)
    return dyy + dxx


def compute_tpi(elevation: np.ndarray, radius_px: int = 5) -> np.ndarray:
    """Topographic Position Index: elevation minus mean of neighbors."""
    from scipy.ndimage import uniform_filter
    kernel = 2 * radius_px + 1
    mean_elev = uniform_filter(elevation, size=kernel, mode="nearest")
    return elevation - mean_elev


def compute_tri(elevation: np.ndarray) -> np.ndarray:
    """Terrain Ruggedness Index: mean absolute difference from neighbors.
    Vectorized implementation (~100x faster than scipy generic_filter).
    """
    if elevation.shape[0] < 3 or elevation.shape[1] < 3:
        return np.zeros_like(elevation)
    padded = np.pad(elevation, 1, mode='edge')
    center = elevation
    h, w = elevation.shape
    total = np.zeros_like(elevation, dtype=np.float64)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            total += np.abs(padded[1 + dr:h + 1 + dr, 1 + dc:w + 1 + dc] - center)
    return total / 9.0  # /9 to match nanmean over 3x3=9 elements (including center=0)


def compute_roughness(elevation: np.ndarray) -> np.ndarray:
    """Roughness: max elevation difference in 3×3 window."""
    from scipy.ndimage import maximum_filter, minimum_filter
    return maximum_filter(elevation, size=3) - minimum_filter(elevation, size=3)


def compute_lobateness(slope: np.ndarray) -> float:
    """
    Lobateness proxy: ratio of max to mean slope in the window.
    High lobateness suggests flow-like morphology (LDA, GLF).
    """
    mean_slope = np.nanmean(slope)
    if mean_slope < 0.1:
        return 0.0
    return float(np.nanmax(slope) / mean_slope)


def extract_features_at_scale(
    ds, lat: float, lon: float, radius_km: float
) -> Dict[str, float]:
    """Extract all 7 features at a single spatial scale."""
    window = extract_window(ds, lat, lon, radius_km)

    if window.size == 0 or np.all(np.isnan(window)):
        return {
            "slope_mean": 0.0, "slope_std": 0.0, "curvature_mean": 0.0,
            "TPI": 0.0, "TRI": 0.0, "roughness": 0.0, "lobateness": 0.0,
        }

    # Replace NaN with median for computation
    valid = window[~np.isnan(window)]
    if len(valid) == 0:
        median_val = 0.0
    else:
        median_val = float(np.median(valid))
    window_filled = np.where(np.isnan(window), median_val, window)

    slope = compute_slope(window_filled)
    curvature = compute_curvature(window_filled)

    # TPI uses half the window radius
    tpi_radius = max(1, window.shape[0] // 4)
    tpi = compute_tpi(window_filled, radius_px=tpi_radius)

    tri = compute_tri(window_filled)
    roughness = compute_roughness(window_filled)

    return {
        "slope_mean": float(np.nanmean(slope)),
        "slope_std": float(np.nanstd(slope)),
        "curvature_mean": float(np.nanmean(curvature)),
        "TPI": float(np.nanmean(tpi)),
        "TRI": float(np.nanmean(tri)),
        "roughness": float(np.nanmean(roughness)),
        "lobateness": compute_lobateness(slope),
    }


def extract_image_features(
    ds,
    lat: float,
    lon: float,
    scales_km: List[float] = None,
) -> np.ndarray:
    """
    Extract full 23-feature vector for one image location.
    Returns: np.ndarray of shape (23,)
    """
    if scales_km is None:
        scales_km = [1.0, 5.0, 20.0]

    features = []

    # Multi-scale features (7 × 3 = 21)
    for scale in scales_km:
        scale_feats = extract_features_at_scale(ds, lat, lon, scale)
        features.extend([
            scale_feats["slope_mean"],
            scale_feats["slope_std"],
            scale_feats["curvature_mean"],
            scale_feats["TPI"],
            scale_feats["TRI"],
            scale_feats["roughness"],
            scale_feats["lobateness"],
        ])

    # Global features (+2 = 23)
    # Elevation at 1km scale
    window_1km = extract_window(ds, lat, lon, 1.0)
    elev_mean = float(np.nanmean(window_1km)) if window_1km.size > 0 else 0.0
    features.append(elev_mean)

    # Absolute latitude
    features.append(abs(lat))

    return np.array(features, dtype=np.float32)


def extract_all_features(
    image_ids: List[str],
    metadata: Dict[str, dict],
    dem_path: str,
    scales_km: List[float] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract MOLA features for all images.
    Returns: (features_array [N×23], feature_names [23])
    """
    if scales_km is None:
        scales_km = [1.0, 5.0, 20.0]

    ds = load_dem(dem_path)
    logger.info(f"Loaded DEM: {ds.width}×{ds.height} pixels")

    feature_names = []
    for scale in scales_km:
        for feat in ["slope_mean", "slope_std", "curvature_mean", "TPI", "TRI", "roughness", "lobateness"]:
            feature_names.append(f"{feat}_{scale}km")
    feature_names.extend(["elevation_mean", "abs_latitude"])

    all_features = []
    valid_ids = []
    skipped = 0

    for i, img_id in enumerate(image_ids):
        if img_id not in metadata:
            skipped += 1
            continue

        meta = metadata[img_id]
        lat = meta.get("lat")
        lon = meta.get("lon")
        if lat is None or lon is None:
            skipped += 1
            continue

        try:
            feats = extract_image_features(ds, float(lat), float(lon), scales_km)
            all_features.append(feats)
            valid_ids.append(img_id)
        except Exception as e:
            logger.warning(f"Failed to extract features for {img_id}: {e}")
            skipped += 1

        if (i + 1) % 100 == 0:
            logger.info(f"  Extracted {i + 1}/{len(image_ids)} ({skipped} skipped)")

    ds.close()

    features_array = np.stack(all_features) if all_features else np.empty((0, 23))
    logger.info(f"Extracted {features_array.shape[0]} feature vectors, {skipped} skipped")
    return features_array, feature_names, valid_ids


def main():
    parser = argparse.ArgumentParser(description="Extract MOLA DEM features at image-level")
    parser.add_argument("--labels", type=str,
                       default=str(V2_OUTPUT / "unified_labels.json"))
    parser.add_argument("--dem", type=str, default=str(MOLA_DEM))
    parser.add_argument("--output-dir", type=str, default=str(V2_OUTPUT))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load labels
    with open(args.labels) as f:
        labels_data = json.load(f)
    if isinstance(labels_data, list):
        metadata = {m["image_id"]: m for m in labels_data}
    else:
        metadata = labels_data

    image_ids = list(metadata.keys())
    if args.limit:
        image_ids = image_ids[: args.limit]

    logger.info(f"Extracting MOLA features for {len(image_ids)} images")

    features, feature_names, valid_ids = extract_all_features(
        image_ids, metadata, args.dem
    )

    # Save
    np.save(output_dir / "mola_features.npy", features)
    with open(output_dir / "mola_feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)
    with open(output_dir / "mola_image_ids.json", "w") as f:
        json.dump(valid_ids, f)

    logger.info(f"Saved: mola_features.npy ({features.shape}), mola_feature_names.json, mola_image_ids.json")


if __name__ == "__main__":
    main()
