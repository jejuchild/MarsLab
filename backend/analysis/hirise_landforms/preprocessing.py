from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any

import numpy as np
import requests
from PIL import Image

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = ROOT / "backend" / "data" / "hirise_landforms"
BROWSE_CACHE_DIR = DATA_ROOT / "cache" / "browse"
LEGACY_BROWSE_DIR = ROOT / "Data" / "HiRISE" / "midlat_browse"
MOLA_DEM_PATH = ROOT / "Mars_HRSC_MOLA_BlendDEM_Global_200mp_v2.tif"
MOLA_FEATURE_DIM = 23

# Lazy-loaded MOLA DEM dataset (rasterio handle — opened once, reused)
_mola_ds: Any = None
_mola_load_attempted: bool = False


def _candidate_local_paths(product_id: str) -> list[Path]:
    lowered = product_id.lower()
    names = [
        product_id,
        product_id.upper(),
        product_id.lower(),
        f"{product_id}_browse",
        f"{product_id}_RED",
    ]
    extensions = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
    out: list[Path] = []
    for base in (BROWSE_CACHE_DIR, LEGACY_BROWSE_DIR):
        for name in names:
            for ext in extensions:
                out.append(base / f"{name}{ext}")
        if base.exists():
            for path in base.glob(f"*{lowered}*"):
                out.append(path)
            for path in base.glob(f"*{product_id.upper()}*"):
                out.append(path)
    return out


def _candidate_urls(product_id: str) -> list[str]:
    return [
        f"https://www.uahirise.org/jpeg/{product_id}",
        f"https://www.uahirise.org/images/{product_id}.jpg",
    ]


def fetch_hirise_browse(product_id: str) -> Image.Image:
    BROWSE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    for path in _candidate_local_paths(product_id):
        if path.is_file():
            with Image.open(path) as img:
                return img.convert("RGB")

    errors: list[str] = []
    for url in _candidate_urls(product_id):
        try:
            response = requests.get(url, timeout=20)
            if response.status_code != 200:
                errors.append(f"{url}: HTTP {response.status_code}")
                continue
            with Image.open(io.BytesIO(response.content)) as img:
                rgb = img.convert("RGB")
                cache_path = BROWSE_CACHE_DIR / f"{product_id}.jpg"
                rgb.save(cache_path, format="JPEG", quality=95)
                return rgb
        except Exception as exc:
            errors.append(f"{url}: {exc}")

    error_text = "; ".join(errors) if errors else "no candidate URLs available"
    raise FileNotFoundError(f"Could not fetch HiRISE browse image for '{product_id}'. {error_text}")


def tile_image(
    image: Image.Image,
    tile_size: int = 224,
    min_content: float = 0.3,
) -> list[tuple[int, int, Image.Image]]:
    """
    Extract non-overlapping 224×224 tiles from a browse image.
    Filters out tiles with >70% black pixels (edges of browse images).
    Returns list of (grid_x, grid_y, PIL.Image) tuples.
    """
    if tile_size <= 0:
        raise ValueError("tile_size must be > 0")

    rgb = image.convert("RGB")
    width, height = rgb.size
    tiles: list[tuple[int, int, Image.Image]] = []
    arr = np.array(rgb)

    for y0 in range(0, height - tile_size + 1, tile_size):
        for x0 in range(0, width - tile_size + 1, tile_size):
            tile_arr = arr[y0 : y0 + tile_size, x0 : x0 + tile_size]

            # Skip mostly-black tiles (edges of browse images)
            content_frac = float(np.mean(tile_arr > 10))
            if content_frac < min_content:
                continue

            crop = Image.fromarray(tile_arr)
            tiles.append((x0 // tile_size, y0 // tile_size, crop))

    if not tiles:
        resized = rgb.resize((tile_size, tile_size), Image.Resampling.BICUBIC)
        tiles.append((0, 0, resized))

    return tiles


def _get_mola_ds() -> Any:
    """Lazy-load MOLA DEM rasterio dataset."""
    global _mola_ds, _mola_load_attempted
    if _mola_ds is not None:
        return _mola_ds
    if _mola_load_attempted:
        return None

    _mola_load_attempted = True
    if not MOLA_DEM_PATH.exists():
        logger.warning("MOLA DEM not found at %s", MOLA_DEM_PATH)
        return None

    try:
        import rasterio
        _mola_ds = rasterio.open(str(MOLA_DEM_PATH))
        logger.info("Loaded MOLA DEM: %d×%d pixels", _mola_ds.width, _mola_ds.height)
        return _mola_ds
    except Exception as exc:
        logger.warning("Failed to open MOLA DEM: %s", exc)
        return None


# ── Geomorphometric feature extraction ────────────────────────────────────────
# Mirrors scripts/marslandform_v2/data/mola.py exactly.

DEM_RESOLUTION_M = 200.0  # 200m/pixel MOLA DEM


def _latlon_to_pixel(ds: Any, lat: float, lon: float) -> tuple[int, int]:
    """Convert Mars lat/lon to pixel coordinates in DEM."""
    row, col = ds.index(lon, lat)
    row = max(0, min(row, ds.height - 1))
    col = max(0, min(col, ds.width - 1))
    return int(row), int(col)


def _extract_window(ds: Any, lat: float, lon: float, radius_km: float) -> np.ndarray:
    """Extract a square window from DEM centered on (lat, lon)."""
    radius_px = max(1, int(radius_km * 1000 / DEM_RESOLUTION_M))
    row, col = _latlon_to_pixel(ds, lat, lon)

    row_min = max(0, row - radius_px)
    row_max = min(ds.height, row + radius_px + 1)
    col_min = max(0, col - radius_px)
    col_max = min(ds.width, col + radius_px + 1)

    window = ds.read(1, window=((row_min, row_max), (col_min, col_max)))
    nodata = ds.nodata
    if nodata is not None:
        window = window.astype(np.float64)
        window[window == nodata] = np.nan
    return window


def _compute_slope(elevation: np.ndarray) -> np.ndarray:
    if elevation.shape[0] < 3 or elevation.shape[1] < 3:
        return np.zeros_like(elevation)
    dy, dx = np.gradient(elevation, DEM_RESOLUTION_M)
    slope_rad = np.arctan(np.sqrt(dx ** 2 + dy ** 2))
    return np.degrees(slope_rad)


def _compute_curvature(elevation: np.ndarray) -> np.ndarray:
    if elevation.shape[0] < 3 or elevation.shape[1] < 3:
        return np.zeros_like(elevation)
    dy, dx = np.gradient(elevation, DEM_RESOLUTION_M)
    dyy, _ = np.gradient(dy, DEM_RESOLUTION_M)
    _, dxx = np.gradient(dx, DEM_RESOLUTION_M)
    return dyy + dxx


def _compute_tpi(elevation: np.ndarray, radius_px: int = 5) -> np.ndarray:
    from scipy.ndimage import uniform_filter
    kernel = 2 * radius_px + 1
    mean_elev = uniform_filter(elevation, size=kernel, mode="nearest")
    return elevation - mean_elev


def _compute_tri(elevation: np.ndarray) -> np.ndarray:
    from scipy.ndimage import generic_filter

    def _tri_func(x: np.ndarray) -> float:
        center = x[len(x) // 2]
        return float(np.nanmean(np.abs(x - center)))

    return generic_filter(elevation, _tri_func, size=3, mode="nearest")


def _compute_roughness(elevation: np.ndarray) -> np.ndarray:
    from scipy.ndimage import maximum_filter, minimum_filter
    return maximum_filter(elevation, size=3) - minimum_filter(elevation, size=3)


def _compute_lobateness(slope: np.ndarray) -> float:
    mean_slope = float(np.nanmean(slope))
    if mean_slope < 0.1:
        return 0.0
    return float(np.nanmax(slope) / mean_slope)


def _extract_features_at_scale(
    ds: Any, lat: float, lon: float, radius_km: float,
) -> dict[str, float]:
    """Extract 7 geomorphometric features at one spatial scale."""
    window = _extract_window(ds, lat, lon, radius_km)

    if window.size == 0 or np.all(np.isnan(window)):
        return {
            "slope_mean": 0.0, "slope_std": 0.0, "curvature_mean": 0.0,
            "TPI": 0.0, "TRI": 0.0, "roughness": 0.0, "lobateness": 0.0,
        }

    valid = window[~np.isnan(window)]
    median_val = float(np.median(valid)) if len(valid) > 0 else 0.0
    window_filled = np.where(np.isnan(window), median_val, window)

    slope = _compute_slope(window_filled)
    curvature = _compute_curvature(window_filled)
    tpi_radius = max(1, window.shape[0] // 4)
    tpi = _compute_tpi(window_filled, radius_px=tpi_radius)
    tri = _compute_tri(window_filled)
    roughness = _compute_roughness(window_filled)

    return {
        "slope_mean": float(np.nanmean(slope)),
        "slope_std": float(np.nanstd(slope)),
        "curvature_mean": float(np.nanmean(curvature)),
        "TPI": float(np.nanmean(tpi)),
        "TRI": float(np.nanmean(tri)),
        "roughness": float(np.nanmean(roughness)),
        "lobateness": _compute_lobateness(slope),
    }


def extract_mola_features(lat: float, lon: float) -> np.ndarray:
    """
    Extract 23-dim MOLA geomorphometric feature vector at (lat, lon).
    Returns np.ndarray of shape (23,).

    Features: 7 per scale × 3 scales (1, 5, 20 km) + 2 global
      - slope_mean, slope_std, curvature_mean, TPI, TRI, roughness, lobateness
      - elevation_mean, abs_latitude
    """
    ds = _get_mola_ds()
    if ds is None:
        logger.warning("MOLA DEM unavailable — returning zero features for lat=%s lon=%s", lat, lon)
        return np.zeros((MOLA_FEATURE_DIM,), dtype=np.float32)

    scales_km = [1.0, 5.0, 20.0]
    features: list[float] = []

    for scale in scales_km:
        try:
            scale_feats = _extract_features_at_scale(ds, lat, lon, scale)
        except Exception as exc:
            logger.warning("MOLA feature extraction failed at scale %.1f km: %s", scale, exc)
            scale_feats = {
                "slope_mean": 0.0, "slope_std": 0.0, "curvature_mean": 0.0,
                "TPI": 0.0, "TRI": 0.0, "roughness": 0.0, "lobateness": 0.0,
            }
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
    try:
        window_1km = _extract_window(ds, lat, lon, 1.0)
        elev_mean = float(np.nanmean(window_1km)) if window_1km.size > 0 else 0.0
    except Exception:
        elev_mean = 0.0
    features.append(elev_mean)
    features.append(abs(lat))

    return np.array(features, dtype=np.float32)
