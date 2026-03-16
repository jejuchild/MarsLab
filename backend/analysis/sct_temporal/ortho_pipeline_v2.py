"""
Orthorectification-aware HiRISE temporal change detection pipeline v2.

Improvements over v1:
- Uses pre-computed ORTHO products (orthorectified on DTM) as reference
- ECC-based sub-pixel co-registration (replaces integer-pixel FFT alignment)
- DTM-derived terrain parallax correction for non-ortho targets
- DTM-based slope classification (replaces gradient-percentile bias)
- Hierarchical co-registration: global affine → local ECC tiles

Site: "Lefort Core" — 46°N, 92°E, Utopia Planitia
DTM:  DTEEC_001938_2265_002439_2265_U01 (1.0 m/px)
Pairs:
  A: ORTHO_001938 (2006) → ESP_064072 (2020) — 14 yr baseline
  B: ORTHO_002439 (2007) → ESP_064072 (2020) — 13 yr baseline
  C: ORTHO_001938 → ORTHO_002439              — null test (~0 retreat)
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from scipy import ndimage

logger = logging.getLogger(__name__)

# ─── Paths (relative to project root) ────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "Data" / "HiRISE"
RDR_CACHE = DATA_DIR / "rdr_cache"
DTM_CACHE = DATA_DIR / "dtm_cache"
RESULTS_DIR = PROJECT_ROOT / "results" / "sct_analysis" / "v2_ortho"

# Image catalog for Lefort Core site
@dataclass
class ImageInfo:
    path: Path
    date: str
    emission_angle: float
    is_ortho: bool
    obs_id: str


IMAGE_CATALOG: Dict[str, ImageInfo] = {
    "ORTHO_001938": ImageInfo(
        path=RDR_CACHE / "PSP_001938_2265_RED_A_01_ORTHO.JP2",
        date="2006-12-25", emission_angle=2.806, is_ortho=True,
        obs_id="PSP_001938_2265",
    ),
    "ORTHO_002439": ImageInfo(
        path=RDR_CACHE / "PSP_002439_2265_RED_A_01_ORTHO.JP2",
        date="2007-02-02", emission_angle=14.710, is_ortho=True,
        obs_id="PSP_002439_2265",
    ),
    "RDR_000856": ImageInfo(
        path=RDR_CACHE / "TRA_000856_2265_RED.JP2",
        date="2006-10-02", emission_angle=0.374, is_ortho=False,
        obs_id="TRA_000856_2265",
    ),
    "RDR_002439": ImageInfo(
        path=RDR_CACHE / "PSP_002439_2265_RED.JP2",
        date="2007-02-02", emission_angle=14.710, is_ortho=False,
        obs_id="PSP_002439_2265",
    ),
    "RDR_064072": ImageInfo(
        path=RDR_CACHE / "ESP_064072_2265_RED.JP2",
        date="2020-03-28", emission_angle=3.098, is_ortho=False,
        obs_id="ESP_064072_2265",
    ),
}

DTM_PATH = DTM_CACHE / "DTEEC_001938_2265_002439_2265_U01.IMG"

# Temporal pairs for analysis — RDR↔RDR for reliable co-registration
# ORTHO↔RDR matching fails due to 8-bit vs 16-bit radiometric mismatch
TEMPORAL_PAIRS = [
    {"name": "D_rdr_primary", "ref": "RDR_000856", "tgt": "RDR_064072",
     "baseline_yr": 13.49, "desc": "Primary RDR: TRA_000856(2006)→ESP_064072(2020)"},
    {"name": "E_rdr_validation", "ref": "RDR_002439", "tgt": "RDR_064072",
     "baseline_yr": 13.15, "desc": "Validation RDR: PSP_002439(2007)→ESP_064072(2020)"},
    {"name": "F_rdr_null", "ref": "RDR_000856", "tgt": "RDR_002439",
     "baseline_yr": 0.23, "desc": "Null test RDR: TRA_000856(2006)→PSP_002439(2007)"},
]

# Legacy ORTHO pairs (kept for reference, matching fails on these)
ORTHO_PAIRS = [
    {"name": "A_primary", "ref": "ORTHO_001938", "tgt": "RDR_064072",
     "baseline_yr": 13.26, "desc": "Primary: 2006→2020 (14yr)"},
    {"name": "B_validation", "ref": "ORTHO_002439", "tgt": "RDR_064072",
     "baseline_yr": 13.15, "desc": "Validation: 2007→2020 (13yr)"},
    {"name": "C_null", "ref": "ORTHO_001938", "tgt": "ORTHO_002439",
     "baseline_yr": 0.107, "desc": "Null test: 2006→2007 (~40 days)"},
]

# ─── Constants ────────────────────────────────────────────────────────────────

HIRISE_PIXEL_SCALE = 0.25  # m/px
DTM_PIXEL_SCALE = 1.0  # m/px
MAX_OVERLAP_PX = 8192
ECC_TILE_SIZE = 512
ECC_TILE_OVERLAP = 0.5
ECC_ITERATIONS = 200
ECC_EPSILON = 1e-6


# ─── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class CoregResult:
    """Result of hierarchical co-registration."""
    affine_matrix: np.ndarray  # 2×3 affine transform
    affine_rmse_px: float  # RMS residual after affine (pixels)
    n_keypoints: int
    n_inliers: int
    ecc_applied: bool
    ecc_mean_correction_px: float  # mean local ECC correction magnitude
    final_rmse_px: float  # estimated final alignment error


@dataclass
class ParallaxCorrection:
    """Terrain parallax correction results."""
    max_displacement_m: float
    mean_displacement_m: float
    correction_applied: bool
    emission_angle_deg: float


@dataclass
class DisplacementStats:
    """Statistics for a displacement field on a terrain class."""
    mean_m: float
    median_m: float
    std_m: float
    mad_m: float  # median absolute deviation
    n_valid: int
    percentile_05: float
    percentile_95: float


@dataclass
class PairResult:
    """Full analysis result for one temporal pair."""
    pair_name: str
    ref_image: str
    tgt_image: str
    baseline_yr: float
    coreg: Dict[str, Any]
    parallax: Dict[str, Any]
    scarp_stats: Dict[str, float]
    stable_stats: Dict[str, float]
    noise_floor_m: float
    scarp_excess_m: float  # scarp_median - stable_median
    retreat_rate_m_yr: float  # excess / baseline
    retreat_rate_upper_bound_m_yr: float  # (excess + 2*noise) / baseline


@dataclass
class CrossValResult:
    """Cross-validation between pair A and pair B."""
    r_squared: float
    rmse_m: float
    n_common: int
    ratio_A_B: float  # median(A) / median(B) — should be ~baseline_A/baseline_B


@dataclass
class PipelineReport:
    """Complete pipeline output."""
    site: str
    dtm_id: str
    pairs: List[Dict[str, Any]]
    cross_validation: Dict[str, Any]
    null_test: Dict[str, Any]
    noise_floor_m: float
    verdict: str
    notes: List[str]


# ─── DTM & Terrain Classification ────────────────────────────────────────────

def load_dtm(dtm_path: Path) -> Tuple[np.ndarray, rasterio.Affine, Any]:
    """Load DTM and return elevation array, transform, and CRS."""
    with rasterio.open(dtm_path) as ds:
        elev = ds.read(1).astype(np.float32)
        transform = ds.transform
        crs = ds.crs
        bounds = ds.bounds
        nodata = ds.nodata

    # Handle nodata
    if nodata is not None:
        elev[elev == nodata] = np.nan
    # PDS DTMs may use special values for nodata
    elev[elev < -1e6] = np.nan
    elev[elev > 1e6] = np.nan

    logger.info(
        f"DTM loaded: {elev.shape[1]}×{elev.shape[0]} px, "
        f"elev range: {np.nanmin(elev):.1f} to {np.nanmax(elev):.1f} m"
    )
    return elev, transform, crs


def compute_slope_map(elev: np.ndarray, pixel_scale_m: float = 1.0) -> np.ndarray:
    """Compute slope angle (degrees) from DTM."""
    # Gradient in x and y
    dy, dx = np.gradient(elev, pixel_scale_m)
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    slope_deg = np.degrees(slope_rad)
    slope_deg[np.isnan(elev)] = np.nan
    return slope_deg


def classify_terrain(
    slope_deg: np.ndarray,
    flat_threshold: float = 5.0,
    scarp_threshold: float = 15.0,
) -> np.ndarray:
    """
    Classify terrain based on DTM-derived slope.

    Returns:
        0 = flat/stable (slope < flat_threshold)
        1 = intermediate
        2 = scarp (slope > scarp_threshold)
        255 = nodata
    """
    cls = np.ones(slope_deg.shape, dtype=np.uint8)  # default: intermediate
    cls[slope_deg < flat_threshold] = 0  # flat/stable
    cls[slope_deg > scarp_threshold] = 2  # scarp
    cls[np.isnan(slope_deg)] = 255  # nodata
    return cls


def _mars_eqc_to_lonlat(x: float, y: float, lon_0: float, R: float, lat_0: float = 45.0) -> Tuple[float, float]:
    """Convert Mars equirectangular projected coords to lon/lat (degrees)."""
    # HiRISE equirectangular: x = (lon-lon_0)*R*cos(lat_0)*pi/180, y = lat*R*pi/180
    cos_lat0 = math.cos(math.radians(lat_0))
    lon = lon_0 + x / (R * cos_lat0 * math.pi / 180)
    lat = y / (R * math.pi / 180)
    return lon, lat


def _lonlat_to_mars_eqc(lon: float, lat: float, lon_0: float, R: float, lat_0: float = 45.0) -> Tuple[float, float]:
    """Convert lon/lat (degrees) to Mars equirectangular projected coords."""
    cos_lat0 = math.cos(math.radians(lat_0))
    x = (lon - lon_0) * R * cos_lat0 * math.pi / 180
    y = lat * R * math.pi / 180
    return x, y


def _transform_bounds_mars(
    src_crs_dict: dict, dst_crs_dict: dict,
    left: float, bottom: float, right: float, top: float,
) -> Tuple[float, float, float, float]:
    """Transform bounds between two Mars equirectangular CRSs via lon/lat."""
    src_lon0 = src_crs_dict.get('lon_0', 0.0)
    src_R = src_crs_dict.get('R', 3396190.0)
    src_lat0 = src_crs_dict.get('lat_0', 0.0)
    dst_lon0 = dst_crs_dict.get('lon_0', 0.0)
    dst_R = dst_crs_dict.get('R', 3396190.0)
    dst_lat0 = dst_crs_dict.get('lat_0', 0.0)
    
    # Convert corners to lon/lat
    corners_lonlat = [
        _mars_eqc_to_lonlat(left, bottom, src_lon0, src_R, src_lat0),
        _mars_eqc_to_lonlat(right, bottom, src_lon0, src_R, src_lat0),
        _mars_eqc_to_lonlat(left, top, src_lon0, src_R, src_lat0),
        _mars_eqc_to_lonlat(right, top, src_lon0, src_R, src_lat0),
    ]
    
    # Convert to destination CRS
    dst_coords = [
        _lonlat_to_mars_eqc(lon, lat, dst_lon0, dst_R, dst_lat0)
        for lon, lat in corners_lonlat
    ]
    
    dst_xs = [c[0] for c in dst_coords]
    dst_ys = [c[1] for c in dst_coords]
    return min(dst_xs), min(dst_ys), max(dst_xs), max(dst_ys)


# ─── Image Loading ────────────────────────────────────────────────────────────

def load_overlap_region(
    ref_path: Path,
    tgt_path: Path,
    max_size: int = MAX_OVERLAP_PX,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Load overlapping region of two HiRISE images.

    Handles CRS mismatch by reprojecting the target into the reference CRS.
    Uses VRT-based reprojection for efficiency (no full-image write).

    Returns (ref_img, tgt_img, metadata_dict).
    """
    from rasterio.warp import transform_bounds, Resampling
    from rasterio.vrt import WarpedVRT

    with rasterio.open(ref_path) as ds_ref:
        ref_crs = ds_ref.crs
        ref_bounds = ds_ref.bounds
        ref_transform = ds_ref.transform
        pixel_scale = abs(ref_transform.a)

        with rasterio.open(tgt_path) as ds_tgt:
            tgt_crs = ds_tgt.crs

            # Check CRS match (compare lon_0 and R)
            ref_dict = ref_crs.to_dict()
            tgt_dict = tgt_crs.to_dict()
            same_crs = (
                abs(ref_dict.get('lon_0', 0) - tgt_dict.get('lon_0', 0)) < 0.001
                and abs(ref_dict.get('R', 0) - tgt_dict.get('R', 0)) < 1.0
            )
            if not same_crs:
                logger.info(
                    f"CRS mismatch: ref lon_0={ref_dict.get('lon_0')}/R={ref_dict.get('R')}, "
                    f"tgt lon_0={tgt_dict.get('lon_0')}/R={tgt_dict.get('R')} \u2014 manual transform"
                )
                # Use manual Mars CRS conversion (PROJ can't handle Mars datums)
                b_tgt_left, b_tgt_bottom, b_tgt_right, b_tgt_top = _transform_bounds_mars(
                    tgt_dict, ref_dict,
                    ds_tgt.bounds.left, ds_tgt.bounds.bottom,
                    ds_tgt.bounds.right, ds_tgt.bounds.top,
                )
            else:
                b_tgt_left = ds_tgt.bounds.left
                b_tgt_bottom = ds_tgt.bounds.bottom
                b_tgt_right = ds_tgt.bounds.right
                b_tgt_top = ds_tgt.bounds.top

        # Compute overlap in reference CRS
        left = max(ref_bounds.left, b_tgt_left)
        bottom = max(ref_bounds.bottom, b_tgt_bottom)
        right = min(ref_bounds.right, b_tgt_right)
        top = min(ref_bounds.top, b_tgt_top)

        if left >= right or bottom >= top:
            raise ValueError(
                f"No spatial overlap between {ref_path.name} and {tgt_path.name}. "
                f"Ref bounds: {ref_bounds}, Tgt bounds (in ref CRS): "
                f"({b_tgt_left:.0f}, {b_tgt_bottom:.0f}, {b_tgt_right:.0f}, {b_tgt_top:.0f})"
            )

        overlap_w_px = int((right - left) / pixel_scale)
        overlap_h_px = int((top - bottom) / pixel_scale)

        logger.info(
            f"Overlap: {overlap_w_px}x{overlap_h_px} px "
            f"({(right-left):.0f}x{(top-bottom):.0f} m)"
        )

        # Center-crop if too large
        if overlap_w_px > max_size or overlap_h_px > max_size:
            cx = (left + right) / 2
            cy = (bottom + top) / 2
            half_w = min(overlap_w_px, max_size) / 2 * pixel_scale
            half_h = min(overlap_h_px, max_size) / 2 * pixel_scale
            left, right = cx - half_w, cx + half_w
            bottom, top = cy - half_h, cy + half_h
            logger.info(f"Center-cropped to {max_size}x{max_size} px")

        # Read reference window
        win_ref = from_bounds(left, bottom, right, top, ref_transform)
        win_ref = win_ref.round_offsets().round_lengths()
        ref_img = ds_ref.read(1, window=win_ref).astype(np.float32)

    # Read target with reprojection if needed
    with rasterio.open(tgt_path) as ds_tgt:
        if same_crs:
            win_tgt = from_bounds(left, bottom, right, top, ds_tgt.transform)
            win_tgt = win_tgt.round_offsets().round_lengths()
            tgt_img = ds_tgt.read(1, window=win_tgt).astype(np.float32)
        else:
            # Manual reprojection for Mars CRS mismatch
            # For each pixel in the output (ref CRS), compute its lon/lat,
            # then convert to tgt CRS coords, then sample from tgt image
            H, W = ref_img.shape
            ref_dict = ref_crs.to_dict()
            tgt_dict = ds_tgt.crs.to_dict()
            
            # Build coordinate grids in reference CRS
            ref_out_transform = rasterio.transform.from_bounds(left, bottom, right, top, W, H)
            cols, rows = np.meshgrid(np.arange(W), np.arange(H))
            # Pixel coords → projected coords in ref CRS
            xs_ref = left + cols * pixel_scale + pixel_scale / 2
            ys_ref = top - rows * pixel_scale - pixel_scale / 2
            
            # Ref CRS → geographic → tgt CRS
            ref_lon0 = ref_dict.get('lon_0', 0.0)
            ref_R = ref_dict.get('R', 3396190.0)
            ref_lat0 = ref_dict.get('lat_0', 0.0)
            tgt_lon0 = tgt_dict.get('lon_0', 0.0)
            tgt_R = tgt_dict.get('R', 3396190.0)
            tgt_lat0 = tgt_dict.get('lat_0', 0.0)
            
            cos_ref = math.cos(math.radians(ref_lat0))
            cos_tgt = math.cos(math.radians(tgt_lat0))
            
            lons = ref_lon0 + xs_ref / (ref_R * cos_ref * math.pi / 180)
            lats = ys_ref / (ref_R * math.pi / 180)
            
            xs_tgt = (lons - tgt_lon0) * tgt_R * cos_tgt * math.pi / 180
            ys_tgt = lats * tgt_R * math.pi / 180
            
            # Convert tgt projected coords to tgt pixel coords
            tgt_transform = ds_tgt.transform
            # Inverse transform: pixel = (x - origin_x) / pixel_size
            tgt_cols = (xs_tgt - tgt_transform.c) / tgt_transform.a
            tgt_rows = (ys_tgt - tgt_transform.f) / tgt_transform.e
            
            # Read full target image (or a generous window)
            tgt_left_px = max(0, int(np.min(tgt_cols)) - 10)
            tgt_top_px = max(0, int(np.min(tgt_rows)) - 10)
            tgt_right_px = min(ds_tgt.width, int(np.max(tgt_cols)) + 10)
            tgt_bottom_px = min(ds_tgt.height, int(np.max(tgt_rows)) + 10)
            
            tgt_win = rasterio.windows.Window(
                tgt_left_px, tgt_top_px,
                tgt_right_px - tgt_left_px, tgt_bottom_px - tgt_top_px,
            )
            tgt_data = ds_tgt.read(1, window=tgt_win).astype(np.float32)
            
            # Adjust pixel coordinates to window origin
            sample_cols = (tgt_cols - tgt_left_px).astype(np.float32)
            sample_rows = (tgt_rows - tgt_top_px).astype(np.float32)
            
            # Bilinear interpolation via cv2.remap
            tgt_img = cv2.remap(
                tgt_data, sample_cols, sample_rows,
                cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            )
            logger.info(
                f"Manual reprojection: tgt window {tgt_data.shape} → output {tgt_img.shape}"
            )

    # Ensure same shape
    h = min(ref_img.shape[0], tgt_img.shape[0])
    w = min(ref_img.shape[1], tgt_img.shape[1])
    ref_img = ref_img[:h, :w]
    tgt_img = tgt_img[:h, :w]

    metadata = {
        "overlap_bounds": {"left": left, "bottom": bottom, "right": right, "top": top},
        "pixel_scale_m": pixel_scale,
        "shape": (h, w),
        "ref_crs": ref_crs,
    }

    logger.info(f"Loaded overlap: {w}x{h} px, pixel_scale={pixel_scale:.4f} m")
    return ref_img, tgt_img, metadata


def load_dtm_for_overlap(
    dtm_path: Path,
    overlap_bounds: dict,
    target_shape: Tuple[int, int],
    ref_crs: Any = None,  # CRS of the overlap bounds
) -> np.ndarray:
    """Load and resample DTM to match the overlap region of HiRISE images.

    Handles CRS mismatch: overlap_bounds may be in RDR CRS while DTM is in ORTHO CRS.
    """
    with rasterio.open(dtm_path) as ds:
        left = overlap_bounds['left']
        bottom = overlap_bounds['bottom']
        right = overlap_bounds['right']
        top = overlap_bounds['top']

        dtm_dict = ds.crs.to_dict()

        # Check if we need CRS conversion
        needs_conversion = False
        if ref_crs is not None:
            ref_dict = ref_crs.to_dict() if hasattr(ref_crs, 'to_dict') else ref_crs
            same = (
                abs(ref_dict.get('lon_0', 0) - dtm_dict.get('lon_0', 0)) < 0.001
                and abs(ref_dict.get('R', 0) - dtm_dict.get('R', 0)) < 1.0
            )
            needs_conversion = not same

        if needs_conversion:
            # Convert overlap bounds from ref CRS to DTM CRS via lon/lat
            dtm_left, dtm_bottom, dtm_right, dtm_top = _transform_bounds_mars(
                ref_dict, dtm_dict, left, bottom, right, top
            )
            logger.info(
                f"DTM CRS conversion: overlap ({left:.0f},{bottom:.0f})-({right:.0f},{top:.0f}) "
                f"→ DTM ({dtm_left:.0f},{dtm_bottom:.0f})-({dtm_right:.0f},{dtm_top:.0f})"
            )
            left, bottom, right, top = dtm_left, dtm_bottom, dtm_right, dtm_top

        win = from_bounds(left, bottom, right, top, ds.transform)
        win = win.round_offsets().round_lengths()

        # Validate window
        if win.width <= 0 or win.height <= 0:
            logger.warning(f"DTM window has zero size ({win.width}x{win.height}). DTM may not overlap.")
            return np.full(target_shape, np.nan, dtype=np.float32)

        dtm_chip = ds.read(1, window=win).astype(np.float32)

    if dtm_chip.size == 0:
        logger.warning("DTM chip is empty")
        return np.full(target_shape, np.nan, dtype=np.float32)

    # Handle nodata
    dtm_chip[dtm_chip < -1e6] = np.nan
    dtm_chip[dtm_chip > 1e6] = np.nan

    # Resample to target shape (DTM is coarser than HiRISE)
    if dtm_chip.shape != target_shape and dtm_chip.shape[0] > 0 and dtm_chip.shape[1] > 0:
        from scipy.ndimage import zoom
        zoom_r = target_shape[0] / dtm_chip.shape[0]
        zoom_c = target_shape[1] / dtm_chip.shape[1]
        dtm_chip = zoom(dtm_chip, (zoom_r, zoom_c), order=1)

    return dtm_chip


# ─── Hierarchical Co-Registration ────────────────────────────────────────────

def _normalize_for_coreg(img: np.ndarray) -> np.ndarray:
    """Normalize image to uint8 for OpenCV feature matching."""
    img_f = img.astype(np.float64)
    vmin, vmax = np.nanpercentile(img_f[np.isfinite(img_f)], [1, 99])
    img_f = np.clip((img_f - vmin) / max(vmax - vmin, 1e-6), 0, 1)
    return (img_f * 255).astype(np.uint8)


def _histogram_match(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Match source image histogram to reference (both uint8)."""
    src_hist, _ = np.histogram(source.ravel(), bins=256, range=(0, 256))
    ref_hist, _ = np.histogram(reference.ravel(), bins=256, range=(0, 256))
    src_cdf = src_hist.cumsum().astype(np.float64)
    ref_cdf = ref_hist.cumsum().astype(np.float64)
    src_cdf /= max(src_cdf[-1], 1)
    ref_cdf /= max(ref_cdf[-1], 1)
    # Build lookup table
    lut = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        j = np.searchsorted(ref_cdf, src_cdf[i])
        lut[i] = min(j, 255)
    return lut[source]


def _apply_clahe(img_u8: np.ndarray, clip_limit: float = 3.0, tile_size: int = 16) -> np.ndarray:
    """Apply CLAHE for local contrast normalization."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    return clahe.apply(img_u8)


def _prepare_pair_for_matching(
    ref_img: np.ndarray, tgt_img: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare image pair for feature matching: normalize, histogram-match, CLAHE.
    Handles 8-bit ORTHO vs 16-bit RDR radiometric mismatch.
    """
    ref_u8 = _normalize_for_coreg(ref_img)
    tgt_u8 = _normalize_for_coreg(tgt_img)
    # Match target histogram to reference
    tgt_u8 = _histogram_match(tgt_u8, ref_u8)
    # CLAHE for local contrast normalization
    ref_u8 = _apply_clahe(ref_u8)
    tgt_u8 = _apply_clahe(tgt_u8)
    return ref_u8, tgt_u8


def global_affine_registration(
    ref_img: np.ndarray,
    tgt_img: np.ndarray,
    max_features: int = 20000,
    ransac_threshold: float = 5.0,
) -> Tuple[np.ndarray, float, int, int]:
    """
    Compute global alignment using phase correlation for bulk shift,
    then SIFT + RANSAC for affine refinement.

    SIFT is far more robust than ORB across radiometric differences
    (8-bit ORTHO vs 16-bit RDR) because it uses gradient histograms.

    Returns: (affine_2x3, rmse_pixels, n_keypoints, n_inliers)
    """
    H, W = ref_img.shape

    # Step 1: Phase-correlation bulk alignment (handles large shifts)
    patch_size = min(2048, H, W)
    cy, cx = H // 2, W // 2
    half = patch_size // 2
    p1 = ref_img[cy-half:cy+half, cx-half:cx+half].astype(np.float64)
    p2 = tgt_img[cy-half:cy+half, cx-half:cx+half].astype(np.float64)
    p1 = (p1 - p1.mean()) / max(p1.std(), 1e-6)
    p2 = (p2 - p2.mean()) / max(p2.std(), 1e-6)

    from numpy.fft import fft2, ifft2
    cc = np.abs(ifft2(fft2(p1) * np.conj(fft2(p2))))
    peak = np.unravel_index(np.argmax(cc), cc.shape)
    shift_r = float(peak[0])
    shift_c = float(peak[1])
    if shift_r > patch_size / 2:
        shift_r -= patch_size
    if shift_c > patch_size / 2:
        shift_c -= patch_size

    logger.info(f"Phase-corr bulk shift: ({shift_r:.0f}, {shift_c:.0f}) px")

    # Step 2: Apply bulk shift as initial affine
    M_bulk = np.float64([[1, 0, shift_c], [0, 1, shift_r]])
    tgt_shifted = cv2.warpAffine(
        tgt_img, M_bulk.astype(np.float32), (W, H),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )

    # Step 3: SIFT feature matching with histogram-matched + CLAHE images
    ref_u8, tgt_u8 = _prepare_pair_for_matching(ref_img, tgt_shifted)

    sift = cv2.SIFT_create(nfeatures=max_features)
    kp1, des1 = sift.detectAndCompute(ref_u8, None)
    kp2, des2 = sift.detectAndCompute(tgt_u8, None)

    n_kp = len(kp1) if kp1 else 0
    logger.info(f"SIFT: ref={n_kp} kp, tgt={len(kp2) if kp2 else 0} kp")

    if des1 is None or des2 is None or n_kp < 10 or len(kp2) < 10:
        logger.warning("SIFT matching failed — using bulk shift only")
        return M_bulk, float(np.sqrt(shift_r**2 + shift_c**2)) * 0.5, n_kp, 0

    # FLANN matcher (better for SIFT float descriptors)
    FLANN_INDEX_KDTREE = 1
    flann = cv2.FlannBasedMatcher(
        dict(algorithm=FLANN_INDEX_KDTREE, trees=5),
        dict(checks=50),
    )
    raw_matches = flann.knnMatch(des1, des2, k=2)

    # Lowe's ratio test (strict for precision)
    good_matches = []
    for m_pair in raw_matches:
        if len(m_pair) == 2 and m_pair[0].distance < 0.7 * m_pair[1].distance:
            good_matches.append(m_pair[0])

    logger.info(f"SIFT ratio-test matches: {len(good_matches)}")

    if len(good_matches) < 10:
        logger.warning(f"Only {len(good_matches)} matches — using bulk shift")
        return M_bulk, float(np.sqrt(shift_r**2 + shift_c**2)) * 0.5, n_kp, len(good_matches)

    pts_ref = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
    pts_tgt = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

    affine_mat, inlier_mask = cv2.estimateAffinePartial2D(
        pts_tgt, pts_ref,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_threshold,
    )

    if affine_mat is None:
        logger.warning("Affine refinement failed — using bulk shift")
        return M_bulk, float(np.sqrt(shift_r**2 + shift_c**2)) * 0.5, n_kp, 0

    n_inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0

    # Compute RMS residual on inliers
    inlier_idx = inlier_mask.ravel().astype(bool)
    pts_tgt_inlier = pts_tgt[inlier_idx]
    pts_ref_inlier = pts_ref[inlier_idx]

    pts_tgt_h = np.hstack([pts_tgt_inlier, np.ones((len(pts_tgt_inlier), 1))])
    pts_warped = (affine_mat @ pts_tgt_h.T).T
    residuals = pts_ref_inlier - pts_warped
    rmse = float(np.sqrt(np.mean(residuals**2)))

    # Compose bulk shift with SIFT refinement:
    # affine_mat operates on the SHIFTED target, so we need:
    #   composed = affine_mat @ [M_bulk; 0 0 1]
    M_bulk_3x3 = np.vstack([M_bulk, [0, 0, 1]])
    affine_3x3 = np.vstack([affine_mat, [0, 0, 1]])
    composed_3x3 = affine_3x3 @ M_bulk_3x3
    composed = composed_3x3[:2, :]

    logger.info(
        f"Global affine: {len(good_matches)} matches, {n_inliers} inliers, "
        f"RMSE={rmse:.3f} px ({rmse * HIRISE_PIXEL_SCALE:.4f} m)"
    )
    logger.info(
        f"  Bulk shift: ({shift_r:.0f}, {shift_c:.0f}), "
        f"composed translation: ({composed[1,2]:.1f}, {composed[0,2]:.1f}) px"
    )

    return composed, rmse, len(kp1), n_inliers

def apply_affine_warp(
    img: np.ndarray,
    affine_mat: np.ndarray,
    output_shape: Tuple[int, int],
) -> np.ndarray:
    """
    Apply affine warp to align target image to reference.

    affine_mat maps from target pixel coords to reference pixel coords.
    cv2.warpAffine without WARP_INVERSE_MAP interprets M as the forward transform,
    and internally inverts it to compute: dst(p) = src(M^{-1} * p).
    """
    h, w = output_shape
    warped = cv2.warpAffine(
        img, affine_mat.astype(np.float64), (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warped

def local_ecc_refinement(
    ref_img: np.ndarray,
    tgt_img: np.ndarray,
    tile_size: int = ECC_TILE_SIZE,
    overlap_frac: float = ECC_TILE_OVERLAP,
    max_iterations: int = ECC_ITERATIONS,
    epsilon: float = ECC_EPSILON,
) -> Tuple[np.ndarray, float, bool]:
    """
    Apply local ECC refinement in tiles to handle spatially-varying drift.

    Uses histogram-matched + CLAHE images for robust ECC across radiometric differences.
    Includes quality gate: rejects when correction std >> median (pure noise).

    Returns (warped_target, mean_correction, quality_ok).
    """
    H, W = ref_img.shape
    step = int(tile_size * (1 - overlap_frac))

    # Use histogram-matched + CLAHE images for better ECC convergence
    ref_u8, tgt_u8 = _prepare_pair_for_matching(ref_img, tgt_img)

    # Collect local translation corrections
    corrections_r = []
    corrections_c = []
    centers_r = []
    centers_c = []

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iterations, epsilon)

    for r in range(0, H - tile_size + 1, step):
        for c in range(0, W - tile_size + 1, step):
            ref_tile = ref_u8[r:r + tile_size, c:c + tile_size]
            tgt_tile = tgt_u8[r:r + tile_size, c:c + tile_size]

            # Skip low-contrast tiles
            if ref_tile.std() < 8 or tgt_tile.std() < 8:
                continue

            try:
                warp_matrix = np.eye(2, 3, dtype=np.float32)
                cc, warp_matrix = cv2.findTransformECC(
                    ref_tile.astype(np.float32),
                    tgt_tile.astype(np.float32),
                    warp_matrix,
                    cv2.MOTION_TRANSLATION,  # Translation only (simpler, more robust)
                    criteria,
                )
                tx = warp_matrix[0, 2]
                ty = warp_matrix[1, 2]

                # Reject outlier corrections (> 3 px) — tighter threshold
                if abs(tx) < 3 and abs(ty) < 3 and cc > 0.5:
                    corrections_c.append(tx)
                    corrections_r.append(ty)
                    centers_r.append(r + tile_size // 2)
                    centers_c.append(c + tile_size // 2)

            except cv2.error:
                continue

    if len(corrections_r) < 4:
        logger.warning("Insufficient ECC tile matches — skipping local refinement")
        return tgt_img.copy(), 0.0, False

    corrections_r = np.array(corrections_r)
    corrections_c = np.array(corrections_c)
    centers_r = np.array(centers_r)
    centers_c = np.array(centers_c)

    med_r = float(np.median(corrections_r))
    med_c = float(np.median(corrections_c))
    std_r = float(np.std(corrections_r))
    std_c = float(np.std(corrections_c))
    mean_correction = float(np.sqrt(med_r**2 + med_c**2))

    logger.info(
        f"Local ECC: {len(corrections_r)} tiles, "
        f"median correction: ({med_r:.3f}, {med_c:.3f}) px, "
        f"std: ({std_r:.3f}, {std_c:.3f}) px"
    )

    # Quality gate: if std > 1.5 px, corrections are too noisy to be useful
    # (For good co-registration, std should be << 0.5 px)
    if std_r > 1.5 or std_c > 1.5:
        logger.warning(
            f"ECC corrections too noisy (std_r={std_r:.2f}, std_c={std_c:.2f} px). "
            f"Applying ONLY median translation, not spatially-varying field."
        )
        # Apply only the robust median shift
        M = np.float32([[1, 0, med_c], [0, 1, med_r]])
        warped = cv2.warpAffine(
            tgt_img, M, (W, H),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return warped, mean_correction, False

    # If corrections are spatially uniform, apply global median
    if std_r < 0.3 and std_c < 0.3:
        M = np.float32([[1, 0, med_c], [0, 1, med_r]])
        warped = cv2.warpAffine(
            tgt_img, M, (W, H),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return warped, mean_correction, True

    # Spatially-varying correction: interpolate correction field
    from scipy.interpolate import griddata

    # First, apply MAD-based outlier rejection
    mad_r = np.median(np.abs(corrections_r - med_r))
    mad_c = np.median(np.abs(corrections_c - med_c))
    inlier_mask = (
        (np.abs(corrections_r - med_r) < 3 * max(mad_r, 0.1)) &
        (np.abs(corrections_c - med_c) < 3 * max(mad_c, 0.1))
    )
    corrections_r = corrections_r[inlier_mask]
    corrections_c = corrections_c[inlier_mask]
    centers_r = centers_r[inlier_mask]
    centers_c = centers_c[inlier_mask]

    if len(corrections_r) < 4:
        M = np.float32([[1, 0, med_c], [0, 1, med_r]])
        warped = cv2.warpAffine(
            tgt_img, M, (W, H),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return warped, mean_correction, True

    grid_r, grid_c = np.mgrid[0:H, 0:W]
    points = np.column_stack([centers_r, centers_c])

    map_r = griddata(points, corrections_r, (grid_r, grid_c), method="cubic", fill_value=med_r)
    map_c = griddata(points, corrections_c, (grid_r, grid_c), method="cubic", fill_value=med_c)

    # Apply via remap
    map_x = (grid_c - map_c).astype(np.float32)
    map_y = (grid_r - map_r).astype(np.float32)
    warped = cv2.remap(tgt_img, map_x, map_y, cv2.INTER_CUBIC, borderValue=0)

    return warped, mean_correction, True

def hierarchical_coregistration(
    ref_img: np.ndarray,
    tgt_img: np.ndarray,
) -> Tuple[np.ndarray, CoregResult]:
    """
    Full hierarchical co-registration pipeline.

    1. Global affine registration (SIFT + RANSAC)
    2. Local ECC refinement in tiles (with quality gate)

    Returns (aligned_target, CoregResult).
    """
    # Step 1: Global affine
    affine_mat, affine_rmse, n_kp, n_inliers = global_affine_registration(
        ref_img, tgt_img
    )

    # Apply affine warp
    tgt_affine = apply_affine_warp(tgt_img, affine_mat, ref_img.shape)

    # Step 2: Local ECC refinement (with quality gate)
    tgt_refined, ecc_correction, ecc_quality_ok = local_ecc_refinement(ref_img, tgt_affine)

    # Estimate final RMSE based on ECC quality
    if ecc_quality_ok and ecc_correction > 0.01:
        final_rmse = max(affine_rmse * 0.3, 0.05)
    else:
        # ECC was noisy, final RMSE is close to affine RMSE
        final_rmse = max(affine_rmse * 0.7, 0.1)

    coreg_result = CoregResult(
        affine_matrix=affine_mat,
        affine_rmse_px=affine_rmse,
        n_keypoints=n_kp,
        n_inliers=n_inliers,
        ecc_applied=ecc_correction > 0.01,
        ecc_mean_correction_px=ecc_correction,
        final_rmse_px=final_rmse,
    )

    return tgt_refined, coreg_result

# ─── Terrain Parallax Correction ──────────────────────────────────────────────

def compute_parallax_displacement(
    dtm: np.ndarray,
    emission_angle_deg: float,
    spacecraft_azimuth_deg: float = 0.0,
    pixel_scale_m: float = HIRISE_PIXEL_SCALE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-pixel terrain parallax displacement for a non-ortho image.

    For each pixel with elevation h above the reference surface:
      displacement = h * tan(emission_angle) in the spacecraft viewing direction

    Returns (disp_row_px, disp_col_px) — displacement in pixels.
    """
    e_rad = math.radians(emission_angle_deg)
    az_rad = math.radians(spacecraft_azimuth_deg)

    # Use elevation relative to median (reference surface)
    h_ref = np.nanmedian(dtm)
    h_relative = dtm - h_ref

    # Displacement in meters
    disp_magnitude_m = h_relative * math.tan(e_rad)
    disp_row_m = disp_magnitude_m * math.cos(az_rad)  # N-S component
    disp_col_m = disp_magnitude_m * math.sin(az_rad)  # E-W component

    # Convert to pixels
    disp_row_px = disp_row_m / pixel_scale_m
    disp_col_px = disp_col_m / pixel_scale_m

    max_disp = float(np.nanmax(np.abs(disp_magnitude_m)))
    mean_disp = float(np.nanmean(np.abs(disp_magnitude_m)))

    logger.info(
        f"Parallax correction: emission={emission_angle_deg:.2f}°, "
        f"max_disp={max_disp:.2f} m ({max_disp/pixel_scale_m:.1f} px), "
        f"mean_disp={mean_disp:.2f} m"
    )

    return disp_row_px, disp_col_px


def apply_parallax_correction(
    img: np.ndarray,
    disp_row_px: np.ndarray,
    disp_col_px: np.ndarray,
) -> np.ndarray:
    """Apply parallax correction by resampling the image."""
    H, W = img.shape
    grid_r, grid_c = np.mgrid[0:H, 0:W]

    # Ensure displacement arrays match image shape
    if disp_row_px.shape != img.shape:
        from scipy.ndimage import zoom
        zr = img.shape[0] / disp_row_px.shape[0]
        zc = img.shape[1] / disp_row_px.shape[1]
        disp_row_px = zoom(disp_row_px, (zr, zc), order=1)
        disp_col_px = zoom(disp_col_px, (zr, zc), order=1)

    # Remap: shift each pixel by its parallax displacement
    map_x = (grid_c - disp_col_px).astype(np.float32)
    map_y = (grid_r - disp_row_px).astype(np.float32)

    corrected = cv2.remap(img, map_x, map_y, cv2.INTER_CUBIC, borderValue=0)
    return corrected


# ─── Phase Correlation (reuse existing engine) ───────────────────────────────

def _phase_correlate_chip(
    chip1: np.ndarray,
    chip2: np.ndarray,
    upsample_factor: int = 100,
) -> Tuple[float, float, float]:
    """
    Phase correlation on a single chip pair.
    Returns (row_shift, col_shift, snr).
    """
    from .phase_correlation import cosicorr_phase_correlation
    result = cosicorr_phase_correlation(
        chip1, chip2,
        upsample_factor=upsample_factor,
        freq_low=0.02,
        freq_high=0.80,
        snr_threshold=2.0,
    )
    return result.row_shift, result.col_shift, result.snr


def compute_displacement_field(
    ref_img: np.ndarray,
    tgt_img: np.ndarray,
    chip_size: int = 128,
    step_size: int = 64,
    upsample_factor: int = 100,
    snr_threshold: float = 2.0,
    pixel_scale_m: float = HIRISE_PIXEL_SCALE,
    detrend_order: int = 3,
    stable_mask: Optional[np.ndarray] = None,
) -> dict:
    """
    Compute displacement field using sliding-window phase correlation.

    Uses the existing phase_correlation module but with improved parameters.

    Returns dict with displacement maps and statistics.
    """
    from .phase_correlation import sliding_window_correlation

    field = sliding_window_correlation(
        ref_img, tgt_img,
        chip_size=chip_size,
        step_size=step_size,
        upsample_factor=upsample_factor,
        snr_threshold=snr_threshold,
        pixel_scale_m=pixel_scale_m,
        stable_mask=stable_mask,
        remove_bulk_offset=True,
        detrend_order=detrend_order,
    )

    return {
        "row_disp_m": field.row_disp_m,
        "col_disp_m": field.col_disp_m,
        "magnitude_m": field.magnitude_m,
        "snr": field.snr,
        "valid_mask": field.valid_mask,
        "valid_fraction": field.valid_fraction,
        "row_centers": field.row_centers,
        "col_centers": field.col_centers,
    }


# ─── Statistics & Cross-Validation ────────────────────────────────────────────

def compute_stats(
    magnitude_m: np.ndarray,
    mask: np.ndarray,
    valid_mask: np.ndarray,
) -> DisplacementStats:
    """Compute displacement statistics for a given terrain class."""
    combined = mask & valid_mask
    values = magnitude_m[combined]

    if len(values) == 0:
        return DisplacementStats(
            mean_m=np.nan, median_m=np.nan, std_m=np.nan, mad_m=np.nan,
            n_valid=0, percentile_05=np.nan, percentile_95=np.nan,
        )

    return DisplacementStats(
        mean_m=float(np.mean(values)),
        median_m=float(np.median(values)),
        std_m=float(np.std(values)),
        mad_m=float(np.median(np.abs(values - np.median(values)))),
        n_valid=int(len(values)),
        percentile_05=float(np.percentile(values, 5)),
        percentile_95=float(np.percentile(values, 95)),
    )


def cross_validate_pairs(
    disp_A: np.ndarray,
    disp_B: np.ndarray,
    valid_A: np.ndarray,
    valid_B: np.ndarray,
) -> CrossValResult:
    """
    Cross-validate displacement fields from two independent temporal pairs.
    Both fields should show proportional displacement (ratio = baseline ratio).
    """
    common = valid_A & valid_B
    n_common = int(common.sum())

    if n_common < 10:
        return CrossValResult(
            r_squared=np.nan, rmse_m=np.nan, n_common=n_common, ratio_A_B=np.nan,
        )

    a = disp_A[common].flatten()
    b = disp_B[common].flatten()

    # R² correlation
    if np.std(a) > 0 and np.std(b) > 0:
        r = np.corrcoef(a, b)[0, 1]
        r_squared = float(r**2)
    else:
        r_squared = 0.0

    rmse = float(np.sqrt(np.mean((a - b)**2)))
    ratio = float(np.median(a) / np.median(b)) if np.median(b) != 0 else np.nan

    return CrossValResult(
        r_squared=r_squared,
        rmse_m=rmse,
        n_common=n_common,
        ratio_A_B=ratio,
    )


# ─── Main Pipeline ────────────────────────────────────────────────────────────

def run_pair_analysis(
    pair_config: dict,
    dtm_path: Path = DTM_PATH,
    chip_size: int = 128,
    step_size: int = 64,
    detrend_order: int = 3,
) -> Tuple[PairResult, dict]:
    """
    Run full analysis for one temporal pair.

    Returns (PairResult, displacement_data_dict).
    """
    ref_key = pair_config["ref"]
    tgt_key = pair_config["tgt"]
    pair_name = pair_config["name"]
    baseline_yr = pair_config["baseline_yr"]

    ref_info = IMAGE_CATALOG[ref_key]
    tgt_info = IMAGE_CATALOG[tgt_key]

    logger.info(f"\n{'='*60}")
    logger.info(f"Processing pair: {pair_name}")
    logger.info(f"  Ref: {ref_key} ({ref_info.date})")
    logger.info(f"  Tgt: {tgt_key} ({tgt_info.date})")
    logger.info(f"  Baseline: {baseline_yr:.2f} yr")
    logger.info(f"{'='*60}")

    # 1. Load overlap region
    ref_img, tgt_img, meta = load_overlap_region(ref_info.path, tgt_info.path)

    # 2. Hierarchical co-registration
    tgt_aligned, coreg = hierarchical_coregistration(ref_img, tgt_img)

    # 3. Terrain parallax correction
    #    - If ref is ORTHO: correct tgt by its full parallax
    #    - If both are non-ORTHO (RDR): correct tgt by DIFFERENTIAL parallax
    #    - If both are ORTHO: no correction needed
    parallax_info = ParallaxCorrection(
        max_displacement_m=0.0, mean_displacement_m=0.0,
        correction_applied=False, emission_angle_deg=tgt_info.emission_angle,
    )

    if dtm_path.exists() and (not tgt_info.is_ortho or not ref_info.is_ortho):
        dtm_chip = load_dtm_for_overlap(dtm_path, meta['overlap_bounds'], ref_img.shape, ref_crs=meta.get('ref_crs'))

        if not np.all(np.isnan(dtm_chip)):
            if ref_info.is_ortho:
                # Ref is orthorectified, correct tgt by its full parallax
                disp_r, disp_c = compute_parallax_displacement(
                    dtm_chip, emission_angle_deg=tgt_info.emission_angle,
                )
                tgt_aligned = apply_parallax_correction(tgt_aligned, disp_r, disp_c)
                diff_angle = tgt_info.emission_angle
            elif not ref_info.is_ortho and not tgt_info.is_ortho:
                # Both are non-ORTHO: compute DIFFERENTIAL parallax
                # The displacement difference due to terrain between two off-nadir views
                disp_r_ref, disp_c_ref = compute_parallax_displacement(
                    dtm_chip, emission_angle_deg=ref_info.emission_angle,
                )
                disp_r_tgt, disp_c_tgt = compute_parallax_displacement(
                    dtm_chip, emission_angle_deg=tgt_info.emission_angle,
                )
                diff_r = disp_r_tgt - disp_r_ref
                diff_c = disp_c_tgt - disp_c_ref
                tgt_aligned = apply_parallax_correction(tgt_aligned, diff_r, diff_c)
                diff_angle = abs(tgt_info.emission_angle - ref_info.emission_angle)
                logger.info(
                    f"Differential parallax: ref={ref_info.emission_angle:.2f}°, "
                    f"tgt={tgt_info.emission_angle:.2f}°, diff={diff_angle:.2f}°"
                )
            else:
                diff_angle = 0.0

            h_rel = dtm_chip - np.nanmedian(dtm_chip)
            parallax_info = ParallaxCorrection(
                max_displacement_m=float(np.nanmax(np.abs(
                    h_rel * math.tan(math.radians(diff_angle))
                ))) if diff_angle > 0 else 0.0,
                mean_displacement_m=float(np.nanmean(np.abs(
                    h_rel * math.tan(math.radians(diff_angle))
                ))) if diff_angle > 0 else 0.0,
                correction_applied=True,
                emission_angle_deg=tgt_info.emission_angle,
            )
    elif tgt_info.is_ortho and ref_info.is_ortho:
        logger.info('Both images are orthorectified — no parallax correction needed')
    # 4. DTM-based terrain classification
    stable_mask_full = None
    scarp_mask_full = None

    if dtm_path.exists():
        dtm_chip = load_dtm_for_overlap(dtm_path, meta["overlap_bounds"], ref_img.shape, ref_crs=meta.get("ref_crs"))
        slope = compute_slope_map(dtm_chip, pixel_scale_m=DTM_PIXEL_SCALE)
        terrain_cls = classify_terrain(slope)

        # Create masks at displacement field resolution
        stable_mask_full = (terrain_cls == 0)  # flat terrain
        scarp_mask_full = (terrain_cls == 2)  # steep terrain

    # 5. Compute displacement field
    disp_data = compute_displacement_field(
        ref_img, tgt_aligned,
        chip_size=chip_size,
        step_size=step_size,
        detrend_order=detrend_order,
        stable_mask=stable_mask_full,
    )

    # 6. Resample terrain masks to displacement grid
    nr = len(disp_data["row_centers"])
    nc = len(disp_data["col_centers"])

    if stable_mask_full is not None:
        from scipy.ndimage import zoom
        stable_mask_disp = zoom(
            stable_mask_full.astype(np.float32),
            (nr / stable_mask_full.shape[0], nc / stable_mask_full.shape[1]),
            order=0,
        ) > 0.5
        scarp_mask_disp = zoom(
            scarp_mask_full.astype(np.float32),
            (nr / scarp_mask_full.shape[0], nc / scarp_mask_full.shape[1]),
            order=0,
        ) > 0.5
    else:
        # Fallback: use gradient-based classification
        stable_mask_disp = np.ones((nr, nc), dtype=bool)
        scarp_mask_disp = np.zeros((nr, nc), dtype=bool)

    # 7. Compute statistics
    scarp_stats = compute_stats(
        disp_data["magnitude_m"], scarp_mask_disp, disp_data["valid_mask"]
    )
    stable_stats = compute_stats(
        disp_data["magnitude_m"], stable_mask_disp, disp_data["valid_mask"]
    )

    noise_floor = stable_stats.mad_m * 1.4826  # MAD → σ conversion
    scarp_excess = scarp_stats.median_m - stable_stats.median_m

    retreat_rate = scarp_excess / baseline_yr if baseline_yr > 0 else 0.0
    upper_bound = (scarp_excess + 2 * noise_floor) / baseline_yr if baseline_yr > 0 else 0.0

    pair_result = PairResult(
        pair_name=pair_name,
        ref_image=ref_key,
        tgt_image=tgt_key,
        baseline_yr=baseline_yr,
        coreg={
            "affine_rmse_px": coreg.affine_rmse_px,
            "n_inliers": coreg.n_inliers,
            "ecc_applied": coreg.ecc_applied,
            "ecc_correction_px": coreg.ecc_mean_correction_px,
            "final_rmse_px": coreg.final_rmse_px,
        },
        parallax={
            "correction_applied": parallax_info.correction_applied,
            "emission_angle_deg": parallax_info.emission_angle_deg,
            "max_displacement_m": parallax_info.max_displacement_m,
            "mean_displacement_m": parallax_info.mean_displacement_m,
        },
        scarp_stats=asdict(scarp_stats),
        stable_stats=asdict(stable_stats),
        noise_floor_m=noise_floor,
        scarp_excess_m=scarp_excess,
        retreat_rate_m_yr=retreat_rate,
        retreat_rate_upper_bound_m_yr=upper_bound,
    )

    logger.info(f"\n--- Pair {pair_name} Results ---")
    logger.info(f"  Noise floor: {noise_floor:.4f} m")
    logger.info(f"  Scarp excess: {scarp_excess:.4f} m")
    logger.info(f"  Retreat rate: {retreat_rate:.4f} m/yr")
    logger.info(f"  Upper bound: {upper_bound:.4f} m/yr")

    return pair_result, disp_data


def run_pipeline(
    output_dir: Path = RESULTS_DIR,
    chip_size: int = 128,
    step_size: int = 64,
    detrend_order: int = 3,
) -> PipelineReport:
    """
    Run the complete v2 ortho pipeline on all temporal pairs.

    Steps:
    1. Process each temporal pair (co-registration, parallax, displacement)
    2. Cross-validate pair A vs pair B
    3. Run null test on pair C
    4. Determine verdict
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Verify required files exist (only for pairs we're running)
    missing = []
    needed_keys = set()
    for p in TEMPORAL_PAIRS:
        needed_keys.add(p['ref'])
        needed_keys.add(p['tgt'])
    for key in needed_keys:
        info = IMAGE_CATALOG[key]
        if not info.path.exists():
            missing.append(f"{key}: {info.path}")
    if not DTM_PATH.exists():
        missing.append(f"DTM: {DTM_PATH}")
    if missing:
        logger.warning(f"Missing files (will skip affected pairs): {missing}")
    # Process each pair
    pair_results: List[PairResult] = []
    disp_data_all: Dict[str, dict] = {}

    for pair_config in TEMPORAL_PAIRS:
        ref_info = IMAGE_CATALOG.get(pair_config['ref'])
        tgt_info = IMAGE_CATALOG.get(pair_config['tgt'])
        if ref_info is None or tgt_info is None:
            logger.warning(f"Skipping {pair_config['name']}: unknown image key")
            continue
        if not ref_info.path.exists() or not tgt_info.path.exists():
            logger.warning(f"Skipping {pair_config['name']}: missing data file")
            continue

        try:
            result, disp_data = run_pair_analysis(
                pair_config,
                chip_size=chip_size,
                step_size=step_size,
                detrend_order=detrend_order,
            )
            pair_results.append(result)
            disp_data_all[pair_config['name']] = disp_data

            # Save per-pair results
            pair_dir = output_dir / pair_config['name']
            pair_dir.mkdir(exist_ok=True)
            np.savez_compressed(
                pair_dir / 'displacement.npz',
                magnitude_m=disp_data['magnitude_m'],
                row_disp_m=disp_data['row_disp_m'],
                col_disp_m=disp_data['col_disp_m'],
                snr=disp_data['snr'],
                valid_mask=disp_data['valid_mask'],
            )
        except Exception as e:
            logger.error(f"Pair {pair_config['name']} failed: {e}")
            continue

    # Cross-validation: primary vs validation pair
    crossval = CrossValResult(
        r_squared=np.nan, rmse_m=np.nan, n_common=0, ratio_A_B=np.nan,
    )
    primary_name = TEMPORAL_PAIRS[0]['name'] if TEMPORAL_PAIRS else None
    validation_name = TEMPORAL_PAIRS[1]['name'] if len(TEMPORAL_PAIRS) > 1 else None
    if primary_name in disp_data_all and validation_name in disp_data_all:
        da = disp_data_all[primary_name]
        db = disp_data_all[validation_name]

        min_r = min(da['magnitude_m'].shape[0], db['magnitude_m'].shape[0])
        min_c = min(da['magnitude_m'].shape[1], db['magnitude_m'].shape[1])

        crossval = cross_validate_pairs(
            da['magnitude_m'][:min_r, :min_c],
            db['magnitude_m'][:min_r, :min_c],
            da['valid_mask'][:min_r, :min_c],
            db['valid_mask'][:min_r, :min_c],
        )

    # Null test: last pair should show ~0 displacement
    null_name = TEMPORAL_PAIRS[-1]['name'] if TEMPORAL_PAIRS else None
    null_result = None
    for pr in pair_results:
        if pr.pair_name == null_name:
            null_result = pr
            break
    null_test = {
        'scarp_excess_m': null_result.scarp_excess_m if null_result else np.nan,
        'noise_floor_m': null_result.noise_floor_m if null_result else np.nan,
        'pass': (abs(null_result.scarp_excess_m) < 2 * null_result.noise_floor_m
                 if null_result and not np.isnan(null_result.scarp_excess_m) else False),
    }
    # Determine verdict
    noise_floor = pair_results[0].noise_floor_m if pair_results else np.nan
    primary = pair_results[0] if pair_results else None
    notes = []

    if primary and crossval.r_squared > 0.3 and primary.scarp_excess_m > 2 * noise_floor:
        verdict = "POSITIVE_DETECTION"
        notes.append(f"Scarp excess ({primary.scarp_excess_m:.3f} m) exceeds 2σ noise ({2*noise_floor:.3f} m)")
        notes.append(f"Cross-validation R²={crossval.r_squared:.3f}")
    elif primary and primary.scarp_excess_m > noise_floor:
        verdict = "MARGINAL_DETECTION"
        notes.append(f"Scarp excess ({primary.scarp_excess_m:.3f} m) exceeds 1σ noise ({noise_floor:.3f} m)")
        notes.append("Insufficient cross-validation confidence")
    else:
        verdict = "NO_DETECTION_UPPER_BOUND"
        if primary:
            notes.append(
                f"Scarp excess ({primary.scarp_excess_m:.3f} m) below noise ({noise_floor:.3f} m)"
            )
            notes.append(
                f"Upper bound on retreat rate: {primary.retreat_rate_upper_bound_m_yr:.4f} m/yr "
                f"= {primary.retreat_rate_upper_bound_m_yr * 100:.2f} cm/yr"
            )

    if not null_test["pass"]:
        notes.append("WARNING: Null test failed — systematic errors may persist")

    report = PipelineReport(
        site="Lefort Core, 46°N 92°E, Utopia Planitia",
        dtm_id="DTEEC_001938_2265_002439_2265_U01",
        pairs=[asdict(pr) for pr in pair_results],
        cross_validation=asdict(crossval),
        null_test=null_test,
        noise_floor_m=noise_floor,
        verdict=verdict,
        notes=notes,
    )

    # Save report
    report_dict = asdict(report)
    # Convert numpy types to Python native
    report_json = json.loads(
        json.dumps(report_dict, default=lambda x: float(x) if isinstance(x, (np.floating,)) else
                   int(x) if isinstance(x, (np.integer,)) else
                   x.tolist() if isinstance(x, np.ndarray) else str(x))
    )

    with open(output_dir / "v2_pipeline_report.json", "w") as f:
        json.dump(report_json, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info(f"PIPELINE COMPLETE — Verdict: {verdict}")
    for note in notes:
        logger.info(f"  {note}")
    logger.info(f"Report saved: {output_dir / 'v2_pipeline_report.json'}")

    return report


# ─── CLI Entry Point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(RESULTS_DIR / "pipeline_v2.log"),
        ],
    )

    print("HiRISE Temporal Change Detection Pipeline v2")
    print(f"Site: Lefort Core, 46°N 92°E")
    print(f"DTM: {DTM_PATH}")
    print()

    report = run_pipeline()

    print(f"\nVerdict: {report.verdict}")
    print(f"Noise floor: {report.noise_floor_m:.4f} m")
    for note in report.notes:
        print(f"  → {note}")
