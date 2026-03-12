"""
Co-registration for HiRISE temporal pairs.

Both images are in the same projected CRS (Mars equirectangular, meters)
with identical pixel scale (0.25 m). Uses efficient windowed reading to
extract only the overlapping region — no reprojection needed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple, cast

import numpy as np

logger = logging.getLogger(__name__)

MAX_OVERLAP_PX = 8192


@dataclass
class CoregisteredPair:
    img1: np.ndarray
    img2: np.ndarray
    transform: dict[str, Any]
    pixel_scale_m: float
    overlap_bounds: dict[str, float]
    stable_mask: Optional[np.ndarray] = None


def coregister_geotiffs(
    path1: Path,
    path2: Path,
    classification_mask: Optional[np.ndarray] = None,
    target_resolution: Optional[float] = None,
    max_size: int = MAX_OVERLAP_PX,
) -> CoregisteredPair:
    """
    Co-register two HiRISE images by extracting their overlapping region.

    Both images must share the same projected CRS and pixel scale.
    Uses windowed reading for efficiency (no full-image decompression).

    Parameters
    ----------
    path1, path2 : Path
        Paths to rasterio-readable files (JP2, GeoTIFF).
    classification_mask : optional array
        Landform classification where 0=OTHER (stable terrain).
    target_resolution : optional float
        Force specific pixel size (meters). Default: native resolution.
    max_size : int
        Maximum overlap dimension in pixels. If larger, center-crop.
        Default: 8192 (gives ~4000 correlation chips at step=128).
    """
    import rasterio
    from rasterio.windows import from_bounds

    with rasterio.open(path1) as ds1, rasterio.open(path2) as ds2:
        b1 = ds1.bounds
        b2 = ds2.bounds

        overlap_left = max(b1.left, b2.left)
        overlap_bottom = max(b1.bottom, b2.bottom)
        overlap_right = min(b1.right, b2.right)
        overlap_top = min(b1.top, b2.top)

        if overlap_left >= overlap_right or overlap_bottom >= overlap_top:
            raise ValueError(
                f"No spatial overlap: {path1.name} bounds={b1}, "
                f"{path2.name} bounds={b2}"
            )

        pixel_scale = abs(ds1.transform.a)
        is_projected = ds1.crs.is_projected if ds1.crs else False

        if is_projected:
            pixel_scale_m = pixel_scale
        else:
            pixel_scale_m = pixel_scale * 59274.0

        overlap_w = overlap_right - overlap_left
        overlap_h = overlap_top - overlap_bottom
        overlap_w_px = int(overlap_w / pixel_scale)
        overlap_h_px = int(overlap_h / pixel_scale)

        logger.info(
            f"Overlap: {overlap_w_px}x{overlap_h_px} px "
            f"({overlap_w:.0f}x{overlap_h:.0f} {'m' if is_projected else 'deg'}), "
            f"pixel_scale={pixel_scale_m:.3f} m/px"
        )

        if overlap_w_px > max_size or overlap_h_px > max_size:
            cx = (overlap_left + overlap_right) / 2
            cy = (overlap_bottom + overlap_top) / 2
            half_w = min(overlap_w_px, max_size) / 2 * pixel_scale
            half_h = min(overlap_h_px, max_size) / 2 * pixel_scale
            overlap_left = cx - half_w
            overlap_right = cx + half_w
            overlap_bottom = cy - half_h
            overlap_top = cy + half_h

            crop_w = int((overlap_right - overlap_left) / pixel_scale)
            crop_h = int((overlap_top - overlap_bottom) / pixel_scale)
            logger.info(f"Center-cropped to {crop_w}x{crop_h} px (max_size={max_size})")

        win1 = from_bounds(
            overlap_left, overlap_bottom, overlap_right, overlap_top,
            ds1.transform,
        )
        win2 = from_bounds(
            overlap_left, overlap_bottom, overlap_right, overlap_top,
            ds2.transform,
        )

        win1 = win1.round_offsets().round_lengths()
        win2 = win2.round_offsets().round_lengths()

        img1 = ds1.read(1, window=win1).astype(np.float32)
        img2 = ds2.read(1, window=win2).astype(np.float32)

    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])
    img1 = img1[:h, :w]
    img2 = img2[:h, :w]

    img1, img2 = _bulk_align(img1, img2, pixel_scale_m)

    logger.info(f"Co-registered: {img1.shape[1]}x{img1.shape[0]} px, {pixel_scale_m:.3f} m/px")

    stable_mask = None
    if classification_mask is not None:
        from scipy.ndimage import zoom

        scale_r = h / classification_mask.shape[0]
        scale_c = w / classification_mask.shape[1]
        resized = zoom(classification_mask.astype(np.float32), (scale_r, scale_c), order=0)
        stable_mask = np.asarray(resized < 0.5)

    overlap_bounds = {
        "left": overlap_left,
        "bottom": overlap_bottom,
        "right": overlap_right,
        "top": overlap_top,
    }

    return CoregisteredPair(
        img1=img1,
        img2=img2,
        transform={
            "width": w,
            "height": h,
            "pixel_scale_native": pixel_scale,
        },
        pixel_scale_m=pixel_scale_m,
        overlap_bounds=overlap_bounds,
        stable_mask=cast(Optional[np.ndarray], stable_mask),
    )


def _bulk_align(
    img1: np.ndarray,
    img2: np.ndarray,
    pixel_scale_m: float,
    patch_size: int = 1024,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove bulk translational offset between two geocoded images.

    HiRISE geocoding accuracy is ~10-60m, causing multi-pixel offsets
    that exceed the detection range of small phase-correlation chips.
    Uses FFT cross-correlation on a central patch to estimate and remove
    the integer-pixel offset, then crops to the aligned overlap.
    """
    from numpy.fft import fft2, ifft2

    H, W = img1.shape
    cy, cx = H // 2, W // 2
    half = patch_size // 2
    p1 = img1[cy - half:cy + half, cx - half:cx + half].astype(np.float64)
    p2 = img2[cy - half:cy + half, cx - half:cx + half].astype(np.float64)

    p1 = (p1 - p1.mean()) / max(p1.std(), 1e-6)
    p2 = (p2 - p2.mean()) / max(p2.std(), 1e-6)

    cc = np.abs(ifft2(fft2(p1) * np.conj(fft2(p2))))
    peak = np.unravel_index(np.argmax(cc), cc.shape)
    shift_r = float(peak[0])
    shift_c = float(peak[1])
    if shift_r > patch_size / 2:
        shift_r -= patch_size
    if shift_c > patch_size / 2:
        shift_c -= patch_size
    dr = int(round(shift_r))
    dc = int(round(shift_c))

    logger.info(
        f"Bulk alignment: ({dr}, {dc}) px = ({dr * pixel_scale_m:.1f}, {dc * pixel_scale_m:.1f}) m"
    )

    r1_start = max(0, dr)
    r2_start = max(0, -dr)
    c1_start = max(0, dc)
    c2_start = max(0, -dc)

    h = min(H - abs(dr), H)
    w = min(W - abs(dc), W)
    img1_aligned = img1[r1_start:r1_start + h, c1_start:c1_start + w]
    img2_aligned = img2[r2_start:r2_start + h, c2_start:c2_start + w]

    final_h = min(img1_aligned.shape[0], img2_aligned.shape[0])
    final_w = min(img1_aligned.shape[1], img2_aligned.shape[1])
    return img1_aligned[:final_h, :final_w], img2_aligned[:final_h, :final_w]


def estimate_parallax_correction(
    emission_angle_1: float,
    emission_angle_2: float,
    azimuth_1: float,
    azimuth_2: float,
    slope_angle: float,
    slope_aspect: float,
    depth_m: float = 10.0,
) -> Tuple[float, float]:
    """
    Estimate parallax displacement from viewing geometry difference.

    For a surface feature at given depth below a sloped surface,
    compute the expected apparent displacement between two viewing
    angles. Returns (row_px, col_px) correction to subtract from
    measured displacement.

    Parameters
    ----------
    emission_angle_1, emission_angle_2 : float
        Emission (viewing) angles in degrees.
    azimuth_1, azimuth_2 : float
        Spacecraft azimuth angles in degrees (0=N, 90=E).
    slope_angle : float
        Local surface slope in degrees.
    slope_aspect : float
        Slope facing direction in degrees (0=N, 90=E).
    depth_m : float
        Depth of the feature (scallop depression depth).
    """
    e1 = np.radians(emission_angle_1)
    e2 = np.radians(emission_angle_2)
    a1 = np.radians(azimuth_1)
    a2 = np.radians(azimuth_2)
    sa = np.radians(slope_aspect)

    dx1 = depth_m * np.tan(e1) * np.sin(a1 - sa)
    dx2 = depth_m * np.tan(e2) * np.sin(a2 - sa)
    dy1 = depth_m * np.tan(e1) * np.cos(a1 - sa)
    dy2 = depth_m * np.tan(e2) * np.cos(a2 - sa)

    parallax_ew = dx1 - dx2
    parallax_ns = dy1 - dy2

    return parallax_ns, parallax_ew
