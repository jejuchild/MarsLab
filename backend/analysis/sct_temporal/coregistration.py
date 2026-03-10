"""
GeoTIFF-based co-registration for HiRISE temporal pairs.

Extracts overlapping regions using geographic coordinates, resamples
to a common grid, and optionally corrects residual parallax using
emission angle metadata and estimated surface slope.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple, cast

import numpy as np

logger = logging.getLogger(__name__)


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
) -> CoregisteredPair:
    """
    Co-register two HiRISE GeoTIFFs to a common grid.

    Uses geographic coordinates from GeoTIFF metadata to extract the
    overlapping region. Both images are resampled to the same pixel grid.

    Parameters
    ----------
    path1, path2 : Path
        Paths to GeoTIFF files (output of hirise_download).
    classification_mask : optional array
        Landform classification where 0=OTHER (stable terrain).
        Used to build stable_mask for bulk offset removal.
    target_resolution : optional float
        Force a specific pixel size (meters). If None, uses the
        finer resolution of the two inputs.
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject, calculate_default_transform

    with rasterio.open(path1) as ds1, rasterio.open(path2) as ds2:
        bounds1 = ds1.bounds
        bounds2 = ds2.bounds

        overlap_left = max(bounds1.left, bounds2.left)
        overlap_bottom = max(bounds1.bottom, bounds2.bottom)
        overlap_right = min(bounds1.right, bounds2.right)
        overlap_top = min(bounds1.top, bounds2.top)

        if overlap_left >= overlap_right or overlap_bottom >= overlap_top:
            raise ValueError(
                f"No spatial overlap between {path1.name} and {path2.name}. "
                f"Bounds1: {bounds1}, Bounds2: {bounds2}"
            )

        overlap_width = overlap_right - overlap_left
        overlap_height = overlap_top - overlap_bottom
        logger.info(
            f"Overlap region: {overlap_width:.4f}° × {overlap_height:.4f}° "
            f"({overlap_width * 59274:.0f}m × {overlap_height * 59274:.0f}m at 45°N)"
        )

        res1 = abs(ds1.transform.a)
        res2 = abs(ds2.transform.a)
        if target_resolution is not None:
            deg_per_m = 1.0 / 59274.0  # approximate at 45°N
            target_res = target_resolution * deg_per_m
        else:
            target_res = min(res1, res2)

        width = int((overlap_right - overlap_left) / target_res)
        height = int((overlap_top - overlap_bottom) / target_res)

        if width < 64 or height < 64:
            raise ValueError(
                f"Overlap too small: {width}×{height} px. Need at least 64×64."
            )

        from rasterio.transform import from_bounds
        dst_transform = from_bounds(
            overlap_left, overlap_bottom, overlap_right, overlap_top,
            width, height,
        )

        img1 = np.zeros((height, width), dtype=np.float32)
        img2 = np.zeros((height, width), dtype=np.float32)

        reproject(
            source=rasterio.band(ds1, 1),
            destination=img1,
            dst_transform=dst_transform,
            dst_crs=ds1.crs,
            resampling=Resampling.bilinear,
        )
        reproject(
            source=rasterio.band(ds2, 1),
            destination=img2,
            dst_transform=dst_transform,
            dst_crs=ds2.crs,
            resampling=Resampling.bilinear,
        )

        pixel_scale_m = target_res * 59274.0  # degrees to meters at ~45°N

    logger.info(f"Co-registered: {width}×{height} px, {pixel_scale_m:.3f} m/px")

    stable_mask = None
    if classification_mask is not None:
        from scipy.ndimage import zoom
        scale_r = height / classification_mask.shape[0]
        scale_c = width / classification_mask.shape[1]
        resized = zoom(classification_mask.astype(np.float32), (scale_r, scale_c), order=0)
        stable_mask = np.asarray(resized < 0.5)  # 0 = OTHER = stable

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
            "width": width,
            "height": height,
            "pixel_scale_deg": target_res,
        },
        pixel_scale_m=pixel_scale_m,
        overlap_bounds=overlap_bounds,
        stable_mask=cast(Optional[np.ndarray], stable_mask),
    )


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
