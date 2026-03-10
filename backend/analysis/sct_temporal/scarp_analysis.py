"""
Scarp retreat rate extraction from displacement fields.

Identifies scarp edges in scalloped terrain, measures displacement
perpendicular to the scarp direction, and computes retreat rates
normalized by time gap.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple, cast

import numpy as np
from scipy import ndimage

from .phase_correlation import DisplacementField

logger = logging.getLogger(__name__)


@dataclass
class ScarpSegment:
    row_start: int
    col_start: int
    row_end: int
    col_end: int
    orientation_deg: float  # 0=N-S, 90=E-W
    length_m: float
    mean_retreat_m: float
    median_retreat_m: float
    max_retreat_m: float
    std_retreat_m: float
    n_valid_points: int


@dataclass
class RetreatAnalysis:
    segments: List[ScarpSegment]
    time_gap_mars_years: float
    mean_retreat_rate_m_per_yr: float
    median_retreat_rate_m_per_yr: float
    max_retreat_rate_m_per_yr: float
    total_scarp_length_m: float
    valid_measurement_pct: float
    displacement_field: DisplacementField

    @property
    def summary(self) -> dict[str, object]:
        return {
            "n_scarp_segments": len(self.segments),
            "time_gap_mars_years": round(self.time_gap_mars_years, 2),
            "mean_retreat_rate_m_per_yr": round(self.mean_retreat_rate_m_per_yr, 3),
            "median_retreat_rate_m_per_yr": round(self.median_retreat_rate_m_per_yr, 3),
            "max_retreat_rate_m_per_yr": round(self.max_retreat_rate_m_per_yr, 3),
            "total_scarp_length_m": round(self.total_scarp_length_m, 1),
            "valid_measurement_pct": round(self.valid_measurement_pct, 1),
            "pixel_scale_m": self.displacement_field.pixel_scale_m,
        }


def detect_scarps(
    img: np.ndarray,
    gradient_threshold: float = 0.15,
    min_segment_length_px: int = 10,
) -> np.ndarray:
    """
    Detect scarp edges in a HiRISE image using gradient magnitude.

    Returns a binary mask where True = scarp edge pixel.
    """
    img_norm = img.astype(np.float64)
    if img_norm.max() > img_norm.min():
        img_norm = (img_norm - img_norm.min()) / (img_norm.max() - img_norm.min())

    gy = ndimage.sobel(img_norm, axis=0)
    gx = ndimage.sobel(img_norm, axis=1)
    grad_mag = np.sqrt(gx**2 + gy**2)

    threshold = gradient_threshold * grad_mag.max()
    edge_mask = grad_mag > threshold

    labeled, n_features = cast(Tuple[np.ndarray, int], ndimage.label(edge_mask))
    for label_id in range(1, n_features + 1):
        component = labeled == label_id
        if np.sum(component) < min_segment_length_px:
            edge_mask[component] = False

    return edge_mask


def compute_scarp_orientation(edge_mask: np.ndarray) -> np.ndarray:
    """
    Compute local scarp orientation at each edge pixel.

    Uses structure tensor (gradient covariance) to estimate
    the dominant edge direction. Returns orientation in degrees (0-180).
    """
    gy = ndimage.sobel(edge_mask.astype(np.float64), axis=0)
    gx = ndimage.sobel(edge_mask.astype(np.float64), axis=1)

    sigma = 3.0
    Jxx = ndimage.gaussian_filter(gx * gx, sigma)
    Jxy = ndimage.gaussian_filter(gx * gy, sigma)
    Jyy = ndimage.gaussian_filter(gy * gy, sigma)

    orientation = 0.5 * np.arctan2(2 * Jxy, Jxx - Jyy)
    return np.degrees(orientation) % 180


def measure_retreat(
    displacement: DisplacementField,
    reference_image: np.ndarray,
    time_gap_mars_years: float,
    gradient_threshold: float = 0.15,
    min_segment_length_px: int = 10,
) -> RetreatAnalysis:
    """
    Measure scarp retreat rates from a displacement field.

    Steps:
    1. Detect scarp edges in the reference image
    2. For each scarp segment, project displacement perpendicular to scarp
    3. Normalize by time gap to get retreat rate

    Parameters
    ----------
    displacement : DisplacementField
        Output of sliding_window_correlation.
    reference_image : np.ndarray
        The T1 (reference) image, same dimensions as displacement inputs.
    time_gap_mars_years : float
        Time between observations in Mars years.
    gradient_threshold : float
        Edge detection sensitivity (fraction of max gradient).
    min_segment_length_px : int
        Minimum connected edge component size.
    """
    edge_mask = detect_scarps(reference_image, gradient_threshold, min_segment_length_px)
    orientation = compute_scarp_orientation(edge_mask)

    scale = displacement.pixel_scale_m
    step = displacement.row_centers[1] - displacement.row_centers[0] if len(displacement.row_centers) > 1 else 1

    labeled_edges, n_segments = cast(Tuple[np.ndarray, int], ndimage.label(edge_mask))
    segments: List[ScarpSegment] = []

    for seg_id in range(1, n_segments + 1):
        seg_pixels = np.argwhere(labeled_edges == seg_id)
        if len(seg_pixels) < min_segment_length_px:
            continue

        r_min, c_min = seg_pixels.min(axis=0)
        r_max, c_max = seg_pixels.max(axis=0)
        seg_length = np.sqrt((r_max - r_min)**2 + (c_max - c_min)**2) * scale

        perpendicular_displacements = []
        for r, c in seg_pixels:
            di = r // step if step > 0 else 0
            dj = c // step if step > 0 else 0
            di = min(di, displacement.row_disp.shape[0] - 1)
            dj = min(dj, displacement.row_disp.shape[1] - 1)

            if not displacement.valid_mask[di, dj]:
                continue

            dr = displacement.row_disp[di, dj] * scale
            dc = displacement.col_disp[di, dj] * scale
            theta = np.radians(orientation[r, c])

            perp = -dr * np.sin(theta) + dc * np.cos(theta)
            perpendicular_displacements.append(abs(perp))

        if len(perpendicular_displacements) < 3:
            continue

        retreats = np.array(perpendicular_displacements)
        segments.append(ScarpSegment(
            row_start=int(r_min),
            col_start=int(c_min),
            row_end=int(r_max),
            col_end=int(c_max),
            orientation_deg=float(np.median(orientation[labeled_edges == seg_id])),
            length_m=float(seg_length),
            mean_retreat_m=float(np.mean(retreats)),
            median_retreat_m=float(np.median(retreats)),
            max_retreat_m=float(np.max(retreats)),
            std_retreat_m=float(np.std(retreats)),
            n_valid_points=len(retreats),
        ))

    if not segments:
        logger.warning("No scarp segments detected with sufficient measurements")
        return RetreatAnalysis(
            segments=[],
            time_gap_mars_years=time_gap_mars_years,
            mean_retreat_rate_m_per_yr=0.0,
            median_retreat_rate_m_per_yr=0.0,
            max_retreat_rate_m_per_yr=0.0,
            total_scarp_length_m=0.0,
            valid_measurement_pct=0.0,
            displacement_field=displacement,
        )

    all_retreats = [s.mean_retreat_m for s in segments]
    total_length = sum(s.length_m for s in segments)
    total_points = sum(s.n_valid_points for s in segments)
    total_edge_pixels = int(edge_mask.sum())

    return RetreatAnalysis(
        segments=segments,
        time_gap_mars_years=time_gap_mars_years,
        mean_retreat_rate_m_per_yr=float(np.mean(all_retreats) / time_gap_mars_years) if time_gap_mars_years > 0 else 0.0,
        median_retreat_rate_m_per_yr=float(np.median(all_retreats) / time_gap_mars_years) if time_gap_mars_years > 0 else 0.0,
        max_retreat_rate_m_per_yr=float(np.max(all_retreats) / time_gap_mars_years) if time_gap_mars_years > 0 else 0.0,
        total_scarp_length_m=total_length,
        valid_measurement_pct=100.0 * total_points / max(total_edge_pixels, 1),
        displacement_field=displacement,
    )
