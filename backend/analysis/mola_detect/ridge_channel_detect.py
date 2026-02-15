"""
Channel/valley and wrinkle-ridge detection from MOLA 200 m/pixel DEM.

Uses plan curvature (Laplacian) to identify concave (valley) and convex
(ridge) landforms, then labels connected regions, filters by geometry,
and returns simplified polylines / polygons suitable for map rendering.

Dependencies: numpy, scipy.ndimage, scipy.spatial, math, dataclasses.
"""

from __future__ import annotations

import logging
import math
import uuid
from typing import Callable, Optional

import numpy as np
from scipy import ndimage
from .common import (
    extract_dem_window,
    compute_slope_map,
    pixel_to_latlon,
    disk_footprint,
    pixels_to_km,
    km_to_pixels,
)
from .crater_detect import DetectedFeature

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fill_nan_with_mean(elev: np.ndarray) -> np.ndarray:
    """Replace NaN pixels with the array mean for gradient computation."""
    arr = elev.copy()
    mask = np.isnan(arr)
    if mask.any():
        arr[mask] = np.nanmean(arr)
    return arr


def _compute_laplacian(
    elev_smooth: np.ndarray,
    px_m_ew: float,
    px_m_ns: float,
) -> np.ndarray:
    """
    Compute the negative Laplacian of the elevation surface.

    Returns ``-(d2z/dx2 + d2z/dy2)``.
    Positive values indicate concave-up terrain (valleys);
    negative values indicate convex-up terrain (ridges).
    """
    # First derivatives (row = NS, col = EW)
    dz_dy = np.gradient(elev_smooth, px_m_ns, axis=0)
    dz_dx = np.gradient(elev_smooth, px_m_ew, axis=1)

    # Second derivatives
    d2z_dy2 = np.gradient(dz_dy, px_m_ns, axis=0)
    d2z_dx2 = np.gradient(dz_dx, px_m_ew, axis=1)

    laplacian = d2z_dx2 + d2z_dy2  # positive = convex, negative = concave
    return -laplacian  # flip sign: positive = valley, negative = ridge


def _inertia_axes(rows: np.ndarray, cols: np.ndarray):
    """
    Compute major/minor axis lengths and orientation from pixel coordinates
    using the eigenvalues of the 2x2 inertia tensor.

    Returns (major_len_px, minor_len_px, angle_rad).
    """
    if len(rows) < 3:
        return 0.0, 0.0, 0.0

    cr = np.mean(rows)
    cc = np.mean(cols)
    dr = rows - cr
    dc = cols - cc

    Irr = np.sum(dc ** 2)
    Icc = np.sum(dr ** 2)
    Irc = -np.sum(dr * dc)

    tensor = np.array([[Irr, Irc], [Irc, Icc]])
    eigvals, eigvecs = np.linalg.eigh(tensor)

    # Sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    n = len(rows)
    major = 4.0 * math.sqrt(max(eigvals[0] / n, 0))
    minor = 4.0 * math.sqrt(max(eigvals[1] / n, 0))
    angle = math.atan2(eigvecs[1, 0], eigvecs[0, 0])

    return major, minor, angle


def _extract_centerline(
    rows: np.ndarray,
    cols: np.ndarray,
    angle_rad: float,
    major_len_px: float,
) -> list[tuple[float, float]]:
    """
    Compute a simplified centerline through an elongated component.

    Projects all component pixels onto the major axis, bins them, and
    returns the median transverse position per bin as the centerline.
    """
    cr = np.mean(rows)
    cc = np.mean(cols)

    # Unit vector along major axis (in row/col space)
    u_row = math.sin(angle_rad)
    u_col = math.cos(angle_rad)

    # Project each pixel onto major axis
    dr = rows - cr
    dc = cols - cc
    proj = dr * u_row + dc * u_col  # signed distance along major axis

    # Number of bins: aim for ~40 segments
    n_bins = min(50, max(10, int(major_len_px / 3)))
    proj_min, proj_max = proj.min(), proj.max()
    if proj_max - proj_min < 1e-6:
        return [(cr, cc)]

    bin_edges = np.linspace(proj_min, proj_max, n_bins + 1)

    centerline = []
    for i in range(n_bins):
        mask = (proj >= bin_edges[i]) & (proj < bin_edges[i + 1])
        if i == n_bins - 1:
            mask |= proj == bin_edges[i + 1]
        if mask.sum() == 0:
            continue
        centerline.append((np.median(rows[mask]), np.median(cols[mask])))

    return centerline


def _simplify_path(
    coords: list[list[float]],
    max_vertices: int = 50,
) -> list[list[float]]:
    """
    Downsample a polyline to at most *max_vertices* points using uniform
    index sampling while preserving endpoints.
    """
    n = len(coords)
    if n <= max_vertices:
        return coords

    indices = np.round(np.linspace(0, n - 1, max_vertices)).astype(int)
    return [coords[i] for i in indices]


def _path_length(coords: list[list[float]]) -> float:
    """
    Geodesic-approximated path length in km from a list of [lat, lon] pairs.

    Uses the Mars mean radius for a spherical approximation.
    """
    MARS_R_KM = 3389.5
    total = 0.0
    for i in range(1, len(coords)):
        lat1, lon1 = math.radians(coords[i - 1][0]), math.radians(coords[i - 1][1])
        lat2, lon2 = math.radians(coords[i][0]), math.radians(coords[i][1])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(max(1 - a, 0)))
        total += MARS_R_KM * c
    return total


def _straight_line_km(coords: list[list[float]]) -> float:
    """Straight-line distance in km between first and last coordinate."""
    if len(coords) < 2:
        return 0.0
    return _path_length([coords[0], coords[-1]])


# ---------------------------------------------------------------------------
# Channel / valley detection
# ---------------------------------------------------------------------------

def detect_channels(
    lat0: float,
    lon0: float,
    radius_km: float,
    min_length_km: float = 5.0,
    on_progress: Optional[Callable] = None,
) -> list[DetectedFeature]:
    """
    Detect channel / valley features in a MOLA DEM window.

    Parameters
    ----------
    lat0, lon0 : float
        Center latitude/longitude in degrees.
    radius_km : float
        Search radius in km.
    min_length_km : float
        Minimum major-axis length to keep a candidate.
    on_progress : callable, optional
        ``(stage_name, info_dict)`` callback for UI progress reporting.

    Returns
    -------
    list[DetectedFeature]
        Detected channels sorted by descending confidence.
    """
    _progress = on_progress or (lambda *a: None)

    # 1. Extract DEM
    _progress("extracting_dem", {})
    elev, meta = extract_dem_window(lat0, lon0, radius_km)
    px_m_ew = meta["px_m_ew"]
    px_m_ns = meta["px_m_ns"]

    if elev.size == 0:
        return []

    # 2. Fill NaN
    elev_filled = _fill_nan_with_mean(elev)

    # 3. Gaussian smooth (sigma = 2 px)
    elev_smooth = ndimage.gaussian_filter(elev_filled, sigma=2.0)

    # 4. Compute curvature (negative Laplacian)
    _progress("computing_curvature", {})
    neg_laplacian = _compute_laplacian(elev_smooth, px_m_ew, px_m_ns)

    # 5. Threshold for valleys (neg_laplacian > 0 = valley)
    valley_values = neg_laplacian[neg_laplacian > 0]
    if valley_values.size == 0:
        return []
    threshold = np.percentile(valley_values, 85)  # top 15% of valley signal
    valley_mask = neg_laplacian > threshold

    # 6. Morphological cleanup
    valley_mask = ndimage.binary_erosion(valley_mask, iterations=1)
    valley_mask = ndimage.binary_dilation(valley_mask, iterations=1)

    # 7. Label connected components
    labels, n_labels = ndimage.label(valley_mask)
    _progress("detecting_channels", {"n_candidates": int(n_labels)})

    if n_labels == 0:
        return []

    # 8. Analyse each component
    features: list[DetectedFeature] = []

    for label_id in range(1, n_labels + 1):
        component_mask = labels == label_id
        rows, cols = np.where(component_mask)

        if len(rows) < 10:
            continue

        # Bounding box area in km^2
        area_px = len(rows)
        area_km2 = area_px * px_m_ew * px_m_ns / 1e6

        # Principal axes
        major_px, minor_px, angle = _inertia_axes(rows, cols)
        if minor_px < 1e-6:
            continue

        aspect_ratio = major_px / max(minor_px, 1e-6)
        major_km = pixels_to_km(major_px, (px_m_ew + px_m_ns) / 2)
        minor_km = pixels_to_km(minor_px, (px_m_ew + px_m_ns) / 2)

        # Filter: must be elongated and long enough
        if aspect_ratio < 3.0:
            continue
        if major_km < min_length_km:
            continue

        # Centerline path in pixel space
        centerline_px = _extract_centerline(rows, cols, angle, major_px)
        if len(centerline_px) < 2:
            continue

        # Convert to lat/lon
        path_latlon = []
        for r, c in centerline_px:
            lat, lon = pixel_to_latlon(int(round(r)), int(round(c)), meta)
            path_latlon.append([round(lat, 5), round(lon, 5)])

        path_latlon = _simplify_path(path_latlon, max_vertices=50)

        # Sinuosity
        total_len_km = _path_length(path_latlon)
        straight_km = _straight_line_km(path_latlon)
        sinuosity = total_len_km / max(straight_km, 0.01)

        # Depth: median elevation deficit within the channel relative to edges
        component_elev = elev_filled[component_mask]
        # Dilate the component to get surrounding terrain
        surround_mask = ndimage.binary_dilation(
            component_mask, structure=disk_footprint(3)
        ) & ~component_mask
        if surround_mask.any():
            rim_elev = float(np.nanmedian(elev_filled[surround_mask]))
            floor_elev = float(np.nanmedian(component_elev))
            depth_m = max(rim_elev - floor_elev, 0.0)
        else:
            rim_elev = float(np.nanmedian(component_elev))
            floor_elev = rim_elev
            depth_m = 0.0

        # Centroid lat/lon
        centroid_row = float(np.mean(rows))
        centroid_col = float(np.mean(cols))
        centroid_lat, centroid_lon = pixel_to_latlon(
            int(round(centroid_row)), int(round(centroid_col)), meta
        )

        # Confidence scoring
        # Factors: aspect ratio (max contribution at AR>=6), length, depth, sinuosity
        ar_score = min(aspect_ratio / 8.0, 1.0)
        len_score = min(total_len_km / 50.0, 1.0)
        depth_score = min(depth_m / 200.0, 1.0)
        sin_score = min(sinuosity / 2.0, 1.0) if sinuosity > 1.0 else 0.3
        confidence = 0.35 * ar_score + 0.30 * len_score + 0.20 * depth_score + 0.15 * sin_score
        confidence = round(min(max(confidence, 0.0), 1.0), 3)

        # Morphology description
        if sinuosity > 1.5:
            morphology = "sinuous_channel"
        elif depth_m > 100:
            morphology = "incised_valley"
        else:
            morphology = "linear_channel"

        description = (
            f"Channel/valley: length {total_len_km:.1f} km, width ~{minor_km:.1f} km, "
            f"depth ~{depth_m:.0f} m, sinuosity {sinuosity:.2f}"
        )

        features.append(DetectedFeature(
            id=f"ch_{uuid.uuid4().hex[:8]}",
            feature_type="channel",
            lat=round(centroid_lat, 5),
            lon=round(centroid_lon, 5),
            length_km=round(total_len_km, 2),
            width_km=round(minor_km, 2),
            aspect_ratio=round(aspect_ratio, 2),
            depth_m=round(depth_m, 1),
            rim_elevation_m=round(rim_elev, 1),
            floor_elevation_m=round(floor_elev, 1),
            sinuosity=round(sinuosity, 3),
            confidence=confidence,
            morphology=morphology,
            description=description,
            path=path_latlon,
            area_km2=round(area_km2, 2),
        ))

    # Sort by confidence descending
    features.sort(key=lambda f: f.confidence, reverse=True)
    logger.info(
        "detect_channels: %d features found in %.1f km radius at (%.3f, %.3f)",
        len(features), radius_km, lat0, lon0,
    )
    return features


# ---------------------------------------------------------------------------
# Wrinkle ridge detection
# ---------------------------------------------------------------------------

def detect_ridges(
    lat0: float,
    lon0: float,
    radius_km: float,
    min_length_km: float = 5.0,
    on_progress: Optional[Callable] = None,
) -> list[DetectedFeature]:
    """
    Detect wrinkle-ridge features in a MOLA DEM window.

    Wrinkle ridges are low-relief, elongated convexities on relatively
    flat volcanic plains.  They are identified by positive Laplacian
    (convex surface) combined with low background slope.

    Parameters
    ----------
    lat0, lon0 : float
        Center latitude/longitude in degrees.
    radius_km : float
        Search radius in km.
    min_length_km : float
        Minimum major-axis length to keep a candidate.
    on_progress : callable, optional
        ``(stage_name, info_dict)`` callback for UI progress reporting.

    Returns
    -------
    list[DetectedFeature]
        Detected ridges sorted by descending confidence.
    """
    _progress = on_progress or (lambda *a: None)

    # 1. Extract DEM + slope
    _progress("extracting_dem", {})
    elev, meta = extract_dem_window(lat0, lon0, radius_km)
    px_m_ew = meta["px_m_ew"]
    px_m_ns = meta["px_m_ns"]

    if elev.size == 0:
        return []

    elev_filled = _fill_nan_with_mean(elev)
    slope_deg = compute_slope_map(elev_filled, px_m_ns, px_m_ew)

    # 2. Gaussian smooth (sigma = 2 px)
    elev_smooth = ndimage.gaussian_filter(elev_filled, sigma=2.0)

    # 3. Compute curvature
    _progress("computing_curvature", {})
    neg_laplacian = _compute_laplacian(elev_smooth, px_m_ew, px_m_ns)

    # Ridges: neg_laplacian < 0 (convex up)
    ridge_values = -neg_laplacian  # positive = convex
    convex_vals = ridge_values[ridge_values > 0]
    if convex_vals.size == 0:
        return []
    threshold = np.percentile(convex_vals, 85)  # top 15% of convexity

    # 4. Ridge mask: strong convexity AND low local slope (flat plains)
    ridge_mask = (ridge_values > threshold) & (slope_deg < 5.0)

    # Morphological cleanup
    ridge_mask = ndimage.binary_erosion(ridge_mask, iterations=1)
    ridge_mask = ndimage.binary_dilation(ridge_mask, iterations=1)

    # 5. Label connected components
    labels, n_labels = ndimage.label(ridge_mask)
    _progress("detecting_ridges", {"n_candidates": int(n_labels)})

    if n_labels == 0:
        return []

    # 6. Neighbourhood slope check (mean slope in a window around each component)
    # Pre-compute a smoothed slope for neighbourhood checks
    smooth_slope = ndimage.uniform_filter(slope_deg, size=max(11, km_to_pixels(5, (px_m_ew + px_m_ns) / 2)))

    features: list[DetectedFeature] = []

    for label_id in range(1, n_labels + 1):
        component_mask = labels == label_id
        rows, cols = np.where(component_mask)

        if len(rows) < 10:
            continue

        area_px = len(rows)
        area_km2 = area_px * px_m_ew * px_m_ns / 1e6

        # Principal axes
        major_px, minor_px, angle = _inertia_axes(rows, cols)
        if minor_px < 1e-6:
            continue

        aspect_ratio = major_px / max(minor_px, 1e-6)
        major_km = pixels_to_km(major_px, (px_m_ew + px_m_ns) / 2)
        minor_km = pixels_to_km(minor_px, (px_m_ew + px_m_ns) / 2)

        # Filter: elongated and long enough
        if aspect_ratio < 3.0:
            continue
        if major_km < min_length_km:
            continue

        # Local relief check: relief within component must be < 500 m
        component_elev = elev_filled[component_mask]
        local_relief = float(np.nanmax(component_elev) - np.nanmin(component_elev))
        if local_relief > 500.0:
            continue

        # Neighbourhood slope check: mean slope around the feature must be < 5 deg
        mean_nbr_slope = float(np.nanmean(smooth_slope[component_mask]))
        if mean_nbr_slope > 5.0:
            continue

        # Centerline
        centerline_px = _extract_centerline(rows, cols, angle, major_px)
        if len(centerline_px) < 2:
            continue

        path_latlon = []
        for r, c in centerline_px:
            lat, lon = pixel_to_latlon(int(round(r)), int(round(c)), meta)
            path_latlon.append([round(lat, 5), round(lon, 5)])

        path_latlon = _simplify_path(path_latlon, max_vertices=50)

        # Path metrics
        total_len_km = _path_length(path_latlon)
        straight_km = _straight_line_km(path_latlon)
        sinuosity = total_len_km / max(straight_km, 0.01)

        # Ridge height: elevation difference between ridge crest and flanking terrain
        surround_mask = ndimage.binary_dilation(
            component_mask, structure=disk_footprint(3)
        ) & ~component_mask
        if surround_mask.any():
            crest_elev = float(np.nanmedian(component_elev))
            flank_elev = float(np.nanmedian(elev_filled[surround_mask]))
            height_m = max(crest_elev - flank_elev, 0.0)
        else:
            crest_elev = float(np.nanmedian(component_elev))
            flank_elev = crest_elev
            height_m = 0.0

        # Centroid
        centroid_row = float(np.mean(rows))
        centroid_col = float(np.mean(cols))
        centroid_lat, centroid_lon = pixel_to_latlon(
            int(round(centroid_row)), int(round(centroid_col)), meta
        )

        # Confidence
        ar_score = min(aspect_ratio / 8.0, 1.0)
        len_score = min(total_len_km / 50.0, 1.0)
        height_score = min(height_m / 100.0, 1.0)
        flat_score = max(0.0, 1.0 - mean_nbr_slope / 5.0)
        confidence = (
            0.30 * ar_score
            + 0.25 * len_score
            + 0.20 * height_score
            + 0.25 * flat_score
        )
        confidence = round(min(max(confidence, 0.0), 1.0), 3)

        # Morphology
        if height_m > 50:
            morphology = "prominent_ridge"
        elif sinuosity > 1.3:
            morphology = "sinuous_ridge"
        else:
            morphology = "linear_ridge"

        description = (
            f"Wrinkle ridge: length {total_len_km:.1f} km, width ~{minor_km:.1f} km, "
            f"height ~{height_m:.0f} m, on plains with mean slope {mean_nbr_slope:.1f} deg"
        )

        features.append(DetectedFeature(
            id=f"wr_{uuid.uuid4().hex[:8]}",
            feature_type="wrinkle_ridge",
            lat=round(centroid_lat, 5),
            lon=round(centroid_lon, 5),
            length_km=round(total_len_km, 2),
            width_km=round(minor_km, 2),
            aspect_ratio=round(aspect_ratio, 2),
            depth_m=round(height_m, 1),  # repurpose depth_m for ridge height
            rim_elevation_m=round(crest_elev, 1),
            floor_elevation_m=round(flank_elev, 1),
            sinuosity=round(sinuosity, 3),
            confidence=confidence,
            morphology=morphology,
            description=description,
            path=path_latlon,
            area_km2=round(area_km2, 2),
        ))

    features.sort(key=lambda f: f.confidence, reverse=True)
    logger.info(
        "detect_ridges: %d features found in %.1f km radius at (%.3f, %.3f)",
        len(features), radius_km, lat0, lon0,
    )
    return features
