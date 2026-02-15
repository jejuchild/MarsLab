"""
Lobate Debris Apron (LDA) detection from MOLA 200 m/pixel DEM.

LDAs are ice-rich mass-wasting features found primarily at mid-latitudes
(30-60 deg N/S).  They appear as gently sloping, lobate surfaces that
originate at the base of steep escarpments (mesas, crater walls, scarps).

Detection algorithm:
    1. Identify low-slope regions adjacent to steep terrain.
    2. Filter by area, lobateness (area / convex-hull area), and
       optionally latitude.
    3. Return simplified boundary polygons for map rendering.

Dependencies: numpy, scipy.ndimage, scipy.spatial (ConvexHull), math,
              dataclasses.
"""

from __future__ import annotations

import logging
import math
import uuid
from typing import Callable, Optional

import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull

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

_LDA_LATITUDE_BANDS = ((30.0, 60.0), (-60.0, -30.0))  # N and S mid-latitudes


def _in_lda_latitude(lat: float) -> bool:
    """Return True if *lat* falls within a typical LDA latitude band."""
    for lo, hi in _LDA_LATITUDE_BANDS:
        if lo <= lat <= hi:
            return True
    return False


def _boundary_pixels(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract the outer boundary pixels of a binary mask.

    A boundary pixel is any True pixel that has at least one False
    4-connected neighbour.
    """
    eroded = ndimage.binary_erosion(mask, structure=ndimage.generate_binary_structure(2, 1))
    boundary = mask & ~eroded
    return np.where(boundary)


def _simplify_polygon(
    coords: list[list[float]],
    max_vertices: int = 80,
) -> list[list[float]]:
    """
    Downsample a polygon to at most *max_vertices* points using uniform
    index sampling while preserving the closure.
    """
    n = len(coords)
    if n <= max_vertices:
        return coords

    # Sample uniformly (indices), always keep first point
    indices = np.round(np.linspace(0, n - 1, max_vertices)).astype(int)
    simplified = [coords[i] for i in indices]

    # Ensure closure
    if simplified[0] != simplified[-1]:
        simplified.append(simplified[0])

    return simplified


def _order_boundary_pixels(
    rows: np.ndarray,
    cols: np.ndarray,
) -> list[tuple[int, int]]:
    """
    Order boundary pixels into a roughly sequential contour by nearest-
    neighbour traversal.  This is not a perfect contour tracer but is
    sufficient for producing a renderable polygon.
    """
    n = len(rows)
    if n == 0:
        return []
    if n <= 3:
        return list(zip(rows.tolist(), cols.tolist()))

    visited = np.zeros(n, dtype=bool)
    ordered: list[tuple[int, int]] = []

    # Start at the topmost-leftmost pixel
    start_idx = 0
    for i in range(n):
        if (rows[i], cols[i]) < (rows[start_idx], cols[start_idx]):
            start_idx = i

    current = start_idx
    for _ in range(n):
        visited[current] = True
        ordered.append((int(rows[current]), int(cols[current])))

        # Find nearest unvisited
        remaining = ~visited
        if not remaining.any():
            break

        dr = rows[remaining] - rows[current]
        dc = cols[remaining] - cols[current]
        dists = dr * dr + dc * dc
        nearest_local = np.argmin(dists)

        # Map back to global index
        remaining_indices = np.where(remaining)[0]
        current = remaining_indices[nearest_local]

    return ordered


def _compute_lobateness(rows: np.ndarray, cols: np.ndarray) -> float:
    """
    Compute lobateness = pixel_area / convex_hull_area.

    Returns a value in (0, 1]; values < 1 indicate non-convex (lobate) shapes.
    A perfectly convex region returns 1.0.
    """
    if len(rows) < 4:
        return 1.0

    points = np.column_stack((cols.astype(float), rows.astype(float)))

    # Remove duplicate points
    unique_points = np.unique(points, axis=0)
    if len(unique_points) < 4:
        return 1.0

    try:
        hull = ConvexHull(unique_points)
        hull_area = hull.volume  # In 2D, .volume gives area
    except Exception:
        return 1.0

    if hull_area < 1e-6:
        return 1.0

    pixel_area = float(len(rows))
    return min(pixel_area / hull_area, 1.0)


def _adjacency_fraction(
    component_mask: np.ndarray,
    steep_mask: np.ndarray,
    dilation_px: int = 3,
) -> float:
    """
    Fraction of the component boundary that is adjacent to steep terrain.

    Dilates the component boundary and measures overlap with *steep_mask*.
    """
    bnd_rows, bnd_cols = _boundary_pixels(component_mask)
    n_boundary = len(bnd_rows)
    if n_boundary == 0:
        return 0.0

    # For each boundary pixel, check if any steep pixel lies within
    # dilation_px Manhattan-approximate distance.  Use the dilated
    # steep_mask to vectorise the lookup: dilate the steep mask by the
    # same radius and check which boundary pixels fall inside.
    steep_dilated = ndimage.binary_dilation(
        steep_mask, structure=disk_footprint(dilation_px)
    )
    bnd_adjacent = int(steep_dilated[bnd_rows, bnd_cols].sum())

    return bnd_adjacent / n_boundary


# ---------------------------------------------------------------------------
# Main detection function
# ---------------------------------------------------------------------------

def detect_ldas(
    lat0: float,
    lon0: float,
    radius_km: float,
    min_area_km2: float = 10.0,
    latitude_filter: bool = True,
    on_progress: Optional[Callable] = None,
) -> list[DetectedFeature]:
    """
    Detect Lobate Debris Apron features in a MOLA DEM window.

    Parameters
    ----------
    lat0, lon0 : float
        Center latitude/longitude in degrees.
    radius_km : float
        Search radius in km.
    min_area_km2 : float
        Minimum LDA area in km^2.
    latitude_filter : bool
        If True, only retain candidates whose centroid falls in the
        30-60 deg N or S latitude band (typical for LDAs).
    on_progress : callable, optional
        ``(stage_name, info_dict)`` callback for UI progress reporting.

    Returns
    -------
    list[DetectedFeature]
        Detected LDAs sorted by descending confidence.
    """
    _progress = on_progress or (lambda *a: None)

    # 1. Extract DEM + slope
    _progress("extracting_dem", {})
    elev, meta = extract_dem_window(lat0, lon0, radius_km)
    px_m_ew = meta["px_m_ew"]
    px_m_ns = meta["px_m_ns"]

    if elev.size == 0:
        return []

    # Fill NaN for gradient computation
    elev_filled = elev.copy()
    nan_mask = np.isnan(elev_filled)
    if nan_mask.any():
        elev_filled[nan_mask] = np.nanmean(elev_filled)

    slope_deg = compute_slope_map(elev_filled, px_m_ns, px_m_ew)

    # 2. Slope masks
    _progress("computing_curvature", {})
    low_slope_mask = slope_deg < 5.0   # gentle apron surface
    steep_mask = slope_deg > 15.0       # escarpment / source cliff

    # 3. Label low-slope regions
    labels, n_labels = ndimage.label(low_slope_mask)
    _progress("detecting_ridges", {"n_candidates": int(n_labels)})

    if n_labels == 0:
        return []

    # Pixel area in km^2
    pixel_area_km2 = px_m_ew * px_m_ns / 1e6

    features: list[DetectedFeature] = []

    for label_id in range(1, n_labels + 1):
        component_mask = labels == label_id
        rows, cols = np.where(component_mask)

        if len(rows) < 10:
            continue

        # --- Area filter ---
        area_px = len(rows)
        area_km2 = area_px * pixel_area_km2

        if area_km2 < min_area_km2:
            continue

        # --- Centroid ---
        centroid_row = float(np.mean(rows))
        centroid_col = float(np.mean(cols))
        centroid_lat, centroid_lon = pixel_to_latlon(
            int(round(centroid_row)), int(round(centroid_col)), meta
        )

        # --- Latitude filter ---
        if latitude_filter and not _in_lda_latitude(centroid_lat):
            continue

        # --- Adjacency to steep terrain ---
        adj_frac = _adjacency_fraction(component_mask, steep_mask, dilation_px=3)
        if adj_frac < 0.15:
            continue

        # --- Lobateness ---
        lobateness = _compute_lobateness(rows, cols)
        if lobateness > 0.95:
            # Nearly convex; LDAs should be somewhat lobate / irregular
            # But we also allow lobateness > 0.4 (i.e., must fill at least
            # 40 % of its convex hull -- very lobate shapes score lower)
            pass
        if lobateness < 0.4:
            continue

        # --- Extract boundary polygon ---
        bnd_rows, bnd_cols = _boundary_pixels(component_mask)
        if len(bnd_rows) < 4:
            continue

        ordered = _order_boundary_pixels(bnd_rows, bnd_cols)

        boundary_latlon = []
        for r, c in ordered:
            lat, lon = pixel_to_latlon(r, c, meta)
            boundary_latlon.append([round(lat, 5), round(lon, 5)])

        # Close the polygon
        if boundary_latlon and boundary_latlon[0] != boundary_latlon[-1]:
            boundary_latlon.append(boundary_latlon[0])

        boundary_latlon = _simplify_polygon(boundary_latlon, max_vertices=80)

        # --- Dimensions ---
        # Bounding-box extent
        row_extent = (rows.max() - rows.min()) * px_m_ns / 1000.0
        col_extent = (cols.max() - cols.min()) * px_m_ew / 1000.0
        length_km = max(row_extent, col_extent)
        width_km = min(row_extent, col_extent)
        aspect_ratio = length_km / max(width_km, 0.01)

        # --- Elevation statistics ---
        component_elev = elev_filled[component_mask]
        mean_elev = float(np.nanmean(component_elev))
        elev_range = float(np.nanmax(component_elev) - np.nanmin(component_elev))

        # Surface slope statistics within the LDA
        component_slope = slope_deg[component_mask]
        mean_slope = float(np.nanmean(component_slope))

        # --- Confidence ---
        # Weighted composite of adjacency, lobateness, latitude, area
        adj_score = min(adj_frac / 0.5, 1.0)  # 50 % adjacency = max
        lob_score = 1.0 - lobateness  # more lobate (lower ratio) = higher score
        lob_score = max(lob_score, 0.0)
        lat_match = 1.0 if _in_lda_latitude(centroid_lat) else 0.3
        area_score = min(area_km2 / 200.0, 1.0)  # 200 km^2 = max

        confidence = (
            0.35 * adj_score
            + 0.25 * lob_score
            + 0.20 * lat_match
            + 0.20 * area_score
        )
        confidence = round(min(max(confidence, 0.0), 1.0), 3)

        # --- Morphology ---
        if adj_frac > 0.4 and lobateness < 0.7:
            morphology = "classic_lda"
        elif mean_slope < 2.0:
            morphology = "flat_apron"
        else:
            morphology = "lobate_apron"

        description = (
            f"LDA: area {area_km2:.1f} km2, "
            f"{length_km:.1f} x {width_km:.1f} km, "
            f"lobateness {lobateness:.2f}, "
            f"steep-terrain adjacency {adj_frac:.0%}, "
            f"mean slope {mean_slope:.1f} deg"
        )

        features.append(DetectedFeature(
            id=f"lda_{uuid.uuid4().hex[:8]}",
            feature_type="lda",
            lat=round(centroid_lat, 5),
            lon=round(centroid_lon, 5),
            area_km2=round(area_km2, 2),
            length_km=round(length_km, 2),
            width_km=round(width_km, 2),
            aspect_ratio=round(aspect_ratio, 2),
            depth_m=round(elev_range, 1),  # repurpose: elevation range within LDA
            rim_elevation_m=round(mean_elev, 1),  # repurpose: mean LDA elevation
            confidence=confidence,
            morphology=morphology,
            description=description,
            boundary=boundary_latlon,
            circularity=round(lobateness, 3),  # repurpose: lobateness metric
        ))

    features.sort(key=lambda f: f.confidence, reverse=True)
    logger.info(
        "detect_ldas: %d features found in %.1f km radius at (%.3f, %.3f)",
        len(features), radius_km, lat0, lon0,
    )
    return features
