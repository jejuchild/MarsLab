"""
Crater, terraced crater, volcanic construct, and graben detection from MOLA DEM.

Uses multi-scale morphological closing on the 200 m/pixel MOLA DEM to detect
negative-relief (craters, graben) and positive-relief (volcanic constructs)
landforms.  Craters >10 km are further classified via radial-profile terrace
analysis adapted from the HiRISE-scale terrace detector.

Dependencies: numpy, scipy.ndimage, math, dataclasses (no scikit-image).
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from scipy.ndimage import gaussian_filter, grey_closing, label

from .common import (
    SharedDEMContext,
    compute_slope_map,
    disk_footprint,
    extract_dem_window,
    km_to_pixels,
    pixel_to_latlon,
    pixels_to_km,
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class DetectedFeature:
    """Universal feature descriptor for MOLA landform detections."""

    id: str
    feature_type: str  # "crater", "terraced_crater", "volcanic", "graben"
    lat: float
    lon: float
    diameter_km: float = 0.0
    depth_m: float = 0.0
    depth_diameter_ratio: float = 0.0
    circularity: float = 0.0
    rim_elevation_m: float = 0.0
    floor_elevation_m: float = 0.0
    n_terraces: int = 0
    morphology: str = ""  # "simple", "complex", "terraced", "shield", "dome"
    confidence: float = 0.0
    description: str = ""
    length_km: float = 0.0  # for graben / polylines
    width_km: float = 0.0  # for graben
    aspect_ratio: float = 0.0  # for graben
    path: list = field(default_factory=list)  # [[lat,lon], ...] for polylines
    area_km2: float = 0.0
    boundary: list = field(default_factory=list)  # [[lat,lon], ...] for polygons
    sinuosity: float = 0.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

ProgressCallback = Optional[Callable[[str, dict], None]]

# Morphological closing scales (disk radii in pixels).
# At 200 m/px these target ~5 km, ~15 km, and ~30 km diameter craters.
_CLOSING_RADII_PX = (12, 37, 75)

# Minimum connected-component pixel count to consider a candidate.
_MIN_COMPONENT_PX = 50

# Terrace detection constants (MOLA scale)
_TERRACE_N_AZIMUTHS = 18
_TERRACE_STEP_M = 200.0  # 1 pixel at 200 m/px
_TERRACE_SMOOTH_WIN = 5  # pixels
_TERRACE_SLOPE_THRESH = 0.005  # dimensionless
_TERRACE_MIN_BENCH_PX = 3  # 600 m
_TERRACE_INNER_FRAC = 0.30
_TERRACE_OUTER_FRAC = 0.85


def _emit(on_progress: ProgressCallback, phase: str, data: dict) -> None:
    """Safely call the progress callback if provided."""
    if on_progress is not None:
        try:
            on_progress(phase, data)
        except Exception:
            pass


def _uid() -> str:
    """Short unique id for a detected feature."""
    return uuid.uuid4().hex[:12]


def _smooth_1d(arr: np.ndarray, window: int = 5) -> np.ndarray:
    """Moving-average smooth of a 1-D array, NaN-aware, vectorized."""
    valid = np.where(np.isnan(arr), 0.0, arr)
    counts = np.where(np.isnan(arr), 0.0, 1.0)
    kernel = np.ones(window)
    sum_vals = np.convolve(valid, kernel, mode="same")
    sum_counts = np.convolve(counts, kernel, mode="same")
    out = np.where(sum_counts > 0, sum_vals / sum_counts, arr)
    return out


def _component_properties(
    labels: np.ndarray,
    depth_map: np.ndarray,
    elev_smooth: np.ndarray,
    label_id: int,
) -> Optional[dict]:
    """Compute geometric / topographic properties for one labelled region.

    Returns *None* if the component is too small or otherwise degenerate.
    """
    mask = labels == label_id
    n_pixels = int(np.sum(mask))
    if n_pixels < _MIN_COMPONENT_PX:
        return None

    rows, cols = np.where(mask)

    # Bounding box
    r_min, r_max = int(rows.min()), int(rows.max())
    c_min, c_max = int(cols.min()), int(cols.max())
    bb_height = r_max - r_min + 1
    bb_width = c_max - c_min + 1

    # Centroid
    centroid_r = float(np.mean(rows))
    centroid_c = float(np.mean(cols))

    # Area-based equivalent circle diameter (in pixels)
    eq_diameter_px = math.sqrt(4.0 * n_pixels / math.pi)

    # Perimeter estimation: count boundary pixels (pixels with at least one
    # 4-connected neighbour outside the mask).
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    interior = (
        padded[1:-1, 1:-1]
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    boundary = mask & ~interior
    perimeter_px = float(np.sum(boundary))

    # Circularity = 4*pi*area / perimeter^2 (1.0 for a perfect circle)
    if perimeter_px > 0:
        circularity = (4.0 * math.pi * n_pixels) / (perimeter_px ** 2)
    else:
        circularity = 0.0
    circularity = min(circularity, 1.0)

    # Elevations
    floor_elev = float(np.nanmin(elev_smooth[mask]))
    # Rim: max elevation in boundary pixels
    if np.any(boundary):
        rim_elev = float(np.nanmax(elev_smooth[boundary]))
    else:
        rim_elev = floor_elev
    depth = rim_elev - floor_elev

    # Aspect ratio of bounding box (major / minor)
    major = max(bb_height, bb_width)
    minor = max(1, min(bb_height, bb_width))
    aspect_ratio = major / minor

    return {
        "n_pixels": n_pixels,
        "centroid_r": centroid_r,
        "centroid_c": centroid_c,
        "r_min": r_min,
        "r_max": r_max,
        "c_min": c_min,
        "c_max": c_max,
        "bb_height": bb_height,
        "bb_width": bb_width,
        "eq_diameter_px": eq_diameter_px,
        "perimeter_px": perimeter_px,
        "circularity": circularity,
        "floor_elev": floor_elev,
        "rim_elev": rim_elev,
        "depth": depth,
        "aspect_ratio": aspect_ratio,
        "major_axis_px": major,
        "minor_axis_px": minor,
        "mask": mask,
        "boundary": boundary,
    }


_DOWNSAMPLE_THRESHOLD = 40  # Kernels larger than this use 2x downsampled DEM


def _depression_depth_map(
    elev_smooth: np.ndarray,
    closing_radii: tuple[int, ...] = _CLOSING_RADII_PX,
) -> np.ndarray:
    """Compute max depression depth across multiple closing scales.

    For each scale the morphological closing fills depressions up to the
    disk size; the residual (closed - input) gives depression depth.

    For large kernels (radius > 40 px), uses 2x downsampled DEM to reduce
    computational cost by ~4x, then upsamples the result back.
    """
    from scipy.ndimage import zoom

    depth_max = np.zeros_like(elev_smooth)
    for r_px in closing_radii:
        if r_px > _DOWNSAMPLE_THRESHOLD:
            # 2x downsample: halve kernel and array, ~4x fewer operations
            ds_factor = 2
            small = zoom(elev_smooth, 1.0 / ds_factor, order=1)
            small_r = max(1, r_px // ds_factor)
            footprint = disk_footprint(small_r)
            closed_small = grey_closing(small, footprint=footprint)
            residual_small = closed_small - small
            residual = zoom(residual_small, ds_factor, order=1)
            # Trim/pad to match original shape
            residual = residual[:elev_smooth.shape[0], :elev_smooth.shape[1]]
            if residual.shape != elev_smooth.shape:
                padded = np.zeros_like(elev_smooth)
                h = min(residual.shape[0], elev_smooth.shape[0])
                w = min(residual.shape[1], elev_smooth.shape[1])
                padded[:h, :w] = residual[:h, :w]
                residual = padded
        else:
            footprint = disk_footprint(r_px)
            closed = grey_closing(elev_smooth, footprint=footprint)
            residual = closed - elev_smooth
        depth_max = np.maximum(depth_max, residual)
    return depth_max


def _classify_terraces(
    elev: np.ndarray,
    centroid_r: float,
    centroid_c: float,
    radius_px: float,
    px_m: float,
) -> int:
    """Radial-profile terrace detection at MOLA scale.

    Returns the number of detected terrace benches (0 if none).

    Algorithm mirrors the HiRISE terrace detector but uses parameters tuned
    for the coarser 200 m/px MOLA DEM.
    """
    nrows, ncols = elev.shape
    step_px = 1  # 200 m per pixel
    max_r_px = int(radius_px * 1.1)  # sample slightly beyond estimated rim
    if max_r_px < 3:
        return 0

    azimuths = np.linspace(0, 2 * math.pi, _TERRACE_N_AZIMUTHS, endpoint=False)
    n_samples = max(1, max_r_px // step_px)
    all_profiles = np.full((_TERRACE_N_AZIMUTHS, n_samples), np.nan)

    for i, az in enumerate(azimuths):
        cos_a = math.cos(az)
        sin_a = math.sin(az)
        for j in range(n_samples):
            r = j * step_px
            ri = int(round(centroid_r - r * cos_a))
            ci = int(round(centroid_c + r * sin_a))
            if 0 <= ri < nrows and 0 <= ci < ncols:
                val = elev[ri, ci]
                if not np.isnan(val):
                    all_profiles[i, j] = val

    # Median profile across azimuths
    with np.errstate(all="ignore"):
        median_prof = np.nanmedian(all_profiles, axis=0)

    if np.all(np.isnan(median_prof)):
        return 0

    # Smooth
    smooth = _smooth_1d(median_prof, window=_TERRACE_SMOOTH_WIN)

    # Compute slope (dimensionless: dz per step distance)
    step_m = _TERRACE_STEP_M
    slope = np.gradient(smooth, step_m)

    # Search bounds: between 30% and 85% of crater radius in pixels
    inner_idx = max(0, int(_TERRACE_INNER_FRAC * radius_px / step_px))
    outer_idx = min(n_samples, int(_TERRACE_OUTER_FRAC * radius_px / step_px))

    if outer_idx <= inner_idx:
        return 0

    # Find contiguous flat segments
    is_flat = np.abs(slope) < _TERRACE_SLOPE_THRESH
    benches: list[tuple[int, int]] = []
    in_seg = False
    seg_start = 0
    for k in range(inner_idx, outer_idx):
        if is_flat[k]:
            if not in_seg:
                seg_start = k
                in_seg = True
        else:
            if in_seg:
                benches.append((seg_start, k - 1))
                in_seg = False
    if in_seg:
        benches.append((seg_start, outer_idx - 1))

    # Filter: minimum bench width
    benches = [(s, e) for s, e in benches if (e - s + 1) >= _TERRACE_MIN_BENCH_PX]

    return len(benches)


def _assign_confidence(
    circularity: float,
    depth_diameter_ratio: float,
    diameter_km: float,
) -> float:
    """Heuristic confidence score (0-1) for a crater detection.

    Rewards high circularity and d/D ratios consistent with known Martian
    craters.  Typical fresh simple craters have d/D ~ 0.1-0.2; complex
    craters are shallower (0.01-0.05).
    """
    # Circularity component: 0 at 0.55, 1 at 0.95+
    circ_score = min(1.0, max(0.0, (circularity - 0.55) / 0.40))

    # d/D consistency: simple craters peak near 0.15, complex near 0.03
    if diameter_km < 10:
        ideal_dd = 0.15
    elif diameter_km < 50:
        ideal_dd = 0.06
    else:
        ideal_dd = 0.03
    dd_diff = abs(depth_diameter_ratio - ideal_dd)
    dd_score = max(0.0, 1.0 - dd_diff / ideal_dd) if ideal_dd > 0 else 0.5

    return round(min(1.0, 0.55 * circ_score + 0.45 * dd_score), 3)


def _assign_volcanic_confidence(
    circularity: float,
    mean_flank_slope: float,
    height_m: float,
) -> float:
    """Heuristic confidence score for a volcanic construct detection."""
    circ_score = min(1.0, max(0.0, (circularity - 0.55) / 0.40))
    # Prefer moderate slopes (2-12 deg); penalise very flat or very steep
    slope_score = max(0.0, 1.0 - abs(mean_flank_slope - 7.0) / 10.0)
    # Prefer substantial relief
    height_score = min(1.0, height_m / 2000.0)
    return round(min(1.0, 0.35 * circ_score + 0.35 * slope_score + 0.30 * height_score), 3)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_craters_and_volcanics(
    lat0: float,
    lon0: float,
    radius_km: float,
    min_diameter_km: float = 5.0,
    min_depth_m: float = 100.0,
    on_progress: ProgressCallback = None,
    ctx: SharedDEMContext | None = None,
) -> list[DetectedFeature]:
    """Detect craters, terraced craters, volcanic constructs, and graben.

    Parameters
    ----------
    lat0, lon0 : float
        Center of the search window (degrees).
    radius_km : float
        Search radius in km.
    min_diameter_km : float
        Minimum equivalent-circle diameter to keep for craters / volcanics.
    min_depth_m : float
        Minimum depression depth (metres) to seed candidate labelling.
    on_progress : callable, optional
        ``on_progress(phase: str, data: dict)`` called at key stages.
    ctx : SharedDEMContext, optional
        Pre-computed DEM data. If provided, skips DEM extraction and
        derivative computation. Dramatically faster for multi-detector scans.

    Returns
    -------
    list[DetectedFeature]
        All detected features, sorted by diameter (largest first).
    """
    features: list[DetectedFeature] = []

    # ------------------------------------------------------------------
    # 1. Use shared context or extract DEM
    # ------------------------------------------------------------------
    if ctx is not None:
        meta = ctx.meta
        px_m_ns = ctx.px_m_ns
        px_m_ew = ctx.px_m_ew
        px_m = ctx.px_m
        elev_filled = ctx.elev_filled
        elev_smooth = ctx.elev_smooth
    else:
        _emit(on_progress, "extracting_dem", {"radius_km": radius_km})
        elev, meta = extract_dem_window(lat0, lon0, radius_km)
        px_m_ns = meta["px_m_ns"]
        px_m_ew = meta["px_m_ew"]
        px_m = (px_m_ns + px_m_ew) / 2.0
        nan_mask = np.isnan(elev)
        fill_val = float(np.nanmean(elev)) if not np.all(nan_mask) else 0.0
        elev_filled = np.where(nan_mask, fill_val, elev)
        elev_smooth = gaussian_filter(elev_filled, sigma=2.0)

    # ------------------------------------------------------------------
    # 4-5. Multi-scale morphological closing -> depression depth
    # ------------------------------------------------------------------
    _emit(on_progress, "detecting_depressions", {"n_scales": len(_CLOSING_RADII_PX)})
    depth_map = _depression_depth_map(elev_smooth, _CLOSING_RADII_PX)

    # ------------------------------------------------------------------
    # 6. Threshold depth
    # ------------------------------------------------------------------
    binary = depth_map > min_depth_m

    # ------------------------------------------------------------------
    # 7. Label connected components
    # ------------------------------------------------------------------
    labels, n_labels = label(binary)
    _emit(on_progress, "labeling_craters", {"n_candidates": int(n_labels)})

    # ------------------------------------------------------------------
    # 8. Analyse each component
    # ------------------------------------------------------------------
    crater_candidates: list[tuple[dict, DetectedFeature]] = []
    graben_candidates: list[tuple[dict, DetectedFeature]] = []

    for lbl in range(1, n_labels + 1):
        props = _component_properties(labels, depth_map, elev_smooth, lbl)
        if props is None:
            continue

        diameter_km_val = props["eq_diameter_px"] * px_m / 1000.0
        depth_val = props["depth"]
        rim_elev = props["rim_elev"]
        floor_elev = props["floor_elev"]
        circularity = props["circularity"]

        # d/D ratio
        diam_m = diameter_km_val * 1000.0
        dd_ratio = depth_val / diam_m if diam_m > 0 else 0.0

        # Geographic centre
        c_lat, c_lon = pixel_to_latlon(
            int(round(props["centroid_r"])),
            int(round(props["centroid_c"])),
            meta,
        )

        # ----- Crater branch -----
        if circularity > 0.55:
            if diameter_km_val < min_diameter_km or diameter_km_val > 300.0:
                continue
            if dd_ratio < 0.005 or dd_ratio > 0.4:
                continue

            confidence = _assign_confidence(circularity, dd_ratio, diameter_km_val)

            feat = DetectedFeature(
                id=_uid(),
                feature_type="crater",  # may upgrade to terraced_crater later
                lat=round(c_lat, 5),
                lon=round(c_lon, 5),
                diameter_km=round(diameter_km_val, 2),
                depth_m=round(depth_val, 1),
                depth_diameter_ratio=round(dd_ratio, 5),
                circularity=round(circularity, 4),
                rim_elevation_m=round(rim_elev, 1),
                floor_elevation_m=round(floor_elev, 1),
                morphology="simple" if diameter_km_val < 10 else "complex",
                confidence=confidence,
                description="",
            )
            crater_candidates.append((props, feat))

        # ----- Graben branch: elongated, non-circular -----
        else:
            aspect = props["aspect_ratio"]
            major_px = props["major_axis_px"]
            if aspect > 3.0 and major_px > 25:
                width_km_val = props["minor_axis_px"] * px_m / 1000.0
                if width_km_val >= 10.0:
                    continue  # too wide to be a graben
                length_km_val = props["major_axis_px"] * px_m / 1000.0

                # Graben confidence: weighted combination of elongation,
                # length, and depth signals.
                conf_aspect = min(1.0, aspect / 5.0)
                conf_length = min(1.0, length_km_val / 20.0)
                conf_depth = min(1.0, depth_val / 500.0)
                graben_conf = round(
                    min(1.0, 0.40 + 0.15 * conf_aspect
                              + 0.15 * conf_length
                              + 0.15 * conf_depth),
                    3,
                )

                graben_candidates.append((props, DetectedFeature(
                    id=_uid(),
                    feature_type="graben",
                    lat=round(c_lat, 5),
                    lon=round(c_lon, 5),
                    depth_m=round(depth_val, 1),
                    length_km=round(length_km_val, 2),
                    width_km=round(width_km_val, 2),
                    aspect_ratio=round(aspect, 2),
                    rim_elevation_m=round(rim_elev, 1),
                    floor_elevation_m=round(floor_elev, 1),
                    confidence=graben_conf,
                    description="",
                )))

    # ------------------------------------------------------------------
    # Terrace classification for craters >10 km
    # ------------------------------------------------------------------
    n_terrace_candidates = sum(
        1 for _, f in crater_candidates if f.diameter_km > 10.0
    )
    terrace_idx = 0
    for props, feat in crater_candidates:
        if feat.diameter_km <= 10.0:
            feat.morphology = "simple"
            feat.description = (
                f"Simple crater, d/D={feat.depth_diameter_ratio:.4f}, "
                f"circularity={feat.circularity:.3f}"
            )
            continue

        terrace_idx += 1
        _emit(on_progress, "classifying_terraces", {
            "current": terrace_idx,
            "total": n_terrace_candidates,
        })

        radius_px = props["eq_diameter_px"] / 2.0
        n_terraces = _classify_terraces(
            elev_filled,
            props["centroid_r"],
            props["centroid_c"],
            radius_px,
            px_m,
        )
        feat.n_terraces = n_terraces
        if n_terraces >= 1:
            feat.feature_type = "terraced_crater"
            feat.morphology = "terraced"
            feat.description = (
                f"Terraced crater ({n_terraces} "
                f"bench{'es' if n_terraces > 1 else ''}), "
                f"d/D={feat.depth_diameter_ratio:.4f}, "
                f"circularity={feat.circularity:.3f}"
            )
        else:
            feat.morphology = "complex"
            feat.description = (
                f"Complex crater (no terrace benches detected), "
                f"d/D={feat.depth_diameter_ratio:.4f}, "
                f"circularity={feat.circularity:.3f}"
            )

    # Fill descriptions for any craters that were not yet described
    for _, feat in crater_candidates:
        if feat.description == "":
            feat.description = (
                f"{feat.morphology.capitalize()} crater, "
                f"d/D={feat.depth_diameter_ratio:.4f}, "
                f"circularity={feat.circularity:.3f}"
            )

    features.extend(f for _, f in crater_candidates)

    # ------------------------------------------------------------------
    # Graben: add descriptions
    # ------------------------------------------------------------------
    _emit(on_progress, "detecting_graben", {"n_candidates": len(graben_candidates)})
    for _, feat in graben_candidates:
        feat.description = (
            f"Graben, {feat.length_km:.1f} km x {feat.width_km:.1f} km, "
            f"aspect ratio {feat.aspect_ratio:.1f}, depth {feat.depth_m:.0f} m"
        )
    features.extend(f for _, f in graben_candidates)

    # ------------------------------------------------------------------
    # Volcanic construct detection (positive-relief features)
    # ------------------------------------------------------------------
    _emit(on_progress, "detecting_volcanics", {})

    inv_elev = float(np.nanmax(elev_smooth)) - elev_smooth
    inv_depth_map = _depression_depth_map(inv_elev, _CLOSING_RADII_PX)
    inv_binary = inv_depth_map > min_depth_m
    inv_labels, inv_n = label(inv_binary)

    # Slope map for flank angle filtering (reuse shared if available)
    slope_map = ctx.slope_deg if ctx is not None else compute_slope_map(elev_smooth, px_m_ns, px_m_ew)

    # Collect centres of already-detected craters so we can avoid
    # double-counting calderas as volcanic constructs.
    crater_centres: set[tuple[float, float]] = set()
    for _, feat in crater_candidates:
        # Quantise to ~2-pixel grid to catch near-overlaps
        crater_centres.add((round(feat.lat, 3), round(feat.lon, 3)))

    for lbl in range(1, inv_n + 1):
        props = _component_properties(inv_labels, inv_depth_map, elev_smooth, lbl)
        if props is None:
            continue

        diameter_km_val = props["eq_diameter_px"] * px_m / 1000.0
        if diameter_km_val < min_diameter_km or diameter_km_val > 600.0:
            continue
        if props["circularity"] < 0.55:
            continue

        c_lat, c_lon = pixel_to_latlon(
            int(round(props["centroid_r"])),
            int(round(props["centroid_c"])),
            meta,
        )

        # Filter: the centre of a volcanic construct should be ELEVATED,
        # not a depression.  If the original depth_map shows a significant
        # depression at the centroid the feature is more likely a caldera
        # already captured as a crater.
        cr = int(round(props["centroid_r"]))
        cc = int(round(props["centroid_c"]))
        if 0 <= cr < depth_map.shape[0] and 0 <= cc < depth_map.shape[1]:
            if depth_map[cr, cc] > min_depth_m:
                continue  # depression at centre -> skip

        # Also skip if too close to an existing crater detection
        key = (round(c_lat, 3), round(c_lon, 3))
        if key in crater_centres:
            continue

        # Mean flank slope (on original DEM) within the construct mask
        mask = props["mask"]
        flank_slopes = slope_map[mask]
        valid_slopes = flank_slopes[~np.isnan(flank_slopes)]
        if len(valid_slopes) == 0:
            continue
        mean_flank_slope = float(np.mean(valid_slopes))
        if mean_flank_slope > 15.0:
            continue  # flanks too steep for a shield / dome

        # Height = max(original) - min(original) over the footprint
        region_elev = elev_smooth[mask]
        height_m = float(np.nanmax(region_elev) - np.nanmin(region_elev))

        confidence = _assign_volcanic_confidence(
            props["circularity"], mean_flank_slope, height_m
        )

        features.append(DetectedFeature(
            id=_uid(),
            feature_type="volcanic",
            lat=round(c_lat, 5),
            lon=round(c_lon, 5),
            diameter_km=round(diameter_km_val, 2),
            depth_m=round(height_m, 1),  # store relief as "depth" (positive)
            circularity=round(props["circularity"], 4),
            rim_elevation_m=round(float(np.nanmax(region_elev)), 1),
            floor_elevation_m=round(float(np.nanmin(region_elev)), 1),
            confidence=confidence,
            morphology="shield" if mean_flank_slope < 5.0 else "dome",
            description=(
                f"Volcanic construct ({round(diameter_km_val, 1)} km), "
                f"mean flank slope {mean_flank_slope:.1f} deg, "
                f"relief {height_m:.0f} m"
            ),
        ))

    # ------------------------------------------------------------------
    # Sort by diameter (largest first) and return
    # ------------------------------------------------------------------
    features.sort(key=lambda f: f.diameter_km, reverse=True)
    return features
