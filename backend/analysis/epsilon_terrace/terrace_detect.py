"""
Terrace detection from HiRISE DTM radial profiles.

Extracts radial elevation profiles from crater center across multiple azimuths,
identifies terrace benches (near-zero slope bounded by slope breaks), and
computes the depth between terraces as a "true depth" proxy.
"""

import math
import numpy as np
import rasterio
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class TerraceDetection:
    """A detected terrace in a radial profile."""
    radius_m: float           # Distance from crater center
    elevation_m: float        # Elevation at terrace
    azimuth_deg: float        # Azimuth of this ray
    width_m: float = 0.0      # Radial extent of the bench


@dataclass
class TerraceResult:
    """Result of terrace analysis for a single crater DTM."""
    dtm_file: str
    crater_center_lat: float
    crater_center_lon: float
    crater_radius_m: float
    n_azimuths: int
    terraces: List[TerraceDetection] = field(default_factory=list)
    median_profile: Optional[np.ndarray] = None     # shape: (n_radial_bins,)
    radial_distances: Optional[np.ndarray] = None   # shape: (n_radial_bins,)
    all_profiles: Optional[np.ndarray] = None       # shape: (n_azimuths, n_radial_bins)
    terrace_depths: List[dict] = field(default_factory=list)
    error: Optional[str] = None


def _equirect_params(crs_wkt: str) -> Tuple[float, float, float]:
    """Extract central_meridian, standard_parallel_1, radius from CRS WKT."""
    import re
    cm_match = re.search(r'central_meridian["\s,]+(-?[\d.]+)', crs_wkt, re.IGNORECASE)
    sp_match = re.search(r'standard_parallel_1["\s,]+(-?[\d.]+)', crs_wkt, re.IGNORECASE)
    radius_match = re.search(r'SPHEROID\["[^"]*",\s*([\d.]+)', crs_wkt)
    cm = float(cm_match.group(1)) if cm_match else 0.0
    sp = float(sp_match.group(1)) if sp_match else 0.0
    radius = float(radius_match.group(1)) if radius_match else 3389500.0
    return cm, sp, radius


def latlon_to_pixel(ds, lat: float, lon: float) -> Tuple[int, int]:
    """Convert geographic lat/lon to pixel row/col in the DTM raster."""
    wkt = ds.crs.to_wkt()
    cm, sp, radius = _equirect_params(wkt)
    cos_sp = math.cos(math.radians(sp))
    # Normalize longitude to same convention as CRS central_meridian
    # CRS may use 0-360 (e.g., cm=193.36) while input uses -180/180 (e.g., lon=-166.64)
    if cm > 180 and lon < 0:
        lon = lon + 360
    elif cm < -180 and lon > 0:
        lon = lon - 360
    # Geographic → projected (equirectangular)
    x = radius * cos_sp * math.radians(lon - cm)
    y = radius * math.radians(lat)
    # Projected → pixel via inverse affine
    inv = ~ds.transform
    col, row = inv * (x, y)
    return int(round(row)), int(round(col))


def _estimate_crater_center(elev: np.ndarray, res_m: float) -> Tuple[int, int, float]:
    """Estimate crater center as the location of minimum elevation.

    Excludes edge regions to avoid boundary artifacts.
    Returns (row, col, estimated_radius_m).
    """
    from scipy.ndimage import uniform_filter
    # Fill NaN before smoothing to prevent NaN propagation
    fill_val = np.nanmean(elev) if np.any(~np.isnan(elev)) else 0.0
    filled = np.where(np.isnan(elev), fill_val, elev)
    smooth = uniform_filter(filled, size=max(5, int(50 / res_m)))
    # Mask edges (10% border on each side) to avoid boundary artifacts
    margin_r = max(10, elev.shape[0] // 10)
    margin_c = max(10, elev.shape[1] // 10)
    mask = np.full_like(smooth, np.inf)
    mask[margin_r:-margin_r, margin_c:-margin_c] = smooth[margin_r:-margin_r, margin_c:-margin_c]
    mask[np.isnan(elev)] = np.inf
    min_idx = np.unravel_index(np.argmin(mask), mask.shape)
    center_row, center_col = int(min_idx[0]), int(min_idx[1])

    # Estimate radius: distance from center to edge where elevation stops increasing
    max_r_px = min(center_row - margin_r, center_col - margin_c,
                   elev.shape[0] - center_row - 1 - margin_r,
                   elev.shape[1] - center_col - 1 - margin_c,
                   int(5000 / res_m))  # cap at 5km
    max_r_px = max(max_r_px, 50)
    r_m = max_r_px * res_m * 0.5
    return center_row, center_col, r_m


def extract_radial_profiles(
    dtm_path: str,
    center_lat: Optional[float] = None,
    center_lon: Optional[float] = None,
    n_azimuths: int = 36,
    max_radius_m: float = 3000.0,
    radial_step_m: float = 10.0,
) -> TerraceResult:
    """
    Extract radial elevation profiles from a DTM centered on a crater.

    If center_lat/lon not provided, estimates crater center as the elevation minimum.

    Args:
        dtm_path: Path to HiRISE DTM .IMG file
        center_lat/lon: Crater center in geographic coordinates
        n_azimuths: Number of radial rays (default 36 = every 10°)
        max_radius_m: Maximum radial distance to sample
        radial_step_m: Step size along each ray

    Returns:
        TerraceResult with profiles and detected terraces
    """
    result = TerraceResult(
        dtm_file=dtm_path,
        crater_center_lat=center_lat or 0.0,
        crater_center_lon=center_lon or 0.0,
        crater_radius_m=0.0,
        n_azimuths=n_azimuths,
    )

    try:
        ds = rasterio.open(dtm_path)
    except Exception as e:
        result.error = f"Cannot open DTM: {e}"
        return result

    elev = ds.read(1).astype(np.float64)
    res_m = abs(ds.res[0])  # pixel size in meters

    # Handle nodata
    if ds.nodata is not None:
        elev[elev == ds.nodata] = np.nan

    # Get crater center in pixel space
    if center_lat is not None and center_lon is not None:
        center_row, center_col = latlon_to_pixel(ds, center_lat, center_lon)
        # Clamp to raster
        center_row = max(0, min(center_row, elev.shape[0] - 1))
        center_col = max(0, min(center_col, elev.shape[1] - 1))
    else:
        center_row, center_col, est_radius = _estimate_crater_center(elev, res_m)
        # Convert pixel back to lat/lon for result
        wkt = ds.crs.to_wkt()
        cm, sp, radius = _equirect_params(wkt)
        cos_sp = math.cos(math.radians(sp))
        x, y = ds.transform * (center_col, center_row)
        result.crater_center_lon = math.degrees(x / (radius * cos_sp)) + cm
        result.crater_center_lat = math.degrees(y / radius)

    # Build radial sample grid
    n_radial = int(max_radius_m / radial_step_m)
    radii = np.arange(0, n_radial) * radial_step_m
    azimuths = np.linspace(0, 360, n_azimuths, endpoint=False)

    all_profiles = np.full((n_azimuths, n_radial), np.nan)

    for i, az in enumerate(azimuths):
        az_rad = math.radians(az)
        for j, r in enumerate(radii):
            # Offset in pixels
            dr = r / res_m
            row = center_row - dr * math.cos(az_rad)  # N is -row
            col = center_col + dr * math.sin(az_rad)
            ri, ci = int(round(row)), int(round(col))
            if 0 <= ri < elev.shape[0] and 0 <= ci < elev.shape[1]:
                all_profiles[i, j] = elev[ri, ci]

    result.all_profiles = all_profiles
    result.radial_distances = radii

    # Compute median profile (robust across azimuths)
    median_profile = np.nanmedian(all_profiles, axis=0)
    result.median_profile = median_profile

    # Estimate crater radius from median profile
    # Crater rim = first significant local max going outward from center
    if not np.all(np.isnan(median_profile)):
        smooth_med = _smooth_1d(median_profile, window=max(5, int(100 / radial_step_m)))
        grad = np.gradient(smooth_med)
        # Rim: where gradient goes from positive to negative (local max)
        for k in range(10, len(grad) - 1):
            if grad[k] > 0 and grad[k + 1] <= 0 and radii[k] > 200:
                result.crater_radius_m = float(radii[k])
                break
        if result.crater_radius_m == 0:
            result.crater_radius_m = float(radii[len(radii) // 2])

    ds.close()

    # Detect terraces in median profile
    _detect_terraces(result, radial_step_m)

    # Compute terrace depths
    _compute_depths(result)

    return result


def _smooth_1d(arr: np.ndarray, window: int = 11) -> np.ndarray:
    """Smooth 1D array with a moving average, ignoring NaNs."""
    result = np.copy(arr)
    half = window // 2
    for i in range(len(arr)):
        lo = max(0, i - half)
        hi = min(len(arr), i + half + 1)
        segment = arr[lo:hi]
        valid = segment[~np.isnan(segment)]
        if len(valid) > 0:
            result[i] = np.mean(valid)
    return result


def _detect_terraces(result: TerraceResult, step_m: float):
    """
    Detect terraces as near-zero slope segments (benches) in the median profile.

    Algorithm:
    1. Smooth median profile
    2. Compute slope (first derivative) and curvature (second derivative)
    3. Find segments where |slope| is small (bench) bounded by slope breaks
    4. Filter by minimum bench width and position (must be inside crater rim)
    """
    if result.median_profile is None or result.radial_distances is None:
        return

    profile = result.median_profile
    radii = result.radial_distances

    # Smooth for robust derivative estimation
    window = max(7, int(150 / step_m))  # ~150m smoothing window
    smooth = _smooth_1d(profile, window=window)

    # Slope: elevation change per meter
    slope = np.gradient(smooth, step_m)

    # Curvature (second derivative)
    curvature = np.gradient(slope, step_m)

    # Find bench segments: |slope| < threshold
    slope_thresh = 0.02  # ~1.1° — flat bench threshold
    is_flat = np.abs(slope) < slope_thresh

    # Search from just outside the crater pit outward
    # For terraced craters, benches are ABOVE the floor, beyond the steep wall
    # min_r: skip the crater floor and steep wall; start where slope first flattens
    # max_r: extend to 80% of max profile distance (exclude very far terrain)
    floor_elev = float(np.nanmin(smooth))
    half_relief = (float(np.nanmax(smooth)) - floor_elev) * 0.3
    # min_r: first radius where elevation is above floor + 30% of relief
    min_r_idx = 0
    for k in range(len(smooth)):
        if not np.isnan(smooth[k]) and smooth[k] > floor_elev + half_relief:
            min_r_idx = k
            break
    min_r = max(50.0, float(radii[min_r_idx]) if min_r_idx > 0 else 50.0)
    max_r = radii[-1] * 0.85

    # Find contiguous flat segments
    flat_segments = []
    in_segment = False
    seg_start = 0
    for k in range(len(is_flat)):
        if is_flat[k] and radii[k] >= min_r and radii[k] <= max_r:
            if not in_segment:
                seg_start = k
                in_segment = True
        else:
            if in_segment:
                flat_segments.append((seg_start, k - 1))
                in_segment = False
    if in_segment:
        flat_segments.append((seg_start, len(is_flat) - 1))

    # Filter: minimum bench width of 50m
    min_width = max(3, int(50 / step_m))
    filtered = [(s, e) for s, e in flat_segments if (e - s) >= min_width]

    # Convert to TerraceDetection objects
    terraces = []
    for s, e in filtered:
        mid = (s + e) // 2
        terrace = TerraceDetection(
            radius_m=float(radii[mid]),
            elevation_m=float(np.nanmean(smooth[s:e + 1])),
            azimuth_deg=0.0,  # median profile
            width_m=float(radii[e] - radii[s]),
        )
        terraces.append(terrace)

    # Also detect terraces per-azimuth for uncertainty estimation
    if result.all_profiles is not None:
        n_az = result.all_profiles.shape[0]
        azimuths = np.linspace(0, 360, n_az, endpoint=False)

        for i in range(n_az):
            ray = result.all_profiles[i]
            if np.sum(~np.isnan(ray)) < 20:
                continue
            ray_smooth = _smooth_1d(ray, window=window)
            ray_slope = np.gradient(ray_smooth, step_m)
            ray_flat = np.abs(ray_slope) < slope_thresh

            ray_segs = []
            in_seg = False
            ss = 0
            for k in range(len(ray_flat)):
                if ray_flat[k] and radii[k] >= min_r and radii[k] <= max_r:
                    if not in_seg:
                        ss = k
                        in_seg = True
                else:
                    if in_seg:
                        ray_segs.append((ss, k - 1))
                        in_seg = False
            if in_seg:
                ray_segs.append((ss, len(ray_flat) - 1))

            for s, e in ray_segs:
                if (e - s) >= min_width:
                    mid = (s + e) // 2
                    terraces.append(TerraceDetection(
                        radius_m=float(radii[mid]),
                        elevation_m=float(np.nanmean(ray_smooth[s:e + 1])),
                        azimuth_deg=float(azimuths[i]),
                        width_m=float(radii[e] - radii[s]),
                    ))

    result.terraces = terraces


def _compute_depths(result: TerraceResult):
    """
    Compute depth metrics between terraces.

    Metrics:
    A) Upper terrace → lower terrace Δz
    B) Terrace → crater floor Δz
    C) Rim → first terrace Δz
    """
    if result.median_profile is None or result.radial_distances is None:
        return
    if not result.terraces:
        return

    profile = result.median_profile
    radii = result.radial_distances

    # Get median terraces (azimuth_deg == 0.0 means from median profile)
    med_terraces = [t for t in result.terraces if t.azimuth_deg == 0.0]
    per_az_terraces = [t for t in result.terraces if t.azimuth_deg != 0.0]

    if not med_terraces:
        return

    # Sort by radius (inner to outer)
    med_terraces.sort(key=lambda t: t.radius_m)

    # Floor elevation (minimum of profile in inner region)
    inner_mask = radii < med_terraces[0].radius_m * 0.5
    if np.any(inner_mask) and np.any(~np.isnan(profile[inner_mask])):
        floor_elev = float(np.nanmin(profile[inner_mask]))
    else:
        floor_elev = float(np.nanmin(profile[~np.isnan(profile)]))

    # Rim elevation (max elevation near crater radius)
    rim_idx = np.argmin(np.abs(radii - result.crater_radius_m))
    rim_region = profile[max(0, rim_idx - 5):rim_idx + 5]
    rim_elev = float(np.nanmax(rim_region)) if len(rim_region) > 0 else float(np.nanmax(profile))

    # Per-azimuth depth estimates for uncertainty
    az_depths = {
        'terrace_to_floor': [],
        'rim_to_terrace': [],
        'upper_to_lower': [],
    }

    # Group per-azimuth terraces by azimuth and compute depths
    from collections import defaultdict
    az_groups = defaultdict(list)
    for t in per_az_terraces:
        az_groups[t.azimuth_deg].append(t)
    for az, grp in az_groups.items():
        grp.sort(key=lambda t: t.radius_m)
        if grp:
            az_depths['terrace_to_floor'].append(grp[0].elevation_m - floor_elev)
            az_depths['rim_to_terrace'].append(rim_elev - grp[0].elevation_m)
        if len(grp) >= 2:
            az_depths['upper_to_lower'].append(grp[-1].elevation_m - grp[0].elevation_m)

    depths = []

    # Metric A: Upper → Lower terrace
    if len(med_terraces) >= 2:
        dz = med_terraces[-1].elevation_m - med_terraces[0].elevation_m
        unc = float(np.std(az_depths['upper_to_lower'])) if az_depths['upper_to_lower'] else 0.0
        depths.append({
            'metric': 'upper_to_lower_terrace',
            'depth_m': round(abs(dz), 1),
            'uncertainty_m': round(unc, 1),
            'upper_elev_m': round(med_terraces[-1].elevation_m, 1),
            'lower_elev_m': round(med_terraces[0].elevation_m, 1),
            'n_azimuth_samples': len(az_depths['upper_to_lower']),
        })

    # Metric B: First terrace → floor
    dz_floor = med_terraces[0].elevation_m - floor_elev
    unc_floor = float(np.std(az_depths['terrace_to_floor'])) if az_depths['terrace_to_floor'] else 0.0
    depths.append({
        'metric': 'terrace_to_floor',
        'depth_m': round(abs(dz_floor), 1),
        'uncertainty_m': round(unc_floor, 1),
        'terrace_elev_m': round(med_terraces[0].elevation_m, 1),
        'floor_elev_m': round(floor_elev, 1),
        'n_azimuth_samples': len(az_depths['terrace_to_floor']),
    })

    # Metric C: Rim → first terrace
    dz_rim = rim_elev - med_terraces[0].elevation_m
    unc_rim = float(np.std(az_depths['rim_to_terrace'])) if az_depths['rim_to_terrace'] else 0.0
    depths.append({
        'metric': 'rim_to_terrace',
        'depth_m': round(abs(dz_rim), 1),
        'uncertainty_m': round(unc_rim, 1),
        'rim_elev_m': round(rim_elev, 1),
        'terrace_elev_m': round(med_terraces[0].elevation_m, 1),
        'n_azimuth_samples': len(az_depths['rim_to_terrace']),
    })

    result.terrace_depths = depths
