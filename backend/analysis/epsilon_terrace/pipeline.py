"""
CraterStratigraphyAnalyzer — crater-centric dielectric stratigraphy from SHARAD + DTM.

Algorithm:
  1. Search for SHARAD tracks within buffer_km of crater center
  2. Search for HiRISE DTMs; fall back to MOLA DEM if none
  3. Extract radial elevation profiles + detect terraces
  4. Pick subsurface SHARAD reflectors near the crater
  5. Compute εr = (c·t / 2d)² from terrace depth vs radar TWT
  6. Build overlay + profile for map and chart rendering
"""

import json
import logging
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from analysis.shared.base import AnalysisModule
from analysis.shared.coordinates import haversine_km
from analysis.shared.overlay import interpolate_colormap
from .models import (
    CraterInfo,
    EpsilonEstimateInfo,
    RadialProfilePoint,
    SharadPickInfo,
    StratigraphyOverlayPoint,
    StratigraphyParameters,
    StratigraphyResult,
    StratigraphySummary,
)

logger = logging.getLogger(__name__)

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Epsilon colormap: εr value → RGBA
CMAP_EPSILON: List[Tuple[float, List[int]]] = [
    (1.5,  [100, 180, 255, 220]),   # light blue (very low)
    (2.5,  [70, 130, 230, 220]),    # blue (porous ice / dry regolith)
    (3.1,  [80, 220, 200, 220]),    # cyan (water ice ~3.1)
    (4.0,  [100, 200, 100, 220]),   # green (ice-cemented regolith)
    (5.5,  [240, 220, 80, 220]),    # yellow (basaltic regolith)
    (8.0,  [230, 130, 50, 220]),    # orange (dense rock)
    (12.0, [210, 50, 50, 220]),     # red (very dense / artifact)
]


class CraterStratigraphyAnalyzer(AnalysisModule):
    """Crater-centric dielectric stratigraphy analysis using SHARAD + DTM."""

    def __init__(self):
        self._result: Optional[StratigraphyResult] = None

    # ────────────────────────────────────────────────────────────────
    # AnalysisModule interface
    # ────────────────────────────────────────────────────────────────

    def run(
        self,
        crater_lat: float,
        crater_lon: float,
        diameter_km: float,
        buffer_km: float = 30.0,
        n_azimuths: int = 36,
        max_radius_m: float = 3000.0,
        min_snr: float = 3.5,
    ) -> StratigraphyResult:
        """Execute the full crater stratigraphy pipeline."""
        try:
            self._result = self._run_impl(
                crater_lat, crater_lon, diameter_km,
                buffer_km, n_azimuths, max_radius_m, min_snr,
            )
        except Exception as exc:
            logger.exception("Stratigraphy pipeline failed for (%.3f, %.3f)", crater_lat, crater_lon)
            self._result = StratigraphyResult(success=False, error=str(exc))
        return self._result

    def generate_profile(self) -> List[Dict]:
        if not self._result or not self._result.radial_profile:
            return []
        return [p.model_dump() for p in self._result.radial_profile]

    def generate_overlay(self) -> List[Dict]:
        if not self._result or not self._result.overlay_points:
            return []
        return [p.model_dump() for p in self._result.overlay_points]

    def generate_summary(self) -> Dict:
        if not self._result:
            return {"success": False, "error": "Not run yet"}
        d = {"success": self._result.success, "error": self._result.error}
        if self._result.summary:
            d.update(self._result.summary.model_dump())
        return d

    # ────────────────────────────────────────────────────────────────
    # Core implementation
    # ────────────────────────────────────────────────────────────────

    def _run_impl(
        self,
        crater_lat: float,
        crater_lon: float,
        diameter_km: float,
        buffer_km: float,
        n_azimuths: int,
        max_radius_m: float,
        min_snr: float,
    ) -> StratigraphyResult:
        from .sharad_pick import pick_subsurface_interface, SharadPickResult
        from .terrace_detect import extract_radial_profiles, TerraceResult
        from .epsilon_calc import run_epsilon_analysis

        logger.info(
            "Stratigraphy: crater=(%.3f, %.3f) d=%.1fkm buffer=%.0fkm",
            crater_lat, crater_lon, diameter_km, buffer_km,
        )

        # Phase 2: Adaptive max_radius_m scaled to crater diameter
        # Use at least 1.5x the crater radius, capped at user-specified max
        adaptive_max_radius_m = max(max_radius_m, diameter_km * 500.0 * 1.5)
        # Don't exceed 10 km (practical limit for DTM coverage)
        adaptive_max_radius_m = min(adaptive_max_radius_m, 10000.0)
        if adaptive_max_radius_m != max_radius_m:
            logger.info("Stratigraphy: adaptive max_radius = %.0f m (from %.0f m)",
                        adaptive_max_radius_m, max_radius_m)
            max_radius_m = adaptive_max_radius_m

        params = StratigraphyParameters(
            crater_lat=crater_lat, crater_lon=crater_lon,
            diameter_km=diameter_km, buffer_km=buffer_km,
            n_azimuths=n_azimuths, max_radius_m=max_radius_m,
            min_snr=min_snr,
        )

        # ── Step 1: Find SHARAD tracks near crater ────────────────
        nearby_tracks = _find_sharad_tracks_near_crater(
            crater_lat, crater_lon, buffer_km,
        )
        logger.info("Stratigraphy: %d SHARAD tracks within %.0fkm",
                     len(nearby_tracks), buffer_km)

        # ── Step 2: Find HiRISE DTMs or fall back to MOLA ─────────
        terrace_results: List[TerraceResult] = []
        dtm_source = "none"

        # Try HiRISE DTM directory
        dtm_dir = os.path.join(_BACKEND_DIR, "hirise_dtm_data")
        dtm_paths = _find_dtms_near_crater(
            crater_lat, crater_lon, dtm_dir, buffer_km,
        )

        if dtm_paths:
            dtm_source = "HiRISE"
            for dtm_path in dtm_paths[:2]:  # max 2 DTMs
                tr = extract_radial_profiles(
                    dtm_path,
                    center_lat=crater_lat,
                    center_lon=crater_lon,
                    n_azimuths=n_azimuths,
                    max_radius_m=max_radius_m,
                )
                if not tr.error:
                    terrace_results.append(tr)
                    logger.info("Stratigraphy: HiRISE DTM terraces=%d (%s)",
                                len([t for t in tr.terraces if t.azimuth_deg == 0.0]),
                                os.path.basename(dtm_path))
        else:
            # MOLA DEM fallback
            tr = _mola_radial_profiles(
                crater_lat, crater_lon, diameter_km,
                n_azimuths, max_radius_m,
            )
            if tr is not None and not tr.error:
                dtm_source = "MOLA"
                terrace_results.append(tr)
                n_med = len([t for t in tr.terraces if t.azimuth_deg == 0.0])
                logger.info("Stratigraphy: MOLA fallback terraces=%d", n_med)

        # ── Step 3: SHARAD subsurface picks ────────────────────────
        # Phase 2: Adaptive window based on crater diameter
        # Larger craters → wider SHARAD window to capture full subsurface extent
        adaptive_window_km = max(5.0, min(diameter_km * 1.5, buffer_km / 2))
        logger.info("Stratigraphy: adaptive SHARAD window = %.1f km (diameter=%.1f km)",
                     adaptive_window_km, diameter_km)

        pick_pairs: List[Tuple[TerraceResult, SharadPickResult]] = []
        all_pick_results: List[SharadPickResult] = []

        for product_id, dist_km in nearby_tracks[:3]:  # max 3 tracks
            pr = pick_subsurface_interface(
                product_id, crater_lat, crater_lon,
                window_km=adaptive_window_km,
                min_snr=min_snr,
            )
            all_pick_results.append(pr)

            if not pr.error and pr.picks:
                # Pair with the best terrace result (if any)
                for tr in terrace_results:
                    if tr.terrace_depths:
                        pick_pairs.append((tr, pr))
                        break

        logger.info("Stratigraphy: %d SHARAD pick results, %d paired with terraces",
                     len(all_pick_results), len(pick_pairs))

        # ── Step 4: Compute εr ─────────────────────────────────────
        eps_result = run_epsilon_analysis(terrace_results, pick_pairs)

        # ── Step 5: Build crater info ──────────────────────────────
        n_terraces = 0
        terrace_depths_out: List[Dict] = []
        if terrace_results:
            tr = terrace_results[0]
            n_terraces = len([t for t in tr.terraces if t.azimuth_deg == 0.0])
            terrace_depths_out = tr.terrace_depths

        crater_info = CraterInfo(
            lat=crater_lat, lon=crater_lon,
            diameter_km=diameter_km,
            depth_m=terrace_depths_out[0]["depth_m"] if terrace_depths_out else None,
            n_terraces=n_terraces,
            terrace_depths=terrace_depths_out,
            dtm_source=dtm_source,
        )

        # ── Step 6: Build radial profile for chart ─────────────────
        radial_profile: List[RadialProfilePoint] = []
        if terrace_results:
            tr = terrace_results[0]
            if tr.median_profile is not None and tr.radial_distances is not None:
                for j in range(len(tr.radial_distances)):
                    med = float(tr.median_profile[j])
                    if np.isnan(med):
                        continue
                    pt = RadialProfilePoint(
                        radius_m=round(float(tr.radial_distances[j]), 1),
                        elevation_m=round(med, 1),
                    )
                    if tr.all_profiles is not None:
                        col = tr.all_profiles[:, j]
                        valid = col[~np.isnan(col)]
                        if len(valid) > 0:
                            pt.elevation_min = round(float(valid.min()), 1)
                            pt.elevation_max = round(float(valid.max()), 1)
                    radial_profile.append(pt)

                # Downsample if too many points
                if len(radial_profile) > 500:
                    step = len(radial_profile) // 500
                    radial_profile = radial_profile[::step]

        # ── Step 7: Build epsilon estimates ────────────────────────
        epsilon_estimates: List[EpsilonEstimateInfo] = []
        for est in eps_result.estimates:
            epsilon_estimates.append(EpsilonEstimateInfo(
                depth_metric=est.depth_metric,
                depth_m=est.depth_true_m,
                depth_unc_m=est.depth_unc_m,
                twt_us=est.twt_us,
                epsilon_r=est.epsilon_r,
                epsilon_low=est.epsilon_low,
                epsilon_high=est.epsilon_high,
                interpretation=est.interpretation,
                quality=est.quality,
                quality_note=est.quality_note,
                sharad_product=est.sharad_product,
                n_picks=est.n_sharad_picks,
            ))

        # ── Step 8: Build SHARAD pick summaries ────────────────────
        sharad_picks: List[SharadPickInfo] = []
        for pr in all_pick_results:
            if pr.picks:
                min_dist = min(p.distance_to_crater_km for p in pr.picks)
                sharad_picks.append(SharadPickInfo(
                    product_id=pr.picks[0].product_id,
                    n_picks=len(pr.picks),
                    median_twt_us=pr.median_twt_us,
                    twt_unc_us=pr.twt_unc_us,
                    min_distance_km=round(min_dist, 1),
                ))

        # ── Step 9: Overlay points (SHARAD pick locations) ─────────
        overlay_points: List[StratigraphyOverlayPoint] = []
        best_eps = None
        if epsilon_estimates:
            good = [e for e in epsilon_estimates if e.quality == "good"]
            best_eps = good[0].epsilon_r if good else epsilon_estimates[0].epsilon_r

        for pr in all_pick_results:
            for p in pr.picks:
                overlay_points.append(StratigraphyOverlayPoint(
                    lat=p.lat, lon=p.lon,
                    epsilon_r=best_eps,
                    color=interpolate_colormap(best_eps, CMAP_EPSILON),
                ))
        # Cap overlay points
        if len(overlay_points) > 200:
            step = len(overlay_points) // 200
            overlay_points = overlay_points[::step]

        # ── Step 10: Summary (with Phase 2 quality-weighted aggregate) ─
        best_interp = None
        best_quality = None
        weighted_eps = None
        weighted_std = None
        ci_68 = None
        ci_95 = None
        n_good = 0
        n_marginal = 0

        if epsilon_estimates:
            good = [e for e in epsilon_estimates if e.quality == "good"]
            best = good[0] if good else epsilon_estimates[0]
            best_eps = best.epsilon_r
            best_interp = best.interpretation
            best_quality = best.quality

            # Phase 2: Use quality-weighted aggregate if available
            if eps_result.aggregate:
                agg = eps_result.aggregate
                weighted_eps = agg.get("weighted_median_epsilon_r")
                weighted_std = agg.get("weighted_std")
                ci_68_raw = agg.get("confidence_interval_68")
                ci_95_raw = agg.get("confidence_interval_95")
                ci_68 = list(ci_68_raw) if ci_68_raw else None
                ci_95 = list(ci_95_raw) if ci_95_raw else None
                n_good = agg.get("n_good", 0)
                n_marginal = agg.get("n_marginal", 0)
                # Override best interpretation with aggregate interpretation
                best_interp = agg.get("interpretation", best_interp)

        total_picks = sum(len(pr.picks) for pr in all_pick_results)

        summary = StratigraphySummary(
            n_sharad_tracks=len(nearby_tracks),
            n_epsilon_estimates=len(epsilon_estimates),
            best_epsilon_r=best_eps,
            best_interpretation=best_interp,
            best_quality=best_quality,
            dem_source=dtm_source,
            total_sharad_picks=total_picks,
            weighted_epsilon_r=weighted_eps,
            weighted_epsilon_std=weighted_std,
            confidence_interval_68=ci_68,
            confidence_interval_95=ci_95,
            n_good_estimates=n_good,
            n_marginal_estimates=n_marginal,
        )

        # Build error/warning message for partial results
        error_msg = None
        if not nearby_tracks:
            error_msg = f"No SHARAD tracks within {buffer_km:.0f} km of crater"
        elif not pick_pairs:
            if not terrace_results or all(not tr.terrace_depths for tr in terrace_results):
                error_msg = "No terraces detected — cannot compute εr"
            elif all(pr.error for pr in all_pick_results):
                error_msg = "No SHARAD subsurface reflectors found near crater"

        logger.info(
            "Stratigraphy: done — %d estimates, best εr=%s",
            len(epsilon_estimates),
            f"{best_eps:.2f}" if best_eps else "none",
        )

        return StratigraphyResult(
            success=True,
            error=error_msg,
            crater_info=crater_info,
            summary=summary,
            radial_profile=radial_profile,
            epsilon_estimates=epsilon_estimates,
            sharad_picks=sharad_picks,
            overlay_points=overlay_points,
            parameters=params,
        )


# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────

def _find_sharad_tracks_near_crater(
    crater_lat: float,
    crater_lon: float,
    buffer_km: float,
) -> List[Tuple[str, float]]:
    """Find SHARAD product IDs whose tracks pass within buffer_km of a crater.

    Returns list of (product_id, min_distance_km) sorted by distance.
    """
    index_path = os.path.join(_BACKEND_DIR, "sharad_highres_data", "index.geojson")
    if not os.path.isfile(index_path):
        logger.warning("SHARAD index not found: %s", index_path)
        return []

    with open(index_path) as f:
        idx = json.load(f)

    # Quick lat pre-filter: buffer_km → degrees (approx)
    buffer_deg = buffer_km / 59.3  # ~59.3 km/deg on Mars

    results: List[Tuple[str, float]] = []
    for feat in idx.get("features", []):
        product_id = feat.get("properties", {}).get("product_id", "")
        if not product_id:
            continue

        coords = feat.get("geometry", {}).get("coordinates", [])
        if not coords:
            continue

        # Quick latitude check on sampled coordinates
        min_dist = float("inf")
        step = max(1, len(coords) // 50)  # sample every ~50 points

        # Check if any point is in lat range first
        lats = [c[1] for c in coords[::step]]
        if not lats:
            continue
        lat_min, lat_max = min(lats), max(lats)
        if lat_max < crater_lat - buffer_deg or lat_min > crater_lat + buffer_deg:
            continue

        # Detailed haversine check on sampled points
        for i in range(0, len(coords), step):
            lon_c, lat_c = coords[i]
            lon_180 = lon_c if lon_c <= 180 else lon_c - 360
            dist = haversine_km(crater_lat, crater_lon, lat_c, lon_180)
            if dist < min_dist:
                min_dist = dist

        if min_dist <= buffer_km:
            results.append((product_id, round(min_dist, 1)))

    results.sort(key=lambda x: x[1])
    return results


def _find_dtms_near_crater(
    crater_lat: float,
    crater_lon: float,
    dtm_dir: str,
    buffer_km: float,
) -> List[str]:
    """Find HiRISE DTM .IMG files near the crater location."""
    import glob as globmod

    if not os.path.isdir(dtm_dir):
        return []

    dtm_files = globmod.glob(os.path.join(dtm_dir, "*.IMG"))
    if not dtm_files:
        dtm_files = globmod.glob(os.path.join(dtm_dir, "*.img"))
    if not dtm_files:
        return []

    # Use existing run.py helper for geo bounds
    from .run import _get_dtm_geo_bounds

    results: List[Tuple[str, float]] = []
    for dtm_path in dtm_files:
        geo = _get_dtm_geo_bounds(dtm_path)
        if geo is None:
            continue
        dist = haversine_km(
            crater_lat, crater_lon,
            geo["center_lat"], geo["center_lon"],
        )
        if dist <= buffer_km:
            results.append((dtm_path, dist))

    results.sort(key=lambda x: x[1])
    return [r[0] for r in results]


def _mola_radial_profiles(
    crater_lat: float,
    crater_lon: float,
    diameter_km: float,
    n_azimuths: int = 36,
    max_radius_m: float = 3000.0,
) -> Optional["TerraceResult"]:
    """Extract radial profiles from MOLA DEM when no HiRISE DTM is available.

    Returns a TerraceResult-compatible object for use with run_epsilon_analysis.
    """
    from .terrace_detect import TerraceResult, _smooth_1d, _detect_terraces, _compute_depths
    from analysis.mola_detect.common import extract_dem_window, pixel_to_latlon

    radius_km = max(diameter_km, max_radius_m / 1000.0) * 1.5
    radius_km = max(radius_km, 5.0)  # at least 5km

    try:
        elev, meta = extract_dem_window(crater_lat, crater_lon, radius_km)
    except Exception as exc:
        logger.warning("MOLA DEM extraction failed: %s", exc)
        return None

    px_m = (meta["px_m_ew"] + meta["px_m_ns"]) / 2.0
    center_row = meta["row_c_local"]
    center_col = meta["col_c_local"]

    # Build radial profiles
    radial_step_m = max(px_m, 50.0)  # MOLA is ~200m/px, step at least pixel size
    n_radial = int(max_radius_m / radial_step_m)
    if n_radial < 10:
        return None

    radii = np.arange(0, n_radial) * radial_step_m
    azimuths = np.linspace(0, 360, n_azimuths, endpoint=False)
    all_profiles = np.full((n_azimuths, n_radial), np.nan)

    for i, az in enumerate(azimuths):
        az_rad = math.radians(az)
        for j, r in enumerate(radii):
            dr = r / px_m
            row = center_row - dr * math.cos(az_rad)
            col = center_col + dr * math.sin(az_rad)
            ri, ci = int(round(row)), int(round(col))
            if 0 <= ri < elev.shape[0] and 0 <= ci < elev.shape[1]:
                val = elev[ri, ci]
                if not np.isnan(val):
                    all_profiles[i, j] = val

    median_profile = np.nanmedian(all_profiles, axis=0)

    result = TerraceResult(
        dtm_file="MOLA_DEM",
        crater_center_lat=crater_lat,
        crater_center_lon=crater_lon,
        crater_radius_m=diameter_km * 500.0,  # radius_m = diameter_km / 2 * 1000
        n_azimuths=n_azimuths,
        all_profiles=all_profiles,
        radial_distances=radii,
        median_profile=median_profile,
    )

    # Estimate crater radius from profile
    if not np.all(np.isnan(median_profile)):
        smooth_med = _smooth_1d(median_profile, window=max(3, int(300 / radial_step_m)))
        grad = np.gradient(smooth_med)
        for k in range(3, len(grad) - 1):
            if grad[k] > 0 and grad[k + 1] <= 0 and radii[k] > 200:
                result.crater_radius_m = float(radii[k])
                break

    # Terrace detection (works on TerraceResult in-place)
    _detect_terraces(result, radial_step_m)
    _compute_depths(result)

    # Warn if MOLA resolution is insufficient for small craters
    if diameter_km < 5.0:
        logger.warning(
            "MOLA resolution (~200m/px) insufficient for %.1fkm crater; "
            "terrace detection may be unreliable", diameter_km,
        )

    return result
