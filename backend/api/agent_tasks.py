"""
Agentic AI Task Implementations.

Each task is an atomic operation that the agent orchestrator can execute:
- search_region: Find products for an instrument in a geographic area
- check_local_data: Check which products are already downloaded
- download_data: Download missing products
- slope_analysis: Compute slope/terrain feasibility
- subsurface_scan: Analyze SHARAD subsurface data availability
- mineral_analysis: Check CRISM ice/hydration scores
- synthesize: Combine all results into a composite assessment
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Callable, Awaitable
from dataclasses import dataclass, field

import aiohttp

from .ode_client import (
    Instrument,
    search_ode_spatial,
    search_sharad_spatial,
    search_sharad_highres_spatial,
)
from .download_manager import (
    download_manager,
    check_local_existence,
    check_local_existence_detailed,
    DownloadStatus,
)
from .ai_search import _search_local_index, haversine_distance_km
from .terrain_router import compute_slope_stats
from .mars_regions import find_region

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TaskResult:
    """Result from executing an agent task."""
    task_type: str
    instrument: Optional[str] = None
    success: bool = True
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    summary: str = ""


@dataclass
class RegionBBox:
    """Bounding box for a region."""
    min_lat: float
    max_lat: float
    min_lon: float  # -180 to 180
    max_lon: float  # -180 to 180

    @property
    def center_lat(self) -> float:
        return (self.min_lat + self.max_lat) / 2

    @property
    def center_lon(self) -> float:
        return (self.min_lon + self.max_lon) / 2

    @property
    def western_lon_360(self) -> float:
        """ODE uses 0-360 longitude."""
        w = self.min_lon
        if w < 0:
            w += 360
        return w

    @property
    def eastern_lon_360(self) -> float:
        e = self.max_lon
        if e < 0:
            e += 360
        return e


# =============================================================================
# Region Resolution
# =============================================================================

def resolve_region(region_name: str) -> Optional[RegionBBox]:
    """Resolve a named region to a bounding box."""
    region = find_region(region_name)
    if region:
        return RegionBBox(
            min_lat=region.lat_min,
            max_lat=region.lat_max,
            min_lon=region.lon_min,
            max_lon=region.lon_max,
        )
    return None


def bbox_from_coords(lat: float, lon: float, radius_deg: float = 2.0) -> RegionBBox:
    """Create a bounding box from center coordinates and radius."""
    return RegionBBox(
        min_lat=max(-90, lat - radius_deg),
        max_lat=min(90, lat + radius_deg),
        min_lon=lon - radius_deg,
        max_lon=lon + radius_deg,
    )


# =============================================================================
# Task: Search Region
# =============================================================================

async def search_region(
    instrument: str,
    bbox: RegionBBox,
    max_results: int = 500,
) -> TaskResult:
    """
    Search for products of a given instrument within a bounding box.

    Works with both ODE-based instruments (CRISM, HIRISE, SHARAD)
    and local-index instruments (CTX, HIRISE_DTM, SHARAD_HIGHRES).
    """
    instrument_upper = instrument.upper()
    results = []

    try:
        # Local-index instruments
        if instrument_upper in ("CTX", "HIRISE_DTM", "SHARAD_HIGHRES"):
            local_results = _search_local_index(
                instrument_upper,
                bbox.min_lat, bbox.max_lat,
                bbox.min_lon, bbox.max_lon,
                bbox.center_lat, bbox.center_lon,
                max_results,
            )
            for r in local_results:
                results.append({
                    "product_id": r.product_id,
                    "instrument": r.instrument,
                    "lat": r.lat,
                    "lon": r.lon,
                    "distance_km": r.distance_km,
                    "local": True,
                })
        else:
            # ODE-based instruments
            async with aiohttp.ClientSession() as session:
                products = []
                if instrument_upper == "CRISM":
                    products = await search_ode_spatial(
                        bbox.min_lat, bbox.max_lat,
                        bbox.western_lon_360, bbox.eastern_lon_360,
                        Instrument.CRISM,
                        max_results=max_results,
                        session=session,
                    )
                elif instrument_upper == "HIRISE":
                    products = await search_ode_spatial(
                        bbox.min_lat, bbox.max_lat,
                        bbox.western_lon_360, bbox.eastern_lon_360,
                        Instrument.HIRISE,
                        max_results=max_results,
                        session=session,
                    )
                elif instrument_upper == "SHARAD":
                    products = await search_sharad_spatial(
                        bbox.min_lat, bbox.max_lat,
                        bbox.western_lon_360, bbox.eastern_lon_360,
                        max_results=max_results,
                        session=session,
                    )

                for p in products:
                    dist = None
                    if p.lat is not None and p.lon is not None:
                        dist = round(haversine_distance_km(
                            bbox.center_lat, bbox.center_lon, p.lat, p.lon
                        ), 2)
                    results.append({
                        "product_id": p.product_id,
                        "instrument": instrument_upper,
                        "lat": p.lat,
                        "lon": p.lon,
                        "distance_km": dist,
                        "local": False,
                    })

        return TaskResult(
            task_type="search_region",
            instrument=instrument_upper,
            success=True,
            data={"products": results, "count": len(results)},
            summary=f"Found {len(results)} {instrument_upper} products",
        )
    except Exception as e:
        logger.error(f"search_region error for {instrument}: {e}")
        return TaskResult(
            task_type="search_region",
            instrument=instrument_upper,
            success=False,
            error=str(e),
            summary=f"Failed to search {instrument_upper}: {e}",
        )


# =============================================================================
# Task: Check Local Data
# =============================================================================

def check_local_data(products: List[Dict[str, Any]]) -> TaskResult:
    """
    Check which products are already downloaded locally.

    Args:
        products: List of product dicts with 'product_id' and 'instrument' keys.

    Returns:
        TaskResult with lists of available and missing products.
    """
    available = []
    missing = []

    for p in products:
        pid = p["product_id"]
        inst_str = p["instrument"].upper()

        try:
            inst_enum = Instrument(inst_str.lower())
        except ValueError:
            # Local-index instruments (CTX, HIRISE_DTM) are always "available"
            if inst_str in ("CTX", "HIRISE_DTM", "SHARAD_HIGHRES"):
                available.append(p)
            else:
                missing.append(p)
            continue

        existence = check_local_existence_detailed(pid, inst_enum)
        if existence.exists:
            available.append({**p, "has_core": True, "has_browse": existence.has_browse})
        else:
            missing.append({**p, "missing_files": existence.missing_files})

    return TaskResult(
        task_type="check_local_data",
        success=True,
        data={
            "available": available,
            "missing": missing,
            "available_count": len(available),
            "missing_count": len(missing),
        },
        summary=f"{len(available)} available locally, {len(missing)} need download",
    )


# =============================================================================
# Task: Download Data
# =============================================================================

async def download_data(
    products: List[Dict[str, Any]],
    max_concurrent: int = 3,
    on_progress: Optional[Callable[[int, int, int, int], Awaitable[None]]] = None,
) -> TaskResult:
    """
    Download missing products. Starts downloads and waits for completion.

    Args:
        products: List of product dicts with 'product_id', 'instrument', 'lat', 'lon'.
        max_concurrent: Max concurrent downloads.

    Returns:
        TaskResult with download statuses.
    """
    if not products:
        return TaskResult(
            task_type="download_data",
            success=True,
            data={"downloaded": [], "failed": [], "skipped": []},
            summary="No products to download",
        )

    downloaded = []
    failed = []
    skipped = []

    semaphore = asyncio.Semaphore(max_concurrent)

    async def download_one(p: Dict[str, Any]):
        async with semaphore:
            pid = p["product_id"]
            inst_str = p["instrument"].upper()

            # Skip local-index instruments (already available)
            if inst_str in ("CTX", "HIRISE_DTM"):
                skipped.append(pid)
                if on_progress:
                    await on_progress(len(downloaded), len(failed), len(skipped), len(products))
                return

            try:
                inst_enum = Instrument(inst_str.lower())
            except ValueError:
                skipped.append(pid)
                if on_progress:
                    await on_progress(len(downloaded), len(failed), len(skipped), len(products))
                return

            # Skip if already exists
            if check_local_existence(pid, inst_enum):
                skipped.append(pid)
                if on_progress:
                    await on_progress(len(downloaded), len(failed), len(skipped), len(products))
                return

            try:
                task = await download_manager.start_download(
                    product_id=pid,
                    instrument=inst_enum,
                    lat=p.get("lat"),
                    lon=p.get("lon"),
                )

                # Poll until complete (max 10 min per product)
                for _ in range(600):
                    t = download_manager.get_task(task.task_id)
                    if not t:
                        break
                    if t.status == DownloadStatus.COMPLETED:
                        downloaded.append(pid)
                        if on_progress:
                            await on_progress(len(downloaded), len(failed), len(skipped), len(products))
                        return
                    if t.status == DownloadStatus.FAILED:
                        failed.append({"product_id": pid, "error": t.error})
                        if on_progress:
                            await on_progress(len(downloaded), len(failed), len(skipped), len(products))
                        return
                    await asyncio.sleep(1)

                failed.append({"product_id": pid, "error": "Timeout"})
            except Exception as e:
                failed.append({"product_id": pid, "error": str(e)})
            if on_progress:
                await on_progress(len(downloaded), len(failed), len(skipped), len(products))

    await asyncio.gather(*[download_one(p) for p in products])

    return TaskResult(
        task_type="download_data",
        success=len(failed) == 0,
        data={
            "downloaded": downloaded,
            "failed": failed,
            "skipped": skipped,
            "downloaded_count": len(downloaded),
            "failed_count": len(failed),
        },
        summary=f"Downloaded {len(downloaded)}, failed {len(failed)}, skipped {len(skipped)}",
    )


# =============================================================================
# Task: Slope Analysis — Multi-point grid
# =============================================================================

def _assess_safety(stats: dict) -> str:
    """Classify a single slope-stats dict into a safety rating."""
    if stats["count"] == 0:
        return "UNKNOWN"
    pct_below_5 = stats["distribution"]["0_3"] + stats["distribution"]["3_5"]
    if pct_below_5 >= 95:
        return "FAVORABLE"
    if stats["mean_slope"] < 5 and stats["distribution"]["5_plus"] < 10:
        return "MARGINAL"
    return "UNFAVORABLE"


def slope_analysis(
    lat: float,
    lon: float,
    radius_m: float = 5000,
    bbox: Optional['RegionBBox'] = None,
) -> TaskResult:
    """
    Multi-point slope grid analysis for engineering feasibility.

    Samples a 5×5 grid across the bbox (or around lat/lon),
    identifies safest zones with coordinates, and provides
    detailed slope statistics.
    """
    try:
        grid_points = []

        if bbox:
            # Sample a 5×5 grid across the bounding box
            n_lat, n_lon = 5, 5
            lat_step = (bbox.max_lat - bbox.min_lat) / (n_lat - 1) if n_lat > 1 else 0
            lon_step = (bbox.max_lon - bbox.min_lon) / (n_lon - 1) if n_lon > 1 else 0
            for i in range(n_lat):
                for j in range(n_lon):
                    g_lat = bbox.min_lat + i * lat_step
                    g_lon = bbox.min_lon + j * lon_step
                    grid_points.append((round(g_lat, 3), round(g_lon, 3)))
        else:
            # Fallback: 3×3 grid around centre
            for dlat in [-0.5, 0, 0.5]:
                for dlon in [-0.5, 0, 0.5]:
                    grid_points.append((round(lat + dlat, 3), round(lon + dlon, 3)))

        # Analyse each grid point
        grid_results = []
        best_point = None
        best_score = -1  # higher = safer

        for g_lat, g_lon in grid_points:
            try:
                stats = compute_slope_stats(g_lat, g_lon, radius_m=2000)
                safety = _assess_safety(stats)
                pct_safe = stats["distribution"]["0_3"] + stats["distribution"]["3_5"]

                point_data = {
                    "lat": g_lat,
                    "lon": g_lon,
                    "mean_slope": stats["mean_slope"],
                    "max_slope": stats["max_slope"],
                    "elevation_m": stats["elevation_m"],
                    "pct_below_5deg": round(pct_safe, 1),
                    "safety": safety,
                }
                grid_results.append(point_data)

                if pct_safe > best_score:
                    best_score = pct_safe
                    best_point = point_data
            except Exception:
                # Skip points outside DEM coverage
                continue

        if not grid_results:
            raise RuntimeError("No valid slope data at any grid point")

        # Summary stats across grid
        mean_slopes = [p["mean_slope"] for p in grid_results]
        favorable_count = sum(1 for p in grid_results if p["safety"] == "FAVORABLE")
        marginal_count = sum(1 for p in grid_results if p["safety"] == "MARGINAL")

        overall_safety = "FAVORABLE" if favorable_count > len(grid_results) * 0.6 else \
                         "MARGINAL" if (favorable_count + marginal_count) > len(grid_results) * 0.5 else \
                         "UNFAVORABLE"

        result_data = {
            "grid_points": grid_results,
            "grid_size": len(grid_results),
            "best_point": best_point,
            "safety": overall_safety,
            "mean_slope": round(sum(mean_slopes) / len(mean_slopes), 2),
            "max_slope": round(max(p["max_slope"] for p in grid_results), 2),
            "elevation_m": best_point["elevation_m"] if best_point else 0,
            "favorable_zones": favorable_count,
            "marginal_zones": marginal_count,
            "unfavorable_zones": len(grid_results) - favorable_count - marginal_count,
            "distribution": {
                "0_3": round(sum(p["pct_below_5deg"] for p in grid_results) / len(grid_results) * 0.6, 1),
                "3_5": round(sum(p["pct_below_5deg"] for p in grid_results) / len(grid_results) * 0.4, 1),
                "5_plus": round(100 - sum(p["pct_below_5deg"] for p in grid_results) / len(grid_results), 1),
            },
        }

        summary = (
            f"Grid {len(grid_results)} pts: {favorable_count} FAVORABLE, "
            f"{marginal_count} MARGINAL. Best site ({best_point['lat']:.2f}, "
            f"{best_point['lon']:.2f}) mean {best_point['mean_slope']}deg, "
            f"{best_point['pct_below_5deg']:.0f}% below 5deg"
        ) if best_point else f"Overall: {overall_safety}"

        return TaskResult(
            task_type="slope_analysis",
            success=True,
            data=result_data,
            summary=summary,
        )
    except Exception as e:
        logger.error(f"slope_analysis error: {e}")
        return TaskResult(
            task_type="slope_analysis",
            success=False,
            error=str(e),
            summary=f"Slope analysis failed: {e}",
        )


# =============================================================================
# Task: Subsurface Analysis (SHARAD — real radargram analysis)
# =============================================================================

# Physics constants for SHARAD depth conversion
_SPEED_OF_LIGHT = 299_792_458.0          # m/s
_SHARAD_DT_US   = 3.0 / 80.0             # 0.0375 µs per range-bin (1/26.67 MHz ADC)
_ICE_EPSILON    = 3.15                    # pure water-ice εr

def subsurface_scan(products: List[Dict[str, Any]]) -> TaskResult:
    """
    Real SHARAD subsurface analysis.

    For each locally-available SHARAD_HIGHRES product:
      1. Load RDR binary (power array + geometry)
      2. Auto-pick surface return
      3. Scan for subsurface reflectors below surface
      4. Estimate ice-table depth using εr = 3.15 (water-ice)

    Falls back to coverage count for products not downloaded.
    """
    import numpy as np
    import os

    sharad = [p for p in products if p.get("instrument") in ("SHARAD", "SHARAD_HIGHRES")]
    sharad_standard = [p for p in sharad if p.get("instrument") == "SHARAD"]
    sharad_highres = [p for p in sharad if p.get("instrument") == "SHARAD_HIGHRES"]
    total = len(sharad)

    # Coverage quality
    if total >= 10:
        coverage = "EXCELLENT"
    elif total >= 5:
        coverage = "GOOD"
    elif total >= 1:
        coverage = "LIMITED"
    else:
        coverage = "NONE"

    # ── Real analysis for downloaded hi-res products ──
    analyzed_tracks: List[Dict[str, Any]] = []
    subsurface_detections = 0

    try:
        from .sharad_highres_router import (
            _get_power, _get_geometry, _pick_surface, _lon_to_180,
            SHARAD_HR_DIR,
        )
    except ImportError:
        SHARAD_HR_DIR = None

    for p in sharad_highres:
        pid = p["product_id"]
        track_info: Dict[str, Any] = {
            "product_id": pid,
            "analyzed": False,
            "lat": p.get("lat"),
            "lon": p.get("lon"),
        }

        if SHARAD_HR_DIR is None:
            analyzed_tracks.append(track_info)
            continue

        # Check if RDR data file exists locally
        dat_path = os.path.join(SHARAD_HR_DIR, f"{pid.lower()}.dat")
        if not os.path.exists(dat_path):
            analyzed_tracks.append(track_info)
            continue

        try:
            power, n_traces = _get_power(pid)
            geom, _ = _get_geometry(pid)
            surface = _pick_surface(pid, power)
            lons180 = _lon_to_180(geom["lon"])

            valid_surface = surface >= 0
            n_valid = int(valid_surface.sum())

            if n_valid < 10:
                track_info["analyzed"] = True
                track_info["surface_detection_pct"] = round(n_valid / n_traces * 100, 1)
                track_info["subsurface_detected"] = False
                analyzed_tracks.append(track_info)
                continue

            # ── Scan for subsurface reflector ──
            # Look 20-200 bins below surface for secondary power peaks
            n_bins = power.shape[1]
            sub_detect_count = 0
            depth_estimates: List[float] = []
            subsurface_picks: List[Dict[str, Any]] = []

            # Sample every 10th valid trace for speed
            valid_indices = np.where(valid_surface)[0]
            sample_step = max(1, len(valid_indices) // 100)
            sampled = valid_indices[::sample_step]

            for idx in sampled:
                s_bin = int(surface[idx])
                search_lo = min(s_bin + 20, n_bins - 1)
                search_hi = min(s_bin + 200, n_bins)
                if search_hi <= search_lo:
                    continue

                band = power[idx, search_lo:search_hi].astype(np.float64)
                noise = float(np.median(band)) + 1e-12
                peak_idx = int(np.argmax(band))
                peak_val = float(band[peak_idx])
                snr = peak_val / noise

                if snr >= 2.5:  # subsurface reflector threshold
                    sub_detect_count += 1
                    delta_bins = (search_lo + peak_idx) - s_bin
                    dt_s = delta_bins * _SHARAD_DT_US * 1e-6
                    v_ice = _SPEED_OF_LIGHT / np.sqrt(_ICE_EPSILON)
                    depth_m = v_ice * dt_s / 2.0
                    depth_estimates.append(float(depth_m))
                    subsurface_picks.append({
                        "trace_idx": int(idx),
                        "bin_idx": search_lo + peak_idx,
                        "depth_m": round(float(depth_m), 1),
                        "snr": round(float(snr), 2),
                    })

            detection_pct = round(sub_detect_count / len(sampled) * 100, 1) if sampled.size > 0 else 0

            track_info["analyzed"] = True
            track_info["n_traces"] = int(n_traces)
            track_info["surface_detection_pct"] = round(n_valid / n_traces * 100, 1)
            track_info["subsurface_detected"] = detection_pct > 15
            track_info["subsurface_detection_pct"] = detection_pct
            track_info["lat_range"] = [round(float(geom["lat"].min()), 3), round(float(geom["lat"].max()), 3)]
            track_info["lon_range"] = [round(float(lons180.min()), 3), round(float(lons180.max()), 3)]

            if depth_estimates:
                track_info["estimated_depth_m"] = {
                    "min": round(min(depth_estimates), 1),
                    "max": round(max(depth_estimates), 1),
                    "median": round(float(np.median(depth_estimates)), 1),
                    "epsilon_r": _ICE_EPSILON,
                }
                subsurface_detections += 1
            if subsurface_picks:
                track_info["subsurface_picks"] = subsurface_picks

        except Exception as e:
            logger.warning(f"SHARAD analysis failed for {pid}: {e}")
            track_info["error"] = str(e)

        analyzed_tracks.append(track_info)

    # For standard SHARAD — just count (no local data typically)
    for p in sharad_standard:
        analyzed_tracks.append({
            "product_id": p["product_id"],
            "analyzed": False,
            "lat": p.get("lat"),
            "lon": p.get("lon"),
        })

    n_analyzed = sum(1 for t in analyzed_tracks if t.get("analyzed"))
    n_with_subsurface = sum(1 for t in analyzed_tracks if t.get("subsurface_detected"))

    # Compute depth summary across all tracks
    all_depths = []
    for t in analyzed_tracks:
        if "estimated_depth_m" in t:
            all_depths.append(t["estimated_depth_m"]["median"])

    depth_summary = None
    if all_depths:
        depth_summary = {
            "min_depth_m": round(min(all_depths), 1),
            "max_depth_m": round(max(all_depths), 1),
            "median_depth_m": round(float(np.median(all_depths)), 1),
            "n_tracks": len(all_depths),
            "epsilon_r_assumed": _ICE_EPSILON,
        }

    summary_parts = [f"SHARAD: {coverage} ({total} tracks)"]
    if n_analyzed > 0:
        summary_parts.append(f"{n_analyzed} analyzed")
    if n_with_subsurface > 0:
        summary_parts.append(f"{n_with_subsurface} with subsurface reflectors")
    if depth_summary:
        summary_parts.append(
            f"depth {depth_summary['min_depth_m']}-{depth_summary['max_depth_m']}m "
            f"(εr={_ICE_EPSILON})"
        )

    return TaskResult(
        task_type="subsurface_scan",
        success=True,
        data={
            "total_tracks": total,
            "sharad_standard": len(sharad_standard),
            "sharad_highres": len(sharad_highres),
            "coverage": coverage,
            "analyzed_count": n_analyzed,
            "subsurface_detections": n_with_subsurface,
            "depth_summary": depth_summary,
            "tracks": analyzed_tracks,
        },
        summary=", ".join(summary_parts),
    )


# =============================================================================
# Task: Mineral Analysis (CRISM ice/hydration scores)
# =============================================================================

def mineral_analysis(products: List[Dict[str, Any]]) -> TaskResult:
    """
    Detailed CRISM ice/hydration analysis with spatial coordinates.

    Per-product scores with lat/lon, ranked by ice probability.
    Identifies spatial clusters of high-ice observations.
    """
    import os
    import json
    import re

    crism = [p for p in products if p.get("instrument") == "CRISM"]

    if not crism:
        return TaskResult(
            task_type="mineral_analysis",
            success=True,
            data={"crism_count": 0, "scored": [], "coverage": "NONE"},
            summary="No CRISM products for mineral analysis",
        )

    # Load score stats
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    score_stats_file = os.path.join(base_dir, "crism_score", "score_stats.json")

    score_stats = {}
    if os.path.exists(score_stats_file):
        with open(score_stats_file, "r") as f:
            score_stats = json.load(f)

    scored_products = []
    high_ice = 0
    high_hyd = 0

    for p in crism:
        pid = p["product_id"]
        match = re.match(r'^([a-z]{3}[0-9a-f]{8})', pid.lower())
        obs_id = match.group(1) if match else pid.lower()

        stats = score_stats.get(obs_id, {})
        ice_stats = stats.get("ice", {})
        hyd_stats = stats.get("hyd", {})

        ice_pct = 0
        hyd_pct = 0
        ice_mean = 0
        ice_max = 0
        valid = ice_stats.get("valid_pixels", 0)
        if valid > 0:
            ice_above = ice_stats.get("threshold_counts", {}).get("0.3", 0)
            hyd_above = hyd_stats.get("threshold_counts", {}).get("0.3", 0)
            ice_pct = round((ice_above / valid) * 100, 1)
            hyd_pct = round((hyd_above / valid) * 100, 1)
            ice_mean = round(ice_stats.get("mean_score", 0), 3)
            ice_max = round(ice_stats.get("max_score", 0), 3)

        if ice_pct >= 5:
            high_ice += 1
        if hyd_pct >= 5:
            high_hyd += 1

        scored_products.append({
            "product_id": pid,
            "obs_id": obs_id,
            "lat": p.get("lat"),
            "lon": p.get("lon"),
            "ice_percent": ice_pct,
            "ice_mean_score": ice_mean,
            "ice_max_score": ice_max,
            "hyd_percent": hyd_pct,
            "has_score": bool(stats),
        })

    # Rank by ice score (highest first)
    scored_products.sort(key=lambda x: x["ice_percent"], reverse=True)

    # Identify top ice candidates with locations
    top_ice = [s for s in scored_products if s["ice_percent"] >= 5 and s["lat"] is not None and s["lon"] is not None][:5]

    # Spatial cluster: find the center of high-ice observations
    ice_hotspot = None
    if top_ice:
        avg_lat = sum(s["lat"] for s in top_ice) / len(top_ice)
        avg_lon = sum(s["lon"] for s in top_ice) / len(top_ice)
        ice_hotspot = {
            "center_lat": round(avg_lat, 3),
            "center_lon": round(avg_lon, 3),
            "n_products": len(top_ice),
            "max_ice_pct": top_ice[0]["ice_percent"],
        }

    summary_parts = [f"{len(crism)} CRISM: {high_ice} high-ice, {high_hyd} high-hyd"]
    if top_ice:
        best = top_ice[0]
        summary_parts.append(
            f"Best: {best['obs_id']} ({best['ice_percent']}% ice at {best['lat']:.2f}, {best['lon']:.2f})"
        )
    if ice_hotspot:
        summary_parts.append(
            f"Ice hotspot near ({ice_hotspot['center_lat']:.2f}, {ice_hotspot['center_lon']:.2f})"
        )

    return TaskResult(
        task_type="mineral_analysis",
        success=True,
        data={
            "crism_count": len(crism),
            "scored": scored_products,
            "high_ice_count": high_ice,
            "high_hyd_count": high_hyd,
            "scored_count": sum(1 for s in scored_products if s["has_score"]),
            "top_ice_candidates": top_ice,
            "ice_hotspot": ice_hotspot,
        },
        summary=", ".join(summary_parts),
    )


# =============================================================================
# Task: CNN Mineral Classification (1D CNN-Attention on TRR3+DDR)
# =============================================================================

async def mineral_cnn_classify(products: List[Dict[str, Any]]) -> TaskResult:
    """
    Run 1D CNN-Attention mineral classification on locally-available CRISM TRR3 data.

    For each CRISM product with local TRR3 data:
      1. Check for cached CNN result on disk
      2. If not cached, run full classification pipeline (TRR3 + DDR + JCAT + CNN)
      3. Aggregate mineral class distributions across all observations

    Returns mineral distribution, top minerals, and ice detection flags.
    """
    import os
    import re
    import numpy as np

    crism = [p for p in products if p.get("instrument") == "CRISM"]

    if not crism:
        return TaskResult(
            task_type="mineral_cnn",
            success=True,
            data={"crism_count": 0, "classified_count": 0, "top_minerals": [], "ice_detected": False},
            summary="No CRISM products for CNN mineral classification",
        )

    try:
        from .mineral_cnn.pipeline import has_cached_result, load_cached_result, run_classification
        from .mineral_cnn.constants import CLASS_NAME, TRR_DATA_DIR
    except ImportError as e:
        return TaskResult(
            task_type="mineral_cnn",
            success=False,
            error=f"CNN module not available: {e}",
            summary="CNN mineral classification unavailable",
        )

    classified_obs: List[Dict[str, Any]] = []
    total_mineral_pixels: Dict[int, int] = {}  # class_id -> pixel count
    ice_types_found: List[str] = []

    for p in crism:
        pid = p["product_id"]
        match = re.match(r'^([a-z]{3}[0-9a-f]{8})', pid.lower())
        obs_id = match.group(1) if match else pid.lower()

        # Check if TRR3 data directory exists locally (try _07, then any suffix)
        data_dir = os.path.join(TRR_DATA_DIR, f"{obs_id}_07")
        if not os.path.isdir(data_dir):
            import glob as _glob
            candidates = _glob.glob(os.path.join(TRR_DATA_DIR, f"{obs_id}_*"))
            dirs = [c for c in candidates if os.path.isdir(c)]
            if dirs:
                data_dir = dirs[0]
            else:
                continue

        try:
            if has_cached_result(obs_id):
                result = load_cached_result(obs_id)
            else:
                # Run classification pipeline, consume all events
                result = None
                async for event in run_classification(obs_id):
                    ev_type = event.get("event", "")
                    if ev_type == "error":
                        logger.warning(f"CNN classification error for {obs_id}: {event.get('data', {}).get('error')}")
                        break
                    if ev_type in ("complete", "cached"):
                        result = load_cached_result(obs_id)
                        break

                if result is None:
                    continue

            # Aggregate class stats
            total_classified = 0
            top_class_id = -1
            top_class_count = 0

            for cls_id_str, count in result.class_stats.items():
                cls_id = int(cls_id_str)
                if cls_id < 0:
                    continue  # skip unclassified
                total_mineral_pixels[cls_id] = total_mineral_pixels.get(cls_id, 0) + count
                total_classified += count
                if count > top_class_count:
                    top_class_count = count
                    top_class_id = cls_id

            # Check for ice
            for ice_id, ice_name in [(1, "CO2 Ice"), (2, "H2O Ice")]:
                ice_str = str(ice_id)
                if ice_str in result.class_stats and result.class_stats[ice_str] > 0:
                    if ice_name not in ice_types_found:
                        ice_types_found.append(ice_name)

            classified_obs.append({
                "obs_id": obs_id,
                "lat": p.get("lat"),
                "lon": p.get("lon"),
                "total_classified_pixels": total_classified,
                "top_mineral": CLASS_NAME.get(top_class_id, "Unknown"),
                "top_mineral_pixels": top_class_count,
            })

        except Exception as e:
            logger.warning(f"CNN classification failed for {obs_id}: {e}")
            continue

    # Build top minerals list
    sorted_minerals = sorted(total_mineral_pixels.items(), key=lambda x: x[1], reverse=True)
    top_minerals = [
        {"class_id": cid, "name": CLASS_NAME.get(cid, f"Class_{cid}"), "total_pixels": count}
        for cid, count in sorted_minerals[:5]
    ]

    ice_detected = bool(ice_types_found)

    summary_parts = [f"CNN classified {len(classified_obs)}/{len(crism)} CRISM observations"]
    if top_minerals:
        summary_parts.append(f"Top: {top_minerals[0]['name']} ({top_minerals[0]['total_pixels']} px)")
    if ice_detected:
        summary_parts.append(f"Ice detected: {', '.join(ice_types_found)}")

    return TaskResult(
        task_type="mineral_cnn",
        success=True,
        data={
            "crism_count": len(crism),
            "classified_count": len(classified_obs),
            "observations": classified_obs,
            "top_minerals": top_minerals,
            "ice_detected": ice_detected,
            "ice_types": ice_types_found,
            "total_mineral_pixels": {str(k): v for k, v in total_mineral_pixels.items()},
        },
        summary=", ".join(summary_parts),
    )


# =============================================================================
# Task: Dielectric Constant Estimation (SHARAD + HiRISE DTM)
# =============================================================================

def dielectric_analysis(
    subsurface_result: Optional[TaskResult],
    products: List[Dict[str, Any]],
    bbox: "RegionBBox",
) -> TaskResult:
    """
    Estimate dielectric constant by combining SHARAD radar two-way travel time
    with HiRISE DTM elevation data.

    For each SHARAD track with subsurface reflectors:
      1. Find nearest HiRISE DTM product
      2. Extract terrain relief from DTM as geometric depth proxy
      3. Back-compute raw SHARAD two-way travel time from assumed-εr depth
      4. Compute εr = (c * t / (2 * geometric_depth))²
      5. Classify material based on εr value

    εr interpretation:
      < 2.5  → dry regolith or porous ice
      2.5-3.5 → ice-rich subsurface
      3.5-5.0 → ice-cemented regolith
      > 5.0  → basalt / solid rock
    """
    import numpy as np
    import os
    import math

    if not subsurface_result or not subsurface_result.success:
        return TaskResult(
            task_type="dielectric",
            success=True,
            data={"estimates_count": 0},
            summary="No subsurface scan data available for dielectric analysis",
        )

    tracks = subsurface_result.data.get("tracks", [])
    tracks_with_reflectors = [
        t for t in tracks
        if t.get("subsurface_detected") and t.get("estimated_depth_m")
    ]

    if not tracks_with_reflectors:
        return TaskResult(
            task_type="dielectric",
            success=True,
            data={"estimates_count": 0, "reason": "no_subsurface_reflectors"},
            summary="No SHARAD subsurface reflectors for dielectric estimation",
        )

    # Find HiRISE DTM products
    dtm_products = [p for p in products if p.get("instrument") == "HIRISE_DTM"]

    if not dtm_products:
        return TaskResult(
            task_type="dielectric",
            success=True,
            data={"estimates_count": 0, "reason": "no_dtm_products"},
            summary="No HiRISE DTM products for dielectric estimation",
        )

    try:
        from .terrain_router import compute_hirise_dtm_patch
    except ImportError:
        return TaskResult(
            task_type="dielectric",
            success=False,
            error="terrain_router not available",
            summary="Dielectric estimation unavailable (terrain module missing)",
        )

    estimates: List[Dict[str, Any]] = []

    for track in tracks_with_reflectors:
        track_lat = track.get("lat")
        track_lon = track.get("lon")
        depth_info = track.get("estimated_depth_m", {})
        median_depth = depth_info.get("median")

        if track_lat is None or track_lon is None or median_depth is None:
            continue

        # Find nearest DTM product
        best_dtm = None
        best_dist = float("inf")
        for dtm in dtm_products:
            dlat = dtm.get("lat")
            dlon = dtm.get("lon")
            if dlat is None or dlon is None:
                continue
            dist = math.sqrt((track_lat - dlat) ** 2 + (track_lon - dlon) ** 2)
            if dist < best_dist and dist < 0.5:  # within 0.5 degrees
                best_dist = dist
                best_dtm = dtm

        if best_dtm is None:
            continue

        try:
            dtm_id = best_dtm["product_id"]
            patch = compute_hirise_dtm_patch(dtm_id, track_lat, track_lon, radius_m=2000, grid_size=64)
            elev_grid = patch.get("elevations")
            if elev_grid is None:
                continue

            elev_arr = np.array(elev_grid, dtype=np.float64)
            elev_valid = elev_arr[~np.isnan(elev_arr)]
            if len(elev_valid) < 10:
                continue

            terrain_relief = float(np.max(elev_valid) - np.min(elev_valid))

            if terrain_relief < 50:  # Need significant relief for meaningful εr
                continue

            # Back-compute raw SHARAD two-way travel time from assumed-εr depth
            # depth_m = v_ice * dt_s / 2  where v_ice = c / sqrt(εr_assumed)
            # => dt_s = 2 * depth_m * sqrt(εr_assumed) / c
            dt_s = 2.0 * median_depth * math.sqrt(_ICE_EPSILON) / _SPEED_OF_LIGHT

            # Compute εr = (c * dt_s / (2 * geometric_depth))²
            epsilon_r = (_SPEED_OF_LIGHT * dt_s / (2.0 * terrain_relief)) ** 2

            # Classify material
            if epsilon_r < 2.5:
                material = "dry regolith or porous ice"
            elif epsilon_r < 3.5:
                material = "ice-rich subsurface"
            elif epsilon_r < 5.0:
                material = "ice-cemented regolith"
            else:
                material = "basalt or solid rock"

            estimates.append({
                "track_id": track.get("product_id"),
                "dtm_id": dtm_id,
                "track_lat": track_lat,
                "track_lon": track_lon,
                "sharad_depth_m": median_depth,
                "terrain_relief_m": round(terrain_relief, 1),
                "epsilon_r": round(epsilon_r, 2),
                "material": material,
                "dt_s": dt_s,
            })

        except Exception as e:
            logger.warning(f"Dielectric estimation failed for track {track.get('product_id')}: {e}")
            continue

    if not estimates:
        return TaskResult(
            task_type="dielectric",
            success=True,
            data={
                "estimates_count": 0,
                "reason": "no_colocated_dtm_relief",
                "tracks_attempted": len(tracks_with_reflectors),
                "dtm_count": len(dtm_products),
            },
            summary=f"No co-located DTM with sufficient relief ({len(tracks_with_reflectors)} tracks, {len(dtm_products)} DTMs)",
        )

    eps_values = [e["epsilon_r"] for e in estimates]
    mean_eps = round(float(np.mean(eps_values)), 2)
    median_eps = round(float(np.median(eps_values)), 2)

    # Overall interpretation
    if median_eps < 2.5:
        interpretation = "dry regolith or porous ice"
    elif median_eps < 3.5:
        interpretation = "ice-rich subsurface"
    elif median_eps < 5.0:
        interpretation = "ice-cemented regolith"
    else:
        interpretation = "basalt or solid rock"

    summary_parts = [
        f"Dielectric: εr = {median_eps} (median, {len(estimates)} estimates)",
        f"Interpretation: {interpretation}",
    ]

    return TaskResult(
        task_type="dielectric",
        success=True,
        data={
            "estimates_count": len(estimates),
            "mean_epsilon_r": mean_eps,
            "median_epsilon_r": median_eps,
            "min_epsilon_r": round(min(eps_values), 2),
            "max_epsilon_r": round(max(eps_values), 2),
            "interpretation": interpretation,
            "estimates": estimates,
        },
        summary=", ".join(summary_parts),
    )


# =============================================================================
# Task: Synthesize Results
# =============================================================================

def synthesize_results(
    region_name: str,
    all_results: Dict[str, TaskResult],
    region_context: Optional[str] = None,
) -> TaskResult:
    """
    Combine all analysis results into a composite assessment.

    Produces a structured summary suitable for LLM narrative generation.
    region_context: Optional science context string from mars_science_context.json
    """
    synthesis = {
        "region": region_name,
        "region_science_context": region_context or "",
        "instruments_searched": [],
        "total_products_found": 0,
        "total_available_locally": 0,
        "total_downloaded": 0,
        "engineering_feasibility": {},
        "ice_indicators": {},
        "subsurface_coverage": {},
        "mineral_signatures": {},
        "overall_score": 0,
        "recommendation": "",
    }

    # Collect search results
    for key, result in all_results.items():
        if result.task_type == "search_region" and result.success:
            synthesis["instruments_searched"].append(result.instrument)
            synthesis["total_products_found"] += result.data.get("count", 0)

    # Local data check
    if "check_local" in all_results:
        r = all_results["check_local"]
        synthesis["total_available_locally"] = r.data.get("available_count", 0)

    # Downloads
    if "download" in all_results:
        r = all_results["download"]
        synthesis["total_downloaded"] = r.data.get("downloaded_count", 0)

    # Slope analysis (now with grid data)
    if "slope" in all_results and all_results["slope"].success:
        slope = all_results["slope"].data
        best_pt = slope.get("best_point", {})
        synthesis["engineering_feasibility"] = {
            "safety": slope.get("safety", "UNKNOWN"),
            "mean_slope": slope.get("mean_slope", 0),
            "max_slope": slope.get("max_slope", 0),
            "elevation_m": slope.get("elevation_m", 0),
            "favorable_zones": slope.get("favorable_zones", 0),
            "grid_size": slope.get("grid_size", 0),
            "best_site": {
                "lat": best_pt.get("lat"),
                "lon": best_pt.get("lon"),
                "mean_slope": best_pt.get("mean_slope", 0),
                "pct_below_5deg": best_pt.get("pct_below_5deg", 0),
            } if best_pt else None,
        }

    # SHARAD subsurface (now with real analysis data)
    if "subsurface" in all_results and all_results["subsurface"].success:
        sub = all_results["subsurface"].data
        depth = sub.get("depth_summary")
        synthesis["subsurface_coverage"] = {
            "coverage": sub.get("coverage", "NONE"),
            "total_tracks": sub.get("total_tracks", 0),
            "analyzed_count": sub.get("analyzed_count", 0),
            "subsurface_detections": sub.get("subsurface_detections", 0),
            "depth_summary": depth,
        }

    # CRISM mineral analysis (now with spatial data)
    if "mineral" in all_results and all_results["mineral"].success:
        mineral = all_results["mineral"].data
        synthesis["mineral_signatures"] = {
            "crism_count": mineral.get("crism_count", 0),
            "high_ice_count": mineral.get("high_ice_count", 0),
            "high_hyd_count": mineral.get("high_hyd_count", 0),
            "top_ice_candidates": mineral.get("top_ice_candidates", []),
            "ice_hotspot": mineral.get("ice_hotspot"),
        }

    # CNN mineral classification (deep learning)
    if "mineral_cnn" in all_results and all_results["mineral_cnn"].success:
        cnn = all_results["mineral_cnn"].data
        synthesis["cnn_mineral_classification"] = {
            "observations_classified": cnn.get("classified_count", 0),
            "top_minerals": cnn.get("top_minerals", []),
            "ice_detected": cnn.get("ice_detected", False),
            "ice_types": cnn.get("ice_types", []),
        }

    # Dielectric constant estimation (SHARAD + DTM)
    if "dielectric" in all_results and all_results["dielectric"].success:
        diel = all_results["dielectric"].data
        synthesis["dielectric_analysis"] = {
            "estimates_count": diel.get("estimates_count", 0),
            "mean_epsilon_r": diel.get("mean_epsilon_r"),
            "median_epsilon_r": diel.get("median_epsilon_r"),
            "interpretation": diel.get("interpretation", ""),
        }

    # Climate analysis (MCD parametric model)
    if "climate" in all_results and all_results["climate"].success:
        clim = all_results["climate"].data
        synthesis["climate"] = {
            "climate_score": clim.get("climate_score", 0),
            "climate_summary": clim.get("climate_summary", ""),
            "annual_stats": clim.get("annual_stats", {}),
            "elevation_m": clim.get("elevation_m", 0),
        }

    # Thermal inertia analysis (TES)
    if "thermal_inertia" in all_results and all_results["thermal_inertia"].success:
        ti = all_results["thermal_inertia"].data
        synthesis["thermal_inertia"] = {
            "ti_score": ti.get("ti_score", 0),
            "ti_explanation": ti.get("ti_explanation", ""),
            "ti_median": ti.get("ti_median"),
            "ti_mean": ti.get("ti_mean"),
            "classification": ti.get("classification", ""),
            "distribution_pct": ti.get("distribution_pct", {}),
        }

    # Site recommendation (if available)
    if "recommend" in all_results and all_results["recommend"].success:
        synthesis["recommended_site"] = all_results["recommend"].data

    # Cross-instrument consistency analysis
    cross = _compute_cross_instrument(all_results)
    synthesis["cross_instrument"] = cross

    # ── Composite scoring (max ~95, never 100) ──
    # Rebalanced weights:
    #   SHARAD 0-18, CRISM 0-18, CNN 0-5, Dielectric 0-5,
    #   Cross-inst 0-13, Engineering 0-22, Data 0-9,
    #   Climate 0-10, Thermal Inertia 0-10  → max 110, capped at 95
    score = 0
    strengths = []
    uncertainties = []

    # SHARAD subsurface (0-18) — direct ice evidence, depth-weighted
    sub_cov = synthesis["subsurface_coverage"]
    n_detections = sub_cov.get("subsurface_detections", 0)
    depth_summary = sub_cov.get("depth_summary", {})
    median_depth = depth_summary.get("median_depth_m")

    if n_detections >= 1:
        if median_depth is not None:
            try:
                d = float(median_depth)
            except (TypeError, ValueError):
                d = 50.0
            if d <= 10:
                depth_score = 18 if n_detections >= 2 else 14
                strengths.append(
                    f"Shallow subsurface ice detected at ~{d:.0f} m depth "
                    f"({n_detections} track(s)) — accessible for extraction"
                )
            elif d <= 30:
                depth_score = 12 if n_detections >= 2 else 9
                strengths.append(
                    f"Subsurface reflector at ~{d:.0f} m depth ({n_detections} track(s)) "
                    f"— moderate drilling required"
                )
            elif d <= 80:
                depth_score = 5
                uncertainties.append(
                    f"Subsurface reflector at ~{d:.0f} m depth — deep, "
                    f"significant drilling infrastructure required"
                )
            else:
                depth_score = 2
                uncertainties.append(
                    f"Subsurface reflector at ~{d:.0f} m depth — too deep "
                    f"for practical ice extraction"
                )
        else:
            depth_score = 9 if n_detections >= 2 else 6
            strengths.append(f"SHARAD subsurface reflector detected ({n_detections} track(s)), depth unknown")
        score += depth_score
    elif sub_cov.get("analyzed_count", 0) > 0:
        score += 3
        uncertainties.append("SHARAD radargrams analyzed but no subsurface reflectors detected")
    else:
        uncertainties.append("No SHARAD radargram data available for direct subsurface analysis")

    # CRISM ice/hydration (0-18) — indirect/proxy evidence
    ice_count = synthesis["mineral_signatures"].get("high_ice_count", 0)
    if ice_count >= 3:
        score += 18
        strengths.append(f"Strong CRISM spectral ice signatures ({ice_count} products above threshold)")
    elif ice_count >= 1:
        score += 10
        strengths.append(f"CRISM ice signatures present ({ice_count} product(s))")
    else:
        uncertainties.append("No significant CRISM ice signatures detected")

    # CNN mineral classification bonus (0-5)
    cnn_data = synthesis.get("cnn_mineral_classification", {})
    if cnn_data.get("ice_detected"):
        score += 5
        ice_types = ", ".join(cnn_data.get("ice_types", []))
        strengths.append(f"CNN mineral classifier detected ice phases: {ice_types}")

    # Dielectric constant bonus (0-5)
    diel_data = synthesis.get("dielectric_analysis", {})
    if diel_data.get("estimates_count", 0) > 0:
        eps = diel_data.get("median_epsilon_r", 0)
        interp = diel_data.get("interpretation", "")
        if 2.5 <= eps <= 3.5:
            score += 5
            strengths.append(f"Dielectric constant εr={eps} consistent with ice-rich subsurface")
        elif eps > 5.0:
            uncertainties.append(f"Dielectric constant εr={eps} suggests rock/basalt ({interp})")
        else:
            strengths.append(f"Dielectric constant εr={eps} measured ({interp})")

    # Cross-instrument consistency (0-13)
    consistency = cross.get("evidence_consistency", "insufficient_data")
    if consistency == "consistent":
        score += 13
        strengths.append("SHARAD and CRISM ice evidence spatially consistent")
    elif consistency == "partial":
        score += 7
    elif consistency in ("sharad_only", "crism_only"):
        score += 4
        uncertainties.append(f"Ice evidence from single instrument only ({consistency.replace('_', ' ')})")
    else:
        uncertainties.append("No cross-instrument corroboration available")

    # Engineering feasibility (0-22) — final filter, not driver
    safety = synthesis["engineering_feasibility"].get("safety", "UNKNOWN")
    if safety == "FAVORABLE":
        score += 22
        strengths.append("Terrain engineering assessment: FAVORABLE (majority below 5 deg)")
    elif safety == "MARGINAL":
        score += 10
        uncertainties.append("Terrain is marginal — some zones exceed 5 deg slope threshold")
    elif safety == "UNFAVORABLE":
        score += 3
        uncertainties.append("Terrain is largely unfavorable for landing (steep slopes)")
    else:
        uncertainties.append("Slope data unavailable for engineering assessment")

    # Data confidence (0-9) — de-emphasized
    total = synthesis["total_products_found"]
    if total >= 20:
        score += 9
    elif total >= 10:
        score += 6
    elif total >= 5:
        score += 3
    else:
        uncertainties.append(f"Low data coverage ({total} products) limits confidence")

    # Climate assessment (0-10) — NEW
    clim_data = synthesis.get("climate", {})
    if clim_data:
        clim_score = clim_data.get("climate_score", 0)
        score += clim_score
        clim_summary = clim_data.get("climate_summary", "")
        if clim_score >= 8:
            strengths.append(f"Favorable climate conditions: {clim_summary}")
        elif clim_score >= 5:
            strengths.append(f"Acceptable climate: {clim_summary}")
        elif clim_score > 0:
            uncertainties.append(f"Climate challenges: {clim_summary}")
        else:
            uncertainties.append(f"Harsh climate conditions: {clim_summary}")

    # Thermal inertia (0-10) — NEW
    ti_data = synthesis.get("thermal_inertia", {})
    if ti_data:
        ti_score = ti_data.get("ti_score", 0)
        score += ti_score
        ti_expl = ti_data.get("ti_explanation", "")
        if ti_score >= 8:
            strengths.append(ti_expl)
        elif ti_score >= 4:
            strengths.append(ti_expl)
        elif ti_score > 0:
            uncertainties.append(ti_expl)
        # Score 0 with explanation means unfavorable
        elif ti_expl:
            uncertainties.append(ti_expl)

    # Cap at 95 — always leave room for uncertainty
    score = min(score, 95)

    # Score range: +/- based on uncertainty count
    uncertainty_margin = min(len(uncertainties) * 3, 12)
    confidence_bonus = min(len(strengths) * 2, 6)
    score_low = max(0, score - uncertainty_margin)
    score_high = min(95, score + confidence_bonus)

    synthesis["overall_score"] = score
    synthesis["score_range"] = {"low": score_low, "high": score_high}
    synthesis["strengths"] = strengths
    synthesis["uncertainties"] = uncertainties

    # Recommendation — nuanced labels
    if score >= 75:
        synthesis["recommendation"] = "STRONG_CANDIDATE"
    elif score >= 55:
        synthesis["recommendation"] = "PROMISING_WITH_CAVEATS"
    elif score >= 35:
        synthesis["recommendation"] = "REQUIRES_FURTHER_INVESTIGATION"
    else:
        synthesis["recommendation"] = "LOW_PRIORITY"

    # Extract ice_confidence from region context if available
    if region_context and "Ice confidence:" in region_context:
        for line in region_context.split("\n"):
            if line.strip().startswith("Ice confidence:"):
                synthesis["known_ice_confidence"] = line.split(":", 1)[1].strip()
                break

    return TaskResult(
        task_type="synthesize",
        success=True,
        data=synthesis,
        summary=f"Score: {score_low}-{score_high}/100 — {synthesis['recommendation']}",
    )


# =============================================================================
# Task: Recommend Best Rover Site
# =============================================================================

def recommend_site(all_results: Dict[str, TaskResult]) -> TaskResult:
    """
    Cross-reference slope, SHARAD, and CRISM results to recommend
    primary + secondary landing sites and science targets.

    Scoring weights: 40 slope safety, 35 ice proximity, 25 SHARAD proximity.
    Also identifies science targets (high-value CRISM locations regardless of slope).
    """
    candidates: List[Dict[str, Any]] = []

    # Gather slope grid points
    slope_points = {}
    if "slope" in all_results and all_results["slope"].success:
        for pt in all_results["slope"].data.get("grid_points", []):
            key = (pt["lat"], pt["lon"])
            slope_points[key] = pt

    # Gather CRISM ice hotspot
    ice_hotspot = None
    top_ice = []
    if "mineral" in all_results and all_results["mineral"].success:
        ice_hotspot = all_results["mineral"].data.get("ice_hotspot")
        top_ice = all_results["mineral"].data.get("top_ice_candidates", [])

    # Gather SHARAD tracks with subsurface detection
    sharad_detections = []
    if "subsurface" in all_results and all_results["subsurface"].success:
        for t in all_results["subsurface"].data.get("tracks", []):
            if t.get("subsurface_detected"):
                sharad_detections.append(t)

    # Score each slope grid point as a candidate site
    for key, pt in slope_points.items():
        score = 0
        reasons = []

        # Slope score (0-40)
        if pt["safety"] == "FAVORABLE":
            score += 40
            reasons.append(f"Flat terrain: {pt['pct_below_5deg']:.0f}% below 5 deg, mean {pt['mean_slope']} deg")
        elif pt["safety"] == "MARGINAL":
            score += 20
            reasons.append(f"Moderate terrain: mean {pt['mean_slope']} deg")
        else:
            reasons.append(f"Steep terrain: mean {pt['mean_slope']} deg — not ideal for landing")

        # Proximity to ice indicators (0-35)
        if ice_hotspot and pt.get("lat") is not None:
            dist = haversine_distance_km(
                pt["lat"], pt["lon"],
                ice_hotspot["center_lat"], ice_hotspot["center_lon"],
            )
            if dist < 50:
                score += 35
                reasons.append(f"Within {dist:.0f} km of CRISM ice hotspot ({ice_hotspot['max_ice_pct']}% ice)")
            elif dist < 150:
                score += 20
                reasons.append(f"{dist:.0f} km from CRISM ice hotspot")
            elif dist < 300:
                score += 10
                reasons.append(f"{dist:.0f} km from ice indicators")

        # SHARAD subsurface detection nearby (0-25)
        if sharad_detections and pt.get("lat") is not None:
            nearest_sharad = None
            min_dist = float("inf")
            for t in sharad_detections:
                if t.get("lat") is not None:
                    d = haversine_distance_km(pt["lat"], pt["lon"], t["lat"], t["lon"])
                    if d < min_dist:
                        min_dist = d
                        nearest_sharad = t
            if nearest_sharad and min_dist < 100:
                score += 25
                depth_info = nearest_sharad.get("estimated_depth_m", {})
                depth_str = f"~{depth_info.get('median', '?')}m deep" if depth_info else ""
                reasons.append(f"SHARAD subsurface reflector {min_dist:.0f} km away {depth_str}")
            elif nearest_sharad and min_dist < 300:
                score += 12
                reasons.append(f"SHARAD subsurface detection {min_dist:.0f} km away")

        candidates.append({
            "lat": pt["lat"],
            "lon": pt["lon"],
            "score": score,
            "elevation_m": pt["elevation_m"],
            "mean_slope": pt["mean_slope"],
            "reasons": reasons,
        })

    # Sort by score
    candidates.sort(key=lambda c: c["score"], reverse=True)
    primary = candidates[0] if candidates else None
    secondary = candidates[1] if len(candidates) > 1 else None

    # Science targets: high-ice CRISM locations that may not be landable
    science_targets = []
    for c in top_ice[:3]:
        if c.get("lat") is None:
            continue
        # Skip if too close to primary/secondary
        is_landing = False
        for site in [primary, secondary]:
            if site and abs(site["lat"] - c["lat"]) < 0.3 and abs(site["lon"] - c["lon"]) < 0.3:
                is_landing = True
                break
        if not is_landing:
            science_targets.append({
                "lat": c["lat"],
                "lon": c["lon"],
                "obs_id": c.get("obs_id"),
                "ice_percent": c.get("ice_percent"),
                "reason": f"High CRISM ice signature ({c.get('ice_percent')}% ice) — potential traverse target",
            })

    # Trade-off statements
    trade_offs = []
    if primary and secondary:
        if primary["score"] - secondary["score"] < 10:
            trade_offs.append(
                f"Primary and secondary sites are similarly scored "
                f"({primary['score']} vs {secondary['score']}). "
                f"Final selection may depend on mission-specific constraints."
            )
        if primary.get("mean_slope", 99) > secondary.get("mean_slope", 99):
            trade_offs.append(
                f"Primary site has steeper terrain "
                f"({primary['mean_slope']} deg vs {secondary['mean_slope']} deg) "
                f"but was preferred for ice proximity."
            )
    if science_targets:
        trade_offs.append(
            f"{len(science_targets)} science target(s) identified outside the primary "
            f"landing zone that may require extended traverse capability."
        )
    if not sharad_detections and top_ice:
        trade_offs.append(
            "Landing site selection relies on indirect (CRISM spectral) ice evidence; "
            "no SHARAD subsurface reflectors were detected for direct confirmation."
        )

    summary = "No candidate sites found"
    if primary:
        summary = (
            f"Primary: ({primary['lat']:.2f}, {primary['lon']:.2f}) "
            f"score {primary['score']}/100, slope {primary['mean_slope']} deg"
        )
        if secondary:
            summary += f" | Backup: ({secondary['lat']:.2f}, {secondary['lon']:.2f})"

    return TaskResult(
        task_type="recommend_site",
        success=bool(candidates),
        data={
            "candidates": candidates[:10],
            "best_site": primary,  # backward compat
            "primary_site": primary,
            "secondary_site": secondary,
            "science_targets": science_targets,
            "trade_offs": trade_offs,
        },
        summary=summary,
    )


# =============================================================================
# Cross-Instrument Consistency Analysis
# =============================================================================

def _compute_cross_instrument(all_results: Dict[str, 'TaskResult']) -> Dict[str, Any]:
    """
    Compute spatial consistency between SHARAD subsurface detections
    and CRISM ice/hydration signatures.

    Returns a dict with:
    - sharad_crism_min_distance_km: closest approach
    - evidence_consistency: consistent / partial / inconsistent / *_only / insufficient
    - direct_ice_evidence: list of SHARAD reflector detections (direct)
    - indirect_ice_evidence: list of CRISM spectral indicators (proxy)
    - coincident_detections: count of SHARAD tracks near CRISM ice
    - notes: explanatory text for the report
    """
    cross: Dict[str, Any] = {
        "sharad_crism_min_distance_km": None,
        "evidence_consistency": "insufficient_data",
        "direct_ice_evidence": [],
        "indirect_ice_evidence": [],
        "coincident_detections": 0,
        "notes": [],
    }

    # SHARAD detections with locations
    sharad_detections = []
    if "subsurface" in all_results and all_results["subsurface"].success:
        for t in all_results["subsurface"].data.get("tracks", []):
            if t.get("subsurface_detected") and t.get("lat") is not None:
                sharad_detections.append(t)
                cross["direct_ice_evidence"].append({
                    "type": "SHARAD_reflector",
                    "product_id": t["product_id"],
                    "lat": t["lat"],
                    "lon": t["lon"],
                    "depth_m": t.get("estimated_depth_m", {}).get("median"),
                })

    # CRISM ice data
    ice_hotspot = None
    top_ice = []
    crism_has_ice = False  # true if any CRISM ice detected (even without coordinates)
    if "mineral" in all_results and all_results["mineral"].success:
        ice_hotspot = all_results["mineral"].data.get("ice_hotspot")
        top_ice = all_results["mineral"].data.get("top_ice_candidates", [])
        crism_has_ice = all_results["mineral"].data.get("high_ice_count", 0) > 0
        for c in top_ice:
            cross["indirect_ice_evidence"].append({
                "type": "CRISM_spectral",
                "obs_id": c.get("obs_id"),
                "lat": c.get("lat"),
                "lon": c.get("lon"),
                "ice_percent": c.get("ice_percent"),
            })

    # Compute spatial proximity
    if sharad_detections and ice_hotspot:
        min_dist = float("inf")
        for t in sharad_detections:
            d = haversine_distance_km(
                t["lat"], t["lon"],
                ice_hotspot["center_lat"], ice_hotspot["center_lon"],
            )
            min_dist = min(min_dist, d)

        cross["sharad_crism_min_distance_km"] = round(min_dist, 1)

        # Count SHARAD tracks within 100 km of any CRISM high-ice observation
        for t in sharad_detections:
            for c in top_ice:
                if c.get("lat") is not None:
                    d = haversine_distance_km(t["lat"], t["lon"], c["lat"], c["lon"])
                    if d < 100:
                        cross["coincident_detections"] += 1
                        break

        if min_dist < 50:
            cross["evidence_consistency"] = "consistent"
            cross["notes"].append(
                f"SHARAD subsurface reflectors and CRISM ice signatures spatially coincide "
                f"(within {min_dist:.0f} km), providing strong mutual corroboration. "
                f"The SHARAD reflectors represent direct evidence of a dielectric interface "
                f"consistent with buried ice, while CRISM detections indicate surface or "
                f"near-surface ice/hydration signatures."
            )
        elif min_dist < 200:
            cross["evidence_consistency"] = "partial"
            cross["notes"].append(
                f"SHARAD and CRISM evidence are moderately correlated "
                f"({min_dist:.0f} km separation). The ice deposit may be laterally "
                f"discontinuous, or the surface expression (CRISM) may not directly "
                f"overlie the subsurface interface (SHARAD)."
            )
        else:
            cross["evidence_consistency"] = "inconsistent"
            cross["notes"].append(
                f"SHARAD reflectors and CRISM ice signatures do not spatially coincide "
                f"({min_dist:.0f} km apart). This may indicate distinct ice reservoirs — "
                f"a buried deposit detected by radar and a separate surface/near-surface "
                f"exposure detected spectrally."
            )

    elif sharad_detections and not top_ice and not crism_has_ice:
        cross["evidence_consistency"] = "sharad_only"
        cross["notes"].append(
            "Subsurface reflectors detected by SHARAD but no corresponding CRISM surface "
            "ice signatures. This is consistent with buried ice beneath a protective "
            "regolith layer that masks the spectral signature."
        )

    elif sharad_detections and crism_has_ice and not ice_hotspot:
        # CRISM detected ice but without spatial coordinates for proximity analysis
        cross["evidence_consistency"] = "partial"
        cross["notes"].append(
            "Both SHARAD subsurface reflectors and CRISM ice signatures are present, "
            "but CRISM ice candidates lack spatial coordinates for proximity analysis. "
            "Cross-instrument spatial correlation cannot be fully evaluated."
        )

    elif (top_ice or crism_has_ice) and not sharad_detections:
        cross["evidence_consistency"] = "crism_only"
        cross["notes"].append(
            "CRISM spectral signatures suggest surface or near-surface ice/hydration, "
            "but no SHARAD subsurface reflectors were detected. The ice may be too "
            "shallow or too thin for SHARAD vertical resolution (~15 m)."
        )

    else:
        cross["notes"].append(
            "Insufficient data for cross-instrument comparison. Neither SHARAD "
            "subsurface reflectors nor CRISM ice signatures were identified."
        )

    return cross
