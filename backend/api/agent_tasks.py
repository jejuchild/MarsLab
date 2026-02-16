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
from .scoring_methodology import compute_composite_score, classify_evidence_strength, classify_recommendation

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
            # ODE-based instruments — with fallback to local index on failure
            ode_failed = False
            try:
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
            except Exception as ode_err:
                logger.warning(f"ODE search failed for {instrument_upper}, trying local index: {ode_err}")
                ode_failed = True

            # Fallback: try local index if ODE returned nothing or failed
            if ode_failed or len(results) == 0:
                try:
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
                    if local_results:
                        logger.info(f"Local index fallback found {len(local_results)} {instrument_upper} products")
                except Exception:
                    pass  # No local index for this instrument

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

            max_retries = 2
            for attempt in range(max_retries + 1):
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
                            raise RuntimeError(t.error or "Download failed")
                        await asyncio.sleep(1)
                    else:
                        raise RuntimeError("Timeout")

                    # If we got here without returning, task disappeared
                    raise RuntimeError("Download task vanished")

                except Exception as e:
                    if attempt < max_retries:
                        wait = 2 ** (attempt + 1)  # 2s, 4s
                        logger.warning(f"Download {pid} failed (attempt {attempt + 1}), retrying in {wait}s: {e}")
                        await asyncio.sleep(wait)
                        continue
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
            # Sample a 15×15 grid across the bounding box (225 points)
            n_lat, n_lon = 15, 15
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

                # Roughness: use std_slope if available, else proxy estimate
                roughness = stats.get("std_slope", 0)
                if roughness == 0:
                    # Proxy from distribution buckets
                    dist = stats["distribution"]
                    roughness = (dist.get("5_plus", 0) * 10 + dist.get("3_5", 0) * 4 + dist.get("0_3", 0) * 1.5) / 100.0

                point_data = {
                    "lat": g_lat,
                    "lon": g_lon,
                    "mean_slope": stats["mean_slope"],
                    "max_slope": stats["max_slope"],
                    "elevation_m": stats["elevation_m"],
                    "pct_below_5deg": round(pct_safe, 1),
                    "safety": safety,
                    "roughness": round(roughness, 3),
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

        # Roughness statistics across grid
        roughness_values = [p.get("roughness", 0) for p in grid_results]
        roughness_stats = {
            "mean": round(sum(roughness_values) / len(roughness_values), 3) if roughness_values else 0,
            "max": round(max(roughness_values), 3) if roughness_values else 0,
        }

        # Hazard density: fraction of grid points with >5° mean slope
        hazard_density = round(
            sum(1 for p in grid_results if p["mean_slope"] > 5) / len(grid_results), 4
        )

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
            "roughness_stats": roughness_stats,
            "hazard_density": hazard_density,
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
            sample_step = max(1, len(valid_indices) // 10)
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

                if snr >= 4.0:  # subsurface reflector threshold (strict to reduce false positives)
                    sub_detect_count += 1
                    delta_bins = (search_lo + peak_idx) - s_bin

                    # Three dielectric scenarios for depth estimation
                    DIELECTRIC_SCENARIOS = [
                        {"epsilon_r": 2.8, "label": "porous_ice", "description": "Porous ice/regolith mix"},
                        {"epsilon_r": 3.15, "label": "pure_ice", "description": "Pure water ice"},
                        {"epsilon_r": 4.0, "label": "basaltic", "description": "Basaltic contrast interface"},
                    ]
                    depths_multi = {}
                    for scenario in DIELECTRIC_SCENARIOS:
                        eps_r = scenario["epsilon_r"]
                        d_m = (_SPEED_OF_LIGHT * delta_bins * _SHARAD_DT_US * 1e-6) / (2.0 * np.sqrt(eps_r))
                        depths_multi[scenario["label"]] = round(float(d_m), 1)

                    # Keep pure_ice as the backward-compat default depth
                    depth_m = depths_multi["pure_ice"]
                    depth_estimates.append(float(depth_m))
                    subsurface_picks.append({
                        "trace_idx": int(idx),
                        "bin_idx": search_lo + peak_idx,
                        "delta_bins": delta_bins,
                        "snr": round(float(snr), 2),
                        "depths": depths_multi,
                        # backward compat
                        "depth_m": round(float(depth_m), 1),
                        "epsilon_r_source": "assumed",
                    })

            # Coherence filter: real subsurface interfaces have consistent depth
            # Discard if depth varies too wildly (noise, not a real horizontal interface)
            if len(depth_estimates) >= 3:
                depth_arr = np.array(depth_estimates)
                depth_std = float(np.std(depth_arr))
                depth_median = float(np.median(depth_arr))
                # Keep only picks within 2 std of median depth (remove outliers)
                if depth_std > 0 and depth_median > 0:
                    mask = np.abs(depth_arr - depth_median) <= 2.0 * depth_std
                    depth_estimates = depth_arr[mask].tolist()
                    subsurface_picks = [p for p, m in zip(subsurface_picks, mask) if m]
                    sub_detect_count = len(subsurface_picks)
                # If remaining depths still too scattered (CV > 30%), likely noise
                if len(depth_estimates) >= 3:
                    final_std = float(np.std(depth_estimates))
                    final_mean = float(np.mean(depth_estimates))
                    if final_mean > 0 and (final_std / final_mean) > 0.30:
                        depth_estimates = []
                        subsurface_picks = []
                        sub_detect_count = 0

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
                    "epsilon_r_source": "assumed",
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

    # ── SNR Distribution across all detected reflectors ──
    all_snr = [pick["snr"] for t in analyzed_tracks for pick in t.get("subsurface_picks", [])]
    snr_distribution = None
    if all_snr:
        snr_distribution = {
            "mean": round(float(np.mean(all_snr)), 2),
            "median": round(float(np.median(all_snr)), 2),
            "std": round(float(np.std(all_snr)), 2),
            "min": round(float(min(all_snr)), 2),
            "max": round(float(max(all_snr)), 2),
            "histogram": np.histogram(all_snr, bins=[4, 6, 8, 10, 15, 20, 50])[0].tolist(),
        }

    # ── Reflector Spatial Density (detections per km of analyzed track) ──
    total_track_km = 0
    for t in analyzed_tracks:
        if t.get("analyzed") and t.get("lat_range"):
            lat_span = abs(t["lat_range"][1] - t["lat_range"][0])
            track_km = lat_span * 59.27  # 1 deg lat ~ 59.27 km on Mars
            total_track_km += track_km
    reflector_density = round(n_with_subsurface / total_track_km, 3) if total_track_km > 0 else 0

    # ── Multi-scenario depth summary ──
    DIELECTRIC_SCENARIOS_SUMMARY = [
        {"epsilon_r": 2.8, "label": "porous_ice", "description": "Porous ice/regolith mix"},
        {"epsilon_r": 3.15, "label": "pure_ice", "description": "Pure water ice"},
        {"epsilon_r": 4.0, "label": "basaltic", "description": "Basaltic contrast interface"},
    ]

    # Collect per-scenario depths from all picks across all tracks
    all_depths_by_scenario: Dict[str, List[float]] = {"porous_ice": [], "pure_ice": [], "basaltic": []}
    for t in analyzed_tracks:
        for pick in t.get("subsurface_picks", []):
            depths_dict = pick.get("depths", {})
            for label in all_depths_by_scenario:
                val = depths_dict.get(label)
                if val is not None:
                    all_depths_by_scenario[label].append(float(val))

    # Also collect backward-compat median depths per track
    all_depths = []
    for t in analyzed_tracks:
        if "estimated_depth_m" in t:
            all_depths.append(t["estimated_depth_m"]["median"])

    depth_summary = None
    if all_depths_by_scenario["pure_ice"]:
        depth_ranges = {}
        for scenario in DIELECTRIC_SCENARIOS_SUMMARY:
            label = scenario["label"]
            vals = all_depths_by_scenario.get(label, [])
            if vals:
                depth_ranges[label] = {
                    "min": round(min(vals), 1),
                    "max": round(max(vals), 1),
                    "median": round(float(np.median(vals)), 1),
                    "epsilon_r": scenario["epsilon_r"],
                }
        pure_vals = all_depths_by_scenario["pure_ice"]
        depth_summary = {
            "dielectric_scenarios": DIELECTRIC_SCENARIOS_SUMMARY,
            "depth_ranges": depth_ranges,
            "n_tracks": len(all_depths) if all_depths else len(pure_vals),
            # Backward-compat fields (using pure_ice scenario)
            "min_depth_m": round(min(pure_vals), 1),
            "max_depth_m": round(max(pure_vals), 1),
            "median_depth_m": round(float(np.median(pure_vals)), 1),
            "epsilon_r_assumed": _ICE_EPSILON,
            "epsilon_r_source": "assumed",
            "physics_warning": (
                "Depth computed from assumed εr=3.15 (water-ice). "
                "This is NOT an independent physical measurement. "
                "Physics-based dielectric inversion required for validated depth."
            ),
        }
    elif all_depths:
        # Fallback: no multi-scenario picks but old-style depths exist
        depth_summary = {
            "min_depth_m": round(min(all_depths), 1),
            "max_depth_m": round(max(all_depths), 1),
            "median_depth_m": round(float(np.median(all_depths)), 1),
            "n_tracks": len(all_depths),
            "epsilon_r_assumed": _ICE_EPSILON,
            "epsilon_r_source": "assumed",
            "physics_warning": (
                "Depth computed from assumed εr=3.15 (water-ice). "
                "This is NOT an independent physical measurement. "
                "Physics-based dielectric inversion required for validated depth."
            ),
        }

    # ── Clutter rejection methodology note ──
    clutter_rejection = (
        "Subsurface reflectors are identified by SNR >= 4.0 threshold relative to "
        "local noise floor (median power in search window). Coherence filtering "
        "removes detections with coefficient of variation > 30% across sampled traces "
        "to reject off-nadir surface clutter. Remaining outliers beyond 2 sigma of median "
        "depth are removed."
    )

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
            "snr_distribution": snr_distribution,
            "clutter_rejection": clutter_rejection,
            "reflector_density_per_km": reflector_density,
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

    # ── Threshold Statistical Justification ──
    import numpy as np
    all_ice_means = [
        s.get("ice", {}).get("mean_score", 0)
        for s in score_stats.values()
        if s.get("ice", {}).get("valid_pixels", 0) > 0
    ]
    threshold_justification = None
    if all_ice_means:
        bg_mean = float(np.mean(all_ice_means))
        bg_std = float(np.std(all_ice_means))
        sigma_above = (0.3 - bg_mean) / bg_std if bg_std > 0 else float('inf')
        threshold_justification = {
            "threshold": 0.3,
            "background_mean": round(bg_mean, 4),
            "background_std": round(bg_std, 4),
            "sigma_above_background": round(sigma_above, 2),
            "note": f"Threshold 0.3 is {sigma_above:.1f} sigma above the global background mean ({bg_mean:.4f})",
        }

    # ── False Positive Estimate ──
    all_ice_pcts = [s["ice_percent"] for s in scored_products if s["has_score"]]
    false_positive_rate = 0.0
    if all_ice_pcts:
        q25 = float(np.percentile(all_ice_pcts, 25)) if len(all_ice_pcts) >= 4 else 0
        false_positive_products = [p for p in all_ice_pcts if p < q25 and p > 0]
        false_positive_rate = (
            len([p for p in false_positive_products if p >= 5]) / max(len(false_positive_products), 1)
        )

    # ── Spatial Coherence (connected-component proxy) ──
    high_ice_products = [
        p for p in scored_products
        if p["ice_percent"] >= 5 and p.get("lat") is not None and p.get("lon") is not None
    ]
    clusters: List[List[Dict[str, Any]]] = []
    used: set = set()
    for i, p in enumerate(high_ice_products):
        if i in used:
            continue
        cluster = [p]
        used.add(i)
        for j, q in enumerate(high_ice_products):
            if j in used:
                continue
            d = haversine_distance_km(p["lat"], p["lon"], q["lat"], q["lon"])
            if d < 50:
                cluster.append(q)
                used.add(j)
        clusters.append(cluster)

    largest_cluster_extent_km = 0.0
    if clusters:
        largest = max(clusters, key=len)
        if len(largest) >= 2:
            for ci in range(len(largest)):
                for cj in range(ci + 1, len(largest)):
                    d = haversine_distance_km(
                        largest[ci]["lat"], largest[ci]["lon"],
                        largest[cj]["lat"], largest[cj]["lon"],
                    )
                    largest_cluster_extent_km = max(largest_cluster_extent_km, d)

    spatial_coherence = {
        "cluster_count": len(clusters),
        "largest_cluster_size": max(len(c) for c in clusters) if clusters else 0,
        "largest_cluster_extent_km": round(largest_cluster_extent_km, 1),
    }

    # ── Evidence Separation ──
    evidence_separation: Dict[str, List[str]] = {
        "spectral_hydration": [],   # products with hyd_pct >= 5 but ice_pct < 5
        "water_ice": [],            # products with ice_pct >= 5
        "atmospheric_suspect": [],  # products where ice score is suspiciously uniform
    }
    for s in scored_products:
        if s["ice_percent"] >= 5:
            evidence_separation["water_ice"].append(s["obs_id"])
        elif s["hyd_percent"] >= 5:
            evidence_separation["spectral_hydration"].append(s["obs_id"])
        # Atmospheric suspect: ice_mean close to ice_max (uniform = atmospheric artifact)
        if s["ice_mean_score"] > 0 and s["ice_max_score"] > 0:
            ratio = s["ice_mean_score"] / s["ice_max_score"] if s["ice_max_score"] > 0 else 0
            if ratio > 0.85 and s["ice_percent"] >= 5:
                evidence_separation["atmospheric_suspect"].append(s["obs_id"])

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
            "threshold_justification": threshold_justification,
            "false_positive_rate": round(false_positive_rate, 4),
            "spatial_coherence": spatial_coherence,
            "evidence_separation": evidence_separation,
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

    # H2O / CO2 Ice tracking (class IDs: H2O=2, CO2=1)
    H2O_ICE_ID, CO2_ICE_ID = 2, 1
    h2o_total_pixels = 0
    co2_total_pixels = 0

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
            for ice_id, ice_name in [(CO2_ICE_ID, "CO2 Ice"), (H2O_ICE_ID, "H2O Ice")]:
                ice_str = str(ice_id)
                if ice_str in result.class_stats and result.class_stats[ice_str] > 0:
                    if ice_name not in ice_types_found:
                        ice_types_found.append(ice_name)

            # Per-observation H2O / CO2 pixel counts
            obs_h2o = int(result.class_stats.get(str(H2O_ICE_ID), 0))
            obs_co2 = int(result.class_stats.get(str(CO2_ICE_ID), 0))
            h2o_pct = round(obs_h2o / total_classified * 100, 2) if total_classified > 0 else 0.0
            co2_pct = round(obs_co2 / total_classified * 100, 2) if total_classified > 0 else 0.0

            h2o_total_pixels += obs_h2o
            co2_total_pixels += obs_co2

            classified_obs.append({
                "obs_id": obs_id,
                "lat": p.get("lat"),
                "lon": p.get("lon"),
                "total_classified_pixels": total_classified,
                "top_mineral": CLASS_NAME.get(top_class_id, "Unknown"),
                "top_mineral_pixels": top_class_count,
                "h2o_pixels": obs_h2o,
                "h2o_percent": h2o_pct,
                "co2_pixels": obs_co2,
                "co2_percent": co2_pct,
                "h2o_rich": h2o_pct >= 1.0,
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

    # H2O spatial aggregation
    h2o_obs = [o for o in classified_obs if o["h2o_pixels"] > 0]
    h2o_rich_obs = [o for o in classified_obs if o.get("h2o_rich")]
    h2o_hotspot = None
    if h2o_rich_obs:
        lats = [o["lat"] for o in h2o_rich_obs if o.get("lat") is not None]
        lons = [o["lon"] for o in h2o_rich_obs if o.get("lon") is not None]
        if lats and lons:
            h2o_hotspot = {
                "center_lat": round(sum(lats) / len(lats), 4),
                "center_lon": round(sum(lons) / len(lons), 4),
                "n_observations": len(h2o_rich_obs),
                "max_h2o_percent": max(o["h2o_percent"] for o in h2o_rich_obs),
            }

    summary_parts = [f"CNN classified {len(classified_obs)}/{len(crism)} CRISM observations"]
    if top_minerals:
        summary_parts.append(f"Top: {top_minerals[0]['name']} ({top_minerals[0]['total_pixels']} px)")
    if h2o_total_pixels > 0:
        summary_parts.append(f"H2O Ice: {h2o_total_pixels} px across {len(h2o_obs)} obs ({len(h2o_rich_obs)} H2O-rich)")
    if co2_total_pixels > 0:
        summary_parts.append(f"CO2 Ice: {co2_total_pixels} px (seasonal frost)")
    if ice_detected and not h2o_total_pixels and not co2_total_pixels:
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
            # H2O-specific fields
            "h2o_total_pixels": h2o_total_pixels,
            "co2_total_pixels": co2_total_pixels,
            "h2o_observations": len(h2o_obs),
            "h2o_rich_observations": len(h2o_rich_obs),
            "h2o_hotspot": h2o_hotspot,
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
# Task: Terraced Crater Dielectric Estimation (SHARAD + HiRISE DTM)
# =============================================================================

def terrace_dielectric_analysis(
    subsurface_result: Optional[TaskResult],
    products: List[Dict[str, Any]],
    bbox: "RegionBBox",
) -> TaskResult:
    """
    Estimate dielectric constant using terraced craters as "true depth" constraints.

    This is a more rigorous version of dielectric_analysis that:
    1. Finds HiRISE DTMs with terraced morphology near SHARAD tracks
    2. Extracts radial elevation profiles and identifies terrace benches
    3. Uses terrace-to-floor depth as a geometric depth proxy
    4. Picks SHARAD subsurface reflector two-way travel time near the crater
    5. Back-computes εr = (c * t / (2 * depth))²

    Quality flags indicate whether the terrace boundary plausibly corresponds
    to the SHARAD reflector (εr in 2-8 range = plausible).
    """
    import os

    if not subsurface_result or not subsurface_result.success:
        return TaskResult(
            task_type="terrace_dielectric",
            success=True,
            data={"estimates_count": 0},
            summary="No subsurface data for terrace dielectric analysis",
        )

    tracks = subsurface_result.data.get("tracks", [])
    tracks_with_reflectors = [
        t for t in tracks
        if t.get("subsurface_detected") and t.get("estimated_depth_m")
    ]

    if not tracks_with_reflectors:
        return TaskResult(
            task_type="terrace_dielectric",
            success=True,
            data={"estimates_count": 0, "reason": "no_subsurface_reflectors"},
            summary="No SHARAD subsurface reflectors for terrace analysis",
        )

    # Find SHARAD hi-res products
    sharad_products = [p for p in products if p.get("instrument") == "SHARAD_HIGHRES"]
    if not sharad_products:
        return TaskResult(
            task_type="terrace_dielectric",
            success=True,
            data={"estimates_count": 0, "reason": "no_sharad_highres"},
            summary="No SHARAD high-res products for terrace analysis",
        )

    try:
        from backend.analysis.epsilon_terrace.run import run_pipeline
    except ImportError:
        try:
            from analysis.epsilon_terrace.run import run_pipeline
        except ImportError:
            return TaskResult(
                task_type="terrace_dielectric",
                success=False,
                error="epsilon_terrace module not available",
                summary="Terrace dielectric module unavailable",
            )

    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dtm_dir = os.path.join(backend_dir, "hirise_dtm_data")
    output_dir = os.path.join(dtm_dir, "epsilon_results")

    if not os.path.isdir(dtm_dir):
        return TaskResult(
            task_type="terrace_dielectric",
            success=True,
            data={"estimates_count": 0, "reason": "no_dtm_dir"},
            summary="No HiRISE DTM directory for terrace analysis",
        )

    all_estimates = []
    analyzed_tracks = []

    for sp in sharad_products[:3]:  # Limit to 3 tracks for speed
        pid = sp["product_id"]
        try:
            result = run_pipeline(
                sharad_product_id=pid,
                dtm_dir=dtm_dir,
                buffer_km=30,
                output_dir=output_dir,
            )
            estimates = result.get("epsilon_estimates", [])
            # Only keep good/marginal quality estimates
            good_estimates = [e for e in estimates if e.get("quality") != "unreliable"]
            all_estimates.extend(good_estimates)
            analyzed_tracks.append({
                "product_id": pid,
                "dtms_found": result.get("dtms_found", 0),
                "estimates": len(estimates),
                "good_estimates": len(good_estimates),
            })
        except Exception as e:
            logger.warning(f"Terrace dielectric failed for {pid}: {e}")
            analyzed_tracks.append({
                "product_id": pid,
                "error": str(e),
            })

    if not all_estimates:
        return TaskResult(
            task_type="terrace_dielectric",
            success=True,
            data={
                "estimates_count": 0,
                "tracks_analyzed": analyzed_tracks,
                "reason": "no_reliable_estimates",
            },
            summary=f"No reliable terrace εr estimates from {len(analyzed_tracks)} tracks",
        )

    eps_values = [e["epsilon_r"] for e in all_estimates]
    import numpy as np
    mean_eps = round(float(np.mean(eps_values)), 2)
    median_eps = round(float(np.median(eps_values)), 2)

    if median_eps < 2.5:
        interp = "dry regolith or porous ice"
    elif median_eps < 3.5:
        interp = "ice-rich subsurface (consistent with water ice)"
    elif median_eps < 5.0:
        interp = "ice-cemented regolith"
    else:
        interp = "basaltic regolith or dense rock"

    return TaskResult(
        task_type="terrace_dielectric",
        success=True,
        data={
            "estimates_count": len(all_estimates),
            "mean_epsilon_r": mean_eps,
            "median_epsilon_r": median_eps,
            "min_epsilon_r": round(min(eps_values), 2),
            "max_epsilon_r": round(max(eps_values), 2),
            "interpretation": interp,
            "estimates": all_estimates,
            "tracks_analyzed": analyzed_tracks,
        },
        summary=f"Terrace εr = {median_eps} (median, {len(all_estimates)} estimates): {interp}",
    )


# =============================================================================
# Task: SHARAD Physics-Based Inversion
# =============================================================================

def sharad_physics_inversion(
    products: List[Dict[str, Any]],
    bbox: "RegionBBox",
) -> TaskResult:
    """
    Physics-based SHARAD dielectric inversion using terraced crater depth constraints.

    Mandatory pipeline: DTM geometry → SHARAD travel time → εr = (c·Δt / 2d)²
    Depth MUST come from DTM measurement, εr is NEVER assumed as 3.15.
    Includes clutter filtering and hyperbola curvature validation.
    """
    import os

    sharad_pids = [p["product_id"] for p in products if p.get("instrument") == "SHARAD_HIGHRES"]
    if not sharad_pids:
        return TaskResult(
            task_type="sharad_physics_inversion",
            success=True,
            data={
                "inversions_completed": 0,
                "reason": "no_sharad_highres_products",
                "physics_note": "Inversion cannot proceed without SHARAD_HIGHRES radargram data.",
            },
            summary="No SHARAD_HIGHRES products for physics-based inversion",
        )

    try:
        from analysis.sharad_inversion import run_inversion_pipeline
    except ImportError:
        try:
            from backend.analysis.sharad_inversion import run_inversion_pipeline
        except ImportError:
            return TaskResult(
                task_type="sharad_physics_inversion",
                success=False,
                error="sharad_inversion module not available",
                summary="SHARAD physics inversion module unavailable",
            )

    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dtm_dir = os.path.join(backend_dir, "hirise_dtm_data")

    if not os.path.isdir(dtm_dir):
        return TaskResult(
            task_type="sharad_physics_inversion",
            success=True,
            data={
                "inversions_completed": 0,
                "reason": "no_dtm_directory",
                "physics_note": "DTM data directory not found. Cannot perform inversion without independent depth constraints.",
            },
            summary="No DTM directory for physics-based inversion",
        )

    try:
        pipeline_result = run_inversion_pipeline(
            sharad_product_ids=sharad_pids[:5],
            dtm_dir=dtm_dir,
            buffer_km=30,
            min_snr=3.0,
            do_hyperbola_validation=True,
            do_clutter_filter=True,
        )

        # Convert Pydantic models to serializable dicts
        assumptions_list = []
        for a in pipeline_result.assumptions:
            assumptions_list.append({
                "param": a.param,
                "value": a.value,
                "source": a.source,
                "justification": a.justification,
                "uncertainty": a.uncertainty,
            })

        derivation_log = []
        for d in pipeline_result.derivation_log:
            derivation_log.append({
                "step": d.step,
                "equation": d.equation,
                "inputs": d.inputs,
                "output": d.output,
                "notes": d.notes,
            })

        inversion_details = []
        for inv in pipeline_result.results:
            detail = {
                "sharad_product_id": inv.sharad_product_id,
                "dtm_product_id": inv.dtm_product_id,
                "epsilon_r": inv.epsilon_r,
                "epsilon_r_ci": inv.epsilon_r_ci,
                "depth_m": inv.depth_m,
                "twt_us": inv.twt_us,
                "quality": inv.quality,
                "material_interpretation": inv.material_interpretation,
            }
            if inv.hyperbola_validation:
                detail["hyperbola_validation"] = {
                    "epsilon_r_hyperbola": inv.hyperbola_validation.epsilon_r_hyperbola,
                    "agreement": inv.hyperbola_validation.agreement,
                    "delta_epsilon": inv.hyperbola_validation.delta_epsilon,
                }
            if inv.clutter_assessment:
                detail["clutter_assessment"] = {
                    "is_clutter": inv.clutter_assessment.is_clutter,
                    "clutter_score": inv.clutter_assessment.clutter_score,
                    "snr_at_pick": inv.clutter_assessment.snr_at_pick,
                }
            inversion_details.append(detail)

        data = {
            "inversions_completed": pipeline_result.inversions_completed,
            "sharad_products_analyzed": pipeline_result.sharad_products_analyzed,
            "dtm_intersections_found": pipeline_result.dtm_intersections_found,
            "best_epsilon_r": pipeline_result.best_epsilon_r,
            "best_epsilon_r_ci": pipeline_result.best_epsilon_r_ci,
            "reflector_confidence": pipeline_result.reflector_confidence,
            "assumptions": assumptions_list,
            "derivation_log": derivation_log,
            "inversion_details": inversion_details,
            "methodology": (
                "Physics-based inversion: depth from DTM terrace geometry (independent measurement), "
                "εr = (c · Δt / (2 · depth))². Clutter filtered via cluttergram comparison. "
                "Cross-validated with hyperbola curvature fitting."
            ),
        }

        # Build summary
        if pipeline_result.inversions_completed > 0:
            summary = (
                f"Physics inversion: {pipeline_result.inversions_completed} estimates, "
                f"best εr={pipeline_result.best_epsilon_r:.2f} "
                f"(CI: {pipeline_result.best_epsilon_r_ci}), "
                f"confidence={pipeline_result.reflector_confidence}"
            )
        else:
            summary = (
                f"Analyzed {pipeline_result.sharad_products_analyzed} SHARAD products, "
                f"found {pipeline_result.dtm_intersections_found} DTM intersections, "
                f"but no successful inversions"
            )

        return TaskResult(
            task_type="sharad_physics_inversion",
            success=True,
            data=data,
            summary=summary,
        )

    except Exception as e:
        logger.error(f"SHARAD physics inversion failed: {e}")
        return TaskResult(
            task_type="sharad_physics_inversion",
            success=False,
            error=str(e),
            summary=f"SHARAD physics inversion failed: {e}",
        )


# =============================================================================
# Task: Find SHARAD Track Geometric Intersections with Other Instruments
# =============================================================================

def find_sharad_intersections(
    products: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Find which SHARAD tracks geometrically intersect CRISM/HiRISE/DTM footprints.

    Uses Liang-Barsky line-vs-bbox clipping for precise intersection testing.
    Returns intersection counts and pairs for cross-instrument correlation.
    """
    import json
    import os

    result: Dict[str, Any] = {
        "sharad_crism_intersections": 0,
        "sharad_hirise_intersections": 0,
        "sharad_dtm_intersections": 0,
        "intersection_pairs": [],
        "crism_ice_with_sharad": [],
    }

    try:
        from .proximity_router import _bbox_from_geometry, _line_intersects_bbox
    except ImportError:
        logger.warning("proximity_router not available for intersection analysis")
        return result

    # Load SHARAD_HIGHRES index for track geometries
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sharad_index_path = os.path.join(backend_dir, "sharad_highres_data", "index.geojson")
    if not os.path.exists(sharad_index_path):
        return result

    with open(sharad_index_path) as f:
        sharad_index = json.load(f)

    def _norm_lon(lon: float) -> float:
        """Normalize longitude to -180..180."""
        while lon > 180:
            lon -= 360
        while lon < -180:
            lon += 360
        return lon

    # Build SHARAD track lookup: product_id → coordinates (normalized to -180..180)
    sharad_tracks: Dict[str, list] = {}
    for feat in sharad_index.get("features", []):
        pid = feat.get("properties", {}).get("product_id", "")
        coords = feat.get("geometry", {}).get("coordinates", [])
        if pid and coords:
            sharad_tracks[pid] = [[_norm_lon(c[0]), c[1]] for c in coords]

    if not sharad_tracks:
        return result

    # Get product IDs in the session grouped by instrument
    non_sharad_products = [
        p for p in products
        if p.get("instrument") not in ("SHARAD", "SHARAD_HIGHRES")
    ]

    if not non_sharad_products:
        return result

    # Load footprint indices for non-SHARAD instruments
    try:
        from .registry import get_registry
        registry = get_registry()
    except Exception:
        return result

    # Build footprint lookup: product_id → {bbox, instrument}
    # Use lowercase keys for case-insensitive matching
    footprint_cache: Dict[str, Dict] = {}
    loaded_instruments: set = set()

    for p in non_sharad_products:
        inst = p.get("instrument", "").upper()
        if inst in loaded_instruments:
            continue
        loaded_instruments.add(inst)

        try:
            index = registry.load_index(inst)
        except Exception:
            continue

        for feat in index.get("features", []):
            props = feat.get("properties", {})
            fp_id = (props.get("product_id") or props.get("ProductId") or
                     props.get("id") or props.get("PRODUCT_ID") or "")
            geom = feat.get("geometry")
            if not fp_id or not geom:
                continue
            bbox = _bbox_from_geometry(geom)
            if bbox:
                # Buffer Point geometries to approximate footprint size
                # CRISM targeted obs ~10 km, HiRISE ~6 km
                if geom.get("type") == "Point":
                    import math as _math
                    lat_c = (bbox["lat_min"] + bbox["lat_max"]) / 2
                    buf_deg = 0.1  # ~6 km at equator
                    cos_lat = max(_math.cos(_math.radians(lat_c)), 0.3)
                    bbox["lat_min"] -= buf_deg
                    bbox["lat_max"] += buf_deg
                    bbox["lon_min"] -= buf_deg / cos_lat
                    bbox["lon_max"] += buf_deg / cos_lat
                # Normalize bbox longitudes to -180..180
                bbox["lon_min"] = _norm_lon(bbox["lon_min"])
                bbox["lon_max"] = _norm_lon(bbox["lon_max"])
                footprint_cache[fp_id.lower()] = {
                    "bbox": bbox,
                    "instrument": inst,
                    "geom": geom,
                }

    def _find_footprint(product_id: str) -> Optional[Dict]:
        """Find footprint by exact or prefix match (case-insensitive)."""
        pid_lower = product_id.lower()
        # Exact match
        if pid_lower in footprint_cache:
            return footprint_cache[pid_lower]
        # Prefix match: registry IDs may have suffixes (e.g., frt00009326_07_if164j_mtr3)
        for cache_id, fp_data in footprint_cache.items():
            if cache_id.startswith(pid_lower):
                return fp_data
        return None

    # Test intersections: each session product vs all SHARAD tracks
    for p in non_sharad_products:
        pid = p["product_id"]
        fp = _find_footprint(pid)
        if not fp:
            continue

        bbox = fp["bbox"]
        inst = fp["instrument"]

        matching_sharad = []
        for sharad_pid, track_coords in sharad_tracks.items():
            if _line_intersects_bbox(track_coords, bbox):
                matching_sharad.append(sharad_pid)
                result["intersection_pairs"].append({
                    "sharad_id": sharad_pid,
                    "target_id": pid,
                    "target_instrument": inst,
                })

        if matching_sharad:
            if inst == "CRISM" or inst == "CRISM_TRR3":
                result["sharad_crism_intersections"] += len(matching_sharad)
            elif inst == "HIRISE":
                result["sharad_hirise_intersections"] += len(matching_sharad)
            elif inst == "HIRISE_DTM":
                result["sharad_dtm_intersections"] += len(matching_sharad)

    # Limit pairs list to avoid huge output
    result["intersection_pairs"] = result["intersection_pairs"][:50]

    return result


# =============================================================================
# Task: Targeted Subsurface Analysis at CRISM Ice Locations
# =============================================================================

def targeted_subsurface_at_ice(
    products: List[Dict[str, Any]],
    all_results: Dict[str, Any],
) -> TaskResult:
    """
    Check SHARAD subsurface at locations where CRISM/CNN detected surface ice.

    For each CRISM/CNN ice location:
    1. Find the nearest SHARAD_HIGHRES track (within 30 km)
    2. Call pick_subsurface_interface at the ice lat/lon
    3. Report whether a subsurface reflector exists at that location
    """
    import json
    import os
    import math

    ice_locations = []

    # Gather ice locations from mineral analysis
    mineral_result = all_results.get("mineral")
    if mineral_result and hasattr(mineral_result, 'data'):
        mineral_data = mineral_result.data if hasattr(mineral_result, 'data') else mineral_result
        # Ice hotspot
        hotspot = mineral_data.get("ice_hotspot") if isinstance(mineral_data, dict) else None
        if hotspot and hotspot.get("center_lat") is not None:
            ice_locations.append({
                "source": "CRISM",
                "lat": hotspot["center_lat"],
                "lon": hotspot["center_lon"],
                "product_id": "ice_hotspot",
                "ice_percent": hotspot.get("max_ice_pct"),
            })
        # Top ice candidates
        top_ice = mineral_data.get("top_ice_candidates", []) if isinstance(mineral_data, dict) else []
        for c in top_ice[:5]:
            if c.get("lat") is not None and c.get("lon") is not None:
                ice_locations.append({
                    "source": "CRISM",
                    "lat": c["lat"],
                    "lon": c["lon"],
                    "product_id": c.get("obs_id", c.get("product_id", "")),
                    "ice_percent": c.get("ice_percent"),
                })

    # Gather ice locations from CNN H2O
    cnn_result = all_results.get("mineral_cnn")
    if cnn_result and hasattr(cnn_result, 'data'):
        cnn_data = cnn_result.data if hasattr(cnn_result, 'data') else cnn_result
        h2o_hotspot = cnn_data.get("h2o_hotspot") if isinstance(cnn_data, dict) else None
        if h2o_hotspot and h2o_hotspot.get("center_lat") is not None:
            ice_locations.append({
                "source": "CNN_H2O",
                "lat": h2o_hotspot["center_lat"],
                "lon": h2o_hotspot["center_lon"],
                "product_id": "h2o_hotspot",
                "ice_percent": h2o_hotspot.get("max_h2o_percent"),
            })

    # Deduplicate ice locations (same lat/lon within 0.01°)
    unique_locs: list = []
    for loc in ice_locations:
        dup = False
        for u in unique_locs:
            if abs(loc["lat"] - u["lat"]) < 0.01 and abs(loc["lon"] - u["lon"]) < 0.01:
                dup = True
                break
        if not dup:
            unique_locs.append(loc)
    ice_locations = unique_locs

    if not ice_locations:
        return TaskResult(
            task_type="targeted_subsurface_at_ice",
            success=True,
            data={
                "ice_locations_checked": 0,
                "ice_locations_with_sharad": 0,
                "reflectors_at_ice": 0,
                "targeted_picks": [],
                "note": "No CRISM/CNN ice locations available to target.",
            },
            summary="No CRISM/CNN ice locations to check against SHARAD.",
        )

    # Load SHARAD index
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sharad_index_path = os.path.join(backend_dir, "sharad_highres_data", "index.geojson")
    if not os.path.exists(sharad_index_path):
        return TaskResult(
            task_type="targeted_subsurface_at_ice",
            success=True,
            data={"ice_locations_checked": len(ice_locations), "targeted_picks": [],
                  "note": "SHARAD index not found."},
            summary="SHARAD index not found for targeted subsurface analysis.",
        )

    with open(sharad_index_path) as f:
        sharad_index = json.load(f)

    # Build track center lookup
    tracks = []
    for feat in sharad_index.get("features", []):
        pid = feat.get("properties", {}).get("product_id", "")
        coords = feat.get("geometry", {}).get("coordinates", [])
        if not pid or not coords:
            continue
        # Compute track center for quick distance filtering
        mid = coords[len(coords) // 2]
        tracks.append({"product_id": pid, "coords": coords, "mid_lat": mid[1], "mid_lon": mid[0]})

    # Import pick function
    try:
        from backend.analysis.epsilon_terrace.sharad_pick import pick_subsurface_interface
    except ImportError:
        try:
            from analysis.epsilon_terrace.sharad_pick import pick_subsurface_interface
        except ImportError:
            return TaskResult(
                task_type="targeted_subsurface_at_ice",
                success=False,
                error="sharad_pick module not available",
                summary="SHARAD pick module unavailable for targeted analysis.",
            )

    # For each ice location, find nearest SHARAD track and pick
    targeted_picks = []
    locations_with_sharad = 0
    reflectors_at_ice = 0

    for ice_loc in ice_locations:
        ice_lat = ice_loc["lat"]
        ice_lon = ice_loc["lon"]

        # Find nearest SHARAD track point within 30 km
        best_track = None
        best_dist = 30.0  # km threshold

        for track in tracks:
            # Quick pre-filter: skip tracks whose latitude range doesn't include target
            track_lats = [c[1] for c in track["coords"]]
            if ice_lat < min(track_lats) - 1 or ice_lat > max(track_lats) + 1:
                continue
            for coord in track["coords"][::3]:  # Sample every 3rd point
                d = haversine_distance_km(ice_lat, ice_lon, coord[1], coord[0])
                if d < best_dist:
                    best_dist = d
                    best_track = track

        if best_track is None:
            targeted_picks.append({
                "ice_source": ice_loc["source"],
                "ice_lat": ice_lat,
                "ice_lon": ice_lon,
                "ice_product_id": ice_loc["product_id"],
                "sharad_product_id": None,
                "distance_km": None,
                "reflector_detected": False,
                "note": "No SHARAD track within 30 km",
            })
            continue

        locations_with_sharad += 1

        # Run subsurface pick at ice location
        try:
            pick_result = pick_subsurface_interface(
                best_track["product_id"],
                ice_lat, ice_lon,
                window_km=15.0,
                min_snr=2.0,
            )

            reflector_detected = bool(pick_result.picks) and pick_result.median_twt_us > 0
            if reflector_detected:
                reflectors_at_ice += 1

            # Compute assumed depth (clearly labeled)
            depth_assumed = None
            if pick_result.median_twt_us > 0:
                _ICE_EPSILON = 3.15
                _C = 299_792_458.0
                v_ice = _C / math.sqrt(_ICE_EPSILON)
                depth_assumed = round(v_ice * pick_result.median_twt_us * 1e-6 / 2, 1)

            pick_entry = {
                "ice_source": ice_loc["source"],
                "ice_lat": ice_lat,
                "ice_lon": ice_lon,
                "ice_product_id": ice_loc["product_id"],
                "ice_percent": ice_loc.get("ice_percent"),
                "sharad_product_id": best_track["product_id"],
                "distance_km": round(best_dist, 1),
                "reflector_detected": reflector_detected,
                "n_picks": len(pick_result.picks),
                "median_snr": round(max((p.snr for p in pick_result.picks), default=0), 1),
                "twt_us": round(pick_result.median_twt_us, 4) if pick_result.median_twt_us else None,
                "depth_m_assumed": depth_assumed,
                "epsilon_r_note": "Depth uses assumed εr=3.15 (not measured)",
            }
            if pick_result.error:
                pick_entry["error"] = pick_result.error
            targeted_picks.append(pick_entry)

        except Exception as e:
            logger.warning(f"Targeted pick failed at ({ice_lat}, {ice_lon}): {e}")
            targeted_picks.append({
                "ice_source": ice_loc["source"],
                "ice_lat": ice_lat,
                "ice_lon": ice_lon,
                "ice_product_id": ice_loc["product_id"],
                "sharad_product_id": best_track["product_id"],
                "distance_km": round(best_dist, 1),
                "reflector_detected": False,
                "error": str(e),
            })

    # Build summary
    summary_parts = [
        f"Checked {len(ice_locations)} ice locations: {locations_with_sharad} had SHARAD coverage, "
        f"{reflectors_at_ice} showed subsurface reflectors."
    ]
    for p in targeted_picks:
        if p.get("reflector_detected"):
            summary_parts.append(
                f"  Reflector at ({p['ice_lat']:.2f}, {p['ice_lon']:.2f}): "
                f"SHARAD {p['sharad_product_id']}, depth~{p.get('depth_m_assumed', '?')}m, "
                f"SNR={p.get('median_snr', '?')}"
            )

    return TaskResult(
        task_type="targeted_subsurface_at_ice",
        success=True,
        data={
            "ice_locations_checked": len(ice_locations),
            "ice_locations_with_sharad": locations_with_sharad,
            "reflectors_at_ice": reflectors_at_ice,
            "targeted_picks": targeted_picks,
        },
        summary="\n".join(summary_parts),
    )


# =============================================================================
# Task: CRISM Spectral Analysis (SAM + Band Parameters)
# =============================================================================

def crism_spectral_analysis(products: List[Dict[str, Any]]) -> TaskResult:
    """
    CRISM spectral analysis with continuum removal, band parameters, and SAM classification.

    Replaces threshold-based ice scoring with physics-based mineral classification:
    1. Continuum removal (convex hull)
    2. Band parameter extraction (BD1500, BD1900, BD2100, BD2200)
    3. Spectral Angle Mapper against USGS endmembers
    4. Per-pixel classification with mineral class and probability
    """
    import re

    crism = [p for p in products if p.get("instrument") == "CRISM"]
    if not crism:
        return TaskResult(
            task_type="crism_spectral",
            success=True,
            data={"observations_analyzed": 0, "reason": "no_crism_products"},
            summary="No CRISM products for spectral analysis",
        )

    try:
        from analysis.crism_spectral import run_spectral_analysis
    except ImportError:
        try:
            from backend.analysis.crism_spectral import run_spectral_analysis
        except ImportError:
            return TaskResult(
                task_type="crism_spectral",
                success=False,
                error="crism_spectral module not available",
                summary="CRISM spectral analysis module unavailable",
            )

    analyzed = []
    aggregate_class_counts: Dict[str, int] = {}
    total_water_ice_pixels = 0
    total_pixels_analyzed = 0
    all_bd1500 = []
    all_bd1900 = []

    for p in crism:
        pid = p["product_id"]
        match = re.match(r'^([a-z]{3}[0-9a-f]{8})', pid.lower())
        obs_id = match.group(1) if match else pid.lower()

        try:
            result = run_spectral_analysis(obs_id, max_sam_angle=0.3, subsample=1)

            obs_data = {
                "obs_id": result.obs_id,
                "lat": p.get("lat"),
                "lon": p.get("lon"),
                "class_counts": result.class_counts,
                "class_fractions": result.class_fractions,
                "water_ice_fraction": result.water_ice_fraction,
                "water_ice_pixel_count": result.water_ice_pixel_count,
                "mean_band_params": {
                    "BD1500": result.mean_band_params.BD1500,
                    "BD1900": result.mean_band_params.BD1900,
                    "BD2100": result.mean_band_params.BD2100,
                    "BD2200": result.mean_band_params.BD2200,
                },
                "atmospheric_uncertainty_notes": result.atmospheric_uncertainty_notes,
            }
            analyzed.append(obs_data)

            # Aggregate
            for cls, count in result.class_counts.items():
                aggregate_class_counts[cls] = aggregate_class_counts.get(cls, 0) + count
            total_water_ice_pixels += result.water_ice_pixel_count
            total_pixels_analyzed += sum(result.class_counts.values())
            if result.mean_band_params.BD1500 is not None:
                all_bd1500.append(result.mean_band_params.BD1500)
            if result.mean_band_params.BD1900 is not None:
                all_bd1900.append(result.mean_band_params.BD1900)

        except Exception as e:
            logger.warning(f"CRISM spectral analysis failed for {obs_id}: {e}")
            continue

    if not analyzed:
        return TaskResult(
            task_type="crism_spectral",
            success=True,
            data={
                "observations_analyzed": 0,
                "reason": "no_trr3_data_available",
                "physics_note": "No TRR3 cube data available locally for spectral analysis.",
            },
            summary="No TRR3 data available for spectral analysis",
        )

    # Compute aggregate fractions
    aggregate_fractions = {}
    if total_pixels_analyzed > 0:
        for cls, count in aggregate_class_counts.items():
            aggregate_fractions[cls] = round(count / total_pixels_analyzed, 4)

    water_ice_overall_fraction = (
        round(total_water_ice_pixels / total_pixels_analyzed, 4)
        if total_pixels_analyzed > 0 else 0.0
    )

    # Mean band parameters across all observations
    import numpy as np
    mean_bd1500 = round(float(np.mean(all_bd1500)), 4) if all_bd1500 else None
    mean_bd1900 = round(float(np.mean(all_bd1900)), 4) if all_bd1900 else None

    data = {
        "observations_analyzed": len(analyzed),
        "total_pixels_analyzed": total_pixels_analyzed,
        "observations": analyzed,
        "aggregate_class_counts": aggregate_class_counts,
        "aggregate_class_fractions": aggregate_fractions,
        "water_ice_total_pixels": total_water_ice_pixels,
        "water_ice_overall_fraction": water_ice_overall_fraction,
        "mean_band_params": {
            "BD1500": mean_bd1500,
            "BD1900": mean_bd1900,
        },
        "methodology": (
            "Continuum removal (convex hull) → band parameter extraction (BD1500, BD1900, BD2100, BD2200) "
            "→ Spectral Angle Mapper classification against synthetic USGS endmembers "
            "(water ice, gypsum, polyhydrated sulfate, basalt). "
            "Band strengths cross-validate SAM mineral ID."
        ),
    }

    summary_parts = [f"Spectral analysis: {len(analyzed)}/{len(crism)} CRISM observations"]
    if water_ice_overall_fraction > 0:
        summary_parts.append(f"water ice {water_ice_overall_fraction:.1%} of pixels")
    if mean_bd1500 is not None:
        summary_parts.append(f"mean BD1500={mean_bd1500:.3f}")

    return TaskResult(
        task_type="crism_spectral",
        success=True,
        data=data,
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
            "hazard_density": slope.get("hazard_density"),
            "roughness_stats": slope.get("roughness_stats"),
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
            "snr_distribution": sub.get("snr_distribution"),
            "clutter_rejection": sub.get("clutter_rejection"),
            "reflector_density_per_km": sub.get("reflector_density_per_km"),
            "epsilon_r_source": depth.get("epsilon_r_source", "assumed") if depth else "no_data",
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

    # CNN mineral classification (deep learning) — with H2O detail
    if "mineral_cnn" in all_results and all_results["mineral_cnn"].success:
        cnn = all_results["mineral_cnn"].data
        synthesis["cnn_mineral_classification"] = {
            "observations_classified": cnn.get("classified_count", 0),
            "top_minerals": cnn.get("top_minerals", []),
            "ice_detected": cnn.get("ice_detected", False),
            "ice_types": cnn.get("ice_types", []),
            "h2o_total_pixels": cnn.get("h2o_total_pixels", 0),
            "co2_total_pixels": cnn.get("co2_total_pixels", 0),
            "h2o_observations": cnn.get("h2o_observations", 0),
            "h2o_rich_observations": cnn.get("h2o_rich_observations", 0),
            "h2o_hotspot": cnn.get("h2o_hotspot"),
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

    # Terrace-based dielectric estimation (more rigorous)
    if "terrace_dielectric" in all_results and all_results["terrace_dielectric"].success:
        tdiel = all_results["terrace_dielectric"].data
        synthesis["terrace_dielectric"] = {
            "estimates_count": tdiel.get("estimates_count", 0),
            "mean_epsilon_r": tdiel.get("mean_epsilon_r"),
            "median_epsilon_r": tdiel.get("median_epsilon_r"),
            "interpretation": tdiel.get("interpretation", ""),
            "estimates": tdiel.get("estimates", []),
        }
        # Override basic dielectric if terrace version has estimates
        if tdiel.get("estimates_count", 0) > 0:
            synthesis["dielectric_analysis"] = {
                "estimates_count": tdiel.get("estimates_count", 0),
                "mean_epsilon_r": tdiel.get("mean_epsilon_r"),
                "median_epsilon_r": tdiel.get("median_epsilon_r"),
                "interpretation": tdiel.get("interpretation", ""),
                "method": "terraced_crater",
            }

    # SHARAD Physics Inversion (new physics-based pipeline)
    if "sharad_physics_inversion" in all_results and all_results["sharad_physics_inversion"].success:
        inv = all_results["sharad_physics_inversion"].data
        synthesis["sharad_physics_inversion"] = {
            "inversions_completed": inv.get("inversions_completed", 0),
            "best_epsilon_r": inv.get("best_epsilon_r"),
            "best_epsilon_r_ci": inv.get("best_epsilon_r_ci"),
            "reflector_confidence": inv.get("reflector_confidence"),
            "assumptions": inv.get("assumptions", []),
            "derivation_log": inv.get("derivation_log", []),
            "methodology": inv.get("methodology", ""),
        }
        # Override dielectric_analysis if physics inversion succeeded
        if inv.get("inversions_completed", 0) > 0:
            synthesis["dielectric_analysis"] = {
                "estimates_count": inv.get("inversions_completed", 0),
                "mean_epsilon_r": inv.get("best_epsilon_r"),
                "median_epsilon_r": inv.get("best_epsilon_r"),
                "interpretation": f"Physics-based inversion εr={inv.get('best_epsilon_r')}",
                "method": "physics_inversion",
                "assumptions": inv.get("assumptions", []),
            }

    # ── Dielectric Method Hierarchy: track what was attempted and outcomes ──
    method_hierarchy = []
    physics_inv_data = synthesis.get("sharad_physics_inversion", {})
    terrace_diel_data = synthesis.get("terrace_dielectric", {})
    basic_diel_data = synthesis.get("dielectric_analysis", {})

    if physics_inv_data.get("inversions_completed", 0) > 0:
        method_hierarchy.append({
            "method": "physics_inversion",
            "status": "success",
            "epsilon_r": physics_inv_data.get("best_epsilon_r"),
        })
    elif "sharad_physics_inversion" in all_results:
        method_hierarchy.append({
            "method": "physics_inversion",
            "status": "attempted_failed",
            "reason": all_results["sharad_physics_inversion"].data.get("reason", "unknown"),
        })

    if terrace_diel_data.get("estimates_count", 0) > 0:
        method_hierarchy.append({
            "method": "terraced_crater",
            "status": "success",
            "epsilon_r": terrace_diel_data.get("median_epsilon_r"),
        })
    elif "terrace_dielectric" in all_results:
        method_hierarchy.append({
            "method": "terraced_crater",
            "status": "attempted_failed",
            "reason": all_results["terrace_dielectric"].data.get("reason", "unknown"),
        })

    if basic_diel_data.get("estimates_count", 0) > 0 and basic_diel_data.get("method") not in ("physics_inversion", "terraced_crater"):
        method_hierarchy.append({
            "method": "standard_dtm_relief",
            "status": "success",
            "epsilon_r": basic_diel_data.get("median_epsilon_r"),
        })

    # Determine effective εr source for the entire synthesis
    successful_physics = [m for m in method_hierarchy if m["status"] == "success" and m["method"] in ("physics_inversion", "terraced_crater")]
    if successful_physics:
        synthesis["epsilon_r_source"] = successful_physics[0]["method"]
        synthesis["is_fallback"] = False
        # Update subsurface_coverage source to reflect physics override
        if synthesis.get("subsurface_coverage"):
            synthesis["subsurface_coverage"]["epsilon_r_source"] = successful_physics[0]["method"]
    else:
        synthesis["epsilon_r_source"] = "assumed"
        synthesis["is_fallback"] = True
        # Check if physics was attempted but failed
        attempted_methods = [m for m in method_hierarchy if m["status"] == "attempted_failed"]
        synthesis["physics_inversion_attempted"] = len(attempted_methods) > 0

    synthesis["dielectric_method_hierarchy"] = method_hierarchy

    # Propagate physics_attempted flag into dielectric_analysis for scoring
    if synthesis.get("dielectric_analysis") is not None:
        synthesis["dielectric_analysis"]["physics_attempted"] = synthesis.get("physics_inversion_attempted", False)

    # CRISM Spectral Analysis (new physics-based pipeline)
    if "crism_spectral" in all_results and all_results["crism_spectral"].success:
        spec = all_results["crism_spectral"].data
        synthesis["crism_spectral_analysis"] = {
            "observations_analyzed": spec.get("observations_analyzed", 0),
            "aggregate_class_fractions": spec.get("aggregate_class_fractions", {}),
            "water_ice_total_pixels": spec.get("water_ice_total_pixels", 0),
            "water_ice_overall_fraction": spec.get("water_ice_overall_fraction", 0.0),
            "mean_band_params": spec.get("mean_band_params", {}),
            "methodology": spec.get("methodology", ""),
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

    # SHARAD geometric intersections with other instruments
    all_products_flat = []
    for key, res in all_results.items():
        if key.startswith("search_") and res.success:
            all_products_flat.extend(res.data.get("products", []))
    if all_products_flat:
        intersection_stats = find_sharad_intersections(all_products_flat)
        synthesis["sharad_intersections"] = intersection_stats
        # Propagate counts to cross_instrument
        cross["sharad_crism_geometric_intersections"] = intersection_stats.get("sharad_crism_intersections", 0)
        cross["sharad_hirise_geometric_intersections"] = intersection_stats.get("sharad_hirise_intersections", 0)
        cross["sharad_dtm_geometric_intersections"] = intersection_stats.get("sharad_dtm_intersections", 0)

    # Targeted subsurface analysis at CRISM/CNN ice locations
    targeted_result = all_results.get("targeted_subsurface_at_ice")
    if targeted_result and hasattr(targeted_result, "data") and targeted_result.data:
        synthesis["targeted_ice_subsurface"] = targeted_result.data

    # ── Science Distance Computation ──
    # Distance between best landing site and nearest science target
    science_distance = None
    best_site = synthesis.get("engineering_feasibility", {}).get("best_site")
    if best_site and best_site.get("lat") is not None:
        # Distance to nearest SHARAD reflector
        subsurface_tracks = all_results.get("subsurface", TaskResult(task_type="subsurface_scan")).data.get("tracks", [])
        for t in subsurface_tracks:
            if t.get("subsurface_detected") and t.get("lat") is not None:
                d = haversine_distance_km(best_site["lat"], best_site["lon"], t["lat"], t["lon"])
                if science_distance is None or d < science_distance:
                    science_distance = d
        # Distance to CRISM ice hotspot
        hotspot = synthesis.get("mineral_signatures", {}).get("ice_hotspot")
        if hotspot:
            d = haversine_distance_km(best_site["lat"], best_site["lon"], hotspot["center_lat"], hotspot["center_lon"])
            if science_distance is None or d < science_distance:
                science_distance = d

    # ── Composite Scoring v2 (weighted, transparent, multi-dimensional) ──
    scoring_result = compute_composite_score(
        subsurface_data=synthesis["subsurface_coverage"],
        dielectric_data=synthesis.get("dielectric_analysis", {}),
        cross_instrument_data=cross,
        mineral_data=synthesis.get("mineral_signatures", {}),
        cnn_data=synthesis.get("cnn_mineral_classification", {}),
        engineering_data=synthesis.get("engineering_feasibility", {}),
        climate_data=synthesis.get("climate", {}),
        science_distance_km=science_distance,
    )

    evidence_strength = classify_evidence_strength(
        scoring_result["sub_scores"],
        cross,
    )

    recommendation = classify_recommendation(
        scoring_result["final_score"],
        evidence_strength,
        cross.get("evidence_consistency", "insufficient_data"),
    )

    synthesis["scoring_model"] = scoring_result
    synthesis["evidence_strength"] = evidence_strength
    synthesis["recommendation_v2"] = recommendation

    # Backward-compat: overall_score as integer 0-95
    score = min(scoring_result["final_score_100"], 95)
    synthesis["overall_score"] = score

    # Generate strengths and uncertainties from sub-score breakdown
    strengths = []
    uncertainties = []
    sub = scoring_result["sub_scores"]

    def _extract_notes(sub_key):
        """Extract notes from a sub-score breakdown."""
        entry = sub.get(sub_key, {})
        bd = entry.get("breakdown", {})
        return bd.get("notes", [])

    # Subsurface
    sub_score_val = sub.get("subsurface_potential", {}).get("score", 0)
    sub_notes = _extract_notes("subsurface_potential")
    if sub_score_val >= 0.5:
        strengths.append(f"SHARAD subsurface: {' '.join(sub_notes)}" if sub_notes else "SHARAD subsurface reflectors detected")
    elif sub_score_val > 0:
        uncertainties.append(f"Limited subsurface evidence: {' '.join(sub_notes)}" if sub_notes else "Limited subsurface evidence")
    else:
        uncertainties.append("No SHARAD radargram data available for direct subsurface analysis")

    # Surface ice (CRISM + CNN combined)
    ice_score_val = sub.get("surface_ice", {}).get("score", 0)
    ice_notes = _extract_notes("surface_ice")
    if ice_score_val >= 0.5:
        strengths.append(f"Surface ice evidence: {' '.join(ice_notes)}" if ice_notes else "Strong surface ice evidence")
    elif ice_score_val > 0:
        uncertainties.append(f"Weak surface ice signal: {' '.join(ice_notes)}" if ice_notes else "Weak surface ice signal")
    else:
        uncertainties.append("No significant surface ice signatures detected")

    # Terrain
    terrain_score_val = sub.get("terrain_safety", {}).get("score", 0)
    terrain_notes = _extract_notes("terrain_safety")
    if terrain_score_val >= 0.5:
        strengths.append(f"Terrain: {' '.join(terrain_notes)}" if terrain_notes else "Favorable terrain for landing")
    elif terrain_score_val > 0:
        uncertainties.append(f"Terrain challenges: {' '.join(terrain_notes)}" if terrain_notes else "Terrain is marginal")
    else:
        uncertainties.append("Slope data unavailable for engineering assessment")

    # Climate
    climate_score_val = sub.get("climate", {}).get("score", 0)
    climate_notes = _extract_notes("climate")
    if climate_score_val >= 0.5:
        strengths.append(f"Climate: {' '.join(climate_notes)}" if climate_notes else "Favorable climate")
    elif climate_score_val > 0:
        uncertainties.append(f"Climate: {' '.join(climate_notes)}" if climate_notes else "Climate challenges noted")

    # Cross-instrument consistency (from cross dict, not sub-scores)
    consistency = cross.get("evidence_consistency", "insufficient_data")
    if consistency in ("consistent", "surface_multi"):
        strengths.append(f"Cross-instrument: {consistency.replace('_', ' ')}")
    elif consistency in ("partial",):
        uncertainties.append("Partial cross-instrument agreement")
    elif consistency != "insufficient_data":
        uncertainties.append(f"Cross-instrument: {consistency.replace('_', ' ')}")

    # Thermal inertia (from synthesis, not in scoring_methodology)
    ti_data = synthesis.get("thermal_inertia", {})
    if ti_data:
        ti_expl = ti_data.get("ti_explanation", "")
        ti_score = ti_data.get("ti_score", 0)
        if ti_score >= 4:
            strengths.append(ti_expl)
        elif ti_score > 0:
            uncertainties.append(ti_expl)
        elif ti_expl:
            uncertainties.append(ti_expl)

    # Score range: +/- based on uncertainty count
    uncertainty_margin = min(len(uncertainties) * 3, 12)
    confidence_bonus = min(len(strengths) * 2, 6)
    score_low = max(0, score - uncertainty_margin)
    score_high = min(95, score + confidence_bonus)

    synthesis["score_range"] = {"low": score_low, "high": score_high}
    synthesis["strengths"] = strengths
    synthesis["uncertainties"] = uncertainties

    # Backward-compat recommendation (map new classification to old labels)
    classification = recommendation.get("classification", "Low priority")
    _classification_to_label = {
        "Science-ready candidate": "STRONG_CANDIDATE",
        "Screening-level": "PROMISING_WITH_CAVEATS",
        "Engineering-safe only": "REQUIRES_FURTHER_INVESTIGATION",
        "Low priority": "LOW_PRIORITY",
    }
    synthesis["recommendation"] = _classification_to_label.get(classification, "REQUIRES_FURTHER_INVESTIGATION")

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
    Compute spatial consistency between SHARAD subsurface detections,
    CRISM ice/hydration signatures, and CNN H2O Ice classification.

    Three evidence lines:
    - SHARAD reflectors (direct subsurface)
    - CRISM spectral ice indices (indirect proxy)
    - CNN H2O Ice classification on TRR3 (direct surface, 95% confidence)

    Returns a dict with:
    - sharad_crism_min_distance_km: closest approach
    - evidence_consistency: consistent / partial / inconsistent / *_only / insufficient
    - direct_ice_evidence: list of SHARAD reflector detections (direct subsurface)
    - indirect_ice_evidence: list of CRISM spectral indicators (proxy)
    - cnn_surface_ice_evidence: list of CNN H2O detections (direct surface)
    - coincident_detections: count of SHARAD tracks near CRISM/CNN ice
    - notes: explanatory text for the report
    """
    cross: Dict[str, Any] = {
        "sharad_crism_min_distance_km": None,
        "evidence_consistency": "insufficient_data",
        "direct_ice_evidence": [],
        "indirect_ice_evidence": [],
        "cnn_surface_ice_evidence": [],
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

    # CRISM spectral ice data
    ice_hotspot = None
    top_ice = []
    crism_has_ice = False
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

    # CNN H2O Ice detections (direct surface evidence from TRR3 classification)
    cnn_h2o_hotspot = None
    cnn_h2o_obs = []
    cnn_has_h2o = False
    if "mineral_cnn" in all_results and all_results["mineral_cnn"].success:
        cnn_data = all_results["mineral_cnn"].data
        cnn_has_h2o = cnn_data.get("h2o_total_pixels", 0) > 0
        cnn_h2o_hotspot = cnn_data.get("h2o_hotspot")
        for obs in cnn_data.get("observations", []):
            if obs.get("h2o_pixels", 0) > 0:
                cnn_h2o_obs.append(obs)
                cross["cnn_surface_ice_evidence"].append({
                    "type": "CNN_H2O_Ice",
                    "obs_id": obs["obs_id"],
                    "lat": obs.get("lat"),
                    "lon": obs.get("lon"),
                    "h2o_pixels": obs["h2o_pixels"],
                    "h2o_percent": obs["h2o_percent"],
                    "h2o_rich": obs.get("h2o_rich", False),
                })

    # Combine all surface ice reference points (CRISM spectral + CNN H2O)
    # for proximity computation against SHARAD
    all_surface_ice_points = []
    if ice_hotspot:
        all_surface_ice_points.append({
            "source": "CRISM_spectral",
            "lat": ice_hotspot["center_lat"],
            "lon": ice_hotspot["center_lon"],
        })
    if cnn_h2o_hotspot:
        all_surface_ice_points.append({
            "source": "CNN_H2O",
            "lat": cnn_h2o_hotspot["center_lat"],
            "lon": cnn_h2o_hotspot["center_lon"],
        })

    # ── Multi-Scale Proximity Test ──
    PROXIMITY_SCALES = [25, 50, 100, 200]  # km
    SCALE_WEIGHTS = {25: 0.4, 50: 0.3, 100: 0.2, 200: 0.1}

    multi_scale_results = {}
    min_dist = float("inf")

    if sharad_detections and all_surface_ice_points:
        for scale in PROXIMITY_SCALES:
            coincidences = 0
            possible_pairs = 0
            for t in sharad_detections:
                for pt in all_surface_ice_points:
                    possible_pairs += 1
                    d = haversine_distance_km(t["lat"], t["lon"], pt["lat"], pt["lon"])
                    min_dist = min(min_dist, d)
                    if d <= scale:
                        coincidences += 1
            multi_scale_results[scale] = {
                "coincidences": coincidences,
                "possible_pairs": possible_pairs,
                "fraction": round(coincidences / max(possible_pairs, 1), 4),
            }

        cross["sharad_crism_min_distance_km"] = round(min_dist, 1)
        cross["multi_scale_proximity"] = multi_scale_results

        # Count SHARAD tracks within 100 km of any surface ice observation (backward compat)
        all_surface_obs = top_ice + cnn_h2o_obs
        for t in sharad_detections:
            for c in all_surface_obs:
                if c.get("lat") is not None:
                    d = haversine_distance_km(t["lat"], t["lon"], c["lat"], c["lon"])
                    if d < 100:
                        cross["coincident_detections"] += 1
                        break

    # ── Probabilistic Consistency Score (0-1) ──
    consistency_score = 0.0
    if sharad_detections and all_surface_ice_points and multi_scale_results:
        consistency_score = sum(
            SCALE_WEIGHTS[scale] * multi_scale_results[scale]["fraction"]
            for scale in PROXIMITY_SCALES
        )
    cross["consistency_score"] = round(consistency_score, 4)

    # ── Instrument Consistency Matrix ──
    # Compute pairwise min distances between instrument detections
    def _min_pairwise_dist(points_a, points_b):
        """Compute minimum pairwise distance between two lists of {lat,lon} dicts."""
        best = float("inf")
        for a in points_a:
            if a.get("lat") is None:
                continue
            for b in points_b:
                if b.get("lat") is None:
                    continue
                d = haversine_distance_km(a["lat"], a["lon"], b["lat"], b["lon"])
                best = min(best, d)
        return best if best < float("inf") else None

    sharad_pts = [{"lat": t["lat"], "lon": t["lon"]} for t in sharad_detections]
    crism_pts = [{"lat": c.get("lat"), "lon": c.get("lon")} for c in top_ice]
    cnn_pts = [{"lat": o.get("lat"), "lon": o.get("lon")} for o in cnn_h2o_obs]

    sharad_crism_dist = _min_pairwise_dist(sharad_pts, crism_pts)
    sharad_cnn_dist = _min_pairwise_dist(sharad_pts, cnn_pts)
    crism_cnn_dist = _min_pairwise_dist(crism_pts, cnn_pts)

    instrument_matrix = {
        "SHARAD_CRISM": {
            "distance_km": round(sharad_crism_dist, 1) if sharad_crism_dist is not None else None,
            "coincident": sharad_crism_dist is not None and sharad_crism_dist < 100,
        },
        "SHARAD_CNN": {
            "distance_km": round(sharad_cnn_dist, 1) if sharad_cnn_dist is not None else None,
            "coincident": sharad_cnn_dist is not None and sharad_cnn_dist < 100,
        },
        "CRISM_CNN": {
            "distance_km": round(crism_cnn_dist, 1) if crism_cnn_dist is not None else None,
            "coincident": crism_cnn_dist is not None and crism_cnn_dist < 100,
        },
    }
    cross["instrument_matrix"] = instrument_matrix

    # ── Non-Correlated Classification ──
    if multi_scale_results and all(
        multi_scale_results[s]["coincidences"] == 0 for s in PROXIMITY_SCALES
    ):
        cross["evidence_consistency"] = "non_correlated"
        consistency_score = 0.0
        cross["consistency_score"] = 0.0

    # ── Separate System Flag ──
    cross["separate_system"] = False
    if min_dist > 300 and min_dist < float("inf"):
        cross["separate_system"] = True
        cross["notes"].append(
            "CRISM ice cluster >300 km from nearest SHARAD reflector — "
            "classified as separate system"
        )

    # Determine evidence consistency (only if not already set to non_correlated)
    has_sharad = bool(sharad_detections)
    has_crism = crism_has_ice
    has_cnn_h2o = cnn_has_h2o
    surface_ice = has_crism or has_cnn_h2o  # any surface evidence

    if cross["evidence_consistency"] != "non_correlated":
        if has_sharad and surface_ice and min_dist < float("inf"):
            if min_dist < 50:
                cross["evidence_consistency"] = "consistent"
                sources = []
                if has_crism:
                    sources.append("CRISM spectral")
                if has_cnn_h2o:
                    sources.append("CNN H2O classification")
                src_str = " and ".join(sources)
                cross["notes"].append(
                    f"SHARAD subsurface reflectors and surface ice evidence ({src_str}) "
                    f"spatially coincide (within {min_dist:.0f} km), providing strong "
                    f"mutual corroboration of an ice deposit extending from the surface "
                    f"to the subsurface."
                )
            elif min_dist < 200:
                cross["evidence_consistency"] = "partial"
                cross["notes"].append(
                    f"SHARAD and surface ice evidence are moderately correlated "
                    f"({min_dist:.0f} km separation). The ice deposit may be laterally "
                    f"discontinuous, or the surface expression may not directly "
                    f"overlie the subsurface interface."
                )
            else:
                cross["evidence_consistency"] = "inconsistent"
                cross["notes"].append(
                    f"SHARAD reflectors and surface ice signatures do not spatially coincide "
                    f"({min_dist:.0f} km apart). This may indicate distinct ice reservoirs."
                )
        elif has_sharad and not surface_ice:
            cross["evidence_consistency"] = "sharad_only"
            cross["notes"].append(
                "Subsurface reflectors detected by SHARAD but no corresponding surface "
                "ice signatures from CRISM spectral or CNN classification. This is "
                "consistent with buried ice beneath a protective regolith layer."
            )
        elif has_sharad and surface_ice and min_dist == float("inf"):
            # Surface ice detected but without usable coordinates
            cross["evidence_consistency"] = "partial"
            cross["notes"].append(
                "Both SHARAD subsurface reflectors and surface ice evidence are present, "
                "but spatial coordinates are insufficient for proximity analysis."
            )
        elif surface_ice and not has_sharad:
            if has_cnn_h2o and has_crism:
                cross["evidence_consistency"] = "surface_multi"
                cross["notes"].append(
                    "Both CRISM spectral indices and CNN H2O classification confirm surface "
                    "ice, but no SHARAD subsurface reflectors were detected. The ice may be "
                    "too shallow or too thin for SHARAD vertical resolution (~15 m). "
                    "The CNN confirmation at 95% confidence strengthens the surface ice case."
                )
            elif has_cnn_h2o:
                cross["evidence_consistency"] = "cnn_h2o_only"
                cross["notes"].append(
                    "CNN mineral classifier detected surface H2O ice on TRR3 data (95% "
                    "confidence), but no SHARAD subsurface reflectors or CRISM spectral "
                    "ice indices were found. This indicates exposed water ice visible "
                    "in the infrared spectrum."
                )
            else:
                cross["evidence_consistency"] = "crism_only"
                cross["notes"].append(
                    "CRISM spectral signatures suggest surface or near-surface ice/hydration, "
                    "but no SHARAD subsurface reflectors were detected. The ice may be too "
                    "shallow or too thin for SHARAD vertical resolution (~15 m)."
                )
        else:
            cross["notes"].append(
                "Insufficient data for cross-instrument comparison. Neither SHARAD "
                "subsurface reflectors nor surface ice signatures were identified."
            )

    # CNN H2O + CRISM spectral mutual reinforcement note
    if has_cnn_h2o and has_crism:
        cross["notes"].append(
            "CNN H2O classification and CRISM spectral ice indices independently "
            "confirm surface ice presence — two distinct detection methods agree, "
            "increasing confidence in surface water ice."
        )

    # ── Stratigraphic Interpretation (physics-based cross-instrument) ──
    sharad_inv = all_results.get("sharad_physics_inversion")
    crism_spec = all_results.get("crism_spectral")

    stratigraphic = {"interpretation": "insufficient_data", "notes": []}

    has_sharad_inversion = (
        sharad_inv and sharad_inv.success
        and sharad_inv.data.get("inversions_completed", 0) > 0
    )
    has_crism_spectral = (
        crism_spec and crism_spec.success
        and crism_spec.data.get("observations_analyzed", 0) > 0
    )

    if has_sharad_inversion and has_crism_spectral:
        eps_r = sharad_inv.data.get("best_epsilon_r")
        water_frac = crism_spec.data.get("water_ice_overall_fraction", 0)

        if eps_r is not None and eps_r < 4.0 and water_frac > 0.01:
            stratigraphic["interpretation"] = "deep_reservoir_shallow_exposure"
            stratigraphic["notes"].append(
                f"SHARAD εr={eps_r:.2f} indicates ice-rich subsurface (deep reservoir). "
                f"CRISM water ice fraction={water_frac:.1%} confirms shallow surface exposure. "
                f"Consistent with obliquity-driven ice deposition model."
            )
        elif eps_r is not None and eps_r < 4.0 and water_frac <= 0.01:
            stratigraphic["interpretation"] = "buried_ice_no_surface_exposure"
            stratigraphic["notes"].append(
                f"SHARAD εr={eps_r:.2f} suggests subsurface ice, but CRISM shows no surface "
                f"water ice (fraction={water_frac:.1%}). Lateral heterogeneity or dune cover "
                f"may obscure surface expression."
            )
        elif eps_r is not None and eps_r >= 4.0 and water_frac > 0.01:
            stratigraphic["interpretation"] = "surface_ice_rocky_subsurface"
            stratigraphic["notes"].append(
                f"SHARAD εr={eps_r:.2f} indicates rocky/basaltic subsurface. "
                f"CRISM water ice fraction={water_frac:.1%} shows surface ice present "
                f"but not extending to depth. Thin ice veneer or seasonal frost."
            )
        elif eps_r is not None and eps_r >= 4.0:
            stratigraphic["interpretation"] = "no_ice_evidence"
            stratigraphic["notes"].append(
                f"Neither instrument supports ice presence: SHARAD εr={eps_r:.2f} (rocky), "
                f"CRISM water ice fraction={water_frac:.1%} (negligible)."
            )
    elif has_sharad_inversion:
        eps_r = sharad_inv.data.get("best_epsilon_r")
        if eps_r is not None:
            stratigraphic["interpretation"] = "sharad_only"
            stratigraphic["notes"].append(
                f"SHARAD physics inversion: εr={eps_r:.2f}. "
                f"No CRISM spectral data to cross-validate surface composition."
            )
    elif has_crism_spectral:
        water_frac = crism_spec.data.get("water_ice_overall_fraction", 0)
        stratigraphic["interpretation"] = "crism_only"
        stratigraphic["notes"].append(
            f"CRISM spectral analysis: water ice fraction={water_frac:.1%}. "
            f"No SHARAD physics inversion to constrain subsurface."
        )

    cross["stratigraphic_interpretation"] = stratigraphic

    return cross


# =============================================================================
# Task: Terrain ε Inversion (MOLA terraced crater → SHARAD pipeline)
# =============================================================================

def terrain_epsilon_inversion(
    lat: float,
    lon: float,
    diameter_km: float,
    terrace_depth_m: float,
    products: List[Dict[str, Any]],
    bbox: "RegionBBox",
) -> TaskResult:
    """
    Run εr inversion for a terraced crater detected by MOLA scan.

    Bridges MOLA landform detection to the existing epsilon_terrace and
    sharad_inversion pipelines:
    1. Search SHARAD_HIGHRES + HIRISE_DTM near the crater
    2. Run terrace_dielectric_analysis (terrace depth + SHARAD TWT → εr)
    3. Cross-validate with sharad_physics_inversion if tracks exist
    4. Return combined results with formula: εr = (c·Δt / 2d)²

    Parameters
    ----------
    lat, lon : float
        Terraced crater center coordinates.
    diameter_km : float
        Crater diameter in km (from MOLA detection).
    terrace_depth_m : float
        Terrace bench depth below rim (from MOLA detection).
    products : list
        Available instrument products in the region.
    bbox : RegionBBox
        Region bounding box for searches.
    """
    import numpy as np

    # Find SHARAD and DTM products near this crater
    sharad_nearby = []
    dtm_nearby = []
    search_radius_km = max(diameter_km * 2, 50.0)

    for p in products:
        p_lat = p.get("lat") or p.get("center_lat", 0)
        p_lon = p.get("lon") or p.get("center_lon", 0)
        if p_lat == 0 and p_lon == 0:
            continue
        dist = haversine_distance_km(lat, lon, p_lat, p_lon)
        if dist > search_radius_km:
            continue
        inst = p.get("instrument", "")
        if inst == "SHARAD_HIGHRES":
            sharad_nearby.append(p)
        elif inst == "HIRISE_DTM":
            dtm_nearby.append(p)

    if not sharad_nearby:
        return TaskResult(
            task_type="terrain_epsilon_inversion",
            success=True,
            data={
                "crater_lat": lat,
                "crater_lon": lon,
                "diameter_km": diameter_km,
                "terrace_depth_m": terrace_depth_m,
                "sharad_tracks_nearby": 0,
                "dtm_products_nearby": len(dtm_nearby),
                "epsilon_r": None,
                "reason": "no_sharad_tracks",
            },
            summary=(
                f"No SHARAD tracks within {search_radius_km:.0f} km of "
                f"terraced crater at ({lat:.3f}, {lon:.3f}). "
                f"Cannot compute εr without radar data."
            ),
        )

    # Run terrace dielectric analysis using existing pipeline
    # Build a minimal subsurface result for the terrace pipeline
    sub_result = TaskResult(
        task_type="subsurface",
        success=True,
        data={
            "tracks": [{
                "subsurface_detected": True,
                "estimated_depth_m": terrace_depth_m,
                "product_id": sp["product_id"],
            } for sp in sharad_nearby[:3]],
        },
    )

    terrace_result = terrace_dielectric_analysis(sub_result, products, bbox)

    # Also run physics inversion for cross-validation
    physics_result = sharad_physics_inversion(products, bbox)

    # Combine results
    terrace_eps = terrace_result.data.get("median_epsilon_r")
    physics_eps = physics_result.data.get("best_epsilon_r")

    combined_eps = None
    interpretation = "insufficient_data"
    method_used = []

    if terrace_eps is not None:
        combined_eps = terrace_eps
        interpretation = terrace_result.data.get("interpretation", "unknown")
        method_used.append("terrace_dielectric")

    if physics_eps is not None:
        if combined_eps is not None:
            # Average the two estimates
            combined_eps = round((combined_eps + physics_eps) / 2, 2)
            method_used.append("physics_inversion")
        else:
            combined_eps = physics_eps
            interpretation = "physics_inversion_only"
            method_used.append("physics_inversion")

    # Generate interpretation from combined εr
    if combined_eps is not None:
        if combined_eps < 2.5:
            interpretation = "dry regolith or porous ice"
        elif combined_eps < 3.5:
            interpretation = "ice-rich subsurface (consistent with water ice)"
        elif combined_eps < 5.0:
            interpretation = "ice-cemented regolith"
        else:
            interpretation = "basaltic regolith or dense rock"

    data = {
        "crater_lat": lat,
        "crater_lon": lon,
        "diameter_km": diameter_km,
        "terrace_depth_m": terrace_depth_m,
        "sharad_tracks_nearby": len(sharad_nearby),
        "dtm_products_nearby": len(dtm_nearby),
        "epsilon_r": combined_eps,
        "interpretation": interpretation,
        "method_used": method_used,
        "formula": "εr = (c · Δt / (2 · depth))²",
        "terrace_analysis": terrace_result.data,
        "physics_analysis": physics_result.data,
    }

    if combined_eps is not None:
        summary = (
            f"Terrain εr inversion at ({lat:.3f}, {lon:.3f}): "
            f"εr = {combined_eps:.2f} ({interpretation}). "
            f"Methods: {', '.join(method_used)}. "
            f"Crater: {diameter_km:.1f} km, terrace depth {terrace_depth_m:.0f} m."
        )
    else:
        summary = (
            f"Terrain εr inversion at ({lat:.3f}, {lon:.3f}): "
            f"No reliable εr estimate. "
            f"{len(sharad_nearby)} SHARAD tracks, {len(dtm_nearby)} DTMs nearby."
        )

    return TaskResult(
        task_type="terrain_epsilon_inversion",
        success=True,
        data=data,
        summary=summary,
    )
