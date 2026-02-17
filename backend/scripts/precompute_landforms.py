#!/usr/bin/env python3
"""
Pre-compute landform detection for the entire Mars surface.

Tiles Mars into a latitude-adaptive grid, runs all 4 detectors
(craters, channels, ridges, LDAs) on each tile using multiprocessing,
deduplicates features across tile boundaries, and saves results to a
persistent JSON cache.

Usage:
    cd backend
    python scripts/precompute_landforms.py [--resume] [--radius 100] [--workers 16]

Output:
    cache/landforms_precomputed.json
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Ensure backend/ is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MARS_MEAN_RADIUS_KM = 3389.5
CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
OUTPUT_FILE = CACHE_DIR / "landforms_precomputed.json"
PROGRESS_FILE = CACHE_DIR / "landforms_progress.json"

# Default scan parameters matching ScanRequest defaults
DEFAULT_MIN_DIAMETER_KM = 5.0
DEFAULT_MIN_DEPTH_M = 100.0


# ---------------------------------------------------------------------------
# Tile grid generation
# ---------------------------------------------------------------------------

def generate_tile_grid(
    lat_step_deg: float = 2.5,
    lon_step_deg: float = 3.0,
    lat_min: float = -75.0,
    lat_max: float = 75.0,
) -> list[tuple[float, float]]:
    """
    Generate a latitude-adaptive tile grid covering Mars.

    At higher latitudes, longitude steps are widened proportionally
    to 1/cos(lat) to maintain roughly equal tile areas.

    Latitude range capped at ±75° to avoid oversized DEM windows
    at the poles (where equirectangular projection inflates pixel counts).

    Returns list of (lat, lon) center coordinates.
    """
    tiles = []
    lat = lat_min
    while lat <= lat_max:
        # Adjust longitude step for latitude convergence
        cos_lat = max(math.cos(math.radians(lat)), 0.25)
        effective_lon_step = min(lon_step_deg / cos_lat, 30.0)

        lon = -180.0
        while lon < 180.0:
            tiles.append((round(lat, 2), round(lon, 2)))
            lon += effective_lon_step

        lat += lat_step_deg

    return tiles


# ---------------------------------------------------------------------------
# Feature serialization
# ---------------------------------------------------------------------------

def feature_to_dict(f) -> dict:
    """Convert DetectedFeature to the same JSON format as the SSE endpoint."""
    d = {
        "id": f.id,
        "type": f.feature_type,
        "lat": round(f.lat, 5),
        "lon": round(f.lon, 5),
        "confidence": round(f.confidence, 2),
        "description": f.description,
        "morphology": f.morphology,
    }
    if f.diameter_km > 0:
        d["diameter_km"] = round(f.diameter_km, 1)
    if f.depth_m > 0:
        d["depth_m"] = round(f.depth_m, 1)
    if f.depth_diameter_ratio > 0:
        d["depth_diameter_ratio"] = round(f.depth_diameter_ratio, 4)
    if f.circularity > 0:
        d["circularity"] = round(f.circularity, 2)
    if f.rim_elevation_m != 0:
        d["rim_elevation_m"] = round(f.rim_elevation_m, 1)
    if f.floor_elevation_m != 0:
        d["floor_elevation_m"] = round(f.floor_elevation_m, 1)
    if f.n_terraces > 0:
        d["n_terraces"] = f.n_terraces
    if f.length_km > 0:
        d["length_km"] = round(f.length_km, 1)
    if f.width_km > 0:
        d["width_km"] = round(f.width_km, 1)
    if f.area_km2 > 0:
        d["area_km2"] = round(f.area_km2, 1)
    if f.sinuosity > 0:
        d["sinuosity"] = round(f.sinuosity, 2)
    if f.path:
        path = f.path
        if len(path) > 80:
            step = max(1, len(path) // 80)
            path = path[::step]
        d["path"] = [[round(p[0], 4), round(p[1], 4)] for p in path]
    if f.boundary:
        boundary = f.boundary
        if len(boundary) > 80:
            step = max(1, len(boundary) // 80)
            boundary = boundary[::step]
        d["boundary"] = [[round(p[0], 4), round(p[1], 4)] for p in boundary]
    if f.terrace_depth_m > 0:
        d["terrace_depth_m"] = round(f.terrace_depth_m, 1)
    if f.terrace_ring_radii_km:
        d["terrace_ring_radii_km"] = [round(r, 2) for r in f.terrace_ring_radii_km]
    return d


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km on Mars."""
    R = MARS_MEAN_RADIUS_KM
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(max(1 - a, 0)))


def deduplicate_features(
    features: list[dict],
    proximity_km: float = 5.0,
) -> list[dict]:
    """
    Remove duplicate detections from overlapping tiles.

    Two features are considered duplicates if they have the same type
    and their centroids are within proximity_km. The instance with
    higher confidence is kept.
    """
    if not features:
        return []

    # Sort by confidence descending so we keep higher-confidence first
    features_sorted = sorted(features, key=lambda f: f.get("confidence", 0), reverse=True)
    kept: list[dict] = []
    # Group by type for faster comparison
    by_type: dict[str, list[dict]] = {}

    for f in features_sorted:
        ftype = f["type"]
        lat = f["lat"]
        lon = f["lon"]

        is_dup = False
        for existing in by_type.get(ftype, []):
            dist = _haversine_km(lat, lon, existing["lat"], existing["lon"])
            if dist < proximity_km:
                is_dup = True
                break

        if not is_dup:
            kept.append(f)
            by_type.setdefault(ftype, []).append(f)

    return kept


# ---------------------------------------------------------------------------
# Single tile processing (called in worker processes)
# ---------------------------------------------------------------------------

def _process_tile_worker(args: tuple) -> tuple[float, float, list[dict]]:
    """
    Worker function for multiprocessing. Takes (lat, lon, radius_km).
    Returns (lat, lon, features_list).

    Each worker re-imports modules to avoid shared state issues.
    """
    lat, lon, radius_km = args

    # Import in worker to get fresh module state
    from analysis.mola_detect.common import build_shared_context
    from analysis.mola_detect.crater_detect import detect_craters_and_volcanics
    from analysis.mola_detect.ridge_channel_detect import detect_channels, detect_ridges
    from analysis.mola_detect.lda_detect import detect_ldas

    features_raw = []

    try:
        ctx = build_shared_context(lat, lon, radius_km)
    except Exception as e:
        return (lat, lon, [])

    # 1. Craters / volcanics / graben
    try:
        features_raw.extend(detect_craters_and_volcanics(
            lat, lon, radius_km,
            min_diameter_km=DEFAULT_MIN_DIAMETER_KM,
            min_depth_m=DEFAULT_MIN_DEPTH_M,
            ctx=ctx,
        ))
    except Exception:
        pass

    # 2. Channels
    try:
        features_raw.extend(detect_channels(lat, lon, radius_km, ctx=ctx))
    except Exception:
        pass

    # 3. Ridges
    try:
        features_raw.extend(detect_ridges(lat, lon, radius_km, ctx=ctx))
    except Exception:
        pass

    # 4. LDAs (latitude filter disabled for full coverage)
    try:
        features_raw.extend(detect_ldas(lat, lon, radius_km, latitude_filter=False, ctx=ctx))
    except Exception:
        pass

    return (lat, lon, [feature_to_dict(f) for f in features_raw])


# ---------------------------------------------------------------------------
# Progress tracking (for resume)
# ---------------------------------------------------------------------------

def load_progress() -> tuple[set[tuple[float, float]], list[dict]]:
    """Load progress file tracking completed tiles."""
    if PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE) as f:
                data = json.load(f)
            completed = set(tuple(t) for t in data.get("completed_tiles", []))
            features = data.get("features", [])
            return completed, features
        except Exception:
            pass
    return set(), []


def save_progress(completed: set[tuple[float, float]], features: list[dict]):
    """Save progress to disk."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "completed_tiles": [list(t) for t in completed],
        "features": features,
    }
    tmp = PROGRESS_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    tmp.rename(PROGRESS_FILE)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Pre-compute Mars landform detection")
    parser.add_argument("--resume", action="store_true", help="Resume from progress file")
    parser.add_argument("--radius", type=float, default=100.0, help="Scan radius per tile (km)")
    parser.add_argument("--lat-step", type=float, default=2.5, help="Latitude step (degrees)")
    parser.add_argument("--lon-step", type=float, default=3.0, help="Longitude step (degrees)")
    parser.add_argument("--workers", type=int, default=16, help="Number of parallel workers")
    args = parser.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    tiles = generate_tile_grid(
        lat_step_deg=args.lat_step,
        lon_step_deg=args.lon_step,
    )
    logger.info(
        "Generated %d tiles (radius=%.0f km, lat_step=%.1f°, lon_step=%.1f°, workers=%d)",
        len(tiles), args.radius, args.lat_step, args.lon_step, args.workers,
    )

    # Resume support
    if args.resume:
        completed_set, all_features = load_progress()
        logger.info(
            "Resuming: %d tiles already completed, %d features so far",
            len(completed_set), len(all_features),
        )
    else:
        completed_set: set[tuple[float, float]] = set()
        all_features: list[dict] = []

    remaining = [(lat, lon) for (lat, lon) in tiles if (lat, lon) not in completed_set]
    logger.info("%d tiles remaining", len(remaining))

    if not remaining:
        logger.info("All tiles already completed!")
    else:
        t_start = time.time()
        save_interval = 50  # Save progress every N completed tiles
        tiles_since_save = 0

        # Build worker args
        worker_args = [(lat, lon, args.radius) for lat, lon in remaining]

        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_process_tile_worker, a): a for a in worker_args}
            done_count = 0
            total = len(futures)

            for future in as_completed(futures):
                done_count += 1
                tiles_since_save += 1

                try:
                    lat, lon, tile_features = future.result()
                    all_features.extend(tile_features)
                    completed_set.add((lat, lon))

                    elapsed_total = time.time() - t_start
                    rate = elapsed_total / done_count
                    eta = rate * (total - done_count)

                    if done_count % 10 == 0 or done_count == total:
                        logger.info(
                            "[%d/%d] (%.2f°, %.2f°) → %d features | "
                            "Total: %d | Rate: %.1fs/tile | ETA: %.0f min",
                            done_count, total, lat, lon, len(tile_features),
                            len(all_features), rate, eta / 60,
                        )
                except Exception as e:
                    lat, lon, _ = futures[future]
                    logger.warning("Tile (%.2f, %.2f) failed: %s", lat, lon, e)

                # Periodic progress save
                if tiles_since_save >= save_interval:
                    save_progress(completed_set, all_features)
                    tiles_since_save = 0
                    logger.info("  Progress saved (%d tiles)", len(completed_set))

        # Final progress save
        save_progress(completed_set, all_features)
        elapsed_total = time.time() - t_start
        logger.info("All tiles processed in %.0f min", elapsed_total / 60)

    # Deduplicate
    logger.info("Deduplicating %d raw features ...", len(all_features))
    deduped = deduplicate_features(all_features, proximity_km=5.0)
    logger.info("After deduplication: %d features", len(deduped))

    # Type counts
    type_counts: dict[str, int] = {}
    for f in deduped:
        type_counts[f["type"]] = type_counts.get(f["type"], 0) + 1
    for ftype, count in sorted(type_counts.items()):
        logger.info("  %s: %d", ftype, count)

    # Write final output
    output = {
        "version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tile_count": len(tiles),
        "scan_radius_km": args.radius,
        "total_features": len(deduped),
        "type_counts": type_counts,
        "features": deduped,
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, separators=(",", ":"))

    file_size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    total_time_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - (time.time() - 1)))
    logger.info(
        "Done! %d features written to %s (%.1f MB)",
        len(deduped), OUTPUT_FILE, file_size_mb,
    )

    # Clean up progress file
    if PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()
        logger.info("Cleaned up progress file")


if __name__ == "__main__":
    main()
