"""
Epsilon-terrace analysis pipeline.

CLI entrypoint and programmatic API for estimating dielectric constant (εr)
of Mars subsurface using terraced craters as "true depth" constraints.

Usage:
    python -m backend.analysis.epsilon_terrace.run \
        --sharad R_7748801_001_SS4_700_A \
        --dtm_dir backend/hirise_dtm_data \
        --buffer_km 30
"""

import os
import sys
import json
import math
import argparse
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import asdict

import numpy as np

# Ensure project root is on path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from backend.analysis.epsilon_terrace.terrace_detect import (
    extract_radial_profiles,
    TerraceResult,
)
from backend.analysis.epsilon_terrace.sharad_pick import (
    pick_subsurface_interface,
    SharadPickResult,
)
from backend.analysis.epsilon_terrace.epsilon_calc import (
    compute_epsilon,
    run_epsilon_analysis,
    EpsilonResult,
)

logger = logging.getLogger(__name__)

MARS_RADIUS_KM = 3389.5


def _haversine_km(lat1, lon1, lat2, lon2):
    rlat1, rlat2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2
    return MARS_RADIUS_KM * 2 * math.asin(min(1.0, math.sqrt(a)))


def _equirect_params(crs_wkt):
    import re
    cm_match = re.search(r'central_meridian["\s,]+(-?[\d.]+)', crs_wkt, re.IGNORECASE)
    sp_match = re.search(r'standard_parallel_1["\s,]+(-?[\d.]+)', crs_wkt, re.IGNORECASE)
    radius_match = re.search(r'SPHEROID\["[^"]*",\s*([\d.]+)', crs_wkt)
    cm = float(cm_match.group(1)) if cm_match else 0.0
    sp = float(sp_match.group(1)) if sp_match else 0.0
    radius = float(radius_match.group(1)) if radius_match else 3389500.0
    return cm, sp, radius


def _get_dtm_geo_bounds(dtm_path: str) -> Optional[Dict]:
    """Get geographic bounds of a DTM file."""
    import rasterio
    try:
        with rasterio.open(dtm_path) as ds:
            b = ds.bounds
            wkt = ds.crs.to_wkt()
            cm, sp, radius = _equirect_params(wkt)
            cos_sp = math.cos(math.radians(sp))
            west = math.degrees(b.left / (radius * cos_sp)) + cm
            east = math.degrees(b.right / (radius * cos_sp)) + cm
            south = math.degrees(b.bottom / radius)
            north = math.degrees(b.top / radius)
            if west > 180: west -= 360
            if east > 180: east -= 360
            return {
                'west': west, 'east': east, 'south': south, 'north': north,
                'center_lat': (south + north) / 2,
                'center_lon': (west + east) / 2,
            }
    except Exception:
        return None


def find_nearby_dtms(
    sharad_product_id: str,
    dtm_dir: str,
    buffer_km: float = 30.0,
    sharad_index_path: Optional[str] = None,
) -> List[Dict]:
    """
    Find DTM files that are near a SHARAD track.

    Returns list of dicts with dtm_path, center_lat, center_lon, distance_km, title.
    """
    import glob as globmod

    # Load SHARAD track geometry
    if sharad_index_path is None:
        backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        sharad_index_path = os.path.join(backend_dir, "sharad_highres_data", "index.geojson")

    with open(sharad_index_path) as f:
        idx = json.load(f)

    # Find this track
    track_coords = None
    for feat in idx['features']:
        if feat['properties'].get('product_id') == sharad_product_id:
            track_coords = feat['geometry']['coordinates']
            break

    if track_coords is None:
        logger.warning(f"SHARAD product {sharad_product_id} not found in index")
        return []

    # Get all DTM files
    dtm_files = globmod.glob(os.path.join(dtm_dir, "*.IMG"))
    if not dtm_files:
        dtm_files = globmod.glob(os.path.join(dtm_dir, "*.img"))

    # Load DTM metadata if available
    meta_path = os.path.join(dtm_dir, "crawl_metadata.json")
    dtm_meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta_list = json.load(f)
        for m in meta_list:
            dtm_meta[m.get('dtm_filename', '')] = m

    # Check each DTM against track
    buffer_deg = buffer_km / (MARS_RADIUS_KM * math.pi / 180)
    results = []

    for dtm_path in dtm_files:
        geo = _get_dtm_geo_bounds(dtm_path)
        if geo is None:
            continue

        # Check if any SHARAD track point is within buffer of DTM
        for lon, lat in track_coords:
            if (geo['west'] - buffer_deg <= lon <= geo['east'] + buffer_deg and
                geo['south'] - buffer_deg <= lat <= geo['north'] + buffer_deg):
                dist = _haversine_km(geo['center_lat'], geo['center_lon'], lat, lon)
                fname = os.path.basename(dtm_path)
                meta = dtm_meta.get(fname, {})
                results.append({
                    'dtm_path': dtm_path,
                    'dtm_filename': fname,
                    'center_lat': geo['center_lat'],
                    'center_lon': geo['center_lon'],
                    'distance_km': round(dist, 1),
                    'title': meta.get('title', fname),
                    'bounds': geo,
                })
                break

    results.sort(key=lambda x: x['distance_km'])
    return results


def run_pipeline(
    sharad_product_id: str,
    dtm_dir: str,
    buffer_km: float = 30.0,
    n_azimuths: int = 36,
    max_radius_m: float = 3000.0,
    output_dir: Optional[str] = None,
) -> Dict:
    """
    Full epsilon-terrace analysis pipeline.

    1. Find DTMs near the SHARAD track
    2. Extract terrace profiles from each DTM
    3. Pick subsurface interface from SHARAD near each crater
    4. Compute εr for each crater

    Returns summary dict suitable for JSON serialization.
    """
    print(f"\n{'='*60}")
    print(f"  Epsilon-Terrace Analysis Pipeline")
    print(f"  SHARAD: {sharad_product_id}")
    print(f"  DTM dir: {dtm_dir}")
    print(f"  Buffer: {buffer_km} km")
    print(f"{'='*60}\n")

    # Step 1: Find nearby DTMs
    print("[1/4] Finding DTMs near SHARAD track...")
    dtms = find_nearby_dtms(sharad_product_id, dtm_dir, buffer_km)
    print(f"  Found {len(dtms)} DTM(s) within {buffer_km}km")
    for d in dtms:
        print(f"    {d['dtm_filename']} ({d['title']}) - {d['distance_km']}km")

    if not dtms:
        return {'error': 'No DTMs found near SHARAD track', 'sharad': sharad_product_id}

    # Step 2: Extract terrace profiles
    print(f"\n[2/4] Extracting terrace profiles...")
    terrace_results: List[TerraceResult] = []
    for d in dtms:
        print(f"  Processing {d['dtm_filename']}...")
        # Auto-detect crater center from DTM (metadata center is extent centroid, not crater center)
        tr = extract_radial_profiles(
            d['dtm_path'],
            center_lat=None,
            center_lon=None,
            n_azimuths=n_azimuths,
            max_radius_m=max_radius_m,
        )
        if tr.error:
            print(f"    ERROR: {tr.error}")
        else:
            n_med_terraces = len([t for t in tr.terraces if t.azimuth_deg == 0.0])
            print(f"    Crater radius: {tr.crater_radius_m:.0f}m")
            print(f"    Terraces found (median profile): {n_med_terraces}")
            for depth in tr.terrace_depths:
                print(f"    Depth [{depth['metric']}]: {depth['depth_m']:.1f}m ± {depth['uncertainty_m']:.1f}m")
        terrace_results.append(tr)

    # Step 3: SHARAD subsurface interface picks
    print(f"\n[3/4] Picking SHARAD subsurface interfaces...")
    pick_pairs: List[Tuple[TerraceResult, SharadPickResult]] = []
    for tr, d in zip(terrace_results, dtms):
        if tr.error or not tr.terrace_depths:
            print(f"  Skipping {d['dtm_filename']} (no terraces)")
            continue

        # Use auto-detected crater center for SHARAD picks
        crater_lat = tr.crater_center_lat
        crater_lon = tr.crater_center_lon
        print(f"  Picking near {d['dtm_filename']} (crater at {crater_lat:.3f}, {crater_lon:.3f})...")
        pr = pick_subsurface_interface(
            sharad_product_id,
            crater_lat,
            crater_lon,
            window_km=max(5.0, buffer_km / 2),
        )
        if pr.error:
            print(f"    {pr.error}")
        else:
            print(f"    {len(pr.picks)} picks, median twt = {pr.median_twt_us:.4f} µs ± {pr.twt_unc_us:.4f}")
        pick_pairs.append((tr, pr))

    # Step 4: Compute εr
    print(f"\n[4/4] Computing dielectric constants...")
    eps_result = run_epsilon_analysis(terrace_results, pick_pairs)

    for est in eps_result.estimates:
        print(f"  {est.crater_id} [{est.depth_metric}]:")
        print(f"    depth = {est.depth_true_m}m, twt = {est.twt_us} µs")
        print(f"    εr = {est.epsilon_r} [{est.epsilon_low}, {est.epsilon_high}]")
        print(f"    → {est.interpretation}")

    print(f"\n{'='*60}")
    print(f"  Summary: {eps_result.summary}")
    print(f"{'='*60}\n")

    # Build output
    output = {
        'sharad_product': sharad_product_id,
        'buffer_km': buffer_km,
        'dtms_found': len(dtms),
        'dtms': [{
            'filename': d['dtm_filename'],
            'title': d['title'],
            'center_lat': d['center_lat'],
            'center_lon': d['center_lon'],
            'distance_km': d['distance_km'],
        } for d in dtms],
        'terrace_analysis': [],
        'sharad_picks': [],
        'epsilon_estimates': [],
        'summary': eps_result.summary,
    }

    for tr in terrace_results:
        ta = {
            'dtm_file': os.path.basename(tr.dtm_file),
            'crater_center': [tr.crater_center_lat, tr.crater_center_lon],
            'crater_radius_m': tr.crater_radius_m,
            'n_terraces_median': len([t for t in tr.terraces if t.azimuth_deg == 0.0]),
            'terrace_depths': tr.terrace_depths,
            'error': tr.error,
        }
        output['terrace_analysis'].append(ta)

    for tr, pr in pick_pairs:
        sp = {
            'dtm_file': os.path.basename(tr.dtm_file),
            'crater_lat': pr.crater_lat,
            'crater_lon': pr.crater_lon,
            'n_picks': len(pr.picks),
            'median_twt_us': pr.median_twt_us,
            'twt_unc_us': pr.twt_unc_us,
            'error': pr.error,
            'picks': [{
                'trace_idx': p.trace_idx,
                'lat': p.lat,
                'lon': p.lon,
                'delta_bins': p.delta_bins,
                'twt_us': p.twt_us,
                'snr': p.snr,
                'distance_to_crater_km': p.distance_to_crater_km,
            } for p in pr.picks],
        }
        output['sharad_picks'].append(sp)

    for est in eps_result.estimates:
        output['epsilon_estimates'].append({
            'crater_id': est.crater_id,
            'depth_metric': est.depth_metric,
            'depth_true_m': est.depth_true_m,
            'depth_unc_m': est.depth_unc_m,
            'twt_us': est.twt_us,
            'twt_unc_us': est.twt_unc_us,
            'epsilon_r': est.epsilon_r,
            'epsilon_low': est.epsilon_low,
            'epsilon_high': est.epsilon_high,
            'interpretation': est.interpretation,
            'quality': est.quality,
            'quality_note': est.quality_note,
            'sharad_product': est.sharad_product,
            'n_sharad_picks': est.n_sharad_picks,
        })

    # Save output
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"epsilon_terrace_{sharad_product_id}.json")
        with open(out_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"Results saved to: {out_path}")

    return output


# Async wrapper for agentic AI integration
async def run_pipeline_async(
    sharad_product_id: str,
    dtm_dir: str,
    buffer_km: float = 30.0,
    output_dir: Optional[str] = None,
) -> Dict:
    """Async wrapper that runs the pipeline in a thread pool."""
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: run_pipeline(sharad_product_id, dtm_dir, buffer_km, output_dir=output_dir),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Estimate dielectric constant (εr) via terraced crater analysis"
    )
    parser.add_argument("--sharad", required=True, help="SHARAD high-res product ID")
    parser.add_argument("--dtm_dir", required=True, help="Path to HiRISE DTM directory")
    parser.add_argument("--buffer_km", type=float, default=30.0, help="Search buffer in km")
    parser.add_argument("--output_dir", default=None, help="Output directory (default: dtm_dir/epsilon_results)")
    parser.add_argument("--n_azimuths", type=int, default=36, help="Number of radial profile azimuths")
    parser.add_argument("--max_radius_m", type=float, default=3000.0, help="Max radial distance from crater center")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    output_dir = args.output_dir or os.path.join(args.dtm_dir, "epsilon_results")

    result = run_pipeline(
        sharad_product_id=args.sharad,
        dtm_dir=args.dtm_dir,
        buffer_km=args.buffer_km,
        n_azimuths=args.n_azimuths,
        max_radius_m=args.max_radius_m,
        output_dir=output_dir,
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
