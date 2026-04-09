#!/usr/bin/env python3
"""Batch process Mastcam-Z frames per sol through the coregistration pipeline.

By default processes all left-eye linearized XYZ frames per sol.

For each sol:
  1. Find all left-eye linearized XYZ products (ZLF_*XYZLN*)
  2. Download XYZ + matching RAS products
  3. SPICE transform: SITE_FRAME → ROVER_NAV_FRAME → IAU_MARS → lon/lat
  4. Save per-frame and combined lon/lat maps
"""

import sys
import time
import json
import argparse
import traceback
from pathlib import Path

from coregister.pds_fetch import (
    list_available_sols,
    search_mastcamz_products,
    download_mastcamz_product,
)
from coregister.spice_setup import init_spice, cleanup_spice
from coregister.mastcam_xyz import process_mastcamz_xyz
from coregister.config import OUTPUT_DIR, PDS_CACHE

import numpy as np


def get_left_linearized_products(sol: int) -> list[dict]:
    """Get only left-eye linearized XYZ products (ZLF_*XYZLN*).

    Linearized (L) products have corrected lens distortion.
    We skip right-eye (ZRF) and non-linearized (XYZ_) variants.
    """
    all_xyz = search_mastcamz_products(sol, "XYZ")
    # Filter: ZLF = left, XYZLN = linearized XYZ
    left_lin = [p for p in all_xyz if "ZLF" in p["product_id"] and "XYZLN" in p["product_id"]]
    # Deduplicate by SCLK timestamp
    seen = set()
    unique = []
    for p in left_lin:
        parts = p["product_id"].split("_")
        sclk = parts[2] if len(parts) > 2 else p["product_id"]
        if sclk not in seen:
            seen.add(sclk)
            unique.append(p)
    return unique


def get_matching_ras(sol: int, xyz_product: dict) -> dict | None:
    """Find the RAS texture matching an XYZ product by SCLK timestamp."""
    ras_products = search_mastcamz_products(sol, "RAS")
    if not ras_products:
        return None

    parts = xyz_product["product_id"].split("_")
    xyz_sclk = parts[2] if len(parts) > 2 else ""

    # Prefer left linearized RAS with same SCLK
    for rp in ras_products:
        if xyz_sclk in rp["product_id"] and "ZLF" in rp["product_id"] and "RASLN" in rp["product_id"]:
            return rp

    # Fallback: any RAS with same SCLK
    for rp in ras_products:
        if xyz_sclk in rp["product_id"]:
            return rp

    return None


def process_sol(sol: int, validate_jezero: bool = True) -> dict:
    """Process all Mastcam-Z frames for a sol.

    Returns metadata dict with per-frame and combined results.
    """
    sol_dir = OUTPUT_DIR / f"sol{sol:05d}"
    sol_dir.mkdir(parents=True, exist_ok=True)

    products = get_left_linearized_products(sol)
    if not products:
        raise ValueError(f"No left linearized XYZ products for sol {sol}")

    print(f"  Found {len(products)} unique XYZ frames")

    init_spice(sol)

    frames = []
    all_lon = []
    all_lat = []
    all_px = []
    all_py = []
    all_frame_idx = []

    for i, xyz_prod in enumerate(products):
        frame_id = xyz_prod["product_id"]
        print(f"  [{i+1}/{len(products)}] {frame_id}")

        # Download XYZ
        xyz_files = download_mastcamz_product(xyz_prod)
        if "data_path" not in xyz_files:
            print(f"    SKIP: download failed")
            continue

        # Download matching RAS
        ras_prod = get_matching_ras(sol, xyz_prod)
        ras_path = None
        if ras_prod:
            ras_files = download_mastcamz_product(ras_prod)
            ras_path = ras_files.get("data_path")

        # SPICE transform
        try:
            result = process_mastcamz_xyz(
                data_path=xyz_files["data_path"],
                label_path=xyz_files.get("label_path"),
            )
        except Exception as e:
            print(f"    SKIP: {e}")
            continue

        lon, lat = result["lon"], result["lat"]
        valid = ~np.isnan(lon)
        valid_count = int(np.sum(valid))

        if valid_count == 0:
            print(f"    SKIP: no valid points")
            continue

        # Validate: must be in Jezero area
        if validate_jezero:
            lon_med = np.nanmedian(lon)
            lat_med = np.nanmedian(lat)
            if not (77.0 < lon_med < 78.0 and 18.0 < lat_med < 19.0):
                print(f"    SKIP: coordinates outside Jezero (lon={lon_med:.2f}, lat={lat_med:.2f})")
                continue

        # Save per-frame lon/lat
        frame_npz = sol_dir / f"frame_{i:03d}_lonlat.npz"
        np.savez_compressed(str(frame_npz), lon=lon, lat=lat)

        # Collect valid points for combined map
        ys, xs = np.where(valid)
        all_lon.append(lon[valid])
        all_lat.append(lat[valid])
        all_px.append(xs)
        all_py.append(ys)
        all_frame_idx.append(np.full(valid_count, i, dtype=np.int32))

        frames.append({
            "index": i,
            "product_id": frame_id,
            "ras_product_id": ras_prod["product_id"] if ras_prod else None,
            "valid_points": valid_count,
            "lon_range": [float(np.nanmin(lon)), float(np.nanmax(lon))],
            "lat_range": [float(np.nanmin(lat)), float(np.nanmax(lat))],
            "width": int(lon.shape[1]),
            "height": int(lon.shape[0]),
        })

    cleanup_spice()

    if not frames:
        raise ValueError(f"No valid frames processed for sol {sol}")

    # Build combined point cloud
    combined_lon = np.concatenate(all_lon)
    combined_lat = np.concatenate(all_lat)
    combined_px = np.concatenate(all_px)
    combined_py = np.concatenate(all_py)
    combined_frame = np.concatenate(all_frame_idx)

    combined_npz = sol_dir / "combined_lonlat.npz"
    np.savez_compressed(
        str(combined_npz),
        lon=combined_lon,
        lat=combined_lat,
        pixel_x=combined_px,
        pixel_y=combined_py,
        frame_idx=combined_frame,
    )

    total_valid = int(combined_lon.size)
    lon_span_m = (combined_lon.max() - combined_lon.min()) * 59274 * np.cos(np.radians(18.44))
    lat_span_m = (combined_lat.max() - combined_lat.min()) * 59274

    result_meta = {
        "sol": sol,
        "n_frames": len(frames),
        "total_valid_points": total_valid,
        "combined_lon_range": [float(combined_lon.min()), float(combined_lon.max())],
        "combined_lat_range": [float(combined_lat.min()), float(combined_lat.max())],
        "coverage_m": [round(lon_span_m, 1), round(lat_span_m, 1)],
        "frames": frames,
    }

    with open(sol_dir / "metadata.json", "w") as f:
        json.dump(result_meta, f, indent=2)

    print(f"  Combined: {total_valid:,} points, {len(frames)} frames")
    print(f"  Coverage: {lon_span_m:.1f}m x {lat_span_m:.1f}m")

    return result_meta


def main():
    parser = argparse.ArgumentParser(description="Batch Mastcam-Z coregistration")
    parser.add_argument("sols", nargs="*", type=int, help="Specific sol numbers (default: all)")
    parser.add_argument("--results-file", default="batch_results.json",
                        help="Output results filename (default: batch_results.json)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.sols:
        sols = args.sols
    else:
        sols = list_available_sols("stereo")

    print(f"Processing {len(sols)} sols (all frames per sol)")

    results = {}
    results_path = OUTPUT_DIR / args.results_file

    t0 = time.time()
    success = 0
    failed = 0

    for i, sol in enumerate(sols, 1):
        elapsed = time.time() - t0
        if success > 0:
            avg = elapsed / success
            eta = avg * (len(sols) - i)
            print(f"\n[{i}/{len(sols)}] Sol {sol}  (elapsed: {elapsed/60:.1f}m, ETA: {eta/60:.1f}m)")
        else:
            print(f"\n[{i}/{len(sols)}] Sol {sol}")

        # Skip if already done
        sol_dir = OUTPUT_DIR / f"sol{sol:05d}"
        if (sol_dir / "combined_lonlat.npz").exists():
            print(f"  [skip] Already processed")
            if (sol_dir / "metadata.json").exists():
                with open(sol_dir / "metadata.json") as f:
                    results[str(sol)] = json.load(f)
            success += 1
            continue

        try:
            meta = process_sol(sol)
            results[str(sol)] = meta
            success += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            failed += 1
            cleanup_spice()

        # Save results incrementally
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

    total_time = time.time() - t0
    print(f"\n{'='*60}")
    print(f"BATCH COMPLETE!")
    print(f"  Success: {success}")
    print(f"  Failed:  {failed}")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"  Results: {results_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
