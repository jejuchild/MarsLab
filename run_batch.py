#!/usr/bin/env python3
"""Batch process all available Mastcam-Z sols through the coregistration pipeline.

Phase 1: XYZ + SPICE → lon/lat + georeferenced overlay (no DTM)
Phase 2: Apply DTM draping to all results (run separately after)
"""

import sys
import time
import json
import traceback
from pathlib import Path

from coregister.pds_fetch import list_available_sols, search_mastcamz_products, download_mastcamz_product
from coregister.spice_setup import init_spice, cleanup_spice
from coregister.mastcam_xyz import process_mastcamz_xyz
from coregister.drape import drape_simple_overlay, load_mastcamz_texture, _read_pds_img_texture
from coregister.config import OUTPUT_DIR, PDS_CACHE

import numpy as np


def process_sol(sol: int, results: dict) -> bool:
    """Process a single sol. Returns True on success."""
    output_path = OUTPUT_DIR / f"sol{sol:05d}_coregistered.tif"
    npz_path = OUTPUT_DIR / f"sol{sol:05d}_lonlat.npz"

    if output_path.exists() and npz_path.exists():
        print(f"  [skip] Already done")
        return True

    # Search XYZ products
    xyz_products = search_mastcamz_products(sol, "XYZ")
    if not xyz_products:
        print(f"  No XYZ products")
        return False

    # Download first XYZ
    xyz_product = xyz_products[0]
    print(f"  XYZ: {xyz_product['product_id']}")
    xyz_files = download_mastcamz_product(xyz_product)
    if "data_path" not in xyz_files:
        print(f"  FAIL: XYZ download failed")
        return False

    # Download matching RAS
    ras_products = search_mastcamz_products(sol, "RAS")
    ras_path = None
    if ras_products:
        # Match by SCLK timestamp (chars 10-23 of product ID)
        xyz_sclk = xyz_product["product_id"][10:23]
        matching = [p for p in ras_products if xyz_sclk in p["product_id"]]
        ras_to_dl = matching[0] if matching else ras_products[0]
        ras_files = download_mastcamz_product(ras_to_dl)
        ras_path = ras_files.get("data_path")

    # Init SPICE (reuse if same sol range)
    init_spice(sol)

    # Process XYZ
    result = process_mastcamz_xyz(
        data_path=xyz_files["data_path"],
        label_path=xyz_files.get("label_path"),
    )

    lon, lat = result["lon"], result["lat"]

    # Save lon/lat for later DTM draping
    np.savez_compressed(str(npz_path), lon=lon, lat=lat)

    # Load texture
    if ras_path and ras_path.exists():
        try:
            texture = _read_pds_img_texture(ras_path)
            if texture.shape[:2] != lon.shape:
                from PIL import Image as PILImage
                texture = np.array(
                    PILImage.fromarray(texture).resize((lon.shape[1], lon.shape[0]))
                )
        except Exception as e:
            print(f"  Texture load failed ({e}), using grayscale")
            texture = None
    else:
        texture = None

    if texture is None:
        mag = np.sqrt(np.nansum(result["xyz_rover"] ** 2, axis=-1))
        denom = np.nanmax(mag) - np.nanmin(mag)
        if denom == 0:
            denom = 1.0
        mag_norm = np.clip((mag - np.nanmin(mag)) / denom * 255, 0, 255).astype(np.uint8)
        texture = np.stack([mag_norm] * 3, axis=-1)

    # Generate overlay
    drape_simple_overlay(lon, lat, texture, output_path)

    cleanup_spice()

    # Record result
    valid = ~np.isnan(lon)
    results[str(sol)] = {
        "sol": sol,
        "product_id": xyz_product["product_id"],
        "valid_points": int(np.sum(valid)),
        "lon_range": [float(np.nanmin(lon)), float(np.nanmax(lon))],
        "lat_range": [float(np.nanmin(lat)), float(np.nanmax(lat))],
        "output": str(output_path),
    }

    return True


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sols = list_available_sols("stereo")
    print(f"Found {len(sols)} sols with stereo data")
    print(f"Sols: {sols}\n")

    results = {}
    results_path = OUTPUT_DIR / "batch_results.json"

    # Load existing results
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)

    success = 0
    failed = 0
    skipped = 0
    t0 = time.time()

    for i, sol in enumerate(sols, 1):
        elapsed = time.time() - t0
        if success > 0:
            avg = elapsed / success
            remaining = avg * (len(sols) - i)
            eta_min = remaining / 60
            print(f"\n[{i}/{len(sols)}] Sol {sol}  (elapsed: {elapsed/60:.1f}m, ETA: {eta_min:.1f}m)")
        else:
            print(f"\n[{i}/{len(sols)}] Sol {sol}")

        try:
            if str(sol) in results:
                print(f"  [skip] Already in results")
                skipped += 1
                continue

            ok = process_sol(sol, results)
            if ok:
                success += 1
            else:
                failed += 1

            # Save results incrementally
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)

        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            failed += 1
            cleanup_spice()

    total_time = time.time() - t0
    print(f"\n{'='*60}")
    print(f"BATCH COMPLETE!")
    print(f"  Success: {success}")
    print(f"  Failed:  {failed}")
    print(f"  Skipped: {skipped}")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"  Results: {results_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
