#!/usr/bin/env python3
"""Phase 2: Apply HiRISE DTM draping to all processed sol results.

Reads lon/lat .npz files from Phase 1 and drapes textures onto the DTM grid.
Run this after run_batch.py completes and DTM is downloaded.
"""

import json
import time
import traceback
from pathlib import Path

import numpy as np

from coregister.config import OUTPUT_DIR, PDS_CACHE
from coregister.hirise_dtm import HiRISEDTM
from coregister.drape import drape_on_dtm, _read_pds_img_texture

DTM_PATH = PDS_CACHE / "hirise" / "DTEEC_048842_1985_048908_1985_U01.IMG"


def main():
    # Load batch results from Phase 1
    results_path = OUTPUT_DIR / "batch_results.json"
    if not results_path.exists():
        print("ERROR: No batch_results.json found. Run run_batch.py first.")
        return

    with open(results_path) as f:
        results = json.load(f)

    print(f"Found {len(results)} processed sols")

    # Load DTM
    print(f"\nLoading HiRISE DTM: {DTM_PATH.name}")
    dtm = HiRISEDTM(DTM_PATH)
    dtm_bounds = dtm.bounds
    print(f"  DTM bounds: lon=[{dtm_bounds['min_lon']:.4f}, {dtm_bounds['max_lon']:.4f}], "
          f"lat=[{dtm_bounds['min_lat']:.4f}, {dtm_bounds['max_lat']:.4f}]")

    # Filter sols that fall within DTM coverage
    valid_sols = {}
    for sol_str, info in results.items():
        lon_range = info.get("lon_range", [0, 0])
        lat_range = info.get("lat_range", [0, 0])

        # Check if Mastcam-Z coverage overlaps DTM (with generous margin)
        margin = 0.05  # ~3 km margin
        in_lon = (lon_range[0] >= dtm_bounds["min_lon"] - margin and
                  lon_range[1] <= dtm_bounds["max_lon"] + margin)
        in_lat = (lat_range[0] >= dtm_bounds["min_lat"] - margin and
                  lat_range[1] <= dtm_bounds["max_lat"] + margin)

        # Also filter out obviously wrong coordinates
        reasonable = (77.3 < lon_range[0] < 77.7 and 18.0 < lat_range[0] < 18.7)

        if in_lon and in_lat and reasonable:
            valid_sols[sol_str] = info

    print(f"Sols within DTM coverage: {len(valid_sols)}/{len(results)}")

    success = 0
    failed = 0
    t0 = time.time()

    for i, (sol_str, info) in enumerate(sorted(valid_sols.items()), 1):
        sol = int(sol_str)
        npz_path = OUTPUT_DIR / f"sol{sol:05d}_lonlat.npz"
        output_path = OUTPUT_DIR / f"sol{sol:05d}_dtm_draped.tif"

        if output_path.exists() and output_path.stat().st_size > 0:
            print(f"[{i}/{len(valid_sols)}] Sol {sol} — already done")
            success += 1
            continue

        if not npz_path.exists():
            print(f"[{i}/{len(valid_sols)}] Sol {sol} — no lon/lat data, skip")
            failed += 1
            continue

        print(f"[{i}/{len(valid_sols)}] Sol {sol}  ({info['product_id']})")

        try:
            # Load lon/lat
            data = np.load(str(npz_path))
            lon, lat = data["lon"], data["lat"]

            # Load texture from cached RAS product
            sol_dir = PDS_CACHE / "mastcamz" / f"sol{sol:05d}"
            ras_files = sorted(sol_dir.glob("*RAS*.IMG"))
            texture = None
            if ras_files:
                try:
                    texture = _read_pds_img_texture(ras_files[0])
                    if texture.shape[:2] != lon.shape:
                        from PIL import Image
                        texture = np.array(
                            Image.fromarray(texture).resize((lon.shape[1], lon.shape[0]))
                        )
                except Exception as e:
                    print(f"  Texture failed: {e}")

            if texture is None:
                # Grayscale fallback
                r = np.sqrt(lon**2 + lat**2)
                r_norm = np.clip((r - np.nanmin(r)) / max(np.nanmax(r) - np.nanmin(r), 1) * 255, 0, 255)
                texture = np.stack([r_norm.astype(np.uint8)] * 3, axis=-1)

            # Drape onto DTM
            drape_on_dtm(lon, lat, texture, dtm, output_path)
            success += 1

        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            failed += 1

    dtm.close()

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"PHASE 2 COMPLETE!")
    print(f"  Success: {success}")
    print(f"  Failed:  {failed}")
    print(f"  Time: {elapsed/60:.1f} minutes")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
