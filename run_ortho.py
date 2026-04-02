#!/usr/bin/env python3
"""Generate Mastcam-Z orthoimages for all processed sols.

Projects RAS textures onto HiRISE DTM grid using SPICE-derived lon/lat.
Runs independently of run_batch.py — processes whatever sols are already done.
"""

import sys
import json
import time
from pathlib import Path

from coregister.hirise_dtm import HiRISEDTM
from coregister.ortho import generate_ortho
from coregister.config import OUTPUT_DIR, PDS_CACHE

HIRISE_DTM_PATH = PDS_CACHE / "hirise" / "DTEEC_048842_1985_048908_1985_U01.IMG"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load DTM if available
    dtm = None
    if HIRISE_DTM_PATH.exists():
        print("Loading HiRISE DTM...")
        dtm = HiRISEDTM(HIRISE_DTM_PATH)
        print(f"  Resolution: {dtm.resolution_m:.2f} m/px")
        print(f"  Bounds: {dtm.bounds}")
    else:
        print("No HiRISE DTM found, using self-generated grid")

    # Find all sols with combined data
    if len(sys.argv) > 1:
        sols = [int(s) for s in sys.argv[1:]]
    else:
        sols = []
        for sol_dir in sorted(OUTPUT_DIR.glob("sol?????")):
            if (sol_dir / "combined_lonlat.npz").exists():
                sol = int(sol_dir.name.replace("sol", ""))
                # Skip if ortho already exists
                if not (OUTPUT_DIR / f"sol{sol:05d}_ortho.png").exists():
                    sols.append(sol)

    print(f"\nGenerating orthoimages for {len(sols)} sols")

    t0 = time.time()
    success = 0
    failed = 0

    for i, sol in enumerate(sols, 1):
        print(f"\n[{i}/{len(sols)}] Sol {sol}")
        try:
            result = generate_ortho(sol, dtm=dtm)
            if result:
                success += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    total_time = time.time() - t0
    print(f"\n{'='*60}")
    print(f"ORTHO GENERATION COMPLETE!")
    print(f"  Success: {success}")
    print(f"  Failed:  {failed}")
    print(f"  Time: {total_time/60:.1f} minutes")
    print(f"{'='*60}")

    if dtm:
        dtm.close()


if __name__ == "__main__":
    main()
