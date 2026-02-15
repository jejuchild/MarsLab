"""
Batch CNN mineral classification + quickview generation for all TRR3 observations.
Auto-discovers obs_ids from mineral_cnn_data/ directory.
Runs directly via Python (no HTTP overhead), processes sequentially.

Usage:
    python batch_classify.py                       # All observations
    python batch_classify.py --quickview-only      # Just generate quickviews (skip CNN)
    python batch_classify.py --no-quickview        # CNN only, no quickview
    python batch_classify.py --limit 20            # Process at most 20
"""
import sys
import os
import re
import argparse
import asyncio
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.mineral_cnn.pipeline import run_classification, has_cached_result
from api.mineral_cnn.acquire import _generate_quickview
from api.mineral_cnn.data_loader import resolve_trr_files
from api.mineral_cnn.constants import TRR_DATA_DIR

import numpy as np


def _has_l_sensor(obs_dir: str) -> bool:
    """Check if observation has L-sensor (IR) TRR3 data (needed for CNN).

    PDS filename convention: xxx_ifNNNl_trr3.img = L-sensor, xxx_ifNNNs_trr3.img = S-sensor.
    """
    for f in os.listdir(obs_dir):
        fl = f.lower()
        if fl.endswith("_trr3.img") and re.search(r"_if\d+l_", fl):
            return True
    return False


def discover_obs_ids(l_sensor_only: bool = False) -> tuple[list[str], int]:
    """Scan mineral_cnn_data/ for observation directories with TRR3+DDR files.

    Returns (obs_ids, skipped_s_sensor_count).
    """
    obs_ids = []
    skipped = 0
    if not os.path.isdir(TRR_DATA_DIR):
        return obs_ids, 0

    for entry in sorted(os.listdir(TRR_DATA_DIR)):
        d = os.path.join(TRR_DATA_DIR, entry)
        if not os.path.isdir(d):
            continue
        files = os.listdir(d)
        has_trr3 = any(f.upper().endswith("_TRR3.IMG") for f in files)
        has_ddr = any(f.upper().endswith("_DDR1.IMG") for f in files)
        if has_trr3 and has_ddr:
            if l_sensor_only and not _has_l_sensor(d):
                skipped += 1
                continue
            obs_ids.append(entry)

    return obs_ids, skipped


def generate_quickview(obs_id: str) -> bool:
    """Generate quickview PNG for an observation. Returns True on success."""
    try:
        files = resolve_trr_files(obs_id)
        obs_dir = os.path.dirname(files["trr_img"])
        cache_path = os.path.join(obs_dir, "quickview.png")
        if os.path.exists(cache_path):
            return True  # Already exists
        result = _generate_quickview(obs_dir, files["trr_img"], files["trr_lbl"])
        return bool(result)
    except Exception as e:
        print(f"    Quickview error: {e}")
        return False


async def run_classify(obs_id: str) -> bool:
    """Run CNN classification for an observation. Returns True on success."""
    try:
        async for event in run_classification(obs_id):
            evt_type = event.get("event", "")
            data = event.get("data", {})

            if evt_type == "status":
                print(f"    {data.get('message', '')}")
            elif evt_type == "progress":
                pct = data.get("percent", 0)
                if pct % 25 < 1:
                    print(f"    JCAT progress: {pct:.0f}%")
            elif evt_type == "complete":
                classified = data.get("classified_pixels", 0)
                valid = data.get("valid_pixels", 0)
                print(f"    Classified {classified}/{valid} pixels")
                return True
            elif evt_type == "cached":
                return True
            elif evt_type == "error":
                print(f"    ERROR: {data.get('error', 'unknown')}")
                return False
        return True
    except Exception as e:
        print(f"    Classification error: {e}")
        return False


async def main():
    parser = argparse.ArgumentParser(description="Batch classify + quickview for TRR3 data")
    parser.add_argument("--quickview-only", action="store_true",
                        help="Only generate quickviews, skip CNN classification")
    parser.add_argument("--no-quickview", action="store_true",
                        help="Skip quickview generation")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max observations to process (0=all)")
    parser.add_argument("--l-sensor-only", action="store_true", default=True,
                        help="Skip S-sensor (VNIR) observations that can't be classified (default: True)")
    parser.add_argument("--all-sensors", action="store_true",
                        help="Include S-sensor observations (CNN will fail on these)")
    args = parser.parse_args()

    l_only = args.l_sensor_only and not args.all_sensors
    obs_ids, skipped_s = discover_obs_ids(l_sensor_only=l_only)
    if not obs_ids:
        print("No TRR3 observations found in mineral_cnn_data/")
        if skipped_s:
            print(f"  ({skipped_s} S-sensor/VNIR observations skipped — CNN requires L-sensor/IR data)")
        return

    if args.limit > 0:
        obs_ids = obs_ids[:args.limit]

    total = len(obs_ids)
    do_cnn = not args.quickview_only
    do_qv = not args.no_quickview

    print(f"=== Batch Processing: {total} observations ===")
    if skipped_s:
        print(f"  Skipped: {skipped_s} S-sensor/VNIR observations (no IR data for CNN)")
    print(f"  CNN classification: {'YES' if do_cnn else 'SKIP'}")
    print(f"  Quickview generation: {'YES' if do_qv else 'SKIP'}")
    print()

    cnn_ok, cnn_skip, cnn_fail = 0, 0, 0
    qv_ok, qv_skip, qv_fail = 0, 0, 0
    t_start = time.time()

    for i, obs_id in enumerate(obs_ids, 1):
        print(f"[{i}/{total}] {obs_id}")
        t0 = time.time()

        # Quickview
        if do_qv:
            qv_cache = os.path.join(TRR_DATA_DIR, obs_id, "quickview.png")
            if os.path.exists(qv_cache):
                qv_skip += 1
            elif generate_quickview(obs_id):
                qv_ok += 1
                print(f"    Quickview: generated")
            else:
                qv_fail += 1
                print(f"    Quickview: FAILED")

        # CNN classification
        if do_cnn:
            if has_cached_result(obs_id):
                cnn_skip += 1
                print(f"    CNN: cached")
            else:
                t_cnn = time.time()
                if await run_classify(obs_id):
                    cnn_ok += 1
                    print(f"    CNN: done ({time.time() - t_cnn:.1f}s)")
                else:
                    cnn_fail += 1
                    print(f"    CNN: FAILED")

        elapsed = time.time() - t0
        if elapsed > 1:
            print(f"    Total: {elapsed:.1f}s")

    elapsed_total = time.time() - t_start
    print(f"\n=== Done ({elapsed_total:.0f}s) ===")
    if do_qv:
        print(f"Quickview: {qv_ok} generated, {qv_skip} cached, {qv_fail} failed")
    if do_cnn:
        print(f"CNN:       {cnn_ok} classified, {cnn_skip} cached, {cnn_fail} failed")


if __name__ == "__main__":
    asyncio.run(main())
