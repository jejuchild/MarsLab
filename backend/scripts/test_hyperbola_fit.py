#!/usr/bin/env python3
"""
Test script for hyperbola fitting.

Loads a SHARAD_HIGHRES product, auto-detects diffraction apexes,
runs hyperbola fit on the best candidate, and saves overlay PNG.

Usage:
    cd backend
    python -m scripts.test_hyperbola_fit [PRODUCT_ID]

If no product ID given, uses the first available local product.
"""

import json
import os
import sys

# Add backend to path
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_DIR)


def find_first_product() -> str:
    """Find first available SHARAD_HIGHRES product."""
    data_dir = os.path.join(BACKEND_DIR, "sharad_highres_data")
    if not os.path.isdir(data_dir):
        print(f"ERROR: {data_dir} not found")
        sys.exit(1)

    for fname in os.listdir(data_dir):
        if fname.lower().endswith(".dat"):
            pid = fname.replace(".dat", "").replace(".DAT", "")
            print(f"Found product: {pid}")
            return pid

    print("ERROR: No .dat files found in sharad_highres_data/")
    sys.exit(1)


def main():
    product_id = sys.argv[1] if len(sys.argv) > 1 else find_first_product()
    print(f"\n{'='*60}")
    print(f"  Hyperbola Fit Test: {product_id}")
    print(f"{'='*60}\n")

    # Step 1: Auto-detect apex candidates
    print("[1/3] Auto-detecting diffraction apex candidates...")
    from analysis.ice_evidence.hyperbola_fit import auto_detect_apexes
    apexes = auto_detect_apexes(product_id, n_candidates=3)

    if not apexes:
        print("  No diffraction apexes detected. Try a different product.")
        return

    for i, apex in enumerate(apexes):
        print(f"  Candidate {i+1}: trace={apex['trace']}, bin={apex['bin']}, energy={apex['energy']:.2f}")

    # Step 2: Fit hyperbola on best candidate
    best = apexes[0]
    print(f"\n[2/3] Fitting hyperbola at trace={best['trace']}, bin={best['bin']}...")

    from analysis.ice_evidence.models import HyperbolaFitRequest
    from analysis.ice_evidence.hyperbola_fit import fit_hyperbola

    req = HyperbolaFitRequest(
        product_id=product_id,
        apex_trace=best["trace"],
        apex_bin=best["bin"],
        roi_traces=80,
        roi_bins=100,
    )

    result = fit_hyperbola(req)

    print(f"\n  Results:")
    print(f"  ─────────────────────────────")
    print(f"  Velocity:     {result.v_mps:.0f} m/s ({result.v_mps/1e6:.3f} Mm/s)")
    print(f"  εr:           {result.epsr:.3f}")
    print(f"  εr CI95:      [{result.epsr_ci95[0]:.3f}, {result.epsr_ci95[1]:.3f}]")
    print(f"  Depth:        {result.depth_m:.1f} m")
    print(f"  Depth CI95:   [{result.depth_ci95[0]:.1f}, {result.depth_ci95[1]:.1f}]")
    print(f"  SNR:          {result.quality.snr:.2f}")
    print(f"  Residual:     {result.quality.residual:.2e} s")
    print(f"  Support:      {result.quality.support_traces} traces")
    print(f"  Flags:        {', '.join(result.flags)}")
    print(f"  Overlay pts:  {len(result.overlay_polyline)}")

    # Step 3: Save overlay PNG
    print(f"\n[3/3] Saving overlay PNG...")
    from analysis.ice_evidence.io import save_hyperbola_overlay_png, save_hyperbola_fit
    from api.sharad_highres_router import _get_power
    import numpy as np

    power, _ = _get_power(product_id)
    png_path = save_hyperbola_overlay_png(
        product_id, power, result,
        best["trace"], best["bin"], 80, 100,
    )

    json_path = save_hyperbola_fit(product_id, result)

    print(f"\n  JSON saved: {json_path}")
    if png_path:
        print(f"  PNG saved:  {png_path}")
    else:
        print("  PNG save failed (PIL may not be available)")

    print(f"\n{'='*60}")
    print(f"  DONE — εr = {result.epsr:.3f}")
    if "ICE_CONSISTENT" in result.flags:
        print(f"  → ICE-CONSISTENT dielectric (evidence supports water ice)")
    elif result.epsr < 2.7:
        print(f"  → Below ice range (very porous or low-density)")
    elif result.epsr > 3.4:
        print(f"  → Above ice range (denser material)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
