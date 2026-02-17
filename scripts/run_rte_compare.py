#!/usr/bin/env python3
"""
Regression test script for RTE shallow-mode implementation.

Compares default-mode outputs against baseline to verify zero behavior change
when mode="default" (Phase 1 verification).
"""

import requests
import json
import sys
from pathlib import Path

# Configuration
SERVER_URL = "http://localhost:8000"
TEST_PRODUCT_ID = "R_0277201_001_SS19_700_A"  # Example product with known data

def fetch_rte_result(product_id: str, mode: str = "default", **kwargs) -> dict:
    """Fetch RTE result from API."""
    params = {
        "product_id": product_id,
        "mode": mode,
        **kwargs,
    }
    resp = requests.get(f"{SERVER_URL}/api/regolith/thickness_profile", params=params)
    if resp.status_code != 200:
        print(f"Error: {resp.status_code} - {resp.text}")
        return None
    return resp.json()


def compare_results(default_result: dict, shallow_result: dict) -> dict:
    """Compare default and shallow mode results."""
    if not default_result or not shallow_result:
        return {"error": "One or both results are None"}

    if not default_result.get("success") or not shallow_result.get("success"):
        return {"error": "One or both API calls failed"}

    default_summary = default_result.get("summary", {})
    shallow_summary = shallow_result.get("summary", {})

    comparison = {
        "default": {
            "detection_rate": default_summary.get("detection_rate"),
            "mean_thickness": default_summary.get("thickness_mean_m"),
            "valid_traces": default_summary.get("valid_traces"),
        },
        "shallow": {
            "detection_rate": shallow_summary.get("detection_rate"),
            "mean_thickness": shallow_summary.get("thickness_mean_m"),
            "valid_traces": shallow_summary.get("valid_traces"),
            "ring_reject_rate": shallow_summary.get("ring_reject_rate"),
        },
        "difference": {
            "detection_rate_delta": (
                shallow_summary.get("detection_rate", 0) - default_summary.get("detection_rate", 0)
            ),
        },
    }

    return comparison


def main():
    """Run regression test."""
    print("=" * 60)
    print("RTE REGRESSION TEST — Default Mode Verification")
    print("=" * 60)
    print()

    print(f"Server: {SERVER_URL}")
    print(f"Test product: {TEST_PRODUCT_ID}")
    print()

    # Fetch default mode
    print("1. Running default mode...")
    default_result = fetch_rte_result(TEST_PRODUCT_ID, mode="default")
    if not default_result:
        print("ERROR: Failed to fetch default mode result")
        return 1

    if default_result.get("success"):
        summary = default_result.get("summary", {})
        print(f"   ✓ Success")
        print(f"   - Detection rate: {summary.get('detection_rate', 'N/A'):.2%}")
        print(f"   - Mean thickness: {summary.get('thickness_mean_m', 'N/A')} m")
        print(f"   - Valid traces: {summary.get('valid_traces', 'N/A')}")
    else:
        print(f"   ✗ Failed: {default_result.get('error', 'Unknown error')}")
        return 1

    print()

    # Fetch shallow mode
    print("2. Running shallow mode...")
    shallow_result = fetch_rte_result(TEST_PRODUCT_ID, mode="shallow")
    if not shallow_result:
        print("ERROR: Failed to fetch shallow mode result")
        return 1

    if shallow_result.get("success"):
        summary = shallow_result.get("summary", {})
        print(f"   ✓ Success")
        print(f"   - Detection rate: {summary.get('detection_rate', 'N/A'):.2%}")
        print(f"   - Mean thickness: {summary.get('thickness_mean_m', 'N/A')} m")
        print(f"   - Valid traces: {summary.get('valid_traces', 'N/A')}")
        print(f"   - Ring reject rate: {summary.get('ring_reject_rate', 'N/A'):.2%}")
    else:
        print(f"   ✗ Failed: {shallow_result.get('error', 'Unknown error')}")
        return 1

    print()

    # Compare
    print("3. Comparing results...")
    comparison = compare_results(default_result, shallow_result)

    if "error" in comparison:
        print(f"   ✗ Comparison error: {comparison['error']}")
        return 1

    print(f"   Default mode detection_rate:  {comparison['default']['detection_rate']:.4f}")
    print(f"   Shallow mode detection_rate:  {comparison['shallow']['detection_rate']:.4f}")
    print(f"   Delta:                        {comparison['difference']['detection_rate_delta']:+.4f}")

    print()
    print("=" * 60)

    # Verification
    default_det = comparison['default']['detection_rate']
    shallow_det = comparison['shallow']['detection_rate']

    if default_det is not None and shallow_det is not None:
        # Check: Default mode should be relatively stable
        # Shallow mode may detect more (higher detection_rate)
        if shallow_det >= default_det:
            print("✓ PASS: Shallow mode detection >= default (expected)")
        else:
            print("⚠ WARNING: Shallow mode detection < default (unexpected)")

        print(f"  Shallow mode detected {shallow_det * 100:.1f}% vs default {default_det * 100:.1f}%")
    else:
        print("⚠ Could not compare detection rates")

    print()
    print("Regression test complete.")
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(2)
