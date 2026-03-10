#!/usr/bin/env python3
"""CLI for SCT Temporal Change Detection Pipeline."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="SCT Scarp Retreat Measurement Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-find temporal pairs for a known SCT region
  python -m backend.analysis.sct_temporal.cli --product-id ESP_016142_2240

  # Process a specific pair
  python -m backend.analysis.sct_temporal.cli --pair ESP_016142_2240 ESP_025000_2240

  # Custom parameters
  python -m backend.analysis.sct_temporal.cli --product-id ESP_016142_2240 \\
    --chip-size 128 --max-emission-diff 3.0 --min-time-gap 600
        """,
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--product-id", help="HiRISE observation ID for auto pair search")
    mode.add_argument("--pair", nargs=2, metavar=("PRODUCT_A", "PRODUCT_B"),
                       help="Specific pair of product IDs")

    parser.add_argument("--output", "-o", default="results/sct_retreat",
                        help="Output directory (default: results/sct_retreat)")
    parser.add_argument("--max-pairs", type=int, default=5,
                        help="Max pairs to process in auto mode (default: 5)")
    parser.add_argument("--chip-size", type=int, default=64,
                        help="Correlation window size in pixels (default: 64)")
    parser.add_argument("--step-size", type=int, default=16,
                        help="Window step size in pixels (default: 16)")
    parser.add_argument("--upsample", type=int, default=100,
                        help="Sub-pixel upsample factor (default: 100)")
    parser.add_argument("--max-emission-diff", type=float, default=5.0,
                        help="Max emission angle difference for pairs (default: 5.0°)")
    parser.add_argument("--min-time-gap", type=float, default=300,
                        help="Min time gap between pairs in days (default: 300)")
    parser.add_argument("--time-gap-days", type=float, default=None,
                        help="Explicit time gap for --pair mode (days)")
    parser.add_argument("--search-radius", type=float, default=30.0,
                        help="ODE search radius in km (default: 30)")
    parser.add_argument("--no-frost-filter", action="store_true",
                        help="Disable CO2 frost-free season filter")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    from .pipeline import SCTTemporalPipeline, PipelineConfig

    config = PipelineConfig(
        chip_size=args.chip_size,
        step_size=args.step_size,
        upsample_factor=args.upsample,
        max_emission_diff=args.max_emission_diff,
        min_time_gap_days=args.min_time_gap,
        frost_free_only=not args.no_frost_filter,
        search_radius_km=args.search_radius,
    )

    pipeline = SCTTemporalPipeline(config)
    output_dir = Path(args.output)

    if args.product_id:
        results = asyncio.run(
            pipeline.run(args.product_id, output_dir, max_pairs=args.max_pairs)
        )
    else:
        result = asyncio.run(
            pipeline.run_pair(
                args.pair[0], args.pair[1], output_dir,
                time_gap_days=args.time_gap_days,
            )
        )
        results = [result]

    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    print(f"\n{'='*60}")
    print(f"SCT Temporal Change Detection — Results")
    print(f"{'='*60}")
    print(f"Pairs processed: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")

    for r in successful:
        retreat = r.retreat
        if retreat is None:
            continue
        print(f"\n  {r.pair.product_a.observation_id} → {r.pair.product_b.observation_id}")
        print(f"  Time gap: {r.pair.time_gap_mars_years:.1f} Mars years")
        print(f"  Mean retreat: {retreat.mean_retreat_rate_m_per_yr:.3f} m/Mars yr")
        print(f"  Max retreat:  {retreat.max_retreat_rate_m_per_yr:.3f} m/Mars yr")
        print(f"  Scarp segments: {len(retreat.segments)}")
        print(f"  Valid measurements: {retreat.valid_measurement_pct:.1f}%")
        print(f"  Output: {r.output_dir}")

    for r in failed:
        print(f"\n  FAILED: {r.pair.product_a.observation_id} → {r.pair.product_b.observation_id}")
        print(f"  Error: {r.error}")

    print(f"\nResults saved to: {output_dir}")
    return 0 if all(r.success for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
