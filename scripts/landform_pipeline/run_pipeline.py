#!/usr/bin/env python3
"""
Mars Landform Classification Pipeline — Full Orchestrator.

Runs all steps in sequence:
  1. Tile HiRISE browse images + extract DINOv2 embeddings
  2. Extract MOLA geomorphometric features per tile
  3. K-Means clustering + visualization

Usage:
    # Full pipeline (all downloaded images)
    python scripts/landform_pipeline/run_pipeline.py

    # Labeled images only (faster, ~1-2h)
    python scripts/landform_pipeline/run_pipeline.py --labeled-only

    # Quick test (first 20 images)
    python scripts/landform_pipeline/run_pipeline.py --limit 20

    # Custom output directory
    python scripts/landform_pipeline/run_pipeline.py --output-dir /path/to/output
"""

import argparse
import os
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))


def run_step(name: str, cmd: list[str], env: dict | None = None):
    """Run a pipeline step, streaming output. Exits on failure."""
    print(f"\n{'='*60}")
    print(f"  STEP: {name}")
    print(f"{'='*60}\n")
    
    start = time.time()
    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        env={**os.environ, **(env or {})},
    )
    elapsed = time.time() - start
    
    if result.returncode != 0:
        print(f"\n✗ FAILED: {name} (exit code {result.returncode})")
        print(f"  Command: {' '.join(cmd)}")
        sys.exit(1)
    
    print(f"\n✓ DONE: {name} ({elapsed:.0f}s)")
    return elapsed


def main():
    parser = argparse.ArgumentParser(
        description="Mars Landform Classification Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--image-dirs",
        type=str,
        default="Data/HiRISE/midlat_browse,arcadia_hirise/jpeg",
        help="Comma-separated image directories",
    )
    parser.add_argument(
        "--metadata-json",
        type=str,
        default="Data/HiRISE/midlat_metadata.json",
        help="Path to metadata JSON",
    )
    parser.add_argument(
        "--dem-path",
        type=str,
        default="Mars_HRSC_MOLA_BlendDEM_Global_200mp_v2.tif",
        help="Path to MOLA DEM",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="Data/HiRISE/pipeline_output",
        help="Pipeline output directory",
    )
    parser.add_argument("--labeled-only", action="store_true", help="Process only labeled images (LDA/CCF/LVF/GLF)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images (for testing)")
    parser.add_argument("--batch-size", type=int, default=64, help="DINOv2 batch size")
    parser.add_argument("--n-clusters", type=int, default=40, help="Number of K-Means clusters")
    parser.add_argument("--mola-weight", type=float, default=1.0, help="MOLA feature weight in clustering")
    parser.add_argument("--skip-umap", action="store_true", help="Skip UMAP/t-SNE visualization")
    parser.add_argument("--skip-mola", action="store_true", help="Skip MOLA feature extraction (DINO-only clustering)")

    args = parser.parse_args()
    
    os.makedirs(os.path.join(PROJECT_ROOT, args.output_dir), exist_ok=True)
    
    total_start = time.time()
    timings = {}

    # ── Step 1: Tile + DINOv2 Extract ─────────────────────────────
    cmd = [
        sys.executable,
        os.path.join(SCRIPT_DIR, "tile_and_extract.py"),
        "--image-dirs", args.image_dirs,
        "--metadata-json", args.metadata_json,
        "--output-dir", args.output_dir,
        "--batch-size", str(args.batch_size),
    ]
    if args.labeled_only:
        cmd.append("--labeled-only")
    if args.limit:
        cmd.extend(["--limit", str(args.limit)])
    
    timings["tile_and_extract"] = run_step("Tile images + Extract DINOv2 embeddings", cmd)

    # ── Step 2: MOLA Feature Extraction ───────────────────────────
    if not args.skip_mola:
        cmd = [
            sys.executable,
            os.path.join(SCRIPT_DIR, "extract_mola_features.py"),
            "--tile-metadata", os.path.join(args.output_dir, "tile_metadata.csv"),
            "--dem-path", args.dem_path,
            "--output-dir", args.output_dir,
        ]
        timings["mola_features"] = run_step("Extract MOLA geomorphometric features", cmd)

    # ── Step 3: Cluster + Visualize ───────────────────────────────
    cmd = [
        sys.executable,
        os.path.join(SCRIPT_DIR, "cluster_and_visualize.py"),
        "--embeddings", os.path.join(args.output_dir, "embeddings.npy"),
        "--tile-metadata", os.path.join(args.output_dir, "tile_metadata.csv"),
        "--image-dirs", args.image_dirs,
        "--output-dir", os.path.join(args.output_dir, "clusters"),
        "--n-clusters", str(args.n_clusters),
        "--mola-weight", str(args.mola_weight),
    ]
    if not args.skip_mola:
        cmd.extend(["--mola-features", os.path.join(args.output_dir, "mola_features.npy")])
    if args.skip_umap:
        cmd.append("--skip-umap")
    
    timings["cluster"] = run_step("K-Means clustering + visualization", cmd)

    # ── Summary ───────────────────────────────────────────────────
    total_elapsed = time.time() - total_start
    
    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"  Total time: {total_elapsed/3600:.1f}h ({total_elapsed:.0f}s)")
    for step, t in timings.items():
        print(f"    {step}: {t:.0f}s")
    print(f"\n  Output: {os.path.join(PROJECT_ROOT, args.output_dir)}")
    print(f"  Clusters: {os.path.join(PROJECT_ROOT, args.output_dir, 'clusters')}")
    print(f"\n  Next steps:")
    print(f"    1. Open clusters/cluster_*/grid.png to see tile patterns")
    print(f"    2. Check clusters/cluster_summary.json for enrichment scores")
    print(f"    3. Assign class labels to enriched clusters")
    print(f"    4. Run fine-tuning with pseudo-labels")


if __name__ == "__main__":
    main()
