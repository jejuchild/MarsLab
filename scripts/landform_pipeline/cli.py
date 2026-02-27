#!/usr/bin/env python3
"""
Mars Landform Classification Pipeline — Unified CLI.

Entry point for all pipeline steps. Run individual steps or the full pipeline.

Usage:
    # Full pipeline (existing images)
    python scripts/landform_pipeline/cli.py run-all

    # Individual steps
    python scripts/landform_pipeline/cli.py download    # download browse JPEGs
    python scripts/landform_pipeline/cli.py metadata    # regenerate metadata JSON
    python scripts/landform_pipeline/cli.py tile        # tile + DINOv2 extract
    python scripts/landform_pipeline/cli.py mola        # MOLA feature extraction
    python scripts/landform_pipeline/cli.py cluster     # K-Means clustering
    python scripts/landform_pipeline/cli.py train       # train classifier
    python scripts/landform_pipeline/cli.py fuse        # Bayesian fusion
    python scripts/landform_pipeline/cli.py predict     # run inference
    python scripts/landform_pipeline/cli.py export      # export GeoJSON

    # Quick test (5 images, labeled only)
    python scripts/landform_pipeline/cli.py run-all --limit 5 --labeled-only

    # Status check
    python scripts/landform_pipeline/cli.py status
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

# Default paths
DEFAULTS = {
    "image_dirs": "Data/HiRISE/midlat_browse,arcadia_hirise/jpeg",
    "metadata_json": "Data/HiRISE/midlat_metadata.json",
    "dem_path": "Mars_HRSC_MOLA_BlendDEM_Global_200mp_v2.tif",
    "output_dir": "Data/HiRISE/pipeline_output",
    "db_path": "backend/data/hirise_ice.db",
}


def run_step(name: str, script: str, extra_args: list[str] | None = None) -> float:
    """Run a pipeline step. Returns elapsed seconds."""
    cmd = [sys.executable, str(SCRIPT_DIR / script)]
    if extra_args:
        cmd.extend(extra_args)

    print(f"\n{'='*60}")
    print(f"  STEP: {name}")
    print(f"  CMD:  {' '.join(cmd)}")
    print(f"{'='*60}\n")

    start = time.time()
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\n✗ FAILED: {name} (exit code {result.returncode})")
        sys.exit(1)

    print(f"\n✓ DONE: {name} ({elapsed:.0f}s)")
    return elapsed


def cmd_download(args: argparse.Namespace) -> None:
    """Download browse JPEGs."""
    extra = ["--concurrency", str(args.concurrency)]
    if args.limit:
        extra.extend(["--limit", str(args.limit)])
    run_step("Download browse JPEGs", "download_browse.py", extra)


def cmd_metadata(args: argparse.Namespace) -> None:
    """Regenerate metadata JSON."""
    run_step("Generate metadata JSON", "generate_metadata.py", [
        "--db-path", str(PROJECT_ROOT / DEFAULTS["db_path"]),
        "--output", str(PROJECT_ROOT / DEFAULTS["metadata_json"]),
    ])


def cmd_tile(args: argparse.Namespace) -> None:
    """Tile images + extract DINOv2 embeddings."""
    extra = [
        "--image-dirs", args.image_dirs,
        "--metadata-json", args.metadata_json,
        "--output-dir", args.output_dir,
        "--batch-size", str(args.batch_size),
    ]
    if args.labeled_only:
        extra.append("--labeled-only")
    if args.limit:
        extra.extend(["--limit", str(args.limit)])
    run_step("Tile images + DINOv2 extraction", "tile_and_extract.py", extra)


def cmd_mola(args: argparse.Namespace) -> None:
    """Extract MOLA geomorphometric features."""
    run_step("MOLA feature extraction", "extract_mola_features.py", [
        "--tile-metadata", os.path.join(args.output_dir, "tile_metadata.csv"),
        "--dem-path", args.dem_path,
        "--output-dir", args.output_dir,
    ])


def cmd_cluster(args: argparse.Namespace) -> None:
    """K-Means clustering + visualization."""
    extra = [
        "--embeddings", os.path.join(args.output_dir, "embeddings.npy"),
        "--mola-features", os.path.join(args.output_dir, "mola_features.npy"),
        "--tile-metadata", os.path.join(args.output_dir, "tile_metadata.csv"),
        "--image-dirs", args.image_dirs,
        "--output-dir", os.path.join(args.output_dir, "clusters"),
        "--n-clusters", str(args.n_clusters),
        "--mola-weight", str(args.mola_weight),
    ]
    if args.skip_umap:
        extra.append("--skip-umap")
    run_step("K-Means clustering", "cluster_and_visualize.py", extra)


def cmd_train(args: argparse.Namespace) -> None:
    """Train supervised classifier."""
    extra = [
        "--cluster-dir", os.path.join(args.output_dir, "clusters"),
        "--embeddings", os.path.join(args.output_dir, "embeddings.npy"),
        "--tile-metadata", os.path.join(args.output_dir, "tile_metadata.csv"),
        "--output-dir", os.path.join(args.output_dir, "classifier"),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
    ]
    if args.use_mola:
        extra.extend(["--use-mola", "--mola-features", os.path.join(args.output_dir, "mola_features.npy")])
    run_step("Train classifier", "train_classifier.py", extra)


def cmd_fuse(args: argparse.Namespace) -> None:
    """Bayesian fusion."""
    run_step("Bayesian fusion", "fusion.py", [
        "--embeddings", os.path.join(args.output_dir, "embeddings.npy"),
        "--mola-features", os.path.join(args.output_dir, "mola_features.npy"),
        "--tile-metadata", os.path.join(args.output_dir, "tile_metadata.csv"),
        "--classifier-model", os.path.join(args.output_dir, "classifier", "best_model.pt"),
        "--cluster-dir", os.path.join(args.output_dir, "clusters"),
        "--output-dir", os.path.join(args.output_dir, "fusion"),
    ])


def cmd_predict(args: argparse.Namespace) -> None:
    """Run inference."""
    extra = [
        "--image-dirs", args.image_dirs,
        "--metadata-json", args.metadata_json,
        "--classifier-model", os.path.join(args.output_dir, "classifier", "best_model.pt"),
        "--fusion-model", os.path.join(args.output_dir, "fusion", "fusion_model.pt"),
        "--dem-path", args.dem_path,
        "--output-dir", os.path.join(args.output_dir, "predictions"),
        "--batch-size", str(args.batch_size),
    ]
    if args.limit:
        extra.extend(["--limit", str(args.limit)])
    if args.save_maps:
        extra.append("--save-maps")
    run_step("Prediction", "predict.py", extra)


def cmd_export(args: argparse.Namespace) -> None:
    """Export GeoJSON."""
    run_step("GeoJSON export", "export_geojson.py", [
        "--predictions-dir", os.path.join(args.output_dir, "predictions"),
        "--output-dir", os.path.join(args.output_dir, "geojson"),
    ])


def cmd_run_all(args: argparse.Namespace) -> None:
    """Run full pipeline end-to-end."""
    total_start = time.time()
    timings: dict[str, float] = {}

    # Step 1: Metadata (quick)
    cmd_metadata(args)
    timings["metadata"] = 0.0

    # Step 2: Tile + DINOv2
    t = time.time()
    cmd_tile(args)
    timings["tile_extract"] = time.time() - t

    # Step 3: MOLA features
    t = time.time()
    cmd_mola(args)
    timings["mola"] = time.time() - t

    # Step 4: Clustering
    t = time.time()
    cmd_cluster(args)
    timings["cluster"] = time.time() - t

    # Step 5: Train classifier
    t = time.time()
    cmd_train(args)
    timings["train"] = time.time() - t

    # Step 6: Fusion
    t = time.time()
    cmd_fuse(args)
    timings["fuse"] = time.time() - t

    # Step 7: Predict
    t = time.time()
    cmd_predict(args)
    timings["predict"] = time.time() - t

    # Step 8: Export GeoJSON
    t = time.time()
    cmd_export(args)
    timings["export"] = time.time() - t

    total = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  FULL PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"  Total: {total/3600:.1f}h ({total:.0f}s)")
    for step, t_val in timings.items():
        print(f"    {step}: {t_val:.0f}s")
    print(f"\n  Output: {os.path.join(str(PROJECT_ROOT), args.output_dir)}")


def cmd_status(args: argparse.Namespace) -> None:
    """Show pipeline status — what exists, what's missing."""
    out = Path(args.output_dir)
    checks = [
        ("Metadata JSON", Path(args.metadata_json)),
        ("Embeddings", out / "embeddings.npy"),
        ("Tile metadata", out / "tile_metadata.csv"),
        ("MOLA features", out / "mola_features.npy"),
        ("Cluster summary", out / "clusters" / "cluster_summary.json"),
        ("Cluster assignments", out / "clusters" / "cluster_assignments.csv"),
        ("Classifier model", out / "classifier" / "best_model.pt"),
        ("Fusion model", out / "fusion" / "fusion_model.pt"),
        ("Fused predictions", out / "fusion" / "fused_predictions.csv"),
        ("GeoJSON (combined)", out / "geojson" / "landforms_all.geojson"),
    ]

    print("\n=== Pipeline Status ===\n")

    # Count images
    image_dirs = [Path(d.strip()) for d in args.image_dirs.split(",") if d.strip()]
    total_images = 0
    for d in image_dirs:
        if d.is_dir():
            total_images += sum(1 for f in d.iterdir() if f.suffix.lower() in {".jpg", ".jpeg"})
    print(f"  Images on disk: {total_images}")

    # Check download status
    db_path = PROJECT_ROOT / DEFAULTS["db_path"]
    if db_path.exists():
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        db_count = conn.execute(
            "SELECT COUNT(*) FROM hirise_ice WHERE quickview_url IS NOT NULL AND quickview_url != ''"
        ).fetchone()[0]
        conn.close()
        print(f"  Images in DB:   {db_count}")
        print(f"  Download:       {total_images}/{db_count} ({100*total_images/db_count:.0f}%)")

    print()
    for name, path in checks:
        exists = path.exists()
        size = ""
        if exists:
            sz = path.stat().st_size
            if sz > 1_000_000:
                size = f" ({sz/1_000_000:.1f} MB)"
            elif sz > 1_000:
                size = f" ({sz/1_000:.1f} KB)"
            else:
                size = f" ({sz} B)"
        status = f"✓{size}" if exists else "✗ missing"
        print(f"  {name:25s} {status}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mars Landform Classification Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Global args
    parser.add_argument("--image-dirs", default=DEFAULTS["image_dirs"], help="Comma-separated image dirs")
    parser.add_argument("--metadata-json", default=DEFAULTS["metadata_json"], help="Metadata JSON path")
    parser.add_argument("--dem-path", default=DEFAULTS["dem_path"], help="MOLA DEM path")
    parser.add_argument("--output-dir", default=DEFAULTS["output_dir"], help="Pipeline output dir")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for DINOv2 / training")
    parser.add_argument("--labeled-only", action="store_true", help="Process only labeled images")
    parser.add_argument("--limit", type=int, default=None, help="Limit images (for testing)")
    parser.add_argument("--n-clusters", type=int, default=40, help="K-Means clusters")
    parser.add_argument("--mola-weight", type=float, default=1.0, help="MOLA weight in clustering")
    parser.add_argument("--skip-umap", action="store_true", help="Skip UMAP/t-SNE scatter")
    parser.add_argument("--use-mola", action="store_true", help="Include MOLA features in classifier")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--concurrency", type=int, default=6, help="Download concurrency")
    parser.add_argument("--save-maps", action="store_true", help="Save per-image prediction maps")

    subparsers = parser.add_subparsers(dest="command", help="Pipeline step to run")

    subparsers.add_parser("download", help="Download browse JPEGs")
    subparsers.add_parser("metadata", help="Regenerate metadata JSON from DB")
    subparsers.add_parser("tile", help="Tile images + extract DINOv2 embeddings")
    subparsers.add_parser("mola", help="Extract MOLA geomorphometric features")
    subparsers.add_parser("cluster", help="K-Means clustering + visualization")
    subparsers.add_parser("train", help="Train supervised classifier on pseudo-labels")
    subparsers.add_parser("fuse", help="Bayesian fusion (CNN + DEM)")
    subparsers.add_parser("predict", help="Run inference on images")
    subparsers.add_parser("export", help="Export predictions to GeoJSON")
    subparsers.add_parser("run-all", help="Run full pipeline end-to-end")
    subparsers.add_parser("status", help="Show pipeline status")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    dispatch = {
        "download": cmd_download,
        "metadata": cmd_metadata,
        "tile": cmd_tile,
        "mola": cmd_mola,
        "cluster": cmd_cluster,
        "train": cmd_train,
        "fuse": cmd_fuse,
        "predict": cmd_predict,
        "export": cmd_export,
        "run-all": cmd_run_all,
        "status": cmd_status,
    }

    handler = dispatch.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
