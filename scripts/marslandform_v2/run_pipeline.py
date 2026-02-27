"""
MarsLandformNet V2 — Pipeline Orchestrator.

Runs the full pipeline or individual stages:
  Stage 1: collect_labels   — Unify label sources
  Stage 2: tile             — Extract 224×224 tiles from browse images
  Stage 3: mola             — Extract MOLA features (image-level)
  Stage 4: embed            — Extract DINOv2 embeddings (frozen or LoRA)
  Stage 5: train_mil        — Train MIL classifier
  Stage 6: ingest_rag       — Build RAG knowledge base
  Stage 7: predict          — Classify images (fast or agent mode)
  Stage 8: export           — Export GeoJSON + evaluation report

Usage:
  python run_pipeline.py --stages all
  python run_pipeline.py --stages labels,tile,embed
  python run_pipeline.py --stages predict --mode agent
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.marslandform_v2.config import (
    BROWSE_DIR, METADATA_JSON, MOLA_DEM, V2_OUTPUT,
    RAG_CORPUS_DIR, get_config,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent


def run_stage(name: str, cmd: list, timeout: int = 3600) -> bool:
    """Run a pipeline stage as subprocess."""
    logger.info(f"{'='*60}")
    logger.info(f"STAGE: {name}")
    logger.info(f"CMD: {' '.join(str(c) for c in cmd)}")
    logger.info(f"{'='*60}")

    start = time.time()
    try:
        result = subprocess.run(
            [sys.executable] + [str(c) for c in cmd],
            cwd=str(SCRIPT_DIR.parents[1]),
            timeout=timeout,
            capture_output=False,
        )
        elapsed = time.time() - start

        if result.returncode == 0:
            logger.info(f"✓ {name} completed in {elapsed:.1f}s")
            return True
        else:
            logger.error(f"✗ {name} FAILED (exit code {result.returncode}) after {elapsed:.1f}s")
            return False
    except subprocess.TimeoutExpired:
        logger.error(f"✗ {name} TIMED OUT after {timeout}s")
        return False
    except Exception as e:
        logger.error(f"✗ {name} ERROR: {e}")
        return False


def stage_labels(args) -> bool:
    """Stage 1: Collect and unify labels."""
    return run_stage("collect_labels", [
        SCRIPT_DIR / "data" / "collect_labels.py",
        "--metadata", str(METADATA_JSON),
        "--output-dir", str(V2_OUTPUT),
    ])


def stage_tile(args) -> bool:
    """Stage 2: Tile HiRISE browse images."""
    return run_stage("tile", [
        SCRIPT_DIR / "data" / "tiler.py",
        "--labels", str(V2_OUTPUT / "unified_labels.json"),
        "--browse-dir", str(BROWSE_DIR),
        "--output-dir", str(V2_OUTPUT),
    ] + (["--limit", str(args.limit)] if args.limit else []),
        timeout=7200,
    )


def stage_mola(args) -> bool:
    """Stage 3: Extract MOLA features."""
    return run_stage("mola", [
        SCRIPT_DIR / "data" / "mola.py",
        "--labels", str(V2_OUTPUT / "unified_labels.json"),
        "--dem", str(MOLA_DEM),
        "--output-dir", str(V2_OUTPUT),
    ] + (["--limit", str(args.limit)] if args.limit else []),
        timeout=3600,
    )


def stage_embed(args) -> bool:
    """Stage 4: Extract DINOv2 embeddings."""
    cmd = [
        SCRIPT_DIR / "models" / "embedder.py",
        "--image-dir", str(V2_OUTPUT / "tiles"),
        "--output-dir", str(V2_OUTPUT / "embeddings"),
        "--batch-size", str(args.batch_size),
    ]
    if args.model_path:
        cmd.extend(["--model-path", str(args.model_path)])
    return run_stage("embed", cmd, timeout=14400)


def stage_train_mil(args) -> bool:
    """Stage 5: Train MIL classifier."""
    return run_stage("train_mil", [
        SCRIPT_DIR / "models" / "mil_classifier.py",
        "--embeddings-dir", str(V2_OUTPUT / "embeddings"),
        "--mola-path", str(V2_OUTPUT / "mola_features.npy"),
        "--labels-path", str(V2_OUTPUT / "unified_labels.json"),
        "--output-dir", str(V2_OUTPUT / "models"),
        "--epochs", str(args.epochs),
        "--patience", str(args.patience),
    ], timeout=7200)


def stage_ingest_rag(args) -> bool:
    """Stage 6: Build RAG knowledge base."""
    return run_stage("ingest_rag", [
        SCRIPT_DIR / "rag" / "ingest.py",
        "--corpus-dir", str(RAG_CORPUS_DIR),
        "--reset",
    ])


def stage_predict(args) -> bool:
    """Stage 7: Run classification (fast or agent mode)."""
    # Load test set
    splits_path = V2_OUTPUT / "dataset_splits.json"
    if splits_path.exists():
        with open(splits_path) as f:
            splits = json.load(f)
        test_ids = splits.get("test", [])
    else:
        logger.warning("No splits file found, running on all labeled images")
        labels_path = V2_OUTPUT / "unified_labels.json"
        with open(labels_path) as f:
            test_ids = list(json.load(f).keys())

    logger.info(f"Running prediction on {len(test_ids)} images (mode={args.mode})")

    # Import and run agent
    from scripts.marslandform_v2.agent.react_agent import MarsLandformAgent
    from scripts.marslandform_v2.config import PipelineConfig
    import asyncio

    config = PipelineConfig()
    config.agent.mode = args.mode

    # This is a simplified prediction loop
    # Full implementation would load all models and run inference
    logger.info(f"Prediction stage placeholder — requires trained models")
    logger.info(f"Would classify {len(test_ids)} test images in '{args.mode}' mode")
    return True


def stage_export(args) -> bool:
    """Stage 8: Export GeoJSON + evaluation report."""
    logger.info("Export stage — requires prediction results")
    predictions_dir = V2_OUTPUT / "predictions"
    if not predictions_dir.exists():
        logger.warning("No predictions found. Run predict stage first.")
        return False

    logger.info("Export stage placeholder — will generate GeoJSON + report")
    return True


STAGES = {
    "labels": stage_labels,
    "tile": stage_tile,
    "mola": stage_mola,
    "embed": stage_embed,
    "train_mil": stage_train_mil,
    "ingest_rag": stage_ingest_rag,
    "predict": stage_predict,
    "export": stage_export,
}

ALL_STAGES = ["labels", "tile", "mola", "embed", "train_mil", "ingest_rag", "predict", "export"]


def main():
    parser = argparse.ArgumentParser(description="MarsLandformNet V2 Pipeline")
    parser.add_argument("--stages", type=str, default="all",
                       help="Comma-separated stages or 'all'")
    parser.add_argument("--mode", type=str, default="fast",
                       choices=["fast", "agent"],
                       help="Prediction mode: fast (classifier only) or agent (ReACT)")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit images (for testing)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--model-path", type=str, default=None,
                       help="Path to LoRA adapter (None = frozen DINOv2)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--stop-on-failure", action="store_true",
                       help="Stop pipeline if any stage fails")
    args = parser.parse_args()

    # Determine stages to run
    if args.stages == "all":
        stages = ALL_STAGES
    else:
        stages = [s.strip() for s in args.stages.split(",")]
        for s in stages:
            if s not in STAGES:
                logger.error(f"Unknown stage: {s}. Available: {', '.join(STAGES.keys())}")
                sys.exit(1)

    logger.info(f"MarsLandformNet V2 Pipeline")
    logger.info(f"Stages: {', '.join(stages)}")
    logger.info(f"Output: {V2_OUTPUT}")

    results = {}
    for stage_name in stages:
        success = STAGES[stage_name](args)
        results[stage_name] = success
        if not success and args.stop_on_failure:
            logger.error(f"Pipeline stopped due to failure in {stage_name}")
            break

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"PIPELINE SUMMARY")
    logger.info(f"{'='*60}")
    for stage, success in results.items():
        status = "✓" if success else "✗"
        logger.info(f"  {status} {stage}")

    all_success = all(results.values())
    if all_success:
        logger.info(f"\nAll stages completed successfully!")
    else:
        failed = [s for s, ok in results.items() if not ok]
        logger.warning(f"\nFailed stages: {', '.join(failed)}")

    return 0 if all_success else 1


if __name__ == "__main__":
    sys.exit(main())
