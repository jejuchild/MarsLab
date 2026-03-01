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
    import asyncio
    import numpy as np
    import torch
    from scripts.marslandform_v2.config import PipelineConfig, CLASS_ORDER, METADATA_JSON
    from scripts.marslandform_v2.models.mil_classifier import (
        AttentionMILClassifier, load_embeddings, load_mola_features,
        load_labels, compute_metrics, MILConfig,
    )

    config = PipelineConfig()
    device = torch.device(config.device)

    # --- Find best model checkpoint ---
    model_dirs = [
        V2_OUTPUT / "models" / "multihead_improved",
        V2_OUTPUT / "models" / "cleaned_focal",
        V2_OUTPUT / "models",
    ]
    model_path = None
    for d in model_dirs:
        candidate = d / "best_mil_model.pt"
        if candidate.exists():
            model_path = candidate
            break
    if model_path is None:
        logger.error("No trained model found. Run train_mil stage first.")
        return False
    logger.info(f"Loading model from {model_path}")

    # --- Load checkpoint and reconstruct model ---
    checkpoint = torch.load(model_path, map_location=device)
    mil_cfg_dict = checkpoint.get("mil_config", {})
    mil_cfg = MILConfig(**{k: v for k, v in mil_cfg_dict.items() if k in MILConfig.__dataclass_fields__})
    model = AttentionMILClassifier(mil_cfg).to(device)

    # Handle backward compat: old checkpoints have different layer names
    state_dict = checkpoint["model_state_dict"]
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())
    missing = model_keys - ckpt_keys
    unexpected = ckpt_keys - model_keys
    if missing or unexpected:
        logger.warning(f"Checkpoint mismatch: {len(missing)} missing, {len(unexpected)} unexpected keys")
        logger.warning("Old checkpoint detected. Will run inference with reinitialized layers.")
        logger.warning("For best results, retrain with: python run_pipeline.py --stages train_mil")
        compatible = {k: v for k, v in state_dict.items() if k in model_keys and model.state_dict()[k].shape == v.shape}
        model.load_state_dict(compatible, strict=False)
        logger.info(f"Loaded {len(compatible)}/{len(model_keys)} compatible weights")
    else:
        model.load_state_dict(state_dict)
    model.eval()
    logger.info(f"Model loaded (best_epoch={checkpoint.get('best_epoch', '?')}, best_f1={checkpoint.get('best_landform_macro_f1', '?')})")

    # --- Load embeddings and MOLA features ---
    emb_paths = [
        V2_OUTPUT / "embeddings_mil" / "embeddings_by_image.npy",
        V2_OUTPUT / "embeddings" / "embeddings.npy",
    ]
    embeddings_dict = None
    for p in emb_paths:
        if p.exists():
            data = np.load(p, allow_pickle=True)
            if data.dtype == object and data.shape == ():
                embeddings_dict = data.item()
            break
    if embeddings_dict is None:
        logger.error("No embeddings found.")
        return False
    logger.info(f"Loaded embeddings for {len(embeddings_dict)} images")

    mola_paths = [
        V2_OUTPUT / "mola_features_by_image.npy",
        V2_OUTPUT / "mola_features.npy",
    ]
    mola_dict = None
    for p in mola_paths:
        if p.exists():
            data = np.load(p, allow_pickle=True)
            if data.dtype == object and data.shape == ():
                mola_dict = data.item()
            break
    if mola_dict is None:
        logger.error("No MOLA features found.")
        return False
    logger.info(f"Loaded MOLA features for {len(mola_dict)} images")

    # --- Determine test IDs ---
    split_paths = [
        V2_OUTPUT / "models" / "multihead_improved" / "data_split.json",
        V2_OUTPUT / "models" / "cleaned_focal" / "data_split.json",
        V2_OUTPUT / "dataset_splits.json",
    ]
    test_ids = []
    for sp in split_paths:
        if sp.exists():
            with open(sp) as f:
                splits = json.load(f)
            test_ids = splits.get("test_ids", splits.get("test", []))
            if test_ids:
                break
    if not test_ids:
        logger.warning("No splits file found, running on all labeled images")
        labels_path = V2_OUTPUT / "unified_labels.json"
        if labels_path.exists():
            with open(labels_path) as f:
                test_ids = list(json.load(f).keys())

    # Filter to images that have embeddings and MOLA
    test_ids = [tid for tid in test_ids if tid in embeddings_dict and tid in mola_dict]
    logger.info(f"Running prediction on {len(test_ids)} test images (mode={args.mode})")

    if not test_ids:
        logger.error("No valid test images found.")
        return False

    # --- Load labels for metrics (if available) ---
    labels_dict = {}
    for lp in [V2_OUTPUT / "labels_simple.json", V2_OUTPUT / "unified_labels.json"]:
        if lp.exists():
            with open(lp) as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                # unified_labels has nested structure
                if "labels" in raw and isinstance(raw["labels"], dict):
                    raw = raw["labels"]
                for k, v in raw.items():
                    if isinstance(v, str) and v in CLASS_ORDER:
                        labels_dict[k] = CLASS_ORDER.index(v)
                    elif isinstance(v, int):
                        labels_dict[k] = v
                    elif isinstance(v, dict):
                        cls = v.get("final_class") or v.get("class") or v.get("label")
                        if cls and cls in CLASS_ORDER:
                            labels_dict[k] = CLASS_ORDER.index(cls)
            break

    # --- Prediction function ---
    @torch.no_grad()
    def predict_image(image_id: str) -> dict:
        tiles = np.asarray(embeddings_dict[image_id], dtype=np.float32)
        mola = np.asarray(mola_dict[image_id], dtype=np.float32)
        # Handle tile count
        if tiles.shape[0] > mil_cfg.max_tiles_per_image:
            keep = np.random.choice(tiles.shape[0], size=mil_cfg.max_tiles_per_image, replace=False)
            tiles = tiles[keep]
        tiles_t = torch.from_numpy(tiles).unsqueeze(0).to(device)
        mask_t = torch.ones(1, tiles_t.shape[1], dtype=torch.bool, device=device)
        mola_t = torch.from_numpy(mola).unsqueeze(0).to(device)
        logits, att_weights = model(tiles_t, mask_t, mola_t)
        probs = torch.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)
        pred_idx = int(pred.item())
        return {
            "class": CLASS_ORDER[pred_idx],
            "confidence": float(conf.item()),
            "probabilities": probs[0].cpu().tolist(),
            "attention_weights": att_weights[0, :tiles.shape[0]].cpu().tolist(),
            "pred_label": pred_idx,
            "pred_label_name": CLASS_ORDER[pred_idx],
        }

    # --- Run predictions ---
    predictions = []
    if args.mode == "fast":
        logger.info("Running FAST mode (MIL classifier only)")
        for i, image_id in enumerate(test_ids):
            result = predict_image(image_id)
            pred_entry = {
                "image_id": image_id,
                "predicted_class": result["class"],
                "confidence": result["confidence"],
                "probabilities": result["probabilities"],
                "attention_weights": result["attention_weights"],
                "mode": "fast",
            }
            if image_id in labels_dict:
                pred_entry["true_label"] = labels_dict[image_id]
                pred_entry["true_label_name"] = CLASS_ORDER[labels_dict[image_id]]
            predictions.append(pred_entry)
            if (i + 1) % 20 == 0 or (i + 1) == len(test_ids):
                logger.info(f"  Predicted {i+1}/{len(test_ids)}")
    else:
        # Agent mode
        logger.info("Running AGENT mode (ReACT loop)")
        from scripts.marslandform_v2.agent.react_agent import MarsLandformAgent, MockVLM

        # Create MILPredictor wrapper for ClassifyTool
        class MILPredictor:
            def predict_image(self, image_id: str) -> dict:
                return predict_image(image_id)

        # Convert MOLA numpy arrays to dicts for AnalyzeMOLATool
        mola_feature_names = [
            "slope_mean_1km", "slope_std_1km", "curvature_mean_1km",
            "TPI_1km", "TRI_1km", "roughness_1km", "lobateness_1km",
            "slope_mean_5km", "slope_std_5km", "curvature_mean_5km",
            "TPI_5km", "TRI_5km", "roughness_5km", "lobateness_5km",
            "slope_mean_20km", "slope_std_20km", "curvature_mean_20km",
            "TPI_20km", "TRI_20km", "roughness_20km", "lobateness_20km",
            "elevation_mean", "abs_latitude",
        ]
        mola_as_dicts = {}
        for img_id, arr in mola_dict.items():
            mola_as_dicts[img_id] = {
                mola_feature_names[i]: float(arr[i])
                for i in range(min(len(mola_feature_names), len(arr)))
            }

        # Load metadata for RegionalContextTool
        metadata = None
        if METADATA_JSON.exists():
            with open(METADATA_JSON) as f:
                meta_raw = json.load(f)
            if isinstance(meta_raw, list):
                metadata = {item["image_id"]: item for item in meta_raw if "image_id" in item}
            else:
                metadata = meta_raw

        config.agent.mode = args.mode
        agent = MarsLandformAgent(
            config=config.agent,
            classifier=MILPredictor(),
            rag=None,
            mola_features=mola_as_dicts,
            metadata=metadata or {},
            vlm=MockVLM(),  # Use mock VLM for offline; set ANTHROPIC_API_KEY for real
        )

        async def run_agent_predictions():
            results = []
            for i, image_id in enumerate(test_ids):
                try:
                    agent_result = await agent.classify_image(image_id)
                    pred_entry = {
                        "image_id": image_id,
                        "predicted_class": agent_result.landform_class,
                        "confidence": agent_result.confidence,
                        "mode": agent_result.mode,
                        "num_steps": agent_result.num_steps,
                        "tools_used": agent_result.tools_used,
                        "reasoning_chain": agent_result.reasoning_chain,
                    }
                    if image_id in labels_dict:
                        pred_entry["true_label"] = labels_dict[image_id]
                        pred_entry["true_label_name"] = CLASS_ORDER[labels_dict[image_id]]
                    results.append(pred_entry)
                except Exception as e:
                    logger.error(f"Agent failed for {image_id}: {e}")
                    results.append({
                        "image_id": image_id,
                        "predicted_class": "BACKGROUND",
                        "confidence": 0.0,
                        "mode": "agent",
                        "error": str(e),
                    })
                if (i + 1) % 10 == 0 or (i + 1) == len(test_ids):
                    logger.info(f"  Agent predicted {i+1}/{len(test_ids)}")
            return results

        predictions = asyncio.run(run_agent_predictions())

    # --- Save predictions ---
    predictions_dir = V2_OUTPUT / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    preds_path = predictions_dir / "predictions.json"
    preds_path.write_text(json.dumps(predictions, indent=2))
    logger.info(f"Predictions saved to {preds_path} ({len(predictions)} images)")

    # --- Compute and save summary metrics ---
    y_true = [p["true_label"] for p in predictions if "true_label" in p]
    y_pred = [CLASS_ORDER.index(p["predicted_class"]) for p in predictions if "true_label" in p]
    if y_true:
        metrics = compute_metrics(y_true, y_pred, num_classes=mil_cfg.num_classes)
        summary = {
            "num_predictions": len(predictions),
            "num_with_labels": len(y_true),
            "mode": args.mode,
            "macro_f1_all": metrics["macro_f1_all"],
            "landform_macro_f1": metrics["landform_macro_f1"],
            "per_class_f1": {CLASS_ORDER[i]: metrics["f1"][i] for i in range(len(CLASS_ORDER))},
            "confusion_matrix": metrics["confusion_matrix"],
        }
        summary_path = predictions_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        logger.info(f"Macro-F1: {metrics['macro_f1_all']:.4f}, Landform F1: {metrics['landform_macro_f1']:.4f}")
    else:
        logger.warning("No labeled test images — cannot compute metrics")

    return True


def stage_export(args) -> bool:
    """Stage 8: Export GeoJSON + evaluation report."""
    predictions_dir = V2_OUTPUT / "predictions"
    predictions_path = predictions_dir / "predictions.json"
    if not predictions_path.exists():
        logger.warning("No predictions found. Run predict stage first.")
        return False

    try:
        from scripts.marslandform_v2.export.geojson import export_geojson
        from scripts.marslandform_v2.export.report import generate_report
        from scripts.marslandform_v2.config import GEOJSON_DIR, EVAL_DIR

        # Export GeoJSON
        geojson_out = GEOJSON_DIR / "mars_landform_predictions.geojson"
        export_geojson(
            predictions_path=predictions_path,
            metadata_path=METADATA_JSON,
            output_path=geojson_out,
        )
        logger.info(f"GeoJSON exported to {geojson_out}")

        # Generate evaluation report
        metrics_path = predictions_dir / "summary.json"
        if metrics_path.exists():
            report_path = generate_report(
                metrics_path=metrics_path,
                output_dir=EVAL_DIR,
                format="html",
            )
            logger.info(f"Report generated at {report_path}")

        # Print summary
        with open(predictions_path) as f:
            preds = json.load(f)
        from collections import Counter
        class_counts = Counter(p["predicted_class"] for p in preds)
        avg_conf = sum(p.get("confidence", 0) for p in preds) / max(len(preds), 1)
        logger.info(f"Export summary: {len(preds)} images, avg_confidence={avg_conf:.3f}")
        for cls, count in sorted(class_counts.items()):
            logger.info(f"  {cls}: {count} images")

    except Exception as e:
        logger.error(f"Export failed: {e}")
        import traceback
        traceback.print_exc()
        return False

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
