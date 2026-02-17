"""
CNN Mineral Classification — Inference Pipeline Orchestrator

Runs the full classification pipeline as an async generator yielding SSE events:
1. Resolve TRR3/DDR files
2. Load cubes
3. JCAT atmospheric correction (with time-based ADR selection)
4. CNN inference with confidence filtering
5. Cache results to disk
"""

import os
import json
import math
import time
import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Optional, AsyncGenerator

import numpy as np
import torch

from .constants import (
    RESULTS_DIR, IR_START, IR_COUNT, FILL_VALUE,
    BATCH_SIZE, CONFIDENCE_THRESHOLD, GROUPS_IDX,
    CLASS_NAME, GROUPS_UM,
)
from .data_loader import resolve_trr_files, load_trr_cube, load_ddr_ina, resolve_best_adr, get_obs_start_time
from .model import get_model
from .jcat_bridge import correct_cube

logger = logging.getLogger(__name__)

# Observation ID must be alphanumeric + underscores only (prevent path traversal)
_OBS_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_]+$")


def _validate_obs_id(obs_id: str):
    """Validate observation ID to prevent path traversal attacks."""
    if not _OBS_ID_PATTERN.match(obs_id):
        raise ValueError(f"Invalid observation ID: {obs_id!r} (alphanumeric + underscores only)")


# ============================================================
# Result Data Model
# ============================================================
@dataclass
class ClassificationResult:
    obs_id: str
    rows: int
    cols: int
    mineral_map: np.ndarray       # (rows, cols) int32, -1 = unclassified/invalid
    confidence_map: np.ndarray    # (rows, cols) float32
    attention_map: np.ndarray     # (rows, cols, n_branches) float32
    valid_mask: np.ndarray        # (rows, cols) bool
    class_stats: dict             # {mineral_id_str: pixel_count}
    elapsed_seconds: float
    confidence_threshold: float
    adr_used: str = ""


# ============================================================
# Disk Cache
# ============================================================
def _result_dir(obs_id: str) -> str:
    """Get result directory path for obs_id, with case-insensitive fallback.

    First tries exact match, then case-insensitive search if not found.
    This handles cases where obs_id is provided in different case.
    """
    exact_path = os.path.join(RESULTS_DIR, obs_id)
    if os.path.exists(exact_path):
        return exact_path

    # Fallback: case-insensitive search
    if os.path.exists(RESULTS_DIR):
        obs_id_lower = obs_id.lower()
        try:
            for dirname in os.listdir(RESULTS_DIR):
                if dirname.lower() == obs_id_lower:
                    return os.path.join(RESULTS_DIR, dirname)
        except (OSError, PermissionError) as e:
            logging.debug(f"Case-insensitive search failed: {e}")

    # Not found; return exact path (will fail gracefully later)
    return exact_path


def has_cached_result(obs_id: str) -> bool:
    _validate_obs_id(obs_id)
    d = _result_dir(obs_id)
    return (
        os.path.exists(os.path.join(d, "mineral_map.npy"))
        and os.path.exists(os.path.join(d, "confidence_map.npy"))
    )


def load_cached_result(obs_id: str) -> ClassificationResult:
    """Load previously cached classification from disk."""
    d = _result_dir(obs_id)
    mineral_map = np.load(os.path.join(d, "mineral_map.npy"))
    confidence_map = np.load(os.path.join(d, "confidence_map.npy"))
    attention_map = np.load(os.path.join(d, "attention_map.npy"))
    valid_mask = np.load(os.path.join(d, "valid_mask.npy"))
    with open(os.path.join(d, "metadata.json")) as f:
        meta = json.load(f)
    return ClassificationResult(
        obs_id=obs_id,
        rows=meta["rows"],
        cols=meta["cols"],
        mineral_map=mineral_map,
        confidence_map=confidence_map,
        attention_map=attention_map,
        valid_mask=valid_mask,
        class_stats=meta.get("class_stats", {}),
        elapsed_seconds=meta.get("elapsed_seconds", 0),
        confidence_threshold=meta.get("confidence_threshold", CONFIDENCE_THRESHOLD),
        adr_used=meta.get("adr_used", ""),
    )


def _save_result(result: ClassificationResult):
    """Persist classification result to disk."""
    d = _result_dir(result.obs_id)
    os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, "mineral_map.npy"), result.mineral_map)
    np.save(os.path.join(d, "confidence_map.npy"), result.confidence_map)
    np.save(os.path.join(d, "attention_map.npy"), result.attention_map)
    np.save(os.path.join(d, "valid_mask.npy"), result.valid_mask)
    meta = {
        "obs_id": result.obs_id,
        "rows": result.rows,
        "cols": result.cols,
        "class_stats": result.class_stats,
        "elapsed_seconds": result.elapsed_seconds,
        "confidence_threshold": result.confidence_threshold,
        "adr_used": result.adr_used,
    }
    with open(os.path.join(d, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)


# ============================================================
# Main Pipeline (async generator → SSE events)
# ============================================================
async def run_classification(
    obs_id: str,
    force: bool = False,
) -> AsyncGenerator[dict, None]:
    """
    Run the full CNN mineral classification pipeline.

    Yields SSE event dicts:
      {"event": "status",   "data": {"step": ..., "message": ...}}
      {"event": "progress", "data": {"step": "jcat", "done": N, "total": M, "percent": ...}}
      {"event": "complete", "data": {"obs_id": ..., "stats": ..., ...}}
      {"event": "cached",   "data": {"obs_id": ..., "stats": ..., ...}}
      {"event": "error",    "data": {"error": ..., "type": ...}}
    """
    try:
        _validate_obs_id(obs_id)

        # --- Check cache ---
        if not force and has_cached_result(obs_id):
            result = load_cached_result(obs_id)
            total_valid = int(result.valid_mask.sum())
            total_classified = int(np.sum(result.mineral_map >= 0))
            yield {
                "event": "cached",
                "data": {
                    "obs_id": obs_id,
                    "rows": result.rows,
                    "cols": result.cols,
                    "valid_pixels": total_valid,
                    "classified_pixels": total_classified,
                    "unclassified_pixels": total_valid - total_classified,
                    "stats": result.class_stats,
                    "elapsed": result.elapsed_seconds,
                    "threshold": result.confidence_threshold,
                    "adr_used": result.adr_used,
                },
            }
            return

        t0 = time.time()
        loop = asyncio.get_running_loop()

        # --- Step 1: Resolve files ---
        yield {"event": "status", "data": {
            "step": "resolve", "message": f"Resolving TRR3/DDR files for {obs_id}",
        }}
        files = resolve_trr_files(obs_id)

        # --- Step 2: Load TRR cube ---
        yield {"event": "status", "data": {
            "step": "loading_trr", "message": "Loading TRR3 cube...",
        }}
        cube, rows, cols = await loop.run_in_executor(
            None, load_trr_cube, files["trr_img"], files["trr_lbl"],
        )
        total_bands = cube.shape[2]
        yield {"event": "status", "data": {
            "step": "trr_loaded",
            "message": f"TRR3 loaded: {rows} x {cols}, {total_bands} bands",
        }}

        # Validate band count: model requires IR bands starting at index IR_START
        min_bands = IR_START + IR_COUNT
        if total_bands < min_bands:
            raise ValueError(
                f"S-sensor (VNIR) TRR3 data has {total_bands} bands but mineral classification "
                f"requires L-sensor (IR) data with at least {min_bands} bands (1.0–3.5 µm). "
                f"Download the L-sensor TRR3 for this observation."
            )

        # --- Step 3: Load DDR ---
        yield {"event": "status", "data": {
            "step": "loading_ddr", "message": "Loading DDR incidence angle...",
        }}
        ina_rad = await loop.run_in_executor(
            None, load_ddr_ina, files["ddr_img"], files["ddr_lbl"],
        )
        yield {"event": "status", "data": {
            "step": "ddr_loaded",
            "message": f"Incidence angle: {math.degrees(ina_rad):.2f} deg",
        }}

        # --- Step 4: Select best ADR based on observation time ---
        obs_time = get_obs_start_time(files["trr_lbl"])
        adr_path = resolve_best_adr(obs_time)
        adr_basename = os.path.basename(adr_path)
        yield {"event": "status", "data": {
            "step": "adr_selected",
            "message": f"ADR selected: {adr_basename}"
                + (f" (obs time: {obs_time.isoformat()})" if obs_time else " (default)"),
        }}

        # --- Step 5: JCAT correction (THE SLOW PART) ---
        total_pixels = rows * cols
        yield {"event": "status", "data": {
            "step": "jcat_start",
            "message": f"JCAT atmospheric correction: {total_pixels:,} pixels...",
        }}

        progress_events = []

        def on_progress(done, total):
            progress_events.append({
                "event": "progress",
                "data": {
                    "step": "jcat",
                    "done": done,
                    "total": total,
                    "percent": round(100 * done / total, 1),
                },
            })

        cube_corrected = await loop.run_in_executor(
            None, correct_cube, cube, ina_rad, adr_path, on_progress,
        )

        # Yield accumulated progress events
        for evt in progress_events:
            yield evt

        yield {"event": "status", "data": {
            "step": "jcat_done", "message": "JCAT correction complete",
        }}

        # Free original cube memory
        del cube

        # --- Step 6: CNN inference with confidence filtering ---
        yield {"event": "status", "data": {
            "step": "inference_start", "message": "Loading CNN model...",
        }}

        model, class_map, inv_class_map, device = get_model()

        # Flatten for batch processing
        X_flat = cube_corrected.reshape(-1, IR_COUNT)

        # Validity mask: require >= 250 valid bands
        valid_band_count = np.sum(
            (X_flat < FILL_VALUE) & np.isfinite(X_flat), axis=1,
        )
        valid_mask = valid_band_count >= 250

        # Clean fill values for valid pixels
        X_flat = np.where(X_flat >= FILL_VALUE, 0, X_flat)
        X_flat = np.nan_to_num(X_flat, nan=0.0, posinf=0.0, neginf=0.0)

        # Init output arrays
        pred_flat = np.full(len(X_flat), -1, dtype=np.int32)
        conf_flat = np.zeros(len(X_flat), dtype=np.float32)
        attn_flat = np.zeros((len(X_flat), len(GROUPS_IDX)), dtype=np.float32)

        valid_idx = np.where(valid_mask)[0]
        X_valid = X_flat[valid_idx]

        yield {"event": "status", "data": {
            "step": "inference",
            "message": f"Classifying {len(X_valid):,} valid pixels "
                       f"(threshold={CONFIDENCE_THRESHOLD})...",
        }}

        def _do_inference():
            with torch.no_grad():
                for i in range(0, len(X_valid), BATCH_SIZE):
                    batch = X_valid[i:i + BATCH_SIZE]
                    xg = [
                        torch.from_numpy(batch[:, s:e]).float().to(device)
                        for s, e in GROUPS_IDX
                    ]
                    logits, attn = model(*xg, return_attn=True)

                    # Confidence filtering: softmax → max prob → threshold
                    probs = torch.softmax(logits, dim=1)
                    confidence, predicted = probs.max(dim=1)

                    # Apply 95% threshold — sub-threshold → -1 (unclassified)
                    predicted_np = predicted.cpu().numpy().astype(np.int32)
                    confidence_np = confidence.cpu().numpy()
                    predicted_np[confidence_np < CONFIDENCE_THRESHOLD] = -1

                    batch_idx = valid_idx[i:i + BATCH_SIZE]
                    pred_flat[batch_idx] = predicted_np
                    conf_flat[batch_idx] = confidence_np
                    attn_flat[batch_idx] = attn.cpu().numpy()

        await loop.run_in_executor(None, _do_inference)

        # Map training indices back to original mineral IDs
        mineral_map = np.full_like(pred_flat, -1)
        classified_mask = pred_flat >= 0
        for idx_val in np.unique(pred_flat[classified_mask]):
            mineral_map[pred_flat == idx_val] = inv_class_map[int(idx_val)]

        mineral_map_2d = mineral_map.reshape(rows, cols)
        confidence_map = conf_flat.reshape(rows, cols)
        attention_map = attn_flat.reshape(rows, cols, -1)
        valid_2d = valid_mask.reshape(rows, cols)

        # Compute class statistics
        class_stats = {}
        unique_ids, counts = np.unique(
            mineral_map_2d[mineral_map_2d >= 0], return_counts=True,
        )
        for cls_id, cnt in zip(unique_ids, counts):
            class_stats[str(int(cls_id))] = int(cnt)

        elapsed = round(time.time() - t0, 1)

        # --- Step 7: Save to disk ---
        result = ClassificationResult(
            obs_id=obs_id,
            rows=rows,
            cols=cols,
            mineral_map=mineral_map_2d,
            confidence_map=confidence_map,
            attention_map=attention_map,
            valid_mask=valid_2d,
            class_stats=class_stats,
            elapsed_seconds=elapsed,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            adr_used=adr_basename,
        )
        _save_result(result)

        total_valid = int(valid_mask.sum())
        total_classified = int(classified_mask.sum())

        yield {
            "event": "complete",
            "data": {
                "obs_id": obs_id,
                "rows": rows,
                "cols": cols,
                "valid_pixels": total_valid,
                "classified_pixels": total_classified,
                "unclassified_pixels": total_valid - total_classified,
                "stats": class_stats,
                "elapsed": elapsed,
                "threshold": CONFIDENCE_THRESHOLD,
                "adr_used": adr_basename,
                "class_labels": {
                    str(k): CLASS_NAME.get(int(k), f"Class {k}")
                    for k in class_stats
                },
            },
        }

    except FileNotFoundError as e:
        yield {"event": "error", "data": {"error": str(e), "type": "not_found"}}
    except ValueError as e:
        yield {"event": "error", "data": {"error": str(e), "type": "validation"}}
    except Exception as e:
        logger.error(f"Classification failed for {obs_id}: {e}", exc_info=True)
        yield {"event": "error", "data": {"error": str(e), "type": "internal"}}
