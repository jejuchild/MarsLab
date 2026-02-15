"""
I/O utilities for Ice Evidence Synthesizer.

Handles saving JSON results and optional PNG snapshots.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import List, Optional

import numpy as np

from .models import IceEvidenceResult, HyperbolaFitResult

logger = logging.getLogger(__name__)

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "results", "ice_evidence",
)


def ensure_results_dir():
    """Create results directory if it doesn't exist."""
    os.makedirs(RESULTS_DIR, exist_ok=True)


def save_evidence_result(result: IceEvidenceResult) -> str:
    """Save IceEvidenceResult to JSON.

    Returns path to saved file.
    """
    ensure_results_dir()
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"{ts}_{result.candidate_id}.json"
    filepath = os.path.join(RESULTS_DIR, filename)

    data = result.model_dump()
    data["_saved_at"] = datetime.utcnow().isoformat()

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)

    logger.info(f"Saved ice evidence result: {filepath}")
    return filepath


def save_hyperbola_fit(product_id: str, result: HyperbolaFitResult) -> str:
    """Save HyperbolaFitResult to JSON."""
    ensure_results_dir()
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"hyperbola_{product_id}_{ts}.json"
    filepath = os.path.join(RESULTS_DIR, filename)

    data = result.model_dump()
    data["_product_id"] = product_id
    data["_saved_at"] = datetime.utcnow().isoformat()

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)

    logger.info(f"Saved hyperbola fit: {filepath}")
    return filepath


def save_hyperbola_overlay_png(
    product_id: str,
    power: np.ndarray,
    result: HyperbolaFitResult,
    apex_trace: int,
    apex_bin: int,
    roi_traces: int,
    roi_bins: int,
) -> Optional[str]:
    """Save a PNG snapshot of the ROI with fitted hyperbola overlay.

    Returns path to PNG or None on failure.
    """
    try:
        from PIL import Image, ImageDraw

        ensure_results_dir()
        n_traces, n_bins = power.shape
        t_lo = max(0, apex_trace - roi_traces)
        t_hi = min(n_traces, apex_trace + roi_traces)
        b_lo = max(0, apex_bin - roi_bins)
        b_hi = min(n_bins, apex_bin + roi_bins)

        roi = power[t_lo:t_hi, b_lo:b_hi].astype(np.float64)
        roi = np.where(roi > 0, np.log10(roi + 1e-30), 0)

        # Normalize to 0-255
        if roi.max() > roi.min():
            roi_norm = ((roi - roi.min()) / (roi.max() - roi.min()) * 255).astype(np.uint8)
        else:
            roi_norm = np.zeros_like(roi, dtype=np.uint8)

        # Create image (transpose so traces = x, bins = y)
        img = Image.fromarray(roi_norm.T, mode="L").convert("RGB")
        draw = ImageDraw.Draw(img)

        # Draw fitted curve
        for pt in result.overlay_polyline:
            x = pt.trace - t_lo
            y = pt.bin - b_lo
            if 0 <= x < img.width and 0 <= y < img.height:
                draw.ellipse([x - 1, y - 1, x + 1, y + 1], fill="red")

        # Draw CI band
        if result.overlay_ci_band:
            for pt in result.overlay_ci_band:
                x = pt.trace - t_lo
                y_lo = pt.bin_lo - b_lo
                y_hi = pt.bin_hi - b_lo
                if 0 <= x < img.width:
                    y_lo_c = max(0, min(img.height - 1, int(y_lo)))
                    y_hi_c = max(0, min(img.height - 1, int(y_hi)))
                    draw.line([(x, y_lo_c), (x, y_hi_c)], fill=(255, 100, 100, 80), width=1)

        # Draw preset curves if present
        colors = {"2.8": (100, 200, 255), "3.15": (100, 255, 100), "4.0": (255, 255, 100)}
        if result.preset_curves:
            for key, curve in result.preset_curves.items():
                color = colors.get(key, (200, 200, 200))
                for pt in curve:
                    x = pt.trace - t_lo
                    y = pt.bin - b_lo
                    if 0 <= x < img.width and 0 <= y < img.height:
                        draw.point((x, int(y)), fill=color)

        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"hyperbola_overlay_{product_id}_{ts}.png"
        filepath = os.path.join(RESULTS_DIR, filename)
        img.save(filepath)
        logger.info(f"Saved hyperbola overlay PNG: {filepath}")
        return filepath

    except Exception as e:
        logger.warning(f"Failed to save overlay PNG: {e}")
        return None


def load_latest_result(candidate_id: str) -> Optional[dict]:
    """Load most recent IceEvidenceResult for a candidate."""
    ensure_results_dir()

    matching = []
    for fname in os.listdir(RESULTS_DIR):
        if fname.endswith(f"_{candidate_id}.json") and not fname.startswith("hyperbola_"):
            matching.append(fname)

    if not matching:
        return None

    matching.sort(reverse=True)  # most recent first (timestamp prefix)
    filepath = os.path.join(RESULTS_DIR, matching[0])

    with open(filepath, "r") as f:
        return json.load(f)


def list_results(limit: int = 50) -> List[dict]:
    """List recent ice evidence results."""
    ensure_results_dir()

    results = []
    for fname in sorted(os.listdir(RESULTS_DIR), reverse=True):
        if fname.endswith(".json") and not fname.startswith("hyperbola_"):
            filepath = os.path.join(RESULTS_DIR, fname)
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                results.append({
                    "file": fname,
                    "candidate_id": data.get("candidate_id", ""),
                    "ice_probability": data.get("ice_probability", 0),
                    "confidence": data.get("confidence", 0),
                    "saved_at": data.get("_saved_at", ""),
                })
            except Exception:
                continue
        if len(results) >= limit:
            break

    return results
