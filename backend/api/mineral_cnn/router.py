"""
CNN Mineral Classification — FastAPI Router

Endpoints for running CNN mineral classification on CRISM TRR3 observations,
retrieving results (stats, mineral map, confidence map, per-pixel queries),
and checking model status.

Route prefix: /api/mineral-cnn
"""

import json
import os
import io
import logging

import numpy as np
from PIL import Image

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse, Response, JSONResponse

from api.validation import validate_product_id

from .pipeline import (
    run_classification, has_cached_result, load_cached_result, _result_dir,
)
from .model import model_status
from .constants import CLASS_NAME, CONFIDENCE_THRESHOLD, RESULTS_DIR, GROUPS_UM

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/mineral-cnn", tags=["CNN Mineral Classification"])


def _validate_obs_id_or_400(obs_id: str) -> str:
    try:
        return validate_product_id(obs_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid observation ID")


# ============================================================
# POST /classify/{obs_id} — Run classification (SSE stream)
# ============================================================
@router.post("/classify/{obs_id}")
async def classify_observation(
    obs_id: str,
    force: bool = Query(False, description="Force re-run even if cached"),
):
    """
    Run CNN mineral classification on a CRISM TRR3 observation.
    Returns an SSE stream with progress events.
    """
    obs_id = _validate_obs_id_or_400(obs_id)

    async def event_generator():
        async for event in run_classification(obs_id, force=force):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ============================================================
# POST /acquire/{obs_id} — Full pipeline: download + classify
# ============================================================
@router.post("/acquire/{obs_id}")
async def acquire_observation(
    obs_id: str,
    force: bool = Query(False, description="Force re-download and re-classify"),
):
    """
    Full pipeline: download TRR3+DDR from ODE, run JCAT+CNN, generate quickview, update index.
    Returns an SSE stream with progress events.
    """
    obs_id = _validate_obs_id_or_400(obs_id)

    from .acquire import acquire_and_classify

    async def event_generator():
        async for event in acquire_and_classify(obs_id, force=force):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ============================================================
# GET /acquire/{obs_id}/status — Check data availability
# ============================================================
@router.get("/acquire/{obs_id}/status")
async def acquire_status(obs_id: str):
    """Check if TRR3 data exists locally and if classification results are cached."""
    obs_id = _validate_obs_id_or_400(obs_id)

    from .acquire import check_acquire_status
    try:
        return JSONResponse(check_acquire_status(obs_id))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================
# GET /result/{obs_id}/stats — Classification summary
# ============================================================
@router.get("/result/{obs_id}/stats")
async def get_result_stats(obs_id: str):
    """Get classification summary statistics."""
    obs_id = _validate_obs_id_or_400(obs_id)

    try:
        if not has_cached_result(obs_id):
            raise HTTPException(
                status_code=404,
                detail=f"No classification result for {obs_id}. Run POST /classify/{obs_id} first.",
            )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    result = load_cached_result(obs_id)

    # Build human-readable mineral stats
    minerals = {}
    total_classified = 0
    for cls_id_str, count in result.class_stats.items():
        cls_id = int(cls_id_str)
        name = CLASS_NAME.get(cls_id, f"Class {cls_id}")
        minerals[name] = {
            "mineral_id": cls_id,
            "pixel_count": count,
            "percent_of_image": round(100 * count / (result.rows * result.cols), 2),
        }
        total_classified += count

    total_valid = int(result.valid_mask.sum())

    return JSONResponse({
        "obs_id": obs_id,
        "dimensions": {"rows": result.rows, "cols": result.cols},
        "total_pixels": result.rows * result.cols,
        "valid_pixels": total_valid,
        "classified_pixels": total_classified,
        "unclassified_pixels": total_valid - total_classified,
        "confidence_threshold": result.confidence_threshold,
        "elapsed_seconds": result.elapsed_seconds,
        "adr_used": result.adr_used,
        "minerals": minerals,
        "top_minerals": sorted(
            [{"name": k, **v} for k, v in minerals.items()],
            key=lambda x: x["pixel_count"],
            reverse=True,
        )[:10],
    })


# ============================================================
# GET /result/{obs_id}/mineral-map.png — Colored mineral map
# ============================================================
# Fixed color palette for consistent mineral coloring across observations
_MINERAL_COLORS = {
    1: (0, 191, 255),      # CO2 Ice — deep sky blue
    2: (135, 206, 250),    # H2O Ice — light sky blue
    3: (255, 255, 0),      # Gypsum — yellow
    4: (255, 140, 0),      # Ferric Hydroxysulfate — dark orange
    6: (139, 69, 19),      # Fe smectite — saddle brown
    7: (107, 142, 35),     # Mg smectite — olive drab
    8: (0, 128, 0),        # Prehnite — green
    9: (255, 69, 0),       # Jarosite — orange-red
    10: (34, 139, 34),     # Serpentine — forest green
    11: (220, 20, 60),     # Alunite — crimson
    12: (178, 34, 34),     # Akaganeite — firebrick
    14: (210, 105, 30),    # Al smectite 1 — chocolate
    15: (255, 228, 181),   # Kaolinite — moccasin
    16: (255, 218, 185),   # Bassanite — peachpuff
    17: (127, 255, 0),     # Epidote — chartreuse
    18: (244, 164, 96),    # Al smectite 2 — sandy brown
    19: (148, 0, 211),     # Polyhydrated sulfate — dark violet
    23: (160, 82, 45),     # Illite — sienna
    25: (0, 206, 209),     # Analcime — dark turquoise
    26: (186, 85, 211),    # Monohydrated sulfate — medium orchid
    27: (176, 224, 230),   # Hydrated silica — powder blue
    29: (255, 99, 71),     # Ferricopiapite — tomato
    31: (46, 139, 87),     # Chlorite — sea green
    38: (85, 107, 47),     # Chlorite-smectite — dark olive green
    100: (169, 169, 169),  # Water-unrelated — dark gray
}


@router.get("/result/{obs_id}/mineral-map.png")
async def get_mineral_map_png(obs_id: str):
    """Serve colored PNG mineral classification map."""
    obs_id = _validate_obs_id_or_400(obs_id)

    try:
        _ = has_cached_result(obs_id)  # validates obs_id
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    cache_png = os.path.join(_result_dir(obs_id), "mineral_map.png")

    if os.path.exists(cache_png):
        with open(cache_png, "rb") as f:
            return Response(
                f.read(),
                media_type="image/png",
                headers={"Cache-Control": "public, max-age=86400"},
            )

    if not has_cached_result(obs_id):
        raise HTTPException(status_code=404, detail=f"No classification result for {obs_id}")

    result = load_cached_result(obs_id)

    # Build RGB image using fixed mineral color palette
    rgb = np.zeros((result.rows, result.cols, 4), dtype=np.uint8)

    for mineral_id, color in _MINERAL_COLORS.items():
        mask = result.mineral_map == mineral_id
        if mask.any():
            rgb[mask, 0] = color[0]
            rgb[mask, 1] = color[1]
            rgb[mask, 2] = color[2]
            rgb[mask, 3] = 255

    # Invalid/unclassified pixels remain transparent (alpha=0)

    img = Image.fromarray(rgb, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    png_bytes = buf.read()

    # Cache to disk
    os.makedirs(os.path.dirname(cache_png), exist_ok=True)
    with open(cache_png, "wb") as f:
        f.write(png_bytes)

    return Response(
        png_bytes,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=86400"},
    )


# ============================================================
# GET /result/{obs_id}/confidence-map.png — Grayscale confidence
# ============================================================
@router.get("/result/{obs_id}/confidence-map.png")
async def get_confidence_map_png(obs_id: str):
    """Serve grayscale PNG confidence map (brighter = higher confidence)."""
    obs_id = _validate_obs_id_or_400(obs_id)

    try:
        _ = has_cached_result(obs_id)  # validates obs_id
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    cache_png = os.path.join(_result_dir(obs_id), "confidence_map.png")

    if os.path.exists(cache_png):
        with open(cache_png, "rb") as f:
            return Response(
                f.read(),
                media_type="image/png",
                headers={"Cache-Control": "public, max-age=86400"},
            )

    if not has_cached_result(obs_id):
        raise HTTPException(status_code=404, detail=f"No classification result for {obs_id}")

    result = load_cached_result(obs_id)

    # Scale confidence [0, 1] → [0, 255], invalid pixels = 0
    conf_u8 = (result.confidence_map * 255).astype(np.uint8)
    conf_u8[~result.valid_mask] = 0

    img = Image.fromarray(conf_u8, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    png_bytes = buf.read()

    os.makedirs(os.path.dirname(cache_png), exist_ok=True)
    with open(cache_png, "wb") as f:
        f.write(png_bytes)

    return Response(
        png_bytes,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=86400"},
    )


# ============================================================
# GET /result/{obs_id}/pixel — Single pixel classification
# ============================================================
@router.get("/result/{obs_id}/pixel")
async def get_pixel_classification(
    obs_id: str,
    line: int = Query(..., description="Row index (0-based)"),
    sample: int = Query(..., description="Column index (0-based)"),
):
    """Query CNN classification for a single pixel."""
    obs_id = _validate_obs_id_or_400(obs_id)

    try:
        cached = has_cached_result(obs_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not cached:
        raise HTTPException(status_code=404, detail=f"No classification result for {obs_id}")

    result = load_cached_result(obs_id)

    if line < 0 or line >= result.rows or sample < 0 or sample >= result.cols:
        raise HTTPException(
            status_code=400,
            detail=f"Pixel ({line}, {sample}) out of bounds "
                   f"({result.rows} x {result.cols})",
        )

    mineral_id = int(result.mineral_map[line, sample])
    confidence = float(result.confidence_map[line, sample])
    attention = result.attention_map[line, sample].tolist()
    is_valid = bool(result.valid_mask[line, sample])

    return JSONResponse({
        "obs_id": obs_id,
        "line": line,
        "sample": sample,
        "is_valid": is_valid,
        "mineral_id": mineral_id,
        "mineral_name": CLASS_NAME.get(
            mineral_id,
            "Unclassified" if mineral_id == -1 else f"Class {mineral_id}",
        ),
        "confidence": round(confidence, 4),
        "above_threshold": confidence >= CONFIDENCE_THRESHOLD,
        "threshold": CONFIDENCE_THRESHOLD,
        "attention_weights": {
            f"{a:.2f}-{b:.2f} um": round(w, 4)
            for (a, b), w in zip(GROUPS_UM, attention)
        },
    })


# ============================================================
# GET /status — Model and system status
# ============================================================
@router.get("/status")
async def get_status():
    """Check CNN model status and list available results."""
    status = model_status()

    # List cached results
    available = []
    if os.path.isdir(RESULTS_DIR):
        for d in sorted(os.listdir(RESULTS_DIR)):
            meta_path = os.path.join(RESULTS_DIR, d, "metadata.json")
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                available.append({
                    "obs_id": d,
                    "rows": meta.get("rows"),
                    "cols": meta.get("cols"),
                    "elapsed": meta.get("elapsed_seconds"),
                    "threshold": meta.get("confidence_threshold"),
                })

    status["available_results"] = available
    status["confidence_threshold"] = CONFIDENCE_THRESHOLD
    return JSONResponse(status)


# ============================================================
# GET /result/{obs_id}/legend — Color legend for mineral map
# ============================================================
@router.get("/result/{obs_id}/legend")
async def get_mineral_legend(obs_id: str):
    """Get the color legend for a classified observation's mineral map."""
    obs_id = _validate_obs_id_or_400(obs_id)

    try:
        cached = has_cached_result(obs_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not cached:
        raise HTTPException(status_code=404, detail=f"No classification result for {obs_id}")

    result = load_cached_result(obs_id)

    legend = []
    for cls_id_str in sorted(result.class_stats, key=lambda x: result.class_stats[x], reverse=True):
        cls_id = int(cls_id_str)
        color = _MINERAL_COLORS.get(cls_id, (128, 128, 128))
        legend.append({
            "mineral_id": cls_id,
            "name": CLASS_NAME.get(cls_id, f"Class {cls_id}"),
            "color_rgb": list(color),
            "color_hex": f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}",
            "pixel_count": result.class_stats[cls_id_str],
        })

    return JSONResponse({"obs_id": obs_id, "legend": legend})


# ============================================================
# GET /quickview/{obs_id} — VNIR single-band quickview PNG
# ============================================================
@router.get("/quickview/{obs_id}")
async def get_trr3_quickview(obs_id: str, band: int = Query(None, description="Band index (0-based). Default: middle VNIR band")):
    """Serve a grayscale PNG quickview from a TRR3 VNIR band."""
    obs_id = _validate_obs_id_or_400(obs_id)

    from .constants import TRR_DATA_DIR

    obs_dir = os.path.join(TRR_DATA_DIR, obs_id)
    if not os.path.isdir(obs_dir):
        raise HTTPException(status_code=404, detail=f"Observation {obs_id} not found")

    # Find TRR3 files
    trr_img = trr_lbl = None
    for fname in os.listdir(obs_dir):
        upper = fname.upper()
        if upper.endswith("_TRR3.IMG"):
            trr_img = os.path.join(obs_dir, fname)
        elif upper.endswith("_TRR3.LBL"):
            trr_lbl = os.path.join(obs_dir, fname)

    if not trr_img or not trr_lbl:
        raise HTTPException(status_code=404, detail="TRR3 files not found")

    # Check for cached quickview
    cache_path = os.path.join(obs_dir, "quickview.png")
    if band is None and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return Response(
                f.read(),
                media_type="image/png",
                headers={"Cache-Control": "public, max-age=86400"},
            )

    try:
        from .data_loader import load_trr_cube

        cube, rows, cols = load_trr_cube(trr_img, trr_lbl)
        n_bands = cube.shape[2]

        # Pick band: default to ~60% through bands (roughly 0.7 µm for S-sensor)
        if band is None:
            band = int(n_bands * 0.6)
        band = max(0, min(band, n_bands - 1))

        single = cube[:, :, band].astype(np.float64)

        # Replace fill/invalid values
        valid = (single > 0) & (single < 1.5) & np.isfinite(single)
        if valid.sum() < 10:
            raise HTTPException(status_code=500, detail="No valid data in selected band")

        # Stretch to 0-255 using 2-98 percentile
        p2 = float(np.percentile(single[valid], 2))
        p98 = float(np.percentile(single[valid], 98))
        if p98 <= p2:
            p98 = p2 + 0.01

        stretched = np.clip((single - p2) / (p98 - p2) * 255, 0, 255).astype(np.uint8)
        stretched[~valid] = 0

        img = Image.fromarray(stretched, mode="L")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        png_bytes = buf.read()

        # Cache default band
        if band == int(n_bands * 0.6):
            with open(cache_path, "wb") as f:
                f.write(png_bytes)

        return Response(
            png_bytes,
            media_type="image/png",
            headers={"Cache-Control": "public, max-age=86400"},
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to generate quickview for %s", obs_id)
        raise HTTPException(status_code=500, detail=str(e))
