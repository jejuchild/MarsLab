from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from analysis.hirise_landforms.models import ClassifyRequest

router = APIRouter(prefix="/api/hirise-landforms", tags=["HiRISE Landform Classification"])

_job_queue: Any = None


def _get_job_queue() -> Any:
    global _job_queue
    if _job_queue is not None:
        return _job_queue
    try:
        from analysis.hirise_landforms.job_queue import LandformJobQueue
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"HiRISE job queue unavailable: {exc}")
    _job_queue = LandformJobQueue()
    # Start the async worker
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_job_queue.start_worker())
    except RuntimeError:
        pass
    return _job_queue


def _flatten_result(raw: dict[str, Any]) -> dict[str, Any]:
    """
    Flatten the nested ClassifyResult into the shape the frontend expects:
    { product_id, model, top_class, confidence, classes[], processing_time_s, heatmap_url, agent_reasoning, ... }
    """
    result = raw.get("result")
    if result is None:
        return raw

    # If result is already a dict (from model_dump), flatten prediction
    if isinstance(result, dict):
        prediction = result.get("prediction", {})
        probabilities = prediction.get("probabilities", {})

        # Build classes array for frontend
        classes = []
        class_descriptions = {
            "LDA": "Lobate Debris Apron",
            "LVF": "Lineated Valley Fill",
            "CCF": "Concentric Crater Fill",
            "GLF": "Glacier-Like Form",
            "BACKGROUND": "Background",
        }
        for class_code, prob in probabilities.items():
            classes.append({
                "class_code": class_code,
                "class_name": class_descriptions.get(class_code, class_code),
                "probability": prob,
            })

        flat_result = {
            "product_id": result.get("product_id", ""),
            "model": result.get("model_used", "v2"),
            "top_class": prediction.get("top_class", "BACKGROUND"),
            "confidence": prediction.get("confidence", 0.0),
            "classes": classes,
            "heatmap_url": result.get("heatmap_url"),
            "processing_time_s": result.get("processing_time_s", 0.0),
            "agent_reasoning": result.get("agent_reasoning"),
            "num_tiles": result.get("num_tiles", 0),
            "device": result.get("device", "cpu"),
        }

        return {**raw, "result": flat_result}
    return raw


@router.post("/classify")
async def classify_hirise_landforms(request: ClassifyRequest):
    queue = _get_job_queue()

    try:
        job_id = queue.submit(request)
    except ValueError as exc:
        raise HTTPException(status_code=429, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to submit job: {exc}")

    # Ensure worker is running
    try:
        await queue.start_worker()
    except Exception:
        pass

    return JSONResponse(content={
        "job_id": job_id,
        "status": "queued",
        "estimated_seconds": 30,
    })


@router.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    queue = _get_job_queue()

    try:
        status = queue.get_status(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    response = status.model_dump() if hasattr(status, "model_dump") else dict(status)
    response = _flatten_result(response)
    return JSONResponse(content=response)


@router.get("/status")
async def get_hirise_landforms_status():
    queue = _get_job_queue()

    queue_length = queue._queue.qsize() if hasattr(queue, "_queue") else 0
    active_job = getattr(queue, "active_job", None)

    pipeline_status: dict[str, Any] = {}
    pipeline = getattr(queue, "_pipeline", None)
    if pipeline is not None and hasattr(pipeline, "status"):
        try:
            pipeline_status = pipeline.status()
        except Exception:
            pass

    return JSONResponse(content={
        "models_loaded": pipeline_status.get("models_loaded", []),
        "device": pipeline_status.get("device", "unknown"),
        "memory_mb": pipeline_status.get("memory_mb", 0.0),
        "queue_length": queue_length,
        "active_job": active_job,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


@router.get("/classify/{product_id}")
async def get_cached_classification(product_id: str):
    queue = _get_job_queue()

    # Check cache in job queue
    cache = getattr(queue, "_cache", {})
    for key, (expires_at, result) in cache.items():
        if key[0] == product_id and expires_at > datetime.now(timezone.utc):
            result_dict = result.model_dump() if hasattr(result, "model_dump") else dict(result)
            flat = _flatten_result({"result": result_dict})
            return JSONResponse(content=flat.get("result", result_dict))

    raise HTTPException(status_code=404, detail=f"No cached classification for product_id={product_id}")
