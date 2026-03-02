from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from analysis.hirise_landforms.models import ClassifyRequest

router = APIRouter(prefix="/api/hirise-landforms", tags=["HiRISE Landform Classification"])

_pipeline: object | None = None
_job_queue: object | None = None


def _to_dict(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, dict):
            return dumped
    return {"value": value}


def _get_pipeline() -> object:
    global _pipeline
    if _pipeline is None:
        try:
            from analysis.hirise_landforms.pipeline import HiriseLandformPipeline
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"HiRISE landform pipeline unavailable: {exc}")
        _pipeline = HiriseLandformPipeline()
    return _pipeline


def _get_job_queue() -> object:
    global _job_queue
    if _job_queue is None:
        try:
            from analysis.hirise_landforms.job_queue import LandformJobQueue
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"HiRISE job queue unavailable: {exc}")
        _job_queue = LandformJobQueue()
    return _job_queue


def _find_callable(target: object, names: tuple[str, ...]):
    for name in names:
        method = getattr(target, name, None)
        if callable(method):
            return method
    return None


@router.post("/classify")
async def classify_hirise_landforms(request: ClassifyRequest):
    queue = _get_job_queue()
    pipeline = _get_pipeline()
    submit = _find_callable(queue, ("submit", "submit_job", "enqueue", "enqueue_job"))
    if submit is None:
        raise HTTPException(status_code=503, detail="HiRISE classify submission endpoint unavailable")

    try:
        submitted = submit(request, pipeline)
    except TypeError:
        submitted = submit(req=request, pipeline=pipeline)

    data = _to_dict(submitted)
    job_id = data.get("job_id")
    if not isinstance(job_id, str) or not job_id:
        raise HTTPException(status_code=500, detail="Failed to create classification job")

    estimate = data.get("estimated_seconds", 20)
    if not isinstance(estimate, int):
        try:
            if isinstance(estimate, (float, str)):
                estimate = int(estimate)
            else:
                estimate = 20
        except Exception:
            estimate = 20

    return JSONResponse(content={"job_id": job_id, "status": data.get("status", "queued"), "estimated_seconds": estimate})


@router.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    queue = _get_job_queue()
    getter = _find_callable(queue, ("get_job", "get_job_status", "status", "get_status"))
    if getter is None:
        raise HTTPException(status_code=503, detail="HiRISE job status endpoint unavailable")
    status_data = getter(job_id)
    if status_data is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    response = _to_dict(status_data)
    response.setdefault("job_id", job_id)
    return JSONResponse(content=response)


@router.get("/status")
async def get_hirise_landforms_status():
    queue = _get_job_queue()
    pipeline = _get_pipeline()

    queue_length = 0
    for name in ("queue_length", "pending_count"):
        value = getattr(queue, name, None)
        if isinstance(value, int):
            queue_length = value
            break
    if queue_length == 0:
        queue_length_method = _find_callable(queue, ("get_queue_length",))
        if queue_length_method is not None:
            raw_length = queue_length_method()
            if isinstance(raw_length, int):
                queue_length = raw_length

    active_job = None
    for name in ("active_job", "current_job_id", "processing_job"):
        value = getattr(queue, name, None)
        if isinstance(value, str) and value:
            active_job = value
            break

    pipeline_status: dict[str, object] = {}
    status_getter = _find_callable(pipeline, ("status", "get_status", "runtime_status", "get_runtime_status"))
    if status_getter is not None:
        pipeline_status = _to_dict(status_getter())

    return JSONResponse(
        content={
            "models_loaded": pipeline_status.get("models_loaded", []),
            "device": pipeline_status.get("device", "unknown"),
            "memory_mb": pipeline_status.get("memory_mb", 0.0),
            "queue_length": queue_length,
            "active_job": active_job,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )


@router.get("/classify/{product_id}")
async def get_cached_classification(product_id: str):
    queue = _get_job_queue()
    pipeline = _get_pipeline()

    for target in (queue, pipeline):
        cached_getter = _find_callable(target, ("get_cached_result", "get_cached", "cached_result"))
        if cached_getter is None:
            continue
        cached = cached_getter(product_id)
        if cached is not None:
            return JSONResponse(content=_to_dict(cached))

    raise HTTPException(status_code=404, detail=f"No cached classification for product_id={product_id}")
