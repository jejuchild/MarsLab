from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from .models import ClassifyRequest, ClassifyResult, JobStatus
from .pipeline import HiriseLandformPipeline


class LandformJobQueue:
    max_depth: int
    _queue: asyncio.Queue[tuple[str, ClassifyRequest]]
    _jobs: dict[str, JobStatus]
    _pipeline: HiriseLandformPipeline
    _worker_task: asyncio.Task[None] | None
    _cache: dict[tuple[str, str], tuple[datetime, ClassifyResult]]
    _cache_ttl: timedelta
    active_job: str | None

    def __init__(self, max_depth: int = 10) -> None:
        self.max_depth = max_depth
        self._queue = asyncio.Queue(maxsize=max_depth)
        self._jobs = {}
        self._pipeline = HiriseLandformPipeline()
        self._worker_task = None
        self._cache = {}
        self._cache_ttl = timedelta(hours=24)
        self.active_job = None

    def submit(self, request: ClassifyRequest) -> str:
        self._prune_cache()
        key = (request.product_id, request.model)
        cached = self._cache.get(key)

        job_id = str(uuid4())
        now = datetime.now(timezone.utc)
        if cached is not None and cached[0] > now:
            self._jobs[job_id] = JobStatus(
                job_id=job_id,
                status="completed",
                progress=1.0,
                submitted_at=now.isoformat(),
                result=cached[1],
                error=None,
            )
            return job_id

        if self._queue.qsize() >= self.max_depth:
            raise ValueError(f"HiRISE landform queue is full (max {self.max_depth} jobs).")

        self._jobs[job_id] = JobStatus(
            job_id=job_id,
            status="queued",
            progress=0.0,
            submitted_at=now.isoformat(),
            result=None,
            error=None,
        )
        self._queue.put_nowait((job_id, request))
        return job_id

    def get_status(self, job_id: str) -> JobStatus:
        status = self._jobs.get(job_id)
        if status is None:
            raise KeyError(f"Unknown job_id '{job_id}'.")
        return status

    async def start_worker(self) -> None:
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._worker_loop())

    async def _worker_loop(self) -> None:
        while True:
            job_id, request = await self._queue.get()
            status = self._jobs[job_id]
            self.active_job = job_id
            status.status = "processing"
            status.progress = 0.1
            try:
                result = await asyncio.to_thread(self._pipeline.classify, request)
                status.status = "completed"
                status.progress = 1.0
                status.result = result
                status.error = None
                self._cache[(request.product_id, request.model)] = (
                    datetime.now(timezone.utc) + self._cache_ttl,
                    result,
                )
            except Exception as exc:
                status.status = "failed"
                status.progress = 1.0
                status.result = None
                status.error = str(exc)
            finally:
                self.active_job = None
                self._queue.task_done()

    def _prune_cache(self) -> None:
        now = datetime.now(timezone.utc)
        expired = [key for key, (expires_at, _) in self._cache.items() if expires_at <= now]
        for key in expired:
            _ = self._cache.pop(key, None)


landform_job_queue = LandformJobQueue()
