from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from litestar import Controller, get
from litestar.response import Response

from omoai.api import services as _svc
from omoai.api.models import OutputFormatParams, PipelineRequest


@dataclass
class JobRecord:
    id: str
    status: str = "PENDING"  # PENDING | RUNNING | SUCCEEDED | FAILED
    submitted_at: float = field(default_factory=lambda: time.time())
    started_at: float | None = None
    ended_at: float | None = None
    result: dict[str, Any] | None = None
    error: str | None = None


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, JobRecord] = {}
        self._lock = asyncio.Lock()
        # Track background tasks to satisfy linter and allow optional lifecycle management
        self._tasks: set[asyncio.Task[None]] = set()

    async def submit_pipeline_job(
        self,
        file_bytes: bytes,
        filename: str,
        content_type: str,
        output_params: OutputFormatParams | None,
    ) -> str:
        job_id = uuid.uuid4().hex
        record = JobRecord(id=job_id)
        async with self._lock:
            self._jobs[job_id] = record

        async def _runner() -> None:
            record.started_at = time.time()
            record.status = "RUNNING"
            try:
                upload = _svc._BytesUploadFile(
                    data=file_bytes, filename=filename, content_type=content_type
                )
                req = PipelineRequest(audio_file=upload)
                result = await _svc.run_full_pipeline(req, output_params)
                record.result = result.model_dump()
                record.status = "SUCCEEDED"
            except Exception as e:  # noqa: BLE001 - we want error surface
                record.error = str(e)
                record.status = "FAILED"
            finally:
                record.ended_at = time.time()

        # Fire and forget, but keep a reference per linter guidance
        task: asyncio.Task[None] = asyncio.create_task(_runner())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return job_id

    async def get(self, job_id: str) -> JobRecord | None:
        async with self._lock:
            return self._jobs.get(job_id)


job_manager = JobManager()


class JobsController(Controller):
    path = "/jobs"

    @get(path="/{job_id:str}")
    async def get_job(self, job_id: str) -> Response:
        rec = await job_manager.get(job_id)
        if rec is None:
            return Response({"error": "job not found", "job_id": job_id}, status_code=404)
        body: dict[str, Any] = {
            "job_id": rec.id,
            "status": rec.status.lower(),
            "submitted_at": rec.submitted_at,
            "started_at": rec.started_at,
            "ended_at": rec.ended_at,
        }
        if rec.status == "SUCCEEDED" and rec.result is not None:
            body["result"] = rec.result
        if rec.status == "FAILED" and rec.error is not None:
            body["error"] = rec.error
        return Response(body, status_code=200)

