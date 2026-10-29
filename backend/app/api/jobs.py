"""Job status and a Server-Sent-Events progress stream."""
from __future__ import annotations

import json
import time
from typing import Iterator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.api.deps import get_workspace
from app.db import Database, get_database
from app.errors import APIError
from app.serialization import parse_id, serialize

router = APIRouter(tags=["jobs"])
PUBLIC = {"type": 1, "status": 1, "progress": 1, "message": 1, "attempts": 1, "error": 1, "result": 1, "created_at": 1, "updated_at": 1}


def _job(database: Database, ws: dict, job_id: str) -> dict:
    j = database.jobs.find_one({"_id": parse_id(job_id, "Job"), "workspace_id": ws["_id"]}, PUBLIC)
    if j is None:
        raise APIError(404, "not_found", "Job not found.")
    return j


@router.get("/jobs/{job_id}")
def status(job_id: str, database: Database = Depends(get_database), ws: dict = Depends(get_workspace)):
    return serialize(_job(database, ws, job_id))


@router.get("/jobs/{job_id}/events")
def events(job_id: str, database: Database = Depends(get_database), ws: dict = Depends(get_workspace)):
    """SSE: one `progress` event whenever something changes, then a final `done`/`failed` event. Polls MongoDB,
    so it works no matter which API process or worker handles the job (and survives worker restarts)."""
    _job(database, ws, job_id)          # authorise before streaming

    def stream() -> Iterator[str]:
        last, started, ticks = None, time.monotonic(), 0
        while time.monotonic() - started < 3600:
            j = serialize(database.jobs.find_one({"_id": parse_id(job_id, "Job"), "workspace_id": ws["_id"]}, PUBLIC))
            if j is None:
                yield "event: failed\ndata: {\"message\": \"Job disappeared.\"}\n\n"
                return
            snap = (j["status"], j["progress"], j["message"], j["attempts"])
            if snap != last:
                last = snap
                yield f"event: progress\ndata: {json.dumps(j)}\n\n"
            if j["status"] in ("succeeded", "failed"):
                yield f"event: {'done' if j['status'] == 'succeeded' else 'failed'}\ndata: {json.dumps(j)}\n\n"
                return
            ticks += 1
            if ticks % 20 == 0:
                yield ": keep-alive\n\n"
            time.sleep(0.5)

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
