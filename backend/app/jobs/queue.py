"""MongoDB-backed job queue: atomic claiming, heartbeats, retries with backoff, stale-job recovery, TTL cleanup.

The `JobQueue` interface is deliberately tiny (enqueue / claim / heartbeat / complete / fail / get) so a
Redis or Arq implementation can replace this class without touching handlers, the worker or the API.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Protocol

from bson import ObjectId
from pymongo import ReturnDocument

from app.db import Database


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class JobQueueProtocol(Protocol):
    def enqueue(self, type: str, payload: Dict[str, Any], workspace_id: ObjectId) -> ObjectId: ...
    def claim(self, worker_id: str) -> Optional[Dict[str, Any]]: ...
    def heartbeat(self, job_id: ObjectId, worker_id: str, progress: float, message: str) -> bool: ...
    def complete(self, job_id: ObjectId, result: Optional[Dict[str, Any]] = None) -> None: ...
    def fail(self, job_id: ObjectId, error: str) -> None: ...
    def get(self, job_id: ObjectId) -> Optional[Dict[str, Any]]: ...


class MongoJobQueue:
    def __init__(self, database: Database, max_attempts: int = 3, stale_seconds: int = 45, ttl_days: int = 7):
        self.col = database.jobs
        self.max_attempts, self.stale_seconds, self.ttl_days = max_attempts, stale_seconds, ttl_days

    def enqueue(self, type: str, payload: Dict[str, Any], workspace_id: ObjectId) -> ObjectId:
        now = _now()
        doc = {"_id": ObjectId(), "type": type, "payload": payload, "workspace_id": workspace_id, "status": "queued",
               "progress": 0.0, "message": "Queued", "attempts": 0, "locked_by": None, "heartbeat_at": None, "error": None,
               "result": None, "run_after": now, "created_at": now, "updated_at": now}
        self.col.insert_one(doc)
        return doc["_id"]

    def claim(self, worker_id: str) -> Optional[Dict[str, Any]]:
        """Atomically take one runnable job: queued and due, OR running but its worker stopped heartbeating."""
        now = _now()
        stale = now - timedelta(seconds=self.stale_seconds)
        job = self.col.find_one_and_update(
            {"$or": [{"status": "queued", "run_after": {"$lte": now}},
                     {"status": "running", "heartbeat_at": {"$lt": stale}}]},
            {"$set": {"status": "running", "locked_by": worker_id, "heartbeat_at": now, "updated_at": now,
                      "message": "Started"}, "$inc": {"attempts": 1}},
            sort=[("created_at", 1)], return_document=ReturnDocument.AFTER)
        if job and job["attempts"] > self.max_attempts:       # a job that keeps killing workers must not loop forever
            self.fail(job["_id"], "Gave up after repeated worker failures.", final=True)
            return self.claim(worker_id)
        return job

    def heartbeat(self, job_id: ObjectId, worker_id: str, progress: float, message: str) -> bool:
        now = _now()
        res = self.col.update_one({"_id": job_id, "locked_by": worker_id, "status": "running"},
                                  {"$set": {"heartbeat_at": now, "updated_at": now, "progress": round(float(progress), 4), "message": message}})
        return res.matched_count == 1       # False => someone else took the job over; stop working on it

    def complete(self, job_id: ObjectId, result: Optional[Dict[str, Any]] = None) -> None:
        now = _now()
        self.col.update_one({"_id": job_id}, {"$set": {"status": "succeeded", "progress": 1.0, "message": "Done", "result": result,
                                                      "updated_at": now, "finished_at": now,
                                                      "expire_at": now + timedelta(days=self.ttl_days)}})

    def fail(self, job_id: ObjectId, error: str, final: bool = False) -> None:
        now = _now()
        job = self.col.find_one({"_id": job_id})
        if job and not final and job["attempts"] < self.max_attempts:
            backoff = 2 ** job["attempts"]            # 2s, 4s, ...
            self.col.update_one({"_id": job_id}, {"$set": {"status": "queued", "locked_by": None, "heartbeat_at": None,
                                                          "run_after": now + timedelta(seconds=backoff), "error": error,
                                                          "message": f"Retrying after error (attempt {job['attempts']} of {self.max_attempts})",
                                                          "updated_at": now}})
            return
        self.col.update_one({"_id": job_id}, {"$set": {"status": "failed", "error": error, "message": "Failed", "updated_at": now,
                                                      "finished_at": now, "expire_at": now + timedelta(days=self.ttl_days)}})

    def get(self, job_id: ObjectId) -> Optional[Dict[str, Any]]:
        return self.col.find_one({"_id": job_id})
