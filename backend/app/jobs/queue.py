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
    def complete(self, job_id: ObjectId, worker_id: str, result: Optional[Dict[str, Any]] = None) -> bool: ...
    def fail(self, job_id: ObjectId, worker_id: str, error: str, final: bool = False) -> Optional[str]: ...
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
        return job

    def heartbeat(self, job_id: ObjectId, worker_id: str, progress: float, message: str) -> bool:
        now = _now()
        res = self.col.update_one({"_id": job_id, "locked_by": worker_id, "status": "running"},
                                  {"$set": {"heartbeat_at": now, "updated_at": now, "progress": round(float(progress), 4), "message": message}})
        return res.matched_count == 1       # False => someone else took the job over; stop working on it

    def complete(self, job_id: ObjectId, worker_id: str, result: Optional[Dict[str, Any]] = None) -> bool:
        now = _now()
        res = self.col.update_one({"_id": job_id, "locked_by": worker_id, "status": "running"},
                                  {"$set": {"status": "succeeded", "progress": 1.0, "message": "Done", "result": result,
                                            "locked_by": None, "heartbeat_at": None, "updated_at": now, "finished_at": now,
                                            "expire_at": now + timedelta(days=self.ttl_days)}})
        return res.matched_count == 1

    def fail(self, job_id: ObjectId, worker_id: str, error: str, final: bool = False) -> Optional[str]:
        now = _now()
        owned = {"_id": job_id, "locked_by": worker_id, "status": "running"}
        job = self.col.find_one(owned)
        if job is None:
            return None
        if not final and job["attempts"] < self.max_attempts:
            backoff = 2 ** job["attempts"]            # 2s, 4s, ...
            res = self.col.update_one(owned, {"$set": {"status": "queued", "locked_by": None, "heartbeat_at": None,
                                                       "run_after": now + timedelta(seconds=backoff), "error": error,
                                                       "message": f"Retrying after error (attempt {job['attempts']} of {self.max_attempts})",
                                                       "updated_at": now}})
            return "queued" if res.matched_count == 1 else None
        res = self.col.update_one(owned, {"$set": {"status": "failed", "locked_by": None, "heartbeat_at": None,
                                                   "error": error, "message": "Failed", "updated_at": now,
                                                   "finished_at": now, "expire_at": now + timedelta(days=self.ttl_days)}})
        return "failed" if res.matched_count == 1 else None

    def get(self, job_id: ObjectId) -> Optional[Dict[str, Any]]:
        return self.col.find_one({"_id": job_id})
