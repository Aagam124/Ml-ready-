"""Worker process: python -m app.jobs.worker

Claims jobs atomically, heartbeats while working, retries with backoff, recovers jobs from dead workers, and
shuts down gracefully on SIGTERM/SIGINT (finishes or releases the current job). Never logs row values.
"""
from __future__ import annotations

import logging
import os
import signal
import socket
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from typing import Any, Dict

from app.config import Settings, get_settings
from app.db import Database, build_database, ensure_indexes
from app.jobs.handlers import HANDLERS, mark_target_failed
from app.jobs.queue import MongoJobQueue

log = logging.getLogger("mlready.worker")


class Worker:
    def __init__(self, database: Database, settings: Settings):
        self.db, self.settings = database, settings
        self.queue = MongoJobQueue(database, settings.job_max_attempts, settings.job_stale_seconds, settings.job_ttl_days)
        self.id = f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:6]}"
        self.stop = threading.Event()
        self._last_janitor = 0.0

    def run_once(self) -> bool:
        job = self.queue.claim(self.id)
        if job is None:
            return False
        if job["attempts"] > self.queue.max_attempts:
            self._fail_job(job, "Gave up after repeated worker failures.", final=True)
            return True
        handler = HANDLERS.get(job["type"])
        if handler is None:
            self._fail_job(job, f"Unknown job type '{job['type']}'", final=True)
            return True
        lost = threading.Event()
        last = {"t": 0.0}

        def progress(fraction: float, message: str) -> None:
            if time.monotonic() - last["t"] < 0.4 and fraction < 0.99:       # throttle writes
                return
            last["t"] = time.monotonic()
            if not self.queue.heartbeat(job["_id"], self.id, fraction, message):
                lost.set()
                raise RuntimeError("Job was taken over by another worker.")

        # Heartbeat thread keeps the lease alive during long single steps (e.g. fitting a model).
        beat_stop = threading.Event()

        def beat() -> None:
            while not beat_stop.wait(max(1.0, self.settings.job_stale_seconds / 3)):
                cur = self.queue.get(job["_id"]) or {}
                if not self.queue.heartbeat(job["_id"], self.id, cur.get("progress", 0.0), cur.get("message", "Working")):
                    lost.set()
                    return
        t = threading.Thread(target=beat, daemon=True)
        t.start()
        try:
            log.info("job %s type=%s attempt=%s", job["_id"], job["type"], job["attempts"])
            result = handler(self.db, self.settings, job, progress)
            if not lost.is_set():
                if not self.queue.complete(job["_id"], self.id, result):
                    lost.set()
                    log.warning("job %s completion ignored because this worker no longer owns it", job["_id"])
        except Exception as e:       # noqa: BLE001 - the job record carries the error; never crash the worker
            err = f"{type(e).__name__}: {e}"
            log.error("job %s failed: %s\n%s", job["_id"], err, traceback.format_exc())
            if not lost.is_set():
                self._fail_job(job, err)
        finally:
            beat_stop.set()
        return True

    def _fail_job(self, job: Dict[str, Any], error: str, final: bool = False) -> None:
        status = self.queue.fail(job["_id"], self.id, error, final=final)
        if status == "failed":
            mark_target_failed(self.db, job, error)

    def janitor(self) -> None:
        """Delete expired demo (guest) workspaces with all their data, every 10 minutes."""
        if time.monotonic() - self._last_janitor < 600:
            return
        self._last_janitor = time.monotonic()
        from app.storage import delete_dataset_everything
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        self.db.reports.update_many({"share_token": {"$ne": None}, "share_expires_at": {"$lt": now}},
                                    {"$set": {"share_token": None, "share_expires_at": None}})
        for ws in list(self.db.workspaces.find({"expires_at": {"$lt": now}})):
            for ds in list(self.db.datasets.find({"workspace_id": ws["_id"]})):
                delete_dataset_everything(self.db, ds)
            self.db.users.delete_many({"_id": {"$in": [m["user_id"] for m in ws.get("members", [])]}, "is_guest": True})
            self.db.workspaces.delete_one({"_id": ws["_id"]})
            log.info("expired demo workspace %s removed", ws["_id"])

    def serve(self) -> None:
        log.info("worker %s started", self.id)
        while not self.stop.is_set():
            try:
                self.janitor()
                if not self.run_once():
                    self.stop.wait(self.settings.worker_poll_seconds)
            except Exception:      # noqa: BLE001 - e.g. MongoDB briefly unreachable
                log.exception("worker loop error; retrying")
                self.stop.wait(2.0)
        log.info("worker %s stopped cleanly", self.id)


def main() -> None:
    from app.main import configure_logging
    configure_logging()
    settings = get_settings()
    database = build_database(settings=settings)
    ensure_indexes(database)
    worker = Worker(database, settings)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: worker.stop.set())       # finish the current job, then exit
    worker.serve()


if __name__ == "__main__":
    main()
