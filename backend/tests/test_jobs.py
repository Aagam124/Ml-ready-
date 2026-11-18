"""Queue semantics against mongomock: atomic claim, recovery after a worker dies, retries, abandonment."""
from datetime import datetime, timedelta

from bson import ObjectId

from app.jobs.handlers import HANDLERS
from app.jobs.queue import MongoJobQueue
from app.jobs.worker import Worker


def test_claim_is_fifo_and_exclusive(database):
    q, ws = MongoJobQueue(database), ObjectId()
    a, b = q.enqueue("x", {}, ws), q.enqueue("x", {}, ws)
    assert q.claim("w1")["_id"] == a and q.claim("w2")["_id"] == b and q.claim("w3") is None


def test_dead_worker_job_is_recovered_and_zombie_cannot_write(database):
    q, ws = MongoJobQueue(database, stale_seconds=45), ObjectId()
    jid = q.enqueue("x", {}, ws)
    q.claim("dead")
    assert q.claim("w2") is None
    database.jobs.update_one({"_id": jid}, {"$set": {"heartbeat_at": datetime.utcnow() - timedelta(seconds=120)}})
    again = q.claim("w2")
    assert again["_id"] == jid and again["attempts"] == 2 and again["locked_by"] == "w2"
    assert q.heartbeat(jid, "dead", 0.9, "zombie") is False


def test_stale_worker_cannot_complete_or_fail_reclaimed_job(database):
    q, ws = MongoJobQueue(database, stale_seconds=45), ObjectId()
    jid = q.enqueue("x", {}, ws)
    q.claim("worker-a")
    database.jobs.update_one({"_id": jid}, {"$set": {"heartbeat_at": datetime.utcnow() - timedelta(seconds=120)}})
    reclaimed = q.claim("worker-b")

    assert reclaimed and reclaimed["locked_by"] == "worker-b"
    assert q.complete(jid, "worker-a", {"worker": "a"}) is False
    assert q.fail(jid, "worker-a", "stale failure", final=True) is None
    current = q.get(jid)
    assert current["status"] == "running" and current["locked_by"] == "worker-b"
    assert q.complete(jid, "worker-b", {"worker": "b"}) is True
    assert q.get(jid)["status"] == "succeeded"


def test_retry_backoff_then_permanent_failure_with_ttl_field(database):
    q, ws = MongoJobQueue(database, max_attempts=2), ObjectId()
    jid = q.enqueue("x", {}, ws)
    q.claim("w"); assert q.fail(jid, "w", "boom") == "queued"
    j = q.get(jid)
    assert j["status"] == "queued" and j["run_after"] > datetime.utcnow() and q.claim("w") is None
    database.jobs.update_one({"_id": jid}, {"$set": {"run_after": datetime.utcnow()}})
    q.claim("w"); assert q.fail(jid, "w", "boom again") == "failed"
    j = q.get(jid)
    assert j["status"] == "failed" and j["expire_at"] > datetime.utcnow()


def test_handler_failure_after_attempt_limit_marks_analysis_failed(database, settings, monkeypatch):
    analysis_id, workspace_id = ObjectId(), ObjectId()
    database.analyses.insert_one({"_id": analysis_id, "status": "running"})
    worker = Worker(database, settings)
    worker.queue.max_attempts = 1
    worker.queue.enqueue("analysis", {"analysis_id": str(analysis_id)}, workspace_id)

    def broken_handler(database, settings, job, progress):
        raise RuntimeError("handler failed")

    monkeypatch.setitem(HANDLERS, "analysis", broken_handler)
    assert worker.run_once()

    failed_job = database.jobs.find_one({"type": "analysis"})
    failed_analysis = database.analyses.find_one({"_id": analysis_id})
    assert failed_job["status"] == "failed"
    assert failed_analysis["status"] == "failed" and "handler failed" in failed_analysis["error"]


def test_worker_crash_exhaustion_marks_analysis_failed(database, settings):
    analysis_id, workspace_id = ObjectId(), ObjectId()
    database.analyses.insert_one({"_id": analysis_id, "status": "running"})
    worker = Worker(database, settings)
    worker.queue.max_attempts = 1
    jid = worker.queue.enqueue("analysis", {"analysis_id": str(analysis_id)}, workspace_id)
    assert worker.queue.claim("dead-worker")["attempts"] == 1
    database.jobs.update_one({"_id": jid}, {"$set": {"heartbeat_at": datetime.utcnow() - timedelta(seconds=999)}})

    assert worker.run_once()
    failed_job = worker.queue.get(jid)
    failed_analysis = database.analyses.find_one({"_id": analysis_id})
    assert failed_job["status"] == "failed"
    assert failed_analysis["status"] == "failed" and "repeated worker failures" in failed_analysis["error"]


def test_unknown_job_type_marks_associated_analysis_failed(database, settings):
    analysis_id, workspace_id = ObjectId(), ObjectId()
    database.analyses.insert_one({"_id": analysis_id, "status": "queued"})
    worker = Worker(database, settings)
    jid = worker.queue.enqueue("unknown", {"analysis_id": str(analysis_id)}, workspace_id)

    assert worker.run_once()
    failed_job = worker.queue.get(jid)
    failed_analysis = database.analyses.find_one({"_id": analysis_id})
    assert failed_job["status"] == "failed"
    assert failed_analysis["status"] == "failed" and "Unknown job type" in failed_analysis["error"]
