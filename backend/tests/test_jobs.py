"""Queue semantics against mongomock: atomic claim, recovery after a worker dies, retries, abandonment."""
from datetime import datetime, timedelta

from bson import ObjectId

from app.jobs.queue import MongoJobQueue


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


def test_retry_backoff_then_permanent_failure_with_ttl_field(database):
    q, ws = MongoJobQueue(database, max_attempts=2), ObjectId()
    jid = q.enqueue("x", {}, ws)
    q.claim("w"); q.fail(jid, "boom")
    j = q.get(jid)
    assert j["status"] == "queued" and j["run_after"] > datetime.utcnow() and q.claim("w") is None
    database.jobs.update_one({"_id": jid}, {"$set": {"run_after": datetime.utcnow()}})
    q.claim("w"); q.fail(jid, "boom again")
    j = q.get(jid)
    assert j["status"] == "failed" and j["expire_at"] > datetime.utcnow()


def test_job_that_keeps_killing_workers_is_abandoned(database):
    q, ws = MongoJobQueue(database, max_attempts=2), ObjectId()
    jid = q.enqueue("x", {}, ws)
    for _ in range(4):
        database.jobs.update_one({"_id": jid}, {"$set": {"heartbeat_at": datetime.utcnow() - timedelta(seconds=999)}})
        q.claim("w")
    assert q.get(jid)["status"] == "failed"
