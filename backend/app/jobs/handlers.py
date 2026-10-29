"""Job handlers. Each takes (database, settings, job, progress) and returns a small result dict.

Handlers are idempotent per attempt: they write their outputs at the end (findings are replaced, never appended
twice), so a retried or recovered job cannot double-count anything.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List

from bson import ObjectId

from app.cleaning.executor import plan_options, run_plan
from app.config import Settings
from app.db import Database
from app.engine.config import EngineConfig
from app.engine.pipeline import run_analysis
from app.modeling.runner import run_before_after
from app.storage import create_derived_version, load_version_df, now

log = logging.getLogger("mlready.jobs")
Progress = Callable[[float, str], None]


def handle_analysis(database: Database, settings: Settings, job: Dict[str, Any], progress: Progress) -> Dict[str, Any]:
    aid = ObjectId(job["payload"]["analysis_id"])
    analysis = database.analyses.find_one({"_id": aid})
    if analysis is None:
        raise LookupError("Analysis no longer exists.")
    database.analyses.update_one({"_id": aid}, {"$set": {"status": "running"}})
    version = database.versions.find_one({"_id": analysis["version_id"]})
    progress(0.02, "Loading dataset")
    df = load_version_df(database, version)
    cfg = EngineConfig(**{k: v for k, v in analysis["config"].items() if k in EngineConfig.__dataclass_fields__})
    t0 = time.perf_counter()
    result = run_analysis(df, analysis.get("target_column"), analysis.get("task_type"), profile=version.get("profile"), config=cfg,
                          progress=lambda f, m: progress(0.05 + 0.9 * f, m))
    progress(0.97, "Saving findings")
    database.findings.delete_many({"analysis_id": aid})
    docs = [{"_id": ObjectId(), "analysis_id": aid, "workspace_id": analysis["workspace_id"], **f} for f in result["findings"]]
    if docs:
        database.findings.insert_many(docs)
    database.analyses.update_one({"_id": aid}, {"$set": {
        "status": "done", "score": result["score"], "summary": result["summary"], "fix_first": result["fix_first"],
        "target_candidates": result["target_candidates"], "task_type": result["task_type"], "sampling": result["sampling"],
        "check_timings_ms": result["check_timings_ms"], "check_errors": result["check_errors"], "checks_skipped": result["checks_skipped"],
        "leakage_columns": result["leakage_columns"], "n_findings": len(docs), "config_snapshot": result["config_snapshot"],
        "duration_ms": round((time.perf_counter() - t0) * 1000), "finished_at": now(), "score_version": result["score_version"]}})
    return {"analysis_id": str(aid), "score": result["score"]["value"]}


def handle_apply_plan(database: Database, settings: Settings, job: Dict[str, Any], progress: Progress) -> Dict[str, Any]:
    pid = ObjectId(job["payload"]["plan_id"])
    plan = database.plans.find_one({"_id": pid})
    if plan is None:
        raise LookupError("Plan no longer exists.")
    analysis = database.analyses.find_one({"_id": plan["analysis_id"]})
    dataset = database.datasets.find_one({"_id": plan["dataset_id"]})
    parent = database.versions.find_one({"_id": plan["version_id"]})
    database.plans.update_one({"_id": pid}, {"$set": {"status": "applying"}})
    progress(0.1, "Loading dataset")
    df = load_version_df(database, parent)
    progress(0.3, "Applying steps")
    cleaned, _, report = run_plan(df, plan["steps"], analysis.get("target_column"), analysis.get("task_type"))
    progress(0.7, "Saving new version")
    ver = create_derived_version(database, dataset, parent, cleaned, step=f"cleaning_plan:{pid}", created_by=plan.get("created_by"),
                                 extra={"plan_id": pid})
    # Chain a fresh analysis of the new version so the user sees the score change immediately.
    new_analysis = {"_id": ObjectId(), "workspace_id": plan["workspace_id"], "dataset_id": dataset["_id"], "version_id": ver["_id"],
                    "target_column": analysis.get("target_column"), "task_type": analysis.get("task_type"), "config": analysis["config"],
                    "status": "queued", "compared_to": analysis["_id"], "created_at": now(), "created_by": plan.get("created_by")}
    database.analyses.insert_one(new_analysis)
    from app.jobs.queue import MongoJobQueue
    q = MongoJobQueue(database, settings.job_max_attempts, settings.job_stale_seconds, settings.job_ttl_days)
    jid = q.enqueue("analysis", {"analysis_id": str(new_analysis["_id"]), "dataset_id": str(dataset["_id"])}, plan["workspace_id"])
    database.analyses.update_one({"_id": new_analysis["_id"]}, {"$set": {"job_id": jid}})
    database.plans.update_one({"_id": pid}, {"$set": {"status": "applied", "result_version_id": ver["_id"], "apply_report": report,
                                                     "result_analysis_id": new_analysis["_id"], "applied_at": now()}})
    return {"version_id": str(ver["_id"]), "analysis_id": str(new_analysis["_id"]), "job_id": str(jid), "n_rows": ver["n_rows"]}


def handle_model_run(database: Database, settings: Settings, job: Dict[str, Any], progress: Progress) -> Dict[str, Any]:
    mid = ObjectId(job["payload"]["model_run_id"])
    run = database.model_runs.find_one({"_id": mid})
    if run is None:
        raise LookupError("Model run no longer exists.")
    plan = database.plans.find_one({"_id": run["plan_id"]})
    analysis = database.analyses.find_one({"_id": run["analysis_id"]})
    version = database.versions.find_one({"_id": plan["version_id"]})
    database.model_runs.update_one({"_id": mid}, {"$set": {"status": "running"}})
    progress(0.05, "Loading dataset")
    before = load_version_df(database, version)
    after, _, _ = run_plan(before, plan["steps"], analysis["target_column"], analysis["task_type"])   # same row index as `before`
    result = run_before_after(before, after, analysis["target_column"], analysis["task_type"], plan_options(plan["steps"]),
                              analysis.get("leakage_columns", []), seed=run.get("seed", 42),
                              progress=lambda f, m: progress(0.1 + 0.88 * f, m))
    database.model_runs.update_one({"_id": mid}, {"$set": {"status": "done", "result": result, "finished_at": now(),
                                                          "leakage_columns_excluded": result["comparison"]["leakage_columns_excluded"]}})
    return {"model_run_id": str(mid), "verdict": result["comparison"]["verdict"]}


HANDLERS = {"analysis": handle_analysis, "apply_plan": handle_apply_plan, "model_run": handle_model_run}


def mark_target_failed(database: Database, job: Dict[str, Any], error: str) -> None:
    """When a job finally fails, the object the user is looking at must say so (never a spinner forever)."""
    p = job.get("payload", {})
    for key, col in (("analysis_id", database.analyses), ("plan_id", database.plans), ("model_run_id", database.model_runs)):
        if key in p:
            col.update_one({"_id": ObjectId(p[key])}, {"$set": {"status": "failed", "error": error[:500]}})
