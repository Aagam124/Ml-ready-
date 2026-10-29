"""Analyses: start (background job), read, filter findings, compare two runs ("What changed?")."""
from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional

from bson import ObjectId
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from app.api.datasets import _get_dataset, _get_version
from app.api.deps import audit, editor_workspace, get_workspace
from app.config import Settings, get_settings
from app.db import Database, get_database
from app.engine.config import EngineConfig
from app.engine.target import suggest_targets_from_profile, task_type_from_profile
from app.errors import APIError
from app.jobs.queue import MongoJobQueue
from app.serialization import parse_id, serialize
from app.storage import now

router = APIRouter(tags=["analyses"])
SEVERITY_ORDER = ["critical", "high", "medium", "low", "info"]


class AnalysisRequest(BaseModel):
    version_id: Optional[str] = None
    target_column: Optional[str] = None
    task_type: Optional[str] = None


def queue_for(database: Database, settings: Settings) -> MongoJobQueue:
    return MongoJobQueue(database, settings.job_max_attempts, settings.job_stale_seconds, settings.job_ttl_days)


def get_analysis(database: Database, ws: dict, analysis_id: str) -> dict:
    a = database.analyses.find_one({"_id": parse_id(analysis_id, "Analysis"), "workspace_id": ws["_id"]})
    if a is None:
        raise APIError(404, "not_found", "Analysis not found.")
    return a


@router.get("/datasets/{dataset_id}/target-suggestions")
def target_suggestions(dataset_id: str, version: Optional[str] = None, database: Database = Depends(get_database),
                       ws: dict = Depends(get_workspace)):
    ds = _get_dataset(database, ws, dataset_id)
    ver = _get_version(database, ws, ds, version)
    return {"version_id": str(ver["_id"]), "candidates": suggest_targets_from_profile(ver["profile"])}


@router.post("/datasets/{dataset_id}/analyses", status_code=202)
def create_analysis(dataset_id: str, body: AnalysisRequest, database: Database = Depends(get_database),
                    ws: dict = Depends(editor_workspace), settings: Settings = Depends(get_settings)):
    ds = _get_dataset(database, ws, dataset_id)
    ver = _get_version(database, ws, ds, body.version_id)
    task = None
    if body.target_column:
        col = next((c for c in ver["profile"]["columns"] if c["name"] == body.target_column), None)
        if col is None:
            raise APIError(422, "unknown_target", f"There is no column named '{body.target_column}'.")
        task = task_type_from_profile(col)
        if task is None:
            raise APIError(422, "unsuitable_target", f"'{body.target_column}' cannot be predicted (it is an ID, constant, text or date). Pick another column.")
        if body.task_type and body.task_type != task:
            if body.task_type not in ("binary", "multiclass", "regression"):
                raise APIError(422, "bad_task_type", "task_type must be binary, multiclass or regression.")
            task = body.task_type
    cfg = EngineConfig(as_of=date.today().isoformat(), sample_threshold=settings.analysis_sample_threshold)
    analysis = {"_id": ObjectId(), "workspace_id": ws["_id"], "dataset_id": ds["_id"], "version_id": ver["_id"],
                "target_column": body.target_column, "task_type": task, "config": cfg.snapshot(), "status": "queued",
                "created_at": now(), "created_by": ws.get("_user_id")}
    database.analyses.insert_one(analysis)
    job_id = queue_for(database, settings).enqueue("analysis", {"analysis_id": str(analysis["_id"]), "dataset_id": str(ds["_id"])}, ws["_id"])
    database.analyses.update_one({"_id": analysis["_id"]}, {"$set": {"job_id": job_id}})
    audit(database, ws, "analysis.create", str(analysis["_id"]))
    return {"analysis_id": str(analysis["_id"]), "job_id": str(job_id), "status": "queued"}


@router.get("/datasets/{dataset_id}/analyses")
def list_analyses(dataset_id: str, limit: int = Query(20, ge=1, le=100), database: Database = Depends(get_database),
                  ws: dict = Depends(get_workspace)):
    ds = _get_dataset(database, ws, dataset_id)
    items = database.analyses.find({"workspace_id": ws["_id"], "dataset_id": ds["_id"]},
                                   {"config": 0, "target_candidates": 0, "check_timings_ms": 0}).sort("created_at", -1).limit(limit)
    return {"items": serialize(list(items))}


@router.get("/analyses/{analysis_id}")
def read_analysis(analysis_id: str, database: Database = Depends(get_database), ws: dict = Depends(get_workspace)):
    return serialize(get_analysis(database, ws, analysis_id))


@router.get("/analyses/{analysis_id}/findings")
def read_findings(analysis_id: str, severity: Optional[str] = Query(None), category: Optional[str] = Query(None),
                  column: Optional[str] = Query(None), database: Database = Depends(get_database), ws: dict = Depends(get_workspace)):
    a = get_analysis(database, ws, analysis_id)
    q: Dict[str, Any] = {"analysis_id": a["_id"], "workspace_id": ws["_id"]}
    if severity:
        sev = [s for s in severity.split(",") if s in SEVERITY_ORDER]
        q["severity"] = {"$in": sev}
    if category:
        q["category"] = {"$in": category.split(",")}
    if column:
        q["columns"] = column
    items = list(database.findings.find(q))
    items.sort(key=lambda f: (SEVERITY_ORDER.index(f["severity"]), -f.get("score_impact", 0), f["code"]))
    return {"items": serialize(items), "total": len(items)}


def diff_analyses(a_findings: List[dict], b_findings: List[dict], a: dict, b: dict) -> Dict[str, Any]:
    key = lambda f: (f["code"], tuple(sorted(f["columns"])))
    ak, bk = {key(f): f for f in a_findings}, {key(f): f for f in b_findings}
    resolved = [ak[k] for k in ak if k not in bk]
    new = [bk[k] for k in bk if k not in ak]
    changed = [{"code": k[0], "columns": list(k[1]), "from_severity": ak[k]["severity"], "to_severity": bk[k]["severity"],
                "from_points": ak[k].get("score_impact", 0), "to_points": bk[k].get("score_impact", 0)}
               for k in ak if k in bk and (ak[k]["severity"] != bk[k]["severity"] or ak[k].get("score_impact") != bk[k].get("score_impact"))]
    unchanged = [k for k in ak if k in bk and k not in {(c["code"], tuple(c["columns"])) for c in changed}]
    slim = lambda f: {"code": f["code"], "columns": f["columns"], "severity": f["severity"], "points": f.get("score_impact", 0), "plain_text": f["plain_text"]}
    sa, sb = (a.get("score") or {}).get("value"), (b.get("score") or {}).get("value")
    return {"score_from": sa, "score_to": sb, "score_delta": None if sa is None or sb is None else round(sb - sa, 2),
            "resolved": [slim(f) for f in resolved], "new": [slim(f) for f in new], "changed": changed, "unchanged_count": len(unchanged)}


@router.get("/analyses/compare/{a_id}/{b_id}")
def compare(a_id: str, b_id: str, database: Database = Depends(get_database), ws: dict = Depends(get_workspace)):
    a, b = get_analysis(database, ws, a_id), get_analysis(database, ws, b_id)
    if a["status"] != "done" or b["status"] != "done":
        raise APIError(409, "analysis_not_ready", "Both analyses must be finished before they can be compared.")
    fa = list(database.findings.find({"analysis_id": a["_id"], "workspace_id": ws["_id"]}))
    fb = list(database.findings.find({"analysis_id": b["_id"], "workspace_id": ws["_id"]}))
    return serialize({"a": a_id, "b": b_id, **diff_analyses(fa, fb, a, b)})
