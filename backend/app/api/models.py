"""Honest before/after model runs (background)."""
from __future__ import annotations

from typing import Optional

from bson import ObjectId
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.api.analyses import get_analysis, queue_for
from app.api.cleaning import get_plan
from app.api.deps import audit, editor_workspace, get_workspace
from app.config import Settings, get_settings
from app.db import Database, get_database
from app.errors import APIError
from app.serialization import parse_id, serialize
from app.storage import now

router = APIRouter(tags=["models"])


class ModelRunRequest(BaseModel):
    plan_id: str
    seed: int = 42


@router.post("/model-runs", status_code=202)
def train(body: ModelRunRequest, database: Database = Depends(get_database), ws: dict = Depends(editor_workspace),
          settings: Settings = Depends(get_settings)):
    plan = get_plan(database, ws, body.plan_id)
    analysis = get_analysis(database, ws, str(plan["analysis_id"]))
    if not analysis.get("target_column"):
        raise APIError(422, "target_required", "Select a target column before training models.")
    run = {"_id": ObjectId(), "workspace_id": ws["_id"], "dataset_id": plan["dataset_id"], "version_id": plan["version_id"],
           "analysis_id": analysis["_id"], "plan_id": plan["_id"], "target": analysis["target_column"], "seed": body.seed,
           "status": "queued", "created_at": now(), "created_by": ws.get("_user_id")}
    database.model_runs.insert_one(run)
    job_id = queue_for(database, settings).enqueue("model_run", {"model_run_id": str(run["_id"]), "dataset_id": str(plan["dataset_id"])}, ws["_id"])
    database.model_runs.update_one({"_id": run["_id"]}, {"$set": {"job_id": job_id}})
    audit(database, ws, "model_run.create", str(run["_id"]))
    return {"model_run_id": str(run["_id"]), "job_id": str(job_id)}


def get_run(database: Database, ws: dict, run_id: str) -> dict:
    r = database.model_runs.find_one({"_id": parse_id(run_id, "Model run"), "workspace_id": ws["_id"]})
    if r is None:
        raise APIError(404, "not_found", "Model run not found.")
    return r


@router.get("/model-runs/{run_id}")
def results(run_id: str, database: Database = Depends(get_database), ws: dict = Depends(get_workspace)):
    return serialize(get_run(database, ws, run_id))


@router.get("/datasets/{dataset_id}/model-runs")
def list_runs(dataset_id: str, database: Database = Depends(get_database), ws: dict = Depends(get_workspace)):
    items = database.model_runs.find({"workspace_id": ws["_id"], "dataset_id": parse_id(dataset_id, "Dataset")}, {"result.arms": 0}).sort("created_at", -1).limit(20)
    return {"items": serialize(list(items))}


@router.get("/model-runs/compare/{a_id}/{b_id}")
def compare_runs(a_id: str, b_id: str, database: Database = Depends(get_database), ws: dict = Depends(get_workspace)):
    a, b = get_run(database, ws, a_id), get_run(database, ws, b_id)
    if a.get("status") != "done" or b.get("status") != "done":
        raise APIError(409, "run_not_ready", "Both model runs must be finished.")
    def best(run):
        c = run["result"]["comparison"]
        row = next((p for p in c["per_model"] if p["name"] == c["best_model"]), None)
        return {"id": str(run["_id"]), "primary_metric": c["primary_metric"], "best_model": c["best_model"], "after": row and row["after"],
                "before": row and row["before"], "verdict": c["verdict"]}
    return {"a": best(a), "b": best(b)}
