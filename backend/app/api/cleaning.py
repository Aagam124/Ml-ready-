"""Cleaning plans: propose, edit/toggle, preview a step, apply (background), export code."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from app.api.analyses import get_analysis, queue_for
from app.api.deps import audit, editor_workspace, get_workspace
from app.cleaning.export import generate_pipeline, generate_script
from app.cleaning.plan import propose_plan
from app.cleaning.preview import preview_step
from app.cleaning.registry import OPS, validate_step
from app.config import Settings, get_settings
from app.db import Database, get_database
from app.errors import APIError
from app.serialization import parse_id, serialize
from app.storage import load_version_df, now

router = APIRouter(tags=["cleaning"])


class StepPatch(BaseModel):
    step_id: str
    enabled: Optional[bool] = None
    params: Optional[Dict[str, Any]] = None
    columns: Optional[List[str]] = None


class PlanPatch(BaseModel):
    steps: List[StepPatch]


class NewStep(BaseModel):
    op: str
    columns: List[str] = []
    params: Dict[str, Any] = {}
    label: Optional[str] = None


def get_plan(database: Database, ws: dict, plan_id: str) -> dict:
    p = database.plans.find_one({"_id": parse_id(plan_id, "Plan"), "workspace_id": ws["_id"]})
    if p is None:
        raise APIError(404, "not_found", "Cleaning plan not found.")
    return p


@router.post("/analyses/{analysis_id}/plan", status_code=201)
def propose(analysis_id: str, database: Database = Depends(get_database), ws: dict = Depends(editor_workspace)):
    a = get_analysis(database, ws, analysis_id)
    if a["status"] != "done":
        raise APIError(409, "analysis_not_ready", "Wait for the analysis to finish before building a plan.")
    ver = database.versions.find_one({"_id": a["version_id"], "workspace_id": ws["_id"]})
    findings = list(database.findings.find({"analysis_id": a["_id"], "workspace_id": ws["_id"]}))
    plan_doc = propose_plan([{k: v for k, v in f.items() if k != "_id"} for f in findings], ver["profile"], a.get("target_column"))
    plan = {"workspace_id": ws["_id"], "dataset_id": a["dataset_id"], "analysis_id": a["_id"], "version_id": a["version_id"],
            "steps": plan_doc["steps"], "status": "draft", "result_version_id": None, "created_at": now(), "created_by": ws.get("_user_id")}
    res = database.plans.insert_one(plan)
    plan["_id"] = res.inserted_id
    return serialize(plan)


@router.get("/plans/{plan_id}")
def read_plan(plan_id: str, database: Database = Depends(get_database), ws: dict = Depends(get_workspace)):
    return serialize(get_plan(database, ws, plan_id))


@router.patch("/plans/{plan_id}")
def edit_plan(plan_id: str, body: PlanPatch, database: Database = Depends(get_database), ws: dict = Depends(editor_workspace)):
    plan = get_plan(database, ws, plan_id)
    if plan["status"] not in ("draft", "failed"):
        raise APIError(409, "plan_locked", "This plan was already applied. Build a new plan from the new version.")
    by_id = {s["step_id"]: s for s in plan["steps"]}
    for patch in body.steps:
        s = by_id.get(patch.step_id)
        if s is None:
            raise APIError(404, "step_not_found", f"No step '{patch.step_id}' in this plan.")
        if patch.enabled is not None:
            s["enabled"] = patch.enabled
        if patch.params is not None:
            s["params"] = patch.params
        if patch.columns is not None:
            s["columns"] = patch.columns
        s["preview"] = None
    database.plans.update_one({"_id": plan["_id"]}, {"$set": {"steps": plan["steps"]}})
    return serialize(plan)


@router.post("/plans/{plan_id}/steps", status_code=201)
def add_step(plan_id: str, body: NewStep, database: Database = Depends(get_database), ws: dict = Depends(editor_workspace)):
    import uuid
    plan = get_plan(database, ws, plan_id)
    err = validate_step({"op": body.op})
    if err or OPS[body.op]["kind"] == "option":
        raise APIError(422, "bad_step", err or "That operation cannot be added manually.")
    step = {"step_id": f"custom-{uuid.uuid4().hex[:8]}", "op": body.op, "columns": body.columns, "params": body.params, "enabled": True,
            "label": body.label or OPS[body.op]["label"], "finding_code": None, "preview": None}
    database.plans.update_one({"_id": plan["_id"]}, {"$push": {"steps": step}})
    return serialize(step)


@router.post("/plans/{plan_id}/steps/{step_id}/preview")
def preview(plan_id: str, step_id: str, database: Database = Depends(get_database), ws: dict = Depends(get_workspace)):
    plan = get_plan(database, ws, plan_id)
    a = database.analyses.find_one({"_id": plan["analysis_id"], "workspace_id": ws["_id"]})
    ver = database.versions.find_one({"_id": plan["version_id"], "workspace_id": ws["_id"]})
    df = load_version_df(database, ver)
    try:
        result = preview_step(df, plan["steps"], step_id, a.get("target_column"), a.get("task_type"))
    except KeyError:
        raise APIError(404, "step_not_found", "That step is not in the plan.")
    except Exception as e:     # a bad parameter should say what to fix, not 500
        raise APIError(422, "preview_failed", f"This step cannot run on your data as configured: {type(e).__name__}: {e}")
    database.plans.update_one({"_id": plan["_id"], "steps.step_id": step_id}, {"$set": {"steps.$.preview": result}})
    return result


@router.post("/plans/{plan_id}/apply", status_code=202)
def apply(plan_id: str, database: Database = Depends(get_database), ws: dict = Depends(editor_workspace),
          settings: Settings = Depends(get_settings)):
    plan = get_plan(database, ws, plan_id)
    if plan["status"] not in ("draft", "failed"):
        raise APIError(409, "plan_locked", "This plan is already applying or applied.")
    if not any(s.get("enabled", True) for s in plan["steps"]):
        raise APIError(422, "empty_plan", "Turn on at least one step first.")
    job_id = queue_for(database, settings).enqueue("apply_plan", {"plan_id": str(plan["_id"]), "dataset_id": str(plan["dataset_id"])}, ws["_id"])
    database.plans.update_one({"_id": plan["_id"]}, {"$set": {"status": "queued", "job_id": job_id}})
    audit(database, ws, "plan.apply", str(plan["_id"]))
    return {"plan_id": str(plan["_id"]), "job_id": str(job_id)}


@router.get("/plans/{plan_id}/export")
def export_code(plan_id: str, kind: str = Query("script", pattern="^(script|pipeline)$"), database: Database = Depends(get_database),
                ws: dict = Depends(get_workspace)):
    plan = get_plan(database, ws, plan_id)
    a = database.analyses.find_one({"_id": plan["analysis_id"], "workspace_id": ws["_id"]})
    ver = database.versions.find_one({"_id": plan["version_id"], "workspace_id": ws["_id"]})
    ds = database.datasets.find_one({"_id": plan["dataset_id"], "workspace_id": ws["_id"]})
    if kind == "script":
        read = {"ext": ver.get("file_ext"), "delimiter": ver["ingest"].get("delimiter"), "encoding": ver["ingest"].get("encoding"),
                "sheet": ver["ingest"].get("sheet"), "columns": [c["name"] for c in ver["schema"]]}
        code, fname = generate_script(plan["steps"], a.get("target_column"), a.get("task_type"), read, ds["name"]), "clean_data.py"
    else:
        code, fname = generate_pipeline(plan["steps"], a.get("target_column"), a.get("task_type")), "mlready_pipeline.py"
    return PlainTextResponse(code, headers={"Content-Disposition": f'attachment; filename="{fname}"'})
