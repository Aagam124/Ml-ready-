"""Workspace endpoints: current workspace metadata, usage metering, plan upgrades, and audit logs."""
from __future__ import annotations


from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.api.deps import audit, editor_workspace, get_workspace
from app.db import Database, get_database
from app.serialization import serialize

router = APIRouter(prefix="/workspaces", tags=["workspaces"])

PLAN_LIMITS = {
    "guest": {"max_datasets": 3, "max_rows": 50_000, "features": ["basic_analysis", "demo_data"]},
    "free": {"max_datasets": 10, "max_rows": 250_000, "features": ["full_engine", "cleaning_plans", "pdf_reports"]},
    "pro": {"max_datasets": 50, "max_rows": 2_000_000, "features": ["1.2m_large_workloads", "custom_rules", "unlimited_reports", "cross_version_diff"]},
    "enterprise": {"max_datasets": 1000, "max_rows": 10_000_000, "features": ["all_features", "priority_worker", "audit_retention", "dedicated_sla"]},
}


class PlanUpgradeRequest(BaseModel):
    plan: str = Field(pattern="^(free|pro|enterprise)$")


@router.get("/current")
def get_current_workspace(
    database: Database = Depends(get_database),
    ws: dict = Depends(get_workspace),
):
    ws_id = ws["_id"]
    dataset_count = database.datasets.count_documents({"workspace_id": ws_id})
    analysis_count = database.analyses.count_documents({"workspace_id": ws_id})
    rules_count = database.quality_rules.count_documents({"workspace_id": ws_id})

    # Workspace plan inherits from owner or defaults to 'free'
    plan = ws.get("plan", "free")
    limits = PLAN_LIMITS.get(plan, PLAN_LIMITS["free"])

    return serialize({
        "id": ws_id,
        "name": ws.get("name", "Workspace"),
        "role": ws.get("_role", "owner"),
        "plan": plan,
        "limits": limits,
        "usage": {
            "datasets": dataset_count,
            "analyses": analysis_count,
            "rules": rules_count,
            "limit_reached": dataset_count >= limits["max_datasets"],
        },
        "created_at": ws.get("created_at"),
    })


@router.get("/audit-log")
def get_audit_log(
    limit: int = 50,
    database: Database = Depends(get_database),
    ws: dict = Depends(get_workspace),
):
    limit = min(max(1, limit), 200)
    events = list(database.audit.find({"workspace_id": ws["_id"]}).sort("created_at", -1).limit(limit))
    return {"items": serialize(events), "total": len(events)}


@router.post("/plan")
def upgrade_plan(
    body: PlanUpgradeRequest,
    database: Database = Depends(get_database),
    ws: dict = Depends(editor_workspace),
):
    ws_id = ws["_id"]
    target_plan = body.plan
    database.workspaces.update_one({"_id": ws_id}, {"$set": {"plan": target_plan}})
    if ws.get("_user_id"):
        database.users.update_one({"_id": ws["_user_id"]}, {"$set": {"plan": target_plan}})

    audit(database, ws, "workspace.upgrade_plan", target_plan)
    return {
        "success": True,
        "new_plan": target_plan,
        "limits": PLAN_LIMITS[target_plan],
    }
