"""Quality Rules API: declarative dataset constraints, quality checks, and benchmark endpoints."""
from __future__ import annotations

import io
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import pandas as pd
from bson import ObjectId
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.api.deps import audit, editor_workspace, get_workspace
from app.db import Database, get_database
from app.engine.benchmark_1_2m import run_benchmark
from app.engine.evaluation import EvaluationSuite
from app.engine.rules import QualityRule, QualityRulesEngine
from app.errors import APIError
from app.serialization import parse_id, serialize

router = APIRouter(tags=["quality-rules"])


class CreateRuleBody(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    rule_type: str = Field(pattern="^(not_null|numeric_range|unique|allowed_values|regex_match|min_row_count|date_interval)$")
    column: Optional[str] = None
    params: Dict[str, Any] = Field(default_factory=dict)
    severity: str = Field(default="high", pattern="^(critical|high|medium|low)$")
    description: str = Field(default="", max_length=500)


def _load_dataset_df(database: Database, ws: dict, dataset_id: str, version_id: Optional[str] = None) -> pd.DataFrame:
    ds = database.datasets.find_one({"_id": parse_id(dataset_id, "Dataset"), "workspace_id": ws["_id"]})
    if ds is None:
        raise APIError(404, "not_found", "Dataset not found.")
    vid = parse_id(version_id, "Version") if version_id else ds["current_version_id"]
    ver = database.versions.find_one({"_id": vid, "dataset_id": ds["_id"], "workspace_id": ws["_id"]})
    if ver is None:
        raise APIError(404, "not_found", "Version not found.")
    grid_out = database.fs.get(ver["parquet_file_id"])
    return pd.read_parquet(io.BytesIO(grid_out.read()))


@router.get("/quality-rules")
def list_rules(ws: dict = Depends(get_workspace), database: Database = Depends(get_database)):
    rules = list(database.quality_rules.find({"workspace_id": ws["_id"]}).sort("created_at", -1))
    return serialize({"rules": rules, "total": len(rules)})


@router.post("/quality-rules", status_code=201)
def create_rule(body: CreateRuleBody, ws: dict = Depends(editor_workspace), database: Database = Depends(get_database)):
    doc = {
        "_id": ObjectId(),
        "workspace_id": ws["_id"],
        "name": body.name,
        "rule_type": body.rule_type,
        "column": body.column,
        "params": body.params,
        "severity": body.severity,
        "description": body.description,
        "enabled": True,
        "created_at": datetime.now(timezone.utc).replace(tzinfo=None),
    }
    database.quality_rules.insert_one(doc)
    audit(database, ws, "quality_rule.create", f"quality_rule:{doc['_id']}")
    return serialize({"rule": doc})


@router.delete("/quality-rules/{rule_id}")
def delete_rule(rule_id: str, ws: dict = Depends(editor_workspace), database: Database = Depends(get_database)):
    res = database.quality_rules.delete_one({"_id": parse_id(rule_id, "Rule"), "workspace_id": ws["_id"]})
    if res.deleted_count == 0:
        raise APIError(404, "not_found", "Rule not found.")
    audit(database, ws, "quality_rule.delete", f"quality_rule:{rule_id}")
    return {"ok": True}


@router.post("/datasets/{dataset_id}/quality-checks")
def run_quality_checks(dataset_id: str, version_id: Optional[str] = None, ws: dict = Depends(get_workspace),
                       database: Database = Depends(get_database)):
    df = _load_dataset_df(database, ws, dataset_id, version_id)
    raw_rules = list(database.quality_rules.find({"workspace_id": ws["_id"], "enabled": True}))
    rules = [
        QualityRule(
            id=str(r["_id"]),
            name=r["name"],
            rule_type=r["rule_type"],
            column=r.get("column"),
            params=r.get("params", {}),
            severity=r.get("severity", "high"),
            description=r.get("description", ""),
        )
        for r in raw_rules
    ]
    engine = QualityRulesEngine(rules)
    results = engine.evaluate(df)
    passed_count = sum(1 for r in results if r.status == "passed")
    failed_count = sum(1 for r in results if r.status == "failed")
    return {
        "dataset_id": dataset_id,
        "total_rules": len(results),
        "passed_count": passed_count,
        "failed_count": failed_count,
        "all_passed": failed_count == 0,
        "checks": [
            {
                "rule_id": r.rule_id,
                "name": r.name,
                "rule_type": r.rule_type,
                "column": r.column,
                "severity": r.severity,
                "status": r.status,
                "total_checked": r.total_checked,
                "failure_count": r.failure_count,
                "failure_pct": r.failure_pct,
                "sample_failures": r.sample_failures,
                "message": r.message,
                "evaluated_at": r.evaluated_at,
            }
            for r in results
        ],
    }


@router.get("/benchmarks/evaluation-suite")
def get_evaluation_metrics(seeds: int = 5, ws: dict = Depends(get_workspace)):
    """Runs the synthetic detection-quality suite; results are not real-world accuracy estimates."""
    suite = EvaluationSuite(seeds=min(max(1, seeds), 10), sample_size=1000)
    return suite.run_evaluations()


@router.get("/benchmarks/1-2m-workload")
def get_large_workload_benchmark(workload: str = "mixed", n_rows: int = 1_200_000, ws: dict = Depends(get_workspace)):
    """Runs the 1.2M+ row workload benchmark measuring execution runtime, peak RAM, and throughput."""
    res = run_benchmark(n_rows=min(max(10_000, n_rows), 1_500_000), workload=workload)
    return {
        "workload_name": res.workload_name,
        "n_rows": res.n_rows,
        "n_cols": res.n_cols,
        "file_format": res.file_format,
        "file_size_mb": res.file_size_mb,
        "read_seconds": res.read_seconds,
        "profile_seconds": res.profile_seconds,
        "analysis_seconds": res.analysis_seconds,
        "total_seconds": res.total_seconds,
        "peak_memory_mb": res.peak_memory_mb,
        "throughput_rows_per_sec": res.throughput_rows_per_sec,
        "throughput_mb_per_sec": res.throughput_mb_per_sec,
        "correctness_verified": res.correctness_verified,
        "details": res.details,
    }
