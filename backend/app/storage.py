"""Move DataFrames in and out of GridFS (Parquet) and create derived dataset versions."""
from __future__ import annotations

import io
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from bson import ObjectId

from app.db import Database
from app.profiling.profile import profile_dataframe, schema_from_profile


def now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def encode_df(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    try:
        df.to_parquet(buf, index=False)
    except Exception:       # mixed-type object columns: stringify only those
        fixed = df.copy()
        for c in fixed.columns:
            if fixed[c].dtype == object:
                fixed[c] = fixed[c].map(lambda v: v if v is None or v != v else str(v))
        buf = io.BytesIO()
        fixed.to_parquet(buf, index=False)
    return buf.getvalue()


def decode_df(data: bytes) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(data))


def load_version_df(database: Database, version: Dict[str, Any]) -> pd.DataFrame:
    grid_out = database.fs.get(version["parquet_file_id"])
    try:
        return decode_df(grid_out.read())
    finally:
        grid_out.close()


def create_derived_version(database: Database, dataset: Dict[str, Any], parent: Dict[str, Any], df: pd.DataFrame,
                           step: str, created_by: Optional[ObjectId] = None, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Every change is a NEW version chained to its parent. The original is never touched."""
    import hashlib
    profile = profile_dataframe(df.reset_index(drop=True))
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    stored = []
    try:
        stem = parent.get("original_filename", "dataset").rsplit(".", 1)[0]
        raw_id = database.fs.put(csv_bytes, filename=f"{stem}.cleaned.csv", workspace_id=dataset["workspace_id"], kind="derived_csv")
        stored.append(raw_id)
        pq_id = database.fs.put(encode_df(df.reset_index(drop=True)), filename=f"{stem}.cleaned.parquet",
                                workspace_id=dataset["workspace_id"], kind="parquet")
        stored.append(pq_id)
        ver = {
            "_id": ObjectId(), "dataset_id": dataset["_id"], "workspace_id": dataset["workspace_id"], "parent_version_id": parent["_id"],
            "gridfs_file_id": raw_id, "parquet_file_id": pq_id, "original_filename": f"{stem}.cleaned.csv", "file_ext": ".csv",
            "size_bytes": len(csv_bytes), "content_hash": hashlib.sha256(csv_bytes).hexdigest(),
            "n_rows": profile["n_rows"], "n_cols": profile["n_cols"], "schema": schema_from_profile(profile), "profile": profile,
            "ingest": {"timings_ms": {}, "warnings": [], "renamed_columns": [], "encoding": "utf-8", "delimiter": ",",
                       "sheet": None, "sheets_available": []},
            "created_by_step": step, "created_by": created_by, "created_at": now(), **(extra or {}),
        }
        database.versions.insert_one(ver)
        database.datasets.update_one({"_id": dataset["_id"]}, {"$set": {"current_version_id": ver["_id"], "updated_at": now()},
                                                              "$inc": {"n_versions": 1}})
    except BaseException:
        for fid in stored:
            try:
                database.fs.delete(fid)
            except Exception:
                pass
        raise
    return ver


def delete_dataset_everything(database: Database, dataset: Dict[str, Any]) -> Dict[str, int]:
    """One-click 'delete everything': versions, files, analyses, findings, plans, model runs, reports, jobs."""
    ws, ds = dataset["workspace_id"], dataset["_id"]
    counts: Dict[str, int] = {}
    versions = list(database.versions.find({"dataset_id": ds, "workspace_id": ws}))
    vids = [v["_id"] for v in versions]
    analyses = list(database.analyses.find({"workspace_id": ws, "dataset_id": ds}, {"_id": 1}))
    aids = [a["_id"] for a in analyses]
    for rep in database.reports.find({"workspace_id": ws, "dataset_id": ds}):
        if rep.get("gridfs_file_id"):
            database.fs.delete(rep["gridfs_file_id"])
    counts["reports"] = database.reports.delete_many({"workspace_id": ws, "dataset_id": ds}).deleted_count
    counts["findings"] = database.findings.delete_many({"analysis_id": {"$in": aids}}).deleted_count
    counts["plans"] = database.plans.delete_many({"workspace_id": ws, "dataset_id": ds}).deleted_count
    counts["model_runs"] = database.model_runs.delete_many({"workspace_id": ws, "dataset_id": ds}).deleted_count
    counts["analyses"] = database.analyses.delete_many({"workspace_id": ws, "dataset_id": ds}).deleted_count
    counts["jobs"] = database.jobs.delete_many({"workspace_id": ws, "payload.dataset_id": str(ds)}).deleted_count
    for v in versions:
        for key in ("gridfs_file_id", "parquet_file_id"):
            if v.get(key):
                database.fs.delete(v[key])
    counts["versions"] = database.versions.delete_many({"dataset_id": ds, "workspace_id": ws}).deleted_count
    database.datasets.delete_one({"_id": ds, "workspace_id": ws})
    return counts
