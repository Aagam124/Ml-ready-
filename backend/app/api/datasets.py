"""Dataset endpoints: upload, demo, list, get, versions, profile, download, delete."""
from __future__ import annotations

import os
import re
from typing import Iterator, Optional
from urllib.parse import quote

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile
from fastapi.responses import StreamingResponse

from app.api.deps import audit, editor_workspace, get_workspace
from app.auth.ratelimit import upload_rate_limit
from app.config import Settings, get_settings
from app.db import Database, get_database
from app.demo import generate_demo_dataframe
from app.errors import APIError
from app.ingestion.service import ingest_file
from app.ingestion.storage import save_stream_to_tempfile
from app.ingestion.validate import sanitize_filename, validate_extension, validate_magic
from app.serialization import parse_id, serialize

router = APIRouter(prefix="/datasets", tags=["datasets"])

_VERSION_LIST_FIELDS = {"profile": 0}


def _summary(dataset: dict, version: dict) -> dict:
    prof = version["profile"]
    return serialize({
        "dataset": dataset,
        "version": {k: v for k, v in version.items() if k not in ("profile", "gridfs_file_id", "parquet_file_id")},
        "overview": {
            "n_rows": prof["n_rows"], "n_cols": prof["n_cols"],
            "missing": prof["missing"], "duplicate_rows": prof["duplicate_rows"],
            "semantic_summary": prof["semantic_summary"],
        },
    })


def _get_dataset(database: Database, ws: dict, dataset_id: str) -> dict:
    ds = database.datasets.find_one({"_id": parse_id(dataset_id, "Dataset"), "workspace_id": ws["_id"]})
    if ds is None:
        raise APIError(404, "not_found", "Dataset not found.")
    return ds


def _get_version(database: Database, ws: dict, ds: dict, version_id: Optional[str]) -> dict:
    vid = parse_id(version_id, "Version") if version_id else ds["current_version_id"]
    v = database.versions.find_one({"_id": vid, "dataset_id": ds["_id"], "workspace_id": ws["_id"]})
    if v is None:
        raise APIError(404, "not_found", "Version not found.")
    return v


@router.post("/upload", status_code=201, dependencies=[Depends(upload_rate_limit)])
def upload_dataset(
    file: UploadFile = File(...),
    name: Optional[str] = Form(None),
    sheet: Optional[str] = Form(None),
    database: Database = Depends(get_database),
    ws: dict = Depends(editor_workspace),
    settings: Settings = Depends(get_settings),
):
    filename = sanitize_filename(file.filename)
    ext = validate_extension(filename)
    from app.api.workspaces import PLAN_LIMITS
    plan = ws.get("plan", "free")
    limits = PLAN_LIMITS.get(plan, PLAN_LIMITS["free"])
    if database.datasets.count_documents({"workspace_id": ws["_id"]}) >= limits["max_datasets"]:
        raise APIError(403, "plan_limit_reached", f"Your {plan.capitalize()} plan allows up to {limits['max_datasets']} datasets. Please upgrade your plan to continue.")
    saved = save_stream_to_tempfile(file.file, settings.max_upload_bytes, settings.upload_chunk_bytes, suffix=ext)
    try:
        validate_magic(saved.path, ext, settings)
        ds, ver = ingest_file(database, ws["_id"], saved.path, filename, ext, saved.size,
                              saved.sha256, settings, name=name, sheet=sheet)
    finally:
        os.unlink(saved.path)
    return _summary(ds, ver)


@router.post("/demo", status_code=201)
def create_demo_dataset(
    database: Database = Depends(get_database),
    ws: dict = Depends(editor_workspace),
    settings: Settings = Depends(get_settings),
):
    import hashlib
    import tempfile
    csv_bytes = generate_demo_dataframe().to_csv(index=False).encode("utf-8")
    fd, path = tempfile.mkstemp(prefix="mlready_demo_", suffix=".csv")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(csv_bytes)
        ds, ver = ingest_file(database, ws["_id"], path, "demo_customers.csv", ".csv", len(csv_bytes),
                              hashlib.sha256(csv_bytes).hexdigest(), settings, name="Demo: customer churn")
    finally:
        os.unlink(path)
    return _summary(ds, ver)


@router.get("")
def list_datasets(
    q: Optional[str] = Query(None, max_length=100),
    limit: int = Query(20, ge=1, le=100),
    skip: int = Query(0, ge=0),
    database: Database = Depends(get_database),
    ws: dict = Depends(get_workspace),
):
    query: dict = {"workspace_id": ws["_id"]}
    if q:
        query["name"] = {"$regex": re.escape(q), "$options": "i"}
    total = database.datasets.count_documents(query)
    items = list(database.datasets.find(query).sort("created_at", -1).skip(skip).limit(limit))
    return {"items": serialize(items), "total": total, "limit": limit, "skip": skip}


@router.get("/{dataset_id}")
def get_dataset(dataset_id: str, database: Database = Depends(get_database), ws: dict = Depends(get_workspace)):
    ds = _get_dataset(database, ws, dataset_id)
    ver = _get_version(database, ws, ds, None)
    return _summary(ds, ver)


@router.get("/{dataset_id}/versions")
def list_versions(dataset_id: str, database: Database = Depends(get_database), ws: dict = Depends(get_workspace)):
    ds = _get_dataset(database, ws, dataset_id)
    versions = database.versions.find({"dataset_id": ds["_id"], "workspace_id": ws["_id"]},
                                      {"profile": 0}).sort("created_at", 1)
    out = [{k: v for k, v in ver.items() if k not in ("gridfs_file_id", "parquet_file_id")} for ver in versions]
    return {"items": serialize(out)}


@router.get("/{dataset_id}/profile")
def get_profile(dataset_id: str, version: Optional[str] = None,
                database: Database = Depends(get_database), ws: dict = Depends(get_workspace)):
    ds = _get_dataset(database, ws, dataset_id)
    ver = _get_version(database, ws, ds, version)
    return serialize({"version_id": ver["_id"], "profile": ver["profile"], "ingest": ver["ingest"]})


@router.get("/{dataset_id}/download")
def download(dataset_id: str, version: Optional[str] = None, kind: str = Query("original", pattern="^(original|parquet)$"),
             spreadsheet_safe: bool = Query(False, description="Prefix formula-like text cells so Excel/Sheets cannot execute them."),
             database: Database = Depends(get_database), ws: dict = Depends(get_workspace),
             settings: Settings = Depends(get_settings)):
    ds = _get_dataset(database, ws, dataset_id)
    ver = _get_version(database, ws, ds, version)
    if spreadsheet_safe and kind == "original":
        from fastapi.responses import Response
        from app.ingestion.csvsafe import sanitize_for_spreadsheet
        from app.storage import load_version_df
        data = sanitize_for_spreadsheet(load_version_df(database, ver)).to_csv(index=False).encode("utf-8")
        name = os.path.splitext(ver["original_filename"])[0] + ".safe.csv"
        return Response(data, media_type="text/csv", headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(name)}"})
    file_id = ver["gridfs_file_id"] if kind == "original" else ver["parquet_file_id"]
    grid_out = database.fs.get(file_id)
    filename = ver["original_filename"] if kind == "original" else os.path.splitext(ver["original_filename"])[0] + ".parquet"

    def stream() -> Iterator[bytes]:
        try:
            while True:
                chunk = grid_out.read(settings.upload_chunk_bytes)
                if not chunk:
                    break
                yield chunk
        finally:
            grid_out.close()

    headers = {"Content-Disposition": f"attachment; filename*=UTF-8''{quote(filename)}"}
    return StreamingResponse(stream(), media_type="application/octet-stream", headers=headers)


@router.delete("/{dataset_id}")
def delete_dataset(dataset_id: str, database: Database = Depends(get_database), ws: dict = Depends(editor_workspace)):
    """One-click 'delete everything': versions, files, analyses, findings, plans, model runs, reports, jobs."""
    from app.storage import delete_dataset_everything
    ds = _get_dataset(database, ws, dataset_id)
    counts = delete_dataset_everything(database, ds)
    audit(database, ws, "dataset.delete_everything", dataset_id)
    return {"deleted": True, "counts": counts}


@router.post("/{dataset_id}/versions/{version_id}/select")
def select_version(dataset_id: str, version_id: str, database: Database = Depends(get_database), ws: dict = Depends(editor_workspace)):
    """Undo / redo = choose which version is current. Nothing is deleted."""
    ds = _get_dataset(database, ws, dataset_id)
    ver = _get_version(database, ws, ds, version_id)
    database.datasets.update_one({"_id": ds["_id"], "workspace_id": ws["_id"]}, {"$set": {"current_version_id": ver["_id"]}})
    audit(database, ws, "dataset.select_version", version_id)
    return {"current_version_id": str(ver["_id"])}


@router.post("/{dataset_id}/undo")
def undo(dataset_id: str, database: Database = Depends(get_database), ws: dict = Depends(editor_workspace)):
    ds = _get_dataset(database, ws, dataset_id)
    cur = _get_version(database, ws, ds, None)
    if not cur.get("parent_version_id"):
        raise APIError(409, "nothing_to_undo", "This is the original data. There is nothing earlier to go back to.")
    database.datasets.update_one({"_id": ds["_id"], "workspace_id": ws["_id"]}, {"$set": {"current_version_id": cur["parent_version_id"]}})
    return {"current_version_id": str(cur["parent_version_id"])}
