"""Reports: generate (HTML / PDF), download, share links with expiry. Shared links are public but unguessable."""
from __future__ import annotations

from datetime import timedelta
from typing import Iterator, Optional

from bson import ObjectId
from fastapi import APIRouter, Depends, Query
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from app.api.analyses import get_analysis
from app.api.deps import audit, editor_workspace, get_workspace
from app.auth.security import new_share_token
from app.db import Database, get_database
from app.errors import APIError
from app.reports.html import render_report
from app.reports.pdf import render_pdf
from app.serialization import parse_id, serialize
from app.storage import now

router = APIRouter(tags=["reports"])


class ReportRequest(BaseModel):
    format: str = "html"
    model_run_id: Optional[str] = None


class ShareRequest(BaseModel):
    days: int = 7


@router.post("/analyses/{analysis_id}/reports", status_code=201)
def generate(analysis_id: str, body: ReportRequest, database: Database = Depends(get_database), ws: dict = Depends(editor_workspace)):
    if body.format not in ("html", "pdf"):
        raise APIError(422, "bad_format", "format must be 'html' or 'pdf'.")
    a = get_analysis(database, ws, analysis_id)
    if a["status"] != "done":
        raise APIError(409, "analysis_not_ready", "Wait for the analysis to finish.")
    ds = database.datasets.find_one({"_id": a["dataset_id"], "workspace_id": ws["_id"]})
    findings = [{k: v for k, v in f.items() if k != "_id"} for f in database.findings.find({"analysis_id": a["_id"], "workspace_id": ws["_id"]})]
    order = ["critical", "high", "medium", "low", "info"]
    findings.sort(key=lambda f: (order.index(f["severity"]), -f.get("score_impact", 0)))
    run = None
    if body.model_run_id:
        run = database.model_runs.find_one({"_id": parse_id(body.model_run_id, "Model run"), "workspace_id": ws["_id"], "analysis_id": a["_id"]})
        if run is None or run.get("status") != "done":
            raise APIError(409, "run_not_ready", "That model run is not finished or does not belong to this analysis.")
    if body.format == "html":
        data, mime, ext = render_report(a, findings, ds["name"], run).encode("utf-8"), "text/html; charset=utf-8", "html"
    else:
        data, mime, ext = render_pdf(a, findings, ds["name"], run), "application/pdf", "pdf"
    fid = database.fs.put(data, filename=f"mlready-report.{ext}", workspace_id=ws["_id"], kind="report")
    rep = {"workspace_id": ws["_id"], "dataset_id": a["dataset_id"], "analysis_id": a["_id"], "format": body.format, "mime": mime,
           "gridfs_file_id": fid, "size_bytes": len(data), "share_token": None, "created_at": now(), "created_by": ws.get("_user_id")}
    rep["_id"] = database.reports.insert_one(rep).inserted_id
    audit(database, ws, "report.generate", str(rep["_id"]))
    return serialize({k: v for k, v in rep.items() if k != "gridfs_file_id"})


def _stream(database: Database, rep: dict, disposition: str) -> Response:
    grid = database.fs.get(rep["gridfs_file_id"])
    data = grid.read()
    grid.close()
    return Response(data, media_type=rep["mime"], headers={
        "Content-Disposition": f'{disposition}; filename="mlready-report.{rep["format"]}"',
        "X-Content-Type-Options": "nosniff", "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; img-src data:"})


@router.get("/reports/{report_id}/download")
def download(report_id: str, database: Database = Depends(get_database), ws: dict = Depends(get_workspace)):
    rep = database.reports.find_one({"_id": parse_id(report_id, "Report"), "workspace_id": ws["_id"]})
    if rep is None:
        raise APIError(404, "not_found", "Report not found.")
    return _stream(database, rep, "attachment")


@router.post("/reports/{report_id}/share")
def share(report_id: str, body: ShareRequest, database: Database = Depends(get_database), ws: dict = Depends(editor_workspace)):
    if not 1 <= body.days <= 90:
        raise APIError(422, "bad_expiry", "Choose between 1 and 90 days.")
    rep = database.reports.find_one({"_id": parse_id(report_id, "Report"), "workspace_id": ws["_id"]})
    if rep is None:
        raise APIError(404, "not_found", "Report not found.")
    token, expires = new_share_token(), now() + timedelta(days=body.days)
    database.reports.update_one({"_id": rep["_id"]}, {"$set": {"share_token": token, "share_expires_at": expires}})
    audit(database, ws, "report.share", report_id)
    return {"share_url_path": f"/api/v1/shared/{token}", "token": token, "expires_at": expires.isoformat() + "Z"}


@router.delete("/reports/{report_id}/share", status_code=204)
def unshare(report_id: str, database: Database = Depends(get_database), ws: dict = Depends(editor_workspace)):
    database.reports.update_one({"_id": parse_id(report_id, "Report"), "workspace_id": ws["_id"]}, {"$set": {"share_token": None, "share_expires_at": None}})
    return Response(status_code=204)


@router.get("/shared/{token}")
def shared(token: str, database: Database = Depends(get_database)):
    """Public. Unknown, revoked and expired tokens all look identical (404)."""
    rep = database.reports.find_one({"share_token": token, "share_expires_at": {"$gt": now()}}) if len(token) >= 20 else None
    if rep is None:
        raise APIError(404, "not_found", "This link does not exist or has expired.")
    return _stream(database, rep, "inline")
