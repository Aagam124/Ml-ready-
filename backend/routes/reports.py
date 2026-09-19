"""
routes/reports.py
-------------------
GET /datasets/{id}/full-report          - JSON bundle for the Report page
GET /datasets/{id}/report/export?format=html|pdf - downloadable file
"""

import re

from fastapi import APIRouter, Depends, HTTPException, Query, Response

from database.mongodb import MLReadyDatabase
from dependencies import get_current_user, get_db
from reports.builder import build_report_bundle
from reports.html_renderer import render_html
from reports.pdf_renderer import render_pdf
from routes.datasets import _get_owned_dataset

router = APIRouter(prefix="/datasets", tags=["reports"])


def _assemble_bundle(dataset_id: str, dataset: dict, db: MLReadyDatabase) -> dict:
    latest = db.analysis_reports.get_latest_for_dataset(dataset_id)
    cleaning_history = db.cleaning_history.list_for_dataset(dataset_id)
    comparison = db.model_results.compare_before_after(dataset_id)
    if comparison["before"] is None:
        comparison = None
    return build_report_bundle(
        dataset=dataset,
        latest_analysis=latest["report"] if latest else None,
        cleaning_history=cleaning_history,
        model_comparison=comparison,
    )


def _safe_filename(filename: str) -> str:
    stem = filename.rsplit(".", 1)[0]
    return re.sub(r"[^A-Za-z0-9_-]+", "_", stem) or "report"


@router.get("/{dataset_id}/full-report")
def get_full_report(dataset_id: str, user: dict = Depends(get_current_user), db: MLReadyDatabase = Depends(get_db)):
    dataset = _get_owned_dataset(dataset_id, user, db)
    return _assemble_bundle(dataset_id, dataset, db)


@router.get("/{dataset_id}/report/export")
def export_report(
    dataset_id: str,
    format: str = Query("html", pattern="^(html|pdf)$"),
    user: dict = Depends(get_current_user),
    db: MLReadyDatabase = Depends(get_db),
):
    dataset = _get_owned_dataset(dataset_id, user, db)
    bundle = _assemble_bundle(dataset_id, dataset, db)
    stem = _safe_filename(dataset["filename"])

    if format == "pdf":
        pdf_bytes = render_pdf(bundle)
        return Response(
            content=pdf_bytes, media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{stem}_report.pdf"'},
        )

    html_str = render_html(bundle)
    return Response(
        content=html_str, media_type="text/html",
        headers={"Content-Disposition": f'attachment; filename="{stem}_report.html"'},
    )
