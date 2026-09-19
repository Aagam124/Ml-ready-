"""
routes/analysis.py
-------------------
POST /datasets/{id}/analyze
GET  /datasets/{id}/report
POST /datasets/{id}/clean
"""

import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from analysis.pipeline import run_full_analysis
from cleaning.cleaner import CleaningError, DataCleaner
from database.mongodb import MLReadyDatabase
from dependencies import get_current_user, get_db
from io_utils import DatasetLoadError, load_dataframe
from routes.datasets import _get_owned_dataset

router = APIRouter(prefix="/datasets", tags=["analysis"])


class AnalyzeRequest(BaseModel):
    target_column: Optional[str] = None


class CleaningOperation(BaseModel):
    type: str
    column: Optional[str] = None
    columns: Optional[List[str]] = None
    strategy: Optional[str] = None
    method: Optional[str] = None


class CleanRequest(BaseModel):
    operations: List[CleaningOperation]


@router.post("/{dataset_id}/analyze")
def analyze_dataset(
    dataset_id: str,
    body: AnalyzeRequest = AnalyzeRequest(),
    user: dict = Depends(get_current_user),
    db: MLReadyDatabase = Depends(get_db),
):
    dataset = _get_owned_dataset(dataset_id, user, db)

    try:
        df = load_dataframe(dataset["file_path"])
    except DatasetLoadError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if body.target_column and body.target_column not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"target_column '{body.target_column}' not found in dataset columns",
        )

    report_payload = run_full_analysis(df, body.target_column)
    report = db.analysis_reports.create_report(dataset_id, user["id"], report_payload)
    db.datasets.update_dataset(dataset_id, {"status": "analyzed"})
    return report


@router.get("/{dataset_id}/report")
def get_report(dataset_id: str, user: dict = Depends(get_current_user), db: MLReadyDatabase = Depends(get_db)):
    _get_owned_dataset(dataset_id, user, db)  # ownership check, 404s if not theirs
    report = db.analysis_reports.get_latest_for_dataset(dataset_id)
    if report is None:
        raise HTTPException(status_code=404, detail="No analysis report yet - call analyze first")
    return report


@router.get("/{dataset_id}/reports")
def list_reports(dataset_id: str, user: dict = Depends(get_current_user), db: MLReadyDatabase = Depends(get_db)):
    """Full analysis history for a dataset (Analysis History page), not just the latest."""
    _get_owned_dataset(dataset_id, user, db)
    return db.analysis_reports.list_for_dataset(dataset_id)


@router.get("/{dataset_id}/cleaning-history")
def list_cleaning_history(dataset_id: str, user: dict = Depends(get_current_user), db: MLReadyDatabase = Depends(get_db)):
    _get_owned_dataset(dataset_id, user, db)
    return db.cleaning_history.list_for_dataset(dataset_id)


@router.post("/{dataset_id}/clean")
def clean_dataset(
    dataset_id: str,
    body: CleanRequest,
    user: dict = Depends(get_current_user),
    db: MLReadyDatabase = Depends(get_db),
):
    dataset = _get_owned_dataset(dataset_id, user, db)

    try:
        df = load_dataframe(dataset["file_path"])
    except DatasetLoadError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    operations: List[Dict[str, Any]] = [op.model_dump(exclude_none=True) for op in body.operations]

    cleaner = DataCleaner(df)
    try:
        cleaner.apply_operations(operations)
    except CleaningError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # Never overwrite the original file - write the clean version alongside it.
    base, ext = os.path.splitext(dataset["file_path"])
    clean_path = f"{base}_cleaned{ext}"
    cleaned_df = cleaner.result()
    if ext == ".csv":
        cleaned_df.to_csv(clean_path, index=False)
    else:
        cleaned_df.to_excel(clean_path, index=False)

    summary = cleaner.summary()
    entry = db.cleaning_history.create_entry(
        dataset_id=dataset_id, user_id=user["id"],
        operations=summary["operations"], resulting_shape=summary["resulting_shape"],
    )
    db.datasets.update_dataset(dataset_id, {
        "status": "cleaned",
        "cleaned_file_path": clean_path,
    })
    return entry
