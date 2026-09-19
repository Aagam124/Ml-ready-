"""
routes/models.py
-----------------
POST /datasets/{id}/train
GET  /datasets/{id}/model-results

Trains the same baseline model on the original file, and - if a cleaned
version exists (i.e. POST /clean has been called) - on the cleaned file
too, so the before/after comparison is measuring the effect of cleaning
and nothing else (same model, same random_state, same target).
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from database.mongodb import MLReadyDatabase
from dependencies import get_current_user, get_db
from io_utils import DatasetLoadError, load_dataframe
from models.baseline import TrainingError, train_baseline
from routes.datasets import _get_owned_dataset

router = APIRouter(prefix="/datasets", tags=["models"])


class TrainRequest(BaseModel):
    target_column: str


def _split_result(result: dict, target_column: str) -> tuple[dict, dict]:
    """Splits a train_baseline() result into (metrics, details) - metrics
    stays purely numeric so the repository's delta/chart logic never has to
    guess what's a real metric versus metadata. target_column is stored so
    compare_before_after can refuse to pair results trained on different
    targets (e.g. 'before' on 'price', 'after' on 'bedrooms')."""
    details = {
        "target_column": target_column,
        "problem_type": result["problem_type"],
        "train_rows": result["train_rows"],
        "test_rows": result["test_rows"],
    }
    if result["confusion_matrix"] is not None:
        details["confusion_matrix"] = result["confusion_matrix"]
    return result["metrics"], details


@router.post("/{dataset_id}/train")
def train(
    dataset_id: str,
    body: TrainRequest,
    user: dict = Depends(get_current_user),
    db: MLReadyDatabase = Depends(get_db),
):
    dataset = _get_owned_dataset(dataset_id, user, db)

    try:
        original_df = load_dataframe(dataset["file_path"])
    except DatasetLoadError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    try:
        before_result = train_baseline(original_df, body.target_column)
    except TrainingError as exc:
        raise HTTPException(status_code=400, detail=f"Could not train on the original data: {exc}")

    before_metrics, before_details = _split_result(before_result, body.target_column)
    db.model_results.create_result(
        dataset_id=dataset_id, user_id=user["id"], stage="before",
        model_type=before_result["model_type"], metrics=before_metrics, details=before_details,
    )

    after_skipped_reason: Optional[str] = None
    cleaned_path = dataset.get("cleaned_file_path")

    if cleaned_path:
        try:
            cleaned_df = load_dataframe(cleaned_path)
            after_result = train_baseline(cleaned_df, body.target_column)
            after_metrics, after_details = _split_result(after_result, body.target_column)
            db.model_results.create_result(
                dataset_id=dataset_id, user_id=user["id"], stage="after",
                model_type=after_result["model_type"], metrics=after_metrics, details=after_details,
            )
        except (DatasetLoadError, TrainingError) as exc:
            # The "before" result is still useful on its own - don't fail
            # the whole request just because the cleaned version couldn't
            # be trained on (e.g. cleaning dropped the target column).
            after_skipped_reason = str(exc)
    else:
        after_skipped_reason = "No cleaned version of this dataset yet - run cleaning first to get a comparison."

    comparison = db.model_results.compare_before_after(dataset_id, target_column=body.target_column)
    comparison["after_skipped_reason"] = after_skipped_reason if comparison["after"] is None else None
    return comparison


@router.get("/{dataset_id}/model-results/history")
def list_model_results(dataset_id: str, user: dict = Depends(get_current_user), db: MLReadyDatabase = Depends(get_db)):
    """Every training run on this dataset, not just the latest before/after
    pair - so Analysis History can show what was trained, on what target,
    and when."""
    _get_owned_dataset(dataset_id, user, db)
    return db.model_results.list_for_dataset(dataset_id)


@router.get("/{dataset_id}/model-results")
def get_model_results(dataset_id: str, user: dict = Depends(get_current_user), db: MLReadyDatabase = Depends(get_db)):
    dataset = _get_owned_dataset(dataset_id, user, db)
    comparison = db.model_results.compare_before_after(dataset_id)
    if comparison["before"] is None:
        raise HTTPException(status_code=404, detail="No model has been trained on this dataset yet")
    if comparison["after"] is None:
        if dataset.get("cleaned_file_path"):
            comparison["after_skipped_reason"] = (
                "The cleaned version hasn't been trained on with this target column yet - train again."
            )
        else:
            comparison["after_skipped_reason"] = "No cleaned version of this dataset yet - run cleaning first to get a comparison."
    else:
        comparison["after_skipped_reason"] = None
    return comparison
