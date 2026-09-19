"""
routes/datasets.py
-------------------
POST   /datasets/upload
GET    /datasets
GET    /datasets/{id}
DELETE /datasets/{id}
"""

import os
import uuid
from typing import List

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File

import config
from database.mongodb import MLReadyDatabase
from dependencies import get_current_user, get_db
from io_utils import DatasetLoadError, load_dataframe

router = APIRouter(prefix="/datasets", tags=["datasets"])


def _get_owned_dataset(dataset_id: str, user: dict, db: MLReadyDatabase) -> dict:
    try:
        dataset = db.datasets.get_dataset(dataset_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if dataset is None or dataset["user_id"] != user["id"]:
        # Same 404 whether it doesn't exist or belongs to someone else -
        # don't leak which one to an unauthorized caller.
        raise HTTPException(status_code=404, detail="Dataset not found")
    return dataset


@router.post("/upload", status_code=201)
async def upload_dataset(
    file: UploadFile = File(...),
    user: dict = Depends(get_current_user),
    db: MLReadyDatabase = Depends(get_db),
):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in config.ALLOWED_UPLOAD_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(config.ALLOWED_UPLOAD_EXTENSIONS)}",
        )

    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > config.MAX_UPLOAD_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({size_mb:.1f} MB). Max is {config.MAX_UPLOAD_SIZE_MB} MB",
        )

    user_dir = os.path.join(config.UPLOAD_DIR, user["id"])
    os.makedirs(user_dir, exist_ok=True)
    stored_name = f"{uuid.uuid4().hex}{ext}"
    file_path = os.path.join(user_dir, stored_name)
    with open(file_path, "wb") as f:
        f.write(contents)

    try:
        df = load_dataframe(file_path)
    except DatasetLoadError as exc:
        os.remove(file_path)
        raise HTTPException(status_code=400, detail=str(exc))

    metadata = {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "size_bytes": len(contents),
        "column_names": df.columns.tolist(),
    }
    dataset = db.datasets.create_dataset(
        user_id=user["id"], filename=file.filename, file_path=file_path, metadata=metadata,
    )
    return dataset


@router.get("", response_model=List[dict])
def list_datasets(user: dict = Depends(get_current_user), db: MLReadyDatabase = Depends(get_db)):
    return db.datasets.list_datasets_for_user(user["id"])


@router.get("/{dataset_id}")
def get_dataset(dataset_id: str, user: dict = Depends(get_current_user), db: MLReadyDatabase = Depends(get_db)):
    return _get_owned_dataset(dataset_id, user, db)


@router.delete("/{dataset_id}", status_code=204)
def delete_dataset(dataset_id: str, user: dict = Depends(get_current_user), db: MLReadyDatabase = Depends(get_db)):
    dataset = _get_owned_dataset(dataset_id, user, db)
    if os.path.exists(dataset["file_path"]):
        os.remove(dataset["file_path"])
    db.datasets.delete_dataset(dataset_id)
    return None
