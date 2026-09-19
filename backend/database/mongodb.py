"""
mongodb.py
----------
Phase 2 - MongoDB connection + CRUD repositories.

Design notes:
- get_client()/MLReadyDatabase both accept an optional pre-built client so
  the exact same code path can run against a real MongoDB Community Server
  (production / student laptop) or against mongomock.MongoClient() (tests,
  or this sandbox which has no local mongod available). Phase 3's FastAPI
  routes will import MLReadyDatabase directly - no changes needed there.
- Every repository method returns plain dicts/strings (ObjectId -> str,
  datetime -> isoformat) so results are JSON-serializable straight out of
  the box for FastAPI responses.
- Raw dataset *contents* are never stored here - only metadata/paths, per
  the project spec. Actual files live on disk (backend/uploads/).
"""

from __future__ import annotations

import hashlib
import os
import secrets
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from bson import ObjectId
from bson.errors import InvalidId
from pymongo import MongoClient, ReturnDocument

import config


# --------------------------------------------------------------------------
# Connection
# --------------------------------------------------------------------------

def get_client(client: Optional[MongoClient] = None) -> MongoClient:
    """Return the given client (e.g. a mongomock client for tests) or build
    a real one from config. Never call MongoClient() at import time - only
    when a database is actually requested, so importing this module never
    requires a running MongoDB instance."""
    if client is not None:
        return client
    return MongoClient(config.MONGO_URI, serverSelectionTimeoutMS=5000)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _oid(id_value: Any) -> ObjectId:
    """Coerce a string/ObjectId into a valid ObjectId, raising a clear error
    on malformed input instead of letting pymongo throw deep inside a query."""
    if isinstance(id_value, ObjectId):
        return id_value
    try:
        return ObjectId(str(id_value))
    except (InvalidId, TypeError):
        raise ValueError(f"'{id_value}' is not a valid ObjectId")


def _serialize(doc: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Make a Mongo document JSON-friendly: _id -> id (str), datetimes -> ISO."""
    if doc is None:
        return None
    out = dict(doc)
    if "_id" in out:
        out["id"] = str(out.pop("_id"))
    for key, value in out.items():
        if isinstance(value, datetime):
            out[key] = value.isoformat()
        elif isinstance(value, ObjectId):
            out[key] = str(value)
    return out


# --------------------------------------------------------------------------
# Password hashing (stdlib PBKDF2 - no native-build dependency)
# --------------------------------------------------------------------------

def hash_password(password: str) -> str:
    salt = secrets.token_hex(config.PBKDF2_SALT_BYTES)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"),
        config.PBKDF2_ITERATIONS,
    ).hex()
    return f"{salt}${digest}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        salt, digest = stored_hash.split("$", 1)
    except ValueError:
        return False
    candidate = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"),
        config.PBKDF2_ITERATIONS,
    ).hex()
    return secrets.compare_digest(candidate, digest)


# --------------------------------------------------------------------------
# users
# --------------------------------------------------------------------------

class UserRepository:
    def __init__(self, db):
        self.collection = db["users"]
        self.collection.create_index("username", unique=True)
        self.collection.create_index("email", unique=True)

    def create_user(self, username: str, email: str, password: str) -> Dict[str, Any]:
        doc = {
            "username": username,
            "email": email,
            "password_hash": hash_password(password),
            "created_at": _now(),
        }
        result = self.collection.insert_one(doc)
        return self.get_by_id(result.inserted_id)

    def get_by_id(self, user_id: Any) -> Optional[Dict[str, Any]]:
        doc = self.collection.find_one({"_id": _oid(user_id)})
        return _serialize(doc)

    def get_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        return _serialize(self.collection.find_one({"username": username}))

    def verify_credentials(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        raw = self.collection.find_one({"username": username})
        if raw is None or not verify_password(password, raw["password_hash"]):
            return None
        return _serialize(raw)

    def update_user(self, user_id: Any, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        updates = {k: v for k, v in updates.items() if k not in ("_id", "password_hash")}
        doc = self.collection.find_one_and_update(
            {"_id": _oid(user_id)}, {"$set": updates},
            return_document=ReturnDocument.AFTER,
        )
        return _serialize(doc)

    def delete_user(self, user_id: Any) -> bool:
        result = self.collection.delete_one({"_id": _oid(user_id)})
        return result.deleted_count == 1


# --------------------------------------------------------------------------
# datasets
# --------------------------------------------------------------------------

class DatasetRepository:
    def __init__(self, db):
        self.collection = db["datasets"]
        self.collection.create_index("user_id")

    def create_dataset(self, user_id: Any, filename: str, file_path: str,
                        metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        doc = {
            "user_id": _oid(user_id),
            "filename": filename,
            "file_path": file_path,
            "metadata": metadata or {},   # n_rows, n_columns, size_bytes, etc.
            "status": "uploaded",
            "uploaded_at": _now(),
        }
        result = self.collection.insert_one(doc)
        return self.get_dataset(result.inserted_id)

    def get_dataset(self, dataset_id: Any) -> Optional[Dict[str, Any]]:
        return _serialize(self.collection.find_one({"_id": _oid(dataset_id)}))

    def list_datasets_for_user(self, user_id: Any) -> List[Dict[str, Any]]:
        cursor = self.collection.find({"user_id": _oid(user_id)}).sort("uploaded_at", -1)
        return [_serialize(d) for d in cursor]

    def update_dataset(self, dataset_id: Any, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        doc = self.collection.find_one_and_update(
            {"_id": _oid(dataset_id)}, {"$set": updates},
            return_document=ReturnDocument.AFTER,
        )
        return _serialize(doc)

    def delete_dataset(self, dataset_id: Any) -> bool:
        result = self.collection.delete_one({"_id": _oid(dataset_id)})
        return result.deleted_count == 1


# --------------------------------------------------------------------------
# analysis_reports
# --------------------------------------------------------------------------

class AnalysisReportRepository:
    def __init__(self, db):
        self.collection = db["analysis_reports"]
        self.collection.create_index("dataset_id")

    def create_report(self, dataset_id: Any, user_id: Any, report: Dict[str, Any]) -> Dict[str, Any]:
        doc = {
            "dataset_id": _oid(dataset_id),
            "user_id": _oid(user_id),
            "report": report,   # the full Phase 1 profiler/quality/risk/score payload
            "created_at": _now(),
        }
        result = self.collection.insert_one(doc)
        return self.get_report(result.inserted_id)

    def get_report(self, report_id: Any) -> Optional[Dict[str, Any]]:
        return _serialize(self.collection.find_one({"_id": _oid(report_id)}))

    def get_latest_for_dataset(self, dataset_id: Any) -> Optional[Dict[str, Any]]:
        # _id as a secondary sort key breaks ties when two docs share the
        # same created_at millisecond (BSON datetime resolution) - ObjectId
        # is monotonically increasing even within the same millisecond.
        doc = self.collection.find({"dataset_id": _oid(dataset_id)}) \
            .sort([("created_at", -1), ("_id", -1)]).limit(1)
        results = list(doc)
        return _serialize(results[0]) if results else None

    def list_for_dataset(self, dataset_id: Any) -> List[Dict[str, Any]]:
        """Oldest-first, matching CleaningHistoryRepository and
        ModelResultRepository - the frontend reverses all three the same
        way to show newest-first, so they must agree on this order."""
        cursor = self.collection.find({"dataset_id": _oid(dataset_id)}).sort("created_at", 1)
        return [_serialize(d) for d in cursor]

    def delete_report(self, report_id: Any) -> bool:
        result = self.collection.delete_one({"_id": _oid(report_id)})
        return result.deleted_count == 1


# --------------------------------------------------------------------------
# cleaning_history
# --------------------------------------------------------------------------

class CleaningHistoryRepository:
    def __init__(self, db):
        self.collection = db["cleaning_history"]
        self.collection.create_index("dataset_id")

    def create_entry(self, dataset_id: Any, user_id: Any, operations: List[Dict[str, Any]],
                      resulting_shape: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        doc = {
            "dataset_id": _oid(dataset_id),
            "user_id": _oid(user_id),
            "operations": operations,   # e.g. [{"type": "impute_missing", "column": "age", "strategy": "mean"}]
            "resulting_shape": resulting_shape or {},
            "created_at": _now(),
        }
        result = self.collection.insert_one(doc)
        return self.get_entry(result.inserted_id)

    def get_entry(self, entry_id: Any) -> Optional[Dict[str, Any]]:
        return _serialize(self.collection.find_one({"_id": _oid(entry_id)}))

    def list_for_dataset(self, dataset_id: Any) -> List[Dict[str, Any]]:
        cursor = self.collection.find({"dataset_id": _oid(dataset_id)}).sort("created_at", 1)
        return [_serialize(d) for d in cursor]

    def delete_entry(self, entry_id: Any) -> bool:
        result = self.collection.delete_one({"_id": _oid(entry_id)})
        return result.deleted_count == 1


# --------------------------------------------------------------------------
# model_results
# --------------------------------------------------------------------------

class ModelResultRepository:
    def __init__(self, db):
        self.collection = db["model_results"]
        self.collection.create_index("dataset_id")

    def create_result(self, dataset_id: Any, user_id: Any, stage: str,
                       model_type: str, metrics: Dict[str, Any],
                       details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if stage not in ("before", "after"):
            raise ValueError("stage must be 'before' or 'after'")
        doc = {
            "dataset_id": _oid(dataset_id),
            "user_id": _oid(user_id),
            "stage": stage,
            "model_type": model_type,
            "metrics": metrics,          # purely numeric - safe to diff/chart
            "details": details or {},    # problem_type, row counts, confusion matrix, etc.
            "created_at": _now(),
        }
        result = self.collection.insert_one(doc)
        return self.get_result(result.inserted_id)

    def get_result(self, result_id: Any) -> Optional[Dict[str, Any]]:
        return _serialize(self.collection.find_one({"_id": _oid(result_id)}))

    def list_for_dataset(self, dataset_id: Any) -> List[Dict[str, Any]]:
        cursor = self.collection.find({"dataset_id": _oid(dataset_id)}).sort("created_at", 1)
        return [_serialize(d) for d in cursor]

    def compare_before_after(self, dataset_id: Any, target_column: Optional[str] = None) -> Dict[str, Any]:
        """Latest 'before' vs latest 'after' *for the same target column* -
        never pairs results trained on different targets. If target_column
        isn't given, it's taken from whichever target was used most
        recently (the latest 'before' entry)."""
        if target_column is None:
            latest_before_any = list(self.collection.find(
                {"dataset_id": _oid(dataset_id), "stage": "before"}
            ).sort([("created_at", -1), ("_id", -1)]).limit(1))
            if not latest_before_any:
                return {"before": None, "after": None, "deltas": {}, "target_column": None}
            target_column = latest_before_any[0]["details"].get("target_column")

        before = list(self.collection.find(
            {"dataset_id": _oid(dataset_id), "stage": "before", "details.target_column": target_column}
        ).sort([("created_at", -1), ("_id", -1)]).limit(1))
        after = list(self.collection.find(
            {"dataset_id": _oid(dataset_id), "stage": "after", "details.target_column": target_column}
        ).sort([("created_at", -1), ("_id", -1)]).limit(1))

        before_doc = _serialize(before[0]) if before else None
        after_doc = _serialize(after[0]) if after else None

        deltas = {}
        if before_doc and after_doc:
            for metric, before_val in before_doc["metrics"].items():
                after_val = after_doc["metrics"].get(metric)
                if isinstance(before_val, (int, float)) and isinstance(after_val, (int, float)):
                    deltas[metric] = round(after_val - before_val, 4)

        return {"before": before_doc, "after": after_doc, "deltas": deltas, "target_column": target_column}

    def delete_result(self, result_id: Any) -> bool:
        result = self.collection.delete_one({"_id": _oid(result_id)})
        return result.deleted_count == 1


# --------------------------------------------------------------------------
# Facade - one object Phase 3's FastAPI routes can depend-inject
# --------------------------------------------------------------------------

class MLReadyDatabase:
    def __init__(self, client: Optional[MongoClient] = None):
        self.client = get_client(client)
        self.db = self.client[config.MONGO_DB_NAME]
        self.users = UserRepository(self.db)
        self.datasets = DatasetRepository(self.db)
        self.analysis_reports = AnalysisReportRepository(self.db)
        self.cleaning_history = CleaningHistoryRepository(self.db)
        self.model_results = ModelResultRepository(self.db)

    def close(self):
        self.client.close()


if __name__ == "__main__":
    # Smoke test against a real local MongoDB (only runs if one is up).
    db = MLReadyDatabase()
    try:
        db.client.admin.command("ping")
        print("Connected to MongoDB at", config.MONGO_URI)
    except Exception as exc:
        print("Could not reach MongoDB:", exc)
        print("Start MongoDB Community Server locally, or run test_phase2.py "
              "which uses mongomock and needs no server.")
    finally:
        db.close()
