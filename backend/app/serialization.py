"""Turn Mongo documents into JSON-safe API payloads and parse ids safely."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from bson import ObjectId
from bson.errors import InvalidId

from app.errors import APIError


def parse_id(value: str, what: str = "resource") -> ObjectId:
    try:
        return ObjectId(value)
    except (InvalidId, TypeError):
        raise APIError(404, "not_found", f"{what} not found.")


def serialize(value: Any) -> Any:
    if isinstance(value, ObjectId):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            out["id" if k == "_id" else k] = serialize(v)
        return out
    if isinstance(value, (list, tuple)):
        return [serialize(v) for v in value]
    return value
