"""Mongo-backed fixed-window rate limiter (correct across several API processes)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import Depends, Request

from app.config import Settings, get_settings
from app.db import Database, get_database
from app.errors import APIError


def _check(database: Database, key: str, limit: int, window_s: int = 60) -> None:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    bucket = int(now.timestamp() // window_s)
    doc = database.rate_limits.find_one_and_update(
        {"_id": f"{key}:{bucket}"},
        {"$inc": {"n": 1}, "$setOnInsert": {"expire_at": now + timedelta(seconds=window_s * 2)}},
        upsert=True, return_document=True)
    if doc["n"] > limit:
        raise APIError(429, "rate_limited", f"Too many requests. Try again in {window_s - int(now.timestamp() % window_s)} seconds.")


def _ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    return (fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else "unknown"))


def auth_rate_limit(request: Request, database: Database = Depends(get_database), settings: Settings = Depends(get_settings)) -> None:
    _check(database, f"auth:{_ip(request)}", settings.auth_rate_limit_per_min)


def upload_rate_limit(request: Request, database: Database = Depends(get_database), settings: Settings = Depends(get_settings)) -> None:
    _check(database, f"upload:{_ip(request)}", settings.upload_rate_limit_per_min)
