"""Shared FastAPI dependencies: authentication and per-workspace authorisation.

Every route resolves the caller's workspace here and passes `ws["_id"]` into EVERY query.
"""
from __future__ import annotations

from datetime import datetime, timezone

from bson import ObjectId
from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pymongo import ReturnDocument

from app.auth.security import decode_token
from app.config import Settings, get_settings
from app.db import Database, get_database
from app.errors import APIError

bearer = HTTPBearer(auto_error=False)
ROLE_RANK = {"viewer": 0, "editor": 1, "owner": 2}


def get_current_user(creds: HTTPAuthorizationCredentials = Depends(bearer), database: Database = Depends(get_database),
                     settings: Settings = Depends(get_settings)) -> dict:
    if creds is None:
        raise APIError(401, "not_authenticated", "Sign in to continue.")
    payload = decode_token(creds.credentials, settings, "access")
    user = database.users.find_one({"_id": ObjectId(payload["sub"])})
    if user is None:
        raise APIError(401, "invalid_token", "That account no longer exists.")
    return user


def get_workspace(user: dict = Depends(get_current_user), database: Database = Depends(get_database)) -> dict:
    ws = database.workspaces.find_one({"members.user_id": user["_id"]}, sort=[("created_at", 1)])
    if ws is None:
        raise APIError(403, "no_workspace", "Your account has no workspace.")
    ws = dict(ws)
    ws["_role"] = next(m["role"] for m in ws["members"] if m["user_id"] == user["_id"])
    ws["_user_id"] = user["_id"]
    return ws


def editor_workspace(ws: dict = Depends(get_workspace)) -> dict:
    if ROLE_RANK.get(ws.get("_role", "owner"), 2) < ROLE_RANK["editor"]:
        raise APIError(403, "forbidden", "Your role in this workspace is read-only.")
    return ws


def default_workspace(database: Database = Depends(get_database)) -> dict:
    """Unauthenticated single-workspace mode, used only by tests and local tooling via dependency override."""
    return database.workspaces.find_one_and_update(
        {"slug": "default"},
        {"$setOnInsert": {"slug": "default", "name": "Default workspace", "members": [],
                          "created_at": datetime.now(timezone.utc).replace(tzinfo=None)}},
        upsert=True, return_document=ReturnDocument.AFTER)


def audit(database: Database, ws: dict, action: str, ref: str) -> None:
    database.audit.insert_one({"user_id": ws.get("_user_id"), "workspace_id": ws["_id"], "action": action, "object_ref": ref,
                               "at": datetime.now(timezone.utc).replace(tzinfo=None)})
