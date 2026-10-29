"""Register, login, refresh, guest (demo) sessions, me."""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

from bson import ObjectId
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field, field_validator

from app.api.deps import get_current_user
from app.auth.ratelimit import auth_rate_limit
from app.auth.security import hash_password, rotate_refresh_token, token_pair, verify_password
from app.config import Settings, get_settings
from app.db import Database, get_database
from app.errors import APIError
from app.serialization import serialize

router = APIRouter(prefix="/auth", tags=["auth"])
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Credentials(BaseModel):
    email: str = Field(max_length=254)
    password: str = Field(min_length=10, max_length=200)

    @field_validator("email")
    @classmethod
    def _email(cls, v: str) -> str:
        v = v.strip().lower()
        if not EMAIL_RE.match(v):
            raise ValueError("Enter a valid email address.")
        return v


class RefreshBody(BaseModel):
    refresh_token: str


def _new_workspace(database: Database, user_id: ObjectId, name: str, expires_at=None) -> ObjectId:
    ws = {"_id": ObjectId(), "name": name, "owner_id": user_id, "members": [{"user_id": user_id, "role": "owner"}], "created_at": _now()}
    if expires_at:
        ws["expires_at"] = expires_at
    database.workspaces.insert_one(ws)
    return ws["_id"]


@router.post("/register", status_code=201, dependencies=[Depends(auth_rate_limit)])
def register(body: Credentials, database: Database = Depends(get_database), settings: Settings = Depends(get_settings)):
    user = {"_id": ObjectId(), "email": body.email, "password_hash": hash_password(body.password), "created_at": _now(),
            "plan": "free", "settings": {}, "is_guest": False}
    try:
        database.users.insert_one(user)
    except Exception as e:      # duplicate key
        if "duplicate" in str(e).lower() or "E11000" in str(e):
            raise APIError(409, "email_taken", "An account with that email already exists. Try signing in.")
        raise
    _new_workspace(database, user["_id"], "My workspace")
    return {**token_pair(database, user["_id"], settings), "user": serialize({"id": user["_id"], "email": user["email"]})}


@router.post("/login", dependencies=[Depends(auth_rate_limit)])
def login(body: Credentials, database: Database = Depends(get_database), settings: Settings = Depends(get_settings)):
    user = database.users.find_one({"email": body.email})
    # Same message for unknown email and wrong password: do not reveal which accounts exist.
    if user is None or not verify_password(user["password_hash"], body.password):
        raise APIError(401, "invalid_credentials", "Email or password is not correct.")
    return {**token_pair(database, user["_id"], settings), "user": serialize({"id": user["_id"], "email": user["email"]})}


@router.post("/guest", status_code=201, dependencies=[Depends(auth_rate_limit)])
def guest(database: Database = Depends(get_database), settings: Settings = Depends(get_settings)):
    """Instant demo session. Everything it creates is deleted automatically after guest_ttl_hours."""
    expires = _now() + timedelta(hours=settings.guest_ttl_hours)
    user = {"_id": ObjectId(), "email": f"guest-{ObjectId()}@guest.invalid", "password_hash": "", "created_at": _now(), "plan": "guest",
            "settings": {}, "is_guest": True, "expires_at": expires}
    database.users.insert_one(user)
    _new_workspace(database, user["_id"], "Demo workspace", expires_at=expires)
    return {**token_pair(database, user["_id"], settings), "user": {"id": str(user["_id"]), "email": None, "guest": True},
            "expires_at": expires.isoformat() + "Z"}


@router.post("/refresh", dependencies=[Depends(auth_rate_limit)])
def refresh(body: RefreshBody, database: Database = Depends(get_database), settings: Settings = Depends(get_settings)):
    return rotate_refresh_token(database, body.refresh_token, settings)


@router.get("/me")
def me(user: dict = Depends(get_current_user)):
    return serialize({"id": user["_id"], "email": None if user.get("is_guest") else user["email"], "plan": user.get("plan"),
                      "guest": bool(user.get("is_guest")), "expires_at": user.get("expires_at"), "settings": user.get("settings", {})})
