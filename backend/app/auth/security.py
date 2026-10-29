"""Password hashing (Argon2id), short-lived JWT access tokens, rotating refresh tokens."""
from __future__ import annotations

import hashlib
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError
from bson import ObjectId

from app.config import Settings
from app.db import Database
from app.errors import APIError

_hasher = PasswordHasher()


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password_hash: str, password: str) -> bool:
    try:
        return _hasher.verify(password_hash, password)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


def _now() -> datetime:
    return datetime.now(timezone.utc)


def create_access_token(user_id: ObjectId, settings: Settings) -> str:
    payload = {"sub": str(user_id), "type": "access", "iat": _now(), "exp": _now() + timedelta(minutes=settings.access_ttl_minutes)}
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_token(token: str, settings: Settings, expected: str) -> Dict[str, Any]:
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except jwt.ExpiredSignatureError:
        raise APIError(401, "token_expired", "Your session expired. Sign in again.")
    except jwt.InvalidTokenError:
        raise APIError(401, "invalid_token", "That sign-in token is not valid.")
    if payload.get("type") != expected:
        raise APIError(401, "invalid_token", "That sign-in token is not valid for this action.")
    return payload


def issue_refresh_token(database: Database, user_id: ObjectId, settings: Settings) -> str:
    jti = uuid.uuid4().hex
    expires = _now() + timedelta(days=settings.refresh_ttl_days)
    token = jwt.encode({"sub": str(user_id), "type": "refresh", "jti": jti, "exp": expires}, settings.jwt_secret, algorithm=settings.jwt_algorithm)
    database.refresh_tokens.insert_one({"jti": jti, "user_id": user_id, "expires_at": expires.replace(tzinfo=None), "used": False})
    return token


def rotate_refresh_token(database: Database, token: str, settings: Settings) -> Dict[str, Any]:
    """One-time use: a replayed refresh token revokes the whole family for that user."""
    payload = decode_token(token, settings, "refresh")
    doc = database.refresh_tokens.find_one_and_update({"jti": payload["jti"], "used": False}, {"$set": {"used": True}})
    if doc is None:
        database.refresh_tokens.delete_many({"user_id": ObjectId(payload["sub"])})
        raise APIError(401, "invalid_token", "That sign-in token was already used. Sign in again.")
    user_id = ObjectId(payload["sub"])
    return {"access_token": create_access_token(user_id, settings), "refresh_token": issue_refresh_token(database, user_id, settings),
            "token_type": "bearer", "expires_in": settings.access_ttl_minutes * 60}


def token_pair(database: Database, user_id: ObjectId, settings: Settings) -> Dict[str, Any]:
    return {"access_token": create_access_token(user_id, settings), "refresh_token": issue_refresh_token(database, user_id, settings),
            "token_type": "bearer", "expires_in": settings.access_ttl_minutes * 60}


def new_share_token() -> str:
    return secrets.token_urlsafe(24)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()
