"""
dependencies.py
----------------
Shared FastAPI dependencies: the MongoDB connection and JWT-based auth.

get_db() is overridden in tests (see test_phase3.py) via
app.dependency_overrides[get_db] to point at a mongomock-backed database
instead of a real MongoDB server - no route code needs to know or care.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

import config
from database.mongodb import MLReadyDatabase

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

# Lazily-created singleton so importing this module (or running tests, which
# override get_db entirely) never requires a live MongoDB connection.
_db_instance: Optional[MLReadyDatabase] = None


def get_db() -> MLReadyDatabase:
    global _db_instance
    if _db_instance is None:
        _db_instance = MLReadyDatabase()
    return _db_instance


def create_access_token(user_id: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=config.JWT_EXPIRE_MINUTES)
    payload = {"sub": user_id, "exp": expire}
    return jwt.encode(payload, config.JWT_SECRET_KEY, algorithm=config.JWT_ALGORITHM)


def decode_access_token(token: str) -> str:
    """Returns the user_id ('sub' claim) or raises HTTPException(401)."""
    credentials_error = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, config.JWT_SECRET_KEY, algorithms=[config.JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired",
                             headers={"WWW-Authenticate": "Bearer"})
    except jwt.InvalidTokenError:
        raise credentials_error

    user_id = payload.get("sub")
    if user_id is None:
        raise credentials_error
    return user_id


def get_current_user(token: str = Depends(oauth2_scheme), db: MLReadyDatabase = Depends(get_db)) -> dict:
    user_id = decode_access_token(token)
    try:
        user = db.users.get_by_id(user_id)
    except ValueError:
        user = None
    if user is None:
        raise HTTPException(status_code=401, detail="User not found",
                             headers={"WWW-Authenticate": "Bearer"})
    return user
