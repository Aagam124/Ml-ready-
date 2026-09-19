"""
routes/auth.py
---------------
POST /auth/register
POST /auth/login
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, EmailStr, Field

from database.mongodb import MLReadyDatabase
from dependencies import create_access_token, get_current_user, get_db

router = APIRouter(prefix="/auth", tags=["auth"])


class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=32)
    email: EmailStr
    password: str = Field(min_length=8, description="Minimum 8 characters")


class LoginRequest(BaseModel):
    username: str
    password: str


class UserResponse(BaseModel):
    id: str
    username: str
    email: str
    created_at: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


@router.post("/register", response_model=UserResponse, status_code=201)
def register(body: RegisterRequest, db: MLReadyDatabase = Depends(get_db)):
    if db.users.get_by_username(body.username) is not None:
        raise HTTPException(status_code=409, detail="Username already taken")
    try:
        user = db.users.create_user(body.username, body.email, body.password)
    except Exception:
        # Covers the unique-index race / duplicate email case
        raise HTTPException(status_code=409, detail="Username or email already registered")
    return user


@router.post("/login", response_model=TokenResponse)
def login(body: LoginRequest, db: MLReadyDatabase = Depends(get_db)):
    user = db.users.verify_credentials(body.username, body.password)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    token = create_access_token(user_id=user["id"])
    return TokenResponse(access_token=token)


@router.get("/me", response_model=UserResponse)
def get_me(user: dict = Depends(get_current_user)):
    """Lets the frontend restore a session on page refresh - it only has the
    JWT in localStorage, not the user's profile, until it calls this."""
    return user
