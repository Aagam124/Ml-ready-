"""Central settings, read from MLREADY_* environment variables."""
from __future__ import annotations

from functools import lru_cache

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_JWT_SECRET = "dev-only-change-me"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MLREADY_", env_file=".env", extra="ignore")

    env: str = "development"
    mongo_uri: str = "mongodb://localhost:27017"
    db_name: str = "mlready"
    jwt_secret: str = DEFAULT_JWT_SECRET  # used by auth (Phase 0 follow-up)
    cors_origins: list[str] = [
        "http://localhost:5173",
        "https://mlready-app-2026.web.app",
        "https://mlready-app-2026.firebaseapp.com",
    ]

    # Auth
    jwt_algorithm: str = "HS256"
    access_ttl_minutes: int = 15
    refresh_ttl_days: int = 7
    guest_ttl_hours: int = 24
    auth_rate_limit_per_min: int = 10
    upload_rate_limit_per_min: int = 20

    # Jobs / worker
    job_max_attempts: int = 3
    job_stale_seconds: int = 45          # a running job without a heartbeat this long is re-queued
    job_ttl_days: int = 7
    worker_poll_seconds: float = 0.5
    analysis_sample_threshold: int = 200_000

    # Upload limits
    max_upload_mb: int = 200
    upload_chunk_bytes: int = 1024 * 1024
    max_columns: int = 2000
    max_xlsx_uncompressed_mb: int = 1000

    @model_validator(mode="after")
    def _refuse_default_secret_in_production(self) -> "Settings":
        if self.env.lower() == "production" and self.jwt_secret == DEFAULT_JWT_SECRET:
            raise ValueError("MLREADY_JWT_SECRET must be set to a strong value in production.")
        return self

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
