"""MongoDB access: one Database object bundling the collections and GridFS."""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Optional

import gridfs
from pymongo import ASCENDING, DESCENDING, MongoClient

from app.config import Settings, get_settings


class Database:
    def __init__(self, client: Any, db_name: str):
        self.client = client
        self.db = client[db_name]
        self.fs = gridfs.GridFS(self.db)
        self.workspaces = self.db["workspaces"]
        self.datasets = self.db["datasets"]
        self.versions = self.db["dataset_versions"]
        self.users = self.db["users"]
        self.refresh_tokens = self.db["refresh_tokens"]
        self.analyses = self.db["analyses"]
        self.findings = self.db["findings"]
        self.plans = self.db["cleaning_plans"]
        self.model_runs = self.db["model_runs"]
        self.reports = self.db["reports"]
        self.jobs = self.db["jobs"]
        self.audit = self.db["audit_log"]
        self.rate_limits = self.db["rate_limits"]
        self.quality_rules = self.db["quality_rules"]

    def ping(self) -> None:
        self.client.admin.command("ping")


def build_database(client: Optional[Any] = None, settings: Optional[Settings] = None) -> Database:
    settings = settings or get_settings()
    client = client or MongoClient(settings.mongo_uri, serverSelectionTimeoutMS=5000)
    return Database(client, settings.db_name)


def ensure_indexes(database: Database) -> None:
    try:
        if "username_1" in [idx.get("name") for idx in database.users.list_indexes()]:
            database.users.drop_index("username_1")
    except Exception:
        pass
    database.workspaces.create_index([("slug", ASCENDING)], unique=True, sparse=True)
    database.datasets.create_index([("workspace_id", ASCENDING), ("created_at", DESCENDING)])
    database.datasets.create_index([("name", "text")])
    database.versions.create_index([("dataset_id", ASCENDING), ("created_at", ASCENDING)])
    database.versions.create_index([("workspace_id", ASCENDING), ("content_hash", ASCENDING)])
    database.workspaces.create_index([("members.user_id", ASCENDING)])
    database.workspaces.create_index([("expires_at", ASCENDING)], sparse=True)
    database.users.create_index([("email", ASCENDING)], unique=True)
    database.refresh_tokens.create_index([("expires_at", ASCENDING)], expireAfterSeconds=0)
    database.refresh_tokens.create_index([("jti", ASCENDING)], unique=True)
    database.analyses.create_index([("workspace_id", ASCENDING), ("created_at", DESCENDING)])
    database.analyses.create_index([("version_id", ASCENDING), ("created_at", DESCENDING)])
    database.findings.create_index([("analysis_id", ASCENDING), ("severity", ASCENDING)])
    database.plans.create_index([("analysis_id", ASCENDING)])
    database.model_runs.create_index([("workspace_id", ASCENDING), ("created_at", DESCENDING)])
    database.reports.create_index([("share_token", ASCENDING)], sparse=True)
    database.reports.create_index([("share_expires_at", ASCENDING)], sparse=True)   # expired share tokens are cleared by the worker janitor
    database.jobs.create_index([("status", ASCENDING), ("run_after", ASCENDING), ("created_at", ASCENDING)])
    database.jobs.create_index([("expire_at", ASCENDING)], expireAfterSeconds=0)        # TTL: only set on finished jobs
    database.audit.create_index([("user_id", ASCENDING), ("at", DESCENDING)])
    database.rate_limits.create_index([("expire_at", ASCENDING)], expireAfterSeconds=0)
    database.quality_rules.create_index([("workspace_id", ASCENDING), ("created_at", DESCENDING)])


@lru_cache
def get_database() -> Database:
    """FastAPI dependency. Tests override it with a mongomock-backed Database."""
    return build_database()
