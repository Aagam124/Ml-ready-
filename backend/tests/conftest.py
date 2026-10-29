import os
import mongomock
import pytest
from fastapi.testclient import TestClient
from mongomock.gridfs import enable_gridfs_integration

enable_gridfs_integration()

from app.config import Settings, get_settings  # noqa: E402
from app.db import Database, ensure_indexes, get_database  # noqa: E402
from app.api.deps import default_workspace, get_workspace  # noqa: E402
from app.main import create_app  # noqa: E402


@pytest.fixture()
def settings():
    return Settings(max_upload_mb=5) if not os.environ.get("MLREADY_TEST_BIG") else Settings(max_upload_mb=300)


@pytest.fixture()
def database():
    db = Database(mongomock.MongoClient(), "mlready_test")
    ensure_indexes(db)
    return db


@pytest.fixture()
def client(database, settings):
    """Single default workspace, no sign-in (Phase 1/2 style tests). Auth is covered by `auth_client`."""
    app = create_app()
    app.dependency_overrides[get_database] = lambda: database
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_workspace] = default_workspace
    return TestClient(app)


@pytest.fixture()
def auth_client(database, settings):
    """Real authentication and per-workspace authorisation."""
    app = create_app()
    app.dependency_overrides[get_database] = lambda: database
    app.dependency_overrides[get_settings] = lambda: settings
    return TestClient(app)
