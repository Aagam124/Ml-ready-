"""MLReady API entry point.  Run:  uvicorn app.main:app --reload --port 8000"""
from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, RedirectResponse

from app.api import analyses, auth, cleaning, datasets, jobs, models, reports, rules, workspaces
from app.config import get_settings
from app.db import Database, ensure_indexes, get_database
from app.errors import APIError, install_error_handlers


APP_VERSION = "1.0.0"
METRICS = {"requests": {}, "seconds": 0.0}      # per-process counters; the worker has its own


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps({"level": record.levelname, "logger": record.name, "message": record.getMessage()})


def configure_logging() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(logging.INFO)


@asynccontextmanager
async def lifespan(_: FastAPI):
    configure_logging()
    ensure_indexes(get_database())  # fail fast if MongoDB is unreachable
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="MLReady API", version=APP_VERSION, lifespan=lifespan,
                  description="MLReady: tells you whether a dataset is ready for machine learning, why not, how to fix it, and proves the fix worked.")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_origin_regex=r"https://.*\.onrender\.com",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    install_error_handlers(app)

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        start = time.perf_counter()
        response = await call_next(request)
        response.headers["x-request-id"] = rid
        METRICS["requests"][(request.method, response.status_code)] = METRICS["requests"].get((request.method, response.status_code), 0) + 1
        METRICS["seconds"] += time.perf_counter() - start
        logging.getLogger("mlready.http").info(
            "%s %s -> %s in %dms rid=%s", request.method, request.url.path,
            response.status_code, (time.perf_counter() - start) * 1000, rid)
        return response

    for r in (auth.router, datasets.router, analyses.router, cleaning.router, models.router, jobs.router, reports.router, rules.router, workspaces.router):
        app.include_router(r, prefix="/api/v1")

    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse(url="/docs")

    @app.get("/api/v1/version", tags=["ops"])
    def version():
        from app.engine.config import ENGINE_VERSION
        from app.engine.scoring import SCORE_VERSION
        return {"api": APP_VERSION, "engine": ENGINE_VERSION, "score": SCORE_VERSION}

    @app.get("/metrics", tags=["ops"], response_class=PlainTextResponse)
    def metrics(database: Database = Depends(get_database)):
        lines = ["# HELP mlready_http_requests_total HTTP requests by method and status.", "# TYPE mlready_http_requests_total counter"]
        for (m, st), n in sorted(METRICS["requests"].items()):
            lines.append(f'mlready_http_requests_total{{method="{m}",status="{st}"}} {n}')
        lines += ["# TYPE mlready_http_request_seconds_sum counter", f'mlready_http_request_seconds_sum {METRICS["seconds"]:.4f}',
                  "# TYPE mlready_jobs gauge"]
        for status in ("queued", "running", "failed"):
            lines.append(f'mlready_jobs{{status="{status}"}} {database.jobs.count_documents({"status": status})}')
        return "\n".join(lines) + "\n"

    @app.get("/health", tags=["ops"])
    def health():
        return {"status": "ok", "version": app.version}

    @app.get("/ready", tags=["ops"])
    def ready(database: Database = Depends(get_database)):
        try:
            database.ping()
        except Exception:
            raise APIError(503, "db_unavailable", "MongoDB is not reachable.")
        return {"status": "ready"}

    return app


app = create_app()
