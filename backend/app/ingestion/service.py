"""Ingest one file: read -> profile -> store raw + Parquet in GridFS -> create docs."""
from __future__ import annotations

import logging
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

from bson import ObjectId

from app.config import Settings
from app.db import Database
from app.ingestion.reader import read_dataframe, write_parquet
from app.profiling.profile import profile_dataframe, schema_from_profile

log = logging.getLogger("mlready.ingest")


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def ingest_file(
    database: Database,
    workspace_id: ObjectId,
    path: str,
    filename: str,
    ext: str,
    size: int,
    sha256: str,
    settings: Settings,
    name: Optional[str] = None,
    sheet: Optional[str] = None,
) -> Tuple[dict, dict]:
    t0 = time.perf_counter()
    df, report = read_dataframe(path, ext, settings, sheet=sheet)
    t_read = time.perf_counter()
    profile = profile_dataframe(df)
    t_profile = time.perf_counter()

    fd, parquet_path = tempfile.mkstemp(prefix="mlready_", suffix=".parquet")
    os.close(fd)
    stored = []
    try:
        write_parquet(df, parquet_path)
        with open(path, "rb") as raw, open(parquet_path, "rb") as pq:
            raw_id = database.fs.put(raw, filename=filename, workspace_id=workspace_id, kind="raw")
            stored.append(raw_id)
            pq_id = database.fs.put(pq, filename=Path(filename).stem + ".parquet",
                                    workspace_id=workspace_id, kind="parquet")
            stored.append(pq_id)
        t_store = time.perf_counter()

        now = _now()
        dataset_id, version_id = ObjectId(), ObjectId()
        version = {
            "_id": version_id,
            "dataset_id": dataset_id,
            "workspace_id": workspace_id,
            "parent_version_id": None,
            "gridfs_file_id": raw_id,
            "parquet_file_id": pq_id,
            "original_filename": filename,
            "file_ext": ext,
            "size_bytes": size,
            "content_hash": sha256,
            "n_rows": profile["n_rows"],
            "n_cols": profile["n_cols"],
            "schema": schema_from_profile(profile),
            "profile": profile,
            "ingest": {
                "encoding": report.encoding, "delimiter": report.delimiter,
                "sheet": report.sheet, "sheets_available": report.sheets_available,
                "renamed_columns": report.renamed_columns, "warnings": report.warnings,
                "timings_ms": {
                    "read": round((t_read - t0) * 1000),
                    "profile": round((t_profile - t_read) * 1000),
                    "store": round((t_store - t_profile) * 1000),
                    "total": round((t_store - t0) * 1000),
                },
            },
            "created_by_step": "upload",
            "created_at": now,
        }
        dataset = {
            "_id": dataset_id,
            "workspace_id": workspace_id,
            "name": (name or Path(filename).stem)[:200],
            "current_version_id": version_id,
            "n_versions": 1,
            "created_by": None,
            "created_at": now,
            "updated_at": now,
        }
        database.datasets.insert_one(dataset)
        database.versions.insert_one(version)
    except BaseException:
        for fid in stored:
            try:
                database.fs.delete(fid)
            except Exception:  # best-effort cleanup
                pass
        raise
    finally:
        os.unlink(parquet_path)

    log.info("ingested dataset=%s rows=%d cols=%d bytes=%d total_ms=%d",
             dataset_id, profile["n_rows"], profile["n_cols"], size, version["ingest"]["timings_ms"]["total"])
    return dataset, version
