"""Stream an upload to a temp file in chunks, hashing and size-limiting as we go."""
from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from typing import BinaryIO

from app.errors import APIError


@dataclass
class SavedUpload:
    path: str
    size: int
    sha256: str


def save_stream_to_tempfile(stream: BinaryIO, max_bytes: int, chunk_bytes: int, suffix: str = "") -> SavedUpload:
    fd, path = tempfile.mkstemp(prefix="mlready_", suffix=suffix)
    digest, size = hashlib.sha256(), 0
    try:
        with os.fdopen(fd, "wb") as out:
            while True:
                chunk = stream.read(chunk_bytes)
                if not chunk:
                    break
                size += len(chunk)
                if size > max_bytes:
                    raise APIError(
                        413, "file_too_large",
                        f"File exceeds the {max_bytes // (1024 * 1024)} MB upload limit.",
                    )
                digest.update(chunk)
                out.write(chunk)
    except BaseException:
        os.unlink(path)
        raise
    return SavedUpload(path=path, size=size, sha256=digest.hexdigest())
