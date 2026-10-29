"""Upload safety: filename, extension, magic bytes, zip-bomb guard."""
from __future__ import annotations

import os
import re
import zipfile
from pathlib import Path

from app.config import Settings
from app.errors import APIError

ALLOWED_EXTENSIONS = {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".parquet"}


def sanitize_filename(filename: str | None) -> str:
    base = os.path.basename((filename or "").replace("\\", "/")).strip()
    base = re.sub(r"[^\w.\- ]+", "_", base)[:150].strip(" .")
    if not base:
        raise APIError(422, "invalid_filename", "The uploaded file has no usable name.")
    return base


def validate_extension(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise APIError(
            422, "unsupported_file_type",
            f"Unsupported file type '{ext or 'none'}'.",
            {"allowed": sorted(ALLOWED_EXTENSIONS)},
        )
    return ext


def validate_magic(path: str, ext: str, settings: Settings) -> None:
    """Check the real content matches the extension, and refuse zip bombs."""
    with open(path, "rb") as f:
        head = f.read(4096)
    if not head:
        raise APIError(422, "empty_file", "The uploaded file is empty.")

    if ext == ".xlsx":
        if not head.startswith(b"PK"):
            raise APIError(422, "content_mismatch", "File is not a valid .xlsx workbook.")
        try:
            with zipfile.ZipFile(path) as z:
                total = sum(i.file_size for i in z.infolist())
        except zipfile.BadZipFile:
            raise APIError(422, "content_mismatch", "File is not a valid .xlsx workbook.")
        if total > settings.max_xlsx_uncompressed_mb * 1024 * 1024:
            raise APIError(413, "file_too_large", "Workbook expands to an unsafe size.")
    elif ext == ".xls":
        if not head.startswith(b"\xd0\xcf\x11\xe0"):
            raise APIError(422, "content_mismatch", "File is not a valid .xls workbook.")
    elif ext == ".parquet":
        if not head.startswith(b"PAR1"):
            raise APIError(422, "content_mismatch", "File is not a valid Parquet file.")
    else:  # delimited text
        if b"\x00" in head or head.startswith((b"PK", b"\xd0\xcf\x11\xe0", b"PAR1", b"%PDF")):
            raise APIError(422, "content_mismatch", "File does not look like delimited text.")
