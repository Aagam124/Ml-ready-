"""
io_utils.py
-----------
Shared helpers for reading uploaded dataset files safely.
"""

import os

import pandas as pd


class DatasetLoadError(ValueError):
    """Raised for anything wrong with a dataset file - missing, empty, corrupted, unsupported."""


def load_dataframe(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise DatasetLoadError(f"File not found: {file_path}")
    if os.path.getsize(file_path) == 0:
        raise DatasetLoadError("File is empty")

    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == ".csv":
            df = pd.read_csv(file_path)
        elif ext in (".xlsx", ".xls"):
            df = pd.read_excel(file_path)
        else:
            raise DatasetLoadError(f"Unsupported file extension '{ext}'")
    except DatasetLoadError:
        raise
    except Exception as exc:
        raise DatasetLoadError(f"Could not parse file - it may be corrupted: {exc}")

    if df.shape[0] == 0 or df.shape[1] == 0:
        raise DatasetLoadError("File parsed but contains no data")

    return df
