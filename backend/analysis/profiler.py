"""
MLReady - Dataset Profiler

Pure-function profiling: takes a pandas DataFrame, returns a JSON-serializable
dict describing its shape, types, and basic statistics. No side effects,
no I/O - this makes it trivial to unit test and to later wrap in a FastAPI
route (Phase 3) without changing this file.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_float(value) -> float | None:
    """Convert numpy/pandas scalars to plain python floats, handling NaN."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(f) else round(f, 4)


def infer_column_groups(df: pd.DataFrame) -> dict:
    """Split columns into numerical vs categorical.

    Booleans and low-unique integer columns are still treated as numerical
    here; ML-risk-specific reclassification (e.g. treating a 0/1 int column
    as a categorical target) happens later in ml_risks.py, not here, since
    profiling should describe the data as-is.
    """
    numerical_columns = df.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_columns = [c for c in df.columns if c not in numerical_columns]
    return {
        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns,
    }


def numeric_summary(df: pd.DataFrame, numerical_columns: list[str]) -> dict:
    """Min/max/mean/median/std/quartiles for every numerical column."""
    summary = {}
    for col in numerical_columns:
        series = df[col].dropna()
        if series.empty:
            summary[col] = {
                "min": None, "max": None, "mean": None, "median": None,
                "std": None, "q1": None, "q3": None,
            }
            continue
        summary[col] = {
            "min": _safe_float(series.min()),
            "max": _safe_float(series.max()),
            "mean": _safe_float(series.mean()),
            "median": _safe_float(series.median()),
            "std": _safe_float(series.std()),
            "q1": _safe_float(series.quantile(0.25)),
            "q3": _safe_float(series.quantile(0.75)),
        }
    return summary


def _semantic_type(series: pd.Series, column: str, n_rows: int) -> dict:
    """Infer a useful display/analysis type without silently changing values."""
    non_null = series.dropna()
    name = str(column).lower()
    if not len(non_null):
        return {"type": "empty", "confidence": 1.0, "evidence": "all values are missing"}
    if pd.api.types.is_bool_dtype(series):
        return {"type": "boolean", "confidence": 1.0, "evidence": "native boolean dtype"}
    if pd.api.types.is_datetime64_any_dtype(series):
        return {"type": "datetime", "confidence": 1.0, "evidence": "native datetime dtype"}
    if pd.api.types.is_numeric_dtype(series):
        if any(token in name for token in ("id", "uuid", "key")) and non_null.nunique() / max(1, len(non_null)) > .95:
            return {"type": "identifier", "confidence": .92, "evidence": "identifier-like name and near-unique values"}
        return {"type": "numeric", "confidence": 1.0, "evidence": "native numeric dtype"}

    text = non_null.astype(str).str.strip()
    unique_ratio = text.nunique() / max(1, len(text))
    date_ratio = pd.to_datetime(text, errors="coerce", format="mixed").notna().mean()
    email_ratio = text.str.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$").mean()
    url_ratio = text.str.match(r"^https?://", case=False).mean()
    if email_ratio >= .9:
        return {"type": "email", "confidence": round(float(email_ratio), 3), "evidence": "email pattern"}
    if url_ratio >= .9:
        return {"type": "url", "confidence": round(float(url_ratio), 3), "evidence": "URL pattern"}
    if date_ratio >= .9:
        return {"type": "datetime", "confidence": round(float(date_ratio), 3), "evidence": "parseable date values"}
    if unique_ratio >= .98 and len(text) >= 20:
        return {"type": "identifier", "confidence": round(float(unique_ratio), 3), "evidence": "near-unique values"}
    if unique_ratio <= .2 or text.nunique() <= 20:
        return {"type": "categorical", "confidence": round(float(1 - unique_ratio), 3), "evidence": "low cardinality"}
    return {"type": "text", "confidence": round(float(min(1, unique_ratio)), 3), "evidence": "free-form string values"}


def profile_dataset(df: pd.DataFrame) -> dict:
    """Build a JSON-safe profile with conservative semantic type evidence."""
    n_rows = int(len(df))
    n_columns = int(df.shape[1])
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / n_rows * 100) if n_rows else missing_counts * 0
    duplicate_count = int(df.duplicated().sum())
    duplicate_pct = round((duplicate_count / n_rows * 100), 2) if n_rows else 0.0
    groups = infer_column_groups(df)
    numerical_set = set(groups["numerical_columns"])
    numeric_summary_map = numeric_summary(df, groups["numerical_columns"])
    column_details = []
    ui_columns = []
    for col in df.columns:
        series = df[col]
        semantic = _semantic_type(series, col, n_rows)
        detail = {
            "column": str(col), "dtype": str(series.dtype),
            "semantic_type": semantic["type"], "semantic_confidence": semantic["confidence"],
            "semantic_evidence": semantic["evidence"],
            "missing_count": int(missing_counts[col]),
            "missing_pct": round(float(missing_pct[col]), 2) if n_rows else 0.0,
            "unique_count": int(series.nunique(dropna=True)),
        }
        if col in numerical_set and semantic["type"] == "numeric":
            stats = numeric_summary_map.get(col, {})
            detail.update({"min": stats.get("min"), "max": stats.get("max"), "mean": stats.get("mean")})
        column_details.append(detail)

        # Shape expected by the dashboard's column-profile table.
        kind = "numerical" if col in numerical_set else "categorical"
        ui_col = {
            "column": str(col), "kind": kind,
            "missing_pct": detail["missing_pct"], "unique_count": detail["unique_count"],
        }
        if kind == "numerical":
            ui_col.update({"min": detail.get("min"), "mean": detail.get("mean"), "max": detail.get("max")})
        else:
            value_counts = series.dropna().astype(str).value_counts()
            ui_col["top_values"] = {k: int(v) for k, v in value_counts.head(5).items()}
        ui_columns.append(ui_col)
    return {
        "n_rows": n_rows, "n_columns": n_columns, "columns": ui_columns,
        "column_details": column_details,
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": {col: {"count": int(missing_counts[col]), "percentage": round(float(missing_pct[col]), 2) if n_rows else 0.0} for col in df.columns},
        "unique_values": {col: int(df[col].nunique(dropna=True)) for col in df.columns},
        "duplicate_rows": {"count": duplicate_count, "percentage": duplicate_pct},
        "numerical_columns": groups["numerical_columns"], "categorical_columns": groups["categorical_columns"],
        "numeric_summary": numeric_summary_map,
    }
