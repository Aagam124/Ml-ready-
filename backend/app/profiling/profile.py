"""Dataset and column profiling. Output is plain JSON-safe dicts stored on the version."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from app.profiling.semantic import infer_semantic_type

NUMERIC_TYPES = {"continuous", "count", "discrete"}
CATEGORY_TYPES = {"categorical", "binary"}


def _num(x: Any) -> Optional[float]:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return None if (math.isnan(f) or math.isinf(f)) else round(f, 6)


def _pct(part: float, whole: float) -> float:
    return round(part / whole * 100, 2) if whole else 0.0


def _numeric_stats(values: pd.Series) -> Dict[str, Any]:
    v = values.dropna().astype(float)
    if v.empty:
        return {}
    q = v.quantile([0.25, 0.5, 0.75])
    return {
        "min": _num(v.min()), "max": _num(v.max()), "mean": _num(v.mean()),
        "std": _num(v.std()), "median": _num(q[0.5]),
        "q25": _num(q[0.25]), "q75": _num(q[0.75]), "skew": _num(v.skew()),
        "n_negative": int((v < 0).sum()), "n_zero": int((v == 0).sum()),
    }


def _column_profile(s: pd.Series, name: str, n_rows: int) -> Dict[str, Any]:
    n_missing = int(s.isna().sum())
    n_nonnull = n_rows - n_missing
    n_unique = int(s.nunique())
    is_text = pd.api.types.is_string_dtype(s) or s.dtype == object
    n_blank = int((s.dropna().astype(str).str.strip() == "").sum()) if is_text else 0

    sem = infer_semantic_type(s, name, n_unique=n_unique, n_nonnull=n_nonnull)
    flags = list(sem.flags)
    if n_blank:
        flags.append("blank_strings")

    col: Dict[str, Any] = {
        "name": name,
        "dtype": str(s.dtype),
        "semantic_type": sem.type,
        "semantic_confidence": round(sem.confidence, 3),
        "semantic_evidence": sem.evidence,
        "flags": sorted(set(flags)),
        "n_missing": n_missing,
        "missing_pct": _pct(n_missing, n_rows),
        "n_unique": n_unique,
        "unique_pct": _pct(n_unique, n_nonnull),
    }
    if n_blank:
        col["n_blank_strings"] = n_blank

    if sem.type in NUMERIC_TYPES:
        values = s if pd.api.types.is_numeric_dtype(s) else pd.to_numeric(
            s.astype(str).str.replace(",", "", regex=False), errors="coerce")
        col["stats"] = _numeric_stats(values)
    elif sem.type in CATEGORY_TYPES:
        counts = s.value_counts(dropna=True).head(5)
        col["top_values"] = [
            {"value": str(k), "count": int(c), "pct": _pct(c, n_nonnull)} for k, c in counts.items()
        ]
    elif sem.type == "datetime":
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(s, errors="coerce", format="mixed", utc=True)
        if parsed.notna().any():
            col["stats"] = {"min": parsed.min().isoformat(), "max": parsed.max().isoformat()}
    elif sem.type == "text":
        col["stats"] = {"avg_length": _num(s.dropna().astype(str).str.len().mean())}
    return col


def profile_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    n_rows, n_cols = df.shape
    cells = n_rows * n_cols
    n_missing_cells = int(df.isna().sum().sum())
    n_dupes = int(df.duplicated().sum())

    columns: List[Dict[str, Any]] = [_column_profile(df[c], str(c), n_rows) for c in df.columns]

    summary: Dict[str, int] = {}
    for c in columns:
        summary[c["semantic_type"]] = summary.get(c["semantic_type"], 0) + 1

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "memory_bytes": int(df.memory_usage(deep=True).sum()),
        "missing": {
            "cells": n_missing_cells,
            "pct": _pct(n_missing_cells, cells),
            "denominator": f"{cells} cells ({n_rows} rows x {n_cols} columns)",
            "columns_with_missing": sum(1 for c in columns if c["n_missing"] > 0),
        },
        "duplicate_rows": {
            "count": n_dupes,
            "pct": _pct(n_dupes, n_rows),
            "denominator": f"{n_rows} rows",
        },
        "semantic_summary": summary,
        "columns": columns,
    }


def schema_from_profile(profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        {"name": c["name"], "dtype": c["dtype"], "semantic_type": c["semantic_type"],
         "confidence": c["semantic_confidence"]}
        for c in profile["columns"]
    ]
