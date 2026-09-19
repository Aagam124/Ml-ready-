"""
MLReady - Data Quality Checks

Everything here is a "is something structurally wrong with this column/row"
check, as opposed to ml_risks.py which is "will this hurt a trained model".
The split matters because quality issues apply to any tabular use case,
not just ML.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import NEAR_CONSTANT_THRESHOLD, HIGH_CARDINALITY_RATIO, HIGH_CARDINALITY_MIN_UNIQUE


def missing_value_report(df: pd.DataFrame) -> dict:
    n_rows = len(df)
    counts = df.isnull().sum()
    flagged = {}
    for col in df.columns:
        pct = round(float(counts[col] / n_rows * 100), 2) if n_rows else 0.0
        if counts[col] > 0:
            flagged[col] = {"count": int(counts[col]), "percentage": pct}
    return {
        "columns_with_missing": flagged,
        "total_missing_cells": int(counts.sum()),
        "overall_percentage": round(float(counts.sum() / (n_rows * len(df.columns)) * 100), 2)
        if n_rows and len(df.columns) else 0.0,
    }


def duplicate_report(df: pd.DataFrame) -> dict:
    n_rows = len(df)
    dup_count = int(df.duplicated().sum())
    return {
        "count": dup_count,
        "percentage": round(dup_count / n_rows * 100, 2) if n_rows else 0.0,
    }


def dtype_issue_report(df: pd.DataFrame) -> dict:
    """Flag columns that LOOK numeric but are stored as text/object.

    Heuristic: for object columns, try coercing to numeric; if the vast
    majority of non-null values convert cleanly, the column is probably
    mistyped (e.g. "1,200" strings, or numbers read in with stray whitespace).
    """
    issues = {}
    for col in df.select_dtypes(include=["object", "str"]).columns:
        series = df[col].dropna()
        if series.empty:
            continue
        coerced = pd.to_numeric(series.astype(str).str.replace(",", "", regex=False), errors="coerce")
        convertible_ratio = coerced.notna().mean()
        if convertible_ratio >= 0.9:
            issues[col] = {
                "detected_dtype": "object",
                "likely_intended_dtype": "numeric",
                "convertible_ratio": round(float(convertible_ratio), 3),
            }
    return issues


def constant_and_near_constant_columns(df: pd.DataFrame) -> dict:
    """Split into fully-constant columns and near-constant (low-variance) ones."""
    constant_cols = []
    near_constant_cols = {}
    for col in df.columns:
        series = df[col].dropna()
        if series.empty:
            continue
        top_freq_ratio = series.value_counts(normalize=True).iloc[0]
        n_unique = series.nunique()
        if n_unique == 1:
            constant_cols.append(col)
        elif top_freq_ratio >= NEAR_CONSTANT_THRESHOLD:
            near_constant_cols[col] = {
                "dominant_value_ratio": round(float(top_freq_ratio), 4),
            }
    return {"constant_columns": constant_cols, "near_constant_columns": near_constant_cols}


def high_cardinality_columns(df: pd.DataFrame) -> dict:
    """Categorical columns where almost every row has a distinct value.

    Requires BOTH a high ratio and an absolute floor of unique values so a
    50-row dataset doesn't get every text column flagged.
    """
    n_rows = len(df)
    flagged = {}
    if n_rows == 0:
        return flagged
    categorical_cols = df.select_dtypes(exclude=[np.number, "bool"]).columns
    for col in categorical_cols:
        n_unique = df[col].nunique(dropna=True)
        ratio = n_unique / n_rows
        if ratio >= HIGH_CARDINALITY_RATIO and n_unique >= HIGH_CARDINALITY_MIN_UNIQUE:
            flagged[col] = {"unique_count": int(n_unique), "unique_ratio": round(float(ratio), 3)}
    return flagged


def invalid_value_report(df: pd.DataFrame) -> dict:
    """Find non-finite numerics and common missing-value sentinels."""
    non_finite = {}
    sentinels = {}
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            count = int((~np.isfinite(series.dropna().to_numpy(dtype=float))).sum())
            if count:
                non_finite[col] = {"count": count, "kind": "infinite_or_nan"}
        else:
            values = series.dropna().astype(str).str.strip().str.lower()
            count = int(values.isin({"na", "n/a", "null", "none", "unknown", "missing", "-999", "999999"}).sum())
            if count:
                sentinels[col] = {"count": count, "values": sorted(values[values.isin({"na", "n/a", "null", "none", "unknown", "missing", "-999", "999999"})].unique().tolist())}
    return {"non_finite_values": non_finite, "possible_missing_sentinels": sentinels}


def missingness_patterns(df: pd.DataFrame) -> dict:
    """Summarize rows with multiple missing fields and co-missing pairs."""
    if df.empty or not len(df.columns):
        return {"rows_with_missing": 0, "complete_rows": 0, "top_patterns": [], "co_missing_pairs": []}
    mask = df.isna()
    pattern_counts = mask.astype(int).astype(str).agg("".join, axis=1).value_counts().head(10)
    pairs = []
    for i, a in enumerate(df.columns):
        for b in df.columns[i + 1:]:
            both = int((mask[a] & mask[b]).sum())
            if both:
                pairs.append({"columns": [a, b], "rows": both, "percentage": round(both / len(df) * 100, 2)})
    return {
        "rows_with_missing": int(mask.any(axis=1).sum()), "complete_rows": int((~mask.any(axis=1)).sum()),
        "top_patterns": [{"pattern": pattern, "rows": int(count)} for pattern, count in pattern_counts.items()],
        "co_missing_pairs": sorted(pairs, key=lambda x: x["rows"], reverse=True)[:10],
    }


def run_quality_checks(df: pd.DataFrame) -> dict:
    """Run every data-quality check and bundle results into one report."""
    return {
        "missing_values": missing_value_report(df),
        "missingness_patterns": missingness_patterns(df),
        "duplicates": duplicate_report(df),
        "dtype_issues": dtype_issue_report(df),
        "invalid_values": invalid_value_report(df),
        **constant_and_near_constant_columns(df),
        "high_cardinality_columns": high_cardinality_columns(df),
    }
