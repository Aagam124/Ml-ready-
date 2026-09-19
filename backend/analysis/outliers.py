"""
MLReady - Outlier Detection

Three complementary methods on purpose: IQR and Z-score are univariate and
cheap, Isolation Forest is multivariate and catches points that only look
weird in combination. We report all three per column/dataset rather than
picking a winner - the recommendations layer explains that outliers may be
legitimate and should be reviewed, not blindly deleted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from config import IQR_MULTIPLIER, ZSCORE_THRESHOLD, ISOLATION_FOREST_CONTAMINATION, RANDOM_STATE


def iqr_outliers(df: pd.DataFrame, numerical_columns: list[str]) -> dict:
    result = {}
    n_rows = len(df)
    for col in numerical_columns:
        series = df[col].dropna()
        if series.empty:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower, upper = q1 - IQR_MULTIPLIER * iqr, q3 + IQR_MULTIPLIER * iqr
        mask = (series < lower) | (series > upper)
        count = int(mask.sum())
        if count > 0:
            result[col] = {
                "count": count,
                "percentage": round(count / n_rows * 100, 2) if n_rows else 0.0,
                "lower_bound": round(float(lower), 4),
                "upper_bound": round(float(upper), 4),
            }
    return result


def zscore_outliers(df: pd.DataFrame, numerical_columns: list[str]) -> dict:
    result = {}
    n_rows = len(df)
    for col in numerical_columns:
        series = df[col].dropna()
        if series.empty or series.std() == 0:
            continue
        z = (series - series.mean()) / series.std()
        count = int((z.abs() > ZSCORE_THRESHOLD).sum())
        if count > 0:
            result[col] = {
                "count": count,
                "percentage": round(count / n_rows * 100, 2) if n_rows else 0.0,
            }
    return result


def isolation_forest_outliers(df: pd.DataFrame, numerical_columns: list[str]) -> dict:
    """Multivariate outlier detection. Needs at least 2 numeric columns and
    enough rows to be meaningful; otherwise skipped rather than forced."""
    usable_cols = [c for c in numerical_columns if df[c].notna().sum() > 0]
    if len(usable_cols) < 2 or len(df) < 20:
        return {"applied": False, "reason": "Needs >=2 numeric columns and >=20 rows."}

    subset = df[usable_cols].dropna()
    if len(subset) < 20:
        return {"applied": False, "reason": "Too many rows dropped due to missing values."}

    model = IsolationForest(
        contamination=ISOLATION_FOREST_CONTAMINATION,
        random_state=RANDOM_STATE,
        n_estimators=100,
    )
    preds = model.fit_predict(subset)
    outlier_count = int((preds == -1).sum())
    return {
        "applied": True,
        "columns_used": usable_cols,
        "rows_evaluated": int(len(subset)),
        "outlier_count": outlier_count,
        "percentage": round(outlier_count / len(subset) * 100, 2),
    }


def robust_mad_outliers(df: pd.DataFrame, numerical_columns: list[str]) -> dict:
    """Median absolute deviation flags, robust to skew and extreme values."""
    result = {}
    for col in numerical_columns:
        values = df[col].dropna()
        if len(values) < 8:
            continue
        median = values.median()
        mad = (values - median).abs().median()
        if not mad or not np.isfinite(mad):
            continue
        modified_z = 0.6745 * (values - median) / mad
        flagged = modified_z.abs() > 3.5
        count = int(flagged.sum())
        if count:
            result[col] = {"count": count, "percentage": round(count / len(df) * 100, 2), "threshold": 3.5}
    return result


def detection_consensus(outlier_methods: dict, n_rows: int) -> dict:
    """Count columns independently supported by multiple detectors."""
    columns = set().union(*(set(v) for v in outlier_methods.values() if isinstance(v, dict)))
    consensus = {}
    for col in columns:
        methods = [name for name, result in outlier_methods.items() if isinstance(result, dict) and col in result]
        if methods:
            consensus[col] = {"methods": methods, "agreement": round(len(methods) / max(1, len(outlier_methods)), 2)}
    return consensus


def detect_outliers(df: pd.DataFrame, numerical_columns: list[str]) -> dict:
    iqr = iqr_outliers(df, numerical_columns)
    zscore = zscore_outliers(df, numerical_columns)
    mad = robust_mad_outliers(df, numerical_columns)
    methods = {"iqr": iqr, "zscore": zscore, "mad": mad}
    return {
        **methods,
        "consensus": detection_consensus(methods, len(df)),
        "isolation_forest": isolation_forest_outliers(df, numerical_columns),
        "note": "Outlier flags are evidence for review, not proof of invalid data. Consensus across methods is stronger evidence than a single flag.",
    }
