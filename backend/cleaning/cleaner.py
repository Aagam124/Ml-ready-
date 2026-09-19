"""
cleaner.py
----------
Phase 3 - Data cleaning operations.

DataCleaner never mutates the DataFrame it's given - it works on a copy and
returns a new one, so callers always keep:

    original_df -> DataCleaner(original_df).apply_operations([...]) -> clean_df

Every operation is logged (what ran, what changed) so the log can be stored
verbatim in the cleaning_history collection (Phase 2) and shown to the user.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

import config


class CleaningError(ValueError):
    """Raised when a requested operation can't be applied (bad column, bad strategy, etc.)."""


class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.log: List[Dict[str, Any]] = []

    # -- individual operations -------------------------------------------

    def impute_missing(self, column: str, strategy: str) -> "DataCleaner":
        if column not in self.df.columns:
            raise CleaningError(f"Column '{column}' not found")

        series = self.df[column]
        missing_before = int(series.isna().sum())
        if missing_before == 0:
            self.log.append({
                "type": "impute_missing", "column": column, "strategy": strategy,
                "rows_affected": 0, "note": "no missing values - no-op",
            })
            return self

        if strategy == "mean":
            if not pd.api.types.is_numeric_dtype(series):
                raise CleaningError(f"'mean' requires a numeric column; '{column}' is not numeric")
            fill_value = series.mean()
        elif strategy == "median":
            if not pd.api.types.is_numeric_dtype(series):
                raise CleaningError(f"'median' requires a numeric column; '{column}' is not numeric")
            fill_value = series.median()
        elif strategy == "mode":
            mode_values = series.mode(dropna=True)
            if mode_values.empty:
                raise CleaningError(f"Column '{column}' has no mode (all values missing)")
            fill_value = mode_values.iloc[0]
        else:
            raise CleaningError(f"Unknown imputation strategy '{strategy}' "
                                 f"(use mean, median, or mode)")

        self.df[column] = series.fillna(fill_value)
        self.log.append({
            "type": "impute_missing", "column": column, "strategy": strategy,
            "rows_affected": missing_before,
            "fill_value": float(fill_value) if isinstance(fill_value, (int, float, np.number)) else str(fill_value),
        })
        return self

    def drop_rows_with_missing(self, columns: List[str] | None = None) -> "DataCleaner":
        subset = columns if columns else None
        rows_before = len(self.df)
        self.df = self.df.dropna(subset=subset)
        self.log.append({
            "type": "drop_rows_with_missing", "columns": columns,
            "rows_affected": rows_before - len(self.df),
        })
        return self

    def drop_columns(self, columns: List[str]) -> "DataCleaner":
        missing_cols = [c for c in columns if c not in self.df.columns]
        if missing_cols:
            raise CleaningError(f"Column(s) not found: {missing_cols}")
        self.df = self.df.drop(columns=columns)
        self.log.append({"type": "drop_columns", "columns": columns})
        return self

    def remove_duplicates(self) -> "DataCleaner":
        rows_before = len(self.df)
        self.df = self.df.drop_duplicates()
        self.log.append({
            "type": "remove_duplicates",
            "rows_affected": rows_before - len(self.df),
        })
        return self

    def handle_outliers(self, column: str, method: str) -> "DataCleaner":
        """method: 'remove' drops flagged rows, 'cap' winsorizes to the IQR bounds."""
        if column not in self.df.columns:
            raise CleaningError(f"Column '{column}' not found")
        if not pd.api.types.is_numeric_dtype(self.df[column]):
            raise CleaningError(f"Outlier handling requires a numeric column; '{column}' is not numeric")

        series = self.df[column]
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - config.IQR_MULTIPLIER * iqr
        upper = q3 + config.IQR_MULTIPLIER * iqr
        flagged = (series < lower) | (series > upper)
        flagged_count = int(flagged.sum())

        if method == "remove":
            rows_before = len(self.df)
            self.df = self.df[~flagged]
            self.log.append({
                "type": "handle_outliers", "column": column, "method": "remove",
                "rows_affected": rows_before - len(self.df),
                "bounds": [round(float(lower), 4), round(float(upper), 4)],
            })
        elif method == "cap":
            self.df[column] = series.clip(lower=lower, upper=upper)
            self.log.append({
                "type": "handle_outliers", "column": column, "method": "cap",
                "rows_affected": flagged_count,
                "bounds": [round(float(lower), 4), round(float(upper), 4)],
            })
        elif method == "keep":
            self.log.append({
                "type": "handle_outliers", "column": column, "method": "keep",
                "rows_affected": 0, "note": "outliers left as-is",
            })
        else:
            raise CleaningError(f"Unknown outlier method '{method}' (use remove, cap, or keep)")
        return self

    # -- batch entry point --------------------------------------------------

    def apply_operations(self, operations: List[Dict[str, Any]]) -> "DataCleaner":
        """
        operations: list of dicts like
          {"type": "impute_missing", "column": "age", "strategy": "mean"}
          {"type": "drop_rows_with_missing", "columns": ["salary"]}
          {"type": "drop_columns", "columns": ["notes"]}
          {"type": "remove_duplicates"}
          {"type": "handle_outliers", "column": "salary", "method": "cap"}
        Applied in the order given - order matters (e.g. impute before outlier capping).
        """
        dispatch = {
            "impute_missing": lambda op: self.impute_missing(op["column"], op["strategy"]),
            "drop_rows_with_missing": lambda op: self.drop_rows_with_missing(op.get("columns")),
            "drop_columns": lambda op: self.drop_columns(op["columns"]),
            "remove_duplicates": lambda op: self.remove_duplicates(),
            "handle_outliers": lambda op: self.handle_outliers(op["column"], op["method"]),
        }
        for op in operations:
            op_type = op.get("type")
            if op_type not in dispatch:
                raise CleaningError(f"Unknown operation type '{op_type}'")
            dispatch[op_type](op)
        return self

    def result(self) -> pd.DataFrame:
        return self.df

    def summary(self) -> Dict[str, Any]:
        return {
            "operations": self.log,
            "resulting_shape": {"n_rows": len(self.df), "n_columns": len(self.df.columns)},
        }


if __name__ == "__main__":
    sample = pd.DataFrame({
        "age": [25, 30, np.nan, 40, 200],
        "salary": [50000, 60000, 55000, 58000, 59000],
    })
    cleaner = DataCleaner(sample)
    cleaner.apply_operations([
        {"type": "impute_missing", "column": "age", "strategy": "mean"},
        {"type": "handle_outliers", "column": "age", "method": "cap"},
    ])
    print(cleaner.result())
    print(cleaner.summary())
