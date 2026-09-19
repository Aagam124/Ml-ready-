"""
baseline.py
-----------
Phase 5 - baseline model training, used to measure the before/after impact
of cleaning (section 10-11 of the spec).

Deliberately simple: one preprocessing pipeline (median/most-frequent
impute + scale/one-hot), one model per problem type (Random Forest, which
tolerates mixed numeric/categorical data and unscaled features better than
a linear baseline), and a fixed reproducible random_state so "before" and
"after" runs are only measuring the effect of cleaning, not randomness.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
    mean_squared_error, precision_score, r2_score, recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config
from json_utils import sanitize_for_json

MIN_TRAINABLE_ROWS = 20
# A numeric target with more distinct values than this is treated as a
# regression target rather than a many-class classification target.
MAX_NUMERIC_CLASSES = 15


class TrainingError(ValueError):
    """Raised when the dataset/target combination can't be trained on."""


def _infer_problem_type(y: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(y) and y.nunique() > MAX_NUMERIC_CLASSES:
        return "regression"
    return "classification"


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    transformers = []
    if numeric_cols:
        transformers.append((
            "num",
            Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
            numeric_cols,
        ))
    if categorical_cols:
        transformers.append((
            "cat",
            Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical_cols,
        ))
    if not transformers:
        raise TrainingError("No usable feature columns found")
    return ColumnTransformer(transformers)


def train_baseline(df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
    """Trains one baseline model and returns its problem type, model name,
    and metrics. Raises TrainingError for anything that makes training
    impossible (missing target, too few rows, no features left, etc.)."""
    if target_col not in df.columns:
        raise TrainingError(f"Target column '{target_col}' not found")

    working = df.dropna(subset=[target_col])
    if len(working) < MIN_TRAINABLE_ROWS:
        raise TrainingError(
            f"Only {len(working)} rows have a non-missing target - "
            f"need at least {MIN_TRAINABLE_ROWS} to train"
        )

    y = working[target_col]
    X = working.drop(columns=[target_col])
    if X.shape[1] == 0:
        raise TrainingError("No feature columns remain after removing the target")

    problem_type = _infer_problem_type(y)

    # Stratify keeps class proportions stable across train/test - but only
    # if every class has enough members to survive the split.
    stratify = y if (problem_type == "classification" and y.value_counts().min() >= 2) else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=config.RANDOM_STATE, stratify=stratify,
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=config.RANDOM_STATE,
        )

    if len(X_test) == 0:
        raise TrainingError("Test split ended up empty - dataset is too small to evaluate")

    preprocessor = _build_preprocessor(X)
    estimator = (
        RandomForestClassifier(n_estimators=200, random_state=config.RANDOM_STATE)
        if problem_type == "classification"
        else RandomForestRegressor(n_estimators=200, random_state=config.RANDOM_STATE)
    )
    pipeline = Pipeline([("preprocess", preprocessor), ("model", estimator)])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    if problem_type == "classification":
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        }
        confusion = {
            "labels": [item.item() if hasattr(item, "item") else item
                       for item in sorted(pd.unique(y_test), key=str)],
            "matrix": confusion_matrix(y_test, y_pred).tolist(),
        }
    else:
        mse = mean_squared_error(y_test, y_pred)
        metrics = {
            "mae": mean_absolute_error(y_test, y_pred),
            "mse": mse,
            "rmse": np.sqrt(mse),
            "r2": r2_score(y_test, y_pred),
        }
        confusion = None

    metrics = {k: round(float(v), 4) for k, v in metrics.items()}

    result = {
        "problem_type": problem_type,
        "model_type": estimator.__class__.__name__,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "metrics": metrics,
        "confusion_matrix": confusion,
    }
    return sanitize_for_json(result)


if __name__ == "__main__":
    rng = np.random.default_rng(config.RANDOM_STATE)
    n = 200
    sample = pd.DataFrame({
        "age": rng.normal(35, 10, n),
        "income": rng.normal(50000, 15000, n),
        "city": rng.choice(["Delhi", "Mumbai", "Pune"], n),
        "churned": rng.choice([0, 1], n, p=[0.7, 0.3]),
    })
    print(train_baseline(sample, "churned"))
