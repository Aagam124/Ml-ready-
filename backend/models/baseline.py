"""
baseline.py
-----------
Phase 5 - baseline model training, used to measure the before/after impact
of cleaning (section 10-11 of the spec).

Deliberately reproducible: fixed random_state so "before" and "after" runs
only measure the effect of cleaning, not randomness.

Upgrades vs the original simple baseline:
  - Richer classification metrics (balanced accuracy, ROC-AUC when binary)
  - Majority-class / mean baseline so users can see if the model beats chance
  - Optional cross-validation summary for more stable estimates
  - Train/test class distribution (classification)
  - Top feature importances (when the model exposes them)
  - Safer handling of high-cardinality categoricals and constant columns
  - Clear warnings when the split is fragile (tiny test set, rare classes)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config
from json_utils import sanitize_for_json

MIN_TRAINABLE_ROWS = 20
# A numeric target with more distinct values than this is treated as a
# regression target rather than a many-class classification target.
MAX_NUMERIC_CLASSES = 15
# Cap one-hot expansion so high-cardinality IDs don't explode memory.
MAX_ONEHOT_CATEGORIES = 50
# Cross-validation folds (skipped when the dataset is too small).
CV_FOLDS = 3


class TrainingError(ValueError):
    """Raised when the dataset/target combination can't be trained on."""


def _infer_problem_type(y: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(y) and y.nunique() > MAX_NUMERIC_CLASSES:
        return "regression"
    return "classification"


def _drop_unusable_features(X: pd.DataFrame) -> tuple[pd.DataFrame, List[str]]:
    """Drop constant columns and pure-ID-like columns that only hurt the model."""
    dropped: List[str] = []
    keep = []
    n = len(X)
    for col in X.columns:
        series = X[col]
        nunique = series.nunique(dropna=True)
        if nunique <= 1:
            dropped.append(col)
            continue
        # Near-100% unique object/string columns are usually IDs
        if (
            not pd.api.types.is_numeric_dtype(series)
            and n > 0
            and nunique / n >= 0.98
            and nunique >= 20
        ):
            dropped.append(col)
            continue
        keep.append(col)
    return X[keep].copy(), dropped


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    transformers = []
    if numeric_cols:
        transformers.append((
            "num",
            Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]),
            numeric_cols,
        ))
    if categorical_cols:
        transformers.append((
            "cat",
            Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                (
                    "encode",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        max_categories=MAX_ONEHOT_CATEGORIES,
                    ),
                ),
            ]),
            categorical_cols,
        ))
    if not transformers:
        raise TrainingError("No usable feature columns found")
    return ColumnTransformer(transformers)


def _classification_metrics(
    y_test: pd.Series,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
) -> Dict[str, float]:
    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "precision": float(
            precision_score(y_test, y_pred, average="weighted", zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, y_pred, average="weighted", zero_division=0)
        ),
        "f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
    }

    # Binary ROC-AUC when probabilities are available
    classes = pd.unique(y_test)
    if y_proba is not None and len(classes) == 2 and y_proba.ndim == 2 and y_proba.shape[1] >= 2:
        try:
            pos_idx = 1 if y_proba.shape[1] > 1 else 0
            metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba[:, pos_idx]))
        except Exception:
            pass

    return metrics


def _regression_metrics(y_test: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    mse = float(mean_squared_error(y_test, y_pred))
    return {
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_test, y_pred)),
    }


def _majority_baseline_accuracy(y_train: pd.Series, y_test: pd.Series) -> float:
    """Accuracy of always predicting the training majority class."""
    majority = y_train.value_counts().index[0]
    return float((y_test == majority).mean())


def _mean_baseline_mae(y_train: pd.Series, y_test: pd.Series) -> float:
    """MAE of always predicting the training mean."""
    pred = float(y_train.mean())
    return float(np.mean(np.abs(y_test.astype(float) - pred)))


def _class_distribution(series: pd.Series) -> Dict[str, int]:
    return {str(k): int(v) for k, v in series.value_counts().items()}


def _top_feature_importances(
    pipeline: Pipeline,
    X: pd.DataFrame,
    top_n: int = 10,
) -> Optional[List[Dict[str, Any]]]:
    """Extract feature importances from the fitted RF when possible."""
    try:
        model = pipeline.named_steps["model"]
        if not hasattr(model, "feature_importances_"):
            return None
        pre = pipeline.named_steps["preprocess"]
        try:
            names = list(pre.get_feature_names_out())
        except Exception:
            names = [f"f{i}" for i in range(len(model.feature_importances_))]
        importances = model.feature_importances_
        if len(names) != len(importances):
            names = [f"f{i}" for i in range(len(importances))]
        pairs = sorted(
            zip(names, importances),
            key=lambda x: x[1],
            reverse=True,
        )[:top_n]
        return [
            {"feature": name, "importance": round(float(imp), 4)}
            for name, imp in pairs
        ]
    except Exception:
        return None


def train_baseline(df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
    """Trains one baseline model and returns its problem type, model name,
    and metrics. Raises TrainingError for anything that makes training
    impossible (missing target, too few rows, no features left, etc.).

    Return shape (backward compatible with routes/models.py):
      problem_type, model_type, train_rows, test_rows, metrics, confusion_matrix

    Extra keys (safe for clients that ignore unknowns):
      warnings, dropped_features, class_distribution, naive_baseline,
      cv_scores, feature_importances
    """
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

    X, dropped_features = _drop_unusable_features(X)
    if X.shape[1] == 0:
        raise TrainingError(
            "No usable feature columns remain after removing constant/ID-like columns"
        )

    problem_type = _infer_problem_type(y)
    warnings: List[str] = []

    if dropped_features:
        warnings.append(
            f"Dropped {len(dropped_features)} constant or ID-like column(s) before training: "
            + ", ".join(dropped_features[:8])
            + ("…" if len(dropped_features) > 8 else "")
        )

    # Stratify keeps class proportions stable across train/test - but only
    # if every class has enough members to survive the split.
    stratify = None
    if problem_type == "classification":
        counts = y.value_counts()
        if counts.min() < 2:
            warnings.append(
                "At least one class has fewer than 2 samples; stratification disabled."
            )
        else:
            stratify = y

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

    if len(X_test) < 10:
        warnings.append(
            f"Test set is very small (n={len(X_test)}). Metrics may be unstable."
        )

    preprocessor = _build_preprocessor(X)
    if problem_type == "classification":
        estimator = RandomForestClassifier(
            n_estimators=200,
            random_state=config.RANDOM_STATE,
            class_weight="balanced_subsample",
            n_jobs=-1,
        )
    else:
        estimator = RandomForestRegressor(
            n_estimators=200,
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        )

    pipeline = Pipeline([("preprocess", preprocessor), ("model", estimator)])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    y_proba = None
    if problem_type == "classification" and hasattr(pipeline, "predict_proba"):
        try:
            y_proba = pipeline.predict_proba(X_test)
        except Exception:
            y_proba = None

    if problem_type == "classification":
        metrics = _classification_metrics(y_test, y_pred, y_proba)
        labels_sorted = sorted(pd.unique(pd.concat([y_test, y_train])), key=str)
        confusion = {
            "labels": [
                item.item() if hasattr(item, "item") else item
                for item in labels_sorted
            ],
            "matrix": confusion_matrix(y_test, y_pred, labels=labels_sorted).tolist(),
        }
        naive = {
            "name": "majority_class",
            "accuracy": round(_majority_baseline_accuracy(y_train, y_test), 4),
        }
        class_dist = {
            "train": _class_distribution(y_train),
            "test": _class_distribution(y_test),
        }
        if metrics["accuracy"] <= naive["accuracy"] + 0.01:
            warnings.append(
                "Model accuracy is not meaningfully better than always predicting "
                "the majority class. Check leakage, features, or class imbalance."
            )
    else:
        metrics = _regression_metrics(y_test, y_pred)
        confusion = None
        naive = {
            "name": "train_mean",
            "mae": round(_mean_baseline_mae(y_train, y_test), 4),
        }
        class_dist = None
        if metrics["r2"] < 0:
            warnings.append(
                "R² is negative: the model is worse than predicting the mean on the test set."
            )

    metrics = {k: round(float(v), 4) for k, v in metrics.items()}

    # Lightweight cross-validation on the full (post-drop) data for stability signal
    cv_scores: Optional[Dict[str, Any]] = None
    if len(X) >= MIN_TRAINABLE_ROWS * 2:
        try:
            scoring = "f1_weighted" if problem_type == "classification" else "r2"
            scores = cross_val_score(
                pipeline, X, y, cv=min(CV_FOLDS, len(X) // 10 or 2), scoring=scoring,
                n_jobs=-1,
            )
            cv_scores = {
                "scoring": scoring,
                "mean": round(float(scores.mean()), 4),
                "std": round(float(scores.std()), 4),
                "folds": [round(float(s), 4) for s in scores],
            }
        except Exception:
            cv_scores = None

    feature_importances = _top_feature_importances(pipeline, X)

    result = {
        # --- required by routes/models.py ---
        "problem_type": problem_type,
        "model_type": estimator.__class__.__name__,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "metrics": metrics,
        "confusion_matrix": confusion,
        # --- extra (ignored by old clients) ---
        "warnings": warnings,
        "dropped_features": dropped_features,
        "class_distribution": class_dist,
        "naive_baseline": naive,
        "cv_scores": cv_scores,
        "feature_importances": feature_importances,
    }
    return sanitize_for_json(result)


if __name__ == "__main__":
    rng = np.random.default_rng(config.RANDOM_STATE)
    n = 200
    sample = pd.DataFrame({
        "age": rng.normal(35, 10, n),
        "income": rng.normal(50000, 15000, n),
        "city": rng.choice(["Delhi", "Mumbai", "Pune"], n),
        "customer_id": [f"c{i}" for i in range(n)],
        "churned": rng.choice([0, 1], n, p=[0.7, 0.3]),
    })
    print(train_baseline(sample, "churned"))