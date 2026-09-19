"""
MLReady - ML-Specific Risk Analysis

This is the differentiating layer: it doesn't just describe the data
(that's profiler.py/quality.py), it estimates whether the data will
actively hurt a trained model. Every heuristic here is intentionally
conservative in its language - nothing is described as "proven", only
"detected" / "potential", because these are statistical heuristics, not
guarantees. This matters especially for leakage detection (see
potential_leakage below): false positives are cheap, false confidence
in a leaky model is expensive.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

import re

from config import (
    CLASS_IMBALANCE_RATIO,
    MULTICOLLINEARITY_CORR_THRESHOLD,
    VIF_THRESHOLD,
    LEAKAGE_CORR_THRESHOLD,
    LEAKAGE_NAME_SUBSTRINGS,
    LEAKAGE_NAME_TOKENS,
    ID_LIKE_UNIQUE_RATIO,
)


def _name_looks_leaky(column_name: str) -> bool:
    """True if the column name itself hints at target/outcome/future info.

    Two tiers on purpose: short generic words (like "y" or "class") are only
    flagged as whole tokens so we don't match "tenure_years" or "classroom";
    longer, more specific words (like "target" or "leak") are matched as
    substrings since they rarely appear by accident.
    """
    lower_name = column_name.lower()
    if any(keyword in lower_name for keyword in LEAKAGE_NAME_SUBSTRINGS):
        return True
    tokens = re.split(r"[_\-\s]+", lower_name)
    return any(token in LEAKAGE_NAME_TOKENS for token in tokens)


def detect_class_imbalance(df: pd.DataFrame, target_col: str | None) -> dict:
    if not target_col or target_col not in df.columns:
        return {"applicable": False, "reason": "No target column selected."}

    series = df[target_col].dropna()
    n_unique = series.nunique()

    # Only meaningful for classification-shaped targets: few distinct values
    # relative to row count. A continuous target isn't "imbalanced".
    if n_unique < 2 or n_unique > min(20, max(2, int(len(series) * 0.05))):
        return {"applicable": False, "reason": "Target does not look categorical."}

    counts = series.value_counts()
    majority, minority = counts.iloc[0], counts.iloc[-1]
    ratio = round(float(majority / minority), 2) if minority else float("inf")

    return {
        "applicable": True,
        "class_distribution": {str(k): int(v) for k, v in counts.items()},
        "majority_minority_ratio": ratio,
        "is_imbalanced": ratio >= CLASS_IMBALANCE_RATIO,
        "threshold_used": CLASS_IMBALANCE_RATIO,
    }


def detect_multicollinearity(df: pd.DataFrame, numerical_columns: list[str]) -> dict:
    """Pairwise correlation screen + VIF for numeric features.

    VIF is computed manually (regress each feature on the rest, VIF = 1/(1-R^2))
    rather than pulling in statsmodels, to keep the dependency list minimal.
    """
    usable = [c for c in numerical_columns if df[c].notna().sum() > 1]
    if len(usable) < 2:
        return {"high_correlation_pairs": [], "vif": {}, "note": "Fewer than 2 usable numeric columns."}

    subset = df[usable].dropna()
    high_corr_pairs = []
    if len(subset) >= 2:
        corr_matrix = subset.corr(numeric_only=True)
        seen = set()
        for col_a in corr_matrix.columns:
            for col_b in corr_matrix.columns:
                if col_a == col_b or (col_b, col_a) in seen:
                    continue
                seen.add((col_a, col_b))
                value = corr_matrix.loc[col_a, col_b]
                if pd.notna(value) and abs(value) >= MULTICOLLINEARITY_CORR_THRESHOLD:
                    high_corr_pairs.append({
                        "feature_a": col_a, "feature_b": col_b,
                        "correlation": round(float(value), 3),
                    })

    vif_scores = {}
    if len(subset) > len(usable) and len(usable) >= 2:
        # Standardize to keep the regression well-conditioned.
        standardized = (subset - subset.mean()) / subset.std(ddof=0).replace(0, np.nan)
        standardized = standardized.dropna(axis=1, how="all").fillna(0)
        cols = standardized.columns.tolist()
        for col in cols:
            others = [c for c in cols if c != col]
            if not others:
                continue
            X = standardized[others].values
            y = standardized[col].values
            if X.shape[0] <= X.shape[1]:
                continue
            reg = LinearRegression().fit(X, y)
            r2 = reg.score(X, y)
            vif = float("inf") if r2 >= 0.9999 else round(1 / (1 - r2), 2)
            vif_scores[col] = vif

    high_vif = {c: v for c, v in vif_scores.items() if isinstance(v, (int, float)) and v >= VIF_THRESHOLD}

    return {
        "high_correlation_pairs": high_corr_pairs,
        "vif": vif_scores,
        "high_vif_features": high_vif,
        "correlation_threshold_used": MULTICOLLINEARITY_CORR_THRESHOLD,
        "vif_threshold_used": VIF_THRESHOLD,
    }


def detect_potential_leakage(df: pd.DataFrame, target_col: str | None) -> dict:
    """Heuristic leakage screen. Flags features that are SUSPICIOUS, never
    claims proof - see module docstring. Three independent signals:
      1. Column name contains a target-like keyword
      2. Near-perfect correlation with the target (numeric target only)
      3. Column looks ID-like (near-100% unique values) - not leakage per se,
         but a classic "accidentally predictive of row identity" risk.
    """
    flagged = []
    n_rows = len(df)

    for col in df.columns:
        if col == target_col:
            continue
        reasons = []

        if _name_looks_leaky(col):
            reasons.append("Column name resembles a target/outcome/future-information term.")

        if target_col and target_col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]) and pd.api.types.is_numeric_dtype(df[target_col]):
                paired = df[[col, target_col]].dropna()
                if len(paired) > 2 and paired[col].std() > 0 and paired[target_col].std() > 0:
                    corr = paired[col].corr(paired[target_col])
                    if pd.notna(corr) and abs(corr) >= LEAKAGE_CORR_THRESHOLD:
                        reasons.append(
                            f"Near-perfect correlation with target ({round(float(corr), 3)})."
                        )

        if n_rows and df[col].nunique(dropna=True) / n_rows >= ID_LIKE_UNIQUE_RATIO:
            reasons.append("Column is ID-like (nearly every value is unique).")

        if reasons:
            flagged.append({"column": col, "reasons": reasons})

    return {
        "potential_leakage_columns": flagged,
        "disclaimer": (
            "These are heuristic flags based on naming patterns, correlation, and "
            "uniqueness. Leakage is NOT confirmed automatically - review each "
            "flagged feature manually before trusting model results."
        ),
    }


def detect_high_cardinality_features(df: pd.DataFrame, categorical_columns: list[str]) -> dict:
    """ML-context version: any categorical feature likely to blow up
    one-hot encoding or overfit a tree model."""
    n_rows = len(df)
    flagged = {}
    if n_rows == 0:
        return flagged
    for col in categorical_columns:
        n_unique = df[col].nunique(dropna=True)
        if n_unique > 50 or (n_unique / n_rows) > 0.5:
            flagged[col] = {
                "unique_count": int(n_unique),
                "unique_ratio": round(float(n_unique / n_rows), 3),
                "risk": "May cause sparse one-hot encodings or tree-model overfitting.",
            }
    return flagged


def train_test_distribution_diff(df: pd.DataFrame, numerical_columns: list[str],
                                  test_size: float = 0.2, random_state: int = 42) -> dict:
    """Split the data the same way model training will, then run a
    Kolmogorov-Smirnov test per numeric column to flag columns whose
    train/test distributions differ enough to worry about (e.g. a dataset
    sorted by time or by a categorical group before splitting)."""
    from sklearn.model_selection import train_test_split

    if len(df) < 20 or not numerical_columns:
        return {"applicable": False, "reason": "Needs >=20 rows and >=1 numeric column."}

    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    results = {}
    for col in numerical_columns:
        train_vals = train_df[col].dropna()
        test_vals = test_df[col].dropna()
        if len(train_vals) < 5 or len(test_vals) < 5:
            continue
        ks_stat, p_value = stats.ks_2samp(train_vals, test_vals)
        if p_value < 0.05:
            results[col] = {"ks_statistic": round(float(ks_stat), 4), "p_value": round(float(p_value), 4)}

    return {
        "applicable": True,
        "split_used": f"{int((1 - test_size) * 100)}/{int(test_size * 100)}",
        "columns_with_significant_shift": results,
        "note": "p < 0.05 on a KS test flags a meaningful train/test distribution shift.",
    }


def run_ml_risk_analysis(df: pd.DataFrame, numerical_columns: list[str],
                          categorical_columns: list[str], target_col: str | None) -> dict:
    return {
        "class_imbalance": detect_class_imbalance(df, target_col),
        "multicollinearity": detect_multicollinearity(df, numerical_columns),
        "potential_leakage": detect_potential_leakage(df, target_col),
        "high_cardinality_features": detect_high_cardinality_features(df, categorical_columns),
        "train_test_distribution_diff": train_test_distribution_diff(df, numerical_columns),
    }
