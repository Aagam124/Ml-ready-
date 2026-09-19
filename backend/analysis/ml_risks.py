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

This module now also produces a per-feature risk profile
(feature_risk_analysis) so a data scientist can see exactly which
columns need REVIEW / TRANSFORM / IMPUTE / REMOVE and why.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

from config import (
    CLASS_IMBALANCE_RATIO,
    MULTICOLLINEARITY_CORR_THRESHOLD,
    VIF_THRESHOLD,
    LEAKAGE_CORR_THRESHOLD,
    LEAKAGE_NAME_SUBSTRINGS,
    LEAKAGE_NAME_TOKENS,
    ID_LIKE_UNIQUE_RATIO,
)


# ============================================================
# Utility helpers
# ============================================================

def _name_looks_leaky(column_name: str) -> bool:
    """True if the column name itself hints at target/outcome/future info.

    Two tiers on purpose: short generic words (like "y" or "class") are only
    flagged as whole tokens so we don't match "tenure_years" or "classroom";
    longer, more specific words (like "target" or "leak") are matched as
    substrings since they rarely appear by accident.
    """
    lower_name = str(column_name).lower()
    if any(keyword in lower_name for keyword in LEAKAGE_NAME_SUBSTRINGS):
        return True
    tokens = re.split(r"[_\-\s]+", lower_name)
    return any(token in LEAKAGE_NAME_TOKENS for token in tokens)


def _is_identifier_like(series: pd.Series, column_name: str) -> bool:
    """
    Conservative identifier-like detection.

    A feature is considered ID-like when:
    - its uniqueness ratio is extremely high, OR
    - its name strongly resembles an identifier.
    """
    if series.empty:
        return False

    n = len(series)
    unique_ratio = series.nunique(dropna=True) / max(n, 1)
    name = str(column_name).lower()

    identifier_terms = [
        "id", "uuid", "guid", "identifier",
        "customer_number", "account_number", "user_number",
        "transaction_id", "record_id", "reference_id",
        "pk", "primary_key", "row_id",
    ]

    name_signal = any(term in name for term in identifier_terms)
    uniqueness_signal = (
        unique_ratio >= ID_LIKE_UNIQUE_RATIO
        and series.nunique(dropna=True) >= 20
    )
    return bool(name_signal or uniqueness_signal)


def _is_datetime_like(series: pd.Series, column_name: str) -> bool:
    """Heuristic for datetime / temporal columns that can leak future info."""
    name = str(column_name).lower()
    temporal_terms = [
        "date", "time", "timestamp", "datetime",
        "created", "updated", "modified", "deleted",
        "start", "end", "expiry", "expiration",
        "year", "month", "day", "hour",
    ]
    if any(term in name for term in temporal_terms):
        return True

    if pd.api.types.is_datetime64_any_dtype(series):
        return True

    sample = series.dropna().astype(str).head(30)
    if len(sample) < 5:
        return False
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(sample, errors="coerce")
        return float(parsed.notna().mean()) > 0.8
    except Exception:
        return False


def _detect_near_zero_variance(series: pd.Series, threshold: float = 0.01) -> bool:
    """Near-zero variance: the most frequent value occupies almost the column."""
    clean = series.dropna()
    if len(clean) < 5:
        return False
    top_freq = clean.value_counts(normalize=True).iloc[0]
    return bool(top_freq >= (1.0 - threshold))


def _outlier_pct_iqr(series: pd.Series) -> float:
    """Return percentage of IQR outliers for a numeric series (0 if not usable)."""
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if len(clean) < 8:
        return 0.0
    q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return 0.0
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mask = (clean < lower) | (clean > upper)
    return float(mask.mean() * 100)


# ============================================================
# Class imbalance  (shape kept for scoring / frontend)
# ============================================================

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
        "majority_class": str(counts.index[0]),
        "minority_class": str(counts.index[-1]),
        "majority_percentage": round(float(counts.iloc[0] / len(series) * 100), 2),
        "minority_percentage": round(float(counts.iloc[-1] / len(series) * 100), 2),
    }


# ============================================================
# Multicollinearity  (shape kept for scoring / frontend)
# ============================================================

def detect_multicollinearity(df: pd.DataFrame, numerical_columns: list[str]) -> dict:
    """Pairwise correlation screen + VIF for numeric features.

    VIF is computed manually (regress each feature on the rest, VIF = 1/(1-R^2))
    rather than pulling in statsmodels, to keep the dependency list minimal.
    """
    usable = [c for c in numerical_columns if c in df.columns and df[c].notna().sum() > 1]
    if len(usable) < 2:
        return {
            "high_correlation_pairs": [],
            "vif": {},
            "high_vif_features": {},
            "note": "Fewer than 2 usable numeric columns.",
            "correlation_threshold_used": MULTICOLLINEARITY_CORR_THRESHOLD,
            "vif_threshold_used": VIF_THRESHOLD,
        }

    subset = df[usable].apply(pd.to_numeric, errors="coerce").dropna()
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
                        "feature_a": col_a,
                        "feature_b": col_b,
                        "correlation": round(float(value), 3),
                    })

    vif_scores: dict[str, float | None] = {}
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
            try:
                reg = LinearRegression().fit(X, y)
                r2 = reg.score(X, y)
                vif = float("inf") if r2 >= 0.9999 else round(1 / (1 - r2), 2)
                vif_scores[col] = vif
            except Exception:
                vif_scores[col] = None

    high_vif = {
        c: v for c, v in vif_scores.items()
        if isinstance(v, (int, float)) and np.isfinite(v) and v >= VIF_THRESHOLD
    }

    return {
        "high_correlation_pairs": high_corr_pairs,
        "vif": vif_scores,
        "high_vif_features": high_vif,
        "correlation_threshold_used": MULTICOLLINEARITY_CORR_THRESHOLD,
        "vif_threshold_used": VIF_THRESHOLD,
    }


# ============================================================
# Potential leakage  (shape kept for scoring / frontend)
# ============================================================

def detect_potential_leakage(df: pd.DataFrame, target_col: str | None) -> dict:
    """Heuristic leakage screen. Flags features that are SUSPICIOUS, never
    claims proof - see module docstring. Signals used:
      1. Column name contains a target-like keyword
      2. Near-perfect correlation with the target (numeric target only)
      3. Column looks ID-like (near-100% unique values)
      4. Column looks temporal (created_at, updated_at, etc.)
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

        if _is_datetime_like(df[col], col):
            reasons.append(
                "Column appears temporal. Confirm it does not contain "
                "future information relative to prediction time."
            )

        if reasons:
            flagged.append({"column": col, "reasons": reasons})

    return {
        "potential_leakage_columns": flagged,
        "disclaimer": (
            "These are heuristic flags based on naming patterns, correlation, uniqueness, "
            "and temporal signals. Leakage is NOT confirmed automatically - review each "
            "flagged feature manually before trusting model results."
        ),
    }


# ============================================================
# High-cardinality features  (shape kept for scoring / frontend)
# ============================================================

def detect_high_cardinality_features(df: pd.DataFrame, categorical_columns: list[str]) -> dict:
    """ML-context version: any categorical feature likely to blow up
    one-hot encoding or overfit a tree model."""
    n_rows = len(df)
    flagged = {}
    if n_rows == 0:
        return flagged
    for col in categorical_columns:
        if col not in df.columns:
            continue
        n_unique = df[col].nunique(dropna=True)
        if n_unique > 50 or (n_unique / n_rows) > 0.5:
            flagged[col] = {
                "unique_count": int(n_unique),
                "unique_ratio": round(float(n_unique / n_rows), 3),
                "risk": (
                    "May cause sparse one-hot encodings or tree-model overfitting. "
                    "Consider grouping rare categories, frequency encoding, or "
                    "target encoding (with proper cross-validation)."
                ),
            }
    return flagged


# ============================================================
# Train / test distribution difference  (shape kept)
# ============================================================

def train_test_distribution_diff(
    df: pd.DataFrame,
    numerical_columns: list[str],
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """Split the data the same way model training will, then run a
    Kolmogorov-Smirnov test per numeric column to flag columns whose
    train/test distributions differ enough to worry about."""
    from sklearn.model_selection import train_test_split

    if len(df) < 20 or not numerical_columns:
        return {"applicable": False, "reason": "Needs >=20 rows and >=1 numeric column."}

    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    results = {}
    for col in numerical_columns:
        if col not in df.columns:
            continue
        train_vals = pd.to_numeric(train_df[col], errors="coerce").dropna()
        test_vals = pd.to_numeric(test_df[col], errors="coerce").dropna()
        if len(train_vals) < 5 or len(test_vals) < 5:
            continue
        try:
            ks_stat, p_value = stats.ks_2samp(train_vals, test_vals)
            if p_value < 0.05:
                results[col] = {
                    "ks_statistic": round(float(ks_stat), 4),
                    "p_value": round(float(p_value), 4),
                }
        except Exception:
            continue

    return {
        "applicable": True,
        "split_used": f"{int((1 - test_size) * 100)}/{int(test_size * 100)}",
        "columns_with_significant_shift": results,
        "note": "p < 0.05 on a KS test flags a meaningful train/test distribution shift.",
    }


# ============================================================
# NEW: Duplicate / near-duplicate columns
# ============================================================

def detect_duplicate_columns(
    df: pd.DataFrame,
    numerical_columns: list[str],
    corr_threshold: float = 0.999,
) -> dict[str, Any]:
    """Find exact duplicate columns and numerical near-duplicates."""
    result: dict[str, Any] = {
        "exact_duplicates": [],
        "near_duplicates": [],
    }

    cols = list(df.columns)
    seen: dict[tuple, str] = {}
    for col in cols:
        # Hash by stringified values so mixed dtypes compare safely
        key = tuple(df[col].astype(str).fillna("__NA__").tolist())
        if key in seen:
            result["exact_duplicates"].append(
                {"feature_1": seen[key], "feature_2": col}
            )
        else:
            seen[key] = col

    valid = [c for c in numerical_columns if c in df.columns]
    if len(valid) >= 2:
        numeric = df[valid].apply(pd.to_numeric, errors="coerce")
        corr = numeric.corr()
        for i, a in enumerate(valid):
            for b in valid[i + 1 :]:
                val = corr.loc[a, b]
                if pd.notna(val) and abs(val) >= corr_threshold:
                    result["near_duplicates"].append(
                        {
                            "feature_1": a,
                            "feature_2": b,
                            "correlation": round(float(val), 6),
                        }
                    )

    return result


# ============================================================
# NEW: Sample-size / dimensionality risk
# ============================================================

def detect_dimensionality_risk(
    df: pd.DataFrame,
    numerical_columns: list[str],
    categorical_columns: list[str],
) -> dict[str, Any]:
    """Warn when #features is large relative to #rows (overfitting risk)."""
    n_rows = len(df)
    n_features = len(set(numerical_columns) | set(categorical_columns))
    ratio = n_features / max(n_rows, 1)

    if n_rows < 50:
        risk = "HIGH"
        message = (
            f"Very small sample size (n={n_rows}). "
            "Models are likely to overfit; prefer simple models or collect more data."
        )
    elif ratio > 0.5:
        risk = "HIGH"
        message = (
            f"Feature-to-sample ratio is high ({n_features} features / {n_rows} rows). "
            "Strong risk of overfitting; consider feature selection or dimensionality reduction."
        )
    elif ratio > 0.2:
        risk = "MEDIUM"
        message = (
            f"Moderately high feature-to-sample ratio ({n_features}/{n_rows}). "
            "Regularization or feature selection is recommended."
        )
    else:
        risk = "LOW"
        message = "Sample size appears adequate relative to feature count."

    return {
        "n_rows": n_rows,
        "n_features": n_features,
        "feature_to_sample_ratio": round(ratio, 4),
        "risk": risk,
        "message": message,
    }


# ============================================================
# NEW: Per-feature risk analysis (the main upgrade)
# ============================================================

def analyze_feature_risks(
    df: pd.DataFrame,
    numerical_columns: list[str],
    categorical_columns: list[str],
    target_col: str | None = None,
    leakage: dict | None = None,
    multicollinearity: dict | None = None,
    high_cardinality: dict | None = None,
    duplicates: dict | None = None,
) -> dict[str, Any]:
    """
    Build an explainable risk profile for every feature.

    Risk scores are internal evidence scores, NOT probabilities.

    Severity bands:
      0-14  -> LOW
     15-29  -> MEDIUM
     30-49  -> HIGH
     50+    -> CRITICAL

    Recommendations: KEEP | TRANSFORM | IMPUTE | REVIEW | REMOVE
    """
    if leakage is None:
        leakage = detect_potential_leakage(df, target_col)
    if multicollinearity is None:
        multicollinearity = detect_multicollinearity(df, numerical_columns)
    if high_cardinality is None:
        high_cardinality = detect_high_cardinality_features(df, categorical_columns)
    if duplicates is None:
        duplicates = detect_duplicate_columns(df, numerical_columns)

    # Map correlated pairs for quick lookup
    correlated_features: dict[str, list[dict]] = {}
    for item in multicollinearity.get("high_correlation_pairs", []):
        a, b = item["feature_a"], item["feature_b"]
        correlated_features.setdefault(a, []).append(item)
        correlated_features.setdefault(b, []).append(item)

    high_vif = set(multicollinearity.get("high_vif_features", {}).keys())
    leakage_cols = {
        entry["column"] for entry in leakage.get("potential_leakage_columns", [])
    }
    high_card_cols = set(high_cardinality.keys())

    exact_dup_set: set[str] = set()
    for pair in duplicates.get("exact_duplicates", []):
        exact_dup_set.add(pair["feature_1"])
        exact_dup_set.add(pair["feature_2"])

    near_dup_set: set[str] = set()
    for pair in duplicates.get("near_duplicates", []):
        near_dup_set.add(pair["feature_1"])
        near_dup_set.add(pair["feature_2"])

    features = [c for c in df.columns if c != target_col]
    analysis: dict[str, Any] = {}
    summary = {
        "low": 0, "medium": 0, "high": 0, "critical": 0,
        "keep": 0, "review": 0, "transform": 0, "impute": 0, "remove": 0,
    }

    for column in features:
        series = df[column]
        score = 0
        evidence: list[str] = []
        suggested_actions: list[str] = []

        # ---- Missing values ----
        missing_pct = float(series.isna().mean() * 100) if len(series) > 0 else 0.0
        if missing_pct >= 40:
            score += 30
            evidence.append(f"{missing_pct:.1f}% missing values.")
            suggested_actions.append(
                "Evaluate whether the feature should be removed or carefully imputed."
            )
        elif missing_pct >= 20:
            score += 20
            evidence.append(f"{missing_pct:.1f}% missing values.")
            suggested_actions.append("Consider an appropriate imputation strategy.")
        elif missing_pct > 0:
            score += 8
            evidence.append(f"{missing_pct:.1f}% missing values.")
            suggested_actions.append("Handle missing values before model training.")

        # ---- Constant / near-constant ----
        unique_count = int(series.nunique(dropna=True))
        if unique_count <= 1:
            score += 35
            evidence.append("Feature is constant or contains only one usable value.")
            suggested_actions.append(
                "Remove the feature; it provides little predictive information."
            )
        elif _detect_near_zero_variance(series):
            score += 18
            evidence.append(
                "Near-zero variance: almost all non-null values are identical."
            )
            suggested_actions.append(
                "Review whether the feature adds useful signal; consider dropping."
            )

        # ---- Identifier-like ----
        if _is_identifier_like(series, column):
            score += 25
            evidence.append(
                "Feature appears identifier-like based on its name and/or uniqueness."
            )
            suggested_actions.append(
                "Confirm this feature is available and meaningful at prediction time."
            )

        # ---- Potential leakage ----
        if column in leakage_cols:
            score += 40
            for entry in leakage.get("potential_leakage_columns", []):
                if entry["column"] == column:
                    for r in entry["reasons"]:
                        evidence.append(r)
            suggested_actions.append(
                "Review carefully for target leakage before training."
            )

        # ---- High cardinality ----
        if column in high_card_cols:
            score += 18
            evidence.append("Categorical feature has high cardinality.")
            suggested_actions.append(
                "Review encoding strategy; consider grouping rare categories "
                "or using frequency / target / embedding encodings."
            )

        # ---- VIF ----
        if column in high_vif:
            score += 15
            vif_value = multicollinearity.get("vif", {}).get(column)
            evidence.append(f"High multicollinearity detected (VIF={vif_value}).")
            suggested_actions.append("Review redundant numerical features.")

        # ---- Strong pairwise correlation ----
        if column in correlated_features:
            score += 8
            for item in correlated_features[column]:
                other = (
                    item["feature_b"]
                    if item["feature_a"] == column
                    else item["feature_a"]
                )
                evidence.append(
                    f"Strong correlation with '{other}' (r={item['correlation']})."
                )
            suggested_actions.append(
                "Review whether correlated features provide redundant information."
            )

        # ---- Exact / near duplicates ----
        if column in exact_dup_set:
            score += 30
            evidence.append("Exact duplicate of another column detected.")
            suggested_actions.append("Drop one of the duplicate columns.")
        elif column in near_dup_set:
            score += 20
            evidence.append(
                "Near-duplicate of another numerical column (correlation ≈ 1)."
            )
            suggested_actions.append(
                "Consider dropping one of the near-duplicate columns."
            )

        # ---- Outliers (numeric) ----
        if column in numerical_columns:
            out_pct = _outlier_pct_iqr(series)
            if out_pct >= 5:
                score += 12
                evidence.append(f"{out_pct:.1f}% potential outliers (IQR method).")
                suggested_actions.append(
                    "Inspect outliers; consider winsorizing, robust scaling, "
                    "or tree-based models that are less sensitive."
                )
            elif out_pct >= 1:
                score += 5
                evidence.append(f"{out_pct:.1f}% potential outliers (IQR method).")

        # ---- Skewness ----
        if column in numerical_columns and pd.api.types.is_numeric_dtype(series):
            numeric_values = pd.to_numeric(series, errors="coerce").dropna()
            if len(numeric_values) >= 8:
                try:
                    skewness = float(numeric_values.skew())
                    if abs(skewness) >= 2:
                        score += 10
                        evidence.append(
                            f"Strongly skewed numerical distribution "
                            f"(skewness={skewness:.2f})."
                        )
                        suggested_actions.append(
                            "Consider log / Box-Cox / Yeo-Johnson transform "
                            "if required by the chosen model."
                        )
                    elif abs(skewness) >= 1:
                        score += 5
                        evidence.append(
                            f"Moderately skewed numerical distribution "
                            f"(skewness={skewness:.2f})."
                        )
                except Exception:
                    pass

        # ---- Datetime-like (extra signal even if already in leakage) ----
        if _is_datetime_like(series, column) and column not in leakage_cols:
            score += 10
            evidence.append("Column appears to be datetime / temporal.")
            suggested_actions.append(
                "Extract useful components (year, month, day-of-week, etc.) "
                "and ensure no future information is present."
            )

        # ---- Severity ----
        score = min(int(score), 100)
        if score >= 50:
            severity = "CRITICAL"
        elif score >= 30:
            severity = "HIGH"
        elif score >= 15:
            severity = "MEDIUM"
        else:
            severity = "LOW"

        # ---- Recommendation ----
        if column in leakage_cols or _is_identifier_like(series, column):
            recommendation = "REVIEW"
        elif unique_count <= 1 or column in exact_dup_set:
            recommendation = "REMOVE"
        elif missing_pct >= 20:
            recommendation = "IMPUTE"
        elif (
            column in high_card_cols
            or any("skewed" in e.lower() for e in evidence)
            or any("outlier" in e.lower() for e in evidence)
        ):
            recommendation = "TRANSFORM"
        elif column in high_vif or column in correlated_features or column in near_dup_set:
            recommendation = "REVIEW"
        else:
            recommendation = "KEEP"

        if not evidence:
            evidence.append("No major heuristic risk signal detected.")
        if not suggested_actions:
            suggested_actions.append("No immediate preprocessing action suggested.")

        summary[severity.lower()] += 1
        summary[recommendation.lower()] += 1

        analysis[column] = {
            "risk_score": score,
            "severity": severity,
            "recommendation": recommendation,
            "missing_percentage": round(missing_pct, 2),
            "unique_count": unique_count,
            "evidence": evidence,
            "suggested_actions": list(dict.fromkeys(suggested_actions)),
        }

    return {
        "features": analysis,
        "summary": summary,
        "methodology": {
            "description": (
                "Feature risk is calculated from multiple heuristic signals "
                "including missingness, identifier-like patterns, potential "
                "leakage, cardinality, multicollinearity, correlation, "
                "duplicates, outliers, skewness, and temporal signals."
            ),
            "score_interpretation": (
                "The risk score is an internal evidence score and "
                "is not a probability of failure."
            ),
            "important_note": (
                "Potential leakage, identifier detection, and outlier flags "
                "are heuristics and must be reviewed by the user."
            ),
            "severity_bands": {
                "LOW": "0-14",
                "MEDIUM": "15-29",
                "HIGH": "30-49",
                "CRITICAL": "50+",
            },
        },
    }


# ============================================================
# Complete ML risk analysis (entry point used by pipeline)
# ============================================================

def run_ml_risk_analysis(
    df: pd.DataFrame,
    numerical_columns: list[str],
    categorical_columns: list[str],
    target_col: str | None,
) -> dict:
    """
    Run the complete ML risk analysis pipeline.

    Existing keys (class_imbalance, multicollinearity, potential_leakage,
    high_cardinality_features, train_test_distribution_diff) keep their
    previous shapes so scoring.py, recommendations.py, and the React UI
    continue to work without changes.

    New keys:
      - feature_risk_analysis  → per-feature scores, severity, evidence, actions
      - duplicate_columns      → exact + near-duplicate pairs
      - dimensionality_risk    → sample-size vs feature-count warning
    """
    class_imbalance = detect_class_imbalance(df, target_col)
    multicollinearity = detect_multicollinearity(df, numerical_columns)
    potential_leakage = detect_potential_leakage(df, target_col)
    high_cardinality = detect_high_cardinality_features(df, categorical_columns)
    distribution_diff = train_test_distribution_diff(df, numerical_columns)
    duplicates = detect_duplicate_columns(df, numerical_columns)
    dimensionality = detect_dimensionality_risk(
        df, numerical_columns, categorical_columns
    )
    feature_risk_analysis = analyze_feature_risks(
        df=df,
        numerical_columns=numerical_columns,
        categorical_columns=categorical_columns,
        target_col=target_col,
        leakage=potential_leakage,
        multicollinearity=multicollinearity,
        high_cardinality=high_cardinality,
        duplicates=duplicates,
    )

    return {
        # --- existing (backward-compatible) ---
        "class_imbalance": class_imbalance,
        "multicollinearity": multicollinearity,
        "potential_leakage": potential_leakage,
        "high_cardinality_features": high_cardinality,
        "train_test_distribution_diff": distribution_diff,
        # --- new ---
        "feature_risk_analysis": feature_risk_analysis,
        "duplicate_columns": duplicates,
        "dimensionality_risk": dimensionality,
    }