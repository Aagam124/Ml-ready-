"""
MLReady - Recommendation Engine

Pure rule-based text generation from the already-computed reports. No LLM,
no external API - every sentence here is a template filled from numbers
we already calculated, which is what makes the recommendations reproducible
and free to run.
"""

from __future__ import annotations

import pandas as pd


def _missing_value_recommendations(df: pd.DataFrame, quality: dict) -> list[str]:
    recs = []
    for col, info in quality.get("missing_values", {}).get("columns_with_missing", {}).items():
        pct = info["percentage"]
        is_numeric = col in df.columns and pd.api.types.is_numeric_dtype(df[col])
        if is_numeric:
            has_outliers = df[col].dropna().skew() if df[col].dropna().size > 2 else 0
            strategy = "median imputation" if abs(has_outliers) > 1 else "mean imputation"
            reason = "because the feature contains extreme values" if abs(has_outliers) > 1 else "since the feature is roughly symmetric"
            recs.append(f"'{col}' contains {pct}% missing values. {strategy.capitalize()} may be suitable {reason}.")
        else:
            recs.append(f"'{col}' contains {pct}% missing values. Consider mode imputation or a dedicated 'Unknown' category.")
        if pct > 40:
            recs.append(f"'{col}' is missing in over 40% of rows - consider dropping the column instead of imputing.")
    return recs


def _duplicate_recommendations(quality: dict) -> list[str]:
    dup = quality.get("duplicates", {})
    if dup.get("count", 0) > 0:
        return [f"{dup['count']} duplicate rows detected ({dup['percentage']}%). Remove them before training to avoid inflating certain patterns."]
    return []


def _constant_column_recommendations(quality: dict) -> list[str]:
    recs = []
    if quality.get("constant_columns"):
        cols = ", ".join(f"'{c}'" for c in quality["constant_columns"])
        recs.append(f"{cols} contain only a single value and add no predictive information. Drop them before training.")
    if quality.get("near_constant_columns"):
        cols = ", ".join(f"'{c}'" for c in quality["near_constant_columns"])
        recs.append(f"{cols} are near-constant (one value dominates). Review whether they carry any useful signal.")
    return recs


def _dtype_recommendations(quality: dict) -> list[str]:
    recs = []
    for col in quality.get("dtype_issues", {}):
        recs.append(f"'{col}' looks numeric but is stored as text. Convert it to a numeric dtype before modeling.")
    return recs


def _high_cardinality_recommendations(quality: dict, risks: dict) -> list[str]:
    recs = []
    for col in quality.get("high_cardinality_columns", {}):
        recs.append(f"'{col}' has very high cardinality. Consider target/frequency encoding instead of one-hot encoding.")
    for col in risks.get("high_cardinality_features", {}):
        if col not in quality.get("high_cardinality_columns", {}):
            recs.append(f"'{col}' may produce a sparse feature space after encoding. Consider grouping rare categories.")
    return recs


def _outlier_recommendations(outliers: dict) -> list[str]:
    recs = []
    iqr = outliers.get("iqr", {})
    if iqr:
        cols = ", ".join(f"'{c}'" for c in list(iqr.keys())[:5])
        recs.append(f"Outliers detected (IQR method) in {cols}. Review these points individually - they may be valid extreme observations rather than errors.")
    iso = outliers.get("isolation_forest", {})
    if iso.get("applied") and iso.get("outlier_count", 0) > 0:
        recs.append(f"Isolation Forest flagged {iso['outlier_count']} rows ({iso['percentage']}%) as multivariate outliers across {', '.join(iso['columns_used'][:5])}.")
    return recs


def _class_imbalance_recommendations(risks: dict) -> list[str]:
    ci = risks.get("class_imbalance", {})
    if ci.get("applicable") and ci.get("is_imbalanced"):
        return [
            f"Class imbalance detected (majority:minority ≈ {ci['majority_minority_ratio']}:1). "
            "Consider class weights, stratified splitting, or resampling (SMOTE/undersampling)."
        ]
    return []


def _multicollinearity_recommendations(risks: dict) -> list[str]:
    recs = []
    mc = risks.get("multicollinearity", {})
    for pair in mc.get("high_correlation_pairs", []):
        recs.append(
            f"'{pair['feature_a']}' and '{pair['feature_b']}' have correlation {pair['correlation']}. "
            "Consider removing one feature or investigating multicollinearity."
        )
    high_vif = mc.get("high_vif_features", {})
    if high_vif:
        cols = ", ".join(f"'{c}' (VIF={v})" for c, v in list(high_vif.items())[:5])
        recs.append(f"High variance inflation detected: {cols}. These features are largely explained by other features in the dataset.")
    return recs


def _leakage_recommendations(risks: dict) -> list[str]:
    recs = []
    for entry in risks.get("potential_leakage", {}).get("potential_leakage_columns", []):
        reason_text = " ".join(entry["reasons"])
        recs.append(f"Potential target leakage detected in '{entry['column']}': {reason_text} Review this feature before model training.")
    return recs


def _distribution_shift_recommendations(risks: dict) -> list[str]:
    shift = risks.get("train_test_distribution_diff", {})
    cols = shift.get("columns_with_significant_shift", {}) if shift.get("applicable") else {}
    if cols:
        col_list = ", ".join(f"'{c}'" for c in cols)
        return [f"Train/test distributions differ significantly for {col_list}. Consider shuffling before splitting or using stratified/time-aware splitting."]
    return []


def generate_recommendations(df: pd.DataFrame, quality: dict, outliers: dict, risks: dict) -> list[str]:
    """Aggregate every rule-based recommendation into a single ordered list.

    Order roughly matches severity: leakage and imbalance first (biggest
    model impact), structural/cleanup items last.
    """
    recs: list[str] = []
    recs += _leakage_recommendations(risks)
    recs += _class_imbalance_recommendations(risks)
    recs += _multicollinearity_recommendations(risks)
    recs += _distribution_shift_recommendations(risks)
    recs += _missing_value_recommendations(df, quality)
    recs += _duplicate_recommendations(quality)
    recs += _outlier_recommendations(outliers)
    recs += _high_cardinality_recommendations(quality, risks)
    recs += _dtype_recommendations(quality)
    recs += _constant_column_recommendations(quality)

    if not recs:
        recs.append("No significant data quality or ML risk issues were detected. This dataset looks ready for baseline modeling.")
    return recs
