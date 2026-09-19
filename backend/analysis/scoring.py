"""
MLReady - ML Readiness Score

Deterministic, explainable scoring: start at 100, subtract a capped penalty
per issue category, floor at 0. Every deduction is returned alongside the
final score so the UI can show "why" - never just a bare number. Caps exist
so one extremely messy column can't single-handedly zero out an otherwise
fine dataset, matching the spec's "small/moderate" language for lower-impact
issues (outliers) vs "large penalty" for the highest-impact one (leakage).
"""

from __future__ import annotations

from config import SCORE_BANDS


def _band_for(score: float) -> str:
    for low, high, label in SCORE_BANDS:
        if low <= score <= high:
            return label
    return "Poor"


def calculate_readiness_score(quality: dict, outliers: dict, risks: dict) -> dict:
    score = 100.0
    deductions: list[dict] = []

    def deduct(issue: str, points: float):
        nonlocal score
        points = round(points, 1)
        if points <= 0:
            return
        score -= points
        deductions.append({"issue": issue, "points": -points})

    # --- Data quality penalties -------------------------------------------------
    missing_pct = quality.get("missing_values", {}).get("overall_percentage", 0)
    if missing_pct > 0:
        deduct(f"Missing values ({missing_pct}% of all cells)", min(20, missing_pct * 0.6))

    dup_pct = quality.get("duplicates", {}).get("percentage", 0)
    if dup_pct > 0:
        deduct(f"Duplicate records ({dup_pct}% of rows)", min(10, dup_pct * 0.5))

    n_constant = len(quality.get("constant_columns", []))
    if n_constant:
        deduct(f"Constant columns ({n_constant})", min(10, n_constant * 3))

    n_near_constant = len(quality.get("near_constant_columns", {}))
    if n_near_constant:
        deduct(f"Near-constant columns ({n_near_constant})", min(6, n_near_constant * 2))

    n_dtype_issues = len(quality.get("dtype_issues", {}))
    if n_dtype_issues:
        deduct(f"Likely mistyped columns ({n_dtype_issues})", min(6, n_dtype_issues * 2))

    n_hc_quality = len(quality.get("high_cardinality_columns", {}))
    if n_hc_quality:
        deduct(f"High-cardinality columns ({n_hc_quality})", min(8, n_hc_quality * 3))

    # --- Outlier penalty: require cross-method evidence -------------------------
    consensus = outliers.get("consensus", {})
    strong_outlier_columns = sum(1 for info in consensus.values() if info.get("agreement", 0) >= 0.67)
    if strong_outlier_columns:
        deduct(f"Cross-method outlier evidence ({strong_outlier_columns} column(s))", min(6, strong_outlier_columns * 1.5))
    iso = outliers.get("isolation_forest", {})
    if iso.get("applied") and iso.get("percentage", 0) > 0:
        deduct(f"Multivariate outliers (~{iso['percentage']}% of rows)", min(4, iso["percentage"] * 0.2))

    # --- ML risk penalties --------------------------------------------------------
    class_imbalance = risks.get("class_imbalance", {})
    if class_imbalance.get("applicable") and class_imbalance.get("is_imbalanced"):
        ratio = class_imbalance.get("majority_minority_ratio", 0)
        deduct(f"Class imbalance (majority:minority ≈ {ratio}:1)", min(15, 5 + max(0, ratio - 4)))

    multicollinearity = risks.get("multicollinearity", {})
    n_corr_pairs = len(multicollinearity.get("high_correlation_pairs", []))
    n_high_vif = len(multicollinearity.get("high_vif_features", {}))
    if n_corr_pairs or n_high_vif:
        deduct(
            f"Multicollinearity ({n_corr_pairs} correlated pair(s), {n_high_vif} high-VIF feature(s))",
            min(10, (n_corr_pairs + n_high_vif) * 2),
        )

    leakage = risks.get("potential_leakage", {})
    n_leak = len(leakage.get("potential_leakage_columns", []))
    if n_leak:
        deduct(f"Potential data leakage ({n_leak} suspicious column(s))", min(25, n_leak * 8))

    n_hc_ml = len(risks.get("high_cardinality_features", {}))
    if n_hc_ml:
        deduct(f"High-cardinality features affecting ML encoding ({n_hc_ml})", min(6, n_hc_ml * 2))

    dist_shift = risks.get("train_test_distribution_diff", {})
    n_shifted = len(dist_shift.get("columns_with_significant_shift", {})) if dist_shift.get("applicable") else 0
    if n_shifted:
        deduct(f"Train/test distribution shift ({n_shifted} column(s))", min(8, n_shifted * 2))

    final_score = max(0.0, round(score, 1))
    return {
        "score": final_score,
        "category": _band_for(final_score),
        "deductions": deductions,
        "starting_score": 100,
    }
