"""The readiness score: deterministic, versioned, and fully itemised.

score = 100 - sum(deductions). Each finding deducts
    severity_weight x confidence_factor x breadth_factor
scaled down proportionally when its category hits the category cap, so one messy column or
one noisy category cannot zero the score. A finding is the unit of deduction, so a problem is
counted once. Money is kept in integer hundredths so deductions sum EXACTLY to 100 - score.
"""
from __future__ import annotations

from typing import Any, Dict, List

from app.engine.findings import Finding

SCORE_VERSION = "1.0"
SEVERITY_WEIGHT = {"critical": 18.0, "high": 10.0, "medium": 5.0, "low": 1.5, "info": 0.0}
CONFIDENCE_FACTOR = {"high": 1.0, "medium": 0.75, "low": 0.5}
EFFORT_COST = {"low": 1.0, "medium": 2.0, "high": 4.0}
CATEGORY_CAP = {"quality": 35.0, "outliers": 10.0, "leakage": 30.0, "ml_risk": 25.0}   # sums to 100
BANDS = (("Excellent", 85), ("Good", 70), ("Fair", 50), ("Poor", 0))


def breadth_factor(share: float) -> float:
    """0.6 (touches little) .. 1.2 (touches >= 30% of rows/columns, or is dataset-wide)."""
    return 0.6 + 0.6 * min(1.0, share / 0.3)


def band_for(value: float) -> str:
    return next(name for name, lo in BANDS if value >= lo)


def _raw_cents(f: Finding) -> int:
    raw = SEVERITY_WEIGHT[f.severity] * CONFIDENCE_FACTOR[f.confidence] * breadth_factor(f.breadth)
    return int(round(raw * 100))


def compute_deductions(findings: List[Finding]) -> List[int]:
    """Integer cents deducted per finding (same order as input), caps applied, exact."""
    cents = [_raw_cents(f) for f in findings]
    out = list(cents)
    for cat, cap in CATEGORY_CAP.items():
        idx = [i for i, f in enumerate(findings) if f.category == cat and cents[i] > 0]
        total, cap_c = sum(cents[i] for i in idx), int(cap * 100)
        if total <= cap_c:
            continue
        # largest-remainder allocation so the category sums to exactly cap_c
        exact = [cents[i] * cap_c / total for i in idx]
        floors = [int(e) for e in exact]
        rest = cap_c - sum(floors)
        order = sorted(range(len(idx)), key=lambda j: (-(exact[j] - floors[j]), idx[j]))
        for j in order[:rest]:
            floors[j] += 1
        for j, i in enumerate(idx):
            out[i] = floors[j]
    return out


def _apply_critical_leakage_cap(cents: List[int], critical_idx: List[int], target_total: int = 6000) -> List[int]:
    """Add only the extra deduction needed to reach the critical-leakage cap. Unrelated findings remain unchanged."""
    out = list(cents)
    current_total = sum(out)
    if not critical_idx or current_total >= target_total:
        return out
    delta = target_total - current_total
    weights = [max(0, out[i]) for i in critical_idx]
    total_weight = sum(weights)
    if total_weight == 0:
        weights = [1] * len(critical_idx)
        total_weight = len(critical_idx)
    exact = [delta * w / total_weight for w in weights]
    floors = [int(e) for e in exact]
    rest = delta - sum(floors)
    order = sorted(range(len(critical_idx)), key=lambda j: (-(exact[j] - floors[j]), critical_idx[j]))
    for j in order[:rest]:
        floors[j] += 1
    for pos, idx in enumerate(critical_idx):
        out[idx] += floors[pos]
    return out


def score_findings(findings: List[Finding], provisional: bool = False) -> Dict[str, Any]:
    """Sets finding.score_impact and returns the score block stored on the analysis."""
    base_cents = compute_deductions(findings)
    score_cents = list(base_cents)
    critical_idx = [i for i, f in enumerate(findings) if f.category == "leakage" and f.severity == "critical"]
    has_critical_leakage = bool(critical_idx)
    if has_critical_leakage:
        score_cents = _apply_critical_leakage_cap(score_cents, critical_idx)
    for f, c in zip(findings, score_cents):
        f.score_impact = c / 100
    total = sum(score_cents)
    value = (10000 - total) / 100
    if has_critical_leakage:
        value = min(value, 40.0)
    deductions = []
    for i, (f, c) in enumerate(zip(findings, score_cents)):
        pts = c / 100
        if pts <= 0:
            continue
        cap_adjustment = (c - base_cents[i]) / 100 if i in critical_idx else 0.0
        deductions.append({
            "code": f.code, "columns": f.columns, "category": f.category, "severity": f.severity,
            "confidence": f.confidence, "points": pts, "cap_adjustment": cap_adjustment,
        })
    deductions.sort(key=lambda d: -d["points"])
    dim_map = {
        "missing_values": "completeness", "empty_column": "completeness", "disguised_missing_values": "completeness", "target_missing": "completeness",
        "invalid_values": "validity", "future_dates": "validity", "unexpected_categorical_value": "validity",
        "mixed_types": "consistency", "inconsistent_categories": "consistency", "mixed_units": "consistency", "encoding_damage": "consistency",
        "exact_duplicates": "uniqueness", "near_duplicates": "uniqueness", "duplicate_entities_different_ids": "uniqueness",
    }
    dim_deductions = {"completeness": 0.0, "validity": 0.0, "consistency": 0.0, "uniqueness": 0.0, "model_readiness": 0.0}
    for f, c in zip(findings, score_cents):
        pts = c / 100
        dim = dim_map.get(f.code, "model_readiness")
        dim_deductions[dim] = round(dim_deductions[dim] + pts, 2)
    dimensions = {dim: max(0.0, round(100.0 - min(100.0, ded * 2.5), 1)) for dim, ded in dim_deductions.items()}
    data_quality_score = round(sum(dimensions[d] for d in ("completeness", "validity", "consistency", "uniqueness")) / 4, 1)

    unresolved_critical = sum(1 for f in findings if f.severity == "critical")
    unresolved_high = sum(1 for f in findings if f.severity == "high")

    # Specific validation check statuses
    val_missing = "Failed" if any(f.code in ("missing_values", "empty_column", "target_missing") and f.severity in ("critical", "high") for f in findings) else "Passed"
    val_target = "Not evaluated" if provisional else ("Failed" if any(f.category == "ml_risk" and f.severity in ("critical", "high") and "target" in f.code for f in findings) else "Passed")
    val_leakage = "Not evaluated" if provisional else ("Failed" if any(f.category == "leakage" and f.severity in ("critical", "high") for f in findings) else "Passed")
    val_categories = "Failed" if any(f.code in ("unexpected_categorical_value", "inconsistent_categories") and f.severity in ("critical", "high") for f in findings) else "Passed"
    val_duplicates = "Review needed" if any("duplicate" in f.code for f in findings) else "Passed"
    val_outliers = "Review needed" if any("outlier" in f.code for f in findings) else "Passed"

    validation_checks = {
        "missing_values": val_missing,
        "target_column": val_target,
        "data_leakage": val_leakage,
        "category_validity": val_categories,
        "duplicate_records": val_duplicates,
        "outlier_observations": val_outliers,
    }

    if has_critical_leakage:
        verdict = "Not Ready - Critical Leakage"
    elif unresolved_critical > 0:
        verdict = "Not Ready"
    elif unresolved_high > 0:
        verdict = "Requires Review"
    elif provisional:
        verdict = "Provisional"
    elif value >= 85:
        verdict = "Ready for ML"
    else:
        verdict = "Needs Improvement"

    return {"value": value, "band": band_for(value), "provisional": provisional, "version": SCORE_VERSION,
            "data_quality_score": data_quality_score, "ml_readiness": verdict,
            "unresolved_critical": unresolved_critical, "unresolved_high": unresolved_high,
            "validation_checks": validation_checks,
            "deductions": deductions, "total_deducted": total / 100, "dimensions": dimensions,
            "category_totals": {cat: sum(c for f, c in zip(findings, score_cents) if f.category == cat) / 100 for cat in CATEGORY_CAP}}


def fix_first(findings: List[Finding], limit: int = 3) -> List[Dict[str, Any]]:
    """Findings with a fix, ranked by (points gained / effort)."""
    ranked = [f for f in findings if f.score_impact > 0 and f.fixes and f.fixes[0]["op"] != "manual"]
    ranked.sort(key=lambda f: (-(f.score_impact / EFFORT_COST[f.effort]), f.code, f.columns))
    return [{"code": f.code, "columns": f.columns, "points_gain": f.score_impact, "effort": f.effort,
             "headline": f.plain_text.split(". ")[0].rstrip(".") + ".", "fix": f.fixes[0]} for f in ranked[:limit]]
