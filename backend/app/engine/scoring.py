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


def score_findings(findings: List[Finding], provisional: bool = False) -> Dict[str, Any]:
    """Sets finding.score_impact and returns the score block stored on the analysis."""
    cents = compute_deductions(findings)
    for f, c in zip(findings, cents):
        f.score_impact = c / 100
    total = sum(cents)
    value = (10000 - total) / 100
    deductions = [
        {"code": f.code, "columns": f.columns, "category": f.category, "severity": f.severity, "confidence": f.confidence,
         "points": c / 100}
        for f, c in zip(findings, cents) if c > 0
    ]
    deductions.sort(key=lambda d: -d["points"])
    # Map finding codes to 5 enterprise data quality dimensions
    dim_map = {
        "missing_values": "completeness", "disguised_missing_values": "completeness", "target_missing": "completeness",
        "invalid_values": "validity", "future_dates": "validity", "unexpected_categorical_value": "validity",
        "mixed_types": "consistency", "inconsistent_categories": "consistency", "mixed_units": "consistency", "encoding_damage": "consistency",
        "exact_duplicates": "uniqueness", "near_duplicates": "uniqueness", "duplicate_entities_different_ids": "uniqueness",
    }
    dim_deductions = {"completeness": 0.0, "validity": 0.0, "consistency": 0.0, "uniqueness": 0.0, "model_readiness": 0.0}
    for f, c in zip(findings, cents):
        pts = c / 100
        dim = dim_map.get(f.code, "model_readiness")
        dim_deductions[dim] = round(dim_deductions[dim] + pts, 2)
    dimensions = {dim: max(0.0, round(100.0 - min(100.0, ded * 2.5), 1)) for dim, ded in dim_deductions.items()}

    return {"value": value, "band": band_for(value), "provisional": provisional, "version": SCORE_VERSION,
            "deductions": deductions, "total_deducted": total / 100, "dimensions": dimensions,
            "category_totals": {cat: sum(c for f, c in zip(findings, cents) if f.category == cat) / 100 for cat in CATEGORY_CAP}}


def fix_first(findings: List[Finding], limit: int = 3) -> List[Dict[str, Any]]:
    """Findings with a fix, ranked by (points gained / effort)."""
    ranked = [f for f in findings if f.score_impact > 0 and f.fixes and f.fixes[0]["op"] != "manual"]
    ranked.sort(key=lambda f: (-(f.score_impact / EFFORT_COST[f.effort]), f.code, f.columns))
    return [{"code": f.code, "columns": f.columns, "points_gain": f.score_impact, "effort": f.effort,
             "headline": f.plain_text.split(". ")[0].rstrip(".") + ".", "fix": f.fixes[0]} for f in ranked[:limit]]
