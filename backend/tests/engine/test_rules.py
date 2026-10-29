"""Tests for the reusable Quality Rules Engine."""
import pandas as pd
import pytest

from app.engine.rules import QualityRule, QualityRulesEngine


def test_quality_rules_not_null_and_min_row_count():
    df = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", None, "David", "Eve"],
        "age": [25, 30, 35, 40, 45],
    })

    engine = QualityRulesEngine([
        QualityRule(id="r1", name="ID Not Null", rule_type="not_null", column="id", severity="critical"),
        QualityRule(id="r2", name="Name Not Null", rule_type="not_null", column="name", severity="high"),
        QualityRule(id="r3", name="Min 3 Rows", rule_type="min_row_count", params={"min_rows": 3}),
        QualityRule(id="r4", name="Min 10 Rows", rule_type="min_row_count", params={"min_rows": 10}),
    ])

    results = {r.rule_id: r for r in engine.evaluate(df)}

    assert results["r1"].status == "passed" and results["r1"].failure_count == 0
    assert results["r2"].status == "failed" and results["r2"].failure_count == 1
    assert results["r3"].status == "passed"
    assert results["r4"].status == "failed" and results["r4"].failure_count == 5


def test_quality_rules_range_unique_and_allowed():
    df = pd.DataFrame({
        "account_id": [101, 102, 103, 101],  # duplicate 101
        "score": [85.0, 92.5, 110.0, -5.0],   # -5 and 110 out of [0, 100]
        "tier": ["Gold", "Silver", "Unknown", "Platinum"],
    })

    engine = QualityRulesEngine([
        QualityRule(id="u1", name="Unique Account ID", rule_type="unique", column="account_id"),
        QualityRule(id="rng1", name="Score 0-100", rule_type="numeric_range", column="score", params={"min": 0, "max": 100}),
        QualityRule(id="cat1", name="Allowed Tiers", rule_type="allowed_values", column="tier", params={"allowed": ["Bronze", "Silver", "Gold", "Platinum"]}),
    ])

    results = {r.rule_id: r for r in engine.evaluate(df)}

    assert results["u1"].status == "failed" and results["u1"].failure_count == 1
    assert results["rng1"].status == "failed" and results["rng1"].failure_count == 2
    assert results["cat1"].status == "failed" and results["cat1"].failure_count == 1
    assert "Unknown" in results["cat1"].sample_failures


def test_quality_rules_regex_and_date():
    df = pd.DataFrame({
        "email": ["user@example.com", "valid.name@domain.co", "bad_email_at_nowhere"],
        "created_at": ["2023-01-15", "2023-06-20", "2025-12-01"],
    })

    engine = QualityRulesEngine([
        QualityRule(id="reg1", name="Valid Email", rule_type="regex_match", column="email", params={"pattern": r"^[^@\s]+@[^@\s]+\.[^@\s]+$"}),
        QualityRule(id="dt1", name="Date Before 2024", rule_type="date_interval", column="created_at", params={"max_date": "2023-12-31"}),
    ])

    results = {r.rule_id: r for r in engine.evaluate(df)}

    assert results["reg1"].status == "failed" and results["reg1"].failure_count == 1
    assert results["dt1"].status == "failed" and results["dt1"].failure_count == 1
