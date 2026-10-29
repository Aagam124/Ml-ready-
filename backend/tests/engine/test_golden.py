"""Golden benchmark: every planted problem is found; clean datasets raise nothing serious."""
import numpy as np
import pandas as pd

from app.benchmark import PLANTS, make_clean, make_planted
from app.engine.pipeline import run_analysis

SERIOUS = ("critical", "high")


def _found(result, code, col):
    return any(f["code"] == code and (col is None or col in f["columns"]) for f in result["findings"])


def test_every_planted_problem_is_found_together():
    df, target, expected = make_planted(list(PLANTS))
    r = run_analysis(df, target)
    assert not r["check_errors"], r["check_errors"]
    for plant, (code, col) in expected.items():
        assert _found(r, code, col), f"planted '{plant}' was not found as {code}/{col}"


def test_every_planted_problem_is_found_alone():
    for plant in PLANTS:
        df, target, expected = make_planted([plant], seed=3)
        r = run_analysis(df, target)
        code, col = expected[plant]
        assert _found(r, code, col), f"planted '{plant}' alone was not found"


def test_clean_datasets_have_no_serious_or_medium_findings():
    for kind in ("binary", "multiclass", "regression"):
        for seed in range(6):
            df, target = make_clean(kind, seed=seed)
            r = run_analysis(df, target)
            bad = [(f["severity"], f["code"], f["columns"]) for f in r["findings"] if f["severity"] in SERIOUS]
            assert not bad, (kind, seed, bad)
            medium = [f["code"] for f in r["findings"] if f["severity"] == "medium"]
            assert not medium, (kind, seed, medium)
            assert r["score"]["value"] >= 85 or r["score"]["band"] in ("Excellent", "Good")


def test_clean_datasets_without_target_have_no_serious_findings():
    for seed in range(3):
        df, _ = make_clean("binary", seed=seed)
        r = run_analysis(df)
        assert not [f for f in r["findings"] if f["severity"] in SERIOUS]
        assert r["score"]["provisional"] is True


def test_negative_salary_is_an_error_not_an_outlier():
    df, target, _ = make_planted(["negative_salary"])
    r = run_analysis(df, target)
    inv = [f for f in r["findings"] if f["code"] == "invalid_values" and "salary" in f["columns"]]
    assert inv and inv[0]["severity"] in ("high", "medium")
    assert not [f for f in r["findings"] if f["code"] == "outliers" and "salary" in f["columns"]], "double counted"


def test_missingness_pattern_detects_dependence_on_another_column():
    df, target, _ = make_planted(["missingness"])
    f = next(f for f in run_analysis(df, target)["findings"] if f["code"] == "missing_values")
    assert f["evidence"]["pattern"]["kind"] == "possibly_MAR"
    assert "income" in [h["column"] for h in f["evidence"]["pattern"]["associated_with"]]


def test_imbalance_ratio_and_denominator():
    df, target, _ = make_planted(["imbalance"])
    f = next(f for f in run_analysis(df, target)["findings"] if f["code"] == "class_imbalance")
    assert f["evidence"]["ratio"] > 8 and "labelled rows" in f["evidence"]["denominator"]
    assert f["fixes"][0]["op"] == "modeling_option"


def test_rare_classes_multiclass():
    rng = np.random.default_rng(0)
    df, _ = make_clean("binary", n=600)
    df["grade"] = rng.choice(["a", "b", "c"], 600, p=[0.5, 0.49, 0.01])
    df.loc[df.index[:4], "grade"] = "z"
    r = run_analysis(df, "grade")
    assert any(f["code"] == "rare_classes" for f in r["findings"])
