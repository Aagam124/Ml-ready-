"""One test per false positive / bug ever found. Never delete these."""
import numpy as np
import pandas as pd

from app.benchmark import make_clean
from app.engine.pipeline import run_analysis


def codes(r):
    return {(f["code"], tuple(f["columns"])) for f in r["findings"]}


def test_tenure_years_and_months_are_redundant_not_temporal_leakage():
    df, target = make_clean("binary", seed=5)
    df["tenure_months"] = df["tenure_years"] * 12
    r = run_analysis(df, target)
    assert any(f["code"] == "redundant_columns" and set(f["columns"]) == {"tenure_years", "tenure_months"} for f in r["findings"])
    assert not any("leak" in f["code"] or "temporal" in f["code"] for f in r["findings"])
    sem = {c["name"]: c["semantic_type"] for c in r["profile"]["columns"]}
    assert sem["tenure_years"] != "datetime" and sem["tenure_months"] != "datetime"


def test_name_hint_alone_never_produces_a_high_severity_leakage_finding():
    df, target = make_clean("binary", seed=6)
    df["final_outcome_label"] = np.random.default_rng(1).normal(size=len(df))   # name screams leak, values are noise
    r = run_analysis(df, target)
    leaks = [f for f in r["findings"] if f["code"] == "target_leakage"]
    for f in leaks:
        assert f["severity"] in ("low", "info") and f["evidence"]["name_only"]
        assert f["confidence"] == "low"


def test_substring_names_do_not_trigger_hints():
    df, target = make_clean("binary", seed=7)
    df["resultant_noise"] = np.random.default_rng(2).normal(size=len(df))     # 'result' is only a substring
    df["targeted_ads"] = np.random.default_rng(3).integers(0, 4, len(df))
    r = run_analysis(df, target)
    assert not [f for f in r["findings"] if f["code"] == "target_leakage"]


def test_target_never_appears_in_feature_findings():
    from app.benchmark import make_planted
    from app.engine.pipeline import TARGET_CODES
    df, target, _ = make_planted(["imbalance", "leakage", "duplicates"])
    r = run_analysis(df, target)
    for f in r["findings"]:
        if target in f["columns"]:
            assert f["code"] in TARGET_CODES, f


def test_strong_but_legit_feature_is_only_possible_not_critical():
    rng = np.random.default_rng(0)
    n = 2000
    x = rng.normal(size=n)
    y = (x + rng.normal(0, 0.9, n) > 0).astype(int)        # AUC ~0.85-0.9: strong, not leaky
    df = pd.DataFrame({"x": x, "z": rng.normal(size=n), "w": rng.normal(size=n), "y": y})
    r = run_analysis(df, "y")
    assert not [f for f in r["findings"] if f["code"] == "target_leakage" and f["severity"] in ("critical", "high")]


def test_numeric_id_column_is_not_leakage_even_if_sequential_with_target_order():
    rng = np.random.default_rng(0)
    n = 800
    y = np.sort(rng.integers(0, 2, n))                       # target sorted -> row id "predicts" it
    df = pd.DataFrame({"row_id": np.arange(n), "a": rng.normal(size=n), "b": rng.normal(size=n), "y": y})
    r = run_analysis(df, "y")
    assert not [f for f in r["findings"] if f["code"] == "target_leakage"]
    assert any(f["code"] == "identifier_feature" for f in r["findings"])


def test_age_column_below_zero_or_over_120_flagged_but_normal_ages_not():
    df, target = make_clean("binary", seed=8)
    assert not [f for f in run_analysis(df, target)["findings"] if f["code"] == "invalid_values"]
    df.loc[0, "age"] = 190
    f = [f for f in run_analysis(df, target)["findings"] if f["code"] == "invalid_values"]
    assert f and f[0]["columns"] == ["age"]


def test_signed_columns_may_be_negative():
    df, target = make_clean("binary", seed=9)       # balance_change is signed by design
    assert not [f for f in run_analysis(df, target)["findings"] if "balance_change" in f["columns"] and f["code"] == "invalid_values"]


def test_numeric_text_with_placeholders_is_one_finding_not_two():
    from app.benchmark import make_planted
    df, target, _ = make_planted(["disguised_missing"])
    r = run_analysis(df, target)
    got = [f["code"] for f in r["findings"] if "monthly_usage" in f["columns"]]
    assert got.count("disguised_missing_values") == 1 and "mistyped_column" not in got and "mixed_types" not in got


def test_future_dates_check_uses_fixed_as_of_and_ignores_expiry_columns():
    from app.engine.config import EngineConfig
    df, target = make_clean("binary", n=400, seed=1)
    df["signup_date"] = pd.date_range("2020-01-01", periods=400, freq="D").astype(str)
    df["contract_expiry"] = pd.date_range("2030-01-01", periods=400, freq="D").astype(str)
    r = run_analysis(df, target, config=EngineConfig(as_of="2021-01-01"))
    fut = [f for f in r["findings"] if f["code"] == "future_dates"]
    assert [f["columns"] for f in fut] == [["signup_date"]]
