"""Focused unit tests for individual checks."""
import numpy as np
import pandas as pd

from app.benchmark import make_clean
from app.engine.config import EngineConfig
import app.engine.pipeline as pipeline_module
from app.engine.pipeline import run_analysis
from app.engine.registry import Check
from app.engine.target import suggest_targets, task_type_for
from app.profiling.profile import profile_dataframe


def find(r, code):
    return [f for f in r["findings"] if f["code"] == code]


def test_target_detection_ranks_named_binary_last_column_first():
    df, target = make_clean("binary")
    c = suggest_targets(df, profile_dataframe(df))
    assert c[0]["column"] == "churned" and c[0]["task_type"] == "binary" and c[0]["confidence"] > 0.5


def test_task_type_for_identifier_and_constant_is_none():
    df = pd.DataFrame({"id": range(50), "c": 1, "y": [0, 1] * 25})
    p = {c["name"]: c["semantic_type"] for c in profile_dataframe(df)["columns"]}
    assert task_type_for(df["id"], p["id"]) is None and task_type_for(df["c"], p["c"]) is None
    assert task_type_for(df["y"], p["y"]) == "binary"


def test_exact_duplicates_with_shared_ids():
    df, target = make_clean("binary", n=500)
    df = pd.concat([df, df.iloc[:30]], ignore_index=True)
    f = find(run_analysis(df, target), "exact_duplicates")[0]
    assert f["evidence"]["n_duplicate_rows"] == 30 and f["evidence"]["share_identifier"] is True
    assert "of 530 rows" in f["plain_text"]


def test_low_cardinality_data_does_not_treat_repeats_as_errors():
    df = pd.DataFrame({"a": np.random.default_rng(0).integers(0, 2, 300), "b": np.random.default_rng(1).integers(0, 2, 300),
                       "y": np.random.default_rng(2).integers(0, 2, 300)})
    f = find(run_analysis(df, "y"), "exact_duplicates")
    assert f and f[0]["confidence"] == "low" and f[0]["severity"] == "low"


def test_near_duplicates_after_case_and_whitespace():
    df, target = make_clean("binary", n=400)
    d2 = df.iloc[:20].copy()
    d2["region"] = d2["region"].str.upper() + " "
    r = run_analysis(pd.concat([df, d2], ignore_index=True), target)
    assert find(r, "near_duplicates") or find(r, "inconsistent_categories")


def test_inconsistent_categories_mapping_merges_abbreviations():
    df, target = make_clean("binary", n=600)
    df["sex"] = np.random.default_rng(0).choice(["Male", "male", "M", "Female", "F"], 600)
    f = find(run_analysis(df, target), "inconsistent_categories")[0]
    m = f["evidence"]["mapping"]
    assert m.get("M") == "Male" and m.get("male") == "Male" and m.get("F") == "Female"
    assert set(m.values()) == {"Male", "Female"}


def test_mixed_units_detected():
    df, target = make_clean("binary", n=400)
    df["weight"] = np.random.default_rng(0).choice(["70 kg", "150 lb", "65 kg", "180 lb"], 400)
    f = find(run_analysis(df, target), "inconsistent_units")
    assert f and set(f[0]["evidence"]["units"]) == {"kg", "lb"} and f[0]["evidence"]["convertible_family"] == "mass"


def test_encoding_damage_detected():
    df, target = make_clean("binary", n=400)
    df["city"] = np.random.default_rng(0).choice(["Zürich", "MÃ¼nchen", "Berlin"], 400)
    assert find(run_analysis(df, target), "encoding_issues")


def test_outliers_report_denominator_and_need_agreement():
    rng = np.random.default_rng(0)
    df, target = make_clean("binary", n=800)
    df.loc[rng.choice(800, 25, replace=False), "feature_a"] = 40.0
    f = find(run_analysis(df, target), "outliers")
    f = [x for x in f if x["columns"] == ["feature_a"]][0]
    assert f["evidence"]["n_values"] == 800 and "non-missing values" in f["plain_text"]
    assert f["evidence"]["n_both_iqr_and_mad"] >= 20


def test_normal_data_produces_no_outlier_findings():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=2000), "b": rng.normal(size=2000), "y": rng.integers(0, 2, 2000)})
    assert not find(run_analysis(df, "y"), "outliers")


def test_log_normal_and_gamma_clean_data_do_not_trigger_outliers():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "income": np.exp(rng.normal(10.8, 0.45, 2000)).round(2),
        "monthly_usage": rng.gamma(3.0, 40.0, 2000).round(1),
        "y": rng.integers(0, 2, 2000),
    })
    assert not find(run_analysis(df, "y"), "outliers")


def test_extreme_positive_tail_values_still_trigger_outliers():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"income": np.exp(rng.normal(10.8, 0.45, 800)), "y": rng.integers(0, 2, 800)})
    df.loc[rng.choice(800, 30, replace=False), "income"] = df["income"].quantile(0.99) * 15
    f = find(run_analysis(df, "y"), "outliers")
    assert any(item["columns"] == ["income"] for item in f)


def test_leakage_deterministic_categorical_mirror():
    df, target = make_clean("binary", n=800)
    df["churn_label_text"] = np.where(df[target] == 1, "yes", "no")
    f = find(run_analysis(df, target), "target_leakage")[0]
    assert f["columns"] == ["churn_label_text"] and f["evidence"]["tier"] == "Likely" and f["severity"] == "critical"


def test_leakage_regression_target_transform():
    df, target = make_clean("regression", n=800)
    df["spend_usd_x2"] = df[target] * 2
    f = [x for x in find(run_analysis(df, target), "target_leakage") if x["columns"] == ["spend_usd_x2"]]
    assert f and f[0]["evidence"]["tier"] == "Likely"


def test_leakage_multiclass():
    df, target = make_clean("multiclass", n=900)
    df["grade_code"] = df[target].map({"bronze": 1, "silver": 2, "gold": 3})
    f = [x for x in find(run_analysis(df, target), "target_leakage") if x["columns"] == ["grade_code"]]
    assert f and f[0]["severity"] in ("critical", "high")


def test_multicollinearity_vif_without_perfect_redundancy():
    rng = np.random.default_rng(0)
    n = 1000
    a = rng.normal(size=n)
    df = pd.DataFrame({"a": a, "b": a * 0.8 + rng.normal(0, 0.25, n), "c": rng.normal(size=n), "y": rng.integers(0, 2, n)})
    r = run_analysis(df, "y")
    f = find(r, "multicollinearity")
    assert f and not find(r, "redundant_columns") and set(f[0]["columns"]) >= {"a", "b"}


def test_redundant_categorical_one_to_one():
    df, target = make_clean("binary", n=600)
    df["region_code"] = df["region"].map({"north": 1, "south": 2, "east": 3, "west": 4}).astype(str)
    f = find(run_analysis(df, target), "redundant_columns")
    assert f and set(f[0]["columns"]) == {"region", "region_code"}


def test_adversarial_validation_finds_shift_between_train_and_test():
    df, target = make_clean("binary", n=800, seed=1)
    test = df.sample(300, random_state=0).copy()
    test["income"] = test["income"] * 3
    r = run_analysis(df, target, test_df=test)
    f = find(r, "train_test_shift")
    assert f and f[0]["evidence"]["adversarial_auc"] > 0.8 and f[0]["columns"][0] == "income"


def test_no_shift_finding_when_distributions_match():
    df, target = make_clean("binary", n=1000, seed=2)
    r = run_analysis(df, target, test_df=df.sample(400, random_state=1))
    assert not find(r, "train_test_shift")


def test_temporal_drift_is_found():
    df, target = make_clean("binary", n=1000, seed=3)
    df["event_date"] = pd.date_range("2022-01-01", periods=1000, freq="D").astype(str)
    df.loc[800:, "income"] = df.loc[800:, "income"] * 4
    r = run_analysis(df, target)
    assert find(r, "temporal_shift") and find(r, "temporal_ordering")


def test_repeated_entity_ids_group_leakage():
    rng = np.random.default_rng(0)
    n = 600
    df = pd.DataFrame({"patient_id": rng.integers(0, 100, n), "a": rng.normal(size=n), "b": rng.normal(size=n), "y": rng.integers(0, 2, n)})
    f = find(run_analysis(df, "y"), "repeated_entity_ids")
    assert f and f[0]["evidence"]["n_rows_in_repeated_ids"] >= 0.95 * n and f[0]["fixes"][0]["params"]["split"] == "group"


def test_dimensionality_wide_data():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(40, 60)), columns=[f"f{i}" for i in range(60)])
    df["y"] = rng.integers(0, 2, 40)
    f = find(run_analysis(df, "y"), "dimensionality_risk")
    assert f and f[0]["severity"] == "high"


def test_sensitive_attribute_is_informational_with_proxy():
    df, target = make_clean("binary", n=800)
    df["gender"] = np.random.default_rng(0).choice(["m", "f"], 800)
    df["pincode"] = np.where(df["gender"] == "m", "100", "200")
    r = run_analysis(df, target)
    s = [f for f in find(r, "sensitive_attribute") if f["columns"] == ["gender"]][0]
    assert s["severity"] == "info" and s["score_impact"] == 0 and s["evidence"]["proxies"]


def test_high_cardinality_advises_encoding_and_is_scored_once():
    df, target = make_clean("binary", n=1200)
    df["zip"] = np.random.default_rng(0).choice([f"z{i}" for i in range(150)], 1200)
    r = run_analysis(df, target)
    f = find(r, "high_cardinality")
    assert len(f) == 1 and {x["params"]["method"] for x in f[0]["fixes"]} == {"frequency", "target"}


def test_target_missing_labels():
    df, target = make_clean("binary", n=500)
    df.loc[:49, target] = np.nan
    f = find(run_analysis(df, target), "target_missing")
    assert f and f[0]["evidence"]["n_missing"] == 50 and f[0]["severity"] == "high"


def test_target_missing_labels_with_placeholder_tokens_and_nans():
    df = pd.DataFrame({
        "a": range(12),
        "y": ["yes", "no", "N/A", "NA", "n/a", None, np.nan, "null", "none", "?", "  ", "--"],
    })
    f = find(run_analysis(df, "y"), "target_missing")
    assert f and f[0]["evidence"]["n_missing"] == 10 and f[0]["severity"] == "high"


def test_all_placeholder_target_is_missing_and_not_inferred():
    df = pd.DataFrame({"a": range(6), "y": ["N/A", "null", "?", " ", "--", None]})
    before = df.copy(deep=True)
    r = run_analysis(df, "y")
    f = find(r, "target_missing")
    assert f and f[0]["evidence"]["n_missing"] == 6 and f[0]["severity"] == "high"
    assert r["task_type"] is None
    assert df.equals(before)


def test_profile_missing_counts_include_placeholder_values_consistently():
    df = pd.DataFrame({
        "feature_a": [1, 2, "N/A", 4, 5],
        "feature_b": ["yes", "no", "?", "yes", None],
        "feature_c": [10, np.nan, 12, 13, 14],
    })
    prof = profile_dataframe(df)
    expected_total = sum(col["n_missing"] for col in prof["columns"])
    assert prof["missing"]["cells"] == expected_total == 4
    assert prof["missing"]["pct"] == 26.67
    assert prof["missing"]["denominator"] == "15 cells (5 rows x 3 columns)"


def test_valid_binary_target_values_do_not_trigger_target_missing():
    df = pd.DataFrame({"a": range(20), "y": ["yes", "no"] * 10})
    assert not find(run_analysis(df, "y"), "target_missing")


def test_valid_multiclass_target_values_do_not_trigger_target_missing():
    df = pd.DataFrame({"a": range(18), "y": ["a", "b", "c"] * 6})
    assert not find(run_analysis(df, "y"), "target_missing")


def test_numeric_target_values_remain_valid_when_not_missing():
    df = pd.DataFrame({"a": range(14), "y": [0, 1, -1, 1, 0, -1, 1, 0, 1, -1, 0, 1, 0, 1]})
    assert not find(run_analysis(df, "y"), "target_missing")


def test_target_missing_does_not_mutate_original_dataframe():
    df = pd.DataFrame({"a": range(10), "y": ["yes", "no", "N/A", "yes", "no", "?", "yes", "no", "None", "yes"]})
    before = df.copy(deep=True)
    run_analysis(df, "y")
    assert df.equals(before)


def test_target_missing_with_mixed_valid_and_placeholder_labels():
    df = pd.DataFrame({"a": range(9), "y": ["yes", "no", "N/A", "yes", "no", "?", "yes", "no", "na"]})
    f = find(run_analysis(df, "y"), "target_missing")
    assert f and f[0]["evidence"]["n_missing"] == 3


def test_sampling_is_reported_and_stratified():
    df, target = make_clean("binary", n=3000)
    r = run_analysis(df, target, config=EngineConfig(sample_threshold=1000))
    assert r["sampling"] == {"sampled": True, "n_rows_original": 3000, "n_rows_analysed": r["sampling"]["n_rows_analysed"]}
    assert 990 <= r["sampling"]["n_rows_analysed"] <= 1010
    assert any("stratified sample" in n for n in r["summary"]["notes"])


def test_a_crashing_check_is_reported_not_swallowed():
    from app.engine import registry

    class Boom(Check):
        id = "boom_test"
        def run(self, df, ctx):
            raise RuntimeError("kaboom")
    registry.REGISTRY["boom_test"] = Boom()
    try:
        df, target = make_clean("binary", n=200)
        r = run_analysis(df, target)
        assert {"check": "boom_test", "error": "RuntimeError: kaboom"} in r["check_errors"] and r["findings"] is not None
        assert r["analysis_status"] == "partial"
        assert r["checks_completed"]
        assert r["score"]["ml_readiness"] == "Incomplete validation"
        assert "Incomplete analysis" in r["summary"]["headline"]
    finally:
        del registry.REGISTRY["boom_test"]


def test_applies_to_exception_is_isolated_and_other_checks_continue(monkeypatch):
    class BrokenApplicability(Check):
        id = "broken_applicability_test"

        def applies_to(self, ctx):
            raise RuntimeError("applicability failed")

        def run(self, df, ctx):
            raise AssertionError("run must not be called")

    class HealthyCheck(Check):
        id = "healthy_after_applicability_test"

        def run(self, df, ctx):
            return []

    monkeypatch.setattr(pipeline_module, "ordered_checks", lambda: [BrokenApplicability(), HealthyCheck()])
    df, target = make_clean("binary", n=200)
    result = run_analysis(df, target)

    assert result["check_errors"] == [{"check": "broken_applicability_test", "error": "RuntimeError: applicability failed"}]
    assert result["checks_completed"] == ["healthy_after_applicability_test"]
    assert result["analysis_status"] == "partial"


def test_analysis_fails_when_no_check_completes(monkeypatch):
    class BrokenCheck(Check):
        id = "only_broken_test"

        def run(self, df, ctx):
            raise RuntimeError("no usable checks")

    monkeypatch.setattr(pipeline_module, "ordered_checks", lambda: [BrokenCheck()])
    df, target = make_clean("binary", n=200)
    result = run_analysis(df, target)

    assert result["analysis_status"] == "failed"
    assert result["checks_completed"] == []
    assert result["check_errors"] == [{"check": "only_broken_test", "error": "RuntimeError: no usable checks"}]
    assert result["score"]["ml_readiness"] == "Unavailable"


def test_every_finding_has_required_fields_and_a_fix():
    from app.benchmark import PLANTS, make_planted
    df, target, _ = make_planted(list(PLANTS))
    for f in run_analysis(df, target)["findings"]:
        for k in ("code", "category", "severity", "confidence", "columns", "evidence", "plain_text", "technical_text", "fixes"):
            assert k in f
        assert f["fixes"] or f["code"] in ("multivariate_outliers", "dimensionality_risk"), f["code"]
