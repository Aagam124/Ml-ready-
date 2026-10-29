"""Honest modeling: leakage is called out, improvements are measured on identical folds, no-change is reported honestly."""
import numpy as np
import pandas as pd

from app.benchmark import make_clean, make_planted
from app.cleaning.executor import plan_options, run_plan
from app.cleaning.plan import propose_plan
from app.engine.pipeline import run_analysis
from app.modeling.runner import run_before_after


def _run(df, target, task="binary", extra_steps=None, enable=None):
    r = run_analysis(df, target)
    plan = propose_plan(r["findings"], r["profile"], target)
    steps = plan["steps"] + (extra_steps or [])
    after, _, _ = run_plan(df, steps, target, task)
    return r, run_before_after(df, after, target, task, plan_options(steps), r["leakage_columns"])


def test_planted_leak_dataset_is_reported_as_leaky_not_as_success():
    df, target, _ = make_planted(["leakage"], n=1500)
    r, res = _run(df, target)
    cmp = res["comparison"]
    assert cmp["leakage_inflated"] and cmp["verdict"] == "leakage_removed"
    assert "post_outcome_score" in cmp["leakage_columns_excluded"]
    before = max(m["cv"]["roc_auc"]["mean"] for m in res["arms"]["before"]["models"])
    honest = max(m["cv"]["roc_auc"]["mean"] for m in res["arms"]["before_without_leakage"]["models"])
    assert before > 0.97 and honest < 0.9
    assert any("not real" in h or "inflated" in n for h, n in [(cmp["headline"], " ".join(cmp["notes"]))])
    assert "post_outcome_score" not in res["arms"]["after"]["features_used"]


def test_clean_data_reports_no_meaningful_improvement():
    df, target = make_clean("binary", n=1500, seed=11)
    _, res = _run(df, target)
    assert res["comparison"]["verdict"] == "no_meaningful_change"
    assert "No meaningful improvement" in res["comparison"]["headline"]


def test_same_folds_and_test_rows_for_every_arm_and_ci_present():
    df, target, _ = make_planted(["leakage", "duplicates", "negative_salary"], n=1200)
    _, res = _run(df, target)
    arms = res["arms"]
    assert arms["before"]["n_test"] == arms["after"]["n_test"] == res["common_test_rows"]
    for arm in arms.values():
        for m in arm["models"]:
            cv = m["cv"]["roc_auc"]
            assert len(cv["folds"]) == 5 and cv["ci_low"] <= cv["mean"] <= cv["ci_high"]   # equal when every fold scores identically (e.g. a perfect leak)
            t = m["test"]["roc_auc"]
            assert t["ci_low"] <= t["value"] <= t["ci_high"]
    pair = res["comparison"]["per_model"][0]["paired"]
    assert pair["n_folds"] == 5


def test_modeling_is_deterministic():
    df, target = make_clean("binary", n=800, seed=3)
    a = run_before_after(df, df, target, "binary", {}, [], seed=7)
    b = run_before_after(df, df, target, "binary", {}, [], seed=7)
    assert a["arms"]["before"]["models"] == b["arms"]["before"]["models"]


def test_regression_and_multiclass_metrics():
    df, target = make_clean("regression", n=900, seed=2)
    res = run_before_after(df, df, target, "regression", {}, [])
    m = res["arms"]["before"]["models"][0]["cv"]
    assert set(m) == {"mae", "rmse", "r2"} and res["comparison"]["primary_metric"] == "r2"
    df, target = make_clean("multiclass", n=900, seed=2)
    res = run_before_after(df, df, target, "multiclass", {}, [])
    assert {"accuracy", "balanced_accuracy", "f1", "roc_auc", "pr_auc"} <= set(res["arms"]["before"]["models"][0]["cv"])


def test_real_improvement_is_detected_with_paired_folds():
    rng = np.random.default_rng(0)
    n = 1500
    x = rng.normal(size=n)
    y = (x + rng.normal(0, 0.5, n) > 0).astype(int)
    broken = x.copy()
    broken[rng.random(n) < 0.35] = 9999.0            # corrupted sentinel values hide the signal
    before = pd.DataFrame({"x": broken, "z": rng.normal(size=n), "y": y})
    after = before.copy()
    after["x"] = x                                    # the "fix"
    res = run_before_after(before, after, "y", "binary", {}, [])
    assert res["comparison"]["verdict"] == "improved"


def test_class_weight_option_is_used():
    df, target, _ = make_planted(["imbalance"], n=1500)
    res = run_before_after(df, df, target, "binary", {"class_weight": "balanced", "primary_metric": "pr_auc"}, [])
    assert res["arms"]["before"]["models"][0]["params"]["class_weight"] == "balanced"
    assert res["comparison"]["primary_metric"] == "pr_auc"
