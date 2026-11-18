import copy

import pandas as pd
import pytest

import app.engine.evaluation as evaluation
from app.benchmark import (
    PLANTS,
    PLANT_FINDINGS,
    make_clean,
    make_planted,
    planted_defect_present,
)
from app.engine.evaluation import (
    EvaluationSuite,
    TaskMetric,
    finding_matches_plant,
    maximum_matching,
    severity_counts,
)
from app.engine.pipeline import run_analysis


def test_task_metric_calculations_and_zero_denominators():
    metric = TaskMetric(task="test", true_positives=10, false_positives=0, false_negatives=0, true_negatives=10)
    assert metric.precision == metric.recall == metric.f1_score == metric.accuracy == 1.0
    assert metric.false_positive_rate == 0.0

    empty = TaskMetric(task="empty")
    assert empty.precision == empty.recall == empty.f1_score == empty.accuracy == 0.0
    assert empty.specificity == empty.false_positive_rate == 0.0
    empty_metrics = empty.as_dict()
    assert not empty_metrics["precision_defined"]
    assert not empty_metrics["recall_defined"]
    assert not empty_metrics["false_positive_rate_defined"]
    assert not empty_metrics["accuracy_defined"]

    metric = TaskMetric(task="quality", true_positives=95, false_positives=5, false_negatives=5, true_negatives=95)
    assert metric.precision == metric.recall == metric.f1_score == metric.accuracy == 0.95
    assert metric.false_positive_rate == 0.05


def test_finding_match_requires_correct_code_column_and_relevant_evidence():
    df, target, expected = make_planted(["negative_salary"], seed=4, n=800)
    finding = next(
        finding for finding in run_analysis(df, target)["findings"]
        if finding["code"] == "invalid_values" and "salary" in finding["columns"]
    )
    assert finding_matches_plant("negative_salary", expected["negative_salary"], finding, df, target)

    wrong_code = copy.deepcopy(finding)
    wrong_code["code"] = "outliers"
    assert not finding_matches_plant("negative_salary", expected["negative_salary"], wrong_code, df, target)

    wrong_column = copy.deepcopy(finding)
    wrong_column["columns"] = ["age"]
    assert not finding_matches_plant("negative_salary", expected["negative_salary"], wrong_column, df, target)

    unrelated_evidence = copy.deepcopy(finding)
    unrelated_evidence["evidence"]["examples"] = [12.0, 50.0]
    assert not finding_matches_plant("negative_salary", expected["negative_salary"], unrelated_evidence, df, target)


@pytest.mark.parametrize(
    ("plant", "valid_example"),
    [("negative_salary", 1.0), ("impossible_age", 40.0)],
)
def test_numeric_planted_finding_requires_all_examples_and_count_to_match(
    plant, valid_example
):
    df, target, expected = make_planted([plant], seed=4, n=800)
    code, column = expected[plant]
    finding = next(
        finding for finding in run_analysis(df, target)["findings"]
        if finding["code"] == code and column in finding["columns"]
    )
    match = lambda candidate: finding_matches_plant(plant, expected[plant], candidate, df, target)

    assert match(finding)

    fabricated_example = copy.deepcopy(finding)
    fabricated_example["evidence"]["examples"].append(valid_example)
    assert not match(fabricated_example)

    incorrect_count = copy.deepcopy(finding)
    incorrect_count["evidence"]["n_invalid"] += 1
    assert not match(incorrect_count)

    empty_examples = copy.deepcopy(finding)
    empty_examples["evidence"]["examples"] = []
    assert not match(empty_examples)

    malformed_examples = copy.deepcopy(finding)
    malformed_examples["evidence"]["examples"] = ["not-a-number"]
    assert not match(malformed_examples)

    malformed_examples["evidence"]["examples"] = "not-a-list"
    assert not match(malformed_examples)


@pytest.mark.parametrize(
    "plant",
    ["negative_salary", "impossible_age"],
)
@pytest.mark.parametrize(
    "malformed_value",
    [True, float("nan"), float("inf"), float("-inf")],
    ids=["boolean", "nan", "positive-infinity", "negative-infinity"],
)
def test_numeric_planted_finding_rejects_non_finite_or_boolean_examples(
    plant, malformed_value
):
    df, target, expected = make_planted([plant], seed=4, n=800)
    code, column = expected[plant]
    finding = next(
        finding for finding in run_analysis(df, target)["findings"]
        if finding["code"] == code and column in finding["columns"]
    )
    match = lambda candidate: finding_matches_plant(plant, expected[plant], candidate, df, target)

    mixed_evidence = copy.deepcopy(finding)
    mixed_evidence["evidence"]["examples"].append(malformed_value)
    assert not match(mixed_evidence)


def test_matching_is_one_to_one_for_duplicate_findings_and_overlapping_labels():
    df, target, expected = make_planted(["negative_salary"], seed=2, n=800)
    finding = next(
        finding for finding in run_analysis(df, target)["findings"]
        if finding["code"] == "invalid_values" and "salary" in finding["columns"]
    )
    duplicates = [finding, copy.deepcopy(finding)]
    assigned = maximum_matching(
        ["negative_salary"],
        duplicates,
        lambda plant, item: finding_matches_plant(plant, expected[plant], item, df, target),
    )
    assert len(assigned) == 1
    assert len(set(assigned.values())) == 1

    overlaps = maximum_matching(["first_defect", "second_defect"], [{}], lambda _label, _item: True)
    assert len(overlaps) == 1
    assert list(overlaps.values()) == [0]


def test_clean_dataset_findings_are_separated_by_severity():
    findings = [{"severity": severity} for severity in ("critical", "high", "medium", "low", "info")]
    assert severity_counts(findings) == {
        "critical": 1, "high": 1, "medium": 1, "low": 1, "info": 1,
    }


def test_generator_reproducibility_skew_and_planted_ground_truth():
    for kind in ("binary", "multiclass", "regression"):
        first, target = make_clean(kind, seed=13, n=1200)
        second, second_target = make_clean(kind, seed=13, n=1200)
        pd.testing.assert_frame_equal(first, second)
        assert target == second_target
        assert first["income"].skew() > 1
        assert first["monthly_usage"].skew() > 0.5
        assert first["age"].between(18, 80).all()
        assert first["customer_id"].is_unique
        assert not first.duplicated().any()
        assert not {"salary", "gender", "post_outcome_score", "tenure_months", "country"}.intersection(first.columns)
        assert not any(planted_defect_present(plant, first, target) for plant in PLANTS)

    for plant in PLANTS:
        planted, target, expected = make_planted([plant], seed=13, n=1200)
        repeated, repeated_target, repeated_expected = make_planted([plant], seed=13, n=1200)
        pd.testing.assert_frame_equal(planted, repeated)
        assert target == repeated_target and expected == repeated_expected
        assert planted_defect_present(plant, planted, target)
        assert expected[plant] == PLANT_FINDINGS[plant] or (
            plant == "imbalance" and expected[plant] == ("class_imbalance", target)
        )


def test_missing_planted_defect_is_detected_before_evaluation(monkeypatch):
    clean, target = make_clean("binary", seed=1, n=800)
    assert not planted_defect_present("leakage", clean, target)
    monkeypatch.setattr(
        evaluation, "make_planted",
        lambda *_args, **_kwargs: (clean, target, {"leakage": PLANT_FINDINGS["leakage"]}),
    )
    with pytest.raises(ValueError, match="is absent"):
        EvaluationSuite(seeds=1, sample_size=800).run_evaluations()


def test_clean_findings_report_actionable_false_positives_and_observations(monkeypatch):
    clean, target = make_clean("binary", seed=1, n=800)
    def fake_make_planted(plants, seed, n):
        return make_planted(plants, seed=seed, n=n)

    def fake_analysis(_df, _target):
        return {"findings": [
            {"code": "unrelated_medium", "columns": [], "severity": "medium", "evidence": {}},
            {"code": "harmless_low", "columns": [], "severity": "low", "evidence": {}},
            {"code": "harmless_info", "columns": [], "severity": "info", "evidence": {}},
        ]}

    monkeypatch.setattr(evaluation, "make_planted", fake_make_planted)
    monkeypatch.setattr(evaluation, "run_analysis", fake_analysis)
    result = EvaluationSuite(seeds=1, sample_size=800).run_evaluations()
    assert result["clean_false_positives"]["actionable_total"] == 3
    assert result["clean_false_positives"]["medium"] == 3
    assert result["clean_false_positives"]["observations_by_severity"] == {"low": 3, "info": 3}
    assert result["overall"]["clean_dataset_false_positive_rate"] == 1.0
    assert result["overall"]["unmatched_actionable_findings_fp"] == 13


def test_evaluation_reports_per_defect_and_dataset_metrics_as_limited_synthetic_evidence():
    result = EvaluationSuite(seeds=1, sample_size=800).run_evaluations()
    assert result["methodology"]["scope"] == "synthetic_benchmark_only"
    assert result["methodology"]["real_world_accuracy_claim"] is False
    assert result["methodology"]["small_sample"] is True
    assert set(result["tasks"]) == set(PLANTS)
    assert len(result["datasets"]) == len(PLANTS) + 3
    assert all("exact_match" in dataset for dataset in result["datasets"])
    assert all("false_positive_rate" in task for task in result["tasks"].values())
    assert result["overall"]["precision_defined"] is True
    assert result["overall"]["recall_defined"] is True
