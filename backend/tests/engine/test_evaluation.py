"""Tests for the Precision / Recall (>=90%) Evaluation Framework."""

from app.engine.evaluation import EvaluationSuite, TaskMetric


def test_task_metric_calculations():
    # Test perfect detection
    m = TaskMetric(task="test_leak", true_positives=10, false_positives=0, false_negatives=0, true_negatives=10)
    assert m.precision == 1.0
    assert m.recall == 1.0
    assert m.f1_score == 1.0
    assert m.accuracy == 1.0
    assert m.as_dict()["meets_target_90pct"] is True

    # Test edge case: zero denominator
    empty = TaskMetric(task="empty")
    assert empty.precision == 0.0
    assert empty.recall == 0.0
    assert empty.f1_score == 0.0

    # Test realistic metrics
    m2 = TaskMetric(task="test_quality", true_positives=95, false_positives=5, false_negatives=5, true_negatives=95)
    assert m2.precision == 0.95
    assert m2.recall == 0.95
    assert m2.f1_score == 0.95
    assert m2.as_dict()["meets_target_90pct"] is True


def test_evaluation_suite_run_and_accuracy():
    suite = EvaluationSuite(seeds=3, sample_size=800)
    res = suite.run_evaluations()

    overall = res["overall"]
    assert overall["total_evaluations"] > 0
    assert overall["precision"] >= 0.90, f"Precision below 90%: {overall['precision']}"
    assert overall["recall"] >= 0.90, f"Recall below 90%: {overall['recall']}"
    assert overall["f1"] >= 0.90, f"F1 below 90%: {overall['f1']}"
    assert overall["meets_target_90pct"] is True
