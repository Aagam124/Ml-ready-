"""Synthetic evaluation of MLReady findings against planted and clean datasets."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd

from app.benchmark import PLANTS, PLANT_FINDINGS, make_clean, make_planted, planted_defect_present
from app.engine.pipeline import run_analysis

SEVERITIES = ("critical", "high", "medium", "low", "info")
ACTIONABLE_SEVERITIES = frozenset(("critical", "high", "medium"))
OBSERVATION_SEVERITIES = frozenset(("low", "info"))


@dataclass
class TaskMetric:
    task: str
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    true_negatives: int = 0
    total_evaluations: int = 0

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return round(self.true_positives / denom, 4) if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return round(self.true_positives / denom, 4) if denom else 0.0

    @property
    def specificity(self) -> float:
        denom = self.true_negatives + self.false_positives
        return round(self.true_negatives / denom, 4) if denom else 0.0

    @property
    def false_positive_rate(self) -> float:
        denom = self.true_negatives + self.false_positives
        return round(self.false_positives / denom, 4) if denom else 0.0

    @property
    def f1_score(self) -> float:
        p, r = self.precision, self.recall
        return round(2 * p * r / (p + r), 4) if p + r else 0.0

    @property
    def accuracy(self) -> float:
        total = self.true_positives + self.true_negatives + self.false_positives + self.false_negatives
        return round((self.true_positives + self.true_negatives) / total, 4) if total else 0.0

    def as_dict(self) -> Dict[str, Any]:
        precision_denominator = self.true_positives + self.false_positives
        recall_denominator = self.true_positives + self.false_negatives
        rate_denominator = self.true_negatives + self.false_positives
        accuracy_denominator = self.true_positives + self.true_negatives + self.false_positives + self.false_negatives
        return {
            "task": self.task,
            "tp": self.true_positives,
            "fp": self.false_positives,
            "fp_definition": "Per-defect candidate false positives on clean datasets.",
            "fn": self.false_negatives,
            "tn": self.true_negatives,
            "precision": self.precision,
            "precision_defined": precision_denominator > 0,
            "precision_denominator": precision_denominator,
            "recall": self.recall,
            "recall_defined": recall_denominator > 0,
            "recall_denominator": recall_denominator,
            "specificity": self.specificity,
            "specificity_defined": rate_denominator > 0,
            "false_positive_rate": self.false_positive_rate,
            "false_positive_rate_defined": rate_denominator > 0,
            "false_positive_rate_denominator": rate_denominator,
            "f1": self.f1_score,
            "accuracy": self.accuracy,
            "accuracy_defined": accuracy_denominator > 0,
            "accuracy_denominator": accuracy_denominator,
            "total_evaluations": self.total_evaluations,
        }


def _claims_plant(plant: str, expected: tuple[str, str | None], finding: Mapping[str, Any]) -> bool:
    code, expected_col = expected
    columns = finding.get("columns", [])
    evidence = finding.get("evidence")
    if finding.get("code") != code or not isinstance(columns, list) or not isinstance(evidence, dict):
        return False
    if expected_col is None:
        if columns:
            return False
    elif expected_col not in columns:
        return False

    if plant == "leakage":
        signals = evidence.get("signals", {})
        return (
            evidence.get("name_only") is False
            and isinstance(signals, dict)
            and any(
                isinstance(signals.get(key), (int, float)) and signals[key] >= threshold
                for key, threshold in (("single_feature_score", 0.8), ("association", 0.5), ("mutual_info_norm", 0.5))
            )
        )
    if plant == "duplicates":
        return isinstance(evidence.get("n_duplicate_rows"), int) and evidence["n_duplicate_rows"] > 0
    if plant == "negative_salary":
        examples = evidence.get("examples", [])
        return evidence.get("rule") == "negative" and any(_number(value) and float(value) < 0 for value in examples)
    if plant == "impossible_age":
        examples = evidence.get("examples", [])
        return evidence.get("rule") == "negative or above 120" and any(
            _number(value) and (float(value) < 0 or float(value) > 120) for value in examples
        )
    if plant == "inconsistent_categories":
        groups = evidence.get("groups", {})
        return isinstance(groups, dict) and any(
            isinstance(values, list) and len(set(values)) > 1 for values in groups.values()
        ) and evidence.get("n_values_to_change", 0) > 0
    if plant == "imbalance":
        counts = evidence.get("class_counts", {})
        return (
            isinstance(counts, dict)
            and len(counts) >= 2
            and min(counts.values()) > 0
            and max(counts.values()) / min(counts.values()) >= 3
            and _number(evidence.get("ratio"))
        )
    if plant == "redundant_columns":
        return evidence.get("exact_copy") is True and expected_col in evidence.get("drop", [])
    if plant == "missingness":
        return isinstance(evidence.get("n_missing"), int) and evidence["n_missing"] > 0
    if plant == "disguised_missing":
        tokens = {str(token).strip().lower() for token in evidence.get("tokens", [])}
        return evidence.get("n_placeholders", 0) > 0 and bool(tokens & {"n/a", "na", "null", "none", "?"})
    if plant == "constant_column":
        return evidence.get("n_nonnull", 0) > 0 and evidence.get("value") == "IN"
    return False


def _number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def _supports_ground_truth(
    plant: str,
    finding: Mapping[str, Any],
    df: pd.DataFrame,
    target: str,
) -> bool:
    evidence = finding["evidence"]
    code, col = PLANT_FINDINGS[plant]
    if not planted_defect_present(plant, df, target):
        return False

    if plant == "leakage":
        return bool(df["post_outcome_score"].corr(pd.to_numeric(df[target], errors="coerce")) >= 0.5)
    if plant == "duplicates":
        return int(df.duplicated(keep="first").sum()) == evidence["n_duplicate_rows"]
    if plant in ("negative_salary", "impossible_age"):
        values = pd.to_numeric(df[col], errors="coerce")
        bad = values < 0 if plant == "negative_salary" else (values < 0) | (values > 120)
        actual = set(values[bad].dropna().astype(float))
        examples = evidence.get("examples")
        invalid_count = evidence.get("n_invalid")
        if (
            not isinstance(examples, list)
            or not examples
            or not all(_number(value) and np.isfinite(value) for value in examples)
            or not isinstance(invalid_count, int)
            or isinstance(invalid_count, bool)
        ):
            return False
        reported_values = {float(value) for value in examples}
        return int(bad.sum()) == invalid_count and reported_values.issubset(actual)
    if plant == "inconsistent_categories":
        actual_values = set(df[col].dropna().astype(str))
        return all(set(values).issubset(actual_values) for values in evidence["groups"].values())
    if plant == "imbalance":
        counts = {str(key): int(value) for key, value in df[target].value_counts().items()}
        observed_ratio = max(counts.values()) / min(counts.values())
        return counts == evidence["class_counts"] and round(observed_ratio, 2) == round(evidence["ratio"], 2)
    if plant == "redundant_columns":
        return bool(np.allclose(df["tenure_months"], df["tenure_years"] * 12))
    if plant == "missingness":
        return int(df[col].isna().sum()) == evidence["n_missing"] and evidence["n_rows"] == len(df)
    if plant == "disguised_missing":
        values = df[col].astype("string").str.strip().str.lower()
        count = int(values.eq("n/a").sum())
        return count == evidence["n_placeholders"] and "n/a" in evidence["tokens"]
    if plant == "constant_column":
        return int(df[col].notna().sum()) == evidence["n_nonnull"] and str(df[col].dropna().iloc[0]) == evidence["value"]
    return False


def finding_matches_plant(
    plant: str,
    expected: tuple[str, str | None],
    finding: Mapping[str, Any],
    df: pd.DataFrame,
    target: str,
) -> bool:
    """Require the finding to claim the right defect and its evidence to agree with the data."""
    expected_matches_label = expected == (
        (PLANT_FINDINGS[plant][0], target) if plant == "imbalance" else PLANT_FINDINGS[plant]
    )
    return expected_matches_label and _claims_plant(plant, expected, finding) and _supports_ground_truth(
        plant, finding, df, target
    )


def maximum_matching(
    labels: Sequence[str],
    findings: Sequence[Mapping[str, Any]],
    matches: Callable[[str, Mapping[str, Any]], bool],
) -> Dict[str, int]:
    """Assign findings to labels one-to-one using maximum bipartite matching."""
    owner_by_finding: Dict[int, str] = {}

    def assign(label: str, visited: set[int]) -> bool:
        for finding_index, finding in enumerate(findings):
            if finding_index in visited or not matches(label, finding):
                continue
            visited.add(finding_index)
            previous = owner_by_finding.get(finding_index)
            if previous is None or assign(previous, visited):
                owner_by_finding[finding_index] = label
                return True
        return False

    for label in labels:
        assign(label, set())
    return {label: index for index, label in owner_by_finding.items()}


def severity_counts(findings: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    return {severity: sum(finding.get("severity") == severity for finding in findings) for severity in SEVERITIES}


class EvaluationSuite:
    """Evaluate synthetic planted defects and clean datasets; not a real-world accuracy estimate."""

    def __init__(self, seeds: int = 10, sample_size: int = 1500):
        if seeds < 1 or sample_size < 1:
            raise ValueError("seeds and sample_size must be positive.")
        self.seeds = seeds
        self.sample_size = sample_size

    def run_evaluations(self) -> Dict[str, Any]:
        metrics = {plant: TaskMetric(task=plant) for plant in PLANTS}
        datasets: List[Dict[str, Any]] = []
        clean_findings_by_severity = {severity: 0 for severity in SEVERITIES}
        clean_actionable_by_severity = {severity: 0 for severity in SEVERITIES}
        clean_observations_by_severity = {severity: 0 for severity in OBSERVATION_SEVERITIES}
        total_tp = total_fp = total_fn = 0
        correct_datasets = clean_dataset_fp_count = 0
        positive_defect_trials = 0

        for seed in range(self.seeds):
            for plant in PLANTS:
                df, target, expected = make_planted([plant], seed=seed, n=self.sample_size)
                if set(expected) != {plant} or expected[plant] != PLANT_FINDINGS[plant]:
                    raise ValueError(f"Ground-truth label mismatch for planted defect '{plant}'.")
                if not planted_defect_present(plant, df, target):
                    raise ValueError(f"Planted defect '{plant}' is absent from generated data for seed {seed}.")

                findings = run_analysis(df, target)["findings"]
                assigned = maximum_matching(
                    list(expected),
                    findings,
                    lambda label, finding, expected=expected, df=df, target=target: finding_matches_plant(
                        label, expected[label], finding, df, target
                    ),
                )
                matched_indices = set(assigned.values())
                missing = sorted(set(expected) - set(assigned))
                actionable_fp = sum(
                    finding.get("severity") in ACTIONABLE_SEVERITIES
                    for index, finding in enumerate(findings)
                    if index not in matched_indices
                )
                false_positive_findings = [
                    {
                        "code": finding.get("code"),
                        "columns": finding.get("columns", []),
                        "severity": finding.get("severity"),
                        "evidence": finding.get("evidence", {}),
                    }
                    for index, finding in enumerate(findings)
                    if index not in matched_indices and finding.get("severity") in ACTIONABLE_SEVERITIES
                ]
                metrics[plant].true_positives += len(assigned)
                metrics[plant].false_negatives += len(missing)
                metrics[plant].total_evaluations += 1
                total_tp += len(assigned)
                total_fn += len(missing)
                total_fp += actionable_fp
                positive_defect_trials += len(expected)
                exact_match = not missing and actionable_fp == 0
                correct_datasets += int(exact_match)
                datasets.append({
                    "dataset": f"planted_{plant}_seed{seed}",
                    "kind": "planted",
                    "seed": seed,
                    "expected_defects": sorted(expected),
                    "detected_defects": sorted(assigned),
                    "false_negatives": missing,
                    "actionable_false_positives": actionable_fp,
                    "actionable_false_positive_findings": false_positive_findings,
                    "false_positives_by_severity": severity_counts([
                        finding for index, finding in enumerate(findings)
                        if index not in matched_indices and finding.get("severity") in ACTIONABLE_SEVERITIES
                    ]),
                    "observations": sum(
                        finding.get("severity") in OBSERVATION_SEVERITIES
                        for index, finding in enumerate(findings) if index not in matched_indices
                    ),
                    "exact_match": exact_match,
                })

            for kind in ("binary", "multiclass", "regression"):
                clean_seed = seed + 500
                df, target = make_clean(kind, seed=clean_seed, n=self.sample_size)
                findings = run_analysis(df, target)["findings"]
                expected_clean = dict(PLANT_FINDINGS)
                expected_clean["imbalance"] = (PLANT_FINDINGS["imbalance"][0], target)
                claimed = maximum_matching(
                    list(expected_clean),
                    findings,
                    lambda label, finding, expected_clean=expected_clean: _claims_plant(
                        label, expected_clean[label], finding
                    ),
                )
                severity_totals = severity_counts(findings)
                for severity, count in severity_totals.items():
                    clean_findings_by_severity[severity] += count
                actionable_fp = sum(finding.get("severity") in ACTIONABLE_SEVERITIES for finding in findings)
                total_fp += actionable_fp
                false_positive_findings = [
                    {
                        "code": finding.get("code"),
                        "columns": finding.get("columns", []),
                        "severity": finding.get("severity"),
                        "evidence": finding.get("evidence", {}),
                    }
                    for finding in findings if finding.get("severity") in ACTIONABLE_SEVERITIES
                ]
                for severity in ACTIONABLE_SEVERITIES:
                    clean_actionable_by_severity[severity] += sum(
                        finding.get("severity") == severity for finding in findings
                    )
                for severity in OBSERVATION_SEVERITIES:
                    clean_observations_by_severity[severity] += severity_totals[severity]

                for plant, metric in metrics.items():
                    metric.total_evaluations += 1
                    match_index = claimed.get(plant)
                    if match_index is not None and findings[match_index].get("severity") in ACTIONABLE_SEVERITIES:
                        metric.false_positives += 1
                    else:
                        metric.true_negatives += 1

                exact_match = actionable_fp == 0
                correct_datasets += int(exact_match)
                clean_dataset_fp_count += int(actionable_fp > 0)
                datasets.append({
                    "dataset": f"clean_{kind}_seed{clean_seed}",
                    "kind": "clean",
                    "seed": clean_seed,
                    "expected_defects": [],
                    "candidate_defects": sorted(
                        plant for plant, index in claimed.items()
                        if findings[index].get("severity") in ACTIONABLE_SEVERITIES
                    ),
                    "actionable_false_positives": actionable_fp,
                    "actionable_false_positive_findings": false_positive_findings,
                    "false_positives_by_severity": {
                        severity: severity_totals[severity] if severity in ACTIONABLE_SEVERITIES else 0
                        for severity in SEVERITIES
                    },
                    "observations_by_severity": {
                        severity: severity_totals[severity] for severity in OBSERVATION_SEVERITIES
                    },
                    "exact_match": exact_match,
                })

        tp_denominator = total_tp + total_fp
        fn_denominator = total_tp + total_fn
        task_values = [metric.as_dict() for metric in metrics.values()]
        macro = {
            "precision": round(sum(item["precision"] for item in task_values) / len(task_values), 4)
            if task_values else 0.0,
            "recall": round(sum(item["recall"] for item in task_values) / len(task_values), 4)
            if task_values else 0.0,
            "f1": round(sum(item["f1"] for item in task_values) / len(task_values), 4)
            if task_values else 0.0,
            "defect_opportunity_false_positive_rate": (
                round(sum(item["false_positive_rate"] for item in task_values) / len(task_values), 4)
                if task_values else 0.0
            ),
        }
        task_fp = sum(metric.false_positives for metric in metrics.values())
        task_tn = sum(metric.true_negatives for metric in metrics.values())
        fpr_denominator = task_fp + task_tn
        clean_dataset_count = self.seeds * 3
        dataset_count = len(datasets)
        small_sample = self.seeds < 30 or self.sample_size < 1000
        precision = round(total_tp / tp_denominator, 4) if tp_denominator else 0.0
        recall = round(total_tp / fn_denominator, 4) if fn_denominator else 0.0

        return {
            "methodology": {
                "scope": "synthetic_benchmark_only",
                "real_world_accuracy_claim": False,
                "small_sample": small_sample,
                "evidence_strength": "limited" if small_sample else "synthetic_only",
                "seeds": self.seeds,
                "sample_size": self.sample_size,
                "positive_defect_trials": positive_defect_trials,
                "clean_dataset_runs": clean_dataset_count,
                "dataset_runs": dataset_count,
                "aggregation": "micro detection metrics; per-defect metrics; exact-match dataset accuracy",
                "metric_definitions": {
                    "micro_precision_fp": "Unmatched actionable findings across planted and clean datasets.",
                    "defect_opportunity_fp": "Per-defect candidate findings on clean datasets, matched one-to-one.",
                    "clean_dataset_false_positive_rate": (
                        "Clean dataset runs with at least one actionable finding divided by clean dataset runs."
                    ),
                },
            },
            "overall": {
                "tp": total_tp,
                "unmatched_actionable_findings_fp": total_fp,
                "actionable_fp_definition": (
                    "Unmatched actionable findings across planted and clean datasets; used in micro precision."
                ),
                "fn": total_fn,
                "precision": precision,
                "precision_defined": tp_denominator > 0,
                "precision_denominator": tp_denominator,
                "recall": recall,
                "recall_defined": fn_denominator > 0,
                "recall_denominator": fn_denominator,
                "f1": round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0,
                "false_positive_rate": (
                    round(clean_dataset_fp_count / clean_dataset_count, 4) if clean_dataset_count else 0.0
                ),
                "false_positive_rate_defined": clean_dataset_count > 0,
                "false_positive_rate_denominator": clean_dataset_count,
                "false_positive_rate_unit": "clean_dataset_runs_with_any_actionable_finding",
                "defect_opportunity_false_positive_rate": (
                    round(task_fp / fpr_denominator, 4) if fpr_denominator else 0.0
                ),
                "defect_opportunity_false_positive_rate_defined": fpr_denominator > 0,
                "defect_opportunity_false_positive_rate_denominator": fpr_denominator,
                "dataset_detection_accuracy": round(correct_datasets / dataset_count, 4) if dataset_count else 0.0,
                "dataset_detection_accuracy_defined": dataset_count > 0,
                "clean_dataset_false_positive_rate": (
                    round(clean_dataset_fp_count / clean_dataset_count, 4) if clean_dataset_count else 0.0
                ),
                "clean_dataset_false_positive_rate_defined": clean_dataset_count > 0,
                "total_evaluations": sum(metric.total_evaluations for metric in metrics.values()),
                "observed_threshold_met": precision >= 0.90 and recall >= 0.90,
                "evidence_strength": "limited" if small_sample else "synthetic_only",
            },
            "macro": macro,
            "clean_false_positives": {
                "definition": "Actionable findings on clean datasets; low and info findings are reported as observations.",
                "by_severity": clean_actionable_by_severity,
                "actionable_total": sum(clean_actionable_by_severity.values()),
                "critical_high": clean_actionable_by_severity["critical"] + clean_actionable_by_severity["high"],
                "medium": clean_actionable_by_severity["medium"],
                "observations_by_severity": clean_observations_by_severity,
                "all_findings_by_severity": clean_findings_by_severity,
            },
            "tasks": {plant: metric.as_dict() for plant, metric in metrics.items()},
            "datasets": datasets,
        }
