"""Empirical evaluation framework for MLReady quality checks and detectors.

Measures Precision, Recall, Specificity, and F1 score against ground-truth labeled evaluation sets
(planted defects vs clean datasets) to verify the target of >= 90% accuracy without fabrication.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from app.benchmark import PLANTS, make_clean, make_planted
from app.engine.pipeline import run_analysis


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
        if denom == 0:
            return 0.0 if self.true_positives == 0 else 1.0
        return round(self.true_positives / denom, 4)

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        if denom == 0:
            return 0.0 if self.true_positives == 0 else 1.0
        return round(self.true_positives / denom, 4)

    @property
    def specificity(self) -> float:
        denom = self.true_negatives + self.false_positives
        return round(self.true_negatives / denom, 4) if denom > 0 else 1.0

    @property
    def f1_score(self) -> float:
        p, r = self.precision, self.recall
        return round(2 * (p * r) / (p + r), 4) if (p + r) > 0 else 0.0

    @property
    def accuracy(self) -> float:
        total = self.true_positives + self.true_negatives + self.false_positives + self.false_negatives
        return round((self.true_positives + self.true_negatives) / total, 4) if total > 0 else 1.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "tp": self.true_positives,
            "fp": self.false_positives,
            "fn": self.false_negatives,
            "tn": self.true_negatives,
            "precision": self.precision,
            "recall": self.recall,
            "specificity": self.specificity,
            "f1": self.f1_score,
            "accuracy": self.accuracy,
            "meets_target_90pct": self.precision >= 0.90 and self.recall >= 0.90,
        }


class EvaluationSuite:
    """Rigorous evaluation suite evaluating detector accuracy on defined ground-truth tasks."""

    def __init__(self, seeds: int = 10, sample_size: int = 1500):
        self.seeds = seeds
        self.sample_size = sample_size
        self.metrics: Dict[str, TaskMetric] = {}

    def _get_metric(self, task: str) -> TaskMetric:
        if task not in self.metrics:
            self.metrics[task] = TaskMetric(task=task)
        return self.metrics[task]

    def run_evaluations(self) -> Dict[str, Any]:
        """Runs detector evaluations over synthetic ground truth and clean datasets across multiple seeds."""
        self.metrics.clear()

        # 1. Evaluate Planted Problems (Recall & True Positives)
        for plant in PLANTS:
            metric = self._get_metric(plant)
            for s in range(self.seeds):
                df, target, expected = make_planted([plant], seed=s, n=self.sample_size)
                res = run_analysis(df, target)
                code, expected_col = expected[plant]
                detected = any(
                    f["code"] == code and (expected_col is None or expected_col in f["columns"])
                    for f in res["findings"]
                )
                if detected:
                    metric.true_positives += 1
                else:
                    metric.false_negatives += 1
                metric.total_evaluations += 1

        # 2. Evaluate Clean Datasets (False Positives & True Negatives)
        for s in range(self.seeds):
            for kind in ("binary", "multiclass", "regression"):
                df, target = make_clean(kind, seed=s + 500, n=self.sample_size)
                res = run_analysis(df, target)
                raised_codes = {f["code"] for f in res["findings"] if f["severity"] in ("critical", "high")}

                # Map plant types to codes
                plant_code_map = {
                    "leakage": "target_leakage",
                    "duplicates": "exact_duplicates",
                    "negative_salary": "invalid_values",
                    "inconsistent_categories": "inconsistent_categories",
                    "redundant_columns": "redundant_columns",
                    "missingness": "missing_values",
                    "impossible_age": "invalid_values",
                    "disguised_missing": "disguised_missing_values",
                    "constant_column": "constant_column",
                    "imbalance": "class_imbalance",
                }

                for plant, expected_code in plant_code_map.items():
                    metric = self._get_metric(plant)
                    if expected_code in raised_codes:
                        metric.false_positives += 1
                    else:
                        metric.true_negatives += 1
                    metric.total_evaluations += 1

        overall_tp = sum(m.true_positives for m in self.metrics.values())
        overall_fp = sum(m.false_positives for m in self.metrics.values())
        overall_fn = sum(m.false_negatives for m in self.metrics.values())
        overall_tn = sum(m.true_negatives for m in self.metrics.values())

        p_denom = overall_tp + overall_fp
        r_denom = overall_tp + overall_fn
        overall_prec = round(overall_tp / p_denom, 4) if p_denom > 0 else 1.0
        overall_rec = round(overall_tp / r_denom, 4) if r_denom > 0 else 1.0
        overall_f1 = (
            round(2 * (overall_prec * overall_rec) / (overall_prec + overall_rec), 4)
            if (overall_prec + overall_rec) > 0
            else 0.0
        )

        return {
            "overall": {
                "precision": overall_prec,
                "recall": overall_rec,
                "f1": overall_f1,
                "total_evaluations": overall_tp + overall_fp + overall_fn + overall_tn,
                "meets_target_90pct": overall_prec >= 0.90 and overall_rec >= 0.90,
            },
            "tasks": {task: m.as_dict() for task, m in self.metrics.items()},
        }
