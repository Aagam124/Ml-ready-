"""Reusable enterprise quality rules engine.

Allows teams to define declarative constraints and quality gates on datasets.
Supported rules:
- not_null: column must not have missing values
- numeric_range: column values must be between min_val and max_val
- unique: values in column must be unique (no duplicates)
- allowed_values: categorical values must belong to an approved set
- regex_match: string values must match a specified pattern
- min_row_count: dataset must meet a minimum row count threshold
- date_interval: date values must fall between min_date and max_date
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd


@dataclass
class QualityRule:
    id: str
    name: str
    rule_type: str  # not_null, numeric_range, unique, allowed_values, regex_match, min_row_count, date_interval
    column: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    severity: str = "high"  # critical, high, medium, low
    description: str = ""
    enabled: bool = True
    version: str = "1.0"


@dataclass
class RuleEvaluationResult:
    rule_id: str
    name: str
    rule_type: str
    column: Optional[str]
    severity: str
    status: str  # "passed" | "failed" | "skipped" | "error"
    total_checked: int
    failure_count: int
    failure_pct: float
    sample_failures: List[Any]
    message: str
    evaluated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class QualityRulesEngine:
    def __init__(self, rules: Optional[List[QualityRule]] = None):
        self.rules: List[QualityRule] = rules or []

    def add_rule(self, rule: QualityRule) -> None:
        self.rules.append(rule)

    def evaluate(self, df: pd.DataFrame) -> List[RuleEvaluationResult]:
        results: List[RuleEvaluationResult] = []
        n_rows = len(df)

        for rule in self.rules:
            if not rule.enabled:
                continue

            try:
                res = self._evaluate_rule(rule, df, n_rows)
            except Exception as e:
                res = RuleEvaluationResult(
                    rule_id=rule.id,
                    name=rule.name,
                    rule_type=rule.rule_type,
                    column=rule.column,
                    severity=rule.severity,
                    status="error",
                    total_checked=n_rows,
                    failure_count=0,
                    failure_pct=0.0,
                    sample_failures=[],
                    message=f"Rule evaluation error: {str(e)}",
                )
            results.append(res)
        return results

    def _evaluate_rule(self, rule: QualityRule, df: pd.DataFrame, n_rows: int) -> RuleEvaluationResult:
        col = rule.column
        rtype = rule.rule_type
        params = rule.params

        # Dataset-level checks
        if rtype == "min_row_count":
            min_rows = int(params.get("min_rows", 1))
            passed = n_rows >= min_rows
            return RuleEvaluationResult(
                rule_id=rule.id,
                name=rule.name,
                rule_type=rtype,
                column=None,
                severity=rule.severity,
                status="passed" if passed else "failed",
                total_checked=n_rows,
                failure_count=0 if passed else (min_rows - n_rows),
                failure_pct=0.0 if passed else round(max(0.0, (min_rows - n_rows) / min_rows * 100), 2),
                sample_failures=[],
                message=f"Dataset has {n_rows:,} rows (required at least {min_rows:,})."
                if passed
                else f"Dataset has only {n_rows:,} rows, failing the minimum requirement of {min_rows:,}.",
            )

        if col is None or col not in df.columns:
            return RuleEvaluationResult(
                rule_id=rule.id,
                name=rule.name,
                rule_type=rtype,
                column=col,
                severity=rule.severity,
                status="skipped",
                total_checked=n_rows,
                failure_count=0,
                failure_pct=0.0,
                sample_failures=[],
                message=f"Column '{col}' not found in dataset.",
            )

        series = df[col]

        # 1. not_null
        if rtype == "not_null":
            null_mask = series.isna()
            fail_count = int(null_mask.sum())
            pct = round((fail_count / n_rows) * 100, 2) if n_rows else 0.0
            return RuleEvaluationResult(
                rule_id=rule.id,
                name=rule.name,
                rule_type=rtype,
                column=col,
                severity=rule.severity,
                status="passed" if fail_count == 0 else "failed",
                total_checked=n_rows,
                failure_count=fail_count,
                failure_pct=pct,
                sample_failures=[None] if fail_count > 0 else [],
                message=f"Column '{col}' has no missing values."
                if fail_count == 0
                else f"Column '{col}' contains {fail_count:,} missing values ({pct}%).",
            )

        # 2. unique
        if rtype == "unique":
            non_null = series.dropna()
            dups_mask = non_null.duplicated(keep="first")
            fail_count = int(dups_mask.sum())
            sample = list(non_null[dups_mask].iloc[:5].to_numpy())
            pct = round((fail_count / len(non_null)) * 100, 2) if len(non_null) else 0.0
            return RuleEvaluationResult(
                rule_id=rule.id,
                name=rule.name,
                rule_type=rtype,
                column=col,
                severity=rule.severity,
                status="passed" if fail_count == 0 else "failed",
                total_checked=len(non_null),
                failure_count=fail_count,
                failure_pct=pct,
                sample_failures=sample,
                message=f"Values in '{col}' are unique."
                if fail_count == 0
                else f"Column '{col}' contains {fail_count:,} duplicate values ({pct}%).",
            )

        # 3. numeric_range
        if rtype == "numeric_range":
            num_s = pd.to_numeric(series, errors="coerce")
            valid = num_s.dropna()
            min_v = params.get("min")
            max_v = params.get("max")
            mask = pd.Series(False, index=valid.index)
            if min_v is not None:
                mask |= valid < float(min_v)
            if max_v is not None:
                mask |= valid > float(max_v)
            fail_count = int(mask.sum())
            sample = list(valid[mask].iloc[:5].to_numpy())
            pct = round((fail_count / len(valid)) * 100, 2) if len(valid) else 0.0
            bound_desc = f"[{min_v if min_v is not None else '-inf'}, {max_v if max_v is not None else '+inf'}]"
            return RuleEvaluationResult(
                rule_id=rule.id,
                name=rule.name,
                rule_type=rtype,
                column=col,
                severity=rule.severity,
                status="passed" if fail_count == 0 else "failed",
                total_checked=len(valid),
                failure_count=fail_count,
                failure_pct=pct,
                sample_failures=sample,
                message=f"All values in '{col}' fall within expected range {bound_desc}."
                if fail_count == 0
                else f"Column '{col}' has {fail_count:,} values ({pct}%) outside expected range {bound_desc}.",
            )

        # 4. allowed_values
        if rtype == "allowed_values":
            allowed = set(params.get("allowed", []))
            non_null = series.dropna().astype(str).str.strip()
            mask = ~non_null.isin(allowed)
            fail_count = int(mask.sum())
            sample = list(non_null[mask].unique()[:5])
            pct = round((fail_count / len(non_null)) * 100, 2) if len(non_null) else 0.0
            return RuleEvaluationResult(
                rule_id=rule.id,
                name=rule.name,
                rule_type=rtype,
                column=col,
                severity=rule.severity,
                status="passed" if fail_count == 0 else "failed",
                total_checked=len(non_null),
                failure_count=fail_count,
                failure_pct=pct,
                sample_failures=sample,
                message=f"All values in '{col}' belong to allowed set."
                if fail_count == 0
                else f"Column '{col}' contains {fail_count:,} unexpected values ({pct}%): {sample[:3]}...",
            )

        # 5. regex_match
        if rtype == "regex_match":
            pattern = str(params.get("pattern", ".*"))
            regex = re.compile(pattern)
            non_null = series.dropna().astype(str)
            mask = ~non_null.apply(lambda s: bool(regex.search(s)))
            fail_count = int(mask.sum())
            sample = list(non_null[mask].iloc[:5].to_numpy())
            pct = round((fail_count / len(non_null)) * 100, 2) if len(non_null) else 0.0
            return RuleEvaluationResult(
                rule_id=rule.id,
                name=rule.name,
                rule_type=rtype,
                column=col,
                severity=rule.severity,
                status="passed" if fail_count == 0 else "failed",
                total_checked=len(non_null),
                failure_count=fail_count,
                failure_pct=pct,
                sample_failures=sample,
                message=f"All non-null values in '{col}' match pattern '{pattern}'."
                if fail_count == 0
                else f"Column '{col}' has {fail_count:,} values ({pct}%) that do not match pattern '{pattern}'.",
            )

        # 6. date_interval
        if rtype == "date_interval":
            dates = pd.to_datetime(series, errors="coerce")
            valid = dates.dropna()
            min_d = pd.to_datetime(params["min_date"]) if params.get("min_date") else None
            max_d = pd.to_datetime(params["max_date"]) if params.get("max_date") else None
            mask = pd.Series(False, index=valid.index)
            if min_d is not None:
                mask |= valid < min_d
            if max_d is not None:
                mask |= valid > max_d
            fail_count = int(mask.sum())
            sample = [str(x) for x in valid[mask].iloc[:5]]
            pct = round((fail_count / len(valid)) * 100, 2) if len(valid) else 0.0
            return RuleEvaluationResult(
                rule_id=rule.id,
                name=rule.name,
                rule_type=rtype,
                column=col,
                severity=rule.severity,
                status="passed" if fail_count == 0 else "failed",
                total_checked=len(valid),
                failure_count=fail_count,
                failure_pct=pct,
                sample_failures=sample,
                message=f"All dates in '{col}' fall within allowed interval."
                if fail_count == 0
                else f"Column '{col}' has {fail_count:,} dates ({pct}%) outside allowed interval.",
            )

        return RuleEvaluationResult(
            rule_id=rule.id,
            name=rule.name,
            rule_type=rtype,
            column=col,
            severity=rule.severity,
            status="skipped",
            total_checked=n_rows,
            failure_count=0,
            failure_pct=0.0,
            sample_failures=[],
            message=f"Unknown rule type '{rtype}'.",
        )
