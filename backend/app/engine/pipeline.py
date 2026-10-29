"""Run every applicable check in dependency order, then score. Never silently skips a failure."""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

import app.engine.checks  # noqa: F401  (registers built-in checks)
from app.engine.config import EngineConfig
from app.engine.context import AnalysisContext
from app.engine.findings import Finding
from app.engine.registry import ordered_checks
from app.engine.scoring import SCORE_VERSION, fix_first, score_findings
from app.engine.target import suggest_targets, task_type_for
from app.profiling.profile import profile_dataframe

# Codes that may legitimately name the target column. Everything else is a feature finding.
TARGET_CODES = {"target_missing", "target_constant", "class_imbalance", "rare_classes"}
Progress = Callable[[float, str], None]


def stratified_sample(df: pd.DataFrame, n: int, target: Optional[str], seed: int) -> pd.DataFrame:
    if len(df) <= n:
        return df
    if target and target in df.columns and df[target].nunique() <= 50:
        frac = n / len(df)
        parts = [g.sample(n=max(1, int(round(len(g) * frac))), random_state=seed) for _, g in df.groupby(target, dropna=False)]
        return pd.concat(parts).sort_index()
    return df.sample(n=n, random_state=seed).sort_index()


def merge_findings(findings: List[Finding]) -> List[Finding]:
    """One real problem -> one finding. Same (code, columns) from two checks is merged (keep the stronger)."""
    order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
    best: Dict[tuple, Finding] = {}
    for f in findings:
        k = f.key
        if k not in best or order[f.severity] < order[best[k].severity]:
            best[k] = f
    return list(best.values())


def run_analysis(df: pd.DataFrame, target: Optional[str] = None, task_type: Optional[str] = None,
                 profile: Optional[Dict[str, Any]] = None, config: Optional[EngineConfig] = None,
                 test_df: Optional[pd.DataFrame] = None, progress: Optional[Progress] = None) -> Dict[str, Any]:
    cfg = config or EngineConfig()
    t_start = time.perf_counter()
    n_original = len(df)
    sampled = n_original > cfg.sample_threshold
    work = stratified_sample(df, cfg.sample_threshold, target, cfg.seed) if sampled else df
    prof = profile if (profile is not None and not sampled) else profile_dataframe(work)
    candidates = suggest_targets(work, prof)
    if target is not None and target not in work.columns:
        raise ValueError(f"Target column '{target}' not found.")
    if target is not None and task_type is None:
        task_type = task_type_for(work[target], next(c["semantic_type"] for c in prof["columns"] if c["name"] == target))
        if task_type is None:
            raise ValueError(f"Column '{target}' cannot be used as a target (type unsuitable).")
    ctx = AnalysisContext(df=work, target=target, task_type=task_type if target else None, profile=prof, config=cfg,
                          n_rows_original=n_original, sampled=sampled, test_df=test_df)

    checks = [c for c in ordered_checks()]
    findings: List[Finding] = []
    timings: Dict[str, int] = {}
    errors: List[Dict[str, str]] = []
    skipped: List[str] = []
    for i, check in enumerate(checks):
        if progress:
            progress(i / len(checks), f"Running {check.id}")
        if not check.applies_to(ctx):
            skipped.append(check.id)
            continue
        t0 = time.perf_counter()
        try:
            res = check.run(work, ctx)
            for f in res:
                f.check_id = check.id
            findings.extend(res)
        except Exception as e:   # one broken check must not hide the rest; the failure is reported
            errors.append({"check": check.id, "error": f"{type(e).__name__}: {e}"})
        timings[check.id] = round((time.perf_counter() - t0) * 1000)

    # Guard: a feature finding must never be raised on the target column.
    kept = []
    for f in findings:
        if target in f.columns and f.code not in TARGET_CODES:
            errors.append({"check": f.check_id, "error": f"dropped '{f.code}' raised on target column"})
            continue
        kept.append(f)
    findings = merge_findings(kept)
    findings.sort(key=lambda f: (["critical", "high", "medium", "low", "info"].index(f.severity), f.code, f.columns))
    provisional = target is None
    score = score_findings(findings, provisional=provisional)
    top = fix_first(findings)
    summary = build_summary(work, ctx, findings, score, top, candidates)
    if progress:
        progress(1.0, "Done")
    return {
        "target_column": target, "task_type": ctx.task_type, "target_candidates": candidates,
        "findings": [f.to_dict() for f in findings], "score": score, "fix_first": top, "summary": summary,
        "leakage_columns": ctx.shared.get("leakage_likely", []),
        "check_timings_ms": timings, "check_errors": errors, "checks_skipped": skipped,
        "sampling": {"sampled": sampled, "n_rows_original": n_original, "n_rows_analysed": len(work)},
        "profile": prof, "duration_ms": round((time.perf_counter() - t_start) * 1000), "score_version": SCORE_VERSION,
        "config_snapshot": cfg.snapshot(),
    }


def build_summary(df, ctx: AnalysisContext, findings: List[Finding], score: Dict[str, Any], top, candidates) -> Dict[str, Any]:
    counts = {s: sum(1 for f in findings if f.severity == s) for s in ("critical", "high", "medium", "low", "info")}
    headlines = {"Excellent": "Your data looks ready for machine learning.",
                 "Good": "Your data is mostly ready; a few fixes will help.",
                 "Fair": "Your data needs work before modelling.",
                 "Poor": "Your data is not ready for machine learning yet."}
    if score.get("unresolved_critical", 0) > 0:
        headline = f"Not ready: {score['unresolved_critical']} critical issue(s) must be resolved before modeling."
    elif score.get("unresolved_high", 0) > 0:
        headline = f"Requires review: {score['unresolved_high']} high-severity issue(s) need attention."
    else:
        headline = headlines[score["band"]]
    notes = []
    if score["provisional"]:
        notes.append("Select a target to unlock imbalance, leakage and modeling checks. This score is provisional.")
    if ctx.sampled:
        notes.append(f"Analysed a stratified sample of {len(df)} of {ctx.n_rows_original} rows; percentages refer to the sample.")
    return {
        "headline": headline,
        "score": score["value"],
        "data_quality_score": score.get("data_quality_score", score["value"]),
        "ml_readiness": score.get("ml_readiness", "Requires Review"),
        "band": score["band"],
        "provisional": score["provisional"],
        "counts": counts,
        "notes": notes,
        "unresolved_critical": score.get("unresolved_critical", 0),
        "unresolved_high": score.get("unresolved_high", 0),
        "validation_checks": score.get("validation_checks", {}),
        "top_fixes": [t["headline"] for t in top]
    }
