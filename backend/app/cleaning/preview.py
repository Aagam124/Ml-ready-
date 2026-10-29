"""Preview a step on a sample: before/after rows, rows and columns affected. Never touches stored data."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from app.cleaning.executor import ordered_enabled, run_plan

PREVIEW_ROWS = 2000


def _changed_cells(before: pd.DataFrame, after: pd.DataFrame, columns: List[str]) -> int:
    idx = before.index.intersection(after.index)
    n = 0
    for c in columns:
        if c in before.columns and c in after.columns:
            a, b = before.loc[idx, c], after.loc[idx, c]
            same = (a == b) | (a.isna() & b.isna())
            n += int((~same.fillna(False)).sum())
    return n


def _jsonable_frame(df: pd.DataFrame, n: int = 8) -> List[Dict[str, Any]]:
    out = df.head(n).astype(object).where(df.head(n).notna(), None)
    return [{str(k): (v if isinstance(v, (int, float, str, bool)) or v is None else str(v)) for k, v in row.items()}
            for row in out.to_dict(orient="records")]


def preview_step(df: pd.DataFrame, steps: List[Dict[str, Any]], step_id: str, target: Optional[str], task_type: Optional[str],
                 seed: int = 42) -> Dict[str, Any]:
    """Effect of ONE step in the context of the steps that precede it in execution order."""
    sample = df if len(df) <= PREVIEW_ROWS else df.sample(PREVIEW_ROWS, random_state=seed).sort_index()
    live = ordered_enabled([dict(s, enabled=True) if s["step_id"] == step_id else s for s in steps])
    ids = [s["step_id"] for s in live]
    if step_id not in ids:
        raise KeyError(step_id)
    upto = live[: ids.index(step_id) + 1]
    prior = live[: ids.index(step_id)]
    before, _, _ = run_plan(sample, prior, target, task_type, seed)
    after, _, rep = run_plan(sample, upto, target, task_type, seed)
    step = next(s for s in live if s["step_id"] == step_id)
    cols = sorted(set(step.get("columns", [])) | set(rep["steps"][-1]["columns_added"]) | set(rep["steps"][-1]["columns_removed"]))
    shown = [c for c in cols if c in before.columns or c in after.columns]
    rows_removed = len(before) - len(after)
    cells = _changed_cells(before, after, [c for c in shown if c in before.columns and c in after.columns])
    sample_idx = after.index[:8]
    if shown:     # show rows that actually changed first (more useful than the first rows of the file)
        common = before.index.intersection(after.index)
        both = [c for c in shown if c in before.columns and c in after.columns]
        if both:
            a, b = before.loc[common, both], after.loc[common, both]
            diff = ~(((a == b) | (a.isna() & b.isna())).all(axis=1).fillna(False))
            if diff.any():
                sample_idx = common[diff.to_numpy()][:8]
    return {
        "step_id": step_id, "op": step["op"], "sampled": len(df) > PREVIEW_ROWS, "n_rows_sampled": len(sample),
        "rows_before": len(before), "rows_after": len(after), "rows_removed": rows_removed,
        "cells_changed": cells, "columns_affected": shown,
        "columns_added": rep["steps"][-1]["columns_added"], "columns_removed": rep["steps"][-1]["columns_removed"],
        "before_sample": _jsonable_frame(before.loc[before.index.intersection(sample_idx)][[c for c in shown if c in before.columns]]) if shown else [],
        "after_sample": _jsonable_frame(after.loc[sample_idx][[c for c in shown if c in after.columns]]) if shown else [],
        "denominator": f"{len(sample)} sampled rows" if len(df) > PREVIEW_ROWS else f"{len(df)} rows",
    }
