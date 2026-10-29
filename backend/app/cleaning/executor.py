"""Run a cleaning plan. Stateless steps first, then ONE train/test split, then stateful steps fitted on train only."""
from __future__ import annotations

import io
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from app.cleaning import runtime as rt
from app.cleaning.registry import OPS, impl, phase_of, sort_key


def plan_options(steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge the (enabled) modeling_option steps into one dict: class_weight, split, group_column, ..."""
    opts: Dict[str, Any] = {}
    for st in steps:
        if st.get("enabled", True) and st["op"] == "modeling_option":
            opts.update({k: v for k, v in st.get("params", {}).items() if v is not None})
            if st.get("columns") and "split" in st.get("params", {}):
                opts.setdefault("split_columns", st["columns"])
    return opts


def split_spec(opts: Dict[str, Any], seed: int = 42, test_size: float = 0.2) -> Dict[str, Any]:
    return {"strategy": opts.get("split", "random"), "group_column": opts.get("group_column"),
            "time_column": opts.get("time_column"), "stratify": bool(opts.get("stratify", True)),
            "seed": seed, "test_size": test_size}


def ordered_enabled(steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    live = [(i, s) for i, s in enumerate(steps) if s.get("enabled", True) and OPS[s["op"]]["kind"] != "option"]
    return [s for _, s in sorted(live, key=sort_key)]


def apply_stateless(df: pd.DataFrame, step: Dict[str, Any]) -> pd.DataFrame:
    return impl(OPS[step["op"]]["impl"])(df, step.get("columns", []), **step.get("params", {}))


def make_stateful(step: Dict[str, Any]):
    return impl(OPS[step["op"]]["impl"])(step.get("columns", []), **step.get("params", {}))


def run_plan(df: pd.DataFrame, steps: List[Dict[str, Any]], target: Optional[str] = None, task_type: Optional[str] = None,
             seed: int = 42, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """Returns (cleaned_df, is_train_mask, report). The original frame is never modified; row index is preserved."""
    ordered = ordered_enabled(steps)
    opts = plan_options(steps)
    spec = split_spec(opts, seed, test_size)
    out = df.copy()
    log: List[Dict[str, Any]] = []
    audit_trail: List[Dict[str, Any]] = []
    split_done, is_train = False, None
    total_cells_changed = 0

    for step in ordered:
        kind = OPS[step["op"]]["kind"]
        prev_out = out.copy()
        before_rows, before_cols = len(out), list(out.columns)
        if phase_of(step) >= 2 and not split_done:   # after row ops, BEFORE any column is dropped
            is_train = rt.make_split(out, target, task_type, spec["seed"], spec["test_size"], spec["strategy"],
                                     spec["group_column"], spec["time_column"], spec["stratify"])
            split_done = True
        if kind == "stateless":
            out = apply_stateless(out, step)
        else:
            obj = make_stateful(step).fit(out.loc[is_train.reindex(out.index).fillna(True).astype(bool)],
                                          target=out[target] if target in out.columns else None)
            out = obj.transform(out)

        # Audit comparison
        common_cols = [c for c in before_cols if c in out.columns]
        common_idx = prev_out.index.intersection(out.index)
        cells_changed = 0
        rows_with_changes = 0
        if len(common_idx) and len(common_cols):
            a, b = prev_out.loc[common_idx, common_cols], out.loc[common_idx, common_cols]
            diff = (a != b) & ~(a.isna() & b.isna())
            cells_changed = int(diff.sum().sum())
            rows_with_changes = int(diff.any(axis=1).sum())
            total_cells_changed += cells_changed

        rows_removed = before_rows - len(out)
        cols_dropped = [c for c in before_cols if c not in out.columns]
        cols_added = [c for c in out.columns if c not in before_cols]

        # Target Column Preservation: never silently drop the target column
        if target and target in before_cols and target not in out.columns:
            if not step.get("params", {}).get("force_drop_target", False):
                out[target] = prev_out[target].reindex(out.index)
                if target in cols_dropped:
                    cols_dropped.remove(target)

        # Categorize action status: Removed | Corrected | Flagged | Unchanged
        if rows_removed > 0 or cols_dropped:
            action_status = "Removed"
        elif cells_changed > 0:
            action_status = "Corrected"
        elif not step.get("enabled", True):
            action_status = "Flagged"
        else:
            action_status = "Unchanged"

        action_summary = f"Applied {step.get('label') or step['op']} on {step.get('columns', [])}"
        if rows_removed > 0:
            action_summary += f": removed {rows_removed} row(s)"
        if cols_dropped:
            action_summary += f": dropped column(s) {cols_dropped}"
        if cells_changed > 0:
            action_summary += f": modified {cells_changed} cell(s) across {rows_with_changes} row(s)"

        entry = {
            "step_id": step.get("step_id"),
            "op": step["op"],
            "label": step.get("label", step["op"]),
            "columns": step.get("columns", []),
            "status": action_status,
            "cleaning_method": step.get("label", step["op"]),
            "reason": step.get("finding_code", step["op"]),
            "rows_affected": rows_removed if rows_removed > 0 else rows_with_changes,
            "rows_before": before_rows,
            "rows_after": len(out),
            "rows_removed": rows_removed,
            "columns_added": cols_added,
            "columns_removed": cols_dropped,
            "cells_changed": cells_changed,
            "summary": action_summary,
        }
        log.append(entry)
        audit_trail.append(entry)

    if not split_done:
        is_train = rt.make_split(out, target, task_type, spec["seed"], spec["test_size"], spec["strategy"],
                                 spec["group_column"], spec["time_column"], spec["stratify"])

    # Dataset integrity verification
    target_integrity = {}
    if target and target in df.columns:
        preserved = target in out.columns
        labels_intact = True
        unmodified_rows = 0
        total_common = 0
        if preserved:
            common_idx = df.index.intersection(out.index)
            orig_y = df.loc[common_idx, target]
            clean_y = out.loc[common_idx, target]
            match = (orig_y == clean_y) | (orig_y.isna() & clean_y.isna())
            unmodified_rows = int(match.sum())
            total_common = len(common_idx)
            labels_intact = bool(match.all())
        target_integrity = {
            "target": target,
            "preserved": preserved,
            "labels_intact": labels_intact,
            "unmodified_rows": unmodified_rows,
            "total_rows": total_common,
        }

    # Verify CSV reopenability
    csv_reopenable = True
    try:
        csv_buffer = io.StringIO()
        out.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        reopened = pd.read_csv(csv_buffer)
        csv_reopenable = len(reopened) == len(out) and len(reopened.columns) == len(out.columns)
    except Exception:
        csv_reopenable = False

    dataset_integrity = {
        "target_integrity": target_integrity,
        "rows_original": len(df),
        "rows_cleaned": len(out),
        "columns_original": list(df.columns),
        "columns_cleaned": list(out.columns),
        "csv_export_valid": csv_reopenable,
    }

    return out, is_train.reindex(out.index).fillna(True).astype(bool), {
        "steps": log,
        "audit_trail": audit_trail,
        "total_cells_changed": total_cells_changed,
        "split": spec,
        "options": opts,
        "target_integrity": target_integrity,
        "dataset_integrity": dataset_integrity,
    }
