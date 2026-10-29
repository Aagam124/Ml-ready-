"""Target and task-type detection. Returns ranked candidates, each with confidence and reasons."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from app.engine.utils import numeric_view, tokens

TARGET_NAME_HINTS = {
    "churn", "churned", "label", "target", "outcome", "y", "class", "default", "defaulted", "fraud",
    "survived", "response", "converted", "conversion", "diagnosis", "price", "sale_price", "saleprice",
    "income", "salary", "loan_status", "status", "is_fraud", "attrition", "clicked", "approved",
}
NEVER_TARGET = {"identifier", "constant", "empty", "email", "url", "phone", "text", "datetime"}
MAX_CLASSES = 20


def task_type_for(series: pd.Series, semantic: str) -> Optional[str]:
    """binary / multiclass / regression for a column, or None when it cannot be a target."""
    s = series.dropna()
    if s.empty or semantic in NEVER_TARGET:
        return None
    n_unique = s.nunique()
    if n_unique < 2:
        return None
    if semantic == "binary" or n_unique == 2:
        return "binary"
    if semantic == "categorical" and n_unique <= MAX_CLASSES:
        return "multiclass"
    if semantic in ("continuous", "count", "discrete", "currency", "percentage"):
        return "regression"
    return None


def suggest_targets(df: pd.DataFrame, profile: Dict[str, Any], limit: int = 5) -> List[Dict[str, Any]]:
    cols = {c["name"]: c for c in profile["columns"]}
    names = list(df.columns)
    out: List[Dict[str, Any]] = []
    for pos, name in enumerate(names):
        sem = cols[name]["semantic_type"]
        task = task_type_for(df[name], sem)
        if task is None:
            continue
        score, reasons = 0.0, []
        hint = tokens(name) & TARGET_NAME_HINTS
        if hint:
            score += 0.45
            reasons.append(f"Name contains the target-like word '{sorted(hint)[0]}'.")
        if pos == len(names) - 1:
            score += 0.2
            reasons.append("It is the last column, where targets usually sit.")
        if task == "binary":
            score += 0.2
            reasons.append("It has exactly two values, typical of a yes/no outcome.")
        elif task == "multiclass":
            score += 0.1
            reasons.append(f"It has {df[name].nunique()} distinct categories.")
        else:
            score += 0.05
        if cols[name]["missing_pct"] > 30:
            score -= 0.2
        if score > 0.0:
            out.append({"column": name, "task_type": task, "confidence": round(min(score, 0.99), 2), "reasons": reasons})
    out.sort(key=lambda c: (-c["confidence"], names.index(c["column"])))
    return out[:limit]


def task_type_from_profile(col: Dict[str, Any]) -> Optional[str]:
    """Same decision as task_type_for, using only the stored column profile (no data load)."""
    sem, n_unique = col["semantic_type"], col["n_unique"]
    if sem in NEVER_TARGET or n_unique < 2:
        return None
    if sem == "binary" or n_unique == 2:
        return "binary"
    if sem == "categorical" and n_unique <= MAX_CLASSES:
        return "multiclass"
    if sem in ("continuous", "count", "discrete", "currency", "percentage"):
        return "regression"
    return None


def suggest_targets_from_profile(profile: Dict[str, Any], limit: int = 5) -> List[Dict[str, Any]]:
    names = [c["name"] for c in profile["columns"]]
    out = []
    for pos, c in enumerate(profile["columns"]):
        task = task_type_from_profile(c)
        if task is None:
            continue
        score, reasons = 0.0, []
        hint = tokens(c["name"]) & TARGET_NAME_HINTS
        if hint:
            score += 0.45
            reasons.append(f"Name contains the target-like word '{sorted(hint)[0]}'.")
        if pos == len(names) - 1:
            score += 0.2
            reasons.append("It is the last column, where targets usually sit.")
        if task == "binary":
            score += 0.2
            reasons.append("It has exactly two values, typical of a yes/no outcome.")
        elif task == "multiclass":
            score += 0.1
            reasons.append(f"It has {c['n_unique']} distinct categories.")
        else:
            score += 0.05
        if c["missing_pct"] > 30:
            score -= 0.2
        if score > 0:
            out.append({"column": c["name"], "task_type": task, "confidence": round(min(score, 0.99), 2), "reasons": reasons})
    out.sort(key=lambda x: (-x["confidence"], names.index(x["column"])))
    return out[:limit]
