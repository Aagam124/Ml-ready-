"""Turn findings into a cleaning plan: ordered, reversible steps, each tied to the finding it fixes."""
from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional

from app.cleaning.registry import OPS

NUMERIC_SEM = ("continuous", "count", "discrete", "currency", "percentage")


def _step_id(code: str, op: str, columns: List[str], params: Dict[str, Any]) -> str:
    h = hashlib.sha1(f"{code}|{op}|{sorted(columns)}|{sorted(params.items(), key=lambda kv: kv[0])}".encode()).hexdigest()[:8]
    return f"{code}-{h}"


def propose_plan(findings: List[Dict[str, Any]], profile: Dict[str, Any], target: Optional[str]) -> Dict[str, Any]:
    """Default plan = apply the first suggested fix of every finding that has an automatic one.

    Conservative defaults (principle: trust over cleverness): confident fixes are enabled; 'Possible'
    leakage, outlier clipping and encodings are included but switched OFF for the user to decide.
    """
    sem = {c["name"]: c["semantic_type"] for c in profile["columns"]}
    steps: List[Dict[str, Any]] = []
    seen = set()
    imputed: set = set()
    needs_impute: List[str] = []
    for f in findings:
        if not f.get("fixes"):
            continue
        fx = f["fixes"][0]
        if fx["op"] == "manual":
            continue
        op, cols, params = fx["op"], fx.get("columns") or f.get("columns") or [], dict(fx.get("params") or {})
        if op == "impute":
            imputed |= set(cols)
        if op == "fix_invalid" and params.get("action") == "set_missing":
            needs_impute += cols
        enabled = True
        if f["code"] == "target_leakage" and f["evidence"].get("tier") != "Likely":
            enabled = False
        if f["code"] in ("outliers", "high_cardinality", "identifier_feature", "free_text_column", "near_duplicates", "multicollinearity"):
            enabled = False
        if f["code"] == "invalid_values" and params.get("action") == "remove":
            enabled = False
        key = (op, tuple(sorted(cols)), repr(sorted(params.items(), key=lambda kv: kv[0])))
        if key in seen:
            continue
        seen.add(key)
        steps.append({"step_id": _step_id(f["code"], op, cols, params), "op": op, "params": params, "columns": cols,
                      "enabled": enabled, "label": fx["label"], "finding_code": f["code"], "finding_columns": f.get("columns", []),
                      "severity": f["severity"], "preview": None})
    # Values set to missing by a fix must not be left as gaps: add an impute step if none exists.
    SENSITIVE_OR_NOMINAL = {"gender", "sex", "race", "ethnicity", "religion", "city", "state", "country", "location"}
    for col in dict.fromkeys(needs_impute):
        if col in imputed or col == target:
            continue
        is_num = sem.get(col) in NUMERIC_SEM
        col_lower = col.lower()
        is_sensitive = any(tok in col_lower for tok in SENSITIVE_OR_NOMINAL)
        if is_num:
            strategy, params, label = "median", {"strategy": "median"}, f"Fill gaps in '{col}' with median"
        elif is_sensitive:
            strategy, params, label = "constant", {"strategy": "constant", "value": "Unknown"}, f"Fill gaps in '{col}' with 'Unknown' (unbiased)"
        else:
            strategy, params, label = "mode", {"strategy": "mode"}, f"Fill gaps in '{col}' with most frequent value"
        steps.append({"step_id": _step_id("fill_after_fix", "impute", [col], params), "op": "impute",
                      "params": params, "columns": [col], "enabled": True,
                      "label": label, "finding_code": "invalid_values", "finding_columns": [col],
                      "severity": "medium", "preview": None})
    # merge several impute steps with the same strategy? keep separate: each is toggleable by its finding.
    for s in steps:
        assert s["op"] in OPS, s["op"]
    return {"steps": steps}
