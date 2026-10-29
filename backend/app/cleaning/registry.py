"""Op catalogue: one source of truth for the executor AND the code generator."""
from __future__ import annotations

from typing import Any, Dict, Optional

from app.cleaning import runtime as rt

# kind: stateless -> function(df, columns, **params); stateful -> Class(columns, **params).fit(train).transform(df)
OPS: Dict[str, Dict[str, Any]] = {
    "strip_whitespace":     {"kind": "stateless", "phase": 0, "impl": "strip_whitespace", "label": "Trim spaces"},
    "normalize_categories": {"kind": "stateless", "phase": 0, "impl": "normalize_categories", "label": "Merge category spellings"},
    "fix_encoding":         {"kind": "stateless", "phase": 0, "impl": "fix_encoding", "label": "Repair text encoding"},
    "cast_type":            {"kind": "stateless", "phase": 0, "impl": "cast_type", "label": "Convert type"},
    "parse_dates":          {"kind": "stateless", "phase": 0, "impl": "parse_dates", "label": "Parse dates"},
    "convert_units":        {"kind": "stateless", "phase": 0, "impl": "convert_units", "label": "Convert units"},
    "merge_rare_classes":   {"kind": "stateless", "phase": 0, "impl": "merge_rare_classes", "label": "Merge rare classes"},
    "fix_invalid":          {"kind": "stateless", "phase": 0, "impl": "fix_invalid", "label": "Fix invalid values"},
    "drop_missing_target":  {"kind": "stateless", "phase": 1, "impl": "drop_missing_target", "label": "Remove rows without a target"},
    "drop_duplicates":      {"kind": "stateless", "phase": 1, "impl": "drop_duplicates", "label": "Remove duplicate rows"},
    "drop_columns":         {"kind": "stateless", "phase": 2, "impl": "drop_columns", "label": "Drop columns"},
    "winsorize":            {"kind": "stateful", "phase": 3, "order": 0, "impl": "Winsorize", "label": "Clip outliers"},
    "impute":               {"kind": "stateful", "phase": 3, "order": 1, "impl": "Impute", "label": "Fill missing values"},
    "bin":                  {"kind": "stateful", "phase": 3, "order": 2, "impl": "Bin", "label": "Bin values"},
    "encode":               {"kind": "stateful", "phase": 3, "order": 3, "impl": "Encode", "label": "Encode categories"},
    "scale":                {"kind": "stateful", "phase": 3, "order": 4, "impl": "Scale", "label": "Scale numbers"},
    "modeling_option":      {"kind": "option", "phase": 9, "impl": None, "label": "Modeling setting"},
    "manual":               {"kind": "option", "phase": 9, "impl": None, "label": "Manual action"},
}


def phase_of(step: Dict[str, Any]) -> int:
    meta = OPS[step["op"]]
    if step["op"] == "fix_invalid" and step.get("params", {}).get("action") == "remove":
        return 1
    return meta["phase"]


def sort_key(indexed_step):
    i, step = indexed_step
    return (phase_of(step), OPS[step["op"]].get("order", 0), i)


def validate_step(step: Dict[str, Any]) -> Optional[str]:
    if step.get("op") not in OPS:
        return f"Unknown operation '{step.get('op')}'."
    return None


def impl(name: str):
    return getattr(rt, name)
