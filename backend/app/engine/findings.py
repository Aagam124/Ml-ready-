"""Core data types shared by every check."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

SEVERITIES = ("critical", "high", "medium", "low", "info")
CONFIDENCES = ("high", "medium", "low")
CATEGORIES = ("quality", "outliers", "leakage", "ml_risk")
EFFORTS = ("low", "medium", "high")


def jsonable(value: Any) -> Any:
    """Recursively convert NumPy / pandas scalars so evidence is always JSON-safe."""
    import numpy as np
    import pandas as pd

    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        f = float(value)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, 6)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [jsonable(v) for v in value.tolist()]
    if value is pd.NaT:
        return None
    return str(value)


@dataclass
class Finding:
    """One real problem. `fixes` use the cleaning-engine op names (see app/cleaning)."""

    code: str
    category: str
    severity: str
    confidence: str
    columns: List[str]
    evidence: Dict[str, Any]
    plain_text: str
    technical_text: str
    fixes: List[Dict[str, Any]] = field(default_factory=list)
    breadth: float = 1.0          # share (0..1) of rows or columns affected; 1.0 = dataset-wide
    effort: str = "low"           # effort to fix: low | medium | high
    score_impact: float = 0.0     # filled in by the scorer: points deducted for this finding
    check_id: str = ""

    def __post_init__(self) -> None:
        assert self.severity in SEVERITIES, self.severity
        assert self.confidence in CONFIDENCES, self.confidence
        assert self.category in CATEGORIES, self.category
        self.evidence = jsonable(self.evidence)
        self.columns = [str(c) for c in self.columns]
        self.breadth = float(min(1.0, max(0.0, self.breadth)))

    @property
    def key(self) -> tuple:
        return (self.code, tuple(sorted(self.columns)))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code, "category": self.category, "severity": self.severity,
            "confidence": self.confidence, "columns": self.columns, "evidence": self.evidence,
            "plain_text": self.plain_text, "technical_text": self.technical_text,
            "fixes": self.fixes, "score_impact": self.score_impact, "effort": self.effort,
            "breadth": round(self.breadth, 4), "check_id": self.check_id,
        }


def fix(fix_id: str, label: str, op: str, columns: Optional[List[str]] = None, **params: Any) -> Dict[str, Any]:
    """Build a fix suggestion. `op` + `params` map 1:1 onto a cleaning-engine step."""
    return {"fix_id": fix_id, "label": label, "op": op, "columns": columns or [], "params": jsonable(params)}
