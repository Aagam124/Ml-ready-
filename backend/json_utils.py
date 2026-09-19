"""
json_utils.py
-------------
Recursively replaces inf/-inf/NaN with None so results are safe to hand to
FastAPI's default JSONResponse (which sets allow_nan=False) and to Mongo's
JSON-facing layers. Shared by analysis/pipeline.py and models/baseline.py -
both can legitimately produce non-finite floats (infinite VIF, degenerate
R² on a near-constant target).
"""

import math
from typing import Any


def sanitize_for_json(obj: Any) -> Any:
    if isinstance(obj, float):
        return None if (math.isinf(obj) or math.isnan(obj)) else obj
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    return obj
