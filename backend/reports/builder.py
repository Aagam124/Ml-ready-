"""
builder.py
----------
Phase 6 - assembles everything section 15 of the spec asks a report to
contain, in one place, so the JSON API endpoint and the HTML/PDF export
render from the exact same data - no duplicated assembly logic, no risk of
the export saying something different from what's on screen.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def build_report_bundle(
    dataset: Dict[str, Any],
    latest_analysis: Optional[Dict[str, Any]],
    cleaning_history: List[Dict[str, Any]],
    model_comparison: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "filename": dataset["filename"],
            "status": dataset["status"],
            "uploaded_at": dataset["uploaded_at"],
            "n_rows": dataset.get("metadata", {}).get("n_rows"),
            "n_columns": dataset.get("metadata", {}).get("n_columns"),
            "size_bytes": dataset.get("metadata", {}).get("size_bytes"),
        },
        "analysis": latest_analysis,        # full Phase 1 report dict, or None
        "cleaning_history": cleaning_history,  # list of cleaning_history entries, oldest first
        "model_comparison": model_comparison,  # {before, after, deltas, after_skipped_reason}, or None
    }
