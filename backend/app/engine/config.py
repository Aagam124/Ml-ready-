"""Engine configuration. A snapshot of this is stored on every analysis (reproducibility)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

ENGINE_VERSION = "1.0.0"


@dataclass
class EngineConfig:
    seed: int = 42
    sample_threshold: int = 200_000      # above this many rows, analyse a stratified sample
    ml_check_max_rows: int = 20_000      # cap for model-based checks (leakage, adversarial)
    as_of: Optional[str] = None          # ISO date used for "future date" checks; fixed => deterministic
    missing_min_pct: float = 0.0         # report all missingness (even < 1%)
    max_categorical_cardinality_for_groups: int = 200
    outlier_min_rate_pct: float = 1.0
    multivariate_contamination: float = 0.02  # targeted contamination for multivariate Isolation Forest
    leakage_model_auc: float = 0.95
    leakage_corr: float = 0.85
    leakage_mi: float = 0.70
    vif_threshold: float = 10.0
    redundant_corr: float = 0.999
    collinear_corr: float = 0.95
    approved_replacement_categories: Tuple[str, ...] = (
        "Unknown", "unknown", "Other", "other", "N/A", "n/a", "Unspecified", "Missing"
    )
    allowed_categories: Dict[str, List[str]] = field(default_factory=dict)

    def snapshot(self) -> Dict[str, Any]:
        d = asdict(self)
        d["engine_version"] = ENGINE_VERSION
        return d
