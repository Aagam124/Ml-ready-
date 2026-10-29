"""Engine configuration. A snapshot of this is stored on every analysis (reproducibility)."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

ENGINE_VERSION = "1.0.0"


@dataclass
class EngineConfig:
    seed: int = 42
    sample_threshold: int = 200_000      # above this many rows, analyse a stratified sample
    ml_check_max_rows: int = 20_000      # cap for model-based checks (leakage, adversarial)
    as_of: Optional[str] = None          # ISO date used for "future date" checks; fixed => deterministic
    missing_min_pct: float = 1.0
    max_categorical_cardinality_for_groups: int = 200
    outlier_min_rate_pct: float = 1.0
    leakage_model_auc: float = 0.95
    leakage_corr: float = 0.85
    leakage_mi: float = 0.70
    vif_threshold: float = 10.0
    redundant_corr: float = 0.999
    collinear_corr: float = 0.95

    def snapshot(self) -> Dict[str, Any]:
        d = asdict(self)
        d["engine_version"] = ENGINE_VERSION
        return d
