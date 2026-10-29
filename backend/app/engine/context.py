"""AnalysisContext: everything a check may read. Checks never mutate it (except `shared`)."""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from app.engine.config import EngineConfig
from app.engine.utils import tokens

CLASSIFICATION = ("binary", "multiclass")


@dataclass
class AnalysisContext:
    df: pd.DataFrame
    target: Optional[str]
    task_type: Optional[str]                 # binary | multiclass | regression | None
    profile: Dict[str, Any]
    config: EngineConfig
    n_rows_original: int
    sampled: bool = False
    test_df: Optional[pd.DataFrame] = None   # optional hold-out for adversarial validation
    shared: Dict[str, Any] = field(default_factory=dict)  # check -> check hand-off (e.g. invalid masks)
    _dt_cache: Dict[str, pd.Series] = field(default_factory=dict)

    # ---- column metadata -------------------------------------------------
    @property
    def has_target(self) -> bool:
        return self.target is not None and self.target in self.df.columns

    @property
    def is_classification(self) -> bool:
        return self.task_type in CLASSIFICATION

    @property
    def col_profile(self) -> Dict[str, Dict[str, Any]]:
        return {c["name"]: c for c in self.profile["columns"]}

    def semantic(self, col: str) -> str:
        return self.col_profile[col]["semantic_type"]

    def flags(self, col: str) -> List[str]:
        return self.col_profile[col].get("flags", [])

    @property
    def feature_columns(self) -> List[str]:
        """Every column except the target. Feature-only checks must iterate this, never df.columns."""
        return [c for c in self.df.columns if c != self.target]

    def columns_of(self, *semantic_types: str, include_target: bool = False) -> List[str]:
        cols = self.df.columns if include_target else self.feature_columns
        return [c for c in cols if self.semantic(c) in semantic_types]

    @property
    def identifier_columns(self) -> List[str]:
        return [c for c in self.feature_columns if self.semantic(c) == "identifier"]

    @property
    def id_like_columns(self) -> List[str]:
        """Identifier semantics, plus ID-named columns that may repeat (so duplicated rows still reveal them)."""
        from app.engine.utils import ID_TOKENS_FOR_NAMES
        out = []
        for c in self.feature_columns:
            nn = self.df[c].dropna()
            if self.semantic(c) == "identifier" or (tokens(c) & ID_TOKENS_FOR_NAMES and len(nn) and nn.nunique() / len(nn) >= 0.5
                                                    and self.semantic(c) not in ("constant", "empty", "continuous")):
                out.append(c)
        return out

    # ---- parsed dates ----------------------------------------------------
    def datetimes(self, col: str) -> pd.Series:
        """Tz-naive datetimes parsed from VALUES. Unparseable cells become NaT."""
        if col not in self._dt_cache:
            s = self.df[col]
            if pd.api.types.is_datetime64_any_dtype(s):
                parsed = s
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    parsed = pd.to_datetime(s, errors="coerce", format="mixed", utc=True)
            try:
                parsed = parsed.dt.tz_convert(None)
            except (TypeError, AttributeError):
                try:
                    parsed = parsed.dt.tz_localize(None)
                except (TypeError, AttributeError):
                    pass
            self._dt_cache[col] = parsed
        return self._dt_cache[col]

    @property
    def datetime_columns(self) -> List[str]:
        return self.columns_of("datetime")

    @property
    def as_of(self) -> pd.Timestamp:
        return pd.Timestamp(self.config.as_of) if self.config.as_of else pd.Timestamp.now().normalize()

    def name_tokens(self, col: str) -> set:
        return tokens(col)
