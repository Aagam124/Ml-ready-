"""Plugin registry. A check is a small class: id, applies_to(ctx), run(df, ctx) -> findings."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Type

import pandas as pd

from app.engine.context import AnalysisContext
from app.engine.findings import Finding


class Check(ABC):
    id: str = ""
    category: str = "quality"
    requires_target: bool = False
    depends_on: Tuple[str, ...] = ()

    def applies_to(self, ctx: AnalysisContext) -> bool:
        return ctx.has_target or not self.requires_target

    @abstractmethod
    def run(self, df: pd.DataFrame, ctx: AnalysisContext) -> List[Finding]:
        ...


REGISTRY: Dict[str, Check] = {}


def register(cls: Type[Check]) -> Type[Check]:
    inst = cls()
    if not inst.id:
        raise ValueError(f"{cls.__name__} must define an id")
    if inst.id in REGISTRY:
        raise ValueError(f"Duplicate check id: {inst.id}")
    REGISTRY[inst.id] = inst
    return cls


def ordered_checks() -> List[Check]:
    """Topological order by depends_on (stable: ties keep registration order)."""
    done: List[str] = []
    remaining = list(REGISTRY)
    while remaining:
        progressed = False
        for cid in list(remaining):
            if all(d in done or d not in REGISTRY for d in REGISTRY[cid].depends_on):
                done.append(cid)
                remaining.remove(cid)
                progressed = True
        if not progressed:
            raise ValueError(f"Circular check dependency among: {remaining}")
    return [REGISTRY[c] for c in done]
