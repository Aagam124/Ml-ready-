"""Small helpers shared by checks: tokenising names, coercing values, associations."""
from __future__ import annotations

import re
from typing import Iterable, Set

import numpy as np
import pandas as pd
from scipy import stats

CURRENCY_STRIP = re.compile(r"[$€£¥₹,%\s]|USD|EUR|INR|GBP")
ID_TOKENS_FOR_NAMES = {"id", "uuid", "guid", "pk", "key", "identifier"}
PLACEHOLDERS = {"n/a", "na", "null", "none", "nan", "?", "-", "--", "missing", "#n/a", "undefined", "nil", "#na"}


def tokens(name: str) -> Set[str]:
    """Whole-token split of a column name: 'tenure_years' -> {'tenure','years'}; camelCase aware."""
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(name))
    return {t for t in re.split(r"[^a-z0-9]+", spaced.lower()) if t}


def has_token(name: str, vocab: Iterable[str]) -> bool:
    return bool(tokens(name) & set(vocab))


def is_texty(s: pd.Series) -> bool:
    return s.dtype == object or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype)


def numeric_view(s: pd.Series) -> pd.Series:
    """Float view of a column. Text is cleaned of currency symbols, commas, % and spaces."""
    if pd.api.types.is_bool_dtype(s):
        return s.astype(float)
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    cleaned = s.astype("string").str.replace(CURRENCY_STRIP, "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce").astype(float)


def effective_missing(s: pd.Series) -> pd.Series:
    """True where a value is NaN or a blank string."""
    m = s.isna()
    if is_texty(s):
        m = m | (s.astype("string").str.strip() == "").fillna(False)
    return m


def placeholder_mask(s: pd.Series) -> pd.Series:
    """True where a value is a missing-value placeholder using the project’s canonical semantics."""
    text = s.astype("string")
    stripped = text.str.strip()
    lowered = stripped.str.lower()
    return lowered.isna() | (stripped == "") | lowered.isin(PLACEHOLDERS)


def pct(part: float, whole: float) -> float:
    return round(part / whole * 100, 2) if whole else 0.0


def ratio_text(part: int, whole: int, what: str) -> str:
    """Every percentage states its denominator: '35 of 696 complete rows (5.03%)'."""
    return f"{part} of {whole} {what} ({pct(part, whole)}%)"


def cramers_v(a: pd.Series, b: pd.Series) -> float:
    m = a.notna() & b.notna()
    if m.sum() < 5:
        return 0.0
    table = pd.crosstab(a[m], b[m])
    if table.shape[0] < 2 or table.shape[1] < 2:
        return 0.0
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.values.sum()
    r, k = table.shape
    return float(np.sqrt((chi2 / n) / max(1, min(r - 1, k - 1))))


def correlation_ratio(cat: pd.Series, num: pd.Series) -> float:
    """eta: how much of a numeric column's variance is explained by a category (0..1)."""
    m = cat.notna() & num.notna()
    if m.sum() < 5:
        return 0.0
    c, x = cat[m].astype(str), num[m].astype(float)
    total = ((x - x.mean()) ** 2).sum()
    if total == 0:
        return 0.0
    means = x.groupby(c).transform("mean")
    return float(np.sqrt(max(0.0, 1 - ((x - means) ** 2).sum() / total)))


def association(a: pd.Series, b: pd.Series, a_num: bool, b_num: bool) -> float:
    """Strength of association 0..1 for any column-type pairing."""
    if a_num and b_num:
        m = a.notna() & b.notna()
        if m.sum() < 5 or a[m].nunique() < 2 or b[m].nunique() < 2:
            return 0.0
        return float(abs(stats.spearmanr(a[m], b[m])[0]))
    if a_num != b_num:
        return correlation_ratio(b if a_num else a, a if a_num else b)
    return cramers_v(a, b)


def bin_series(x: pd.Series, q: int = 10) -> pd.Series:
    try:
        return pd.qcut(x, q=q, duplicates="drop", labels=False)
    except ValueError:
        return pd.Series(0, index=x.index)


def entropy(counts: np.ndarray) -> float:
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log(p)).sum())


def safe_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    return df if len(df) <= n else df.sample(n=n, random_state=seed)
