"""
Semantic type inference.

Goes beyond pandas dtypes: tells apart identifiers, counts, continuous
numbers, binary flags, categories, free text, dates, emails, URLs, phones,
currency and percentages.

Core rule (learned the hard way): a column is NEVER called a datetime from its
name alone. `tenure_years`, `start_amount` and `month_count` are numbers.
Dates are only recognised when the VALUES parse as dates.

Name hints are used only for one thing: deciding whether a nearly-unique
integer column is an identifier. Uniqueness is always required as well.
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from app.engine.utils import placeholder_mask

SAMPLE_SIZE = 1000
ID_TOKENS = {"id", "uuid", "guid", "pk", "key", "identifier"}

EMAIL_RE = r"[^@\s]+@[^@\s]+\.[A-Za-z]{2,}"
URL_RE = r"(?:https?|ftp)://[^\s/$.?#][^\s]*"
CURRENCY_RE = (
    r"(?:[-(]?[$€£¥₹]\s?-?\d[\d,]*(?:\.\d+)?\)?"
    r"|-?\d[\d,]*(?:\.\d+)?\s?(?:USD|EUR|INR|GBP))"
)
PERCENT_RE = r"-?\d+(?:\.\d+)?\s?%"
PHONE_RE = r"\+?\d[\d\s().\-]{6,18}\d"
DATE_LIKE_RE = (
    r"(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2}"
    r"|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}"
    r"|[A-Za-z]{3,9}\.? \d{1,2},? \d{4}"
    r"|\d{1,2} [A-Za-z]{3,9}\.? \d{4})"
)


@dataclass
class SemanticResult:
    type: str
    confidence: float
    evidence: List[str] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)


def _tokens(name: str) -> set:
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(name))
    return {t for t in re.split(r"[^a-z0-9]+", spaced.lower()) if t}


def _name_is_id_like(name: str) -> bool:
    lower = str(name).lower()
    return bool(_tokens(name) & ID_TOKENS) or (lower.endswith("id") and len(lower) > 2)


def _ratio_match(strs: pd.Series, pattern: str) -> float:
    return float(strs.str.fullmatch(pattern).fillna(False).mean())


def _datetime_ratio(strs: pd.Series) -> float:
    looks_like = float(strs.str.match(DATE_LIKE_RE).fillna(False).mean())
    if looks_like < 0.9:
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            parsed = pd.to_datetime(strs, errors="coerce", format="mixed", utc=True)
        except (ValueError, TypeError):
            return 0.0
    return float(parsed.notna().mean())


def _classify_numeric(
    sample: pd.Series,
    full: pd.Series,
    name: str,
    n_unique: int,
    n_nonnull: int,
    flags: List[str],
) -> SemanticResult:
    unique_ratio = n_unique / n_nonnull
    if n_unique == 2:
        return SemanticResult("binary", 0.95, ["Exactly two distinct numeric values."], flags)

    vals = sample.astype(float)
    if not np.all(np.isfinite(vals)):
        flags = flags + ["has_infinite_values"]
        return SemanticResult("continuous", 0.7, ["Contains infinite values."], flags)

    is_int = bool(np.all(vals == np.floor(vals)))
    if not is_int:
        return SemanticResult("continuous", 0.9, ["Non-integer numeric values."], flags)

    # Integer-valued from here on.
    if unique_ratio >= 0.98 and n_nonnull >= 10:
        numeric_full = pd.to_numeric(full, errors="coerce").dropna()
        sequential = bool(numeric_full.sort_values().diff().dropna().eq(1).mean() > 0.99)
        if _name_is_id_like(name) or sequential:
            why = "name looks like an id" if _name_is_id_like(name) else "values form a sequence"
            return SemanticResult(
                "identifier", 0.9, [f"Nearly every value is unique and the {why}."], flags
            )
    span = float(vals.max() - vals.min())
    name_lower = name.lower()
    is_numeric_name = any(tok in name_lower for tok in ("income", "salary", "wage", "amount", "price", "revenue", "spend", "balance", "cost", "years", "tenure", "age", "count", "num"))
    if n_unique <= 10 and not is_numeric_name and float(vals.max()) <= 20:
        return SemanticResult(
            "categorical", 0.7, ["Few distinct integer values; likely category codes or ratings."], flags
        )
    if unique_ratio > 0.5 and span > 1000:
        return SemanticResult("continuous", 0.75, ["Integer-valued with a wide range."], flags)
    if float(vals.min()) >= 0:
        return SemanticResult("count", 0.8, ["Non-negative whole numbers."], flags)
    return SemanticResult("discrete", 0.75, ["Whole numbers including negatives."], flags)


def _classify_text(
    sample: pd.Series,
    full: pd.Series,
    name: str,
    n_unique: int,
    n_nonnull: int,
    flags: List[str],
) -> SemanticResult:
    strs = sample.astype(str).str.strip()
    strs = strs[strs != ""]
    if strs.empty:
        return SemanticResult("empty", 1.0, ["Only blank strings."], flags + ["blank_strings"])
    unique_ratio = n_unique / n_nonnull

    r = _ratio_match(strs, EMAIL_RE)
    if r >= 0.9:
        extra = ["identifier_like"] if unique_ratio >= 0.95 else []
        return SemanticResult("email", r, [f"{r:.0%} of values look like email addresses."], flags + extra)
    r = _ratio_match(strs, URL_RE)
    if r >= 0.9:
        return SemanticResult("url", r, [f"{r:.0%} of values look like URLs."], flags)
    r = _ratio_match(strs, CURRENCY_RE)
    if r >= 0.9:
        return SemanticResult("currency", r, [f"{r:.0%} of values look like money amounts."], flags + ["stored_as_text"])
    r = _ratio_match(strs, PERCENT_RE)
    if r >= 0.9:
        return SemanticResult("percentage", r, [f"{r:.0%} of values look like percentages."], flags + ["stored_as_text"])
    r = _datetime_ratio(strs)
    if r >= 0.9:
        return SemanticResult("datetime", r, [f"{r:.0%} of values parse as dates."], flags + ["stored_as_text"])
    phone_like = strs.str.fullmatch(PHONE_RE).fillna(False) & strs.str.contains(r"[+\s().\-]", regex=True)
    if float(phone_like.mean()) >= 0.9:
        return SemanticResult("phone", float(phone_like.mean()), ["Values look like phone numbers."], flags)

    leading_zeros = float(strs.str.match(r"^0\d+$").fillna(False).mean())
    numeric = pd.to_numeric(strs.str.replace(",", "", regex=False), errors="coerce")
    numeric_ratio = float(numeric.notna().mean())
    if numeric_ratio >= 0.95 and leading_zeros <= 0.05:
        result = _classify_numeric(
            numeric.dropna(), full, name, n_unique, n_nonnull, flags + ["stored_as_text"]
        )
        result.evidence.append("Numbers stored as text.")
        return result
    if 0.1 < numeric_ratio < 0.95:
        flags = flags + ["mixed_types"]
    if leading_zeros > 0.05:
        flags = flags + ["leading_zeros"]

    if n_unique == 2:
        return SemanticResult("binary", 0.95, ["Exactly two distinct values."], flags)

    words = strs.str.split().str.len()
    if float(words.mean()) >= 4 or float(strs.str.len().mean()) > 60:
        return SemanticResult("text", 0.85, ["Long multi-word values; free text."], flags)

    if unique_ratio >= 0.95 and n_nonnull >= 10:
        why = "name looks like an id; " if _name_is_id_like(name) else ""
        return SemanticResult("identifier", 0.9 if why else 0.7, [f"{why}Nearly every value is unique."], flags)

    if n_unique > 50 and unique_ratio > 0.5:
        flags = flags + ["high_cardinality"]
    return SemanticResult("categorical", 0.85, [f"{n_unique} distinct text values."], flags)


def infer_semantic_type(
    series: pd.Series,
    name: str = "",
    n_unique: Optional[int] = None,
    n_nonnull: Optional[int] = None,
) -> SemanticResult:
    s = series.loc[~placeholder_mask(series)].dropna()
    n_nonnull = len(s) if n_nonnull is None else n_nonnull
    if n_nonnull == 0:
        return SemanticResult("empty", 1.0, ["All values are missing."])
    n_unique = int(s.nunique()) if n_unique is None else n_unique
    if n_unique == 1:
        return SemanticResult("constant", 1.0, ["Every non-missing value is identical."])

    sample = s if len(s) <= SAMPLE_SIZE else s.sample(SAMPLE_SIZE, random_state=42)
    flags: List[str] = []

    if pd.api.types.is_bool_dtype(series):
        return SemanticResult("binary", 1.0, ["Boolean column."])
    if pd.api.types.is_datetime64_any_dtype(series):
        return SemanticResult("datetime", 1.0, ["Native datetime dtype."])
    if pd.api.types.is_numeric_dtype(series):
        return _classify_numeric(sample, s, name, n_unique, n_nonnull, flags)
    if series.dtype == object and sample.map(type).nunique() > 1:
        flags.append("mixed_types")
    return _classify_text(sample, s, name, n_unique, n_nonnull, flags)
