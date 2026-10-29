"""CSV-injection protection for spreadsheet users.

A text cell starting with = + - @ (or tab / carriage return) can run as a formula when a CSV is opened in
Excel or Sheets. Numeric cells are never touched; only STRING cells are prefixed with a single quote.
Default downloads stay byte-identical to what was stored, so this is opt-in (?spreadsheet_safe=true).
"""
from __future__ import annotations

import pandas as pd

_DANGEROUS = ("=", "+", "-", "@", "\t", "\r")


def sanitize_for_spreadsheet(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        s = out[c]
        if s.dtype == object or pd.api.types.is_string_dtype(s):
            txt = s.astype("string")
            risky = txt.str.startswith(_DANGEROUS).fillna(False).astype(bool)
            out[c] = txt.mask(risky, "'" + txt)
    out.columns = [("'" + str(c)) if str(c).startswith(_DANGEROUS) else c for c in out.columns]
    return out
