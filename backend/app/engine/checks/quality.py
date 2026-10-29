"""Data-quality checks: missing values, duplicates, constants, invalid values, mistyped columns."""
from __future__ import annotations

import re
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy import stats

from app.engine.context import AnalysisContext
from app.engine.findings import Finding, fix
from app.engine.registry import Check, register
from app.engine.utils import (PLACEHOLDERS, association, effective_missing, is_texty,
                              numeric_view, pct, ratio_text, tokens)

NON_NEGATIVE = {"salary", "wage", "income", "price", "cost", "fare", "age", "quantity", "qty", "count",
                "weight", "height", "revenue", "sales", "population", "tenure", "duration", "distance",
                "area", "volume", "length", "size", "years", "months", "days", "spend", "experience"}
SIGNED_OK = {"change", "delta", "diff", "difference", "balance", "profit", "loss", "temperature", "temp",
             "return", "growth", "net", "lat", "lon", "latitude", "longitude", "residual", "error"}
PAST_EVENT = {"birth", "dob", "created", "signup", "joined", "registered", "purchase", "purchased", "order",
              "transaction", "hire", "hired", "login", "visit", "event"}
FUTURE_OK = {"expiry", "expire", "expires", "due", "deadline", "next", "renewal", "scheduled", "end",
             "until", "delivery", "forecast", "target", "valid"}
UNIT_RE = re.compile(r"^\s*(-?\d+(?:[.,]\d+)?)\s*([A-Za-z°/]+)\s*$")
UNIT_FAMILIES = {"mass": {"kg": 1.0, "g": 0.001, "lb": 0.45359237, "lbs": 0.45359237, "oz": 0.028349523},
                 "length": {"m": 1.0, "cm": 0.01, "mm": 0.001, "km": 1000.0, "in": 0.0254, "ft": 0.3048, "mi": 1609.344}}
MOJIBAKE = re.compile(r"Ã.|â€.|Â.|\ufffd")


def _example_values(values: pd.Series, n: int = 5) -> List[Any]:
    return [v if isinstance(v, (int, float)) else str(v) for v in values.head(n).tolist()]


@register
class MissingValuesCheck(Check):
    id = "missing_values"
    category = "quality"

    def run(self, df, ctx: AnalysisContext) -> List[Finding]:
        out: List[Finding] = []
        n = len(df)
        miss = {c: effective_missing(df[c]) for c in df.columns}
        for col in ctx.feature_columns:
            m = miss[col]
            k = int(m.sum())
            share = pct(k, n)
            if k == 0 or share < ctx.config.missing_min_pct or ctx.semantic(col) in ("constant", "empty"):
                continue
            severity = "high" if share >= 60 else "medium" if share >= 20 else "low"
            pattern = self._pattern(df, ctx, col, m)
            drop = share >= 60
            is_num = ctx.semantic(col) in ("continuous", "count", "discrete", "currency", "percentage")
            col_lower = col.lower()
            is_sensitive_or_nominal = any(w in col_lower for w in ("gender", "sex", "race", "ethnicity", "religion", "city", "state", "country", "location"))
            if is_num:
                fixes = [fix("impute_median", f"Fill missing '{col}' with median", "impute", [col], strategy="median")]
            elif is_sensitive_or_nominal:
                fixes = [
                    fix("impute_unknown", f"Fill missing '{col}' with 'Unknown' (avoids demographic/geographic bias)", "impute", [col], strategy="constant", value="Unknown"),
                    fix("impute_mode", f"Fill missing '{col}' with mode", "impute", [col], strategy="mode")
                ]
            else:
                fixes = [fix("impute_mode", f"Fill missing '{col}' with mode", "impute", [col], strategy="mode")]
            if drop:
                fixes.insert(0, fix("drop_column", f"Drop '{col}' (mostly empty)", "drop_columns", [col]))
            out.append(Finding(
                code="missing_values", category="quality", severity=severity, confidence="high", columns=[col],
                evidence={"n_missing": k, "n_rows": n, "pct": share, "denominator": f"{n} rows",
                          "n_blank_strings": int((m & df[col].notna()).sum()), "pattern": pattern},
                plain_text=f"'{col}' is missing in {ratio_text(k, n, 'rows')}. " + pattern["plain"],
                technical_text=f"Effective missing (NaN or blank) = {k}/{n}. {pattern['technical']}",
                fixes=fixes, breadth=1 / max(1, len(ctx.feature_columns)), effort="medium"))
        return out

    @staticmethod
    def _pattern(df: pd.DataFrame, ctx: AnalysisContext, col: str, m: pd.Series) -> Dict[str, Any]:
        """MAR hint: is missingness associated with other columns? Bonferroni-corrected, effect-size gated."""
        if m.sum() < 10 or (~m).sum() < 10:
            return {"kind": "unknown", "associated_with": [], "plain": "", "technical": "Too few rows to assess the pattern."}
        indicator = m.astype(int)
        hits = []
        others = [c for c in ctx.feature_columns if c != col and ctx.semantic(c) not in ("identifier", "constant", "empty", "text", "email", "url")][:25]
        for o in others:
            s = df[o]
            if pd.api.types.is_numeric_dtype(s):
                a, b = s[m].dropna(), s[~m].dropna()
                if len(a) < 5 or len(b) < 5 or s.nunique() < 2:
                    continue
                p = stats.mannwhitneyu(a, b).pvalue
                effect = abs(stats.rankdata(np.r_[a, b])[:len(a)].mean() / (len(a) + len(b)) - 0.5) * 2
            else:
                t = pd.crosstab(s.fillna("__nan__"), indicator)
                if t.shape[0] < 2 or t.shape[1] < 2 or t.shape[0] > 30:
                    continue
                p = stats.chi2_contingency(t)[1]
                effect = association(s.astype("string"), indicator, False, False)
            if p < 0.001 / max(1, len(others)) and effect >= 0.15:
                hits.append({"column": o, "p_value": float(p), "effect": round(float(effect), 3)})
        if hits:
            names = ", ".join(h["column"] for h in hits[:3])
            return {"kind": "possibly_MAR", "associated_with": hits,
                    "plain": f"The gaps are not random: they relate to {names}, so simple averages may bias results.",
                    "technical": f"Missingness indicator associated with {names} (Bonferroni-corrected p<0.001, effect>=0.15): consistent with MAR."}
        return {"kind": "no_evidence_of_dependence", "associated_with": [],
                "plain": "No sign that the gaps depend on other columns (consistent with random, though not proof).",
                "technical": "No association with other columns passed corrected significance and effect-size gates; consistent with MCAR but not proven."}


@register
class DuplicatesCheck(Check):
    id = "duplicates"
    category = "quality"

    def run(self, df, ctx: AnalysisContext) -> List[Finding]:
        out: List[Finding] = []
        n = len(df)
        exact = df.duplicated(keep="first")
        k = int(exact.sum())
        combos = 1.0
        for c in df.columns:
            combos *= max(1, df[c].nunique())
            if combos > 1e12:
                break
        natural = combos < 100 * n  # low-cardinality data: duplicates are expected, not an error
        ids = ctx.id_like_columns
        if k > 0:
            share = pct(k, n)
            severity = "high" if share >= 10 else "medium" if share >= 2 else "low"
            if natural:
                severity = "low"
            shares_id = bool(ids) and bool(df.loc[exact, ids[0]].isin(df.loc[~exact, ids[0]]).all())
            out.append(Finding(
                code="exact_duplicates", category="quality", severity=severity,
                confidence="low" if natural else "high", columns=[],
                evidence={"n_duplicate_rows": k, "n_rows": n, "pct": share, "denominator": f"{n} rows",
                          "share_identifier": shares_id, "low_cardinality_data": natural},
                plain_text=f"{ratio_text(k, n, 'rows')} are exact copies of another row."
                           + (" The copies share the same ID, which points to an ingestion error." if shares_id else "")
                           + (" Your columns have few distinct values, so some repeats can be natural." if natural else "")
                           + " Copies can leak between training and test sets and inflate scores.",
                technical_text=f"DataFrame.duplicated(keep='first') = {k}/{n}. Cardinality product {combos:.3g}.",
                fixes=[fix("drop_duplicates", "Remove duplicate rows (keep first)", "drop_duplicates")],
                breadth=share / 100, effort="low"))

        # Exact duplicate entities where feature columns match but identifier columns differ
        if ids and not natural:
            no_ids = df.drop(columns=ids, errors="ignore")
            id_dup_mask = no_ids.duplicated(keep="first") & (~exact)
            id_dups = int(id_dup_mask.sum())
            if id_dups > 0:
                share = pct(id_dups, n)
                severity = "medium" if share < 5 else "high"
                matching_cols = list(no_ids.columns)
                preview_indices = [int(i) for i in id_dup_mask[id_dup_mask].index[:10]]
                out.append(Finding(
                    code="duplicate_entities_different_ids", category="quality", severity=severity,
                    confidence="medium", columns=ids,
                    evidence={"n_duplicate_rows": id_dups, "n_rows": n, "pct": share, "ignored_ids": ids,
                              "matching_columns": matching_cols, "denominator": f"{n} rows", "row_indices": preview_indices},
                    plain_text=f"{ratio_text(id_dups, n, 'rows')} share identical attributes across all non-identifier columns ({', '.join(matching_cols[:4])}) but have different {', '.join(ids)}. "
                               "They may represent duplicate entries created under new IDs, or distinct entities with matching profiles; review candidate records before removal.",
                    technical_text=f"Exact duplicate records across all non-identifier columns ({len(matching_cols)} cols): {id_dups}/{n}.",
                    fixes=[fix("drop_duplicate_entities", f"Review and remove duplicate records (ignore {', '.join(ids)})", "drop_duplicates",
                               ignore_columns=ids)],
                    breadth=share / 100, effort="medium"))

        # Near-duplicates after normalising whitespace/case; skipped when collisions are natural.
        if len(ctx.feature_columns) >= 5 and not natural and n >= 20:
            norm = df.drop(columns=ids, errors="ignore").copy()
            for c in norm.columns:
                if is_texty(norm[c]):
                    norm[c] = norm[c].astype("string").str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
                elif pd.api.types.is_float_dtype(norm[c]):
                    norm[c] = norm[c].round(6)
            near = int(norm.duplicated(keep="first").sum())
            extra = near - int(df.drop(columns=ids, errors="ignore").duplicated(keep="first").sum())
            if extra >= max(2, int(0.005 * n)) or (near > k and near - k >= 2 and ids):
                diff = max(extra, near - k)
                out.append(Finding(
                    code="near_duplicates", category="quality", severity="medium" if pct(diff, n) >= 2 else "low",
                    confidence="medium", columns=[],
                    evidence={"n_near_duplicate_rows": near, "beyond_exact": diff, "n_rows": n,
                              "ignored_identifier_columns": ids, "denominator": f"{n} rows"},
                    plain_text=f"{ratio_text(diff, n, 'rows')} are copies of another row apart from spacing, capitalisation"
                               + (" or a new ID." if ids else "."),
                    technical_text="Rows compared after strip/lower/whitespace-collapse, floats rounded to 6dp"
                                   + (", identifier columns ignored." if ids else "."),
                    fixes=[fix("drop_near_duplicates", "Normalise text then remove duplicates", "drop_duplicates",
                               normalize=True, ignore_columns=ids)],
                    breadth=pct(diff, n) / 100, effort="low"))
        return out


@register
class ConstantColumnsCheck(Check):
    id = "constant_columns"
    category = "quality"

    def run(self, df, ctx: AnalysisContext) -> List[Finding]:
        out = []
        n = len(df)
        for col in ctx.feature_columns:
            s = df[col]
            nn = s.dropna()
            if nn.empty:
                out.append(self._f(col, "empty_column", "high", {"n_rows": n, "all_missing": True, "pct": 100.0},
                                   f"'{col}' is completely empty (100% missing values in every row). It carries zero predictive signal.", ctx))
                continue
            vc = nn.value_counts()
            top_share = vc.iloc[0] / len(nn)
            if len(vc) == 1 and len(nn) >= 2:
                out.append(self._f(col, "constant_column", "low", {"value": str(vc.index[0]), "n_nonnull": len(nn)},
                                   f"'{col}' has the same value ({vc.index[0]}) in every row, so it carries no information.", ctx))
            elif top_share >= 0.995 and len(nn) >= 200:
                out.append(self._f(col, "near_constant_column", "low",
                                   {"top_share_pct": round(top_share * 100, 2), "n_nonnull": len(nn),
                                    "denominator": f"{len(nn)} non-missing rows"},
                                   f"'{col}' is {top_share:.1%} one value ({len(nn)} non-missing rows), so it adds almost nothing.", ctx))
        return out

    @staticmethod
    def _f(col, code, sev, ev, plain, ctx) -> Finding:
        return Finding(code=code, category="quality", severity=sev, confidence="high", columns=[col], evidence=ev,
                       plain_text=plain, technical_text=f"{code}: {ev}",
                       fixes=[fix("drop_column", f"Drop '{col}'", "drop_columns", [col])],
                       breadth=1 / max(1, len(ctx.feature_columns)), effort="low")


@register
class InvalidValuesCheck(Check):
    id = "invalid_values"
    category = "quality"
    depends_on = ("missing_values",)

    def run(self, df, ctx: AnalysisContext) -> List[Finding]:
        out: List[Finding] = []
        masks: Dict[str, pd.Series] = {}
        n = len(df)
        for col in ctx.feature_columns:
            sem, s = ctx.semantic(col), df[col]
            toks = tokens(col)
            if sem in ("continuous", "count", "discrete", "currency", "percentage") or pd.api.types.is_numeric_dtype(s):
                v = numeric_view(s)
                nn = v.dropna()
                if len(nn) < 5:
                    continue
                if toks & NON_NEGATIVE and not toks & SIGNED_OK:
                    bad = v < 0
                    if "age" in toks:
                        bad = bad | (v > 120)
                    k = int(bad.sum())
                    nonneg_share = float((nn >= 0).mean())
                    if k and nonneg_share >= 0.8:
                        masks[col] = bad.fillna(False)
                        out.append(self._invalid(col, "age" in toks, bad, v, n, len(nn), ctx))
                elif toks & {"percent", "percentage", "pct"} and sem != "percentage":
                    bad = (v < 0) | (v > 100)
                    if int(bad.sum()) and float(((nn >= 0) & (nn <= 100)).mean()) >= 0.8:
                        masks[col] = bad.fillna(False)
                        out.append(self._invalid(col, False, bad, v, n, len(nn), ctx, bounds=(0, 100)))
            if sem == "datetime" or (toks & {"date", "time", "dob", "created", "signup", "joined", "timestamp", "expiry", "due"}):
                f = self._future_dates(col, ctx, n)
                if f:
                    out.append(f)
            if is_texty(s) and sem not in ("text", "email", "url", "phone", "datetime", "identifier"):
                out.extend(self._text_problems(col, s, sem, ctx, n))
        ctx.shared["invalid_masks"] = masks
        return out

    @staticmethod
    def _invalid(col, is_age, bad, v, n, n_valid, ctx, bounds=None) -> Finding:
        k = int(bad.sum())
        share = pct(k, n_valid)
        rule = (f"outside {bounds[0]}-{bounds[1]}" if bounds else "negative or above 120" if is_age else "negative")
        lo, hi = (bounds if bounds else (0, 120 if is_age else None))
        return Finding(
            code="invalid_values", category="quality", severity="high" if (share >= 1 or k >= 5) else "medium",
            confidence="high", columns=[col],
            evidence={"n_invalid": k, "n_valid_values": n_valid, "pct": share, "rule": rule,
                      "denominator": f"{n_valid} non-missing values", "examples": _example_values(v[bad].dropna()),
                      "row_indices": [int(i) for i in v[bad].dropna().index[:20]]},
            plain_text=f"'{col}' has {ratio_text(k, n_valid, 'non-missing values')} that are {rule}. These are data errors, not outliers.",
            technical_text=f"Values {rule} in a column whose name implies a non-negative quantity; >=80% of values are valid.",
            fixes=[fix("set_missing", "Set impossible values to missing, then fill", "fix_invalid", [col],
                       rule="range", min=lo, max=hi, action="set_missing"),
                   fix("remove_rows", "Remove the rows", "fix_invalid", [col], rule="range", min=lo, max=hi, action="remove")],
            breadth=1 / max(1, len(ctx.feature_columns)), effort="low")

    @staticmethod
    def _future_dates(col, ctx, n):
        dt = ctx.datetimes(col)
        valid = dt.dropna()
        if valid.empty:
            return None
        future = dt > ctx.as_of + pd.Timedelta(days=1)
        k = int(future.sum())
        toks = tokens(col)
        if k == 0:
            return None

        # Check for sentinel future dates (year >= 2100, such as 2999-01-01 or 9999-12-31)
        # Even in columns where future dates are normally allowed (e.g. expiry, end_date),
        # sentinel years >= 2100 are placeholder errors that must not escape cleaning.
        sentinel_mask = dt.dt.year >= 2100
        n_sentinel = int(sentinel_mask.sum())
        is_future_ok = bool(toks & FUTURE_OK)

        if is_future_ok and n_sentinel == 0:
            return None  # Legitimate future expiry/due dates in the 21st century

        target_mask = sentinel_mask if is_future_ok else future
        n_flagged = int(target_mask.sum())
        if n_flagged == 0:
            return None

        past_named = bool(toks & PAST_EVENT)
        sev = "high" if (past_named or n_sentinel > 0) else "medium"
        conf = "high"

        flagged_examples = [str(d.date()) for d in dt[target_mask].dropna().head(3)]
        desc = (f"'{col}' contains sentinel future dates like {flagged_examples} (year >= 2100), which are placeholder values requiring cleaning."
                if n_sentinel > 0 else
                f"'{col}' has {ratio_text(k, len(valid), 'dates')} after {ctx.as_of.date()}, which cannot have happened yet.")

        return Finding(
            code="future_dates", category="quality", severity=sev,
            confidence=conf, columns=[col],
            evidence={"n_future": k, "n_sentinel": n_sentinel, "n_dates": len(valid), "as_of": str(ctx.as_of.date()),
                      "denominator": f"{len(valid)} parsed dates", "examples": flagged_examples},
            plain_text=desc,
            technical_text=f"Dates > as_of + 1 day ({ctx.as_of.date()}); sentinel dates (>= 2100) = {n_sentinel}.",
            fixes=[fix("future_to_missing", "Set future dates to missing", "fix_invalid", [col], rule="future_date",
                       as_of=str(ctx.as_of.date()), action="set_missing")],
            breadth=1 / max(1, len(ctx.feature_columns)), effort="low")

    def _text_problems(self, col: str, s: pd.Series, sem: str, ctx, n: int) -> List[Finding]:
        out: List[Finding] = []
        nn = s.dropna().astype("string")
        nn = nn[nn.str.strip() != ""]
        if nn.empty:
            return out
        # disguised missing / mixed types
        lowered = nn.str.strip().str.lower()
        ph = lowered.isin(PLACEHOLDERS)
        numeric_share = float(pd.to_numeric(nn.str.replace(",", "", regex=False), errors="coerce").notna().mean())
        if ph.any() and (numeric_share >= 0.5 or sem == "categorical"):
            k = int(ph.sum())
            if k >= 1 and (numeric_share >= 0.5 or k / len(nn) >= 0.001):
                ctx.shared.setdefault("disguised_cols", set()).add(col)
                out.append(Finding(
                    code="disguised_missing_values", category="quality", severity="medium" if numeric_share >= 0.5 else "low",
                    confidence="high", columns=[col],
                    evidence={"n_placeholders": k, "n_nonblank": len(nn), "tokens": sorted(set(lowered[ph])),
                              "denominator": f"{len(nn)} non-blank values"},
                    plain_text=f"'{col}' uses placeholders like {sorted(set(lowered[ph]))[:3]} for missing data ({ratio_text(k, len(nn), 'non-blank values')}).",
                    technical_text="Strings in {n/a, null, none, ?, -, ...} are missing values stored as text.",
                    fixes=[fix("placeholders_to_missing", "Treat placeholders as missing" + (" and convert to numbers" if numeric_share >= 0.5 else ""),
                               "fix_invalid", [col], rule="placeholders", action="set_missing",
                               then_cast="numeric" if numeric_share >= 0.5 else None)],
                    breadth=1 / max(1, len(ctx.feature_columns)), effort="low"))
        elif 0.1 < numeric_share < 0.95 and "mixed_types" in ctx.flags(col):
            non_num = nn[pd.to_numeric(nn.str.replace(",", "", regex=False), errors="coerce").isna()]
            k = len(non_num)
            out.append(Finding(
                code="mixed_types", category="quality", severity="medium", confidence="high", columns=[col],
                evidence={"n_non_numeric": k, "n_nonblank": len(nn), "examples": _example_values(non_num.drop_duplicates()),
                          "denominator": f"{len(nn)} non-blank values"},
                plain_text=f"'{col}' mixes numbers and text ({ratio_text(k, len(nn), 'non-blank values')} are not numbers).",
                technical_text="Numeric share between 10% and 95% with object dtype.",
                fixes=[fix("coerce_numeric", "Convert to numbers, text becomes missing", "cast_type", [col], to="numeric")],
                breadth=1 / max(1, len(ctx.feature_columns)), effort="medium"))
        # inconsistent categories / whitespace
        if sem in ("categorical", "binary") and nn.nunique() <= ctx.config.max_categorical_cardinality_for_groups:
            groups: Dict[str, List[str]] = {}
            for v in nn.unique():
                key = re.sub(r"\s+", " ", v.strip().lower())
                groups.setdefault(key, []).append(v)
            # single-letter abbreviations that map uniquely onto a longer value (M -> male)
            keys = list(groups)
            for key in [k for k in keys if len(k) == 1]:
                full = [k for k in keys if len(k) > 1 and k.startswith(key)]
                if len(full) == 1:
                    groups[full[0]] += groups.pop(key)
            counts = nn.value_counts()
            multi = {k: v for k, v in groups.items() if len(v) > 1}
            if multi:
                mapping = {}
                for variants in multi.values():
                    canon = max(variants, key=lambda x: (len(x.strip()) > 1, counts.get(x, 0), len(x)))
                    canon = canon.strip()
                    for x in variants:
                        if x != canon:
                            mapping[x] = canon
                case_issue = any(re.sub(r"\s+", " ", a.strip()) != re.sub(r"\s+", " ", b.strip())
                                 for v in multi.values() for a in v for b in v)
                affected = int(nn.isin(list(mapping)).sum())
                out.append(Finding(
                    code="inconsistent_categories" if case_issue else "whitespace_issues", category="quality",
                    severity="medium" if case_issue else "low", confidence="high", columns=[col],
                    evidence={"groups": {k: v for k, v in list(multi.items())[:10]}, "n_values_to_change": affected,
                              "n_nonblank": len(nn), "mapping": mapping, "denominator": f"{len(nn)} non-blank values"},
                    plain_text=(f"'{col}' writes the same category several ways, e.g. {list(multi.values())[0][:3]}. "
                                if case_issue else f"'{col}' has stray spaces around some values. ")
                               + f"{ratio_text(affected, len(nn), 'non-blank values')} would change.",
                    technical_text="Values grouped by lower-cased, whitespace-collapsed key; abbreviations matched by unique prefix.",
                    fixes=[fix("normalize_categories", "Merge variants into one spelling", "normalize_categories", [col], mapping=mapping)],
                    breadth=1 / max(1, len(ctx.feature_columns)), effort="low"))

        # unexpected or rare categories in predominantly binary/canonical categorical columns
        approved = set(ctx.config.approved_replacement_categories) | set(ctx.config.allowed_categories.get(col, []))
        if col in ctx.config.allowed_categories and ctx.config.allowed_categories[col]:
            allowed = set(ctx.config.allowed_categories[col]) | set(ctx.config.approved_replacement_categories)
            disallowed_mask = ~nn.isin(allowed)
            disallowed_count = int(disallowed_mask.sum())
            if disallowed_count > 0:
                rare_cats = sorted(set(nn[disallowed_mask]))
                out.append(Finding(
                    code="unexpected_categorical_value", category="quality",
                    severity="high" if disallowed_count / len(nn) >= 0.05 else "medium", confidence="high", columns=[col],
                    evidence={"unexpected_categories": rare_cats, "allowed_categories": ctx.config.allowed_categories[col],
                              "n_unexpected": disallowed_count, "n_nonblank": len(nn),
                              "pct_unexpected": pct(disallowed_count, len(nn)),
                              "denominator": f"{len(nn)} non-blank values",
                              "row_indices": [int(i) for i in nn[disallowed_mask].index[:10]]},
                    plain_text=f"'{col}' contains values outside its configured allowed categories: {rare_cats} ({ratio_text(disallowed_count, len(nn), 'non-blank values')}).",
                    technical_text=f"Values not in allowed set {ctx.config.allowed_categories[col]}: count {disallowed_count}.",
                    fixes=[fix("set_unexpected_to_missing", f"Set unapproved values {rare_cats} to missing", "fix_invalid",
                               [col], rule="categories", unexpected=rare_cats, action="set_missing")],
                    breadth=1 / max(1, len(ctx.feature_columns)), effort="low"))
        elif sem in ("categorical", "binary") and 2 <= nn.nunique() <= 10:
            vc = nn.value_counts()
            total_nn = len(nn)
            if len(vc) > 2:
                top2_share = float(vc.iloc[:2].sum() / total_nn)
                # If top 2 categories make up >= 85% of rows, any 3rd+ category with frequency <= 5% or count <= 2 is unexpected
                # Exclude explicitly approved replacement categories (e.g. "Unknown" from safe imputation)
                if top2_share >= 0.85:
                    unapproved_mask = ~nn.isin(approved)
                    rare_mask = unapproved_mask & nn.isin(vc.index[2:]) & nn.map(vc).le(max(2, int(0.05 * total_nn)))
                    rare_count = int(rare_mask.sum())
                    if rare_count > 0:
                        rare_cats = sorted(set(nn[rare_mask]))
                        canonical = list(vc.index[:2])
                        out.append(Finding(
                            code="unexpected_categorical_value", category="quality",
                            severity="medium" if top2_share >= 0.95 else "low", confidence="high", columns=[col],
                            evidence={"unexpected_categories": rare_cats, "canonical_categories": canonical,
                                      "n_unexpected": rare_count, "n_nonblank": total_nn,
                                      "pct_unexpected": pct(rare_count, total_nn),
                                      "dominant_share_pct": round(top2_share * 100, 2),
                                      "denominator": f"{total_nn} non-blank values",
                                      "row_indices": [int(i) for i in nn[rare_mask].index[:10]]},
                            plain_text=f"'{col}' is {top2_share:.1%} {canonical[0]}/{canonical[1]}, but contains unexpected category values: {rare_cats} ({ratio_text(rare_count, total_nn, 'non-blank values')}). "
                                       "This is likely an invalid entry or stray response.",
                            technical_text=f"Dominant categories {canonical} represent {top2_share:.1%}; unexpected values {rare_cats} occur <= 5% (count {rare_count}).",
                            fixes=[fix("set_unexpected_to_missing", f"Set unexpected values {rare_cats} to missing", "fix_invalid",
                                       [col], rule="categories", unexpected=rare_cats, action="set_missing")],
                            breadth=1 / max(1, len(ctx.feature_columns)), effort="low"))
        elif nn.str.contains(r"^\s|\s$", regex=True).any():
            k = int(nn.str.contains(r"^\s|\s$", regex=True).sum())
            out.append(Finding(
                code="whitespace_issues", category="quality", severity="low", confidence="high", columns=[col],
                evidence={"n_with_whitespace": k, "n_nonblank": len(nn), "denominator": f"{len(nn)} non-blank values"},
                plain_text=f"'{col}' has leading or trailing spaces in {ratio_text(k, len(nn), 'non-blank values')}.",
                technical_text="Leading/trailing whitespace detected.",
                fixes=[fix("strip", "Trim spaces", "strip_whitespace", [col])],
                breadth=1 / max(1, len(ctx.feature_columns)), effort="low"))
        # encoding damage
        if nn.str.contains(MOJIBAKE).any():
            k = int(nn.str.contains(MOJIBAKE).sum())
            out.append(Finding(
                code="encoding_issues", category="quality", severity="low", confidence="medium", columns=[col],
                evidence={"n_affected": k, "n_nonblank": len(nn), "denominator": f"{len(nn)} non-blank values"},
                plain_text=f"'{col}' has garbled characters in {ratio_text(k, len(nn), 'non-blank values')}, a sign of a wrong text encoding on import.",
                technical_text="Pattern match for UTF-8-read-as-Latin-1 sequences or U+FFFD.",
                fixes=[fix("fix_encoding", "Repair text encoding", "fix_encoding", [col])],
                breadth=1 / max(1, len(ctx.feature_columns)), effort="low"))
        # inconsistent units
        m = nn.str.extract(UNIT_RE)
        if m[1].notna().mean() >= 0.9 and m[1].dropna().str.lower().nunique() >= 2:
            units = sorted(m[1].dropna().str.lower().unique())
            fam = next((f for f, tbl in UNIT_FAMILIES.items() if set(units) <= set(tbl)), None)
            out.append(Finding(
                code="inconsistent_units", category="quality", severity="medium", confidence="high", columns=[col],
                evidence={"units": units, "n_nonblank": len(nn), "convertible_family": fam, "denominator": f"{len(nn)} non-blank values"},
                plain_text=f"'{col}' mixes units ({', '.join(units)}), so the numbers are not comparable.",
                technical_text=f"Regex number+unit matched >=90% of values; units: {units}.",
                fixes=[fix("convert_units", "Convert to one unit", "convert_units", [col], family=fam, target_unit=units[0])]
                if fam else [fix("manual_units", "Review and convert units manually", "manual", [col])],
                breadth=1 / max(1, len(ctx.feature_columns)), effort="medium"))
        return out


@register
class MistypedColumnsCheck(Check):
    id = "mistyped_columns"
    category = "quality"
    depends_on = ("invalid_values",)

    def run(self, df, ctx: AnalysisContext) -> List[Finding]:
        out = []
        for col in ctx.feature_columns:
            sem, flags = ctx.semantic(col), ctx.flags(col)
            if col in ctx.shared.get("disguised_cols", set()):
                continue      # same problem already reported once (placeholders in a numeric column)
            if "stored_as_text" in flags:
                to = "datetime" if sem == "datetime" else "numeric"
                out.append(Finding(
                    code="mistyped_column", category="quality", severity="low", confidence="high", columns=[col],
                    evidence={"detected_type": sem, "stored_as": str(df[col].dtype), "to": to},
                    plain_text=f"'{col}' holds {sem if sem != 'continuous' else 'numbers'} stored as text, so models and statistics cannot use it directly.",
                    technical_text=f"dtype {df[col].dtype} but values parse as {sem}.",
                    fixes=[fix("cast", f"Convert to {to}", "parse_dates" if to == "datetime" else "cast_type", [col], to=to)],
                    breadth=1 / max(1, len(ctx.feature_columns)), effort="low"))
            elif sem == "identifier" and pd.api.types.is_float_dtype(df[col]):
                out.append(Finding(
                    code="id_stored_as_float", category="quality", severity="low", confidence="high", columns=[col],
                    evidence={"stored_as": str(df[col].dtype)},
                    plain_text=f"The ID column '{col}' is stored as decimals, which can corrupt long IDs.",
                    technical_text="Identifier-like column with float dtype.",
                    fixes=[fix("id_to_text", "Store as text", "cast_type", [col], to="string")],
                    breadth=1 / max(1, len(ctx.feature_columns)), effort="low"))
        return out
