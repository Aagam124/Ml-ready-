"""ML-risk checks: target quality, imbalance, leakage, redundancy, collinearity, shift, groups, cardinality."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from app.engine.checks.association import (category_purity, linear_strength, mutual_info_norm,
                                            single_feature_score)
from app.engine.context import AnalysisContext
from app.engine.findings import Finding, fix
from app.engine.registry import Check, register
from app.engine.utils import association, numeric_view, pct, ratio_text, safe_sample, tokens, ID_TOKENS_FOR_NAMES

NUMERIC_SEM = ("continuous", "count", "discrete", "currency", "percentage")
LEAK_HINTS = {"leak", "leaked", "leakage", "outcome", "result", "actual", "prediction", "predicted", "label", "target"}
GENERIC_TOKENS = {"is", "has", "flag", "id", "label", "target", "y", "the", "of", "a", "score", "value", "type"}
SENSITIVE = {"gender", "sex", "race", "ethnicity", "religion", "caste", "nationality", "marital", "disability",
             "pincode", "zipcode", "zip", "postal", "orientation", "age"}


def _share(n_cols: int, ctx: AnalysisContext) -> float:
    return n_cols / max(1, len(ctx.feature_columns))


# --------------------------------------------------------------------------- target
@register
class TargetQualityCheck(Check):
    id = "target_quality"
    category = "ml_risk"
    requires_target = True

    def run(self, df, ctx: AnalysisContext) -> List[Finding]:
        t, y = ctx.target, df[ctx.target]
        n = len(df)
        out: List[Finding] = []
        miss = y.isna()
        if ctx.is_classification or ctx.task_type == "regression":
            miss = miss | (y.astype("string").str.strip() == "").fillna(False)
        k = int(miss.sum())
        if k:
            share = pct(k, n)
            out.append(Finding(
                code="target_missing", category="ml_risk", severity="high" if share >= 5 else "medium", confidence="high",
                columns=[t], evidence={"n_missing": k, "n_rows": n, "pct": share, "denominator": f"{n} rows"},
                plain_text=f"The target '{t}' is empty in {ratio_text(k, n, 'rows')}. Those rows have no answer to learn from.",
                technical_text=f"Target NaN/blank count {k}/{n}.",
                fixes=[fix("drop_missing_target", "Remove rows without a target", "drop_missing_target", [t])],
                breadth=share / 100, effort="low"))
        labelled = y[~miss]
        if labelled.nunique() < 2:
            out.append(Finding(
                code="target_constant", category="ml_risk", severity="critical", confidence="high", columns=[t],
                evidence={"n_labelled": int(len(labelled)), "n_classes": int(labelled.nunique())},
                plain_text=f"The target '{t}' has only one value, so nothing can be learned.",
                technical_text="Fewer than 2 distinct non-missing target values.",
                fixes=[fix("choose_target", "Pick a different target column", "manual", [t])], breadth=1.0, effort="high"))
        return out


@register
class ClassImbalanceCheck(Check):
    id = "class_imbalance"
    category = "ml_risk"
    requires_target = True
    depends_on = ("target_quality",)

    def applies_to(self, ctx) -> bool:
        return ctx.has_target and ctx.is_classification

    def run(self, df, ctx: AnalysisContext) -> List[Finding]:
        y = df[ctx.target].dropna()
        counts = y.value_counts()
        if len(counts) < 2:
            return []
        n = int(counts.sum())
        major, minor = int(counts.iloc[0]), int(counts.iloc[-1])
        ratio = major / minor
        out: List[Finding] = []
        opts = fix("class_weights", "Use balanced class weights, stratified split, PR-AUC", "modeling_option", [ctx.target],
                   class_weight="balanced", stratify=True, primary_metric="pr_auc")
        if ratio >= 3:
            sev = "high" if ratio >= 20 else "medium" if ratio >= 5 else "low"
            out.append(Finding(
                code="class_imbalance", category="ml_risk", severity=sev, confidence="high", columns=[ctx.target],
                evidence={"ratio": round(ratio, 2), "minority_share_pct": pct(minor, n), "majority_class": str(counts.index[0]),
                          "minority_class": str(counts.index[-1]), "class_counts": {str(k): int(v) for k, v in counts.items()},
                          "n_labelled": n, "denominator": f"{n} labelled rows"},
                plain_text=f"Your target is {ratio:.0f}:1 imbalanced ({pct(minor, n)}% of {n} rows are '{counts.index[-1]}'). "
                           f"A model that always says '{counts.index[0]}' would score {pct(major, n)}% accuracy, so accuracy will mislead.",
                technical_text=f"majority/minority = {major}/{minor}. Prefer stratified splits, class weights, PR-AUC / balanced accuracy.",
                fixes=[opts], breadth=1.0, effort="low"))
        rare = counts[counts < 10]
        if len(counts) > 2 and len(rare):
            out.append(Finding(
                code="rare_classes", category="ml_risk", severity="medium", confidence="high", columns=[ctx.target],
                evidence={"rare_classes": {str(k): int(v) for k, v in rare.items()}, "n_labelled": n, "min_examples": 10},
                plain_text=f"{len(rare)} classes have fewer than 10 examples, too few to learn or to evaluate.",
                technical_text="Classes with <10 samples break stratified k-fold and give unstable metrics.",
                fixes=[fix("merge_rare", "Merge rare classes into 'other'", "merge_rare_classes", [ctx.target], min_count=10)],
                breadth=len(rare) / len(counts), effort="low"))
        elif len(counts) == 2 and minor < 10:
            out.append(Finding(
                code="rare_classes", category="ml_risk", severity="high", confidence="high", columns=[ctx.target],
                evidence={"rare_classes": {str(counts.index[-1]): minor}, "n_labelled": n},
                plain_text=f"Only {minor} rows are '{counts.index[-1]}'. That is too few to learn from or test on.",
                technical_text="Minority class < 10 samples.", fixes=[fix("collect_more", "Collect more minority examples", "manual", [ctx.target])],
                breadth=1.0, effort="high"))
        return out


# --------------------------------------------------------------------------- leakage
@register
class LeakageCheck(Check):
    id = "target_leakage"
    category = "leakage"
    requires_target = True
    depends_on = ("target_quality", "invalid_values")

    def run(self, df, ctx: AnalysisContext) -> List[Finding]:
        cfg = ctx.config
        t = ctx.target
        sub = safe_sample(df, cfg.ml_check_max_rows, cfg.seed)
        y = sub[t]
        task = ctx.task_type
        target_toks = tokens(t) - GENERIC_TOKENS
        out: List[Finding] = []
        likely: List[str] = []
        for col in ctx.feature_columns:
            sem = ctx.semantic(col)
            if sem in ("identifier", "constant", "empty", "text", "email", "url", "phone"):
                continue
            x = sub[col]
            if sem == "datetime":
                dt = ctx.datetimes(col).loc[sub.index]
                x, numeric = (dt.astype("int64").where(dt.notna()) / 1e9), True
            else:
                numeric = sem in NUMERIC_SEM or pd.api.types.is_numeric_dtype(x)
                x = numeric_view(x) if numeric else x
            sig: Dict[str, Any] = {}
            model = single_feature_score(x, y, task, numeric, cfg.seed)
            corr = linear_strength(x, y, task, numeric)
            mi = mutual_info_norm(x, y, task, numeric)
            sig["single_feature_score"] = None if model is None else round(model, 4)
            sig["association"] = round(corr, 4)
            sig["mutual_info_norm"] = round(mi, 4)
            model_thr = cfg.leakage_model_auc if task == "binary" else 0.90 if task == "multiclass" else 0.95
            strong = {"model": model is not None and model >= model_thr,
                      "association": corr >= cfg.leakage_corr, "mutual_info": mi >= cfg.leakage_mi}
            deterministic = False
            if model is not None and model >= 0.9995:
                deterministic = True
            if task != "regression" and x.nunique() <= 50 and x.nunique() <= 0.2 * max(1, len(x)):
                p = category_purity(x, y)
                sig["purity"] = round(p["purity"], 4)
                deterministic = deterministic or (p["purity"] >= 0.995 and p["n_values"] >= 2)
            if task == "regression" and corr >= 0.9995:
                deterministic = True
            hints = sorted((tokens(col) & LEAK_HINTS) | (tokens(col) & target_toks))
            n_strong = sum(strong.values())
            if deterministic or n_strong >= 2:
                tier, sev, conf = "Likely", ("critical" if deterministic or (model or 0) >= 0.99 else "high"), "high"
                likely.append(col)
            elif n_strong == 1:
                tier, sev, conf = "Possible", "medium", "medium" if hints else "low"
            elif hints:
                tier, sev, conf = "Possible", "low", "low"      # name-only: weak, can never be high severity
            else:
                continue
            name_only = n_strong == 0 and not deterministic
            out.append(Finding(
                code="target_leakage", category="leakage", severity=sev, confidence=conf, columns=[col],
                evidence={"tier": tier, "signals": sig, "strong_signals": [k for k, v in strong.items() if v],
                          "deterministic": deterministic, "name_hint_tokens": hints, "name_only": name_only,
                          "n_rows_scored": int(len(sub)), "denominator": f"{len(sub)} rows scored"},
                plain_text=(f"'{col}' {'almost perfectly gives away' if tier == 'Likely' else 'may reveal'} the answer '{t}'. "
                            + ("Models using it will look brilliant in testing and fail in real use. "
                               if tier == "Likely" else "Check whether it is known before the outcome happens. ")
                            + ("Only its name raised this flag." if name_only else "")),
                technical_text=f"{tier} leakage. signals={sig}; deterministic={deterministic}; whole-token name hints={hints}.",
                fixes=[fix("drop_leak", f"Drop '{col}' (not available at prediction time)", "drop_columns", [col])],
                breadth=1.0, effort="low"))
        ctx.shared["leakage_likely"] = likely
        return out


# --------------------------------------------------------------------------- redundancy / collinearity
def _numeric_features(ctx: AnalysisContext, df: pd.DataFrame, exclude: Optional[set] = None) -> pd.DataFrame:
    cols = [c for c in ctx.columns_of(*NUMERIC_SEM) if c in df.columns and c not in (exclude or set())]
    return pd.DataFrame({c: numeric_view(df[c]) for c in cols})


@register
class RedundancyCheck(Check):
    id = "redundant_columns"
    category = "ml_risk"

    def run(self, df, ctx: AnalysisContext) -> List[Finding]:
        out: List[Finding] = []
        groups: List[List[str]] = []
        num = _numeric_features(ctx, df)
        invalid = ctx.shared.get("invalid_masks", {})
        for c, m in invalid.items():
            if c in num:
                num.loc[m, c] = np.nan
        if num.shape[1] >= 2:
            corr = num.corr(min_periods=20).abs()
            seen: set = set()
            for c in corr.columns:
                if c in seen:
                    continue
                comp = {c} | {o for o in corr.columns if o != c and corr.loc[c, o] >= ctx.config.redundant_corr}
                if len(comp) > 1:
                    seen |= comp
                    groups.append([x for x in num.columns if x in comp])
        cats = [c for c in ctx.columns_of("categorical", "binary") if df[c].nunique() <= 50 and ctx.semantic(c) != "constant"]
        used = {c for g in groups for c in g}
        for i, a in enumerate(cats):
            for b in cats[i + 1:]:
                if a in used or b in used:
                    continue
                m = df[a].notna() & df[b].notna()
                if m.sum() >= 20:
                    tab = pd.crosstab(df.loc[m, a].astype(str), df.loc[m, b].astype(str))
                    if (tab.gt(0).sum(axis=1) == 1).all() and (tab.gt(0).sum(axis=0) == 1).all() and tab.shape[0] > 1:
                        groups.append([a, b])
                        used |= {a, b}
        redundant_drops: List[str] = []
        for g in groups:
            miss = {c: int(df[c].isna().sum()) for c in g}
            keep = min(g, key=lambda c: (miss[c], list(df.columns).index(c)))
            drop = [c for c in g if c != keep]
            redundant_drops += drop
            rel = {}
            exact_copy = False
            if all(c in num.columns for c in g) and len(g) == 2:
                m = num[g].dropna()
                if len(m) > 2:
                    slope, icpt = np.polyfit(m[keep], m[drop[0]], 1)
                    rel = {"relation": f"{drop[0]} = {slope:.6g} x {keep} + {icpt:.6g}"}
                    residuals = (m[drop[0]] - (slope * m[keep] + icpt)).abs()
                    exact_copy = bool((residuals < 1e-4).all())
            plain = (
                f"{', '.join(repr(c) for c in g)} carry identical information (one is an exact rescaled copy of the other). Keep '{keep}' and drop the rest."
                if exact_copy else
                f"{', '.join(repr(c) for c in g)} are strongly correlated (r >= {ctx.config.redundant_corr}). In small datasets, verify whether this relationship is genuine domain redundancy before dropping either column."
            )
            out.append(Finding(
                code="redundant_columns", category="ml_risk", severity="medium", confidence="high" if exact_copy else "medium", columns=g,
                evidence={"keep": keep, "drop": drop, "kind": "linear" if all(c in num.columns for c in g) else "one_to_one", **rel,
                          "exact_copy": exact_copy, "denominator": f"{len(df)} rows"},
                plain_text=plain,
                technical_text=f"|r| >= {ctx.config.redundant_corr} (or exact one-to-one mapping). {rel.get('relation', '')}",
                fixes=[fix("drop_redundant", f"Drop {', '.join(drop)}", "drop_columns", drop)],
                breadth=_share(len(g), ctx), effort="low"))
        ctx.shared["redundant_drops"] = redundant_drops
        return out


@register
class MulticollinearityCheck(Check):
    id = "multicollinearity"
    category = "ml_risk"
    depends_on = ("redundant_columns",)

    def run(self, df, ctx: AnalysisContext) -> List[Finding]:
        num = _numeric_features(ctx, df, exclude=set(ctx.shared.get("redundant_drops", [])))
        num = num.loc[:, num.nunique() > 1]
        if num.shape[1] < 2 or len(num) < 30:
            return []
        x = num.fillna(num.median())
        corr = np.corrcoef(x.to_numpy(float), rowvar=False)
        vifs = np.clip(np.diag(np.linalg.pinv(corr)), 1.0, 1e6)
        vif = dict(zip(num.columns, vifs, strict=False))
        high = {c: float(v) for c, v in vif.items() if v > ctx.config.vif_threshold}
        pairs = [(a, b, float(corr[i, j])) for i, a in enumerate(num.columns) for j, b in enumerate(num.columns)
                 if i < j and ctx.config.collinear_corr <= abs(corr[i, j])]
        if not high and not pairs:
            return []
        cols = sorted(set(high) | {p[0] for p in pairs} | {p[1] for p in pairs})
        drops = sorted({max((a, b), key=lambda c: vif.get(c, 0)) for a, b, _ in pairs} | ({max(high, key=high.get)} if high and not pairs else set()))
        worst = max(list(high.values()) + [1.0])
        return [Finding(
            code="multicollinearity", category="ml_risk", severity="medium" if worst >= 20 or any(abs(p[2]) >= 0.98 for p in pairs) else "low",
            confidence="high", columns=cols,
            evidence={"vif": {c: round(v, 2) for c, v in high.items()}, "correlated_pairs": [{"a": a, "b": b, "r": round(r, 4)} for a, b, r in pairs],
                      "vif_threshold": ctx.config.vif_threshold, "n_rows": len(num), "denominator": f"{len(num)} rows, medians imputed"},
            plain_text=f"{len(cols)} columns overlap heavily ({', '.join(cols[:3])}...). Linear models become unstable and importances misleading.",
            technical_text=f"VIF > {ctx.config.vif_threshold} for {sorted(high)}; |r| >= {ctx.config.collinear_corr} for {len(pairs)} pairs.",
            fixes=[fix("drop_collinear", f"Drop {', '.join(drops)}", "drop_columns", drops)] if drops else [],
            breadth=_share(len(cols), ctx), effort="medium")]


# --------------------------------------------------------------------------- shift and groups
def _adversarial_auc(a: pd.DataFrame, b: pd.DataFrame, cols: List[str], seed: int):
    both = pd.concat([a[cols], b[cols]], ignore_index=True)
    X = pd.DataFrame(index=both.index)
    for c in cols:
        s = both[c]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            X[c] = s.astype(float).fillna(s.median())
        else:
            X[c] = pd.factorize(s.astype(str))[0]
    y = np.r_[np.zeros(len(a)), np.ones(len(b))]
    rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=seed, n_jobs=1)
    pred = cross_val_predict(rf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed), method="predict_proba")[:, 1]
    rf.fit(X, y)
    imp = sorted(zip(cols, rf.feature_importances_, strict=False), key=lambda t: -t[1])[:3]
    return float(roc_auc_score(y, pred)), [(c, round(float(v), 3)) for c, v in imp]


@register
class SplitShiftCheck(Check):
    id = "split_shift"
    category = "ml_risk"
    requires_target = True

    def run(self, df, ctx: AnalysisContext) -> List[Finding]:
        cfg = ctx.config
        feats = [c for c in ctx.feature_columns if ctx.semantic(c) not in ("identifier", "constant", "empty", "text", "email", "url", "phone", "datetime")]
        if len(feats) < 1:
            return []
        out = []
        if ctx.test_df is not None and len(ctx.test_df) >= 30:
            cols = [c for c in feats if c in ctx.test_df.columns]
            auc, top = _adversarial_auc(safe_sample(df, cfg.ml_check_max_rows, cfg.seed), safe_sample(ctx.test_df, cfg.ml_check_max_rows, cfg.seed), cols, cfg.seed)
            f = self._finding("train_test_shift", auc, top, "your training and test files", ctx, len(df), len(ctx.test_df))
            if f:
                out.append(f)
        for dtc in ctx.datetime_columns[:1]:
            dt = ctx.datetimes(dtc)
            if dt.notna().mean() < 0.9 or len(df) < 100:
                continue
            order = dt.sort_values().index
            cut = int(len(order) * 0.8)
            early, late = df.loc[order[:cut]], df.loc[order[cut:]]
            auc, top = _adversarial_auc(safe_sample(early, cfg.ml_check_max_rows, cfg.seed), safe_sample(late, cfg.ml_check_max_rows, cfg.seed), feats, cfg.seed)
            f = self._finding("temporal_shift", auc, top, f"the earliest 80% and latest 20% of rows by '{dtc}'", ctx, len(early), len(late))
            if f:
                out.append(f)
        return out

    @staticmethod
    def _finding(code, auc, top, what, ctx, n_a, n_b) -> Optional[Finding]:
        if auc < 0.65:
            return None
        sev = "high" if auc >= 0.9 else "medium" if auc >= 0.75 else "low"
        return Finding(
            code=code, category="ml_risk", severity=sev, confidence="medium", columns=[c for c, _ in top],
            evidence={"adversarial_auc": round(auc, 4), "n_train_side": n_a, "n_test_side": n_b, "top_drifting_features": top,
                      "denominator": f"{n_a} vs {n_b} rows"},
            plain_text=f"A classifier can tell {what} apart (AUC {auc:.2f}; 0.50 means identical). "
                       f"The data changes over time or between files, led by {top[0][0]}, so test scores will look better than real life.",
            technical_text=f"Adversarial validation, RandomForest 5-fold CV AUC={auc:.3f}. Top importances: {top}.",
            fixes=[fix("time_split", "Evaluate with a time-based split", "modeling_option", [], split="time"),
                   fix("drop_drifting", f"Drop '{top[0][0]}' if it only tracks time", "drop_columns", [top[0][0]])],
            breadth=1.0, effort="medium")


@register
class RepeatedIdsCheck(Check):
    id = "repeated_entity_ids"
    category = "ml_risk"
    depends_on = ("duplicates",)

    def run(self, df, ctx: AnalysisContext) -> List[Finding]:
        out = []
        n = len(df)
        for col in ctx.feature_columns:
            s = df[col]
            name_id = bool(tokens(col) & ID_TOKENS_FOR_NAMES)
            if not name_id or ctx.semantic(col) in ("constant", "empty", "continuous"):
                continue
            nn = s.dropna()
            if len(nn) < 20:
                continue
            vc = nn.value_counts()
            repeated = vc[vc > 1]
            if repeated.empty or vc.size / len(nn) > 0.95 and len(repeated) < 3:
                continue
            rows_in = int(repeated.sum())
            # identical rows are already reported as exact duplicates; look for repeated IDs with different content
            dup_ids = df[s.isin(repeated.index)]
            non_exact = int((~dup_ids.duplicated(keep=False)).sum())
            if non_exact < max(3, int(0.01 * n)) and len(repeated) / vc.size < 0.05:
                continue
            out.append(Finding(
                code="repeated_entity_ids", category="ml_risk", severity="medium", confidence="high" if non_exact else "medium",
                columns=[col],
                evidence={"n_ids_repeated": int(len(repeated)), "n_distinct_ids": int(vc.size), "n_rows_in_repeated_ids": rows_in,
                          "n_rows": n, "mean_rows_per_repeated_id": round(float(repeated.mean()), 2),
                          "denominator": f"{n} rows"},
                plain_text=f"{ratio_text(rows_in, n, 'rows')} share an ID in '{col}' with other rows. If the same {col} lands in both training and test data, scores will be too optimistic.",
                technical_text="Repeated entity IDs imply grouped observations; use GroupKFold on this column.",
                fixes=[fix("group_split", f"Split by '{col}' so groups never straddle train/test", "modeling_option", [col], split="group", group_column=col)],
                breadth=pct(rows_in, n) / 100, effort="low"))
        return out


@register
class TemporalOrderingCheck(Check):
    id = "temporal_ordering"
    category = "ml_risk"
    requires_target = True

    def run(self, df, ctx: AnalysisContext) -> List[Finding]:
        for col in ctx.datetime_columns[:1]:
            dt = ctx.datetimes(col)
            if dt.notna().mean() < 0.9 or len(df) < 100:
                continue
            drift = None
            y = df[ctx.target]
            q = pd.qcut(dt.rank(method="first"), 4, labels=False)
            try:
                if ctx.is_classification:
                    tab = pd.crosstab(q, y.astype(str))
                    p = stats.chi2_contingency(tab)[1]
                else:
                    p = stats.kruskal(*[y[q == i].dropna() for i in range(4)])[1]
                drift = float(p)
            except Exception:
                drift = None
            drifting = drift is not None and drift < 0.001
            return [Finding(
                code="temporal_ordering", category="ml_risk", severity="low" if drifting else "info", confidence="medium" if drifting else "low",
                columns=[col],
                evidence={"target_changes_over_time": drifting, "p_value": drift, "n_dates": int(dt.notna().sum()), "denominator": f"{int(dt.notna().sum())} dated rows"},
                plain_text=f"'{col}' orders your rows in time" + (" and the target shifts over time" if drifting else "") +
                           ". Predicting the future from the past needs a time-based split, not a random one.",
                technical_text="Datetime column present; target distribution compared across time quartiles (chi2 / Kruskal-Wallis).",
                fixes=[fix("time_split", "Use a time-based train/test split", "modeling_option", [col], split="time", time_column=col)],
                breadth=1.0, effort="low")]
        return []


# --------------------------------------------------------------------------- features
@register
class CardinalityCheck(Check):
    id = "high_cardinality"
    category = "ml_risk"

    def run(self, df, ctx: AnalysisContext) -> List[Finding]:
        out = []
        n = len(df)
        for col in ctx.feature_columns:
            sem, k = ctx.semantic(col), int(df[col].nunique())
            if sem == "identifier":
                out.append(Finding(
                    code="identifier_feature", category="ml_risk", severity="low", confidence="high", columns=[col],
                    evidence={"n_unique": k, "n_rows": n, "denominator": f"{n} rows"},
                    plain_text=f"'{col}' is an ID (almost every value is unique). Models memorise IDs instead of learning patterns.",
                    technical_text="Identifier-semantic column used as a feature.",
                    fixes=[
                        fix("exclude_feature", f"Exclude '{col}' from training features (retained in dataset)", "modeling_option", [col], exclude_columns=[col]),
                        fix("drop_id", f"Delete '{col}' completely from dataset", "drop_columns", [col]),
                    ],
                    breadth=_share(1, ctx), effort="low"))
            elif sem == "categorical" and k > 50:
                out.append(Finding(
                    code="high_cardinality", category="ml_risk", severity="low", confidence="high", columns=[col],
                    evidence={"n_unique": k, "n_rows": n, "denominator": f"{n} rows"},
                    plain_text=f"'{col}' has {k} distinct categories. One-hot encoding would add {k} columns; use frequency or target encoding instead.",
                    technical_text=f"Cardinality {k} > 50.",
                    fixes=[fix("freq_encode", "Frequency-encode", "encode", [col], method="frequency"),
                           fix("target_encode", "Target-encode (cross-validated)", "encode", [col], method="target")],
                    breadth=_share(1, ctx), effort="medium"))
            elif sem == "text":
                out.append(Finding(
                    code="free_text_column", category="ml_risk", severity="info", confidence="high", columns=[col],
                    evidence={"avg_length": ctx.col_profile[col].get("stats", {}).get("avg_length")},
                    plain_text=f"'{col}' is free text. Standard models cannot use it as is; use text features (e.g. TF-IDF) or drop it.",
                    technical_text="Semantic type 'text'.", fixes=[fix("drop_text", f"Drop '{col}'", "drop_columns", [col])],
                    breadth=_share(1, ctx), effort="medium"))
        return out


@register
class DimensionalityCheck(Check):
    id = "dimensionality"
    category = "ml_risk"

    def run(self, df, ctx: AnalysisContext) -> List[Finding]:
        n = len(df)
        use = [c for c in ctx.feature_columns if ctx.semantic(c) not in ("identifier", "constant", "empty", "text")]
        p = sum(min(int(df[c].nunique()), 50) if ctx.semantic(c) in ("categorical",) else 1 for c in use)
        causes = []
        if p > n:
            sev = "high"
            causes.append("more features than rows")
        elif n / max(1, p) < 5:
            sev = "medium"
            causes.append("fewer than 5 rows per feature")
        elif n / max(1, p) < 10:
            sev = "low"
            causes.append("fewer than 10 rows per feature")
        else:
            sev = None
        if n < 100:
            sev = "medium" if sev in (None, "low") else sev
            causes.append(f"only {n} rows")
        if sev is None:
            return []
        return [Finding(
            code="dimensionality_risk", category="ml_risk", severity=sev, confidence="medium", columns=[],
            evidence={"n_rows": n, "n_features_after_encoding": p, "rows_per_feature": round(n / max(1, p), 2), "causes": causes,
                      "denominator": f"{n} rows, {p} features after one-hot encoding"},
            plain_text=f"You have {n} rows for about {p} features ({', '.join(causes)}). Models will overfit and scores will be unstable.",
            technical_text=f"rows/features = {n / max(1, p):.1f}. Features counted after one-hot encoding (categories capped at 50).",
            fixes=[fix("reduce_features", "Drop weak or redundant columns, or collect more rows", "manual")],
            breadth=1.0, effort="high")]


@register
class SensitiveAttributesCheck(Check):
    id = "sensitive_attributes"
    category = "ml_risk"

    def run(self, df, ctx: AnalysisContext) -> List[Finding]:
        sens = [c for c in ctx.feature_columns if tokens(c) & SENSITIVE and ctx.semantic(c) not in ("constant", "empty")]
        if not sens:
            return []
        sub = safe_sample(df, 5000, ctx.config.seed)
        candidates = [c for c in ctx.feature_columns if ctx.semantic(c) not in ("identifier", "constant", "empty", "text", "email", "url", "phone", "datetime")][:40]
        out = []
        for s in sens:
            sn = pd.api.types.is_numeric_dtype(sub[s]) and sub[s].nunique() > 10
            proxies = []
            for o in [c for c in candidates if c != s]:
                on = pd.api.types.is_numeric_dtype(sub[o]) and sub[o].nunique() > 10
                a = association(sub[s] if sn else sub[s].astype("string"), sub[o] if on else sub[o].astype("string"), sn, on)
                if a >= 0.7:
                    proxies.append({"column": o, "association": round(a, 3)})
            out.append(Finding(
                code="sensitive_attribute", category="ml_risk", severity="info", confidence="medium", columns=[s],
                evidence={"proxies": proxies, "denominator": f"{len(sub)} rows"},
                plain_text=f"'{s}' looks like a sensitive attribute." + (f" {', '.join(p['column'] for p in proxies[:3])} track it closely and could act as a stand-in." if proxies else "")
                           + " Check that using it is fair and legal for your purpose.",
                technical_text="Whole-token name match against a sensitive-attribute vocabulary; proxies: association >= 0.7.",
                fixes=[fix("review_sensitive", "Review fairness impact (no automatic change)", "manual", [s])], breadth=_share(1, ctx), effort="medium"))
        return out
