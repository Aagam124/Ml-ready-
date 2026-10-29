"""Outliers: IQR + robust z-score (MAD) + Isolation Forest. Report only on agreement or extremes."""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from app.engine.context import AnalysisContext
from app.engine.findings import Finding, fix
from app.engine.registry import Check, register
from app.engine.utils import numeric_view, pct, ratio_text

NUMERIC = ("continuous", "count", "discrete", "currency", "percentage")


@register
class OutliersCheck(Check):
    id = "outliers"
    category = "outliers"
    depends_on = ("invalid_values", "missing_values", "target_leakage")

    def run(self, df, ctx: AnalysisContext) -> List[Finding]:
        cols = [c for c in ctx.columns_of(*NUMERIC) if c in df.columns]
        if not cols:
            return []
        invalid = ctx.shared.get("invalid_masks", {})
        num = pd.DataFrame({c: numeric_view(df[c]) for c in cols})
        for c, m in invalid.items():       # impossible values are errors, already reported once
            if c in num:
                num.loc[m, c] = np.nan
        if_flags = self._isolation_forest(num, ctx)
        out: List[Finding] = []
        covered = pd.Series(False, index=df.index)
        skip = set(ctx.shared.get("leakage_likely", []))   # leaked columns get dropped; not worth outlier noise
        for c in cols:
            if c in skip:
                continue
            v = num[c]
            nn = v.dropna()
            if len(nn) < 15 or nn.nunique() < 5:
                continue
            work = np.log1p(nn) if (nn.min() > 0 and abs(nn.skew()) > 1.5) else nn
            log_scale = work is not nn
            q1, q3 = work.quantile(0.25), work.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            iqr_flag = (work < q1 - 1.5 * iqr) | (work > q3 + 1.5 * iqr)
            extreme = (work < q1 - 3 * iqr) | (work > q3 + 3 * iqr)
            med = work.median()
            mad = (work - med).abs().median()
            mad_flag = ((0.6745 * (work - med).abs() / mad) > 3.5) if mad > 0 else pd.Series(False, index=work.index)
            agree = iqr_flag & mad_flag
            flagged = agree | extreme
            k = int(flagged.sum())
            if k == 0 or (pct(k, len(nn)) < ctx.config.outlier_min_rate_pct and int(extreme.sum()) == 0):
                continue
            n_if = int((flagged & if_flags.reindex(nn.index).fillna(False)).sum())
            rate = pct(k, len(nn))
            if rate > 10:      # that many "outliers" is a distribution shape (bimodal/heavy-tailed), not stray points
                continue
            covered.loc[flagged[flagged].index] = True
            out.append(Finding(
                code="outliers", category="outliers", severity="medium" if rate >= 5 else "low",
                confidence="high" if (agree.sum() and n_if) else "medium", columns=[c],
                evidence={"n_outliers": k, "n_values": int(len(nn)), "pct": rate,
                          "denominator": f"{len(nn)} non-missing values", "n_iqr": int(iqr_flag.sum()),
                          "n_mad": int(mad_flag.sum()), "n_both_iqr_and_mad": int(agree.sum()),
                          "n_extreme_3iqr": int(extreme.sum()), "n_also_isolation_forest": n_if,
                          "evaluated_on_log_scale": log_scale,
                          "bounds": [round(float(np.expm1(q1 - 1.5 * iqr) if log_scale else q1 - 1.5 * iqr), 4),
                                     round(float(np.expm1(q3 + 1.5 * iqr) if log_scale else q3 + 1.5 * iqr), 4)]},
                plain_text=f"'{c}' has {ratio_text(k, len(nn), 'non-missing values')} far outside the usual range. "
                           "Check whether they are real before changing them.",
                technical_text=f"IQR(1.5x) and MAD(z>3.5) agree on {int(agree.sum())}; 3x-IQR extremes: {int(extreme.sum())}; "
                               f"Isolation Forest agrees on {n_if}." + (" Evaluated on log1p scale (skewed positive data)." if log_scale else ""),
                fixes=[fix("winsorize", "Clip to the 1st-99th percentile", "winsorize", [c], lower_q=0.01, upper_q=0.99)],
                breadth=1 / max(1, len(ctx.feature_columns)), effort="low"))
        # Multivariate-only outliers: rows Isolation Forest flags that no single column explains.
        only_mv = if_flags & ~covered
        k = int(only_mv.sum())
        if len(cols) >= 3 and k >= max(5, int(0.02 * len(df))):
            out.append(Finding(
                code="multivariate_outliers", category="outliers", severity="low", confidence="low", columns=[],
                evidence={"n_rows_flagged": k, "n_rows": len(df), "denominator": f"{len(df)} rows",
                          "method": "IsolationForest(contamination=auto); medians imputed so all rows are scored"},
                plain_text=f"{ratio_text(k, len(df), 'rows')} look unusual only when columns are combined.",
                technical_text="Isolation Forest on median-imputed numeric columns, all rows scored; rows already flagged by a single column excluded.",
                fixes=[], breadth=pct(k, len(df)) / 100, effort="high"))
        return out

    @staticmethod
    def _isolation_forest(num: pd.DataFrame, ctx: AnalysisContext) -> pd.Series:
        """Every row is scored: missing cells are median-imputed, never dropped."""
        if num.shape[1] < 1 or len(num) < 50:
            return pd.Series(False, index=num.index)
        x = num.fillna(num.median()).fillna(0.0)
        x = x.iloc[:, :50]
        model = IsolationForest(n_estimators=100, contamination="auto", random_state=ctx.config.seed, n_jobs=1)
        return pd.Series(model.fit_predict(x) == -1, index=num.index)
