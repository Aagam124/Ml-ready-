"""Feature-vs-target association signals used by leakage detection."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import balanced_accuracy_score, mutual_info_score, r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from app.engine.utils import bin_series, cramers_v


def single_feature_score(x: pd.Series, y: pd.Series, task: str, is_numeric: bool, seed: int = 42, folds: int = 3) -> Optional[float]:
    """Out-of-fold predictive power of ONE feature. binary: ROC-AUC; multiclass: chance-corrected
    balanced accuracy; regression: R^2 (floored at 0). High-cardinality categories score ~chance."""
    m = x.notna() & y.notna()
    x, y = x[m].reset_index(drop=True), y[m].reset_index(drop=True)
    if len(x) < 30 or x.nunique() < 2:
        return None
    yy = y.astype(float) if task == "regression" else pd.Series(pd.factorize(y)[0])
    if task != "regression" and yy.value_counts().min() < folds:
        return None
    splitter = (KFold(folds, shuffle=True, random_state=seed) if task == "regression"
                else StratifiedKFold(folds, shuffle=True, random_state=seed))
    pred = np.zeros(len(x)) if task != "multiclass" else np.zeros(len(x), dtype=int)
    codes = None if is_numeric else pd.Series(pd.factorize(x.astype(str))[0])
    k = int(yy.nunique())
    for tr, te in splitter.split(x, yy if task != "regression" else None):
        if is_numeric:
            xtr, xte = x.iloc[tr].to_numpy(float).reshape(-1, 1), x.iloc[te].to_numpy(float).reshape(-1, 1)
            if task == "regression":
                mdl = DecisionTreeRegressor(max_depth=5, min_samples_leaf=5, random_state=seed).fit(xtr, yy.iloc[tr])
                pred[te] = mdl.predict(xte)
            else:
                mdl = DecisionTreeClassifier(max_depth=5, min_samples_leaf=5, random_state=seed).fit(xtr, yy.iloc[tr])
                pred[te] = mdl.predict_proba(xte)[:, 1] if task == "binary" else mdl.predict(xte)
        else:
            ctr, cte = codes.iloc[tr].to_numpy(), codes.iloc[te].to_numpy()
            ytr = yy.iloc[tr].to_numpy()
            if task == "multiclass":
                tab = pd.crosstab(ctr, ytr)
                best = tab.idxmax(axis=1)
                fallback = int(pd.Series(ytr).mode().iloc[0])
                pred[te] = [int(best.get(c, fallback)) for c in cte]
            else:
                prior = ytr.mean()
                g = pd.DataFrame({"c": ctr, "y": ytr}).groupby("c")["y"].agg(["sum", "count"])
                smooth = (g["sum"] + prior * 2) / (g["count"] + 2)   # additive smoothing, m=2
                pred[te] = [float(smooth.get(c, prior)) for c in cte]
    try:
        if task == "binary":
            return float(roc_auc_score(yy, pred))
        if task == "multiclass":
            return float(max(0.0, (balanced_accuracy_score(yy, pred) - 1 / k) / (1 - 1 / k)))
        return float(max(0.0, r2_score(yy, pred)))
    except ValueError:
        return None


def linear_strength(x: pd.Series, y: pd.Series, task: str, is_numeric: bool) -> float:
    """|Spearman| (numeric feature) or Cramer's V / eta (categorical)."""
    m = x.notna() & y.notna()
    if m.sum() < 10:
        return 0.0
    if is_numeric:
        yv = y[m].astype(float) if task == "regression" else pd.Series(pd.factorize(y[m])[0], index=y[m].index)
        if task == "multiclass":
            # eta from class membership
            means = x[m].astype(float).groupby(yv).transform("mean")
            tot = ((x[m] - x[m].mean()) ** 2).sum()
            return float(np.sqrt(max(0.0, 1 - ((x[m] - means) ** 2).sum() / tot))) if tot else 0.0
        if x[m].nunique() < 2 or yv.nunique() < 2:
            return 0.0
        return float(abs(stats.spearmanr(x[m], yv)[0]))
    if task == "regression":
        yb = bin_series(y[m].astype(float), 10)
        return cramers_v(x[m].astype(str), yb)
    return cramers_v(x[m].astype(str), y[m].astype(str))


def mutual_info_norm(x: pd.Series, y: pd.Series, task: str, is_numeric: bool) -> float:
    """Mutual information divided by the target's entropy (0..1)."""
    m = x.notna() & y.notna()
    if m.sum() < 10:
        return 0.0
    xs = bin_series(x[m].astype(float), 10) if is_numeric else pd.Series(pd.factorize(x[m].astype(str))[0], index=x[m].index)
    ys = bin_series(y[m].astype(float), 10) if task == "regression" else pd.Series(pd.factorize(y[m])[0], index=y[m].index)
    h = float(-(ys.value_counts(normalize=True) * np.log(ys.value_counts(normalize=True))).sum())
    if h == 0:
        return 0.0
    # Unseen-per-row categories give inflated MI; discount by the number of cells vs rows.
    if xs.nunique() > 0.5 * len(xs):
        return 0.0
    return float(min(1.0, mutual_info_score(xs, ys) / h))


def category_purity(x: pd.Series, y: pd.Series) -> Dict[str, float]:
    """Share of rows whose class equals the majority class of their feature value (1.0 = deterministic)."""
    m = x.notna() & y.notna()
    tab = pd.crosstab(x[m].astype(str), y[m].astype(str))
    return {"purity": float(tab.max(axis=1).sum() / tab.values.sum()) if tab.values.sum() else 0.0, "n_values": int(tab.shape[0])}
