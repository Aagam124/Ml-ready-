"""Honest before/after modeling.

* Same split and same CV folds for every arm (folds are defined on ORIGINAL row ids).
* Preprocessing (impute / scale / encode) is fitted inside each CV fold, on training rows only.
* Likely-leakage columns are excluded from the "after" arm and a leak-free "before" arm is added so the
  effect of the leak is measured separately from the effect of cleaning. A leaky 99% is called out, not celebrated.
* Metrics come with 95% intervals: t-intervals across CV folds, bootstrap intervals on the held-out test rows.
"""
from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score)
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from app.cleaning import runtime as rt
from app.profiling.profile import profile_dataframe

CLF_METRICS = ("accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")
REG_METRICS = ("mae", "rmse", "r2")
LOWER_IS_BETTER = {"mae", "rmse"}
UNUSABLE = {"identifier", "email", "url", "text", "phone", "empty", "constant"}
MAX_TRAIN_ROWS = 30_000
N_FOLDS = 5
N_BOOT = 200
MIN_PRACTICAL_DELTA = 0.01
Progress = Callable[[float, str], None]


# ------------------------------------------------------------------------------------ data prep
def _usable_columns(df: pd.DataFrame, target: str, forced_out: List[str]) -> List[str]:
    prof = profile_dataframe(df.drop(columns=[target]))
    return [c["name"] for c in prof["columns"] if c["semantic_type"] not in UNUSABLE and c["name"] not in forced_out]


def _feature_frame(df: pd.DataFrame, cols: List[str]) -> Tuple[pd.DataFrame, List[str], List[str]]:
    x = df[cols].copy()
    for c in cols:
        if pd.api.types.is_datetime64_any_dtype(x[c]):
            x[c] = x[c].astype("int64").where(x[c].notna()) / 1e9
        elif pd.api.types.is_bool_dtype(x[c]):
            x[c] = x[c].astype(float)
    num = [c for c in cols if pd.api.types.is_numeric_dtype(x[c])]
    cat = [c for c in cols if c not in num]
    for c in cat:
        x[c] = x[c].astype("string").fillna("__missing__").astype(object)
    return x, num, cat


def _preprocessor(num: List[str], cat: List[str], scale: bool) -> ColumnTransformer:
    num_steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))] + ([("scale", StandardScaler())] if scale else [])
    parts = []
    if num:
        parts.append(("num", Pipeline(num_steps), num))
    if cat:
        parts.append(("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", max_categories=20, sparse_output=False), cat))
    return ColumnTransformer(parts, remainder="drop")


def _models(task: str, seed: int, class_weight: Optional[str]) -> List[Tuple[str, Any, bool, Dict[str, Any]]]:
    cw = class_weight if class_weight == "balanced" else None
    if task == "regression":
        return [("linear_regression", LinearRegression(), True, {}),
                ("random_forest", RandomForestRegressor(n_estimators=100, min_samples_leaf=2, random_state=seed, n_jobs=1), False, {"n_estimators": 100, "min_samples_leaf": 2}),
                ("gradient_boosting", HistGradientBoostingRegressor(max_iter=100, random_state=seed), False, {"max_iter": 100})]
    return [("logistic_regression", LogisticRegression(max_iter=1000, class_weight=cw, random_state=seed), True, {"max_iter": 1000, "class_weight": cw}),
            ("random_forest", RandomForestClassifier(n_estimators=100, min_samples_leaf=2, class_weight=cw, random_state=seed, n_jobs=1), False, {"n_estimators": 100, "min_samples_leaf": 2, "class_weight": cw}),
            ("gradient_boosting", HistGradientBoostingClassifier(max_iter=100, class_weight=cw, random_state=seed), False, {"max_iter": 100, "class_weight": cw})]


# ------------------------------------------------------------------------------------ metrics
def _metrics(task: str, y_true: np.ndarray, pred: np.ndarray, proba: Optional[np.ndarray], positive: int = 1) -> Dict[str, float]:
    if task == "regression":
        return {"mae": float(mean_absolute_error(y_true, pred)), "rmse": float(np.sqrt(mean_squared_error(y_true, pred))),
                "r2": float(r2_score(y_true, pred))}
    avg = "binary" if task == "binary" else "macro"
    kw = {"pos_label": positive} if task == "binary" else {}
    out = {"accuracy": float(accuracy_score(y_true, pred)), "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
           "precision": float(precision_score(y_true, pred, average=avg, zero_division=0, **kw)),
           "recall": float(recall_score(y_true, pred, average=avg, zero_division=0, **kw)),
           "f1": float(f1_score(y_true, pred, average=avg, zero_division=0, **kw))}
    try:
        if task == "binary":
            p = proba[:, 1 if positive == 1 else 0]
            out["roc_auc"], out["pr_auc"] = float(roc_auc_score(y_true == positive, p)), float(average_precision_score(y_true == positive, p))
        else:
            classes = np.arange(proba.shape[1])
            out["roc_auc"] = float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=classes))
            yb = (y_true[:, None] == classes[None, :]).astype(int)
            out["pr_auc"] = float(np.mean([average_precision_score(yb[:, k], proba[:, k]) for k in range(len(classes)) if yb[:, k].any()]))
    except ValueError:
        out["roc_auc"] = out["pr_auc"] = float("nan")
    return out


def _t_interval(values: List[float]) -> Dict[str, Any]:
    v = np.array([x for x in values if not np.isnan(x)], dtype=float)
    if len(v) == 0:
        return {"mean": None, "ci_low": None, "ci_high": None, "folds": []}
    if len(v) < 2:
        return {"mean": float(v[0]), "ci_low": None, "ci_high": None, "folds": [float(x) for x in v]}
    half = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return {"mean": float(v.mean()), "ci_low": float(v.mean() - half), "ci_high": float(v.mean() + half), "folds": [float(x) for x in v]}


# ------------------------------------------------------------------------------------ folds on ORIGINAL row ids
def define_folds(df: pd.DataFrame, is_train: pd.Series, target: str, task: str, spec: Dict[str, Any], seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Folds over the training rows of the ORIGINAL data, expressed as row labels (index values)."""
    tr = df.loc[is_train & df[target].notna()]
    idx = tr.index.to_numpy()
    y = tr[target]
    if spec.get("strategy") == "group" and spec.get("group_column") in tr.columns:
        groups = tr[spec["group_column"]].astype("string").fillna("__nan__")
        splits = GroupKFold(n_splits=min(N_FOLDS, groups.nunique())).split(tr, groups=groups)
    elif spec.get("strategy") == "time" and spec.get("time_column") in tr.columns:
        order = rt._parse_dates(tr[spec["time_column"]]).sort_values(kind="stable").index
        pos = pd.Series(np.arange(len(order)), index=order).loc[tr.index].to_numpy()
        ordered = np.argsort(pos)
        splits = ((ordered[a], ordered[b]) for a, b in TimeSeriesSplit(n_splits=min(N_FOLDS, 4)).split(ordered))
    elif task in ("binary", "multiclass") and y.astype("string").value_counts().min() >= N_FOLDS and spec.get("stratify", True):
        splits = StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed).split(tr, y.astype("string"))
    else:
        splits = KFold(N_FOLDS, shuffle=True, random_state=seed).split(tr)
    return [(idx[a], idx[b]) for a, b in splits]


# ------------------------------------------------------------------------------------ one arm
def run_arm(name: str, df: pd.DataFrame, target: str, task: str, folds: List[Tuple[np.ndarray, np.ndarray]],
            train_labels: np.ndarray, test_labels: np.ndarray, forced_out: List[str], seed: int, class_weight: Optional[str],
            progress: Optional[Progress] = None) -> Dict[str, Any]:
    df = df.loc[df[target].notna()]
    cols = _usable_columns(df, target, forced_out)
    x_all, num, cat = _feature_frame(df, cols)
    if not cols:
        raise ValueError("No usable feature columns remain.")
    y_raw = df[target]
    if task == "regression":
        y = pd.Series(pd.to_numeric(y_raw, errors="coerce"), index=df.index)
        classes: List[str] = []
        positive = 1
    else:
        tr_labels = df.index.intersection(train_labels)
        le = LabelEncoder().fit(y_raw.loc[tr_labels].astype(str))
        classes = list(le.classes_)
        known = y_raw.astype(str).isin(classes)
        y = pd.Series(np.where(known, le.transform(y_raw.astype(str).where(known, classes[0])), -1), index=df.index)
        counts = y.loc[tr_labels].value_counts()
        positive = int(counts.idxmin()) if task == "binary" else 1       # positive = minority class (the one that matters)
        df, x_all, y = df.loc[known], x_all.loc[known], y.loc[known]
    tr_all = df.index.intersection(train_labels)
    te = df.index.intersection(test_labels)
    if len(tr_all) > MAX_TRAIN_ROWS:
        tr_all = pd.Index(np.random.default_rng(seed).choice(tr_all, MAX_TRAIN_ROWS, replace=False))
    models_out: List[Dict[str, Any]] = []
    metric_names = REG_METRICS if task == "regression" else CLF_METRICS
    rng = np.random.default_rng(seed)
    for mi, (mname, est, scale, params) in enumerate(_models(task, seed, class_weight)):
        fold_scores: Dict[str, List[float]] = {m: [] for m in metric_names}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for a, b in folds:
                a, b = df.index.intersection(a).intersection(tr_all), df.index.intersection(b).intersection(tr_all)
                if len(a) < 10 or len(b) < 5 or (task != "regression" and y.loc[a].nunique() < 2):
                    continue
                pipe = Pipeline([("pre", _preprocessor(num, cat, scale)), ("model", _clone(est))]).fit(x_all.loc[a], y.loc[a])
                proba = pipe.predict_proba(x_all.loc[b]) if task != "regression" else None
                if proba is not None and proba.shape[1] != len(classes):
                    full = np.zeros((len(b), len(classes)))
                    full[:, pipe.classes_] = proba
                    proba = full
                for k, v in _metrics(task, y.loc[b].to_numpy(), pipe.predict(x_all.loc[b]), proba, positive).items():
                    fold_scores[k].append(v)
            final = Pipeline([("pre", _preprocessor(num, cat, scale)), ("model", _clone(est))]).fit(x_all.loc[tr_all], y.loc[tr_all])
            test: Dict[str, Any] = {}
            if len(te) >= 5:
                yt = y.loc[te].to_numpy()
                proba = final.predict_proba(x_all.loc[te]) if task != "regression" else None
                if proba is not None and proba.shape[1] != len(classes):
                    full = np.zeros((len(te), len(classes)))
                    full[:, final.classes_] = proba
                    proba = full
                pred = final.predict(x_all.loc[te])
                point = _metrics(task, yt, pred, proba, positive)
                boots: Dict[str, List[float]] = {m: [] for m in point}
                for _ in range(N_BOOT):
                    ii = rng.integers(0, len(yt), len(yt))
                    if task != "regression" and len(np.unique(yt[ii])) < 2:
                        continue
                    for k, v in _metrics(task, yt[ii], pred[ii], None if proba is None else proba[ii], positive).items():
                        boots[k].append(v)
                test = {k: {"value": v, "ci_low": float(np.nanpercentile(boots[k], 2.5)) if boots[k] else None,
                            "ci_high": float(np.nanpercentile(boots[k], 97.5)) if boots[k] else None} for k, v in point.items()}
        models_out.append({"name": mname, "params": params, "cv": {m: _t_interval(v) for m, v in fold_scores.items()}, "test": test})
        if progress:
            progress((mi + 1) / 3, f"{name}: trained {mname}")
    return {"arm": name, "n_train": int(len(tr_all)), "n_test": int(len(te)), "features_used": cols, "features_excluded": forced_out,
            "classes": classes, "positive_class": classes[positive] if task == "binary" and classes else None, "models": models_out}


def _clone(est):
    from sklearn.base import clone
    return clone(est)


# ------------------------------------------------------------------------------------ the comparison
def choose_primary(task: str, y: pd.Series, option: Optional[str]) -> str:
    if option and option in (REG_METRICS if task == "regression" else CLF_METRICS):
        return option
    if task == "regression":
        return "r2"
    if task == "binary":
        share = y.value_counts(normalize=True).min()
        return "pr_auc" if share < 0.2 else "roc_auc"
    return "balanced_accuracy"


def _paired(before_folds: List[float], after_folds: List[float], lower_better: bool) -> Dict[str, Any]:
    n = min(len(before_folds), len(after_folds))
    if n < 2:
        return {"mean_diff": None, "ci_low": None, "ci_high": None, "n_folds": n}
    d = np.array(after_folds[:n]) - np.array(before_folds[:n])
    if lower_better:
        d = -d                                   # positive always means "better"
    half = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n)
    return {"mean_diff": float(d.mean()), "ci_low": float(d.mean() - half), "ci_high": float(d.mean() + half), "n_folds": n}


def compare_arms(before: Dict[str, Any], after: Dict[str, Any], primary: str, leakage: List[str],
                 before_no_leak: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    lower = primary in LOWER_IS_BETTER
    per_model, notes = [], []
    for mb in before["models"]:
        ma = next((m for m in after["models"] if m["name"] == mb["name"]), None)
        if not ma:
            continue
        cb, ca = mb["cv"][primary], ma["cv"][primary]
        pr = _paired(cb["folds"], ca["folds"], lower)
        delta = None if cb["mean"] is None or ca["mean"] is None else ca["mean"] - cb["mean"]
        better = pr["mean_diff"] is not None and pr["ci_low"] > 0 and abs(delta) >= MIN_PRACTICAL_DELTA
        worse = pr["mean_diff"] is not None and pr["ci_high"] < 0 and abs(delta) >= MIN_PRACTICAL_DELTA
        per_model.append({"name": mb["name"], "before": cb["mean"], "after": ca["mean"], "delta": delta, "paired": pr,
                          "verdict": "improved" if better else "worse" if worse else "no_meaningful_change"})
    best = max(after["models"], key=lambda m: (m["cv"][primary]["mean"] if m["cv"][primary]["mean"] is not None else -1e9) * (-1 if lower else 1)) if after["models"] else None
    best_row = next((p for p in per_model if best and p["name"] == best["name"]), None)
    verdict = best_row["verdict"] if best_row else "no_meaningful_change"
    inflated = False
    if leakage and before_no_leak:
        nb = max((m["cv"][primary]["mean"] or -1e9) * (-1 if lower else 1) for m in before["models"]) * (-1 if lower else 1)
        nn = max((m["cv"][primary]["mean"] or -1e9) * (-1 if lower else 1) for m in before_no_leak["models"]) * (-1 if lower else 1)
        gap = (nb - nn) * (-1 if lower else 1)
        inflated = gap >= 0.03
        if inflated:
            notes.append(f"The 'before' score is inflated by leakage ({', '.join(leakage)}): {primary} {nb:.3f} with the leaked column(s), "
                         f"{nn:.3f} without. The honest starting point is {nn:.3f}.")
            verdict = "leakage_removed"
    if best and best["cv"][primary]["mean"] is not None and not lower and best["cv"][primary]["mean"] >= 0.98 and not leakage:
        notes.append(f"{primary} of {best['cv'][primary]['mean']:.3f} is unusually high. Check for leakage MLReady cannot see, such as a column that is only known after the outcome.")
    headline = {"improved": f"Cleaning measurably improved {primary} (paired across {best_row['paired']['n_folds'] if best_row else 0} identical folds).",
                "worse": f"{primary} got worse after cleaning. Review the steps before using this version.",
                "no_meaningful_change": f"No meaningful improvement in {primary}. The data was already usable for this task, or the issues do not affect these models.",
                "leakage_removed": "The headline score dropped because a leaked column was removed. That is the correct outcome: the old number was not real."}[verdict]
    return {"primary_metric": primary, "per_model": per_model, "best_model": best["name"] if best else None, "verdict": verdict,
            "headline": headline, "notes": notes, "leakage_inflated": inflated, "leakage_columns_excluded": leakage}


def run_before_after(df_before: pd.DataFrame, df_after: pd.DataFrame, target: str, task: str, options: Dict[str, Any],
                     leakage_columns: List[str], seed: int = 42, progress: Optional[Progress] = None) -> Dict[str, Any]:
    """df_before / df_after must share the original row index (cleaning preserves it)."""
    spec = {"strategy": options.get("split", "random"), "group_column": options.get("group_column"),
            "time_column": options.get("time_column"), "stratify": bool(options.get("stratify", True))}
    is_train = rt.make_split(df_before, target, task, seed, 0.2, spec["strategy"], spec["group_column"], spec["time_column"], spec["stratify"])
    labelled = df_before[target].notna()
    folds = define_folds(df_before, is_train, target, task, spec, seed)
    train_labels = df_before.index[is_train & labelled].to_numpy()
    test_before = df_before.index[~is_train & labelled]
    test_labels = test_before.intersection(df_after.index)          # identical test rows for every arm
    primary = choose_primary(task, df_before.loc[labelled, target], options.get("primary_metric"))
    cw = options.get("class_weight")
    present_leaks = [c for c in leakage_columns if c in df_before.columns]
    tick = lambda frac, msg: progress and progress(frac, msg)
    before = run_arm("before", df_before, target, task, folds, train_labels, test_labels, [], seed, cw, lambda f, m: tick(f * 0.33, m))
    no_leak = (run_arm("before_without_leakage", df_before, target, task, folds, train_labels, test_labels, present_leaks, seed, cw,
                       lambda f, m: tick(0.33 + f * 0.33, m)) if present_leaks else None)
    after = run_arm("after", df_after, target, task, folds, train_labels, test_labels, [c for c in present_leaks if c in df_after.columns],
                    seed, cw, lambda f, m: tick(0.66 + f * 0.34, m))
    comparison = compare_arms(before, after, primary, present_leaks, no_leak)
    arms = {"before": before, "after": after}
    if no_leak:
        arms["before_without_leakage"] = no_leak
    return {"task_type": task, "target": target, "seed": seed, "split": {**spec, "test_size": 0.2}, "arms": arms,
            "comparison": comparison, "common_test_rows": int(len(test_labels)), "n_folds": len(folds),
            "notes": ["Free text, URLs, e-mails and ID columns are excluded from every arm because no model can use them directly.",
                      "Imputation, scaling and one-hot encoding are fitted inside each fold on training rows only.",
                      "All arms are scored on the same held-out test rows; CV folds are defined once on the original row ids."]}
