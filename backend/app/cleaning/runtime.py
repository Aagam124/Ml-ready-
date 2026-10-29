# --- MLREADY RUNTIME BEGIN ---------------------------------------------------------------
# Self-contained cleaning helpers (pandas / NumPy / scikit-learn only). The MLReady engine
# and every exported script use exactly this code, so results are identical by construction.
# Stateless functions: fn(df, columns, **params) -> new DataFrame (row index is preserved).
# Stateful classes: .fit(train_df, target=None) learns ONLY from the training rows; .transform(df) applies.
import re

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.model_selection import GroupShuffleSplit, KFold, train_test_split

_CURRENCY = re.compile(r"[$€£¥₹,%\s]|USD|EUR|INR|GBP")
_PLACEHOLDERS = {"n/a", "na", "null", "none", "nan", "?", "-", "--", "missing", "#n/a", "undefined", "nil", "#na"}
_MOJIBAKE = re.compile(r"Ã.|â€.|Â.|\ufffd")
_UNIT_RE = re.compile(r"^\s*(-?\d+(?:[.,]\d+)?)\s*([A-Za-z°/]+)\s*$")
_UNITS = {"mass": {"kg": 1.0, "g": 0.001, "lb": 0.45359237, "lbs": 0.45359237, "oz": 0.028349523},
          "length": {"m": 1.0, "cm": 0.01, "mm": 0.001, "km": 1000.0, "in": 0.0254, "ft": 0.3048, "mi": 1609.344}}


def _present(df, columns):
    return [c for c in columns if c in df.columns]


def _is_texty(s):
    return s.dtype == object or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype)


def _to_numeric(s):
    """Float view of a column; text is stripped of currency symbols, commas, % and spaces."""
    if pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    cleaned = s.astype("string").str.replace(_CURRENCY, "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce").astype(float)


def _parse_dates(s):
    if pd.api.types.is_datetime64_any_dtype(s):
        parsed = s
    else:
        parsed = pd.to_datetime(s, errors="coerce", format="mixed", utc=True)
    try:
        return parsed.dt.tz_convert(None)
    except (TypeError, AttributeError):
        try:
            return parsed.dt.tz_localize(None)
        except (TypeError, AttributeError):
            return parsed


# ----------------------------------------------------------------------------- stateless steps
def strip_whitespace(df, columns):
    df = df.copy()
    for c in _present(df, columns):
        if _is_texty(df[c]):
            df[c] = df[c].astype("string").str.strip()
    return df


def normalize_categories(df, columns, mapping):
    df = df.copy()
    for c in _present(df, columns):
        df[c] = df[c].astype("string").map(lambda v: v if v is pd.NA or v != v else mapping.get(v, v)).astype("string")
    return df


def cast_type(df, columns, to):
    df = df.copy()
    for c in _present(df, columns):
        if to == "numeric":
            df[c] = _to_numeric(df[c])
        elif to == "string":
            s = df[c]
            if pd.api.types.is_float_dtype(s) and bool(((s.dropna() % 1) == 0).all()):
                df[c] = s.astype("Int64").astype("string")
            else:
                df[c] = s.astype("string")
    return df


def parse_dates(df, columns, to=None):
    df = df.copy()
    for c in _present(df, columns):
        df[c] = _parse_dates(df[c])
    return df


def fix_encoding(df, columns):
    def fx(v):
        if isinstance(v, str) and _MOJIBAKE.search(v):
            for enc in ("cp1252", "latin-1"):
                try:
                    return v.encode(enc).decode("utf-8")
                except (UnicodeEncodeError, UnicodeDecodeError):
                    continue
        return v
    df = df.copy()
    for c in _present(df, columns):
        if _is_texty(df[c]):
            df[c] = df[c].astype("string").map(fx).astype("string")
    return df


def convert_units(df, columns, family, target_unit):
    table = _UNITS[family]
    def conv(v):
        if v is pd.NA or v != v:
            return np.nan
        m = _UNIT_RE.match(str(v))
        if not m or m.group(2).lower() not in table:
            return np.nan
        return float(m.group(1).replace(",", ".")) * table[m.group(2).lower()] / table[target_unit.lower()]
    df = df.copy()
    for c in _present(df, columns):
        df[c] = df[c].map(conv).astype(float)
    return df


def merge_rare_classes(df, columns, min_count=10):
    df = df.copy()
    for c in _present(df, columns):
        counts = df[c].value_counts()
        rare = counts[counts < min_count].index
        df[c] = df[c].astype("string").where(~df[c].isin(rare), "other")
    return df


def fix_invalid(df, columns, rule, action="set_missing", min=None, max=None, as_of=None, then_cast=None, unexpected=None, mapping=None, **kwargs):
    """rule: range | future_date | placeholders | categories. action: set_missing | clip | remove."""
    df = df.copy()
    drop = pd.Series(False, index=df.index)
    for c in _present(df, columns):
        if rule == "placeholders":
            mask = df[c].astype("string").str.strip().str.lower().isin(_PLACEHOLDERS).fillna(False).astype(bool)
            col = df[c].astype("string").mask(mask) if _is_texty(df[c]) else df[c]
            df[c] = _to_numeric(col) if then_cast == "numeric" else col
        elif rule == "categories":
            if mapping and isinstance(mapping, dict):
                df[c] = df[c].replace(mapping)
            bad = df[c].isin(unexpected or [])
            if action == "remove":
                drop = drop | bad.fillna(False)
            else:
                df[c] = df[c].mask(bad)
        elif rule == "range":
            v = _to_numeric(df[c])
            bad = pd.Series(False, index=df.index)
            if min is not None:
                bad = bad | (v < min)
            if max is not None:
                bad = bad | (v > max)
            if action == "clip":
                df[c] = v.clip(lower=min, upper=max)
            elif action == "remove":
                drop = drop | bad
            else:
                df[c] = v.mask(bad)
        elif rule == "future_date":
            dt = _parse_dates(df[c])
            bad = (dt > pd.Timestamp(as_of) + pd.Timedelta(days=1)) | (dt.dt.year >= 2100)
            if action == "remove":
                drop = drop | bad.fillna(False)
            else:
                df[c] = df[c].mask(bad)
    return df.loc[~drop] if drop.any() else df


def drop_missing_target(df, columns):
    keep = pd.Series(True, index=df.index)
    for c in _present(df, columns):
        s = df[c]
        keep = keep & s.notna()
        if _is_texty(s):
            keep = keep & (s.astype("string").str.strip() != "").fillna(False)
    return df.loc[keep]


def drop_duplicates(df, columns=(), normalize=False, ignore_columns=()):
    key = df.drop(columns=_present(df, ignore_columns)) if ignore_columns else df
    if normalize:
        key = key.copy()
        for c in key.columns:
            if _is_texty(key[c]):
                key[c] = key[c].astype("string").str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
            elif pd.api.types.is_float_dtype(key[c]):
                key[c] = key[c].round(6)
    return df.loc[~key.duplicated(keep="first")]


def drop_columns(df, columns):
    return df.drop(columns=_present(df, columns))


# ----------------------------------------------------------------------------- the train/test split
def make_split(df, target=None, task_type=None, seed=42, test_size=0.2, strategy="random",
               group_column=None, time_column=None, stratify=True):
    """Boolean Series (True = training row). Deterministic. Rows without a target always train."""
    idx = df.index
    is_train = pd.Series(True, index=idx)
    if len(df) < 10:
        return is_train
    labelled = df.index if target is None or target not in df.columns else df.index[df[target].notna()]
    sub = df.loc[labelled]
    if strategy == "group" and group_column in df.columns:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        _, te = next(splitter.split(sub, groups=sub[group_column].astype("string").fillna("__nan__")))
        is_train.loc[sub.index[te]] = False
    elif strategy == "time" and time_column in df.columns:
        order = _parse_dates(sub[time_column]).sort_values(kind="stable").index
        is_train.loc[order[int(len(order) * (1 - test_size)):]] = False
    else:
        strat = None
        if stratify and task_type in ("binary", "multiclass") and target in df.columns:
            vc = sub[target].astype("string").value_counts()
            if len(vc) > 1 and vc.min() >= 2:
                strat = sub[target].astype("string")
        _, te = train_test_split(np.arange(len(sub)), test_size=test_size, random_state=seed, stratify=strat)
        is_train.loc[sub.index[te]] = False
    return is_train


# ----------------------------------------------------------------------------- stateful steps (fit on TRAIN only)
class Winsorize:
    def __init__(self, columns, lower_q=0.01, upper_q=0.99):
        self.columns, self.lower_q, self.upper_q = list(columns), lower_q, upper_q

    def fit(self, train, target=None):
        self.bounds_ = {}
        for c in _present(train, self.columns):
            v = _to_numeric(train[c]).dropna()
            if len(v):
                self.bounds_[c] = (float(v.quantile(self.lower_q)), float(v.quantile(self.upper_q)))
        return self

    def transform(self, df):
        df = df.copy()
        for c, (lo, hi) in self.bounds_.items():
            if c in df.columns:
                df[c] = _to_numeric(df[c]).clip(lower=lo, upper=hi)
        return df


class Impute:
    def __init__(self, columns, strategy="median", value=None, group_by=None, n_neighbors=5):
        self.columns, self.strategy, self.value, self.group_by, self.n_neighbors = list(columns), strategy, value, group_by, n_neighbors

    def _stat(self, s):
        if self.strategy in ("mean", "median"):
            v = _to_numeric(s).dropna()
            return float(v.mean() if self.strategy == "mean" else v.median()) if len(v) else np.nan
        if self.strategy == "mode":
            m = s.dropna().mode()
            return m.iloc[0] if len(m) else np.nan
        return self.value

    def fit(self, train, target=None):
        cols = _present(train, self.columns)
        self.cols_ = cols
        if self.strategy == "knn":
            self.knn_ = KNNImputer(n_neighbors=self.n_neighbors).fit(pd.DataFrame({c: _to_numeric(train[c]) for c in cols}))
        elif self.strategy == "group" and self.group_by in train.columns:
            self.stats_ = {}
            for c in cols:
                base = train[c]
                inner = Impute([c], strategy=self.value or "median")
                per_group = {k: inner._stat(g) for k, g in base.groupby(train[self.group_by].astype("string"))}
                self.stats_[c] = (per_group, inner._stat(base))
        else:
            self.stats_ = {c: self._stat(train[c]) for c in cols}
        return self

    def transform(self, df):
        df = df.copy()
        if self.strategy == "knn":
            num = pd.DataFrame({c: _to_numeric(df[c]) for c in self.cols_ if c in df.columns})
            filled = self.knn_.transform(num[self.cols_]) if list(num.columns) == self.cols_ else num.values
            for j, c in enumerate(self.cols_):
                df[c] = filled[:, j]
        elif self.strategy == "group" and self.group_by in df.columns:
            g = df[self.group_by].astype("string")
            numeric = self.value in (None, "median", "mean")
            for c, (per_group, overall) in self.stats_.items():
                if c not in df.columns or not df[c].isna().any():
                    continue
                col = _to_numeric(df[c]) if numeric else df[c]
                fill = g.map(per_group).astype(float) if numeric else g.map(per_group)
                df[c] = col.fillna(fill).fillna(overall)
        else:
            for c, stat in self.stats_.items():
                if c not in df.columns or not df[c].isna().any():
                    continue
                col = _to_numeric(df[c]) if self.strategy in ("mean", "median") else df[c]
                df[c] = col.fillna(stat)
        return df


class Bin:
    """Adds `<col>_bin` (0..k-1). Edges come from the training rows."""
    def __init__(self, columns, n_bins=5, strategy="quantile"):
        self.columns, self.n_bins, self.strategy = list(columns), n_bins, strategy

    def fit(self, train, target=None):
        self.edges_ = {}
        for c in _present(train, self.columns):
            v = _to_numeric(train[c]).dropna()
            if len(v) == 0:
                continue
            if self.strategy == "uniform":
                e = np.linspace(v.min(), v.max(), self.n_bins + 1)[1:-1]
            else:
                e = v.quantile(np.linspace(0, 1, self.n_bins + 1)[1:-1]).to_numpy()
            self.edges_[c] = np.unique(e)
        return self

    def transform(self, df):
        df = df.copy()
        for c, e in self.edges_.items():
            if c in df.columns:
                v = _to_numeric(df[c])
                b = pd.Series(np.searchsorted(e, v.to_numpy(), side="right"), index=df.index).astype("Int64")
                df[f"{c}_bin"] = b.mask(v.isna())
        return df


class Scale:
    def __init__(self, columns, method="standard"):
        self.columns, self.method = list(columns), method

    def fit(self, train, target=None):
        self.params_ = {}
        for c in _present(train, self.columns):
            v = _to_numeric(train[c]).dropna()
            if not len(v):
                continue
            if self.method == "minmax":
                a, b = float(v.min()), float(v.max())
                self.params_[c] = (a, (b - a) or 1.0)
            elif self.method == "robust":
                q1, q3 = float(v.quantile(0.25)), float(v.quantile(0.75))
                self.params_[c] = (float(v.median()), (q3 - q1) or 1.0)
            else:
                self.params_[c] = (float(v.mean()), float(v.std(ddof=0)) or 1.0)
        return self

    def transform(self, df):
        df = df.copy()
        for c, (center, scale) in self.params_.items():
            if c in df.columns:
                df[c] = (_to_numeric(df[c]) - center) / scale
        return df


class Encode:
    """method: onehot | ordinal | frequency | target (cross-validated, binary/regression targets)."""
    def __init__(self, columns, method="onehot", max_categories=50, smoothing=10.0, folds=5, seed=42):
        self.columns, self.method, self.max_categories = list(columns), method, max_categories
        self.smoothing, self.folds, self.seed = smoothing, folds, seed

    @staticmethod
    def _key(s):
        return s.astype("string").fillna("__nan__")

    def fit(self, train, target=None):
        self.maps_, self.oof_ = {}, {}
        for c in _present(train, self.columns):
            k = self._key(train[c])
            if self.method == "onehot":
                top = k[k != "__nan__"].value_counts().sort_index().sort_values(ascending=False, kind="stable")
                self.maps_[c] = sorted(top.index[: self.max_categories])
            elif self.method == "ordinal":
                self.maps_[c] = {v: i for i, v in enumerate(sorted(set(k[k != "__nan__"])))}
            elif self.method == "frequency":
                self.maps_[c] = k.value_counts(normalize=True).to_dict()
            elif self.method == "target":
                if target is None:
                    raise ValueError("target encoding needs the target column")
                y = target.loc[train.index]
                if y.dtype == object or pd.api.types.is_string_dtype(y) or pd.api.types.is_bool_dtype(y):
                    classes = sorted(y.dropna().astype("string").unique())
                    if len(classes) != 2:
                        raise ValueError("target encoding supports binary or numeric targets only")
                    y = (y.astype("string") == classes[-1]).astype(float)
                else:
                    y = y.astype(float)
                ok = y.notna()
                kk, yy = k[ok], y[ok]
                prior = float(yy.mean())
                self.maps_[c] = (self._smooth(kk, yy, prior), prior)
                oof = pd.Series(np.nan, index=kk.index)
                for tr, te in KFold(self.folds, shuffle=True, random_state=self.seed).split(kk):
                    p = float(yy.iloc[tr].mean())
                    m = self._smooth(kk.iloc[tr], yy.iloc[tr], p)
                    oof.iloc[te] = kk.iloc[te].map(m).fillna(p).to_numpy()
                self.oof_[c] = oof
        return self

    def _smooth(self, k, y, prior):
        g = y.groupby(k).agg(["sum", "count"])
        return ((g["sum"] + prior * self.smoothing) / (g["count"] + self.smoothing)).to_dict()

    def transform(self, df):
        df = df.copy()
        for c in list(self.maps_):
            if c not in df.columns:
                continue
            k = self._key(df[c])
            if self.method == "onehot":
                for cat in self.maps_[c]:
                    df[f"{c}__{cat}"] = (k == cat).astype(int)
                df = df.drop(columns=[c])
            elif self.method == "ordinal":
                missing = df[c].isna()
                df[c] = k.map(self.maps_[c]).astype(float).fillna(-1.0).mask(missing, np.nan)
            elif self.method == "frequency":
                df[c] = k.map(self.maps_[c]).fillna(0.0).astype(float)
            else:
                full, prior = self.maps_[c]
                out = k.map(full).fillna(prior).astype(float).copy()
                common = df.index.intersection(self.oof_[c].index)
                if len(common):
                    out.loc[common] = self.oof_[c].loc[common].to_numpy()
                df[c] = out
        return df
# --- MLREADY RUNTIME END -----------------------------------------------------------------
