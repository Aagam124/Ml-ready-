"""Cleaning engine: ops, no-leakage fitting, immutability, reproducibility of the exported script."""
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

from app.benchmark import PLANTS, make_planted
from app.cleaning import runtime as rt
from app.cleaning.executor import run_plan
from app.cleaning.export import generate_pipeline, generate_script
from app.cleaning.plan import propose_plan
from app.cleaning.preview import preview_step
from app.engine.pipeline import run_analysis


def _plan(df, target):
    r = run_analysis(df, target)
    return r, propose_plan(r["findings"], r["profile"], target)


def test_original_is_never_modified():
    df, target, _ = make_planted(list(PLANTS))
    snapshot = df.copy(deep=True)
    _, plan = _plan(df, target)
    run_plan(df, plan["steps"], target, "binary")
    pd.testing.assert_frame_equal(df, snapshot)


def test_applying_default_plan_fixes_planted_problems_and_raises_score():
    df, target, _ = make_planted(list(PLANTS))
    r, plan = _plan(df, target)
    out, _, rep = run_plan(df, plan["steps"], target, "binary")
    r2 = run_analysis(out, target)
    codes = {f["code"] for f in r2["findings"]}
    assert not codes & {"target_leakage", "exact_duplicates", "invalid_values", "inconsistent_categories",
                        "redundant_columns", "disguised_missing_values", "constant_column", "empty_column"}, codes
    assert r2["score"]["value"] > r["score"]["value"] + 20


def test_imputer_is_fitted_on_training_rows_only():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": np.r_[rng.normal(0, 1, 160), rng.normal(100, 1, 40)], "y": [0, 1] * 100})
    df.loc[[0, 1, 2], "x"] = np.nan
    steps = [{"step_id": "s1", "op": "impute", "params": {"strategy": "mean"}, "columns": ["x"], "enabled": True}]
    out, is_train, _ = run_plan(df, steps, "y", "binary")
    expected = df.loc[is_train, "x"].mean()
    filled = out.loc[[0, 1, 2], "x"]
    assert np.allclose(filled[is_train[[0, 1, 2]].to_numpy()], expected)
    assert not np.isclose(expected, df["x"].mean())            # train-only mean differs from all-rows mean


def test_scaler_and_encoder_learn_from_train_only_and_handle_unseen():
    train = pd.DataFrame({"a": [1.0, 2, 3, 4], "c": ["x", "y", "x", "y"]})
    test = pd.DataFrame({"a": [10.0], "c": ["z"]})
    sc = rt.Scale(["a"], "standard").fit(train)
    assert np.isclose(sc.transform(test)["a"].iloc[0], (10 - 2.5) / np.std([1, 2, 3, 4]))
    enc = rt.Encode(["c"], "onehot").fit(train)
    t = enc.transform(test)
    assert t[["c__x", "c__y"]].to_numpy().sum() == 0
    assert rt.Encode(["c"], "ordinal").fit(train).transform(test)["c"].iloc[0] == -1
    assert rt.Encode(["c"], "frequency").fit(train).transform(test)["c"].iloc[0] == 0.0


def test_target_encoding_is_out_of_fold_for_train_rows():
    rng = np.random.default_rng(0)
    n = 400
    cat = rng.choice([f"c{i}" for i in range(40)], n)
    y = pd.Series(rng.integers(0, 2, n), name="y")
    df = pd.DataFrame({"cat": cat})
    enc = rt.Encode(["cat"], "target", seed=1).fit(df, target=y)
    out = enc.transform(df)["cat"]
    full = df["cat"].map(enc.maps_["cat"][0])
    assert not np.allclose(out, full)                 # train rows use out-of-fold values, not their own label
    assert abs(np.corrcoef(out, y)[0, 1]) < 0.15      # random category carries no signal => no leak into train


def test_every_op_runs_and_preview_reports_effect():
    rng = np.random.default_rng(1)
    n = 300
    df = pd.DataFrame({
        "price": rng.normal(100, 20, n).round(2), "city": rng.choice(["a", "b", "c "], n),
        "txt": rng.choice(["$1,200", "300", "45%"], n), "when": pd.date_range("2020-01-01", periods=n).astype(str),
        "w": rng.choice(["70 kg", "150 lb"], n), "bad": rng.choice(["N/A", "1", "2", "3"], n), "y": rng.integers(0, 2, n)})
    df.loc[:20, "price"] = np.nan
    steps = [
        {"step_id": "a", "op": "strip_whitespace", "columns": ["city"], "params": {}},
        {"step_id": "b", "op": "normalize_categories", "columns": ["city"], "params": {"mapping": {"a": "A"}}},
        {"step_id": "c", "op": "cast_type", "columns": ["txt"], "params": {"to": "numeric"}},
        {"step_id": "d", "op": "parse_dates", "columns": ["when"], "params": {}},
        {"step_id": "e", "op": "convert_units", "columns": ["w"], "params": {"family": "mass", "target_unit": "kg"}},
        {"step_id": "f", "op": "fix_invalid", "columns": ["bad"], "params": {"rule": "placeholders", "action": "set_missing", "then_cast": "numeric"}},
        {"step_id": "g", "op": "impute", "columns": ["price"], "params": {"strategy": "median"}},
        {"step_id": "h", "op": "impute", "columns": ["bad"], "params": {"strategy": "knn", "n_neighbors": 3}},
        {"step_id": "i", "op": "winsorize", "columns": ["price"], "params": {"lower_q": 0.05, "upper_q": 0.95}},
        {"step_id": "j", "op": "bin", "columns": ["price"], "params": {"n_bins": 4}},
        {"step_id": "k", "op": "encode", "columns": ["city"], "params": {"method": "onehot"}},
        {"step_id": "l", "op": "scale", "columns": ["price", "txt"], "params": {"method": "robust"}},
        {"step_id": "m", "op": "drop_columns", "columns": ["w"], "params": {}},
    ]
    out, _, rep = run_plan(df, steps, "y", "binary")
    assert "w" not in out and {"city__A", "price_bin"} <= set(out.columns) and out["price"].notna().all() and out["bad"].notna().all()
    p = preview_step(df, steps, "g", "y", "binary")
    assert p["cells_changed"] == 21 and p["rows_removed"] == 0 and "price" in p["columns_affected"]


def test_group_and_mode_and_constant_impute():
    df = pd.DataFrame({"g": ["a", "a", "b", "b", "a", "b"] * 5, "v": [1.0, np.nan, 10.0, np.nan, 3.0, 12.0] * 5,
                       "c": ["u", np.nan, "v", "u", "u", np.nan] * 5, "y": [0, 1] * 15})
    steps = [{"step_id": "1", "op": "impute", "columns": ["v"], "params": {"strategy": "group", "group_by": "g", "value": "median"}},
             {"step_id": "2", "op": "impute", "columns": ["c"], "params": {"strategy": "mode"}}]
    out, _, _ = run_plan(df, steps, "y", "binary")
    assert out["v"].notna().all() and out["c"].notna().all()
    assert out.loc[(df.g == "b") & df.v.isna(), "v"].median() > 5


def test_split_is_deterministic_stratified_and_group_safe():
    df, target, _ = make_planted(["imbalance"], n=1200)
    a = rt.make_split(df, target, "binary", 42, 0.2)
    b = rt.make_split(df, target, "binary", 42, 0.2)
    assert a.equals(b)
    assert abs(df.loc[~a, target].mean() - df[target].mean()) < 0.01
    g = pd.DataFrame({"gid": np.repeat(np.arange(100), 5), "v": np.arange(500), "y": [0, 1] * 250})
    s = rt.make_split(g, "y", "binary", 1, 0.2, "group", "gid")
    assert not set(g.loc[s, "gid"]) & set(g.loc[~s, "gid"])
    t = pd.DataFrame({"d": pd.date_range("2020", periods=100).astype(str), "y": [0, 1] * 50})
    s = rt.make_split(t, "y", "binary", 1, 0.2, "time", time_column="d")
    assert t.loc[s, "d"].max() < t.loc[~s, "d"].min()


def _write_csv(df, path):
    df.to_csv(path, index=False)


def test_exported_script_reproduces_cleaned_dataset_byte_for_byte():
    df, target, _ = make_planted(list(PLANTS))
    _, plan = _plan(df, target)
    # also switch on some stateful learned steps so the no-leak fitting is exercised
    plan["steps"] += [
        {"step_id": "x1", "op": "winsorize", "columns": ["income"], "params": {"lower_q": 0.01, "upper_q": 0.99}, "enabled": True},
        {"step_id": "x2", "op": "encode", "columns": ["region"], "params": {"method": "onehot"}, "enabled": True},
        {"step_id": "x3", "op": "encode", "columns": ["plan"], "params": {"method": "target"}, "enabled": True},
        {"step_id": "x4", "op": "scale", "columns": ["income", "monthly_usage"], "params": {"method": "standard"}, "enabled": True},
        {"step_id": "x5", "op": "bin", "columns": ["age"], "params": {"n_bins": 5}, "enabled": True},
    ]
    with tempfile.TemporaryDirectory() as d:
        src, dst = os.path.join(d, "orig.csv"), os.path.join(d, "out.csv")
        _write_csv(df, src)
        loaded = pd.read_csv(src, sep=",", encoding="utf-8-sig", low_memory=False, skip_blank_lines=True, header=0, names=list(df.columns))
        out, _, _ = run_plan(loaded, plan["steps"], target, "binary")
        script = generate_script(plan["steps"], target, "binary",
                                 {"ext": ".csv", "delimiter": ",", "encoding": "utf-8-sig", "columns": list(df.columns)}, "benchmark")
        script_path = os.path.join(d, "clean_data.py")
        open(script_path, "w", encoding="utf-8").write(script)
        res = subprocess.run([sys.executable, script_path, src, dst], capture_output=True, text=True)
        assert res.returncode == 0, res.stderr[-2000:]
        assert open(dst, "rb").read() == out.to_csv(index=False).encode("utf-8")
        compile(script, "clean_data.py", "exec")
        assert "TRAIN rows only" in script and "Step 1/" in script


def test_exported_sklearn_pipeline_fits_on_train_and_matches_script_for_native_steps():
    df, target, _ = make_planted(["negative_salary", "missingness", "duplicates"])
    steps = [
        {"step_id": "a", "op": "fix_invalid", "columns": ["salary"], "params": {"rule": "range", "min": 0, "max": None, "action": "set_missing"}, "enabled": True},
        {"step_id": "b", "op": "drop_duplicates", "columns": [], "params": {}, "enabled": True},
        {"step_id": "c", "op": "drop_columns", "columns": ["customer_id"], "params": {}, "enabled": True},
        {"step_id": "d", "op": "impute", "columns": ["salary"], "params": {"strategy": "median"}, "enabled": True},
        {"step_id": "e", "op": "encode", "columns": ["region"], "params": {"method": "ordinal"}, "enabled": True},
        {"step_id": "f", "op": "scale", "columns": ["income"], "params": {"method": "standard"}, "enabled": True},
    ]
    code = generate_pipeline(steps, target, "binary")
    ns = {}
    exec(compile(code, "pipeline.py", "exec"), ns)
    clean = ns["structural_clean"](df)
    X, y = clean.drop(columns=[target]), clean[target]
    train = rt.make_split(clean, target, "binary", 42, 0.2)
    pre = ns["build_preprocessor"]().fit(X[train], y[train])
    Xt = pre.transform(X)
    assert Xt["salary"].notna().all()
    expected, _, _ = run_plan(df, steps, target, "binary")
    assert np.allclose(Xt["income"].to_numpy(float), expected["income"].to_numpy(float), atol=1e-9)
    assert np.allclose(Xt["salary"].to_numpy(float), expected["salary"].to_numpy(float), atol=1e-6)


def test_csv_injection_protection_only_touches_formula_like_text():
    from app.ingestion.csvsafe import sanitize_for_spreadsheet
    df = pd.DataFrame({"name": ["=cmd|' /C calc'!A0", "+1+1", "-2+3", "@SUM(A1)", "normal", None], "n": [-5, 1, 2, 3, 4, 5]})
    out = sanitize_for_spreadsheet(df)
    assert out["name"].tolist()[:5] == ["'=cmd|' /C calc'!A0", "'+1+1", "'-2+3", "'@SUM(A1)", "normal"]
    assert out["n"].tolist() == [-5, 1, 2, 3, 4, 5] and df["name"].iloc[0].startswith("=")      # input untouched


def test_hostile_column_names_cannot_inject_code_into_exported_scripts():
    import ast
    evil = 'x"""\nimport os; os.system("echo pwned")  #'
    steps = [{"step_id": "a", "op": "drop_columns", "columns": [evil], "params": {}, "enabled": True, "label": f"Drop '{evil}'", "finding_code": "constant_column\nimport os"},
             {"step_id": "b", "op": "impute", "columns": [evil], "params": {"strategy": "median"}, "enabled": True, "label": f"Fill {evil}"}]
    for code in (generate_script(steps, None, None, {"ext": ".csv", "columns": [evil]}, "name\"\"\"\nimport os"), generate_pipeline(steps, None, None)):
        tree = ast.parse(code)
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "system"]
        imports = [n for n in ast.walk(tree) if isinstance(n, ast.Import) and any(a.name == "os" for a in n.names)]
        assert not calls and not imports
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        assert not any("pwned" in n or n == "system" for n in names)      # the payload exists only inside string literals


def test_remaining_ops_behave():
    df = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0, 100.0] * 8, "b": [2.0, np.nan, 6.0, 8.0, 200.0] * 8, "when": ["2020-01-01", "2999-01-01"] * 20,
                       "price": [10, -5, 20, 30, 5000] * 8, "y": ["a", "b", "c", "a", "b"] * 8, "w": ["70 kg", "150 lb", "100 g", "1 kg", "x"] * 8})
    knn = rt.Impute(["a", "b"], strategy="knn", n_neighbors=2).fit(df).transform(df)
    assert knn[["a", "b"]].notna().all().all()
    const = rt.Impute(["a"], strategy="constant", value=-1).fit(df).transform(df)
    assert (const["a"] == -1).sum() == 8
    assert set(rt.Bin(["a"], 3, "uniform").fit(df).transform(df)["a_bin"].dropna().unique()) <= {0, 1, 2}
    mm = rt.Scale(["a"], "minmax").fit(df).transform(df)["a"]
    assert mm.min() == 0 and mm.max() == 1
    w = rt.Winsorize(["price"], 0.1, 0.7).fit(df).transform(df)["price"]
    assert w.max() == 30 and w.min() == -5
    assert rt.fix_invalid(df, ["price"], "range", "clip", min=0, max=100)["price"].min() == 0
    assert len(rt.fix_invalid(df, ["price"], "range", "remove", min=0)) == len(df) - 8
    assert rt.fix_invalid(df, ["when"], "future_date", "set_missing", as_of="2024-01-01")["when"].isna().sum() == 20
    assert len(rt.fix_invalid(df, ["when"], "future_date", "remove", as_of="2024-01-01")) == 20
    assert set(rt.merge_rare_classes(df, ["y"], min_count=10)["y"]) == {"a", "b", "other"}
    d2 = df.copy(); d2.loc[:3, "y"] = np.nan
    assert len(rt.drop_missing_target(d2, ["y"])) == len(df) - 4
    conv = rt.convert_units(df, ["w"], "mass", "kg")["w"]
    assert conv.iloc[0] == 70 and abs(conv.iloc[1] - 68.0388555) < 1e-6 and conv.iloc[2] == 0.1 and np.isnan(conv.iloc[4])
    assert rt.fix_encoding(pd.DataFrame({"c": ["MÃ¼nchen", "Zürich"]}), ["c"])["c"].tolist() == ["München", "Zürich"]
    assert rt.cast_type(pd.DataFrame({"id": [1.0, 2.0, np.nan]}), ["id"], "string")["id"].tolist()[:2] == ["1", "2"]
    assert rt.parse_dates(df, ["when"])["when"].dtype.kind == "M"
    assert rt.strip_whitespace(pd.DataFrame({"c": [" a ", "b"]}), ["c"])["c"].tolist() == ["a", "b"]


def test_target_suggestions_from_stored_profile_match_data_based_ones():
    from app.benchmark import make_clean
    from app.engine.target import suggest_targets, suggest_targets_from_profile, task_type_from_profile
    from app.profiling.profile import profile_dataframe
    for kind in ("binary", "multiclass", "regression"):
        df, target = make_clean(kind, n=400)
        prof = profile_dataframe(df)
        a, b = suggest_targets(df, prof), suggest_targets_from_profile(prof)
        assert [(c["column"], c["task_type"]) for c in a] == [(c["column"], c["task_type"]) for c in b]
        if kind == "binary":
            assert a[0]["column"] == target
    idcol = next(c for c in prof["columns"] if c["name"] == "customer_id")
    assert task_type_from_profile(idcol) is None
