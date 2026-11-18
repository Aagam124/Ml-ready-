"""Synthetic benchmark: datasets with KNOWN planted problems, and clean datasets with none.

Used by the golden test-suite (precision AND recall) and by the demo. Deterministic per seed.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PLANT_FINDINGS = {
    "leakage": ("target_leakage", "post_outcome_score"),
    "duplicates": ("exact_duplicates", None),
    "negative_salary": ("invalid_values", "salary"),
    "inconsistent_categories": ("inconsistent_categories", "gender"),
    "imbalance": ("class_imbalance", "churned"),
    "redundant_columns": ("redundant_columns", "tenure_months"),
    "missingness": ("missing_values", "salary"),
    "impossible_age": ("invalid_values", "age"),
    "disguised_missing": ("disguised_missing_values", "monthly_usage"),
    "constant_column": ("constant_column", "country"),
}
PLANTS = tuple(PLANT_FINDINGS)


def make_clean(kind: str = "binary", n: int = 1500, seed: int = 0) -> Tuple[pd.DataFrame, str]:
    """A realistic dataset with NO deliberate problem. Signal is modest (AUC ~0.8) so nothing looks leaky."""
    rng = np.random.default_rng(seed)
    age = rng.normal(40, 11, n).clip(18, 80).round(0)
    income = np.exp(rng.normal(10.8, 0.45, n)).round(2)
    tenure = rng.integers(0, 20, n)
    region = rng.choice(["north", "south", "east", "west"], n, p=[0.3, 0.3, 0.2, 0.2])
    plan = rng.choice(["basic", "plus", "pro"], n, p=[0.5, 0.3, 0.2])
    usage = rng.gamma(3.0, 40.0, n).round(1)
    sessions = rng.poisson(12, n)
    satisfaction = rng.integers(1, 6, n)
    noise_a, noise_b = rng.normal(size=n), rng.normal(size=n)
    signed = rng.normal(0, 1, n) * 100                       # a signed quantity (not "non-negative")
    base = 0.03 * (age - 40) + 0.00002 * (income - 50000) - 0.08 * tenure + 0.4 * (plan == "basic") - 0.3 * (satisfaction - 3) + rng.normal(0, 1.2, n)
    df = pd.DataFrame({"customer_id": np.arange(100000, 100000 + n), "age": age, "income": income, "tenure_years": tenure,
                       "region": region, "plan": plan, "monthly_usage": usage, "sessions": sessions,
                       "satisfaction": satisfaction, "balance_change": signed.round(2), "feature_a": noise_a.round(4), "feature_b": noise_b.round(4)})
    if kind == "binary":
        p = 1 / (1 + np.exp(-(base - 0.3)))
        df["churned"] = (rng.random(n) < p).astype(int)
        return df, "churned"
    if kind == "multiclass":
        score = base + rng.normal(0, 0.5, n)
        df["segment"] = pd.cut(score, [-np.inf, -1, 0.3, np.inf], labels=["bronze", "silver", "gold"]).astype(str)
        return df, "segment"
    if kind == "regression":
        df["spend"] = (200 + 3 * df["sessions"] + 0.002 * df["income"] + 40 * satisfaction + rng.normal(0, 60, n)).round(2)
        return df, "spend"
    raise ValueError(kind)


def make_planted(plants: List[str], seed: int = 1, n: int = 1500) -> Tuple[pd.DataFrame, str, Dict[str, Tuple[str, Optional[str]]]]:
    """Start clean (binary), plant the requested problems, and return what must be found.

    expected maps plant -> (finding code, column or None).
    """
    unknown = set(plants) - set(PLANTS)
    if unknown:
        raise ValueError(f"Unknown planted defects: {', '.join(sorted(unknown))}")
    df, target = make_clean("binary", n=n, seed=seed)
    rng = np.random.default_rng(seed + 100)
    expected: Dict[str, Tuple[str, Optional[str]]] = {}
    if "negative_salary" in plants or "missingness" in plants or "impossible_age" in plants:
        df["salary"] = rng.normal(60000, 14000, n).round(2)
    if "leakage" in plants:
        df["post_outcome_score"] = (df[target] * 0.7 + rng.normal(0.08, 0.04, n)).clip(0, 1).round(4)
        expected["leakage"] = PLANT_FINDINGS["leakage"]
    if "negative_salary" in plants:
        df.loc[rng.choice(n, 8, replace=False), "salary"] = -np.abs(rng.normal(50000, 15000, 8)).round(2)
        expected["negative_salary"] = PLANT_FINDINGS["negative_salary"]
    if "impossible_age" in plants:
        df.loc[rng.choice(n, 5, replace=False), "age"] = [-3, 150, 200, -1, 135]
        expected["impossible_age"] = PLANT_FINDINGS["impossible_age"]
    if "inconsistent_categories" in plants:
        g = rng.choice(["Male", "male", "M", "Female", "female", "F"], n, p=[0.3, 0.1, 0.1, 0.3, 0.1, 0.1])
        df["gender"] = g
        expected["inconsistent_categories"] = PLANT_FINDINGS["inconsistent_categories"]
    if "redundant_columns" in plants:
        df["tenure_months"] = df["tenure_years"] * 12
        expected["redundant_columns"] = PLANT_FINDINGS["redundant_columns"]
    if "missingness" in plants:
        mask = (df["income"] > df["income"].quantile(0.7)) & (rng.random(n) < 0.6)   # depends on income: MAR
        df.loc[mask, "salary"] = np.nan
        expected["missingness"] = PLANT_FINDINGS["missingness"]
    if "disguised_missing" in plants:
        v = df["monthly_usage"].astype(object)
        v.iloc[rng.choice(n, 40, replace=False)] = "N/A"
        df["monthly_usage"] = v
        expected["disguised_missing"] = PLANT_FINDINGS["disguised_missing"]
    if "constant_column" in plants:
        df["country"] = "IN"
        expected["constant_column"] = PLANT_FINDINGS["constant_column"]
    if "imbalance" in plants:
        keep = df[target] == 0
        pos = np.flatnonzero(df[target] == 1)
        drop = rng.choice(pos, int(len(pos) * 0.93), replace=False)
        df = df.drop(index=df.index[drop]).reset_index(drop=True)
        expected["imbalance"] = (PLANT_FINDINGS["imbalance"][0], target)
    if "duplicates" in plants:
        dups = df.sample(frac=0.05, random_state=seed)
        df = pd.concat([df, dups], ignore_index=True)
        expected["duplicates"] = PLANT_FINDINGS["duplicates"]
    return df, target, expected


def planted_defect_present(plant: str, df: pd.DataFrame, target: str) -> bool:
    """Verify the generated data contains the defect named by its ground-truth label."""
    if plant not in PLANT_FINDINGS:
        raise ValueError(f"Unknown planted defect: {plant}")
    if plant == "leakage":
        if "post_outcome_score" not in df or target not in df:
            return False
        x = pd.to_numeric(df["post_outcome_score"], errors="coerce")
        y = pd.to_numeric(df[target], errors="coerce")
        return x.corr(y) >= 0.5
    if plant == "duplicates":
        return bool(df.duplicated(keep="first").any())
    if plant == "negative_salary":
        return "salary" in df and bool((pd.to_numeric(df["salary"], errors="coerce") < 0).any())
    if plant == "inconsistent_categories":
        if "gender" not in df:
            return False
        values = set(df["gender"].dropna().astype(str))
        return {"Male", "male", "M", "Female", "female", "F"}.issubset(values)
    if plant == "imbalance":
        if target != "churned" or target not in df:
            return False
        counts = df[target].value_counts()
        return len(counts) >= 2 and counts.iloc[0] / counts.iloc[-1] >= 3
    if plant == "redundant_columns":
        return {"tenure_years", "tenure_months"}.issubset(df.columns) and bool(
            np.allclose(df["tenure_months"], df["tenure_years"] * 12)
        )
    if plant == "missingness":
        if not {"salary", "income"}.issubset(df.columns):
            return False
        missing = df["salary"].isna()
        return bool(missing.any() and (df.loc[missing, "income"] > df["income"].quantile(0.7)).all())
    if plant == "impossible_age":
        if "age" not in df:
            return False
        age = pd.to_numeric(df["age"], errors="coerce")
        return bool(((age < 0) | (age > 120)).any())
    if plant == "disguised_missing":
        return "monthly_usage" in df and bool(
            df["monthly_usage"].astype("string").str.strip().str.lower().eq("n/a").any()
        )
    if plant == "constant_column":
        return "country" in df and df["country"].nunique(dropna=True) == 1 and df["country"].iloc[0] == "IN"
    return False
