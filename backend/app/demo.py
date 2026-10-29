"""Deterministic demo dataset with realistic, deliberately planted problems."""
from __future__ import annotations

import numpy as np
import pandas as pd


def generate_demo_dataframe(n_rows: int = 825, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_dups = round(n_rows * 0.03)
    n = n_rows - n_dups

    tenure_months = rng.integers(1, 121, n)
    churned = (rng.random(n) < 0.09).astype(int)
    df = pd.DataFrame({
        "customer_id": [f"CUST{100000 + i}" for i in range(n)],
        "age": np.clip(rng.normal(38, 12, n), 18, 75).round(1),
        "salary": rng.normal(58000, 15000, n).round(2),
        "department": rng.choice(["Sales", "Support", "Engineering", "HR", "Finance"], n),
        "email": [f"user{i}_{rng.integers(1000, 9999)}@example.com" for i in range(n)],
        "country": "IN",
        "account_type": rng.choice(["standard", "premium"], n, p=[0.7, 0.3]),
        "tenure_months": tenure_months,
        "tenure_years": tenure_months / 12,
        "churn_score_leak": np.clip(churned * 0.7 + rng.normal(0.07, 0.04, n), 0, 1).round(4),
        "churned": churned,
    })
    # Planted problems: missing values, impossible values.
    df.loc[rng.choice(n, int(n * 0.05), replace=False), "age"] = np.nan
    df.loc[rng.choice(n, int(n * 0.11), replace=False), "salary"] = np.nan
    bad = rng.choice(n, 6, replace=False)
    df.loc[bad, "salary"] = -np.abs(rng.normal(60000, 20000, 6)).round(2)
    df.loc[rng.choice(n, 2, replace=False), "age"] = [2.4, 140.0]
    # Exact duplicate rows.
    dups = df.iloc[rng.choice(n, n_dups, replace=False)]
    return pd.concat([df, dups], ignore_index=True)
