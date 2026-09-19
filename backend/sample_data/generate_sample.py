"""
Generates sample_customers.csv: a synthetic dataset with known, deliberate
problems so Phase 1 can be tested against every code path:
  - missing values (age, salary)
  - duplicate rows
  - outliers (salary)
  - an ID-like column (customer_id)
  - a constant column (country - always "IN")
  - a near-constant column (account_type)
  - a high-cardinality categorical column (email)
  - class imbalance in the target (churned: ~90/10 split)
  - a near-perfect leakage feature (churn_score_leak, built from the target)
  - two highly correlated numeric features (tenure_months / tenure_years)
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N = 800

customer_id = [f"CUST{100000 + i}" for i in range(N)]
age = RNG.normal(38, 12, N).round(1)
age[RNG.choice(N, size=40, replace=False)] = np.nan  # ~5% missing

salary = RNG.normal(55000, 15000, N).round(2)
salary[RNG.choice(N, size=90, replace=False)] = np.nan  # ~11% missing
# inject some clear outliers
outlier_idx = RNG.choice(N, size=12, replace=False)
salary[outlier_idx] = salary[outlier_idx] + RNG.choice([1, -1], 12) * RNG.uniform(150000, 300000, 12)

department = RNG.choice(
    ["Sales", "Engineering", "Support", "Marketing", "Operations"],
    size=N, p=[0.3, 0.25, 0.2, 0.15, 0.1],
)

email = [f"user{i}_{RNG.integers(1000,9999)}@example.com" for i in range(N)]  # ~unique per row

country = ["IN"] * N  # constant column

account_type = RNG.choice(["standard", "standard", "standard", "standard", "premium"], size=N)  # near-constant

tenure_months = RNG.integers(1, 121, N)
tenure_years = tenure_months / 12.0  # near-perfect correlation with tenure_months

# imbalanced binary target: ~10% churned
churned = RNG.choice([0, 1], size=N, p=[0.9, 0.1])

# leakage feature: derived almost directly from the target
churn_score_leak = churned * RNG.uniform(0.85, 1.0, N) + (1 - churned) * RNG.uniform(0.0, 0.15, N)

df = pd.DataFrame({
    "customer_id": customer_id,
    "age": age,
    "salary": salary,
    "department": department,
    "email": email,
    "country": country,
    "account_type": account_type,
    "tenure_months": tenure_months,
    "tenure_years": tenure_years,
    "churn_score_leak": churn_score_leak.round(4),
    "churned": churned,
})

# inject duplicate rows
dup_rows = df.sample(n=25, random_state=42)
df = pd.concat([df, dup_rows], ignore_index=True)

df.to_csv("sample_customers.csv", index=False)
print(f"Wrote sample_customers.csv with {len(df)} rows, {len(df.columns)} columns.")
