import io
import numpy as np
import pandas as pd
from app.engine.pipeline import run_analysis
from app.cleaning.plan import propose_plan
from app.cleaning.executor import run_plan

csv_data = """customer_id,name,age,gender,city,monthly_income_inr,years_as_customer,uses_mobile_app,has_complaint,churned
1001,Aarav,22,Male,Indore,32000,2,Yes,No,0
1002,Diya,29,Female,Bhopal,45000,5,No,No,1
1003,Kabir,35,Male,Indore,58000,8,Yes,Yes,0
1004,Anaya,41,Female,Pune,72000,12,No,No,1
1005,Ishaan,26,Male,Mumbai,39000,3,Yes,No,0
1006,Meera,31,Female,Indore,51000,7,No,No,0
1007,Vivaan,24,Male,Bhopal,28000,1,Yes,No,1
1008,Saanvi,38,Female,Pune,68000,11,No,Yes,0
1009,Advik,45,Male,Mumbai,85000,16,Yes,No,0
1010,Myra,27,Female,Indore,42000,4,No,No,1
1011,Arjun,33,Male,Bhopal,56000,9,Yes,No,0
1012,Kiara,30,Female,Pune,49000,6,No,No,0
1013,Reyansh,52,Male,Mumbai,99000,22,Yes,Yes,0
1014,Aadhya,23,Female,Indore,31000,2,No,No,1
1015,Atharv,37,Male,Bhopal,63000,10,Yes,No,0
1016,Pari,28,Female,Pune,46000,5,No,No,0
1017,Krishna,34,Male,Indore,60000,9,Yes,Yes,0
1018,Ira,25,Female,Mumbai,37000,3,No,No,1
1019,Rudra,43,Male,Bhopal,78000,14,Yes,No,0
1020,Sara,32,Female,Pune,54000,7,No,No,0
1021,Dev,29,Male,Indore,,6,Yes,No,0
1022,Tara,,Female,Bhopal,44000,4,No,No,1
1023,Neil,36,,Pune,67000,10,Yes,No,0
1024,Zoya,27,Female,,41000,5,No,No,1
1025,Om,31,Male,Indore,52000,7,Maybe,No,0
1026,Siya,24,Female,Mumbai,35000,2,No,No,1
1027,Yuvaan,39,Male,Pune,74000,13,Yes,No,0
1028,Mira,28,Female,Bhopal,47000,5,No,No,0
1029,Aarav,22,Male,Indore,32000,2,Yes,No,0
1030,Rohan,999,Male,Mumbai,9999999,4,No,No,1"""


def test_cleaning_pipeline_safety_and_preservation():
    df = pd.read_csv(io.StringIO(csv_data))
    res = run_analysis(df, target="churned")
    plan = propose_plan(res["findings"], res["profile"], target="churned")

    # 1. Useful columns preserved by default (not dropped)
    drop_steps = [s for s in plan["steps"] if s["op"] == "drop_columns"]
    for s in drop_steps:
        assert s["enabled"] is False, f"Drop step should be disabled by default: {s}"

    # 2. Winsorize disabled by default (preserves legitimate numerical values like 22)
    winsorize_steps = [s for s in plan["steps"] if s["op"] == "winsorize"]
    for s in winsorize_steps:
        assert s["enabled"] is False, f"Winsorize should be disabled by default: {s}"

    # 3. Sensitive / nominal imputation avoids demographic bias ("Unknown")
    impute_gender = [s for s in plan["steps"] if s["op"] == "impute" and "gender" in s["columns"]]
    assert len(impute_gender) > 0
    assert impute_gender[0]["params"].get("value") == "Unknown" or impute_gender[0]["params"].get("strategy") == "constant"

    # 4. Duplicate entity removal is disabled by default (requires user confirmation)
    dup_steps = [s for s in plan["steps"] if s.get("finding_code") == "duplicate_entities_different_ids"]
    for s in dup_steps:
        assert s["enabled"] is False

    # 5. Run the plan with default enabled steps
    cleaned, is_train, rpt = run_plan(df, plan["steps"], target="churned", task_type="binary")

    # 6. Verify columns are NOT deleted from dataset
    assert "customer_id" in cleaned.columns
    assert "name" in cleaned.columns
    assert "monthly_income_inr" in cleaned.columns

    # 7. Verify customers with different IDs are NOT automatically deleted
    assert len(cleaned) == 30

    # 8. Verify years_as_customer is unchanged (22 not clipped to 20.68)
    assert cleaned["years_as_customer"].max() == 22.0

    # 9. Verify audit trail is produced
    assert len(rpt["audit_trail"]) > 0
    assert "total_cells_changed" in rpt

    # 10. Verify chained imputation leaves NO missing values in uses_mobile_app
    assert cleaned["uses_mobile_app"].isna().sum() == 0
    assert "Maybe" not in cleaned["uses_mobile_app"].values


# 1. Acceptance test: Missingness below 1% (4 missing in 1,000 rows = 0.4%)
def test_missing_value_detection_under_one_percent():
    rng = np.random.default_rng(42)
    cities = ["Indore", "Bhopal", "Mumbai", "Pune"]
    df = pd.DataFrame({
        "city": rng.choice(cities, 1000),
        "income": rng.normal(50000, 10000, 1000)
    })
    df.loc[:3, "city"] = np.nan  # Exactly 4 missing values in 1,000 rows
    res = run_analysis(df)
    missing_findings = [f for f in res["findings"] if f["code"] == "missing_values" and "city" in f["columns"]]
    assert len(missing_findings) == 1
    f = missing_findings[0]
    assert f["evidence"]["n_missing"] == 4
    assert f["evidence"]["pct"] == 0.4
    assert f["severity"] == "low"


# 2. Acceptance test: Unknown category is not flagged as unexpected
def test_unknown_category_not_flagged_as_unexpected():
    # Dataset where gender contains Male, Female, and Unknown (e.g. after MLReady safe imputation)
    df = pd.DataFrame({
        "gender": ["Male"] * 50 + ["Female"] * 48 + ["Unknown"] * 2,
        "spend": np.random.default_rng(0).normal(100, 10, 100)
    })
    res = run_analysis(df)
    unexpected = [f for f in res["findings"] if f["code"] == "unexpected_categorical_value" and "gender" in f["columns"]]
    assert len(unexpected) == 0, f"Unknown should not be flagged as unexpected category: {unexpected}"


# 3. Acceptance test: Cleaning-plan conflict resolution
def test_cleaning_plan_conflict_resolution():
    # Value "female" should be normalized to "Female" and not wiped out
    df = pd.DataFrame({
        "id": list(range(62)),
        "gender": ["Female"] * 30 + ["Male"] * 30 + ["female"] * 2,
        "y": [0, 1] * 31
    })
    res = run_analysis(df, target="y")
    plan = propose_plan(res["findings"], res["profile"], target="y")
    cleaned, _, _ = run_plan(df, plan["steps"], target="y", task_type="binary")
    # All 62 rows present, "female" normalized to "Female", no NaNs introduced
    assert len(cleaned) == 62
    assert "female" not in cleaned["gender"].values
    assert cleaned["gender"].isna().sum() == 0
    assert set(cleaned["gender"].unique()) == {"Female", "Male"}


# 4. Acceptance test: Safe duplicate removal
def test_duplicate_removal_safety():
    df = pd.DataFrame({
        "customer_id": [101, 102],
        "age": [30, 30],
        "city": ["Indore", "Indore"],
        "income": [50000, 50000]
    })
    res = run_analysis(df)
    plan = propose_plan(res["findings"], res["profile"], target=None)
    cleaned, _, _ = run_plan(df, plan["steps"])
    # Two customers with different IDs must not be automatically deleted by default
    assert len(cleaned) == 2
    assert 101 in cleaned["customer_id"].values and 102 in cleaned["customer_id"].values


# 5. Acceptance test: High income outlier flagged for review without being assumed incorrect or auto-deleted
def test_outlier_detection_preserves_valid_high_income():
    rng = np.random.default_rng(42)
    incomes = list(rng.normal(50000, 5000, 50)) + [500000]  # Valid high income customer
    df = pd.DataFrame({"income": incomes, "y": [0, 1] * 25 + [1]})
    res = run_analysis(df, target="y")
    plan = propose_plan(res["findings"], res["profile"], target="y")
    # The high income is not deleted
    cleaned, _, _ = run_plan(df, plan["steps"], target="y")
    assert len(cleaned) == 51
    assert 500000 in cleaned["income"].values


# 6. Acceptance test: Score trustworthiness with unresolved critical finding
def test_ml_readiness_score_trustworthiness():
    # Deterministic leakage is a critical finding
    y = [0, 1] * 25
    df = pd.DataFrame({"leak_col": y, "x": range(50), "y": y})
    res = run_analysis(df, target="y")
    assert res["summary"]["unresolved_critical"] >= 1
    assert res["summary"]["ml_readiness"] == "Not Ready"
    assert "Not ready" in res["summary"]["headline"]
    assert "ready for machine learning" not in res["summary"]["headline"].lower()


# 7. Acceptance test: Post-event target leakage (cancellation date in churn)
def test_post_event_leakage_detection():
    rng = np.random.default_rng(0)
    churn = [0] * 50 + [1] * 50
    # cancellation_date only exists when customer churned!
    cancel_date = [None] * 50 + [f"2024-05-{i:02d}" for i in range(1, 51)]
    df = pd.DataFrame({
        "tenure": rng.integers(1, 24, 100),
        "cancellation_date": cancel_date,
        "churn": churn
    })
    res = run_analysis(df, target="churn")
    leak_findings = [f for f in res["findings"] if f["code"] == "target_leakage" and "cancellation_date" in f["columns"]]
    assert len(leak_findings) == 1
    f = leak_findings[0]
    assert f["severity"] in ("critical", "high")
    assert "post-event" in f["plain_text"].lower() or "prediction time" in f["plain_text"].lower()
