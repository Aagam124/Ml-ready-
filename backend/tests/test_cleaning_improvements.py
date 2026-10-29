import io
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

    # 4. Run the plan with enabled steps
    cleaned, is_train, rpt = run_plan(df, plan["steps"], target="churned", task_type="binary")

    # 5. Verify columns are NOT deleted from dataset (CL-03, CL-04)
    assert "customer_id" in cleaned.columns
    assert "name" in cleaned.columns
    assert "monthly_income_inr" in cleaned.columns

    # 6. Verify duplicate entity removed (CL-09)
    assert len(cleaned) == 29

    # 7. Verify years_as_customer is unchanged (CL-02: 22 not clipped to 20.68)
    assert cleaned["years_as_customer"].max() == 22.0

    # 8. Verify audit trail is produced (CL-12)
    assert len(rpt["audit_trail"]) > 0
    assert "total_cells_changed" in rpt

    # 9. Verify chained imputation leaves NO missing values in uses_mobile_app (CL-01)
    assert cleaned["uses_mobile_app"].isna().sum() == 0
    assert "Maybe" not in cleaned["uses_mobile_app"].values
