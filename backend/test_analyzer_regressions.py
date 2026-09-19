import numpy as np
import pandas as pd

from analysis.pipeline import run_full_analysis
from analysis.profiler import profile_dataset
from analysis.quality import run_quality_checks


def test_semantic_profile_and_quality_findings():
    df = pd.DataFrame({
        "user_id": range(1, 21),
        "email": [f"user{i}@example.com" for i in range(20)],
        "signup_date": [f"2025-01-{(i % 9) + 1:02d}" for i in range(20)],
        "amount": [10.0] * 19 + [1000.0],
        "status": ["ok"] * 18 + ["N/A", "N/A"],
    })
    df.loc[2, "amount"] = np.nan
    df.loc[3, "status"] = None
    profile = profile_dataset(df)
    by_name = {item["column"]: item for item in profile["column_details"]}
    assert by_name["email"]["semantic_type"] == "email"
    assert by_name["signup_date"]["semantic_type"] == "datetime"
    assert by_name["user_id"]["semantic_type"] == "identifier"
    ui_col = {item["column"]: item for item in profile["columns"]}
    assert ui_col["email"]["kind"] == "categorical" and ui_col["email"]["top_values"]
    quality = run_quality_checks(df)
    assert quality["missing_values"]["total_missing_cells"] == 2
    assert quality["invalid_values"]["possible_missing_sentinels"]["status"]["count"] == 2
    assert quality["missingness_patterns"]["rows_with_missing"] == 2


def test_pipeline_handles_empty_and_small_frames():
    report = run_full_analysis(pd.DataFrame({"empty": pd.Series(dtype="float64")}))
    assert report["profile"]["n_rows"] == 0
    assert report["readiness_score"]["score"] == 100
    assert report["outliers"]["consensus"] == {}
