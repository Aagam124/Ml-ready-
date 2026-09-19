"""
pipeline.py
-----------
The one function that wires profiling -> quality -> outliers -> ML risks ->
scoring -> recommendations together. Both test_phase1.py and
routes/analysis.py call this so the sequence is defined in exactly one
place.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from analysis.profiler import profile_dataset
from analysis.quality import run_quality_checks
from analysis.outliers import detect_outliers
from analysis.ml_risks import run_ml_risk_analysis
from analysis.scoring import calculate_readiness_score
from analysis.recommendations import generate_recommendations
from json_utils import sanitize_for_json


def run_full_analysis(df: pd.DataFrame, target_col: Optional[str] = None) -> dict:
    profile = profile_dataset(df)
    quality = run_quality_checks(df)
    outliers = detect_outliers(df, profile["numerical_columns"])
    risks = run_ml_risk_analysis(
        df, profile["numerical_columns"], profile["categorical_columns"], target_col
    )
    score = calculate_readiness_score(quality, outliers, risks)
    recommendations = generate_recommendations(df, quality, outliers, risks)

    return sanitize_for_json({
        "profile": profile,
        "quality": quality,
        "outliers": outliers,
        "ml_risks": risks,
        "readiness_score": score,
        "recommendations": recommendations,
    })
