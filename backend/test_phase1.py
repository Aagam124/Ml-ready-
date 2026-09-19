"""
MLReady - Phase 1 test harness.

Run this to prove the Data Science engine works end-to-end on a real CSV,
independent of any API/DB/frontend. This is exactly the function call
sequence that FastAPI routes will wrap in Phase 3, and it's what any
future contributor should run first to sanity-check the engine.

Usage:
    python3 test_phase1.py [path/to.csv] [target_column]

Defaults to sample_data/sample_customers.csv with target 'churned'.
"""

import json
import sys

import pandas as pd

from analysis.pipeline import run_full_analysis

# Windows consoles default to cp1252, which can't encode symbols like "≈"
# used in deduction text. Reconfigure before any report printing.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def run_pipeline(csv_path: str, target_col: str | None) -> dict:
    df = pd.read_csv(csv_path)
    return run_full_analysis(df, target_col)


def print_report(report: dict):
    p = report["profile"]
    print("=" * 70)
    print("MLREADY - PHASE 1 ENGINE TEST")
    print("=" * 70)

    print(f"\nRows: {p['n_rows']}  |  Columns: {p['n_columns']}")
    print(f"Numerical columns ({len(p['numerical_columns'])}): {p['numerical_columns']}")
    print(f"Categorical columns ({len(p['categorical_columns'])}): {p['categorical_columns']}")

    q = report["quality"]
    print("\n--- DATA QUALITY ---")
    print(f"Missing values: {q['missing_values']['overall_percentage']}% of all cells")
    for col, info in q["missing_values"]["columns_with_missing"].items():
        print(f"  - {col}: {info['percentage']}%")
    print(f"Duplicate rows: {q['duplicates']['count']} ({q['duplicates']['percentage']}%)")
    print(f"Constant columns: {q['constant_columns']}")
    print(f"Near-constant columns: {list(q['near_constant_columns'].keys())}")
    print(f"Likely mistyped columns: {list(q['dtype_issues'].keys())}")
    print(f"High-cardinality columns: {list(q['high_cardinality_columns'].keys())}")

    o = report["outliers"]
    print("\n--- OUTLIERS ---")
    print(f"IQR-flagged columns: {o['iqr']}")
    iso = o["isolation_forest"]
    if iso.get("applied"):
        print(f"Isolation Forest: {iso['outlier_count']} rows flagged ({iso['percentage']}%)")
    else:
        print(f"Isolation Forest: skipped ({iso.get('reason')})")

    r = report["ml_risks"]
    print("\n--- ML RISK ANALYSIS ---")
    ci = r["class_imbalance"]
    print(f"Class imbalance: {ci}")
    print(f"High-correlation pairs: {r['multicollinearity']['high_correlation_pairs']}")
    print(f"High-VIF features: {r['multicollinearity']['high_vif_features']}")
    print(f"Potential leakage columns: {[c['column'] for c in r['potential_leakage']['potential_leakage_columns']]}")
    print(f"High-cardinality ML risk features: {list(r['high_cardinality_features'].keys())}")

    s = report["readiness_score"]
    print("\n--- ML READINESS SCORE ---")
    print(f"Score: {s['score']}/100  ({s['category']})")
    for d in s["deductions"]:
        print(f"  {d['issue']}: {d['points']}")

    print("\n--- RECOMMENDATIONS ---")
    for i, rec in enumerate(report["recommendations"], 1):
        print(f"  {i}. {rec}")

    print("\n" + "=" * 70)
    print("Phase 1 engine ran successfully end-to-end.")
    print("=" * 70)


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "sample_data/sample_customers.csv"
    target_col = sys.argv[2] if len(sys.argv) > 2 else "churned"

    report = run_pipeline(csv_path, target_col)
    print_report(report)

    with open("phase1_report_output.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print("\nFull JSON report written to phase1_report_output.json")
