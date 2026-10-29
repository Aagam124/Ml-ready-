"""Tests and benchmarks for the 1.2 Million+ Row Engine."""
import pytest

from app.engine.benchmark_1_2m import generate_benchmark_workload, run_benchmark


def test_workload_generators():
    df_mixed = generate_benchmark_workload(n_rows=10_000, workload="mixed")
    assert len(df_mixed) == 10_000
    assert "salary" in df_mixed.columns and "churned" in df_mixed.columns
    assert df_mixed["salary"].isna().sum() > 0

    df_narrow = generate_benchmark_workload(n_rows=10_000, workload="numeric_narrow")
    assert len(df_narrow) == 10_000
    assert set(df_narrow.columns) == {"id", "v1", "v2", "v3", "v4", "target"}

    df_cat = generate_benchmark_workload(n_rows=10_000, workload="categorical_heavy")
    assert len(df_cat) == 10_000
    assert "city" in df_cat.columns and "membership_tier" in df_cat.columns


def test_benchmark_runner_execution_and_metrics():
    # Fast test run on 25,000 rows to verify benchmark metrics & correctness
    res = run_benchmark(n_rows=25_000, workload="mixed", target_column="churned")

    assert res.n_rows == 25_000
    assert res.correctness_verified is True
    assert res.read_seconds > 0.0
    assert res.profile_seconds > 0.0
    assert res.analysis_seconds > 0.0
    assert res.peak_memory_mb > 0.0
    assert res.throughput_rows_per_sec > 1000.0


@pytest.mark.slow
def test_full_1_2_million_row_benchmark():
    # Full 1.2 Million record benchmark (run with pytest -m slow)
    res = run_benchmark(n_rows=1_200_000, workload="mixed", target_column="churned")

    assert res.n_rows == 1_200_000
    assert res.correctness_verified is True
    assert res.file_size_mb >= 80.0
    assert res.total_seconds < 120.0
    assert res.peak_memory_mb < 3000.0  # bounded memory under 3 GB RAM
