"""Phase 1 acceptance: a ~100 MB CSV must upload and profile in under 60 seconds.
Run with:  pytest -m slow -s
"""
import io
import os
import time

import pytest

from app.demo import generate_demo_dataframe


@pytest.mark.slow
def test_100mb_csv_uploads_and_profiles_under_60_seconds(client, tmp_path):
    df = generate_demo_dataframe(n_rows=1_050_000)
    path = tmp_path / "big.csv"
    df.to_csv(path, index=False)
    size_mb = os.path.getsize(path) / 1024 / 1024
    assert size_mb >= 95, f"test file only {size_mb:.0f} MB"

    start = time.perf_counter()
    with open(path, "rb") as f:
        r = client.post("/api/v1/datasets/upload", files={"file": ("big.csv", f, "text/csv")})
    elapsed = time.perf_counter() - start

    assert r.status_code == 201, r.text
    timings = r.json()["version"]["ingest"]["timings_ms"]
    print(f"\n{size_mb:.0f} MB, {r.json()['overview']['n_rows']:,} rows -> {elapsed:.1f}s total; stages(ms)={timings}")
    assert elapsed < 60
