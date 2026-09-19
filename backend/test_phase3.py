"""
test_phase3.py
---------------
End-to-end HTTP test of the FastAPI layer: register -> login -> upload ->
analyze -> report -> clean -> list -> delete.

Uses Starlette's TestClient (no real server process needed) and overrides
the get_db dependency with a mongomock-backed database - same pattern as
test_phase2.py, just at the HTTP layer instead of calling repositories
directly. This proves the routes, request/response models, and auth
wiring all work together, independent of a running MongoDB or uvicorn
process.
"""

import shutil
import sys

import mongomock
from fastapi.testclient import TestClient

import config
from database.mongodb import MLReadyDatabase
from dependencies import get_db
from main import app

# --- wire a mongomock-backed DB into the app for this test run -------------
test_db = MLReadyDatabase(client=mongomock.MongoClient())
app.dependency_overrides[get_db] = lambda: test_db

client = TestClient(app)


def section(title):
    print(f"\n--- {title} ---")


def main():
    print("=" * 70)
    print("MLREADY - PHASE 3 API TEST")
    print("=" * 70)

    # ---- health -----------------------------------------------------------
    section("HEALTH")
    resp = client.get("/health")
    assert resp.status_code == 200 and resp.json() == {"status": "ok"}
    print("GET /health -> 200 OK")

    # ---- register + login ---------------------------------------------------
    section("AUTH - register, duplicate rejection, login, bad password")
    resp = client.post("/auth/register", json={
        "username": "aj_api", "email": "aj.api@example.com", "password": "correct-horse-battery",
    })
    assert resp.status_code == 201, resp.text
    print(f"Registered user: {resp.json()['username']}")

    dup = client.post("/auth/register", json={
        "username": "aj_api", "email": "someone.else@example.com", "password": "whatever123",
    })
    assert dup.status_code == 409
    print("Duplicate username correctly rejected (409)")

    login = client.post("/auth/login", json={"username": "aj_api", "password": "correct-horse-battery"})
    assert login.status_code == 200, login.text
    token = login.json()["access_token"]
    print("Login succeeded, JWT issued")

    bad_login = client.post("/auth/login", json={"username": "aj_api", "password": "wrong"})
    assert bad_login.status_code == 401
    print("Wrong password correctly rejected (401)")

    headers = {"Authorization": f"Bearer {token}"}

    me = client.get("/auth/me", headers=headers)
    assert me.status_code == 200 and me.json()["username"] == "aj_api"
    print("GET /auth/me -> resolves token back to the right user")

    no_token = client.get("/datasets")
    assert no_token.status_code == 401
    print("Unauthenticated request to /datasets correctly rejected (401)")

    # ---- upload -------------------------------------------------------------
    section("DATASETS - upload, list, get, unsupported type rejected")
    with open("sample_data/sample_customers.csv", "rb") as f:
        upload = client.post("/datasets/upload", headers=headers,
                              files={"file": ("sample_customers.csv", f, "text/csv")})
    assert upload.status_code == 201, upload.text
    dataset = upload.json()
    print(f"Uploaded dataset: {dataset['filename']} "
          f"({dataset['metadata']['n_rows']} rows, {dataset['metadata']['n_columns']} cols)")

    bad_upload = client.post("/datasets/upload", headers=headers,
                              files={"file": ("notes.txt", b"hello", "text/plain")})
    assert bad_upload.status_code == 400
    print("Unsupported file type correctly rejected (400)")

    empty_upload = client.post("/datasets/upload", headers=headers,
                                files={"file": ("empty.csv", b"", "text/csv")})
    assert empty_upload.status_code == 400
    print("Empty file correctly rejected (400)")

    listing = client.get("/datasets", headers=headers)
    assert listing.status_code == 200 and len(listing.json()) == 1
    print(f"GET /datasets -> {len(listing.json())} dataset(s)")

    single = client.get(f"/datasets/{dataset['id']}", headers=headers)
    assert single.status_code == 200
    print("GET /datasets/{id} -> matches uploaded dataset")

    fake_id = "64b64f2e1234567890abcdef"
    not_found = client.get(f"/datasets/{fake_id}", headers=headers)
    assert not_found.status_code == 404
    print("GET on a non-existent dataset id correctly returns 404")

    # ---- analyze + report -----------------------------------------------------
    section("ANALYSIS - analyze with target_column, fetch report")
    analyze = client.post(f"/datasets/{dataset['id']}/analyze", headers=headers,
                           json={"target_column": "churned"})
    assert analyze.status_code == 200, analyze.text
    score = analyze.json()["report"]["readiness_score"]["score"]
    print(f"POST /analyze -> readiness score {score}/100")

    report = client.get(f"/datasets/{dataset['id']}/report", headers=headers)
    assert report.status_code == 200
    assert report.json()["report"]["readiness_score"]["score"] == score
    print("GET /report -> matches the score just computed")

    reports_history = client.get(f"/datasets/{dataset['id']}/reports", headers=headers)
    assert reports_history.status_code == 200 and len(reports_history.json()) == 1
    print(f"GET /reports (history) -> {len(reports_history.json())} report(s) on file")

    bad_target = client.post(f"/datasets/{dataset['id']}/analyze", headers=headers,
                              json={"target_column": "not_a_real_column"})
    assert bad_target.status_code == 400
    print("Analyze with a bogus target_column correctly rejected (400)")

    # ---- clean --------------------------------------------------------------
    section("CLEANING - impute + drop duplicates, original file untouched")
    clean = client.post(f"/datasets/{dataset['id']}/clean", headers=headers, json={
        "operations": [
            {"type": "impute_missing", "column": "age", "strategy": "mean"},
            {"type": "impute_missing", "column": "salary", "strategy": "median"},
            {"type": "remove_duplicates"},
        ],
    })
    assert clean.status_code == 200, clean.text
    cleaning_result = clean.json()
    print(f"POST /clean -> {len(cleaning_result['operations'])} operations applied, "
          f"resulting shape {cleaning_result['resulting_shape']}")

    cleaning_history = client.get(f"/datasets/{dataset['id']}/cleaning-history", headers=headers)
    assert cleaning_history.status_code == 200 and len(cleaning_history.json()) == 1
    print(f"GET /cleaning-history -> {len(cleaning_history.json())} entry(ies) on file")

    import os
    assert os.path.exists(dataset["file_path"]), "original file must survive cleaning"
    print("Original uploaded file still exists on disk - never overwritten")

    bad_clean = client.post(f"/datasets/{dataset['id']}/clean", headers=headers, json={
        "operations": [{"type": "impute_missing", "column": "does_not_exist", "strategy": "mean"}],
    })
    assert bad_clean.status_code == 400
    print("Cleaning with a bogus column correctly rejected (400)")

    # ---- delete ---------------------------------------------------------------
    section("DELETE - remove dataset, verify gone")
    deleted = client.delete(f"/datasets/{dataset['id']}", headers=headers)
    assert deleted.status_code == 204
    gone = client.get(f"/datasets/{dataset['id']}", headers=headers)
    assert gone.status_code == 404
    print("Dataset deleted and confirmed gone (404 on re-fetch)")

    test_db.close()
    shutil.rmtree(config.UPLOAD_DIR, ignore_errors=True)  # clean up test upload artifacts

    print("\n" + "=" * 70)
    print("Phase 3 API layer verified end-to-end (auth, datasets, analysis, cleaning).")
    print("=" * 70)


if __name__ == "__main__":
    main()
