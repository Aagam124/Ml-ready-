import hashlib
import io

import pandas as pd
import pytest

from app.api.deps import get_workspace
from app.db import get_database
from app.demo import generate_demo_dataframe

CSV = generate_demo_dataframe().to_csv(index=False).encode()


def upload(client, data=CSV, name="customers.csv", **kw):
    return client.post("/api/v1/datasets/upload", files={"file": (name, io.BytesIO(data), "text/csv")}, **kw)


def test_health_and_ready(client):
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/ready").status_code == 200


def test_upload_profiles_the_demo_dataset_correctly(client):
    r = upload(client)
    assert r.status_code == 201, r.text
    body = r.json()
    ov = body["overview"]
    assert (ov["n_rows"], ov["n_cols"]) == (825, 11)
    assert ov["duplicate_rows"]["count"] == 25 and ov["duplicate_rows"]["pct"] == 3.03
    assert "denominator" in ov["missing"] and "denominator" in ov["duplicate_rows"]

    prof = client.get(f"/api/v1/datasets/{body['dataset']['id']}/profile").json()["profile"]
    types = {c["name"]: c["semantic_type"] for c in prof["columns"]}
    assert types["customer_id"] == "identifier"
    assert types["email"] == "email"
    assert types["country"] == "constant"
    assert types["churned"] == "binary" and types["account_type"] == "binary"
    assert types["department"] == "categorical"
    assert types["salary"] == "continuous" and types["age"] == "continuous"
    # The bug from the original report: these two are plain numbers, NOT temporal.
    assert types["tenure_years"] == "continuous" and types["tenure_months"] == "count"
    # Impossible values are surfaced in the stats for the later checks to act on.
    sal = next(c for c in prof["columns"] if c["name"] == "salary")
    assert sal["stats"]["n_negative"] == 6


def test_download_returns_identical_bytes_and_parquet_is_valid(client):
    ds_id = upload(client).json()["dataset"]["id"]
    got = client.get(f"/api/v1/datasets/{ds_id}/download")
    assert hashlib.sha256(got.content).hexdigest() == hashlib.sha256(CSV).hexdigest()
    pq = client.get(f"/api/v1/datasets/{ds_id}/download?kind=parquet")
    assert len(pd.read_parquet(io.BytesIO(pq.content))) == 825


def test_list_get_versions_search_and_pagination(client):
    a = upload(client, name="alpha.csv", data=b"x,y\n1,2\n3,4\n").json()["dataset"]["id"]
    upload(client, name="beta.csv", data=b"x,y\n1,2\n3,4\n")
    assert client.get("/api/v1/datasets").json()["total"] == 2
    assert client.get("/api/v1/datasets?q=alp").json()["total"] == 1
    assert len(client.get("/api/v1/datasets?limit=1").json()["items"]) == 1
    assert client.get(f"/api/v1/datasets/{a}").status_code == 200
    versions = client.get(f"/api/v1/datasets/{a}/versions").json()["items"]
    assert len(versions) == 1 and "profile" not in versions[0] and versions[0]["parent_version_id"] is None


def test_demo_endpoint(client):
    r = client.post("/api/v1/datasets/demo")
    assert r.status_code == 201 and r.json()["dataset"]["name"].startswith("Demo")


def test_delete_removes_everything(client, database):
    ds_id = upload(client).json()["dataset"]["id"]
    assert client.delete(f"/api/v1/datasets/{ds_id}").status_code == 200
    assert client.get(f"/api/v1/datasets/{ds_id}").status_code == 404
    assert database.versions.count_documents({}) == 0
    assert database.db["fs.files"].count_documents({}) == 0, "GridFS files must not be left behind"


def test_errors_use_the_uniform_shape(client):
    r = upload(client, data=b"hello", name="notes.docx")
    assert r.status_code == 422 and set(r.json()) == {"code", "message", "details"}
    assert r.json()["code"] == "unsupported_file_type"
    assert client.get("/api/v1/datasets/not-an-id").json()["code"] == "not_found"
    assert client.get("/api/v1/datasets?limit=0").json()["code"] == "validation_error"


def test_content_must_match_extension(client):
    r = upload(client, data=b"PK\x03\x04junk", name="fake.csv")
    assert r.status_code == 422 and r.json()["code"] == "content_mismatch"
    r = upload(client, data=b"not a zip at all", name="fake.xlsx")
    assert r.json()["code"] == "content_mismatch"


def test_empty_file_and_oversize_rejected(client):
    assert upload(client, data=b"", name="e.csv").json()["code"] == "empty_file"
    big = b"a,b\n" + b"1,2\n" * (6 * 1024 * 1024 // 4)   # > 5 MB test limit
    r = upload(client, data=big)
    assert r.status_code == 413 and r.json()["code"] == "file_too_large"


def test_filename_is_sanitised(client):
    r = upload(client, data=b"a,b\n1,2\n3,4\n", name="../../etc/pass wd?.csv")
    assert r.status_code == 201
    assert "/" not in r.json()["version"]["original_filename"]


def test_workspace_isolation(client, database):
    ds_id = upload(client).json()["dataset"]["id"]
    other = database.workspaces.insert_one({"slug": "other", "name": "Other"}).inserted_id
    client.app.dependency_overrides[get_workspace] = lambda: {"_id": other, "slug": "other"}
    assert client.get(f"/api/v1/datasets/{ds_id}").status_code == 404
    assert client.get(f"/api/v1/datasets/{ds_id}/download").status_code == 404
    assert client.delete(f"/api/v1/datasets/{ds_id}").status_code == 404
    assert client.get("/api/v1/datasets").json()["total"] == 0


def test_failed_ingest_leaves_no_orphans(client, database):
    r = upload(client, data=b"a,b,c\n", name="header_only.csv")
    assert r.status_code == 422
    assert database.db["fs.files"].count_documents({}) == 0
    assert database.datasets.count_documents({}) == 0
