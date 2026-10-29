"""Integration tests for the Quality Rules API."""


def test_quality_rules_crud_and_checks(auth_client, database):
    # Register user
    r = auth_client.post("/api/v1/auth/register", json={"email": "rules_user@example.com", "password": "password12345"})
    assert r.status_code == 201
    tok = r.json()["access_token"]
    headers = {"Authorization": f"Bearer {tok}"}

    # 1. List rules (initially empty)
    res = auth_client.get("/api/v1/quality-rules", headers=headers)
    assert res.status_code == 200 and res.json()["total"] == 0

    # 2. Create rule
    res = auth_client.post("/api/v1/quality-rules", headers=headers, json={
        "name": "Age must be >= 18",
        "rule_type": "numeric_range",
        "column": "age",
        "params": {"min": 18, "max": 100},
        "severity": "high",
        "description": "Ensure legal adult age constraint"
    })
    assert res.status_code == 201
    rule_id = res.json()["rule"]["id"]

    # 3. Create second rule
    res2 = auth_client.post("/api/v1/quality-rules", headers=headers, json={
        "name": "Dataset has at least 100 rows",
        "rule_type": "min_row_count",
        "params": {"min_rows": 100},
        "severity": "medium",
        "description": "Statistical significance requirement"
    })
    assert res2.status_code == 201

    # 4. List rules (now 2)
    res = auth_client.get("/api/v1/quality-rules", headers=headers)
    assert res.status_code == 200 and res.json()["total"] == 2

    # 5. Upload demo dataset
    ds_res = auth_client.post("/api/v1/datasets/demo", headers=headers)
    assert ds_res.status_code == 201
    ds_id = ds_res.json()["dataset"]["id"]

    # 6. Run quality checks on the dataset
    chk_res = auth_client.post(f"/api/v1/datasets/{ds_id}/quality-checks", headers=headers)
    assert chk_res.status_code == 200
    data = chk_res.json()
    assert data["total_rules"] == 2
    assert "checks" in data
    # The demo dataset has intentional planted impossible age (2.4, 140.0), so age rule should fail
    age_check = next(c for c in data["checks"] if c["rule_type"] == "numeric_range")
    assert age_check["status"] == "failed"
    assert age_check["failure_count"] > 0

    # 7. Delete rule
    del_res = auth_client.delete(f"/api/v1/quality-rules/{rule_id}", headers=headers)
    assert del_res.status_code == 200

    # 8. List rules (now 1)
    res = auth_client.get("/api/v1/quality-rules", headers=headers)
    assert res.status_code == 200 and res.json()["total"] == 1
