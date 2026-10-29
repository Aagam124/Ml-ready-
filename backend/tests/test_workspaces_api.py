"""Tests for workspaces API: metadata, plan tiers, audit logs, and limits."""


def test_workspace_current_and_plan_upgrade(auth_client):
    # Register user
    r = auth_client.post("/api/v1/auth/register", json={"email": "workspace_test@example.com", "password": "password12345"})
    assert r.status_code == 201
    tok = r.json()["access_token"]
    headers = {"Authorization": f"Bearer {tok}"}

    # Get current workspace info
    r = auth_client.get("/api/v1/workspaces/current", headers=headers)
    assert r.status_code == 200
    data = r.json()
    assert "usage" in data
    assert data["plan"] == "free"
    assert data["limits"]["max_datasets"] == 10

    # Upgrade to pro plan
    r = auth_client.post("/api/v1/workspaces/plan", headers=headers, json={"plan": "pro"})
    assert r.status_code == 200
    res = r.json()
    assert res["success"] is True
    assert res["new_plan"] == "pro"
    assert res["limits"]["max_rows"] == 2_000_000

    # Check audit log reflects the upgrade
    r = auth_client.get("/api/v1/workspaces/audit-log", headers=headers)
    assert r.status_code == 200
    audit_data = r.json()
    assert audit_data["total"] >= 1
    actions = [it["action"] for it in audit_data["items"]]
    assert "workspace.upgrade_plan" in actions
