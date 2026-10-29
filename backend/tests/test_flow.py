"""The core journey through the real API, with the worker run inline: demo -> target -> analysis -> plan -> apply -> models -> report."""
import pytest

from app.jobs.worker import Worker


@pytest.fixture()
def session(auth_client, database, settings):
    tok = auth_client.post("/api/v1/auth/guest").json()
    h = {"Authorization": f"Bearer {tok['access_token']}"}
    worker = Worker(database, settings)

    def drain():
        while worker.run_once():
            pass

    return auth_client, h, drain, database


def test_core_journey(session):
    c, h, drain, database = session
    ds = c.post("/api/v1/datasets/demo", headers=h).json()["dataset"]["id"]

    sug = c.get(f"/api/v1/datasets/{ds}/target-suggestions", headers=h).json()["candidates"]
    assert sug[0]["column"] == "churned" and sug[0]["task_type"] == "binary"

    # analysis without a target is allowed and provisional
    prov = c.post(f"/api/v1/datasets/{ds}/analyses", headers=h, json={}).json()
    drain()
    assert c.get(f"/api/v1/analyses/{prov['analysis_id']}", headers=h).json()["score"]["provisional"] is True

    a = c.post(f"/api/v1/datasets/{ds}/analyses", headers=h, json={"target_column": "churned"})
    assert a.status_code == 202
    aid, jid = a.json()["analysis_id"], a.json()["job_id"]
    drain()
    analysis = c.get(f"/api/v1/analyses/{aid}", headers=h).json()
    assert analysis["status"] == "done" and not analysis["score"]["provisional"] and len(analysis["fix_first"]) == 3
    assert round(analysis["score"]["total_deducted"], 2) == round(100 - analysis["score"]["value"], 2)

    crit = c.get(f"/api/v1/analyses/{aid}/findings?severity=critical,high", headers=h).json()
    assert {f["code"] for f in crit["items"]} >= {"target_leakage", "invalid_values"}
    assert c.get(f"/api/v1/analyses/{aid}/findings?category=leakage", headers=h).json()["total"] == 1

    # SSE for a finished job ends with a done event
    with c.stream("GET", f"/api/v1/jobs/{jid}/events", headers=h) as r:
        text = "".join(r.iter_text())
    assert "event: done" in text

    plan = c.post(f"/api/v1/analyses/{aid}/plan", headers=h).json()
    pid = plan["id"]
    steps = plan["steps"]
    imputes = [s for s in steps if s["op"] == "impute"]
    patch = c.patch(f"/api/v1/plans/{pid}", headers=h, json={"steps": [{"step_id": imputes[0]["step_id"], "enabled": False}]}).json()
    assert [s for s in patch["steps"] if s["step_id"] == imputes[0]["step_id"]][0]["enabled"] is False
    prev = c.post(f"/api/v1/plans/{pid}/steps/{steps[0]['step_id']}/preview", headers=h)
    assert prev.status_code == 200 and "rows_after" in prev.json()
    c.patch(f"/api/v1/plans/{pid}", headers=h, json={"steps": [{"step_id": imputes[0]["step_id"], "enabled": True}]})

    assert "def clean(df)" in c.get(f"/api/v1/plans/{pid}/export?kind=script", headers=h).text
    assert "build_preprocessor" in c.get(f"/api/v1/plans/{pid}/export?kind=pipeline", headers=h).text

    assert c.post(f"/api/v1/plans/{pid}/apply", headers=h).status_code == 202
    drain()
    done = c.get(f"/api/v1/plans/{pid}", headers=h).json()
    assert done["status"] == "applied"
    assert c.patch(f"/api/v1/plans/{pid}", headers=h, json={"steps": []}).status_code == 409      # applied plans are locked

    versions = c.get(f"/api/v1/datasets/{ds}/versions", headers=h).json()["items"]
    assert len(versions) == 2 and versions[1]["parent_version_id"] == versions[0]["id"]
    new_aid = done["result_analysis_id"]
    after = c.get(f"/api/v1/analyses/{new_aid}", headers=h).json()
    assert after["score"]["value"] > analysis["score"]["value"] + 20
    diff = c.get(f"/api/v1/analyses/compare/{aid}/{new_aid}", headers=h).json()
    assert diff["score_delta"] > 20 and any(f["code"] == "target_leakage" for f in diff["resolved"])

    mr = c.post("/api/v1/model-runs", headers=h, json={"plan_id": pid}).json()
    drain()
    run = c.get(f"/api/v1/model-runs/{mr['model_run_id']}", headers=h).json()
    assert run["status"] == "done" and run["result"]["comparison"]["verdict"] == "leakage_removed"

    for fmt, magic in (("html", b"<!doctype html>"), ("pdf", b"%PDF-")):
        rep = c.post(f"/api/v1/analyses/{aid}/reports", headers=h, json={"format": fmt, "model_run_id": mr["model_run_id"]})
        assert rep.status_code == 201, rep.text
        rid = rep.json()["id"]
        body = c.get(f"/api/v1/reports/{rid}/download", headers=h).content
        assert body.startswith(magic)
    share = c.post(f"/api/v1/reports/{rid}/share", headers=h, json={"days": 7}).json()
    assert c.get(share["share_url_path"]).status_code == 200                       # public, no auth
    assert c.get("/api/v1/shared/" + "x" * 30).status_code == 404
    c.delete(f"/api/v1/reports/{rid}/share", headers=h)
    assert c.get(share["share_url_path"]).status_code == 404

    undone = c.post(f"/api/v1/datasets/{ds}/undo", headers=h)
    assert undone.status_code == 200
    assert c.post(f"/api/v1/datasets/{ds}/undo", headers=h).status_code == 409     # already at the original

    gone = c.delete(f"/api/v1/datasets/{ds}", headers=h).json()
    assert gone["deleted"] and gone["counts"]["findings"] > 0
    assert database.db["fs.files"].count_documents({}) == 0          # no leftover files anywhere
    assert c.get(f"/api/v1/datasets/{ds}", headers=h).status_code == 404


def test_unsuitable_or_unknown_target_is_rejected_with_advice(session):
    c, h, _, _ = session
    ds = c.post("/api/v1/datasets/demo", headers=h).json()["dataset"]["id"]
    bad = c.post(f"/api/v1/datasets/{ds}/analyses", headers=h, json={"target_column": "customer_id"})
    assert bad.status_code == 422 and "Pick another column" in bad.json()["message"]
    assert c.post(f"/api/v1/datasets/{ds}/analyses", headers=h, json={"target_column": "nope"}).status_code == 422


def test_plan_requires_finished_analysis_and_enabled_steps(session):
    c, h, drain, _ = session
    ds = c.post("/api/v1/datasets/demo", headers=h).json()["dataset"]["id"]
    aid = c.post(f"/api/v1/datasets/{ds}/analyses", headers=h, json={"target_column": "churned"}).json()["analysis_id"]
    assert c.post(f"/api/v1/analyses/{aid}/plan", headers=h).status_code == 409        # not finished yet
    drain()
    plan = c.post(f"/api/v1/analyses/{aid}/plan", headers=h).json()
    pid = plan["id"]
    c.patch(f"/api/v1/plans/{pid}", headers=h, json={"steps": [{"step_id": s["step_id"], "enabled": False} for s in plan["steps"]]})
    assert c.post(f"/api/v1/plans/{pid}/apply", headers=h).status_code == 422


def test_metrics_version_and_health_endpoints(auth_client):
    assert auth_client.get("/health").json()["status"] == "ok"
    assert auth_client.get("/api/v1/version").json()["score"]
    assert "mlready_http_requests_total" in auth_client.get("/metrics").text
