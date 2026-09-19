"""
test_phase6.py
---------------
Exercises the report assembly and export at two points: right after upload
(nothing analyzed yet - report must degrade gracefully, not error out) and
after the full pipeline has run (analyze, clean, train) - where the report
should contain everything.
"""

import mongomock
from fastapi.testclient import TestClient

import config
from database.mongodb import MLReadyDatabase
from dependencies import get_db
from main import app

test_db = MLReadyDatabase(client=mongomock.MongoClient())
app.dependency_overrides[get_db] = lambda: test_db
client = TestClient(app)


def section(title):
    print(f"\n--- {title} ---")


def main():
    print("=" * 70)
    print("MLREADY - PHASE 6 REPORTS TEST")
    print("=" * 70)

    client.post("/auth/register", json={
        "username": "aj_reports", "email": "aj.reports@example.com", "password": "correct-horse-battery",
    })
    token = client.post("/auth/login", json={
        "username": "aj_reports", "password": "correct-horse-battery",
    }).json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    with open("sample_data/sample_customers.csv", "rb") as f:
        dataset = client.post("/datasets/upload", headers=headers,
                               files={"file": ("sample_customers.csv", f, "text/csv")}).json()

    section("REPORT BEFORE ANY ANALYSIS - must degrade gracefully")
    early = client.get(f"/datasets/{dataset['id']}/full-report", headers=headers)
    assert early.status_code == 200, early.text
    bundle = early.json()
    assert bundle["analysis"] is None
    assert bundle["cleaning_history"] == []
    assert bundle["model_comparison"] is None
    print("Full report before analysis: analysis=None, empty history, no model comparison - as expected")

    early_html = client.get(f"/datasets/{dataset['id']}/report/export?format=html", headers=headers)
    assert early_html.status_code == 200
    assert "not been analyzed yet" in early_html.text
    assert early_html.headers["content-type"].startswith("text/html")
    print(f"HTML export before analysis: {len(early_html.content)} bytes, says 'not analyzed yet'")

    early_pdf = client.get(f"/datasets/{dataset['id']}/report/export?format=pdf", headers=headers)
    assert early_pdf.status_code == 200
    assert early_pdf.headers["content-type"] == "application/pdf"
    assert early_pdf.content[:4] == b"%PDF"
    print(f"PDF export before analysis: {len(early_pdf.content)} bytes, valid PDF header")

    section("RUN THE FULL PIPELINE")
    client.post(f"/datasets/{dataset['id']}/analyze", headers=headers, json={"target_column": "churned"})
    client.post(f"/datasets/{dataset['id']}/clean", headers=headers, json={
        "operations": [
            {"type": "impute_missing", "column": "age", "strategy": "mean"},
            {"type": "impute_missing", "column": "salary", "strategy": "median"},
            {"type": "remove_duplicates"},
        ],
    })
    client.post(f"/datasets/{dataset['id']}/train", headers=headers, json={"target_column": "churned"})
    print("Analyzed, cleaned, and trained")

    section("FULL REPORT AFTER THE PIPELINE")
    full = client.get(f"/datasets/{dataset['id']}/full-report", headers=headers)
    assert full.status_code == 200
    bundle = full.json()
    assert bundle["analysis"] is not None
    assert len(bundle["cleaning_history"]) == 1
    assert bundle["model_comparison"] is not None
    assert bundle["model_comparison"]["after"] is not None
    print(f"Report now has: analysis score {bundle['analysis']['readiness_score']['score']}, "
          f"{len(bundle['cleaning_history'])} cleaning run(s), "
          f"model comparison with before AND after")

    html = client.get(f"/datasets/{dataset['id']}/report/export?format=html", headers=headers)
    assert html.status_code == 200
    assert "Readiness score" in html.text and "Model performance" in html.text
    assert 'attachment; filename="sample_customers_report.html"' in html.headers["content-disposition"]
    print(f"HTML export: {len(html.content)} bytes, correct filename in Content-Disposition")

    pdf = client.get(f"/datasets/{dataset['id']}/report/export?format=pdf", headers=headers)
    assert pdf.status_code == 200
    assert pdf.content[:4] == b"%PDF"
    assert 'attachment; filename="sample_customers_report.pdf"' in pdf.headers["content-disposition"]
    print(f"PDF export: {len(pdf.content)} bytes, valid PDF, correct filename")

    section("BAD FORMAT PARAM")
    bad_format = client.get(f"/datasets/{dataset['id']}/report/export?format=docx", headers=headers)
    assert bad_format.status_code == 422
    print("format=docx correctly rejected (422) - only html/pdf allowed")

    section("OWNERSHIP")
    client.post("/auth/register", json={
        "username": "someone_else2", "email": "else2@example.com", "password": "correct-horse-battery",
    })
    other_token = client.post("/auth/login", json={
        "username": "someone_else2", "password": "correct-horse-battery",
    }).json()["access_token"]
    stolen = client.get(f"/datasets/{dataset['id']}/full-report",
                         headers={"Authorization": f"Bearer {other_token}"})
    assert stolen.status_code == 404
    print("Another user cannot read this dataset's report (404)")

    test_db.close()
    import shutil
    shutil.rmtree(config.UPLOAD_DIR, ignore_errors=True)

    print("\n" + "=" * 70)
    print("Phase 6 reports verified: graceful degradation, full assembly, HTML + PDF export, ownership.")
    print("=" * 70)


if __name__ == "__main__":
    main()
