"""
test_phase5.py
---------------
End-to-end test of model training: train on the original data (before any
cleaning), confirm the comparison honestly has no "after" yet, clean the
dataset, train again, and confirm a real before/after comparison with
deltas shows up.
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
    print("MLREADY - PHASE 5 MODEL TRAINING TEST")
    print("=" * 70)

    client.post("/auth/register", json={
        "username": "aj_models", "email": "aj.models@example.com", "password": "correct-horse-battery",
    })
    token = client.post("/auth/login", json={
        "username": "aj_models", "password": "correct-horse-battery",
    }).json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    with open("sample_data/sample_customers.csv", "rb") as f:
        dataset = client.post("/datasets/upload", headers=headers,
                               files={"file": ("sample_customers.csv", f, "text/csv")}).json()
    print(f"Uploaded {dataset['filename']}")

    section("TRAIN BEFORE ANY CLEANING")
    missing_target = client.post(f"/datasets/{dataset['id']}/train", headers=headers, json={})
    assert missing_target.status_code == 422
    print("POST /train with no target_column correctly rejected (422)")

    bad_target = client.post(f"/datasets/{dataset['id']}/train", headers=headers,
                              json={"target_column": "not_a_real_column"})
    assert bad_target.status_code == 400
    print("POST /train with a bogus target_column correctly rejected (400)")

    train1 = client.post(f"/datasets/{dataset['id']}/train", headers=headers,
                          json={"target_column": "churned"})
    assert train1.status_code == 200, train1.text
    result1 = train1.json()
    assert result1["before"] is not None
    assert result1["after"] is None
    assert "cleaned" in result1["after_skipped_reason"].lower() or "clean" in result1["after_skipped_reason"].lower()
    print(f"Trained on original data: {result1['before']['model_type']}, "
          f"metrics={result1['before']['metrics']}")
    print(f"Honestly reports no 'after' yet: \"{result1['after_skipped_reason']}\"")

    get1 = client.get(f"/datasets/{dataset['id']}/model-results", headers=headers)
    assert get1.status_code == 200
    assert get1.json()["before"]["metrics"] == result1["before"]["metrics"]
    print("GET /model-results matches what POST /train just returned")

    section("CLEAN, THEN TRAIN AGAIN")
    clean = client.post(f"/datasets/{dataset['id']}/clean", headers=headers, json={
        "operations": [
            {"type": "impute_missing", "column": "age", "strategy": "mean"},
            {"type": "impute_missing", "column": "salary", "strategy": "median"},
            {"type": "remove_duplicates"},
        ],
    })
    assert clean.status_code == 200, clean.text
    print("Cleaned the dataset")

    train2 = client.post(f"/datasets/{dataset['id']}/train", headers=headers,
                          json={"target_column": "churned"})
    assert train2.status_code == 200, train2.text
    result2 = train2.json()
    assert result2["before"] is not None and result2["after"] is not None
    assert result2["after_skipped_reason"] is None
    print(f"Before: {result2['before']['metrics']}")
    print(f"After:  {result2['after']['metrics']}")
    print(f"Deltas: {result2['deltas']}")
    assert set(result2["deltas"].keys()) == set(result2["before"]["metrics"].keys())
    print("Deltas cover exactly the real numeric metrics - no metadata leaked in")

    get2 = client.get(f"/datasets/{dataset['id']}/model-results", headers=headers)
    assert get2.status_code == 200 and get2.json()["after"] is not None
    print("GET /model-results now returns the full before/after comparison")

    section("NO ACCESS TO SOMEONE ELSE'S RESULTS")
    client.post("/auth/register", json={
        "username": "someone_else", "email": "else@example.com", "password": "correct-horse-battery",
    })
    other_token = client.post("/auth/login", json={
        "username": "someone_else", "password": "correct-horse-battery",
    }).json()["access_token"]
    stolen = client.get(f"/datasets/{dataset['id']}/model-results",
                         headers={"Authorization": f"Bearer {other_token}"})
    assert stolen.status_code == 404
    print("Another user cannot read this dataset's model results (404)")

    section("TARGET-MISMATCH PROTECTION - train on a different target, must not cross-pair")
    train_diff_target = client.post(f"/datasets/{dataset['id']}/train", headers=headers,
                                     json={"target_column": "account_type"})
    assert train_diff_target.status_code == 200, train_diff_target.text
    result3 = train_diff_target.json()
    assert result3["target_column"] == "account_type"
    assert result3["before"]["details"]["target_column"] == "account_type"
    assert result3["after"]["details"]["target_column"] == "account_type"
    print(f"Trained on 'account_type' instead: before={result3['before']['metrics']}, "
          f"after={result3['after']['metrics']}")

    # The dataset now has TWO complete before/after pairs stored (churned,
    # account_type). GET /model-results with no target specified must return
    # the most RECENT one (account_type) - matched, not mixed with 'churned'.
    latest = client.get(f"/datasets/{dataset['id']}/model-results", headers=headers).json()
    assert latest["target_column"] == "account_type"
    assert latest["before"]["details"]["target_column"] == "account_type"
    assert latest["after"]["details"]["target_column"] == "account_type"
    print("GET /model-results with no target specified correctly returns the most recent "
          "target's matched before/after pair - not a stale cross-target mix")

    section("DIRECT PROOF: a stale cross-target 'after' must NOT get paired")
    # Simulates the exact bug scenario: training on target A succeeded fully
    # (before_A + after_A stored), then training on target B later only got
    # as far as storing before_B (e.g. the cleaned version failed to train).
    # The old compare_before_after took the latest of EACH stage independently,
    # which would have wrongly paired before_B with the stale after_A.
    test_db.model_results.create_result(
        dataset_id=dataset["id"], user_id="000000000000000000000001", stage="before",
        model_type="RandomForestClassifier", metrics={"accuracy": 0.5},
        details={"target_column": "STALE_TARGET_A"},
    )
    test_db.model_results.create_result(
        dataset_id=dataset["id"], user_id="000000000000000000000001", stage="after",
        model_type="RandomForestClassifier", metrics={"accuracy": 0.6},
        details={"target_column": "STALE_TARGET_A"},
    )
    test_db.model_results.create_result(
        dataset_id=dataset["id"], user_id="000000000000000000000001", stage="before",
        model_type="RandomForestClassifier", metrics={"accuracy": 0.7},
        details={"target_column": "TARGET_B_NO_AFTER_YET"},
    )
    direct = test_db.model_results.compare_before_after(dataset["id"])
    assert direct["target_column"] == "TARGET_B_NO_AFTER_YET"
    assert direct["before"]["metrics"]["accuracy"] == 0.7
    assert direct["after"] is None, (
        f"BUG: paired before(target=TARGET_B_NO_AFTER_YET) with a stale "
        f"after(target=STALE_TARGET_A) instead of correctly reporting no after yet: {direct}"
    )
    assert direct["deltas"] == {}
    print("Confirmed: a stale 'after' from a different target is correctly ignored, "
          "not silently cross-paired with the new target's 'before'")

    test_db.close()
    import shutil
    shutil.rmtree(config.UPLOAD_DIR, ignore_errors=True)

    print("\n" + "=" * 70)
    print("Phase 5 model training verified end-to-end: before, honest no-after, clean, after, deltas.")
    print("=" * 70)


if __name__ == "__main__":
    main()
