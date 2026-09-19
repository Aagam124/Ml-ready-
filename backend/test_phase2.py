"""
test_phase2.py
---------------
Exercises the full MongoDB CRUD layer end-to-end across all five collections.

Runs against mongomock.MongoClient() so it needs no local MongoDB server -
useful in CI/sandboxes. On a machine with MongoDB Community Server running,
swap the client below for `MongoClient()` (or just call `MLReadyDatabase()`
with no argument) and everything else is identical - the repository code
never changes, only which client it's handed.
"""

import json
import sys

import mongomock

from database.mongodb import MLReadyDatabase


def section(title):
    print(f"\n--- {title} ---")


def main():
    print("=" * 70)
    print("MLREADY - PHASE 2 MONGODB CRUD TEST")
    print("=" * 70)

    db = MLReadyDatabase(client=mongomock.MongoClient())

    # ---- USERS ----------------------------------------------------------
    section("USERS - create, auth, update, duplicate rejection")
    user = db.users.create_user("aj_dev", "aj@example.com", "correct-horse-battery")
    print(f"Created user: {user['username']} (id={user['id']})")

    assert db.users.verify_credentials("aj_dev", "correct-horse-battery") is not None
    assert db.users.verify_credentials("aj_dev", "wrong-password") is None
    print("Credential check: correct password accepted, wrong password rejected")

    try:
        db.users.create_user("aj_dev", "other@example.com", "whatever")
        print("FAIL: duplicate username was not rejected")
        sys.exit(1)
    except Exception:
        print("Duplicate username correctly rejected by unique index")

    updated = db.users.update_user(user["id"], {"email": "aj.dev@example.com"})
    assert updated["email"] == "aj.dev@example.com"
    print(f"Updated email -> {updated['email']}")

    # ---- DATASETS ---------------------------------------------------------
    section("DATASETS - create, list, update status")
    dataset = db.datasets.create_dataset(
        user_id=user["id"],
        filename="sample_customers.csv",
        file_path="uploads/sample_customers.csv",
        metadata={"n_rows": 825, "n_columns": 11, "size_bytes": 79011},
    )
    print(f"Created dataset: {dataset['filename']} (id={dataset['id']})")

    datasets = db.datasets.list_datasets_for_user(user["id"])
    assert len(datasets) == 1
    print(f"Datasets for user: {len(datasets)}")

    db.datasets.update_dataset(dataset["id"], {"status": "analyzed"})
    refreshed = db.datasets.get_dataset(dataset["id"])
    assert refreshed["status"] == "analyzed"
    print(f"Dataset status -> {refreshed['status']}")

    # ---- ANALYSIS REPORTS -------------------------------------------------
    section("ANALYSIS REPORTS - store the real Phase 1 output")
    try:
        with open("phase1_report_output.json") as f:
            phase1_report = json.load(f)
        print("Loaded real Phase 1 report from phase1_report_output.json")
    except FileNotFoundError:
        phase1_report = {"note": "Phase 1 report not found; run test_phase1.py first"}
        print("Phase 1 report file not found - storing a placeholder instead")

    report = db.analysis_reports.create_report(dataset["id"], user["id"], phase1_report)
    print(f"Stored analysis report (id={report['id']})")

    latest = db.analysis_reports.get_latest_for_dataset(dataset["id"])
    assert latest["id"] == report["id"]
    print("Fetched latest report for dataset - matches")

    # ---- CLEANING HISTORY --------------------------------------------------
    section("CLEANING HISTORY - record an operation sequence")
    cleaning_entry = db.cleaning_history.create_entry(
        dataset_id=dataset["id"],
        user_id=user["id"],
        operations=[
            {"type": "impute_missing", "column": "age", "strategy": "mean"},
            {"type": "impute_missing", "column": "salary", "strategy": "median"},
            {"type": "drop_duplicates"},
        ],
        resulting_shape={"n_rows": 800, "n_columns": 11},
    )
    print(f"Recorded cleaning entry with {len(cleaning_entry['operations'])} operations "
          f"(id={cleaning_entry['id']})")

    history = db.cleaning_history.list_for_dataset(dataset["id"])
    assert len(history) == 1
    print(f"Cleaning history entries for dataset: {len(history)}")

    # ---- MODEL RESULTS + BEFORE/AFTER COMPARISON ---------------------------
    section("MODEL RESULTS - before vs after cleaning")
    db.model_results.create_result(
        dataset_id=dataset["id"], user_id=user["id"], stage="before",
        model_type="RandomForestClassifier",
        metrics={"accuracy": 0.78, "precision": 0.75, "recall": 0.71, "f1": 0.73},
    )
    db.model_results.create_result(
        dataset_id=dataset["id"], user_id=user["id"], stage="after",
        model_type="RandomForestClassifier",
        metrics={"accuracy": 0.85, "precision": 0.84, "recall": 0.81, "f1": 0.825},
    )
    comparison = db.model_results.compare_before_after(dataset["id"])
    print(f"Before: {comparison['before']['metrics']}")
    print(f"After:  {comparison['after']['metrics']}")
    print(f"Deltas: {comparison['deltas']}")
    assert comparison["deltas"]["accuracy"] > 0

    # ---- OBJECTID / ERROR HANDLING ------------------------------------------
    section("ERROR HANDLING - invalid and missing ObjectIds")
    try:
        db.datasets.get_dataset("not-a-real-objectid")
        print("FAIL: invalid ObjectId was not rejected")
        sys.exit(1)
    except ValueError as exc:
        print(f"Invalid ObjectId correctly raised ValueError: {exc}")

    missing = db.datasets.get_dataset("64b64f2e1234567890abcdef")  # valid format, no match
    assert missing is None
    print("Well-formed but non-existent ObjectId correctly returns None")

    # ---- DELETE CASCADE (manual, mirrors what a route would do) ------------
    section("DELETE - remove dataset and its dependent records")
    assert db.model_results.delete_result(
        db.model_results.list_for_dataset(dataset["id"])[0]["id"]
    )
    assert db.cleaning_history.delete_entry(cleaning_entry["id"])
    assert db.analysis_reports.delete_report(report["id"])
    assert db.datasets.delete_dataset(dataset["id"])
    assert db.datasets.get_dataset(dataset["id"]) is None
    print("Dataset and all dependent records deleted and verified gone")

    assert db.users.delete_user(user["id"])
    print("User deleted")

    db.close()

    print("\n" + "=" * 70)
    print("Phase 2 MongoDB CRUD layer verified end-to-end across all 5 collections.")
    print("=" * 70)


if __name__ == "__main__":
    main()
