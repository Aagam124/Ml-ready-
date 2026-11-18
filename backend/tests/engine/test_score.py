"""Score invariants (property-style, seeded random scenarios)."""
import random

from app.engine.findings import CATEGORIES, CONFIDENCES, SEVERITIES, Finding
from app.engine.scoring import CATEGORY_CAP, score_findings
from app.benchmark import make_planted
from app.engine.pipeline import run_analysis


def rand_findings(rng: random.Random):
    out = []
    for i in range(rng.randint(0, 40)):
        out.append(Finding(code=f"c{i}", category=rng.choice(CATEGORIES), severity=rng.choice(SEVERITIES),
                           confidence=rng.choice(CONFIDENCES), columns=[f"col{i}"], evidence={}, plain_text="p", technical_text="t",
                           breadth=rng.random()))
    return out


def test_score_within_bounds_and_deductions_sum_exactly():
    rng = random.Random(0)
    for _ in range(500):
        fs = rand_findings(rng)
        s = score_findings(fs)
        assert 0 <= s["value"] <= 100
        assert round(100 - s["value"], 2) == round(sum(d["points"] for d in s["deductions"]), 2) == round(sum(f.score_impact for f in fs), 2)


def test_no_findings():
    s = score_findings([])
    assert s["value"] == 100.0
    assert s["total_deducted"] == 0.0
    assert s["deductions"] == []
    assert sum(s["category_totals"].values()) == 0.0


def test_one_critical_leakage_finding_caps_at_40():
    fs = [
        Finding(code="target_leakage", category="leakage", severity="critical", confidence="high",
                columns=["leak_col"], evidence={}, plain_text="critical leakage", technical_text="t",
                breadth=1.0),
    ]
    s = score_findings(fs)
    assert s["value"] == 40.0
    assert s["total_deducted"] == 60.0
    assert sum(d["points"] for d in s["deductions"]) == 60.0
    assert sum(f.score_impact for f in fs) == 60.0
    assert s["category_totals"]["leakage"] == 60.0


def test_multiple_critical_leakage_findings_split_cap_adjustment_only_on_leakage():
    fs = [
        Finding(code="target_leakage_a", category="leakage", severity="critical", confidence="high",
                columns=["a"], evidence={}, plain_text="critical leakage a", technical_text="t", breadth=1.0),
        Finding(code="target_leakage_b", category="leakage", severity="critical", confidence="high",
                columns=["b"], evidence={}, plain_text="critical leakage b", technical_text="t", breadth=0.7),
    ]
    s = score_findings(fs)
    assert s["value"] == 40.0
    assert sum(d["points"] for d in s["deductions"]) == 60.0
    assert sum(f.score_impact for f in fs) == 60.0
    assert all(f.score_impact > 0 for f in fs)
    assert sum(s["category_totals"].values()) == 60.0


def test_critical_leakage_keeps_unrelated_findings_unchanged():
    fs = [
        Finding(code="target_leakage", category="leakage", severity="critical", confidence="high",
                columns=["leak_col"], evidence={}, plain_text="critical leakage", technical_text="t",
                breadth=1.0),
        Finding(code="missing_values", category="quality", severity="medium", confidence="high",
                columns=["x"], evidence={}, plain_text="missing values", technical_text="t",
                breadth=0.2),
        Finding(code="outlier", category="outliers", severity="low", confidence="medium",
                columns=["y"], evidence={}, plain_text="outlier", technical_text="t",
                breadth=0.1),
    ]
    ordinary = score_findings([f for f in fs if f.code != "target_leakage"])
    capped = score_findings(fs)
    assert capped["value"] == 40.0
    assert capped["deductions"][1]["points"] == ordinary["deductions"][0]["points"]
    assert capped["deductions"][2]["points"] == ordinary["deductions"][1]["points"]
    assert sum(d["points"] for d in capped["deductions"]) == 60.0
    assert sum(f.score_impact for f in fs) == 60.0
    assert sum(capped["category_totals"].values()) == 60.0


def test_critical_leakage_with_tiny_penalty_findings_keeps_them_unchanged():
    fs = [
        Finding(code="minor", category="quality", severity="low", confidence="low",
                columns=["a"], evidence={}, plain_text="minor", technical_text="t", breadth=0.01),
        Finding(code="tiny2", category="quality", severity="low", confidence="low",
                columns=["b"], evidence={}, plain_text="tiny2", technical_text="t", breadth=0.01),
        Finding(code="target_leakage", category="leakage", severity="critical", confidence="high",
                columns=["leak_col"], evidence={}, plain_text="critical leakage", technical_text="t", breadth=1.0),
    ]
    s = score_findings(fs)
    assert sum(d["points"] for d in s["deductions"]) == 60.0
    assert s["value"] == 40.0
    assert max(d["points"] for d in s["deductions"] if d["code"] != "target_leakage") == 0.46
    assert sum(f.score_impact for f in fs) == 60.0


def test_category_totals_reconcile_with_final_total():
    fs = [
        Finding(code="target_leakage", category="leakage", severity="critical", confidence="high",
                columns=["leak_col"], evidence={}, plain_text="critical leakage", technical_text="t", breadth=1.0),
        Finding(code="missing_values", category="quality", severity="medium", confidence="high",
                columns=["x"], evidence={}, plain_text="missing values", technical_text="t", breadth=0.2),
        Finding(code="future_dates", category="quality", severity="low", confidence="medium",
                columns=["date_col"], evidence={}, plain_text="future_dates", technical_text="t", breadth=0.1),
    ]
    s = score_findings(fs)
    assert round(sum(s["category_totals"].values()), 2) == round(s["total_deducted"], 2) == round(100 - s["value"], 2)


def test_no_cap_when_leakage_is_not_critical():
    fs = [
        Finding(code="target_leakage", category="leakage", severity="high", confidence="high",
                columns=["leak_col"], evidence={}, plain_text="leakage", technical_text="t", breadth=1.0),
        Finding(code="missing_values", category="quality", severity="medium", confidence="high",
                columns=["x"], evidence={}, plain_text="missing values", technical_text="t", breadth=0.2),
    ]
    s = score_findings(fs)
    assert s["value"] > 40.0
    assert s["total_deducted"] < 60.0


def test_no_cap_for_unrelated_critical_findings():
    fs = [
        Finding(code="missing_values", category="quality", severity="critical", confidence="high",
                columns=["x"], evidence={}, plain_text="missing values", technical_text="t", breadth=0.2),
        Finding(code="future_dates", category="quality", severity="critical", confidence="medium",
                columns=["date_col"], evidence={}, plain_text="future dates", technical_text="t", breadth=0.1),
    ]
    s = score_findings(fs)
    assert s["value"] > 40.0
    assert s["total_deducted"] < 60.0


def test_critical_leakage_cap_reconciles_score_and_deductions():
    fs = [
        Finding(code="target_leakage", category="leakage", severity="critical", confidence="high",
                columns=["leak_col"], evidence={}, plain_text="critical leakage", technical_text="t",
                breadth=1.0),
        Finding(code="missing_values", category="quality", severity="medium", confidence="high",
                columns=["x"], evidence={}, plain_text="missing values", technical_text="t",
                breadth=0.2),
        Finding(code="outlier", category="outliers", severity="low", confidence="medium",
                columns=["y"], evidence={}, plain_text="outlier", technical_text="t",
                breadth=0.1),
    ]
    s = score_findings(fs)
    assert round(100 - s["value"], 2) == round(s["total_deducted"], 2) == round(sum(d["points"] for d in s["deductions"]), 2) == round(sum(f.score_impact for f in fs), 2)
    assert round(sum(s["category_totals"].values()), 2) == round(s["total_deducted"], 2)


def test_category_cap_respected():
    rng = random.Random(1)
    for _ in range(200):
        fs = rand_findings(rng)
        s = score_findings(fs)
        critical_leakage = any(f.category == "leakage" and f.severity == "critical" for f in fs)
        for cat, tot in s["category_totals"].items():
            expected_cap = 60.0 if critical_leakage and cat == "leakage" else CATEGORY_CAP[cat]
            assert tot <= expected_cap + 1e-9


def test_fixing_a_finding_never_lowers_the_score():
    rng = random.Random(2)
    for _ in range(200):
        fs = rand_findings(rng)
        if not fs:
            continue
        before = score_findings(fs)["value"]
        rest = fs[:]
        rest.pop(rng.randrange(len(rest)))
        assert score_findings(rest)["value"] >= before


def test_rerun_is_identical():
    df, target, _ = make_planted(["leakage", "duplicates", "negative_salary", "imbalance"])
    a, b = run_analysis(df, target), run_analysis(df.copy(), target)
    strip = lambda r: ({k: v for k, v in r.items() if k not in ("duration_ms", "check_timings_ms")})
    assert strip(a) == strip(b)


def test_every_point_maps_to_a_named_finding_once():
    df, target, _ = make_planted(["leakage", "duplicates", "negative_salary", "missingness"])
    r = run_analysis(df, target)
    keys = [(f["code"], tuple(f["columns"])) for f in r["findings"]]
    assert len(keys) == len(set(keys))
    assert round(r["score"]["total_deducted"], 2) == round(sum(f["score_impact"] for f in r["findings"]), 2)
    assert {(d["code"], tuple(d["columns"])) for d in r["score"]["deductions"]} <= set(keys)


def test_fixing_planted_problem_raises_score():
    df, target, _ = make_planted(["leakage", "duplicates"])
    before = run_analysis(df, target)["score"]["value"]
    fixed = df.drop(columns=["post_outcome_score"]).drop_duplicates()
    assert run_analysis(fixed, target)["score"]["value"] > before


def test_no_target_makes_score_provisional_and_skips_target_checks():
    df, target, _ = make_planted(["leakage", "imbalance"])
    r = run_analysis(df)
    assert r["score"]["provisional"] and "target_leakage" in r["checks_skipped"]
    assert any("Select a target" in n for n in r["summary"]["notes"])
