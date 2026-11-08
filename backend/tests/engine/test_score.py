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


def test_critical_leakage_cap_keeps_deductions_exact():
    fs = [
        Finding(code="target_leakage_a", category="leakage", severity="critical", confidence="high",
                columns=["a"], evidence={}, plain_text="p", technical_text="t", breadth=0.2),
        Finding(code="target_leakage_b", category="leakage", severity="critical", confidence="high",
                columns=["b"], evidence={}, plain_text="p", technical_text="t", breadth=0.2),
    ]
    s = score_findings(fs)
    assert s["value"] == 40.0
    assert round(100 - s["value"], 2) == round(sum(d["points"] for d in s["deductions"]), 2) == round(sum(f.score_impact for f in fs), 2)
    assert s["category_totals"]["leakage"] <= CATEGORY_CAP["leakage"] + 1e-9


def test_category_cap_respected():
    rng = random.Random(1)
    for _ in range(200):
        s = score_findings(rand_findings(rng))
        for cat, tot in s["category_totals"].items():
            assert tot <= CATEGORY_CAP[cat] + 1e-9


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
