"""HTML / PDF reports: content, escaping, honesty notes."""

from app.benchmark import make_planted
from app.cleaning.executor import plan_options, run_plan
from app.cleaning.plan import propose_plan
from app.engine.pipeline import run_analysis
from app.modeling.runner import run_before_after
from app.reports.html import render_report
from app.reports.pdf import render_pdf


def _inputs(with_model=True):
    df, target, _ = make_planted(["leakage", "negative_salary", "duplicates"], n=900)
    r = run_analysis(df, target)
    analysis = {k: r[k] for k in ("score", "summary", "fix_first", "sampling", "target_column", "task_type")}
    findings = r["findings"]
    run = None
    if with_model:
        plan = propose_plan(findings, r["profile"], target)
        after, _, _ = run_plan(df, plan["steps"], target, "binary")
        run = {"status": "done", "result": run_before_after(df, after, target, "binary", plan_options(plan["steps"]), r["leakage_columns"])}
    return analysis, findings, run


def test_html_report_is_self_contained_and_states_the_leak():
    a, f, run = _inputs()
    html = render_report(a, f, "Bench <script>alert(1)</script>", run)
    assert "<script" not in html.lower().replace("<script>alert", "") and "&lt;script&gt;" in html          # name escaped
    assert "http://" not in html and "https://" not in html                                                  # no external assets
    assert "Fix these first" in html and "Where the points went" in html and "post_outcome_score" in html
    assert "not real" in html or "inflated" in html
    assert f'{a["score"]["value"]:.0f}' in html


def test_pdf_report_is_a_valid_pdf_with_model_section():
    a, f, run = _inputs()
    pdf = render_pdf(a, f, "Bench", run)
    assert pdf[:5] == b"%PDF-" and len(pdf) > 3000 and pdf.rstrip().endswith(b"%%EOF")


def test_report_without_model_run_and_provisional_score():
    df, _, _ = make_planted(["duplicates"])
    r = run_analysis(df)
    a = {k: r[k] for k in ("score", "summary", "fix_first", "sampling", "target_column", "task_type")}
    html = render_report(a, r["findings"], "No target")
    assert "provisional" in html.lower() and "Select a target" in html
    assert render_pdf(a, r["findings"], "No target")[:5] == b"%PDF-"
