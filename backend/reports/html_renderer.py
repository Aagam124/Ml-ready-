"""
html_renderer.py
-----------------
Renders a report bundle (see builder.py) to a single self-contained HTML
string - inline CSS only, no external requests, so it opens correctly
whether it's downloaded, emailed, or opened straight from disk.
"""

from html import escape
from typing import Any, Dict, Optional


def _fmt(value: Any, suffix: str = "") -> str:
    if value is None:
        return "—"
    return f"{value}{suffix}"


def _status_color(status: str) -> str:
    return {"good": "#3F7A5C", "warn": "#C98A2E", "critical": "#B3453F"}.get(status, "#5B6168")


def _score_status(category: Optional[str]) -> str:
    if category in ("Excellent", "Good"):
        return "good"
    if category == "Needs Improvement":
        return "warn"
    return "critical"


def _section(title: str, body: str) -> str:
    return f'<section class="section"><h2>{escape(title)}</h2>{body}</section>'


def _table(headers, rows) -> str:
    head = "".join(f"<th>{escape(str(h))}</th>" for h in headers)
    body_rows = "".join(
        "<tr>" + "".join(f"<td>{escape(str(c))}</td>" for c in row) + "</tr>" for row in rows
    )
    return f'<table><thead><tr>{head}</tr></thead><tbody>{body_rows}</tbody></table>'


def _render_dataset_info(dataset: Dict[str, Any]) -> str:
    rows = [
        ("Filename", dataset["filename"]),
        ("Status", dataset["status"]),
        ("Uploaded", dataset["uploaded_at"]),
        ("Rows", _fmt(dataset["n_rows"])),
        ("Columns", _fmt(dataset["n_columns"])),
    ]
    body = "".join(f'<div class="kv"><span>{escape(k)}</span><strong>{escape(str(v))}</strong></div>' for k, v in rows)
    return _section("Dataset information", f'<div class="kv-grid">{body}</div>')


def _render_readiness_score(score: Dict[str, Any]) -> str:
    status = _score_status(score.get("category"))
    color = _status_color(status)
    deductions = "".join(
        f'<div class="ded-row"><span>{escape(d["issue"])}</span><span style="color:{color}">{d["points"]}</span></div>'
        for d in score.get("deductions", [])
    )
    body = f"""
      <div class="score-hero" style="border-color:{color}">
        <div class="score-number" style="color:{color}">{score['score']}</div>
        <div>
          <div class="score-category" style="color:{color}">{escape(score['category'])}</div>
          <div class="muted">ML Readiness Score, out of 100</div>
        </div>
      </div>
      {f'<div class="ded-list">{deductions}</div>' if deductions else ''}
    """
    return _section("Readiness score", body)


def _render_quality(quality: Dict[str, Any]) -> str:
    missing = quality["missing_values"]
    missing_rows = [(col, f'{info["percentage"]}%') for col, info in missing["columns_with_missing"].items()]
    parts = [
        f'<p><strong>{missing["overall_percentage"]}%</strong> of all cells are missing across '
        f'{len(missing["columns_with_missing"])} column(s).</p>',
        _table(["Column", "Missing %"], missing_rows) if missing_rows else "<p class='muted'>No missing values.</p>",
        f'<p><strong>{quality["duplicates"]["count"]}</strong> duplicate rows '
        f'({quality["duplicates"]["percentage"]}%).</p>',
    ]
    if quality["constant_columns"]:
        parts.append(f'<p><strong>Constant columns:</strong> {escape(", ".join(quality["constant_columns"]))}</p>')
    if quality["high_cardinality_columns"]:
        parts.append(
            f'<p><strong>High-cardinality columns:</strong> '
            f'{escape(", ".join(quality["high_cardinality_columns"].keys()))}</p>'
        )
    return _section("Data quality", "".join(parts))


def _render_ml_risks(risks: Dict[str, Any]) -> str:
    parts = []
    ci = risks["class_imbalance"]
    if ci["applicable"]:
        ratio = ci["majority_minority_ratio"]
        parts.append(f'<p><strong>Class imbalance:</strong> {ratio}:1 majority-to-minority ratio.</p>')
    else:
        parts.append(f'<p><strong>Class imbalance:</strong> {escape(ci["reason"])}</p>')

    mc = risks["multicollinearity"]
    pairs = mc.get("high_correlation_pairs", [])
    if pairs:
        rows = [(p["feature_a"], p["feature_b"], p["correlation"]) for p in pairs]
        parts.append("<p><strong>Highly correlated feature pairs:</strong></p>")
        parts.append(_table(["Feature A", "Feature B", "Correlation"], rows))
    else:
        parts.append("<p><strong>Multicollinearity:</strong> no highly correlated pairs found.</p>")

    leakage = risks["potential_leakage"]["potential_leakage_columns"]
    if leakage:
        rows = [(f["column"], " ".join(f["reasons"])) for f in leakage]
        parts.append("<p><strong>Potential data leakage:</strong></p>")
        parts.append(_table(["Column", "Reason(s)"], rows))
        parts.append(f'<p class="muted">{escape(risks["potential_leakage"]["disclaimer"])}</p>')
    else:
        parts.append("<p><strong>Potential data leakage:</strong> no suspicious columns found.</p>")

    return _section("ML risk analysis", "".join(parts))


def _render_recommendations(recommendations) -> str:
    if not recommendations:
        return _section("Recommendations", "<p class='muted'>None - no issues detected.</p>")
    items = "".join(f"<li>{escape(r)}</li>" for r in recommendations)
    return _section("Recommendations", f"<ol class='rec-list'>{items}</ol>")


def _render_cleaning_history(entries) -> str:
    if not entries:
        return _section("Cleaning operations", "<p class='muted'>No cleaning has been applied yet.</p>")
    parts = []
    for i, entry in enumerate(entries, start=1):
        op_items = []
        for op in entry["operations"]:
            label = op.get("type", "")
            if op.get("column"):
                label += f" — {op['column']}"
            if "rows_affected" in op:
                label += f" ({op['rows_affected']} rows)"
            op_items.append(f"<li>{escape(label)}</li>")
        ops = "".join(op_items)
        shape = entry["resulting_shape"]
        parts.append(
            f'<p><strong>Run {i}</strong> — {escape(entry["created_at"])} → '
            f'{shape["n_rows"]} rows × {shape["n_columns"]} columns</p><ul>{ops}</ul>'
        )
    return _section("Cleaning operations", "".join(parts))


def _render_model_comparison(comparison: Optional[Dict[str, Any]]) -> str:
    if comparison is None:
        return _section("Model performance", "<p class='muted'>No model has been trained on this dataset yet.</p>")

    before, after, deltas = comparison["before"], comparison["after"], comparison.get("deltas") or {}
    rows = []
    for metric, before_val in before["metrics"].items():
        after_val = after["metrics"].get(metric) if after else None
        delta = deltas.get(metric)
        rows.append((
            metric, before_val,
            after_val if after_val is not None else "—",
            f"{'+' if delta and delta >= 0 else ''}{delta}" if delta is not None else "—",
        ))
    body = f'<p><strong>Model:</strong> {escape(before["model_type"])} ' \
           f'({escape(before["details"]["problem_type"])})</p>'
    body += _table(["Metric", "Before", "After", "Change"], rows)
    if not after:
        reason = comparison.get("after_skipped_reason") or "No cleaned version was available to compare against."
        body += f'<p class="muted">{escape(reason)}</p>'
    else:
        body += '<p class="muted">Cleaning does not always improve performance - changes are shown as measured.</p>'
    return _section("Model performance: before vs. after cleaning", body)


CSS = """
  body { font-family: -apple-system, 'Segoe UI', sans-serif; color: #1C2024; margin: 0; background: #EFEAE2; }
  .page { max-width: 860px; margin: 0 auto; padding: 40px 32px 80px; background: #FBFAF7; }
  h1 { font-size: 22px; margin-bottom: 4px; }
  h2 { font-size: 16px; border-bottom: 1px solid #DCD6C9; padding-bottom: 8px; margin: 0 0 14px; }
  .muted { color: #5B6168; font-size: 13px; }
  .section { margin-bottom: 32px; }
  .kv-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px 24px; }
  .kv { display: flex; justify-content: space-between; border-bottom: 1px solid #EEE; padding: 4px 0; font-size: 13px; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; margin: 8px 0; }
  th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid #EEE; }
  th { color: #5B6168; font-weight: 500; font-size: 11px; text-transform: uppercase; }
  .score-hero { display: flex; align-items: center; gap: 20px; border: 1px solid; border-radius: 4px; padding: 16px 20px; }
  .score-number { font-size: 40px; font-weight: 700; font-family: monospace; }
  .score-category { font-size: 16px; font-weight: 600; }
  .ded-list { margin-top: 12px; }
  .ded-row { display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #EEE; font-size: 13px; }
  .rec-list { padding-left: 20px; font-size: 13.5px; line-height: 1.6; }
  .header-meta { color: #5B6168; font-size: 12px; margin-bottom: 28px; }
"""


def render_html(bundle: Dict[str, Any]) -> str:
    dataset = bundle["dataset"]
    analysis = bundle["analysis"]

    parts = [f"<h1>MLReady Report — {escape(dataset['filename'])}</h1>",
              f'<div class="header-meta">Generated {escape(bundle["generated_at"])}</div>',
              _render_dataset_info(dataset)]

    if analysis:
        parts.append(_render_readiness_score(analysis["readiness_score"]))
        parts.append(_render_quality(analysis["quality"]))
        parts.append(_render_ml_risks(analysis["ml_risks"]))
        parts.append(_render_recommendations(analysis["recommendations"]))
    else:
        parts.append(_section("Analysis", "<p class='muted'>This dataset has not been analyzed yet.</p>"))

    parts.append(_render_cleaning_history(bundle["cleaning_history"]))
    parts.append(_render_model_comparison(bundle["model_comparison"]))

    body = "".join(parts)
    return f"<!doctype html><html><head><meta charset='utf-8'><title>MLReady Report</title><style>{CSS}</style></head><body><div class='page'>{body}</div></body></html>"
