"""Self-contained HTML report (no external assets, safe to email or host). Every dynamic string is escaped."""
from __future__ import annotations

import html
import json
import math
from typing import Any, Dict, List, Optional

BAND_COLOR = {"Excellent": "#1a7f4b", "Good": "#3a7d1f", "Fair": "#b36b00", "Poor": "#b3261e"}
SEV_COLOR = {"critical": "#b3261e", "high": "#c4521a", "medium": "#b36b00", "low": "#5b6b7a", "info": "#6a7fa0"}
E = html.escape


def _dial(value: float, band: str) -> str:
    r, c = 54, 2 * math.pi * 54
    off = c * (1 - max(0, min(100, value)) / 100)
    col = BAND_COLOR.get(band, "#444")
    return (f'<svg viewBox="0 0 140 140" width="160" height="160" role="img" aria-label="Readiness score {value:.0f} out of 100, {E(band)}">'
            f'<circle cx="70" cy="70" r="{r}" fill="none" stroke="#e6e8ec" stroke-width="12"/>'
            f'<circle cx="70" cy="70" r="{r}" fill="none" stroke="{col}" stroke-width="12" stroke-linecap="round" '
            f'stroke-dasharray="{c:.1f}" stroke-dashoffset="{off:.1f}" transform="rotate(-90 70 70)"/>'
            f'<text x="70" y="74" text-anchor="middle" font-size="34" font-weight="700" fill="#111">{value:.0f}</text>'
            f'<text x="70" y="94" text-anchor="middle" font-size="12" fill="#555">{E(band)}</text></svg>')


def _waterfall(deductions: List[Dict[str, Any]]) -> str:
    if not deductions:
        return "<p>No points were deducted.</p>"
    top = deductions[:12]
    mx = max(d["points"] for d in top) or 1
    rows = []
    for d in top:
        w = max(2, d["points"] / mx * 100)
        label = f'{d["code"].replace("_", " ")}' + (f' ({", ".join(d["columns"][:2])})' if d["columns"] else "")
        rows.append(f'<div class="wf"><span class="wl">{E(label)}</span><span class="wb"><i style="width:{w:.1f}%;background:{SEV_COLOR.get(d["severity"], "#555")}"></i></span>'
                    f'<span class="wp">-{d["points"]:.1f}</span></div>')
    rest = sum(d["points"] for d in deductions[12:])
    if rest:
        rows.append(f'<div class="wf"><span class="wl">{len(deductions) - 12} smaller findings</span><span class="wb"></span><span class="wp">-{rest:.1f}</span></div>')
    return "".join(rows)


def _finding(f: Dict[str, Any]) -> str:
    ev = json.dumps(f.get("evidence", {}), indent=2, default=str)
    cols = f", ".join(f.get("columns", [])[:6])
    fixes = "".join(f"<li>{E(x['label'])}</li>" for x in f.get("fixes", []))
    return (f'<details class="f"><summary><b class="sev" style="background:{SEV_COLOR.get(f["severity"], "#555")}">{E(f["severity"])}</b> '
            f'{E(f["plain_text"])} <span class="pts">-{f.get("score_impact", 0):.1f} pts</span></summary>'
            f'<p class="meta">Confidence: {E(f["confidence"])} &middot; Columns: {E(cols) or "dataset-wide"} &middot; Code: <code>{E(f["code"])}</code></p>'
            f'<p>{E(f["technical_text"])}</p>' + (f"<p><b>Suggested fixes</b></p><ul>{fixes}</ul>" if fixes else "") +
            f'<pre>{E(ev)}</pre></details>')


def _fmt(v: Optional[float]) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def _model_section(run: Dict[str, Any]) -> str:
    res = run.get("result")
    if not res:
        return ""
    c = res["comparison"]
    rows = "".join(
        f'<tr><td>{E(p["name"].replace("_", " "))}</td><td>{_fmt(p["before"])}</td><td>{_fmt(p["after"])}</td>'
        f'<td>{"" if p["delta"] is None else f"{p["delta"]:+.3f}"}</td><td>{E(p["verdict"].replace("_", " "))}</td></tr>' for p in c["per_model"])
    notes = "".join(f"<li>{E(n)}</li>" for n in c["notes"] + res.get("notes", []))
    cls = "warn" if c["verdict"] in ("leakage_removed", "worse") else "ok"
    return (f'<h2>Before and after</h2><p class="callout {cls}">{E(c["headline"])}</p>'
            f'<p>Primary metric: <b>{E(c["primary_metric"])}</b> (mean over {res["n_folds"]} identical cross-validation folds; '
            f'{res["common_test_rows"]} shared held-out test rows).</p>'
            f'<table><thead><tr><th>Model</th><th>Before</th><th>After</th><th>Change</th><th>Verdict</th></tr></thead><tbody>{rows}</tbody></table>'
            f'<ul class="notes">{notes}</ul>')


CSS = """
:root{--bg:#fff;--fg:#14171c;--mut:#5b6573;--line:#e3e6ea;--card:#f7f8fa}
@media(prefers-color-scheme:dark){:root{--bg:#111418;--fg:#eceff3;--mut:#a3adba;--line:#2a3038;--card:#181c22}}
body{font:16px/1.55 system-ui,-apple-system,Segoe UI,sans-serif;background:var(--bg);color:var(--fg);max-width:900px;margin:0 auto;padding:32px 20px}
h1{font-size:28px;margin:.2em 0}h2{margin-top:2em;font-size:20px}.hero{display:flex;gap:28px;align-items:center;flex-wrap:wrap}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px 20px}
.wf{display:grid;grid-template-columns:260px 1fr 56px;gap:10px;align-items:center;font-size:14px;margin:6px 0}
.wb{background:var(--line);height:10px;border-radius:6px;overflow:hidden;display:block}.wb i{display:block;height:100%}.wp{text-align:right;font-variant-numeric:tabular-nums}
details.f{border:1px solid var(--line);border-radius:12px;padding:10px 14px;margin:8px 0;background:var(--card)}
.sev{color:#fff;border-radius:6px;padding:1px 8px;font-size:12px;text-transform:uppercase}.pts{color:var(--mut);float:right}.meta{color:var(--mut);font-size:13px}
pre{overflow:auto;background:var(--bg);border:1px solid var(--line);padding:10px;border-radius:8px;font-size:12px}
table{border-collapse:collapse;width:100%}th,td{border-bottom:1px solid var(--line);padding:8px;text-align:left;font-size:14px}
.callout{padding:12px 16px;border-radius:12px}.callout.ok{background:rgba(26,127,75,.12)}.callout.warn{background:rgba(179,107,0,.15)}
.foot{color:var(--mut);font-size:13px;margin-top:3em}@media(max-width:640px){.wf{grid-template-columns:1fr 1fr 48px}}
"""


def render_report(analysis: Dict[str, Any], findings: List[Dict[str, Any]], dataset_name: str, model_run: Optional[Dict[str, Any]] = None) -> str:
    score = analysis["score"]
    summ = analysis["summary"]
    top = "".join(f'<li><b>{E(t["headline"])}</b> <span class="meta">+{t["points_gain"]:.1f} points, effort: {E(t["effort"])}</span></li>' for t in analysis.get("fix_first", []))
    notes = "".join(f"<p class='callout warn'>{E(n)}</p>" for n in summ.get("notes", []))
    counts = summ.get("counts", {})
    chips = " ".join(f'<span class="sev" style="background:{SEV_COLOR[s]}">{counts.get(s, 0)} {s}</span>' for s in ("critical", "high", "medium", "low") if counts.get(s))
    target = analysis.get("target_column")
    body = (f'<p class="meta">MLReady readiness report</p><h1>{E(dataset_name)}</h1>'
            f'<div class="hero card">{_dial(score["value"], score["band"])}<div><h2 style="margin:0">{E(summ["headline"])}</h2>'
            f'<p>Target: <b>{E(target) if target else "not selected"}</b>' + (f' ({E(analysis.get("task_type") or "")})' if target else "") +
            f'. Score {score["value"]:.1f} / 100{" (provisional)" if score.get("provisional") else ""}, scoring v{E(score.get("version", ""))}.</p><p>{chips}</p></div></div>{notes}'
            f'<h2>Fix these first</h2><ol>{top or "<li>Nothing urgent.</li>"}</ol>'
            f'<h2>Where the points went</h2><div class="card">{_waterfall(score.get("deductions", []))}'
            f'<p class="meta">Total deducted {score.get("total_deducted", 0):.1f}. Every point maps to one finding below; no problem is counted twice.</p></div>'
            f'<h2>All findings ({len(findings)})</h2>{"".join(_finding(f) for f in findings)}'
            + (_model_section(model_run) if model_run else "")
            + f'<p class="foot">Percentages state their denominator. Rows analysed: {analysis.get("sampling", {}).get("n_rows_analysed", "n/a")}'
              f'{" (sampled from " + str(analysis["sampling"]["n_rows_original"]) + ")" if analysis.get("sampling", {}).get("sampled") else ""}. '
              f'Generated by MLReady. Findings state evidence and confidence; treat "Possible" items as questions, not facts.</p>')
    return f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{E(dataset_name)}: MLReady report</title><style>{CSS}</style></head><body>{body}</body></html>'
