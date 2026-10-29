"""PDF report via ReportLab (pure Python, no browser needed)."""
from __future__ import annotations

import io
from typing import Any, Dict, List, Optional
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

SEV = {"critical": "#b3261e", "high": "#c4521a", "medium": "#b36b00", "low": "#5b6b7a", "info": "#6a7fa0"}
BAND = {"Excellent": "#1a7f4b", "Good": "#3a7d1f", "Fair": "#b36b00", "Poor": "#b3261e"}


def render_pdf(analysis: Dict[str, Any], findings: List[Dict[str, Any]], dataset_name: str, model_run: Optional[Dict[str, Any]] = None) -> bytes:
    ss = getSampleStyleSheet()
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=10, leading=14)
    small = ParagraphStyle("s", parent=body, fontSize=8.5, textColor=colors.HexColor("#5b6573"))
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], spaceBefore=14, spaceAfter=6)
    P = lambda t, st=body: Paragraph(escape(str(t)), st)
    score, summ = analysis["score"], analysis["summary"]
    out = []
    out += [P("MLReady readiness report", small), Paragraph(escape(dataset_name), ss["Title"])]
    big = ParagraphStyle("big", parent=ss["Title"], fontSize=40, textColor=colors.HexColor(BAND.get(score["band"], "#333")), alignment=0)
    out += [Paragraph(f'{score["value"]:.0f} <font size="14">/ 100 &middot; {escape(score["band"])}{" (provisional)" if score.get("provisional") else ""}</font>', big),
            P(summ["headline"]), Spacer(1, 4)]
    for n in summ.get("notes", []):
        out.append(P(n, small))
    out.append(Paragraph("Fix these first", h2))
    for i, t in enumerate(analysis.get("fix_first", []), 1):
        out.append(P(f'{i}. {t["headline"]}  (+{t["points_gain"]:.1f} points, effort: {t["effort"]})'))
    out.append(Paragraph("Where the points went", h2))
    rows = [["Finding", "Columns", "Severity", "Points"]] + [
        [d["code"].replace("_", " "), ", ".join(d["columns"][:3]) or "dataset-wide", d["severity"], f'-{d["points"]:.1f}'] for d in score.get("deductions", [])[:25]]
    t = Table([[Paragraph(escape(str(c)), small) for c in r] for r in rows], colWidths=[62 * mm, 55 * mm, 25 * mm, 18 * mm], repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef0f3")), ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#d5d9de")),
                           ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    out.append(t)
    out.append(Paragraph(f"All findings ({len(findings)})", h2))
    for f in findings:
        out.append(Paragraph(f'<font color="{SEV.get(f["severity"], "#333")}"><b>{escape(f["severity"].upper())}</b></font> {escape(f["plain_text"])}', body))
        out.append(P(f'Confidence {f["confidence"]} | columns: {", ".join(f["columns"][:5]) or "dataset-wide"} | -{f.get("score_impact", 0):.1f} pts | {f["technical_text"][:220]}', small))
        out.append(Spacer(1, 4))
    if model_run and model_run.get("result"):
        c = model_run["result"]["comparison"]
        out.append(Paragraph("Before and after", h2))
        out.append(P(c["headline"]))
        for n in c["notes"]:
            out.append(P(n, small))
        mrows = [["Model", "Before", "After", "Verdict"]] + [
            [p["name"].replace("_", " "), "n/a" if p["before"] is None else f'{p["before"]:.3f}', "n/a" if p["after"] is None else f'{p["after"]:.3f}', p["verdict"].replace("_", " ")]
            for p in c["per_model"]]
        mt = Table([[Paragraph(escape(str(x)), small) for x in r] for r in mrows], colWidths=[55 * mm, 30 * mm, 30 * mm, 45 * mm])
        mt.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef0f3")), ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#d5d9de"))]))
        out.append(mt)
        out.append(P(f'Primary metric: {c["primary_metric"]}; identical folds and test rows for every arm.', small))
    buf = io.BytesIO()
    SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=16 * mm,
                      title=f"{dataset_name} - MLReady report").build(out)
    return buf.getvalue()
