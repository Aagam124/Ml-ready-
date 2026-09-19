"""
pdf_renderer.py
-----------------
Renders a report bundle (see builder.py) to PDF bytes using reportlab.
Chosen over WeasyPrint/wkhtmltopdf specifically because it's pure
pip-installable with no system-level Cairo/Pango/Chromium dependency -
keeps "runs on a normal student laptop" true without extra setup steps.
"""

from io import BytesIO
from typing import Any, Dict, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

INK = colors.HexColor("#1C2024")
MUTED = colors.HexColor("#5B6168")
LINE = colors.HexColor("#DCD6C9")
GOOD = colors.HexColor("#3F7A5C")
WARN = colors.HexColor("#C98A2E")
CRITICAL = colors.HexColor("#B3453F")


def _score_color(category: Optional[str]):
    if category in ("Excellent", "Good"):
        return GOOD
    if category == "Needs Improvement":
        return WARN
    return CRITICAL


def _styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("title", parent=base["Title"], textColor=INK, fontSize=18, spaceAfter=2),
        "meta": ParagraphStyle("meta", parent=base["Normal"], textColor=MUTED, fontSize=9, spaceAfter=18),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], textColor=INK, fontSize=13, spaceBefore=16, spaceAfter=8,
                              borderWidth=0, borderColor=LINE),
        "score_number": ParagraphStyle("score_number", parent=base["Normal"], fontSize=32, leading=38, spaceAfter=2),
        "score_category": ParagraphStyle("score_category", parent=base["Normal"], fontSize=13, leading=16, spaceAfter=10),
        "body": ParagraphStyle("body", parent=base["Normal"], textColor=INK, fontSize=10, leading=14),
        "muted": ParagraphStyle("muted", parent=base["Normal"], textColor=MUTED, fontSize=9, leading=13),
    }


def _table_style(header_bg=colors.HexColor("#F2F0EA")):
    return TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), header_bg),
        ("TEXTCOLOR", (0, 0), (-1, 0), MUTED),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("LINEBELOW", (0, 0), (-1, -1), 0.5, LINE),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ])


def _data_table(headers, rows, col_widths=None, cell_style=None):
    """Wraps every body cell in a Paragraph so long text (leakage reasons,
    long column names) actually wraps within the column instead of
    overflowing off the page - plain strings in reportlab Table cells only
    wrap unreliably."""
    style = cell_style or ParagraphStyle("cell", fontName="Helvetica", fontSize=9, leading=12, textColor=INK)
    data = [headers] + [[Paragraph(str(c), style) for c in row] for row in rows]
    t = Table(data, colWidths=col_widths, hAlign="LEFT")
    t.setStyle(_table_style())
    return t


def render_pdf(bundle: Dict[str, Any]) -> bytes:
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=letter,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch,
        topMargin=0.75 * inch, bottomMargin=0.75 * inch,
        title="MLReady Report",
    )
    s = _styles()
    story = []

    dataset = bundle["dataset"]
    analysis = bundle["analysis"]

    story.append(Paragraph(f"MLReady Report — {dataset['filename']}", s["title"]))
    story.append(Paragraph(f"Generated {bundle['generated_at']}", s["meta"]))

    story.append(Paragraph("Dataset information", s["h2"]))
    story.append(_data_table(
        ["Field", "Value"],
        [
            ("Status", dataset["status"]),
            ("Uploaded", dataset["uploaded_at"]),
            ("Rows", dataset["n_rows"]),
            ("Columns", dataset["n_columns"]),
        ],
        col_widths=[150, 320],
    ))

    if analysis:
        score = analysis["readiness_score"]
        color = _score_color(score["category"])
        story.append(Paragraph("Readiness score", s["h2"]))
        story.append(Paragraph(f'<font color="{color.hexval()}"><b>{score["score"]}</b></font>', s["score_number"]))
        story.append(Paragraph(
            f'<font color="{color.hexval()}"><b>{score["category"]}</b></font> '
            f'<font color="{MUTED.hexval()}">— ML Readiness Score, out of 100</font>',
            s["score_category"],
        ))
        if score.get("deductions"):
            story.append(Spacer(1, 6))
            story.append(_data_table(
                ["Issue", "Points"],
                [(d["issue"], d["points"]) for d in score["deductions"]],
                col_widths=[380, 90],
            ))

        quality = analysis["quality"]
        story.append(Paragraph("Data quality", s["h2"]))
        missing = quality["missing_values"]
        story.append(Paragraph(
            f'{missing["overall_percentage"]}% of all cells missing across '
            f'{len(missing["columns_with_missing"])} column(s). '
            f'{quality["duplicates"]["count"]} duplicate rows ({quality["duplicates"]["percentage"]}%).',
            s["body"],
        ))
        if missing["columns_with_missing"]:
            story.append(Spacer(1, 6))
            story.append(_data_table(
                ["Column", "Missing %"],
                [(c, f'{info["percentage"]}%') for c, info in missing["columns_with_missing"].items()],
                col_widths=[300, 170],
            ))

        risks = analysis["ml_risks"]
        story.append(Paragraph("ML risk analysis", s["h2"]))
        ci = risks["class_imbalance"]
        ci_text = (f'Class imbalance: {ci["majority_minority_ratio"]}:1 majority-to-minority ratio.'
                   if ci["applicable"] else f'Class imbalance: {ci["reason"]}')
        story.append(Paragraph(ci_text, s["body"]))
        leakage = risks["potential_leakage"]["potential_leakage_columns"]
        if leakage:
            story.append(Spacer(1, 6))
            story.append(_data_table(
                ["Column", "Reason(s)"],
                [(f["column"], " ".join(f["reasons"])) for f in leakage],
                col_widths=[150, 320],
            ))
            story.append(Paragraph(risks["potential_leakage"]["disclaimer"], s["muted"]))
        else:
            story.append(Paragraph("No suspicious columns found for potential leakage.", s["body"]))

        story.append(Paragraph("Recommendations", s["h2"]))
        recs = analysis["recommendations"]
        if recs:
            for i, rec in enumerate(recs, start=1):
                story.append(Paragraph(f"{i}. {rec}", s["body"]))
                story.append(Spacer(1, 3))
        else:
            story.append(Paragraph("None — no issues detected.", s["muted"]))
    else:
        story.append(Paragraph("Analysis", s["h2"]))
        story.append(Paragraph("This dataset has not been analyzed yet.", s["muted"]))

    story.append(PageBreak())

    story.append(Paragraph("Cleaning operations", s["h2"]))
    cleaning = bundle["cleaning_history"]
    if cleaning:
        for i, entry in enumerate(cleaning, start=1):
            shape = entry["resulting_shape"]
            story.append(Paragraph(
                f'<b>Run {i}</b> — {entry["created_at"]} → {shape["n_rows"]} rows × {shape["n_columns"]} columns',
                s["body"],
            ))
            op_labels = []
            for op in entry["operations"]:
                label = op.get("type", "")
                if op.get("column"):
                    label += f" — {op['column']}"
                if "rows_affected" in op:
                    label += f" ({op['rows_affected']} rows)"
                op_labels.append(label)
            story.append(Paragraph("&nbsp;&nbsp;• " + "<br/>&nbsp;&nbsp;• ".join(op_labels), s["muted"]))
            story.append(Spacer(1, 8))
    else:
        story.append(Paragraph("No cleaning has been applied yet.", s["muted"]))

    story.append(Paragraph("Model performance: before vs. after cleaning", s["h2"]))
    comparison = bundle["model_comparison"]
    if comparison is None:
        story.append(Paragraph("No model has been trained on this dataset yet.", s["muted"]))
    else:
        before, after = comparison["before"], comparison["after"]
        deltas = comparison.get("deltas") or {}
        story.append(Paragraph(f'Model: {before["model_type"]} ({before["details"]["problem_type"]})', s["body"]))
        rows = []
        for metric, before_val in before["metrics"].items():
            after_val = after["metrics"].get(metric) if after else "—"
            delta = deltas.get(metric)
            delta_str = f"{'+' if delta is not None and delta >= 0 else ''}{delta}" if delta is not None else "—"
            rows.append((metric, before_val, after_val, delta_str))
        story.append(Spacer(1, 6))
        story.append(_data_table(["Metric", "Before", "After", "Change"], rows, col_widths=[140, 100, 100, 130]))
        if not after:
            story.append(Spacer(1, 6))
            story.append(Paragraph(
                comparison.get("after_skipped_reason") or "No cleaned version was available to compare against.",
                s["muted"],
            ))
        else:
            story.append(Spacer(1, 6))
            story.append(Paragraph(
                "Cleaning does not always improve performance - changes are shown as measured.", s["muted"],
            ))

    doc.build(story)
    return buf.getvalue()
