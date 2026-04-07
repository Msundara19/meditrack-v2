"""PDF report generation for wound scans."""
import io
from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.platypus import Image as RLImage


def generate_scan_report(scan_data: dict, annotated_image_path: str = None) -> bytes:
    """
    Generate a PDF report for a wound scan.

    Parameters
    ----------
    scan_data : dict
        Scan record as returned by WoundScan.to_dict().
    annotated_image_path : str, optional
        Filesystem path to the annotated image to embed in the report.

    Returns
    -------
    bytes
        Raw PDF bytes ready to stream to the client.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )

    styles = getSampleStyleSheet()
    story = []

    # --- Title ---
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=20,
        textColor=colors.HexColor("#1a365d"),
        spaceAfter=4,
    )
    story.append(Paragraph("MediTrack Wound Analysis Report", title_style))

    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.HexColor("#718096"),
        spaceAfter=16,
    )
    story.append(
        Paragraph("For Educational Purposes Only — Not for Clinical Use", subtitle_style)
    )

    # --- Patient / scan info ---
    scan_date = str(scan_data.get("scan_date", ""))
    if "T" in scan_date:
        scan_date = scan_date.split("T")[0]

    risk = scan_data.get("analysis", {}).get("risk_level", "N/A").upper()
    scan_id_short = str(scan_data.get("id", "N/A"))[:8] + "…"

    info_data = [
        ["Patient ID", scan_data.get("patient_id", "N/A"), "Scan ID", scan_id_short],
        ["Scan Date", scan_date, "Risk Level", risk],
    ]
    info_table = Table(info_data, colWidths=[1.2 * inch, 2.3 * inch, 1.2 * inch, 2.3 * inch])
    info_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EBF8FF")),
                ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#EBF8FF")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
                ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
                ("FONTNAME", (3, 0), (3, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                ("PADDING", (0, 0), (-1, -1), 6),
                ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")]),
            ]
        )
    )
    story.append(info_table)
    story.append(Spacer(1, 0.2 * inch))

    # --- Wound classification ---
    metrics = scan_data.get("metrics", {})
    wound_type = metrics.get("wound_type", "unknown").replace("_", " ").title()

    section_style = ParagraphStyle(
        "Section",
        parent=styles["Heading2"],
        fontSize=12,
        textColor=colors.HexColor("#2D3748"),
        spaceAfter=6,
        spaceBefore=10,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#4A5568"),
    )

    story.append(Paragraph("Wound Classification", section_style))
    classified_by = metrics.get("classified_by", "heuristic").upper()
    ml_conf = metrics.get("ml_confidence")
    conf_text = f"<b>{wound_type}</b> — classified by {classified_by}"
    if ml_conf is not None:
        conf_text += f" (confidence: {ml_conf * 100:.1f}%)"
    story.append(Paragraph(conf_text, body_style))
    story.append(Spacer(1, 0.15 * inch))

    # --- Annotated image ---
    if annotated_image_path and Path(annotated_image_path).exists():
        try:
            img = RLImage(annotated_image_path, width=4 * inch, height=3.2 * inch)
            story.append(img)
            story.append(Spacer(1, 0.15 * inch))
        except Exception:
            pass

    # --- Metrics table ---
    story.append(Paragraph("Wound Metrics", section_style))

    measurement_type = metrics.get("measurement_type", "area")
    if measurement_type == "linear" and metrics.get("length_cm") is not None:
        size_label = "Length × Width"
        size_value = f"{metrics.get('length_cm', 0):.1f} cm × {metrics.get('width_cm', 0):.1f} cm"
    else:
        size_label = "Wound Area"
        size_value = f"{metrics.get('wound_area_cm2', 0):.2f} cm²"

    circ = metrics.get("circularity")
    ar = metrics.get("aspect_ratio")

    metrics_data = [
        ["Metric", "Value", "Metric", "Value"],
        ["Healing Score", f"{metrics.get('healing_score', 0):.0f} / 100", size_label, size_value],
        ["Redness Index", f"{metrics.get('redness_index', 0):.3f}", "Edge Sharpness", f"{metrics.get('edge_sharpness', 0):.3f}"],
        ["Circularity", f"{circ:.3f}" if circ is not None else "N/A",
         "Aspect Ratio", f"{ar:.2f}" if ar is not None else "N/A"],
    ]
    metrics_table = Table(metrics_data, colWidths=[1.5 * inch, 1.5 * inch, 1.5 * inch, 1.5 * inch])
    metrics_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2D3748")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (2, 1), (2, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                ("PADDING", (0, 0), (-1, -1), 6),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")]),
            ]
        )
    )
    story.append(metrics_table)
    story.append(Spacer(1, 0.2 * inch))

    # --- AI summary ---
    analysis = scan_data.get("analysis", {})
    story.append(Paragraph("AI Assessment", section_style))
    story.append(Paragraph(analysis.get("summary", "N/A"), body_style))
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("Care Recommendations", section_style))
    story.append(Paragraph(analysis.get("recommendations", "N/A"), body_style))
    story.append(Spacer(1, 0.3 * inch))

    # --- Disclaimer ---
    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontSize=8,
        textColor=colors.HexColor("#A0AEC0"),
    )
    story.append(
        Paragraph(
            "DISCLAIMER: This report is generated by an AI system for educational purposes only. "
            "It should NOT be used for medical diagnosis, treatment decisions, or clinical care. "
            "Always consult qualified healthcare professionals for medical advice.",
            disclaimer_style,
        )
    )
    story.append(Spacer(1, 0.1 * inch))

    footer_style = ParagraphStyle(
        "Footer",
        parent=styles["Normal"],
        fontSize=8,
        textColor=colors.HexColor("#A0AEC0"),
        alignment=TA_CENTER,
    )
    story.append(
        Paragraph(
            f"MediTrack v2.0 | Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} | Educational Project",
            footer_style,
        )
    )

    doc.build(story)
    return buffer.getvalue()
