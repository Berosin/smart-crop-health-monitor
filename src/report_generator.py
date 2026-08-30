"""PDF report export — farmer-shareable diagnostic reports.

Turns an on-screen result into a downloadable PDF someone can hand to a
neighbor, an agri-extension officer, or keep for their own records —
image, diagnosis, severity, recommendation, and (where available) the
yield-loss estimate, laid out as an actual document rather than a
screenshot.

Deliberately kept Streamlit-free and pure: every function here takes
plain dicts/values already computed elsewhere (pages/disease.py,
pages/field_scan.py, src/yield_loss.py, src/gradcam.py) and returns PDF
bytes. Callers own pulling data out of session_state; this module only
ever lays it out.

Two report types, one shared style:
- generate_disease_report_pdf()    — a single Disease Detection result
- generate_field_scan_report_pdf() — an aggregated Field Scan result
"""

from __future__ import annotations

from datetime import datetime
from io import BytesIO

import numpy as np
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, HRFlowable, KeepTogether,
)

from config import APP_CONFIG

# ---------------------------------------------------------------------------
# Shared palette (loosely matches the app's "field journal" earthy theme —
# utils/ui.py's CSS variables) and styles
# ---------------------------------------------------------------------------
COLOR_LEAF = colors.HexColor("#2F6D46")
COLOR_INK = colors.HexColor("#1C2E20")
COLOR_MUTED = colors.HexColor("#5B6353")
COLOR_FAINT = colors.HexColor("#93998A")
COLOR_TABLE_HEADER_BG = colors.HexColor("#EAEFE2")
COLOR_TABLE_BORDER = colors.HexColor("#D8DCCB")

SEVERITY_COLOR = {
    "None": colors.HexColor("#7FA687"),
    "Mild": colors.HexColor("#D6A34B"),
    "Moderate": colors.HexColor("#C97A3B"),
    "High": colors.HexColor("#B5564B"),
}

PAGE_MARGIN = 18 * mm
MAX_IMAGE_WIDTH_PX = 900  # downscale embedded images before writing to keep file size reasonable


def _styles() -> dict:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ReportTitle", parent=base["Title"], textColor=COLOR_INK,
            fontSize=18, spaceAfter=2,
        ),
        "subtitle": ParagraphStyle(
            "ReportSubtitle", parent=base["Normal"], textColor=COLOR_MUTED,
            fontSize=9.5, spaceAfter=10,
        ),
        "h2": ParagraphStyle(
            "ReportH2", parent=base["Heading2"], textColor=COLOR_LEAF,
            fontSize=13, spaceBefore=14, spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "ReportBody", parent=base["Normal"], textColor=COLOR_INK,
            fontSize=10, leading=14,
        ),
        "muted": ParagraphStyle(
            "ReportMuted", parent=base["Normal"], textColor=COLOR_MUTED,
            fontSize=8.5, leading=12,
        ),
        "disclaimer": ParagraphStyle(
            "ReportDisclaimer", parent=base["Normal"], textColor=COLOR_FAINT,
            fontSize=7.5, leading=10, spaceBefore=14,
        ),
        "image_caption": ParagraphStyle(
            "ImageCaption", parent=base["Normal"], textColor=COLOR_MUTED,
            fontSize=8, alignment=1, spaceBefore=3,
        ),
    }


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------
def _prep_image_bytes(image_bytes: bytes) -> BytesIO:
    """Decode arbitrary uploaded image bytes, downscale, re-encode as PNG.

    Normalizes whatever format the browser upload was (JPEG, PNG, possibly
    RGBA/CMYK) into a clean RGB PNG at a bounded resolution, so the PDF
    stays a reasonable size regardless of how large the original photo was.
    """
    img = PILImage.open(BytesIO(image_bytes)).convert("RGB")
    img.thumbnail((MAX_IMAGE_WIDTH_PX, MAX_IMAGE_WIDTH_PX))
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def _prep_image_array(arr: np.ndarray) -> BytesIO:
    """Same normalization as _prep_image_bytes, for an in-memory uint8 array
    (used for Grad-CAM's base image / overlay, which never touch disk).
    """
    img = PILImage.fromarray(np.clip(arr, 0, 255).astype("uint8")).convert("RGB")
    img.thumbnail((MAX_IMAGE_WIDTH_PX, MAX_IMAGE_WIDTH_PX))
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def _rl_image(buf: BytesIO, max_width_mm: float) -> RLImage:
    """A reportlab Image flowable sized to max_width_mm, aspect preserved."""
    pil_img = PILImage.open(buf)
    w_px, h_px = pil_img.size
    buf.seek(0)
    max_width = max_width_mm * mm
    scale = max_width / w_px
    return RLImage(buf, width=w_px * scale, height=h_px * scale)


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------
def _header(styles: dict, title: str, subtitle_bits: list[str]) -> list:
    generated = datetime.now().strftime("%d %b %Y, %H:%M")
    return [
        Paragraph(APP_CONFIG["title"], styles["subtitle"]),
        Paragraph(title, styles["title"]),
        Paragraph(" · ".join(subtitle_bits + [f"Generated {generated}"]), styles["subtitle"]),
        HRFlowable(width="100%", thickness=1.2, color=COLOR_LEAF, spaceAfter=10),
    ]


def _kv_table(rows: list[tuple[str, str]], col_widths=(45 * mm, 0)) -> Table:
    """A clean two-column label/value table (used for the result summary)."""
    data = [[Paragraph(f"<b>{k}</b>", _styles()["body"]), Paragraph(v, _styles()["body"])] for k, v in rows]
    widths = [col_widths[0], None]
    t = Table(data, colWidths=widths, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, COLOR_TABLE_BORDER),
    ]))
    return t


def _disclaimer(styles: dict) -> Paragraph:
    return Paragraph(
        f"Generated by {APP_CONFIG['title']}, an AI-assisted diagnostic tool. "
        "Disease predictions, severity, and yield-loss figures come from a trained "
        "model and published agricultural research ranges — they are planning "
        "information, not a guaranteed outcome or professional agronomic/financial "
        "advice. Actual results depend on variety, timing, weather, and management. "
        "Verify important decisions with a local agricultural extension officer.",
        styles["disclaimer"],
    )


def _yield_loss_block(styles: dict, est: dict) -> list:
    if not est:
        return []
    flow = [
        Paragraph("Estimated Yield Loss If Untreated", styles["h2"]),
        Paragraph(
            f"<b>{est['loss_pct_low']:.0f}\u2013{est['loss_pct_high']:.0f}%</b> of expected yield "
            f"on a {est['field_size_ha']:.2f} ha field "
            f"(expected {est['expected_yield']:.1f} t if healthy).",
            styles["body"],
        ),
        Paragraph(
            f"\u2248 {est['yield_lost_low']:.1f}\u2013{est['yield_lost_high']:.1f} t of yield at risk.",
            styles["body"],
        ),
    ]
    if est.get("revenue_lost_low") is not None:
        flow.append(Paragraph(
            f"\u2248 {est['revenue_lost_low']:,.0f}\u2013{est['revenue_lost_high']:,.0f} "
            "estimated revenue at risk (at the price you entered).",
            styles["body"],
        ))
    flow.append(Paragraph(
        "Based on published crop-disease research ranges — see the in-app "
        "disclaimer for sources. Not a guarantee.",
        styles["muted"],
    ))
    return flow


# ---------------------------------------------------------------------------
# Report 1 — single Disease Detection result
# ---------------------------------------------------------------------------
def generate_disease_report_pdf(pred: dict, yield_loss_estimate: dict | None = None) -> bytes:
    """Build a one-analysis diagnostic report PDF.

    Args:
        pred: the same result dict pages/disease.py already builds and
            renders on screen (disease, confidence, severity, recommendation,
            is_healthy, _crop, _image_bytes, and optionally
            gradcam_heatmap/gradcam_base_image).
        yield_loss_estimate: output of src.yield_loss.estimate_yield_loss(),
            or None to omit that section (e.g. the crop is Healthy, or the
            person never opened/used the calculator).

    Returns:
        The PDF file's raw bytes, ready for st.download_button(data=...).
    """
    styles = _styles()
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        topMargin=PAGE_MARGIN, bottomMargin=PAGE_MARGIN,
        leftMargin=PAGE_MARGIN, rightMargin=PAGE_MARGIN,
    )

    story: list = []
    story += _header(
        styles, "Crop Disease Diagnostic Report",
        [pred.get("_crop", "Unknown crop")],
    )

    # Result summary
    status_color = "#7FA687" if pred["is_healthy"] else "#B5564B"
    story.append(Paragraph(
        f"<font color='{status_color}'><b>"
        f"{'Healthy' if pred['is_healthy'] else pred['disease'].replace('_', ' ')}</b></font>",
        ParagraphStyle("Verdict", parent=styles["title"], fontSize=16, spaceAfter=8),
    ))
    story.append(_kv_table([
        ("Crop", pred.get("_crop", "\u2014")),
        ("Prediction", pred["disease"].replace("_", " ")),
        ("Confidence", f"{pred['confidence'] * 100:.1f}%"),
        ("Severity", pred["severity"]),
        ("Status", "Healthy" if pred["is_healthy"] else "Disease detected"),
    ]))
    story.append(Spacer(1, 10))

    # Image(s): original + Grad-CAM overlay side by side, if available
    image_row = []
    if pred.get("_image_bytes"):
        img_buf = _prep_image_bytes(pred["_image_bytes"])
        image_row.append([_rl_image(img_buf, max_width_mm=75), Paragraph("Analyzed leaf image", styles["image_caption"])])

    if pred.get("gradcam_heatmap") is not None and pred.get("gradcam_base_image") is not None:
        from src.gradcam import overlay_heatmap  # local import: keep reportlab-only callers free of the TF-adjacent gradcam module
        overlay = overlay_heatmap(pred["gradcam_heatmap"], pred["gradcam_base_image"], alpha=0.4)
        overlay_buf = _prep_image_array(overlay)
        image_row.append([_rl_image(overlay_buf, max_width_mm=75), Paragraph("Grad-CAM: what drove this prediction", styles["image_caption"])])

    if image_row:
        story.append(Paragraph("Image", styles["h2"]))
        col_data = [[img, cap] for img, cap in image_row]
        row_table = Table([[c[0] for c in col_data]], hAlign="LEFT")
        caption_table = Table([[c[1] for c in col_data]], hAlign="LEFT")
        story.append(row_table)
        story.append(caption_table)
        story.append(Spacer(1, 6))

    # Recommendation
    story.append(Paragraph("Recommendation", styles["h2"]))
    story.append(Paragraph(pred["recommendation"], styles["body"]))

    # Yield loss (optional)
    story += _yield_loss_block(styles, yield_loss_estimate)

    story.append(_disclaimer(styles))
    doc.build(story)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Report 2 — aggregated Field Scan result
# ---------------------------------------------------------------------------
def generate_field_scan_report_pdf(report: dict, yield_loss_estimate: dict | None = None) -> bytes:
    """Build a field-level report PDF from a Field Scan result.

    Args:
        report: the same aggregate dict pages/field_scan.py already builds
            and renders (crop, n_total, n_healthy, n_diseased, healthy_pct,
            dominant_disease, disease_counts, severity_counts,
            field_health_score, leaves).
        yield_loss_estimate: output of src.yield_loss.estimate_yield_loss()
            for the field's dominant disease, or None to omit that section.

    Per-leaf detail is summarized as a compact table (name, disease,
    confidence, severity) rather than embedding every thumbnail — with up
    to 30 photos in one scan, a table stays a readable, shareable page or
    two; embedding 30 images would not.
    """
    styles = _styles()
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        topMargin=PAGE_MARGIN, bottomMargin=PAGE_MARGIN,
        leftMargin=PAGE_MARGIN, rightMargin=PAGE_MARGIN,
    )

    story: list = []
    story += _header(
        styles, "Field Scan Report",
        [report.get("crop", "Unknown crop"), f"{report.get('n_total', 0)} leaves scanned"],
    )

    story.append(_kv_table([
        ("Crop", report.get("crop", "\u2014")),
        ("Photos scanned", str(report.get("n_total", 0))),
        ("Healthy", f"{report.get('healthy_pct', 0):.0f}% ({report.get('n_healthy', 0)}/{report.get('n_total', 0)})"),
        ("Dominant disease", (report.get("dominant_disease") or "None detected").replace("_", " ")),
        ("Field health score", f"{report.get('field_health_score', '\u2014')}/100"),
    ]))
    story.append(Spacer(1, 8))

    # Disease breakdown table
    disease_counts = report.get("disease_counts") or {}
    if disease_counts:
        story.append(Paragraph("Disease Breakdown", styles["h2"]))
        rows = [["Disease", "Leaves"]] + [
            [name.replace("_", " "), str(count)]
            for name, count in sorted(disease_counts.items(), key=lambda kv: -kv[1])
        ]
        story.append(_styled_table(rows))
        story.append(Spacer(1, 8))

    # Severity breakdown table
    severity_counts = report.get("severity_counts") or {}
    if severity_counts:
        story.append(Paragraph("Severity Breakdown", styles["h2"]))
        order = ["None", "Mild", "Moderate", "High"]
        present = [s for s in order if s in severity_counts] + [s for s in severity_counts if s not in order]
        rows = [["Severity", "Leaves"]] + [[s, str(severity_counts[s])] for s in present]
        story.append(_styled_table(rows))
        story.append(Spacer(1, 8))

    # Per-leaf table
    leaves = report.get("leaves") or []
    if leaves:
        story.append(Paragraph("Individual Leaves", styles["h2"]))
        rows = [["#", "Photo", "Prediction", "Confidence", "Severity"]]
        for i, leaf in enumerate(leaves, start=1):
            rows.append([
                str(i),
                leaf.get("name", "\u2014"),
                leaf.get("disease", "\u2014").replace("_", " "),
                f"{leaf.get('confidence', 0) * 100:.0f}%",
                leaf.get("severity", "\u2014"),
            ])
        story.append(_styled_table(rows, col_widths=(10 * mm, 55 * mm, 45 * mm, 25 * mm, 25 * mm)))
        story.append(Spacer(1, 8))

    # Yield loss (optional)
    story += _yield_loss_block(styles, yield_loss_estimate)

    story.append(_disclaimer(styles))
    doc.build(story)
    return buf.getvalue()


def _styled_table(rows: list[list[str]], col_widths=None) -> Table:
    """A bordered, header-shaded table matching the app's card styling."""
    body_style = _styles()["body"]
    data = [[Paragraph(f"<b>{c}</b>", body_style) for c in rows[0]]]
    for row in rows[1:]:
        data.append([Paragraph(str(c), body_style) for c in row])

    t = Table(data, colWidths=col_widths, hAlign="LEFT", repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), COLOR_TABLE_HEADER_BG),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, COLOR_LEAF),
        ("LINEBELOW", (0, 1), (-1, -1), 0.3, COLOR_TABLE_BORDER),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return t


# ---------------------------------------------------------------------------
# Self-test — builds both report types with synthetic data, no Streamlit
# session or trained model needed.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import numpy as np

    print("--- Building a synthetic single-leaf image ---")
    synthetic_leaf = (np.random.rand(224, 224, 3) * 255).astype("uint8")
    leaf_buf = BytesIO()
    PILImage.fromarray(synthetic_leaf).save(leaf_buf, format="JPEG")

    synthetic_heatmap = np.random.rand(7, 7).astype("float32")

    pred = {
        "_crop": "Tomato",
        "disease": "Late_Blight",
        "confidence": 0.94,
        "severity": "High",
        "is_healthy": False,
        "recommendation": "Apply a copper-based fungicide immediately and remove "
                           "infected foliage to prevent spread.",
        "_image_bytes": leaf_buf.getvalue(),
        "gradcam_heatmap": synthetic_heatmap,
        "gradcam_base_image": synthetic_leaf,
    }

    from src.yield_loss import estimate_yield_loss, REFERENCE_YIELD_T_PER_HA
    yl = estimate_yield_loss("Late_Blight", "High", 2.0, REFERENCE_YIELD_T_PER_HA["Tomato"], 200.0)

    pdf_bytes = generate_disease_report_pdf(pred, yield_loss_estimate=yl)
    print(f"Disease report: {len(pdf_bytes)} bytes")
    assert pdf_bytes[:4] == b"%PDF", "Output should be a valid PDF"
    with open("/tmp/test_disease_report.pdf", "wb") as f:
        f.write(pdf_bytes)
    print("Wrote /tmp/test_disease_report.pdf")

    print("\n--- Building a synthetic field scan report ---")
    field_report = {
        "crop": "Corn",
        "n_total": 12,
        "n_healthy": 5,
        "n_diseased": 7,
        "healthy_pct": 41.7,
        "dominant_disease": "Gray_Leaf_Spot",
        "field_health_score": 58,
        "disease_counts": {"Healthy": 5, "Gray_Leaf_Spot": 5, "Northern_Leaf_Blight": 2},
        "severity_counts": {"None": 5, "Moderate": 5, "High": 2},
        "leaves": [
            {"name": f"leaf_{i}.jpg", "disease": "Gray_Leaf_Spot" if i % 2 else "Healthy",
             "confidence": 0.7 + i * 0.01, "severity": "Moderate" if i % 2 else "None"}
            for i in range(12)
        ],
    }
    yl_field = estimate_yield_loss("Gray_Leaf_Spot", "Moderate", 5.0, REFERENCE_YIELD_T_PER_HA["Corn"], 0.0)
    pdf_bytes2 = generate_field_scan_report_pdf(field_report, yield_loss_estimate=yl_field)
    print(f"Field scan report: {len(pdf_bytes2)} bytes")
    assert pdf_bytes2[:4] == b"%PDF"
    with open("/tmp/test_field_report.pdf", "wb") as f:
        f.write(pdf_bytes2)
    print("Wrote /tmp/test_field_report.pdf")

    print("\nOK — both report types built successfully.")