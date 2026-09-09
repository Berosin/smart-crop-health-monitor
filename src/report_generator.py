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

import os
from datetime import datetime
from io import BytesIO

import numpy as np
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, HRFlowable, KeepTogether,
)

from config import APP_CONFIG
from src.i18n import tr_crop, tr_disease, tr_severity, tr_recommendation, tr_label

# ---------------------------------------------------------------------------
# Tamil font registration
#
# ReportLab's built-in fonts (Helvetica etc.) have zero Tamil glyphs — text
# would render as empty boxes, not an error, so this is easy to miss until
# someone actually opens a Tamil PDF. Noto Sans Tamil (SIL Open Font
# License, bundled at assets/fonts/) is registered once at import time and
# reused for every Tamil-language report. Registered as a full "family"
# (normal/bold/italic all pointing at the same single-weight file) so
# in-paragraph <b> tags don't raise a lookup error — the font just won't
# visually bolden, which is a cosmetic tradeoff, not a rendering failure.
# ---------------------------------------------------------------------------
TAMIL_FONT_NAME = "NotoSansTamil"
_TAMIL_FONT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "fonts", "NotoSansTamil-Regular.ttf",
)
_tamil_font_registered = False


def _ensure_tamil_font() -> bool:
    """Register the Tamil font if it hasn't been already. Returns whether
    it's available — callers fall back to the default (English-only) font
    if the asset is missing rather than crashing the whole report.
    """
    global _tamil_font_registered
    if _tamil_font_registered:
        return True
    if not os.path.exists(_TAMIL_FONT_PATH):
        return False
    try:
        pdfmetrics.registerFont(TTFont(TAMIL_FONT_NAME, _TAMIL_FONT_PATH))
        pdfmetrics.registerFontFamily(
            TAMIL_FONT_NAME, normal=TAMIL_FONT_NAME, bold=TAMIL_FONT_NAME,
            italic=TAMIL_FONT_NAME, boldItalic=TAMIL_FONT_NAME,
        )
        _tamil_font_registered = True
        return True
    except Exception:
        return False


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


def _styles(lang: str = "en") -> dict:
    base = getSampleStyleSheet()
    font_name = None
    if lang == "ta" and _ensure_tamil_font():
        font_name = TAMIL_FONT_NAME
    font_kwargs = {"fontName": font_name} if font_name else {}

    return {
        "title": ParagraphStyle(
            "ReportTitle", parent=base["Title"], textColor=COLOR_INK,
            fontSize=18, spaceAfter=2, **font_kwargs,
        ),
        "subtitle": ParagraphStyle(
            "ReportSubtitle", parent=base["Normal"], textColor=COLOR_MUTED,
            fontSize=9.5, spaceAfter=10, **font_kwargs,
        ),
        "h2": ParagraphStyle(
            "ReportH2", parent=base["Heading2"], textColor=COLOR_LEAF,
            fontSize=13, spaceBefore=14, spaceAfter=6, **font_kwargs,
        ),
        "body": ParagraphStyle(
            "ReportBody", parent=base["Normal"], textColor=COLOR_INK,
            fontSize=10, leading=15 if lang == "ta" else 14, **font_kwargs,
        ),
        "muted": ParagraphStyle(
            "ReportMuted", parent=base["Normal"], textColor=COLOR_MUTED,
            fontSize=8.5, leading=13 if lang == "ta" else 12, **font_kwargs,
        ),
        "disclaimer": ParagraphStyle(
            "ReportDisclaimer", parent=base["Normal"], textColor=COLOR_FAINT,
            fontSize=7.5, leading=11 if lang == "ta" else 10, spaceBefore=14, **font_kwargs,
        ),
        "image_caption": ParagraphStyle(
            "ImageCaption", parent=base["Normal"], textColor=COLOR_MUTED,
            fontSize=8, alignment=1, spaceBefore=3, **font_kwargs,
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
def _header(styles: dict, title: str, subtitle_bits: list[str], lang: str = "en") -> list:
    generated_label = tr_label("Generated", lang)
    generated = datetime.now().strftime("%d %b %Y, %H:%M")
    return [
        Paragraph(APP_CONFIG["title"], styles["subtitle"]),
        Paragraph(title, styles["title"]),
        Paragraph(" · ".join(subtitle_bits + [f"{generated_label} {generated}"]), styles["subtitle"]),
        HRFlowable(width="100%", thickness=1.2, color=COLOR_LEAF, spaceAfter=10),
    ]


def _kv_table(rows: list[tuple[str, str]], col_widths=(45 * mm, 0), lang: str = "en") -> Table:
    """A clean two-column label/value table (used for the result summary)."""
    body_style = _styles(lang)["body"]
    data = [[Paragraph(f"<b>{k}</b>", body_style), Paragraph(v, body_style)] for k, v in rows]
    widths = [col_widths[0], None]
    t = Table(data, colWidths=widths, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, COLOR_TABLE_BORDER),
    ]))
    return t


def _ood_warning_block(styles: dict, reason: str) -> list:
    """A prominent warning box — mirrors pages/disease.py's on-screen
    "is this even a leaf?" banner (see src/ood_detection.py) so the PDF
    someone shares carries the same caveat the screen did, not a
    falsely-confident-looking document.
    """
    box_style = ParagraphStyle(
        "OODWarning", parent=styles["body"], textColor=colors.HexColor("#7C3730"),
        fontSize=10, leading=13, backColor=colors.HexColor("#FBEFED"),
        borderColor=colors.HexColor("#B5564B"), borderWidth=0.6,
        borderPadding=8, spaceAfter=10,
    )
    return [Paragraph(
        f"<b>Uncertain match \u2014 this doesn't look like a confident leaf match.</b> {reason} "
        "Treat the result below as unreliable and consider a clearer, closer photo.",
        box_style,
    )]


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


def _yield_loss_block(styles: dict, est: dict, lang: str = "en") -> list:
    if not est:
        return []
    flow = [
        Paragraph(tr_label("Estimated Yield Loss If Untreated", lang), styles["h2"]),
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
def generate_disease_report_pdf(pred: dict, yield_loss_estimate: dict | None = None, lang: str = "en") -> bytes:
    """Build a one-analysis diagnostic report PDF.

    Args:
        pred: the same result dict pages/disease.py already builds and
            renders on screen (disease, confidence, severity, recommendation,
            is_healthy, _crop, _image_bytes, and optionally
            gradcam_heatmap/gradcam_base_image).
        yield_loss_estimate: output of src.yield_loss.estimate_yield_loss(),
            or None to omit that section (e.g. the crop is Healthy, or the
            person never opened/used the calculator).
        lang: "en" or "ta" — see src/i18n.py. Crop/disease/severity/
            recommendation and the report's own section headers translate;
            the yield-loss explanation and disclaimer stay English (see
            src/i18n.py's module docstring for the scope reasoning). Falls
            back to English silently if the Tamil font asset is missing.

    Returns:
        The PDF file's raw bytes, ready for st.download_button(data=...).
    """
    styles = _styles(lang)
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        topMargin=PAGE_MARGIN, bottomMargin=PAGE_MARGIN,
        leftMargin=PAGE_MARGIN, rightMargin=PAGE_MARGIN,
    )

    crop_label = tr_crop(pred.get("_crop", "Unknown crop"), lang)
    disease_label = tr_disease(pred["disease"], lang)
    severity_label = tr_severity(pred["severity"], lang)
    healthy_label = tr_label("Healthy", lang)
    status_label = tr_label("Disease detected", lang) if not pred["is_healthy"] else healthy_label

    story: list = []
    story += _header(
        styles, tr_label("Crop Disease Diagnostic Report", lang),
        [crop_label], lang=lang,
    )

    if pred.get("is_ood"):
        story += _ood_warning_block(styles, " ".join(pred.get("ood_reasons") or []))

    # Result summary
    status_color = "#7FA687" if pred["is_healthy"] else "#B5564B"
    story.append(Paragraph(
        f"<font color='{status_color}'><b>"
        f"{healthy_label if pred['is_healthy'] else disease_label}</b></font>",
        ParagraphStyle("Verdict", parent=styles["title"], fontSize=16, spaceAfter=8),
    ))
    story.append(_kv_table([
        (tr_label("Crop", lang), crop_label),
        (tr_label("Prediction", lang), disease_label),
        (tr_label("Confidence", lang), f"{pred['confidence'] * 100:.1f}%"),
        (tr_label("Severity", lang), severity_label),
        (tr_label("Status", lang), status_label),
    ], lang=lang))
    story.append(Spacer(1, 10))

    # Image(s): original + Grad-CAM overlay side by side, if available
    image_row = []
    if pred.get("_image_bytes"):
        img_buf = _prep_image_bytes(pred["_image_bytes"])
        image_row.append([_rl_image(img_buf, max_width_mm=75), Paragraph(tr_label("Analyzed leaf image", lang), styles["image_caption"])])

    if pred.get("gradcam_heatmap") is not None and pred.get("gradcam_base_image") is not None:
        from src.gradcam import overlay_heatmap  # local import: keep reportlab-only callers free of the TF-adjacent gradcam module
        overlay = overlay_heatmap(pred["gradcam_heatmap"], pred["gradcam_base_image"], alpha=0.4)
        overlay_buf = _prep_image_array(overlay)
        image_row.append([_rl_image(overlay_buf, max_width_mm=75), Paragraph(tr_label("Grad-CAM: what drove this prediction", lang), styles["image_caption"])])

    if image_row:
        story.append(Paragraph(tr_label("Image", lang), styles["h2"]))
        col_data = [[img, cap] for img, cap in image_row]
        row_table = Table([[c[0] for c in col_data]], hAlign="LEFT")
        caption_table = Table([[c[1] for c in col_data]], hAlign="LEFT")
        story.append(row_table)
        story.append(caption_table)
        story.append(Spacer(1, 6))

    # Recommendation
    story.append(Paragraph(tr_label("Recommendation", lang), styles["h2"]))
    recommendation_text = tr_recommendation(pred["disease"], lang, fallback=pred["recommendation"])
    story.append(Paragraph(recommendation_text, styles["body"]))

    # Yield loss (optional)
    story += _yield_loss_block(styles, yield_loss_estimate, lang=lang)

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
def generate_field_scan_report_pdf(report: dict, yield_loss_estimate: dict | None = None, lang: str = "en") -> bytes:
    """Build a field-level report PDF from a Field Scan result.

    Args:
        report: the same aggregate dict pages/field_scan.py already builds
            and renders (crop, n_total, n_healthy, n_diseased, healthy_pct,
            dominant_disease, disease_counts, severity_counts,
            field_health_score, leaves).
        yield_loss_estimate: output of src.yield_loss.estimate_yield_loss()
            for the field's dominant disease, or None to omit that section.
        lang: "en" or "ta" — see src/i18n.py / generate_disease_report_pdf's
            docstring for the exact translation scope.

    Per-leaf detail is summarized as a compact table (name, disease,
    confidence, severity) rather than embedding every thumbnail — with up
    to 30 photos in one scan, a table stays a readable, shareable page or
    two; embedding 30 images would not.
    """
    styles = _styles(lang)
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        topMargin=PAGE_MARGIN, bottomMargin=PAGE_MARGIN,
        leftMargin=PAGE_MARGIN, rightMargin=PAGE_MARGIN,
    )

    crop_label = tr_crop(report.get("crop", "Unknown crop"), lang)
    dominant_disease_label = tr_disease(report["dominant_disease"], lang) if report.get("dominant_disease") else tr_label("None detected", lang)

    story: list = []
    story += _header(
        styles, tr_label("Field Scan Report", lang),
        [crop_label, f"{report.get('n_total', 0)} {tr_label('leaves scanned', lang)}"],
        lang=lang,
    )

    if report.get("n_uncertain"):
        story += _ood_warning_block(
            styles,
            f"{report['n_uncertain']} of {report.get('n_total', 0)} photo(s) in this scan "
            "didn't look like confident leaf matches (marked \u201cuncertain\u201d in the table below).",
        )

    story.append(_kv_table([
        (tr_label("Crop", lang), crop_label),
        (tr_label("Photos scanned", lang), str(report.get("n_total", 0))),
        (tr_label("Healthy", lang), f"{report.get('healthy_pct', 0):.0f}% ({report.get('n_healthy', 0)}/{report.get('n_total', 0)})"),
        (tr_label("Dominant disease", lang), dominant_disease_label),
        (tr_label("Field health score", lang), f"{report.get('field_health_score', '\u2014')}/100"),
    ], lang=lang))
    story.append(Spacer(1, 8))

    # Disease breakdown table
    disease_counts = report.get("disease_counts") or {}
    if disease_counts:
        story.append(Paragraph(tr_label("Disease Breakdown", lang), styles["h2"]))
        rows = [[tr_label("Disease", lang), tr_label("Leaves", lang)]] + [
            [tr_disease(name, lang), str(count)]
            for name, count in sorted(disease_counts.items(), key=lambda kv: -kv[1])
        ]
        story.append(_styled_table(rows, lang=lang))
        story.append(Spacer(1, 8))

    # Severity breakdown table
    severity_counts = report.get("severity_counts") or {}
    if severity_counts:
        story.append(Paragraph(tr_label("Severity Breakdown", lang), styles["h2"]))
        order = ["None", "Mild", "Moderate", "High"]
        present = [s for s in order if s in severity_counts] + [s for s in severity_counts if s not in order]
        rows = [[tr_label("Severity", lang), tr_label("Leaves", lang)]] + [[tr_severity(s, lang), str(severity_counts[s])] for s in present]
        story.append(_styled_table(rows, lang=lang))
        story.append(Spacer(1, 8))

    # Per-leaf table
    leaves = report.get("leaves") or []
    if leaves:
        story.append(Paragraph(tr_label("Individual Leaves", lang), styles["h2"]))
        rows = [["#", tr_label("Photo", lang), tr_label("Prediction", lang), tr_label("Confidence", lang), tr_label("Severity", lang), tr_label("Note", lang)]]
        for i, leaf in enumerate(leaves, start=1):
            is_uncertain = leaf.get("ood_signal", {}).get("is_likely_ood")
            rows.append([
                str(i),
                leaf.get("name", "\u2014"),
                tr_disease(leaf.get("disease", ""), lang),
                f"{leaf.get('confidence', 0) * 100:.0f}%",
                tr_severity(leaf.get("severity", ""), lang),
                tr_label("Uncertain", lang) if is_uncertain else "",
            ])
        story.append(_styled_table(rows, col_widths=(8 * mm, 40 * mm, 35 * mm, 28 * mm, 22 * mm, 22 * mm), lang=lang))
        story.append(Spacer(1, 8))

    # Yield loss (optional)
    story += _yield_loss_block(styles, yield_loss_estimate, lang=lang)

    story.append(_disclaimer(styles))
    doc.build(story)
    return buf.getvalue()


def _styled_table(rows: list[list[str]], col_widths=None, lang: str = "en") -> Table:
    """A bordered, header-shaded table matching the app's card styling."""
    body_style = _styles(lang)["body"]
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

    print("\n--- Building the SAME disease report in Tamil ---")
    assert _ensure_tamil_font(), "Tamil font asset should be present at assets/fonts/NotoSansTamil-Regular.ttf"
    pdf_bytes_ta = generate_disease_report_pdf(pred, yield_loss_estimate=yl, lang="ta")
    assert pdf_bytes_ta[:4] == b"%PDF"
    with open("/tmp/test_disease_report_ta.pdf", "wb") as f:
        f.write(pdf_bytes_ta)
    print(f"Wrote /tmp/test_disease_report_ta.pdf ({len(pdf_bytes_ta)} bytes)")

    print("\n--- Building the SAME field scan report in Tamil ---")
    pdf_bytes2_ta = generate_field_scan_report_pdf(field_report, yield_loss_estimate=yl_field, lang="ta")
    assert pdf_bytes2_ta[:4] == b"%PDF"
    with open("/tmp/test_field_report_ta.pdf", "wb") as f:
        f.write(pdf_bytes2_ta)
    print(f"Wrote /tmp/test_field_report_ta.pdf ({len(pdf_bytes2_ta)} bytes)")

    print("\nOK — both report types built successfully in English and Tamil.")