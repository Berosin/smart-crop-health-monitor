"""Field Scan — batch disease detection across many leaf photos at once.

Reframes the single-leaf Disease Detection workflow into a field-monitoring
tool: upload a batch of leaf photos (simulating a walk through the field —
10 to 20 or more shots) and get back one aggregated **field health
report** — % healthy vs diseased, dominant disease, severity breakdown, and
a field-level health score — instead of one verdict per photo.

This page is intentionally a thin aggregation layer, not a new model or a
new pipeline:
- Model loading   -> pages.disease.load_model()          (same cache)
- Preprocessing   -> pages.disease.preprocess_image()      (same pipeline)
- Per-leaf scoring -> pages.disease's SEVERITY_MAP + src.health_engine's
                      compute_disease_risk_score() (same 0-100 scale the
                      Crop Health Analysis page already uses, so a "72" here
                      means the same thing it means everywhere else in the
                      app)

One deliberate efficiency choice: rather than looping predict_disease()
once per photo (N separate forward passes), every successfully preprocessed
leaf is stacked into a single batch and passed through model.predict() once
— a real speed win on CPU for a 10-20 image batch, and the reason this page
calls into model.predict() directly instead of pages.disease.predict_disease().

Grad-CAM explainability (see src/gradcam.py) is intentionally NOT run per
leaf here — computing a gradient pass for every one of 20 images on every
scan would slow the batch workflow down for a feature this page's report
doesn't need; that stays a single-leaf-focused tool on the Disease
Detection page.
"""

from __future__ import annotations

import json
import time
import uuid
from collections import Counter

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from config import CONFIDENCE_THRESHOLD, DEFAULT_DISEASE_CROP, get_trained_crops
from pages.disease import load_model, preprocess_image, SEVERITY_MAP, CLASS_COLORS, render_yield_loss_estimator
from src.db import insert_field_scan
from src.errors import logger, safe_action
from src.health_engine import compute_disease_risk_score, classify_health_status
from src.image_preprocessing import ImageValidationError
from src.yield_loss import get_yield_loss_range, estimate_yield_loss, REFERENCE_YIELD_T_PER_HA, HECTARES_PER_ACRE
from utils.ui import (
    page_header, callout, card, footer, metric_tile, health_score_card, pretty_name, CHART_THEME,
)
from utils.icons import icon_html

# Sane ceiling so an accidental huge upload can't hang the app or blow up
# memory — this is a "field walk" batch tool, not a bulk-import tool.
MAX_IMAGES = 30

SEVERITY_COLORS = {
    "None": "#7FA687", "Mild": "#D6A34B", "Moderate": "#C97A3B", "High": "#B5564B",
}
SEVERITY_ORDER = ["None", "Mild", "Moderate", "High"]


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
def render() -> None:
    page_header(
        "field_scan",
        "Field Scan",
        "Upload a batch of leaf photos from a field walk and get one aggregated health report.",
    )

    trained_crops = get_trained_crops()
    if not trained_crops:
        callout(
            f"{icon_html('warning', size=18)}<b>No trained model found.</b> "
            "Train a disease model first — see the Disease Detection page for instructions."
        )
        footer()
        return

    default_index = trained_crops.index(DEFAULT_DISEASE_CROP) if DEFAULT_DISEASE_CROP in trained_crops else 0
    crop = st.selectbox("Crop", trained_crops, index=default_index)

    # Same cached loader Disease Detection uses — switching crops there or
    # here reuses the same in-memory model, no duplicate loading.
    model, class_names = load_model(crop)
    if model is None:
        callout(
            f"{icon_html('warning', size=18)}<b>Model unavailable.</b> "
            f"{crop}'s model file couldn't be loaded even though it's listed as trained — "
            "check the server logs for details."
        )
        footer()
        return

    if st.session_state.get("_field_crop") != crop:
        st.session_state["_field_crop"] = crop
        st.session_state.pop("_field_report", None)
        st.session_state.pop("_field_saved_token", None)
        st.session_state.pop("_field_saved_id", None)

    st.markdown("#### 1 · Upload leaf photos")
    uploaded_files = st.file_uploader(
        f"Leaf images (JPG / PNG) — up to {MAX_IMAGES} at once",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        if len(uploaded_files) > MAX_IMAGES:
            callout(
                f"{icon_html('warning', size=18)}"
                f"{len(uploaded_files)} photos uploaded — only the first {MAX_IMAGES} "
                "will be scanned. Split larger batches into multiple scans."
            )
            uploaded_files = uploaded_files[:MAX_IMAGES]
        st.caption(f"{len(uploaded_files)} photo(s) ready to scan.")
    else:
        st.info("Drop 10-20+ leaf photos here — one field walk, one report.")

    with st.expander("Advanced options"):
        threshold = st.slider(
            "Confidence threshold",
            0.0, 1.0, float(CONFIDENCE_THRESHOLD), 0.05,
            help="Per-leaf predictions below this confidence are flagged as uncertain.",
        )
        st.markdown("**Preprocessing** (applied to every photo in the batch)")
        denoise = st.checkbox("Noise reduction", value=False)
        remove_background = st.checkbox("Background handling", value=False)

    run = st.button(
        "Run Field Scan", type="primary", use_container_width=True,
        disabled=not uploaded_files,
    )

    st.markdown("#### 2 · Field health report")

    report = st.session_state.get("_field_report")

    if run and uploaded_files:
        try:
            report = _run_field_scan(
                model, class_names, uploaded_files, crop, threshold, denoise, remove_background,
            )
            st.session_state["_field_report"] = report
            st.session_state.pop("_field_saved_token", None)
            st.session_state.pop("_field_saved_id", None)
        except Exception:
            logger.exception("Unexpected error during field scan")
            st.error(
                "The field scan failed unexpectedly. Please try again. "
                "If the problem continues, contact the app maintainer."
            )
            report = None

    if report is None:
        card(
            "Awaiting scan",
            "Upload several leaf photos and click **Run Field Scan** to see the "
            "aggregated field health report — % healthy vs diseased, dominant "
            "disease, severity breakdown, and a field health score.",
        )
    else:
        _render_report(report)

    footer()


# ---------------------------------------------------------------------------
# Batch inference + aggregation
# ---------------------------------------------------------------------------
def _run_field_scan(model, class_names, uploaded_files, crop, threshold, denoise, remove_background) -> dict:
    """Preprocess every photo, run one batched prediction, aggregate results.

    A photo that fails preprocessing (corrupt file, unreadable format, ...)
    is skipped and recorded under "failures" rather than aborting the whole
    scan — one bad photo in a 20-photo field walk shouldn't block the other 19.
    """
    batches: list[np.ndarray] = []
    thumbs_and_names: list[dict] = []
    failures: list[tuple[str, str]] = []

    with st.spinner(f"Preprocessing {len(uploaded_files)} photo(s)…"):
        for f in uploaded_files:
            try:
                batch = preprocess_image(f, denoise=denoise, remove_background=remove_background)
                batches.append(batch)
                thumbs_and_names.append({
                    "name": f.name,
                    "thumb": np.clip(batch[0], 0, 255).astype(np.uint8),
                })
            except (ImageValidationError, ValueError) as e:
                failures.append((f.name, str(e)))
            except Exception:
                logger.exception("Unexpected preprocessing error for %s in field scan", f.name)
                failures.append((f.name, "Unexpected preprocessing error."))

    leaves: list[dict] = []
    if batches:
        # One batched forward pass over all N successfully-preprocessed
        # leaves, instead of N separate model.predict() calls.
        stacked = np.concatenate(batches, axis=0)
        with st.spinner(f"Running disease detection on {len(batches)} photo(s)…"):
            preds = model.predict(stacked, verbose=0)

        for meta, row in zip(thumbs_and_names, preds):
            pred_idx = int(np.argmax(row))
            confidence = float(row[pred_idx])
            disease = class_names[pred_idx]
            severity = SEVERITY_MAP.get(disease, "Unknown")
            is_healthy = (disease == "Healthy")
            # Same 0-100 disease scoring src.health_engine already uses for
            # the Crop Health Analysis page, so the field score below means
            # the same thing a single-leaf health score means elsewhere.
            score, _, _ = compute_disease_risk_score(disease, confidence, severity)
            leaves.append({
                "name": meta["name"],
                "thumb": meta["thumb"],
                "disease": disease,
                "confidence": confidence,
                "severity": severity,
                "is_healthy": is_healthy,
                "low_confidence": confidence < threshold and not is_healthy,
                "score": score,
            })

    return _aggregate(leaves, failures, crop)


def _aggregate(leaves: list[dict], failures: list[tuple[str, str]], crop: str) -> dict:
    """Roll individual per-leaf predictions up into a field-level report."""
    n = len(leaves)
    if n == 0:
        return {
            "crop": crop, "leaves": [], "failures": failures, "n_total": 0,
            "n_healthy": 0, "n_diseased": 0, "healthy_pct": 0.0,
            "dominant_disease": None, "disease_counts": {}, "severity_counts": {},
            "field_health_score": None, "field_status": None,
        }

    n_healthy = sum(1 for l in leaves if l["is_healthy"])
    n_diseased = n - n_healthy
    healthy_pct = round(100 * n_healthy / n, 1)

    disease_counts = Counter(l["disease"] for l in leaves)
    diseased_counts = Counter(l["disease"] for l in leaves if not l["is_healthy"])
    dominant_disease = diseased_counts.most_common(1)[0][0] if diseased_counts else None

    severity_counts = Counter(l["severity"] for l in leaves)

    field_health_score = int(round(sum(l["score"] for l in leaves) / n))
    field_status = classify_health_status(field_health_score)

    return {
        "crop": crop,
        "leaves": leaves,
        "failures": failures,
        "n_total": n,
        "n_healthy": n_healthy,
        "n_diseased": n_diseased,
        "healthy_pct": healthy_pct,
        "dominant_disease": dominant_disease,
        "disease_counts": dict(disease_counts),
        "severity_counts": dict(severity_counts),
        "field_health_score": field_health_score,
        "field_status": field_status,
        "_scan_token": uuid.uuid4().hex,
    }


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------
def _render_report(report: dict) -> None:
    if report["n_total"] == 0:
        callout(
            f"{icon_html('warning', size=18)}None of the uploaded photos could "
            "be analyzed. See the issues below."
        )
        _render_failures(report["failures"])
        return

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_tile("Photos scanned", str(report["n_total"]))
    with c2:
        metric_tile("Healthy", f"{report['healthy_pct']:.0f}%",
                     f"{report['n_healthy']} / {report['n_total']} leaves")
    with c3:
        metric_tile("Dominant disease", pretty_name(report["dominant_disease"]) or "None detected")
    with c4:
        metric_tile("Diseased leaves", str(report["n_diseased"]))

    st.write("")
    health_score_card(report["field_health_score"], label="Field health score")

    # Disease breakdown
    st.markdown("#### Disease breakdown across the field")
    dc = report["disease_counts"]
    names = sorted(dc, key=lambda k: dc[k], reverse=True)
    fig = go.Figure(go.Bar(
        orientation="h",
        x=[dc[n] for n in names],
        y=[pretty_name(n) for n in names],
        text=[str(dc[n]) for n in names],
        textposition="outside",
        marker=dict(color=[CLASS_COLORS.get(n, "#7C8571") for n in names]),
    ))
    fig.update_layout(
        **CHART_THEME,
        margin=dict(t=10, b=10),
        xaxis_title="Leaves",
        height=max(200, len(names) * 42),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Severity breakdown
    st.markdown("#### Severity breakdown")
    sc = report["severity_counts"]
    present = [s for s in SEVERITY_ORDER if s in sc] + [s for s in sc if s not in SEVERITY_ORDER]
    sev_cols = st.columns(len(present))
    for col, s in zip(sev_cols, present):
        with col:
            st.markdown(
                f"""
                <div class="metric-tile" style="border-left:5px solid {SEVERITY_COLORS.get(s, '#93998A')}">
                  <div class="label">{s}</div>
                  <div class="value">{sc[s]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Yield loss / economic impact estimate — for the field's dominant
    # disease, at a deliberately conservative representative severity (if
    # any scanned leaf came back High severity, use High rather than
    # averaging it away — matches this app's general "don't understate
    # risk" stance, e.g. src/outbreak_detection.py's risk classification).
    rep_severity = None
    if report["dominant_disease"]:
        if sc.get("High"):
            rep_severity = "High"
        elif sc.get("Moderate"):
            rep_severity = "Moderate"
        else:
            rep_severity = "None"
        render_yield_loss_estimator(
            report["crop"], report["dominant_disease"], rep_severity, key_prefix="fieldscan",
        )

    # Per-leaf thumbnail grid
    st.markdown("#### Individual leaves")
    leaves = report["leaves"]
    n_cols = 4
    for row_start in range(0, len(leaves), n_cols):
        row = leaves[row_start:row_start + n_cols]
        cols = st.columns(n_cols)
        for col, leaf in zip(cols, row):
            with col:
                border = "#7FA687" if leaf["is_healthy"] else SEVERITY_COLORS.get(leaf["severity"], "#B5564B")
                st.image(leaf["thumb"], use_container_width=True)
                low_conf_note = " · low confidence" if leaf["low_confidence"] else ""
                st.markdown(
                    f"""
                    <div style="border-left:4px solid {border};padding:.15rem .5rem;
                                font-size:.78rem;color:#4E5646;margin:-.4rem 0 .8rem">
                      <b>{pretty_name(leaf['disease'])}</b><br/>{leaf['confidence']*100:.0f}% confidence{low_conf_note}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    _render_failures(report["failures"])

    # PDF report export
    st.markdown("---")
    _render_pdf_download(report, rep_severity)

    st.markdown("---")
    _render_save_section(report)

    if st.button("New Scan", use_container_width=True):
        st.session_state["_field_report"] = None
        st.session_state.pop("_field_saved_token", None)
        st.session_state.pop("_field_saved_id", None)
        st.rerun()


def _render_failures(failures: list[tuple[str, str]]) -> None:
    if not failures:
        return
    with st.expander(f"{len(failures)} photo(s) skipped"):
        for name, reason in failures:
            st.markdown(f"- **{name}** — {reason}")


# ---------------------------------------------------------------------------
# Save to database
# ---------------------------------------------------------------------------
def _build_field_scan_record(report: dict) -> dict:
    """Map an aggregated field report to src.db's `field_scans` columns.

    Per-leaf detail (thumbnails, individual filenames) is intentionally not
    persisted — this table stores the field-level summary, matching what
    the report itself is: an aggregate, not a re-storable image gallery.
    """
    return {
        "crop_name": report["crop"],
        "num_images": report["n_total"],
        "num_healthy": report["n_healthy"],
        "num_diseased": report["n_diseased"],
        "healthy_pct": report["healthy_pct"],
        "dominant_disease": report["dominant_disease"],
        "field_health_score": report["field_health_score"],
        "severity_breakdown": json.dumps(report["severity_counts"]),
        "disease_breakdown": json.dumps(report["disease_counts"]),
    }


# ---------------------------------------------------------------------------
# PDF report export
# ---------------------------------------------------------------------------
def _render_pdf_download(report: dict, rep_severity: str | None) -> None:
    """A downloadable, farmer-shareable PDF for this field scan.

    Reuses whatever the yield-loss calculator above is currently set to
    (field size/yield/price, kept in session_state under the same
    `_fieldscan_yl_*` keys render_yield_loss_estimator() writes), the same
    pattern pages/disease.py's PDF export uses.
    """
    from src.report_generator import generate_field_scan_report_pdf

    yield_loss_estimate = None
    if report["dominant_disease"] and rep_severity and get_yield_loss_range(report["dominant_disease"], rep_severity) is not None:
        unit = st.session_state.get("_fieldscan_yl_unit", "Hectares")
        size = st.session_state.get("_fieldscan_yl_size", 1.0)
        yield_per_ha = st.session_state.get("_fieldscan_yl_yield", REFERENCE_YIELD_T_PER_HA.get(report["crop"], 5.0))
        price = st.session_state.get("_fieldscan_yl_price", 0.0)
        field_size_ha = size if unit == "Hectares" else size * HECTARES_PER_ACRE
        yield_loss_estimate = estimate_yield_loss(
            report["dominant_disease"], rep_severity, field_size_ha, yield_per_ha, price,
        )

    try:
        pdf_bytes = generate_field_scan_report_pdf(report, yield_loss_estimate=yield_loss_estimate)
    except Exception:
        logger.exception("Unexpected error generating field scan PDF report")
        st.error("Couldn't generate the PDF report right now. Please try again.")
        return

    file_name = f"field_scan_{report['crop'].lower()}_{int(time.time())}.pdf"
    st.download_button(
        "Download PDF Report",
        data=pdf_bytes,
        file_name=file_name,
        mime="application/pdf",
        use_container_width=True,
        key="_fieldscan_pdf_download",
    )


def _render_save_section(report: dict) -> None:
    """'Save Field Scan' button, guarded against duplicate inserts.

    Mirrors pages/disease.py's save pattern: each freshly *computed* report
    carries a unique `_scan_token`; a save is only allowed once per token.
    """
    token = report["_scan_token"]
    saved_token = st.session_state.get("_field_saved_token")

    if saved_token == token:
        saved_id = st.session_state.get("_field_saved_id")
        st.success(f"Field scan saved to database (ID: {saved_id}).")
        st.button("Saved ✓", use_container_width=True, disabled=True, key="_field_saved_btn")
        return

    if st.button("Save Field Scan", type="primary", use_container_width=True, key="_field_save_btn"):
        with safe_action("Saving field scan"):
            with st.spinner("Saving field scan…"):
                record = _build_field_scan_record(report)
                scan_id = insert_field_scan(record)
            st.session_state["_field_saved_token"] = token
            st.session_state["_field_saved_id"] = scan_id
            st.rerun()


if __name__ == "__main__":
    # Standalone entry for direct viewing
    import streamlit as st
    st.set_page_config(page_title="Field Scan", layout="wide")
    from utils.ui import inject_custom_css, render_sidebar
    inject_custom_css()
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "field_scan"
    render_sidebar()
    st.session_state["current_page"] = "field_scan"
    render()