"""Disease Detection page — upload a leaf image and run detection.

Uses the trained TensorFlow/Keras MobileNetV2 model for inference.
"""

from __future__ import annotations

import re
import time
import uuid
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf

from config import CONFIDENCE_THRESHOLD, IMAGE_SIZE, DISEASE_MODELS, DEFAULT_DISEASE_CROP, UPLOAD_DIR, get_trained_crops
from src.dataset_prep import load_class_names
from src.db import insert_disease_analysis
from src.errors import PredictionError, GradCAMError, logger, safe_action
from src.gradcam import generate_gradcam, overlay_heatmap
from src.image_preprocessing import preprocess_leaf_image, ImageValidationError
from src.i18n import get_language, tr_crop, tr_disease, tr_severity, tr_severity_action, tr_recommendation, tr_label, tr_template
from src.yield_loss import get_yield_loss_range, estimate_yield_loss, REFERENCE_YIELD_T_PER_HA, HECTARES_PER_ACRE
from utils.ui import (
    page_header,
    callout,
    card,
    footer,
    metric_tile,
    pretty_name,
    CHART_THEME,
)
from utils.icons import icon_html

# ---------------------------------------------------------------------------
# Disease catalog (used for recommendations and severity mapping)
# Keyed by class name rather than crop, since every trained crop model so
# far shares the same PlantVillage-style classes: Healthy, Early_Blight,
# Late_Blight. If a new crop introduces a class name not listed here, it
# still works — .get() below falls back to "Unknown"/a generic message —
# but for a good demo, add that class's entry to these maps too.
# ---------------------------------------------------------------------------

# Severity and recommendation per model class
SEVERITY_MAP = {
    "Healthy": "None",
    "Early_Blight": "Moderate",
    "Late_Blight": "High",
    "Brown_Spot": "Moderate",
    "Leaf_Blast": "High",
    "Neck_Blast": "High",
    "Brown_Rust": "Moderate",
    "Yellow_Rust": "Moderate",
    "Common_Rust": "Moderate",
    "Gray_Leaf_Spot": "Moderate",
    "Northern_Leaf_Blight": "High",
}

RECOMMENDATION_MAP = {
    "Healthy": "Crop looks healthy. Maintain regular monitoring and balanced irrigation.",
    "Early_Blight": "Early blight detected. Remove infected leaves, apply copper-based fungicide, and improve air circulation between plants.",
    "Late_Blight": "Late blight detected. Urgent: destroy affected plants, apply mancozeb, and avoid working in the field while wet.",
    "Brown_Spot": "Brown spot detected. Improve field nutrition (especially potassium), apply a recommended fungicide (e.g. propiconazole), and avoid water stress.",
    "Leaf_Blast": "Leaf blast detected. Apply a tricyclazole-based fungicide, avoid excess nitrogen, and maintain proper water management in the field.",
    "Neck_Blast": "Neck blast detected. Urgent: apply fungicide at early panicle stage, avoid dense planting, and monitor closely since this can significantly cut yield.",
    "Brown_Rust": "Brown rust detected. Apply a triazole-based fungicide, monitor for warm humid conditions that favor spread, and consider a resistant variety for the next planting.",
    "Yellow_Rust": "Yellow rust detected. Apply a triazole-based fungicide promptly, monitor cool humid conditions closely (this rust spreads fast in cool weather), and scout neighboring fields.",
    "Common_Rust": "Common rust detected. Apply a strobilurin or triazole fungicide if severe, and favor rust-resistant hybrids in future plantings.",
    "Gray_Leaf_Spot": "Gray leaf spot detected. Rotate crops away from corn/residue for a season, apply a foliar fungicide if disease pressure is high, and improve field airflow.",
    "Northern_Leaf_Blight": "Northern leaf blight detected. Apply a foliar fungicide promptly, especially before tasseling, and consider resistant hybrids next season since yield loss can be severe.",
}

SEVERITY_META = {
    "None":     ("#7FA687", "No action needed"),
    "Mild":     ("#D6A34B", "Monitor closely"),
    "Moderate": ("#C97A3B", "Treat promptly"),
    "High":     ("#B5564B", "Intervene urgently"),
}

# Colors for confidence breakdown chart
CLASS_COLORS = {
    "Healthy": "#7FA687",
    "Early_Blight": "#B5564B",
    "Late_Blight": "#B5564B",
    "Brown_Spot": "#C97A3B",
    "Leaf_Blast": "#B5564B",
    "Neck_Blast": "#7C3730",
    "Brown_Rust": "#8B4A2E",
    "Yellow_Rust": "#C9973B",
    "Common_Rust": "#B5713B",
    "Gray_Leaf_Spot": "#8A8578",
    "Northern_Leaf_Blight": "#7C5A3B",
}


# ---------------------------------------------------------------------------
# Model loading (cached)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading disease detection model…")
def load_model(crop: str):
    """Load the trained Keras model and class labels for one crop.

    Cached per crop (Streamlit keys the cache by argument value), so
    switching crops in the UI loads each model once and reuses it after.
    Returns (None, None) if that crop's model is missing or fails to load —
    the caller shows a friendly "model unavailable" message either way. Any
    unexpected loading error (corrupted file, version mismatch, ...) is
    logged in full server-side, never shown to the user.
    """
    paths = DISEASE_MODELS.get(crop)
    if paths is None or not Path(paths["model_path"]).exists():
        return None, None
    try:
        model = tf.keras.models.load_model(paths["model_path"])
        class_names = load_class_names(paths["labels_path"])
        return model, class_names
    except Exception:
        logger.exception("Failed to load disease detection model for crop=%s", crop)
        return None, None


# ---------------------------------------------------------------------------
# Image preprocessing (OpenCV pipeline — see src/image_preprocessing.py)
# ---------------------------------------------------------------------------
def preprocess_image(uploaded_file, target_size=IMAGE_SIZE,
                     denoise: bool = False, remove_background: bool = False):
    """Preprocess an uploaded leaf image for model inference.

    Delegates to src.image_preprocessing.preprocess_leaf_image — a
    validated, OpenCV-based pipeline (format/corruption/size checks,
    RGB conversion, resize, optional denoise/background handling,
    MobileNetV2-compatible normalization). The trained model itself is
    untouched; this only changes how bytes become its input tensor.
    """
    file_bytes = uploaded_file.read()
    uploaded_file.seek(0)  # reset for potential re-use
    return preprocess_leaf_image(
        file_bytes, target_size=target_size,
        denoise=denoise, remove_background=remove_background,
    )


def predict_disease(model, class_names, image_batch, confidence_threshold=CONFIDENCE_THRESHOLD):
    """Run inference and return prediction dict."""
    try:
        # Run inference
        preds = model.predict(image_batch, verbose=0)[0]

        # Get top prediction
        pred_idx = int(np.argmax(preds))
        confidence = float(preds[pred_idx])
        disease = class_names[pred_idx]

        # Build breakdown
        breakdown = []
        for i, name in enumerate(class_names):
            breakdown.append({
                "name": name,
                "color": CLASS_COLORS.get(name, "#7C8571888"),
                "prob": float(preds[i]),
            })

        # Sort by probability descending
        breakdown.sort(key=lambda b: b["prob"], reverse=True)

        severity = SEVERITY_MAP.get(disease, "Unknown")
        recommendation = RECOMMENDATION_MAP.get(disease, "No recommendation available.")
        is_healthy = (disease == "Healthy")
        low_confidence = confidence < confidence_threshold and not is_healthy

        return {
            "disease": disease,
            "confidence": confidence,
            "severity": severity,
            "recommendation": recommendation,
            "breakdown": breakdown,
            "is_healthy": is_healthy,
            "low_confidence": low_confidence,
            "threshold": confidence_threshold,
        }
    except Exception as e:
        logger.exception("Disease prediction failed")
        raise PredictionError(
            "Disease prediction failed. The image may be incompatible with "
            "the model, or the model file may be corrupted. Please try a "
            "different image."
        ) from e


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
def render() -> None:
    lang = get_language()
    page_header(
        "disease",
        tr_label("Disease Detection", lang),
        tr_label("Upload a crop leaf image to detect diseases with AI.", lang),
    )

    trained_crops = get_trained_crops()

    if not trained_crops:
        no_model_msg = tr_label(
            "If a model file exists but still won't load, check the server logs for details.", lang
        )
        callout(
            f"{icon_html('warning', size=18)}<b>{tr_label('No trained model found.', lang)}</b> "
            f"{tr_label('Train one first using', lang)} "
            "<code>python -m src.model_training --data-dir data/samples --crop Tomato</code> "
            f"{tr_label('(swap', lang)} <code>--crop</code> {tr_label('for any crop in', lang)} "
            f"<code>config.DISEASE_MODELS</code>). "
            f"{no_model_msg}"
        )
        footer()
        return

    default_index = trained_crops.index(DEFAULT_DISEASE_CROP) if DEFAULT_DISEASE_CROP in trained_crops else 0
    crop = st.selectbox(tr_label("Crop", lang), trained_crops, index=default_index, format_func=lambda c: tr_crop(c, lang))

    # Load model for the selected crop
    model, class_names = load_model(crop)

    if model is None:
        callout(
            f"{icon_html('warning', size=18)}<b>{tr_label('Model unavailable.', lang)}</b> "
            + tr_template(
                "{crop}'s model file couldn't be loaded even though it's listed as "
                "trained — check the server logs for details.",
                lang, crop=tr_crop(crop, lang),
            )
        )
        footer()
        return

    class_names_label = ", ".join(tr_disease(n, lang) for n in class_names)
    callout(
        f"{icon_html('success', size=18)}<b>{tr_label('Model loaded', lang)}</b> — {tr_crop(crop, lang)} ({model.name}) "
        + tr_template("with {n} classes:", lang, n=len(class_names))
        + f" {class_names_label}"
    )

    if st.session_state.get("_disease_crop") != crop:
        st.session_state["_disease_crop"] = crop
        st.session_state.pop("_disease_pred", None)
        st.session_state.pop("_disease_saved_token", None)
        st.session_state.pop("_disease_saved_id", None)

    col_input, col_result = st.columns([2, 3])

    # ------------------------------------------------------------------ inputs
    with col_input:
        st.markdown(f"#### {tr_label('1 · Upload leaf image', lang)}")

        uploaded = st.file_uploader(
            tr_label("Leaf image (JPG / PNG)", lang),
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )

        if uploaded is not None:
            st.image(uploaded, caption=tr_label("Uploaded leaf", lang), use_container_width=True)
        else:
            st.info(tr_label("Drop a clear, well-lit photo of a single leaf here.", lang))

        with st.expander(tr_label("Advanced options", lang)):
            threshold = st.slider(
                tr_label("Confidence threshold", lang),
                0.0, 1.0,
                float(CONFIDENCE_THRESHOLD), 0.05,
                help=tr_label("Predictions below this confidence are flagged as uncertain.", lang),
            )
            st.markdown(f"**{tr_label('Preprocessing', lang)}**")
            denoise = st.checkbox(
                tr_label("Noise reduction", lang),
                value=False,
                help=tr_label(
                    "Apply OpenCV non-local-means denoising before inference. "
                    "Useful for grainy or low-light photos.",
                    lang,
                ),
            )
            remove_background = st.checkbox(
                tr_label("Background handling", lang),
                value=False,
                help=tr_label(
                    "Softly flatten non-leaf-colored background toward neutral "
                    "gray so the model focuses on the leaf. Useful for busy "
                    "backgrounds; skip for close-up leaf-only photos.",
                    lang,
                ),
            )

        analyze = st.button(
            tr_label("Analyze", lang), type="primary", use_container_width=True,
            disabled=(uploaded is None),
        )

    if uploaded is None:
        st.session_state.pop("_disease_pred", None)
        st.session_state.pop("_disease_saved_token", None)
        st.session_state.pop("_disease_saved_id", None)

    # ----------------------------------------------------------------- results
    with col_result:
        st.markdown(f"#### {tr_label('2 · Prediction result', lang)}")

        # Session state holds the last prediction
        pred = st.session_state.get("_disease_pred")

        if analyze and uploaded is not None:
            try:
                # Preprocess with loading indicator
                with st.spinner(tr_label("Preprocessing image…", lang)):
                    image_batch = preprocess_image(
                        uploaded, denoise=denoise, remove_background=remove_background,
                    )

                # Run inference with loading indicator
                with st.spinner(tr_label("Running disease detection…", lang)):
                    pred = predict_disease(model, class_names, image_batch, threshold)

                # Explainability: Grad-CAM heatmap over the same image batch
                # that was just classified, explaining the top prediction.
                # A failure here must never hide the (already successful)
                # prediction above — degrade to no heatmap instead.
                with st.spinner(tr_label("Computing explainability heatmap…", lang)):
                    try:
                        pred_idx = class_names.index(pred["disease"])
                        gradcam = generate_gradcam(model, image_batch, pred_index=pred_idx)
                        pred["gradcam_heatmap"] = gradcam["heatmap"]
                        pred["gradcam_base_image"] = gradcam["base_image"]
                        pred["gradcam_error"] = None
                    except GradCAMError as e:
                        pred["gradcam_heatmap"] = None
                        pred["gradcam_base_image"] = None
                        pred["gradcam_error"] = str(e)
                    except Exception:
                        logger.exception("Unexpected error during Grad-CAM generation")
                        pred["gradcam_heatmap"] = None
                        pred["gradcam_base_image"] = None
                        pred["gradcam_error"] = tr_label(
                            "Couldn't generate the explainability heatmap for "
                            "this prediction.",
                            lang,
                        )

                # Stash what's needed to save this analysis later: the crop,
                # the raw image bytes/filename (so "Save Analysis" can persist
                # the exact image that was analyzed, without depending on the
                # file_uploader widget still holding it), and a unique token
                # to guard against double-saving the same result on rerun.
                pred["_crop"] = crop
                pred["_image_bytes"] = uploaded.getvalue()
                pred["_image_name"] = uploaded.name
                pred["_analysis_token"] = uuid.uuid4().hex

                st.session_state["_disease_pred"] = pred
                st.rerun()

            except ImageValidationError as e:
                st.error(str(e))
            except ValueError as e:
                st.error(str(e))
            except PredictionError as e:
                st.error(str(e))
            except Exception:
                logger.exception("Unexpected error during disease analysis")
                st.error(
                    tr_label(
                        "Analyzing this image failed unexpectedly. Please try "
                        "again. If the problem continues, contact the app maintainer.",
                        lang,
                    )
                )

        if pred is None:
            card(
                tr_label("Awaiting analysis", lang),
                tr_label(
                    "Upload an image and click **Analyze** to see the prediction, "
                    "confidence, severity, and recommendation.",
                    lang,
                ),
            )
        else:
            _render_result(pred)


# ---------------------------------------------------------------------------
# Result rendering
# ---------------------------------------------------------------------------
def _render_result(pred: dict) -> None:
    """Render the full prediction result block."""
    lang = get_language()

    # Banner
    sev_color, sev_action_en = SEVERITY_META.get(pred["severity"], ("#93998A", "Unknown"))
    sev_action = tr_severity_action(sev_action_en, lang)
    banner_bg = "#EAEFE2" if pred["is_healthy"] else "#F4EAD9"
    banner_border = "#7FA687" if pred["is_healthy"] else sev_color
    condition_label = tr_label("Detected condition", lang)
    disease_label = tr_disease(pred["disease"], lang)
    st.markdown(
        f"""
        <div style="background:{banner_bg};border-left:5px solid {banner_border};
                    border-radius:12px;padding:1rem 1.25rem;margin-bottom:1rem">
          <div style="font-size:.8rem;color:#4E5646;text-transform:uppercase;
                      letter-spacing:.04em">{condition_label}</div>
          <div style="font-size:1.5rem;font-weight:700;color:{pred.get('color', CLASS_COLORS.get(pred['disease'], '#23291F'))}">
            {disease_label}
          </div>
          <div style="font-size:.85rem;color:#5B6353">{sev_action}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # KPI tiles
    c1, c2, c3 = st.columns(3)
    with c1:
        metric_tile(tr_label("Confidence", lang), f"{pred['confidence']*100:.0f}%", tr_label("model output", lang))
    with c2:
        metric_tile(tr_label("Severity", lang), tr_severity(pred["severity"], lang), sev_action)
    with c3:
        metric_tile(tr_label("Threshold", lang), f"{pred['threshold']*100:.0f}%", tr_label("cutoff for reliable result", lang))

    # Low-confidence warning
    if pred["low_confidence"] and not pred["is_healthy"]:
        callout(
            f"{icon_html('warning', size=18)}"
            + tr_label(
                "Confidence is below the threshold. "
                "The result may be uncertain — consider retaking the photo with "
                "better lighting/focus.",
                lang,
            )
        )

    # Confidence breakdown bar chart
    st.markdown(f"#### {tr_label('Confidence breakdown by class', lang)}")
    bd = pred["breakdown"]
    fig = go.Figure(go.Bar(
        orientation="h",
        x=[b["prob"] * 100 for b in bd],
        y=[tr_disease(b["name"], lang) for b in bd],
        text=[f"{b['prob']*100:.0f}%" for b in bd],
        textposition="outside",
        marker=dict(color=[b["color"] for b in bd]),
    ))
    fig.add_vline(
        x=pred["threshold"] * 100,
        line_dash="dash",
        line_color="#7C8571",
        annotation_text=tr_label("threshold", lang),
        annotation_position="top right"
    )
    fig.update_layout(
        **CHART_THEME,
        margin=dict(t=10, b=10),
        xaxis_title=tr_label("Confidence (%)", lang),
        height=max(220, len(bd) * 42),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Explainability — Grad-CAM
    _render_gradcam(pred, lang)

    # Recommendation
    st.markdown(f"#### {tr_label('Recommendation', lang)}")
    rec_icon = icon_html("healthy" if pred["is_healthy"] else "diseased", size=20)
    recommendation_text = tr_recommendation(pred["disease"], lang, fallback=pred["recommendation"])
    st.markdown(
        f"""
        <div class="card">
          {rec_icon} {recommendation_text}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Yield loss / economic impact estimate
    render_yield_loss_estimator(pred["_crop"], pred["disease"], pred["severity"], key_prefix="disease")

    # PDF report export
    st.markdown("---")
    _render_pdf_download(pred, lang)

    # Save to database
    st.markdown("---")
    _render_save_section(pred, lang)

    if st.button(tr_label("Re-run", lang), use_container_width=True):
        st.session_state["_disease_pred"] = None
        st.session_state.pop("_disease_saved_token", None)
        st.session_state.pop("_disease_saved_id", None)
        st.rerun()


# ---------------------------------------------------------------------------
# PDF report export
# ---------------------------------------------------------------------------
def _render_pdf_download(pred: dict, lang: str) -> None:
    """A downloadable, farmer-shareable PDF for this one result.

    Reuses whatever the yield-loss calculator above is currently set to
    (field size/yield/price, kept in session_state under the same
    `_disease_yl_*` keys render_yield_loss_estimator() writes) so the PDF
    reflects the numbers actually on screen, without asking the person to
    re-enter anything. If the calculator was never opened for this result
    (e.g. the crop is Healthy, so it never rendered), the report simply
    omits that section — src.report_generator handles a None estimate.
    """
    from src.report_generator import generate_disease_report_pdf

    yield_loss_estimate = None
    if get_yield_loss_range(pred["disease"], pred["severity"]) is not None:
        unit = st.session_state.get("_disease_yl_unit", "Hectares")
        size = st.session_state.get("_disease_yl_size", 1.0)
        yield_per_ha = st.session_state.get("_disease_yl_yield", REFERENCE_YIELD_T_PER_HA.get(pred["_crop"], 5.0))
        price = st.session_state.get("_disease_yl_price", 0.0)
        field_size_ha = size if unit == "Hectares" else size * HECTARES_PER_ACRE
        yield_loss_estimate = estimate_yield_loss(pred["disease"], pred["severity"], field_size_ha, yield_per_ha, price)

    try:
        pdf_bytes = generate_disease_report_pdf(pred, yield_loss_estimate=yield_loss_estimate)
    except Exception:
        logger.exception("Unexpected error generating PDF report")
        st.error(tr_label("Couldn't generate the PDF report right now. Please try again.", lang))
        return

    file_name = f"crop_diagnosis_{pred['_crop'].lower()}_{pred['disease'].lower()}_{int(time.time())}.pdf"
    st.download_button(
        tr_label("Download PDF Report", lang),
        data=pdf_bytes,
        file_name=file_name,
        mime="application/pdf",
        use_container_width=True,
        key="_disease_pdf_download",
    )


# ---------------------------------------------------------------------------
# Yield loss / economic impact estimator
#
# Shared by Disease Detection (this page) and Field Scan (pages/field_scan.py
# imports this the same way it already imports SEVERITY_MAP/CLASS_COLORS/
# load_model from here) — one calculator, one place its math can go wrong.
# All the published-data lookup and arithmetic lives in src/yield_loss.py;
# this function is presentation only.
# ---------------------------------------------------------------------------
def render_yield_loss_estimator(crop: str, disease: str, severity: str, key_prefix: str) -> None:
    lang = get_language()
    if disease == "Healthy":
        return  # nothing to estimate
    if get_yield_loss_range(disease, severity) is None:
        return  # this disease/severity isn't in the published-data table

    st.markdown(f"#### {tr_label('Estimated yield loss if untreated', lang)}")
    st.caption(
        tr_template(
            "Based on published agricultural research for {disease} at "
            "{severity} severity. Adjust the figures below to your own field.",
            lang, disease=tr_disease(disease, lang), severity=tr_severity(severity, lang).lower(),
        )
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        unit = st.selectbox(
            tr_label("Field size unit", lang), ["Hectares", "Acres"], key=f"_{key_prefix}_yl_unit",
            format_func=lambda u: tr_label(u, lang),
        )
    with c2:
        field_size = st.number_input(
            tr_template("Field size ({unit})", lang, unit=tr_label(unit.lower(), lang)),
            min_value=0.0, value=1.0, step=0.1,
            key=f"_{key_prefix}_yl_size",
        )
    with c3:
        default_yield = REFERENCE_YIELD_T_PER_HA.get(crop, 5.0)
        yield_per_ha = st.number_input(
            tr_label("Expected yield (t/ha if healthy)", lang), min_value=0.0,
            value=default_yield, step=0.5, key=f"_{key_prefix}_yl_yield",
            help=tr_label(
                "Pre-filled with a rough global reference for this crop — "
                "replace with your farm's typical yield for a more accurate estimate.",
                lang,
            ),
        )
    with c4:
        price_per_unit = st.number_input(
            tr_label("Price per tonne (optional)", lang), min_value=0.0, value=0.0, step=10.0,
            key=f"_{key_prefix}_yl_price",
            help=tr_label("Leave at 0 to see only the yield-loss estimate, with no revenue figure.", lang),
        )

    field_size_ha = field_size if unit == "Hectares" else field_size * HECTARES_PER_ACRE
    est = estimate_yield_loss(disease, severity, field_size_ha, yield_per_ha, price_per_unit)
    if est is None:
        return

    low, high = est["loss_pct_low"], est["loss_pct_high"]
    color = "#B5564B" if high >= 60 else "#C97A3B" if high >= 30 else "#D6A34B"

    revenue_html = ""
    if est["revenue_lost_low"] is not None:
        revenue_html = (
            f'<div style="margin-top:.5rem;font-size:1.15rem;font-weight:700;color:{color}">'
            f'≈ {est["revenue_lost_low"]:,.0f} – {est["revenue_lost_high"]:,.0f} {tr_label("estimated revenue at risk", lang)}'
            f'</div>'
        )

    st.markdown(
        f"""
        <div class="card" style="border-left:5px solid {color}">
          <div style="font-size:1.5rem;font-weight:700;color:var(--ink)">{low:.0f}–{high:.0f}% {tr_label('yield loss', lang)}</div>
          <div style="color:#4E5646;margin-top:.3rem">
            ≈ {est['yield_lost_low']:.1f}–{est['yield_lost_high']:.1f} t {tr_label('on your', lang)} {field_size_ha:.2f} ha {tr_label('field (expected', lang)}
            {est['expected_yield']:.1f} {tr_label('t if healthy)', lang)}
          </div>
          {revenue_html}
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div style="font-size:.78rem;color:#5B6353;margin-top:.4rem">
          {icon_html('info', size=14, margin_right='.3em')}
          {tr_label("Planning estimate from published crop-disease research, not a guarantee — actual loss depends on variety, timing of infection, weather, and management. Not financial advice.", lang)}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Explainability — Grad-CAM
# ---------------------------------------------------------------------------
def _render_gradcam(pred: dict, lang: str) -> None:
    """Render the Grad-CAM heatmap next to the leaf image the model saw.

    The heavy part (gradient computation) already ran once, right after
    inference, and its result is cached on `pred`. Adjusting the overlay
    opacity slider below only re-blends two already-computed arrays
    (src.gradcam.overlay_heatmap) — no re-inference, no gradient tape.
    """
    st.markdown(f"#### {tr_label('Why this prediction? (Grad-CAM)', lang)}")

    heatmap = pred.get("gradcam_heatmap")
    base_image = pred.get("gradcam_base_image")

    if heatmap is None or base_image is None:
        callout(
            f"{icon_html('warning', size=18)}"
            f"{pred.get('gradcam_error') or tr_label('Explainability heatmap unavailable for this prediction.', lang)}"
        )
        return

    alpha = st.slider(
        tr_label("Heatmap intensity", lang), 0.0, 1.0, 0.4, 0.05,
        help=tr_label(
            "How strongly the heatmap is blended over the leaf image below. "
            "This only re-blends the already-computed heatmap — it does not "
            "re-run the model.",
            lang,
        ),
        key="_gradcam_alpha",
    )
    overlay = overlay_heatmap(heatmap, base_image, alpha=alpha)

    c1, c2 = st.columns(2)
    with c1:
        st.image(base_image, caption=tr_label("What the model saw (224×224 input)", lang), use_container_width=True)
    with c2:
        st.image(overlay, caption=f"{tr_label('Grad-CAM for', lang)} '{tr_disease(pred['disease'], lang)}'", use_container_width=True)

    st.markdown(
        f"""
        <div style="font-size:.8rem;color:#5B6353;margin-top:.25rem">
          {icon_html('info', size=14, margin_right='.3em')}
          {tr_label("Warmer regions (red/yellow) contributed most to the prediction above; cooler regions (blue) contributed least. Computed by backpropagating the predicted class score to the model's last convolutional layer (Grad-CAM, Selvaraju et al. 2017).", lang)}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Save to database
# ---------------------------------------------------------------------------
def _save_uploaded_image(image_bytes: bytes, original_name: str, crop: str) -> str:
    """Persist the analyzed leaf image to disk and return its path.

    Filenames are sanitized and timestamped to avoid collisions between
    repeated uploads of the same original filename.
    """
    upload_dir = Path(UPLOAD_DIR) / "disease"
    upload_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", original_name or "leaf.jpg")
    filename = f"{crop.lower()}_{int(time.time() * 1000)}_{safe_name}"
    path = upload_dir / filename
    path.write_bytes(image_bytes)
    return str(path)


def _build_disease_db_record(pred: dict, image_path: str) -> dict:
    """Map a disease prediction result to src.db's `disease_analyses` columns."""
    return {
        "crop_name": pred["_crop"],
        "image_path": image_path,
        "disease": pred["disease"],
        "confidence": pred["confidence"],
        "severity": pred["severity"],
        "is_healthy": int(bool(pred["is_healthy"])),
        "recommendation": pred["recommendation"],
        # created_at (timestamp) is filled in automatically by insert_disease_analysis().
    }


def _render_save_section(pred: dict, lang: str) -> None:
    """'Save Analysis' button, guarded against duplicate inserts.

    Mirrors pages/health.py's save pattern: each freshly *computed*
    prediction carries a unique `_analysis_token`; a save is only allowed
    once per token, and the button is replaced with a confirmation
    afterward so a stray rerun or repeat click can't insert the same
    analysis (and re-save the same image file) twice.
    """
    token = pred["_analysis_token"]
    saved_token = st.session_state.get("_disease_saved_token")

    if saved_token == token:
        saved_id = st.session_state.get("_disease_saved_id")
        st.success(tr_template("Analysis saved to database (ID: {id}).", lang, id=saved_id))
        st.button(f"{tr_label('Saved', lang)} ✓", use_container_width=True, disabled=True, key="_disease_saved_btn")
        return

    if st.button(tr_label("Save Analysis", lang), type="primary", use_container_width=True, key="_disease_save_btn"):
        with safe_action("Saving analysis"):
            with st.spinner(tr_label("Saving analysis…", lang)):
                image_path = _save_uploaded_image(
                    pred["_image_bytes"], pred["_image_name"], pred["_crop"]
                )
                record = _build_disease_db_record(pred, image_path)
                analysis_id = insert_disease_analysis(record)
            st.session_state["_disease_saved_token"] = token
            st.session_state["_disease_saved_id"] = analysis_id
            st.rerun()


if __name__ == "__main__":
    # Standalone entry for direct viewing
    import streamlit as st
    st.set_page_config(page_title="Disease Detection", layout="wide")
    from utils.ui import inject_custom_css, render_sidebar
    inject_custom_css()
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "disease"
    render_sidebar()
    st.session_state["current_page"] = "disease"
    render()