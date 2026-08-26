"""Disease Detection page — upload a leaf image and run detection.

Uses the trained TensorFlow/Keras MobileNetV2 model for inference.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf

from config import CONFIDENCE_THRESHOLD, IMAGE_SIZE, DISEASE_MODELS, DEFAULT_DISEASE_CROP, get_trained_crops
from src.dataset_prep import load_class_names
from src.errors import PredictionError, logger
from src.image_preprocessing import preprocess_leaf_image, ImageValidationError
from utils.ui import (
    page_header,
    callout,
    card,
    footer,
    metric_tile,
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
}

RECOMMENDATION_MAP = {
    "Healthy": "Crop looks healthy. Maintain regular monitoring and balanced irrigation.",
    "Early_Blight": "Early blight detected. Remove infected leaves, apply copper-based fungicide, and improve air circulation between plants.",
    "Late_Blight": "Late blight detected. Urgent: destroy affected plants, apply mancozeb, and avoid working in the field while wet.",
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
    page_header(
        "disease",
        "Disease Detection",
        "Upload a crop leaf image to detect diseases with AI.",
    )

    trained_crops = get_trained_crops()

    if not trained_crops:
        callout(
            f"{icon_html('warning', size=18)}<b>No trained model found.</b> Train one first using "
            "<code>python -m src.model_training --data-dir data/samples --crop Tomato</code> "
            "(swap <code>--crop</code> for any crop in <code>config.DISEASE_MODELS</code>). "
            "If a model file exists but still won't load, check the server logs for details."
        )
        footer()
        return

    default_index = trained_crops.index(DEFAULT_DISEASE_CROP) if DEFAULT_DISEASE_CROP in trained_crops else 0
    crop = st.selectbox("Crop", trained_crops, index=default_index)

    # Load model for the selected crop
    model, class_names = load_model(crop)

    if model is None:
        callout(
            f"{icon_html('warning', size=18)}<b>Model unavailable.</b> "
            f"{crop}'s model file couldn't be loaded even though it's listed as trained — "
            "check the server logs for details."
        )
        footer()
        return

    callout(
        f"{icon_html('success', size=18)}<b>Model loaded</b> — {crop} ({model.name}) with "
        f"{len(class_names)} classes: {', '.join(class_names)}"
    )

    if st.session_state.get("_disease_crop") != crop:
        st.session_state["_disease_crop"] = crop
        st.session_state.pop("_disease_pred", None)

    col_input, col_result = st.columns([2, 3])

    # ------------------------------------------------------------------ inputs
    with col_input:
        st.markdown("#### 1 · Upload leaf image")

        uploaded = st.file_uploader(
            "Leaf image (JPG / PNG)",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )

        if uploaded is not None:
            st.image(uploaded, caption="Uploaded leaf", use_container_width=True)
        else:
            st.info("Drop a clear, well-lit photo of a single leaf here.")

        with st.expander("Advanced options"):
            threshold = st.slider(
                "Confidence threshold",
                0.0, 1.0,
                float(CONFIDENCE_THRESHOLD), 0.05,
                help="Predictions below this confidence are flagged as uncertain.",
            )
            st.markdown("**Preprocessing**")
            denoise = st.checkbox(
                "Noise reduction",
                value=False,
                help="Apply OpenCV non-local-means denoising before inference. "
                     "Useful for grainy or low-light photos.",
            )
            remove_background = st.checkbox(
                "Background handling",
                value=False,
                help="Softly flatten non-leaf-colored background toward neutral "
                     "gray so the model focuses on the leaf. Useful for busy "
                     "backgrounds; skip for close-up leaf-only photos.",
            )

        analyze = st.button(
            "Analyze", type="primary", use_container_width=True,
            disabled=(uploaded is None),
        )

    if uploaded is None:
        st.session_state.pop("_disease_pred", None)

    # ----------------------------------------------------------------- results
    with col_result:
        st.markdown("#### 2 · Prediction result")

        # Session state holds the last prediction
        pred = st.session_state.get("_disease_pred")

        if analyze and uploaded is not None:
            try:
                # Preprocess with loading indicator
                with st.spinner("Preprocessing image…"):
                    image_batch = preprocess_image(
                        uploaded, denoise=denoise, remove_background=remove_background,
                    )

                # Run inference with loading indicator
                with st.spinner("Running disease detection…"):
                    pred = predict_disease(model, class_names, image_batch, threshold)

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
                    "Analyzing this image failed unexpectedly. Please try "
                    "again. If the problem continues, contact the app maintainer."
                )

        if pred is None:
            card(
                "Awaiting analysis",
                "Upload an image and click **Analyze** to see the prediction, "
                "confidence, severity, and recommendation.",
            )
        else:
            _render_result(pred)


# ---------------------------------------------------------------------------
# Result rendering
# ---------------------------------------------------------------------------
def _render_result(pred: dict) -> None:
    """Render the full prediction result block."""
    # Banner
    sev_color, sev_action = SEVERITY_META.get(pred["severity"], ("#93998A", "Unknown"))
    banner_bg = "#EAEFE2" if pred["is_healthy"] else "#F4EAD9"
    banner_border = "#7FA687" if pred["is_healthy"] else sev_color
    st.markdown(
        f"""
        <div style="background:{banner_bg};border-left:5px solid {banner_border};
                    border-radius:12px;padding:1rem 1.25rem;margin-bottom:1rem">
          <div style="font-size:.8rem;color:#4E5646;text-transform:uppercase;
                      letter-spacing:.04em">Detected condition</div>
          <div style="font-size:1.5rem;font-weight:700;color:{pred['color'] if 'color' in pred else CLASS_COLORS.get(pred['disease'], '#23291F')}">
            {pred['disease']}
          </div>
          <div style="font-size:.85rem;color:#5B6353">{sev_action}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # KPI tiles
    c1, c2, c3 = st.columns(3)
    with c1:
        metric_tile("Confidence", f"{pred['confidence']*100:.0f}%", "model output")
    with c2:
        metric_tile("Severity", pred["severity"], sev_action)
    with c3:
        metric_tile("Threshold", f"{pred['threshold']*100:.0f}%", "cutoff for reliable result")

    # Low-confidence warning
    if pred["low_confidence"] and not pred["is_healthy"]:
        callout(
            f"{icon_html('warning', size=18)}Confidence is below the threshold. "
            "The result may be uncertain — consider retaking the photo with "
            "better lighting/focus."
        )

    # Confidence breakdown bar chart
    st.markdown("#### Confidence breakdown by class")
    bd = pred["breakdown"]
    fig = go.Figure(go.Bar(
        orientation="h",
        x=[b["prob"] * 100 for b in bd],
        y=[b["name"] for b in bd],
        text=[f"{b['prob']*100:.0f}%" for b in bd],
        textposition="outside",
        marker=dict(color=[b["color"] for b in bd]),
    ))
    fig.add_vline(
        x=pred["threshold"] * 100,
        line_dash="dash",
        line_color="#7C8571",
        annotation_text="threshold",
        annotation_position="top right"
    )
    fig.update_layout(
        **CHART_THEME,
        margin=dict(t=10, b=10),
        xaxis_title="Confidence (%)",
        height=max(220, len(bd) * 42),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Recommendation
    st.markdown("#### Recommendation")
    rec_icon = icon_html("healthy" if pred["is_healthy"] else "diseased", size=20)
    st.markdown(
        f"""
        <div class="card">
          {rec_icon} {pred['recommendation']}
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("Re-run", use_container_width=True):
        st.session_state["_disease_pred"] = None
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