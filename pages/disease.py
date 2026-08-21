"""Disease Detection page — upload a leaf image and run detection.

Uses the trained TensorFlow/Keras MobileNetV2 model for inference.
"""

from __future__ import annotations

import io
import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf

from config import CONFIDENCE_THRESHOLD, IMAGE_SIZE, IMAGE_CHANNELS, MODEL_PATH, LABELS_PATH
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
# The model was trained on generic classes: Healthy, Early_Blight, Late_Blight
# ---------------------------------------------------------------------------
MODEL_CLASSES = ["Early_Blight", "Healthy", "Late_Blight"]

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
def load_model():
    """Load the trained Keras model and class labels."""
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        # Load label map
        import json
        with open(LABELS_PATH, "r") as f:
            label_map = json.load(f)
        # Convert to list in correct order
        inv_map = {v: k for k, v in label_map.items()}
        class_names = [inv_map[i] for i in range(len(inv_map))]
        return model, class_names
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None, None


# ---------------------------------------------------------------------------
# Image preprocessing (OpenCV-style with TensorFlow)
# ---------------------------------------------------------------------------
def preprocess_image(uploaded_file, target_size=IMAGE_SIZE):
    """Preprocess uploaded image for model inference.

    Steps:
    1. Read bytes
    2. Decode with TensorFlow (handles JPG/PNG)
    3. Resize to target_size
    4. Apply MobileNetV2 preprocessing (scales to [-1, 1])
    5. Add batch dimension
    """
    try:
        # Read file bytes
        file_bytes = uploaded_file.read()
        uploaded_file.seek(0)  # reset for potential re-use

        # Decode image
        img = tf.io.decode_image(file_bytes, channels=IMAGE_CHANNELS, expand_animations=False)

        # Resize
        img = tf.image.resize(img, target_size, method="bilinear")

        # MobileNetV2 preprocessing: scales pixels to [-1, 1]
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)

        # Add batch dimension
        img = tf.expand_dims(img, axis=0)

        return img.numpy()
    except Exception as e:
        raise ValueError(f"Image preprocessing failed: {e}")


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
        raise RuntimeError(f"Prediction failed: {e}")


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
def render() -> None:
    page_header(
        "disease",
        "Disease Detection",
        "Upload a crop leaf image to detect diseases with AI.",
    )

    # Load model
    model, class_names = load_model()

    if model is None:
        callout(
            f"{icon_html('warning', size=18)}<b>Model not found.</b> Train the model first using "
            "<code>python -m src.model_training --data-dir data/samples</code> "
            "or place a trained model at <code>models/crop_disease_model.h5</code>."
        )
        footer()
        return

    callout(
        f"{icon_html('success', size=18)}<b>Model loaded</b> — {model.name} with "
        f"{len(class_names)} classes: {', '.join(class_names)}"
    )

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

        analyze = st.button(
            "Analyze", type="primary", use_container_width=True,
            disabled=(uploaded is None),
        )

    # ----------------------------------------------------------------- results
    with col_result:
        st.markdown("#### 2 · Prediction result")

        # Session state holds the last prediction
        pred = st.session_state.get("_disease_pred")

        if analyze and uploaded is not None:
            try:
                # Preprocess with loading indicator
                with st.spinner("Preprocessing image…"):
                    image_batch = preprocess_image(uploaded)

                # Run inference with loading indicator
                with st.spinner("Running disease detection…"):
                    pred = predict_disease(model, class_names, image_batch, threshold)

                st.session_state["_disease_pred"] = pred
                st.rerun()

            except ValueError as e:
                st.error(f"{e}")
            except RuntimeError as e:
                st.error(f"{e}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")

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

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Save analysis", use_container_width=True):
            _save_analysis(pred)
    with col_b:
        if st.button("Re-run", use_container_width=True):
            st.session_state["_disease_pred"] = None
            st.rerun()


def _save_analysis(pred: dict) -> None:
    """Save analysis to SQLite database."""
    try:
        from src.db import insert_analysis
        analysis_id = insert_analysis({
            "crop_name": "Unknown",  # Model doesn't predict crop type
            "disease": pred["disease"],
            "confidence": pred["confidence"],
            "severity": pred["severity"],
            "health_score": int(pred["confidence"] * 100) if pred["is_healthy"] else int((1 - pred["confidence"]) * 100),
            "disease_risk": pred["severity"],
            "recommendation": pred["recommendation"],
        })
        st.success(f"Analysis saved to database (ID: {analysis_id})")
    except Exception as e:
        st.error(f"Failed to save: {e}")


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