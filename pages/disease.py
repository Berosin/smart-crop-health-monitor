"""Disease Detection page — upload a leaf image and run detection.

This is a UI scaffold: the TensorFlow model is not yet wired up. When the
user uploads an image and clicks Analyze, a *mock* prediction is produced so
the full layout (preview, loading, result, recommendation) can be reviewed.
"""

from __future__ import annotations

import time

import plotly.graph_objects as go
import streamlit as st

from config import CONFIDENCE_THRESHOLD
from utils.ui import (
    page_header,
    callout,
    card,
    footer,
    metric_tile,
)

# ---------------------------------------------------------------------------
# Mock disease catalog keyed by crop
# ---------------------------------------------------------------------------
# Each entry: (class name, display color, default severity, recommendation)
DISEASE_CATALOG: dict[str, list[dict]] = {
    "Tomato": [
        {"name": "Healthy",            "color": "#66bb6a", "severity": "None",
         "rec": "Crop looks healthy. Maintain regular monitoring and balanced irrigation."},
        {"name": "Tomato Early Blight", "color": "#ef5350", "severity": "Moderate",
         "rec": "Remove infected leaves, apply copper-based fungicide, and improve air circulation between plants."},
        {"name": "Tomato Leaf Mold",    "color": "#ba68c8", "severity": "Mild",
         "rec": "Reduce leaf wetness by avoiding overhead watering; apply chlorothalonil if humidity stays high."},
        {"name": "Tomato Late Blight",  "color": "#e53935", "severity": "High",
         "rec": "Urgent: destroy affected plants, apply mancozeb, and avoid working in the field while wet."},
    ],
    "Corn": [
        {"name": "Healthy",            "color": "#66bb6a", "severity": "None",
         "rec": "Crop looks healthy. Keep up scheduled scouting and soil testing."},
        {"name": "Corn Common Rust",   "color": "#ff8a65", "severity": "Mild",
         "rec": "Plant resistant hybrids next season; apply fungicide if infection exceeds 15% leaf area."},
        {"name": "Corn Gray Leaf Spot", "color": "#a1887f", "severity": "Moderate",
         "rec": "Rotate crops, bury crop residue, and apply a strobilurin fungicide during early silking."},
        {"name": "Corn Northern Blight", "color": "#8d6e63", "severity": "High",
         "rec": "Apply foliar fungicide immediately; remove severely infected plants to limit spread."},
    ],
    "Potato": [
        {"name": "Healthy",            "color": "#66bb6a", "severity": "None",
         "rec": "Crop looks healthy. Monitor soil moisture and avoid waterlogging."},
        {"name": "Potato Late Blight",  "color": "#f06292", "severity": "High",
         "rec": "Urgent: destroy infected foliage, apply metalaxyl + mancozeb, and ensure good drainage."},
        {"name": "Potato Early Blight", "color": "#ffb74d", "severity": "Moderate",
         "rec": "Remove lower infected leaves and apply azoxystrobin; rotate away from solanaceous crops."},
    ],
    "Rice": [
        {"name": "Healthy",            "color": "#66bb6a", "severity": "None",
         "rec": "Crop looks healthy. Maintain water level and balanced nitrogen."},
        {"name": "Rice Blast",          "color": "#ec407a", "severity": "Moderate",
         "rec": "Apply tricyclazole at booting stage; avoid excess nitrogen and keep field watered."},
        {"name": "Brown Spot",          "color": "#a1887f", "severity": "Mild",
         "rec": "Improve soil fertility with potash; treat seed with fungicide before sowing."},
    ],
    "Wheat": [
        {"name": "Healthy",            "color": "#66bb6a", "severity": "None",
         "rec": "Crop looks healthy. Continue regular scouting for rust."},
        {"name": "Wheat Leaf Rust",     "color": "#ff7043", "severity": "Moderate",
         "rec": "Apply propiconazole at flag-leaf stage; plant resistant varieties next cycle."},
        {"name": "Wheat Septoria",      "color": "#7e57c2", "severity": "Mild",
         "rec": "Apply a triazole fungicide at flowering; rotate crops and bury stubble."},
    ],
}

CROPS = list(DISEASE_CATALOG.keys())
SEVERITY_META = {
    "None":     ("#66bb6a", "No action needed"),
    "Mild":     ("#fdd835", "Monitor closely"),
    "Moderate": ("#ff9800", "Treat promptly"),
    "High":     ("#ef5350", "Intervene urgently"),
}


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
def render() -> None:
    page_header(
        "🦠",
        "Disease Detection",
        "Upload a crop leaf image to detect diseases with AI.",
    )

    callout(
        "**Mock prediction mode** — the TensorFlow model isn't implemented yet. "
        "Results below are simulated so you can review the full workflow."
    )

    col_input, col_result = st.columns([2, 3])

    # ------------------------------------------------------------------ inputs
    with col_input:
        st.markdown("#### 1 · Select crop & upload image")
        crop = st.selectbox("Crop type", CROPS)

        uploaded = st.file_uploader(
            "Leaf image (JPG / PNG)",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )

        if uploaded is not None:
            st.image(uploaded, caption=f"Uploaded {crop} leaf",
                     use_container_width=True)
        else:
            st.info("📷 Drop a clear, well-lit photo of a single leaf here.")

        with st.expander("Advanced options"):
            threshold = st.slider(
                "Confidence threshold", 0.0, 1.0,
                float(CONFIDENCE_THRESHOLD), 0.05,
            )
            preprocess = st.checkbox(
                "Preprocess image (resize / normalize)", value=True,
            )

        analyze = st.button(
            "🔍 Analyze", type="primary", use_container_width=True,
            disabled=(uploaded is None),
        )
        if analyze and uploaded is None:
            st.warning("Please upload a leaf image first.")

    # ----------------------------------------------------------------- results
    with col_result:
        st.markdown("#### 2 · Prediction result")

        # Session state holds the last prediction so it survives reruns
        # caused by widget interaction (e.g. threshold slider).
        pred = st.session_state.get("_disease_pred")

        if analyze and uploaded is not None:
            with st.spinner("Preprocessing image and running detection…"):
                time.sleep(2.0)  # simulate inference latency
            pred = _mock_predict(crop, threshold, preprocess)
            st.session_state["_disease_pred"] = pred
            st.session_state["_disease_crop"] = crop

        if pred is None:
            card(
                "Awaiting analysis",
                "Upload an image and click **Analyze** to see the prediction, "
                "confidence, severity, and recommendation.",
            )
        else:
            _render_result(pred, threshold)

    footer()


# ---------------------------------------------------------------------------
# Mock prediction
# ---------------------------------------------------------------------------
def _mock_predict(crop: str, threshold: float, preprocess: bool) -> dict:
    """Produce a deterministic mock prediction for the given crop.

    A small fixed offset means repeated analyses of the same crop yield a
    stable result (better for reviewing the UI) while still varying by crop.
    """
    classes = DISEASE_CATALOG[crop]
    # Deterministic pick that favors a non-healthy class so the full layout
    # (severity, recommendation) is exercised, rotating by crop index.
    import hashlib
    seed = int(hashlib.md5(crop.encode()).hexdigest(), 16)
    top_idx = (seed % (len(classes) - 1)) + 1  # never the very first index
    # ensure Healthy is reachable sometimes
    if seed % 5 == 0:
        top_idx = 0

    top = classes[top_idx]
    confidence = 0.62 + (seed % 30) / 100.0   # 0.62 - 0.91
    confidence = round(confidence, 3)

    # Build confidence spread across the rest, summing to ~1.0.
    remaining = round(1.0 - confidence, 3)
    others = [c for i, c in enumerate(classes) if i != top_idx]

    # Assign the remaining probability across the other classes with a
    # geometric decay so the spread looks like a real softmax output.
    breakdown = []
    weights = []
    w = 1.0
    for _ in others:
        weights.append(w)
        w *= 0.5
    wsum = sum(weights) or 1.0
    for i, c in enumerate(classes):
        if i == top_idx:
            p = confidence
        else:
            j = others.index(c)
            p = round(remaining * weights[j] / wsum, 3)
        breakdown.append({"name": c["name"], "color": c["color"], "prob": p})

    # Normalize away any tiny floating drift so probabilities sum to 1.0.
    total = sum(b["prob"] for b in breakdown)
    for b in breakdown:
        b["prob"] = round(b["prob"] / total, 3)

    is_healthy = top["name"] == "Healthy"
    return {
        "crop": crop,
        "disease": top["name"],
        "color": top["color"],
        "confidence": max(b["prob"] for b in breakdown if b["name"] == top["name"]),
        "severity": top["severity"],
        "recommendation": top["rec"],
        "threshold": threshold,
        "preprocess": preprocess,
        "breakdown": breakdown,
        "is_healthy": is_healthy,
        "low_confidence": max(b["prob"] for b in breakdown) < threshold,
    }


# ---------------------------------------------------------------------------
# Result rendering
# ---------------------------------------------------------------------------
def _render_result(pred: dict, threshold: float) -> None:
    """Render the full prediction result block."""
    # Banner
    sev_color, sev_action = SEVERITY_META[pred["severity"]]
    banner_bg = "#e8f5e9" if pred["is_healthy"] else "#fff3e0"
    banner_border = "#66bb6a" if pred["is_healthy"] else sev_color
    st.markdown(
        f"""
        <div style="background:{banner_bg};border-left:5px solid {banner_border};
                    border-radius:12px;padding:1rem 1.25rem;margin-bottom:1rem">
          <div style="font-size:.8rem;color:#555;text-transform:uppercase;
                      letter-spacing:.04em">Detected condition</div>
          <div style="font-size:1.5rem;font-weight:700;color:{pred['color']}">
            {pred['disease']}
          </div>
          <div style="font-size:.85rem;color:#666">{pred['crop']} leaf ·
            {sev_action}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # KPI tiles
    c1, c2, c3 = st.columns(3)
    with c1:
        metric_tile("Confidence", f"{pred['confidence']*100:.0f}%",
                    "model output")
    with c2:
        metric_tile("Severity", pred["severity"], sev_action)
    with c3:
        metric_tile("Threshold", f"{pred['threshold']*100:.0f}%",
                    "cutoff for reliable result")

    # Low-confidence warning
    if pred["low_confidence"] and not pred["is_healthy"]:
        callout(
            "⚠️ Confidence is below the threshold. The result may be uncertain — "
            "consider retaking the photo with better lighting/focus."
        )

    # Confidence breakdown bar chart
    st.markdown("#### Confidence breakdown by class")
    bd = sorted(pred["breakdown"], key=lambda b: b["prob"])
    fig = go.Figure(go.Bar(
        orientation="h",
        x=[b["prob"] * 100 for b in bd],
        y=[b["name"] for b in bd],
        text=[f"{b['prob']*100:.0f}%" for b in bd],
        textposition="outside",
        marker=dict(color=[b["color"] for b in bd]),
    ))
    fig.add_vline(x=threshold * 100, line_dash="dash", line_color="#888",
                  annotation_text="threshold", annotation_position="top right")
    fig.update_layout(
        template="plotly_white",
        margin=dict(t=10, b=10),
        xaxis_title="Confidence (%)",
        height=max(220, len(bd) * 42),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Recommendation
    st.markdown("#### Recommendation")
    icon = "✅" if pred["is_healthy"] else "🌾"
    st.markdown(
        f"""
        <div class="card">
          {icon} {pred['recommendation']}
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("💾 Save analysis", use_container_width=True):
            st.info("SQLite storage not yet connected — this is mock mode.")
    with col_b:
        if st.button("🔄 Re-run", use_container_width=True):
            st.session_state["_disease_pred"] = None
            st.rerun()
