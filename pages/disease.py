"""Disease Detection page — upload a leaf image and run detection.

This is a UI scaffold: the AI model is not yet wired up. When a user uploads
an image we show a simulated confidence breakdown so the layout is reviewable.
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from utils.ui import (
    page_header,
    callout,
    card,
    footer,
    get_dummy_diseases,
)


def render() -> None:
    page_header(
        "🦠",
        "Disease Detection",
        "Upload a crop leaf image to detect diseases with AI.",
    )

    col_upload, col_result = st.columns([2, 3])

    with col_upload:
        card("Step 1 — Upload image",
             "Provide a clear, well-lit photo of a single leaf.")
        uploaded = st.file_uploader(
            "Drop a leaf image (JPG/PNG)",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )
        if uploaded is not None:
            st.image(uploaded, caption="Uploaded leaf", use_container_width=True)

        with st.expander("Detection options"):
            st.checkbox("Preprocess image (resize / normalize)", value=True)
            st.slider("Confidence threshold", 0.0, 1.0, 0.70, 0.05)
            st.selectbox("Target crop", ["Tomato", "Corn", "Potato", "Rice", "Wheat"])

        run = st.button("🔍 Run disease detection", type="primary",
                        use_container_width=True)

    with col_result:
        card("Step 2 — Detection results",
             "The model's confidence per disease class appears here.")
        if uploaded is None or not run:
            callout("Upload an image and click **Run disease detection** to see "
                    "results. (AI not yet implemented — results are simulated.)")
        else:
            _render_simulation()

    footer()


def _render_simulation() -> None:
    import pandas as pd
    from utils.ui import metric_tile, get_dummy_diseases

    diseases = get_dummy_diseases()
    # Fabricate a plausible simulated result
    diseases[0]["confidence"] = 0.82
    diseases[1]["confidence"] = 0.11
    diseases[2]["confidence"] = 0.04
    diseases[3]["confidence"] = 0.02
    diseases[4]["confidence"] = 0.01
    diseases[0]["severity"] = "Moderate"

    top = max(diseases, key=lambda d: d["confidence"])
    c1, c2, c3 = st.columns(3)
    with c1:
        metric_tile("Detected", top["name"], f"{top['crop']}")
    with c2:
        metric_tile("Confidence", f"{top['confidence']*100:.0f}%", "simulated")
    with c3:
        metric_tile("Severity", top["severity"], "needs treatment")

    st.markdown("#### Confidence by class")
    df = pd.DataFrame(diseases).sort_values("confidence", ascending=True)
    fig = go.Figure(go.Bar(
        orientation="h",
        x=df["confidence"] * 100,
        y=df["name"],
        text=[f"{v:.0f}%" for v in df["confidence"] * 100],
        textposition="outside",
        marker=dict(color=df["color"].tolist()),
    ))
    fig.update_layout(template="plotly_white", margin=dict(t=10, b=10),
                      xaxis_title="Confidence (%)", height=260)
    st.plotly_chart(fig, use_container_width=True)
