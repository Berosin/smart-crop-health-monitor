"""About Project page — overview, current capabilities, and model scope."""

from __future__ import annotations

import json

import streamlit as st

from config import APP_CONFIG, LABELS_PATH
from utils.ui import page_header, card, callout, footer
from utils.icons import icon_html


def render() -> None:
    page_header("about", "About Project", APP_CONFIG["subtitle"])

    st.markdown(
        """
        **Smart Crop Health Monitoring** is a software-only Streamlit
        application for checking tomato crop health. It combines trained
        image classification, environmental risk analysis, an explainable
        health score, and practical agricultural recommendations in one
        workflow. No sensors or IoT hardware are required.
        """
    )

    st.markdown("#### Current workflow")
    steps = [
        ("1", "Upload a tomato leaf image", "Submit a clear JPG or PNG image for analysis."),
        ("2", "Classify the leaf", "The trained model recognizes Healthy, Early Blight, or Late Blight."),
        ("3", "Review confidence", "The prediction includes class probabilities and confidence."),
        ("4", "Enter environmental readings", "Provide temperature, humidity, soil moisture, and rainfall."),
        ("5", "Assess environmental risk", "The trained environmental model classifies the current conditions."),
        ("6", "Calculate crop health", "Disease and environmental signals become one explainable score out of 100."),
        ("7", "Review recommendations", "Severity-aware, crop-specific actions explain what to do next."),
        ("8", "Save and review analyses", "Save completed assessments to SQLite and inspect them in History and Dashboard."),
    ]
    for n, title, desc in steps:
        st.markdown(
            f"""
            <div class="card" style="margin-bottom:.6rem">
              <div style="display:flex;gap:.8rem;align-items:center">
                <div style="background:var(--leaf);color:#fff;border-radius:50%;
                            width:28px;height:28px;display:flex;align-items:center;
                            justify-content:center;font-weight:700">{n}</div>
                <div>
                  <b style="color:var(--ink)">{title}</b>
                  <div style="font-size:.85rem;color:#4E5646">{desc}</div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    st.markdown("#### Technology")
    stack = [
        ("Frontend / UI", "Streamlit"),
        ("Programming", "Python"),
        ("AI / Deep Learning", "TensorFlow / Keras"),
        ("Machine Learning", "Scikit-learn"),
        ("Image Processing", "OpenCV"),
        ("Data Processing", "Pandas + NumPy"),
        ("Database", "SQLite"),
        ("Visualization", "Plotly"),
    ]
    cols = st.columns(4)
    for col, (layer, tech) in zip(cols, stack):
        with col:
            card(layer, f"<code>{tech}</code>")

    st.markdown("---")

    st.markdown("#### Current model scope")
    with open(LABELS_PATH, "r", encoding="utf-8") as labels_file:
        label_map = json.load(labels_file)
    disease_classes = [
        name.replace("_", " ")
        for name, _ in sorted(label_map.items(), key=lambda item: item[1])
    ]
    scope_cols = st.columns(3)
    with scope_cols[0]:
        card("Supported crop", "<b>Tomato</b>")
    with scope_cols[1]:
        card("Disease classes", f"<b>{', '.join(disease_classes)}</b>")
    with scope_cols[2]:
        card("Health result", "<b>0-100 score + risk breakdown</b>")

    callout(
        f"{icon_html('info', size=18)}The disease model is currently trained for tomato leaves only. "
        "Additional crops can be added after their own labeled data and model classes are trained."
    )
    footer()
