"""About Project page — overview, current capabilities, and model scope."""

from __future__ import annotations

import streamlit as st

from config import APP_CONFIG, DISEASE_MODELS, get_trained_crops
from src.dataset_prep import load_class_names
from utils.ui import page_header, card, callout, footer
from utils.icons import icon_html


def render() -> None:
    page_header("about", "About Project", APP_CONFIG["subtitle"])

    st.markdown(
        """
        **Smart Crop Health Monitoring** is a software-only Streamlit
        application for checking crop health. It combines trained
        image classification, environmental risk analysis, an explainable
        health score, and practical agricultural recommendations in one
        workflow. No sensors or IoT hardware are required.
        """
    )

    st.markdown("#### Current workflow")
    steps = [
        ("1", "Pick a crop and upload a leaf image", "Choose a trained crop and submit a clear JPG or PNG image for analysis."),
        ("2", "Classify the leaf", "That crop's trained model recognizes Healthy, Early Blight, or Late Blight."),
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
    trained_crops = get_trained_crops()

    if not trained_crops:
        callout(f"{icon_html('warning', size=18)}No trained disease model found yet.")
    else:
        scope_cols = st.columns(len(trained_crops))
        for col, crop in zip(scope_cols, trained_crops):
            disease_classes = [
                name.replace("_", " ")
                for name in load_class_names(DISEASE_MODELS[crop]["labels_path"])
            ]
            with col:
                card(crop, f"<b>{', '.join(disease_classes)}</b>")

    all_crops = list(DISEASE_MODELS.keys())
    untrained = [c for c in all_crops if c not in trained_crops]
    if untrained:
        callout(
            f"{icon_html('info', size=18)}Currently trained: <b>{', '.join(trained_crops) or 'none'}</b>. "
            f"<b>{', '.join(untrained)}</b> can be added by training a model for that crop — "
            "see the Disease Detection page for the training command."
        )
    footer()