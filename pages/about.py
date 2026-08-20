"""About Project page — overview, tech stack, and roadmap."""

from __future__ import annotations

import streamlit as st

from config import APP_CONFIG
from utils.ui import page_header, card, callout, footer


def render() -> None:
    page_header("ℹ️", "About Project", APP_CONFIG["subtitle"])

    st.markdown(
        """
        The **AI-Based Smart Crop Health Monitoring and Early Disease
        Detection System** is a *software-only* platform that helps farmers
        and agriculturalists monitor crop health and detect plant diseases
        early — combining deep-learning image analysis with environmental
        data. **No hardware** (no Arduino, ESP32, Raspberry Pi, sensors,
        cameras, or IoT devices) is involved.
        """
    )

    st.markdown("#### What the system does")
    steps = [
        ("1", "Accept crop leaf images", "User uploads a clear leaf photo."),
        ("2", "Detect diseases with AI", "A deep-learning model scores each disease class."),
        ("3", "Show confidence & severity", "Detection confidence and severity are reported."),
        ("4", "Accept environmental conditions", "Temperature, humidity, soil moisture, rainfall."),
        ("5", "Analyze crop / environmental health", "Combined signal from image + environment."),
        ("6", "Calculate overall health score", "One transparent score out of 100."),
        ("7", "Provide recommendations", "Actionable agricultural guidance."),
        ("8", "Store analysis history", "Every analysis is saved in SQLite."),
        ("9", "Dashboard statistics & charts", "Trends visualized with Plotly."),
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
                  <div style="font-size:.85rem;color:#555">{desc}</div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    st.markdown("#### Tech stack")
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

    st.markdown("#### Project status")
    done, todo = st.columns([1, 1])
    with done:
        st.markdown("**Completed**")
        for t in ["Project structure", "Streamlit layout + navigation",
                  "Agriculture-themed UI", "Reusable UI functions",
                  "Dummy data for all pages"]:
            st.markdown(f"- ✅ {t}")
    with todo:
        st.markdown("**Not yet implemented**")
        for t in ["AI disease detection model", "SQLite persistence layer",
                  "Real health-score logic", "Recommendation engine",
                  "Dashboard wired to live data"]:
            st.markdown(f"- ⬜ {t}")

    callout("ℹ️ This is a layout-only build. AI and database functionality are "
            "intentionally deferred.")
    footer()
