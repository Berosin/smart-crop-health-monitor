"""AI-Based Smart Crop Health Monitoring and Early Disease Detection System.

Entry point and home page of the Streamlit application. A custom in-app
navigation router (driven by a sidebar radio) is used instead of Streamlit's
native multipage file convention so that a single shared styling/layout module
can be applied consistently across every page.
"""

from __future__ import annotations

import streamlit as st

from config import APP_CONFIG
from utils.ui import (
    inject_custom_css,
    set_page_config,
    page_header,
    footer,
    render_sidebar,
    callout,
)
from utils.icons import icon_html

# Page renderers (imported lazily inside the router to keep startup light)
import pages.dashboard as dashboard
import pages.disease as disease
import pages.environment as environment
import pages.health as health
import pages.history as history
import pages.about as about


def render_home() -> None:
    """Render the landing / home view."""
    page_header(
        APP_CONFIG["page_icon"],
        APP_CONFIG["title"],
        APP_CONFIG["subtitle"],
    )

    # Intro
    callout(
        "**Tomato health workflow** · upload a leaf image, review the trained "
        "disease prediction, enter environmental readings, and calculate an "
        "explainable health score."
    )
    st.markdown(
        """
        Monitor crop health and detect plant diseases **early** using AI-driven
        image analysis combined with environmental data — entirely in software,
        with no sensors or cameras required.
        """
    )

    st.markdown("### What you can do here")
    feats = [
        ("disease", "Detect diseases", "Upload a tomato leaf image; the trained model reports disease, confidence, and severity."),
        ("environment", "Track conditions", "Log temperature, humidity, soil moisture and rainfall."),
        ("health", "Score crop health", "Combine the image and environmental signals into one health score."),
        ("dashboard", "Visualize trends", "See stats and charts across all your past analyses."),
        ("history", "Keep history", "Save completed analyses and review them later."),
    ]
    cols = st.columns(len(feats))
    for col, (icon, title, desc) in zip(cols, feats):
        with col:
            icon_tag = icon_html(icon, size=32, margin_right="0")
            st.markdown(
                f"""
                <div class="card" style="text-align:center;height:100%">
                  <div>{icon_tag}</div>
                  <h4 style="color:var(--ink);margin:.4rem 0">{title}</h4>
                  <div style="font-size:.85rem;color:#4E5646">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("### Where to start")
    st.markdown(
        """
        Use the **sidebar on the left** to navigate between modules. A suggested
        flow: **Disease Detection → Environmental Analysis → Crop Health
        Analysis → Dashboard**.
        """
    )

    st.markdown("### Current coverage")
    glances = st.columns(4)
    with glances[0]:
        st.metric("Modules", "6")
    with glances[1]:
        st.metric("Supported crops", "1")
    with glances[2]:
        st.metric("Disease classes", "3")
    with glances[3]:
        st.metric("Analysis storage", "SQLite")

    footer()


def main() -> None:
    set_page_config("Home")
    inject_custom_css()
    current = render_sidebar()

    router = {
        "home":        render_home,
        "dashboard":   dashboard.render,
        "disease":     disease.render,
        "environment": environment.render,
        "health":      health.render,
        "history":     history.render,
        "about":       about.render,
    }
    router.get(current, render_home)()


if __name__ == "__main__":
    main()
