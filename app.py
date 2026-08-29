"""AI-Based Smart Crop Health Monitoring and Early Disease Detection System.

Entry point and home page of the Streamlit application. A custom in-app
navigation router (driven by a sidebar radio) is used instead of Streamlit's
native multipage file convention so that a single shared styling/layout module
can be applied consistently across every page.
"""

from __future__ import annotations

import streamlit as st

from config import APP_CONFIG, get_trained_crops
from src.db import get_analyses, get_disease_analyses, get_environment_analyses
from src.errors import DatabaseError, logger
from src.outbreak_detection import load_outbreak_signals, get_active_alerts
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
import pages.field_scan as field_scan
import pages.alerts as alerts
import pages.environment as environment
import pages.health as health
import pages.history as history
import pages.about as about


def _render_alert_banner() -> None:
    """A compact 'go look at this' banner for any crop trending worse.

    Reuses src.outbreak_detection end to end (same module the dedicated
    Outbreak Alerts page uses) so Home never computes its own, possibly
    inconsistent, version of the same signal. Failing silently here (rather
    than showing an error) is intentional — this is a bonus surface, not
    the primary place to see alerts, so a hiccup here shouldn't block the
    rest of the home page.
    """
    try:
        signals = load_outbreak_signals(window=7)
        active = get_active_alerts(signals)
    except Exception:
        logger.exception("Unexpected error checking outbreak alerts on home page")
        return

    if not active:
        return

    crop_list = ", ".join(f"<b>{s['crop']}</b> ({s['risk_level']})" for s in active)
    st.markdown(
        f"""
        <div class="callout" style="border-left-color:#B5564B;background:#FBEFED">
          {icon_html('diseased', size=18)}<b>{len(active)} crop(s) trending worse:</b> {crop_list}
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("View Outbreak Alerts →", key="_home_view_alerts"):
        st.session_state["current_page"] = "alerts"
        st.rerun()


def render_home() -> None:
    """Render the landing / home view."""
    page_header(
        APP_CONFIG["page_icon"],
        APP_CONFIG["title"],
        APP_CONFIG["subtitle"],
    )

    trained_crops = get_trained_crops()

    _render_alert_banner()

    # Intro
    callout(
        "**Multi-crop health workflow** · pick a crop, upload a leaf image, "
        "review the trained disease prediction, enter environmental readings, "
        "and calculate an explainable health score."
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
        ("disease", "Detect diseases", "Upload a leaf image for any supported crop; the trained model reports disease, confidence, and severity."),
        ("field_scan", "Scan a field", "Upload 10-20+ leaf photos from a field walk and get one aggregated field health report."),
        ("alerts", "Watch for outbreaks", "See which crops are trending worse across your saved analysis history."),
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

    st.markdown("### Supported crops")
    if trained_crops:
        cols = st.columns(len(trained_crops))
        for col, crop in zip(cols, trained_crops):
            with col:
                icon_tag = icon_html("leaf", size=28, margin_right="0")
                st.markdown(
                    f"""
                    <div class="card" style="text-align:center;height:100%">
                      <div>{icon_tag}</div>
                      <h4 style="color:var(--ink);margin:.4rem 0">{crop}</h4>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    else:
        callout(
            "No trained disease models were found yet. Train one with "
            "`src/model_training.py --crop <name>` to see it listed here."
        )

    st.markdown("### Your activity")
    try:
        n_health = len(get_analyses(limit=100000))
        n_disease = len(get_disease_analyses(limit=100000))
        n_env = len(get_environment_analyses(limit=100000))
        glances = st.columns(3)
        with glances[0]:
            st.metric("Health analyses", n_health)
        with glances[1]:
            st.metric("Disease detections", n_disease)
        with glances[2]:
            st.metric("Environmental readings", n_env)
    except DatabaseError as e:
        st.error(str(e))
    except Exception:
        logger.exception("Unexpected error loading home page activity stats")
        st.error(
            "Loading your activity stats failed unexpectedly. Please try again. "
            "If the problem continues, contact the app maintainer."
        )

    footer()


def main() -> None:
    set_page_config("Home")
    inject_custom_css()
    current = render_sidebar()

    router = {
        "home":        render_home,
        "dashboard":   dashboard.render,
        "disease":     disease.render,
        "field_scan":  field_scan.render,
        "alerts":      alerts.render,
        "environment": environment.render,
        "health":      health.render,
        "history":     history.render,
        "about":       about.render,
    }
    router.get(current, render_home)()


if __name__ == "__main__":
    main()