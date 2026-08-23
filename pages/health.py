"""Crop Health Analysis page — combine disease + environment into a score.

Wired to the real pipeline:
  - src.environment_model.predict_environmental_risk() — trained
    Decision Tree / Random Forest classifier for environmental risk.
  - src.health_engine.analyze_crop_health() — the modular, explainable
    scoring engine that blends disease + environmental signals into the
    final 0-100 health score.
  - src.recommendation_engine.generate_recommendations() — pure rule-based
    engine producing the detailed, explainable action list shown below.
  - src.db.insert_analysis() — persists the complete analysis to SQLite
    when the user clicks "Save Analysis".

The disease side of the form is still manual entry (crop/disease/
confidence/severity) since this page isn't wired to an uploaded image —
that's what the Disease Detection page is for. Everything downstream of
those four fields plus the environmental readings now runs through the
same engines used across the app, instead of page-local rule logic.
"""

from __future__ import annotations

import json
import uuid

import streamlit as st

from config import (
    ENV_CROP_RANGES,
    ENV_RANGES,
)
from src.db import insert_analysis
from src.environment_model import predict_environmental_risk
from src.errors import safe_action, logger
from src.health_engine import analyze_crop_health
from src.recommendation_engine import generate_recommendations
from src.validation import (
    validate_crop,
    validate_confidence,
    validate_severity,
    validate_disease_name,
    validate_environmental_reading,
    ValidationError,
)
from utils.ui import (
    page_header,
    callout,
    card,
    footer,
    get_dummy_env_readings,
    health_score_card,
    risk_indicator,
    metric_display,
    score_color,
)
from utils.icons import icon_html

CROPS = list(ENV_CROP_RANGES.keys())

ENV_LABELS = {
    "temperature":   ("temperature", "Temperature",   "°C"),
    "humidity":      ("humidity",    "Humidity",      "%"),
    "soil_moisture": ("soil",        "Soil moisture", "%"),
    "rainfall":      ("rainfall",    "Rainfall",      "mm"),
}

# Recommendation-engine category -> icon key.
CATEGORY_ICON = {"disease": "diseased", "environment": "temperature", "overall": "leaf"}
# Recommendation-engine priority -> badge color (matches the app's severity palette).
PRIORITY_COLOR = {"high": "#B5564B", "medium": "#C97A3B", "low": "#7FA687"}


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
def render() -> None:
    page_header(
        "health",
        "Crop Health Analysis",
        "Combine disease detection and environmental data into an overall health score.",
    )

    col_in, col_out = st.columns([2, 3])

    # ------------------------------------------------------------- inputs
    with col_in:
        st.markdown("#### Inputs")

        with st.expander("Disease result", expanded=True):
            crop = st.selectbox("Crop name", CROPS, index=0)
            disease = st.selectbox(
                "Detected disease",
                ["Healthy", "Tomato Early Blight", "Tomato Leaf Mold",
                 "Corn Common Rust", "Potato Late Blight"],
                index=1,
            )
            confidence = st.slider("Disease confidence", 0.0, 1.0, 0.82, 0.05)
            severity = st.select_slider(
                "Disease severity", options=["None", "Mild", "Moderate", "High"],
                value="Moderate",
            )

        with st.expander("Environmental readings", expanded=True):
            env = get_dummy_env_readings()
            cols = st.columns(2)
            for col, key in zip(cols * 2, ENV_LABELS):
                with col:
                    _icon, label, unit = ENV_LABELS[key]
                    spec = ENV_RANGES[key]
                    env[key] = st.number_input(
                        f"{label} ({unit})",
                        min_value=float(spec["min"]),
                        max_value=float(spec["max"]),
                        value=float(env[key]),
                        step=2.0 if key in ("temperature", "rainfall") else 5.0,
                        format="%.1f",
                    )

        compute = st.button("Calculate crop health", type="primary",
                            use_container_width=True)

    # ------------------------------------------------------------- results
    with col_out:
        st.markdown("#### Analysis result")

        results = st.session_state.get("_health_results")

        if compute:
            try:
                with st.spinner("Calculating crop health…"):
                    results = _analyze(crop, disease, confidence, severity, env)
                st.session_state["_health_results"] = results
            except FileNotFoundError:
                callout(
                    f"{icon_html('warning', size=18)}<b>Environmental risk model not found.</b> "
                    "Train it first with <code>python -m src.environment_model</code>."
                )
                results = None
            except ValidationError as e:
                st.error(str(e))
                results = None
            except Exception:
                logger.exception("Unexpected error computing crop health")
                st.error(
                    "Calculating crop health failed unexpectedly. Please try "
                    "again. If the problem continues, contact the app maintainer."
                )
                results = None
        elif results is None:
            card(
                "Awaiting calculation",
                "Click **Calculate crop health** to combine the disease result "
                "and environmental readings into an overall score and status.",
            )

        if results is not None:
            _render(results)

    footer()


# ---------------------------------------------------------------------------
# Computation — delegates to src.environment_model + src.health_engine
# ---------------------------------------------------------------------------
def _analyze(crop, disease, confidence, severity, env) -> dict:
    # Validate every input before touching any model — one combined,
    # specific error if anything is out of range or malformed.
    crop = validate_crop(crop)
    disease = validate_disease_name(disease)
    confidence = validate_confidence(confidence)
    severity = validate_severity(severity)
    env = validate_environmental_reading(**env)

    env_pred = predict_environmental_risk({
        "crop": crop,
        "temperature": env["temperature"],
        "humidity": env["humidity"],
        "soil_moisture": env["soil_moisture"],
        "rainfall": env["rainfall"],
    })

    result = analyze_crop_health(
        disease_prediction=disease,
        disease_confidence=confidence,
        disease_severity=severity,
        environmental_risk=env_pred["risk_level"],
        temperature=env["temperature"],
        humidity=env["humidity"],
        soil_moisture=env["soil_moisture"],
        rainfall=env["rainfall"],
        crop=crop,
        environmental_probability=env_pred["probability"],
        environmental_probabilities=env_pred["probabilities"],
        environmental_recommendation=env_pred["recommendation"],
    )

    result["crop"] = crop
    result["env"] = env
    result["env_model_used"] = env_pred["model_used"]

    result["rule_based"] = generate_recommendations(
        crop=crop,
        disease=disease,
        severity=severity,
        temperature=env["temperature"],
        humidity=env["humidity"],
        soil_moisture=env["soil_moisture"],
        rainfall=env["rainfall"],
        health_score=result["health_score"],
    )

    # A fresh, unique token per *computed* analysis (not per rerun). This is
    # what the Save button's duplicate-save guard keys off — see
    # _render_save_section() below.
    result["_analysis_token"] = uuid.uuid4().hex
    return result


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _render(results: dict) -> None:
    r = results

    # --- Top: overall score card + status ------------------------------
    sc1, sc2 = st.columns([2, 3])
    with sc1:
        health_score_card(r["health_score"], label="Overall health score")
    with sc2:
        st.markdown("#### Overall crop status")
        status_color = score_color(r["health_score"])
        st.markdown(
            f"""
            <div style="background:#F7F7F1;border:1px solid #E2E5D8;
                        border-radius:12px;padding:1rem 1.2rem;text-align:center">
              <div style="font-size:.8rem;color:#5B6353;text-transform:uppercase;
                          letter-spacing:.05em">Status</div>
              <div style="font-size:1.6rem;font-weight:700;color:{status_color};
                          margin-top:.2rem">{r['health_status']}</div>
              <div style="font-size:.85rem;color:#7C8571">Crop: {r['crop']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(f"Environmental risk model: **{r['env_model_used']}**")

    # --- Disease + environmental risk indicators ------------------------
    st.markdown("#### Risk breakdown")
    rb1, rb2 = st.columns(2)
    with rb1:
        st.markdown("**Disease risk**")
        risk_indicator(r["disease_risk"]["level"], show_bar=True)
        st.caption(f"Disease score: {r['disease_risk']['score']}/100")
    with rb2:
        st.markdown("**Environmental risk**")
        risk_indicator(r["environmental_risk"]["level"], show_bar=True)
        st.caption(f"Env. score: {r['environmental_risk']['score']}/100")

    st.markdown("---")

    # --- All metrics ---------------------------------------------------
    st.markdown("#### Detailed metrics")
    st.markdown("**Disease**")
    d1, d2, d3, d4 = st.columns(4)
    dr = r["disease_risk"]
    with d1:
        metric_display("Crop name", r["crop"], accent="#2F6D46")
    with d2:
        metric_display("Disease", dr["prediction"],
                       accent="#7FA687" if dr["level"] == "Optimal" else "#CE8C82")
    with d3:
        metric_display("Confidence", f"{dr['confidence']*100:.0f}%", "model output")
    with d4:
        metric_display("Severity", dr["severity"],
                       accent="#B5564B" if dr["severity"] == "High" else
                             "#C97A3B" if dr["severity"] == "Moderate" else "#7FA687")

    st.markdown("**Environment**")
    e1, e2, e3, e4 = st.columns(4)
    env_keys = ["temperature", "humidity", "soil_moisture", "rainfall"]
    for col, key in zip([e1, e2, e3, e4], env_keys):
        _icon, label, unit = ENV_LABELS[key]
        with col:
            metric_display(label, f"{r['env'][key]} {unit}")

    # --- Explanation ------------------------------------------------
    st.markdown("#### Why this score?")
    callout(r["explanation"])

    # --- Recommendations (rule-based engine) ----------------------------
    st.markdown("#### Agricultural recommendation")
    _render_recommendations(r["rule_based"])

    # --- Save to database -------------------------------------------
    st.markdown("---")
    _render_save_section(r)


def _build_db_record(r: dict) -> dict:
    """Map a health-engine result to src.db's `analyses` table columns."""
    dr, er = r["disease_risk"], r["environmental_risk"]
    return {
        "crop_name": r["crop"],
        "disease": dr["prediction"],
        "confidence": dr["confidence"],
        "severity": dr["severity"],
        "temperature": r["env"]["temperature"],
        "humidity": r["env"]["humidity"],
        "soil_moisture": r["env"]["soil_moisture"],
        "rainfall": r["env"]["rainfall"],
        "health_score": r["health_score"],
        "disease_risk": f"{dr['level']} ({dr['score']}/100)",
        "environmental_risk": f"{er['level']} ({er['score']}/100)",
        "recommendation": json.dumps(r["rule_based"]),
        # created_at (timestamp) is filled in automatically by insert_analysis().
    }


def _render_save_section(r: dict) -> None:
    """'Save Analysis' button, guarded against duplicate inserts.

    Streamlit reruns the whole script on every interaction, and this
    `results` dict stays in session_state across reruns — so without a
    guard, a stray rerun or a second click on the same computed analysis
    could insert the same row twice. Each freshly *computed* analysis gets
    a unique `_analysis_token` (see _analyze()); we only allow a save when
    that token hasn't already been recorded as saved, and once saved the
    button is replaced with a confirmation instead of staying clickable.
    Recomputing (even with identical inputs) mints a new token, so
    intentionally logging the same reading again later is still allowed.
    """
    token = r["_analysis_token"]
    saved_token = st.session_state.get("_health_saved_token")

    if saved_token == token:
        saved_id = st.session_state.get("_health_saved_id")
        st.success(f"Analysis saved to database (ID: {saved_id}).")
        st.button("Saved ✓", use_container_width=True, disabled=True)
        return

    if st.button("Save Analysis", type="primary", use_container_width=True):
        with safe_action("Saving analysis"):
            with st.spinner("Saving analysis…"):
                record = _build_db_record(r)
                analysis_id = insert_analysis(record)
            st.session_state["_health_saved_token"] = token
            st.session_state["_health_saved_id"] = analysis_id
            st.rerun()


def _render_recommendations(rule_based: dict) -> None:
    """Render the rule-based recommendation engine's output: a summary,
    any priority actions, then the full explainable list (text + why).
    """
    callout(rule_based["summary"])

    if rule_based["priority_actions"]:
        st.markdown("**Priority actions**")
        for text in rule_based["priority_actions"]:
            st.markdown(f"- {text}")
        st.markdown("")

    items = ""
    for rec in rule_based["recommendations"]:
        icon_tag = icon_html(CATEGORY_ICON.get(rec["category"], "leaf"), size=20, margin_right="0")
        badge_color = PRIORITY_COLOR.get(rec["priority"], "#7FA687")
        items += (
            "<div class='rec-item'>"
            f"<div class='rec-icon'>{icon_tag}</div>"
            "<div>"
            f"<div class='rec-title'>{rec['category'].title()} · "
            f"<span style='color:{badge_color}'>{rec['priority'].upper()}</span></div>"
            f"<div class='rec-text'>{rec['text']}</div>"
            f"<div style='font-size:.78rem;color:#7C8571;margin-top:.15rem'>{rec['reason']}</div>"
            "</div></div>"
        )
    st.markdown(items, unsafe_allow_html=True)


if __name__ == "__main__":
    # Standalone entry: minimal Streamlit bootstrap for direct viewing.
    import streamlit as st  # noqa: F811
    st.set_page_config(page_title="Crop Health Analysis", layout="wide")
    from utils.ui import inject_custom_css, render_sidebar
    inject_custom_css()
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "health"
    render_sidebar()
    st.session_state["current_page"] = "health"
    render()