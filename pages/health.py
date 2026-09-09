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
    ENV_RANGES,
    DISEASE_MODELS,
    get_trained_crops,
)
from src.dataset_prep import load_class_names
from src.db import insert_analysis
from src.environment_model import predict_environmental_risk
from src.errors import safe_action, logger
from src.health_engine import analyze_crop_health
from src.i18n import get_language, tr_label, tr_crop, tr_disease
from src.recommendation_engine import generate_recommendations, CATEGORY_ICON, PRIORITY_COLOR
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

# Per-factor icon/label/unit metadata lives in config.ENV_RANGES (single
# source of truth, also used by pages/environment.py).
ENV_LABELS = ENV_RANGES


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _disease_classes_for(crop: str) -> list[str]:
    """Return the disease classes for one crop's trained model, in the
    same order as the model's output — cached per crop."""
    return load_class_names(DISEASE_MODELS[crop]["labels_path"])


def render() -> None:
    lang = get_language()
    page_header(
        "health",
        tr_label("Crop Health Analysis", lang),
        tr_label("Combine disease detection and environmental data into an overall health score.", lang),
    )

    trained_crops = get_trained_crops()

    if not trained_crops:
        callout(
            f"{icon_html('warning', size=18)}<b>{tr_label('No trained disease model found.', lang)}</b> "
            + tr_label("Train one first from the Disease Detection page's instructions before "
            "running a health analysis.", lang)
        )
        footer()
        return

    col_in, col_out = st.columns([2, 3])

    # ------------------------------------------------------------- inputs
    with col_in:
        st.markdown(f"#### {tr_label('Inputs', lang)}")

        with st.expander(tr_label("Disease result", lang), expanded=True):
            crop = st.selectbox(tr_label("Crop name", lang), trained_crops, index=0, format_func=lambda c: tr_crop(c, lang))
            disease_classes = _disease_classes_for(crop)
            disease_labels = {
                disease: tr_disease(disease, lang)
                for disease in disease_classes
            }
            disease = st.selectbox(
                tr_label("Detected disease", lang),
                disease_classes,
                format_func=disease_labels.get,
                index=min(1, len(disease_classes) - 1),
            )
            confidence = st.number_input(
                tr_label("Disease confidence", lang),
                min_value=0.0,
                max_value=1.0,
                value=0.82,
                step=0.01,
                format="%.2f",
                help=tr_label("Use the plus and minus buttons to change confidence by 1%.", lang),
            )
            severity = st.select_slider(
                tr_label("Disease severity", lang), options=["None", "Moderate", "High"],
                value="Moderate", format_func=lambda s: tr_label(s, lang),
            )

        with st.expander(tr_label("Environmental readings", lang), expanded=True):
            env = get_dummy_env_readings()
            cols = st.columns(2)
            for col, key in zip(cols * 2, ENV_LABELS):
                with col:
                    spec = ENV_RANGES[key]
                    env[key] = st.number_input(
                        f"{tr_label(spec['label'], lang)} ({spec['unit']})",
                        min_value=float(spec["min"]),
                        max_value=float(spec["max"]),
                        value=float(env[key]),
                        step=spec["nudge"],
                        format="%.1f",
                    )

        compute = st.button(tr_label("Calculate crop health", lang), type="primary",
                            use_container_width=True)

    # ------------------------------------------------------------- results
    with col_out:
        st.markdown(f"#### {tr_label('Analysis result', lang)}")

        results = st.session_state.get("_health_results")

        if compute:
            try:
                with st.spinner("Calculating crop health…"):
                    results = _analyze(crop, disease, confidence, severity, env, lang=lang)
                st.session_state["_health_results"] = results
            except FileNotFoundError:
                callout(
                    f"{icon_html('warning', size=18)}<b>{tr_label('Environmental risk model not found.', lang)}</b> "
                    "Train it first with <code>python -m src.environment_model</code>."
                )
                results = None
            except ValidationError as e:
                st.error(str(e))
                results = None
            except Exception:
                logger.exception("Unexpected error computing crop health")
                st.error(tr_label(
                    "Calculating crop health failed unexpectedly. Please try "
                    "again. If the problem continues, contact the app maintainer.", lang
                ))
                results = None
        elif results is None:
            card(
                tr_label("Awaiting calculation", lang),
                tr_label("Click **Calculate crop health** to combine the disease result "
                "and environmental readings into an overall score and status.", lang),
            )

        if results is not None:
            _render(results)

    footer()


# ---------------------------------------------------------------------------
# Computation — delegates to src.environment_model + src.health_engine
# ---------------------------------------------------------------------------
def _analyze(crop, disease, confidence, severity, env, lang: str = "en") -> dict:
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
    }, lang=lang)

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
        lang=lang,
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
        lang=lang,
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
    lang = get_language()
    r = results

    # --- Top: overall score card + status ------------------------------
    sc1, sc2 = st.columns([2, 3])
    with sc1:
        health_score_card(r["health_score"], label=tr_label("Overall health score", lang))
    with sc2:
        st.markdown(f"#### {tr_label('Overall crop status', lang)}")
        status_color = score_color(r["health_score"])
        st.markdown(
            f"""
            <div style="background:#F7F7F1;border:1px solid #E2E5D8;
                        border-radius:12px;padding:1rem 1.2rem;text-align:center">
              <div style="font-size:.8rem;color:#5B6353;text-transform:uppercase;
                          letter-spacing:.05em">{tr_label('Status', lang)}</div>
              <div style="font-size:1.6rem;font-weight:700;color:{status_color};
                          margin-top:.2rem">{tr_label(r['health_status'], lang)}</div>
              <div style="font-size:.85rem;color:#7C8571">{tr_label('Crop:', lang)} {tr_crop(r['crop'], lang)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(f"{tr_label('Environmental risk model:', lang)} **{r['env_model_used']}**")

    # --- Disease + environmental risk indicators ------------------------
    st.markdown(f"#### {tr_label('Risk breakdown', lang)}")
    rb1, rb2 = st.columns(2)
    with rb1:
        st.markdown(f"**{tr_label('Disease risk', lang)}**")
        risk_indicator(r["disease_risk"]["level"], show_bar=True)
        st.caption(f"{tr_label('Disease score:', lang)} {r['disease_risk']['score']}/100")
    with rb2:
        st.markdown(f"**{tr_label('Environmental risk', lang)}**")
        risk_indicator(r["environmental_risk"]["level"], show_bar=True)
        st.caption(f"{tr_label('Env. score:', lang)} {r['environmental_risk']['score']}/100")

    st.markdown("---")

    # --- All metrics ---------------------------------------------------
    st.markdown(f"#### {tr_label('Detailed metrics', lang)}")
    st.markdown(f"**{tr_label('Disease', lang)}**")
    d1, d2, d3, d4 = st.columns(4)
    dr = r["disease_risk"]
    with d1:
        metric_display(tr_label("Crop name", lang), tr_crop(r["crop"], lang), accent="#2F6D46")
    with d2:
        metric_display(tr_label("Disease", lang), tr_disease(dr["prediction"], lang),
                       accent="#7FA687" if dr["level"] == "Optimal" else "#CE8C82")
    with d3:
        metric_display(tr_label("Confidence", lang), f"{dr['confidence']*100:.0f}%", tr_label("model output", lang))
    with d4:
        metric_display(tr_label("Severity", lang), tr_label(dr["severity"], lang),
                       accent="#B5564B" if dr["severity"] == "High" else
                             "#C97A3B" if dr["severity"] == "Moderate" else "#7FA687")

    st.markdown(f"**{tr_label('Environment', lang)}**")
    e1, e2, e3, e4 = st.columns(4)
    env_keys = ["temperature", "humidity", "soil_moisture", "rainfall"]
    for col, key in zip([e1, e2, e3, e4], env_keys):
        spec = ENV_LABELS[key]
        with col:
            metric_display(tr_label(spec["label"], lang), f"{r['env'][key]} {spec['unit']}")

    # --- Explanation ------------------------------------------------
    st.markdown(f"#### {tr_label('Why this score?', lang)}")
    callout(r["explanation"])

    # --- Recommendations (rule-based engine) ----------------------------
    st.markdown(f"#### {tr_label('Agricultural recommendation', lang)}")
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
    lang = get_language()
    token = r["_analysis_token"]
    saved_token = st.session_state.get("_health_saved_token")

    if saved_token == token:
        saved_id = st.session_state.get("_health_saved_id")
        st.success(f"{tr_label('Analysis saved to database (ID:', lang)} {saved_id}).")
        st.button(f"{tr_label('Saved', lang)} ✓", use_container_width=True, disabled=True)
        return

    if st.button(tr_label("Save Analysis", lang), type="primary", use_container_width=True):
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
    lang = get_language()
    callout(rule_based["summary"])

    if rule_based["priority_actions"]:
        st.markdown(f"**{tr_label('Priority actions', lang)}**")
        for text in rule_based["priority_actions"]:
            st.markdown(f"- {text}")
        st.markdown("")

    items = ""
    for rec in rule_based["recommendations"]:
        icon_tag = icon_html(CATEGORY_ICON.get(rec["category"], "leaf"), size=20, margin_right="0")
        badge_color = PRIORITY_COLOR.get(rec["priority"], "#7FA687")
        category_label = tr_label(rec["category"], lang) if lang == "ta" else rec["category"].title()
        priority_label = tr_label(rec["priority"].upper(), lang) if lang == "ta" else rec["priority"].upper()
        items += (
            "<div class='rec-item'>"
            f"<div class='rec-icon'>{icon_tag}</div>"
            "<div>"
            f"<div class='rec-title'>{category_label} · "
            f"<span style='color:{badge_color}'>{priority_label}</span></div>"
            f"<div class='rec-text'>{rec['text']}</div>"
            f"<div style='font-size:.78rem;color:#7C8571;margin-top:.15rem'>{rec['reason']}</div>"
            "</div></div>"
        )
    st.markdown(items, unsafe_allow_html=True)


if __name__ == "__main__":
    # Standalone entry: minimal Streamlit bootstrap for direct viewing.
    st.set_page_config(page_title="Crop Health Analysis", layout="wide")
    from utils.ui import inject_custom_css, render_sidebar
    inject_custom_css()
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "health"
    render_sidebar()
    st.session_state["current_page"] = "health"
    render()