"""Crop Health Analysis page — combine disease + environment into a score.

Uses dummy data and transparent rule-based weighting. The ML model and
database are intentionally not connected yet. All display widgets are built
from the four reusable functions in utils/ui.py:
  health_score_card, risk_indicator, metric_display, recommendation_display.
"""

from __future__ import annotations

import streamlit as st

from config import (
    ENV_CROP_RANGES,
    ENV_RANGES,
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
    recommendation_display,
    score_color,
)

CROPS = list(ENV_CROP_RANGES.keys())
SEVERITY_WEIGHT = {           # severity -> impact on disease-risk weight
    "None": 0.0, "Mild": 0.3, "Moderate": 0.6, "High": 0.9,
}

ENV_LABELS = {
    "temperature":   ("🌡️", "Temperature",   "°C"),
    "humidity":      ("💧", "Humidity",      "%"),
    "soil_moisture": ("🌱", "Soil moisture", "%"),
    "rainfall":      ("🌧️", "Rainfall",      "mm"),
}


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
def render() -> None:
    page_header(
        "🌱",
        "Crop Health Analysis",
        "Combine disease detection and environmental data into an overall health score.",
    )

    callout(
        "**Dummy data · no ML / DB yet.** All metrics below are computed with a "
        "transparent rule-based formula so the full layout can be reviewed."
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
                    emoji, label, unit = ENV_LABELS[key]
                    spec = ENV_RANGES[key]
                    env[key] = st.number_input(
                        f"{emoji} {label} ({unit})",
                        min_value=float(spec["min"]),
                        max_value=float(spec["max"]),
                        value=float(env[key]),
                        step=2.0 if key in ("temperature", "rainfall") else 5.0,
                        format="%.1f",
                    )

        compute = st.button("🌱 Calculate crop health", type="primary",
                            use_container_width=True)

    # ------------------------------------------------------------- results
    with col_out:
        st.markdown("#### Analysis result")

        results = st.session_state.get("_health_results")
        trigger = compute or st.session_state.get("_health_dirty", False)

        if compute or (results is not None and trigger):
            results = _analyze(crop, disease, confidence, severity, env)
            st.session_state["_health_results"] = results
            st.session_state["_health_dirty"] = True
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
# Computation (rule-based, transparent)
# ---------------------------------------------------------------------------
def _analyze(crop, disease, confidence, severity, env) -> dict:
    # --- Disease risk ---------------------------------------------------
    sev_w = SEVERITY_WEIGHT.get(severity, 0.6)
    conf_w = confidence                  # higher confidence in a disease = worse
    is_healthy = (disease == "Healthy") or severity == "None"
    if is_healthy:
        disease_risk_score = 100        # 100 = low risk
    else:
        # disease risk score (100 = none, 0 = severe): penalise severity & confidence
        disease_risk_score = round(100 - 60 * sev_w - 30 * conf_w)
    disease_risk_score = max(0, min(100, disease_risk_score))
    disease_risk_level = _score_to_risk(disease_risk_score)

    # --- Environmental risk --------------------------------------------
    env_score = _env_score(crop, env)   # 0-100, 100 = all ideal
    env_risk_level = _score_to_risk(env_score)

    # --- Overall health score -----------------------------------------
    # Weights: disease 55%, environment 45%.
    overall = round(0.55 * disease_risk_score + 0.45 * env_score)
    overall = max(0, min(100, overall))

    status = ("Healthy" if overall >= 80 else "Monitor" if overall >= 60 else
              "At risk" if overall >= 40 else "Critical")

    recs = _recommendations(disease, severity, confidence, env_score, crop,
                            env_risk_level, is_healthy)

    return {
        "crop": crop,
        "disease": disease,
        "confidence": confidence,
        "severity": severity,
        "env": env,
        "disease_risk_score": disease_risk_score,
        "disease_risk_level": disease_risk_level,
        "env_score": env_score,
        "env_risk_level": env_risk_level,
        "overall": overall,
        "status": status,
        "recommendations": recs,
        "is_healthy": is_healthy,
    }


def _env_score(crop: str, env: dict) -> int:
    """0-100 environmental score based on how far each factor is from ideal."""
    ranges = ENV_CROP_RANGES[crop]
    total = 0.0
    for key, (low, opt_min, opt_max, high) in ranges.items():
        v = env[key]
        if opt_min <= v <= opt_max:
            total += 25.0
        elif low <= v < opt_min:
            span = opt_min - low or 1
            total += 25.0 * max(0.0, 1 - (opt_min - v) / span)
        elif opt_max < v <= high:
            span = high - opt_max or 1
            total += 25.0 * max(0.0, 1 - (v - opt_max) / span)
        # else: outside safe band -> 0 for this factor
    return max(0, min(100, round(total)))


def _score_to_risk(score: int) -> str:
    if score >= 85:
        return "Optimal"
    if score >= 65:
        return "Low"
    if score >= 40:
        return "Moderate"
    if score >= 20:
        return "High"
    return "Critical"


def _recommendations(disease, severity, confidence, env_score, crop,
                      env_risk, is_healthy) -> list[tuple[str, str]]:
    recs: list[tuple[str, str]] = []
    if is_healthy:
        recs.append(("✅",
            "Crop appears healthy. Continue routine monitoring and balanced "
            "irrigation to maintain condition."))
    else:
        recs.append(("🌾",
            f"{disease} detected ({severity} severity, "
            f"confidence {confidence*100:.0f}%): apply the appropriate "
            "fungicide / removal strategy and improve air circulation."))
        if severity == "High":
            recs.append(("⚠️",
                "Severity is high. Remove and destroy severely affected plants "
                "to limit spread, and avoid working in the field while wet."))
    if env_risk in ("High", "Critical"):
        recs.append(("🌡️",
            "Environmental conditions are unfavourable — review temperature, "
            "humidity, soil moisture, and drainage before the next irrigation."))
    if env_risk in ("Moderate",):
        recs.append(("👁️",
            "Environmental readings are marginal. Monitor closely and adjust "
            "field practices as conditions change."))
    if not recs:
        recs.append(("🌿",
            f"Keep up current practices for {crop} and continue regular scouting."))
    return recs


# ---------------------------------------------------------------------------
# Rendering (uses the four reusable functions)
# ---------------------------------------------------------------------------
def _render(results: dict) -> None:
    r = results

    # --- Top: overall score card + status ------------------------------
    sc1, sc2 = st.columns([2, 3])
    with sc1:
        health_score_card(
            r["overall"],
            label="Overall health score",
        )
    with sc2:
        st.markdown("#### Overall crop status")
        status_color = score_color(r["overall"])
        st.markdown(
            f"""
            <div style="background:#f9fafb;border:1px solid #ececec;
                        border-radius:12px;padding:1rem 1.2rem;text-align:center">
              <div style="font-size:.8rem;color:#666;text-transform:uppercase;
                          letter-spacing:.05em">Status</div>
              <div style="font-size:1.6rem;font-weight:700;color:{status_color};
                          margin-top:.2rem">{r['status']}</div>
              <div style="font-size:.85rem;color:#888">Crop: {r['crop']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --- Disease + environmental risk indicators ------------------------
    st.markdown("#### Risk breakdown")
    rb1, rb2 = st.columns(2)
    with rb1:
        st.markdown("**Disease risk**")
        risk_indicator(r["disease_risk_level"], show_bar=True)
        st.caption(f"Disease score: {r['disease_risk_score']}/100")
    with rb2:
        st.markdown("**Environmental risk**")
        risk_indicator(r["env_risk_level"], show_bar=True)
        st.caption(f"Env. score: {r['env_score']}/100")

    st.markdown("---")

    # --- All metrics ---------------------------------------------------
    st.markdown("#### Detailed metrics")
    # Disease block
    st.markdown("**Disease**")
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        metric_display("Crop name", r["crop"], accent="#2e7d32")
    with d2:
        metric_display("Disease", r["disease"],
                       accent="#e57373" if not r["is_healthy"] else "#66bb6a")
    with d3:
        metric_display("Confidence", f"{r['confidence']*100:.0f}%", "model output")
    with d4:
        metric_display("Severity", r["severity"],
                       accent="#ef5350" if r["severity"] in ("High",) else
                             "#ff9800" if r["severity"] == "Moderate" else "#66bb6a")

    # Environment block
    st.markdown("**Environment**")
    e1, e2, e3, e4 = st.columns(4)
    env_keys = ["temperature", "humidity", "soil_moisture", "rainfall"]
    for col, key in zip([e1, e2, e3, e4], env_keys):
        emoji, label, unit = ENV_LABELS[key]
        with col:
            metric_display(label, f"{r['env'][key]} {unit}")

    # --- Recommendations ------------------------------------------------
    st.markdown("#### Agricultural recommendation")
    recommendation_display(r["recommendations"], title="Action")


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
