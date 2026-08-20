"""Crop Health Analysis page — combine image + environment into a health score.

Layout scaffold: the score is derived from a transparent placeholder formula
so the layout can be reviewed end-to-end without AI or DB support.
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from utils.ui import (
    page_header,
    callout,
    card,
    footer,
    metric_tile,
    get_dummy_env_readings,
)


def render() -> None:
    page_header(
        "🌱",
        "Crop Health Analysis",
        "Combine disease detection and environmental data into an overall health score.",
    )

    callout("This view brings together the disease result and environmental "
            "readings. Inputs below are dummy values; scoring logic is a "
            "placeholder until the AI + DB layers are built.")

    inputs = get_dummy_env_readings()
    col_in, col_out = st.columns([2, 3])

    with col_in:
        st.markdown("#### Inputs")
        with st.expander("Disease result", expanded=True):
            st.selectbox("Detected condition", ["Healthy", "Early Blight", "Leaf Mold", "Rust"], index=1)
            st.slider("Detection confidence", 0.0, 1.0, 0.80, 0.05)
            st.select_slider("Severity", options=["Low", "Moderate", "High"], value="Moderate")

        with st.expander("Environmental readings", expanded=True):
            inputs["temperature"] = st.number_input("Temperature (°C)", -10.0, 50.0, inputs["temperature"])
            inputs["humidity"] = st.number_input("Humidity (%)", 0.0, 100.0, inputs["humidity"])
            inputs["soil_moisture"] = st.number_input("Soil moisture (%)", 0.0, 100.0, inputs["soil_moisture"])
            inputs["rainfall"] = st.number_input("Rainfall (mm)", 0.0, 500.0, inputs["rainfall"])

        compute = st.button("🌱 Calculate crop health", type="primary")

    with col_out:
        st.markdown("#### Overall score")
        if not compute:
            st.info("Click **Calculate crop health** to compute the score.")
            score, grade = 0, "—"
        else:
            score, grade = _placeholder_score(inputs)

        # Gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score if score else 50,
            number={"suffix": " / 100"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": _score_color(score or 50)},
                "steps": [
                    {"range": [0, 40],  "color": "#ffcdd2"},
                    {"range": [40, 70], "color": "#fff9c4"},
                    {"range": [70, 100], "color": "#c8e6c9"},
                ],
            },
        ))
        fig.update_layout(height=240, margin=dict(t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            metric_tile("Health grade", grade)
        with c2:
            metric_tile("Recommendation", _recommendation(score or 50))

        st.markdown("#### Recommendation breakdown")
        for label, weight, val in _components(inputs, score if compute else 0):
            bar_w = int(score * weight / 100) if compute else 0
            st.markdown(
                f"""
                <div style="margin-bottom:.6rem">
                  <div style="display:flex;justify-content:space-between;font-size:.85rem">
                    <span><b>{label}</b> (weight {weight}%)</span><span>{val}</span>
                  </div>
                  <div style="background:#eee;border-radius:6px;height:8px;margin-top:.25rem">
                    <div style="width:{bar_w}%;height:100%;background:#2e7d32"></div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    footer()


def _placeholder_score(env: dict) -> tuple[int, str]:
    # Transparent placeholder weighting until real logic exists.
    t_ok = 1 if 15 <= env["temperature"] <= 32 else 0.5
    h_ok = 1 if 40 <= env["humidity"] <= 75 else 0.5
    s_ok = 1 if 30 <= env["soil_moisture"] <= 60 else 0.5
    r_ok = 1 if env["rainfall"] <= 30 else 0.5
    score = int(25 * t_ok + 25 * h_ok + 20 * s_ok + 10 * r_ok + 20)  # +20 disease
    score = max(0, min(100, score))
    grade = ("A" if score >= 80 else "B" if score >= 60 else
             "C" if score >= 40 else "D")
    return score, grade


def _score_color(score: int) -> str:
    if score >= 70:
        return "#2e7d32"
    if score >= 40:
        return "#fdd835"
    return "#e57373"


def _recommendation(score: int) -> str:
    if score >= 80:
        return "Maintain practices ✅"
    if score >= 60:
        return "Minor monitoring needed"
    if score >= 40:
        return "Take corrective action"
    return "Intervention required ⚠️"


def _components(env: dict, score: int):
    # label, weight %, display value
    return [
        ("Disease severity", 20, "Moderate"),
        ("Temperature", 25, f"{env['temperature']} °C"),
        ("Humidity", 25, f"{env['humidity']} %"),
        ("Soil moisture", 20, f"{env['soil_moisture']} %"),
        ("Rainfall", 10, f"{env['rainfall']} mm"),
    ]
