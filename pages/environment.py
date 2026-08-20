"""Environmental Analysis page — log and assess environmental conditions.

Layout scaffold: readings are prefilled with dummy values the user can edit.
No persistence or analysis logic yet.
"""

from __future__ import annotations

import streamlit as st

from config import ENV_RANGES
from utils.ui import (
    page_header,
    callout,
    card,
    footer,
    env_gauge,
    get_dummy_env_readings,
)


def render() -> None:
    page_header(
        "🌡️",
        "Environmental Analysis",
        "Log and assess temperature, humidity, soil moisture and rainfall.",
    )

    st.markdown("Enter the current environmental readings for the field.")

    dummy = get_dummy_env_readings()
    inputs: dict = {}
    cols = st.columns(len(ENV_RANGES))
    for col, (key, spec) in zip(cols, ENV_RANGES.items()):
        with col:
            inputs[key] = st.number_input(
                f"{key.replace('_', ' ').title()} ({spec['unit']})",
                min_value=float(spec["min"]),
                max_value=float(spec["max"]),
                value=float(dummy[key]),
                step=0.1,
            )

    analyze = st.button("🌡️ Assess environment", type="primary")

    st.markdown("---")

    if not analyze:
        callout("Use default dummy values or adjust them, then click "
                "**Assess environment** to see gauges and a summary.")
        readings = dummy
    else:
        readings = inputs

    # Gauge row (responsive)
    gcols = st.columns(len(ENV_RANGES))
    for gcol, (key, spec) in zip(gcols, ENV_RANGES.items()):
        with gcol:
            st.plotly_chart(
                env_gauge(
                    readings[key], spec["min"], spec["max"],
                    key.replace("_", " ").title(), spec["unit"],
                ),
                use_container_width=True,
            )

    st.markdown("#### Environmental summary")
    left, right = st.columns(2)
    with left:
        card("Overall conditions", _status_text(readings))
    with right:
        card("Per-factor assessment", _factor_table(readings))

    footer()


def _status_text(readings: dict) -> str:
    # Simple placeholder heuristic so the layout has dynamic content.
    issues = []
    if readings["temperature"] > 35:
        issues.append("high temperature")
    if readings["temperature"] < 10:
        issues.append("low temperature")
    if readings["humidity"] > 80:
        issues.append("high humidity")
    if readings["soil_moisture"] > 70:
        issues.append("excess soil moisture")
    if readings["rainfall"] > 50:
        issues.append("heavy rainfall")
    if not issues:
        return "Conditions are within an acceptable range for most crops."
    return "Watch: " + ", ".join(issues) + "."


def _factor_table(readings: dict) -> str:
    rows = ""
    emoji = {"temperature": "🌡️", "humidity": "💧", "soil_moisture": "🌱", "rainfall": "🌧️"}
    for key, spec in ENV_RANGES.items():
        v = readings[key]
        mid = spec["min"] + 0.5 * (spec["max"] - spec["min"])
        status = "Optimal" if v >= mid * 0.6 and v <= mid * 1.4 else "Watch"
        rows += (
            f"<tr><td>{emoji[key]} {key.replace('_',' ').title()}</td>"
            f"<td>{v} {spec['unit']}</td>"
            f"<td>{status}</td></tr>"
        )
    return (
        "<table style='width:100%;border-collapse:collapse;font-size:.85rem'>"
        "<tr style='color:#888'><th align='left'>Factor</th>"
        "<th align='left'>Value</th><th>Status</th></tr>" + rows +
        "</table>"
    )
