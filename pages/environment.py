"""Environmental Analysis page — log and assess environmental conditions.

Rule-based, no ML model yet. For each chosen crop, the four environmental
readings (temperature, humidity, soil moisture, rainfall) are scored against
that crop's ideal ranges to derive a per-factor status and an overall risk
level. A Plotly radar chart visualizes the four factors together.
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from config import (
    ENV_RANGES,
    ENV_CROP_RANGES,
    ENV_RISK_THRESHOLDS,
)
from utils.ui import (
    page_header,
    callout,
    card,
    footer,
    get_dummy_env_readings,
    CHART_THEME,
)
from utils.icons import icon_html

CROPS = list(ENV_CROP_RANGES.keys())

FACTOR_META = {
    "temperature":   {"icon": "temperature", "label": "Temperature",   "nudge": 2},
    "humidity":      {"icon": "humidity",    "label": "Humidity",      "nudge": 5},
    "soil_moisture": {"icon": "soil",        "label": "Soil moisture", "nudge": 5},
    "rainfall":      {"icon": "rainfall",    "label": "Rainfall",      "nudge": 2},
}


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
def render() -> None:
    page_header(
        "environment",
        "Environmental Analysis",
        "Log environmental conditions and assess crop risk — rule-based for now.",
    )

    callout(
        "**Rule-based mode** — no ML model yet. Each reading is scored against "
        "the selected crop's ideal range to derive a status and risk level."
    )

    col_input, col_summary = st.columns([2, 3])

    # -------------------------------------------------------------- inputs
    with col_input:
        st.markdown("#### Enter environmental readings")
        crop = st.selectbox("Crop type", CROPS)

        dummy = get_dummy_env_readings()
        inputs: dict = {}
        cols = st.columns(2)
        factor_order = ["temperature", "humidity", "soil_moisture", "rainfall"]
        for col, key in zip(cols * 2, factor_order):
            with col:
                spec = ENV_RANGES[key]
                inputs[key] = st.number_input(
                    f"{FACTOR_META[key]['label']} ({spec['unit']})",
                    min_value=float(spec["min"]),
                    max_value=float(spec["max"]),
                    value=float(dummy[key]),
                    step=FACTOR_META[key]["nudge"],
                    format="%.1f",
                )

        analyze = st.button("Assess environment", type="primary",
                           use_container_width=True)

    # ----------------------------------------------------- input summary
    with col_summary:
        st.markdown("#### Input summary")
        s1, s2, s3, s4 = st.columns(4)
        for col, key in zip([s1, s2, s3, s4], factor_order):
            with col:
                card(
                    FACTOR_META[key]["label"],
                    f"<div style='font-size:1.4rem;font-weight:700;color:var(--ink)'>"
                    f"{inputs[key]} {ENV_RANGES[key]['unit']}</div>",
                )

    st.markdown("---")

    # ------------------------------------------------------- assessment
    if not analyze and not st.session_state.get("_env_results"):
        callout("Click **Assess environment** to generate per-factor statuses, "
                "an overall risk level, and a visualization.")
        footer()
        return

    results = _assess(crop, inputs)
    st.session_state["_env_results"] = results
    st.session_state["_env_crop"] = crop

    _render_risk_banner(results)
    _render_factor_table(results)

    st.markdown("#### Factor visualization")
    _render_radar(results)

    footer()


# ---------------------------------------------------------------------------
# Rule-based assessment
# ---------------------------------------------------------------------------
def _assess(crop: str, inputs: dict) -> dict:
    """Score each factor against the crop's ideal ranges (a, b, c, d)."""
    ranges = ENV_CROP_RANGES[crop]
    factors = []
    suboptimal = 0

    for key in ("temperature", "humidity", "soil_moisture", "rainfall"):
        value = inputs[key]
        low, opt_min, opt_max, high = ranges[key]
        unit = ENV_RANGES[key]["unit"]

        if opt_min <= value <= opt_max:
            status, color, note = "Optimal", "#7FA687", "Within ideal range"
            norm = (value - low) / (high - low) if high != low else 0.5
        elif low <= value < opt_min:
            status, color, note = "Low", "#ffb74d", f"Below ideal ({opt_min}–{opt_max}{unit})"
            suboptimal += 1
            norm = (value - low) / (high - low) if high != low else 0.25
        elif opt_max < value <= high:
            status, color, note = "High", "#B5564B", f"Above ideal ({opt_min}–{opt_max}{unit})"
            suboptimal += 1
            norm = (value - low) / (high - low) if high != low else 0.75
        else:
            # Outside the safe band entirely (very low or very high).
            status, color, note = "Extreme", "#7C3730", "Outside safe range"
            suboptimal += 1
            norm = 0.0 if value < low else 1.0

        factors.append({
            "key": key,
            "label": FACTOR_META[key]["label"],
            "icon": FACTOR_META[key]["icon"],
            "value": value,
            "unit": unit,
            "status": status,
            "color": color,
            "note": note,
            "ideal": (opt_min, opt_max),
            # 0..1 normalization within [low, high] for the radar chart
            "norm": round(max(0.0, min(1.0, norm)), 3),
        })

    # Health score (100 = all optimal). Each suboptimal factor costs 25 pts,
    # plus a penalty proportional to how far out of range it lies.
    penalty = 0.0
    for f in factors:
        if f["status"] != "Optimal":
            # distance from the nearer ideal edge, normalized by band width
            lo, hi = f["ideal"]
            if f["value"] < lo:
                dist = (lo - f["value"]) / max(lo - ENV_RANGES[f["key"]]["min"], 1)
            else:
                dist = (f["value"] - hi) / max(ENV_RANGES[f["key"]]["max"] - hi, 1)
            penalty += 25.0 + 25.0 * min(1.0, max(0.0, dist))
    health = max(0, round(100 - penalty))

    risk_level, risk_color = ENV_RISK_THRESHOLDS[min(suboptimal, 4)]
    advice = _advice(crop, factors)

    return {
        "crop": crop,
        "factors": factors,
        "suboptimal": suboptimal,
        "risk_level": risk_level,
        "risk_color": risk_color,
        "health_score": health,
        "advice": advice,
        "inputs": {f["key"]: f["value"] for f in factors},
    }


def _advice(crop: str, factors: list[dict]) -> list[str]:
    """Plain-language recommendations based on out-of-range factors."""
    tips = {
        "temperature": {
            "Low":    ("Consider row covers or warming the soil; cold slows growth"
                       " of {crop}."),
            "High":   ("Provide shade cloth or irrigate in the cool evening to "
                       "reduce heat stress on {crop}."),
            "Extreme": ("Temperature is outside the safe band — protect {crop} "
                        "or delay sensitive field operations."),
        },
        "humidity": {
            "High":   ("High humidity raises fungal-disease risk for {crop}; "
                       "improve airflow and avoid overhead watering."),
            "Low":    ("Humidity is low; mulch around {crop} to reduce moisture "
                       "loss and leaf curl."),
            "Extreme": ("Humidity level is extreme — monitor {crop} closely for "
                        "disease or wilting."),
        },
        "soil_moisture": {
            "Low":    ("Soil is dry for {crop}; schedule irrigation and check "
                       "root zones."),
            "High":   ("Soil is waterlogged for {crop}; improve drainage and "
                       "hold off irrigation."),
            "Extreme": ("Soil moisture is off-scale for {crop}; correct drainage "
                        "or watering immediately."),
        },
        "rainfall": {
            "Low":    ("Rainfall is light; supplement {crop} with irrigation."),
            "High":   ("Heavy rainfall expected; watch for runoff and leaching "
                       "around {crop}; ensure field drainage."),
            "Extreme": ("Extreme rainfall for {crop}; protect low-lying areas "
                        "and check for standing water."),
        },
    }
    out = []
    for f in factors:
        if f["status"] in tips.get(f["key"], {}):
            out.append(tips[f["key"]][f["status"]].format(crop=crop.lower()))
    if not out:
        out.append(f"All environmental factors are within the ideal range for {crop}. "
                   "Maintain current practices and keep monitoring.")
    return out


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _render_risk_banner(results: dict) -> None:
    color = results["risk_color"]
    bg = "#F5E9E6" if results["suboptimal"] >= 2 else "#EAEFE2"
    st.markdown(
        f"""
        <div style="background:{bg};border-left:5px solid {color};
                    border-radius:12px;padding:1rem 1.25rem;margin-bottom:1rem">
          <div style="display:flex;justify-content:space-between;align-items:center">
            <div>
              <div style="font-size:.8rem;color:#4E5646;text-transform:uppercase;
                          letter-spacing:.04em">Overall risk level</div>
              <div style="font-size:1.6rem;font-weight:700;color:{color}">
                {results['risk_level']}
              </div>
            </div>
            <div style="text-align:right">
              <div style="font-size:.8rem;color:#4E5646">Env. health score</div>
              <div style="font-size:1.6rem;font-weight:700;color:#1C2E20">
                {results['health_score']}<span style="font-size:.9rem">/100</span>
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_factor_table(results: dict) -> None:
    rows = ""
    for f in results["factors"]:
        rows += (
            f"<tr>"
            f"<td>{icon_html(f['icon'], size=16, margin_right='.3em')}{f['label']}</td>"
            f"<td><b>{f['value']} {f['unit']}</b></td>"
            f"<td>{f['ideal'][0]}–{f['ideal'][1]} {f['unit']}</td>"
            f"<td><span style='color:{f['color']};font-weight:700'>{f['status']}</span></td>"
            f"<td style='color:#6B7462'>{f['note']}</td>"
            f"</tr>"
        )
    st.markdown(
        "<div class='card'><table style='width:100%;border-collapse:collapse;"
        "font-size:.88rem'>"
        "<tr style='color:#7C8571;border-bottom:1px solid #E6E8DC'>"
        "<th align='left'>Factor</th><th align='left'>Value</th>"
        "<th align='left'>Ideal range</th><th align='left'>Status</th>"
        "<th align='left'>Note</th></tr>" + rows + "</table></div>",
        unsafe_allow_html=True,
    )

    # Recommendations
    st.markdown("#### Recommendations")
    for tip in results["advice"]:
        st.markdown(f"• {tip}")


def _render_radar(results: dict) -> None:
    """Radar chart of the four factors (0..1 normalized to each crop's band)."""
    labels = [f["label"] for f in results["factors"]]
    values = [f["norm"] for f in results["factors"]]
    colors = [f["color"] for f in results["factors"]]

    # Ideal-range band as a shaded area (same for all on the 0..1 axis: the
    # optimal band sits at 0.33–0.67 by construction of low/opt_min/opt_max/high
    # only approximately; instead we compute the actual normalized window).
    ideal_lo, ideal_hi = [], []
    crop = results["crop"]
    for f in results["factors"]:
        low, opt_min, opt_max, high = ENV_CROP_RANGES[crop][f["key"]]
        span = high - low or 1
        ideal_lo.append(round((opt_min - low) / span, 3))
        ideal_hi.append(round((opt_max - low) / span, 3))

    fig = go.Figure()

    # Ideal band (upper edge)
    fig.add_trace(go.Scatterpolar(
        r=ideal_hi + [ideal_hi[0]],
        theta=labels + [labels[0]],
        fill="tonext", fillcolor="rgba(102,187,106,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Ideal band", showlegend=True,
    ))
    # Ideal band (lower edge, base)
    fig.add_trace(go.Scatterpolar(
        r=ideal_lo + [ideal_lo[0]],
        theta=labels + [labels[0]],
        line=dict(color="rgba(0,0,0,0)"),
        name="Ideal (low)", showlegend=False, hoverinfo="skip",
    ))
    # Actual readings
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=labels + [labels[0]],
        fill="toself",
        fillcolor="rgba(46,125,50,0.18)",
        line=dict(color="#2F6D46", width=2),
        marker=dict(color=colors, size=8),
        name="Current", showlegend=True,
    ))

    fig.update_layout(
        **CHART_THEME,
        polar=dict(
            radialaxis=dict(range=[0, 1], showticklabels=False, layer="below traces"),
            angularaxis=dict(tickfont=dict(size=12)),
        ),
        legend=dict(orientation="h", y=1.1),
        height=380,
        margin=dict(t=40, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Readings are normalized 0–1 against each factor's safe band "
        f"for **{crop}**. The shaded green zone is the ideal range."
    )
