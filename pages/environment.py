"""Environmental Analysis page — log and assess environmental conditions.

Wired to the trained environmental risk model (src.environment_model —
Decision Tree / Random Forest, selected automatically at training time).
The per-factor table and radar chart still compare each raw reading
against the crop's ideal band directly (that's a visualization of the
inputs, not a risk classification), but the overall risk level,
confidence, health score, and headline recommendation now come from the
model, matching how pages/health.py is wired.
"""

from __future__ import annotations

import uuid

import plotly.graph_objects as go
import streamlit as st

from config import (
    ENV_RANGES,
    ENV_CROP_RANGES,
)
from src.db import insert_environment_analysis
from src.environment_model import predict_environmental_risk
from src.errors import logger, safe_action
from src.health_engine import compute_environmental_risk_score
from src.validation import validate_crop, validate_environmental_reading, ValidationError
from utils.ui import (
    page_header,
    callout,
    card,
    footer,
    get_dummy_env_readings,
    CHART_THEME,
    RISK_LEVELS,
)
from utils.icons import icon_html

CROPS = list(ENV_CROP_RANGES.keys())

# Per-factor icon/label/unit/step metadata lives in config.ENV_RANGES (single
# source of truth, also used by pages/health.py) — kept as a local alias here
# so the rest of this file doesn't need to change.
FACTOR_META = ENV_RANGES


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
def render() -> None:
    page_header(
        "environment",
        "Environmental Analysis",
        "Log environmental conditions and assess crop risk with the trained model.",
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
        callout("Click **Assess environment** to run the trained risk model "
                "and see per-factor statuses, an overall risk level, and a visualization.")
        footer()
        return

    if analyze:
        try:
            with st.spinner("Assessing environmental risk…"):
                results = _assess(crop, inputs)
            st.session_state["_env_results"] = results
        except FileNotFoundError:
            callout(
                f"{icon_html('warning', size=18)}<b>Environmental risk model not found.</b> "
                "Train it first with <code>python -m src.environment_model</code>."
            )
            footer()
            return
        except ValidationError as e:
            st.error(str(e))
            footer()
            return
        except Exception:
            logger.exception("Unexpected error assessing environment")
            st.error(
                "Assessing environmental risk failed unexpectedly. Please try "
                "again. If the problem continues, contact the app maintainer."
            )
            footer()
            return
    else:
        results = st.session_state["_env_results"]

    _render_risk_banner(results)
    _render_factor_table(results)

    st.markdown("#### Factor visualization")
    _render_radar(results)

    st.markdown("---")
    _render_save_section(results)

    footer()


# ---------------------------------------------------------------------------
# Assessment — per-factor breakdown (rule-based, for display) +
# overall risk (from the trained model)
# ---------------------------------------------------------------------------
def _factor_breakdown(crop: str, inputs: dict) -> list[dict]:
    """Compare each raw reading against the crop's ideal band. This is
    visualization/explanation of the inputs, independent of the trained
    classifier's overall risk call.
    """
    ranges = ENV_CROP_RANGES[crop]
    factors = []

    for key in ("temperature", "humidity", "soil_moisture", "rainfall"):
        value = inputs[key]
        low, opt_min, opt_max, high = ranges[key]
        unit = ENV_RANGES[key]["unit"]

        if opt_min <= value <= opt_max:
            status, color, note = "Optimal", "#7FA687", "Within ideal range"
            norm = (value - low) / (high - low) if high != low else 0.5
        elif low <= value < opt_min:
            status, color, note = "Low", "#ffb74d", f"Below ideal ({opt_min}–{opt_max}{unit})"
            norm = (value - low) / (high - low) if high != low else 0.25
        elif opt_max < value <= high:
            status, color, note = "High", "#B5564B", f"Above ideal ({opt_min}–{opt_max}{unit})"
            norm = (value - low) / (high - low) if high != low else 0.75
        else:
            status, color, note = "Extreme", "#7C3730", "Outside safe range"
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
            "norm": round(max(0.0, min(1.0, norm)), 3),
        })

    return factors


def _assess(crop: str, inputs: dict) -> dict:
    """Run the trained model for overall risk, plus a per-factor breakdown."""
    crop = validate_crop(crop)
    inputs = validate_environmental_reading(**inputs)

    factors = _factor_breakdown(crop, inputs)

    env_pred = predict_environmental_risk({"crop": crop, **inputs})
    health_score, _ = compute_environmental_risk_score(
        env_pred["risk_level"], env_pred["probability"], env_pred["probabilities"],
    )
    risk_color = RISK_LEVELS.get(env_pred["risk_level"], RISK_LEVELS["Moderate"])[1]

    advice = _advice(crop, factors)

    return {
        "crop": crop,
        "factors": factors,
        "risk_level": env_pred["risk_level"],
        "risk_color": risk_color,
        "probability": env_pred["probability"],
        "probabilities": env_pred["probabilities"],
        "health_score": health_score,
        "explanation": env_pred["explanation"],
        "model_recommendation": env_pred["recommendation"],
        "model_used": env_pred["model_used"],
        "advice": advice,
        "inputs": {f["key"]: f["value"] for f in factors},
        "_analysis_token": uuid.uuid4().hex,
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
    bg = "#F5E9E6" if results["risk_level"] in ("High", "Critical") else "#EAEFE2"
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
              <div style="font-size:.78rem;color:#7C8571">
                {results['probability']*100:.0f}% model confidence · {results['model_used']}
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
    callout(results["explanation"])


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


def _build_env_db_record(results: dict) -> dict:
    """Map an environmental assessment result to src.db's
    `environment_analyses` columns."""
    return {
        "crop_name": results["crop"],
        "temperature": results["inputs"]["temperature"],
        "humidity": results["inputs"]["humidity"],
        "soil_moisture": results["inputs"]["soil_moisture"],
        "rainfall": results["inputs"]["rainfall"],
        "risk_level": results["risk_level"],
        "probability": results["probability"],
        "health_score": results["health_score"],
        "recommendation": results["model_recommendation"],
        # created_at (timestamp) is filled in automatically by insert_environment_analysis().
    }


def _render_save_section(results: dict) -> None:
    """'Save Analysis' button, guarded against duplicate inserts.

    Mirrors pages/health.py's save pattern: each freshly computed
    assessment carries a unique `_analysis_token`; a save is only allowed
    once per token, and the button is replaced with a confirmation
    afterward so a stray rerun or repeat click can't insert the same
    reading twice.
    """
    token = results["_analysis_token"]
    saved_token = st.session_state.get("_env_saved_token")

    if saved_token == token:
        saved_id = st.session_state.get("_env_saved_id")
        st.success(f"Analysis saved to database (ID: {saved_id}).")
        st.button("Saved ✓", use_container_width=True, disabled=True, key="_env_saved_btn")
        return

    if st.button("Save Analysis", type="primary", use_container_width=True, key="_env_save_btn"):
        with safe_action("Saving analysis"):
            with st.spinner("Saving analysis…"):
                record = _build_env_db_record(results)
                analysis_id = insert_environment_analysis(record)
            st.session_state["_env_saved_token"] = token
            st.session_state["_env_saved_id"] = analysis_id
            st.rerun()


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