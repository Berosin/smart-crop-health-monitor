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
from src.errors import logger, safe_action, WeatherError
from src.health_engine import compute_environmental_risk_score
from src.i18n import get_language, tr_label, tr_crop, tr_env_tip, tr_env_all_optimal
from src.validation import validate_crop, validate_environmental_reading, ValidationError
from src.weather import get_current_weather, get_forecast, build_forecast_risk, resolve_api_key
from utils.ui import (
    page_header,
    callout,
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

# Left-accent color per factor for the input-summary tiles — drawn from the
# app's existing earthy palette (utils.ui's --clay/--leaf-light/--soil/--leaf)
# so each reading is visually distinct without introducing new colors.
FACTOR_ACCENTS = {
    "temperature":   "#C97A3B",  # clay
    "humidity":      "#7FA687",  # leaf-light
    "soil_moisture": "#8A6A47",  # soil
    "rainfall":      "#2F6D46",  # leaf
}


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
def _render_input_tile(key: str, value: float) -> None:
    """One reading in the 'Input summary' panel: icon + label on top,
    large value with unit below, colored left accent per factor —
    matches the KPI-tile style used on the Dashboard rather than the
    plain bordered card used previously (which felt cramped at 4-across
    and let long labels like "Soil moisture" wrap awkwardly)."""
    spec = FACTOR_META[key]
    accent = FACTOR_ACCENTS.get(key, "var(--leaf)")
    icon_tag = icon_html(spec["icon"], size=16, margin_right=".35em")
    st.markdown(
        f"""
        <div class="metric-tile" style="border-left-color:{accent}; margin-bottom:1rem;">
          <div class="label">{icon_tag}{spec['label']}</div>
          <div class="value">{value:g}
            <span style="font-size:.8rem; font-weight:500; color:var(--text-faint);">{spec['unit']}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render() -> None:
    lang = get_language()
    page_header(
        "environment",
        tr_label("Environmental Analysis", lang),
        tr_label("Log environmental conditions and assess crop risk with the trained model.", lang),
    )

    col_input, col_summary = st.columns([2, 3])

    # -------------------------------------------------------------- inputs
    with col_input:
        st.markdown(f"#### {tr_label('Enter environmental readings', lang)}")
        crop = st.selectbox(tr_label("Crop type", lang), CROPS, format_func=lambda c: tr_crop(c, lang))

        data_source_options = ["Manual Entry", "Live Weather"]
        data_source = st.radio(
            tr_label("Data source", lang),
            data_source_options,
            format_func=lambda o: tr_label(o, lang),
            horizontal=True,
            key="_env_data_source",
            help=tr_label(
                "Live Weather auto-fills temperature, humidity, and rainfall "
                "from OpenWeatherMap for a location you choose. Soil moisture "
                "always needs a manual reading — no weather API measures it. "
                "Manual Entry (the default) needs nothing beyond this page.", lang
            ),
        )

        live_weather = None
        if data_source == "Live Weather":
            live_weather = _render_live_weather_panel()

        # One-shot auto-fill: pre-seed the widget state for temperature/
        # humidity/rainfall right before those widgets are created below.
        # This must happen *before* instantiation, and must be one-shot
        # (popped, not just read) — a keyed number_input ignores its
        # `value=` argument on every rerun after the first, so simply
        # passing the fetched value as `value=` on a later rerun would
        # silently do nothing (this is Streamlit's normal, documented
        # widget-state precedence, not a bug in it) and it would fight the
        # user's own edits on top if we re-applied it on every rerun.
        prefill = st.session_state.pop("_env_prefill_pending", None)
        if prefill:
            st.session_state["_env_input_temperature"] = prefill["temperature"]
            st.session_state["_env_input_humidity"] = prefill["humidity"]
            st.session_state["_env_input_rainfall"] = prefill["rainfall"]

        dummy = get_dummy_env_readings()
        inputs: dict = {}
        cols = st.columns(2)
        factor_order = ["temperature", "humidity", "soil_moisture", "rainfall"]
        prefilled_keys = {"temperature", "humidity", "rainfall"} if prefill else set()
        for col, key in zip(cols * 2, factor_order):
            with col:
                spec = ENV_RANGES[key]
                help_text = (
                    tr_label("Not available from weather data — enter your own reading.", lang)
                    if key == "soil_moisture" and live_weather else None
                )
                widget_kwargs = dict(
                    min_value=float(spec["min"]),
                    max_value=float(spec["max"]),
                    step=FACTOR_META[key]["nudge"],
                    format="%.1f",
                    key=f"_env_input_{key}",
                    help=help_text,
                )
                # Omit `value=` on exactly the rerun where we just pre-seeded
                # this key via session_state above — passing both at once is
                # what Streamlit's widget-state policy warns about, even
                # though the pre-seeded value still wins either way.
                if key not in prefilled_keys:
                    widget_kwargs["value"] = float(dummy[key])
                inputs[key] = st.number_input(
                    f"{tr_label(FACTOR_META[key]['label'], lang)} ({spec['unit']})", **widget_kwargs,
                )

        analyze = st.button(tr_label("Assess environment", lang), type="primary",
                           use_container_width=True)

    # ----------------------------------------------------- input summary
    with col_summary:
        st.markdown(f"#### {tr_label('Input summary', lang)}")
        st.caption(
            tr_label("What will be sent to the risk model.", lang)
            + (f" {tr_label('Live weather for', lang)} {live_weather['location_name']}, "
               f"{tr_label('fetched', lang)} {live_weather['fetched_at']}."
               if live_weather else "")
        )
        s1, s2 = st.columns(2)
        for col, key in zip([s1, s2, s1, s2], factor_order):
            with col:
                _render_input_tile(key, inputs[key])

    st.markdown("---")

    # ---------------------------------------------- forecast risk outlook
    if data_source == "Live Weather" and live_weather:
        _render_forecast_section(crop, inputs["soil_moisture"])
        st.markdown("---")

    # ------------------------------------------------------- assessment
    if not analyze and not st.session_state.get("_env_results"):
        callout(tr_label("Click **Assess environment** to run the trained risk model "
                "and see per-factor statuses, an overall risk level, and a visualization.", lang))
        footer()
        return

    if analyze:
        try:
            with st.spinner("Assessing environmental risk…"):
                results = _assess(crop, inputs, lang=lang)
            st.session_state["_env_results"] = results
        except FileNotFoundError:
            callout(
                f"{icon_html('warning', size=18)}<b>{tr_label('Environmental risk model not found.', lang)}</b> "
                f"{tr_label('Train it first with', lang)} <code>python -m src.environment_model</code>."
            )
            footer()
            return
        except ValidationError as e:
            st.error(str(e))
            footer()
            return
        except Exception:
            logger.exception("Unexpected error assessing environment")
            st.error(tr_label(
                "Assessing environmental risk failed unexpectedly. Please try "
                "again. If the problem continues, contact the app maintainer.", lang
            ))
            footer()
            return
    else:
        results = st.session_state["_env_results"]

    _render_risk_banner(results)
    _render_factor_table(results)

    st.markdown(f"#### {tr_label('Factor visualization', lang)}")
    _render_radar(results)

    st.markdown("---")
    _render_save_section(results)

    footer()


# ---------------------------------------------------------------------------
# Live Weather — location + API key input, current-conditions fetch
# ---------------------------------------------------------------------------
def _render_live_weather_panel() -> dict | None:
    """Location + API-key inputs, a Fetch button, and the fetched reading
    (if any) from st.session_state. Returns the fetched weather dict, or
    None if nothing has been successfully fetched yet this session.

    Never blocks the rest of the page on failure — a bad key or unreachable
    API just leaves live_weather as None, and the number_input fields above
    simply keep their manual-entry placeholder values (see render()).
    """
    lang = get_language()
    session_key = st.session_state.get("_owm_api_key")
    api_key = resolve_api_key(session_key)

    if api_key:
        location = st.text_input(
            tr_label("Location", lang), value=st.session_state.get("_owm_location", "Chennai, IN"),
            placeholder="e.g. Chennai, IN",
            help=tr_label("City name, optionally with a country code, e.g. 'Chennai, IN'.", lang),
            key="_owm_location",
        )
    else:
        loc_col, key_col = st.columns(2)
        with loc_col:
            location = st.text_input(
                tr_label("Location", lang), value=st.session_state.get("_owm_location", "Chennai, IN"),
                placeholder="e.g. Chennai, IN",
                help=tr_label("City name, optionally with a country code, e.g. 'Chennai, IN'.", lang),
                key="_owm_location",
            )
        with key_col:
            typed_key = st.text_input(
                tr_label("OpenWeatherMap API key", lang), type="password",
                key="_owm_api_key_input",
                help="Free at openweathermap.org/api. Kept only for this session — "
                     "never written to disk by this app. Set OPENWEATHERMAP_API_KEY "
                     "as an environment variable or in .streamlit/secrets.toml to "
                     "skip typing this every time.",
            )
            if typed_key:
                st.session_state["_owm_api_key"] = typed_key
                api_key = typed_key

    if not api_key:
        no_reenter_label = tr_label("so you don't have to re-enter it.", lang)
        callout(
            f"{icon_html('info', size=18)}{tr_label('No OpenWeatherMap API key configured yet. Get a free one at', lang)} "
            "<a href='https://openweathermap.org/api' target='_blank'>"
            f"openweathermap.org/api</a> {tr_label('and paste it above, or set', lang)} "
            f"<code>OPENWEATHERMAP_API_KEY</code> {tr_label('as an environment variable / in', lang)} "
            f"<code>.streamlit/secrets.toml</code> {no_reenter_label}"
        )
        return st.session_state.get("_env_live_weather")

    fetch = st.button(tr_label("Fetch current weather", lang), use_container_width=True, key="_owm_fetch_btn")
    if fetch:
        try:
            with st.spinner(f"Fetching weather for {location}…"):
                weather = get_current_weather(location, api_key, lang=lang)
            st.session_state["_env_live_weather"] = weather
            # One-shot: render() pops this right before creating the
            # temperature/humidity/rainfall widgets, so this fetch
            # overwrites them exactly once rather than fighting any
            # edits the person makes afterward.
            st.session_state["_env_prefill_pending"] = weather
        except WeatherError as e:
            st.error(str(e))
        except Exception:
            logger.exception("Unexpected error fetching live weather")
            st.error(tr_label(
                "Fetching weather failed unexpectedly. Please try again, or "
                "switch to Manual Entry above.", lang
            ))

    weather = st.session_state.get("_env_live_weather")
    if weather:
        st.markdown(
            f"""
            <div class="callout">
              {icon_html('weather', size=18)}<b>{weather['location_name']}</b> — {weather['description']},
              {weather['temperature']:.0f}°C, {weather['humidity']:.0f}% {tr_label('humidity', lang)}
              <span style="color:#7C8571;font-size:.8rem"> ({tr_label('fetched', lang)} {weather['fetched_at']})</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    return weather


# ---------------------------------------------------------------------------
# Forecast disease-risk outlook — same trained model, forecast data as input
# ---------------------------------------------------------------------------
def _render_forecast_section(crop: str, soil_moisture: float) -> None:
    lang = get_language()
    st.markdown(f"#### {tr_label('Short-term disease risk forecast', lang)}")
    st.caption(
        tr_label("Runs the same trained risk model against the next few days' forecast "
        "instead of a single reading. Soil moisture is held at your current "
        "value above", lang)
        + f" ({soil_moisture:g}%) "
        + tr_label("for every day — no weather API forecasts soil moisture.", lang)
    )

    api_key = resolve_api_key(st.session_state.get("_owm_api_key"))
    location = st.session_state.get("_owm_location", "")
    if not api_key or not location:
        st.info(tr_label("Fetch current weather above first (needs a location and API key).", lang))
        return

    if st.button(tr_label("Load 5-day risk forecast", lang), key="_owm_forecast_btn"):
        try:
            with st.spinner("Fetching forecast and running the risk model…"):
                forecast_days = get_forecast(location, api_key, days=5, lang=lang)
                forecast_risk = build_forecast_risk(crop, forecast_days, soil_moisture, lang=lang)
            st.session_state["_env_forecast_risk"] = forecast_risk
        except WeatherError as e:
            st.error(str(e))
        except Exception:
            logger.exception("Unexpected error building forecast risk")
            st.error(tr_label("Building the forecast failed unexpectedly. Please try again.", lang))

    forecast_risk = st.session_state.get("_env_forecast_risk")
    if not forecast_risk:
        return

    cols = st.columns(len(forecast_risk))
    for col, day in zip(cols, forecast_risk):
        _, color, _ = RISK_LEVELS.get(day["risk_level"], RISK_LEVELS["Moderate"])
        with col:
            st.markdown(
                f"""
                <div class="card" style="text-align:center;border-top:4px solid {color};padding:.75rem .5rem">
                  <div style="font-size:.8rem;color:#4E5646;font-weight:600">{day['day_label']}</div>
                  <div style="font-size:.78rem;color:#7C8571;margin:.15rem 0">{day['description']}</div>
                  <div style="font-size:1rem;color:var(--ink)">{day['temp_min']:.0f}–{day['temp_max']:.0f}°C</div>
                  <div style="font-size:.78rem;color:#7C8571">{day['rainfall_total']:.0f}mm {tr_label('rain', lang)}</div>
                  <div style="margin-top:.5rem;font-weight:700;color:{color}">{tr_label(day['risk_level'], lang)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    worst = max(forecast_risk, key=lambda d: RISK_LEVELS.get(d["risk_level"], ("", "", 0.5))[2])
    if RISK_LEVELS.get(worst["risk_level"], ("", "", 0))[2] >= RISK_LEVELS["High"][2]:
        callout(
            f"{icon_html('warning', size=18)}<b>{worst['day_label']} {tr_label('looks highest-risk:', lang)}</b> "
            f"{worst['explanation']}"
        )


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
    lang = get_language()

    for key in ("temperature", "humidity", "soil_moisture", "rainfall"):
        value = inputs[key]
        low, opt_min, opt_max, high = ranges[key]
        unit = ENV_RANGES[key]["unit"]

        if opt_min <= value <= opt_max:
            status, color, note = "Optimal", "#7FA687", tr_label("Within ideal range", lang)
            norm = (value - low) / (high - low) if high != low else 0.5
        elif low <= value < opt_min:
            status, color = "Low", "#ffb74d"
            note = f"{tr_label('Below ideal', lang)} ({opt_min}–{opt_max}{unit})"
            norm = (value - low) / (high - low) if high != low else 0.25
        elif opt_max < value <= high:
            status, color = "High", "#B5564B"
            note = f"{tr_label('Above ideal', lang)} ({opt_min}–{opt_max}{unit})"
            norm = (value - low) / (high - low) if high != low else 0.75
        else:
            status, color, note = "Extreme", "#7C3730", tr_label("Outside safe range", lang)
            norm = 0.0 if value < low else 1.0

        factors.append({
            "key": key,
            "label": tr_label(FACTOR_META[key]["label"], lang),
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


def _assess(crop: str, inputs: dict, lang: str = "en") -> dict:
    """Run the trained model for overall risk, plus a per-factor breakdown."""
    crop = validate_crop(crop)
    inputs = validate_environmental_reading(**inputs)

    factors = _factor_breakdown(crop, inputs)

    env_pred = predict_environmental_risk({"crop": crop, **inputs}, lang=lang)
    health_score, _ = compute_environmental_risk_score(
        env_pred["risk_level"], env_pred["probability"], env_pred["probabilities"],
    )
    risk_color = RISK_LEVELS.get(env_pred["risk_level"], RISK_LEVELS["Moderate"])[1]

    advice = _advice(crop, factors, lang=lang)

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


def _advice(crop: str, factors: list[dict], lang: str = "en") -> list[str]:
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
    # f["status"] stays in English internally (set by _factor_breakdown) —
    # only translated at render time in _render_factor_table — so it can
    # be used directly as a lookup key here.
    out = []
    for f in factors:
        status_en = f["status"]
        if status_en in tips.get(f["key"], {}):
            english_tip = tips[f["key"]][status_en].format(crop=crop.lower())
            out.append(tr_env_tip(f["key"], status_en, crop, lang, fallback=english_tip))
    if not out:
        english_fallback = (f"All environmental factors are within the ideal range for {crop}. "
                            "Maintain current practices and keep monitoring.")
        out.append(tr_env_all_optimal(crop, lang, fallback=english_fallback))
    return out


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _render_risk_banner(results: dict) -> None:
    lang = get_language()
    color = results["risk_color"]
    bg = "#F5E9E6" if results["risk_level"] in ("High", "Critical") else "#EAEFE2"
    st.markdown(
        f"""
        <div style="background:{bg};border-left:5px solid {color};
                    border-radius:12px;padding:1rem 1.25rem;margin-bottom:1rem">
          <div style="display:flex;justify-content:space-between;align-items:center">
            <div>
              <div style="font-size:.8rem;color:#4E5646;text-transform:uppercase;
                          letter-spacing:.04em">{tr_label('Overall risk level', lang)}</div>
              <div style="font-size:1.6rem;font-weight:700;color:{color}">
                {tr_label(results['risk_level'], lang)}
              </div>
              <div style="font-size:.78rem;color:#7C8571">
                {results['probability']*100:.0f}% {tr_label('model confidence', lang)} · {results['model_used']}
              </div>
            </div>
            <div style="text-align:right">
              <div style="font-size:.8rem;color:#4E5646">{tr_label('Env. health score', lang)}</div>
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
    lang = get_language()
    rows = ""
    for f in results["factors"]:
        rows += (
            f"<tr>"
            f"<td>{icon_html(f['icon'], size=16, margin_right='.3em')}{f['label']}</td>"
            f"<td><b>{f['value']} {f['unit']}</b></td>"
            f"<td>{f['ideal'][0]}–{f['ideal'][1]} {f['unit']}</td>"
            f"<td><span style='color:{f['color']};font-weight:700'>{tr_label(f['status'], lang)}</span></td>"
            f"<td style='color:#6B7462'>{f['note']}</td>"
            f"</tr>"
        )
    st.markdown(
        "<div class='card'><table style='width:100%;border-collapse:collapse;"
        "font-size:.88rem'>"
        "<tr style='color:#7C8571;border-bottom:1px solid #E6E8DC'>"
        f"<th align='left'>{tr_label('Factor', lang)}</th><th align='left'>{tr_label('Value', lang)}</th>"
        f"<th align='left'>{tr_label('Ideal range', lang)}</th><th align='left'>{tr_label('Status', lang)}</th>"
        f"<th align='left'>{tr_label('Note', lang)}</th></tr>" + rows + "</table></div>",
        unsafe_allow_html=True,
    )

    # Recommendations
    st.markdown(f"#### {tr_label('Recommendations', lang)}")
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
    lang = get_language()
    token = results["_analysis_token"]
    saved_token = st.session_state.get("_env_saved_token")

    if saved_token == token:
        saved_id = st.session_state.get("_env_saved_id")
        st.success(f"{tr_label('Analysis saved to database (ID:', lang)} {saved_id}).")
        st.button(f"{tr_label('Saved', lang)} ✓", use_container_width=True, disabled=True, key="_env_saved_btn")
        return

    if st.button(tr_label("Save Analysis", lang), type="primary", use_container_width=True, key="_env_save_btn"):
        with safe_action("Saving analysis"):
            with st.spinner("Saving analysis…"):
                record = _build_env_db_record(results)
                analysis_id = insert_environment_analysis(record)
            st.session_state["_env_saved_token"] = token
            st.session_state["_env_saved_id"] = analysis_id
            st.rerun()


def _render_radar(results: dict) -> None:
    """Radar chart of the four factors (0..1 normalized to each crop's band)."""
    lang = get_language()
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
        name=tr_label("Ideal band", lang), showlegend=True,
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
        name=tr_label("Current", lang), showlegend=True,
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
    crop_label = tr_crop(crop, lang)
    caption_template = tr_label(
        "Readings are normalized 0–1 against each factor's safe band "
        "for **{crop}**. The shaded green zone is the ideal range.", lang
    )
    st.caption(caption_template.format(crop=crop_label))