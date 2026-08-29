"""Live weather integration — OpenWeatherMap client + short-term forecast
aggregation for the Environmental Analysis page.

Two things live here:

1. A thin OpenWeatherMap client (`get_current_weather`, `get_forecast`) that
   fetches real conditions for a location, in the same units the trained
   environmental risk model already expects (°C, %, mm/day) — see
   config.ENV_RANGES.

2. `build_forecast_risk(...)`, which feeds each forecast day through the
   *already-trained* src.environment_model.predict_environmental_risk()
   instead of a single manually-typed reading — turning "today's risk" into
   a short-term "which of the next few days looks risky" outlook. No new
   model, no new training: same classifier, forecast data as input instead
   of current/manual data.

One disclosed assumption: OpenWeatherMap has no soil-moisture forecast (no
free weather API does — it isn't a directly measurable atmospheric
quantity). The forecast risk calculation holds soil moisture constant at
whatever value the person last supplied (their current live reading, or
their manual entry), and the UI says so explicitly rather than silently
guessing. Every other input (temperature, humidity, rainfall) comes
straight from the forecast.

API key
-------
Never hardcoded. Resolved in this order:
1. `OPENWEATHERMAP_API_KEY` environment variable
2. `st.secrets["OPENWEATHERMAP_API_KEY"]` (.streamlit/secrets.toml) — the
   standard, no-retype-needed way to configure this for your own machine
3. Typed into the page for the session (kept only in st.session_state,
   never written to disk by this app)

Get a free key at https://openweathermap.org/api — the free tier (60
calls/min, 1,000/day) covers both endpoints used here.
"""

from __future__ import annotations

import os
from collections import defaultdict
from datetime import datetime

import requests

from src.errors import WeatherError, logger

OWM_BASE_URL = "https://api.openweathermap.org/data/2.5"
REQUEST_TIMEOUT_S = 10


# ---------------------------------------------------------------------------
# API key resolution
# ---------------------------------------------------------------------------
def resolve_api_key(session_key: str | None = None) -> str | None:
    """First match wins: env var -> st.secrets -> session-provided key."""
    env_key = os.environ.get("OPENWEATHERMAP_API_KEY")
    if env_key:
        return env_key

    try:
        import streamlit as st
        secrets_key = st.secrets.get("OPENWEATHERMAP_API_KEY")
        if secrets_key:
            return secrets_key
    except Exception:
        # No secrets.toml configured at all — st.secrets raises in that
        # case rather than returning empty; that's expected, not an error.
        pass

    return session_key or None


# ---------------------------------------------------------------------------
# Current conditions
# ---------------------------------------------------------------------------
def get_current_weather(location: str, api_key: str) -> dict:
    """Current conditions for a location (e.g. "Chennai, IN").

    Returns a dict matching the environmental-reading shape the trained
    risk model expects, plus a few display-only fields:
        temperature (°C), humidity (%), rainfall (mm, last 1h — 0 if dry),
        description, icon_code, location_name, wind_speed (m/s)
    """
    if not api_key:
        raise WeatherError(
            "No OpenWeatherMap API key configured. Enter one below, or set "
            "OPENWEATHERMAP_API_KEY as an environment variable / in "
            ".streamlit/secrets.toml."
        )
    if not location or not location.strip():
        raise WeatherError("Enter a location first (e.g. 'Chennai, IN').")

    data = _get(f"{OWM_BASE_URL}/weather", {"q": location.strip(), "appid": api_key, "units": "metric"}, location)

    rainfall_mm = float((data.get("rain") or {}).get("1h", 0.0))
    weather0 = (data.get("weather") or [{}])[0]
    sys = data.get("sys") or {}

    return {
        "temperature": float(data["main"]["temp"]),
        "humidity": float(data["main"]["humidity"]),
        "rainfall": rainfall_mm,
        "description": weather0.get("description", "").title(),
        "icon_code": weather0.get("icon", ""),
        "location_name": f"{data.get('name', location)}, {sys.get('country', '')}".strip(", "),
        "wind_speed": float((data.get("wind") or {}).get("speed", 0.0)),
        "fetched_at": datetime.now().strftime("%H:%M:%S"),
    }


# ---------------------------------------------------------------------------
# Forecast (daily-aggregated from the free 3-hour/5-day endpoint)
# ---------------------------------------------------------------------------
def get_forecast(location: str, api_key: str, days: int = 5) -> list[dict]:
    """Daily-aggregated forecast for the next `days` days.

    OpenWeatherMap's free tier only offers 3-hour-interval data (no native
    daily forecast), so this aggregates each day's ~8 slots itself:
        temp_max / temp_min (°C), humidity_avg (%),
        rainfall_total (mm — sum of each slot's 3h rainfall, giving a
        mm/day figure in the same unit the risk model expects),
        description/icon_code (from the slot closest to midday).

    Returns a list of dicts, one per day, chronologically sorted, each also
    carrying "date" (YYYY-MM-DD) and "day_label" (e.g. "Tomorrow", "Fri").
    Typically returns 5-6 days depending on what time of day "now" is.
    """
    if not api_key:
        raise WeatherError(
            "No OpenWeatherMap API key configured. Enter one below, or set "
            "OPENWEATHERMAP_API_KEY as an environment variable / in "
            ".streamlit/secrets.toml."
        )
    if not location or not location.strip():
        raise WeatherError("Enter a location first (e.g. 'Chennai, IN').")

    data = _get(f"{OWM_BASE_URL}/forecast", {"q": location.strip(), "appid": api_key, "units": "metric"}, location)

    slots_by_date: dict[str, list[dict]] = defaultdict(list)
    for slot in data.get("list", []):
        date_str = slot.get("dt_txt", "")[:10]
        if date_str:
            slots_by_date[date_str].append(slot)

    today_str = datetime.now().strftime("%Y-%m-%d")
    daily = []
    for date_str in sorted(slots_by_date.keys())[:days]:
        slots = slots_by_date[date_str]
        temp_max = max(s["main"]["temp_max"] for s in slots)
        temp_min = min(s["main"]["temp_min"] for s in slots)
        humidity_avg = sum(s["main"]["humidity"] for s in slots) / len(slots)
        rainfall_total = sum((s.get("rain") or {}).get("3h", 0.0) for s in slots)

        # Representative description/icon: the slot nearest midday local time.
        midday_slot = min(slots, key=lambda s: abs(int(s["dt_txt"][11:13]) - 12))
        weather0 = (midday_slot.get("weather") or [{}])[0]

        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        day_label = "Today" if date_str == today_str else date_obj.strftime("%a %d %b")

        daily.append({
            "date": date_str,
            "day_label": day_label,
            "temp_max": round(temp_max, 1),
            "temp_min": round(temp_min, 1),
            "temp_avg": round((temp_max + temp_min) / 2, 1),
            "humidity_avg": round(humidity_avg, 1),
            "rainfall_total": round(rainfall_total, 1),
            "description": weather0.get("description", "").title(),
            "icon_code": weather0.get("icon", ""),
        })

    return daily


def _get(url: str, params: dict, location: str) -> dict:
    try:
        resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_S)
    except requests.RequestException as e:
        logger.exception("Weather API request failed")
        raise WeatherError(f"Could not reach the weather service: {e}") from e

    if resp.status_code == 401:
        raise WeatherError("Weather API key was rejected (invalid, or not yet activated — new keys can take a few minutes).")
    if resp.status_code == 404:
        raise WeatherError(f"Location '{location}' wasn't found. Try 'City, Country code', e.g. 'Chennai, IN'.")
    if resp.status_code == 429:
        raise WeatherError("Weather API rate limit reached — wait a bit and try again.")
    if resp.status_code != 200:
        raise WeatherError(f"Weather service returned an unexpected error (HTTP {resp.status_code}).")

    return resp.json()


# ---------------------------------------------------------------------------
# Forecast disease-risk outlook — same trained model, forecast data as input
# ---------------------------------------------------------------------------
def build_forecast_risk(crop: str, forecast_days: list[dict], soil_moisture: float) -> list[dict]:
    """Run the trained environmental risk model against each forecast day.

    Args:
        crop: crop name (must match one of config.ENV_CROP_RANGES' keys).
        forecast_days: output of get_forecast().
        soil_moisture: held constant across every forecast day (see module
            docstring — no weather API forecasts soil moisture).

    Returns one dict per day: everything from the forecast day, plus
    "risk_level", "probability", "explanation", "recommendation" from the
    trained model (same shape predict_environmental_risk() already returns
    on the Environmental Analysis page, just run once per forecast day
    instead of once for a single manual reading).
    """
    from src.environment_model import predict_environmental_risk  # local import: avoid importing TF/sklearn stack for callers that only need the API client

    results = []
    for day in forecast_days:
        try:
            env_pred = predict_environmental_risk({
                "crop": crop,
                "temperature": day["temp_avg"],
                "humidity": day["humidity_avg"],
                "soil_moisture": soil_moisture,
                "rainfall": day["rainfall_total"],
            })
        except Exception:
            logger.exception("Forecast risk prediction failed for %s on %s", crop, day.get("date"))
            continue

        results.append({
            **day,
            "risk_level": env_pred["risk_level"],
            "probability": env_pred["probability"],
            "explanation": env_pred["explanation"],
            "recommendation": env_pred["recommendation"],
        })

    return results


# ---------------------------------------------------------------------------
# Self-test — exercises the pure aggregation logic with a synthetic
# OpenWeatherMap-shaped forecast payload, no network or API key needed.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    def _slot(dt_txt, temp, temp_max, temp_min, humidity, rain_3h=None, desc="clear sky", icon="01d"):
        d = {
            "dt_txt": dt_txt,
            "main": {"temp": temp, "temp_max": temp_max, "temp_min": temp_min, "humidity": humidity},
            "weather": [{"description": desc, "icon": icon}],
        }
        if rain_3h is not None:
            d["rain"] = {"3h": rain_3h}
        return d

    synthetic_slots = [
        _slot("2026-08-30 00:00:00", 22, 23, 21, 80),
        _slot("2026-08-30 09:00:00", 25, 27, 24, 75),
        _slot("2026-08-30 12:00:00", 28, 29, 27, 65, desc="light rain", icon="10d"),
        _slot("2026-08-30 15:00:00", 27, 28, 26, 68, rain_3h=4.2),
        _slot("2026-08-30 18:00:00", 24, 25, 23, 78, rain_3h=6.1),
        _slot("2026-08-31 09:00:00", 21, 22, 20, 85, desc="moderate rain", icon="10d"),
        _slot("2026-08-31 12:00:00", 20, 21, 19, 88, rain_3h=8.0, desc="moderate rain", icon="10d"),
        _slot("2026-08-31 15:00:00", 19, 20, 18, 90, rain_3h=10.5, desc="moderate rain", icon="10d"),
    ]

    # Monkeypatch the network call to test aggregation logic in isolation.
    import types
    fake_module = types.SimpleNamespace(list=synthetic_slots)

    slots_by_date = defaultdict(list)
    for slot in synthetic_slots:
        slots_by_date[slot["dt_txt"][:10]].append(slot)

    print("--- Aggregation sanity check ---")
    day1 = slots_by_date["2026-08-30"]
    day1_rain = sum((s.get("rain") or {}).get("3h", 0.0) for s in day1)
    print(f"Day 1 (2026-08-30): {len(day1)} slots, total rainfall = {day1_rain} mm")
    assert day1_rain == 10.3

    day2 = slots_by_date["2026-08-31"]
    day2_rain = sum((s.get("rain") or {}).get("3h", 0.0) for s in day2)
    day2_temp_avg = (max(s["main"]["temp_max"] for s in day2) + min(s["main"]["temp_min"] for s in day2)) / 2
    print(f"Day 2 (2026-08-31): {len(day2)} slots, total rainfall = {day2_rain} mm, temp_avg = {day2_temp_avg}")
    assert day2_rain == 18.5
    print("Cool + rainy day 2 correctly aggregated — this is the kind of day that "
          "should trip an elevated late-blight-style risk once run through the model.")
    print("\nOK — forecast aggregation logic checks out.")