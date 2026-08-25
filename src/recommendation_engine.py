"""Rule-based agricultural recommendation engine.

Pure, deterministic rules — no ML model, no external API, no LLM. Every
recommendation is produced by a plain if/else rule against known crop
ideal ranges (config.ENV_CROP_RANGES) and disease/severity/health-score
lookup tables defined in this file, and every rule returns *why* it fired
alongside the recommendation text, so the output is fully explainable and
auditable.

Inputs used:
    crop, disease, severity, temperature, humidity, soil_moisture,
    rainfall, health_score

Usage:
    from src.recommendation_engine import generate_recommendations

    result = generate_recommendations(
        crop="Tomato", disease="Late_Blight", severity="High",
        temperature=32, humidity=88, soil_moisture=75, rainfall=90,
        health_score=28,
    )
    result["recommendations"]   # list of {text, reason, category, priority}
    result["priority_actions"]  # the high-priority texts, in order
    result["summary"]           # one-line synthesis
"""

from __future__ import annotations

from typing import Any

from config import ENV_CROP_RANGES

# ---------------------------------------------------------------------------
# Rule tables
# ---------------------------------------------------------------------------

# Substring-matched against the disease name (case-insensitive), so any
# disease label containing these words gets relevant, specific advice —
# not just the exact classes the current model happens to output.
DISEASE_KEYWORD_ADVICE: dict[str, str] = {
    "blight":  "Remove and destroy infected foliage promptly, and avoid overhead "
               "irrigation — blight spreads fastest on wet leaves.",
    "rust":    "Remove heavily infected leaves and avoid dense planting that traps "
               "moisture around the canopy.",
    "mold":    "Improve ventilation and reduce canopy density to lower humidity "
               "around the leaves.",
    "mildew":  "Improve airflow, avoid wetting foliage when watering, and apply a "
               "preventive treatment if it continues to spread.",
    "wilt":    "Check root health and drainage — wilting often signals a "
               "soil-borne pathogen or water stress rather than a leaf disease.",
    "spot":    "Remove affected leaves and avoid overhead watering to reduce how "
               "long leaves stay wet.",
    "rot":     "Improve drainage immediately and remove affected tissue — rot "
               "spreads quickly in waterlogged soil.",
}

# Severity -> (extra action text, priority). Applied in addition to the two
# always-on disease actions (inspect + monitor progression) below.
SEVERITY_ACTIONS: dict[str, list[tuple[str, str]]] = {
    "High": [
        ("Apply an appropriate fungicide or treatment immediately, and isolate or "
         "remove severely affected plants to limit spread.", "high"),
        ("Avoid working in the field while foliage is wet to prevent further "
         "spread.", "high"),
    ],
    "Moderate": [
        ("Apply a targeted treatment and re-inspect the affected area every 2-3 "
         "days to confirm it's working.", "medium"),
    ],
    "Mild": [
        ("Hold off on treatment and monitor for progression first — mild, "
         "early-stage cases often resolve with cultural controls alone.", "low"),
    ],
}

# Environmental factor + direction -> (recommendation, reason template).
# `{crop}` / `{value}` / `{bound}` are filled in per reading.
ENV_ADVICE: dict[str, dict[str, tuple[str, str]]] = {
    "temperature": {
        "Low":  ("Protect the crop from cold — use row covers or delay "
                 "temperature-sensitive field operations.",
                 "Temperature is {value}°C, below {crop}'s ideal minimum of {bound}°C."),
        "High": ("Provide shade or irrigate during the cooler parts of the day "
                 "to reduce heat stress.",
                 "Temperature is {value}°C, above {crop}'s ideal maximum of {bound}°C."),
        "Extreme": ("Temperature is far outside the safe range — take immediate "
                    "protective action or delay sensitive field operations.",
                    "Temperature is {value}°C, well outside {crop}'s safe range."),
    },
    "humidity": {
        "Low":  ("Increase humidity around plants (e.g. mulching) or adjust "
                 "irrigation timing.",
                 "Humidity is {value}%, below {crop}'s ideal minimum of {bound}%."),
        "High": ("Improve ventilation to reduce excess humidity and lower "
                 "fungal-disease risk.",
                 "Humidity is {value}%, above {crop}'s ideal maximum of {bound}%."),
        "Extreme": ("Humidity is far outside the safe range — inspect plants "
                    "closely for stress or disease symptoms.",
                    "Humidity is {value}%, well outside {crop}'s safe range."),
    },
    "soil_moisture": {
        "Low":  ("Maintain suitable soil moisture with a consistent irrigation "
                 "schedule.",
                 "Soil moisture is {value}%, below {crop}'s ideal minimum of {bound}%."),
        "High": ("Avoid excessive irrigation — let the soil drain before "
                 "watering again.",
                 "Soil moisture is {value}%, above {crop}'s ideal maximum of {bound}%."),
        "Extreme": ("Correct drainage or watering immediately — soil moisture is "
                    "far outside the safe range.",
                    "Soil moisture is {value}%, well outside {crop}'s safe range."),
    },
    "rainfall": {
        "Low":  ("Supplement rainfall with irrigation to maintain adequate water "
                 "supply.",
                 "Rainfall is {value}mm, below {crop}'s ideal minimum of {bound}mm."),
        "High": ("Ensure adequate field drainage to prevent waterlogging and "
                 "nutrient runoff.",
                 "Rainfall is {value}mm, above {crop}'s ideal maximum of {bound}mm."),
        "Extreme": ("Protect low-lying areas and check for standing water.",
                    "Rainfall is {value}mm, well outside {crop}'s safe range."),
    },
}

FACTOR_UNITS = {"temperature": "°C", "humidity": "%", "soil_moisture": "%", "rainfall": "mm"}

HEALTH_SCORE_ADVICE: list[tuple[int, int, str, str]] = [
    # (lo, hi, recommendation, reason)
    (80, 100, "Overall health is good — maintain current practices and continue "
              "routine monitoring.",
     "Health score is {score}/100 (Healthy band, 80-100)."),
    (60, 79,  "Overall health is moderate — monitor closely and make incremental "
              "adjustments rather than major changes.",
     "Health score is {score}/100 (Moderate band, 60-79)."),
    (40, 59,  "Overall health is at risk — prioritize the largest contributing "
              "factor (disease or environment) and recheck within a few days.",
     "Health score is {score}/100 (At Risk band, 40-59)."),
    (0,  39,  "Overall health is critical — intervene immediately on both disease "
              "treatment and environmental correction.",
     "Health score is {score}/100 (Critical band, 0-39)."),
]

PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}
CATEGORY_ORDER = {"disease": 0, "environment": 1, "overall": 2}

# Display metadata for rendering recommendations in the UI (shared by
# pages/health.py and pages/history.py, so both render the same
# category icon and priority color for a given recommendation).
CATEGORY_ICON = {"disease": "diseased", "environment": "temperature", "overall": "leaf"}
PRIORITY_COLOR = {"high": "#B5564B", "medium": "#C97A3B", "low": "#7FA687"}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _rec(text: str, reason: str, category: str, priority: str) -> dict[str, str]:
    return {"text": text, "reason": reason, "category": category, "priority": priority}


def _factor_status(crop: str, key: str, value: float) -> tuple[str, float]:
    """Return ('Optimal'|'Low'|'High'|'Extreme', the ideal bound that was crossed)."""
    low, opt_min, opt_max, high = ENV_CROP_RANGES[crop][key]
    if opt_min <= value <= opt_max:
        return "Optimal", opt_min
    if low <= value < opt_min:
        return "Low", opt_min
    if opt_max < value <= high:
        return "High", opt_max
    return "Extreme", (low if value < low else high)


def _is_healthy(disease: str, severity: str) -> bool:
    return str(disease).strip().lower() in ("healthy", "none", "") or severity == "None"


# ---------------------------------------------------------------------------
# 1. Disease-based rules
# ---------------------------------------------------------------------------
def _disease_recommendations(crop: str, disease: str, severity: str) -> list[dict]:
    if _is_healthy(disease, severity):
        return [_rec(
            "No disease detected — keep up preventive practices such as crop "
            "rotation and clean tools to avoid introducing pathogens.",
            f"'{disease}' with severity 'None' — no disease signal present.",
            "disease", "low",
        )]

    recs = [
        _rec(
            "Inspect affected leaves closely to confirm the extent of infection.",
            f"'{disease}' detected at {severity} severity.",
            "disease", "high" if severity == "High" else "medium",
        ),
        _rec(
            "Monitor disease progression over the next several days to check "
            "whether treatment is working.",
            f"'{disease}' detected at {severity} severity.",
            "disease", "medium",
        ),
    ]

    disease_lower = str(disease).lower()
    for keyword, advice in DISEASE_KEYWORD_ADVICE.items():
        if keyword in disease_lower:
            recs.append(_rec(
                advice, f"'{disease}' matches known pattern '{keyword}'.",
                "disease", "high" if severity == "High" else "medium",
            ))
            break  # one specific match is enough; avoid piling on near-duplicates

    for text, priority in SEVERITY_ACTIONS.get(severity, []):
        recs.append(_rec(text, f"Severity is {severity}.", "disease", priority))

    return recs


# ---------------------------------------------------------------------------
# 2. Environment-based rules
# ---------------------------------------------------------------------------
def _environmental_recommendations(
    crop: str, temperature: float, humidity: float,
    soil_moisture: float, rainfall: float,
) -> list[dict]:
    readings = {"temperature": temperature, "humidity": humidity,
                "soil_moisture": soil_moisture, "rainfall": rainfall}
    recs: list[dict] = []
    any_off_range = False

    for key, value in readings.items():
        status, bound = _factor_status(crop, key, value)
        if status == "Optimal":
            continue
        any_off_range = True
        text, reason_template = ENV_ADVICE[key][status]
        priority = "high" if status == "Extreme" else "medium"
        reason = reason_template.format(value=value, bound=bound, crop=crop)
        recs.append(_rec(text, reason, "environment", priority))

    if any_off_range:
        recs.append(_rec(
            "Monitor environmental conditions closely over the next few days "
            "and re-check readings after adjustments.",
            "One or more readings fall outside the ideal range for this crop.",
            "environment", "medium",
        ))
    else:
        recs.append(_rec(
            f"All environmental conditions are within {crop}'s ideal range — "
            "maintain current practices.",
            "Temperature, humidity, soil moisture, and rainfall are all optimal.",
            "environment", "low",
        ))

    return recs


# ---------------------------------------------------------------------------
# 3. Health-score-based rule
# ---------------------------------------------------------------------------
def _health_score_recommendation(health_score: int) -> dict:
    for lo, hi, text, reason_template in HEALTH_SCORE_ADVICE:
        if lo <= health_score <= hi:
            return _rec(text, reason_template.format(score=health_score),
                       "overall", "high" if hi <= 39 else "low" if lo >= 80 else "medium")
    return _rec(
        "Re-check the health score input — it should be between 0 and 100.",
        f"Health score {health_score} is outside the expected 0-100 range.",
        "overall", "medium",
    )


# ---------------------------------------------------------------------------
# 4. Combine, dedupe, sort
# ---------------------------------------------------------------------------
def _dedupe(recs: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out = []
    for r in recs:
        if r["text"] not in seen:
            seen.add(r["text"])
            out.append(r)
    return out


def _sort(recs: list[dict]) -> list[dict]:
    return sorted(recs, key=lambda r: (PRIORITY_ORDER[r["priority"]],
                                        CATEGORY_ORDER[r["category"]]))


def _build_summary(crop: str, is_healthy: bool, any_off_range: bool,
                    health_score: int) -> str:
    if is_healthy and not any_off_range:
        return (f"{crop} looks healthy with favorable conditions "
                f"(health score {health_score}/100) — maintain current practices.")
    if is_healthy and any_off_range:
        return (f"{crop} shows no disease, but environmental conditions need "
                f"attention (health score {health_score}/100).")
    if not is_healthy and not any_off_range:
        return (f"{crop} has a detected disease but environmental conditions are "
                f"favorable (health score {health_score}/100) — focus on treatment.")
    return (f"{crop} needs attention on both disease and environmental fronts "
            f"(health score {health_score}/100).")


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------
def generate_recommendations(
    crop: str,
    disease: str,
    severity: str,
    temperature: float,
    humidity: float,
    soil_moisture: float,
    rainfall: float,
    health_score: int,
) -> dict[str, Any]:
    """Generate explainable, rule-based agricultural recommendations.

    No ML model, no external API or LLM — every recommendation comes from
    a fixed lookup table matched against the inputs, with the matching
    condition returned as `reason` for full explainability.

    Returns:
        {
          "recommendations": [{"text","reason","category","priority"}, ...],
          "priority_actions": [str, ...],   # high-priority texts, in order
          "summary": str,
        }
    """
    if crop not in ENV_CROP_RANGES:
        raise ValueError(f"Unknown crop '{crop}'. Expected one of: "
                         f"{', '.join(ENV_CROP_RANGES)}")

    is_healthy = _is_healthy(disease, severity)
    disease_recs = _disease_recommendations(crop, disease, severity)
    env_recs = _environmental_recommendations(crop, temperature, humidity,
                                              soil_moisture, rainfall)
    health_rec = _health_score_recommendation(health_score)

    any_off_range = any(
        _factor_status(crop, k, v)[0] != "Optimal"
        for k, v in [("temperature", temperature), ("humidity", humidity),
                     ("soil_moisture", soil_moisture), ("rainfall", rainfall)]
    )

    all_recs = _sort(_dedupe(disease_recs + env_recs + [health_rec]))
    priority_actions = [r["text"] for r in all_recs if r["priority"] == "high"]

    return {
        "recommendations": all_recs,
        "priority_actions": priority_actions,
        "summary": _build_summary(crop, is_healthy, any_off_range, health_score),
    }


# ---------------------------------------------------------------------------
# Self-test / demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json

    print("--- Example 1: healthy crop, good environment ---")
    print(json.dumps(generate_recommendations(
        crop="Tomato", disease="Healthy", severity="None",
        temperature=24, humidity=60, soil_moisture=55, rainfall=30,
        health_score=93,
    ), indent=2))

    print("\n--- Example 2: severe blight, poor environment ---")
    print(json.dumps(generate_recommendations(
        crop="Rice", disease="Late_Blight", severity="High",
        temperature=39, humidity=91, soil_moisture=18, rainfall=4,
        health_score=17,
    ), indent=2))

    print("\n--- Example 3: mild disease, one factor off ---")
    print(json.dumps(generate_recommendations(
        crop="Corn", disease="Common_Rust", severity="Mild",
        temperature=27, humidity=65, soil_moisture=85, rainfall=50,
        health_score=68,
    ), indent=2))