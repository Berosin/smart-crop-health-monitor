"""Centralized input validation for user-supplied values across the app.

Every page constrains its widgets (selectbox choices, number_input
min/max) so most bad input can't reach these functions through the UI at
all — but the functions here are the actual source of truth for what's
valid, so:
  - the same rules apply if these values ever arrive another way
    (a different entry point, a future API, direct function calls), and
  - every rejection produces one clear, consistent, user-facing message
    instead of a raw ValueError/TypeError from deep inside a widget or
    a model.

All violations raise ValidationError (a ValueError subclass), so existing
`except ValueError` handling anywhere in the app keeps working unchanged,
and src.errors.safe_action() catches it automatically.
"""

from __future__ import annotations

from config import ENV_CROP_RANGES, ENV_RANGES

VALID_SEVERITIES = {"None", "Moderate", "High"}
VALID_CROPS = list(ENV_CROP_RANGES.keys())


class ValidationError(ValueError):
    """A specific, user-facing input validation failure."""


# ---------------------------------------------------------------------------
# Crop selection
# ---------------------------------------------------------------------------
def validate_crop(crop) -> str:
    if not isinstance(crop, str) or not crop.strip():
        raise ValidationError("Please select a crop.")
    if crop not in ENV_CROP_RANGES:
        raise ValidationError(
            f"Unknown crop '{crop}'. Choose one of: {', '.join(VALID_CROPS)}."
        )
    return crop


# ---------------------------------------------------------------------------
# Environmental readings
# ---------------------------------------------------------------------------
def _validate_numeric_field(name: str, label: str, value, unit: str) -> list[str]:
    """Return a list of problem descriptions for one field (empty = valid)."""
    if value is None or isinstance(value, bool):
        return [f"{label} is required."]
    try:
        value = float(value)
    except (TypeError, ValueError):
        return [f"{label} must be a number."]
    if value != value:  # NaN check without importing math
        return [f"{label} must be a number (received NaN)."]

    bounds = ENV_RANGES[name]
    if value < bounds["min"] or value > bounds["max"]:
        return [
            f"{label} must be between {bounds['min']}{unit} and "
            f"{bounds['max']}{unit} (got {value}{unit})."
        ]
    return []


def validate_environmental_reading(
    temperature, humidity, soil_moisture, rainfall,
) -> dict[str, float]:
    """Validate all four environmental readings together.

    Collects every problem found (not just the first) into one combined
    error message, so the user sees everything wrong at once rather than
    fixing issues one at a time.

    Returns the four values as floats on success.
    """
    fields = {
        "temperature":   ("Temperature", temperature, ENV_RANGES["temperature"]["unit"]),
        "humidity":      ("Humidity", humidity, ENV_RANGES["humidity"]["unit"]),
        "soil_moisture": ("Soil moisture", soil_moisture, ENV_RANGES["soil_moisture"]["unit"]),
        "rainfall":      ("Rainfall", rainfall, ENV_RANGES["rainfall"]["unit"]),
    }

    problems: list[str] = []
    for name, (label, value, unit) in fields.items():
        problems.extend(_validate_numeric_field(name, label, value, unit))

    if problems:
        raise ValidationError(" ".join(problems))

    return {
        "temperature": float(temperature),
        "humidity": float(humidity),
        "soil_moisture": float(soil_moisture),
        "rainfall": float(rainfall),
    }


# ---------------------------------------------------------------------------
# Disease-result fields (confidence / severity)
# ---------------------------------------------------------------------------
def validate_confidence(confidence) -> float:
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        raise ValidationError("Confidence must be a number between 0 and 1.")
    if not (0.0 <= confidence <= 1.0):
        raise ValidationError(
            f"Confidence must be between 0 and 1 (got {confidence})."
        )
    return confidence


def validate_severity(severity) -> str:
    if severity not in VALID_SEVERITIES:
        raise ValidationError(
            f"Unknown severity '{severity}'. Expected one of: "
            f"{', '.join(sorted(VALID_SEVERITIES))}."
        )
    return severity


def validate_disease_name(disease) -> str:
    if not isinstance(disease, str) or not disease.strip():
        raise ValidationError("Disease name is required.")
    return disease.strip()


# ---------------------------------------------------------------------------
# Health score
# ---------------------------------------------------------------------------
def validate_health_score(score) -> int:
    try:
        score = float(score)
    except (TypeError, ValueError):
        raise ValidationError("Health score must be a number between 0 and 100.")
    if not (0 <= score <= 100):
        raise ValidationError(f"Health score must be between 0 and 100 (got {score}).")
    return int(round(score))