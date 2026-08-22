"""Crop health analysis engine — combines disease and environmental signals
into a single explainable 0-100 health score.

Design goals:
- Modular: each input signal is scored independently by its own small,
  pure function (compute_disease_risk_score / compute_environmental_risk_score),
  then combined by compute_health_score(). Every step can be tested and
  reasoned about in isolation.
- Explainable: every score-producing function returns *why* it produced
  that number, not just the number, and analyze_crop_health() rolls those
  up into a single plain-language explanation alongside the structured
  result.

Inputs used:
- Disease prediction (name), disease confidence, disease severity
  (typically the dict returned by pages/disease.py's predict_disease())
- Environmental risk level (+ optional probability/probabilities), typically
  the dict returned by src/environment_model.py's predict_environmental_risk()
- The four raw readings (temperature, humidity, soil moisture, rainfall) —
  used to enrich the explanation/recommendation with concrete figures, not
  to recompute environmental risk (that responsibility stays in
  environment_model.py; this module composes already-scored signals).

Health score classification:
    80-100  Healthy
    60-79   Moderate
    40-59   At Risk
    0-39    Critical
"""

from __future__ import annotations

from typing import Any

from config import ENV_CROP_RANGES, ENV_RANGES
from src.environment_model import FACTOR_LABELS, NUMERIC_COLUMNS, TIPS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# How much each signal contributes to the overall health score.
WEIGHTS = {"disease": 0.55, "environmental": 0.45}

# Disease severity -> how much it drags the disease score down (0 = none).
SEVERITY_WEIGHTS = {"None": 0.0, "Mild": 0.3, "Moderate": 0.6, "High": 0.9}

# Fallback disease advice, used only when the caller doesn't supply its own
# disease_recommendation (e.g. from pages/disease.py's RECOMMENDATION_MAP).
FALLBACK_DISEASE_ADVICE = {
    "None": "Crop appears healthy. Maintain regular monitoring and balanced irrigation.",
    "Mild": "Early-stage symptoms detected; monitor closely and consider a preventive treatment.",
    "Moderate": "Apply an appropriate fungicide/treatment and improve air circulation around affected plants.",
    "High": "Severity is high — remove/destroy severely affected plant material and treat promptly to limit spread.",
}

# Representative 0-100 anchor score for each environmental risk class,
# used to convert the environment model's categorical prediction into a
# numeric contribution. When the model's full probability distribution is
# available, we use a probability-weighted average of these anchors instead
# of just the single predicted class, which is a more honest (and more
# explainable) estimate than snapping to one bucket.
ENV_RISK_ANCHORS = {"Optimal": 95, "Low": 75, "Moderate": 55, "High": 30, "Critical": 10}

# Health score classification bands, in the exact form requested.
HEALTH_STATUS_BANDS = [
    (80, 100, "Healthy"),
    (60, 79,  "Moderate"),
    (40, 59,  "At Risk"),
    (0,  39,  "Critical"),
]

# Same five-bucket vocabulary the environment model uses, reused here so a
# disease score can be described with the same risk language.
_RISK_LABEL_BANDS = [
    (85, 100, "Optimal"),
    (65, 84,  "Low"),
    (40, 64,  "Moderate"),
    (20, 39,  "High"),
    (0,  19,  "Critical"),
]


def _clamp(value: float, lo: float = 0, hi: float = 100) -> float:
    return max(lo, min(hi, value))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _score_to_risk_label(score: float) -> str:
    for lo, hi, label in _RISK_LABEL_BANDS:
        if lo <= score <= hi:
            return label
    return "Critical"


# ---------------------------------------------------------------------------
# 1. Disease signal -> score (modular, independently testable)
# ---------------------------------------------------------------------------
def compute_disease_risk_score(
    disease_prediction: str,
    disease_confidence: float,
    disease_severity: str,
) -> tuple[int, str, bool]:
    """Score the disease signal on a 0-100 scale (100 = no disease risk).

    Returns (score, explanation, is_healthy).
    """
    confidence = _clamp01(disease_confidence)
    is_healthy = (str(disease_prediction).strip().lower() == "healthy"
                  or disease_severity == "None")

    if is_healthy:
        # A confident "healthy" call is reassuring; an unsure one still
        # leaves some doubt, so it's scored slightly lower.
        score = 85 + 15 * confidence
        explanation = (
            f"Model predicts '{disease_prediction}' with {confidence*100:.0f}% confidence — "
            "no disease penalty applied."
        )
    else:
        sev_weight = SEVERITY_WEIGHTS.get(disease_severity, 0.6)
        score = 100 - (sev_weight * 70) - (confidence * 30)
        explanation = (
            f"'{disease_prediction}' detected at {disease_severity} severity with "
            f"{confidence*100:.0f}% confidence — both severity and confidence increase risk."
        )

    return int(round(_clamp(score))), explanation, is_healthy


# ---------------------------------------------------------------------------
# 2. Environmental signal -> score (modular, independently testable)
# ---------------------------------------------------------------------------
def compute_environmental_risk_score(
    environmental_risk: str,
    environmental_probability: float | None = None,
    environmental_probabilities: dict[str, float] | None = None,
) -> tuple[int, str]:
    """Score the environmental signal on a 0-100 scale (100 = no risk).

    Returns (score, explanation).
    """
    if environmental_probabilities:
        score = sum(
            ENV_RISK_ANCHORS.get(label, 50) * prob
            for label, prob in environmental_probabilities.items()
        )
        basis = "a probability-weighted average across all predicted risk classes"
    else:
        score = ENV_RISK_ANCHORS.get(environmental_risk, 50)
        basis = "the single predicted risk class"

    conf_text = (f" ({environmental_probability*100:.0f}% model confidence)"
                 if environmental_probability is not None else "")
    explanation = (
        f"Environmental risk classified as {environmental_risk}{conf_text}, "
        f"scored using {basis}."
    )
    return int(round(_clamp(score))), explanation


# ---------------------------------------------------------------------------
# 3. Combine signals -> overall health score (modular, independently testable)
# ---------------------------------------------------------------------------
def compute_health_score(disease_score: int, environmental_score: int,
                          weights: dict[str, float] = WEIGHTS) -> int:
    """Weighted blend of the two component scores into the final 0-100 score."""
    blended = weights["disease"] * disease_score + weights["environmental"] * environmental_score
    return int(round(_clamp(blended)))


def classify_health_status(score: int) -> str:
    """Bucket a 0-100 health score into Healthy / Moderate / At Risk / Critical."""
    for lo, hi, label in HEALTH_STATUS_BANDS:
        if lo <= score <= hi:
            return label
    return "Critical"


# ---------------------------------------------------------------------------
# 4. Recommendation + explanation builders
# ---------------------------------------------------------------------------
def _out_of_range_factors(crop: str | None, readings: dict[str, float]) -> list[str]:
    """Which of the four readings fall outside `crop`'s ideal band, and how."""
    if not crop or crop not in ENV_CROP_RANGES:
        return []
    tips: list[str] = []
    ranges = ENV_CROP_RANGES[crop]
    for key in NUMERIC_COLUMNS:
        if key not in readings or readings[key] is None:
            continue
        value = readings[key]
        low, opt_min, opt_max, high = ranges[key]
        if opt_min <= value <= opt_max:
            continue
        status = "Low" if value < opt_min and value >= low else \
                 "High" if value > opt_max and value <= high else "Extreme"
        tip = TIPS.get(key, {}).get(status)
        if tip:
            tips.append(tip.format(crop=crop.lower()))
    return tips


def _build_recommendation(
    disease_severity: str,
    is_healthy: bool,
    disease_recommendation: str | None,
    environmental_recommendation: str | None,
    crop: str | None,
    readings: dict[str, float],
    status: str,
) -> str:
    parts: list[str] = []

    parts.append(disease_recommendation or FALLBACK_DISEASE_ADVICE.get(
        disease_severity, FALLBACK_DISEASE_ADVICE["None" if is_healthy else "Moderate"]
    ))

    if environmental_recommendation:
        parts.append(environmental_recommendation)
    else:
        parts.extend(_out_of_range_factors(crop, readings))

    closing = {
        "Healthy":  "Overall health is good — maintain current practices and keep monitoring.",
        "Moderate": "Overall health is moderate — monitor closely and make incremental adjustments.",
        "At Risk":  "Overall health is at risk — address the largest contributing factor first and recheck within a few days.",
        "Critical": "Overall health is critical — prioritize immediate intervention on both disease and environmental factors.",
    }[status]
    parts.append(closing)

    return " ".join(parts)


def _build_explanation(
    disease_score: int, disease_explanation: str,
    environmental_score: int, environmental_explanation: str,
    health_score: int, status: str, weights: dict[str, float],
) -> str:
    return (
        f"{disease_explanation} This contributes a disease score of {disease_score}/100. "
        f"{environmental_explanation} This contributes an environmental score of "
        f"{environmental_score}/100. Weighted {weights['disease']*100:.0f}% disease / "
        f"{weights['environmental']*100:.0f}% environment, the overall health score is "
        f"{health_score}/100, classified as '{status}'."
    )


# ---------------------------------------------------------------------------
# 5. Top-level engine entry point
# ---------------------------------------------------------------------------
def analyze_crop_health(
    disease_prediction: str,
    disease_confidence: float,
    disease_severity: str,
    environmental_risk: str,
    temperature: float,
    humidity: float,
    soil_moisture: float,
    rainfall: float,
    *,
    crop: str | None = None,
    environmental_probability: float | None = None,
    environmental_probabilities: dict[str, float] | None = None,
    disease_recommendation: str | None = None,
    environmental_recommendation: str | None = None,
    weights: dict[str, float] = WEIGHTS,
) -> dict[str, Any]:
    """Combine a disease result and an environmental risk result into one
    explainable crop health assessment.

    Args:
        disease_prediction: predicted disease name (or "Healthy").
        disease_confidence: model confidence for that prediction, 0..1.
        disease_severity: "None" | "Mild" | "Moderate" | "High".
        environmental_risk: "Optimal" | "Low" | "Moderate" | "High" | "Critical"
            (the risk_level already produced by
            src.environment_model.predict_environmental_risk()).
        temperature, humidity, soil_moisture, rainfall: the raw readings
            behind that environmental risk call — used here only to enrich
            the explanation/recommendation with concrete figures, not to
            recompute risk (single-responsibility: environment_model.py
            owns that calculation).
        crop: optional, enables factor-specific recommendations.
        environmental_probability / environmental_probabilities: optional,
            for a probability-weighted (rather than single-bucket) score.
        disease_recommendation / environmental_recommendation: optional
            upstream recommendations (e.g. from predict_disease() /
            predict_environmental_risk()) to fold into the final advice.
        weights: override the disease/environment blend (must sum to 1.0).

    Returns:
        {
          "health_score": int,          # 0-100
          "health_status": str,         # Healthy / Moderate / At Risk / Critical
          "disease_risk": {"score": int, "level": str, "explanation": str, ...},
          "environmental_risk": {"score": int, "level": str, "explanation": str, ...},
          "recommendation": str,
          "explanation": str,           # plain-language, step-by-step reasoning
          "weights": dict,
        }
    """
    readings = {
        "temperature": temperature, "humidity": humidity,
        "soil_moisture": soil_moisture, "rainfall": rainfall,
    }

    disease_score, disease_explanation, is_healthy = compute_disease_risk_score(
        disease_prediction, disease_confidence, disease_severity,
    )
    environmental_score, environmental_explanation = compute_environmental_risk_score(
        environmental_risk, environmental_probability, environmental_probabilities,
    )

    health_score = compute_health_score(disease_score, environmental_score, weights)
    health_status = classify_health_status(health_score)

    recommendation = _build_recommendation(
        disease_severity, is_healthy, disease_recommendation,
        environmental_recommendation, crop, readings, health_status,
    )
    explanation = _build_explanation(
        disease_score, disease_explanation, environmental_score,
        environmental_explanation, health_score, health_status, weights,
    )

    return {
        "health_score": health_score,
        "health_status": health_status,
        "disease_risk": {
            "score": disease_score,
            "level": _score_to_risk_label(disease_score),
            "prediction": disease_prediction,
            "confidence": round(_clamp01(disease_confidence), 4),
            "severity": disease_severity,
            "explanation": disease_explanation,
        },
        "environmental_risk": {
            "score": environmental_score,
            "level": environmental_risk,
            "probability": environmental_probability,
            "readings": readings,
            "explanation": environmental_explanation,
        },
        "recommendation": recommendation,
        "explanation": explanation,
        "weights": weights,
    }


def analyze_from_predictions(disease: dict[str, Any], environmental: dict[str, Any],
                              crop: str | None = None) -> dict[str, Any]:
    """Convenience adapter: build the engine call directly from the two
    model outputs already used elsewhere in the app —
    pages/disease.py's predict_disease() and
    src/environment_model.py's predict_environmental_risk() — with no glue
    code needed at the call site.
    """
    readings = environmental.get("readings", {})
    return analyze_crop_health(
        disease_prediction=disease.get("disease", "Unknown"),
        disease_confidence=disease.get("confidence", 0.0),
        disease_severity=disease.get("severity", "Moderate"),
        environmental_risk=environmental.get("risk_level", "Moderate"),
        temperature=readings.get("temperature", 0.0),
        humidity=readings.get("humidity", 0.0),
        soil_moisture=readings.get("soil_moisture", 0.0),
        rainfall=readings.get("rainfall", 0.0),
        crop=crop,
        environmental_probability=environmental.get("probability"),
        environmental_probabilities=environmental.get("probabilities"),
        disease_recommendation=disease.get("recommendation"),
        environmental_recommendation=environmental.get("recommendation"),
    )


# ---------------------------------------------------------------------------
# Self-test / demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json

    print("--- Example 1: healthy crop, good environment ---")
    result = analyze_crop_health(
        disease_prediction="Healthy", disease_confidence=0.95, disease_severity="None",
        environmental_risk="Optimal", temperature=24, humidity=60,
        soil_moisture=55, rainfall=30, crop="Tomato",
        environmental_probability=0.88,
    )
    print(json.dumps(result, indent=2))

    print("\n--- Example 2: diseased crop, poor environment ---")
    result = analyze_crop_health(
        disease_prediction="Late_Blight", disease_confidence=0.91, disease_severity="High",
        environmental_risk="Critical", temperature=38, humidity=92,
        soil_moisture=20, rainfall=5, crop="Rice",
        environmental_probability=0.79,
    )
    print(json.dumps(result, indent=2))