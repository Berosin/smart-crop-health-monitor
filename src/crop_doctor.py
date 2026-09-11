"""Crop Doctor — lightweight, rule-based retrieval-augmented Q&A over one
saved analysis record.

No external LLM / API calls, no per-message cost, nothing "generic":

1. classify_intent(question) — a small keyword classifier (English + Tamil)
   decides which of a fixed set of question types was asked. This is the
   "retrieval" step: it retrieves *which of this app's own explainable
   engines* should answer, not a document from a vector store.
2. An intent handler answers by calling straight into this app's existing
   explainable domain logic (src.health_engine, src.recommendation_engine,
   src.yield_loss, config.ENV_CROP_RANGES) using the real stored numbers
   from the selected saved record — so every answer is grounded in that
   specific analysis, not a generic chatbot reply.

Same "small pure function, independently testable" shape as
health_engine.py / outbreak_detection.py / yield_loss.py — this module has
no Streamlit dependency; pages/crop_doctor.py is presentation only.
"""

from __future__ import annotations

import json
from typing import Any

from config import ENV_CROP_RANGES
from src.health_engine import (
    compute_disease_risk_score,
    classify_health_status,
    _out_of_range_factors,
)
from src.recommendation_engine import generate_recommendations
from src.yield_loss import get_yield_loss_range, estimate_yield_loss, REFERENCE_YIELD_T_PER_HA

RECORD_TYPES = ("health", "disease", "environment", "field_scan")

# Same action vocabulary pages/disease.py's SEVERITY_META already uses —
# reusing the exact strings means tr_severity_action() (see src/i18n.py)
# already has these translated, no new table needed for this part.
SEVERITY_ACTION_EN = {
    "None": "No action needed",
    "Mild": "Monitor closely",
    "Moderate": "Treat promptly",
    "High": "Intervene urgently",
}

# Rough spread-risk classification, used only for the "will this spread"
# intent — a simple lookup table, same style as pages/disease.py's
# SEVERITY_MAP, not a new model.
SPREAD_RISK = {
    "Late_Blight": "fast", "Leaf_Blast": "fast", "Neck_Blast": "fast", "Yellow_Rust": "fast",
    "Early_Blight": "moderate", "Brown_Spot": "moderate", "Brown_Rust": "moderate",
    "Common_Rust": "moderate", "Gray_Leaf_Spot": "moderate", "Northern_Leaf_Blight": "moderate",
}

ENV_FACTOR_LABELS_EN = {
    "temperature": "Temperature", "humidity": "Humidity",
    "soil_moisture": "Soil moisture", "rainfall": "Rainfall",
}
ENV_FACTOR_UNITS = {"temperature": "°C", "humidity": "%", "soil_moisture": "%", "rainfall": "mm"}


# ---------------------------------------------------------------------------
# Record normalization — flatten any of the four src.db row shapes into one
# uniform "facts" dict, so every handler below works the same regardless of
# which page originally saved the record.
# ---------------------------------------------------------------------------
def normalize_record(record_type: str, row: dict) -> dict:
    """Build the uniform facts dict an answer_* handler reasons over."""
    facts: dict[str, Any] = {
        "record_type": record_type,
        "id": row.get("id"),
        "crop": row.get("crop_name"),
        "created_at": row.get("created_at"),
        "disease": None, "severity": None, "confidence": None,
        "health_score": None, "disease_risk": None, "environmental_risk": None,
        "temperature": None, "humidity": None, "soil_moisture": None, "rainfall": None,
        "risk_level": None, "probability": None,
        "num_images": None, "healthy_pct": None, "dominant_disease": None,
        "severity_breakdown": {}, "disease_breakdown": {},
        "stored_recommendation": row.get("recommendation"),
    }

    if record_type == "health":
        facts.update({
            "disease": row.get("disease"), "severity": row.get("severity"),
            "confidence": row.get("confidence"), "health_score": row.get("health_score"),
            "disease_risk": row.get("disease_risk"), "environmental_risk": row.get("environmental_risk"),
            "temperature": row.get("temperature"), "humidity": row.get("humidity"),
            "soil_moisture": row.get("soil_moisture"), "rainfall": row.get("rainfall"),
        })
    elif record_type == "disease":
        facts.update({
            "disease": row.get("disease"), "severity": row.get("severity"),
            "confidence": row.get("confidence"),
        })
    elif record_type == "environment":
        facts.update({
            "temperature": row.get("temperature"), "humidity": row.get("humidity"),
            "soil_moisture": row.get("soil_moisture"), "rainfall": row.get("rainfall"),
            "risk_level": row.get("risk_level"), "probability": row.get("probability"),
            "health_score": row.get("health_score"),
        })
    elif record_type == "field_scan":
        try:
            facts["severity_breakdown"] = json.loads(row.get("severity_breakdown") or "{}")
        except (TypeError, ValueError):
            pass
        try:
            facts["disease_breakdown"] = json.loads(row.get("disease_breakdown") or "{}")
        except (TypeError, ValueError):
            pass
        facts.update({
            "num_images": row.get("num_images"), "num_healthy": row.get("num_healthy"),
            "num_diseased": row.get("num_diseased"), "healthy_pct": row.get("healthy_pct"),
            "dominant_disease": row.get("dominant_disease"),
            "health_score": row.get("field_health_score"),
        })
        # For severity/yield-loss questions on a field scan, treat the
        # dominant disease at its worst *observed* severity as
        # representative — matches pages/field_scan.py's own "don't
        # understate risk" choice for its yield-loss estimator.
        sb = facts["severity_breakdown"]
        facts["disease"] = facts["dominant_disease"]
        facts["severity"] = "High" if sb.get("High") else "Moderate" if sb.get("Moderate") else "None"

    return facts


def record_summary_line(facts: dict, lang: str) -> str:
    """One-line summary of a record, for a selection dropdown."""
    from src.i18n import tr_crop, tr_disease, tr_label

    crop_label = tr_crop(facts["crop"], lang) if facts["crop"] else "?"
    date = (facts.get("created_at") or "")[:10]
    rt = facts["record_type"]

    if rt == "health":
        disease_label = tr_disease(facts["disease"], lang) if facts["disease"] else tr_label("Unknown", lang)
        return f"#{facts['id']} · {crop_label} · {disease_label} · {facts.get('health_score')}/100 · {date}"
    if rt == "disease":
        disease_label = tr_disease(facts["disease"], lang) if facts["disease"] else tr_label("Unknown", lang)
        conf = facts.get("confidence")
        conf_txt = f"{conf*100:.0f}%" if conf is not None else "—"
        return f"#{facts['id']} · {crop_label} · {disease_label} · {conf_txt} · {date}"
    if rt == "environment":
        risk_label = tr_label(facts["risk_level"], lang) if facts["risk_level"] else tr_label("Unknown", lang)
        return f"#{facts['id']} · {crop_label} · {risk_label} · {date}"
    if rt == "field_scan":
        dom = facts.get("dominant_disease")
        dom_label = tr_disease(dom, lang) if dom else tr_label("None detected", lang)
        return f"#{facts['id']} · {crop_label} · {dom_label} · {facts.get('healthy_pct')}% {tr_label('healthy', lang)} · {date}"
    return f"#{facts['id']} · {crop_label} · {date}"


def _record_kind_label(facts: dict, lang: str) -> str:
    from src.i18n import tr_label
    labels = {
        "health": "Crop Health", "disease": "Disease Detection",
        "environment": "Environmental", "field_scan": "Field Scan",
    }
    return tr_label(labels.get(facts["record_type"], "analysis"), lang)


# ---------------------------------------------------------------------------
# Intent classification — lightweight keyword matching, English + Tamil.
# Checked in order; first match wins, so more specific intents are listed
# before their broader neighbors (e.g. "untreated" before "recommendation",
# since "what if I don't treat it" would otherwise also match "treat").
# ---------------------------------------------------------------------------
INTENT_KEYWORDS: list[tuple[str, list[str]]] = [
    # Checked before "recommendation" and "severity_why": both "treat" and
    # "severity"/"why" can appear in an untreated-outcome question ("why
    # would this get worse if I don't treat it?"), so this must win first.
    ("untreated", [
        "untreated", "skip treatment", "don't treat", "dont treat",
        "if i don't", "if i dont", "not treat",
        "சிகிச்சை அளிக்கவில்லை", "தள்ளிப்போட", "புறக்கணி",
    ]),
    ("spread", [
        "spread", "other plant", "other crop", "neighbo", "contagious", "infect",
        "பரவ", "தொற்ற",
    ]),
    ("confidence_why", [
        "confiden", "how sure", "how accurate", "trust this", "reliable",
        "நம்பகத்தன்மை", "நிச்சயமாக",
    ]),
    ("severity_why", [
        "severity", "how severe", "how bad is",
        "தீவிரம்",
    ]),
    ("health_score_why", [
        "health score", "score mean", "score of",
        "ஆரோக்கிய மதிப்பெண்",
    ]),
    ("cause_why", [
        "caused", "cause of", "why did this happen", "why does this",
        "reason for this", "root cause", "why is my", "why did my",
        "என்ன காரணம்", "ஏன் இது",
    ]),
    ("env_status", [
        "temperature", "humidity", "rainfall", "soil moisture", "weather",
        "conditions", "environment", "out of range", "in range",
        "வெப்பநிலை", "ஈரப்பதம்", "மழை", "மண் ஈரப்பதம்", "வரம்பு",
    ]),
    ("recommendation", [
        "should i do", "what now", "recommend", "advice", "treatment",
        "what to do", "how do i treat", "how to treat", "next step", "help me",
        "என்ன செய்ய", "பரிந்துரை", "சிகிச்சை முறை",
    ]),
]


def classify_intent(question: str) -> str:
    q = (question or "").lower()
    for intent, keywords in INTENT_KEYWORDS:
        if any(kw in q for kw in keywords):
            return intent
    return "fallback"


# ---------------------------------------------------------------------------
# Suggested quick-questions per record type (shown as chips in the UI)
# ---------------------------------------------------------------------------
SUGGESTED_QUESTIONS: dict[str, list[str]] = {
    "health": [
        "Why is this severity level?",
        "What if I don't treat it?",
        "What should I do now?",
        "Why is the health score what it is?",
    ],
    "disease": [
        "Why is this severity level?",
        "How confident are you?",
        "What if I don't treat it?",
        "Will it spread?",
    ],
    "environment": [
        "What's out of range here?",
        "What should I do now?",
    ],
    "field_scan": [
        "What if I don't treat it?",
        "Will it spread?",
        "What should I do now?",
    ],
}


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------
def answer(facts: dict, question: str, lang: str = "en") -> str:
    """Main entry point: classify the question, dispatch to a handler.

    Never raises — a handler hitting missing/unexpected data falls back to
    a graceful "don't have that for this record" reply rather than an
    exception, since this sits directly behind a chat input.
    """
    from src.i18n import tr_template

    intent = classify_intent(question)
    handler = _HANDLERS.get(intent, _answer_fallback)
    try:
        return handler(facts, lang)
    except Exception:
        from src.errors import logger
        logger.exception("Crop Doctor handler failed for intent=%s", intent)
        return tr_template(
            "I couldn't work that out from this record. Try one of the "
            "suggested questions below, or rephrase — e.g. \"why is this "
            "moderate severity?\"",
            lang,
        )


def _answer_severity_why(facts: dict, lang: str) -> str:
    from src.i18n import tr_template, tr_disease, tr_severity, tr_severity_action

    disease, severity, confidence = facts.get("disease"), facts.get("severity"), facts.get("confidence")
    if not disease or severity is None:
        return tr_template(
            "This {kind} record doesn't have a disease/severity call to explain.",
            lang, kind=_record_kind_label(facts, lang),
        )

    is_healthy = str(disease).strip().lower() == "healthy" or severity == "None"
    if confidence is not None:
        _, explanation, _ = compute_disease_risk_score(disease, confidence, severity, lang=lang)
    else:
        explanation = tr_template(
            "'{disease}' was recorded at {severity} severity.",
            lang, disease=tr_disease(disease, lang), severity=tr_severity(severity, lang),
        )

    if is_healthy:
        return explanation

    action = SEVERITY_ACTION_EN.get(severity)
    action_ta = tr_severity_action(action, lang) if action else ""
    if action_ta and not action_ta.endswith((".", "!", "?", "।")):
        action_ta += "."
    return f"{explanation} {action_ta}".strip()


def _answer_confidence_why(facts: dict, lang: str) -> str:
    from src.i18n import tr_template, tr_disease, tr_label

    confidence, disease = facts.get("confidence"), facts.get("disease")
    if confidence is None:
        return tr_template(
            "This {kind} record doesn't have a model confidence value stored.",
            lang, kind=_record_kind_label(facts, lang),
        )
    pct = confidence * 100
    disease_label = tr_disease(disease, lang) if disease else tr_label("this result", lang)

    if pct >= 85:
        return tr_template(
            "The model was {pct:.0f}% confident in '{disease}' — that's a high-confidence call.",
            lang, pct=pct, disease=disease_label,
        )
    if pct >= 60:
        return tr_template(
            "The model was {pct:.0f}% confident in '{disease}' — moderate confidence, usually "
            "reliable, but a second clear photo wouldn't hurt if you want to be sure.",
            lang, pct=pct, disease=disease_label,
        )
    return tr_template(
        "The model was only {pct:.0f}% confident in '{disease}' — that's on the lower side. "
        "Consider retaking the photo with better lighting and a closer, single-leaf shot for "
        "a more reliable result.",
        lang, pct=pct, disease=disease_label,
    )


def _answer_untreated(facts: dict, lang: str) -> str:
    from src.i18n import tr_template, tr_disease, tr_severity

    disease, severity, crop = facts.get("disease"), facts.get("severity"), facts.get("crop")
    if not disease or str(disease).strip().lower() == "healthy" or severity in (None, "None"):
        return tr_template(
            "This record doesn't show a disease to leave untreated — it was healthy, or has "
            "no disease/severity call to reason about.",
            lang,
        )

    loss_range = get_yield_loss_range(disease, severity)
    if loss_range is None:
        return tr_template(
            "'{disease}' at {severity} severity isn't in the yield-loss reference table yet, "
            "so I can't give a data-backed estimate — but as a rule, delaying treatment on a "
            "spreading leaf disease rarely helps and usually costs more the longer it waits.",
            lang, disease=tr_disease(disease, lang), severity=tr_severity(severity, lang),
        )

    low, high = loss_range
    yield_per_ha = REFERENCE_YIELD_T_PER_HA.get(crop, 5.0)
    est = estimate_yield_loss(disease, severity, 1.0, yield_per_ha, 0.0)
    yl_low = est["yield_lost_low"] if est else 0.0
    yl_high = est["yield_lost_high"] if est else 0.0

    return tr_template(
        "Published research on {disease} at {severity} severity suggests {low:.0f}-{high:.0f}% "
        "yield loss if left untreated under comparable conditions — on a 1-hectare field "
        "yielding {yield_per_ha:.1f} t/ha, that's roughly {yl_low:.1f}-{yl_high:.1f} tonnes. It "
        "typically doesn't improve on its own; treating promptly is the lower-risk choice.",
        lang, disease=tr_disease(disease, lang), severity=tr_severity(severity, lang),
        low=low, high=high, yield_per_ha=yield_per_ha, yl_low=yl_low, yl_high=yl_high,
    )


def _answer_spread(facts: dict, lang: str) -> str:
    from src.i18n import tr_template, tr_disease

    disease, severity = facts.get("disease"), facts.get("severity")
    if not disease or str(disease).strip().lower() == "healthy" or severity in (None, "None"):
        return tr_template("No disease was detected on this record, so there's nothing here that would spread.", lang)

    risk = SPREAD_RISK.get(disease, "unknown")
    disease_label = tr_disease(disease, lang)

    if risk == "fast":
        return tr_template(
            "Yes — {disease} is known to spread quickly, especially in humid or wet conditions. "
            "Isolate or remove affected material where possible and treat promptly to limit it "
            "reaching nearby plants.",
            lang, disease=disease_label,
        )
    if risk == "moderate":
        return tr_template(
            "{disease} can spread under favorable (humid/warm) conditions, but usually more "
            "slowly than blast/blight-type diseases. Regular monitoring and timely treatment "
            "should keep it contained.",
            lang, disease=disease_label,
        )
    return tr_template(
        "I don't have a specific spread-risk rule for '{disease}' yet, but as a general "
        "precaution, monitor nearby plants and treat promptly regardless.",
        lang, disease=disease_label,
    )


def _answer_cause_why(facts: dict, lang: str) -> str:
    from src.i18n import tr_template, tr_crop, tr_disease, tr_severity

    crop = facts.get("crop")
    readings = {k: facts.get(k) for k in ("temperature", "humidity", "soil_moisture", "rainfall") if facts.get(k) is not None}
    env_factors = _out_of_range_factors(crop, readings, lang=lang) if readings and crop in ENV_CROP_RANGES else []
    disease, severity = facts.get("disease"), facts.get("severity")
    is_healthy = not disease or str(disease).strip().lower() == "healthy" or severity in (None, "None")

    if is_healthy:
        if env_factors:
            bullets = "\n".join(f"- {t}" for t in env_factors)
            return tr_template(
                "No disease was detected here, but these readings were outside {crop}'s "
                "ideal range, which is worth watching:\n{bullets}",
                lang, crop=tr_crop(crop, lang) if crop else "", bullets=bullets,
            )
        return tr_template(
            "No disease was detected on this record, and the logged readings (if any) were "
            "within range — nothing here points to a cause because there's no problem to "
            "explain.",
            lang,
        )

    if env_factors:
        bullets = "\n".join(f"- {t}" for t in env_factors)
        return tr_template(
            "'{disease}' was detected at {severity} severity. These conditions were outside "
            "{crop}'s ideal range and likely contributed:\n{bullets}",
            lang, disease=tr_disease(disease, lang), severity=tr_severity(severity, lang),
            crop=tr_crop(crop, lang) if crop else "", bullets=bullets,
        )
    if readings:
        return tr_template(
            "'{disease}' was detected at {severity} severity. The logged environmental "
            "readings were within {crop}'s typical range, so the immediate cause here is the "
            "leaf-image classification itself, not an obvious environmental trigger.",
            lang, disease=tr_disease(disease, lang), severity=tr_severity(severity, lang),
            crop=tr_crop(crop, lang) if crop else "",
        )
    return tr_template(
        "'{disease}' was detected at {severity} severity from the leaf image alone — this "
        "record doesn't have environmental readings, so I can't say whether conditions "
        "contributed. Check Environmental Analysis for this crop around the same date.",
        lang, disease=tr_disease(disease, lang), severity=tr_severity(severity, lang),
    )


def _answer_health_score_why(facts: dict, lang: str) -> str:
    from src.i18n import tr_template, tr_label

    score = facts.get("health_score")
    if score is None:
        return tr_template("This {kind} record doesn't have a health score stored.", lang, kind=_record_kind_label(facts, lang))

    status_label = tr_label(classify_health_status(score), lang)
    disease_risk, env_risk = facts.get("disease_risk"), facts.get("environmental_risk")

    if disease_risk and env_risk:
        return tr_template(
            "The health score is {score}/100 ('{status}'), combining a disease-signal risk of "
            "'{disease_risk}' and an environmental-signal risk of '{env_risk}' — disease is "
            "weighed slightly more heavily (55%) than environment (45%) in this blend.",
            lang, score=score, status=status_label,
            disease_risk=tr_label(disease_risk, lang), env_risk=tr_label(env_risk, lang),
        )
    return tr_template(
        "The health score is {score}/100, which falls in the '{status}' band.",
        lang, score=score, status=status_label,
    )


def _answer_env_status(facts: dict, lang: str) -> str:
    from src.i18n import tr_template, tr_crop, tr_label

    readings = {k: facts.get(k) for k in ("temperature", "humidity", "soil_moisture", "rainfall") if facts.get(k) is not None}
    if not readings:
        return tr_template("This {kind} record doesn't have environmental readings stored.", lang, kind=_record_kind_label(facts, lang))

    crop = facts.get("crop")
    lines = [
        f"- {tr_label(ENV_FACTOR_LABELS_EN[k], lang)}: {v}{ENV_FACTOR_UNITS[k]}"
        for k, v in readings.items()
    ]
    factors_text = "\n".join(lines)

    out_of_range = _out_of_range_factors(crop, readings, lang=lang) if crop in ENV_CROP_RANGES else []
    crop_label = tr_crop(crop, lang) if crop else tr_label("this crop", lang)
    if out_of_range:
        issues = "\n".join(f"- {t}" for t in out_of_range)
        return tr_template(
            "Logged readings for this analysis:\n{factors}\n\nOutside {crop}'s ideal range:\n{issues}",
            lang, factors=factors_text, crop=crop_label, issues=issues,
        )
    return tr_template(
        "Logged readings for this analysis:\n{factors}\n\nAll within {crop}'s typical range.",
        lang, factors=factors_text, crop=crop_label,
    )


def _answer_recommendation(facts: dict, lang: str) -> str:
    from src.i18n import tr_template

    # For a Crop Health record with every input field present, regenerate
    # a *live* recommendation in the currently-viewed language from the
    # stored raw numbers, rather than echoing back saved text that may
    # have been generated in a different language at save time.
    if facts["record_type"] == "health" and facts.get("crop") in ENV_CROP_RANGES and all(
        facts.get(k) is not None for k in
        ("disease", "severity", "temperature", "humidity", "soil_moisture", "rainfall", "health_score")
    ):
        result = generate_recommendations(
            facts["crop"], facts["disease"], facts["severity"],
            facts["temperature"], facts["humidity"], facts["soil_moisture"], facts["rainfall"],
            facts["health_score"], lang=lang,
        )
        priority = result.get("priority_actions") or []
        if priority:
            bullets = "\n".join(f"- {t}" for t in priority)
            return tr_template("{summary}\n\nTop priority actions:\n{bullets}", lang, summary=result["summary"], bullets=bullets)
        return result["summary"]

    raw = facts.get("stored_recommendation")
    if not raw:
        return tr_template("No recommendation was stored with this record.", lang)

    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        parsed = None

    if isinstance(parsed, dict):
        summary = parsed.get("summary", "")
        priority_texts = [r["text"] for r in parsed.get("recommendations", []) if r.get("priority") == "high"]
        if priority_texts:
            bullets = "\n".join(f"- {t}" for t in priority_texts)
            return tr_template("{summary}\n\nTop priority actions:\n{bullets}", lang, summary=summary, bullets=bullets)
        return summary or raw

    # Plain-text recommendation, stored as-is by the Disease Detection /
    # Environmental Analysis pages (always in English at save time).
    return raw


def _answer_fallback(facts: dict, lang: str) -> str:
    from src.i18n import tr_template

    return tr_template(
        "I can answer questions about this saved {kind} analysis — its severity, how "
        "confident the model was, what happens if it's left untreated, spread risk, its "
        "recommendation, and (where logged) the environmental readings. Try one of the "
        "suggested questions below, or ask in your own words.",
        lang, kind=_record_kind_label(facts, lang),
    )


_HANDLERS = {
    "severity_why": _answer_severity_why,
    "confidence_why": _answer_confidence_why,
    "untreated": _answer_untreated,
    "spread": _answer_spread,
    "cause_why": _answer_cause_why,
    "health_score_why": _answer_health_score_why,
    "env_status": _answer_env_status,
    "recommendation": _answer_recommendation,
    "fallback": _answer_fallback,
}


# ---------------------------------------------------------------------------
# Self-test / demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample_row = {
        "id": 1, "crop_name": "Tomato", "created_at": "2026-09-01T10:00:00Z",
        "disease": "Late_Blight", "confidence": 0.91, "severity": "High",
        "health_score": 38, "disease_risk": "High", "environmental_risk": "Moderate",
        "temperature": 34, "humidity": 88, "soil_moisture": 60, "rainfall": 5,
        "recommendation": json.dumps({
            "summary": "Tomato health is Critical (38/100).",
            "recommendations": [
                {"text": "Destroy affected plants and apply mancozeb.", "reason": "High severity late blight",
                 "category": "disease", "priority": "high"},
            ],
        }),
    }
    facts = normalize_record("health", sample_row)
    print("--- Facts ---")
    print(facts)

    questions = [
        "Why did you say this is High severity?",
        "How confident are you?",
        "What if I don't treat it this week?",
        "Will it spread to my other plants?",
        "What caused this?",
        "Why is the health score so low?",
        "What's the temperature and humidity here?",
        "What should I do now?",
        "Tell me a joke",
    ]
    for q in questions:
        print(f"\nQ: {q}")
        print(f"  intent = {classify_intent(q)}")
        print(f"  A: {answer(facts, q, lang='en')}")

    print("\nOK — self-test complete.")