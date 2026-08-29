"""Outbreak / trend detection over saved disease-detection history.

Turns the app's saved analysis history into a genuinely new signal: instead
of just listing past analyses (that's what Analysis History already does),
this compares a **recent rolling window of saved analyses** against the
window immediately before it, per crop, and flags when the diseased share
and/or high-severity share is climbing — an early "something's spreading"
signal rather than a static log.

Data sources — and one deliberate exclusion
---------------------------------------------
Two tables feed this: `disease_analyses` (single-leaf saves from the
Disease Detection page) and `field_scans` (aggregated batch saves from the
Field Scan page). Both come from real model inference on an uploaded photo.

`analyses` (the Crop Health Analysis page) is deliberately **excluded**:
that page lets the user manually pick a disease/severity from a dropdown
to explore "what would this environment do to a leaf with X severity"
scenarios (see pages/health.py — it isn't wired to an uploaded image at
all). Feeding manually-chosen values into an outbreak signal would let
someone spike an "outbreak" just by picking scary dropdown options, with
no relationship to anything actually observed in a field. Outbreak
detection should only ever reflect real detections.

Windowing model
----------------
"Rolling window" here means the last N *saved analyses* (matching how the
rest of the app already thinks about history — see Analysis History), not
the last N calendar days and not the last N individual leaves. A single
Field Scan save (which might cover 20 leaves) counts as ONE saved analysis
for windowing purposes, same as a single Disease Detection save — but each
record's contribution to the window's diseased%/high-severity% is weighted
by how many leaves it actually represents, so a 20-leaf field scan isn't
drowned out by (or doesn't drown out) a handful of single-leaf saves.

Risk classification is deliberately simple and threshold-based (same
banded-rules philosophy as src/health_engine.py's classify_health_status),
not a learned model — every number this module produces should be
explainable in one sentence, which matters more than statistical
sophistication for a signal that's meant to prompt a human to go look.
"""

from __future__ import annotations

import json
from collections import Counter

# ---------------------------------------------------------------------------
# Risk levels, ordered most to least urgent (used for sorting alert lists)
# ---------------------------------------------------------------------------
RISK_ORDER = {"High": 0, "Elevated": 1, "Watch": 2, "Low": 3, "Insufficient data": 4}
ACTIVE_ALERT_LEVELS = ("High", "Elevated")


# ---------------------------------------------------------------------------
# Step 1 — normalize two differently-shaped tables into one record shape
# ---------------------------------------------------------------------------
def collect_detection_records(disease_rows: list[dict], field_scan_rows: list[dict]) -> list[dict]:
    """Normalize disease_analyses + field_scans rows into one common shape.

    Each output record represents ONE saved analysis and carries just the
    marginal numbers outbreak detection needs — not a leaf-by-leaf event
    log, since neither table stores individual-leaf timestamps for a batch
    scan and none is needed (diseased%/high-severity%/dominant disease are
    all marginal statistics, not a joint per-leaf distribution).
    """
    records: list[dict] = []

    for row in disease_rows:
        disease = row.get("disease")
        if not disease or not row.get("crop_name"):
            continue
        is_healthy = bool(row.get("is_healthy"))
        records.append({
            "crop": row["crop_name"],
            "created_at": row.get("created_at") or "",
            "_id": row.get("id") or 0,
            "source": "disease_detection",
            "n_leaves": 1,
            "diseased_pct": 0.0 if is_healthy else 100.0,
            "high_pct": 100.0 if row.get("severity") == "High" else 0.0,
            "dominant_disease": None if is_healthy else disease,
        })

    for row in field_scan_rows:
        if not row.get("crop_name"):
            continue
        n_leaves = row.get("num_images") or 0
        if n_leaves <= 0:
            continue

        try:
            severity_counts = json.loads(row.get("severity_breakdown") or "{}")
        except (TypeError, ValueError):
            severity_counts = {}
        high_count = severity_counts.get("High", 0)

        healthy_pct = row.get("healthy_pct")
        diseased_pct = (100.0 - healthy_pct) if healthy_pct is not None else 0.0

        records.append({
            "crop": row["crop_name"],
            "created_at": row.get("created_at") or "",
            "_id": row.get("id") or 0,
            "source": "field_scan",
            "n_leaves": n_leaves,
            "diseased_pct": diseased_pct,
            "high_pct": 100.0 * high_count / n_leaves,
            "dominant_disease": row.get("dominant_disease"),
        })

    # Sort chronologically. created_at has only whole-second resolution, so
    # several rows saved within the same second (very plausible when
    # testing, demoing, or scripting several saves back to back) would tie
    # on that key alone; each table's autoincrement id is a reliable
    # same-table tiebreaker since it only ever increases with insertion
    # order. IDs aren't comparable *across* the two tables, but ties that
    # straddle both tables in the same second are rare enough, and the
    # ambiguity narrow enough (at most a few records out of order by at
    # most a second), that it isn't worth chasing further.
    records.sort(key=lambda r: (r["created_at"], r["_id"]))
    for r in records:
        del r["_id"]
    return records


def _records_for_crop(records: list[dict], crop: str) -> list[dict]:
    return [r for r in records if r["crop"] == crop]


# ---------------------------------------------------------------------------
# Step 2 — summarize one window of records
# ---------------------------------------------------------------------------
def _window_stats(records: list[dict]) -> dict:
    """Aggregate a window of records into diseased%/high-severity%/etc.

    Each record's percentage is weighted by how many leaves it represents
    (n_leaves), so a 20-leaf Field Scan contributes proportionally to a
    single-leaf Disease Detection save rather than counting equally.
    """
    n_records = len(records)
    if n_records == 0:
        return {
            "n_records": 0, "n_leaves": 0, "diseased_pct": 0.0, "high_pct": 0.0,
            "dominant_disease": None, "disease_counts": {},
        }

    total_leaves = sum(r["n_leaves"] for r in records) or n_records
    diseased_pct = sum(r["diseased_pct"] * r["n_leaves"] for r in records) / total_leaves
    high_pct = sum(r["high_pct"] * r["n_leaves"] for r in records) / total_leaves

    disease_counts = Counter(r["dominant_disease"] for r in records if r["dominant_disease"])
    dominant = disease_counts.most_common(1)[0][0] if disease_counts else None

    return {
        "n_records": n_records,
        "n_leaves": total_leaves,
        "diseased_pct": round(diseased_pct, 1),
        "high_pct": round(high_pct, 1),
        "dominant_disease": dominant,
        "disease_counts": dict(disease_counts),
    }


# ---------------------------------------------------------------------------
# Step 3 — classify recent vs. prior window into a risk level
# ---------------------------------------------------------------------------
def _classify_risk(recent: dict, diseased_delta: float | None, high_delta: float | None,
                    insufficient_data: bool, has_prior: bool) -> tuple[str, str]:
    if insufficient_data:
        return "Insufficient data", (
            f"Only {recent['n_records']} saved analysis(es) so far for this crop — "
            "a few more are needed before a trend can be judged."
        )

    diseased_pct = recent["diseased_pct"]

    if not has_prior:
        # Enough records to describe right now, but nothing before this
        # window to compare against yet — report current state, not a trend.
        if diseased_pct >= 60:
            return "Watch", (
                f"{diseased_pct:.0f}% of recent analyses are diseased. "
                "No prior window yet to compare against."
            )
        return "Low", "No prior window yet to compare against; current detections look manageable."

    if diseased_pct >= 80 or (diseased_pct >= 60 and high_delta is not None and high_delta > 10):
        reason = f"{diseased_pct:.0f}% of recent analyses are diseased"
        reason += f", and high-severity share is up {high_delta:+.0f} pts vs. the prior window." if high_delta else "."
        return "High", reason

    if (diseased_delta is not None and diseased_delta > 15) or (high_delta is not None and high_delta > 10):
        reason = f"Diseased share is up {diseased_delta:+.0f} pts vs. the prior window"
        reason += f", high-severity share up {high_delta:+.0f} pts." if high_delta and high_delta > 10 else "."
        return "Elevated", reason

    if (diseased_delta is not None and diseased_delta > 5) or diseased_pct >= 40:
        return "Watch", f"Diseased share is {diseased_pct:.0f}% and trending up {diseased_delta:+.0f} pts."

    delta_txt = f"{diseased_delta:+.0f} pts" if diseased_delta is not None else "stable"
    return "Low", f"Diseased share is {diseased_pct:.0f}%, stable or improving ({delta_txt})."


def compute_outbreak_signal(records: list[dict], crop: str, window: int = 7) -> dict:
    """The full recent-vs-prior comparison for one crop.

    `records` must already be chronologically sorted ascending (as returned
    by collect_detection_records).
    """
    crop_records = _records_for_crop(records, crop)
    n_total = len(crop_records)

    recent_records = crop_records[-window:]
    prior_records = crop_records[-2 * window:-window] if n_total > window else []

    recent = _window_stats(recent_records)
    prior = _window_stats(prior_records)
    has_prior = len(prior_records) > 0

    # Require a small minimum before saying anything about a crop at all —
    # one or two saves is not a trend, it's a data point.
    insufficient_data = n_total < max(3, window // 2)

    diseased_delta = round(recent["diseased_pct"] - prior["diseased_pct"], 1) if has_prior else None
    high_delta = round(recent["high_pct"] - prior["high_pct"], 1) if has_prior else None

    risk_level, risk_reason = _classify_risk(recent, diseased_delta, high_delta, insufficient_data, has_prior)

    return {
        "crop": crop,
        "window": window,
        "n_total_records": n_total,
        "recent": recent,
        "prior": prior,
        "has_prior_window": has_prior,
        "diseased_pct_delta": diseased_delta,
        "high_pct_delta": high_delta,
        "risk_level": risk_level,
        "risk_reason": risk_reason,
    }


# ---------------------------------------------------------------------------
# Step 4 — all crops at once, ready for an alerts view
# ---------------------------------------------------------------------------
def compute_all_outbreak_signals(records: list[dict], window: int = 7) -> list[dict]:
    """One outbreak signal per crop present in `records`, most urgent first."""
    crops = sorted({r["crop"] for r in records if r["crop"]})
    signals = [compute_outbreak_signal(records, crop, window) for crop in crops]
    signals.sort(key=lambda s: RISK_ORDER.get(s["risk_level"], 9))
    return signals


def get_active_alerts(signals: list[dict]) -> list[dict]:
    """The subset of signals worth surfacing as an alert banner (Elevated/High)."""
    return [s for s in signals if s["risk_level"] in ACTIVE_ALERT_LEVELS]


# ---------------------------------------------------------------------------
# Convenience: load straight from the database
# ---------------------------------------------------------------------------
def load_outbreak_signals(window: int = 7, limit: int = 500) -> list[dict]:
    """End-to-end: pull saved history from SQLite and compute every crop's signal.

    Kept separate from compute_all_outbreak_signals() so the pure
    computation stays trivially unit-testable with synthetic data (see the
    self-test below) without touching the database.
    """
    from src.db import get_disease_analyses, get_field_scans  # local import: keep DB optional for callers that already have records

    disease_rows = get_disease_analyses(limit=limit)
    field_scan_rows = get_field_scans(limit=limit)
    records = collect_detection_records(disease_rows, field_scan_rows)
    return compute_all_outbreak_signals(records, window=window)


# ---------------------------------------------------------------------------
# Self-test — synthetic, steadily-worsening history for one crop, verifying
# the risk level actually climbs as severity climbs.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from datetime import datetime, timedelta, timezone

    def _ts(days_ago: int) -> str:
        return (datetime.now(timezone.utc) - timedelta(days=days_ago)).strftime("%Y-%m-%dT%H:%M:%SZ")

    synthetic_disease_rows = []
    # Oldest 7 saves: mostly healthy (prior window)
    for i, healthy in enumerate([True, True, True, False, True, True, False]):
        synthetic_disease_rows.append({
            "crop_name": "Tomato", "created_at": _ts(20 - i),
            "disease": "Healthy" if healthy else "Early_Blight",
            "severity": "None" if healthy else "Moderate",
            "is_healthy": healthy,
        })
    # Most recent 7 saves: worsening — more diseased, more High severity
    for i, (disease, severity) in enumerate([
        ("Early_Blight", "Moderate"), ("Late_Blight", "High"), ("Late_Blight", "High"),
        ("Healthy", "None"), ("Late_Blight", "High"), ("Late_Blight", "High"), ("Late_Blight", "High"),
    ]):
        synthetic_disease_rows.append({
            "crop_name": "Tomato", "created_at": _ts(6 - i),
            "disease": disease, "severity": severity, "is_healthy": disease == "Healthy",
        })

    records = collect_detection_records(synthetic_disease_rows, [])
    signal = compute_outbreak_signal(records, "Tomato", window=7)

    print("--- Outbreak signal for synthetic worsening Tomato history ---")
    print("recent  :", signal["recent"])
    print("prior   :", signal["prior"])
    print("deltas  : diseased %+.1f, high %+.1f" % (signal["diseased_pct_delta"], signal["high_pct_delta"]))
    print("risk    :", signal["risk_level"], "-", signal["risk_reason"])
    assert signal["risk_level"] in ("Elevated", "High"), "Expected worsening history to be flagged Elevated/High"
    print("OK — worsening trend correctly flagged.")