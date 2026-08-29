"""Yield loss / economic impact estimator.

Connects a disease detection to a question the raw classification output
doesn't answer: **what does this actually cost if it's left untreated?**
Combines the detected disease + severity with a published yield-loss
percentage range (see YIELD_LOSS_RANGES below — every entry sourced from
peer-reviewed agronomy research or a university/extension-service crop
disease fact sheet, cited per entry) and the person's own field size,
expected yield, and market price, to produce an estimated yield-loss range
and a rough revenue estimate.

Deliberately NOT a new ML model — this is published domain data (a lookup
table) plus arithmetic, same "small pure function, independently testable"
shape as src/health_engine.py and src/outbreak_detection.py.

Why ranges, not a single number
--------------------------------
Real-world yield loss from a given disease varies hugely with variety,
timing of infection, weather, and management — the literature itself
reports wide ranges (e.g. early blight in tomato: studies report anywhere
from 9% to 79% depending on conditions). Collapsing that into one
false-precision figure ("18.3% loss") would overstate what a leaf photo +
severity label can actually tell you. This module always returns a range,
and the UI that consumes it always shows the assumptions (yield/price/field
size) driving the number, so the person can see exactly what to adjust for
their own farm and market.

This is educational/planning information, not professional agronomic or
financial advice — the UI surfaces that disclaimer alongside every estimate.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Published yield-loss ranges (% of yield, if left untreated), per disease,
# split by severity band ("Moderate" vs "High" — matches the two non-Healthy
# values pages/disease.py's SEVERITY_MAP actually produces). Ranges are
# intentionally wide, matching how widely the underlying literature itself
# reports — see the source note above each entry.
# ---------------------------------------------------------------------------
YIELD_LOSS_RANGES: dict[str, dict[str, tuple[float, float]]] = {
    "Healthy": {"None": (0.0, 0.0)},

    # Tomato / Potato — early blight (Alternaria solani): field trials
    # report 9-52% loss in unsprayed plots, with several reviews citing up
    # to ~79% in severe untreated cases.
    # Sources: Bioscan (2021) epidemiological study; IJMS (2017) EB
    # resistance review; UConn IPM early blight trials.
    "Early_Blight": {"Moderate": (10.0, 30.0), "High": (35.0, 79.0)},

    # Tomato / Potato — late blight (Phytophthora infestans): can destroy
    # a crop within days under favorable weather; up to 100% loss reported
    # in severe outbreaks; global control costs + losses estimated at
    # $6.7B/year.
    # Sources: USABlight (NC State); APS Press (Fry & Grünwald review).
    "Late_Blight": {"Moderate": (20.0, 45.0), "High": (50.0, 100.0)},

    # Rice — brown spot (Bipolaris oryzae): ~5% average loss region-wide
    # across South/Southeast Asia, severely infected fields up to 45-52%.
    # Sources: IRRI Rice Knowledge Bank; PMC11025958 biochemical study.
    "Brown_Spot": {"Moderate": (5.0, 20.0), "High": (25.0, 50.0)},

    # Rice — leaf blast (Magnaporthe / Pyricularia oryzae): commonly
    # 10-30% of global yield, up to 90% under pathogen-favorable
    # conditions.
    # Sources: hyperspectral blast-severity study (PMC9997726);
    # ScienceDirect Bipolaris/blast topic overview.
    "Leaf_Blast": {"Moderate": (10.0, 30.0), "High": (35.0, 90.0)},

    # Rice — neck blast: same pathogen, but infects the panicle neck and
    # can cut off grain fill entirely — historically the more destructive
    # blast phase, so given a higher ceiling than leaf blast here.
    "Neck_Blast": {"Moderate": (15.0, 35.0), "High": (40.0, 100.0)},

    # Wheat — brown/leaf rust (Puccinia triticina): ~15% average loss,
    # up to 40% in severe infections.
    # Sources: Bayer UK/NZ Crop Science wheat disease fact sheets.
    "Brown_Rust": {"Moderate": (10.0, 20.0), "High": (25.0, 40.0)},

    # Wheat — yellow/stripe rust (Puccinia striiformis): 40-50% common in
    # untreated susceptible varieties, up to 100% in extreme cases.
    # Sources: AHDB yellow rust guide; PMC6339203 spectral yield-loss study.
    "Yellow_Rust": {"Moderate": (20.0, 40.0), "High": (45.0, 100.0)},

    # Corn — common rust (Puccinia sorghi): roughly 3-8% loss per 10%
    # severity; modern hybrids are mostly resistant, so severe epidemics
    # are less common here than the other two corn diseases below.
    # Source: Purdue Agronomy corn rust extension article.
    "Common_Rust": {"Moderate": (5.0, 15.0), "High": (15.0, 40.0)},

    # Corn — gray leaf spot: 5-50% documented over a wide area since 1994.
    # Source: Nebraska Extension gray leaf spot publication.
    "Gray_Leaf_Spot": {"Moderate": (10.0, 25.0), "High": (25.0, 50.0)},

    # Corn — northern corn leaf blight: up to 30-50% if established before
    # tasseling, minimal if it develops later in the season.
    # Sources: Ohio State Ohioline fact sheet; Crop Protection Network
    # NCLB overview.
    "Northern_Leaf_Blight": {"Moderate": (15.0, 30.0), "High": (30.0, 50.0)},
}

# Reasonable *global reference* yields (tonnes/hectare) to pre-fill the
# calculator. Actual yields vary hugely by region, variety, irrigation, and
# season — these are starting points the person is expected to adjust to
# their own farm, not an authoritative figure for any specific location.
REFERENCE_YIELD_T_PER_HA: dict[str, float] = {
    "Tomato": 35.0,
    "Potato": 22.0,
    "Rice": 4.5,
    "Wheat": 3.5,
    "Corn": 5.5,
}

HECTARES_PER_ACRE = 0.404686


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------
def get_yield_loss_range(disease: str, severity: str) -> tuple[float, float] | None:
    """The published (low%, high%) yield-loss range for one disease+severity.

    Returns None for "Healthy" (0% loss isn't worth a "range") or for a
    disease name the table doesn't recognize (shouldn't happen for any
    class a trained model here can actually predict, but fails safe rather
    than fabricating a number).
    """
    entry = YIELD_LOSS_RANGES.get(disease)
    if not entry or disease == "Healthy":
        return None
    if severity in entry:
        return entry[severity]
    # Severity not in the table for this disease (e.g. an "Unknown"
    # severity from a class outside SEVERITY_MAP) — fall back to whichever
    # band exists, preferring the wider/higher one so the estimate doesn't
    # quietly understate risk.
    if "High" in entry:
        return entry["High"]
    if "Moderate" in entry:
        return entry["Moderate"]
    return None


# ---------------------------------------------------------------------------
# Full estimate
# ---------------------------------------------------------------------------
def estimate_yield_loss(
    disease: str,
    severity: str,
    field_size_ha: float,
    yield_per_ha: float,
    price_per_unit: float = 0.0,
) -> dict | None:
    """Full estimate: yield-loss %, physical quantity, and revenue impact.

    Args:
        disease: predicted class name (e.g. "Late_Blight").
        severity: "None" / "Moderate" / "High" (matches
            pages/disease.py's SEVERITY_MAP).
        field_size_ha: field area in hectares.
        yield_per_ha: expected yield per hectare if healthy, in whatever
            unit the person wants the result expressed in (e.g.
            tonnes/hectare) — their own figure, or REFERENCE_YIELD_T_PER_HA
            as a starting point.
        price_per_unit: market price per yield unit. Defaults to 0 (no
            default is provided for price itself — market prices are too
            volatile and local to guess responsibly); when 0, the revenue
            fields in the result are omitted rather than shown as a
            meaningless "0 lost".

    Returns:
        None if disease is "Healthy" or not found in the table (nothing to
        estimate). Otherwise a dict with the loss % range, the expected
        total yield if healthy, the estimated yield lost (low/high), and —
        only if price_per_unit > 0 — the estimated revenue lost (low/high).
    """
    loss_range = get_yield_loss_range(disease, severity)
    if loss_range is None:
        return None

    low_pct, high_pct = loss_range
    expected_yield = field_size_ha * yield_per_ha

    yield_lost_low = expected_yield * (low_pct / 100.0)
    yield_lost_high = expected_yield * (high_pct / 100.0)

    result = {
        "disease": disease,
        "severity": severity,
        "field_size_ha": field_size_ha,
        "yield_per_ha": yield_per_ha,
        "loss_pct_low": low_pct,
        "loss_pct_high": high_pct,
        "expected_yield": round(expected_yield, 2),
        "yield_lost_low": round(yield_lost_low, 2),
        "yield_lost_high": round(yield_lost_high, 2),
        "revenue_lost_low": None,
        "revenue_lost_high": None,
    }

    if price_per_unit and price_per_unit > 0:
        result["revenue_lost_low"] = round(yield_lost_low * price_per_unit, 2)
        result["revenue_lost_high"] = round(yield_lost_high * price_per_unit, 2)

    return result


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("--- Sanity checks ---")
    assert get_yield_loss_range("Healthy", "None") is None
    assert get_yield_loss_range("NotARealDisease", "High") is None

    r = get_yield_loss_range("Late_Blight", "High")
    print("Late_Blight / High ->", r)
    assert r == (50.0, 100.0)

    # A crop with an "Unknown" severity should fall back to the wider band.
    fallback = get_yield_loss_range("Gray_Leaf_Spot", "Unknown")
    print("Gray_Leaf_Spot / Unknown (fallback) ->", fallback)
    assert fallback == YIELD_LOSS_RANGES["Gray_Leaf_Spot"]["High"]

    print("\n--- Full estimate: 2 ha tomato field, Late Blight, High severity ---")
    est = estimate_yield_loss(
        disease="Late_Blight", severity="High",
        field_size_ha=2.0, yield_per_ha=REFERENCE_YIELD_T_PER_HA["Tomato"],
        price_per_unit=200.0,  # e.g. $200/tonne
    )
    for k, v in est.items():
        print(f"  {k}: {v}")
    assert est["expected_yield"] == 70.0  # 2 ha * 35 t/ha
    assert est["yield_lost_low"] == 35.0  # 50% of 70
    assert est["yield_lost_high"] == 70.0  # 100% of 70
    assert est["revenue_lost_low"] == 7000.0
    assert est["revenue_lost_high"] == 14000.0

    print("\n--- No price given -> revenue fields omitted, not zeroed ---")
    est_no_price = estimate_yield_loss(
        disease="Early_Blight", severity="Moderate",
        field_size_ha=1.0, yield_per_ha=REFERENCE_YIELD_T_PER_HA["Tomato"],
    )
    print(est_no_price)
    assert est_no_price["revenue_lost_low"] is None

    print("\nOK — all checks passed.")