"""Analysis History page — browse, filter, sort, inspect, and delete past
analyses saved to SQLite via src.db.

Reads real rows via src.db.get_analyses(). Falls back to a friendly empty
state (rather than dummy data) when nothing has been saved yet — analyses
are saved from the "Save Analysis" button on the Crop Health Analysis page.
"""

from __future__ import annotations

import json
from datetime import datetime

import pandas as pd
import streamlit as st

from src.db import get_analyses, delete_analysis
from src.health_engine import classify_health_status
from utils.ui import (
    page_header,
    callout,
    footer,
)
from utils.icons import icon_html

CATEGORY_ICON = {"disease": "diseased", "environment": "temperature", "overall": "leaf"}
PRIORITY_COLOR = {"high": "#B5564B", "medium": "#C97A3B", "low": "#7FA687"}

SORT_OPTIONS = {
    "Date (newest first)":       ("_dt", False),
    "Date (oldest first)":       ("_dt", True),
    "Health score (high to low)": ("health_score", False),
    "Health score (low to high)": ("health_score", True),
    "Crop (A-Z)":                 ("crop", True),
}


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
def render() -> None:
    page_header(
        "history",
        "Analysis History",
        "Review, filter, and manage previously saved crop analyses.",
    )

    rows = get_analyses(limit=1000)

    if not rows:
        callout(
            f"{icon_html('history', size=18)}No analyses saved yet. Go to "
            "<b>Crop Health Analysis</b>, run a calculation, and click "
            "<b>Save Analysis</b> to see records here."
        )
        footer()
        return

    df = _to_dataframe(rows)

    # ------------------------------------------------------------ filters
    st.markdown("#### Filters & sorting")
    f1, f2, f3, f4 = st.columns([1.2, 1.2, 1.4, 1.2])

    with f1:
        crop_filter = st.multiselect("Crop", sorted(df["crop"].unique()),
                                     default=sorted(df["crop"].unique()))
    with f2:
        disease_filter = st.multiselect("Disease", sorted(df["disease"].dropna().unique()),
                                        default=sorted(df["disease"].dropna().unique()))
    with f3:
        min_date, max_date = df["_dt"].min().date(), df["_dt"].max().date()
        date_range = st.date_input("Date range", value=(min_date, max_date),
                                   min_value=min_date, max_value=max_date)
    with f4:
        sort_choice = st.selectbox("Sort by", list(SORT_OPTIONS.keys()))

    mask = df["crop"].isin(crop_filter) & df["disease"].isin(disease_filter)

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = date_range
        mask &= (df["_dt"].dt.date >= start) & (df["_dt"].dt.date <= end)

    sort_col, ascending = SORT_OPTIONS[sort_choice]
    view = df[mask].sort_values(sort_col, ascending=ascending)

    # ------------------------------------------------------------- table
    st.markdown(f"#### {len(view)} analyses")
    if len(view) == 0:
        callout("No analyses match the current filters.")
        footer()
        return

    st.dataframe(
        view[["id", "date", "crop", "disease", "confidence", "severity",
              "health_score", "risk", "recommendation_summary"]],
        use_container_width=True,
        column_config={
            "id": "ID",
            "date": "Date",
            "crop": "Crop",
            "disease": "Disease",
            "confidence": st.column_config.NumberColumn("Confidence", format="%.0f%%"),
            "severity": "Severity",
            "health_score": st.column_config.ProgressColumn(
                "Health score", min_value=0, max_value=100, format="%d"),
            "risk": "Risk",
            "recommendation_summary": "Recommendation",
        },
        hide_index=True,
    )

    # ---------------------------------------------------- expandable rows
    st.markdown("#### Records")
    st.caption("Expand a record to view full details or delete it.")
    for _, row in view.iterrows():
        _render_record(row)

    st.markdown("---")
    footer()


# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------
def _to_dataframe(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df = df.rename(columns={"crop_name": "crop", "created_at": "date"})

    for col in ("disease", "severity", "disease_risk", "environmental_risk", "recommendation"):
        if col not in df.columns:
            df[col] = None

    # Real datetime column for filtering/sorting; formatted string for display.
    df["_dt"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_localize(None)
    df["_dt"] = df["_dt"].fillna(pd.Timestamp.now())
    df["date"] = df["_dt"].dt.strftime("%Y-%m-%d %H:%M")

    df["health_score"] = df["health_score"].fillna(0).astype(int)
    df["status"] = df["health_score"].apply(classify_health_status)
    df["disease"] = df["disease"].fillna("Unknown")
    df["confidence"] = pd.to_numeric(df.get("confidence"), errors="coerce") * 100

    df["risk"] = df.apply(
        lambda r: f"Disease: {r['disease_risk'] or '—'} · Env: {r['environmental_risk'] or '—'}",
        axis=1,
    )
    df["recommendation_summary"] = df["recommendation"].apply(_summary_of)

    return df


def _summary_of(raw: str | None) -> str:
    """One-line preview of the stored recommendation for the table view."""
    if not raw:
        return "—"
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return (raw[:80] + "…") if len(raw) > 80 else raw
    if isinstance(parsed, dict) and "summary" in parsed:
        return parsed["summary"]
    return (raw[:80] + "…") if len(raw) > 80 else raw


# ---------------------------------------------------------------------------
# Per-record expandable section (view details + delete)
# ---------------------------------------------------------------------------
def _render_record(row: pd.Series) -> None:
    row_id = int(row["id"])
    label = f"#{row_id} · {row['crop']} · {row['disease']} · {row['date']}"

    with st.expander(label):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Crop", row["crop"])
        c2.metric("Status", row["status"])
        c3.metric("Health score", int(row["health_score"]))
        c4.metric("Date", row["date"])

        d1, d2, d3 = st.columns(3)
        d1.metric("Disease", row["disease"] or "—")
        d2.metric("Confidence", f"{row['confidence']:.0f}%" if pd.notna(row["confidence"]) else "—")
        d3.metric("Severity", row["severity"] or "—")

        r1, r2 = st.columns(2)
        r1.metric("Disease risk", row["disease_risk"] or "—")
        r2.metric("Environmental risk", row["environmental_risk"] or "—")

        st.markdown("**Recommendation**")
        _render_recommendation(row["recommendation"])

        st.markdown("---")
        _render_delete_control(row_id)


def _render_delete_control(row_id: int) -> None:
    """Two-step delete: first click asks for confirmation, second click
    actually deletes — avoids removing a record from a single accidental
    click.
    """
    confirm_key = f"_history_confirm_delete_{row_id}"

    if not st.session_state.get(confirm_key, False):
        if st.button("Delete analysis", key=f"_history_delete_btn_{row_id}"):
            st.session_state[confirm_key] = True
            st.rerun()
        return

    st.warning(f"Delete analysis #{row_id}? This cannot be undone.")
    yes_col, no_col = st.columns(2)
    with yes_col:
        if st.button("Yes, delete", key=f"_history_confirm_yes_{row_id}", type="primary"):
            delete_analysis(row_id)
            st.session_state.pop(confirm_key, None)
            st.success(f"Analysis #{row_id} deleted.")
            st.rerun()
    with no_col:
        if st.button("Cancel", key=f"_history_confirm_no_{row_id}"):
            st.session_state.pop(confirm_key, None)
            st.rerun()


def _render_recommendation(raw: str | None) -> None:
    """Recommendation is stored as JSON from src.recommendation_engine when
    saved from the Crop Health Analysis page; render it structured if so,
    otherwise fall back to showing the raw text as-is (e.g. older/plain
    saves from the Disease Detection page).
    """
    if not raw:
        st.caption("No recommendation recorded.")
        return

    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        st.markdown(raw)
        return

    if not isinstance(parsed, dict) or "recommendations" not in parsed:
        st.markdown(raw)
        return

    st.caption(parsed.get("summary", ""))
    items = ""
    for rec in parsed.get("recommendations", []):
        icon_tag = icon_html(CATEGORY_ICON.get(rec.get("category"), "leaf"), size=18, margin_right="0")
        badge_color = PRIORITY_COLOR.get(rec.get("priority"), "#7FA687")
        items += (
            "<div class='rec-item'>"
            f"<div class='rec-icon'>{icon_tag}</div>"
            "<div>"
            f"<div class='rec-title'>{rec.get('category','').title()} · "
            f"<span style='color:{badge_color}'>{rec.get('priority','').upper()}</span></div>"
            f"<div class='rec-text'>{rec.get('text','')}</div>"
            "</div></div>"
        )
    st.markdown(items, unsafe_allow_html=True)