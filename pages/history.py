"""Analysis History page — browse past analyses saved to SQLite.

Reads real rows via src.db.get_analyses(). Falls back to a friendly empty
state (rather than dummy data) when nothing has been saved yet — analyses
are saved from the "Save Analysis" button on the Crop Health Analysis page.
"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from src.db import get_analyses
from src.health_engine import classify_health_status
from utils.ui import (
    page_header,
    callout,
    footer,
)
from utils.icons import icon_html

CATEGORY_ICON = {"disease": "diseased", "environment": "temperature", "overall": "leaf"}
PRIORITY_COLOR = {"high": "#B5564B", "medium": "#C97A3B", "low": "#7FA687"}


def render() -> None:
    page_header(
        "history",
        "Analysis History",
        "Review previously saved crop analyses.",
    )

    rows = get_analyses(limit=500)

    if not rows:
        callout(
            f"{icon_html('history', size=18)}No analyses saved yet. Go to "
            "<b>Crop Health Analysis</b>, run a calculation, and click "
            "<b>Save Analysis</b> to see records here."
        )
        footer()
        return

    df = _to_dataframe(rows)

    st.markdown("#### Filters")
    f1, f2, f3 = st.columns(3)
    with f1:
        crop_filter = st.multiselect("Crop", sorted(df["crop"].unique()),
                                     default=sorted(df["crop"].unique()))
    with f2:
        status_filter = st.multiselect("Status", sorted(df["status"].unique()),
                                       default=sorted(df["status"].unique()))
    with f3:
        min_score = st.slider("Min health score", 0, 100, 0, 5)

    mask = (df["crop"].isin(crop_filter)) & (df["status"].isin(status_filter)) & (df["health_score"] >= min_score)
    view = df[mask].sort_values("id", ascending=False)

    st.markdown(f"#### {len(view)} analyses")
    st.dataframe(
        view[["id", "date", "crop", "disease", "severity", "status", "health_score",
              "temperature", "humidity", "soil_moisture", "rainfall"]],
        use_container_width=True,
        column_config={
            "id": "ID",
            "date": "Date",
            "crop": "Crop",
            "disease": "Disease",
            "severity": "Severity",
            "status": st.column_config.TextColumn("Status"),
            "health_score": st.column_config.ProgressColumn(
                "Health score", min_value=0, max_value=100, format="%d"),
            "temperature": "Temp (°C)",
            "humidity": "Humidity (%)",
            "soil_moisture": "Soil moist. (%)",
            "rainfall": "Rainfall (mm)",
        },
        hide_index=True,
    )

    if len(view) == 0:
        callout("No analyses match the current filters.")
        footer()
        return

    # Expandable detail view
    with st.expander("Detail view", expanded=False):
        idx = st.selectbox("Select an analysis to inspect", view["id"].tolist())
        row = view[view["id"] == idx].iloc[0]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Crop", row["crop"])
        c2.metric("Status", row["status"])
        c3.metric("Health score", row["health_score"])
        c4.metric("Date", row["date"])

        d1, d2, d3 = st.columns(3)
        d1.metric("Disease", row["disease"] or "—")
        d2.metric("Confidence", f"{row['confidence']*100:.0f}%" if row["confidence"] is not None else "—")
        d3.metric("Severity", row["severity"] or "—")

        r1, r2 = st.columns(2)
        r1.metric("Disease risk", row["disease_risk"] or "—")
        r2.metric("Environmental risk", row["environmental_risk"] or "—")

        st.markdown("**Recommendation**")
        _render_recommendation(row["recommendation"])

    st.markdown("---")
    footer()


def _to_dataframe(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df = df.rename(columns={"crop_name": "crop", "created_at": "date"})

    # Friendly date (fallback to raw string if parsing fails).
    try:
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        pass

    df["health_score"] = df["health_score"].fillna(0).astype(int)
    df["status"] = df["health_score"].apply(classify_health_status)

    for col in ("disease", "severity", "disease_risk", "environmental_risk", "recommendation"):
        if col not in df.columns:
            df[col] = None

    return df


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