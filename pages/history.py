"""Analysis History page — browse, filter, sort, inspect, and delete past
analyses saved to SQLite via src.db.

Three separate tabs, one per analysis type, each reading from its own table
so a save from one page never mixes into another page's history:
- Crop Health       -> src.db.get_analyses()             (Crop Health Analysis page)
- Disease Detection -> src.db.get_disease_analyses()      (Disease Detection page)
- Environmental     -> src.db.get_environment_analyses()  (Environmental Analysis page)

Each tab shows a friendly empty state (rather than dummy data) when nothing
has been saved yet for that analysis type.
"""

from __future__ import annotations

import json
import os
from html import escape

import pandas as pd
import streamlit as st

from src.db import (
    get_analyses, delete_analysis,
    get_disease_analyses, delete_disease_analysis,
    get_environment_analyses, delete_environment_analysis,
)
from src.errors import safe_action, DatabaseError, logger
from src.health_engine import classify_health_status
from src.recommendation_engine import CATEGORY_ICON, PRIORITY_COLOR
from utils.ui import (
    page_header,
    callout,
    footer,
    RISK_LEVELS,
)
from utils.icons import icon_html

HEALTH_SORT_OPTIONS = {
    "Date (newest first)":        ("_dt", False),
    "Date (oldest first)":        ("_dt", True),
    "Health score (high to low)": ("health_score", False),
    "Health score (low to high)": ("health_score", True),
    "Crop (A-Z)":                  ("crop", True),
}

DISEASE_SORT_OPTIONS = {
    "Date (newest first)":       ("_dt", False),
    "Date (oldest first)":       ("_dt", True),
    "Confidence (high to low)":  ("confidence", False),
    "Confidence (low to high)":  ("confidence", True),
    "Crop (A-Z)":                 ("crop", True),
}

ENV_SORT_OPTIONS = {
    "Date (newest first)":        ("_dt", False),
    "Date (oldest first)":        ("_dt", True),
    "Health score (high to low)": ("health_score", False),
    "Health score (low to high)": ("health_score", True),
    "Crop (A-Z)":                  ("crop", True),
}


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
def render() -> None:
    page_header(
        "history",
        "Analysis History",
        "Review, filter, and manage previously saved analyses.",
    )

    tab_health, tab_disease, tab_env = st.tabs(
        ["Crop Health", "Disease Detection", "Environmental"]
    )
    with tab_health:
        _render_health_tab()
    with tab_disease:
        _render_disease_tab()
    with tab_env:
        _render_env_tab()

    footer()


# ---------------------------------------------------------------------------
# Shared helpers (used across all three tabs)
# ---------------------------------------------------------------------------
def _load_rows(loader, label: str) -> list[dict] | None:
    """Fetch rows for one tab, showing a consistent error state on failure.

    Returns None (caller should stop rendering that tab) if loading failed.
    """
    try:
        return loader(limit=1000)
    except DatabaseError as e:
        st.error(str(e))
        return None
    except Exception:
        logger.exception(f"Unexpected error loading {label} history")
        st.error(
            f"Loading {label} history failed unexpectedly. Please try again. "
            "If the problem continues, contact the app maintainer."
        )
        return None


def _prep_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Add a real datetime column (_dt) for filtering/sorting, and a
    formatted display string (date), from the `created_at` column."""
    df = df.rename(columns={"created_at": "date"})
    df["_dt"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_localize(None)
    df["_dt"] = df["_dt"].fillna(pd.Timestamp.now())
    df["date"] = df["_dt"].dt.strftime("%Y-%m-%d %H:%M")
    return df


def _history_metric(label: str, value: object) -> None:
    """Render a compact, wrapping value for the expanded history view."""
    st.markdown(
        f"<div class='metric-tile history-metric'>"
        f"<div class='label'>{escape(str(label))}</div>"
        f"<div class='value'>{escape(str(value))}</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def _render_delete_control(row_id: int, delete_fn, key_prefix: str, label: str = "analysis") -> None:
    """Two-step delete: first click asks for confirmation, second click
    actually deletes — avoids removing a record from a single accidental
    click. Shared across all three tabs; key_prefix keeps widget keys
    from colliding between tabs when the same row id appears in more than
    one table.
    """
    confirm_key = f"{key_prefix}_confirm_delete_{row_id}"

    if not st.session_state.get(confirm_key, False):
        if st.button(f"Delete {label}", key=f"{key_prefix}_delete_btn_{row_id}"):
            st.session_state[confirm_key] = True
            st.rerun()
        return

    st.warning(f"Delete this {label} (#{row_id})? This cannot be undone.")
    yes_col, no_col = st.columns(2)
    with yes_col:
        if st.button("Yes, delete", key=f"{key_prefix}_confirm_yes_{row_id}", type="primary"):
            with safe_action(f"Deleting {label}"):
                delete_fn(row_id)
                st.session_state.pop(confirm_key, None)
                st.success(f"{label.capitalize()} #{row_id} deleted.")
                st.rerun()
    with no_col:
        if st.button("Cancel", key=f"{key_prefix}_confirm_no_{row_id}"):
            st.session_state.pop(confirm_key, None)
            st.rerun()


def _summary_of(raw: str | None) -> str:
    """One-line preview of a stored recommendation for a table cell.
    Handles both plain-text recommendations (Disease/Environment pages)
    and JSON-structured ones (Crop Health page)."""
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
# Tab 1 — Crop Health
# ---------------------------------------------------------------------------
def _render_health_tab() -> None:
    rows = _load_rows(get_analyses, "crop health analysis")
    if rows is None:
        return

    if not rows:
        callout(
            f"{icon_html('history', size=18)}No crop health analyses saved yet. Go to "
            "<b>Crop Health Analysis</b>, run a calculation, and click "
            "<b>Save Analysis</b> to see records here."
        )
        return

    df = _health_to_dataframe(rows)

    st.markdown("#### Filters & sorting")
    f1, f2, f3, f4 = st.columns([1.2, 1.2, 1.4, 1.2])
    with f1:
        crop_options = ["All"] + sorted(df["crop"].unique())
        crop_filter = st.multiselect("Crop", crop_options, default=["All"], key="_hist_health_crop")
    with f2:
        disease_options = ["All"] + sorted(df["disease"].dropna().unique())
        disease_filter = st.multiselect("Disease", disease_options, default=["All"], key="_hist_health_disease")
    with f3:
        min_date, max_date = df["_dt"].min().date(), df["_dt"].max().date()
        date_range = st.date_input("Date range", value=(min_date, max_date),
                                   min_value=min_date, max_value=max_date, key="_hist_health_dates")
    with f4:
        sort_choice = st.selectbox("Sort by", list(HEALTH_SORT_OPTIONS.keys()), key="_hist_health_sort")

    crop_matches = "All" in crop_filter or df["crop"].isin(crop_filter)
    disease_matches = "All" in disease_filter or df["disease"].isin(disease_filter)
    mask = crop_matches & disease_matches
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = date_range
        mask &= (df["_dt"].dt.date >= start) & (df["_dt"].dt.date <= end)

    sort_col, ascending = HEALTH_SORT_OPTIONS[sort_choice]
    view = df[mask].sort_values(sort_col, ascending=ascending)

    st.markdown(f"#### {len(view)} analyses")
    if len(view) == 0:
        callout("No analyses match the current filters.")
        return

    st.dataframe(
        view[["id", "date", "crop", "disease", "confidence", "severity",
              "health_score", "risk", "recommendation_summary"]],
        use_container_width=True,
        column_config={
            "id": "ID", "date": "Date", "crop": "Crop", "disease": "Disease",
            "confidence": st.column_config.NumberColumn("Confidence", format="%.0f%%"),
            "severity": "Severity",
            "health_score": st.column_config.ProgressColumn(
                "Health score", min_value=0, max_value=100, format="%d"),
            "risk": "Risk", "recommendation_summary": "Recommendation",
        },
        hide_index=True,
    )

    st.markdown("#### Records")
    st.caption("Expand a record to view full details or delete it.")
    for _, row in view.iterrows():
        _render_health_record(row)


def _health_to_dataframe(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df = df.rename(columns={"crop_name": "crop"})
    df = _prep_datetime(df)

    for col in ("disease", "severity", "disease_risk", "environmental_risk", "recommendation"):
        if col not in df.columns:
            df[col] = None

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


def _render_health_record(row: pd.Series) -> None:
    row_id = int(row["id"])
    label = f"#{row_id} · {row['crop']} · {row['disease']} · {row['date']}"

    with st.expander(label):
        c1, c2 = st.columns(2)
        with c1:
            _history_metric("Crop", row["crop"])
        with c2:
            _history_metric("Status", row["status"])

        c3, c4 = st.columns(2)
        with c3:
            _history_metric("Health score", int(row["health_score"]))
        with c4:
            _history_metric("Date", row["date"])

        d1, d2, d3 = st.columns(3)
        with d1:
            _history_metric("Disease", row["disease"] or "—")
        with d2:
            _history_metric("Confidence", f"{row['confidence']:.0f}%" if pd.notna(row["confidence"]) else "—")
        with d3:
            _history_metric("Severity", row["severity"] or "—")

        r1, r2 = st.columns(2)
        with r1:
            _history_metric("Disease risk", row["disease_risk"] or "—")
        with r2:
            _history_metric("Environmental risk", row["environmental_risk"] or "—")

        st.markdown("**Recommendation**")
        _render_structured_recommendation(row["recommendation"])

        st.markdown("---")
        _render_delete_control(row_id, delete_analysis, key_prefix="_hist_health", label="crop health analysis")


def _render_structured_recommendation(raw: str | None) -> None:
    """Recommendation is stored as JSON from src.recommendation_engine when
    saved from the Crop Health Analysis page; render it structured if so,
    otherwise fall back to showing the raw text as-is."""
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


# ---------------------------------------------------------------------------
# Tab 2 — Disease Detection
# ---------------------------------------------------------------------------
def _render_disease_tab() -> None:
    rows = _load_rows(get_disease_analyses, "disease detection")
    if rows is None:
        return

    if not rows:
        callout(
            f"{icon_html('history', size=18)}No disease detection analyses saved yet. Go to "
            "<b>Disease Detection</b>, analyze a leaf image, and click "
            "<b>Save Analysis</b> to see records here."
        )
        return

    df = _disease_to_dataframe(rows)

    st.markdown("#### Filters & sorting")
    f1, f2, f3, f4 = st.columns([1.2, 1.2, 1.4, 1.2])
    with f1:
        crop_options = ["All"] + sorted(df["crop"].unique())
        crop_filter = st.multiselect("Crop", crop_options, default=["All"], key="_hist_disease_crop")
    with f2:
        disease_options = ["All"] + sorted(df["disease"].dropna().unique())
        disease_filter = st.multiselect("Disease", disease_options, default=["All"], key="_hist_disease_disease")
    with f3:
        min_date, max_date = df["_dt"].min().date(), df["_dt"].max().date()
        date_range = st.date_input("Date range", value=(min_date, max_date),
                                   min_value=min_date, max_value=max_date, key="_hist_disease_dates")
    with f4:
        sort_choice = st.selectbox("Sort by", list(DISEASE_SORT_OPTIONS.keys()), key="_hist_disease_sort")

    crop_matches = "All" in crop_filter or df["crop"].isin(crop_filter)
    disease_matches = "All" in disease_filter or df["disease"].isin(disease_filter)
    mask = crop_matches & disease_matches
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = date_range
        mask &= (df["_dt"].dt.date >= start) & (df["_dt"].dt.date <= end)

    sort_col, ascending = DISEASE_SORT_OPTIONS[sort_choice]
    view = df[mask].sort_values(sort_col, ascending=ascending)

    st.markdown(f"#### {len(view)} analyses")
    if len(view) == 0:
        callout("No analyses match the current filters.")
        return

    st.dataframe(
        view[["id", "date", "crop", "disease", "confidence", "severity",
              "health_label", "recommendation_summary"]],
        use_container_width=True,
        column_config={
            "id": "ID", "date": "Date", "crop": "Crop", "disease": "Disease",
            "confidence": st.column_config.NumberColumn("Confidence", format="%.0f%%"),
            "severity": "Severity", "health_label": "Result",
            "recommendation_summary": "Recommendation",
        },
        hide_index=True,
    )

    st.markdown("#### Records")
    st.caption("Expand a record to view the analyzed image and full details, or delete it.")
    for _, row in view.iterrows():
        _render_disease_record(row)


def _disease_to_dataframe(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df = df.rename(columns={"crop_name": "crop"})
    df = _prep_datetime(df)

    for col in ("disease", "severity", "recommendation", "image_path"):
        if col not in df.columns:
            df[col] = None

    df["disease"] = df["disease"].fillna("Unknown")
    df["confidence"] = pd.to_numeric(df.get("confidence"), errors="coerce") * 100
    df["is_healthy"] = pd.to_numeric(df.get("is_healthy"), errors="coerce").fillna(0).astype(bool)
    df["health_label"] = df["is_healthy"].map({True: "Healthy", False: "Diseased"})
    df["recommendation_summary"] = df["recommendation"].apply(_summary_of)
    return df


def _render_disease_record(row: pd.Series) -> None:
    row_id = int(row["id"])
    label = f"#{row_id} · {row['crop']} · {row['disease']} · {row['date']}"

    with st.expander(label):
        img_col, info_col = st.columns([1, 2])
        with img_col:
            image_path = row.get("image_path")
            if image_path and isinstance(image_path, str) and os.path.exists(image_path):
                st.image(image_path, caption="Analyzed leaf", use_container_width=True)
            else:
                st.caption("Image not available (file may have been moved or removed).")

        with info_col:
            c1, c2 = st.columns(2)
            with c1:
                _history_metric("Crop", row["crop"])
            with c2:
                _history_metric("Result", row["health_label"])

            c3, c4 = st.columns(2)
            with c3:
                _history_metric("Disease", row["disease"] or "—")
            with c4:
                _history_metric("Confidence", f"{row['confidence']:.0f}%" if pd.notna(row["confidence"]) else "—")

            c5, c6 = st.columns(2)
            with c5:
                _history_metric("Severity", row["severity"] or "—")
            with c6:
                _history_metric("Date", row["date"])

        st.markdown("**Recommendation**")
        st.markdown(row["recommendation"] or "_No recommendation recorded._")

        st.markdown("---")
        _render_delete_control(row_id, delete_disease_analysis, key_prefix="_hist_disease", label="disease analysis")


# ---------------------------------------------------------------------------
# Tab 3 — Environmental
# ---------------------------------------------------------------------------
def _render_env_tab() -> None:
    rows = _load_rows(get_environment_analyses, "environmental analysis")
    if rows is None:
        return

    if not rows:
        callout(
            f"{icon_html('history', size=18)}No environmental analyses saved yet. Go to "
            "<b>Environmental Analysis</b>, assess a reading, and click "
            "<b>Save Analysis</b> to see records here."
        )
        return

    df = _env_to_dataframe(rows)

    st.markdown("#### Filters & sorting")
    f1, f2, f3, f4 = st.columns([1.2, 1.2, 1.4, 1.2])
    with f1:
        crop_options = ["All"] + sorted(df["crop"].unique())
        crop_filter = st.multiselect("Crop", crop_options, default=["All"], key="_hist_env_crop")
    with f2:
        risk_options = ["All"] + sorted(df["risk_level"].dropna().unique())
        risk_filter = st.multiselect("Risk level", risk_options, default=["All"], key="_hist_env_risk")
    with f3:
        min_date, max_date = df["_dt"].min().date(), df["_dt"].max().date()
        date_range = st.date_input("Date range", value=(min_date, max_date),
                                   min_value=min_date, max_value=max_date, key="_hist_env_dates")
    with f4:
        sort_choice = st.selectbox("Sort by", list(ENV_SORT_OPTIONS.keys()), key="_hist_env_sort")

    crop_matches = "All" in crop_filter or df["crop"].isin(crop_filter)
    risk_matches = "All" in risk_filter or df["risk_level"].isin(risk_filter)
    mask = crop_matches & risk_matches
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = date_range
        mask &= (df["_dt"].dt.date >= start) & (df["_dt"].dt.date <= end)

    sort_col, ascending = ENV_SORT_OPTIONS[sort_choice]
    view = df[mask].sort_values(sort_col, ascending=ascending)

    st.markdown(f"#### {len(view)} analyses")
    if len(view) == 0:
        callout("No analyses match the current filters.")
        return

    st.dataframe(
        view[["id", "date", "crop", "temperature", "humidity", "soil_moisture",
              "rainfall", "risk_level", "health_score", "recommendation_summary"]],
        use_container_width=True,
        column_config={
            "id": "ID", "date": "Date", "crop": "Crop",
            "temperature": st.column_config.NumberColumn("Temp (°C)", format="%.1f"),
            "humidity": st.column_config.NumberColumn("Humidity (%)", format="%.1f"),
            "soil_moisture": st.column_config.NumberColumn("Soil moist. (%)", format="%.1f"),
            "rainfall": st.column_config.NumberColumn("Rainfall (mm)", format="%.1f"),
            "risk_level": "Risk level",
            "health_score": st.column_config.ProgressColumn(
                "Health score", min_value=0, max_value=100, format="%d"),
            "recommendation_summary": "Recommendation",
        },
        hide_index=True,
    )

    st.markdown("#### Records")
    st.caption("Expand a record to view full details or delete it.")
    for _, row in view.iterrows():
        _render_env_record(row)


def _env_to_dataframe(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df = df.rename(columns={"crop_name": "crop"})
    df = _prep_datetime(df)

    for col in ("risk_level", "recommendation"):
        if col not in df.columns:
            df[col] = None

    df["risk_level"] = df["risk_level"].fillna("Unknown")
    df["health_score"] = pd.to_numeric(df.get("health_score"), errors="coerce").fillna(0).astype(int)
    df["probability"] = pd.to_numeric(df.get("probability"), errors="coerce") * 100
    df["recommendation_summary"] = df["recommendation"].apply(_summary_of)
    return df


def _render_env_record(row: pd.Series) -> None:
    row_id = int(row["id"])
    label = f"#{row_id} · {row['crop']} · {row['risk_level']} · {row['date']}"
    risk_color = RISK_LEVELS.get(row["risk_level"], ("Unknown", "#93998A", 0.5))[1]

    with st.expander(label):
        c1, c2 = st.columns(2)
        with c1:
            _history_metric("Crop", row["crop"])
        with c2:
            st.markdown(
                f"<div class='metric-tile history-metric'>"
                f"<div class='label'>Risk level</div>"
                f"<div class='value' style='color:{risk_color}'>{escape(str(row['risk_level']))}</div>"
                "</div>",
                unsafe_allow_html=True,
            )

        c3, c4 = st.columns(2)
        with c3:
            _history_metric("Health score", int(row["health_score"]))
        with c4:
            _history_metric("Model confidence", f"{row['probability']:.0f}%" if pd.notna(row["probability"]) else "—")

        e1, e2, e3, e4 = st.columns(4)
        with e1:
            _history_metric("Temperature", f"{row['temperature']} °C" if pd.notna(row["temperature"]) else "—")
        with e2:
            _history_metric("Humidity", f"{row['humidity']} %" if pd.notna(row["humidity"]) else "—")
        with e3:
            _history_metric("Soil moisture", f"{row['soil_moisture']} %" if pd.notna(row["soil_moisture"]) else "—")
        with e4:
            _history_metric("Rainfall", f"{row['rainfall']} mm" if pd.notna(row["rainfall"]) else "—")

        _history_metric("Date", row["date"])

        st.markdown("**Recommendation**")
        st.markdown(row["recommendation"] or "_No recommendation recorded._")

        st.markdown("---")
        _render_delete_control(row_id, delete_environment_analysis, key_prefix="_hist_env", label="environmental analysis")