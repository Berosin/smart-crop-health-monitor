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
from src.i18n import get_language, tr_label, tr_template, tr_crop, tr_disease, tr_severity
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
    lang = get_language()
    page_header(
        "history",
        tr_label("Analysis History", lang),
        tr_label("Review, filter, and manage previously saved analyses.", lang),
    )

    tab_health, tab_disease, tab_env = st.tabs(
        [tr_label("Crop Health", lang), tr_label("Disease Detection", lang), tr_label("Environmental", lang)]
    )
    with tab_health:
        _render_health_tab(lang)
    with tab_disease:
        _render_disease_tab(lang)
    with tab_env:
        _render_env_tab(lang)

    footer()


# ---------------------------------------------------------------------------
# Shared helpers (used across all three tabs)
# ---------------------------------------------------------------------------
def _load_rows(loader, label: str, lang: str) -> list[dict] | None:
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
            tr_template(
                "Loading {label} history failed unexpectedly. Please try again. "
                "If the problem continues, contact the app maintainer.",
                lang, label=label,
            )
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


def _render_delete_control(row_id: int, delete_fn, key_prefix: str, label: str, lang: str) -> None:
    """Two-step delete: first click asks for confirmation, second click
    actually deletes — avoids removing a record from a single accidental
    click. Shared across all three tabs; key_prefix keeps widget keys
    from colliding between tabs when the same row id appears in more than
    one table.
    """
    confirm_key = f"{key_prefix}_confirm_delete_{row_id}"
    label_ta = tr_label(label, lang)

    if not st.session_state.get(confirm_key, False):
        if st.button(tr_template("Delete {label}", lang, label=label_ta), key=f"{key_prefix}_delete_btn_{row_id}"):
            st.session_state[confirm_key] = True
            st.rerun()
        return

    st.warning(
        tr_template(
            "Delete this {label} (#{row_id})? This cannot be undone.",
            lang, label=label_ta, row_id=row_id,
        )
    )
    yes_col, no_col = st.columns(2)
    with yes_col:
        if st.button(tr_label("Yes, delete", lang), key=f"{key_prefix}_confirm_yes_{row_id}", type="primary"):
            with safe_action(f"Deleting {label}"):
                delete_fn(row_id)
                st.session_state.pop(confirm_key, None)
                st.success(tr_template("{label} #{row_id} deleted.", lang, label=label_ta.capitalize(), row_id=row_id))
                st.rerun()
    with no_col:
        if st.button(tr_label("Cancel", lang), key=f"{key_prefix}_confirm_no_{row_id}"):
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
def _render_health_tab(lang: str) -> None:
    rows = _load_rows(get_analyses, "crop health analysis", lang)
    if rows is None:
        return

    if not rows:
        callout(
            f"{icon_html('history', size=18)}"
            + tr_label(
                "No crop health analyses saved yet. Go to "
                "<b>Crop Health Analysis</b>, run a calculation, and click "
                "<b>Save Analysis</b> to see records here.",
                lang,
            )
        )
        return

    df = _health_to_dataframe(rows, lang)

    st.markdown(f"#### {tr_label('Filters & sorting', lang)}")
    f1, f2, f3, f4 = st.columns([1.2, 1.2, 1.4, 1.2])
    with f1:
        crop_options = ["All"] + sorted(df["crop"].unique())
        crop_filter = st.multiselect(
            tr_label("Crop", lang), crop_options, default=["All"], key="_hist_health_crop",
            format_func=lambda c: tr_label(c, lang) if c == "All" else tr_crop(c, lang),
        )
    with f2:
        disease_options = ["All"] + sorted(df["disease"].dropna().unique())
        disease_filter = st.multiselect(
            tr_label("Disease", lang), disease_options, default=["All"], key="_hist_health_disease",
            format_func=lambda d: tr_label(d, lang) if d in ("All", "Unknown") else tr_disease(d, lang),
        )
    with f3:
        min_date, max_date = df["_dt"].min().date(), df["_dt"].max().date()
        date_range = st.date_input(tr_label("Date range", lang), value=(min_date, max_date),
                                   min_value=min_date, max_value=max_date, key="_hist_health_dates")
    with f4:
        sort_choice = st.selectbox(
            tr_label("Sort by", lang), list(HEALTH_SORT_OPTIONS.keys()), key="_hist_health_sort",
            format_func=lambda o: tr_label(o, lang),
        )

    crop_matches = "All" in crop_filter or df["crop"].isin(crop_filter)
    disease_matches = "All" in disease_filter or df["disease"].isin(disease_filter)
    mask = crop_matches & disease_matches
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = date_range
        mask &= (df["_dt"].dt.date >= start) & (df["_dt"].dt.date <= end)

    sort_col, ascending = HEALTH_SORT_OPTIONS[sort_choice]
    view = df[mask].sort_values(sort_col, ascending=ascending)

    st.markdown(f"#### {len(view)} {tr_label('analyses', lang)}")
    if len(view) == 0:
        callout(tr_label("No analyses match the current filters.", lang))
        return

    display = view.copy()
    display["crop"] = display["crop"].apply(lambda c: tr_crop(c, lang))
    display["disease"] = display["disease"].apply(lambda d: tr_label(d, lang) if d == "Unknown" else tr_disease(d, lang))
    display["severity"] = display["severity"].apply(lambda s: tr_severity(s, lang) if s else s)
    display["risk"] = view.apply(
        lambda r: f"{tr_label('Disease', lang)}: {tr_label(r['disease_risk'], lang) if r['disease_risk'] else '—'} · "
                  f"{tr_label('Environment', lang)}: {tr_label(r['environmental_risk'], lang) if r['environmental_risk'] else '—'}",
        axis=1,
    )

    st.dataframe(
        display[["id", "date", "crop", "disease", "confidence", "severity",
                 "health_score", "risk", "recommendation_summary"]],
        use_container_width=True,
        column_config={
            "id": tr_label("ID", lang), "date": tr_label("Date", lang), "crop": tr_label("Crop", lang),
            "disease": tr_label("Disease", lang),
            "confidence": st.column_config.NumberColumn(tr_label("Confidence", lang), format="%.0f%%"),
            "severity": tr_label("Severity", lang),
            "health_score": st.column_config.ProgressColumn(
                tr_label("Health score", lang), min_value=0, max_value=100, format="%d"),
            "risk": tr_label("Risk", lang), "recommendation_summary": tr_label("Recommendation", lang),
        },
        hide_index=True,
    )

    st.markdown(f"#### {tr_label('Records', lang)}")
    st.caption(tr_label("Expand a record to view full details or delete it.", lang))
    for _, row in view.iterrows():
        _render_health_record(row, lang)


def _health_to_dataframe(rows: list[dict], lang: str) -> pd.DataFrame:
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


def _render_health_record(row: pd.Series, lang: str) -> None:
    row_id = int(row["id"])
    label = f"#{row_id} · {tr_crop(row['crop'], lang)} · {tr_disease(row['disease'], lang)} · {row['date']}"

    with st.expander(label):
        c1, c2 = st.columns(2)
        with c1:
            _history_metric(tr_label("Crop", lang), tr_crop(row["crop"], lang))
        with c2:
            _history_metric(tr_label("Status", lang), tr_label(row["status"], lang))

        c3, c4 = st.columns(2)
        with c3:
            _history_metric(tr_label("Health score", lang), int(row["health_score"]))
        with c4:
            _history_metric(tr_label("Date", lang), row["date"])

        d1, d2, d3 = st.columns(3)
        with d1:
            _history_metric(tr_label("Disease", lang), tr_disease(row["disease"], lang) if row["disease"] else "—")
        with d2:
            _history_metric(tr_label("Confidence", lang), f"{row['confidence']:.0f}%" if pd.notna(row["confidence"]) else "—")
        with d3:
            _history_metric(tr_label("Severity", lang), tr_severity(row["severity"], lang) if row["severity"] else "—")

        r1, r2 = st.columns(2)
        with r1:
            _history_metric(tr_label("Disease risk", lang), tr_label(row["disease_risk"], lang) if row["disease_risk"] else "—")
        with r2:
            _history_metric(tr_label("Environmental risk", lang), tr_label(row["environmental_risk"], lang) if row["environmental_risk"] else "—")

        st.markdown(f"**{tr_label('Recommendation', lang)}**")
        _render_structured_recommendation(row["recommendation"], lang)

        st.markdown("---")
        _render_delete_control(row_id, delete_analysis, key_prefix="_hist_health", label="crop health analysis", lang=lang)


def _render_structured_recommendation(raw: str | None, lang: str) -> None:
    """Recommendation is stored as JSON from src.recommendation_engine when
    saved from the Crop Health Analysis page; render it structured if so,
    otherwise fall back to showing the raw text as-is."""
    if not raw:
        st.caption(tr_label("No recommendation recorded.", lang))
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
        category_label = tr_label(rec.get("category", ""), lang).title()
        priority_label = tr_label(rec.get("priority", ""), lang).upper()
        items += (
            "<div class='rec-item'>"
            f"<div class='rec-icon'>{icon_tag}</div>"
            "<div>"
            f"<div class='rec-title'>{category_label} · "
            f"<span style='color:{badge_color}'>{priority_label}</span></div>"
            f"<div class='rec-text'>{rec.get('text','')}</div>"
            "</div></div>"
        )
    st.markdown(items, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Tab 2 — Disease Detection
# ---------------------------------------------------------------------------
def _render_disease_tab(lang: str) -> None:
    rows = _load_rows(get_disease_analyses, "disease detection", lang)
    if rows is None:
        return

    if not rows:
        callout(
            f"{icon_html('history', size=18)}"
            + tr_label(
                "No disease detection analyses saved yet. Go to "
                "<b>Disease Detection</b>, analyze a leaf image, and click "
                "<b>Save Analysis</b> to see records here.",
                lang,
            )
        )
        return

    df = _disease_to_dataframe(rows)

    st.markdown(f"#### {tr_label('Filters & sorting', lang)}")
    f1, f2, f3, f4 = st.columns([1.2, 1.2, 1.4, 1.2])
    with f1:
        crop_options = ["All"] + sorted(df["crop"].unique())
        crop_filter = st.multiselect(
            tr_label("Crop", lang), crop_options, default=["All"], key="_hist_disease_crop",
            format_func=lambda c: tr_label(c, lang) if c == "All" else tr_crop(c, lang),
        )
    with f2:
        disease_options = ["All"] + sorted(df["disease"].dropna().unique())
        disease_filter = st.multiselect(
            tr_label("Disease", lang), disease_options, default=["All"], key="_hist_disease_disease",
            format_func=lambda d: tr_label(d, lang) if d in ("All", "Unknown") else tr_disease(d, lang),
        )
    with f3:
        min_date, max_date = df["_dt"].min().date(), df["_dt"].max().date()
        date_range = st.date_input(tr_label("Date range", lang), value=(min_date, max_date),
                                   min_value=min_date, max_value=max_date, key="_hist_disease_dates")
    with f4:
        sort_choice = st.selectbox(
            tr_label("Sort by", lang), list(DISEASE_SORT_OPTIONS.keys()), key="_hist_disease_sort",
            format_func=lambda o: tr_label(o, lang),
        )

    crop_matches = "All" in crop_filter or df["crop"].isin(crop_filter)
    disease_matches = "All" in disease_filter or df["disease"].isin(disease_filter)
    mask = crop_matches & disease_matches
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = date_range
        mask &= (df["_dt"].dt.date >= start) & (df["_dt"].dt.date <= end)

    sort_col, ascending = DISEASE_SORT_OPTIONS[sort_choice]
    view = df[mask].sort_values(sort_col, ascending=ascending)

    st.markdown(f"#### {len(view)} {tr_label('analyses', lang)}")
    if len(view) == 0:
        callout(tr_label("No analyses match the current filters.", lang))
        return

    display = view.copy()
    display["crop"] = display["crop"].apply(lambda c: tr_crop(c, lang))
    display["disease"] = display["disease"].apply(lambda d: tr_label(d, lang) if d == "Unknown" else tr_disease(d, lang))
    display["severity"] = display["severity"].apply(lambda s: tr_severity(s, lang) if s else s)
    display["health_label"] = display["health_label"].apply(lambda h: tr_label(h, lang))

    st.dataframe(
        display[["id", "date", "crop", "disease", "confidence", "severity",
                 "health_label", "recommendation_summary"]],
        use_container_width=True,
        column_config={
            "id": tr_label("ID", lang), "date": tr_label("Date", lang), "crop": tr_label("Crop", lang),
            "disease": tr_label("Disease", lang),
            "confidence": st.column_config.NumberColumn(tr_label("Confidence", lang), format="%.0f%%"),
            "severity": tr_label("Severity", lang), "health_label": tr_label("Result", lang),
            "recommendation_summary": tr_label("Recommendation", lang),
        },
        hide_index=True,
    )

    st.markdown(f"#### {tr_label('Records', lang)}")
    st.caption(tr_label("Expand a record to view the analyzed image and full details, or delete it.", lang))
    for _, row in view.iterrows():
        _render_disease_record(row, lang)


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


def _render_disease_record(row: pd.Series, lang: str) -> None:
    row_id = int(row["id"])
    label = f"#{row_id} · {tr_crop(row['crop'], lang)} · {tr_disease(row['disease'], lang)} · {row['date']}"

    with st.expander(label):
        img_col, info_col = st.columns([1, 2])
        with img_col:
            image_path = row.get("image_path")
            if image_path and isinstance(image_path, str) and os.path.exists(image_path):
                st.image(image_path, caption=tr_label("Analyzed leaf", lang), use_container_width=True)
            else:
                st.caption(tr_label("Image not available (file may have been moved or removed).", lang))

        with info_col:
            c1, c2 = st.columns(2)
            with c1:
                _history_metric(tr_label("Crop", lang), tr_crop(row["crop"], lang))
            with c2:
                _history_metric(tr_label("Result", lang), tr_label(row["health_label"], lang))

            c3, c4 = st.columns(2)
            with c3:
                _history_metric(tr_label("Disease", lang), tr_disease(row["disease"], lang) if row["disease"] else "—")
            with c4:
                _history_metric(tr_label("Confidence", lang), f"{row['confidence']:.0f}%" if pd.notna(row["confidence"]) else "—")

            c5, c6 = st.columns(2)
            with c5:
                _history_metric(tr_label("Severity", lang), tr_severity(row["severity"], lang) if row["severity"] else "—")
            with c6:
                _history_metric(tr_label("Date", lang), row["date"])

        st.markdown(f"**{tr_label('Recommendation', lang)}**")
        st.markdown(row["recommendation"] or f"_{tr_label('No recommendation recorded.', lang)}_")

        st.markdown("---")
        _render_delete_control(row_id, delete_disease_analysis, key_prefix="_hist_disease", label="disease analysis", lang=lang)


# ---------------------------------------------------------------------------
# Tab 3 — Environmental
# ---------------------------------------------------------------------------
def _render_env_tab(lang: str) -> None:
    rows = _load_rows(get_environment_analyses, "environmental analysis", lang)
    if rows is None:
        return

    if not rows:
        callout(
            f"{icon_html('history', size=18)}"
            + tr_label(
                "No environmental analyses saved yet. Go to "
                "<b>Environmental Analysis</b>, assess a reading, and click "
                "<b>Save Analysis</b> to see records here.",
                lang,
            )
        )
        return

    df = _env_to_dataframe(rows)

    st.markdown(f"#### {tr_label('Filters & sorting', lang)}")
    f1, f2, f3, f4 = st.columns([1.2, 1.2, 1.4, 1.2])
    with f1:
        crop_options = ["All"] + sorted(df["crop"].unique())
        crop_filter = st.multiselect(
            tr_label("Crop", lang), crop_options, default=["All"], key="_hist_env_crop",
            format_func=lambda c: tr_label(c, lang) if c == "All" else tr_crop(c, lang),
        )
    with f2:
        risk_options = ["All"] + sorted(df["risk_level"].dropna().unique())
        risk_filter = st.multiselect(
            tr_label("Risk level", lang), risk_options, default=["All"], key="_hist_env_risk",
            format_func=lambda r: tr_label(r, lang),
        )
    with f3:
        min_date, max_date = df["_dt"].min().date(), df["_dt"].max().date()
        date_range = st.date_input(tr_label("Date range", lang), value=(min_date, max_date),
                                   min_value=min_date, max_value=max_date, key="_hist_env_dates")
    with f4:
        sort_choice = st.selectbox(
            tr_label("Sort by", lang), list(ENV_SORT_OPTIONS.keys()), key="_hist_env_sort",
            format_func=lambda o: tr_label(o, lang),
        )

    crop_matches = "All" in crop_filter or df["crop"].isin(crop_filter)
    risk_matches = "All" in risk_filter or df["risk_level"].isin(risk_filter)
    mask = crop_matches & risk_matches
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = date_range
        mask &= (df["_dt"].dt.date >= start) & (df["_dt"].dt.date <= end)

    sort_col, ascending = ENV_SORT_OPTIONS[sort_choice]
    view = df[mask].sort_values(sort_col, ascending=ascending)

    st.markdown(f"#### {len(view)} {tr_label('analyses', lang)}")
    if len(view) == 0:
        callout(tr_label("No analyses match the current filters.", lang))
        return

    display = view.copy()
    display["crop"] = display["crop"].apply(lambda c: tr_crop(c, lang))
    display["risk_level"] = display["risk_level"].apply(lambda r: tr_label(r, lang))

    st.dataframe(
        display[["id", "date", "crop", "temperature", "humidity", "soil_moisture",
                 "rainfall", "risk_level", "health_score", "recommendation_summary"]],
        use_container_width=True,
        column_config={
            "id": tr_label("ID", lang), "date": tr_label("Date", lang), "crop": tr_label("Crop", lang),
            "temperature": st.column_config.NumberColumn(tr_label("Temp (°C)", lang), format="%.1f"),
            "humidity": st.column_config.NumberColumn(tr_label("Humidity (%)", lang), format="%.1f"),
            "soil_moisture": st.column_config.NumberColumn(tr_label("Soil moist. (%)", lang), format="%.1f"),
            "rainfall": st.column_config.NumberColumn(tr_label("Rainfall (mm)", lang), format="%.1f"),
            "risk_level": tr_label("Risk level", lang),
            "health_score": st.column_config.ProgressColumn(
                tr_label("Health score", lang), min_value=0, max_value=100, format="%d"),
            "recommendation_summary": tr_label("Recommendation", lang),
        },
        hide_index=True,
    )

    st.markdown(f"#### {tr_label('Records', lang)}")
    st.caption(tr_label("Expand a record to view full details or delete it.", lang))
    for _, row in view.iterrows():
        _render_env_record(row, lang)


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


def _render_env_record(row: pd.Series, lang: str) -> None:
    row_id = int(row["id"])
    label = f"#{row_id} · {tr_crop(row['crop'], lang)} · {tr_label(row['risk_level'], lang)} · {row['date']}"
    risk_color = RISK_LEVELS.get(row["risk_level"], ("Unknown", "#93998A", 0.5))[1]

    with st.expander(label):
        c1, c2 = st.columns(2)
        with c1:
            _history_metric(tr_label("Crop", lang), tr_crop(row["crop"], lang))
        with c2:
            st.markdown(
                f"<div class='metric-tile history-metric'>"
                f"<div class='label'>{tr_label('Risk level', lang)}</div>"
                f"<div class='value' style='color:{risk_color}'>{escape(str(tr_label(row['risk_level'], lang)))}</div>"
                "</div>",
                unsafe_allow_html=True,
            )

        c3, c4 = st.columns(2)
        with c3:
            _history_metric(tr_label("Health score", lang), int(row["health_score"]))
        with c4:
            _history_metric(tr_label("Model confidence", lang), f"{row['probability']:.0f}%" if pd.notna(row["probability"]) else "—")

        e1, e2, e3, e4 = st.columns(4)
        with e1:
            _history_metric(tr_label("Temperature", lang), f"{row['temperature']} °C" if pd.notna(row["temperature"]) else "—")
        with e2:
            _history_metric(tr_label("Humidity", lang), f"{row['humidity']} %" if pd.notna(row["humidity"]) else "—")
        with e3:
            _history_metric(tr_label("Soil moisture", lang), f"{row['soil_moisture']} %" if pd.notna(row["soil_moisture"]) else "—")
        with e4:
            _history_metric(tr_label("Rainfall", lang), f"{row['rainfall']} mm" if pd.notna(row["rainfall"]) else "—")

        _history_metric(tr_label("Date", lang), row["date"])

        st.markdown(f"**{tr_label('Recommendation', lang)}**")
        st.markdown(row["recommendation"] or f"_{tr_label('No recommendation recorded.', lang)}_")

        st.markdown("---")
        _render_delete_control(row_id, delete_environment_analysis, key_prefix="_hist_env", label="environmental analysis", lang=lang)