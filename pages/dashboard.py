"""Dashboard page — statistics and charts over real saved analyses.

Reads from SQLite via src.db.get_analyses(). Shows a friendly empty state
instead of any charts/metrics when nothing has been saved yet.
"""

from __future__ import annotations

import json
from collections import Counter

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.db import get_analyses, get_disease_analyses, get_environment_analyses, get_field_scans
from src.errors import DatabaseError, logger
from src.health_engine import classify_health_status, HEALTH_STATUS_BANDS
from src.outbreak_detection import load_outbreak_signals, get_active_alerts
from src.i18n import get_language, tr_label, tr_template, tr_crop, tr_disease, tr_severity
from utils.ui import (
    page_header,
    callout,
    footer,
    pretty_name,
    CHART_THEME,
    RISK_LEVELS,
)
from utils.icons import icon_html

# Health-status band -> color, consistent with the spectral severity ramp
# used elsewhere in the app (page header / health-card gradient bar).
STATUS_COLORS = {
    "Healthy":  "#2F6D46",
    "Moderate": "#D6A34B",
    "At Risk":  "#C97A3B",
    "Critical": "#7C3730",
}
STATUS_ORDER = [label for _, _, label in HEALTH_STATUS_BANDS]  # Healthy..Critical

# Rotating palette for disease names beyond "Healthy" (which is always green).
DISEASE_PALETTE = ["#CE8C82", "#8D74A6", "#CB8A5C", "#B6708E", "#7FA687", "#B5564B", "#D6A34B"]

# health_score below this = "high-risk" for the KPI tile (kept in sync with
# the At Risk / Critical bands from src.health_engine).
HIGH_RISK_STATUSES = {"At Risk", "Critical"}


def render() -> None:
    lang = get_language()
    page_header(
        "dashboard",
        tr_label("Dashboard", lang),
        tr_label("Statistics and trends across all crop, disease, and environmental analyses.", lang),
    )

    _render_outbreak_banner(lang)

    tab_health, tab_disease, tab_field, tab_env = st.tabs(
        [tr_label("Crop Health", lang), tr_label("Disease Detection", lang),
         tr_label("Field Scans", lang), tr_label("Environmental", lang)]
    )
    with tab_health:
        _render_health_tab(lang)
    with tab_disease:
        _render_disease_tab(lang)
    with tab_field:
        _render_field_scan_tab(lang)
    with tab_env:
        _render_env_tab(lang)

    footer()


def _render_outbreak_banner(lang: str) -> None:
    """Compact 'go check Outbreak Alerts' banner, shown above the tabs.

    Reuses src.outbreak_detection end to end — same signal the dedicated
    Outbreak Alerts page and the Home page banner use — so all three
    surfaces always agree. Silently skipped on error so a hiccup here
    never blocks the rest of the dashboard from rendering.
    """
    try:
        signals = load_outbreak_signals(window=7, lang=lang)
        active = get_active_alerts(signals)
    except Exception:
        logger.exception("Unexpected error checking outbreak alerts on dashboard")
        return

    if not active:
        return

    crop_list = ", ".join(f"<b>{tr_crop(s['crop'], lang)}</b> ({tr_label(s['risk_level'], lang)})" for s in active)
    st.markdown(
        f"""
        <div class="callout" style="border-left-color:#B5564B;background:#FBEFED">
          {icon_html('diseased', size=18)}<b>{len(active)} {tr_label('crop(s) trending worse', lang)}</b> — {crop_list}.
          {tr_label('See the Outbreak Alerts page for details.', lang)}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Tab 1 — Crop Health (existing charts, unchanged, sourced from `analyses`)
# ---------------------------------------------------------------------------
def _render_health_tab(lang: str) -> None:
    try:
        rows = get_analyses(limit=2000)
    except DatabaseError as e:
        st.error(str(e))
        return
    except Exception:
        logger.exception("Unexpected error loading dashboard data")
        st.error(
            tr_label(
                "Loading dashboard data failed unexpectedly. Please try again. "
                "If the problem continues, contact the app maintainer.",
                lang,
            )
        )
        return

    if not rows:
        callout(
            f"{icon_html('dashboard', size=18)}"
            + tr_label(
                "No crop health analyses saved yet. Run a "
                "calculation on the <b>Crop Health Analysis</b> page and click "
                "<b>Save Analysis</b> to populate this tab.",
                lang,
            )
        )
        return

    df = _load_dataframe(rows)

    # ---- Derived KPIs ---------------------------------------------------
    total = len(df)
    healthy = int(df["is_healthy"].sum())
    diseased = total - healthy
    avg_score = round(df["health_score"].mean(), 1)
    high_risk = int(df["status"].isin(HIGH_RISK_STATUSES).sum())

    # ---- KPI cards ------------------------------------------------------
    _kpi_row_health(total, healthy, diseased, avg_score, high_risk, lang)

    st.markdown("---")

    # ---- Charts ---------------------------------------------------------
    _chart_disease_distribution(df, lang)
    _chart_health_score_trend(df, lang)
    _chart_crop_wise(df, lang)
    _chart_risk_distribution(df, lang)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _load_dataframe(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df = df.rename(columns={"crop_name": "crop"})

    df["disease"] = df.get("disease").fillna("Healthy") if "disease" in df else "Healthy"
    df["severity"] = df.get("severity") if "severity" in df else None
    df["health_score"] = pd.to_numeric(df.get("health_score"), errors="coerce").fillna(0).astype(int)
    df["status"] = df["health_score"].apply(classify_health_status)

    df["is_healthy"] = (
        df["disease"].str.strip().str.lower().isin(["healthy", "none", ""])
        | (df["severity"] == "None")
    )

    df["_dt"] = pd.to_datetime(df.get("created_at"), errors="coerce", utc=True).dt.tz_localize(None)
    df["_dt"] = df["_dt"].fillna(pd.Timestamp.now())
    df["date"] = df["_dt"].dt.strftime("%Y-%m-%d")

    return df


# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------
TEMPLATE = (
    """
    <div class="metric-tile" style="border-left-color:{accent}">
      <div class="label">{label}</div>
      <div class="value" style="color:{ink}">{value}</div>
      {delta_html}
    </div>
"""
)


def _tile(label: str, value: str, accent: str, ink: str,
          delta: str | None = None) -> None:
    delta_html = f'<div class="delta">{delta}</div>' if delta else ""
    st.markdown(
        TEMPLATE.format(label=label, value=value, accent=accent, ink=ink,
                        delta_html=delta_html),
        unsafe_allow_html=True,
    )


def _kpi_row_health(total, healthy, diseased, avg_score, high_risk, lang: str) -> None:
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        _tile(tr_label("Total Analyses", lang), str(total), "#2F6D46", "#1C2E20", tr_label("all-time", lang))
    with c2:
        _tile(tr_label("Healthy Plants", lang), str(healthy), "#7FA687", "#1C2E20",
              f"{round(healthy/total*100)}% {tr_label('of total', lang)}")
    with c3:
        _tile(tr_label("Diseased Plants", lang), str(diseased), "#CE8C82", "#7C3730",
              f"{round(diseased/total*100)}% {tr_label('of total', lang)}")
    with c4:
        _tile(tr_label("Avg Health Score", lang), str(avg_score), "#D6A34B", "#1C2E20",
              tr_label("out of 100", lang))
    with c5:
        _tile(tr_label("High-Risk Cases", lang), str(high_risk), "#B5564B", "#7C3730",
              tr_label("At Risk + Critical", lang))


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def _translated_disease_label(name: str, lang: str) -> str:
    if str(name).strip().lower() == "healthy":
        return tr_label("Healthy", lang)
    return tr_disease(name, lang)


def _chart_disease_distribution(df: pd.DataFrame, lang: str) -> None:
    """Horizontal bar of how many saved analyses mention each disease."""
    counts = df["disease"].value_counts()
    colors = []
    palette_i = 0
    for name in counts.index:
        if name.strip().lower() == "healthy":
            colors.append("#2F6D46")
        else:
            colors.append(DISEASE_PALETTE[palette_i % len(DISEASE_PALETTE)])
            palette_i += 1

    fig = go.Figure(go.Bar(
        orientation="h",
        x=counts.values,
        y=[_translated_disease_label(n, lang) for n in counts.index],
        text=counts.values,
        textposition="outside",
        marker=dict(color=colors),
    ))
    fig.update_layout(
        **CHART_THEME,
        margin=dict(t=10, b=10),
        height=max(260, 40 * len(counts)),
        xaxis_title=tr_label("Number of analyses", lang),
        yaxis_title="",
        showlegend=False,
    )
    st.markdown(f"#### {tr_label('Disease distribution', lang)}")
    st.plotly_chart(fig, use_container_width=True)


def _chart_health_score_trend(df: pd.DataFrame, lang: str) -> None:
    """Line chart of health score over time (daily average) + volume bars."""
    trend = (
        df.groupby("date", as_index=False)["health_score"]
          .mean()
          .sort_values("date")
          .rename(columns={"health_score": "avg_score"})
    )
    counts = df.groupby("date", as_index=False).size().rename(columns={"size": "n"})

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=counts["date"], y=counts["n"],
        name=tr_label("Analyses per day", lang),
        yaxis="y2",
        marker=dict(color="#D8E2CC"),
        opacity=0.6,
    ))
    fig.add_trace(go.Scatter(
        x=trend["date"], y=trend["avg_score"],
        mode="lines+markers",
        name=tr_label("Avg health score", lang),
        line=dict(color="#2F6D46", width=3),
        marker=dict(size=8),
    ))
    fig.update_layout(
        **CHART_THEME,
        margin=dict(t=10, b=10),
        height=320,
        legend=dict(orientation="h", y=1.12),
        xaxis_title=tr_label("Date", lang),
        yaxis=dict(title=tr_label("Avg health score", lang), range=[0, 100]),
        yaxis2=dict(title=tr_label("Analyses / day", lang), overlaying="y", side="right",
                    showgrid=False),
    )
    st.markdown(f"#### {tr_label('Health score trend', lang)}")
    st.plotly_chart(fig, use_container_width=True)


def _chart_crop_wise(df: pd.DataFrame, lang: str) -> None:
    """Stacked bar: count of analyses per crop, broken down by health status."""
    tdf = df.copy()
    tdf["crop_ta"] = tdf["crop"].apply(lambda c: tr_crop(c, lang))
    tdf["status_ta"] = tdf["status"].apply(lambda s: tr_label(s, lang))
    status_order_ta = [tr_label(s, lang) for s in STATUS_ORDER]
    status_colors_ta = {tr_label(s, lang): c for s, c in STATUS_COLORS.items()}

    grouped = (
        tdf.groupby(["crop_ta", "status_ta"]).size()
          .reset_index(name="count")
    )
    fig = px.bar(
        grouped, x="crop_ta", y="count", color="status_ta",
        color_discrete_map=status_colors_ta,
        category_orders={"status_ta": status_order_ta},
        text="count",
    )
    fig.update_traces(textposition="inside")
    fig.update_layout(
        **CHART_THEME,
        margin=dict(t=10, b=10),
        height=340,
        xaxis_title=tr_label("Crop", lang),
        yaxis_title=tr_label("Number of analyses", lang),
        legend_title_text=tr_label("Status", lang),
        barmode="stack",
    )
    fig.update_xaxes(title_text=tr_label("Crop", lang))
    st.markdown(f"#### {tr_label('Crop-wise analysis', lang)}")
    st.plotly_chart(fig, use_container_width=True)


def _chart_risk_distribution(df: pd.DataFrame, lang: str) -> None:
    """Pie chart of health-status bands across all saved analyses."""
    counts = df["status"].value_counts().reindex(STATUS_ORDER).dropna()
    fig = go.Figure(go.Pie(
        labels=[tr_label(s, lang) for s in counts.index],
        values=counts.values,
        marker=dict(colors=[STATUS_COLORS[s] for s in counts.index]),
        hole=0.45,
        textinfo="label+percent",
    ))
    fig.update_layout(
        **CHART_THEME,
        margin=dict(t=10, b=10),
        height=340,
        showlegend=True,
        legend=dict(orientation="h", y=-0.1),
    )
    st.markdown(f"#### {tr_label('Risk distribution', lang)}")
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 2 — Disease Detection (sourced from `disease_analyses`)
# ---------------------------------------------------------------------------
SEVERITY_COLORS = {
    "None": "#2F6D46", "Mild": "#7FA687", "Moderate": "#C97A3B",
    "High": "#B5564B", "Unknown": "#93998A",
}
SEVERITY_ORDER = ["None", "Mild", "Moderate", "High", "Unknown"]


def _render_disease_tab(lang: str) -> None:
    try:
        rows = get_disease_analyses(limit=2000)
    except DatabaseError as e:
        st.error(str(e))
        return
    except Exception:
        logger.exception("Unexpected error loading disease detection dashboard data")
        st.error(
            tr_label(
                "Loading disease detection dashboard data failed unexpectedly. "
                "Please try again. If the problem continues, contact the app maintainer.",
                lang,
            )
        )
        return

    if not rows:
        callout(
            f"{icon_html('dashboard', size=18)}"
            + tr_label(
                "No disease detection analyses saved yet. "
                "Analyze a leaf image on the <b>Disease Detection</b> page and click "
                "<b>Save Analysis</b> to populate this tab.",
                lang,
            )
        )
        return

    df = _load_disease_dataframe(rows)

    total = len(df)
    healthy = int(df["is_healthy"].sum())
    diseased = total - healthy
    avg_confidence = round(df["confidence"].mean(), 1)
    high_severity = int((df["severity"] == "High").sum())

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        _tile(tr_label("Total Analyses", lang), str(total), "#2F6D46", "#1C2E20", tr_label("all-time", lang))
    with c2:
        _tile(tr_label("Healthy Leaves", lang), str(healthy), "#7FA687", "#1C2E20",
              f"{round(healthy/total*100)}% {tr_label('of total', lang)}")
    with c3:
        _tile(tr_label("Diseased Leaves", lang), str(diseased), "#CE8C82", "#7C3730",
              f"{round(diseased/total*100)}% {tr_label('of total', lang)}")
    with c4:
        _tile(tr_label("Avg Confidence", lang), f"{avg_confidence}%", "#D6A34B", "#1C2E20",
              tr_label("model output", lang))
    with c5:
        _tile(tr_label("High Severity", lang), str(high_severity), "#B5564B", "#7C3730",
              tr_label("cases flagged High", lang))

    st.markdown("---")

    _dd_chart_disease_distribution(df, lang)
    _dd_chart_confidence_trend(df, lang)
    _dd_chart_crop_wise(df, lang)
    _dd_chart_severity_distribution(df, lang)


def _load_disease_dataframe(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df = df.rename(columns={"crop_name": "crop"})
    df["disease"] = df.get("disease").fillna("Healthy") if "disease" in df else "Healthy"
    df["severity"] = df.get("severity").fillna("Unknown") if "severity" in df else "Unknown"
    df["confidence"] = pd.to_numeric(df.get("confidence"), errors="coerce").fillna(0) * 100
    df["is_healthy"] = pd.to_numeric(df.get("is_healthy"), errors="coerce").fillna(0).astype(bool)

    df["_dt"] = pd.to_datetime(df.get("created_at"), errors="coerce", utc=True).dt.tz_localize(None)
    df["_dt"] = df["_dt"].fillna(pd.Timestamp.now())
    df["date"] = df["_dt"].dt.strftime("%Y-%m-%d")
    return df


def _dd_chart_disease_distribution(df: pd.DataFrame, lang: str) -> None:
    """Horizontal bar of how many saved disease analyses mention each class."""
    counts = df["disease"].value_counts()
    colors = []
    palette_i = 0
    for name in counts.index:
        if name.strip().lower() == "healthy":
            colors.append("#2F6D46")
        else:
            colors.append(DISEASE_PALETTE[palette_i % len(DISEASE_PALETTE)])
            palette_i += 1

    fig = go.Figure(go.Bar(
        orientation="h", x=counts.values, y=[_translated_disease_label(n, lang) for n in counts.index],
        text=counts.values, textposition="outside",
        marker=dict(color=colors),
    ))
    fig.update_layout(
        **CHART_THEME, margin=dict(t=10, b=10), height=max(260, 40 * len(counts)),
        xaxis_title=tr_label("Number of analyses", lang), yaxis_title="", showlegend=False,
    )
    st.markdown(f"#### {tr_label('Disease distribution', lang)}")
    st.plotly_chart(fig, use_container_width=True)


def _dd_chart_confidence_trend(df: pd.DataFrame, lang: str) -> None:
    """Line chart of average model confidence over time + volume bars."""
    trend = (
        df.groupby("date", as_index=False)["confidence"]
          .mean().sort_values("date")
          .rename(columns={"confidence": "avg_confidence"})
    )
    counts = df.groupby("date", as_index=False).size().rename(columns={"size": "n"})

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=counts["date"], y=counts["n"], name=tr_label("Analyses per day", lang),
        yaxis="y2", marker=dict(color="#D8E2CC"), opacity=0.6,
    ))
    fig.add_trace(go.Scatter(
        x=trend["date"], y=trend["avg_confidence"], mode="lines+markers",
        name=tr_label("Avg confidence", lang), line=dict(color="#2F6D46", width=3), marker=dict(size=8),
    ))
    fig.update_layout(
        **CHART_THEME, margin=dict(t=10, b=10), height=320,
        legend=dict(orientation="h", y=1.12), xaxis_title=tr_label("Date", lang),
        yaxis=dict(title=tr_label("Avg confidence (%)", lang), range=[0, 100]),
        yaxis2=dict(title=tr_label("Analyses / day", lang), overlaying="y", side="right", showgrid=False),
    )
    st.markdown(f"#### {tr_label('Confidence trend', lang)}")
    st.plotly_chart(fig, use_container_width=True)


def _dd_chart_crop_wise(df: pd.DataFrame, lang: str) -> None:
    """Stacked bar: count of disease analyses per crop, broken down by disease."""
    tdf = df.copy()
    tdf["crop_ta"] = tdf["crop"].apply(lambda c: tr_crop(c, lang))
    tdf["disease_ta"] = tdf["disease"].apply(lambda d: _translated_disease_label(d, lang))
    grouped = tdf.groupby(["crop_ta", "disease_ta"]).size().reset_index(name="count")
    fig = px.bar(
        grouped, x="crop_ta", y="count", color="disease_ta",
        text="count",
    )
    fig.update_traces(textposition="inside")
    fig.update_layout(
        **CHART_THEME, margin=dict(t=10, b=10), height=340,
        xaxis_title=tr_label("Crop", lang), yaxis_title=tr_label("Number of analyses", lang),
        legend_title_text=tr_label("Disease", lang), barmode="stack",
    )
    st.markdown(f"#### {tr_label('Crop-wise analysis', lang)}")
    st.plotly_chart(fig, use_container_width=True)


def _dd_chart_severity_distribution(df: pd.DataFrame, lang: str) -> None:
    """Pie chart of severity levels across all saved disease analyses."""
    counts = df["severity"].value_counts().reindex(SEVERITY_ORDER).dropna()
    fig = go.Figure(go.Pie(
        labels=[tr_severity(s, lang) for s in counts.index], values=counts.values,
        marker=dict(colors=[SEVERITY_COLORS.get(s, "#93998A") for s in counts.index]),
        hole=0.45, textinfo="label+percent",
    ))
    fig.update_layout(
        **CHART_THEME, margin=dict(t=10, b=10), height=340,
        showlegend=True, legend=dict(orientation="h", y=-0.1),
    )
    st.markdown(f"#### {tr_label('Severity distribution', lang)}")
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 3 — Field Scans (sourced from `field_scans`)
# ---------------------------------------------------------------------------
def _render_field_scan_tab(lang: str) -> None:
    try:
        rows = get_field_scans(limit=2000)
    except DatabaseError as e:
        st.error(str(e))
        return
    except Exception:
        logger.exception("Unexpected error loading field scan dashboard data")
        st.error(
            tr_label(
                "Loading field scan dashboard data failed unexpectedly. "
                "Please try again. If the problem continues, contact the app maintainer.",
                lang,
            )
        )
        return

    if not rows:
        callout(
            f"{icon_html('field_scan', size=18)}"
            + tr_label(
                "No field scans saved yet. Run a batch "
                "scan on the <b>Field Scan</b> page and click <b>Save Field Scan</b> "
                "to populate this tab.",
                lang,
            )
        )
        return

    df = _load_field_scan_dataframe(rows)

    total_scans = len(df)
    total_leaves = int(df["num_images"].sum())
    avg_healthy_pct = round(df["healthy_pct"].mean(), 1)
    avg_score = round(df["field_health_score"].mean(), 1)
    high_risk_scans = int(df["status"].isin(HIGH_RISK_STATUSES).sum())

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        _tile(tr_label("Total Scans", lang), str(total_scans), "#2F6D46", "#1C2E20", tr_label("all-time", lang))
    with c2:
        _tile(tr_label("Leaves Scanned", lang), str(total_leaves), "#7FA687", "#1C2E20",
              tr_label("across all scans", lang))
    with c3:
        _tile(tr_label("Avg Healthy %", lang), f"{avg_healthy_pct}%", "#D6A34B", "#1C2E20",
              tr_label("per scan", lang))
    with c4:
        _tile(tr_label("Avg Field Score", lang), str(avg_score), "#D6A34B", "#1C2E20",
              tr_label("out of 100", lang))
    with c5:
        _tile(tr_label("High-Risk Scans", lang), str(high_risk_scans), "#B5564B", "#7C3730",
              tr_label("At Risk + Critical", lang))

    st.markdown("---")

    _fs_chart_score_trend(df, lang)
    _fs_chart_crop_wise(df, lang)
    _fs_chart_dominant_disease(df, lang)
    _fs_chart_severity_breakdown(rows, lang)


def _load_field_scan_dataframe(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df = df.rename(columns={"crop_name": "crop"})
    df["num_images"] = pd.to_numeric(df.get("num_images"), errors="coerce").fillna(0).astype(int)
    df["healthy_pct"] = pd.to_numeric(df.get("healthy_pct"), errors="coerce").fillna(0)
    df["field_health_score"] = pd.to_numeric(df.get("field_health_score"), errors="coerce").fillna(0).astype(int)
    df["status"] = df["field_health_score"].apply(classify_health_status)
    df["dominant_disease"] = (
        df.get("dominant_disease").fillna("None detected") if "dominant_disease" in df
        else "None detected"
    )

    df["_dt"] = pd.to_datetime(df.get("created_at"), errors="coerce", utc=True).dt.tz_localize(None)
    df["_dt"] = df["_dt"].fillna(pd.Timestamp.now())
    df["date"] = df["_dt"].dt.strftime("%Y-%m-%d")
    return df


def _aggregate_severity_counts(rows: list[dict]) -> dict[str, int]:
    """Sum each scan's stored severity_breakdown JSON into one field-wide total."""
    total: Counter = Counter()
    for r in rows:
        raw = r.get("severity_breakdown")
        if not raw:
            continue
        try:
            total.update(json.loads(raw))
        except (TypeError, ValueError):
            continue
    return dict(total)


def _fs_chart_score_trend(df: pd.DataFrame, lang: str) -> None:
    """Line chart of average field health score over time + scan volume bars."""
    trend = (
        df.groupby("date", as_index=False)["field_health_score"]
          .mean().sort_values("date")
          .rename(columns={"field_health_score": "avg_score"})
    )
    counts = df.groupby("date", as_index=False).size().rename(columns={"size": "n"})

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=counts["date"], y=counts["n"], name=tr_label("Scans per day", lang),
        yaxis="y2", marker=dict(color="#D8E2CC"), opacity=0.6,
    ))
    fig.add_trace(go.Scatter(
        x=trend["date"], y=trend["avg_score"], mode="lines+markers",
        name=tr_label("Avg field score", lang), line=dict(color="#2F6D46", width=3), marker=dict(size=8),
    ))
    fig.update_layout(
        **CHART_THEME, margin=dict(t=10, b=10), height=320,
        legend=dict(orientation="h", y=1.12), xaxis_title=tr_label("Date", lang),
        yaxis=dict(title=tr_label("Avg field score", lang), range=[0, 100]),
        yaxis2=dict(title=tr_label("Scans / day", lang), overlaying="y", side="right", showgrid=False),
    )
    st.markdown(f"#### {tr_label('Field health score trend', lang)}")
    st.plotly_chart(fig, use_container_width=True)


def _fs_chart_crop_wise(df: pd.DataFrame, lang: str) -> None:
    """Bar chart: average field health score per crop."""
    tdf = df.copy()
    tdf["crop_ta"] = tdf["crop"].apply(lambda c: tr_crop(c, lang))
    grouped = (
        tdf.groupby("crop_ta", as_index=False)["field_health_score"]
          .mean().rename(columns={"field_health_score": "avg_score"})
    )
    fig = go.Figure(go.Bar(
        x=grouped["crop_ta"], y=grouped["avg_score"],
        marker=dict(color="#7FA687"),
        text=grouped["avg_score"].round(1), textposition="outside",
    ))
    fig.update_layout(
        **CHART_THEME, margin=dict(t=10, b=10), height=320,
        xaxis_title=tr_label("Crop", lang), yaxis=dict(title=tr_label("Avg field health score", lang), range=[0, 100]),
    )
    st.markdown(f"#### {tr_label('Avg field health score by crop', lang)}")
    st.plotly_chart(fig, use_container_width=True)


def _fs_chart_dominant_disease(df: pd.DataFrame, lang: str) -> None:
    """Horizontal bar of how often each disease was the dominant one in a scan."""
    counts = df["dominant_disease"].value_counts()
    colors = []
    palette_i = 0
    for name in counts.index:
        if str(name).strip().lower() in ("healthy", "none detected"):
            colors.append("#2F6D46")
        else:
            colors.append(DISEASE_PALETTE[palette_i % len(DISEASE_PALETTE)])
            palette_i += 1

    def _label(name: str) -> str:
        if str(name).strip().lower() == "none detected":
            return tr_label("None detected", lang)
        return _translated_disease_label(name, lang)

    fig = go.Figure(go.Bar(
        orientation="h",
        x=counts.values,
        y=[_label(n) for n in counts.index],
        text=counts.values, textposition="outside",
        marker=dict(color=colors),
    ))
    fig.update_layout(
        **CHART_THEME, margin=dict(t=10, b=10), height=max(220, 40 * len(counts)),
        xaxis_title=tr_label("Number of scans", lang), yaxis_title="", showlegend=False,
    )
    st.markdown(f"#### {tr_label('Dominant disease across scans', lang)}")
    st.plotly_chart(fig, use_container_width=True)


def _fs_chart_severity_breakdown(rows: list[dict], lang: str) -> None:
    """Pie chart of severity levels aggregated across every leaf, every scan."""
    counts = _aggregate_severity_counts(rows)
    if not counts:
        return
    order = ["None", "Mild", "Moderate", "High"]
    ordered = {k: counts[k] for k in order if k in counts}
    ordered.update({k: v for k, v in counts.items() if k not in order})

    fig = go.Figure(go.Pie(
        labels=[tr_severity(s, lang) for s in ordered.keys()], values=list(ordered.values()),
        marker=dict(colors=[SEVERITY_COLORS.get(s, "#93998A") for s in ordered]),
        hole=0.45, textinfo="label+percent",
    ))
    fig.update_layout(
        **CHART_THEME, margin=dict(t=10, b=10), height=340,
        showlegend=True, legend=dict(orientation="h", y=-0.1),
    )
    st.markdown(f"#### {tr_label('Aggregate severity breakdown (all scanned leaves)', lang)}")
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 4 — Environmental (sourced from `environment_analyses`)
# ---------------------------------------------------------------------------
def _render_env_tab(lang: str) -> None:
    try:
        rows = get_environment_analyses(limit=2000)
    except DatabaseError as e:
        st.error(str(e))
        return
    except Exception:
        logger.exception("Unexpected error loading environmental dashboard data")
        st.error(
            tr_label(
                "Loading environmental dashboard data failed unexpectedly. "
                "Please try again. If the problem continues, contact the app maintainer.",
                lang,
            )
        )
        return

    if not rows:
        callout(
            f"{icon_html('dashboard', size=18)}"
            + tr_label(
                "No environmental analyses saved yet. "
                "Assess a reading on the <b>Environmental Analysis</b> page and click "
                "<b>Save Analysis</b> to populate this tab.",
                lang,
            )
        )
        return

    df = _load_env_dataframe(rows)

    total = len(df)
    avg_score = round(df["health_score"].mean(), 1)
    optimal = int((df["risk_level"] == "Optimal").sum())
    high_risk = int(df["risk_level"].isin(["High", "Critical"]).sum())
    avg_confidence = round(df["probability"].mean(), 1)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        _tile(tr_label("Total Analyses", lang), str(total), "#2F6D46", "#1C2E20", tr_label("all-time", lang))
    with c2:
        _tile(tr_label("Optimal Readings", lang), str(optimal), "#7FA687", "#1C2E20",
              f"{round(optimal/total*100)}% {tr_label('of total', lang)}")
    with c3:
        _tile(tr_label("High-Risk Readings", lang), str(high_risk), "#CE8C82", "#7C3730",
              tr_label("High + Critical", lang))
    with c4:
        _tile(tr_label("Avg Health Score", lang), str(avg_score), "#D6A34B", "#1C2E20", tr_label("out of 100", lang))
    with c5:
        _tile(tr_label("Avg Model Confidence", lang), f"{avg_confidence}%", "#7FA687", "#1C2E20",
              tr_label("trained risk model", lang))

    st.markdown("---")

    _env_chart_risk_distribution(df, lang)
    _env_chart_health_score_trend(df, lang)
    _env_chart_crop_wise(df, lang)
    _env_chart_factor_ranges(df, lang)


def _load_env_dataframe(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df = df.rename(columns={"crop_name": "crop"})
    df["risk_level"] = df.get("risk_level").fillna("Unknown") if "risk_level" in df else "Unknown"
    df["health_score"] = pd.to_numeric(df.get("health_score"), errors="coerce").fillna(0).astype(int)
    df["probability"] = pd.to_numeric(df.get("probability"), errors="coerce").fillna(0) * 100
    for col in ("temperature", "humidity", "soil_moisture", "rainfall"):
        df[col] = pd.to_numeric(df.get(col), errors="coerce")

    df["_dt"] = pd.to_datetime(df.get("created_at"), errors="coerce", utc=True).dt.tz_localize(None)
    df["_dt"] = df["_dt"].fillna(pd.Timestamp.now())
    df["date"] = df["_dt"].dt.strftime("%Y-%m-%d")
    return df


def _env_chart_risk_distribution(df: pd.DataFrame, lang: str) -> None:
    """Pie chart of risk levels across all saved environmental analyses."""
    order = ["Optimal", "Low", "Moderate", "High", "Critical", "Unknown"]
    counts = df["risk_level"].value_counts().reindex(order).dropna()
    colors = [RISK_LEVELS.get(level, ("Unknown", "#93998A", 0.5))[1] for level in counts.index]
    fig = go.Figure(go.Pie(
        labels=[tr_label(level, lang) for level in counts.index], values=counts.values,
        marker=dict(colors=colors), hole=0.45, textinfo="label+percent",
    ))
    fig.update_layout(
        **CHART_THEME, margin=dict(t=10, b=10), height=340,
        showlegend=True, legend=dict(orientation="h", y=-0.1),
    )
    st.markdown(f"#### {tr_label('Risk level distribution', lang)}")
    st.plotly_chart(fig, use_container_width=True)


def _env_chart_health_score_trend(df: pd.DataFrame, lang: str) -> None:
    """Line chart of environmental health score over time + volume bars."""
    trend = (
        df.groupby("date", as_index=False)["health_score"]
          .mean().sort_values("date")
          .rename(columns={"health_score": "avg_score"})
    )
    counts = df.groupby("date", as_index=False).size().rename(columns={"size": "n"})

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=counts["date"], y=counts["n"], name=tr_label("Analyses per day", lang),
        yaxis="y2", marker=dict(color="#D8E2CC"), opacity=0.6,
    ))
    fig.add_trace(go.Scatter(
        x=trend["date"], y=trend["avg_score"], mode="lines+markers",
        name=tr_label("Avg health score", lang), line=dict(color="#2F6D46", width=3), marker=dict(size=8),
    ))
    fig.update_layout(
        **CHART_THEME, margin=dict(t=10, b=10), height=320,
        legend=dict(orientation="h", y=1.12), xaxis_title=tr_label("Date", lang),
        yaxis=dict(title=tr_label("Avg health score", lang), range=[0, 100]),
        yaxis2=dict(title=tr_label("Analyses / day", lang), overlaying="y", side="right", showgrid=False),
    )
    st.markdown(f"#### {tr_label('Health score trend', lang)}")
    st.plotly_chart(fig, use_container_width=True)


def _env_chart_crop_wise(df: pd.DataFrame, lang: str) -> None:
    """Stacked bar: count of environmental analyses per crop, by risk level."""
    order = ["Optimal", "Low", "Moderate", "High", "Critical", "Unknown"]
    tdf = df.copy()
    tdf["crop_ta"] = tdf["crop"].apply(lambda c: tr_crop(c, lang))
    tdf["risk_ta"] = tdf["risk_level"].apply(lambda r: tr_label(r, lang))
    order_ta = [tr_label(level, lang) for level in order]
    colors_ta = {tr_label(level, lang): RISK_LEVELS.get(level, ("Unknown", "#93998A", 0.5))[1] for level in order}
    grouped = tdf.groupby(["crop_ta", "risk_ta"]).size().reset_index(name="count")
    fig = px.bar(
        grouped, x="crop_ta", y="count", color="risk_ta",
        color_discrete_map=colors_ta, category_orders={"risk_ta": order_ta}, text="count",
    )
    fig.update_traces(textposition="inside")
    fig.update_layout(
        **CHART_THEME, margin=dict(t=10, b=10), height=340,
        xaxis_title=tr_label("Crop", lang), yaxis_title=tr_label("Number of analyses", lang),
        legend_title_text=tr_label("Risk level", lang), barmode="stack",
    )
    st.markdown(f"#### {tr_label('Crop-wise analysis', lang)}")
    st.plotly_chart(fig, use_container_width=True)


def _env_chart_factor_ranges(df: pd.DataFrame, lang: str) -> None:
    """Box plot showing the spread of each logged environmental factor."""
    factors = ["temperature", "humidity", "soil_moisture", "rainfall"]
    labels = {"temperature": "Temp (°C)", "humidity": "Humidity (%)",
              "soil_moisture": "Soil moisture (%)", "rainfall": "Rainfall (mm)"}
    fig = go.Figure()
    for i, factor in enumerate(factors):
        fig.add_trace(go.Box(
            y=df[factor], name=tr_label(labels[factor], lang),
            marker=dict(color=DISEASE_PALETTE[i % len(DISEASE_PALETTE)]),
            boxmean=True,
        ))
    fig.update_layout(
        **CHART_THEME, margin=dict(t=10, b=10), height=340,
        yaxis_title=tr_label("Value", lang), showlegend=False,
    )
    st.markdown(f"#### {tr_label('Logged factor ranges', lang)}")
    st.plotly_chart(fig, use_container_width=True)