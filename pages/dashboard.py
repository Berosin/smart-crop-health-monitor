"""Dashboard page — statistics and charts over real saved analyses.

Reads from SQLite via src.db.get_analyses(). Shows a friendly empty state
instead of any charts/metrics when nothing has been saved yet.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.db import get_analyses, get_disease_analyses, get_environment_analyses
from src.errors import DatabaseError, logger
from src.health_engine import classify_health_status, HEALTH_STATUS_BANDS
from utils.ui import (
    page_header,
    callout,
    footer,
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
    page_header(
        "dashboard",
        "Dashboard",
        "Statistics and trends across all crop, disease, and environmental analyses.",
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
# Tab 1 — Crop Health (existing charts, unchanged, sourced from `analyses`)
# ---------------------------------------------------------------------------
def _render_health_tab() -> None:
    try:
        rows = get_analyses(limit=2000)
    except DatabaseError as e:
        st.error(str(e))
        return
    except Exception:
        logger.exception("Unexpected error loading dashboard data")
        st.error(
            "Loading dashboard data failed unexpectedly. Please try again. "
            "If the problem continues, contact the app maintainer."
        )
        return

    if not rows:
        callout(
            f"{icon_html('dashboard', size=18)}No crop health analyses saved yet. Run a "
            "calculation on the <b>Crop Health Analysis</b> page and click "
            "<b>Save Analysis</b> to populate this tab."
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
    _kpi_row(total, healthy, diseased, avg_score, high_risk)

    st.markdown("---")

    # ---- Charts ---------------------------------------------------------
    _chart_disease_distribution(df)
    _chart_health_score_trend(df)
    _chart_crop_wise(df)
    _chart_risk_distribution(df)


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


def _kpi_row(total, healthy, diseased, avg_score, high_risk) -> None:
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        _tile("Total Analyses", str(total), "#2F6D46", "#1C2E20", "all-time")
    with c2:
        _tile("Healthy Plants", str(healthy), "#7FA687", "#1C2E20",
              f"{round(healthy/total*100)}% of total")
    with c3:
        _tile("Diseased Plants", str(diseased), "#CE8C82", "#7C3730",
              f"{round(diseased/total*100)}% of total")
    with c4:
        _tile("Avg Health Score", str(avg_score), "#D6A34B", "#1C2E20",
              "out of 100")
    with c5:
        _tile("High-Risk Cases", str(high_risk), "#B5564B", "#7C3730",
              "At Risk + Critical")


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def _chart_disease_distribution(df: pd.DataFrame) -> None:
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
        y=counts.index,
        text=counts.values,
        textposition="outside",
        marker=dict(color=colors),
    ))
    fig.update_layout(
        **CHART_THEME,
        margin=dict(t=10, b=10),
        height=max(260, 40 * len(counts)),
        xaxis_title="Number of analyses",
        yaxis_title="",
        showlegend=False,
    )
    st.markdown("#### Disease distribution")
    st.plotly_chart(fig, use_container_width=True)


def _chart_health_score_trend(df: pd.DataFrame) -> None:
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
        name="Analyses per day",
        yaxis="y2",
        marker=dict(color="#D8E2CC"),
        opacity=0.6,
    ))
    fig.add_trace(go.Scatter(
        x=trend["date"], y=trend["avg_score"],
        mode="lines+markers",
        name="Avg health score",
        line=dict(color="#2F6D46", width=3),
        marker=dict(size=8),
    ))
    fig.update_layout(
        **CHART_THEME,
        margin=dict(t=10, b=10),
        height=320,
        legend=dict(orientation="h", y=1.12),
        xaxis_title="Date",
        yaxis=dict(title="Avg health score", range=[0, 100]),
        yaxis2=dict(title="Analyses / day", overlaying="y", side="right",
                    showgrid=False),
    )
    st.markdown("#### Health score trend")
    st.plotly_chart(fig, use_container_width=True)


def _chart_crop_wise(df: pd.DataFrame) -> None:
    """Stacked bar: count of analyses per crop, broken down by health status."""
    grouped = (
        df.groupby(["crop", "status"]).size()
          .reset_index(name="count")
    )
    fig = px.bar(
        grouped, x="crop", y="count", color="status",
        color_discrete_map=STATUS_COLORS,
        category_orders={"status": STATUS_ORDER},
        text="count",
    )
    fig.update_traces(textposition="inside")
    fig.update_layout(
        **CHART_THEME,
        margin=dict(t=10, b=10),
        height=340,
        xaxis_title="Crop",
        yaxis_title="Number of analyses",
        legend_title_text="Status",
        barmode="stack",
    )
    st.markdown("#### Crop-wise analysis")
    st.plotly_chart(fig, use_container_width=True)


def _chart_risk_distribution(df: pd.DataFrame) -> None:
    """Pie chart of health-status bands across all saved analyses."""
    counts = df["status"].value_counts().reindex(STATUS_ORDER).dropna()
    fig = go.Figure(go.Pie(
        labels=counts.index,
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
    st.markdown("#### Risk distribution")
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 2 — Disease Detection (sourced from `disease_analyses`)
# ---------------------------------------------------------------------------
SEVERITY_COLORS = {
    "None": "#2F6D46", "Mild": "#7FA687", "Moderate": "#C97A3B",
    "High": "#B5564B", "Unknown": "#93998A",
}
SEVERITY_ORDER = ["None", "Mild", "Moderate", "High", "Unknown"]


def _render_disease_tab() -> None:
    try:
        rows = get_disease_analyses(limit=2000)
    except DatabaseError as e:
        st.error(str(e))
        return
    except Exception:
        logger.exception("Unexpected error loading disease detection dashboard data")
        st.error(
            "Loading disease detection dashboard data failed unexpectedly. "
            "Please try again. If the problem continues, contact the app maintainer."
        )
        return

    if not rows:
        callout(
            f"{icon_html('dashboard', size=18)}No disease detection analyses saved yet. "
            "Analyze a leaf image on the <b>Disease Detection</b> page and click "
            "<b>Save Analysis</b> to populate this tab."
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
        _tile("Total Analyses", str(total), "#2F6D46", "#1C2E20", "all-time")
    with c2:
        _tile("Healthy Leaves", str(healthy), "#7FA687", "#1C2E20",
              f"{round(healthy/total*100)}% of total")
    with c3:
        _tile("Diseased Leaves", str(diseased), "#CE8C82", "#7C3730",
              f"{round(diseased/total*100)}% of total")
    with c4:
        _tile("Avg Confidence", f"{avg_confidence}%", "#D6A34B", "#1C2E20",
              "model output")
    with c5:
        _tile("High Severity", str(high_severity), "#B5564B", "#7C3730",
              "cases flagged High")

    st.markdown("---")

    _dd_chart_disease_distribution(df)
    _dd_chart_confidence_trend(df)
    _dd_chart_crop_wise(df)
    _dd_chart_severity_distribution(df)


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


def _dd_chart_disease_distribution(df: pd.DataFrame) -> None:
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
        orientation="h", x=counts.values, y=counts.index,
        text=counts.values, textposition="outside",
        marker=dict(color=colors),
    ))
    fig.update_layout(
        **CHART_THEME, margin=dict(t=10, b=10), height=max(260, 40 * len(counts)),
        xaxis_title="Number of analyses", yaxis_title="", showlegend=False,
    )
    st.markdown("#### Disease distribution")
    st.plotly_chart(fig, use_container_width=True)


def _dd_chart_confidence_trend(df: pd.DataFrame) -> None:
    """Line chart of average model confidence over time + volume bars."""
    trend = (
        df.groupby("date", as_index=False)["confidence"]
          .mean().sort_values("date")
          .rename(columns={"confidence": "avg_confidence"})
    )
    counts = df.groupby("date", as_index=False).size().rename(columns={"size": "n"})

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=counts["date"], y=counts["n"], name="Analyses per day",
        yaxis="y2", marker=dict(color="#D8E2CC"), opacity=0.6,
    ))
    fig.add_trace(go.Scatter(
        x=trend["date"], y=trend["avg_confidence"], mode="lines+markers",
        name="Avg confidence", line=dict(color="#2F6D46", width=3), marker=dict(size=8),
    ))
    fig.update_layout(
        **CHART_THEME, margin=dict(t=10, b=10), height=320,
        legend=dict(orientation="h", y=1.12), xaxis_title="Date",
        yaxis=dict(title="Avg confidence (%)", range=[0, 100]),
        yaxis2=dict(title="Analyses / day", overlaying="y", side="right", showgrid=False),
    )
    st.markdown("#### Confidence trend")
    st.plotly_chart(fig, use_container_width=True)


def _dd_chart_crop_wise(df: pd.DataFrame) -> None:
    """Stacked bar: count of disease analyses per crop, broken down by disease."""
    grouped = df.groupby(["crop", "disease"]).size().reset_index(name="count")
    fig = px.bar(
        grouped, x="crop", y="count", color="disease",
        text="count",
    )
    fig.update_traces(textposition="inside")
    fig.update_layout(
        **CHART_THEME, margin=dict(t=10, b=10), height=340,
        xaxis_title="Crop", yaxis_title="Number of analyses",
        legend_title_text="Disease", barmode="stack",
    )
    st.markdown("#### Crop-wise analysis")
    st.plotly_chart(fig, use_container_width=True)


def _dd_chart_severity_distribution(df: pd.DataFrame) -> None:
    """Pie chart of severity levels across all saved disease analyses."""
    counts = df["severity"].value_counts().reindex(SEVERITY_ORDER).dropna()
    fig = go.Figure(go.Pie(
        labels=counts.index, values=counts.values,
        marker=dict(colors=[SEVERITY_COLORS.get(s, "#93998A") for s in counts.index]),
        hole=0.45, textinfo="label+percent",
    ))
    fig.update_layout(
        **CHART_THEME, margin=dict(t=10, b=10), height=340,
        showlegend=True, legend=dict(orientation="h", y=-0.1),
    )
    st.markdown("#### Severity distribution")
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 3 — Environmental (sourced from `environment_analyses`)
# ---------------------------------------------------------------------------
def _render_env_tab() -> None:
    try:
        rows = get_environment_analyses(limit=2000)
    except DatabaseError as e:
        st.error(str(e))
        return
    except Exception:
        logger.exception("Unexpected error loading environmental dashboard data")
        st.error(
            "Loading environmental dashboard data failed unexpectedly. "
            "Please try again. If the problem continues, contact the app maintainer."
        )
        return

    if not rows:
        callout(
            f"{icon_html('dashboard', size=18)}No environmental analyses saved yet. "
            "Assess a reading on the <b>Environmental Analysis</b> page and click "
            "<b>Save Analysis</b> to populate this tab."
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
        _tile("Total Analyses", str(total), "#2F6D46", "#1C2E20", "all-time")
    with c2:
        _tile("Optimal Readings", str(optimal), "#7FA687", "#1C2E20",
              f"{round(optimal/total*100)}% of total")
    with c3:
        _tile("High-Risk Readings", str(high_risk), "#CE8C82", "#7C3730",
              "High + Critical")
    with c4:
        _tile("Avg Health Score", str(avg_score), "#D6A34B", "#1C2E20", "out of 100")
    with c5:
        _tile("Avg Model Confidence", f"{avg_confidence}%", "#7FA687", "#1C2E20",
              "trained risk model")

    st.markdown("---")

    _env_chart_risk_distribution(df)
    _env_chart_health_score_trend(df)
    _env_chart_crop_wise(df)
    _env_chart_factor_ranges(df)


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


def _env_chart_risk_distribution(df: pd.DataFrame) -> None:
    """Pie chart of risk levels across all saved environmental analyses."""
    order = ["Optimal", "Low", "Moderate", "High", "Critical", "Unknown"]
    counts = df["risk_level"].value_counts().reindex(order).dropna()
    colors = [RISK_LEVELS.get(level, ("Unknown", "#93998A", 0.5))[1] for level in counts.index]
    fig = go.Figure(go.Pie(
        labels=counts.index, values=counts.values,
        marker=dict(colors=colors), hole=0.45, textinfo="label+percent",
    ))
    fig.update_layout(
        **CHART_THEME, margin=dict(t=10, b=10), height=340,
        showlegend=True, legend=dict(orientation="h", y=-0.1),
    )
    st.markdown("#### Risk level distribution")
    st.plotly_chart(fig, use_container_width=True)


def _env_chart_health_score_trend(df: pd.DataFrame) -> None:
    """Line chart of environmental health score over time + volume bars."""
    trend = (
        df.groupby("date", as_index=False)["health_score"]
          .mean().sort_values("date")
          .rename(columns={"health_score": "avg_score"})
    )
    counts = df.groupby("date", as_index=False).size().rename(columns={"size": "n"})

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=counts["date"], y=counts["n"], name="Analyses per day",
        yaxis="y2", marker=dict(color="#D8E2CC"), opacity=0.6,
    ))
    fig.add_trace(go.Scatter(
        x=trend["date"], y=trend["avg_score"], mode="lines+markers",
        name="Avg health score", line=dict(color="#2F6D46", width=3), marker=dict(size=8),
    ))
    fig.update_layout(
        **CHART_THEME, margin=dict(t=10, b=10), height=320,
        legend=dict(orientation="h", y=1.12), xaxis_title="Date",
        yaxis=dict(title="Avg health score", range=[0, 100]),
        yaxis2=dict(title="Analyses / day", overlaying="y", side="right", showgrid=False),
    )
    st.markdown("#### Health score trend")
    st.plotly_chart(fig, use_container_width=True)


def _env_chart_crop_wise(df: pd.DataFrame) -> None:
    """Stacked bar: count of environmental analyses per crop, by risk level."""
    order = ["Optimal", "Low", "Moderate", "High", "Critical", "Unknown"]
    colors = {level: RISK_LEVELS.get(level, ("Unknown", "#93998A", 0.5))[1] for level in order}
    grouped = df.groupby(["crop", "risk_level"]).size().reset_index(name="count")
    fig = px.bar(
        grouped, x="crop", y="count", color="risk_level",
        color_discrete_map=colors, category_orders={"risk_level": order}, text="count",
    )
    fig.update_traces(textposition="inside")
    fig.update_layout(
        **CHART_THEME, margin=dict(t=10, b=10), height=340,
        xaxis_title="Crop", yaxis_title="Number of analyses",
        legend_title_text="Risk level", barmode="stack",
    )
    st.markdown("#### Crop-wise analysis")
    st.plotly_chart(fig, use_container_width=True)


def _env_chart_factor_ranges(df: pd.DataFrame) -> None:
    """Box plot showing the spread of each logged environmental factor."""
    factors = ["temperature", "humidity", "soil_moisture", "rainfall"]
    labels = {"temperature": "Temp (°C)", "humidity": "Humidity (%)",
              "soil_moisture": "Soil moisture (%)", "rainfall": "Rainfall (mm)"}
    fig = go.Figure()
    for i, factor in enumerate(factors):
        fig.add_trace(go.Box(
            y=df[factor], name=labels[factor],
            marker=dict(color=DISEASE_PALETTE[i % len(DISEASE_PALETTE)]),
            boxmean=True,
        ))
    fig.update_layout(
        **CHART_THEME, margin=dict(t=10, b=10), height=340,
        yaxis_title="Value", showlegend=False,
    )
    st.markdown("#### Logged factor ranges")
    st.plotly_chart(fig, use_container_width=True)