"""Dashboard page — statistics and charts over real saved analyses.

Reads from SQLite via src.db.get_analyses(). Shows a friendly empty state
instead of any charts/metrics when nothing has been saved yet.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.db import get_analyses
from src.health_engine import classify_health_status, HEALTH_STATUS_BANDS
from utils.ui import (
    page_header,
    callout,
    footer,
    CHART_THEME,
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
        "Statistics and trends across all crop analyses.",
    )

    rows = get_analyses(limit=2000)

    if not rows:
        callout(
            f"{icon_html('dashboard', size=18)}No analyses saved yet. Run a "
            "calculation on the <b>Crop Health Analysis</b> page and click "
            "<b>Save Analysis</b> to populate the dashboard."
        )
        footer()
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

    footer()


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