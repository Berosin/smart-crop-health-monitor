"""Dashboard page — statistics and charts over past analyses.

Layout-only at this stage: all data comes from the dummy history generator.
No SQLite connection yet.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.ui import (
    page_header,
    metric_tile,
    callout,
    footer,
    get_dummy_history,
)


# Threshold below which a plant is considered high risk.
HIGH_RISK_THRESHOLD = 50

# Consistent palette across charts.
PALETTE = {
    "Healthy":      "#2e7d32",
    "Early Blight": "#e57373",
    "Leaf Mold":    "#ba68c8",
    "Rust":         "#ff8a65",
    "Late Blight":  "#f06292",
}
SCORE_BANDS = ["Critical (0-40)", "At risk (40-70)", "Healthy (70-100)"]
SCORE_COLORS = ["#e57373", "#fdd835", "#66bb6a"]


def render() -> None:
    page_header(
        "📊",
        "Dashboard",
        "Statistics and trends across all crop analyses.",
    )

    history = get_dummy_history()
    df = pd.DataFrame(history)

    # ---- Derived KPIs ---------------------------------------------------
    total = len(df)
    healthy = int((df["status"] == "Healthy").sum())
    diseased = total - healthy
    avg_score = round(df["health_score"].mean(), 1)
    high_risk = int((df["health_score"] < HIGH_RISK_THRESHOLD).sum())

    # ---- KPI cards ------------------------------------------------------
    _kpi_row(total, healthy, diseased, avg_score, high_risk)

    st.markdown("---")

    # ---- Charts ---------------------------------------------------------
    _chart_disease_distribution(df)
    _chart_health_score_distribution(df)
    _chart_history_trend(df)

    callout(
        "📊 Showing dummy data. Real statistics will appear once the "
        "database layer is connected."
    )
    footer()


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
        _tile("Total Analyses", str(total), "#2e7d32", "#1b5e20", "all-time")
    with c2:
        _tile("Healthy Plants", str(healthy), "#66bb6a", "#1b5e20",
              f"{round(healthy/total*100)}% of total")
    with c3:
        _tile("Diseased Plants", str(diseased), "#e57373", "#c62828",
              f"{round(diseased/total*100)}% of total")
    with c4:
        _tile("Avg Health Score", str(avg_score), "#fdd835", "#1b5e20",
              "out of 100")
    with c5:
        _tile("High-Risk Plants", str(high_risk), "#ef5350", "#c62828",
              f"score < {HIGH_RISK_THRESHOLD}")


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def _chart_disease_distribution(df: pd.DataFrame) -> None:
    """Horizontal bar of how many analyses fell into each disease class."""
    counts = df["status"].value_counts().reindex(PALETTE.keys()).dropna()
    fig = go.Figure(go.Bar(
        orientation="h",
        x=counts.values,
        y=counts.index,
        text=counts.values,
        textposition="outside",
        marker=dict(color=[PALETTE[s] for s in counts.index]),
    ))
    fig.update_layout(
        template="plotly_white",
        margin=dict(t=10, b=10),
        height=300,
        xaxis_title="Number of analyses",
        yaxis_title="",
        showlegend=False,
    )
    st.markdown("#### Disease distribution")
    st.plotly_chart(fig, use_container_width=True)


def _chart_health_score_distribution(df: pd.DataFrame) -> None:
    """Histogram of health scores banded into Critical / At risk / Healthy."""
    bands = pd.cut(
        df["health_score"],
        bins=[-1, 40, 70, 100],
        labels=SCORE_BANDS,
    )
    band_counts = bands.value_counts().reindex(SCORE_BANDS).fillna(0).astype(int)

    fig = go.Figure(go.Bar(
        x=band_counts.index,
        y=band_counts.values,
        text=band_counts.values,
        textposition="outside",
        marker=dict(color=SCORE_COLORS),
    ))
    fig.update_layout(
        template="plotly_white",
        margin=dict(t=10, b=10),
        height=300,
        xaxis_title="Health score band",
        yaxis_title="Number of analyses",
        showlegend=False,
    )
    st.markdown("#### Health score distribution")
    st.plotly_chart(fig, use_container_width=True)


def _chart_history_trend(df: pd.DataFrame) -> None:
    """Line chart of average health score per day across analyses."""
    trend = (
        df.groupby("date", as_index=False)["health_score"]
          .mean()
          .sort_values("date")
          .rename(columns={"health_score": "avg_score"})
    )
    # Count of analyses per day for a secondary signal.
    counts = df.groupby("date", as_index=False).size().rename(columns={"size": "n"})

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trend["date"], y=trend["avg_score"],
        mode="lines+markers",
        name="Avg health score",
        line=dict(color="#2e7d32", width=3),
        marker=dict(size=8),
    ))
    fig.add_trace(go.Bar(
        x=counts["date"], y=counts["n"],
        name="Analyses per day",
        yaxis="y2",
        marker=dict(color="#c8e6c9"),
        opacity=0.6,
    ))
    fig.update_layout(
        template="plotly_white",
        margin=dict(t=10, b=10),
        height=320,
        legend=dict(orientation="h", y=1.12),
        xaxis_title="Date",
        yaxis=dict(title="Avg health score", range=[0, 100]),
        yaxis2=dict(title="Analyses / day", overlaying="y", side="right",
                    showgrid=False),
    )
    st.markdown("#### Analysis history trend")
    st.plotly_chart(fig, use_container_width=True)
