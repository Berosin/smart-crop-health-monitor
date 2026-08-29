"""Outbreak Alerts — trend detection over your saved analysis history.

Analysis History already lists every past analysis. This page asks a
different question of the same data: **is a crop getting worse lately?**
For every crop with saved Disease Detection or Field Scan history, it
compares a recent rolling window of saved analyses against the window
immediately before it, and flags a rising diseased share and/or rising
high-severity share as an outbreak risk signal — Low / Watch / Elevated /
High — with a plain-language reason for the number.

All the actual math lives in src/outbreak_detection.py (pure, DB-free,
independently unit-tested there via its __main__ self-test); this page is
purely presentation over that module's output.
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from src.errors import logger
from src.outbreak_detection import load_outbreak_signals, get_active_alerts
from utils.ui import page_header, callout, card, footer, pretty_name, CHART_THEME
from utils.icons import icon_html

RISK_COLORS = {
    "High": "#B5564B", "Elevated": "#C97A3B", "Watch": "#D6A34B",
    "Low": "#7FA687", "Insufficient data": "#93998A",
}
RISK_ICON = {
    "High": "diseased", "Elevated": "warning", "Watch": "warning",
    "Low": "healthy", "Insufficient data": "info",
}


def render() -> None:
    page_header(
        "alerts",
        "Outbreak Alerts",
        "Rolling-window trend detection over your saved Disease Detection and Field Scan history.",
    )

    window = st.radio(
        "Rolling window (saved analyses)", [7, 30],
        index=0, horizontal=True,
        help="Each crop's most recent N saved analyses are compared against the N before them.",
    )

    try:
        with st.spinner("Analyzing saved history…"):
            signals = load_outbreak_signals(window=window)
    except Exception:
        logger.exception("Unexpected error computing outbreak signals")
        st.error(
            "Couldn't analyze saved history right now. Please try again. "
            "If the problem continues, contact the app maintainer."
        )
        footer()
        return

    if not signals:
        card(
            "No history yet",
            "Save a few Disease Detection or Field Scan analyses first — "
            "Outbreak Alerts needs some saved history per crop before it "
            "can compare a recent window against a prior one.",
        )
        footer()
        return

    _render_active_banner(signals)
    st.markdown("#### Risk by crop")
    for signal in signals:
        _render_crop_card(signal)

    footer()


# ---------------------------------------------------------------------------
# Active-alert banner
# ---------------------------------------------------------------------------
def _render_active_banner(signals: list[dict]) -> None:
    active = get_active_alerts(signals)
    if not active:
        callout(
            f"{icon_html('healthy', size=18)}No crops are currently trending "
            "worse — everything with enough history is Watch level or better."
        )
        return

    lines = "<br/>".join(
        f"<b>{s['crop']}</b> — {s['risk_level']}: {s['risk_reason']}" for s in active
    )
    st.markdown(
        f"""
        <div class="callout" style="border-left-color:#B5564B;background:#FBEFED">
          {icon_html('diseased', size=18)}<b>{len(active)} crop(s) trending worse:</b><br/>
          {lines}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Per-crop detail card
# ---------------------------------------------------------------------------
def _render_crop_card(signal: dict) -> None:
    crop = signal["crop"]
    level = signal["risk_level"]
    color = RISK_COLORS.get(level, "#93998A")
    recent = signal["recent"]
    prior = signal["prior"]

    with st.container():
        st.markdown(
            f"""
            <div class="card" style="border-left:5px solid {color}">
              <div style="display:flex;justify-content:space-between;align-items:center">
                <h4 style="margin:0;color:var(--ink)">{icon_html(RISK_ICON.get(level, 'info'), size=18, margin_right='.4em')}{crop}</h4>
                <span style="background:{color};color:#fff;padding:.15rem .7rem;border-radius:999px;
                             font-size:.78rem;font-weight:600">{level}</span>
              </div>
              <p style="margin:.5rem 0 0;color:#4E5646;font-size:.9rem">{signal['risk_reason']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if recent["n_records"] > 0:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.caption(f"Recent window ({recent['n_records']} saved · {recent['n_leaves']} leaves)")
                st.markdown(f"**{recent['diseased_pct']:.0f}%** diseased · **{recent['high_pct']:.0f}%** high severity")
                if recent["dominant_disease"]:
                    st.caption(f"Dominant: {pretty_name(recent['dominant_disease'])}")
            with c2:
                if signal["has_prior_window"]:
                    st.caption(f"Prior window ({prior['n_records']} saved · {prior['n_leaves']} leaves)")
                    st.markdown(f"**{prior['diseased_pct']:.0f}%** diseased · **{prior['high_pct']:.0f}%** high severity")
                    if prior["dominant_disease"]:
                        st.caption(f"Dominant: {pretty_name(prior['dominant_disease'])}")
                else:
                    st.caption("Prior window")
                    st.markdown("*Not enough history yet*")
            with c3:
                if signal["diseased_pct_delta"] is not None:
                    st.caption("Change vs. prior window")
                    st.markdown(
                        f"Diseased: **{signal['diseased_pct_delta']:+.0f} pts**  \n"
                        f"High severity: **{signal['high_pct_delta']:+.0f} pts**"
                    )
                else:
                    st.caption("Change vs. prior window")
                    st.markdown("*N/A*")

            if recent["disease_counts"]:
                fig = go.Figure(go.Bar(
                    orientation="h",
                    x=list(recent["disease_counts"].values()),
                    y=[pretty_name(n) for n in recent["disease_counts"].keys()],
                    marker=dict(color=color),
                ))
                fig.update_layout(
                    **CHART_THEME,
                    margin=dict(t=10, b=10, l=10),
                    height=max(120, len(recent["disease_counts"]) * 36),
                    xaxis_title="Saved analyses (recent window)",
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True, key=f"_outbreak_chart_{crop}")

        st.write("")