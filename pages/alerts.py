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
from src.i18n import get_language, tr_label, tr_crop, tr_disease
from src.outbreak_detection import load_outbreak_signals, get_active_alerts
from utils.ui import page_header, callout, card, footer, CHART_THEME
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
    lang = get_language()
    page_header(
        "alerts",
        tr_label("Outbreak Alerts", lang),
        tr_label("Rolling-window trend detection over your saved Disease Detection and Field Scan history.", lang),
    )

    window = st.radio(
        tr_label("Rolling window (saved analyses)", lang), [7, 30],
        index=0, horizontal=True,
        help=tr_label("Each crop's most recent N saved analyses are compared against the N before them.", lang),
    )

    try:
        with st.spinner(tr_label("Analyzing saved history…", lang)):
            signals = load_outbreak_signals(window=window, lang=lang)
    except Exception:
        logger.exception("Unexpected error computing outbreak signals")
        st.error(
            tr_label(
                "Couldn't analyze saved history right now. Please try again. "
                "If the problem continues, contact the app maintainer.",
                lang,
            )
        )
        footer()
        return

    if not signals:
        card(
            tr_label("No history yet", lang),
            tr_label(
                "Save a few Disease Detection or Field Scan analyses first — "
                "Outbreak Alerts needs some saved history per crop before it "
                "can compare a recent window against a prior one.",
                lang,
            ),
        )
        footer()
        return

    _render_active_banner(signals, lang)
    st.markdown(f"#### {tr_label('Risk by crop', lang)}")
    for signal in signals:
        _render_crop_card(signal, lang)

    footer()


# ---------------------------------------------------------------------------
# Active-alert banner
# ---------------------------------------------------------------------------
def _render_active_banner(signals: list[dict], lang: str) -> None:
    active = get_active_alerts(signals)
    if not active:
        callout(
            f"{icon_html('healthy', size=18)}"
            + tr_label(
                "No crops are currently trending worse — everything with enough "
                "history is Watch level or better.",
                lang,
            )
        )
        return

    lines = "<br/>".join(
        f"<b>{tr_crop(s['crop'], lang)}</b> — {tr_label(s['risk_level'], lang)}: {s['risk_reason']}"
        for s in active
    )
    st.markdown(
        f"""
        <div class="callout" style="border-left-color:#B5564B;background:#FBEFED">
          {icon_html('diseased', size=18)}<b>{len(active)} {tr_label('crop(s) trending worse:', lang)}</b><br/>
          {lines}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Per-crop detail card
# ---------------------------------------------------------------------------
def _render_crop_card(signal: dict, lang: str) -> None:
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
                <h4 style="margin:0;color:var(--ink)">{icon_html(RISK_ICON.get(level, 'info'), size=18, margin_right='.4em')}{tr_crop(crop, lang)}</h4>
                <span style="background:{color};color:#fff;padding:.15rem .7rem;border-radius:999px;
                             font-size:.78rem;font-weight:600">{tr_label(level, lang)}</span>
              </div>
              <p style="margin:.5rem 0 0;color:#4E5646;font-size:.9rem">{signal['risk_reason']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if recent["n_records"] > 0:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.caption(
                    f"{tr_label('Recent window', lang)} ({recent['n_records']} {tr_label('saved', lang)} · "
                    f"{recent['n_leaves']} {tr_label('leaves', lang)})"
                )
                st.markdown(
                    f"**{recent['diseased_pct']:.0f}%** {tr_label('diseased', lang)} · "
                    f"**{recent['high_pct']:.0f}%** {tr_label('high severity', lang)}"
                )
                if recent["dominant_disease"]:
                    st.caption(f"{tr_label('Dominant:', lang)} {tr_disease(recent['dominant_disease'], lang)}")
            with c2:
                if signal["has_prior_window"]:
                    st.caption(
                        f"{tr_label('Prior window', lang)} ({prior['n_records']} {tr_label('saved', lang)} · "
                        f"{prior['n_leaves']} {tr_label('leaves', lang)})"
                    )
                    st.markdown(
                        f"**{prior['diseased_pct']:.0f}%** {tr_label('diseased', lang)} · "
                        f"**{prior['high_pct']:.0f}%** {tr_label('high severity', lang)}"
                    )
                    if prior["dominant_disease"]:
                        st.caption(f"{tr_label('Dominant:', lang)} {tr_disease(prior['dominant_disease'], lang)}")
                else:
                    st.caption(tr_label("Prior window", lang))
                    st.markdown(f"*{tr_label('Not enough history yet', lang)}*")
            with c3:
                if signal["diseased_pct_delta"] is not None:
                    st.caption(tr_label("Change vs. prior window", lang))
                    st.markdown(
                        f"{tr_label('Diseased', lang)}: **{signal['diseased_pct_delta']:+.0f} {tr_label('pts', lang)}**  \n"
                        f"{tr_label('High severity', lang)}: **{signal['high_pct_delta']:+.0f} {tr_label('pts', lang)}**"
                    )
                else:
                    st.caption(tr_label("Change vs. prior window", lang))
                    st.markdown(f"*{tr_label('N/A', lang)}*")

            if recent["disease_counts"]:
                dc = recent["disease_counts"]
                names = sorted(dc, key=lambda k: dc[k], reverse=True)
                fig = go.Figure(go.Bar(
                    orientation="h",
                    x=[dc[n] for n in names],
                    y=[tr_disease(n, lang) for n in names],
                    marker=dict(color=color),
                ))
                fig.update_layout(
                    **CHART_THEME,
                    margin=dict(t=10, b=10, l=10),
                    height=max(120, len(dc) * 36),
                    xaxis_title=tr_label("Saved analyses (recent window)", lang),
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True, key=f"_outbreak_chart_{crop}")

        st.write("")