"""Reusable Streamlit UI components and agriculture-themed styling.

This module centralises the look-and-feel so every page renders as one
coherent system. It also exposes dummy data generators used by the layout
while the real AI and database layers are not yet implemented.
"""

from __future__ import annotations

from typing import Callable

import plotly.graph_objects as go
import streamlit as st

from config import APP_CONFIG, ENV_RANGES


# ---------------------------------------------------------------------------
# Look & feel
# ---------------------------------------------------------------------------
def inject_custom_css() -> None:
    """Inject an agriculture-themed stylesheet into the Streamlit app."""
    st.markdown(
        """
        <style>
        :root {
            --leaf:        #2e7d32;
            --leaf-light: #66bb6a;
            --soil:       #6d4c41;
            --sun:        #fdd835;
            --sky:        #e8f5e9;
            --ink:        #1b5e20;
        }

        /* App container breathing room */
        .block-container { padding-top: 2rem; padding-bottom: 3rem; }

        /* Page header card */
        .pg-header {
            background: linear-gradient(135deg, #2e7d32 0%, #66bb6a 100%);
            color: #ffffff;
            padding: 1.5rem 1.75rem;
            border-radius: 14px;
            box-shadow: 0 6px 18px rgba(46,125,50,0.25);
            margin-bottom: 1.25rem;
        }
        .pg-header h1 { margin: 0; font-size: 1.6rem; font-weight: 700; }
        .pg-header p  { margin: .35rem 0 0; opacity: .92; font-size: .95rem; }

        /* Metric tile */
        .metric-tile {
            background: #ffffff;
            border: 1px solid #e0e0e0;
            border-left: 5px solid var(--leaf);
            border-radius: 12px;
            padding: 1rem 1.1rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
            height: 100%;
        }
        .metric-tile .label { font-size: .8rem; color: #555; text-transform: uppercase; letter-spacing: .04em; }
        .metric-tile .value { font-size: 1.6rem; font-weight: 700; color: var(--ink); margin-top: .2rem; }
        .metric-tile .delta { font-size: .8rem; margin-top: .15rem; }

        /* Info / callout box */
        .callout {
            background: var(--sky);
            border-left: 5px solid var(--leaf-light);
            border-radius: 10px;
            padding: .85rem 1rem;
            margin: 1rem 0;
        }

        /* Section card */
        .card {
            background: #ffffff;
            border: 1px solid #ececec;
            border-radius: 12px;
            padding: 1.1rem 1.25rem;
            box-shadow: 0 1px 4px rgba(0,0,0,0.03);
            margin-bottom: 1rem;
        }

        /* Sidebar polish */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f1f8e9 0%, #ffffff 100%);
        }
        .sidebar-brand { font-size: 1.1rem; font-weight: 700; color: var(--ink); }
        .sidebar-sub   { font-size: .8rem; color: #6d4c41; }

        /* Footer */
        .footer {
            margin-top: 2.5rem; text-align: center; color: #8a8a8a;
            font-size: .78rem; border-top: 1px solid #eee; padding-top: .9rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def set_page_config(page_title: str) -> None:
    """Apply a consistent page config across every view."""
    st.set_page_config(
        page_title=f"{APP_CONFIG['title']} · {page_title}",
        page_icon=APP_CONFIG["page_icon"],
        layout="wide",
        initial_sidebar_state="expanded",
    )


def page_header(icon: str, title: str, subtitle: str | None = None) -> None:
    """Render the themed banner used at the top of every page."""
    st.markdown(
        f"""
        <div class="pg-header">
          <h1>{icon} {title}</h1>
          {f"<p>{subtitle}</p>" if subtitle else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Layout primitives
# ---------------------------------------------------------------------------
def metric_tile(label: str, value: str, delta: str | None = None) -> None:
    """A themed KPI tile rendered as raw HTML inside its own column."""
    delta_html = f'<div class="delta">{delta}</div>' if delta else ""
    st.markdown(
        f"""
        <div class="metric-tile">
          <div class="label">{label}</div>
          <div class="value">{value}</div>
          {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def callout(text: str) -> None:
    st.markdown(f'<div class="callout">{text}</div>', unsafe_allow_html=True)


def card(title: str | None = None, body: str | None = None) -> None:
    """A simple bordered card container."""
    inner = ""
    if title:
        inner += f"<h4 style='margin-top:0;color:var(--ink)'>{title}</h4>"
    if body:
        inner += f"<div>{body}</div>"
    st.markdown(f'<div class="card">{inner}</div>', unsafe_allow_html=True)


def footer() -> None:
    st.markdown(
        f'<div class="footer">{APP_CONFIG["title"]} · Software-only demo · '
        "No hardware required</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
PAGES: list[tuple[str, str, str]] = [
    ("Home",                 "home",        "🏠"),
    ("Dashboard",            "dashboard",   "📊"),
    ("Disease Detection",    "disease",     "🦠"),
    ("Environmental Analysis", "environment", "🌡️"),
    ("Crop Health Analysis", "health",      "🌱"),
    ("Analysis History",     "history",     "🗂️"),
    ("About Project",        "about",       "ℹ️"),
]


def render_sidebar() -> str:
    """Render the branded sidebar nav and return the selected page key."""
    with st.sidebar:
        st.markdown(
            f"""
            <div class="sidebar-brand">{APP_CONFIG['page_icon']} {APP_CONFIG['title']}</div>
            <div class="sidebar-sub">{APP_CONFIG['subtitle']}</div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("---")

        labels = [f"{icon}  {name}" for name, _, icon in PAGES]
        keys = [key for _, key, _ in PAGES]
        choice = st.radio("Navigate", labels, label_visibility="collapsed")
        idx = labels.index(choice)
        st.session_state["current_page"] = keys[idx]

        st.markdown("---")
        st.caption("🌿 Agriculture-themed demo build")
        return keys[idx]


# ---------------------------------------------------------------------------
# Dummy data (layout scaffolding only — no AI / DB yet)
# ---------------------------------------------------------------------------
def get_dummy_diseases() -> list[dict]:
    """Placeholder disease catalog used by the detection page."""
    return [
        {"name": "Tomato Early Blight", "confidence": 0.0, "severity": "—",
         "crop": "Tomato", "color": "#e57373"},
        {"name": "Tomato Leaf Mold",     "confidence": 0.0, "severity": "—",
         "crop": "Tomato", "color": "#ba68c8"},
        {"name": "Corn Common Rust",     "confidence": 0.0, "severity": "—",
         "crop": "Corn",   "color": "#ff8a65"},
        {"name": "Potato Late Blight",    "confidence": 0.0, "severity": "—",
         "crop": "Potato", "color": "#f06292"},
        {"name": "Healthy",               "confidence": 0.0, "severity": "—",
         "crop": "Various", "color": "#66bb6a"},
    ]


def get_dummy_env_readings() -> dict:
    """Placeholder environmental readings in the configured units."""
    return {
        "temperature":   28.4,   # °C
        "humidity":      62.5,   # %
        "soil_moisture": 41.0,   # %
        "rainfall":       3.2,   # mm
    }


def get_dummy_history(n: int = 8) -> list[dict]:
    """Placeholder analysis history records for the dashboard/history pages."""
    import itertools
    crops = ["Tomato", "Corn", "Potato", "Rice", "Wheat"]
    statuses = ["Healthy", "Early Blight", "Leaf Mold", "Rust"]
    scores = [88, 72, 64, 45, 91, 58, 76, 50]
    dates = ["2026-08-19", "2026-08-18", "2026-08-17", "2026-08-16",
             "2026-08-15", "2026-08-14", "2026-08-13", "2026-08-12"]
    rows = []
    for i in range(n):
        rows.append({
            "id": i + 1,
            "date": dates[i % len(dates)],
            "crop": crops[i % len(crops)],
            "status": statuses[i % len(statuses)],
            "health_score": scores[i % len(scores)],
            "temperature": round(22 + (i % 5) * 1.8, 1),
            "humidity": round(50 + (i % 6) * 4, 1),
            "soil_moisture": round(35 + (i % 7) * 3, 1),
            "rainfall": round((i % 4) * 1.5, 1),
        })
    return rows


def env_status_bar(label: str, value: float, vmin: float, vmax: float, unit: str) -> None:
    """Render a single environmental reading as a coloured progress bar."""
    ratio = max(0.0, min(1.0, (value - vmin) / (vmax - vmin)))
    pct = int(ratio * 100)
    st.markdown(
        f"""
        <div style="margin-bottom:.7rem">
          <div style="display:flex;justify-content:space-between;font-size:.85rem">
            <span><b>{label}</b></span><span>{value} {unit}</span>
          </div>
          <div style="background:#eee;border-radius:6px;overflow:hidden;height:10px;margin-top:.25rem">
            <div style="width:{pct}%;height:100%;background:linear-gradient(90deg,#66bb6a,#2e7d32)"></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def env_gauge(value: float, vmin: float, vmax: float, title: str, unit: str) -> go.Figure:
    """A Plotly radial gauge for an environmental reading."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": f"<span style='font-size:13px;color:#333'>{title}</span>"},
        number={"suffix": f" {unit}"},
        gauge={
            "axis": {"range": [vmin, vmax]},
            "bar": {"color": "#2e7d32"},
            "bgcolor": "white",
            "borderwidth": 1,
            "steps": [
                {"range": [vmin, vmin + 0.33 * (vmax - vmin)], "color": "#e8f5e9"},
                {"range": [vmin + 0.33 * (vmax - vmin), vmin + 0.66 * (vmax - vmin)], "color": "#c8e6c9"},
                {"range": [vmin + 0.66 * (vmax - vmin), vmax], "color": "#a5d6a7"},
            ],
            "threshold": {
                "line": {"color": "#f57c00", "width": 3},
                "thickness": 0.8,
                "value": value,
            },
        },
    ))
    fig.update_layout(height=230, margin=dict(t=40, b=10, l=20, r=20))
    return fig


def render_env_readings(readings: dict) -> None:
    """Convenience: render the four environmental gauges in a row."""
    cols = st.columns(len(ENV_RANGES))
    for col, (key, spec) in zip(cols, ENV_RANGES.items()):
        with col:
            st.plotly_chart(
                env_gauge(readings[key], spec["min"], spec["max"],
                          key.replace("_", " ").title(), spec["unit"]),
                use_container_width=True,
            )
