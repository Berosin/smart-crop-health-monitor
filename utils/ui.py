"""Reusable Streamlit UI components and agriculture-themed styling.

This module centralises the look-and-feel so every page renders as one
coherent system: the CSS design system, page chrome (header, sidebar,
footer), and small display components (metric tiles, risk badges, the
health-score card) shared across pages. A couple of dummy-data generators
remain for widget default values (e.g. pre-filling the environmental
readings form) — everything else in the app reads from the real trained
models and the SQLite database.
"""

from __future__ import annotations

import re

import streamlit as st

from config import APP_CONFIG
from utils.icons import icon_html, icon_pil

# Shared Plotly theming so every chart sits visually inside the same
# paper/card system as the rest of the UI (transparent surface, Inter
# font, ink-colored text) instead of Plotly's stark default white.
CHART_THEME: dict = dict(
    template="plotly_white",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#FFFFFF",
    font=dict(family="Inter, -apple-system, sans-serif", color="#23291F", size=12),
)


# ---------------------------------------------------------------------------
# Look & feel
# ---------------------------------------------------------------------------
def inject_custom_css() -> None:
    """Inject the app's visual system: palette, type, and component styles.

    Design direction — "field journal & spectral scan": a muted, paper-toned
    light theme (cooler and greener than the generic cream-background AI
    look) paired with a warm serif for headings (evoking a printed
    agronomy notebook) and a monospace face for scores/readings (evoking an
    instrument readout). The one recurring signature motif is a spectral
    scan bar — a gradient sweeping from soil to canopy to chlorophyll —
    used sparingly on the page header and the health-score card, echoing
    the app's core idea: every reading maps onto a stressed→healthy scale.
    """
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap');

        :root {
            /* Canopy — primary green family */
            --leaf:        #2F6D46;
            --leaf-light:  #7FA687;
            --ink:         #1C2E20;

            /* Soil & spectral accents */
            --soil:        #8A6A47;
            --sun:         #D6A34B;
            --clay:        #C97A3B;
            --brick:       #B5564B;
            --brick-deep:  #7C3730;

            /* Paper surfaces */
            --paper:       #F5F6EF;
            --paper-deep:  #EAEFE2;
            --sky:         #EAEFE2;
            --card:        #FFFFFF;
            --line:        #DEE3D2;

            /* Text */
            --text:        #23291F;
            --text-muted:  #5B6353;
            --text-faint:  #7C8571;

            --font-display: 'Fraunces', Georgia, serif;
            --font-body:    'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            --font-mono:    'IBM Plex Mono', 'SFMono-Regular', Menlo, monospace;

            --radius: 10px;
            --shadow-sm: 0 1px 3px rgba(28,46,32,0.06);
            --shadow-md: 0 4px 16px rgba(28,46,32,0.08);
        }

        /* ---------------------------------------------------------------
           App-wide paper background & typography
        --------------------------------------------------------------- */
        [data-testid="stAppViewContainer"],
        [data-testid="stHeader"] {
            background: var(--paper);
        }
        html, body, [class*="css"] { font-family: var(--font-body); color: var(--text); }
        .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1180px; }

        h1, h2, h3, h4 {
            font-family: var(--font-display);
            color: var(--ink);
            font-weight: 600;
            letter-spacing: -.01em;
        }
        h4 { font-size: 1.05rem; margin-top: 1.6rem; }

        /* Streamlit's own widget labels/captions stay on the body face */
        [data-testid="stWidgetLabel"] p { font-family: var(--font-body); font-weight: 500; color: var(--text-muted); }
        [data-testid="stCaptionContainer"] { color: var(--text-faint); }

        /* ---------------------------------------------------------------
           Page header — signature spectral scan bar along the top edge
        --------------------------------------------------------------- */
        .pg-header {
            position: relative;
            background: linear-gradient(155deg, #24422E 0%, #2F6D46 62%, #4C7F52 100%);
            color: #FFFFFF;
            padding: 1.6rem 1.9rem 1.5rem;
            border-radius: var(--radius);
            box-shadow: var(--shadow-md);
            margin-bottom: 1.4rem;
            overflow: hidden;
        }
        .pg-header::before {
            content: "";
            position: absolute; top: 0; left: 0; right: 0; height: 4px;
            background: linear-gradient(90deg, var(--brick-deep) 0%, var(--sun) 32%, var(--leaf-light) 66%, #C9E2B8 100%);
        }
        .pg-header h1 {
            margin: 0; font-size: 1.55rem; font-weight: 600; color: #FFFFFF;
            display: flex; align-items: center; font-family: var(--font-display);
        }
        .pg-header p { margin: .4rem 0 0; opacity: .88; font-size: .93rem; font-family: var(--font-body); }

        /* ---------------------------------------------------------------
           Metric tile
        --------------------------------------------------------------- */
        .metric-tile {
            background: var(--card);
            border: 1px solid var(--line);
            border-left: 4px solid var(--leaf);
            border-radius: var(--radius);
            padding: 1rem 1.1rem;
            box-shadow: var(--shadow-sm);
            height: 100%;
        }
        .metric-tile .label { font-size: .74rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: .06em; font-weight: 600; }
        .metric-tile .value {
            font-family: var(--font-mono); font-size: 1.55rem; font-weight: 600;
            color: var(--ink); margin-top: .25rem; line-height: 1.25;
            word-break: break-word; overflow-wrap: break-word;
        }
        .metric-tile .delta { font-size: .78rem; margin-top: .2rem; color: var(--text-faint); }
        .history-metric .value {
            font-size: 1.3rem; line-height: 1.2; overflow-wrap: anywhere;
        }

        /* ---------------------------------------------------------------
           Info / callout box
        --------------------------------------------------------------- */
        .callout {
            background: var(--paper-deep);
            border-left: 4px solid var(--leaf-light);
            border-radius: 8px;
            padding: .85rem 1rem;
            margin: 1rem 0;
            font-size: .92rem;
            color: var(--text);
        }

        /* ---------------------------------------------------------------
           Section card
        --------------------------------------------------------------- */
        .card {
            background: var(--card);
            border: 1px solid var(--line);
            border-radius: var(--radius);
            padding: 1.1rem 1.25rem;
            box-shadow: var(--shadow-sm);
            margin-bottom: 1rem;
        }

        /* ---------------------------------------------------------------
           Sidebar
        --------------------------------------------------------------- */
        section[data-testid="stSidebar"] {
            background: var(--paper-deep);
            border-right: 1px solid var(--line);
        }
        section[data-testid="stSidebar"] > div { padding-top: 1.2rem; }
        .sidebar-brand {
            font-family: var(--font-display); font-size: 1.12rem; font-weight: 600;
            color: var(--ink); display: flex; align-items: center;
        }
        .sidebar-sub { font-size: .8rem; color: var(--soil); margin-top: .1rem; }

        /* Sidebar nav buttons — quiet/ghost when inactive, filled when the
           current page, giving a clear "you are here" state (a plain
           st.radio only shows a small selected dot, easy to miss). */
        section[data-testid="stSidebar"] .stButton > button {
            text-align: left;
            justify-content: flex-start;
            font-weight: 500;
            border-radius: 8px;
            padding: .5rem .8rem;
            margin-bottom: .15rem;
            transition: background .15s ease, color .15s ease, border-color .15s ease;
        }
        section[data-testid="stSidebar"] button[kind="secondary"] {
            background: transparent;
            border: 1px solid transparent;
            color: var(--text-muted);
            box-shadow: none;
        }
        section[data-testid="stSidebar"] button[kind="secondary"]:hover {
            background: rgba(47,109,70,0.09);
            color: var(--ink);
            border-color: transparent;
        }
        section[data-testid="stSidebar"] button[kind="primary"] {
            font-weight: 600;
            box-shadow: var(--shadow-sm);
        }

        /* ---------------------------------------------------------------
           Footer
        --------------------------------------------------------------- */
        .footer {
            margin-top: 2.5rem; text-align: center; color: var(--text-faint);
            font-size: .78rem; border-top: 1px solid var(--line); padding-top: .9rem;
            font-family: var(--font-body);
        }

        /* ---------------------------------------------------------------
           Health score card — the same spectral bar as a signature echo
        --------------------------------------------------------------- */
        .health-card {
            position: relative;
            background: var(--card); border: 1px solid var(--line);
            border-radius: 14px; padding: 1.5rem 1.5rem 1.4rem;
            box-shadow: var(--shadow-md); text-align: center;
            overflow: hidden;
        }
        .health-card::before {
            content: "";
            position: absolute; top: 0; left: 0; right: 0; height: 4px;
            background: linear-gradient(90deg, var(--brick-deep) 0%, var(--sun) 32%, var(--leaf-light) 66%, #C9E2B8 100%);
        }
        .health-card .score-label { font-size:.76rem; color:var(--text-muted);
            text-transform:uppercase; letter-spacing:.06em; font-weight: 600; }
        .health-card .score-value { font-family: var(--font-mono); font-size:2.7rem; font-weight:600;
            line-height:1.15; margin-top: .2rem; }
        .health-card .score-grade { font-size:1.05rem; font-weight:600;
            margin-top:.25rem; font-family: var(--font-body); color: var(--text-muted); }
        .score-bar-track { background:var(--paper-deep); border-radius:8px; height:12px;
            margin:.8rem 0 .3rem; overflow:hidden; border: 1px solid var(--line); }
        .score-bar-fill  { height:100%; border-radius:8px; }

        /* ---------------------------------------------------------------
           Risk indicator pill
        --------------------------------------------------------------- */
        .risk-pill { display:inline-flex; align-items:center; gap:.45rem;
            padding:.35rem .85rem; border-radius:20px; font-weight:600;
            font-size:.83rem; color:#fff; font-family: var(--font-body); }
        .risk-dot { width:8px; height:8px; border-radius:50%; background:rgba(255,255,255,.9); }

        /* ---------------------------------------------------------------
           Recommendation item
        --------------------------------------------------------------- */
        .rec-item { display:flex; gap:.7rem; align-items:flex-start;
            padding:.75rem .95rem; border:1px solid var(--line); border-radius:10px;
            margin-bottom:.55rem; background:var(--card); }
        .rec-icon { line-height:1.3; display:flex; align-items:flex-start; padding-top:.1rem }
        .rec-text { font-size:.88rem; color:var(--text); }
        .rec-title { font-weight:600; color:var(--ink); font-size:.85rem;
            text-transform: uppercase; letter-spacing: .04em; margin-bottom:.15rem }

        /* ---------------------------------------------------------------
           Native widget refinements (buttons, tabs, expander)
        --------------------------------------------------------------- */
        .stButton > button, .stDownloadButton > button {
            border-radius: 8px; font-weight: 600; font-family: var(--font-body);
            transition: box-shadow .15s ease, transform .1s ease;
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            box-shadow: var(--shadow-sm);
        }
        .stButton > button:active {
            transform: translateY(1px);
        }
        .stButton > button:disabled {
            opacity: .55;
        }
        div[data-testid="stExpander"] {
            border-radius: var(--radius); border: 1px solid var(--line);
            box-shadow: var(--shadow-sm);
        }
        div[data-testid="stExpander"] summary {
            font-family: var(--font-body); font-weight: 600; color: var(--ink);
        }
        div[data-testid="stExpander"] summary:hover { color: var(--leaf); }
        [data-testid="stMetricValue"] { font-family: var(--font-mono); color: var(--ink); }
        [data-testid="stMetricLabel"] { font-family: var(--font-body); color: var(--text-muted); }

        /* ---------------------------------------------------------------
           Status messages (error / warning / success / info) — keep
           Streamlit's universally-understood semantic colors, but bring
           shape/type into the same card system as the rest of the app.
        --------------------------------------------------------------- */
        [data-testid="stAlert"] {
            border-radius: var(--radius);
            box-shadow: var(--shadow-sm);
        }
        [data-testid="stAlert"] p {
            font-family: var(--font-body); font-size: .92rem; margin-bottom: 0;
        }

        /* ---------------------------------------------------------------
           Loading states — keep the spinner text on-brand
        --------------------------------------------------------------- */
        [data-testid="stSpinner"] {
            font-family: var(--font-body); color: var(--text-muted);
            font-size: .9rem;
        }

        /* ---------------------------------------------------------------
           Tables
        --------------------------------------------------------------- */
        [data-testid="stDataFrame"] {
            border-radius: var(--radius); overflow: hidden;
            border: 1px solid var(--line); box-shadow: var(--shadow-sm);
        }

        /* ---------------------------------------------------------------
           Images — uploaded/preview photos get the same card treatment
        --------------------------------------------------------------- */
        [data-testid="stImage"] img {
            border-radius: var(--radius);
            border: 1px solid var(--line);
            box-shadow: var(--shadow-sm);
        }

        /* ---------------------------------------------------------------
           Form inputs — consistent corner radius across widget types
        --------------------------------------------------------------- */
        .stTextInput input, .stNumberInput input,
        .stSelectbox div[data-baseweb="select"] > div,
        .stMultiSelect div[data-baseweb="select"] > div,
        .stDateInput input {
            border-radius: 8px !important;
        }

        /* ---------------------------------------------------------------
           Section dividers — quiet hairline instead of the browser default
        --------------------------------------------------------------- */
        hr {
            border: none; border-top: 1px solid var(--line);
            margin: 1.1rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def set_page_config(page_title: str) -> None:
    """Apply a consistent page config across every view."""
    st.set_page_config(
        page_title=f"{APP_CONFIG['title']} · {page_title}",
        page_icon=icon_pil(APP_CONFIG["page_icon"], size=64),
        layout="wide",
        initial_sidebar_state="expanded",
    )


def page_header(icon: str, title: str, subtitle: str | None = None) -> None:
    """Render the themed banner used at the top of every page.

    `icon` is a semantic key from utils.icons.ICONS (e.g. "disease"),
    rendered as a Tabler Icon rather than an emoji.
    """
    icon_tag = icon_html(icon, size=30, color="#ffffff", margin_right=".5em")
    st.markdown(
        f"""
        <div class="pg-header">
          <h1>{icon_tag}{title}</h1>
          {f"<p>{subtitle}</p>" if subtitle else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Layout primitives
# ---------------------------------------------------------------------------
def metric_tile(label: str, value: str, delta: str | None = None) -> None:
    """A themed KPI tile rendered as raw HTML inside its own column.

    Long values (e.g. a multi-word disease name) get a smaller font-size
    so they wrap cleanly within the tile instead of overflowing or
    crowding the label above them.
    """
    delta_html = f'<div class="delta">{delta}</div>' if delta else ""
    value_style = ' style="font-size:1.1rem"' if len(str(value)) > 12 else ""
    st.markdown(
        f"""
        <div class="metric-tile">
          <div class="label">{label}</div>
          <div class="value"{value_style}>{value}</div>
          {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def callout(text: str) -> None:
    text = _render_inline_markdown(text)
    st.markdown(f'<div class="callout">{text}</div>', unsafe_allow_html=True)


def card(title: str | None = None, body: str | None = None) -> None:
    """A simple bordered card container."""
    inner = ""
    if title:
        inner += f"<h4 style='margin-top:0;color:var(--ink)'>{_render_inline_markdown(title)}</h4>"
    if body:
        inner += f"<div>{_render_inline_markdown(body)}</div>"
    st.markdown(f'<div class="card">{inner}</div>', unsafe_allow_html=True)


def _render_inline_markdown(text: str) -> str:
    """Render the small bold-markdown subset used in HTML UI components."""
    return re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)


def pretty_name(name: str | None) -> str:
    """Turn a model class name like 'Gray_Leaf_Spot' into 'Gray Leaf Spot'.

    Purely a display transform (underscores -> spaces) — every place that
    stores or compares these names (DB, SEVERITY_MAP, CLASS_COLORS, ...)
    keeps using the raw underscored form; only user-facing text runs
    through this. Reused anywhere a disease/class name is shown as a KPI
    value, chart label, or caption, so long names wrap at natural word
    boundaries instead of overflowing or breaking mid-word.
    """
    if not name or not isinstance(name, str):
        return name
    return name.replace("_", " ")


def footer() -> None:
    st.markdown(
        f'<div class="footer">{APP_CONFIG["title"]} · Software-only demo · '
        "No hardware required</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Reusable analysis components (score card, risk indicator, metric, recs)
# ---------------------------------------------------------------------------
def score_color(score: int) -> str:
    """Map a 0-100 score to a status color."""
    if score >= 70:
        return "#2F6D46"
    if score >= 40:
        return "#D6A34B"
    return "#CE8C82"


def score_grade(score: int) -> str:
    """Map a 0-100 score to a letter grade."""
    if score >= 80:
        return "A"
    if score >= 60:
        return "B"
    if score >= 40:
        return "C"
    return "D"


def health_score_card(score: int, grade: str | None = None,
                      label: str = "Overall health score",
                      show_bar: bool = True) -> None:
    """A reusable, themed health-score card with a 0-100 progress bar."""
    if grade is None:
        grade = score_grade(score)
    color = score_color(score)
    status = ("Excellent" if score >= 80 else "Good" if score >= 60 else
              "Poor" if score >= 40 else "Critical")
    bar_html = (
        f"""
        <div class="score-bar-track">
          <div class="score-bar-fill" style="width:{score}%;background:{color}"></div>
        </div>
        <div style="display:flex;justify-content:space-between;
                    font-size:.72rem;color:#7C8571">
          <span>0</span><span>100</span>
        </div>
        """ if show_bar else "")
    st.markdown(
        f"""
        <div class="health-card">
          <div class="score-label">{label}</div>
          <div class="score-value" style="color:{color}">{score}<span
              style="font-size:1.2rem;color:#8E9682">/100</span></div>
          <div class="score-grade">Grade {grade} · {status}</div>
          {bar_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# Risk level -> (label, color, weight 0-1). Shared by all pages.
RISK_LEVELS: dict[str, tuple[str, str, float]] = {
    "Optimal":   ("Optimal",   "#2F6D46", 0.0),
    "Low":       ("Low",       "#7FA687", 0.25),
    "Moderate":  ("Moderate",  "#C97A3B", 0.50),
    "High":      ("High",      "#B5564B", 0.75),
    "Critical":  ("Critical",  "#7C3730", 1.0),
}


def risk_indicator(level: str, label: str | None = None,
                   show_bar: bool = False) -> None:
    """A reusable colored pill + optional bar representing a risk level."""
    lvl, color, weight = RISK_LEVELS.get(level, ("Unknown", "#93998A", 0.5))
    if label is None:
        label = lvl
    bar_html = ""
    if show_bar:
        bar_html = (
            f"""<div class="score-bar-track" style="height:8px">
                  <div class="score-bar-fill"
                       style="width:{int(weight*100)}%;background:{color}"></div>
                </div>""")
    st.markdown(
        f"""
        <div style="margin-bottom:.5rem">
          <span class="risk-pill" style="background:{color}">
            <span class="risk-dot"></span>{label}
          </span>
          {bar_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_display(label: str, value: str, sub: str | None = None,
                   accent: str | None = None) -> None:
    """A reusable metric tile with an optional colored left accent."""
    border = (f"border-left:5px solid {accent};"
              if accent else "border-left:5px solid var(--leaf);")
    sub_html = f'<div class="delta">{sub}</div>' if sub else ""
    st.markdown(
        f"""
        <div class="metric-tile" style="{border}">
          <div class="label">{label}</div>
          <div class="value">{value}</div>
          {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
# (display name, page key, icon key from utils.icons.ICONS)
PAGES: list[tuple[str, str, str]] = [
    ("Home",                 "home",        "home"),
    ("Dashboard",            "dashboard",   "dashboard"),
    ("Disease Detection",    "disease",     "disease"),
    ("Field Scan",           "field_scan",  "field_scan"),
    ("Environmental Analysis", "environment", "environment"),
    ("Crop Health Analysis", "health",      "health"),
    ("Analysis History",     "history",     "history"),
    ("About Project",        "about",       "about"),
]


def render_sidebar() -> str:
    """Render the branded sidebar nav and return the selected page key.

    Uses full-width buttons (not st.radio) so the current page can be
    given a clear filled "you are here" state — a plain radio only shows
    a small selected dot next to the label, which is easy to miss.
    """
    with st.sidebar:
        brand_icon = icon_html(APP_CONFIG["page_icon"], size=24, margin_right=".4em")
        st.markdown(
            f"""
            <div class="sidebar-brand">{brand_icon}{APP_CONFIG['title']}</div>
            <div class="sidebar-sub">{APP_CONFIG['subtitle']}</div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("---")

        current = st.session_state.get("current_page", PAGES[0][1])

        st.caption("Navigate")
        for name, key, _icon in PAGES:
            is_active = key == current
            clicked = st.button(
                name, key=f"_nav_{key}", use_container_width=True,
                type="primary" if is_active else "secondary",
            )
            if clicked and key != current:
                st.session_state["current_page"] = key
                st.rerun()

        st.session_state["current_page"] = current  # always defined, even on first load
        st.markdown("---")
        st.caption("Agriculture-themed demo build")
        return current


# ---------------------------------------------------------------------------
# Dummy data (layout scaffolding only — no AI / DB yet)
# ---------------------------------------------------------------------------
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