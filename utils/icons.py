"""Centralised Tabler Icons (pytablericons) support for the whole app.

Every place in the codebase that previously used an emoji as a stand-in
icon (page headers, sidebar nav, callouts, recommendations, status
messages, buttons, page favicon, ...) resolves its icon through this
module instead. Icons are rendered once per (name, size, color, filled)
combination and cached as base64 PNG data-URIs so re-renders on every
Streamlit rerun stay cheap.

Usage
-----
    from utils.icons import icon_html, icon_pil, ICONS

    st.markdown(f"{icon_html('leaf')} Crop Health", unsafe_allow_html=True)
    st.set_page_config(page_icon=icon_pil("leaf"))
"""

from __future__ import annotations

import base64
from functools import lru_cache
from io import BytesIO

from pytablericons import OutlineIcon, FilledIcon, TablerIcons

# ---------------------------------------------------------------------------
# Semantic name -> Tabler icon. Centralising the mapping means every page
# refers to *what the icon means* ("disease", "save") rather than picking
# a raw enum member, so the visual language stays consistent app-wide.
# ---------------------------------------------------------------------------
ICONS: dict[str, OutlineIcon] = {
    # Brand / navigation
    "leaf":        OutlineIcon.LEAF,
    "home":        OutlineIcon.HOME,
    "dashboard":   OutlineIcon.CHART_BAR,
    "disease":     OutlineIcon.BUG,
    "environment": OutlineIcon.THERMOMETER,
    "health":      OutlineIcon.ACTIVITY_HEARTBEAT,
    "history":     OutlineIcon.FOLDER,
    "about":       OutlineIcon.INFO_CIRCLE,
    "field_scan":  OutlineIcon.MAP_2,
    "alerts":      OutlineIcon.ALERT_TRIANGLE,
    "weather":     OutlineIcon.CLOUD_RAIN,

    # Environmental factors
    "temperature": OutlineIcon.THERMOMETER,
    "humidity":    OutlineIcon.DROPLET,
    "soil":        OutlineIcon.PLANT_2,
    "rainfall":    OutlineIcon.CLOUD_RAIN,

    # Actions
    "camera":      OutlineIcon.CAMERA,
    "search":      OutlineIcon.SEARCH,
    "save":        OutlineIcon.DEVICE_FLOPPY,
    "refresh":     OutlineIcon.REFRESH,
    "eye":         OutlineIcon.EYE,

    # Status
    "success":     OutlineIcon.CHECK,
    "error":       OutlineIcon.X,
    "warning":     OutlineIcon.ALERT_TRIANGLE,
    "info":        OutlineIcon.INFO_CIRCLE,
    "healthy":     OutlineIcon.SHIELD_CHECK,
    "diseased":    OutlineIcon.VIRUS,
    "sparkles":    OutlineIcon.SPARKLES,
}

DEFAULT_COLOR = "#2F6D46"  # matches --leaf in utils/ui.py's theme


@lru_cache(maxsize=256)
def _load_b64(icon_key: str, size: int, color: str, filled: bool) -> str:
    """Render a Tabler icon to a base64-encoded PNG (cached)."""
    icon = (FilledIcon[icon_key] if filled and hasattr(FilledIcon, icon_key)
            else ICONS.get(icon_key, OutlineIcon.LEAF))
    img = TablerIcons.load(icon, size=size, color=color)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def icon_b64(name: str, size: int = 20, color: str = DEFAULT_COLOR,
             filled: bool = False) -> str:
    """Return a base64 PNG string (no data-URI prefix) for an icon."""
    return _load_b64(name, size, color, filled)


def icon_html(name: str, size: int = 20, color: str = DEFAULT_COLOR,
              filled: bool = False, margin_right: str = ".4em") -> str:
    """Return an inline <img> tag for use inside raw-HTML markdown blocks."""
    b64 = icon_b64(name, size=size, color=color, filled=filled)
    return (
        f'<img src="data:image/png;base64,{b64}" width="{size}" height="{size}" '
        f'style="vertical-align:-{int(size*0.2)}px;margin-right:{margin_right}" '
        f'alt="{name}"/>'
    )


def icon_pil(name: str, size: int = 64, color: str = DEFAULT_COLOR,
             filled: bool = False):
    """Return a PIL Image, e.g. for st.set_page_config(page_icon=...)."""
    icon = (FilledIcon[name] if filled and hasattr(FilledIcon, name)
            else ICONS.get(name, OutlineIcon.LEAF))
    return TablerIcons.load(icon, size=size, color=color)