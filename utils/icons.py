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
import io
import os
import re
from functools import lru_cache
from pathlib import Path

import pytablericons
from pytablericons import OutlineIcon, FilledIcon

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
    "crop_doctor": OutlineIcon.STETHOSCOPE,

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

_ICON_DIR = Path(pytablericons.__file__).resolve().parent / "icons"


# ---------------------------------------------------------------------------
# NOTE: we deliberately do NOT use TablerIcons.load(). In pytablericons 1.0.x
# it sets width/height=size AND transform=scale(size/24) on the <svg> root, so
# the icon is scaled twice and only a cropped corner of it is rasterised
# (the "broken icons" bug). We read the raw SVG ourselves instead.
# ---------------------------------------------------------------------------
def _resolve(name: str, filled: bool):
    """Return (Icon enum member, 'outline'|'filled') for a semantic name."""
    if filled and hasattr(FilledIcon, name):
        return getattr(FilledIcon, name), "filled"
    return ICONS.get(name, OutlineIcon.LEAF), "outline"


@lru_cache(maxsize=256)
def _svg_source(name: str, size: int, color: str, filled: bool) -> str:
    """Raw SVG text, sized and coloured, with the original viewBox intact."""
    icon, kind = _resolve(name, filled)
    svg = (_ICON_DIR / kind / icon.value).read_text(encoding="utf-8")
    svg = re.sub(r"<!--.*?-->", "", svg, flags=re.S)  # drop metadata comment
    attr = "fill" if kind == "filled" else "stroke"
    svg = svg.replace(f'{attr}="currentColor"', f'{attr}="{color}"')
    svg = re.sub(r'width="\d+"', f'width="{size}"', svg, count=1)
    svg = re.sub(r'height="\d+"', f'height="{size}"', svg, count=1)
    return svg.strip()


@lru_cache(maxsize=256)
def _load_b64(icon_key: str, size: int, color: str, filled: bool) -> str:
    """Base64 of the SVG (crisp at any DPI, no rasteriser needed)."""
    svg = _svg_source(icon_key, size, color, filled)
    return base64.b64encode(svg.encode("utf-8")).decode("ascii")


def icon_b64(name: str, size: int = 20, color: str = DEFAULT_COLOR,
             filled: bool = False) -> str:
    """Return a base64 SVG string (no data-URI prefix) for an icon."""
    return _load_b64(name, size, color, filled)


def icon_html(name: str, size: int = 20, color: str = DEFAULT_COLOR,
              filled: bool = False, margin_right: str = ".4em") -> str:
    """Return an inline <img> tag for use inside raw-HTML markdown blocks."""
    b64 = icon_b64(name, size=size, color=color, filled=filled)
    return (
        f'<img src="data:image/svg+xml;base64,{b64}" width="{size}" height="{size}" '
        f'style="vertical-align:-{int(size*0.2)}px;margin-right:{margin_right}" '
        f'alt="{name}"/>'
    )


@lru_cache(maxsize=16)
def icon_pil(name: str, size: int = 64, color: str = DEFAULT_COLOR,
             filled: bool = False):
    """PIL Image for st.set_page_config(page_icon=...). Falls back to an
    emoji string if SVG rasterising is unavailable (never crashes the app)."""
    try:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        import pygame
        from PIL import Image
        svg = _svg_source(name, size, color, filled)
        surf = pygame.image.load(io.BytesIO(svg.encode("utf-8")))
        return Image.frombytes(
            "RGBA", surf.get_size(), pygame.image.tobytes(surf, "RGBA"))
    except Exception:
        return "\U0001F33F"