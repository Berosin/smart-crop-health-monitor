"""Crop Doctor — a Q&A chat interface grounded in one saved analysis record.

Presentation layer only; all answering logic lives in src.crop_doctor,
which is pure Python with no Streamlit dependency (same split as every
other engine module in this app: src.health_engine, src.recommendation_engine,
src.yield_loss, src.outbreak_detection).

No external AI/LLM API is called anywhere in this page — every reply comes
from src.crop_doctor's keyword-classified, rule-based handlers running
against this app's own stored data, so there's no per-message cost and no
risk of a reply that isn't grounded in the selected record.
"""

from __future__ import annotations

import streamlit as st

from src.crop_doctor import (
    normalize_record, record_summary_line, answer, SUGGESTED_QUESTIONS,
)
from src.db import get_analyses, get_disease_analyses, get_environment_analyses, get_field_scans
from src.errors import DatabaseError, logger
from src.i18n import get_language, tr_label, tr_crop, tr_disease
from utils.ui import page_header, callout, footer
from utils.icons import icon_html

RECORD_TYPE_META = [
    # (record_type key, display label, loader)
    ("health", "Crop Health", get_analyses),
    ("disease", "Disease Detection", get_disease_analyses),
    ("environment", "Environmental", get_environment_analyses),
    ("field_scan", "Field Scan", get_field_scans),
]


def render() -> None:
    lang = get_language()
    page_header(
        "crop_doctor",
        tr_label("Crop Doctor", lang),
        tr_label(
            "Ask questions about one of your saved analyses — answers are grounded "
            "in that record's actual numbers, not a generic chatbot.",
            lang,
        ),
    )

    st.markdown(f"#### {tr_label('1 · Pick a saved analysis', lang)}")

    c1, c2 = st.columns([1, 2])
    with c1:
        record_type = st.selectbox(
            tr_label("Analysis type", lang),
            [rt for rt, _, _ in RECORD_TYPE_META],
            format_func=lambda rt: tr_label(dict((r, l) for r, l, _ in RECORD_TYPE_META)[rt], lang),
            key="_doctor_record_type",
        )

    loader = dict((rt, fn) for rt, _, fn in RECORD_TYPE_META)[record_type]

    try:
        rows = loader(limit=200)
    except DatabaseError as e:
        st.error(str(e))
        footer()
        return
    except Exception:
        logger.exception("Unexpected error loading records for Crop Doctor")
        st.error(
            tr_label(
                "Loading saved analyses failed unexpectedly. Please try again. "
                "If the problem continues, contact the app maintainer.",
                lang,
            )
        )
        footer()
        return

    if not rows:
        callout(
            f"{icon_html('info', size=18)}"
            + tr_label("No saved analyses of this type yet. Save one from the relevant page first.", lang)
        )
        footer()
        return

    all_facts = [normalize_record(record_type, row) for row in rows]

    with c2:
        selected_id = st.selectbox(
            tr_label("Record", lang),
            [f["id"] for f in all_facts],
            format_func=lambda rid: next(record_summary_line(f, lang) for f in all_facts if f["id"] == rid),
            key=f"_doctor_record_id_{record_type}",
        )

    facts = next(f for f in all_facts if f["id"] == selected_id)

    _render_context_card(facts, lang)

    st.markdown(f"#### {tr_label('2 · Ask a question', lang)}")

    chat_key = f"_doctor_chat_{record_type}_{selected_id}"
    if chat_key not in st.session_state:
        st.session_state[chat_key] = []
    history = st.session_state[chat_key]

    # Quick-question chips
    suggestions = SUGGESTED_QUESTIONS.get(record_type, [])
    if suggestions and not history:
        st.caption(tr_label("Try asking", lang))
        cols = st.columns(len(suggestions))
        for col, q in zip(cols, suggestions):
            with col:
                q_label = tr_label(q, lang)
                if st.button(q_label, key=f"_doctor_suggest_{record_type}_{selected_id}_{q}", use_container_width=True):
                    # Classify/answer using the original English phrasing
                    # (classify_intent's keyword list is most reliable in
                    # English), but show the person the label they actually
                    # clicked — otherwise a Tamil-language user would see
                    # their own chat bubble echo back in English.
                    _ask(history, q, facts, lang, display_text=q_label)
                    st.rerun()

    for msg in history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_question = st.chat_input(tr_label("Ask about this analysis…", lang))
    if user_question:
        _ask(history, user_question, facts, lang)
        st.rerun()

    if history and st.button(tr_label("New conversation", lang)):
        st.session_state[chat_key] = []
        st.rerun()

    st.markdown(
        f"""
        <div style="font-size:.78rem;color:#5B6353;margin-top:1rem">
          {icon_html('info', size=14, margin_right='.3em')}
          {tr_label("This app never calls an external AI chat service for this — every answer is generated locally from this app's own rules and stored data.", lang)}
        </div>
        """,
        unsafe_allow_html=True,
    )

    footer()


def _ask(history: list[dict], question: str, facts: dict, lang: str, display_text: str | None = None) -> None:
    """Append a user question + the grounded answer to this record's
    conversation history (in place, so the caller's session_state list
    updates too).

    `display_text`, when given, is shown in the chat bubble instead of
    `question` — used for the suggested-question chips, where `question`
    is the English phrasing classify_intent() matches against but
    `display_text` is the label actually shown/clicked in the current
    language.
    """
    history.append({"role": "user", "content": display_text or question})
    reply = answer(facts, question, lang=lang)
    history.append({"role": "assistant", "content": reply})


def _render_context_card(facts: dict, lang: str) -> None:
    """A compact 'here's what you're asking about' summary above the chat,
    so the person can see exactly which numbers ground the answers below —
    the whole point of this page vs. a generic chatbot."""
    rt = facts["record_type"]
    crop_label = tr_crop(facts["crop"], lang) if facts["crop"] else "—"
    date = (facts.get("created_at") or "")[:10]

    chips = [(tr_label("Crop", lang), crop_label), (tr_label("Date", lang), date)]

    if rt in ("health", "disease", "field_scan") and facts.get("disease"):
        chips.append((tr_label("Disease", lang), tr_disease(facts["disease"], lang)))
    if facts.get("severity") and facts["severity"] != "None":
        from src.i18n import tr_severity
        chips.append((tr_label("Severity", lang), tr_severity(facts["severity"], lang)))
    if facts.get("confidence") is not None:
        chips.append((tr_label("Confidence", lang), f"{facts['confidence']*100:.0f}%"))
    if facts.get("health_score") is not None:
        chips.append((tr_label("Health score", lang), f"{facts['health_score']}/100"))
    if facts.get("risk_level"):
        chips.append((tr_label("Risk level", lang), tr_label(facts["risk_level"], lang)))
    if facts.get("healthy_pct") is not None:
        chips.append((tr_label("Healthy", lang), f"{facts['healthy_pct']}%"))

    chip_html = "".join(
        f'<div style="display:inline-block;background:var(--card);border:1px solid var(--line);'
        f'border-radius:999px;padding:.3rem .8rem;margin:.2rem .3rem .2rem 0;font-size:.82rem">'
        f'<span style="color:#7C8571">{label}:</span> <b style="color:var(--ink)">{value}</b></div>'
        for label, value in chips
    )
    st.markdown(
        '<div style="margin-bottom:.75rem">'
        f'<span style="font-size:.78rem;color:#7C8571;text-transform:uppercase;letter-spacing:.04em">'
        f'{tr_label("Grounded in", lang)}</span><br/>{chip_html}'
        '</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    # Standalone entry for direct viewing
    import streamlit as st
    st.set_page_config(page_title="Crop Doctor", layout="wide")
    from utils.ui import inject_custom_css, render_sidebar
    inject_custom_css()
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "crop_doctor"
    render_sidebar()
    st.session_state["current_page"] = "crop_doctor"
    render()