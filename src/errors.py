"""Shared error types and a Streamlit-safe error boundary.

Two concerns live here:

1. A small, consistent exception hierarchy so every layer of the app
   (validation, database, model inference) raises something specific and
   catchable, instead of letting raw library exceptions (sqlite3.Error,
   arbitrary numpy/sklearn errors, ...) bubble straight to the UI.

2. `safe_action()` — a context manager every page uses around risky work
   (predictions, DB reads/writes, model loads). Known exception types get
   a clear, specific st.error message. Anything unexpected is logged
   server-side (full traceback, for developers) and shown to the user as
   a short, generic message — the raw exception text and traceback never
   reach the browser.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator

import streamlit as st

# Server-side log only. Never rendered in the UI — that's the whole point.
logger = logging.getLogger("crop_health_app")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    ))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------
class AppError(Exception):
    """Base class for errors with a message that is already safe to show
    the user as-is (no internal details, no stack trace).
    """


class DatabaseError(AppError):
    """Raised by src.db when a SQLite operation fails."""


class ModelNotFoundError(AppError):
    """Raised when a required trained model file is missing."""


class PredictionError(AppError):
    """Raised when a model loads fine but inference itself fails."""


class GradCAMError(AppError):
    """Raised when Grad-CAM heatmap generation fails.

    Kept distinct from PredictionError so callers can catch it separately
    and degrade gracefully — a failed explainability heatmap should never
    hide an otherwise-successful disease prediction.
    """


# ValidationError and ImageValidationError live in their own modules
# (src.validation / src.image_preprocessing) since they're raised in many
# call sites, but both subclass ValueError, so safe_action() catches them
# generically via the ValueError branch below.


# ---------------------------------------------------------------------------
# Streamlit-safe error boundary
# ---------------------------------------------------------------------------
@contextmanager
def safe_action(label: str = "This action") -> Iterator[None]:
    """Run a block of page logic, translating any exception into a clean
    st.error() — never a raw traceback.

    Usage:
        with safe_action("Saving analysis"):
            insert_analysis(record)

    - AppError (and subclasses: DatabaseError, ModelNotFoundError,
      PredictionError, GradCAMError) and ValueError (and subclasses:
      ValidationError, ImageValidationError): shown to the user verbatim — these are
      already written to be safe, specific, and actionable.
    - FileNotFoundError: shown verbatim (used for "model not trained yet"
      messages that already include instructions).
    - Anything else: logged in full server-side with a traceback, and the
      user sees only a short, generic message.
    """
    try:
        yield
    except (AppError, ValueError, FileNotFoundError) as e:
        st.error(str(e))
    except Exception:
        logger.exception("Unhandled error during: %s", label)
        st.error(
            f"{label} failed unexpectedly. Please try again. "
            "If the problem continues, contact the app maintainer."
        )