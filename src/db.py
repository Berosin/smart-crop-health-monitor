"""Persistence layer for saved analyses.

This module is intentionally **independent of Streamlit** so it can be unit
tested and reused from CLI scripts, notebooks, or any non-UI context.

Backend
-------
Two interchangeable backends, chosen automatically at connection time:
- **Turso (libSQL)** — a remote, persistent SQLite-compatible database. Used
  whenever TURSO_DATABASE_URL + TURSO_AUTH_TOKEN are configured (env vars,
  or st.secrets — same env-var-first resolution order as
  src.weather.resolve_api_key() and src.model_fetch.resolve_models_zip_url()).
  This is what makes saved analyses survive a Streamlit Community Cloud
  redeploy/restart, since that platform's local filesystem is ephemeral.
- **Local SQLite file** (config.DB_PATH) — the fallback when no Turso
  credentials are configured, e.g. local development. Behaves exactly as
  this module always has.

Both backends are driven through the same SQL and the same DB-API 2.0 call
shapes (`.execute(sql, params)`, `.commit()`, `.fetchall()`, `.lastrowid`,
`.rowcount`, `cursor.description`) — libSQL is a SQLite-file-format- and
wire-compatible fork, so the schema and every query below run unmodified on
either backend. Row->dict conversion is done manually via `cursor.description`
(see `_rows_to_dicts`/`_row_to_dict`) rather than sqlite3's `Row` row_factory,
since that's a sqlite3-module-specific extension libSQL doesn't guarantee —
`cursor.description` is standard DB-API 2.0 and works identically on both.

Schema
------
Four separate tables, one per analysis type, each saved from its own page:
- `analyses`             — Crop Health Analysis page (disease + environment
                            blended into one score)
- `disease_analyses`      — Disease Detection page (single-image predictions,
                            including the saved leaf image path)
- `environment_analyses`  — Environmental Analysis page (readings + trained
                            risk-model output)
- `field_scans`           — Field Scan page (aggregated batch report across
                            many leaf photos from one field walk)

Keeping them separate (rather than one shared table) means each analysis
type's history and dashboard can filter/chart on columns that only make
sense for that type (e.g. image_path only applies to disease analyses),
and one page's save never gets mixed into another page's history.

Functions
---------
- init_db()                         -> create all tables (idempotent)
- insert_analysis(data)              -> save a crop health analysis, return id
- insert_disease_analysis(data)      -> save a disease detection analysis
- insert_environment_analysis(data)  -> save an environmental analysis
- insert_field_scan(data)            -> save an aggregated field scan
- get_analyses(limit=...)            -> list recent crop health analyses
- get_disease_analyses(limit=...)    -> list recent disease analyses
- get_environment_analyses(limit=...)-> list recent environmental analyses
- get_field_scans(limit=...)         -> list recent field scans
- get_analysis_by_id(id)             -> fetch one crop health analysis
- delete_analysis(id)                -> remove one crop health analysis
- delete_disease_analysis(id)        -> remove one disease analysis
- delete_environment_analysis(id)    -> remove one environmental analysis
- delete_field_scan(id)              -> remove one field scan
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator

from config import DB_PATH
from src.errors import DatabaseError


# ---------------------------------------------------------------------------
# Turso (libSQL) credential resolution
# ---------------------------------------------------------------------------
def resolve_turso_credentials() -> tuple[str, str] | None:
    """(TURSO_DATABASE_URL, TURSO_AUTH_TOKEN) if both are configured, else
    None. Same env-var-first, then st.secrets resolution order the app's
    other deployment secrets use — see src.weather.resolve_api_key() and
    src.model_fetch.resolve_models_zip_url().
    """
    url = os.environ.get("TURSO_DATABASE_URL")
    token = os.environ.get("TURSO_AUTH_TOKEN")
    if url and token:
        return url, token

    try:
        import streamlit as st
        url = st.secrets.get("TURSO_DATABASE_URL")
        token = st.secrets.get("TURSO_AUTH_TOKEN")
        if url and token:
            return url, token
    except Exception:
        # No secrets.toml configured at all — st.secrets raises in that
        # case rather than returning empty; expected locally, not an error.
        pass

    return None


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    crop_name         TEXT    NOT NULL,
    image_path        TEXT,
    disease           TEXT,
    confidence        REAL,
    severity          TEXT,
    temperature       REAL,
    humidity          REAL,
    soil_moisture     REAL,
    rainfall          REAL,
    health_score      INTEGER,
    disease_risk      TEXT,
    environmental_risk TEXT,
    recommendation    TEXT,
    created_at        TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);
"""

DISEASE_SCHEMA = """
CREATE TABLE IF NOT EXISTS disease_analyses (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    crop_name         TEXT    NOT NULL,
    image_path        TEXT,
    disease           TEXT,
    confidence        REAL,
    severity          TEXT,
    is_healthy        INTEGER,
    recommendation    TEXT,
    created_at        TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);
"""

ENVIRONMENT_SCHEMA = """
CREATE TABLE IF NOT EXISTS environment_analyses (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    crop_name         TEXT    NOT NULL,
    temperature       REAL,
    humidity          REAL,
    soil_moisture     REAL,
    rainfall          REAL,
    risk_level        TEXT,
    probability       REAL,
    health_score      INTEGER,
    recommendation    TEXT,
    created_at        TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);
"""

FIELD_SCAN_SCHEMA = """
CREATE TABLE IF NOT EXISTS field_scans (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    crop_name           TEXT    NOT NULL,
    num_images          INTEGER,
    num_healthy         INTEGER,
    num_diseased        INTEGER,
    healthy_pct         REAL,
    dominant_disease    TEXT,
    field_health_score  INTEGER,
    severity_breakdown  TEXT,   -- JSON: {"None": 8, "Moderate": 3, ...}
    disease_breakdown   TEXT,   -- JSON: {"Healthy": 8, "Late_Blight": 3, ...}
    created_at          TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);
"""

# Columns in the order they appear in each table (minus id and created_at,
# which the database manages). Used for row <-> dict mapping and inserts.
COLUMNS = [
    "crop_name", "image_path", "disease", "confidence", "severity",
    "temperature", "humidity", "soil_moisture", "rainfall",
    "health_score", "disease_risk", "environmental_risk", "recommendation",
]

DISEASE_COLUMNS = [
    "crop_name", "image_path", "disease", "confidence", "severity",
    "is_healthy", "recommendation",
]

ENVIRONMENT_COLUMNS = [
    "crop_name", "temperature", "humidity", "soil_moisture", "rainfall",
    "risk_level", "probability", "health_score", "recommendation",
]

FIELD_SCAN_COLUMNS = [
    "crop_name", "num_images", "num_healthy", "num_diseased", "healthy_pct",
    "dominant_disease", "field_health_score", "severity_breakdown", "disease_breakdown",
]


# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------
@contextmanager
def _connect(db_path: str = DB_PATH) -> Iterator["sqlite3.Connection"]:
    """Open a database connection (Turso if configured, else local SQLite
    file), close on exit.

    Turso credentials, when present, always take priority over `db_path` —
    that parameter only matters for the local-file fallback (and for tests,
    which pass an explicit tmp path to keep runs isolated from both a real
    local file and any configured Turso database).

    Wraps any backend failure (locked file, disk full, permissions,
    corrupted database, network/auth error, ...) into a DatabaseError with
    a clean, specific message — callers never see a raw backend exception.
    Caught broadly (not sqlite3.Error specifically) because libsql raises
    plain ValueError/RuntimeError for its failures, not sqlite3's exception
    hierarchy.
    """
    creds = resolve_turso_credentials()
    try:
        if creds:
            import libsql
            url, token = creds
            conn = libsql.connect(database=url, auth_token=token)
        else:
            conn = sqlite3.connect(db_path)
    except Exception as e:
        raise DatabaseError(f"Could not open the database: {e}") from e
    try:
        yield conn
    except Exception as e:
        raise DatabaseError(f"Database operation failed: {e}") from e
    finally:
        conn.close()


def _rows_to_dicts(cursor) -> list[dict]:
    """Convert every remaining row in `cursor` to a dict, keyed by column
    name. Uses `cursor.description` (standard DB-API 2.0) rather than
    sqlite3's `Row` row_factory, so this works identically whether `cursor`
    came from a local sqlite3 connection or a remote libsql one.
    """
    cols = [c[0] for c in cursor.description]
    return [dict(zip(cols, row)) for row in cursor.fetchall()]


def _row_to_dict(cursor, row) -> dict | None:
    """Same conversion as _rows_to_dicts, for a single already-fetched row
    (e.g. from .fetchone())."""
    if row is None:
        return None
    cols = [c[0] for c in cursor.description]
    return dict(zip(cols, row))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def init_db(db_path: str = DB_PATH) -> None:
    """Create the database file and all three analysis tables (idempotent)."""
    with _connect(db_path) as conn:
        conn.execute(SCHEMA)
        conn.execute(DISEASE_SCHEMA)
        conn.execute(ENVIRONMENT_SCHEMA)
        conn.execute(FIELD_SCAN_SCHEMA)
        conn.commit()


def _insert_into(table: str, columns: list[str], data: dict, db_path: str = DB_PATH) -> int:
    """Shared insert logic for any of the three analysis tables.

    `data` may include any of `columns`; missing optional keys are stored as
    NULL. A `created_at` timestamp (ISO-8601 UTC) is added automatically if
    not supplied.
    """
    if not isinstance(data, dict) or not data:
        raise DatabaseError("Cannot save an empty or invalid analysis record.")

    init_db(db_path)  # ensure schema exists on first use

    cols = [c for c in columns if c in data]
    vals = [data[c] for c in cols]

    cols.append("created_at")
    vals.append(data.get("created_at") or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))

    placeholders = ", ".join("?" for _ in cols)
    col_list = ", ".join(cols)

    with _connect(db_path) as conn:
        cur = conn.execute(
            f"INSERT INTO {table} ({col_list}) VALUES ({placeholders})",
            vals,
        )
        conn.commit()
        return int(cur.lastrowid)


def _get_all_from(table: str, limit: int, db_path: str = DB_PATH) -> list[dict]:
    """Shared "most recent N rows" query for any of the three tables."""
    init_db(db_path)
    with _connect(db_path) as conn:
        cur = conn.execute(
            f"SELECT * FROM {table} ORDER BY id DESC LIMIT ?", (limit,)
        )
        return _rows_to_dicts(cur)


def _get_by_id_from(table: str, row_id: int, db_path: str = DB_PATH) -> dict | None:
    """Shared "fetch one row by id" query for any of the three tables."""
    init_db(db_path)
    with _connect(db_path) as conn:
        cur = conn.execute(
            f"SELECT * FROM {table} WHERE id = ?", (row_id,)
        )
        return _row_to_dict(cur, cur.fetchone())


def _delete_from(table: str, row_id: int, db_path: str = DB_PATH) -> bool:
    """Shared "delete one row by id" for any of the three tables."""
    init_db(db_path)
    with _connect(db_path) as conn:
        cur = conn.execute(f"DELETE FROM {table} WHERE id = ?", (row_id,))
        conn.commit()
        return cur.rowcount > 0


# --- Crop Health Analysis (pages/health.py) ---------------------------------
def insert_analysis(data: dict, db_path: str = DB_PATH) -> int:
    """Insert a crop health analysis and return its new row id."""
    return _insert_into("analyses", COLUMNS, data, db_path)


def get_analyses(limit: int = 100, db_path: str = DB_PATH) -> list[dict]:
    """Return the most recent crop health analyses, newest first."""
    return _get_all_from("analyses", limit, db_path)


def get_analysis_by_id(analysis_id: int, db_path: str = DB_PATH) -> dict | None:
    """Return one crop health analysis as a dict, or None if not found."""
    return _get_by_id_from("analyses", analysis_id, db_path)


def delete_analysis(analysis_id: int, db_path: str = DB_PATH) -> bool:
    """Delete a crop health analysis by id. Return True if a row was removed."""
    return _delete_from("analyses", analysis_id, db_path)


# --- Disease Detection (pages/disease.py) -----------------------------------
def insert_disease_analysis(data: dict, db_path: str = DB_PATH) -> int:
    """Insert a disease detection analysis and return its new row id."""
    return _insert_into("disease_analyses", DISEASE_COLUMNS, data, db_path)


def get_disease_analyses(limit: int = 100, db_path: str = DB_PATH) -> list[dict]:
    """Return the most recent disease detection analyses, newest first."""
    return _get_all_from("disease_analyses", limit, db_path)


def get_disease_analysis_by_id(analysis_id: int, db_path: str = DB_PATH) -> dict | None:
    """Return one disease detection analysis as a dict, or None if not found."""
    return _get_by_id_from("disease_analyses", analysis_id, db_path)


def delete_disease_analysis(analysis_id: int, db_path: str = DB_PATH) -> bool:
    """Delete a disease detection analysis by id. Return True if removed."""
    return _delete_from("disease_analyses", analysis_id, db_path)


# --- Environmental Analysis (pages/environment.py) --------------------------
def insert_environment_analysis(data: dict, db_path: str = DB_PATH) -> int:
    """Insert an environmental analysis and return its new row id."""
    return _insert_into("environment_analyses", ENVIRONMENT_COLUMNS, data, db_path)


def get_environment_analyses(limit: int = 100, db_path: str = DB_PATH) -> list[dict]:
    """Return the most recent environmental analyses, newest first."""
    return _get_all_from("environment_analyses", limit, db_path)


def get_environment_analysis_by_id(analysis_id: int, db_path: str = DB_PATH) -> dict | None:
    """Return one environmental analysis as a dict, or None if not found."""
    return _get_by_id_from("environment_analyses", analysis_id, db_path)


def delete_environment_analysis(analysis_id: int, db_path: str = DB_PATH) -> bool:
    """Delete an environmental analysis by id. Return True if removed."""
    return _delete_from("environment_analyses", analysis_id, db_path)


# --- Field Scan (pages/field_scan.py) ----------------------------------------
def insert_field_scan(data: dict, db_path: str = DB_PATH) -> int:
    """Insert an aggregated field scan and return its new row id."""
    return _insert_into("field_scans", FIELD_SCAN_COLUMNS, data, db_path)


def get_field_scans(limit: int = 100, db_path: str = DB_PATH) -> list[dict]:
    """Return the most recent field scans, newest first."""
    return _get_all_from("field_scans", limit, db_path)


def get_field_scan_by_id(scan_id: int, db_path: str = DB_PATH) -> dict | None:
    """Return one field scan as a dict, or None if not found."""
    return _get_by_id_from("field_scans", scan_id, db_path)


def delete_field_scan(scan_id: int, db_path: str = DB_PATH) -> bool:
    """Delete a field scan by id. Return True if a row was removed."""
    return _delete_from("field_scans", scan_id, db_path)