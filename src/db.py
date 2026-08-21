"""SQLite persistence layer for crop health analyses.

This module is intentionally **independent of Streamlit** so it can be unit
tested and reused from CLI scripts, notebooks, or any non-UI context. The
only third-party dependency is the Python standard library (`sqlite3`).

Schema
------
One table, `analyses`, stores every crop health analysis produced by the
Disease Detection, Environmental Analysis, and Crop Health Analysis pages.

Functions
---------
- init_db()                -> create the database and schema (idempotent)
- insert_analysis(data)    -> persist a single analysis, return its row id
- get_analyses(limit=...)  -> list recent analyses (newest first)
- get_analysis_by_id(id)   -> fetch one analysis by primary key
- delete_analysis(id)      -> remove one analysis by primary key
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

from config import DB_PATH


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

# Columns in the order they appear in the table (minus id and created_at,
# which the database manages). Used for row <-> dict mapping and inserts.
COLUMNS = [
    "crop_name", "image_path", "disease", "confidence", "severity",
    "temperature", "humidity", "soil_moisture", "rainfall",
    "health_score", "disease_risk", "environmental_risk", "recommendation",
]


# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------
@contextmanager
def _connect(db_path: str = DB_PATH) -> Iterator[sqlite3.Connection]:
    """Open a SQLite connection, enable FK + Row factory, close on exit."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row          # rows behave like dicts
    try:
        yield conn
    finally:
        conn.close()


def _exists(conn: sqlite3.Connection, analysis_id: int) -> bool:
    row = conn.execute(
        "SELECT 1 FROM analyses WHERE id = ?", (analysis_id,)
    ).fetchone()
    return row is not None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def init_db(db_path: str = DB_PATH) -> None:
    """Create the database file and the analyses table (idempotent)."""
    with _connect(db_path) as conn:
        conn.execute(SCHEMA)
        conn.commit()


def insert_analysis(data: dict, db_path: str = DB_PATH) -> int:
    """Insert a single analysis and return its new row id.

    `data` may include the keys in COLUMNS. Missing optional keys are stored
    as NULL. A `created_at` timestamp (ISO-8601 UTC) is added automatically
    if not supplied.
    """
    init_db(db_path)  # ensure schema exists on first use

    # Build column/value lists, filling only what was provided.
    cols = [c for c in COLUMNS if c in data]
    vals = [data[c] for c in cols]

    if "created_at" not in data:
        cols.append("created_at")
        vals.append(datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    else:
        cols.append("created_at")
        vals.append(data["created_at"])

    placeholders = ", ".join("?" for _ in cols)
    col_list = ", ".join(cols)

    with _connect(db_path) as conn:
        cur = conn.execute(
            f"INSERT INTO analyses ({col_list}) VALUES ({placeholders})",
            vals,
        )
        conn.commit()
        return int(cur.lastrowid)


def get_analyses(limit: int = 100, db_path: str = DB_PATH) -> list[dict]:
    """Return the most recent analyses, newest first, as a list of dicts."""
    init_db(db_path)
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM analyses ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_analysis_by_id(analysis_id: int, db_path: str = DB_PATH) -> dict | None:
    """Return one analysis as a dict, or None if the id does not exist."""
    init_db(db_path)
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM analyses WHERE id = ?", (analysis_id,)
        ).fetchone()
    return dict(row) if row else None


def delete_analysis(analysis_id: int, db_path: str = DB_PATH) -> bool:
    """Delete an analysis by id. Return True if a row was removed."""
    init_db(db_path)
    with _connect(db_path) as conn:
        cur = conn.execute(
            "DELETE FROM analyses WHERE id = ?", (analysis_id,)
        )
        conn.commit()
        return cur.rowcount > 0


# ---------------------------------------------------------------------------
# Convenience: convert a stored row into the {metric: value} shape the UI uses
# ---------------------------------------------------------------------------
def row_to_dict(row: sqlite3.Row | dict) -> dict:
    return dict(row)
