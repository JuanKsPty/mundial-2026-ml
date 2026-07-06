"""SQLite helpers for the WC2026 match store (data/database.db).

Matches are keyed by (date, home_team, away_team) so the same real match
coming from different sources lands on the same row. Higher-priority
sources overwrite lower-priority ones (manual > api > martj42), never the
other way around - except that a row without scores is always completable.
"""

import sqlite3
from datetime import datetime, timezone

import pandas as pd

from src.config import DB_PATH

SOURCE_PRIORITY = {"manual": 3, "api": 2, "martj42": 1}

SCHEMA = """
CREATE TABLE IF NOT EXISTS matches (
    date            TEXT NOT NULL,           -- YYYY-MM-DD
    home_team       TEXT NOT NULL,           -- normalized name
    away_team       TEXT NOT NULL,
    round           TEXT,                    -- group | round_of_32 | round_of_16 |
                                             -- quarterfinal | semifinal | third_place | final
    stage           TEXT,                    -- group | knockout
    group_name      TEXT,                    -- 'A'..'L' or NULL
    home_goals      INTEGER,                 -- incl. extra time; NULL if not played
    away_goals      INTEGER,
    penalty_winner  TEXT,                    -- NULL if no shootout
    status          TEXT NOT NULL,           -- NS | FT | AET | PEN
    fixture_id      INTEGER,                 -- API-Football id when known
    source          TEXT NOT NULL,           -- api | martj42 | manual
    updated_at      TEXT NOT NULL,
    PRIMARY KEY (date, home_team, away_team)
);

CREATE TABLE IF NOT EXISTS sync_log (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    synced_at  TEXT NOT NULL,
    source     TEXT NOT NULL,
    n_matches  INTEGER NOT NULL,
    note       TEXT
);
"""

MATCH_COLUMNS = [
    "date", "home_team", "away_team", "round", "stage", "group_name",
    "home_goals", "away_goals", "penalty_winner", "status",
    "fixture_id", "source", "updated_at",
]


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection | None = None) -> None:
    own = conn is None
    conn = conn or get_connection()
    conn.executescript(SCHEMA)
    conn.commit()
    if own:
        conn.close()


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def upsert_matches(rows: list[dict], source: str, conn: sqlite3.Connection | None = None) -> int:
    """Insert/update match rows, respecting source priority. Returns rows written."""
    if source not in SOURCE_PRIORITY:
        raise ValueError(f"unknown source: {source}")
    own = conn is None
    conn = conn or get_connection()
    init_db(conn)

    written = 0
    for row in rows:
        key = (row["date"], row["home_team"], row["away_team"])
        existing = conn.execute(
            "SELECT source, home_goals, away_goals FROM matches "
            "WHERE date=? AND home_team=? AND away_team=?", key,
        ).fetchone()
        if existing is not None:
            same_or_higher = SOURCE_PRIORITY[source] >= SOURCE_PRIORITY[existing["source"]]
            completes_scores = existing["home_goals"] is None and row.get("home_goals") is not None
            if not (same_or_higher or completes_scores):
                continue
        record = {c: row.get(c) for c in MATCH_COLUMNS}
        record["source"] = source
        record["updated_at"] = _now()
        conn.execute(
            f"INSERT OR REPLACE INTO matches ({', '.join(MATCH_COLUMNS)}) "
            f"VALUES ({', '.join(':' + c for c in MATCH_COLUMNS)})",
            record,
        )
        written += 1

    conn.commit()
    if own:
        conn.close()
    return written


def log_sync(source: str, n_matches: int, note: str = "", conn: sqlite3.Connection | None = None) -> None:
    own = conn is None
    conn = conn or get_connection()
    init_db(conn)
    conn.execute(
        "INSERT INTO sync_log (synced_at, source, n_matches, note) VALUES (?, ?, ?, ?)",
        (_now(), source, n_matches, note),
    )
    conn.commit()
    if own:
        conn.close()


def load_matches(stage: str | None = None, conn: sqlite3.Connection | None = None) -> pd.DataFrame:
    """All WC2026 matches (played and scheduled) as a DataFrame."""
    own = conn is None
    conn = conn or get_connection()
    init_db(conn)
    query = "SELECT * FROM matches"
    params: tuple = ()
    if stage is not None:
        query += " WHERE stage = ?"
        params = (stage,)
    query += " ORDER BY date, home_team"
    df = pd.read_sql_query(query, conn, params=params)
    if own:
        conn.close()
    return df


def last_sync(conn: sqlite3.Connection | None = None) -> dict | None:
    own = conn is None
    conn = conn or get_connection()
    init_db(conn)
    row = conn.execute(
        "SELECT synced_at, source, n_matches, note FROM sync_log ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if own:
        conn.close()
    return dict(row) if row else None
