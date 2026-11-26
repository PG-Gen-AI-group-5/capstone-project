"""The desk database: one SQLite file (config `desk.db`, under data/, so it stays out of git).

Market data is not copied here: prices, features and model forecasts stay in the Parquet lake and data/derived.
This file holds what the desk produces and what people do on it:

    desks           one paper-trading account replaying a stretch of history day by day
    days            each trading day the desk has reached: the Planner's output and LLM usage
    watchlist       the stocks on the desk that day (picked by the Planner, or added by a person)
    plans           one per analysed stock and day: the Forecaster's plan, the debate and the Critic's verdict
    evidence        the records each plan could cite (announcements, forecast, indicators, levels)
    exit_decisions  the Exit Monitor's decision per open position and day
    pending_orders  a person's requests for the next session: buy an analysed stock, or exit a position
    orders          every fill (and refused entry) in a session
    trades          closed round trips
    equity          cash, positions and equity at each close
    events          the desk's feed: plans, agent notes, fills
    trace_steps     every agent step (LLM call or offline rule), as in runs/<id>/trace.jsonl
    jobs            background work started from the app (create a desk, advance days, analyse a stock)

Each morning and each session is written in one transaction, so a crash leaves the desk at the last whole step.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS desks (
    id           INTEGER PRIMARY KEY,
    name         TEXT NOT NULL UNIQUE,
    provider     TEXT NOT NULL,              -- offline | openai_compatible | anthropic
    model        TEXT,                       -- the LLM model name, for display
    start_date   TEXT NOT NULL,
    end_date     TEXT NOT NULL,              -- last trading day the desk may replay
    capital      REAL NOT NULL,
    settings     TEXT NOT NULL,              -- JSON: portfolio, costs and agent settings when created
    current_date TEXT NOT NULL,              -- the trading day on the desk
    phase        TEXT NOT NULL,              -- new | morning (plans ready, session not run) | closed | finished
    book         TEXT NOT NULL,              -- JSON engine state: cash and open positions
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS days (
    desk_id     INTEGER NOT NULL REFERENCES desks(id) ON DELETE CASCADE,
    trade_date  TEXT NOT NULL,
    as_of       TEXT NOT NULL,               -- 08:30 IST, in UTC
    planner     TEXT,                        -- JSON PlannerOutput
    llm_calls   INTEGER NOT NULL DEFAULT 0,
    llm_cost    REAL NOT NULL DEFAULT 0,
    fallbacks   INTEGER NOT NULL DEFAULT 0,  -- agent steps that fell back to the offline rules
    planned_at  TEXT,
    closed_at   TEXT,
    PRIMARY KEY (desk_id, trade_date)
);

CREATE TABLE IF NOT EXISTS watchlist (
    desk_id     INTEGER NOT NULL REFERENCES desks(id) ON DELETE CASCADE,
    trade_date  TEXT NOT NULL,
    symbol      TEXT NOT NULL,
    origin      TEXT NOT NULL,               -- planner | manual
    reasons     TEXT NOT NULL DEFAULT '[]',
    removed     INTEGER NOT NULL DEFAULT 0,  -- removed by a person: its plan is not traded
    added_at    TEXT NOT NULL,
    PRIMARY KEY (desk_id, trade_date, symbol)
);

CREATE TABLE IF NOT EXISTS plans (
    id             INTEGER PRIMARY KEY,
    desk_id        INTEGER NOT NULL REFERENCES desks(id) ON DELETE CASCADE,
    trade_date     TEXT NOT NULL,
    symbol         TEXT NOT NULL,
    setup_type     TEXT NOT NULL,
    direction      TEXT NOT NULL,            -- long | no_trade
    entry          REAL NOT NULL,            -- reference close (raw); the fill is the next open
    stop           REAL NOT NULL,
    target         REAL NOT NULL,
    max_hold_days  INTEGER NOT NULL,
    confidence     REAL NOT NULL,
    vol_forecast   REAL,
    base_p_up      REAL,
    verdict        TEXT NOT NULL,            -- accept | revise | reject
    accepted       INTEGER NOT NULL,         -- verdict accept and direction long
    rounds         INTEGER NOT NULL,
    failed_checks  TEXT NOT NULL DEFAULT '[]',
    critic_reasons TEXT NOT NULL DEFAULT '[]',
    rationale      TEXT NOT NULL,
    evidence_ids   TEXT NOT NULL DEFAULT '[]',
    bull           TEXT,                     -- JSON AnalystCase
    bear           TEXT,
    revisions      TEXT NOT NULL DEFAULT '[]',
    catalysts      TEXT NOT NULL DEFAULT '[]',
    created_at     TEXT NOT NULL,
    UNIQUE (desk_id, trade_date, symbol)
);

CREATE TABLE IF NOT EXISTS evidence (
    desk_id      INTEGER NOT NULL REFERENCES desks(id) ON DELETE CASCADE,
    trade_date   TEXT NOT NULL,
    symbol       TEXT NOT NULL,
    evidence_id  TEXT NOT NULL,
    kind         TEXT NOT NULL,
    title        TEXT NOT NULL,
    text         TEXT NOT NULL DEFAULT '',
    url          TEXT,
    published_at TEXT NOT NULL,
    important    INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (desk_id, trade_date, symbol, evidence_id)
);

CREATE TABLE IF NOT EXISTS exit_decisions (
    desk_id     INTEGER NOT NULL REFERENCES desks(id) ON DELETE CASCADE,
    trade_date  TEXT NOT NULL,
    symbol      TEXT NOT NULL,
    action      TEXT NOT NULL,               -- hold | tighten_stop | exit
    new_stop    REAL,
    reason      TEXT NOT NULL,
    PRIMARY KEY (desk_id, trade_date, symbol)
);

CREATE TABLE IF NOT EXISTS pending_orders (
    id            INTEGER PRIMARY KEY,
    desk_id       INTEGER NOT NULL REFERENCES desks(id) ON DELETE CASCADE,
    trade_date    TEXT NOT NULL,             -- the session it applies to
    symbol        TEXT NOT NULL,
    action        TEXT NOT NULL,             -- buy | exit
    plan_id       INTEGER REFERENCES plans(id) ON DELETE SET NULL,
    stop          REAL,
    target        REAL,
    max_hold_days INTEGER,
    status        TEXT NOT NULL DEFAULT 'pending',   -- pending | done
    created_at    TEXT NOT NULL,
    UNIQUE (desk_id, trade_date, symbol, action)
);

CREATE TABLE IF NOT EXISTS orders (
    id          INTEGER PRIMARY KEY,
    desk_id     INTEGER NOT NULL REFERENCES desks(id) ON DELETE CASCADE,
    trade_date  TEXT NOT NULL,
    at          TEXT NOT NULL,               -- open | intraday | close
    symbol      TEXT NOT NULL,
    side        TEXT NOT NULL,               -- buy | sell
    kind        TEXT NOT NULL,               -- entry | no_fill | exit | gap_stop | target | stop | time
    quantity    INTEGER NOT NULL,
    price       REAL,
    costs       REAL NOT NULL,
    source      TEXT NOT NULL,               -- agents | manual
    plan_id     INTEGER,
    status      TEXT NOT NULL                -- filled | no_fill
);

CREATE TABLE IF NOT EXISTS trades (
    id           INTEGER PRIMARY KEY,
    desk_id      INTEGER NOT NULL REFERENCES desks(id) ON DELETE CASCADE,
    symbol       TEXT NOT NULL,
    source       TEXT NOT NULL,
    plan_id      INTEGER,
    setup_type   TEXT NOT NULL DEFAULT 'none',
    signal_date  TEXT NOT NULL,
    entry_date   TEXT NOT NULL,
    entry_price  REAL NOT NULL,
    quantity     INTEGER NOT NULL,
    exit_date    TEXT NOT NULL,
    exit_price   REAL NOT NULL,
    exit_reason  TEXT NOT NULL,              -- stop | target | time | gap_stop | exit_monitor | manual
    gross_pnl    REAL NOT NULL,
    costs        REAL NOT NULL,
    net_pnl      REAL NOT NULL,
    return_pct   REAL NOT NULL,
    held_days    INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS equity (
    desk_id        INTEGER NOT NULL REFERENCES desks(id) ON DELETE CASCADE,
    trade_date     TEXT NOT NULL,
    cash           REAL NOT NULL,
    positions      REAL NOT NULL,
    equity         REAL NOT NULL,
    open_positions INTEGER NOT NULL,
    PRIMARY KEY (desk_id, trade_date)
);

CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY,
    desk_id     INTEGER NOT NULL REFERENCES desks(id) ON DELETE CASCADE,
    trade_date  TEXT NOT NULL,
    at          TEXT NOT NULL,               -- 08:30 | 09:15 | intraday | 15:30
    kind        TEXT NOT NULL,               -- plan | agent | trade | system
    symbol      TEXT,
    text        TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS trace_steps (
    id             INTEGER PRIMARY KEY,
    desk_id        INTEGER NOT NULL REFERENCES desks(id) ON DELETE CASCADE,
    trade_date     TEXT NOT NULL,
    step           INTEGER NOT NULL,
    at             TEXT NOT NULL,
    agent          TEXT NOT NULL,
    symbol         TEXT,
    kind           TEXT NOT NULL,
    prompt_version TEXT,
    provider       TEXT,
    input_summary  TEXT NOT NULL DEFAULT '',
    output         TEXT NOT NULL DEFAULT '{}',
    input_tokens   INTEGER NOT NULL DEFAULT 0,
    output_tokens  INTEGER NOT NULL DEFAULT 0,
    cost_usd       REAL NOT NULL DEFAULT 0,
    latency_ms     INTEGER NOT NULL DEFAULT 0,
    cached         INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS jobs (
    id          INTEGER PRIMARY KEY,
    desk_id     INTEGER REFERENCES desks(id) ON DELETE CASCADE,
    kind        TEXT NOT NULL,               -- create | advance | analyse
    params      TEXT NOT NULL DEFAULT '{}',
    status      TEXT NOT NULL,               -- queued | running | cancelling | done | failed | cancelled
    message     TEXT NOT NULL DEFAULT '',
    done        INTEGER NOT NULL DEFAULT 0,
    total       INTEGER NOT NULL DEFAULT 0,
    error       TEXT,
    created_at  TEXT NOT NULL,
    started_at  TEXT,
    finished_at TEXT
);

CREATE INDEX IF NOT EXISTS ix_plans_day    ON plans (desk_id, trade_date);
CREATE INDEX IF NOT EXISTS ix_orders_day   ON orders (desk_id, trade_date);
CREATE INDEX IF NOT EXISTS ix_trades_exit  ON trades (desk_id, exit_date);
CREATE INDEX IF NOT EXISTS ix_events_desk  ON events (desk_id, id);
CREATE INDEX IF NOT EXISTS ix_trace_day    ON trace_steps (desk_id, trade_date, symbol);
CREATE INDEX IF NOT EXISTS ix_jobs_status  ON jobs (status, id);
"""

JSON_COLUMNS = {"settings", "book", "planner", "reasons", "failed_checks", "critic_reasons", "evidence_ids", "bull",
                "bear", "revisions", "catalysts", "output", "params"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def dumps(value) -> str:
    return json.dumps(value, default=str)


class Store:
    """Thin access layer: a new connection per call site (SQLite in WAL mode handles one writer and many readers)."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.Lock()
        with self.connect() as con:
            con.executescript(SCHEMA)
            con.execute("INSERT OR IGNORE INTO meta VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))

    def connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA foreign_keys=ON")
        con.execute("PRAGMA busy_timeout=30000")
        return con

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        """One write transaction (writers are serialised in-process, so they never wait on each other's locks)."""
        with self._write_lock:
            con = self.connect()
            try:
                con.execute("BEGIN IMMEDIATE")
                yield con
                con.execute("COMMIT")
            except BaseException:
                con.execute("ROLLBACK")
                raise
            finally:
                con.close()

    # ---- reads
    def rows(self, sql: str, params=()) -> list[dict]:
        con = self.connect()
        try:
            return [decode(r) for r in con.execute(sql, params).fetchall()]
        finally:
            con.close()

    def row(self, sql: str, params=()) -> dict | None:
        out = self.rows(sql, params)
        return out[0] if out else None

    def scalar(self, sql: str, params=()):
        con = self.connect()
        try:
            r = con.execute(sql, params).fetchone()
            return r[0] if r else None
        finally:
            con.close()

    # ---- writes outside a bigger transaction
    def execute(self, sql: str, params=()) -> int:
        with self.tx() as con:
            return con.execute(sql, params).lastrowid


def decode(row: sqlite3.Row) -> dict:
    out = dict(row)
    for k, v in out.items():
        if k in JSON_COLUMNS and isinstance(v, str):
            out[k] = json.loads(v)
    return out


def insert(con: sqlite3.Connection, table: str, values: dict, *, replace: bool = False) -> int:
    cols = list(values)
    data = [dumps(v) if k in JSON_COLUMNS and not isinstance(v, str) else v for k, v in values.items()]
    verb = "INSERT OR REPLACE" if replace else "INSERT"
    sql = f"{verb} INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})"
    return con.execute(sql, data).lastrowid


def update(con: sqlite3.Connection, table: str, values: dict, where: str, params=()) -> None:
    sets = ", ".join(f"{k} = ?" for k in values)
    data = [dumps(v) if k in JSON_COLUMNS and not isinstance(v, str) else v for k, v in values.items()]
    con.execute(f"UPDATE {table} SET {sets} WHERE {where}", [*data, *params])
