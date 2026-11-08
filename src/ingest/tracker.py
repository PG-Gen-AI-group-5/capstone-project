"""Download tracker (SQLite): what to download, what happened on every try, and what needs a person.

Statuses of a task:
    pending -> running -> success | no_data
                       -> retry_wait   (retryable error, attempts left; picked up again later)
                       -> needs_review (retries used up, an error retrying can't fix,
                                        or a blocking data-quality problem)
    needs_review -> (person decides) -> pending | verified_gap | success (fallback) | skipped

An error never stops a run: the task gets its status and the run moves on.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

from src.ingest.errors import ERROR_TYPES
from src.timeutil import iso, now_utc

SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
    source          TEXT PRIMARY KEY,
    task_unit       TEXT NOT NULL,
    max_attempts    INTEGER NOT NULL DEFAULT 3,
    retry_base_sec  INTEGER NOT NULL DEFAULT 300,
    description     TEXT
);

CREATE TABLE IF NOT EXISTS error_types (
    error_type   TEXT PRIMARY KEY,
    retryable    INTEGER NOT NULL,
    description  TEXT
);

CREATE TABLE IF NOT EXISTS runs (
    run_id        INTEGER PRIMARY KEY,
    source        TEXT NOT NULL REFERENCES sources(source),
    triggered_by  TEXT NOT NULL,
    started_at    TEXT NOT NULL,
    finished_at   TEXT,
    status        TEXT NOT NULL DEFAULT 'running',
    machine       TEXT,
    code_version  TEXT,
    n_success     INTEGER NOT NULL DEFAULT 0,
    n_no_data     INTEGER NOT NULL DEFAULT 0,
    n_failed      INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS tasks (
    task_id          INTEGER PRIMARY KEY,
    source           TEXT NOT NULL REFERENCES sources(source),
    task_key         TEXT NOT NULL,
    symbol           TEXT NOT NULL DEFAULT '*',
    period_start     TEXT NOT NULL,
    period_end       TEXT NOT NULL,
    url              TEXT,
    expected_bytes   INTEGER,
    expected_md5     TEXT,
    stage            TEXT NOT NULL DEFAULT 'download',
    status           TEXT NOT NULL DEFAULT 'pending',
    attempts         INTEGER NOT NULL DEFAULT 0,
    next_retry_at    TEXT,
    last_attempt_id  INTEGER,
    rows_saved       INTEGER,
    raw_path         TEXT,
    fallback_source  TEXT,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL,
    UNIQUE (source, task_key)
);
CREATE INDEX IF NOT EXISTS idx_tasks_due ON tasks (source, status, next_retry_at);
CREATE INDEX IF NOT EXISTS idx_tasks_period ON tasks (source, period_start);

CREATE TABLE IF NOT EXISTS attempts (
    attempt_id     INTEGER PRIMARY KEY,
    task_id        INTEGER NOT NULL REFERENCES tasks(task_id),
    run_id         INTEGER REFERENCES runs(run_id),
    stage          TEXT NOT NULL,
    started_at     TEXT NOT NULL,
    finished_at    TEXT,
    outcome        TEXT NOT NULL,
    http_status    INTEGER,
    error_type     TEXT REFERENCES error_types(error_type),
    error_message  TEXT,
    bytes          INTEGER,
    rows           INTEGER,
    duration_ms    INTEGER
);
CREATE INDEX IF NOT EXISTS idx_attempts_task ON attempts (task_id);

CREATE TABLE IF NOT EXISTS review_items (
    review_id    INTEGER PRIMARY KEY,
    task_id      INTEGER NOT NULL REFERENCES tasks(task_id),
    reason       TEXT NOT NULL,
    check_name   TEXT,
    severity     TEXT NOT NULL,
    details      TEXT,
    status       TEXT NOT NULL DEFAULT 'open',
    decision     TEXT,
    reviewed_by  TEXT,
    reviewed_at  TEXT,
    note         TEXT,
    created_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_review_open ON review_items (status, task_id);

CREATE TABLE IF NOT EXISTS trading_calendar (
    trade_date  TEXT PRIMARY KEY,
    is_trading  INTEGER NOT NULL,
    note        TEXT
);

CREATE TABLE IF NOT EXISTS coverage (
    symbol      TEXT NOT NULL,
    trade_date  TEXT NOT NULL,
    source      TEXT NOT NULL,
    n_records   INTEGER NOT NULL,
    status      TEXT NOT NULL,
    PRIMARY KEY (symbol, trade_date, source)
);
"""

DECISIONS = ("retry", "fixed_parser", "accept_gap", "use_fallback", "ignore", "acknowledge")


def _period(value) -> str:
    if isinstance(value, datetime):
        return iso(value)
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


@dataclass
class TaskSpec:
    """A unit of work produced by a source's plan()."""
    task_key: str
    period_start: object
    period_end: object
    symbol: str = "*"
    url: str | None = None
    expected_bytes: int | None = None
    expected_md5: str | None = None


@dataclass
class Task:
    task_id: int
    source: str
    task_key: str
    symbol: str
    period_start: str
    period_end: str
    url: str | None
    expected_bytes: int | None
    expected_md5: str | None
    stage: str
    status: str
    attempts: int
    raw_path: str | None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Task:
        return cls(**{k: row[k] for k in cls.__dataclass_fields__})

    @property
    def start_date(self) -> date:
        return date.fromisoformat(self.period_start[:10])

    @property
    def end_date(self) -> date:
        return date.fromisoformat(self.period_end[:10])


class Tracker:
    def __init__(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    # ---------- setup ----------
    def register_sources(self, sources: list[dict]) -> None:
        with self.conn:
            self.conn.executemany(
                "INSERT INTO sources (source, task_unit, max_attempts, retry_base_sec, description) "
                "VALUES (:source, :task_unit, :max_attempts, :retry_base_sec, :description) "
                "ON CONFLICT(source) DO UPDATE SET task_unit=excluded.task_unit, "
                "max_attempts=excluded.max_attempts, retry_base_sec=excluded.retry_base_sec, "
                "description=excluded.description",
                sources,
            )
            self.conn.executemany(
                "INSERT INTO error_types (error_type, retryable, description) VALUES (?, ?, ?) "
                "ON CONFLICT(error_type) DO UPDATE SET retryable=excluded.retryable, "
                "description=excluded.description",
                [(k, int(v[0]), v[1]) for k, v in ERROR_TYPES.items()],
            )

    def source_settings(self, source: str) -> sqlite3.Row:
        row = self.conn.execute("SELECT * FROM sources WHERE source = ?", (source,)).fetchone()
        if row is None:
            raise KeyError(f"Source {source!r} is not registered; run `python -m src.ingest init` first")
        return row

    # ---------- trading calendar ----------
    def load_calendar(self, rows: list[tuple[date, bool, str | None]]) -> None:
        with self.conn:
            self.conn.executemany(
                "INSERT OR REPLACE INTO trading_calendar (trade_date, is_trading, note) VALUES (?, ?, ?)",
                [(d.isoformat(), int(t), note) for d, t, note in rows],
            )

    def is_trading(self, d: date) -> bool | None:
        row = self.conn.execute(
            "SELECT is_trading FROM trading_calendar WHERE trade_date = ?", (d.isoformat(),)).fetchone()
        return None if row is None else bool(row["is_trading"])

    def trading_days(self, start: date, end: date) -> list[date]:
        rows = self.conn.execute(
            "SELECT trade_date FROM trading_calendar WHERE is_trading = 1 AND trade_date BETWEEN ? AND ? "
            "ORDER BY trade_date", (start.isoformat(), end.isoformat())).fetchall()
        return [date.fromisoformat(r["trade_date"]) for r in rows]

    # ---------- planning ----------
    def plan(self, source: str, specs: list[TaskSpec], refresh: bool = False) -> tuple[int, int]:
        """Create tasks (existing ones are left alone). refresh=True resets existing ones to pending.

        Returns (new tasks, refreshed tasks).
        """
        now = iso(now_utc())
        rows = [(source, s.task_key, s.symbol, _period(s.period_start), _period(s.period_end), s.url,
                 s.expected_bytes, s.expected_md5, now, now) for s in specs]
        with self.conn:
            refreshed = 0
            if refresh:  # reset existing tasks first, so tasks created below aren't counted as refreshed
                before = self.conn.total_changes
                self.conn.executemany(
                    "UPDATE tasks SET stage='download', status='pending', attempts=0, next_retry_at=NULL, "
                    "period_start=?, period_end=?, url=?, expected_bytes=?, expected_md5=?, updated_at=? "
                    "WHERE source=? AND task_key=? AND status != 'running'",
                    [(_period(s.period_start), _period(s.period_end), s.url, s.expected_bytes, s.expected_md5,
                      now, source, s.task_key) for s in specs])
                refreshed = self.conn.total_changes - before
            before = self.conn.total_changes
            self.conn.executemany(
                "INSERT OR IGNORE INTO tasks (source, task_key, symbol, period_start, period_end, url, "
                "expected_bytes, expected_md5, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
            inserted = self.conn.total_changes - before
        return inserted, refreshed

    # ---------- running ----------
    def start_run(self, source: str, triggered_by: str, machine: str | None = None,
                  code_version: str | None = None) -> int:
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO runs (source, triggered_by, started_at, machine, code_version) VALUES (?,?,?,?,?)",
                (source, triggered_by, iso(now_utc()), machine, code_version))
        return cur.lastrowid

    def finish_run(self, run_id: int, status: str, counts: dict[str, int]) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE runs SET finished_at=?, status=?, n_success=?, n_no_data=?, n_failed=? WHERE run_id=?",
                (iso(now_utc()), status, counts.get("success", 0), counts.get("no_data", 0),
                 counts.get("failed", 0), run_id))

    def reset_stale(self, source: str) -> int:
        """Tasks left 'running' by a crashed run go back to 'pending'."""
        with self.conn:
            cur = self.conn.execute(
                "UPDATE tasks SET status='pending', updated_at=? WHERE source=? AND status='running'",
                (iso(now_utc()), source))
        return cur.rowcount

    def due_tasks(self, source: str, limit: int = 500, symbols: list[str] | None = None,
                  exclude: set[int] | None = None) -> list[Task]:
        sql = ("SELECT * FROM tasks WHERE source=? AND (status='pending' OR "
               "(status='retry_wait' AND next_retry_at <= ?))")
        params: list = [source, iso(now_utc())]
        if symbols:
            sql += f" AND (symbol='*' OR symbol IN ({','.join('?' * len(symbols))}))"
            params += symbols
        if exclude:
            sql += f" AND task_id NOT IN ({','.join(str(i) for i in exclude)})"
        sql += " ORDER BY period_start, task_id LIMIT ?"
        params.append(limit)
        return [Task.from_row(r) for r in self.conn.execute(sql, params).fetchall()]

    def next_retry_at(self, source: str) -> str | None:
        row = self.conn.execute(
            "SELECT MIN(next_retry_at) AS t FROM tasks WHERE source=? AND status='retry_wait'", (source,)).fetchone()
        return row["t"]

    def get_task(self, task_id: int) -> Task:
        row = self.conn.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
        if row is None:
            raise KeyError(f"No task {task_id}")
        return Task.from_row(row)

    def mark_running(self, task_id: int) -> None:
        with self.conn:
            self.conn.execute("UPDATE tasks SET status='running', updated_at=? WHERE task_id=?",
                              (iso(now_utc()), task_id))

    def add_attempt(self, task_id: int, run_id: int | None, stage: str, started_at: datetime, outcome: str,
                    http_status: int | None = None, error_type: str | None = None,
                    error_message: str | None = None, nbytes: int | None = None,
                    rows: int | None = None) -> int:
        finished = now_utc()
        duration = int((finished - started_at).total_seconds() * 1000)
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO attempts (task_id, run_id, stage, started_at, finished_at, outcome, http_status, "
                "error_type, error_message, bytes, rows, duration_ms) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (task_id, run_id, stage, iso(started_at), iso(finished), outcome, http_status, error_type,
                 (error_message or "")[:4000] or None, nbytes, rows, duration))
            self.conn.execute("UPDATE tasks SET last_attempt_id=? WHERE task_id=?", (cur.lastrowid, task_id))
        return cur.lastrowid

    def advance(self, task_id: int, stage: str, raw_path: str | None) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE tasks SET stage=?, raw_path=COALESCE(?, raw_path), attempts=0, next_retry_at=NULL, "
                "updated_at=? WHERE task_id=?", (stage, raw_path, iso(now_utc()), task_id))

    def complete(self, task_id: int, status: str, rows: int | None = None, raw_path: str | None = None,
                 fallback_source: str | None = None) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE tasks SET stage='done', status=?, rows_saved=?, raw_path=COALESCE(?, raw_path), "
                "fallback_source=COALESCE(?, fallback_source), attempts=0, next_retry_at=NULL, updated_at=? "
                "WHERE task_id=?", (status, rows, raw_path, fallback_source, iso(now_utc()), task_id))

    def fail(self, task_id: int, error_type: str, create_review: bool = True) -> str:
        """Record a failed attempt's effect on the task. Returns the new status."""
        task = self.get_task(task_id)
        settings = self.source_settings(task.source)
        attempts = task.attempts + 1
        retryable = ERROR_TYPES[error_type][0]
        if retryable and attempts < settings["max_attempts"]:
            wait = settings["retry_base_sec"] * 2 ** (attempts - 1)
            status, next_retry = "retry_wait", iso(now_utc() + timedelta(seconds=wait))
        else:
            status, next_retry = "needs_review", None
        with self.conn:
            self.conn.execute(
                "UPDATE tasks SET status=?, attempts=?, next_retry_at=?, updated_at=? WHERE task_id=?",
                (status, attempts, next_retry, iso(now_utc()), task_id))
        if status == "needs_review" and create_review:
            self.add_review(task_id, "max_attempts" if retryable else "non_retryable_error", "error",
                            check_name=error_type)
        return status

    # ---------- review ----------
    def add_review(self, task_id: int, reason: str, severity: str, check_name: str | None = None,
                   details: dict | None = None) -> int | None:
        details_json = json.dumps(details, default=str) if details else None
        existing = self.conn.execute(
            "SELECT review_id FROM review_items WHERE task_id=? AND status='open' AND reason=? "
            "AND COALESCE(check_name,'')=COALESCE(?,'') AND COALESCE(details,'')=COALESCE(?,'')",
            (task_id, reason, check_name, details_json)).fetchone()
        if existing:
            return None
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO review_items (task_id, reason, check_name, severity, details, created_at) "
                "VALUES (?,?,?,?,?,?)", (task_id, reason, check_name, severity, details_json, iso(now_utc())))
        return cur.lastrowid

    def review_list(self, source: str | None = None) -> list[sqlite3.Row]:
        sql = """
            SELECT t.task_id, t.source, t.symbol, t.task_key, t.stage, t.status, t.attempts,
                   a.error_type, a.error_message, r.review_id, r.reason, r.check_name, r.severity, r.details
            FROM review_items r
            JOIN tasks t ON t.task_id = r.task_id
            LEFT JOIN attempts a ON a.attempt_id = t.last_attempt_id
            WHERE r.status = 'open'"""
        params: list = []
        if source:
            sql += " AND t.source = ?"
            params.append(source)
        sql += " ORDER BY (t.status = 'needs_review') DESC, t.source, t.period_start, r.review_id"
        return self.conn.execute(sql, params).fetchall()

    def resolve(self, task_id: int, decision: str, note: str | None = None, by: str | None = None,
                fallback_source: str | None = None) -> Task:
        if decision not in DECISIONS:
            raise ValueError(f"decision must be one of {DECISIONS}")
        task = self.get_task(task_id)
        now = iso(now_utc())
        if decision == "acknowledge" and task.status == "needs_review":
            raise ValueError("This task is blocked; choose retry, fixed_parser, accept_gap, use_fallback or ignore")
        if decision == "fixed_parser" and not task.raw_path:
            raise ValueError("No saved raw file for this task; use retry to download again")
        updates = {
            "retry": ("stage", "pending"),
            "fixed_parser": ("parse", "pending"),
            "accept_gap": ("done", "verified_gap"),
            "use_fallback": ("done", "success"),
            "ignore": ("done", "skipped"),
        }
        with self.conn:
            if decision in updates:
                stage, status = updates[decision]
                self.conn.execute(
                    "UPDATE tasks SET stage = CASE WHEN ?='stage' THEN stage ELSE ? END, status=?, attempts=0, "
                    "next_retry_at=NULL, fallback_source=COALESCE(?, fallback_source), updated_at=? "
                    "WHERE task_id=?", (stage, stage, status, fallback_source, now, task_id))
            self.conn.execute(
                "UPDATE review_items SET status='resolved', decision=?, reviewed_by=?, reviewed_at=?, note=? "
                "WHERE task_id=? AND status='open'", (decision, by, now, note, task_id))
        return self.get_task(task_id)

    def reparse(self, source: str, start: str | None = None, end: str | None = None) -> int:
        """Re-run only the parse step (from saved raw files), e.g. after fixing a parser."""
        with self.conn:
            cur = self.conn.execute(
                "UPDATE tasks SET stage='parse', status='pending', attempts=0, next_retry_at=NULL, updated_at=? "
                "WHERE source=? AND raw_path IS NOT NULL AND status IN ('success', 'needs_review') "
                "AND period_start >= ? AND period_start <= ?",
                (iso(now_utc()), source, start or "0000", end or "9999"))
        return cur.rowcount

    # ---------- reporting ----------
    def summary(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT source, status, COUNT(*) AS n FROM tasks GROUP BY source, status ORDER BY source, status"
        ).fetchall()

    def tasks_with_status(self, source: str, statuses: tuple[str, ...]) -> list[Task]:
        rows = self.conn.execute(
            f"SELECT * FROM tasks WHERE source=? AND status IN ({','.join('?' * len(statuses))}) "
            "ORDER BY period_start", (source, *statuses)).fetchall()
        return [Task.from_row(r) for r in rows]

    # ---------- coverage ----------
    def replace_coverage(self, source: str, start: date, end: date, rows: list[tuple]) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM coverage WHERE source=? AND trade_date BETWEEN ? AND ?",
                              (source, start.isoformat(), end.isoformat()))
            self.conn.executemany(
                "INSERT INTO coverage (symbol, trade_date, source, n_records, status) VALUES (?,?,?,?,?)", rows)
