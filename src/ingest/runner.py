"""Runs tasks through download -> parse -> validate, applying the tracker rules.

Rules (docs/DATA_COLLECTION.md, section T):
1. An error never stops the run: record the attempt, set the status, move on.
2. Retryable errors wait 5, 10, 20 minutes; after the last attempt the task goes to review.
3. Errors that retrying can't fix go straight to review.
4. Blocking quality issues keep the data out of the lake and send the task to review;
   other issues are written to the review list while the task succeeds.
"""
from __future__ import annotations

import platform
import subprocess
import time
import traceback
from collections import Counter
from collections.abc import Callable
from datetime import date
from pathlib import Path

from src.config import Config
from src.ingest import sources as registry
from src.ingest.errors import IngestError
from src.ingest.http import Http
from src.ingest.lake import Lake
from src.ingest.sources.base import Context, Frame, Source
from src.ingest.tracker import Task, Tracker
from src.timeutil import from_iso, now_utc


def make_context(cfg: Config) -> Context:
    return Context(cfg=cfg, tracker=Tracker(cfg.tracker_db), lake=Lake(cfg.lake_dir), http=Http(cfg))


def source_registrations(cfg: Config) -> list[dict]:
    t = cfg["tracker"]
    return [{"source": s.name, "task_unit": s.task_unit, "max_attempts": s.max_attempts or t["max_attempts"],
             "retry_base_sec": s.retry_base_sec or t["retry_base_sec"], "description": s.description}
            for s in registry.ALL]


def git_sha(root: Path) -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True,
                             timeout=5, check=False)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def plan_source(ctx: Context, source: Source, start: date | None, end: date, symbols: list[str] | None,
                refresh: bool = False) -> tuple[int, int, int]:
    specs = source.plan(ctx, start or source.default_start(ctx), end, symbols)
    new, refreshed = ctx.tracker.plan(source.name, specs, refresh=refresh)
    return len(specs), new, refreshed


def write_frames(ctx: Context, frames: list[Frame]) -> int:
    rows = 0
    stamp = now_utc()
    for f in frames:
        df = f.df.copy()
        df["ingested_at"] = stamp
        rows += ctx.lake.upsert(f.table, df, f.key, f.partition, {**f.schema, "ingested_at": "ts"})
    return rows


def _failed(ctx: Context, task: Task, run_id: int | None, stage: str, started, exc: Exception) -> str:
    if isinstance(exc, IngestError):
        err = exc
    else:
        error_type = "unexpected_error" if stage == "download" else "parse_error"
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__, limit=4))
        err = IngestError(error_type, f"{type(exc).__name__}: {exc}\n{tb}")
    ctx.tracker.add_attempt(task.task_id, run_id, stage, started, "error", err.http_status, err.error_type,
                            err.message)
    ctx.tracker.fail(task.task_id, err.error_type)
    return "failed"


def process_task(ctx: Context, source: Source, task: Task, run_id: int | None) -> str:
    """Run one task as far as it can go. Returns 'success', 'no_data' or 'failed'."""
    tr = ctx.tracker
    tr.mark_running(task.task_id)
    raw = Path(task.raw_path) if task.raw_path else None

    if task.stage == "download":
        started = now_utc()
        try:
            result = source.download(ctx, task)
        except Exception as e:  # noqa: BLE001 - any failure is recorded, never fatal to the run
            return _failed(ctx, task, run_id, "download", started, e)
        if result.no_data:
            tr.add_attempt(task.task_id, run_id, "download", started, "no_data", nbytes=result.nbytes)
            tr.complete(task.task_id, "no_data", rows=0)
            return "no_data"
        tr.add_attempt(task.task_id, run_id, "download", started, "success", nbytes=result.nbytes,
                       rows=result.rows)
        raw = result.raw_path
        tr.advance(task.task_id, "parse", str(raw) if raw else None)

    started = now_utc()
    try:
        parsed = source.parse(ctx, task, raw)
    except Exception as e:  # noqa: BLE001
        return _failed(ctx, task, run_id, "parse", started, e)

    blocking = [i for i in parsed.issues if i.blocking]
    if blocking:
        tr.add_attempt(task.task_id, run_id, "validate", started, "error", error_type="validation_failed",
                       error_message="; ".join(i.check_name for i in blocking))
        tr.fail(task.task_id, "validation_failed", create_review=False)
        for issue in parsed.issues:
            tr.add_review(task.task_id, "quality_check", issue.severity, issue.check_name, issue.details)
        return "failed"

    try:
        rows = write_frames(ctx, parsed.frames)
    except Exception as e:  # noqa: BLE001
        return _failed(ctx, task, run_id, "parse", started, e)
    tr.add_attempt(task.task_id, run_id, "parse", started, "success", rows=rows)
    for issue in parsed.issues:
        tr.add_review(task.task_id, "quality_check", issue.severity, issue.check_name, issue.details)
    tr.complete(task.task_id, "success", rows=rows, raw_path=str(raw) if raw else None)
    return "success"


def run_source(ctx: Context, source: Source, triggered_by: str = "backfill", limit: int | None = None,
               symbols: list[str] | None = None, wait_retries: bool = False,
               log: Callable[[str], None] = print) -> Counter:
    tr = ctx.tracker
    stale = tr.reset_stale(source.name)
    if stale:
        log(f"  {stale} task(s) left running by an earlier crash were reset to pending")
    run_id = tr.start_run(source.name, triggered_by, platform.node(), git_sha(ctx.cfg.root))
    counts: Counter = Counter()
    status = "completed"
    processed = 0
    try:
        while limit is None or processed < limit:
            batch = tr.due_tasks(source.name, limit=min(200, (limit - processed) if limit else 200),
                                 symbols=symbols)
            if not batch:
                next_retry = tr.next_retry_at(source.name)
                if wait_retries and next_retry:
                    wait = max(0.0, (from_iso(next_retry) - now_utc()).total_seconds()) + 1
                    log(f"  waiting {int(wait)}s for tasks to retry...")
                    time.sleep(wait)
                    continue
                break
            for task in batch:
                outcome = process_task(ctx, source, task, run_id)
                counts[outcome] += 1
                processed += 1
                if outcome == "failed":
                    new = tr.get_task(task.task_id)
                    log(f"  {task.task_key}: {new.status} ({_last_error(ctx, task.task_id)})")
                elif processed % 50 == 0:
                    log(f"  ...{processed} done ({dict(counts)})")
                if limit and processed >= limit:
                    break
    except KeyboardInterrupt:
        status = "crashed"
        raise
    finally:
        tr.finish_run(run_id, status, counts)
    waiting = tr.next_retry_at(source.name)
    if waiting:
        log(f"  Some tasks are waiting to retry (next at {waiting} UTC). Run again later, or use --wait-retries.")
    return counts


def _last_error(ctx: Context, task_id: int) -> str:
    row = ctx.tracker.conn.execute(
        "SELECT a.error_type, a.error_message FROM attempts a JOIN tasks t ON t.last_attempt_id = a.attempt_id "
        "WHERE t.task_id = ?", (task_id,)).fetchone()
    if not row:
        return ""
    return f"{row['error_type']}: {(row['error_message'] or '').splitlines()[0][:120]}"


def resolve_task(ctx: Context, task_id: int, decision: str, note: str | None, by: str | None) -> Task:
    if decision != "use_fallback":
        return ctx.tracker.resolve(task_id, decision, note, by)
    task = ctx.tracker.get_task(task_id)
    source = registry.get(task.source)
    parsed = source.fallback(ctx, task)
    rows = write_frames(ctx, parsed.frames)
    resolved = ctx.tracker.resolve(task_id, "use_fallback", note, by,
                                   fallback_source=getattr(source, "fallback_name", "fallback"))
    ctx.tracker.complete(task_id, "success", rows=rows)
    return resolved
