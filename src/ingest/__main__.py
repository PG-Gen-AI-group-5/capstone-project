"""Command line for data collection.

    python -m src.ingest init                                   # tracker database + trading calendar
    python -m src.ingest universe                               # data/manual/universe.csv from NSE lists
    python -m src.ingest pilot                                  # every source for 5 stocks, small ranges
    python -m src.ingest plan --source nse_bhavcopy --from 2016-01-01
    python -m src.ingest run  --source nse_bhavcopy
    python -m src.ingest status | review | coverage | views
    python -m src.ingest resolve 123 --decision accept_gap --note "NSE file missing; checked website"
    python -m src.ingest daily                                  # evening update job
"""
from __future__ import annotations

import argparse
import getpass
import json
import sys
from collections import defaultdict
from datetime import date, timedelta

from src.config import Config
from src.ingest import sources as registry
from src.ingest.calendar import build_calendar
from src.ingest.coverage import build_coverage, coverage_table
from src.ingest.runner import make_context, plan_source, resolve_task, run_source, source_registrations
from src.ingest.tracker import DECISIONS
from src.ingest.universe import build_universe
from src.ingest.views import create_views
from src.timeutil import today_ist

PILOT_SYMBOLS = ["RELIANCE", "HDFCBANK", "INFY", "TCS", "LT"]


def print_table(rows: list[tuple], headers: tuple[str, ...]) -> None:
    rows = [tuple("" if v is None else str(v) for v in r) for r in rows]
    widths = [max([len(h)] + [len(r[i]) for r in rows]) for i, h in enumerate(headers)]
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print("  ".join(v.ljust(w) for v, w in zip(r, widths)))


def symbols_arg(value: str | None) -> list[str] | None:
    return [s.strip().upper() for s in value.split(",") if s.strip()] if value else None


def ensure_ready(ctx, build_calendar_if_empty: bool = True) -> None:
    ctx.tracker.register_sources(source_registrations(ctx.cfg))
    empty = ctx.tracker.conn.execute("SELECT COUNT(*) FROM trading_calendar").fetchone()[0] == 0
    if empty and build_calendar_if_empty:
        load_calendar(ctx)


def load_calendar(ctx) -> int:
    end = date(today_ist().year + 1, 12, 31)
    rows = build_calendar(date(2015, 1, 1), end, ctx.cfg.manual_dir / "calendar_overrides.csv")
    ctx.tracker.load_calendar(rows)
    return sum(1 for _, t, _ in rows if t)


def plan_and_run(ctx, name: str, start: date | None, end: date, symbols=None, refresh=False, limit=None,
                 triggered_by="manual") -> None:
    source = registry.get(name)
    print(f"\n== {name}: {source.description}")
    total, new, refreshed = plan_source(ctx, source, start, end, symbols, refresh)
    print(f"  planned {total} task(s): {new} new, {refreshed} reset to pending")
    counts = run_source(ctx, source, triggered_by=triggered_by, limit=limit, symbols=symbols)
    print(f"  result: {dict(counts) or 'nothing due'}")


def cmd_init(ctx, args):
    ensure_ready(ctx, build_calendar_if_empty=False)
    n = load_calendar(ctx)
    print(f"Tracker ready at {ctx.cfg.tracker_db}")
    print(f"Registered {len(registry.ALL)} sources; trading calendar loaded ({n} trading days 2015 onward).")


def cmd_universe(ctx, args):
    df = build_universe(ctx.http, ctx.cfg.manual_dir / "universe.csv")
    print(f"Wrote {ctx.cfg.manual_dir / 'universe.csv'}: {len(df)} stocks "
          f"({(df['tier'] == 'nifty50').sum()} Nifty 50, {(df['tier'] == 'nifty200').sum()} more in the Nifty 200).")
    print("Next: check the aliases, data_start and old_symbols columns by hand.")


def cmd_sources(ctx, args):
    print_table([(s.name, s.task_unit, s.description) for s in registry.ALL], ("source", "unit", "description"))


def cmd_plan(ctx, args):
    source = registry.get(args.source)
    total, new, refreshed = plan_source(ctx, source, args.start, args.end or today_ist(),
                                        symbols_arg(args.symbols), args.refresh)
    print(f"{args.source}: {total} task(s) in range, {new} new, {refreshed} reset to pending")


def cmd_run(ctx, args):
    counts = run_source(ctx, registry.get(args.source), triggered_by=args.triggered_by, limit=args.limit,
                        symbols=symbols_arg(args.symbols), wait_retries=args.wait_retries)
    print(f"{args.source}: {dict(counts) or 'nothing due'}")


def cmd_pilot(ctx, args):
    symbols = symbols_arg(args.symbols) or PILOT_SYMBOLS
    start, end, today = args.start, args.end, today_ist()
    year_start = date(start.year, 1, 1)
    print(f"Pilot for {', '.join(symbols)}; daily files {start} to {end}")
    plan_and_run(ctx, "nse_bhavcopy", start, end, triggered_by="pilot")
    plan_and_run(ctx, "nse_indices", start, end, triggered_by="pilot")
    for name in ("nse_corporate_actions", "nse_results", "nse_board_meetings", "nse_pledges", "nse_shareholding"):
        plan_and_run(ctx, name, None, today, symbols, triggered_by="pilot")
    plan_and_run(ctx, "nse_announcements", year_start, end, symbols, triggered_by="pilot")
    plan_and_run(ctx, "nse_insider", year_start, end, symbols, triggered_by="pilot")
    plan_and_run(ctx, "nse_pdfs", start, end, symbols, limit=args.pdf_limit, triggered_by="pilot")
    plan_and_run(ctx, "nse_fii_dii", today, today, triggered_by="pilot")
    plan_and_run(ctx, "fred", None, today, triggered_by="pilot")
    plan_and_run(ctx, "rbi_policy", None, today, triggered_by="pilot")
    plan_and_run(ctx, "gdelt_raw", end, end, limit=args.gdelt_files, triggered_by="pilot")
    print()
    cmd_status(ctx, args)
    print("\nNext: `python -m src.ingest review` to see what needs checking.")


def cmd_daily(ctx, args):
    today = today_ist()
    recent, year_start = today - timedelta(days=10), date(today.year, 1, 1)
    plan_and_run(ctx, "nse_bhavcopy", recent, today, triggered_by="daily_job")
    plan_and_run(ctx, "nse_indices", recent, today, triggered_by="daily_job")
    plan_and_run(ctx, "nse_fii_dii", today, today, triggered_by="daily_job")
    plan_and_run(ctx, "nse_announcements", year_start, today, refresh=True, triggered_by="daily_job")
    plan_and_run(ctx, "nse_board_meetings", None, today, refresh=True, triggered_by="daily_job")
    plan_and_run(ctx, "nse_insider", year_start, today, refresh=True, triggered_by="daily_job")
    plan_and_run(ctx, "nse_pdfs", recent, today, triggered_by="daily_job")
    plan_and_run(ctx, "gdelt_raw", today - timedelta(days=1), today, triggered_by="daily_job")
    if today.weekday() == 0:  # weekly on Mondays
        for name in ("nse_corporate_actions", "nse_results", "nse_pledges", "nse_shareholding", "fred"):
            plan_and_run(ctx, name, None, today, refresh=True, triggered_by="daily_job")
    print()
    cmd_review(ctx, argparse.Namespace(source=None))


def cmd_status(ctx, args):
    counts: dict[str, dict[str, int]] = defaultdict(dict)
    for r in ctx.tracker.summary():
        counts[r["source"]][r["status"]] = r["n"]
    statuses = ["success", "no_data", "pending", "running", "retry_wait", "needs_review", "verified_gap", "skipped"]
    rows = []
    for name in registry.SOURCES:
        c = counts.get(name, {})
        if c:
            rows.append((name, sum(c.values()), *(c.get(s, "") for s in statuses)))
    if not rows:
        print("No tasks yet. Start with `python -m src.ingest pilot` or `plan` + `run`.")
        return
    print_table(rows, ("source", "total", *statuses))


def cmd_review(ctx, args):
    rows = ctx.tracker.review_list(getattr(args, "source", None))
    if not rows:
        print("Review list is empty.")
        return
    out = []
    for r in rows:
        if r["status"] == "needs_review" and r["reason"] != "quality_check":
            what = f"{r['error_type']}: {(r['error_message'] or '').splitlines()[0][:90]}" if r["error_type"] else ""
        else:
            details = json.loads(r["details"]) if r["details"] else {}
            what = json.dumps(details, ensure_ascii=False)[:100]
        out.append((r["task_id"], r["source"], r["task_key"], r["status"], r["severity"],
                    r["check_name"] or r["reason"], what))
    print_table(out, ("task", "source", "key", "status", "severity", "check", "details"))
    blocked = sum(1 for r in rows if r["status"] == "needs_review")
    print(f"\n{blocked} blocked task item(s), {len(rows) - blocked} warning(s)/note(s).")
    print("Resolve with: python -m src.ingest resolve TASK --decision "
          "{retry|fixed_parser|accept_gap|use_fallback|ignore|acknowledge} --note \"...\"")


def cmd_resolve(ctx, args):
    task = resolve_task(ctx, args.task_id, args.decision, args.note, args.by or getpass.getuser())
    print(f"Task {task.task_id} ({task.source} {task.task_key}) is now {task.status} at stage {task.stage}.")
    if task.status == "pending":
        print(f"Run `python -m src.ingest run --source {task.source}` to process it.")


def cmd_reparse(ctx, args):
    n = ctx.tracker.reparse(args.source, args.start.isoformat() if args.start else None,
                            args.end.isoformat() if args.end else None)
    print(f"{n} task(s) set to re-parse from their saved raw files. Now run: "
          f"python -m src.ingest run --source {args.source}")


def cmd_coverage(ctx, args):
    end = args.end or today_ist()
    if args.symbol:
        start = args.start or end - timedelta(days=30)
    else:
        start = args.start or ctx.cfg.start_date("prices")
    if not args.no_build:
        summary = build_coverage(ctx, args.start or ctx.cfg.start_date("prices"), end)
        print_table([(src, c.get("ok", 0), c.get("empty", 0), c.get("gap", 0)) for src, c in summary.items()],
                    ("source", "ok", "empty", "gap"))
    if args.symbol:
        print(f"\n{args.symbol}, {start} to {end} (numbers = records that day)")
        print_table(coverage_table(ctx, args.symbol.upper(), start, end), ("date", "prices", "announcements", "news"))


def cmd_views(ctx, args):
    tables = create_views(ctx.cfg, ctx.lake)
    print(f"Created {len(tables)} view(s) in {ctx.cfg.duckdb_path}: {', '.join(tables) or '(no tables yet)'}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m src.ingest", description="Data collection for the NSE project")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("init", help="create the tracker database and load the trading calendar")
    sub.add_parser("universe", help="build data/manual/universe.csv from NSE's index lists")
    sub.add_parser("sources", help="list the data sources")

    s = sub.add_parser("plan", help="create tasks for a source")
    s.add_argument("--source", required=True)
    s.add_argument("--from", dest="start", type=date.fromisoformat)
    s.add_argument("--to", dest="end", type=date.fromisoformat)
    s.add_argument("--symbols", help="comma-separated, for per-stock sources")
    s.add_argument("--refresh", action="store_true", help="reset existing tasks in range to pending")

    s = sub.add_parser("run", help="process due tasks for a source")
    s.add_argument("--source", required=True)
    s.add_argument("--limit", type=int)
    s.add_argument("--symbols")
    s.add_argument("--wait-retries", action="store_true", help="wait for retry_wait tasks instead of exiting")
    s.add_argument("--triggered-by", default="backfill")

    s = sub.add_parser("pilot", help="run every source for a few stocks and a short date range")
    s.add_argument("--symbols", default=",".join(PILOT_SYMBOLS))
    s.add_argument("--from", dest="start", type=date.fromisoformat, default=date(2024, 6, 3))
    s.add_argument("--to", dest="end", type=date.fromisoformat, default=date(2024, 6, 28))
    s.add_argument("--pdf-limit", type=int, default=20)
    s.add_argument("--gdelt-files", type=int, default=8, help="15-minute GDELT files to process")

    sub.add_parser("daily", help="evening update job")
    sub.add_parser("status", help="progress per source")

    s = sub.add_parser("review", help="the 'verify later' list")
    s.add_argument("--source")

    s = sub.add_parser("resolve", help="decide what to do with a task on the review list")
    s.add_argument("task_id", type=int)
    s.add_argument("--decision", required=True, choices=DECISIONS)
    s.add_argument("--note")
    s.add_argument("--by")

    s = sub.add_parser("reparse", help="re-run only the parse step from saved raw files")
    s.add_argument("--source", required=True)
    s.add_argument("--from", dest="start", type=date.fromisoformat)
    s.add_argument("--to", dest="end", type=date.fromisoformat)

    s = sub.add_parser("coverage", help="what we have per stock per day")
    s.add_argument("--symbol")
    s.add_argument("--from", dest="start", type=date.fromisoformat)
    s.add_argument("--to", dest="end", type=date.fromisoformat)
    s.add_argument("--no-build", action="store_true", help="show the last built coverage without rebuilding")

    sub.add_parser("views", help="create DuckDB views over the lake")
    return p


COMMANDS = {
    "init": cmd_init, "universe": cmd_universe, "sources": cmd_sources, "plan": cmd_plan, "run": cmd_run,
    "pilot": cmd_pilot, "daily": cmd_daily, "status": cmd_status, "review": cmd_review,
    "resolve": cmd_resolve, "reparse": cmd_reparse, "coverage": cmd_coverage, "views": cmd_views,
}


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    args = build_parser().parse_args(argv)
    ctx = make_context(Config.load())
    try:
        if args.command not in ("init", "universe", "sources"):
            ensure_ready(ctx)
        COMMANDS[args.command](ctx, args)
    finally:
        ctx.tracker.close()


if __name__ == "__main__":
    main()
