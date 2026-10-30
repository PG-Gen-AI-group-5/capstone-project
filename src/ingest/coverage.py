"""Coverage: what we have for each stock on each trading day, per source.

prices        ok / gap   (a price is expected on every trading day after the stock's first day)
announcements ok / empty (zero announcements on a day is normal)
news          ok / empty (zero articles on a day is normal)

Only collected days are counted: prices on days whose bhavcopy succeeded, announcements for stocks
and years whose task finished, news on days whose GDELT files are all finished. So data not yet
downloaded never shows up as a gap. Missing prices are also flagged on the review list by the bhavcopy check.
"""
from __future__ import annotations

from collections import Counter
from datetime import date, timedelta

from src.ingest.sources.base import Context
from src.timeutil import IST


def build_coverage(ctx: Context, start: date, end: date) -> dict[str, Counter]:
    tr, lake = ctx.tracker, ctx.lake
    trading = tr.trading_days(start, end)
    summary: dict[str, Counter] = {}

    # prices
    downloaded = {t.start_date for t in tr.tasks_with_status("nse_bhavcopy", ("success",))}
    prices = lake.read("prices_daily", ["symbol", "trade_date"])
    have = set(zip(prices["symbol"], prices["trade_date"])) if not prices.empty else set()
    first = prices.groupby("symbol")["trade_date"].min().to_dict() if not prices.empty else {}
    rows = []
    for d in trading:
        if d not in downloaded:
            continue
        for s in ctx.symbols("prices"):
            begin = ctx.universe.data_start(s) or first.get(s)
            if begin is None or d < begin:
                continue
            ok = (s, d) in have
            rows.append((s, d.isoformat(), "prices", int(ok), "ok" if ok else "gap"))
    tr.replace_coverage("prices", start, end, rows)
    summary["prices"] = Counter(r[4] for r in rows)

    # announcements and news: counts per stock per IST day, only where collection is complete,
    # so stocks or days not collected yet don't show up as "empty"
    collected = {"announcements": _announcements_collected(ctx), "news": _news_collected(ctx)}
    for table in ("announcements", "news"):
        df = lake.read(table, ["symbol", "published_at"])
        if df.empty:
            continue
        df = df.dropna(subset=["published_at"])
        df["day"] = df["published_at"].dt.tz_convert(IST).dt.date
        counts = df.groupby(["symbol", "day"]).size().to_dict()
        rows = []
        for d in trading:
            for s in ctx.symbols("agents"):
                if not collected[table](s, d):
                    continue
                n = int(counts.get((s, d), 0))
                rows.append((s, d.isoformat(), table, n, "ok" if n else "empty"))
        tr.replace_coverage(table, start, end, rows)
        summary[table] = Counter(r[4] for r in rows)
    return summary


def _announcements_collected(ctx: Context):
    """(symbol, day) is collected if a finished announcements task for that stock covers the day."""
    ranges: dict[str, list[tuple[date, date]]] = {}
    for t in ctx.tracker.tasks_with_status("nse_announcements", ("success", "no_data")):
        ranges.setdefault(t.symbol, []).append((t.start_date, t.end_date))
    return lambda s, d: any(a <= d <= b for a, b in ranges.get(s, []))


def _news_collected(ctx: Context):
    """An IST day is collected when every GDELT file of the UTC days it spans (d-1 and d) is finished."""
    rows = ctx.tracker.conn.execute(
        "SELECT substr(period_start, 1, 10) AS day, COUNT(*) AS total, "
        "SUM(status IN ('success', 'no_data', 'verified_gap', 'skipped')) AS done "
        "FROM tasks WHERE source = 'gdelt_raw' GROUP BY day").fetchall()
    complete = {date.fromisoformat(r["day"]) for r in rows if r["total"] == r["done"]}
    return lambda s, d: d in complete and (d - timedelta(days=1)) in complete


def coverage_table(ctx: Context, symbol: str, start: date, end: date) -> list[tuple]:
    rows = ctx.tracker.conn.execute(
        "SELECT trade_date, source, n_records, status FROM coverage WHERE symbol=? AND trade_date BETWEEN ? AND ? "
        "ORDER BY trade_date", (symbol, start.isoformat(), end.isoformat())).fetchall()
    by_day: dict[str, dict[str, str]] = {}
    for r in rows:
        cell = "gap" if r["status"] == "gap" else str(r["n_records"])
        by_day.setdefault(r["trade_date"], {})[r["source"]] = cell
    return [(d, v.get("prices", "-"), v.get("announcements", "-"), v.get("news", "-")) for d, v in by_day.items()]
