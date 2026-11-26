"""The live data update: download what changed since the last run, then rebuild the live tables.

1. Ingest into the shared lake: NSE bhavcopy and indices from a few days before the last price day (so a missed
   week is caught up), announcements and board meetings (re-read, since filings arrive late), and weekly the
   corporate actions, results and FRED series.
2. Build the live tables in data/live/derived: adjusted prices, features (with the row for the next trading
   day D, whose prices are not out yet) and labels.
"""
from __future__ import annotations

import json
import time
from datetime import date, timedelta

from src.config import Config
from src.live.env import LiveConfig, last_price_day, live_config, live_settings, next_trading_day
from src.store.db import read_table, write_manifest

DAILY = ("nse_bhavcopy", "nse_indices")
FILINGS = ("nse_announcements", "nse_board_meetings")
WEEKLY = ("nse_corporate_actions", "nse_results", "fred")


def state_path(lcfg: LiveConfig):
    return lcfg.live_dir / "state.json"


def read_state(lcfg: LiveConfig) -> dict:
    p = state_path(lcfg)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def write_state(lcfg: LiveConfig, **values) -> None:
    p = state_path(lcfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({**read_state(lcfg), **values}, indent=1, default=str), encoding="utf-8")


def ingest(cfg: Config, log=print) -> dict:
    """Download into the lake. Each source's tasks are tracked, so a rerun only fetches what is missing."""
    import contextlib
    import io

    from src.ingest.__main__ import ensure_ready, plan_and_run
    from src.ingest.runner import make_context
    from src.timeutil import today_ist

    lcfg = live_config(cfg)
    today = today_ist()
    before = last_price_day(cfg)
    start = min(before - timedelta(days=5), today - timedelta(days=10))
    filings_from = min(date(today.year, 1, 1), today - timedelta(days=10))
    state = read_state(lcfg)
    weekly_due = (not state.get("weekly_sources_at")
                  or (today - date.fromisoformat(state["weekly_sources_at"])).days
                  >= int(live_settings(cfg)["weekly_sources_every_days"]))
    ctx = make_context(cfg)
    out = io.StringIO()
    steps = [(n, start, False) for n in DAILY] + [(n, filings_from if n == "nse_announcements" else None, True)
                                                   for n in FILINGS]
    steps += [(n, None, True) for n in WEEKLY] if weekly_due else []
    try:
        ensure_ready(ctx)
        for name, since, refresh in steps:
            log(f"  downloading {name}" + (f" from {since}" if since else ""))
            with contextlib.redirect_stdout(out):              # the collectors' own progress lines
                plan_and_run(ctx, name, since, today, refresh=refresh, triggered_by="live")
    finally:
        ctx.tracker.close()
    if weekly_due:
        write_state(lcfg, weekly_sources_at=today)
    after = last_price_day(cfg)
    lines = [ln for ln in out.getvalue().splitlines() if "result:" in ln or "failed" in ln.lower()]
    return {"last_price_day_before": before, "last_price_day": after, "weekly": weekly_due, "log": lines[-12:]}


def build(cfg: Config, day: date | None = None, log=print) -> dict:
    """Rebuild data/live/derived. `day`: the next trading day to build a feature row for (default: the trading
    day after the last price day)."""
    from src.features import build as features
    from src.labels import labels
    from src.store import adjust

    lcfg = live_config(cfg)
    last = last_price_day(cfg)
    day = day or next_trading_day(cfg, last)
    t0 = time.time()
    log(f"  building live tables: prices to {last}, features for {day}")
    adjust.build(lcfg)
    f = features.build(lcfg, extra_day=day)
    labels.build(lcfg)
    info = {"last_price_day": last, "day": day, "feature_rows": f["rows"], "sec": round(time.time() - t0, 1)}
    write_manifest(lcfg.derived_dir / "live", stage="live", **info)
    write_state(lcfg, built_for=day, last_price_day=last)
    return info


def built(cfg: Config) -> dict:
    """What the live tables hold now (the manifest of the last build), or {}."""
    p = live_config(cfg).derived_dir / "live" / "_manifest.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def lake_status(cfg: Config) -> dict:
    ann = read_table(cfg, "announcements", columns=["published_at"])
    return {"last_price_day": str(last_price_day(cfg)), "last_announcement": str(ann["published_at"].max())}
