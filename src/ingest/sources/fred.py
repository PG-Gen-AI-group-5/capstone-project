"""FRED macro series (US yields, VIX, USD/INR, dollar index, Brent)."""
from __future__ import annotations

from datetime import datetime, time, timedelta

import pandas as pd

from src.ingest.errors import IngestError
from src.ingest.sources.base import DownloadResult, Frame, ParseResult, Source, load_json, raw_path, save_json
from src.ingest.tracker import TaskSpec
from src.timeutil import UTC, parse_date

BASE = "https://api.stlouisfed.org/fred/series/observations"
MACRO_SCHEMA = {"series_id": "str", "obs_date": "date", "value": "float", "known_from": "ts",
                "realtime_start": "date", "published": "str", "description": "str", "source": "str"}


class Fred(Source):
    name = "fred"
    task_unit = "series"
    description = "FRED macro series, one task per series (needs FRED_API_KEY in .env)"

    def default_start(self, ctx):
        return ctx.cfg.start_date("fred")

    def plan(self, ctx, start, end, symbols):
        # The URL stored in the tracker never contains the API key.
        return [TaskSpec(task_key=sid, period_start=start, period_end=end, url=f"{BASE}?series_id={sid}")
                for sid in ctx.cfg["fred"]["series"]]

    def download(self, ctx, task):
        key = ctx.cfg.secret("FRED_API_KEY")
        if not key:
            raise IngestError("config_missing", "Set FRED_API_KEY in .env (free key from fred.stlouisfed.org)")
        payload = ctx.http.json(BASE, params={
            "series_id": task.task_key, "api_key": key, "file_type": "json",
            "observation_start": task.start_date.isoformat(), "observation_end": task.end_date.isoformat(),
        })
        if "observations" not in payload:
            raise IngestError("bad_response", str(payload.get("error_message", payload))[:300])
        if not payload["observations"]:
            return DownloadResult(no_data=True)
        path = raw_path(ctx, "fred", f"{task.task_key}.json")
        return DownloadResult(path, nbytes=save_json(path, payload), rows=len(payload["observations"]))

    def parse(self, ctx, task, raw):
        meta = ctx.cfg["fred"]["series"][task.task_key]
        obs = pd.DataFrame(load_json(raw)["observations"])
        obs_date = obs["date"].map(parse_date)
        lag = timedelta(days=int(meta.get("lag_days", 1)))
        out = pd.DataFrame({
            "series_id": task.task_key,
            "obs_date": obs_date,
            "value": pd.to_numeric(obs["value"], errors="coerce"),   # FRED uses "." for missing
            # Usable from 00:00 UTC (05:30 IST) on obs_date + lag: after the US close, before the 08:30 IST forecast.
            "known_from": obs_date.map(lambda d: datetime.combine(d + lag, time(0, 0), UTC)),
            "realtime_start": obs.get("realtime_start", pd.Series(dtype=object)).map(parse_date),
            "published": meta.get("published"),
            "description": meta.get("description"),
            "source": "fred",
        })
        return ParseResult([Frame("macro", out, ["series_id", "obs_date"], "series_id", MACRO_SCHEMA)])
