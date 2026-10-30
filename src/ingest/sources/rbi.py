"""RBI policy decisions, entered by hand in data/manual/rbi_policy.csv."""
from __future__ import annotations

import pandas as pd

from src.ingest.errors import IngestError
from src.ingest.sources.base import DownloadResult, Frame, Issue, ParseResult, Source
from src.ingest.tracker import TaskSpec
from src.timeutil import at_ist, parse_date

COLUMNS = ["decision_date", "announce_time_ist", "repo_rate_before", "repo_rate_after", "stance", "source_url"]
RBI_SCHEMA = {"decision_date": "date", "announce_time_ist": "str", "repo_rate_before": "float",
              "repo_rate_after": "float", "stance": "str", "source_url": "str", "published_at": "ts",
              "source": "str", "part": "str"}


class RbiPolicy(Source):
    name = "rbi_policy"
    task_unit = "file"
    description = "RBI monetary policy decisions from data/manual/rbi_policy.csv (re-plan with --refresh after edits)"

    def plan(self, ctx, start, end, symbols):
        return [TaskSpec(task_key="rbi_policy.csv", period_start=start, period_end=end)]

    def download(self, ctx, task):
        path = ctx.cfg.manual_dir / "rbi_policy.csv"
        if not path.exists():
            raise IngestError("config_missing", f"{path} not found")
        return DownloadResult(path, nbytes=path.stat().st_size)

    def parse(self, ctx, task, raw):
        df = pd.read_csv(raw, dtype=str).fillna("")
        missing = set(COLUMNS) - set(df.columns)
        if missing:
            raise IngestError("schema_changed", f"rbi_policy.csv is missing columns {sorted(missing)}")
        if df.empty:
            return ParseResult([], [Issue("empty_manual_file", "info", {"file": "data/manual/rbi_policy.csv"})])
        df["decision_date"] = df["decision_date"].map(parse_date)
        bad = df["decision_date"].isna().sum()
        if bad:
            raise IngestError("validation_failed", f"{bad} rows have an unreadable decision_date (use YYYY-MM-DD)")
        df["announce_time_ist"] = df["announce_time_ist"].replace("", "10:00")
        df["published_at"] = [at_ist(d, t) for d, t in zip(df["decision_date"], df["announce_time_ist"])]
        df["source"] = "rbi_manual"
        df["part"] = "all"
        return ParseResult([Frame("rbi_policy", df, ["decision_date"], "part", RBI_SCHEMA)])
