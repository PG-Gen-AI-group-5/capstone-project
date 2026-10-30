"""GDELT news from the raw 15-minute GKG files (route A in docs/DATA_COLLECTION.md).

Each file covers all news worldwide for 15 minutes (~4.5 MB zipped). We download it, keep only rows
that mention a Nifty 50 company, save those rows as Parquet, and delete the zip. GDELT files are
permanent, so the zip can always be downloaded again (the tracker keeps its URL and MD5).
"""
from __future__ import annotations

import csv
import html
import io
import re
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from src.ingest.errors import IngestError
from src.ingest.sources.base import Context, DownloadResult, Frame, ParseResult, Source, raw_path
from src.ingest.tracker import TaskSpec
from src.timeutil import UTC, now_utc

MASTER_LIST_URL = "http://data.gdeltproject.org/gdeltv2/masterfilelist.txt"

# GKG 2.1 column positions (tab-separated, no header)
COL_ID, COL_DATE, COL_SOURCE, COL_URL, COL_THEMES, COL_ORGS, COL_TONE, COL_EXTRAS = 0, 1, 3, 4, 8, 13, 15, 26

NEWS_SCHEMA = {
    "symbol": "str", "url": "str", "published_at": "ts", "source_domain": "str", "title": "str",
    "tone": "float", "positive": "float", "negative": "float", "polarity": "float", "word_count": "int",
    "themes": "str", "organizations": "str", "n_orgs": "int", "in_title": "bool", "relevance": "float",
    "match_alias": "str", "gkg_record_id": "str", "month": "str", "source": "str",
}

_TITLE = re.compile(r"<PAGE_TITLE>(.*?)</PAGE_TITLE>", re.DOTALL)
csv.field_size_limit(2**31 - 1)


def build_matchers(aliases: dict[str, list[str]]) -> tuple[re.Pattern | None, dict[str, re.Pattern]]:
    """One quick 'any company?' pattern plus one pattern per stock. Matches whole names only."""
    def pattern(names):
        alts = "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))
        return re.compile(rf"(?<![a-z0-9])(?:{alts})(?![a-z0-9])")
    per_symbol = {s: pattern(names) for s, names in aliases.items() if names}
    all_names = [n for names in aliases.values() for n in names]
    return (pattern(all_names) if all_names else None), per_symbol


def relevance(in_title: bool, n_orgs: int) -> float:
    """1.0 = company in the headline; lower when it's one of many organisations mentioned."""
    if in_title:
        return 1.0
    return 0.6 if n_orgs <= 5 else 0.3


def filter_rows(lines, any_pattern, per_symbol) -> list[dict]:
    """Rows of one GKG file that mention a company, one record per (article, company)."""
    out = []
    if any_pattern is None:
        return out
    for row in csv.reader(lines, delimiter="\t", quoting=csv.QUOTE_NONE):
        if len(row) <= COL_TONE:
            continue
        orgs = row[COL_ORGS].lower()
        extras = row[COL_EXTRAS] if len(row) > COL_EXTRAS else ""
        m = _TITLE.search(extras)
        title = html.unescape(m.group(1)).strip() if m else ""
        haystack = f"{orgs} || {title.lower()}"
        if not any_pattern.search(haystack):
            continue
        tone = (row[COL_TONE].split(",") + [""] * 7)[:7]
        org_names = [o for o in dict.fromkeys(orgs.split(";")) if o]
        themes = list(dict.fromkeys(t.split(",")[0] for t in row[COL_THEMES].split(";") if t))[:30]
        for symbol, rx in per_symbol.items():
            hit = rx.search(haystack)
            if not hit:
                continue
            in_title = bool(rx.search(title.lower()))
            out.append({
                "symbol": symbol, "url": row[COL_URL],
                "published_at": datetime.strptime(row[COL_DATE], "%Y%m%d%H%M%S").replace(tzinfo=UTC),
                "source_domain": row[COL_SOURCE], "title": title,
                "tone": tone[0] or None, "positive": tone[1] or None, "negative": tone[2] or None,
                "polarity": tone[3] or None, "word_count": tone[6] or None,
                "themes": ";".join(themes), "organizations": ";".join(org_names)[:1000],
                "n_orgs": len(org_names), "in_title": in_title,
                "relevance": relevance(in_title, len(org_names)), "match_alias": hit.group(0),
                "gkg_record_id": row[COL_ID],
            })
    return out


class GdeltRaw(Source):
    name = "gdelt_raw"
    task_unit = "file_15min"
    description = "GDELT GKG 15-minute files, filtered to Nifty 50 companies (~141k files since Oct 2022)"
    retry_base_sec = 120

    def default_start(self, ctx):
        return ctx.cfg.start_date("gdelt")

    def _master_list(self, ctx: Context) -> Path:
        path = raw_path(ctx, "gdelt", "masterfilelist.txt")
        max_age = timedelta(hours=ctx.cfg["gdelt"]["masterlist_max_age_hours"])
        fresh = path.exists() and datetime.now().timestamp() - path.stat().st_mtime < max_age.total_seconds()
        if not fresh:
            print("  Downloading GDELT master file list (~130 MB)...")
            ctx.http.download_to(MASTER_LIST_URL, path)
        return path

    def plan(self, ctx, start: date, end: date, symbols):
        lo, hi = start.strftime("%Y%m%d") + "000000", end.strftime("%Y%m%d") + "235959"
        specs = []
        with open(self._master_list(ctx), encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) != 3 or not parts[2].endswith(".gkg.csv.zip"):
                    continue
                ts = Path(parts[2]).name[:14]
                if lo <= ts <= hi:
                    t0 = datetime.strptime(ts, "%Y%m%d%H%M%S").replace(tzinfo=UTC)
                    specs.append(TaskSpec(task_key=ts, period_start=t0, period_end=t0 + timedelta(minutes=15),
                                          url=parts[2], expected_bytes=int(parts[0]), expected_md5=parts[1]))
        return specs

    def _matchers(self, ctx: Context):
        if "gdelt_matchers" not in ctx.state:
            ctx.state["gdelt_matchers"] = build_matchers(ctx.universe.aliases(ctx.cfg["universe"]["agents"]))
        return ctx.state["gdelt_matchers"]

    def download(self, ctx, task):
        ts = task.task_key
        zip_path = raw_path(ctx, "gdelt", "tmp", f"{ts}.gkg.csv.zip")
        try:
            size, _ = ctx.http.download_to(task.url, zip_path, task.expected_bytes, task.expected_md5)
        except IngestError as e:
            file_time = datetime.strptime(ts, "%Y%m%d%H%M%S").replace(tzinfo=UTC)
            if e.error_type == "http_404" and file_time > now_utc() - timedelta(hours=3):
                raise IngestError("not_yet_available", f"GDELT file {ts} not published yet", 404) from e
            raise
        any_pattern, per_symbol = self._matchers(ctx)
        try:
            with zipfile.ZipFile(zip_path) as z, z.open(z.namelist()[0]) as f:
                records = filter_rows(io.TextIOWrapper(f, encoding="utf-8", errors="replace", newline=""),
                                      any_pattern, per_symbol)
        except zipfile.BadZipFile as e:
            raise IngestError("bad_response", f"{ts}: corrupt zip") from e
        finally:
            if not ctx.cfg["gdelt"]["keep_zip"]:
                zip_path.unlink(missing_ok=True)
        if not records:
            return DownloadResult(None, nbytes=size, rows=0)
        out = raw_path(ctx, "gdelt", ts[:4], ts[4:6], f"{ts}.parquet")
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(records).to_parquet(out, index=False)
        return DownloadResult(out, nbytes=size, rows=len(records))

    def parse(self, ctx, task, raw):
        if raw is None:
            return ParseResult([])
        df = pd.read_parquet(raw)
        df["month"] = task.task_key[:4] + "-" + task.task_key[4:6]
        df["source"] = "gdelt_raw"
        return ParseResult([Frame("news", df, ["symbol", "url"], "month", NEWS_SCHEMA)])
