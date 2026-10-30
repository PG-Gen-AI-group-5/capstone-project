"""What every collector implements, and the context passed to it."""
from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass, field
from datetime import date, timedelta
from functools import cached_property
from pathlib import Path

import pandas as pd

from src.config import Config
from src.ingest.errors import IngestError
from src.ingest.http import Http, write_atomic
from src.ingest.lake import Lake
from src.ingest.tracker import Task, TaskSpec, Tracker
from src.ingest.universe import Universe
from src.timeutil import today_ist


@dataclass
class Context:
    cfg: Config
    tracker: Tracker
    lake: Lake
    http: Http
    state: dict = field(default_factory=dict)   # per-run caches (e.g. first date each stock was seen)

    @cached_property
    def universe(self) -> Universe:
        return Universe.load(self.cfg.manual_dir / "universe.csv")

    def symbols(self, purpose: str) -> list[str]:
        """purpose: 'agents' or 'prices' (see config.yaml -> universe)."""
        return self.universe.symbols(self.cfg["universe"][purpose])


@dataclass
class Frame:
    """Rows for one lake table."""
    table: str
    df: pd.DataFrame
    key: list[str]
    partition: str
    schema: dict[str, str]


@dataclass
class Issue:
    """A data-quality finding. Blocking issues stop the data being written and send the task to review."""
    check_name: str
    severity: str                 # info, warn, error
    details: dict | None = None
    blocking: bool = False


@dataclass
class DownloadResult:
    raw_path: Path | None = None
    no_data: bool = False         # the request worked but nothing exists (holiday, empty list)
    nbytes: int | None = None
    rows: int | None = None


@dataclass
class ParseResult:
    frames: list[Frame] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)

    @property
    def rows(self) -> int:
        return sum(len(f.df) for f in self.frames)


class Source:
    name: str = ""
    task_unit: str = ""
    description: str = ""
    max_attempts: int | None = None    # None = use config.yaml tracker settings
    retry_base_sec: int | None = None

    def plan(self, ctx: Context, start: date, end: date, symbols: list[str] | None) -> list[TaskSpec]:
        raise NotImplementedError

    def download(self, ctx: Context, task: Task) -> DownloadResult:
        raise NotImplementedError

    def parse(self, ctx: Context, task: Task, raw_path: Path | None) -> ParseResult:
        raise NotImplementedError

    def fallback(self, ctx: Context, task: Task) -> ParseResult:
        raise NotImplementedError(f"{self.name} has no backup source")

    def default_start(self, ctx: Context) -> date:
        return date(2016, 1, 1)


# ---------- helpers shared by collectors ----------

def raw_path(ctx: Context, *parts: str) -> Path:
    return ctx.cfg.raw_dir.joinpath(*parts)


def save_json(path: Path, obj) -> int:
    data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    write_atomic(path, data)
    return len(data)


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def as_records(payload) -> list[dict]:
    """NSE endpoints return either a list or {'data': [...]}."""
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "rows", "results"):
            if isinstance(payload.get(key), list):
                return payload[key]
        return [payload] if payload else []
    raise IngestError("schema_changed", f"Unexpected payload type {type(payload).__name__}")


def read_csv_from_zip(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise IngestError("schema_changed", f"No CSV inside {path.name}")
        with z.open(names[0]) as f:
            return pd.read_csv(io.BytesIO(f.read()), dtype=str)


def daily_404(ctx: Context, d: date) -> DownloadResult:
    """A daily file returned 404: holiday -> no_data; very recent day -> not yet published; else a real gap."""
    trading = ctx.tracker.is_trading(d)
    if trading is False:
        return DownloadResult(no_data=True)
    if d >= today_ist() - timedelta(days=1):
        raise IngestError("not_yet_available", f"File for {d} not published yet", 404)
    raise IngestError("http_404", f"No file for trading day {d}", 404)


def weekday_tasks(ctx: Context, start: date, end: date, url_for) -> list[TaskSpec]:
    """One task per weekday, plus weekend special sessions listed in the trading calendar."""
    from src.timeutil import days

    specials = {d for d in ctx.tracker.trading_days(start, end) if d.weekday() >= 5}
    return [TaskSpec(task_key=d.isoformat(), period_start=d, period_end=d, url=url_for(d))
            for d in days(start, end) if d.weekday() < 5 or d in specials]


def years(start: date, end: date) -> list[tuple[date, date]]:
    return [(max(start, date(y, 1, 1)), min(end, date(y, 12, 31))) for y in range(start.year, end.year + 1)]


def nse_date(d: date) -> str:
    return d.strftime("%d-%m-%Y")
