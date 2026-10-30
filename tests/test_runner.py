from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from src.ingest.errors import IngestError
from src.ingest.runner import run_source
from src.ingest.sources.base import DownloadResult, Frame, Issue, ParseResult, Source, daily_404
from src.ingest.tracker import TaskSpec
from src.timeutil import today_ist

SCHEMA = {"key": "str", "part": "str"}


class FakeSource(Source):
    name = "fake"
    task_unit = "day"

    def __init__(self, downloads, issues=None, parse_error=None):
        self.downloads = downloads          # task_key -> DownloadResult or exception
        self.issues = issues or {}
        self.parse_error = parse_error or set()

    def plan(self, ctx, start, end, symbols):
        return [TaskSpec(task_key=k, period_start=date(2024, 6, 3), period_end=date(2024, 6, 3))
                for k in self.downloads]

    def download(self, ctx, task):
        outcome = self.downloads[task.task_key]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def parse(self, ctx, task, raw):
        if task.task_key in self.parse_error:
            raise KeyError("column missing")
        df = pd.DataFrame([{"key": task.task_key, "part": "p"}])
        return ParseResult([Frame("fake_table", df, ["key"], "part", SCHEMA)], self.issues.get(task.task_key, []))


def setup(ctx, source):
    ctx.tracker.plan(source.name, source.plan(ctx, None, None, None))


def status(ctx, key):
    row = ctx.tracker.conn.execute("SELECT status FROM tasks WHERE task_key=?", (key,)).fetchone()
    return row["status"]


def test_errors_never_stop_the_run(ctx, tmp_path):
    raw = tmp_path / "x.bin"
    raw.write_bytes(b"x")
    source = FakeSource({
        "a": IngestError("http_403", "blocked"),
        "b": DownloadResult(raw),
        "c": IngestError("http_404", "missing"),
        "d": DownloadResult(no_data=True),
    })
    setup(ctx, source)
    counts = run_source(ctx, source, log=lambda _: None)
    assert counts == {"failed": 2, "success": 1, "no_data": 1}
    assert [status(ctx, k) for k in "abcd"] == ["retry_wait", "success", "needs_review", "no_data"]
    assert list(ctx.lake.read("fake_table")["key"]) == ["b"]


def test_blocking_issue_keeps_data_out_of_the_lake(ctx, tmp_path):
    source = FakeSource({"a": DownloadResult(tmp_path)},
                        issues={"a": [Issue("too_few_rows", "error", {"rows": 3}, blocking=True)]})
    setup(ctx, source)
    run_source(ctx, source, log=lambda _: None)
    assert status(ctx, "a") == "needs_review"
    assert ctx.lake.read("fake_table").empty
    assert ctx.tracker.review_list()[0]["check_name"] == "too_few_rows"


def test_warning_is_reviewed_but_task_succeeds(ctx, tmp_path):
    source = FakeSource({"a": DownloadResult(tmp_path)},
                        issues={"a": [Issue("missing_symbol", "warn", {"symbols": ["BBB"]})]})
    setup(ctx, source)
    run_source(ctx, source, log=lambda _: None)
    assert status(ctx, "a") == "success"
    assert ctx.tracker.review_list()[0]["severity"] == "warn"


def test_parser_bug_then_fixed_parser_reparses_without_download(ctx, tmp_path):
    raw = tmp_path / "saved.csv"
    raw.write_text("data")
    source = FakeSource({"a": DownloadResult(raw)}, parse_error={"a"})
    setup(ctx, source)
    run_source(ctx, source, log=lambda _: None)
    task = ctx.tracker.due_tasks("fake") or ctx.tracker.tasks_with_status("fake", ("needs_review",))
    assert task[0].stage == "parse" and task[0].raw_path == str(raw)

    source.parse_error.clear()
    source.downloads["a"] = IngestError("http_403", "a download now would fail")
    ctx.tracker.resolve(task[0].task_id, "fixed_parser", note="parser fixed")
    run_source(ctx, source, log=lambda _: None)
    assert status(ctx, "a") == "success"


def test_daily_404_holiday_recent_and_real_gap(ctx):
    holiday, old_day = date(2024, 3, 25), date(2024, 3, 26)
    ctx.tracker.load_calendar([(holiday, False, "Holi"), (old_day, True, None)])
    assert daily_404(ctx, holiday).no_data
    with pytest.raises(IngestError) as e:
        daily_404(ctx, old_day)
    assert e.value.error_type == "http_404"
    with pytest.raises(IngestError) as e:
        daily_404(ctx, today_ist() - timedelta(days=1))
    assert e.value.error_type == "not_yet_available"
