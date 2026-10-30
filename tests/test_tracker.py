from __future__ import annotations

from datetime import date

import pytest

from src.ingest.tracker import TaskSpec


def plan_one(ctx, key="2024-06-03"):
    ctx.tracker.plan("fake", [TaskSpec(task_key=key, period_start=date(2024, 6, 3), period_end=date(2024, 6, 3))])
    return ctx.tracker.due_tasks("fake")[0]


def test_plan_is_idempotent(ctx):
    specs = [TaskSpec(task_key=f"d{i}", period_start=date(2024, 6, i), period_end=date(2024, 6, i))
             for i in range(1, 4)]
    assert ctx.tracker.plan("fake", specs) == (3, 0)
    assert ctx.tracker.plan("fake", specs) == (0, 0)


def test_refresh_resets_finished_tasks(ctx):
    task = plan_one(ctx)
    ctx.tracker.complete(task.task_id, "success", rows=10)
    _, refreshed = ctx.tracker.plan(
        "fake", [TaskSpec(task_key=task.task_key, period_start=date(2024, 6, 3), period_end=date(2024, 6, 3))],
        refresh=True)
    assert refreshed == 1
    assert ctx.tracker.get_task(task.task_id).status == "pending"


def test_retryable_error_waits_then_goes_to_review(ctx):
    task = plan_one(ctx)
    assert ctx.tracker.fail(task.task_id, "timeout") == "retry_wait"
    assert ctx.tracker.fail(task.task_id, "http_429") == "retry_wait"
    assert ctx.tracker.fail(task.task_id, "timeout") == "needs_review"
    review = ctx.tracker.review_list()
    assert len(review) == 1 and review[0]["reason"] == "max_attempts"


def test_retry_wait_task_is_not_due_until_its_time(ctx):
    task = plan_one(ctx)
    ctx.tracker.fail(task.task_id, "timeout")
    assert ctx.tracker.due_tasks("fake") == []
    assert ctx.tracker.next_retry_at("fake") is not None


def test_non_retryable_error_goes_straight_to_review(ctx):
    task = plan_one(ctx)
    assert ctx.tracker.fail(task.task_id, "schema_changed") == "needs_review"
    assert ctx.tracker.review_list()[0]["reason"] == "non_retryable_error"


def test_resolve_decisions(ctx):
    task = plan_one(ctx)
    ctx.tracker.fail(task.task_id, "http_404")
    with pytest.raises(ValueError):
        ctx.tracker.resolve(task.task_id, "acknowledge")
    t = ctx.tracker.resolve(task.task_id, "retry", note="try again", by="tester")
    assert (t.status, t.attempts) == ("pending", 0)
    assert ctx.tracker.review_list() == []

    ctx.tracker.fail(task.task_id, "http_404")
    t = ctx.tracker.resolve(task.task_id, "accept_gap", note="file missing on NSE")
    assert (t.status, t.stage) == ("verified_gap", "done")


def test_fixed_parser_needs_a_raw_file(ctx):
    task = plan_one(ctx)
    ctx.tracker.fail(task.task_id, "parse_error")
    with pytest.raises(ValueError):
        ctx.tracker.resolve(task.task_id, "fixed_parser")


def test_reset_stale_running_tasks(ctx):
    task = plan_one(ctx)
    ctx.tracker.mark_running(task.task_id)
    assert ctx.tracker.reset_stale("fake") == 1
    assert ctx.tracker.get_task(task.task_id).status == "pending"


def test_duplicate_open_review_items_are_not_repeated(ctx):
    task = plan_one(ctx)
    assert ctx.tracker.add_review(task.task_id, "quality_check", "warn", "missing_symbol", {"symbols": ["A"]})
    assert ctx.tracker.add_review(task.task_id, "quality_check", "warn", "missing_symbol", {"symbols": ["A"]}) is None
