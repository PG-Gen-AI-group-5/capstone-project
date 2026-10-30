from __future__ import annotations

from datetime import date, datetime

import pandas as pd

from src.ingest.coverage import build_coverage
from src.ingest.sources.nse_announcements import ANNOUNCEMENT_SCHEMA
from src.ingest.sources.nse_daily import PRICE_SCHEMA
from src.ingest.tracker import TaskSpec
from src.timeutil import UTC

D1, D2 = date(2024, 6, 3), date(2024, 6, 4)


def test_coverage_counts_only_collected_data_and_finds_price_gaps(ctx):
    tr = ctx.tracker
    tr.load_calendar([(D1, True, None), (D2, True, None)])
    for d in (D1, D2):
        tr.plan("nse_bhavcopy", [TaskSpec(task_key=d.isoformat(), period_start=d, period_end=d)])
    for t in tr.due_tasks("nse_bhavcopy"):
        tr.complete(t.task_id, "success")
    prices = pd.DataFrame({"symbol": ["AAA", "BBB", "AAA"], "trade_date": [D1, D1, D2], "year": "2024"})
    ctx.lake.upsert("prices_daily", prices, ["symbol", "trade_date"], "year", PRICE_SCHEMA)

    tr.plan("nse_announcements", [TaskSpec(task_key="AAA|2024", symbol="AAA", period_start=D1, period_end=D2)])
    tr.complete(tr.due_tasks("nse_announcements")[0].task_id, "success")
    anns = pd.DataFrame({"doc_id": ["nse:1"], "symbol": ["AAA"],
                         "published_at": [datetime(2024, 6, 3, 6, 0, tzinfo=UTC)]})
    ctx.lake.upsert("announcements", anns, ["doc_id"], "symbol", ANNOUNCEMENT_SCHEMA)

    summary = build_coverage(ctx, D1, D2)
    rows = {(r["symbol"], r["trade_date"], r["source"]): r["status"]
            for r in tr.conn.execute("SELECT * FROM coverage").fetchall()}
    assert rows[("BBB", "2024-06-04", "prices")] == "gap"        # traded on D1, missing on D2
    assert rows[("AAA", "2024-06-04", "prices")] == "ok"
    assert rows[("AAA", "2024-06-03", "announcements")] == "ok"
    assert rows[("AAA", "2024-06-04", "announcements")] == "empty"
    assert not any(k[0] == "BBB" and k[2] == "announcements" for k in rows)   # not collected: not counted
    assert summary["prices"]["gap"] == 1
