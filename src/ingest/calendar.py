"""NSE trading calendar: tells a holiday (no data expected) from a missing file (a real gap).

Weekday holidays come from exchange_calendars (BSE calendar 'XBOM'; NSE and BSE share holidays).
Weekend special sessions (Diwali Muhurat trading, Budget Saturdays) come from
data/manual/calendar_overrides.csv, which also wins over the library for any date it lists.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from src.timeutil import days


def build_calendar(start: date, end: date, overrides_path: Path) -> list[tuple[date, bool, str | None]]:
    import exchange_calendars as xc

    cal = xc.get_calendar("XBOM")
    first, last = cal.first_session.date(), cal.last_session.date()
    rows: dict[date, tuple[bool, str | None]] = {}
    for d in days(start, end):
        if d.weekday() >= 5:
            rows[d] = (False, None)
        elif first <= d <= last:
            rows[d] = (bool(cal.is_session(pd.Timestamp(d))), None)
        else:
            rows[d] = (True, "assumed trading day (outside exchange_calendars range)")

    if overrides_path.exists():
        overrides = pd.read_csv(overrides_path, dtype=str).fillna("")
        for _, row in overrides.iterrows():
            d = date.fromisoformat(row["date"])
            if start <= d <= end:
                rows[d] = (row["is_trading"].strip() == "1", row["note"] or None)

    return [(d, t, note) for d, (t, note) in sorted(rows.items())]
