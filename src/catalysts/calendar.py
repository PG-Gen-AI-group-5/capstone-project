"""Catalyst calendar (task 7): dated events known in advance, queried as of a forecast time.

- results: a board-meeting notice (`board_meetings`, from 2024) is known from its published_at; before
  notices exist, the next results are estimated from the stock's quarterly cycle (last release day 0 +
  55..70 trading days), known as soon as the last release is.
- expiry: the monthly F&O expiry from the trading calendar (always known).
- budget: Union Budget sessions in data/manual/calendar_overrides.csv.
RBI policy dates arrive with data/manual/rbi_policy.csv (task 4.3).
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from src.contracts import Catalyst

EPOCH = datetime(2000, 1, 1, tzinfo=timezone.utc)


class CatalystCalendar:
    def __init__(self, calendar: pd.DataFrame, releases: pd.DataFrame, board_meetings: pd.DataFrame | None = None,
                 overrides: pd.DataFrame | None = None, cycle_days: tuple[int, int] = (55, 70)):
        self.days = pd.DatetimeIndex(pd.to_datetime(calendar["trade_date"]).sort_values().unique())
        cal = calendar.assign(trade_date=pd.to_datetime(calendar["trade_date"]))
        self.expiries = pd.DatetimeIndex(cal.loc[cal["days_to_expiry"] == 0, "trade_date"])
        self.releases = releases.assign(day0=pd.to_datetime(releases["day0"])).sort_values("day0")
        bm = board_meetings if board_meetings is not None else pd.DataFrame(
            columns=["symbol", "meeting_date", "purpose", "published_at"])
        bm = bm[bm["purpose"].fillna("").str.contains("result", case=False)].copy()
        bm["meeting_date"] = pd.to_datetime(bm["meeting_date"])
        bm["published_at"] = pd.to_datetime(bm["published_at"], utc=True)
        self.board = bm
        ov = overrides if overrides is not None else pd.DataFrame(columns=["date", "note"])
        self.budget = pd.to_datetime(ov.loc[ov["note"].fillna("").str.contains("Budget"), "date"])
        self.cycle = cycle_days

    def query(self, symbol: str | None, start, end, as_of: datetime) -> list[Catalyst]:
        """Catalysts dated in [start, end] for a stock (plus market-wide ones) that were known at as_of."""
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        out: list[Catalyst] = []
        for d in self.expiries[(self.expiries >= start) & (self.expiries <= end)]:
            out.append(Catalyst(symbol=None, event_date=d.date(), kind="expiry", description="monthly F&O expiry",
                                known_from=EPOCH))
        for d in self.budget[(self.budget >= start) & (self.budget <= end)]:
            out.append(Catalyst(symbol=None, event_date=d.date(), kind="budget", description="Union Budget session",
                                known_from=EPOCH))
        if symbol:
            out += self._results(symbol, start, end, pd.Timestamp(as_of))
        return sorted(out, key=lambda c: c.event_date)

    def _results(self, symbol: str, start, end, as_of: pd.Timestamp) -> list[Catalyst]:
        bm = self.board[(self.board["symbol"] == symbol) & (self.board["published_at"] < as_of)
                        & self.board["meeting_date"].between(start, end)]
        if len(bm):
            return [Catalyst(symbol=symbol, event_date=r.meeting_date.date(), kind="results",
                             description=f"board meeting: {r.purpose}", known_from=r.published_at.to_pydatetime())
                    for r in bm.itertuples()]
        past = self.releases[(self.releases["symbol"] == symbol) & (self.releases["day0"] <= as_of.tz_localize(None)
                                                                     .normalize())]
        if past.empty:
            return []
        last = past["day0"].iloc[-1]
        i = self.days.searchsorted(last)
        lo, hi = i + self.cycle[0], i + self.cycle[1]
        if lo >= len(self.days):
            return []
        window_start, window_end = self.days[lo], self.days[min(hi, len(self.days) - 1)]
        if window_end < start or window_start > end:
            return []
        return [Catalyst(symbol=symbol, event_date=max(window_start, start).date(), kind="results",
                         description=f"results expected {window_start.date()}..{window_end.date()} "
                                     f"(estimated from the last release on {last.date()})",
                         known_from=None)]
