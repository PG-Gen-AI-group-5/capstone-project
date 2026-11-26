"""Point-in-time rules (the time rule in CLAUDE.md): when a record may first inform a forecast.

The forecast for trading day d is made at `as_of(d)` = 08:30 IST on d. A record published at t may
inform it only if t < as_of(d), so a record's day 0 is the first trading day whose as_of is later
than t. Features built from dated records (announcements, results) place each record on its day 0.
The readers below (task 6.6) give agents only what was public at as_of.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

IST = "Asia/Kolkata"


def as_of_utc(trade_dates: pd.Series, hhmm: str = "08:30") -> pd.Series:
    """The forecast time (hhmm IST) on each trading day, as UTC timestamps."""
    h, m = map(int, hhmm.split(":"))
    local = pd.to_datetime(pd.Series(trade_dates)).dt.normalize() + pd.Timedelta(hours=h, minutes=m)
    return local.dt.tz_localize(IST).dt.tz_convert("UTC")


def to_utc(ts: pd.Series) -> pd.Series:
    """Timezone-aware UTC; naive timestamps are read as IST."""
    t = pd.to_datetime(pd.Series(ts))
    if not isinstance(t.dtype, pd.DatetimeTZDtype):
        t = t.dt.tz_localize(IST)
    return t.dt.tz_convert("UTC")


def first_usable_day(published_at: pd.Series, days, hhmm: str = "08:30") -> pd.Series:
    """Day 0 of each record: the first trading day d in `days` with published_at < as_of(d).

    NaT when the record falls outside the calendar: published after the last day's as_of, or before
    the first day's date (its true day 0 can't be known from this calendar).
    """
    days = pd.DatetimeIndex(sorted(pd.to_datetime(pd.Series(days)).unique()))
    t = to_utc(published_at)
    cut = as_of_utc(pd.Series(days), hhmm).dt.tz_localize(None).to_numpy("datetime64[ns]")
    tt = t.dt.tz_localize(None).to_numpy("datetime64[ns]")
    idx = np.searchsorted(cut, tt, side="right")                       # first as_of strictly after t
    first_day_start = pd.Timestamp(days[0]).tz_localize(IST).tz_convert("UTC").tz_localize(None)
    valid = t.notna().to_numpy() & (idx < len(days)) & (tt >= np.datetime64(first_day_start, "ns"))
    out = np.full(len(t), np.datetime64("NaT"), dtype="datetime64[ns]")
    out[valid] = days.values[idx[valid]]
    return pd.Series(out, index=pd.Series(published_at).index)


# ---------------------------------------------------------------- readers (task 6.6)

def _utc(t) -> pd.Timestamp:
    t = pd.Timestamp(t)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


class AnnouncementIndex:
    """Announcements for point-in-time lookups: `as_of` returns only records with published_at < as_of."""

    def __init__(self, announcements: pd.DataFrame):
        df = announcements.assign(published_at=pd.to_datetime(announcements["published_at"], utc=True))
        df = df.dropna(subset=["published_at"]).sort_values("published_at", kind="stable")
        self.columns = list(df.columns)
        self._by_symbol = {s: g.reset_index(drop=True) for s, g in df.groupby("symbol")}

    def as_of(self, symbol: str, as_of, since=None) -> pd.DataFrame:
        g = self._by_symbol.get(symbol)
        if g is None:
            return pd.DataFrame(columns=self.columns)
        times = g["published_at"]
        hi = int(times.searchsorted(_utc(as_of), side="left"))           # strictly before as_of
        lo = 0 if since is None else int(times.searchsorted(_utc(since), side="left"))
        return g.iloc[lo:hi]


def prices_as_of(prices: pd.DataFrame, symbols: list[str], as_of, lookback_days: int = 400) -> pd.DataFrame:
    """Daily bars public at as_of: the bhavcopy for day t is published at 18:00 IST on t, so at 08:30 IST on
    day d the latest bar is d-1 (trade_date strictly before as_of's IST date)."""
    day = _utc(as_of).tz_convert(IST).tz_localize(None).normalize()
    d = pd.to_datetime(prices["trade_date"])
    keep = prices["symbol"].isin(symbols) & (d < day) & (d >= day - pd.Timedelta(days=lookback_days))
    return prices[keep]
