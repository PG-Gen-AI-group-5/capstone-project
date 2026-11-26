"""Point-in-time rules (the time rule in CLAUDE.md): when a record may first inform a forecast.

The forecast for trading day d is made at `as_of(d)` = 08:30 IST on d. A record published at t may
inform it only if t < as_of(d), so a record's day 0 is the first trading day whose as_of is later
than t. Features built from dated records (announcements, results) place each record on its day 0.
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
