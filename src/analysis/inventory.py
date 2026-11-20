"""Data inventory and quality checks for EDA notebook 01.

Pure functions over DataFrames, so they can be tested on small fixtures. The notebook loads the
lake through the DuckDB views (`load_table`) and the tracker through SQLite, then calls these.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time

import numpy as np
import pandas as pd

IST = "Asia/Kolkata"
FORECAST_TIME = time(8, 30)  # forecast for day d is made at 08:30 IST
MARKET_CLOSE = time(15, 30)
SESSIONS = ("before 08:30", "08:30-15:30", "after 15:30")


@dataclass(frozen=True)
class TableSpec:
    """How to read one lake table for the inventory.

    `sql` must return the table's columns; JSON-only tables add parsed `date` / `published_at`.
    `date_col` is the column used for first/last date and the stock x year coverage.
    """

    name: str
    date_col: str
    keys: list[str]
    symbol_col: str | None = "symbol"
    time_cols: list[str] = field(default_factory=list)
    sql: str | None = None

    def query(self) -> str:
        return self.sql or f'SELECT * FROM "{self.name}"'


_SH_SQL = """
SELECT *,
  try_strptime(json_extract_string(record_json, '$.date'), '%d-%b-%Y')::DATE AS period_end,
  try_strptime(json_extract_string(record_json, '$.broadcastDate'), '%d-%b-%Y %H:%M:%S') AS broadcast_ist
FROM shareholding
"""
_PLEDGE_SQL = """
SELECT *,
  try_strptime(json_extract_string(record_json, '$.broadcastDt'), '%d-%b-%Y %H:%M:%S') AS broadcast_ist
FROM pledges
"""
_TEXT_SQL = """
SELECT t.*, a.published_at
FROM announcement_text t LEFT JOIN announcements a USING (doc_id)
"""

TABLES: dict[str, TableSpec] = {s.name: s for s in [
    TableSpec("prices_daily", "trade_date", ["symbol", "trade_date", "open", "high", "low", "close", "volume"],
              time_cols=["published_at"]),
    TableSpec("index_daily", "trade_date", ["index_name", "trade_date", "close"], symbol_col="index_name",
              time_cols=["published_at"]),
    TableSpec("corporate_actions", "ex_date", ["symbol", "ex_date", "ca_type", "adj_factor", "published_at"],
              time_cols=["published_at"]),
    TableSpec("financial_results", "period_to", ["symbol", "period_to", "broadcast_at", "published_at"],
              time_cols=["broadcast_at", "published_at"]),
    TableSpec("board_meetings", "meeting_date", ["symbol", "meeting_date", "purpose", "published_at"],
              time_cols=["announced_at", "published_at"]),
    TableSpec("announcements", "published_at", ["doc_id", "symbol", "category", "published_at"],
              time_cols=["published_at"]),
    TableSpec("announcement_text", "published_at", ["doc_id", "symbol", "text", "published_at"],
              time_cols=["published_at (from announcements)"], sql=_TEXT_SQL),
    TableSpec("insider_trades", "acq_to", ["symbol", "person_category", "quantity", "acq_to", "published_at"],
              time_cols=["published_at"]),
    TableSpec("shareholding", "period_end", ["symbol", "period_end", "broadcast_ist"],
              time_cols=["broadcastDate (in record_json)"], sql=_SH_SQL),
    TableSpec("pledges", "broadcast_ist", ["symbol", "broadcast_ist"],
              time_cols=["broadcastDt (in record_json)"], sql=_PLEDGE_SQL),
    TableSpec("fii_dii", "trade_date", ["trade_date", "category", "net_value_cr", "published_at"],
              symbol_col=None, time_cols=["published_at"]),
    TableSpec("macro", "obs_date", ["series_id", "obs_date", "value", "known_from"], symbol_col="series_id",
              time_cols=["known_from"]),
    TableSpec("news", "published_at", ["symbol", "url", "published_at", "tone"], time_cols=["published_at"]),
]}


def load_table(con, spec: TableSpec) -> pd.DataFrame:
    """Read a lake table through its DuckDB view (`con` is a duckdb connection)."""
    return con.execute(spec.query()).df()


def _as_date(s: pd.Series) -> pd.Series:
    """Dates or timestamps -> naive dates (timestamps are taken in IST)."""
    s = pd.Series(s)
    if isinstance(s.dtype, pd.DatetimeTZDtype):
        s = s.dt.tz_convert(IST).dt.tz_localize(None)
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def inventory_row(df: pd.DataFrame, spec: TableSpec) -> dict:
    """One inventory row: size, span, date/time columns, null share per key column, source."""
    dates = _as_date(df[spec.date_col]) if len(df) else pd.Series(dtype="datetime64[ns]")
    date_cols = [c for c in df.columns if c != "ingested_at" and (
        pd.api.types.is_datetime64_any_dtype(df[c]) or c.endswith(("_date", "date")) or c in ("period_to",
                                                                                              "period_from"))]
    nulls = {k: round(float(df[k].isna().mean()), 3) for k in spec.keys if k in df.columns and len(df)}
    return {
        "table": spec.name,
        "rows": len(df),
        "stocks": int(df[spec.symbol_col].nunique()) if spec.symbol_col and len(df) else None,
        "first": dates.min().date() if dates.notna().any() else None,
        "last": dates.max().date() if dates.notna().any() else None,
        "date_col": spec.date_col,
        "date_columns": ", ".join(date_cols),
        "time_columns": ", ".join(spec.time_cols),
        "null_share": ", ".join(f"{k}={v:.0%}" for k, v in nulls.items() if v > 0) or "none",
        "source": ", ".join(sorted(df["source"].dropna().astype(str).unique())) if "source" in df.columns else "",
    }


def stock_year_counts(df: pd.DataFrame, date_col: str, symbol_col: str = "symbol") -> pd.DataFrame:
    """Rows per stock per year (stocks as rows, years as columns, 0 where none)."""
    d = pd.DataFrame({"symbol": df[symbol_col], "year": _as_date(df[date_col]).dt.year}).dropna()
    out = d.groupby(["symbol", "year"]).size().unstack(fill_value=0)
    out.columns = out.columns.astype(int)
    return out.sort_index()


def trading_days(calendar: pd.DataFrame) -> pd.DatetimeIndex:
    """Trading days from the tracker's trading_calendar table (trade_date, is_trading)."""
    cal = calendar[calendar["is_trading"].astype(int) == 1]
    return pd.DatetimeIndex(pd.to_datetime(cal["trade_date"])).sort_values()


def price_coverage(prices: pd.DataFrame, days: pd.DatetimeIndex, end=None) -> pd.DataFrame:
    """Share of trading days with a price, per stock and year, counted from each stock's first day.

    NaN where the stock had not started trading yet (not a gap). `end` defaults to the last price date.
    """
    p = pd.DataFrame({"symbol": prices["symbol"], "d": pd.to_datetime(prices["trade_date"])})
    end = pd.Timestamp(end) if end is not None else p["d"].max()
    days = days[days <= end]
    rows = {}
    for sym, g in p.groupby("symbol"):
        exp = days[days >= g["d"].min()]
        have = pd.Series(exp.isin(g["d"]), index=exp)
        rows[sym] = have.groupby(have.index.year).mean()
    return pd.DataFrame(rows).T.sort_index()


def series_day_coverage(dates: pd.Series, days: pd.DatetimeIndex, start=None, end=None) -> pd.Series:
    """Share of NSE trading days per year on which the series has an observation."""
    d = set(pd.to_datetime(pd.Series(dates)).dt.normalize())
    start = pd.Timestamp(start) if start is not None else min(d)
    end = pd.Timestamp(end) if end is not None else max(d)
    exp = days[(days >= start) & (days <= end)]
    have = pd.Series(exp.isin(list(d)), index=exp)
    return have.groupby(have.index.year).mean()


def calendar_mismatch(price_dates: pd.Series, days: pd.DatetimeIndex) -> tuple[list, list]:
    """(trading days with no prices at all, price dates the calendar marks as non-trading)."""
    pd_ = pd.DatetimeIndex(pd.to_datetime(pd.Series(price_dates)).unique())
    in_range = days[(days >= pd_.min()) & (days <= pd_.max())]
    missing = sorted(d.date() for d in in_range.difference(pd_))
    extra = sorted(d.date() for d in pd_.difference(days))
    return missing, extra


def has_time(ts: pd.Series) -> pd.Series:
    """True where a timestamp carries a time of day (IST time is not exactly 00:00:00)."""
    t = pd.Series(ts)
    if isinstance(t.dtype, pd.DatetimeTZDtype):
        t = t.dt.tz_convert(IST)
    return t.notna() & ((t.dt.hour != 0) | (t.dt.minute != 0) | (t.dt.second != 0))


def ist_session(ts: pd.Series) -> pd.Series:
    """Bucket timestamps by IST time of day: before 08:30, 08:30-15:30, after 15:30 (NaN if missing)."""
    t = pd.Series(ts)
    if isinstance(t.dtype, pd.DatetimeTZDtype):
        t = t.dt.tz_convert(IST)
    mins = t.dt.hour * 60 + t.dt.minute + t.dt.second / 60
    out = pd.Series(np.where(mins < 8 * 60 + 30, SESSIONS[0], np.where(mins < 15 * 60 + 30, SESSIONS[1],
                                                                          SESSIONS[2])), index=t.index)
    return out.where(t.notna())


def first_informed_day(ts: pd.Series, days: pd.DatetimeIndex) -> pd.Series:
    """First trading day whose 08:30 IST forecast may use each record (published_at < as_of).

    Published before 08:30 IST on a trading day -> that day; otherwise the next trading day.
    """
    t = pd.Series(ts)
    if isinstance(t.dtype, pd.DatetimeTZDtype):
        t = t.dt.tz_convert(IST).dt.tz_localize(None)
    t = pd.to_datetime(t)
    day = t.dt.normalize()
    before = (t - day) < pd.Timedelta(hours=8, minutes=30)
    arr = days.values
    i_same = np.searchsorted(arr, day.values, side="left")    # first trading day >= day
    i_next = np.searchsorted(arr, day.values, side="right")   # first trading day > day
    idx = np.where(before.values, i_same, i_next)
    valid = t.notna().values & (idx < len(arr))
    out = np.full(len(t), np.datetime64("NaT"), dtype="datetime64[ns]")
    out[valid] = arr[idx[valid]]
    return pd.Series(out, index=t.index)


ADJUSTING_CA = ("split", "bonus", "rights", "demerger", "other")


def large_moves(prices: pd.DataFrame, corporate_actions: pd.DataFrame, threshold: float = 0.20) -> pd.DataFrame:
    """Raw close-to-previous-close moves above `threshold`, with the corporate action on that day, if any.

    `explained` is True when a split, bonus, rights, demerger or other action has its ex-date that day.
    """
    p = prices[["symbol", "trade_date", "close", "prev_close"]].copy()
    p["trade_date"] = pd.to_datetime(p["trade_date"])
    p["ret"] = p["close"] / p["prev_close"] - 1
    big = p[p["ret"].abs() > threshold]
    ca = corporate_actions[corporate_actions["ca_type"].isin(ADJUSTING_CA)][["symbol", "ex_date", "ca_type"]].copy()
    ca["trade_date"] = pd.to_datetime(ca["ex_date"])
    ca = ca.groupby(["symbol", "trade_date"])["ca_type"].agg(lambda s: "+".join(sorted(set(s)))).reset_index()
    out = big.merge(ca, on=["symbol", "trade_date"], how="left")
    out["explained"] = out["ca_type"].notna()
    return out.sort_values(["explained", "symbol", "trade_date"]).reset_index(drop=True)


def late_starts(prices: pd.DataFrame, universe: pd.DataFrame, start) -> pd.DataFrame:
    """Stocks whose first price is after `start`, with the universe's data_start, old_symbols and notes."""
    first = pd.to_datetime(prices.groupby("symbol")["trade_date"].min()).rename("first_price")
    late = first[first > pd.Timestamp(start)].reset_index()
    cols = [c for c in ("symbol", "data_start", "old_symbols", "notes") if c in universe.columns]
    out = late.merge(universe[cols], on="symbol", how="left")
    out["first_price"] = out["first_price"].dt.date
    return out.sort_values("first_price").reset_index(drop=True)
