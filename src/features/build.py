"""Feature table for the models: one row per stock per trading day d, holding only what was known
at the 08:30 IST forecast on d.

- stock and market indicators: values at the close of the previous trading day (shift by one row)
- macro: the latest FRED value whose known_from is at or before 08:30 IST on d
- events: results releases and important announcements whose day 0 is at or before d (events.py)
- calendar: facts about d itself that are known in advance (weekday, month, holiday gaps, F&O expiry)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import Config
from src.features.events import (
    EVENT_FEATURES,
    empty_event_features,
    event_features,
    results_records,
    results_releases,
)
from src.features.indicators import STOCK_FEATURES, ema, rms, stock_indicators
from src.store.db import read_table, skipped_sessions, write_manifest, write_table
from src.store.pit import as_of_utc

MARKET_FEATURES = ["nifty_ret_1", "nifty_ret_5", "nifty_ret_20", "nifty_ema200_gap", "nifty_rvol_20",
                   "vix", "vix_chg_5", "rs_nifty_20", "beta_nifty"]
MACRO_FEATURES = ["us10y", "us10y_chg_5", "usvix", "usvix_chg_5", "usdinr_ret_5", "brent_ret_5", "dxy_ret_5"]
CALENDAR_FEATURES = ["dow", "month_sin", "month_cos", "gap_days", "next_gap_days", "days_to_expiry"]
FEATURES = STOCK_FEATURES + MARKET_FEATURES + MACRO_FEATURES + EVENT_FEATURES + CALENDAR_FEATURES
DEFAULT_EXPIRY_WEEKDAY = {"2016-01-01": 3}

# FRED series -> (feature name, transform over the last 5 observations)
MACRO_SERIES = {
    "DGS10": ("us10y", "diff"),
    "VIXCLS": ("usvix", "diff"),
    "DEXINUS": ("usdinr", "logret"),
    "DCOILBRENTEU": ("brent", "logret"),
    "DTWEXBGS": ("dxy", "logret"),
}


def expiry_days(days: pd.Series, expiry_weekday: dict) -> pd.DatetimeIndex:
    """Monthly F&O expiry in each month of `days` (sorted dates): the last expiry weekday of the month,
    or the latest date in `days` before it when that day is missing (a holiday)."""
    rules = sorted((pd.Timestamp(str(k)), int(v)) for k, v in expiry_weekday.items())
    d = pd.Series(pd.to_datetime(days)).reset_index(drop=True)
    out = []
    for month_start in pd.date_range(d.iloc[0].replace(day=1), d.iloc[-1], freq="MS"):
        applicable = [wd for start, wd in rules if start <= month_start]
        weekday = applicable[-1] if applicable else rules[0][1]
        month_end = month_start + pd.offsets.MonthEnd(0)
        target = month_end - pd.Timedelta(days=(month_end.day_of_week - weekday) % 7)
        candidates = d[(d >= month_start) & (d <= target)]
        if len(candidates):
            out.append(candidates.iloc[-1])
    return pd.DatetimeIndex(out)


def calendar_table(trading_days: pd.Series, extra_days: int = 30, expiry_weekday: dict | None = None) -> pd.DataFrame:
    """Known-in-advance calendar features for every trading day, plus `extra_days` future weekdays
    (holidays unknown) so sequence models can look past the last date."""
    days = pd.Series(sorted(pd.to_datetime(trading_days).unique()))
    future = pd.bdate_range(days.iloc[-1] + pd.Timedelta(days=1), periods=extra_days + 30)   # +30: next expiry
    allday = pd.concat([days, pd.Series(future)], ignore_index=True)
    cal = pd.DataFrame({"trade_date": allday, "is_future": [False] * len(days) + [True] * len(future)})
    cal["dow"] = cal["trade_date"].dt.day_of_week / 4.0
    month = cal["trade_date"].dt.month
    cal["month_sin"] = np.sin(2 * np.pi * month / 12)
    cal["month_cos"] = np.cos(2 * np.pi * month / 12)
    cal["gap_days"] = cal["trade_date"].diff().dt.days.fillna(1.0)
    cal["next_gap_days"] = (-cal["trade_date"].diff(-1)).dt.days.fillna(1.0)
    expiries = cal.index[cal["trade_date"].isin(expiry_days(cal["trade_date"], expiry_weekday or DEFAULT_EXPIRY_WEEKDAY))]
    nxt = np.searchsorted(expiries.to_numpy(), cal.index.to_numpy(), side="left")       # next expiry at or after
    ok = nxt < len(expiries)
    cal["days_to_expiry"] = np.nan
    cal.loc[ok, "days_to_expiry"] = expiries.to_numpy()[nxt[ok]] - cal.index.to_numpy()[ok]
    return cal.iloc[:len(days) + extra_days].reset_index(drop=True)


def market_features(index: pd.DataFrame) -> pd.DataFrame:
    """Nifty 50 and India VIX features per trading day, shifted so day d holds day d-1's values."""
    nifty = index[index["index_name"] == "Nifty 50"].set_index("trade_date")["close"].sort_index()
    vix = index[index["index_name"] == "India VIX"].set_index("trade_date")["close"].sort_index()
    logn = np.log(nifty)
    r = logn.diff()
    m = pd.DataFrame(index=nifty.index)
    m["nifty_ret_1"] = r
    m["nifty_ret_5"] = logn - logn.shift(5)
    m["nifty_ret_20"] = logn - logn.shift(20)
    m["nifty_ema200_gap"] = nifty / ema(nifty, 200) - 1
    m["nifty_rvol_20"] = rms(r, 20)
    m["vix"] = vix.reindex(m.index) / 100
    m["vix_chg_5"] = np.log(m["vix"] / m["vix"].shift(5))
    unshifted = m[["nifty_ret_1", "nifty_ret_20"]].copy()
    m = m.shift(1)
    m["nifty_ret_1_today"] = unshifted["nifty_ret_1"]      # for stock-vs-market features, before the stock shift
    m["nifty_ret_20_today"] = unshifted["nifty_ret_20"]
    return m.rename_axis("trade_date").reset_index()


def rolling_beta(r: pd.Series, market: pd.Series, n: int) -> pd.Series:
    """Beta of daily returns r to market returns over the last n days: cov / var."""
    xy = (r * market).rolling(n, min_periods=n).mean()
    x = r.rolling(n, min_periods=n).mean()
    y = market.rolling(n, min_periods=n).mean()
    var = (market ** 2).rolling(n, min_periods=n).mean() - y ** 2
    return (xy - x * y) / var.where(var > 0)


def macro_features(macro: pd.DataFrame, trade_dates: pd.Series, hhmm: str) -> pd.DataFrame:
    """Latest FRED values known at 08:30 IST on each trading day."""
    out = pd.DataFrame({"trade_date": sorted(pd.to_datetime(trade_dates).unique())})
    out["as_of"] = as_of_utc(out["trade_date"], hhmm)
    for series, (name, how) in MACRO_SERIES.items():
        s = macro[macro["series_id"] == series].dropna(subset=["value"]).sort_values("obs_date")
        if s.empty:
            out[name] = np.nan
            out[f"{name}_{'chg' if how == 'diff' else 'ret'}_5"] = np.nan
            continue
        v = s["value"].astype(float)
        f = pd.DataFrame({"known_from": pd.to_datetime(s["known_from"], utc=True)})
        if how == "diff":
            f[name] = v.to_numpy()
            f[f"{name}_chg_5"] = (v - v.shift(5)).to_numpy()
        else:
            f[f"{name}_ret_5"] = np.log(v / v.shift(5)).to_numpy()
        f = f.sort_values("known_from")
        out = pd.merge_asof(out.sort_values("as_of"), f, left_on="as_of", right_on="known_from",
                            direction="backward").drop(columns="known_from")
    keep = ["trade_date"] + [c for c in MACRO_FEATURES if c in out.columns]
    return out[keep]


def build_features(prices_adj: pd.DataFrame, index: pd.DataFrame, macro: pd.DataFrame,
                   min_history: int = 200, hhmm: str = "08:30", *, events: pd.DataFrame | None = None,
                   beta_window: int = 60, expiry_weekday: dict | None = None,
                   days_since_cap: int = 90) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (features, calendar). Features: symbol, trade_date, as_of, history, ready + FEATURES.

    events: EVENT_FEATURES per (symbol, trade_date) from events.event_features; None = no event data.
    """
    index = index.assign(trade_date=pd.to_datetime(index["trade_date"]))
    market = market_features(index)
    parts = []
    for symbol, bars in prices_adj.sort_values(["symbol", "trade_date"]).groupby("symbol", sort=True):
        bars = bars.reset_index(drop=True)
        ind = stock_indicators(bars)
        ind["trade_date"] = bars["trade_date"]
        ind = ind.merge(market[["trade_date", "nifty_ret_1_today", "nifty_ret_20_today"]], on="trade_date", how="left")
        ind["rs_nifty_20"] = ind["ret_20"] - ind["nifty_ret_20_today"]
        ind["beta_nifty"] = rolling_beta(ind["ret_1"], ind["nifty_ret_1_today"], beta_window)
        known = ind[STOCK_FEATURES + ["rs_nifty_20", "beta_nifty"]].shift(1)    # day d holds day d-1's close
        known.insert(0, "trade_date", bars["trade_date"])
        known.insert(0, "symbol", symbol)
        known["history"] = np.arange(len(bars))
        parts.append(known)
    feats = pd.concat(parts, ignore_index=True)
    feats = feats.merge(market.drop(columns=["nifty_ret_1_today", "nifty_ret_20_today"]), on="trade_date", how="left")
    feats = feats.merge(macro_features(macro, index["trade_date"], hhmm), on="trade_date", how="left")
    if events is None:
        events = empty_event_features(feats, days_since_cap)
    events = events.assign(trade_date=pd.to_datetime(events["trade_date"]), symbol=events["symbol"].astype(str))
    feats = feats.merge(events[["symbol", "trade_date"] + EVENT_FEATURES], on=["symbol", "trade_date"], how="left")
    cal = calendar_table(index["trade_date"], expiry_weekday=expiry_weekday)
    feats = feats.merge(cal[["trade_date"] + CALENDAR_FEATURES], on="trade_date", how="left")
    feats["as_of"] = as_of_utc(feats["trade_date"], hhmm)
    feats["ready"] = feats["history"] >= min_history
    cols = ["symbol", "trade_date", "as_of", "history", "ready"] + FEATURES
    return feats[cols].sort_values(["symbol", "trade_date"]).reset_index(drop=True), cal


def drop_sessions(df: pd.DataFrame, skip: pd.DatetimeIndex) -> pd.DataFrame:
    return df[~pd.to_datetime(df["trade_date"]).isin(skip)].reset_index(drop=True)


def add_placeholder_day(prices: pd.DataFrame, index: pd.DataFrame, day) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Append an empty row for a trading day whose prices are not out yet (live forecasting).

    Every stock and index that traded on the last day gets a row for `day` with no prices. Because each
    feature row holds the previous close (shift by one row), the row for `day` is complete: it is exactly
    the row the build will produce once `day`'s prices arrive.
    """
    d = pd.Timestamp(day)
    prices = prices.assign(trade_date=pd.to_datetime(prices["trade_date"]))
    index = index.assign(trade_date=pd.to_datetime(index["trade_date"]))
    if d <= prices["trade_date"].max():
        raise ValueError(f"{d.date()} already has prices; a placeholder day must come after the last one")
    last = prices["trade_date"].max()
    syms = prices.loc[prices["trade_date"] == last, "symbol"].unique()
    prices = pd.concat([prices, pd.DataFrame({"symbol": syms, "trade_date": d})], ignore_index=True)
    names = index.loc[index["trade_date"] == index["trade_date"].max(), "index_name"].unique()
    index = pd.concat([index, pd.DataFrame({"index_name": names, "trade_date": d, "close": np.nan})], ignore_index=True)
    return prices, index


def build(cfg: Config, extra_day=None) -> dict:
    """The features table. `extra_day`: also build the row for this next trading day, before its prices
    exist (the live path, into its own data folder; see add_placeholder_day)."""
    f = cfg["features"]
    hhmm = cfg.get("forecast_time_ist", "08:30")
    skip = skipped_sessions(cfg)
    prices = read_table(cfg, "prices_adj", layer="derived")
    prices["trade_date"] = pd.to_datetime(prices["trade_date"])
    prices = drop_sessions(prices, skip)
    index = drop_sessions(read_table(cfg, "index_daily", columns=["index_name", "trade_date", "close"]), skip)
    macro = read_table(cfg, "macro", columns=["series_id", "obs_date", "value", "known_from"])
    if extra_day is not None:
        prices, index = add_placeholder_day(prices, index, extra_day)

    ev_cfg = f.get("events", {})
    cap = int(ev_cfg.get("days_since_cap", 90))
    days = pd.to_datetime(index["trade_date"]).drop_duplicates().sort_values()
    announcements = read_table(cfg, "announcements",
                               columns=["symbol", "category", "summary", "published_at", "is_important"])
    results = read_table(cfg, "financial_results", columns=["symbol", "broadcast_at"])
    records = results_records(results, announcements, ev_cfg.get("results_categories", []))
    releases = results_releases(records, days, int(ev_cfg.get("results_merge_days", 10)), hhmm)
    events = event_features(prices[["symbol", "trade_date"]], releases, announcements, days, cap, hhmm)

    feats, cal = build_features(prices, index, macro, f["min_history"], hhmm, events=events,
                                beta_window=int(f.get("beta_window", 60)), expiry_weekday=f.get("expiry_weekday"),
                                days_since_cap=cap)
    write_table(cfg, "features_daily", feats)
    write_table(cfg, "calendar", cal)
    write_table(cfg, "results_releases", releases)
    summary = {"rows": len(feats), "ready_rows": int(feats["ready"].sum()), "features": len(FEATURES),
               "first_ready": str(feats.loc[feats["ready"], "trade_date"].min().date()),
               "last_date": str(feats["trade_date"].max().date()),
               "skipped_sessions": [str(d.date()) for d in skip], "results_releases": len(releases)}
    write_manifest(cfg.derived_dir / "features_daily", stage="features", feature_list=FEATURES, **summary)
    return summary

