"""Feature table for the models: one row per stock per trading day d, holding only what was known
at the 08:30 IST forecast on d.

- stock and market indicators: values at the close of the previous trading day (shift by one row)
- macro: the latest FRED value whose known_from is at or before 08:30 IST on d
- calendar: facts about d itself that are known in advance (weekday, month, holiday gaps)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import Config
from src.features.indicators import STOCK_FEATURES, ema, rms, stock_indicators
from src.store.db import read_table, write_manifest, write_table

MARKET_FEATURES = ["nifty_ret_1", "nifty_ret_5", "nifty_ret_20", "nifty_ema200_gap", "nifty_rvol_20",
                   "vix", "vix_chg_5", "rs_nifty_20"]
MACRO_FEATURES = ["us10y", "us10y_chg_5", "usvix", "usvix_chg_5", "usdinr_ret_5", "brent_ret_5", "dxy_ret_5"]
CALENDAR_FEATURES = ["dow", "month_sin", "month_cos", "gap_days", "next_gap_days"]
FEATURES = STOCK_FEATURES + MARKET_FEATURES + MACRO_FEATURES + CALENDAR_FEATURES

# FRED series -> (feature name, transform over the last 5 observations)
MACRO_SERIES = {
    "DGS10": ("us10y", "diff"),
    "VIXCLS": ("usvix", "diff"),
    "DEXINUS": ("usdinr", "logret"),
    "DCOILBRENTEU": ("brent", "logret"),
    "DTWEXBGS": ("dxy", "logret"),
}


def as_of_utc(trade_dates: pd.Series, hhmm: str = "08:30") -> pd.Series:
    """08:30 IST on each trading day, as UTC timestamps."""
    h, m = map(int, hhmm.split(":"))
    local = pd.to_datetime(trade_dates).dt.normalize() + pd.Timedelta(hours=h, minutes=m)
    return local.dt.tz_localize("Asia/Kolkata").dt.tz_convert("UTC")


def calendar_table(trading_days: pd.Series, extra_days: int = 30) -> pd.DataFrame:
    """Known-in-advance calendar features for every trading day, plus `extra_days` future weekdays
    (holidays unknown) so sequence models can look past the last date."""
    days = pd.Series(sorted(pd.to_datetime(trading_days).unique()))
    future = pd.bdate_range(days.iloc[-1] + pd.Timedelta(days=1), periods=extra_days)
    allday = pd.concat([days, pd.Series(future)], ignore_index=True)
    cal = pd.DataFrame({"trade_date": allday, "is_future": [False] * len(days) + [True] * len(future)})
    cal["dow"] = cal["trade_date"].dt.day_of_week / 4.0
    month = cal["trade_date"].dt.month
    cal["month_sin"] = np.sin(2 * np.pi * month / 12)
    cal["month_cos"] = np.cos(2 * np.pi * month / 12)
    cal["gap_days"] = cal["trade_date"].diff().dt.days.fillna(1.0)
    cal["next_gap_days"] = (-cal["trade_date"].diff(-1)).dt.days.fillna(1.0)
    return cal


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
    unshifted_ret20 = m["nifty_ret_20"].copy()
    m = m.shift(1)
    m["nifty_ret_20_today"] = unshifted_ret20        # used for relative strength before the stock shift
    return m.rename_axis("trade_date").reset_index()


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
                   min_history: int = 200, hhmm: str = "08:30") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (features, calendar). Features: symbol, trade_date, as_of, history, ready + FEATURES."""
    index = index.assign(trade_date=pd.to_datetime(index["trade_date"]))
    market = market_features(index)
    parts = []
    for symbol, bars in prices_adj.sort_values(["symbol", "trade_date"]).groupby("symbol", sort=True):
        bars = bars.reset_index(drop=True)
        ind = stock_indicators(bars)
        ind["trade_date"] = bars["trade_date"]
        ind = ind.merge(market[["trade_date", "nifty_ret_20_today"]], on="trade_date", how="left")
        ind["rs_nifty_20"] = ind["ret_20"] - ind["nifty_ret_20_today"]
        known = ind[STOCK_FEATURES + ["rs_nifty_20"]].shift(1)          # day d holds day d-1's close
        known.insert(0, "trade_date", bars["trade_date"])
        known.insert(0, "symbol", symbol)
        known["history"] = np.arange(len(bars))
        parts.append(known)
    feats = pd.concat(parts, ignore_index=True)
    feats = feats.merge(market.drop(columns="nifty_ret_20_today"), on="trade_date", how="left")
    feats = feats.merge(macro_features(macro, index["trade_date"], hhmm), on="trade_date", how="left")
    cal = calendar_table(index["trade_date"])
    feats = feats.merge(cal[["trade_date"] + CALENDAR_FEATURES], on="trade_date", how="left")
    feats["as_of"] = as_of_utc(feats["trade_date"], hhmm)
    feats["ready"] = feats["history"] >= min_history
    cols = ["symbol", "trade_date", "as_of", "history", "ready"] + FEATURES
    return feats[cols].sort_values(["symbol", "trade_date"]).reset_index(drop=True), cal


def build(cfg: Config) -> dict:
    prices = read_table(cfg, "prices_adj", layer="derived")
    prices["trade_date"] = pd.to_datetime(prices["trade_date"])
    index = read_table(cfg, "index_daily", columns=["index_name", "trade_date", "close"])
    macro = read_table(cfg, "macro", columns=["series_id", "obs_date", "value", "known_from"])
    hhmm = cfg.get("forecast_time_ist", "08:30")
    feats, cal = build_features(prices, index, macro, cfg["features"]["min_history"], hhmm)
    write_table(cfg, "features_daily", feats)
    write_table(cfg, "calendar", cal)
    summary = {"rows": len(feats), "ready_rows": int(feats["ready"].sum()), "features": len(FEATURES),
               "first_ready": str(feats.loc[feats["ready"], "trade_date"].min().date()),
               "last_date": str(feats["trade_date"].max().date())}
    write_manifest(cfg.derived_dir / "features_daily", stage="features", feature_list=FEATURES, **summary)
    return summary

