"""Macro helpers for EDA notebook 04: align macro values to trading days under the time rule.

A macro value may inform the forecast for trading day d only if `known_from <= as_of`, where as_of is
08:30 IST on d (the forecast time). `macro_asof` returns, for each day, the latest such value and the
observation date it came from.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

IST = "Asia/Kolkata"


def as_of_times(days: pd.DatetimeIndex, at: str = "08:30") -> pd.Series:
    """The forecast time on each trading day, as UTC timestamps."""
    hh, mm = (int(x) for x in at.split(":"))
    t = pd.DatetimeIndex(days).normalize() + pd.Timedelta(hours=hh, minutes=mm)
    return pd.Series(t.tz_localize(IST).tz_convert("UTC"), index=pd.DatetimeIndex(days))


def macro_asof(macro: pd.DataFrame, days: pd.DatetimeIndex, at: str = "08:30") -> tuple[pd.DataFrame, pd.DataFrame]:
    """(values, obs_dates): day x series frames of the latest value with known_from <= as_of on each day.

    `macro` has columns series_id, obs_date, value, known_from (tz-aware). Rows with no value
    (e.g. US holidays) are skipped, so the previous value carries forward.
    """
    asof = as_of_times(pd.DatetimeIndex(days).rename("day"), at).rename("as_of").reset_index()
    asof = asof.sort_values("as_of")
    vals, obs = {}, {}
    for sid, g in macro.dropna(subset=["value"]).groupby("series_id"):
        g = g.sort_values("known_from")[["known_from", "obs_date", "value"]].copy()
        g["known_from"] = pd.to_datetime(g["known_from"], utc=True)
        m = pd.merge_asof(asof, g, left_on="as_of", right_on="known_from", direction="backward",
                          allow_exact_matches=True)
        m = m.set_index("day").sort_index()
        vals[sid], obs[sid] = m["value"], pd.to_datetime(m["obs_date"])
    return pd.DataFrame(vals), pd.DataFrame(obs)


def overnight_change(values: pd.DataFrame, obs_dates: pd.DataFrame, log_series=()) -> pd.DataFrame:
    """Change in each series since the previous trading day's as-of value.

    Log change for the series in `log_series`, plain difference otherwise. NaN on days where no new
    observation arrived (the as-of value is the same observation as the day before).
    """
    out = {}
    for c in values.columns:
        v = values[c]
        ch = np.log(v / v.shift(1)) if c in log_series else v - v.shift(1)
        new = obs_dates[c].ne(obs_dates[c].shift(1))
        out[c] = ch.where(new)
    return pd.DataFrame(out)


def block_bootstrap_corr(x: pd.Series, y: pd.Series, block: int = 20, n_boot: int = 2000, seed: int = 0,
                         level: float = 0.95) -> dict:
    """Pearson correlation with a moving-block bootstrap interval (keeps serial dependence within blocks)."""
    d = pd.concat([x, y], axis=1, keys=["x", "y"]).dropna()
    n = len(d)
    if n < 2 * block:
        return {"n": n, "corr": np.nan, "lo": np.nan, "hi": np.nan}
    xv, yv = d["x"].to_numpy(), d["y"].to_numpy()
    rng = np.random.default_rng(seed)
    k = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, k))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n_boot, -1)[:, :n]
    xb, yb = xv[idx], yv[idx]
    xb = xb - xb.mean(axis=1, keepdims=True)
    yb = yb - yb.mean(axis=1, keepdims=True)
    cb = (xb * yb).sum(axis=1) / np.sqrt((xb ** 2).sum(axis=1) * (yb ** 2).sum(axis=1))
    a = (1 - level) / 2
    return {"n": n, "corr": float(np.corrcoef(xv, yv)[0, 1]),
            "lo": float(np.quantile(cb, a)), "hi": float(np.quantile(cb, 1 - a))}
