"""Event study helpers (EDA notebook 03; reused by task 9.1, the event impact memory).

Conventions:
- `ar` is a wide frame of daily abnormal returns: rows = trading days (sorted), columns = symbols.
  The return on day t runs from close(t-1) to close(t).
- An event's day 0 is the first trading day whose 08:30 IST forecast may use it
  (`event_days`, which applies the time rule: published_at < 08:30 IST on day 0).
- Windows are indexed by trading-day offset (-pre .. +post). A window that leaves the data or contains
  a missing return (e.g. a dropped ex-date) is incomplete; `complete` keeps only full windows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.inventory import first_informed_day


def event_days(published_at: pd.Series, days: pd.DatetimeIndex) -> pd.Series:
    """Day 0 for each record: the first trading day it can inform (see inventory.first_informed_day)."""
    return first_informed_day(published_at, days)


def abnormal_returns(r: pd.DataFrame, market: pd.Series) -> pd.DataFrame:
    """Market-adjusted abnormal return: stock return minus the market return on the same day."""
    return r.sub(market.reindex(r.index), axis=0)


def event_windows(ar: pd.DataFrame, events: pd.DataFrame, pre: int = 5, post: int = 10,
                  symbol_col: str = "symbol", day_col: str = "event_day") -> pd.DataFrame:
    """One row per event (same index as `events`), one column per offset -pre..post.

    NaN where the offset falls outside the data, the event day is not a trading day in `ar`,
    or the symbol is unknown.
    """
    offsets = np.arange(-pre, post + 1)
    out = np.full((len(events), len(offsets)), np.nan)
    pos_all = ar.index.get_indexer(pd.to_datetime(events[day_col]))
    col_pos = {c: i for i, c in enumerate(ar.columns)}
    values = ar.to_numpy(dtype=float)
    n = len(ar)
    for k, (sym, pos) in enumerate(zip(events[symbol_col], pos_all)):
        j = col_pos.get(sym)
        if j is None or pos < 0:
            continue
        lo, hi = pos - pre, pos + post + 1
        a, b = max(lo, 0), min(hi, n)
        out[k, a - lo: b - lo] = values[a:b, j]
    return pd.DataFrame(out, index=events.index, columns=offsets)


def complete(windows: pd.DataFrame) -> pd.DataFrame:
    """Keep events whose whole window has returns."""
    return windows[windows.notna().all(axis=1)]


def car_path(windows: pd.DataFrame, start: int | None = None) -> pd.DataFrame:
    """Cumulative abnormal return from offset `start` (default: the first offset) to each later offset."""
    w = windows if start is None else windows.loc[:, windows.columns >= start]
    return w.cumsum(axis=1)


def car(windows: pd.DataFrame, start: int, end: int) -> pd.Series:
    """CAR over offsets start..end inclusive, per event."""
    return windows.loc[:, start:end].sum(axis=1, min_count=end - start + 1)


def bootstrap_means(values: pd.DataFrame, clusters: pd.Series, n_boot: int = 2000, seed: int = 0) -> np.ndarray:
    """Cluster bootstrap of column means: resample whole clusters with replacement.

    Events that share a cluster (e.g. the same calendar week of a results season) move together, so
    resampling them as a block gives honest intervals. Returns an (n_boot, n_columns) array.
    """
    v = values.to_numpy(dtype=float)
    codes, uniq = pd.factorize(clusters.reindex(values.index))
    k = len(uniq)
    if k == 0:
        return np.full((n_boot, v.shape[1]), np.nan)
    ok = ~np.isnan(v)
    sums = np.zeros((k, v.shape[1]))
    counts = np.zeros((k, v.shape[1]))
    np.add.at(sums, codes, np.where(ok, v, 0.0))
    np.add.at(counts, codes, ok.astype(float))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, k, size=(n_boot, k))
    w = np.zeros((n_boot, k))
    np.add.at(w, (np.repeat(np.arange(n_boot), k), draws.ravel()), 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (w @ sums) / (w @ counts)


def summarize(values: pd.DataFrame, clusters: pd.Series, n_boot: int = 2000, seed: int = 0,
              level: float = 0.95) -> pd.DataFrame:
    """Per column: n, mean, bootstrap interval, share of positive values."""
    b = bootstrap_means(values, clusters, n_boot, seed)
    a = (1 - level) / 2
    return pd.DataFrame({
        "n": values.notna().sum(), "mean": values.mean(),
        "lo": np.nanquantile(b, a, axis=0), "hi": np.nanquantile(b, 1 - a, axis=0),
        "share_pos": (values > 0).sum() / values.notna().sum(),
    }, index=values.columns)


def difference(values: pd.DataFrame, clusters: pd.Series, control: pd.DataFrame, control_clusters: pd.Series,
               n_boot: int = 2000, seed: int = 0, level: float = 0.95) -> pd.DataFrame:
    """Event mean minus control mean per column, with a bootstrap interval (independent resamples)."""
    b = bootstrap_means(values, clusters, n_boot, seed) - bootstrap_means(control, control_clusters, n_boot, seed + 1)
    a = (1 - level) / 2
    return pd.DataFrame({"diff": values.mean() - control.mean(),
                         "lo": np.nanquantile(b, a, axis=0), "hi": np.nanquantile(b, 1 - a, axis=0)},
                        index=values.columns)


def control_sample(ar: pd.DataFrame, events: pd.DataFrame, pre: int = 5, post: int = 10, seed: int = 0,
                   symbol_col: str = "symbol") -> pd.DataFrame:
    """Random days with the same number of events per stock, drawn from days with a complete window."""
    rng = np.random.default_rng(seed)
    width = pre + post + 1
    full = ar.notna().astype(int).rolling(width, min_periods=width).sum().eq(width).shift(-post, fill_value=False)
    rows = []
    for sym, n in events[symbol_col].value_counts().sort_index().items():
        if sym not in full.columns:
            continue
        valid = full.index[full[sym].to_numpy()]
        if len(valid) == 0:
            continue
        pick = rng.choice(len(valid), size=n, replace=n > len(valid))
        rows.append(pd.DataFrame({symbol_col: sym, "event_day": valid[np.sort(pick)]}))
    if not rows:
        return pd.DataFrame(columns=[symbol_col, "event_day"])
    return pd.concat(rows, ignore_index=True)


def week_cluster(event_day: pd.Series) -> pd.Series:
    """Calendar-week cluster id for the bootstrap."""
    return pd.to_datetime(event_day).dt.to_period("W").astype(str)
