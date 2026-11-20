"""Return helpers for the EDA notebooks (and later the labels and baselines).

Everything works on wide frames: one row per trading day, one column per symbol. Log returns are
built from consecutive closes; days listed in `mask` (corporate-action ex-dates while adjusted
prices don't exist yet) are set to NaN, and any multi-day window that contains a masked day is NaN
too, so no return ever spans an unadjusted price change.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def to_wide(prices: pd.DataFrame, col: str) -> pd.DataFrame:
    """Long prices (symbol, trade_date, ...) -> date x symbol frame of `col`."""
    w = prices.pivot(index="trade_date", columns="symbol", values=col)
    w.index = pd.to_datetime(w.index)
    return w.sort_index()


def ex_date_mask(index: pd.DatetimeIndex, columns, corporate_actions: pd.DataFrame,
                 ca_types: tuple[str, ...] | None = None) -> pd.DataFrame:
    """True on each (day, symbol) that is an ex-date of a corporate action (optionally only `ca_types`)."""
    ca = corporate_actions if ca_types is None else corporate_actions[corporate_actions["ca_type"].isin(ca_types)]
    mask = pd.DataFrame(False, index=index, columns=columns)
    for sym, d in zip(ca["symbol"], pd.to_datetime(ca["ex_date"])):
        if sym in mask.columns and d in mask.index:
            mask.loc[d, sym] = True
    return mask


def log_returns(close: pd.DataFrame, mask: pd.DataFrame | None = None) -> pd.DataFrame:
    """Daily close-to-close log returns; NaN on the first day of each series and where `mask` is True."""
    r = np.log(close / close.shift(1))
    return r.mask(mask) if mask is not None else r


def gap_intraday(open_: pd.DataFrame, close: pd.DataFrame, mask: pd.DataFrame | None = None):
    """(overnight gap = log(open_d / close_{d-1}), intraday = log(close_d / open_d)).

    The gap is masked on ex-dates (it contains the price adjustment); the intraday move is not.
    """
    gap = np.log(open_ / close.shift(1))
    intra = np.log(close / open_)
    return (gap.mask(mask) if mask is not None else gap), intra


def forward_sum(r: pd.DataFrame, h: int) -> pd.DataFrame:
    """Value at day d = sum of r over d .. d+h-1 (NaN if any of them is NaN or missing)."""
    return r.rolling(h, min_periods=h).sum().shift(-(h - 1))


def label_returns(close: pd.DataFrame, open_: pd.DataFrame, mask: pd.DataFrame | None, h: int,
                  kind: str = "close") -> pd.DataFrame:
    """h-day log return attached to the forecast day d.

    kind="close": close(d-1) -> close(d+h-1), the usual time-series convention.
    kind="open":  open(d)    -> close(d+h-1), which matches a trade entered at the open of d.
    """
    r = log_returns(close, mask)
    if kind == "close":
        return forward_sum(r, h)
    if kind == "open":
        intra = np.log(close / open_)
        return intra if h == 1 else intra + forward_sum(r, h - 1).shift(-1)
    raise ValueError(f"kind must be 'close' or 'open', not {kind!r}")


def realised_vol(r: pd.DataFrame, window: int) -> pd.DataFrame:
    """Trailing std of daily log returns over `window` days ending on d (not annualised)."""
    return r.rolling(window, min_periods=window).std()


def forward_vol(r: pd.DataFrame, h: int) -> pd.DataFrame:
    """Std of daily log returns over d .. d+h-1 (the volatility label), attached to day d."""
    return r.rolling(h, min_periods=h).std().shift(-(h - 1))


def every_nth(df: pd.DataFrame, h: int) -> pd.DataFrame:
    """Keep every h-th row, so h-day windows don't overlap (for distribution statistics)."""
    return df.iloc[::h]


def moments(x) -> dict:
    """Mean, std, skew, excess kurtosis and tail shares of a sample, vs a normal with the same std."""
    x = pd.Series(np.asarray(x, dtype=float).ravel()).dropna()
    z = (x - x.mean()) / x.std()
    return {
        "n": len(x), "mean": x.mean(), "std": x.std(), "skew": stats.skew(x), "excess_kurt": stats.kurtosis(x),
        "share_beyond_3sd": float((z.abs() > 3).mean()), "normal_beyond_3sd": float(2 * stats.norm.sf(3)),
        "p01": x.quantile(0.01), "p99": x.quantile(0.99),
    }


def beta_corr(r: pd.DataFrame, market: pd.Series, min_obs: int = 250) -> pd.DataFrame:
    """Per-column beta and correlation of daily returns against the market return, on common days."""
    rows = {}
    for c in r.columns:
        d = pd.concat([r[c], market], axis=1, keys=["s", "m"]).dropna()
        if len(d) < min_obs:
            continue
        cov = np.cov(d["s"], d["m"])
        rows[c] = {"beta": cov[0, 1] / cov[1, 1], "corr": d["s"].corr(d["m"]), "obs": len(d)}
    return pd.DataFrame(rows).T
