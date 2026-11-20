"""Daily technical indicators (task 10.1) for one stock or index, valued at each day's close.

Every indicator is a return or a ratio, so a later corporate-action adjustment never changes a
past value. Callers shift the result by one trading day so the row for day d holds values known
at the 08:30 IST forecast on d.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STOCK_FEATURES = [
    "ret_1", "ret_5", "ret_10", "ret_20", "ret_60",
    "ema20_gap", "ema50_gap", "ema200_gap", "ema20_50",
    "rsi14", "atr14_pct", "bb_pctb", "bb_width",
    "logvol_z20", "dist_52w_high", "rvol_5", "rvol_20", "gap_1", "range_1",
]


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def wilder(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def rms(s: pd.Series, n: int) -> pd.Series:
    """Root mean square over n values: realised volatility of zero-mean daily returns."""
    return np.sqrt((s ** 2).rolling(n, min_periods=n).mean())


def stock_indicators(bars: pd.DataFrame) -> pd.DataFrame:
    """bars: one symbol sorted by date with adjusted open, high, low, close, volume."""
    o, h, low, c, v = (bars[k].astype(float) for k in ("open", "high", "low", "close", "volume"))
    logc = np.log(c)
    r = logc.diff()
    out = pd.DataFrame(index=bars.index)
    out["ret_1"] = r
    for n in (5, 10, 20, 60):
        out[f"ret_{n}"] = logc - logc.shift(n)

    e20, e50, e200 = ema(c, 20), ema(c, 50), ema(c, 200)
    out["ema20_gap"] = c / e20 - 1
    out["ema50_gap"] = c / e50 - 1
    out["ema200_gap"] = c / e200 - 1
    out["ema20_50"] = e20 / e50 - 1

    delta = c.diff()
    gain, loss = wilder(delta.clip(lower=0), 14), wilder(-delta.clip(upper=0), 14)
    out["rsi14"] = (1 - 1 / (1 + gain / loss.replace(0, np.nan))).fillna(1.0).where(gain.notna())

    prev = c.shift(1)
    tr = pd.concat([h - low, (h - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    out["atr14_pct"] = wilder(tr, 14) / c

    ma20, sd20 = c.rolling(20).mean(), c.rolling(20).std()
    out["bb_pctb"] = (c - (ma20 - 2 * sd20)) / (4 * sd20)
    out["bb_width"] = 4 * sd20 / ma20

    lv = np.log1p(v)
    out["logvol_z20"] = (lv - lv.rolling(20).mean()) / lv.rolling(20).std()
    out["dist_52w_high"] = c / h.rolling(252, min_periods=60).max() - 1
    out["rvol_5"] = rms(r, 5)
    out["rvol_20"] = rms(r, 20)
    out["gap_1"] = np.log(o / prev)
    out["range_1"] = (h - low) / c
    return out.replace([np.inf, -np.inf], np.nan)
