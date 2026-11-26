"""Labels for the row of trading day d (forecast at 08:30 IST on d), from adjusted prices.

For each horizon h (trading days, the holding period d .. d+h-1):
- fwd_ret_h: log return to the close of d+h-1, from the open of d (entry='open') or the close of d-1
  (entry='close')
- up_h: 1.0 if fwd_ret_h > 0 else 0.0
- fwd_vol_h: realised volatility, the root mean square of the daily close-to-close log returns of
  days d .. d+h-1
- end_h: the date of d+h-1, the last day the label looks at (used by the walk-forward gap rule)
Labels are NaN where the stock has fewer than h trading days left. Sessions listed in
`features.skip_sessions` (Diwali Muhurat) are not trading days here, as in the features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import Config
from src.store.db import read_table, skipped_sessions, write_manifest, write_table


def label_columns(horizons) -> list[str]:
    return [f"{k}_{h}" for h in horizons for k in ("fwd_ret", "up", "fwd_vol", "end")]


def make_labels(prices_adj: pd.DataFrame, horizons, entry: str = "open") -> pd.DataFrame:
    if entry not in ("open", "close"):
        raise ValueError("labels.entry must be 'open' or 'close'")
    parts = []
    for symbol, g in prices_adj.sort_values(["symbol", "trade_date"]).groupby("symbol", sort=True):
        g = g.reset_index(drop=True)
        close = g["close"].astype(float).to_numpy()
        logc = np.log(close)
        base = np.log(g["open"].astype(float).to_numpy()) if entry == "open" else np.r_[np.nan, logc[:-1]]
        r = np.r_[np.nan, np.diff(logc)]                     # close-to-close log return of each day
        r2 = pd.Series(r ** 2)
        dates = g["trade_date"].to_numpy()
        out = {"symbol": symbol, "trade_date": g["trade_date"]}
        n = len(g)
        for h in horizons:
            end_idx = np.arange(n) + h - 1
            ok = end_idx < n
            safe = np.minimum(end_idx, n - 1)
            ret = np.where(ok, logc[safe] - base, np.nan)
            # mean of r^2 over days d..d+h-1 = rolling mean ending at d+h-1, moved back h-1 rows
            vol = np.sqrt(r2.rolling(h, min_periods=h).mean().shift(-(h - 1)).to_numpy())
            out[f"fwd_ret_{h}"] = ret
            out[f"up_{h}"] = np.where(np.isnan(ret), np.nan, (ret > 0).astype(float))
            out[f"fwd_vol_{h}"] = vol
            out[f"end_{h}"] = pd.Series(np.where(ok, dates[safe], np.datetime64("NaT")))
        parts.append(pd.DataFrame(out))
    return pd.concat(parts, ignore_index=True)


def build(cfg: Config) -> dict:
    prices = read_table(cfg, "prices_adj", layer="derived", columns=["symbol", "trade_date", "open", "close"])
    prices["trade_date"] = pd.to_datetime(prices["trade_date"])
    prices = prices[~prices["trade_date"].isin(skipped_sessions(cfg))]
    lab_cfg = cfg["labels"]
    horizons = list(range(1, max(lab_cfg["horizons"]) + 1))     # every step up to the longest (TFT path)
    labels = make_labels(prices, horizons, lab_cfg["entry"])
    write_table(cfg, "labels", labels)
    summary = {"rows": len(labels), "entry": lab_cfg["entry"], "horizons": horizons}
    write_manifest(cfg.derived_dir / "labels", stage="labels", **summary)
    return summary
