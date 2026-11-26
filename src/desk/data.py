"""What the desk reads from the lake and data/derived, loaded once per process.

- `run`: the agents' RunData (features, announcements, catalysts, base forecasts) over the desk window
- `bars`: raw, as-traded OHLC per day (what orders fill at)
- `chart`: adjusted OHLC and the adjustment factor, for charts. A chart drawn as of day L divides by L's factor,
  so it shows prices on L's share basis and never reveals a split that happens after L.
- `vol`: the LSTM's volatility forecasts (its robust output), for the forecast band

The window is the union of the periods in `desk.periods` (by default the inner folds and validation, where
walk-forward forecasts exist). The test period stays out (task 18.4).
"""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from src.agents.context import RunData
from src.backtest.run import raw_bars
from src.config import Config
from src.store.db import read_table, universe

DESK_DEFAULTS = {"db": "app/desk.sqlite", "periods": ["inner", "validation"], "capital": 1_000_000,
                 "history_days": 400, "vol_model": "lstm"}


def desk_settings(cfg: Config) -> dict:
    return {**DESK_DEFAULTS, **(cfg.get("desk") or {})}


def desk_window(cfg: Config) -> tuple[date, date]:
    """First and last day a desk may replay."""
    s = desk_settings(cfg)
    if "test" in s["periods"]:
        raise ValueError("desk.periods must not include the test period (task 18.4 produces test results once)")
    spans = [cfg.period(p) for p in s["periods"]]
    start = min(pd.Timestamp(a) for a, _ in spans)
    end = max(pd.Timestamp(b) for _, b in spans if b is not None)
    return start.date(), end.date()


@dataclass
class DeskData:
    cfg: Config
    run: RunData
    bars: pd.DataFrame                       # symbol, trade_date, open, high, low, close (raw)
    chart: pd.DataFrame                      # symbol, trade_date, open, high, low, close (adjusted), volume, adj_factor
    vol: pd.DataFrame                        # symbol, trade_date, horizon_days, vol_forecast
    names: dict[str, str]
    window: tuple[date, date]
    days: pd.DatetimeIndex = field(init=False)

    def __post_init__(self):
        lo, hi = pd.Timestamp(self.window[0]), pd.Timestamp(self.window[1])
        d = self.run.days
        self.days = d[(d >= lo) & (d <= hi)]
        b = self.bars.assign(trade_date=pd.to_datetime(self.bars["trade_date"]))
        self.bars = b[(b["trade_date"] >= lo - pd.Timedelta(days=31)) & (b["trade_date"] <= hi + pd.Timedelta(days=31))]
        self._bars_by_day = {k: g.set_index("symbol") for k, g in self.bars.groupby("trade_date")}
        c = self.chart.assign(trade_date=pd.to_datetime(self.chart["trade_date"])).sort_values(["symbol", "trade_date"])
        self._chart = {s: g.reset_index(drop=True) for s, g in c.groupby("symbol")}
        v = self.vol
        wide = v.pivot_table(index=["symbol", "trade_date"], columns="horizon_days", values="vol_forecast") \
            if len(v) else pd.DataFrame()
        self._vol = {k: {int(h): float(x) for h, x in row.items() if pd.notna(x)}
                     for k, row in wide.to_dict("index").items()}

    @classmethod
    def load(cls, cfg: Config) -> DeskData:
        s = desk_settings(cfg)
        start, end = desk_window(cfg)
        run = RunData.load(cfg, pd.Timestamp(start) - pd.Timedelta(days=int(s["history_days"])), end)
        bars = raw_bars(cfg)
        chart = read_table(cfg, "prices_adj", layer="derived",
                           columns=["symbol", "trade_date", "open", "high", "low", "close", "volume", "adj_factor"])
        names = dict(zip(*[universe(cfg)[c] for c in ("symbol", "company_name")]))
        return cls(cfg, run, bars, chart, load_vol(cfg, s["vol_model"], s["periods"]), names, (start, end))

    # ---- trading days
    def first_day_from(self, d) -> date | None:
        i = self.days.searchsorted(pd.Timestamp(d))
        return self.days[i].date() if i < len(self.days) else None

    def next_day(self, d) -> date | None:
        i = self.days.searchsorted(pd.Timestamp(d), side="right")
        return self.days[i].date() if i < len(self.days) else None

    def prev_close_day(self, d) -> date | None:
        """The last trading day before d (whose close is known at d's 08:30)."""
        all_days = self.run.days
        i = all_days.searchsorted(pd.Timestamp(d))
        return all_days[i - 1].date() if i > 0 else None

    def held_days(self, entry, exit_) -> int:
        d = self.run.days
        return int(((d >= pd.Timestamp(entry)) & (d <= pd.Timestamp(exit_))).sum())

    # ---- prices
    def bars_on(self, d) -> pd.DataFrame | None:
        return self._bars_by_day.get(pd.Timestamp(d))

    def raw_close(self, symbol: str, d) -> float | None:
        b = self.bars_on(d)
        return float(b.at[symbol, "close"]) if b is not None and symbol in b.index else None

    def chart_frame(self, symbol: str, through, days: int = 120) -> pd.DataFrame:
        """Daily candles up to `through` on its share basis, with EMA20/EMA50 from the full history."""
        g = self._chart.get(symbol)
        if g is None:
            return pd.DataFrame()
        g = g[g["trade_date"] <= pd.Timestamp(through)]
        if g.empty:
            return g
        scale = 1.0 / float(g["adj_factor"].iloc[-1])
        out = g[["trade_date", "open", "high", "low", "close", "volume"]].copy()
        for k in ("open", "high", "low", "close"):
            out[k] = out[k] * scale
        out["ema20"] = out["close"].ewm(span=20, adjust=False).mean()
        out["ema50"] = out["close"].ewm(span=50, adjust=False).mean()
        return out.tail(days).reset_index(drop=True)

    def vol_forecast(self, symbol: str, d) -> dict[int, float]:
        return dict(self._vol.get((symbol, pd.Timestamp(d)), {}))

    def symbols(self) -> list[str]:
        return sorted(self.names)


def load_vol(cfg: Config, model: str, periods: list[str]) -> pd.DataFrame:
    folder = cfg.derived_dir / "forecasts" / model
    frames = [pd.read_parquet(folder / f"{p}.parquet", columns=["symbol", "trade_date", "horizon_days", "vol_forecast"])
              for p in periods if (folder / f"{p}.parquet").exists()]
    if not frames:
        return pd.DataFrame(columns=["symbol", "trade_date", "horizon_days", "vol_forecast"])
    v = pd.concat(frames, ignore_index=True)
    v["trade_date"] = pd.to_datetime(v["trade_date"])
    return v.drop_duplicates(["symbol", "trade_date", "horizon_days"], keep="last")


def forecast_band(reference: float, horizon: int, vol: dict[int, float], exp_return: dict[int, float] | None,
                  fallback_sigma: float | None) -> list[dict]:
    """Points t = 0..horizon trading days ahead: midpoint and 50% / 80% ranges, log-normal with daily sigma.
    sigma(t) is the forecast for the shortest saved horizon >= t; the midpoint follows the base model's expected
    return, interpolated between its horizons."""
    if not vol and not fallback_sigma:
        return []
    hs = sorted(vol)
    er = {0: 0.0, **(exp_return or {})}
    ek = sorted(er)
    pts = []
    for t in range(horizon + 1):
        sigma = next((vol[h] for h in hs if h >= t), vol[hs[-1]] if hs else fallback_sigma) if hs else fallback_sigma
        drift = float(np.interp(t, ek, [er[k] for k in ek])) if len(ek) > 1 else 0.0
        mid = float(reference) * (1 + drift)
        w = float(sigma) * math.sqrt(t)
        pts.append({"t": t, "mid": round(mid, 2),
                    "lo50": round(mid * math.exp(-0.6745 * w), 2), "hi50": round(mid * math.exp(0.6745 * w), 2),
                    "lo80": round(mid * math.exp(-1.2816 * w), 2), "hi80": round(mid * math.exp(1.2816 * w), 2)})
    return pts


_shared: DeskData | None = None
_lock = threading.Lock()


def shared(cfg: Config) -> DeskData:
    """The process-wide DeskData (loading takes a few seconds, so it happens once)."""
    global _shared
    with _lock:
        if _shared is None:
            _shared = DeskData.load(cfg)
        return _shared
