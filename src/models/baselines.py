"""Simple baselines (tasks 16.3-16.5).

- base_rate: the training share of up moves as a constant forecast. Its direction equals a
  buy-and-hold view whenever that share is above 50%; buy-and-hold returns come from the backtester.
- ma_crossover: long when EMA 20 is above EMA 50 at the previous close, otherwise short. p_up is
  the training share of up moves in each of the two states.
- hist_vol: the last 20 days' realised volatility as the volatility forecast for every horizon.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.base import ForecastModel, forecast_rows


class BaseRate(ForecastModel):
    name = "base_rate"

    def fit(self, panel, train_mask):
        df = panel.df[train_mask]
        self.rate = {h: float(df[f"up_{h}"].mean()) for h in panel.horizons}
        self.mean_ret = {h: float(df[f"fwd_ret_{h}"].mean()) for h in panel.horizons}

    def predict(self, panel, mask):
        n = int(mask.sum())
        return pd.concat([forecast_rows(self.name, panel, mask, h, p_up=np.full(n, self.rate[h]),
                                        direction=np.full(n, 1.0 if self.rate[h] > 0.5 else -1.0),
                                        exp_return=np.full(n, self.mean_ret[h]))
                          for h in panel.horizons], ignore_index=True)

    def save(self, folder: Path):
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "params.json").write_text(json.dumps({"rate": self.rate, "mean_ret": self.mean_ret}, indent=2))


class MACrossover(ForecastModel):
    name = "ma_crossover"

    def fit(self, panel, train_mask):
        df = panel.df[train_mask]
        above = df["ema20_50"] > 0
        self.rate = {h: {"above": float(df.loc[above, f"up_{h}"].mean()),
                         "below": float(df.loc[~above, f"up_{h}"].mean())} for h in panel.horizons}

    def predict(self, panel, mask):
        above = (panel.df.loc[mask, "ema20_50"] > 0).to_numpy()
        out = []
        for h in panel.horizons:
            p = np.where(above, self.rate[h]["above"], self.rate[h]["below"])
            out.append(forecast_rows(self.name, panel, mask, h, p_up=p, direction=np.where(above, 1.0, -1.0)))
        return pd.concat(out, ignore_index=True)

    def save(self, folder: Path):
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "params.json").write_text(json.dumps({"rate": self.rate}, indent=2))


class HistoricalVol(ForecastModel):
    name = "hist_vol"

    def fit(self, panel, train_mask):
        pass                                   # nothing to learn

    def predict(self, panel, mask):
        vol = panel.df.loc[mask, "rvol_20"].to_numpy()
        return pd.concat([forecast_rows(self.name, panel, mask, h, vol_forecast=vol) for h in panel.horizons],
                         ignore_index=True)
