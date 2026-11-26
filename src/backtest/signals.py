"""Turn model forecasts into backtester signals (task 16.8).

A model signals a long on day d when its `p_up` at the chosen horizon reaches its threshold. Stop and target
come from the same code as the agents' setups (task 10.4): the raw close of d-1 and the ATR. The holding
period is the horizon. Thresholds belong to `backtest.thresholds` in config.yaml and are tuned on validation.
"""
from __future__ import annotations

import pandas as pd

from src.contracts import Signal
from src.features.setups import levels


def forecast_signals(forecasts: pd.DataFrame, features: pd.DataFrame, threshold: float, horizon: int = 5,
                     cfg_setups: dict | None = None) -> list[Signal]:
    """forecasts: the forecast table (model, symbol, trade_date, horizon_days, p_up). features: symbol,
    trade_date, ref_close (raw close of d-1), atr14_pct."""
    fc = forecasts[(forecasts["horizon_days"] == horizon) & (forecasts["p_up"] >= threshold)].copy()
    fc["trade_date"] = pd.to_datetime(fc["trade_date"])
    f = features[["symbol", "trade_date", "ref_close", "atr14_pct"]].assign(trade_date=pd.to_datetime(features["trade_date"]))
    fc = fc.merge(f, on=["symbol", "trade_date"], how="inner").dropna(subset=["ref_close", "atr14_pct"])
    out = []
    for r in fc.itertuples():
        _, stop, target, _ = levels(float(r.ref_close), float(r.atr14_pct), cfg_setups)
        out.append(Signal(symbol=r.symbol, signal_date=r.trade_date.date(), stop=stop, target=target,
                          max_hold_days=horizon, source=str(r.model), confidence=float(r.p_up)))
    return out
