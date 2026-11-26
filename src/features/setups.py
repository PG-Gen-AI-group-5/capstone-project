"""Swing-setup detector (task 10.3) and code-computed trade levels (task 10.4).

Each rule reads the feature row for day d, which holds values as of the close of d-1 (known at the
08:30 forecast). Levels are in raw prices, from the raw close of d-1 and the ATR as a share of price:
    stop   = reference x (1 - stop_atr x atr_pct)
    target = reference x (1 + reward_risk x stop_atr x atr_pct)
The actual entry is the next open, filled by the backtester; `entry` here is the reference close.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from src.contracts import Setup

SETUP_TYPES = ("pullback_uptrend", "breakout", "oversold_bounce")
DEFAULTS = {"stop_atr": 1.5, "reward_risk": 2.0,
            "pullback": {"ema20_gap_min": -0.02, "ema20_gap_max": 0.01, "rsi_min": 0.40, "rsi_max": 0.60},
            "breakout": {"dist_52w_high_min": -0.01, "logvol_z_min": 0.5},
            "oversold": {"rsi_max": 0.30}}


def settings(cfg_setups: dict | None) -> dict:
    s = {**DEFAULTS, **(cfg_setups or {})}
    for k in ("pullback", "breakout", "oversold"):
        s[k] = {**DEFAULTS[k], **(cfg_setups or {}).get(k, {})}
    return s


def detect(features: pd.DataFrame, cfg_setups: dict | None = None) -> pd.Series:
    """Setup type per row ('none' when no rule fires). Rules are checked in SETUP_TYPES order."""
    s = settings(cfg_setups)
    f = features
    uptrend = (f["ema200_gap"] > 0) & (f["ema20_50"] > 0)
    p = s["pullback"]
    pullback = uptrend & f["ema20_gap"].between(p["ema20_gap_min"], p["ema20_gap_max"]) & \
        f["rsi14"].between(p["rsi_min"], p["rsi_max"])
    b = s["breakout"]
    breakout = (f["dist_52w_high"] >= b["dist_52w_high_min"]) & (f["logvol_z20"] >= b["logvol_z_min"]) & (f["ret_5"] > 0)
    oversold = (f["ema200_gap"] > 0) & (f["rsi14"] <= s["oversold"]["rsi_max"])
    out = np.select([pullback.fillna(False), breakout.fillna(False), oversold.fillna(False)], list(SETUP_TYPES),
                    "none")
    return pd.Series(out, index=f.index)


def levels(reference_close: float, atr_pct: float, cfg_setups: dict | None = None) -> tuple[float, float, float, float]:
    """(entry, stop, target, risk_reward) in raw prices."""
    s = settings(cfg_setups)
    risk = s["stop_atr"] * atr_pct
    stop = reference_close * (1 - risk)
    target = reference_close * (1 + s["reward_risk"] * risk)
    rr = (target - reference_close) / (reference_close - stop) if reference_close > stop else 0.0
    return round(reference_close, 2), round(stop, 2), round(target, 2), round(rr, 2)


def make_setup(symbol: str, trade_date: date, setup_type: str, reference_close: float, atr_pct: float,
               cfg_setups: dict | None = None) -> Setup:
    entry, stop, target, rr = levels(reference_close, atr_pct, cfg_setups)
    return Setup(symbol=symbol, trade_date=trade_date, setup_type=setup_type, reference_close=reference_close,
                 entry=entry, stop=stop, target=target, atr_pct=float(atr_pct), risk_reward=rr)
