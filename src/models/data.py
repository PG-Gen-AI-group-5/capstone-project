"""The modelling panel: features and labels for every stock and trading day, in one aligned frame."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import Config
from src.features.build import CALENDAR_FEATURES, FEATURES
from src.store.db import read_table, universe

SEQUENCE_INPUTS_VERSION = 2       # bump when the LSTM/TFT inputs change, so saved forecasts are not reused


@dataclass
class Panel:
    df: pd.DataFrame                     # sorted by symbol, trade_date; features + labels + symbol meta
    calendar: pd.DataFrame               # trading days (+ future weekdays) with CALENDAR_FEATURES
    horizons: list[int]                  # evaluation horizons, e.g. [1, 5, 10]
    max_h: int                           # longest horizon; labels exist for every step 1..max_h
    features: list[str] = field(default_factory=lambda: list(FEATURES))
    known: list[str] = field(default_factory=lambda: list(CALENDAR_FEATURES))

    @property
    def dates(self) -> pd.Series:
        return self.df["trade_date"]

    def eval_mask(self, start, end) -> np.ndarray:
        d = self.df["trade_date"]
        return (self.df["ready"].to_numpy(bool) & self.df["evaluate"].to_numpy(bool)
                & (d >= pd.Timestamp(start)).to_numpy() & (d <= pd.Timestamp(end)).to_numpy())

    def subset(self, symbols) -> Panel:
        df = self.df[self.df["symbol"].isin(symbols)].reset_index(drop=True)
        return Panel(df, self.calendar, self.horizons, self.max_h, self.features, self.known)


def load_panel(cfg: Config, symbols: list[str] | None = None) -> Panel:
    feats = read_table(cfg, "features_daily", layer="derived")
    labels = read_table(cfg, "labels", layer="derived")
    cal = read_table(cfg, "calendar", layer="derived")
    for frame in (feats, labels, cal):
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    df = feats.merge(labels, on=["symbol", "trade_date"], how="left")
    meta = universe(cfg, tier=None)[["symbol", "industry", "tier"]].drop_duplicates("symbol")
    df = df.merge(meta, on="symbol", how="left")
    df["industry"] = df["industry"].fillna("Unknown")
    df["evaluate"] = df["tier"].eq("nifty50")                 # evaluation universe; others only train
    if symbols is not None:
        df = df[df["symbol"].isin(symbols)]
    df = df.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    horizons = list(cfg["labels"]["horizons"])
    return Panel(df, cal.sort_values("trade_date").reset_index(drop=True), horizons, max(horizons))


def label_frame(panel: Panel, mask: np.ndarray | None = None) -> pd.DataFrame:
    """Labels in long form: symbol, trade_date, horizon_days, ret, up, vol."""
    df = panel.df if mask is None else panel.df[mask]
    parts = [pd.DataFrame({"symbol": df["symbol"], "trade_date": df["trade_date"], "horizon_days": h,
                           "ret": df[f"fwd_ret_{h}"], "up": df[f"up_{h}"], "vol": df[f"fwd_vol_{h}"]})
             for h in panel.horizons]
    return pd.concat(parts, ignore_index=True)


def prepare(cfg: Config, log=print) -> dict:
    """Build every input the models need: adjusted prices -> features -> labels."""
    from src.features import build as features
    from src.labels import labels
    from src.store import adjust

    out = {}
    for name, step in (("prices_adj", adjust.build), ("features", features.build), ("labels", labels.build)):
        out[name] = step(cfg)
        log(f"{name}: {out[name]}")
    return out
