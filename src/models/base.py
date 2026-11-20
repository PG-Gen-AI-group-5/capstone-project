"""Model interface and the forecast table every model writes (see docs/ARCHITECTURE.md)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Config

FORECAST_COLUMNS = ["model", "model_version", "symbol", "trade_date", "as_of", "horizon_days",
                    "p_up", "direction", "exp_return", "vol_forecast", "train_end"]


class ForecastModel(ABC):
    """fit() sees only rows in train_mask; predict() returns forecast rows for rows in mask.

    Every feature in the panel row for day d is already as of 08:30 IST on d, so a model may use
    any feature of a row it predicts, and labels only of rows in train_mask.
    """

    name: str = "model"

    def __init__(self, cfg: Config, params: dict | None = None):
        self.cfg = cfg
        self.params = dict(params or {})

    @abstractmethod
    def fit(self, panel, train_mask: np.ndarray) -> None: ...

    @abstractmethod
    def predict(self, panel, mask: np.ndarray) -> pd.DataFrame: ...

    def save(self, folder: Path) -> None:
        """Store what is needed to reproduce this fold's forecasts. Default: nothing."""

    def describe(self) -> dict:
        return {"name": self.name, "params": self.params}


def forecast_rows(model: str, panel, mask: np.ndarray, horizon: int, p_up=None, direction=None,
                  exp_return=None, vol_forecast=None) -> pd.DataFrame:
    df = panel.df.loc[mask, ["symbol", "trade_date", "as_of"]].reset_index(drop=True)
    n = len(df)

    def col(v):
        return np.full(n, np.nan) if v is None else np.asarray(v, dtype=float).reshape(n)

    df.insert(0, "model", model)
    df["horizon_days"] = horizon
    df["p_up"] = col(p_up)
    df["direction"] = col(direction)
    df["exp_return"] = col(exp_return)
    df["vol_forecast"] = col(vol_forecast)
    return df


def check_forecasts(df: pd.DataFrame) -> None:
    """Raise ValueError if a forecast table breaks the shared format."""
    missing = [c for c in FORECAST_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"forecast table is missing columns {missing}")
    p = df["p_up"].dropna()
    if ((p < 0) | (p > 1)).any():
        raise ValueError("p_up outside [0, 1]")
    d = df["direction"].dropna()
    if not d.isin([-1.0, 1.0]).all():
        raise ValueError("direction must be +1, -1 or NaN")
    if (df["vol_forecast"].dropna() < 0).any():
        raise ValueError("negative vol_forecast")
    if df.duplicated(["model", "symbol", "trade_date", "horizon_days"]).any():
        raise ValueError("duplicate forecast rows")
    if (pd.to_datetime(df["train_end"]) >= pd.to_datetime(df["trade_date"])).any():
        raise ValueError("a forecast was trained on labels that end on or after its own day")
