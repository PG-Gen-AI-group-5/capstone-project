"""Walk-forward splits with the gap rule.

A training row for day t with horizon h looks at prices up to day t+h-1 (its end_h). For a fold
whose predictions start on day T, a row may train only if its end date for the longest horizon is
before T. Training is an expanding window from the first usable row.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FREQS = {"monthly": "M", "quarterly": "Q", "yearly": "Y"}


@dataclass(frozen=True)
class Fold:
    start: pd.Timestamp        # first prediction day
    end: pd.Timestamp          # last prediction day


def make_folds(trading_days, start, end, freq: str = "yearly") -> list[Fold]:
    """One fold per calendar month/quarter/year of trading days within [start, end]."""
    if freq not in FREQS:
        raise ValueError(f"refit frequency must be one of {sorted(FREQS)}")
    days = pd.Series(sorted(pd.to_datetime(pd.Series(trading_days)).unique()))
    days = days[(days >= pd.Timestamp(start)) & (days <= pd.Timestamp(end))]
    if days.empty:
        return []
    keys = days.dt.to_period(FREQS[freq])
    return [Fold(g.iloc[0], g.iloc[-1]) for _, g in days.groupby(keys, sort=True)]


def train_mask(df: pd.DataFrame, fold_start, max_h: int) -> np.ndarray:
    """Rows usable for training a fold that starts predicting on fold_start."""
    end = pd.to_datetime(df[f"end_{max_h}"])
    return (df["ready"].to_numpy(bool) & end.notna().to_numpy()
            & (end < pd.Timestamp(fold_start)).to_numpy())


def early_stop_split(df: pd.DataFrame, mask: np.ndarray, days: int, max_h: int) -> tuple[np.ndarray, np.ndarray]:
    """Hold out the last `days` trading days of the training rows for early stopping, keeping the
    gap rule between the inner training rows and the held-out rows."""
    dates = pd.to_datetime(df["trade_date"])
    train_days = np.sort(dates[mask].unique())
    if len(train_days) <= days * 2:
        return mask, np.zeros_like(mask)
    cut = pd.Timestamp(train_days[-days])
    inner_val = mask & (dates >= cut).to_numpy()
    inner_train = mask & (pd.to_datetime(df[f"end_{max_h}"]) < cut).to_numpy()
    return inner_train, inner_val
