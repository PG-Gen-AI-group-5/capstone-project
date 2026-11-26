"""Where the live path keeps its data, and which day it plans for.

The live path builds its own copy of the derived tables (prices_adj, features, labels, calendar) and its own
models under data/live/, from the shared lake. The research tables in data/derived and the walk-forward models in
data/models stay exactly as the evaluation left them, so a daily update never makes them look out of date, and
live models trained on 2024+ data never touch the frozen test-period forecasts (task 16.9).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from src.config import Config

LIVE_DEFAULTS = {"models": ["tft", "lstm"], "refit_every_days": 5, "daily_time": "08:00", "refit_day": "SAT",
                 "refit_time": "10:00", "weekly_sources_every_days": 7, "stale_ok": False}


@dataclass
class LiveConfig(Config):
    """The project config with derived tables and models redirected to data/live/."""

    @property
    def live_dir(self) -> Path:
        return self.data_dir / "live"

    @property
    def derived_dir(self) -> Path:
        return self.live_dir / "derived"

    @property
    def models_dir(self) -> Path:
        return self.live_dir / "models"


def live_config(cfg: Config) -> LiveConfig:
    return LiveConfig(copy.deepcopy(cfg.raw), cfg.root)


def live_settings(cfg: Config) -> dict:
    return {**LIVE_DEFAULTS, **(cfg.get("live") or {})}


def trading_days(cfg: Config, start: date, end: date) -> list[date]:
    """NSE trading days (exchange_calendars XBOM plus data/manual/calendar_overrides.csv)."""
    from src.ingest.calendar import build_calendar

    return [d for d, is_trading, _ in build_calendar(start, end, cfg.manual_dir / "calendar_overrides.csv") if is_trading]


def next_trading_day(cfg: Config, after) -> date:
    after = pd.Timestamp(after).date()
    days = [d for d in trading_days(cfg, after + timedelta(days=1), after + timedelta(days=20)) if d > after]
    if not days:
        raise ValueError(f"No trading day in the 20 days after {after}")
    return days[0]


def last_price_day(cfg: Config) -> date:
    from src.store.db import read_table

    p = read_table(cfg, "prices_daily", columns=["trade_date"])
    return pd.Timestamp(p["trade_date"].max()).date()
