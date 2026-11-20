"""A small synthetic market (prices, index, macro) and a modelling panel built from it."""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from src.config import Config
from src.features.build import build_features
from src.labels.labels import make_labels
from src.models.data import Panel

SYMBOLS = {"AAA": "Energy", "BBB": "Financial Services", "CCC": "Information Technology"}
START = "2019-01-01"


def market(n_days: int = 420, seed: int = 0):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(START, periods=n_days)
    rows = []
    for k, symbol in enumerate(SYMBOLS):
        r = rng.normal(0.0004 * (k - 1), 0.015, n_days)
        close = 100 * np.exp(np.cumsum(r))
        open_ = close * np.exp(rng.normal(0, 0.004, n_days))
        rows.append(pd.DataFrame({
            "symbol": symbol, "trade_date": days, "open": open_,
            "high": np.maximum(open_, close) * 1.01, "low": np.minimum(open_, close) * 0.99,
            "close": close, "volume": rng.integers(1e5, 1e6, n_days).astype(float), "turnover": 1e7,
        }))
    prices = pd.concat(rows, ignore_index=True)
    nifty = 10000 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n_days)))
    vix = 15 + np.abs(rng.normal(0, 2, n_days))
    index = pd.concat([pd.DataFrame({"index_name": "Nifty 50", "trade_date": days, "close": nifty}),
                       pd.DataFrame({"index_name": "India VIX", "trade_date": days, "close": vix})],
                      ignore_index=True)
    macro = []
    for s in ("DGS10", "VIXCLS", "DEXINUS", "DCOILBRENTEU", "DTWEXBGS"):
        v = 50 + np.cumsum(rng.normal(0, 0.5, n_days))
        macro.append(pd.DataFrame({"series_id": s, "obs_date": days, "value": v,
                                   "known_from": (days + pd.Timedelta(days=1)).tz_localize("UTC")}))
    return prices, index, pd.concat(macro, ignore_index=True)


def config(**seq) -> Config:
    cfg = Config.load()
    raw = copy.deepcopy(cfg.raw)
    raw["periods"] = {"train": [pd.Timestamp("2019-01-01").date(), pd.Timestamp("2020-02-28").date()],
                      "validation": [pd.Timestamp("2020-03-01").date(), pd.Timestamp("2020-06-30").date()],
                      "test": [pd.Timestamp("2020-07-01").date(), None]}
    raw["features"]["min_history"] = 60
    raw["models"]["n_jobs"] = 1
    raw["models"]["sequence"].update({"window": 10, "batch_size": 64, "max_epochs": 1, "patience": 1,
                                      "early_stop_days": 20, "threads": 1, **seq})
    raw["models"]["lstm"].update({"hidden": 8, "layers": 1, "symbol_dim": 2, "sector_dim": 2})
    raw["models"]["tft"].update({"hidden": 8, "heads": 2, "sequence": {}})
    return Config(raw, cfg.root)


def panel(cfg: Config | None = None, n_days: int = 420) -> Panel:
    cfg = cfg or config()
    prices, index, macro = market(n_days)
    feats, cal = build_features(prices, index, macro, cfg["features"]["min_history"])
    horizons = list(cfg["labels"]["horizons"])
    labels = make_labels(prices, range(1, max(horizons) + 1), cfg["labels"]["entry"])
    df = feats.merge(labels, on=["symbol", "trade_date"], how="left")
    df["industry"] = df["symbol"].map(SYMBOLS)
    df["tier"] = "nifty50"
    df["evaluate"] = True
    return Panel(df.sort_values(["symbol", "trade_date"]).reset_index(drop=True), cal, horizons, max(horizons))
