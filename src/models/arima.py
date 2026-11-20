"""ARIMA baseline (task 16.6): one ARMA(p, q) model per stock on daily log returns.

- Order: chosen by AIC on the first fold's training data from p <= max_p, q <= max_q, then kept.
- Parameters: re-estimated at every walk-forward refit (monthly by default).
- Daily updates: between refits the fitted model is extended with each new return (no re-fit),
  so the forecast for day d uses returns up to the close of d-1.
- Forecast for horizon h: the sum of the next h daily forecasts. Its variance comes from the
  model's MA(infinity) weights, so p_up = P(sum > 0) under a normal forecast distribution.

ARIMA models close-to-close returns, so its forecast starts at the close of d-1 even when the
labels start at the open of d (labels.entry = open).
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.models.base import ForecastModel, forecast_rows


def _fit(y: np.ndarray, order: tuple[int, int]):
    from statsmodels.tsa.arima.model import ARIMA

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ARIMA(y, order=(order[0], 0, order[1]), trend="c").fit()


def select_order(y: np.ndarray, max_p: int, max_q: int) -> tuple[int, int]:
    best, best_aic = (0, 0), np.inf
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            try:
                aic = _fit(y, (p, q)).aic
            except (ValueError, np.linalg.LinAlgError):
                aic = np.inf                     # a failed fit just drops that order
            if aic < best_aic:
                best, best_aic = (p, q), aic
    return best


def cumulative_sd(res, max_h: int) -> np.ndarray:
    """Standard deviation of the sum of the next 1..max_h forecast errors."""
    from statsmodels.tsa.arima_process import arma2ma

    psi = arma2ma(res.polynomial_ar, res.polynomial_ma, lags=max_h)
    cum = np.cumsum(psi)                         # effect of one shock on the running sum
    sigma2 = float(res.params[-1])               # last parameter is sigma2
    return np.sqrt(sigma2 * np.cumsum(cum ** 2))


def forecast_stock(y: np.ndarray, train_end: int, pred_idx: np.ndarray, order, max_p: int, max_q: int,
                   horizons: list[int]) -> dict:
    """y[i] is the return known at row i (the previous day's return); fit on y[:train_end + 1]."""
    hist = y[: train_end + 1]
    hist = hist[~np.isnan(hist)]
    if len(hist) < 100:
        return {"order": order, "mean": None}
    if order is None:
        order = select_order(hist, max_p, max_q)
    res = _fit(hist, order)
    max_h = max(horizons)
    sd = cumulative_sd(res, max_h)
    sigma = float(np.sqrt(res.params[-1]))
    means = np.full((len(pred_idx), len(horizons)), np.nan)
    pos = train_end
    for k, i in enumerate(pred_idx):
        if i > pos:
            new = y[pos + 1: i + 1]
            res = res.extend(np.where(np.isnan(new), 0.0, new))
            pos = i
        path = np.cumsum(np.asarray(res.forecast(max_h)))
        means[k] = [path[h - 1] for h in horizons]
    return {"order": order, "mean": means, "sd": np.array([sd[h - 1] for h in horizons]), "sigma": sigma,
            "params": [float(v) for v in res.params]}


class Arima(ForecastModel):
    name = "arima"

    def __init__(self, cfg, params=None):
        super().__init__(cfg, params)
        a = {**cfg["models"].get("arima", {}), **self.params}
        self.max_p, self.max_q = int(a.get("max_p", 2)), int(a.get("max_q", 2))
        self.n_jobs = int(a.get("n_jobs", cfg["models"].get("n_jobs", -1)))
        self.orders: dict[str, tuple[int, int]] = {}
        self._train_mask = None

    def fit(self, panel, train_mask):
        self._train_mask = train_mask           # estimation happens per stock inside predict

    def predict(self, panel, mask):
        from joblib import Parallel, delayed

        df = panel.df
        jobs, keys = [], []
        for symbol, idx in df.groupby("symbol", sort=True).indices.items():
            pred = idx[mask[idx]]
            trained = idx[self._train_mask[idx]]
            if len(pred) == 0 or len(trained) == 0:
                continue
            y = df["ret_1"].to_numpy()[idx]
            local_pred = pred - idx[0]
            train_end = int(trained.max() - idx[0])
            jobs.append(delayed(forecast_stock)(y, train_end, local_pred, self.orders.get(symbol),
                                                self.max_p, self.max_q, panel.horizons))
            keys.append((symbol, pred))
        results = Parallel(n_jobs=self.n_jobs)(jobs) if jobs else []

        parts = []
        self.fold_params = {}
        for (symbol, pred), r in zip(keys, results):
            self.orders[symbol] = r["order"]
            if r["mean"] is None:
                continue
            self.fold_params[symbol] = {"order": r["order"], "params": r["params"]}
            m = np.zeros(len(df), dtype=bool)
            m[pred] = True
            for j, h in enumerate(panel.horizons):
                mean = r["mean"][:, j]
                parts.append(forecast_rows(self.name, panel, m, h, p_up=norm.cdf(mean / r["sd"][j]),
                                           direction=np.where(mean > 0, 1.0, -1.0), exp_return=mean,
                                           vol_forecast=np.full(len(mean), r["sigma"])))
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    def save(self, folder: Path):
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "params.json").write_text(json.dumps(getattr(self, "fold_params", {}), indent=1))
