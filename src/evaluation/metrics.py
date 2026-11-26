"""Forecast metrics (task 18.1): direction (accuracy, balanced accuracy, F1, Brier, log loss) and
volatility (MAE, RMSE), with the constant majority guess as a reference and a split by market regime."""
from __future__ import annotations

import numpy as np
import pandas as pd

DIRECTION_METRICS = ["n", "accuracy", "balanced_accuracy", "f1_up", "brier", "log_loss", "actual_up",
                     "predicted_up"]
VOL_METRICS = ["n_vol", "vol_mae", "vol_rmse"]


def direction_metrics(up: np.ndarray, direction: np.ndarray, p_up: np.ndarray) -> dict:
    ok = ~np.isnan(up) & ~np.isnan(direction)
    if not ok.any():
        return {k: np.nan for k in DIRECTION_METRICS}
    y, pred = up[ok], (direction[ok] > 0).astype(float)
    tp = float(((pred == 1) & (y == 1)).sum())
    precision = tp / pred.sum() if pred.sum() else 0.0
    recall = tp / y.sum() if y.sum() else 0.0
    down = y == 0
    recall_down = float((pred[down] == 0).mean()) if down.any() else np.nan
    recall_up = float((pred[~down] == 1).mean()) if (~down).any() else np.nan
    out = {"n": int(ok.sum()), "accuracy": float((pred == y).mean()),
           "balanced_accuracy": float(np.nanmean([recall_up, recall_down])),
           "f1_up": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
           "actual_up": float(y.mean()), "predicted_up": float(pred.mean())}
    okp = ok & ~np.isnan(p_up)
    if okp.any():
        p = np.clip(p_up[okp], 1e-6, 1 - 1e-6)
        yy = up[okp]
        out["brier"] = float(((p - yy) ** 2).mean())
        out["log_loss"] = float(-(yy * np.log(p) + (1 - yy) * np.log(1 - p)).mean())
    else:
        out["brier"] = out["log_loss"] = np.nan
    return out


def vol_metrics(actual: np.ndarray, forecast: np.ndarray) -> dict:
    ok = ~np.isnan(actual) & ~np.isnan(forecast)
    if not ok.any():
        return {"n_vol": 0, "vol_mae": np.nan, "vol_rmse": np.nan}
    err = forecast[ok] - actual[ok]
    return {"n_vol": int(ok.sum()), "vol_mae": float(np.abs(err).mean()), "vol_rmse": float(np.sqrt((err ** 2).mean()))}


def join_labels(forecasts: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """labels: long form from src.models.data.label_frame (symbol, trade_date, horizon_days, ret, up, vol)."""
    return forecasts.merge(labels, on=["symbol", "trade_date", "horizon_days"], how="left")


def score(joined: pd.DataFrame) -> dict:
    f = lambda c: joined[c].to_numpy(float)
    return {**direction_metrics(f("up"), f("direction"), f("p_up")), **vol_metrics(f("vol"), f("vol_forecast"))}


def evaluate(forecasts: pd.DataFrame, labels: pd.DataFrame, by=("model", "horizon_days")) -> pd.DataFrame:
    joined = join_labels(forecasts, labels)
    rows = [{**dict(zip(by, key if isinstance(key, tuple) else (key,))), **score(g)}
            for key, g in joined.groupby(list(by), sort=True)]
    return pd.DataFrame(rows)


def majority_guess(labels: pd.DataFrame, by=("horizon_days",)) -> pd.DataFrame:
    """Accuracy of always guessing the period's own majority direction: a hindsight reference line
    (EDA 02 §2), not a model. labels: long form with `up`."""
    g = labels.dropna(subset=["up"]).groupby(list(by))["up"]
    out = g.agg(n="size", actual_up="mean").reset_index()
    out["majority_accuracy"] = np.maximum(out["actual_up"], 1 - out["actual_up"])
    return out


def with_regimes(labels: pd.DataFrame, regimes: pd.DataFrame) -> pd.DataFrame:
    """Add the market regime of each trade_date (src.features.regime.regime_table: trend, vix)."""
    r = regimes.rename(columns={"trend": "trend_regime", "vix": "vix_regime"})
    r = r.rename_axis("trade_date").reset_index()[["trade_date", "trend_regime", "vix_regime"]]
    return labels.merge(r, on="trade_date", how="left")


def calibration_table(joined: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    """Mean forecast p_up vs actual up rate in equal-count bins, per model and horizon."""
    j = joined.dropna(subset=["p_up", "up"]).copy()
    out = []
    for (model, h), g in j.groupby(["model", "horizon_days"]):
        if g["p_up"].nunique() < 2:
            continue
        g = g.assign(bin=pd.qcut(g["p_up"], bins, labels=False, duplicates="drop"))
        t = g.groupby("bin").agg(p_mean=("p_up", "mean"), up_rate=("up", "mean"), n=("up", "size")).reset_index()
        out.append(t.assign(model=model, horizon_days=h))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()
