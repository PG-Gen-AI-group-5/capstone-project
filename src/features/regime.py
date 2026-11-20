"""Market regime labels (task 10.2): Nifty trend and India VIX level.

Every label for day d uses closes up to d-1 only, because the forecast for d is made at 08:30 IST,
before d's close exists. Settings come from the `regime` section of config.yaml.
"""
from __future__ import annotations

import pandas as pd

TREND_LABELS = ("below_ema", "above_ema")


def ema(close: pd.Series, span: int) -> pd.Series:
    """Exponential moving average (recursive form), NaN until `span` observations exist."""
    out = close.ewm(span=span, adjust=False, min_periods=span).mean()
    return out


def trend_regime(close: pd.Series, span: int = 200) -> pd.Series:
    """'above_ema' / 'below_ema': the index close vs its `span`-day EMA, as known before day d opens."""
    e = ema(close, span)
    label = pd.Series(pd.NA, index=close.index, dtype="object")
    ok = e.notna() & close.notna()
    label[ok] = (close[ok] > e[ok]).map({True: TREND_LABELS[1], False: TREND_LABELS[0]})
    return label.shift(1)


def vix_bands(vix: pd.Series, start, end, quantiles=(1 / 3, 2 / 3)) -> list[float]:
    """Cut-points for the VIX regime, fitted on [start, end] only (the training period)."""
    v = vix[(vix.index >= pd.Timestamp(start)) & (vix.index <= pd.Timestamp(end))].dropna()
    if v.empty:
        raise ValueError("no VIX values in the band-fitting period")
    return [float(v.quantile(q)) for q in quantiles]


def vix_regime(vix: pd.Series, bands: list[float], labels=("low", "mid", "high")) -> pd.Series:
    """VIX level bucket as known before day d opens (previous day's close)."""
    if len(labels) != len(bands) + 1:
        raise ValueError("need one more label than band cut-points")
    edges = [-float("inf"), *bands, float("inf")]
    cut = pd.cut(vix, bins=edges, labels=list(labels), right=False).astype("object")
    return cut.where(vix.notna()).shift(1)


def regime_table(nifty_close: pd.Series, vix: pd.Series, cfg_regime: dict) -> pd.DataFrame:
    """Both labels for each day, from the `regime` config section."""
    start, end = cfg_regime["vix_band_period"]
    bands = vix_bands(vix, start, end, cfg_regime.get("vix_quantiles", (1 / 3, 2 / 3)))
    out = pd.DataFrame({
        "trend": trend_regime(nifty_close, cfg_regime["trend_ema_span"]),
        "vix": vix_regime(vix.reindex(nifty_close.index), bands, cfg_regime["vix_labels"]),
    })
    out.attrs["vix_bands"] = bands
    return out
