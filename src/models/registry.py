"""Model names -> classes. Heavy libraries load only when their model is created."""
from __future__ import annotations

from importlib import import_module

MODELS = {
    "base_rate": "src.models.baselines:BaseRate",
    "ma_crossover": "src.models.baselines:MACrossover",
    "hist_vol": "src.models.baselines:HistoricalVol",
    "arima": "src.models.arima:Arima",
    "lstm": "src.models.lstm:LSTMModel",
    "tft": "src.models.tft:TFTModel",
}


def create(name: str, cfg, params: dict | None = None):
    if name not in MODELS:
        raise ValueError(f"Unknown model {name!r}. Known: {', '.join(MODELS)}")
    module, cls = MODELS[name].split(":")
    return getattr(import_module(module), cls)(cfg, params)
