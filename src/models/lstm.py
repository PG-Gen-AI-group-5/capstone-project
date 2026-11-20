"""LSTM (task 16.7): one model shared by all stocks.

Input: a window of scaled features and calendar features, plus learned symbol and sector
embeddings at every step. Two heads on the last hidden state: the probability of an up move and
the log of realised volatility, for each evaluation horizon.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.labels.splits import early_stop_split
from src.models.base import ForecastModel, forecast_rows
from src.models.sequence import SequenceData, predict_net, set_threads, train_net


class LSTMNet(nn.Module):
    def __init__(self, n_feat, n_known, n_sym, n_sec, n_h, hidden=64, layers=2, dropout=0.2, symbol_dim=8,
                 sector_dim=4):
        super().__init__()
        self.sym = nn.Embedding(n_sym, symbol_dim)
        self.sec = nn.Embedding(n_sec, sector_dim)
        self.lstm = nn.LSTM(n_feat + n_known + symbol_dim + sector_dim, hidden, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 2 * n_h))
        self.n_h = n_h

    def forward(self, b):
        W = b["x_past"].shape[1]
        static = torch.cat([self.sym(b["sym"]), self.sec(b["sec"])], dim=-1).unsqueeze(1).expand(-1, W, -1)
        out, _ = self.lstm(torch.cat([b["x_past"], b["k_past"], static], dim=-1))
        z = self.head(out[:, -1])
        return {"logits": z[:, : self.n_h], "logvol": z[:, self.n_h:]}


def lstm_loss(out, b):
    bce = nn.functional.binary_cross_entropy_with_logits(out["logits"], b["y_up"])
    mse = nn.functional.mse_loss(out["logvol"], b["y_logvol"])
    return bce + 0.5 * mse


class LSTMModel(ForecastModel):
    name = "lstm"

    def _settings(self):
        m = self.cfg["models"]
        seq = {**m["sequence"], **self.params.get("sequence", {})}
        arch = {**m["lstm"], **self.params.get("lstm", {})}
        return seq, arch

    def _build(self, panel):
        seq, arch = self._settings()
        set_threads(seq.get("threads", 0))
        self.data = SequenceData(panel, int(seq["window"]))
        torch.manual_seed(int(self.cfg["models"]["seed"]))
        self.net = LSTMNet(len(self.data.features), len(self.data.known), len(self.data.symbols),
                           len(self.data.sectors), len(panel.horizons), **arch)
        return seq

    def fit(self, panel, train_mask):
        seq = self._build(panel)
        inner_train, inner_val = early_stop_split(panel.df, train_mask, int(seq["early_stop_days"]), panel.max_h)
        self.data.fit_scaler(np.flatnonzero(inner_train))
        tr = self.data.sample_rows(inner_train, need_targets=True)
        va = self.data.sample_rows(inner_val, need_targets=True)
        self.history = train_net(self.net, self.data, tr, va, lstm_loss, seq, int(self.cfg["models"]["seed"]))

    @classmethod
    def load(cls, cfg, folder: Path, panel, params=None) -> LSTMModel:
        """Rebuild a saved fold's model on `panel` (same config) for prediction or inspection."""
        model = cls(cfg, params)
        model._build(panel)
        saved = torch.load(Path(folder) / "model.pt", weights_only=False)
        model.net.load_state_dict(saved["state_dict"])
        model.data.set_scaler(saved["mean"], saved["std"])
        model.history = json.loads((Path(folder) / "history.json").read_text())
        return model

    def predict(self, panel, mask):
        seq, _ = self._settings()
        rows = self.data.sample_rows(mask, need_targets=False)
        outs = predict_net(self.net, self.data, rows, int(seq["batch_size"]))
        p = torch.sigmoid(torch.cat([o["logits"] for o in outs])).numpy()
        vol = np.exp(torch.cat([o["logvol"] for o in outs]).numpy())
        m = np.zeros(len(panel.df), dtype=bool)
        m[rows] = True
        return pd.concat([forecast_rows(self.name, panel, m, h, p_up=p[:, j], direction=np.where(p[:, j] > 0.5, 1.0, -1.0),
                                        vol_forecast=vol[:, j]) for j, h in enumerate(panel.horizons)],
                         ignore_index=True)

    def save(self, folder: Path):
        folder.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.net.state_dict(), "mean": self.data.mean, "std": self.data.std,
                    "symbols": self.data.symbols, "sectors": self.data.sectors, "features": self.data.features},
                   folder / "model.pt")
        (folder / "history.json").write_text(json.dumps(self.history, indent=1))
