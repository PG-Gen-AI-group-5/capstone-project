"""Temporal Fusion Transformer (task 16.7b), after Lim et al. (2021), one model shared by all stocks.

Written in plain PyTorch instead of a library so it shares the LSTM's windowed data, walk-forward
runner and forecast table, and keeps the project on its current pandas version.

Inputs
- static: symbol and sector embeddings
- observed (past window only): every feature in the panel
- known (past window and the next H trading days): calendar features
  (the catalyst calendar, task 7.x, adds results and RBI dates here once it exists)
Output: quantiles of the cumulative log return from the entry to each of the next H days. For an
evaluation horizon h: exp_return = median, p_up = P(return > 0) from a normal fitted to the
10%-90% quantile spread, vol_forecast = that spread's standard deviation / sqrt(h).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm
from torch import nn
from torch.nn import functional as F

from src.labels.splits import early_stop_split
from src.models.base import ForecastModel, forecast_rows
from src.models.sequence import SequenceData, predict_net, set_threads, train_net


class GLU(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.fc = nn.Linear(d_in, 2 * d_out)

    def forward(self, x):
        a, b = self.fc(x).chunk(2, dim=-1)
        return a * torch.sigmoid(b)


class GRN(nn.Module):
    """Gated residual network: LayerNorm(skip(x) + GLU(W2 ELU(W1 x + Wc c)))."""

    def __init__(self, d_in, d_hidden, d_out, dropout, d_context=None):
        super().__init__()
        self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.ctx = nn.Linear(d_context, d_hidden, bias=False) if d_context else None
        self.fc2 = nn.Linear(d_hidden, d_hidden)
        self.drop = nn.Dropout(dropout)
        self.glu = GLU(d_hidden, d_out)
        self.norm = nn.LayerNorm(d_out)

    def forward(self, x, c=None):
        h = self.fc1(x)
        if self.ctx is not None and c is not None:
            h = h + self.ctx(c)
        h = self.drop(self.fc2(F.elu(h)))
        return self.norm(self.skip(x) + self.glu(h))


class VariableGRN(nn.Module):
    """n independent GRNs (d -> d), one per variable, applied to (..., n, d) as batched matmuls."""

    def __init__(self, n_vars, d, dropout):
        super().__init__()
        bound = 1 / d ** 0.5
        p = lambda *shape: nn.Parameter(torch.empty(*shape).uniform_(-bound, bound))
        self.w1, self.b1 = p(n_vars, d, d), p(n_vars, d)
        self.w2, self.b2 = p(n_vars, d, d), p(n_vars, d)
        self.wg, self.bg = p(n_vars, d, 2 * d), p(n_vars, 2 * d)
        self.ln_w, self.ln_b = nn.Parameter(torch.ones(n_vars, d)), nn.Parameter(torch.zeros(n_vars, d))
        self.drop = nn.Dropout(dropout)
        self.d = d

    def forward(self, x):
        h = F.elu(torch.einsum("...nd,nde->...ne", x, self.w1) + self.b1)
        h = self.drop(torch.einsum("...nd,nde->...ne", h, self.w2) + self.b2)
        a, b = (torch.einsum("...nd,nde->...ne", h, self.wg) + self.bg).chunk(2, dim=-1)
        return F.layer_norm(x + a * torch.sigmoid(b), (self.d,)) * self.ln_w + self.ln_b


class VariableSelection(nn.Module):
    """Weights n embedded variables (..., n, d) with a softmax from a GRN over all of them."""

    def __init__(self, n_vars, d, dropout, d_context=None):
        super().__init__()
        self.weights = GRN(n_vars * d, d, n_vars, dropout, d_context)
        self.var_grns = VariableGRN(n_vars, d, dropout)

    def forward(self, x, c=None):
        w = torch.softmax(self.weights(x.flatten(-2), c), dim=-1)
        return (w.unsqueeze(-1) * self.var_grns(x)).sum(-2), w


class ContinuousEmbedding(nn.Module):
    """A separate linear embedding R -> R^d for each continuous variable."""

    def __init__(self, n_vars, d):
        super().__init__()
        self.w = nn.Parameter(torch.randn(n_vars, d) * 0.1)
        self.b = nn.Parameter(torch.zeros(n_vars, d))

    def forward(self, x):                      # (..., n) -> (..., n, d)
        return x.unsqueeze(-1) * self.w + self.b


class InterpretableAttention(nn.Module):
    """Multi-head attention whose heads share one value projection, so weights can be averaged."""

    def __init__(self, d, heads, dropout):
        super().__init__()
        self.h, self.dk = heads, d // heads
        self.q = nn.Linear(d, heads * self.dk)
        self.k = nn.Linear(d, heads * self.dk)
        self.v = nn.Linear(d, self.dk)
        self.out = nn.Linear(self.dk, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, q, k, v, mask):
        B, Lq, _ = q.shape
        Lk = k.shape[1]
        Q = self.q(q).view(B, Lq, self.h, self.dk).transpose(1, 2)
        K = self.k(k).view(B, Lk, self.h, self.dk).transpose(1, 2)
        att = (Q @ K.transpose(-1, -2)) / self.dk ** 0.5
        att = self.drop(torch.softmax(att.masked_fill(mask, float("-inf")), dim=-1))
        heads = att @ self.v(v).unsqueeze(1)                       # (B, h, Lq, dk)
        return self.out(heads.mean(1)), att.mean(1)


class TFTNet(nn.Module):
    def __init__(self, n_obs, n_known, n_sym, n_sec, horizon, quantiles, hidden=32, heads=4, dropout=0.1):
        super().__init__()
        d = hidden
        self.H, self.quantiles = horizon, list(quantiles)
        self.sym = nn.Embedding(n_sym, d)
        self.sec = nn.Embedding(n_sec, d)
        self.static_vsn = VariableSelection(2, d, dropout)
        self.c_select, self.c_enrich, self.c_h, self.c_c = (GRN(d, d, d, dropout) for _ in range(4))
        self.past_embed = ContinuousEmbedding(n_obs + n_known, d)
        self.future_embed = ContinuousEmbedding(n_known, d)
        self.past_vsn = VariableSelection(n_obs + n_known, d, dropout, d_context=d)
        self.future_vsn = VariableSelection(n_known, d, dropout, d_context=d)
        self.encoder = nn.LSTM(d, d, batch_first=True)
        self.decoder = nn.LSTM(d, d, batch_first=True)
        self.lstm_gate, self.lstm_norm = GLU(d, d), nn.LayerNorm(d)
        self.enrich = GRN(d, d, d, dropout, d_context=d)
        self.attention = InterpretableAttention(d, heads, dropout)
        self.att_gate, self.att_norm = GLU(d, d), nn.LayerNorm(d)
        self.positionwise = GRN(d, d, d, dropout)
        self.out_gate, self.out_norm = GLU(d, d), nn.LayerNorm(d)
        self.head = nn.Linear(d, len(self.quantiles))

    def forward(self, b, return_weights=False):
        static_in = torch.stack([self.sym(b["sym"]), self.sec(b["sec"])], dim=1)       # (B, 2, d)
        s, static_w = self.static_vsn(static_in)
        cs, ce, ch, cc = self.c_select(s), self.c_enrich(s), self.c_h(s), self.c_c(s)

        past = self.past_embed(torch.cat([b["x_past"], b["k_past"]], dim=-1))          # (B, W, n, d)
        W = past.shape[1]
        past_sel, past_w = self.past_vsn(past, cs.unsqueeze(1).expand(-1, W, -1))
        fut = self.future_embed(b["k_future"])                                        # (B, H, k, d)
        fut_sel, fut_w = self.future_vsn(fut, cs.unsqueeze(1).expand(-1, self.H, -1))

        enc, state = self.encoder(past_sel, (ch.unsqueeze(0), cc.unsqueeze(0)))
        dec, _ = self.decoder(fut_sel, state)
        selected = torch.cat([past_sel, fut_sel], dim=1)
        temporal = self.lstm_norm(selected + self.lstm_gate(torch.cat([enc, dec], dim=1)))

        enriched = self.enrich(temporal, ce.unsqueeze(1).expand(-1, temporal.shape[1], -1))
        L = W + self.H
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=enriched.device), diagonal=1)[W:]
        q = enriched[:, W:]
        att, att_w = self.attention(q, enriched, enriched, causal)
        x = self.att_norm(q + self.att_gate(att))
        x = self.positionwise(x)
        x = self.out_norm(temporal[:, W:] + self.out_gate(x))
        out = {"quantiles": self.head(x)}                                               # (B, H, Q)
        if return_weights:
            out.update(static_w=static_w, past_w=past_w, future_w=fut_w, attention=att_w)
        return out


def quantile_loss(quantiles):
    q = torch.tensor(quantiles).view(1, 1, -1)

    def loss(out, b):
        err = b["y_path"].unsqueeze(-1) - out["quantiles"]
        return torch.maximum(q * err, (q - 1) * err).mean() * 100      # scaled for readable loss values

    return loss


class TFTModel(ForecastModel):
    name = "tft"

    def _settings(self):
        m = self.cfg["models"]
        arch = {**m["tft"], **self.params.get("tft", {})}
        seq = {**m["sequence"], **arch.pop("sequence", {}), **self.params.get("sequence", {})}
        return seq, arch

    def _build(self, panel):
        seq, arch = self._settings()
        set_threads(seq.get("threads", 0))
        self.data = SequenceData(panel, int(seq["window"]))
        self.quantiles = list(arch["quantiles"])
        torch.manual_seed(int(self.cfg["models"]["seed"]))
        self.net = TFTNet(len(self.data.features), len(self.data.known), len(self.data.symbols),
                          len(self.data.sectors), panel.max_h, self.quantiles, arch["hidden"], arch["heads"],
                          arch["dropout"])
        return seq

    def fit(self, panel, train_mask):
        seq = self._build(panel)
        inner_train, inner_val = early_stop_split(panel.df, train_mask, int(seq["early_stop_days"]), panel.max_h)
        self.data.fit_scaler(np.flatnonzero(inner_train))
        tr = self.data.sample_rows(inner_train, need_targets=True)
        va = self.data.sample_rows(inner_val, need_targets=True)
        self.history = train_net(self.net, self.data, tr, va, quantile_loss(self.quantiles), seq,
                                 int(self.cfg["models"]["seed"]))

    @classmethod
    def load(cls, cfg, folder: Path, panel, params=None) -> TFTModel:
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
        qs = np.sort(torch.cat([o["quantiles"] for o in outs]).numpy(), axis=-1)       # no crossing
        lo, mid, hi = (self.quantiles.index(v) for v in (0.1, 0.5, 0.9))
        z = norm.ppf(0.9) - norm.ppf(0.1)
        m = np.zeros(len(panel.df), dtype=bool)
        m[rows] = True
        parts = []
        for h in panel.horizons:
            q = qs[:, h - 1, :]
            sd = np.maximum((q[:, hi] - q[:, lo]) / z, 1e-6)
            med = q[:, mid]
            parts.append(forecast_rows(self.name, panel, m, h, p_up=norm.cdf(med / sd),
                                       direction=np.where(med > 0, 1.0, -1.0), exp_return=med,
                                       vol_forecast=sd / np.sqrt(h)))
        return pd.concat(parts, ignore_index=True)

    def predict_quantiles(self, panel, mask) -> tuple[np.ndarray, np.ndarray]:
        """(rows, quantiles of shape (rows, H, Q)) for the rows in mask."""
        seq, _ = self._settings()
        rows = self.data.sample_rows(mask, need_targets=False)
        outs = predict_net(self.net, self.data, rows, int(seq["batch_size"]))
        return rows, np.sort(torch.cat([o["quantiles"] for o in outs]).numpy(), axis=-1)

    @torch.no_grad()
    def variable_importance(self, mask: np.ndarray, max_rows: int = 20000) -> pd.DataFrame:
        """Mean variable-selection weights over the rows in mask (past and future inputs)."""
        rows = self.data.sample_rows(mask, need_targets=False)
        rows = rows[:: max(1, len(rows) // max_rows)]
        self.net.eval()
        past, fut = [], []
        for i in range(0, len(rows), 1024):
            o = self.net(self.data.batch(rows[i:i + 1024]), return_weights=True)
            past.append(o["past_w"].mean(1))
            fut.append(o["future_w"].mean(1))
        past_names = self.data.features + [f"{k} (past)" for k in self.data.known]
        return pd.concat([
            pd.DataFrame({"input": past_names, "part": "past", "weight": torch.cat(past).mean(0).numpy()}),
            pd.DataFrame({"input": self.data.known, "part": "future", "weight": torch.cat(fut).mean(0).numpy()}),
        ]).sort_values(["part", "weight"], ascending=[True, False]).reset_index(drop=True)

    def save(self, folder: Path):
        folder.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.net.state_dict(), "mean": self.data.mean, "std": self.data.std,
                    "symbols": self.data.symbols, "sectors": self.data.sectors, "features": self.data.features,
                    "quantiles": self.quantiles}, folder / "model.pt")
        (folder / "history.json").write_text(json.dumps(self.history, indent=1))
