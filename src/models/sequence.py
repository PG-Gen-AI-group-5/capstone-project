"""Windowed samples and the training loop shared by the LSTM and TFT models.

A sample for row d (stock s, day d) holds:
- x_past (W, F): the scaled features of rows d-W+1 .. d of stock s. Row d's features are as of
  08:30 IST on d, so the window ends at the forecast time.
- k_past (W, K), k_future (H, K): calendar features of those days and of the holding days
  d .. d+H-1 (known in advance)
- symbol and sector ids
- targets: up_h and log(fwd_vol_h) for the evaluation horizons, and the cumulative return path
  fwd_ret_1 .. fwd_ret_H
"""
from __future__ import annotations

import copy
import os
import time

import numpy as np
import torch


class SequenceData:
    def __init__(self, panel, window: int):
        df = panel.df
        self.panel = panel
        self.window = window
        self.H = panel.max_h
        self.horizons = panel.horizons
        self.known = list(panel.known)
        self.features = [f for f in panel.features if f not in self.known]   # calendar inputs arrive via k_past
        self.X = df[self.features].to_numpy(np.float32)
        self.Xs = None
        self.symbols = sorted(df["symbol"].unique())
        self.sectors = sorted(df["industry"].unique())
        self.sym = df["symbol"].map({s: i for i, s in enumerate(self.symbols)}).to_numpy(np.int64)
        self.sec = df["industry"].map({s: i for i, s in enumerate(self.sectors)}).to_numpy(np.int64)
        self.pos = df.groupby("symbol", sort=False).cumcount().to_numpy()

        cal = panel.calendar
        self.KC = cal[self.known].to_numpy(np.float32)
        lookup = {d: i for i, d in enumerate(cal["trade_date"])}
        self.cal_idx = df["trade_date"].map(lookup).to_numpy(np.int64)

        self.y_up = df[[f"up_{h}" for h in self.horizons]].to_numpy(np.float32)
        vol = df[[f"fwd_vol_{h}" for h in self.horizons]].to_numpy(np.float32)
        self.y_logvol = np.log(vol + 1e-4)
        self.y_path = df[[f"fwd_ret_{k}" for k in range(1, self.H + 1)]].to_numpy(np.float32)

    # ---- scaling: fitted on training rows only ----
    def fit_scaler(self, rows: np.ndarray) -> None:
        x = self.X[rows]
        self.mean = np.nanmean(x, axis=0)
        self.std = np.nanstd(x, axis=0)
        self.std[~np.isfinite(self.std) | (self.std < 1e-8)] = 1.0
        self.mean[~np.isfinite(self.mean)] = 0.0
        xs = np.clip((self.X - self.mean) / self.std, -5, 5)
        self.Xs = np.nan_to_num(xs, nan=0.0).astype(np.float32)

    def set_scaler(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.mean, self.std = mean, std
        self.Xs = np.nan_to_num(np.clip((self.X - mean) / std, -5, 5), nan=0.0).astype(np.float32)

    def sample_rows(self, mask: np.ndarray, need_targets: bool) -> np.ndarray:
        ok = mask & (self.pos >= self.window - 1)
        if need_targets:
            ok &= ~np.isnan(self.y_up).any(axis=1) & ~np.isnan(self.y_path).any(axis=1)
        return np.flatnonzero(ok)

    def batch(self, rows: np.ndarray) -> dict:
        back = np.arange(self.window - 1, -1, -1)
        idx = rows[:, None] - back[None, :]                                  # (B, W) rows of the same stock
        fut = np.minimum(self.cal_idx[rows][:, None] + np.arange(self.H)[None, :], len(self.KC) - 1)
        t = torch.from_numpy
        return {
            "x_past": t(self.Xs[idx]),
            "k_past": t(self.KC[self.cal_idx[idx]]),
            "k_future": t(self.KC[fut]),
            "sym": t(self.sym[rows]),
            "sec": t(self.sec[rows]),
            "y_up": t(self.y_up[rows]),
            "y_logvol": t(self.y_logvol[rows]),
            "y_path": t(self.y_path[rows]),
        }


def set_threads(n: int) -> None:
    torch.set_num_threads(n if n and n > 0 else (os.cpu_count() or 1))


def train_net(net: torch.nn.Module, data: SequenceData, train_rows: np.ndarray, val_rows: np.ndarray,
              loss_fn, seq_cfg: dict, seed: int, log=print) -> list[dict]:
    """Mini-batch AdamW with early stopping on val_rows; restores the best weights."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(net.parameters(), lr=seq_cfg["lr"], weight_decay=seq_cfg["weight_decay"])
    bs = int(seq_cfg["batch_size"])
    per_epoch = int(seq_cfg.get("samples_per_epoch") or 0)
    best, best_state, wait, history = np.inf, None, 0, []
    for epoch in range(int(seq_cfg["max_epochs"])):
        t0 = time.time()
        net.train()
        if 0 < per_epoch < len(train_rows):
            order = rng.choice(train_rows, per_epoch, replace=False)
        else:
            order = rng.permutation(train_rows)
        total, count = 0.0, 0
        for i in range(0, len(order), bs):
            b = data.batch(order[i:i + bs])
            loss = loss_fn(net(b), b)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(b["sym"])
            count += len(b["sym"])
        val = evaluate_loss(net, data, val_rows, loss_fn, bs) if len(val_rows) else total / count
        history.append({"epoch": epoch + 1, "train_loss": total / count, "val_loss": val, "sec": time.time() - t0})
        log(f"  epoch {epoch + 1}: train {total / count:.5f}  early-stop {val:.5f}  ({time.time() - t0:.0f}s)")
        if val < best - 1e-6:
            best, best_state, wait = val, copy.deepcopy(net.state_dict()), 0
        else:
            wait += 1
            if wait >= int(seq_cfg["patience"]):
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    return history


@torch.no_grad()
def evaluate_loss(net, data, rows, loss_fn, bs) -> float:
    net.eval()
    total = 0.0
    for i in range(0, len(rows), bs):
        b = data.batch(rows[i:i + bs])
        total += float(loss_fn(net(b), b)) * len(b["sym"])
    return total / max(len(rows), 1)


@torch.no_grad()
def predict_net(net, data, rows, bs) -> list[dict]:
    net.eval()
    return [net(data.batch(rows[i:i + bs])) for i in range(0, len(rows), bs)]
