"""Block bootstrap over trading days (task 18.3). Days are resampled in blocks so the intervals
respect the correlation between stocks on the same day and between nearby days."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.metrics import score


def block_bootstrap(joined: pd.DataFrame, metrics: list[str], n: int = 500, block: int = 20,
                    seed: int = 42, alpha: float = 0.05) -> dict[str, tuple[float, float]]:
    """(low, high) interval of each metric for one model and horizon (rows of `joined`)."""
    days = np.sort(joined["trade_date"].unique())
    groups = joined.groupby("trade_date").indices
    starts = np.arange(0, max(len(days) - block + 1, 1))
    n_blocks = int(np.ceil(len(days) / block))
    rng = np.random.default_rng(seed)
    values = {m: [] for m in metrics}
    for _ in range(n):
        chosen = rng.choice(starts, n_blocks)
        picked = np.concatenate([days[s:s + block] for s in chosen])
        idx = np.concatenate([groups[d] for d in picked])
        s = score(joined.iloc[idx])
        for m in metrics:
            values[m].append(s[m])
    return {m: (float(np.nanpercentile(v, 100 * alpha / 2)), float(np.nanpercentile(v, 100 * (1 - alpha / 2))))
            for m, v in values.items()}


def metrics_with_intervals(joined: pd.DataFrame, metrics=("accuracy", "brier", "vol_mae"), **kw) -> pd.DataFrame:
    rows = []
    for (model, h), g in joined.groupby(["model", "horizon_days"], sort=True):
        point = score(g)
        present = [m for m in metrics if not np.isnan(point[m])]        # skip metrics this model lacks
        ci = block_bootstrap(g, present, **kw) if present else {}
        row = {"model": model, "horizon_days": h, "n": point["n_vol"] if np.isnan(point["n"]) else point["n"]}
        for m in metrics:
            row[m], row[f"{m}_low"], row[f"{m}_high"] = point[m], *ci.get(m, (np.nan, np.nan))
        rows.append(row)
    return pd.DataFrame(rows)
